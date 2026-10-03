"""Process-restart and adversarial boundaries over a TEST-ONLY SQLite port.

These tests exercise the real training worker against durable local fixture
storage. They do not establish live MINDEX authorization or deployed durability.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from nlm.evidence_training.contracts import ContractError, decode, encode, validate_manifest
from nlm.evidence_training.engine import train
from nlm.evidence_training.worker import ArtifactRef, JobCancelled, LeaseLost, RetentionUnavailable, run_job
from tests.evidence_training.fake_mindex import Scope, SimulatedCrash, SQLiteMindexFake
from tests.evidence_training.fixtures import fixture


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nlm-worker-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "fixture.sqlite"
        self.raw, self.captures, self.config = fixture()
        self.config = {**self.config, "epochs": 4}
        self.scope = Scope()
        self.port = SQLiteMindexFake(self.path, self.scope)
        self.port.seed_job("job-a", self.raw, self.captures, self.config)

    async def output_bytes(self, port, job_id="job-a"):
        return {role: await port.read_artifact(ArtifactRef.from_dict(reference))
                for role, reference in port.job(job_id)["outputs"].items()}

    async def test_success_requires_saved_bytes_and_fresh_store_readback(self):
        receipt = await run_job(self.port, "job-a", "worker-a")
        self.assertTrue(receipt["fixture_only"])
        self.assertEqual(receipt["job"]["state"], "succeeded")
        fresh = SQLiteMindexFake(self.path, self.scope)
        outputs = await self.output_bytes(fresh)
        expected = train(validate_manifest(self.raw), self.config)
        self.assertEqual(outputs, {"model": expected.model_bytes, "evaluation": expected.evaluation_bytes,
                                   "checkpoint": expected.checkpoint_bytes})
        events = fresh.events()
        completion = next(i for i, event in enumerate(events) if event["action"] == "complete")
        for role, reference in fresh.job("job-a")["outputs"].items():
            for action in ("read", "server_read"):
                self.assertTrue(any(event["action"] == action and event.get("artifact_id") == reference["artifact_id"]
                                    for event in events[:completion]), (role, action))
        self.assertEqual((await run_job(fresh, "job-a", "another-worker"))["state"], "not_claimed")

    async def test_resume_at_every_supported_epoch_boundary_is_exact(self):
        expected = train(validate_manifest(self.raw), self.config)
        for budget in range(self.config["epochs"]):
            with self.subTest(completed_epochs=budget):
                path = Path(self.temporary.name) / f"resume-{budget}.sqlite"
                first = SQLiteMindexFake(path, self.scope)
                first.seed_job("resumed-job", self.raw, self.captures, self.config)
                receipt = await run_job(first, "resumed-job", "old-worker", invocation_epoch_budget=budget)
                self.assertEqual(receipt["state"], "checkpointed")
                self.assertEqual(first.job("resumed-job")["state"], "running")
                checkpoint = await first.read_artifact(ArtifactRef.from_dict(receipt["checkpoint"]))
                self.assertEqual(decode(checkpoint)["epoch"], budget)
                fresh = SQLiteMindexFake(path, self.scope)
                self.assertEqual((await run_job(fresh, "resumed-job", "new-worker"))["state"], "not_claimed")
                fresh.advance(60)
                receipt = await run_job(fresh, "resumed-job", "new-worker")
                self.assertEqual(receipt["job"]["state"], "succeeded")
                self.assertEqual(await self.output_bytes(fresh, "resumed-job"), {
                    "model": expected.model_bytes, "evaluation": expected.evaluation_bytes,
                    "checkpoint": expected.checkpoint_bytes})

    async def test_process_death_after_durable_checkpoint_resumes_in_new_instances(self):
        def crash(port, detail):
            raise SimulatedCrash("worker process stopped after checkpoint commit")
        crashing = SQLiteMindexFake(self.path, self.scope, hooks={"after_checkpoint": crash})
        with self.assertRaises(SimulatedCrash):
            await run_job(crashing, "job-a", "worker-before-crash")
        persisted = self.port.job("job-a")
        self.assertEqual(persisted["state"], "running")
        checkpoint = await self.port.read_artifact(ArtifactRef.from_dict(persisted["checkpoint"]))
        self.assertEqual(decode(checkpoint)["epoch"], 1)
        fresh = SQLiteMindexFake(self.path, self.scope)
        fresh.advance(61)
        await run_job(fresh, "job-a", "worker-after-crash")
        expected = train(validate_manifest(self.raw), self.config)
        self.assertEqual((await self.output_bytes(fresh))["model"], expected.model_bytes)
        self.assertFalse(any(event["action"] == "fail" for event in fresh.events()))

    async def test_claim_is_exclusive_and_reclaim_increments_fence_even_for_same_worker(self):
        other = SQLiteMindexFake(self.path, self.scope)
        # Distinct threads and connections contend for the transaction; plain
        # gather on this synchronous-disk fake would run the claims sequentially.
        first, second = await asyncio.gather(
            asyncio.to_thread(lambda: asyncio.run(self.port.claim("job-a", "same-worker", 60))),
            asyncio.to_thread(lambda: asyncio.run(other.claim("job-a", "same-worker", 60))))
        self.assertEqual(sum(lease is not None for lease in (first, second)), 1)
        old = first or second
        self.port.advance(60)
        fresh = await SQLiteMindexFake(self.path, self.scope).claim("job-a", "same-worker", 60)
        self.assertGreater(fresh.epoch, old.epoch)
        self.assertNotEqual(fresh.token, old.token)
        with self.assertRaises(LeaseLost):
            await self.port.heartbeat(old, 60)

    async def test_stale_fence_cannot_heartbeat_checkpoint_fail_or_complete(self):
        claims = []
        port = SQLiteMindexFake(self.path, self.scope, hooks={"after_claim": lambda _, detail: claims.append(detail["lease"])})
        await run_job(port, "job-a", "same-worker", invocation_epoch_budget=1)
        old = claims[0]
        reference = ArtifactRef.from_dict(port.job("job-a")["checkpoint"])
        port.advance(61)
        fresh = await self.port.claim("job-a", "same-worker", 60)
        self.assertGreater(fresh.epoch, old.epoch)
        mutations = [lambda: port.heartbeat(old, 60), lambda: port.checkpoint(old, reference),
                     lambda: port.fail(old, "bad"), lambda: port.complete(old, {})]
        for operation in mutations:
            with self.assertRaises(LeaseLost):
                await operation()
        self.assertEqual(port.job("job-a")["state"], "running")
        self.assertEqual(port.job("job-a")["checkpoint"], reference.to_dict())
        self.assertFalse(any(event["action"] in ("fail", "complete") for event in port.events()))

    async def test_lease_expiry_without_competitor_still_rejects_mutation(self):
        lease = await self.port.claim("job-a", "worker", 10)
        self.port.advance(10)
        with self.assertRaises(LeaseLost):
            await self.port.heartbeat(lease, 60)
        with self.assertRaises(LeaseLost):
            await self.port.fail(lease, "stale")

    async def test_expiry_during_compute_blocks_checkpoint_publication(self):
        heartbeats = 0
        def expire_after_precompute_heartbeat(port, detail):
            nonlocal heartbeats
            heartbeats += 1
            if heartbeats == len(self.captures) + 1:
                port.advance(60)
        port = SQLiteMindexFake(self.path, self.scope,
                                hooks={"after_heartbeat": expire_after_precompute_heartbeat})
        with self.assertRaises(LeaseLost):
            await run_job(port, "job-a", "worker")
        self.assertIsNone(port.job("job-a")["checkpoint"])
        self.assertFalse(any(event["action"] == "put" for event in port.events()))

    async def test_cancellation_after_checkpoint_survives_restart(self):
        def cancel(port, detail):
            SQLiteMindexFake(self.path, self.scope).cancel("job-a")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"after_checkpoint": cancel})
        with self.assertRaises(JobCancelled):
            await run_job(port, "job-a", "worker")
        fresh = SQLiteMindexFake(self.path, self.scope)
        self.assertEqual(fresh.job("job-a")["state"], "cancelled")
        self.assertIsNotNone(fresh.job("job-a")["checkpoint"])
        self.assertIsNone(fresh.job("job-a")["outputs"])
        fresh.advance(61)
        self.assertEqual((await run_job(fresh, "job-a", "worker-new"))["state"], "not_claimed")

    async def test_cancellation_at_completion_race_wins_without_success(self):
        def cancel(port, detail):
            SQLiteMindexFake(self.path, self.scope).cancel("job-a")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"before_complete": cancel})
        with self.assertRaises(JobCancelled):
            await run_job(port, "job-a", "worker")
        self.assertEqual(port.job("job-a")["state"], "cancelled")
        self.assertIsNone(port.job("job-a")["outputs"])
        self.assertFalse(any(event["action"] == "complete" for event in port.events()))

    async def test_successful_completion_is_not_rewritten_by_late_cancellation(self):
        await run_job(self.port, "job-a", "worker")
        self.port.cancel("job-a")
        self.assertEqual(self.port.job("job-a")["state"], "succeeded")

    async def test_corrupted_model_upload_readback_never_marks_success(self):
        def corrupt(port, detail):
            if detail["role"] == "model":
                port.corrupt_artifact(detail["reference"].artifact_id, b"corrupted persisted model")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"after_put": corrupt})
        with self.assertRaises(RetentionUnavailable):
            await run_job(port, "job-a", "worker")
        self.assertEqual(port.job("job-a")["state"], "running")
        self.assertIsNone(port.job("job-a")["outputs"])
        self.assertFalse(any(event["action"] in ("fail", "complete") for event in port.events()))
        fresh = SQLiteMindexFake(self.path, self.scope)
        fresh.advance(61)
        self.assertEqual((await run_job(fresh, "job-a", "retry-worker"))["job"]["state"], "succeeded")

    async def test_server_reverifies_artifacts_after_client_readback(self):
        def corrupt(port, detail):
            port.corrupt_artifact(detail["outputs"]["model"].artifact_id, b"changed between checks")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"before_complete": corrupt})
        with self.assertRaises(RetentionUnavailable):
            await run_job(port, "job-a", "worker")
        self.assertEqual(port.job("job-a")["state"], "running")
        self.assertIsNone(port.job("job-a")["outputs"])

    async def test_lost_artifact_upload_ack_can_resume_from_final_checkpoint(self):
        def disconnect(port, detail):
            if detail["role"] == "model":
                raise ConnectionError("upload committed but acknowledgement lost")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"after_put": disconnect})
        with self.assertRaises(ConnectionError):
            await run_job(port, "job-a", "worker")
        saved = ArtifactRef.from_dict(port.job("job-a")["checkpoint"])
        self.assertEqual(decode(await port.read_artifact(saved))["epoch"], self.config["epochs"])
        self.assertEqual(port.job("job-a")["state"], "running")
        fresh = SQLiteMindexFake(self.path, self.scope)
        fresh.advance(61)
        await run_job(fresh, "job-a", "replacement")
        self.assertEqual((await self.output_bytes(fresh))["checkpoint"], train(validate_manifest(self.raw), self.config).checkpoint_bytes)
        self.assertEqual(sum(event["action"] == "complete" for event in fresh.events()), 1)

    async def test_upload_idempotency_replays_exact_bytes_and_rejects_conflict(self):
        first = await self.port.put_artifact("job-a", "checkpoint", b"fixture", "same-key")
        fresh = SQLiteMindexFake(self.path, self.scope)
        self.assertEqual(first, await fresh.put_artifact("job-a", "checkpoint", b"fixture", "same-key"))
        with self.assertRaises(ContractError):
            await fresh.put_artifact("job-a", "checkpoint", b"different", "same-key")
        with self.assertRaises(ContractError):
            await fresh.put_artifact("job-a", "model", b"fixture", "same-key")

    async def test_upload_receipt_for_different_retained_bytes_is_rejected(self):
        class SwappedReceiptPort(SQLiteMindexFake):
            async def put_artifact(self, job_id, role, raw, idempotency_key):
                await super().put_artifact(job_id, role, raw, idempotency_key)
                return ArtifactRef.from_dict(self.job(job_id)["dataset"])
        port = SwappedReceiptPort(self.path, self.scope)
        with self.assertRaises(RetentionUnavailable):
            await run_job(port, "job-a", "worker")
        self.assertIsNone(port.job("job-a")["checkpoint"])
        self.assertIsNone(port.job("job-a")["outputs"])

    async def test_lost_completion_ack_preserves_one_durable_terminal_result(self):
        def lose_ack(port, detail):
            raise ConnectionError("completion committed; acknowledgement lost")
        port = SQLiteMindexFake(self.path, self.scope, hooks={"after_complete": lose_ack})
        with self.assertRaises(ConnectionError):
            await run_job(port, "job-a", "worker")
        fresh = SQLiteMindexFake(self.path, self.scope)
        fresh.advance(61)
        self.assertEqual((await run_job(fresh, "job-a", "replacement"))["state"], "not_claimed")
        self.assertEqual(fresh.job("job-a")["state"], "succeeded")
        self.assertEqual(sum(event["action"] == "complete" for event in fresh.events()), 1)
        self.assertEqual((await self.output_bytes(fresh))["model"], train(validate_manifest(self.raw), self.config).model_bytes)

    async def test_non_success_completion_receipt_is_rejected(self):
        class QueuedReceiptPort(SQLiteMindexFake):
            async def complete(self, lease, outputs):
                await super().complete(lease, outputs)
                return {"job": {"id": lease.job["id"], "state": "queued"}}
        with self.assertRaises(RetentionUnavailable):
            await run_job(QueuedReceiptPort(self.path, self.scope), "job-a", "worker")

    async def test_malformed_evidence_fails_only_under_current_fence(self):
        bad = decode(self.raw)
        bad["dataset"]["units"][0] = "kelvin"
        self.port.seed_job("invalid", encode(bad), self.captures, self.config)
        with self.assertRaises(ContractError):
            await run_job(self.port, "invalid", "worker")
        job = self.port.job("invalid")
        self.assertEqual((job["state"], job["error_code"]), ("failed", "invalid_training_evidence"))
        self.assertIsNone(job["outputs"])

    async def test_admitted_engine_pin_mismatch_prevents_any_training_artifacts(self):
        pin = {**self.port.job("job-a")["engine_pin"], "source_sha256": "0" * 64}
        self.port.replace_job_payload("job-a", engine_pin=pin)
        with self.assertRaises(ContractError):
            await run_job(self.port, "job-a", "worker")
        self.assertIsNone(self.port.job("job-a")["checkpoint"])
        self.assertIsNone(self.port.job("job-a")["outputs"])
        self.assertFalse(any(event["action"] == "put" for event in self.port.events()))

    async def test_corrupt_capture_fails_readback_before_any_checkpoint(self):
        artifact_id = next(iter(self.captures))
        self.port.corrupt_artifact(artifact_id, b"wrong raw capture bytes")
        with self.assertRaises(RetentionUnavailable):
            await run_job(self.port, "job-a", "worker")
        job = self.port.job("job-a")
        self.assertEqual(job["state"], "running")
        self.assertIsNone(job["checkpoint"])
        self.assertIsNone(job["outputs"])

    async def test_each_identity_dimension_denies_job_artifact_and_mutation_access(self):
        lease = await self.port.claim("job-a", "worker", 60)
        reference = ArtifactRef.from_dict(self.port.job("job-a")["dataset"])
        scopes = [replace(self.scope, issuer="https://other.fixture.invalid"),
                  replace(self.scope, subject="unrelated-user"), replace(self.scope, tenant="tenant-b"),
                  replace(self.scope, project="project-b")]
        for scope in scopes:
            with self.subTest(scope=scope):
                other = SQLiteMindexFake(self.path, scope)
                with self.assertRaises(PermissionError):
                    other.job("job-a")
                with self.assertRaises(PermissionError):
                    other.cancel("job-a")
                operations = [lambda: other.claim("job-a", "worker", 60), lambda: other.read_artifact(reference),
                              lambda: other.heartbeat(lease, 60), lambda: other.checkpoint(lease, reference),
                              lambda: other.complete(lease, {}), lambda: other.fail(lease, "foreign"),
                              lambda: other.put_artifact("job-a", "model", b"foreign", "key")]
                for operation in operations:
                    with self.assertRaises(PermissionError):
                        await operation()
        self.assertEqual(self.port.job("job-a")["state"], "running")

    async def test_foreign_dataset_reference_does_not_confer_artifact_authority(self):
        foreign = SQLiteMindexFake(self.path, Scope(subject="user-b", tenant="tenant-b", project="project-b"))
        ref = foreign.seed_artifact(str(uuid.uuid5(uuid.NAMESPACE_URL, "private-foreign-manifest")), self.raw)
        self.port.replace_job_payload("job-a", dataset=ref.to_dict())
        with self.assertRaises(PermissionError):
            await run_job(self.port, "job-a", "worker")
        self.assertIsNone(self.port.job("job-a")["checkpoint"])
        self.assertIsNone(self.port.job("job-a")["outputs"])

    async def test_foreign_capture_reference_is_rechecked_independently(self):
        raw = decode(self.raw)
        original_id = raw["captures"][0]["artifact_id"]
        foreign_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "foreign-retained-capture"))
        raw["captures"][0]["artifact_id"] = foreign_id
        foreign = SQLiteMindexFake(self.path, Scope(subject="user-b", tenant="tenant-b", project="project-b"))
        foreign.seed_artifact(foreign_id, self.captures[original_id])
        self.port.seed_job("foreign-capture-job", encode(raw), self.captures, self.config)
        with self.assertRaises(PermissionError):
            await run_job(self.port, "foreign-capture-job", "worker")
        self.assertIsNone(self.port.job("foreign-capture-job")["checkpoint"])

    async def test_two_scopes_with_identical_job_ids_keep_distinct_evidence(self):
        raw_b = decode(self.raw)
        raw_b["dataset"]["source"] = "Second private fixture scope"
        scope_b = Scope(subject="user-b", tenant="tenant-b", project="project-b")
        other = SQLiteMindexFake(self.path, scope_b)
        other.seed_job("job-a", encode(raw_b), self.captures, self.config)
        await run_job(self.port, "job-a", "worker-a")
        await run_job(other, "job-a", "worker-b")
        a, b = await self.output_bytes(self.port), await self.output_bytes(other)
        self.assertNotEqual(a["model"], b["model"])
        self.assertEqual(decode(a["evaluation"])["dataset_scope"], decode(self.raw)["dataset"]["source"])
        self.assertEqual(decode(b["evaluation"])["dataset_scope"], "Second private fixture scope")
        with self.assertRaises(PermissionError):
            await other.read_artifact(ArtifactRef.from_dict(self.port.job("job-a")["outputs"]["model"]))


if __name__ == "__main__":
    unittest.main()
