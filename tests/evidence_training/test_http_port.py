import asyncio
import io
import unittest
from urllib.error import HTTPError

from nlm.evidence_training.contracts import ContractError, digest, encode
from nlm.evidence_training.http_port import MindexHttpPort, HttpFailure, _NoRedirect
from nlm.evidence_training.worker import ArtifactRef, Lease, LeaseLost, RetentionUnavailable, read_verified

TENANT = "11111111-1111-4111-8111-111111111111"
PROJECT = "22222222-2222-4222-8222-222222222222"
JOB = "33333333-3333-4333-8333-333333333333"
ARTIFACT = "44444444-4444-4444-8444-444444444444"


class Response:
    def __init__(self, body, headers=None):
        self.body, self.headers = body, headers or {}

    def read(self, limit):
        return self.body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class HttpPortTests(unittest.IsolatedAsyncioTestCase):
    def port(self, *responses, **kwargs):
        port = MindexHttpPort("https://mindex.test/api/mindex", lambda: "test-original-jwt", TENANT, PROJECT, **kwargs)
        port._opener = Opener(*responses)
        return port

    def lease_payload(self, **changes):
        return {"contract_version": "nlm-training.v1", "lease": {"job": {"id": JOB}, "token": "a" * 64,
                "epoch": 1, "expires_at": "2026-10-01T22:00:00Z", **changes}}

    async def test_exact_claim_wire_identity_and_worker_policy_headers(self):
        port = self.port(Response(encode(self.lease_payload())), worker_headers=lambda: {"X-NLM-Worker-Authorization": "test-worker-proof"})
        lease = await port.claim(JOB, "fixture-worker", 60)
        self.assertEqual(lease.token, "a" * 64)
        req = port._opener.requests[0]
        headers = {key.lower(): value for key, value in req.header_items()}
        self.assertEqual(req.full_url, f"https://mindex.test/api/mindex/nlm-training/v1/worker/jobs/{JOB}/claim")
        self.assertEqual(headers["authorization"], "Bearer test-original-jwt")
        self.assertEqual(headers["x-tenant-id"], TENANT)
        self.assertEqual(headers["x-project-id"], PROJECT)
        self.assertEqual(headers["x-nlm-worker-authorization"], "test-worker-proof")
        self.assertEqual(req.data, encode({"worker_id": "fixture-worker", "lease_seconds": 60}))

    async def test_receipt_must_be_verified_scoped_and_content_matches(self):
        raw = b'{"fixture":true}'
        receipt = {"contract_version": "retention.v1", "artifact_id": ARTIFACT, "sha256": digest(raw),
                   "byte_length": len(raw), "state": "archived_verified", "archive_verified": True,
                   "tenant_id": TENANT, "project_id": PROJECT}
        content_headers = {"X-Retention-Contract": "retention.v1", "X-Artifact-Id": ARTIFACT, "X-Artifact-SHA256": digest(raw)}
        port = self.port(Response(encode(receipt)), Response(raw, content_headers))
        ref = await port.put_artifact(JOB, "model", raw, "x" * 300)
        self.assertEqual(await read_verified(port, ref), raw)
        headers = {key.lower(): value for key, value in port._opener.requests[0].header_items()}
        self.assertEqual(len(headers["idempotency-key"]), 68)
        for change in ({"state": "pending", "archive_verified": False}, {"tenant_id": PROJECT}, {"sha256": "0" * 64}):
            bad = self.port(Response(encode({**receipt, **change})), archive_wait_seconds=0)
            with self.assertRaises(RetentionUnavailable):
                await bad.put_artifact(JOB, "model", raw, "fixture")

    async def test_content_hash_and_header_mismatch_fail(self):
        ref = ArtifactRef(ARTIFACT, digest(b"safe"), 4)
        port = self.port(Response(b"safe", {}))
        with self.assertRaises(RetentionUnavailable):
            await read_verified(port, ref)
        port = self.port(Response(b"evil", {"X-Retention-Contract": "retention.v1", "X-Artifact-Id": ARTIFACT, "X-Artifact-SHA256": ref.sha256}))
        with self.assertRaises(RetentionUnavailable):
            await read_verified(port, ref)

    async def test_auth_failure_does_not_fallback_and_no_claimable_is_distinct(self):
        def error(code, payload):
            return HTTPError("https://fixture.invalid", code, "failure", {}, io.BytesIO(encode(payload)))
        for code in (401, 403):
            port = self.port(error(code, {"detail": "NEVER_ECHO_PRIVATE_RESPONSE"}))
            with self.assertRaises(PermissionError) as caught:
                await port.claim(JOB, "worker", 60)
            self.assertNotIn("NEVER_ECHO", str(caught.exception))
            self.assertEqual(len(port._opener.requests), 1)
        port = self.port(error(409, {"detail": {"code": "job_not_claimable"}}))
        self.assertIsNone(await port.claim(JOB, "worker", 60))
        port = self.port(error(409, {"detail": {"code": "stale_or_expired_lease"}}))
        with self.assertRaises(LeaseLost):
            await port.heartbeat(Lease({"id": JOB}, "a" * 64, 1, "unused"), 60)

    async def test_refresh_identity_and_reject_changed_fence(self):
        port = self.port(Response(encode(self.lease_payload())), Response(encode(self.lease_payload(token="b" * 64))))
        values = iter(["jwt-one", "jwt-two"])
        port.token_provider = lambda: next(values)
        lease = await port.claim(JOB, "worker", 60)
        with self.assertRaises(LeaseLost):
            await port.heartbeat(lease, 60)
        self.assertEqual(port._opener.requests[-1].get_header("Authorization"), "Bearer jwt-two")

    async def test_no_redirect_or_identity_override_or_insecure_remote(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.test"))
        with self.assertRaises(ContractError):
            MindexHttpPort("http://remote.test/api/mindex", lambda: "token", TENANT, PROJECT)
        port = self.port(worker_headers=lambda: {"Authorization": "Bearer other-person"})
        with self.assertRaises(ContractError):
            await port.claim(JOB, "worker", 60)
        self.assertEqual(port._opener.requests, [])


if __name__ == "__main__":
    unittest.main()
