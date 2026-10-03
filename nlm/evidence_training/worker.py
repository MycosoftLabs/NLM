"""Worker algorithm over canonical MINDEX retention and atomic lease ports.

No local job database or production store fallback exists. Uploads can outlive a
lost lease, but only an atomic fenced checkpoint/complete can link them to a job.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import ENGINE_ID, HASH, MAX_BYTES, ContractError, decode, digest, validate_manifest
from .engine import engine_pin, predict, train


class LeaseLost(RuntimeError):
    """Worker no longer owns a nonexpired lease fence."""


class JobCancelled(LeaseLost):
    """Owner cancellation invalidated the lease."""


class RetentionUnavailable(RuntimeError):
    """Durable private artifact readback has not been established."""


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    byte_length: int

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactRef":
        if not isinstance(value, dict) or set(value) != {"artifact_id", "sha256", "byte_length"}:
            raise ContractError("Invalid artifact reference")
        if not isinstance(value["artifact_id"], str) or not 1 <= len(value["artifact_id"]) <= 512 or not isinstance(value["sha256"], str) or not HASH.fullmatch(value["sha256"]) or type(value["byte_length"]) is not int or not 0 < value["byte_length"] <= MAX_BYTES:
            raise ContractError("Invalid artifact reference fields")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256, "byte_length": self.byte_length}


@dataclass(frozen=True)
class Lease:
    job: dict[str, Any]
    token: str
    epoch: int
    expires_at: str


class MindexPort(Protocol):
    """All methods use verified original identity + scoped worker authorization.

    claim/heartbeat/checkpoint/complete/fail must be atomic in authoritative MINDEX.
    Token+epoch, expiry, cancellation and current job state guard every mutation.
    Reads and uploads must independently enforce shared09 tenant/project identity.
    complete must independently verify all immutable artifacts before success.
    """

    async def claim(self, job_id: str, worker_id: str, lease_seconds: int) -> Lease | None: ...
    async def heartbeat(self, lease: Lease, lease_seconds: int) -> Lease: ...
    async def read_artifact(self, reference: ArtifactRef) -> bytes: ...
    async def put_artifact(self, job_id: str, role: str, raw: bytes, idempotency_key: str) -> ArtifactRef: ...
    async def checkpoint(self, lease: Lease, reference: ArtifactRef) -> None: ...
    async def complete(self, lease: Lease, outputs: dict[str, ArtifactRef]) -> dict[str, Any]: ...
    async def fail(self, lease: Lease, error_code: str) -> None: ...


async def read_verified(port: MindexPort, reference: ArtifactRef) -> bytes:
    raw = await port.read_artifact(reference)
    if not isinstance(raw, bytes) or len(raw) != reference.byte_length or digest(raw) != reference.sha256:
        raise RetentionUnavailable("Artifact bytes fail immutable length/hash readback")
    return raw


async def run_job(port: MindexPort, job_id: str, worker_id: str, *, lease_seconds: int = 60,
                  invocation_epoch_budget: int | None = None) -> dict[str, Any]:
    """Execute one admitted job. A budget returns a checkpoint, never success.

    A process crash leaves a lease for later expiry/reclaim. Cancellation and lost
    fences stop writes; transient retention failures preserve the resumable job.
    """
    if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id) > 128:
        raise ContractError("Worker id required")
    if type(lease_seconds) is not int or not 10 <= lease_seconds <= 300:
        raise ContractError("Lease must be 10..300 seconds")
    if invocation_epoch_budget is not None and (type(invocation_epoch_budget) is not int or not 0 <= invocation_epoch_budget <= 200):
        raise ContractError("Invocation epoch budget must be 0..200")
    lease = await port.claim(job_id, worker_id, lease_seconds)
    if lease is None:
        return {"state": "not_claimed", "job_id": job_id}
    try:
        if lease.job.get("id") != job_id or lease.job.get("engine_id") != ENGINE_ID or lease.job.get("engine_version") != "1":
            raise ContractError("Lease is not for the requested supported engine job")
        if lease.job.get("engine_pin") != engine_pin():
            raise ContractError("Admitted engine source/license/runtime pin mismatch")
        manifest_ref = ArtifactRef.from_dict(lease.job["dataset"])
        manifest = validate_manifest(await read_verified(port, manifest_ref))
        manifest.require_training_ready()
        # Recheck every retained original capture under this worker's identity.
        for capture in manifest.captures:
            lease = await port.heartbeat(lease, lease_seconds)
            await read_verified(port, ArtifactRef(capture.artifact_id, capture.sha256, capture.byte_length))
        if not isinstance(lease.job["parameters"], dict) or set(lease.job["parameters"]) != {"epochs", "learning_rate", "l2"}:
            raise ContractError("Unexpected job parameters or shadowed seed")
        config = {**lease.job["parameters"], "seed": lease.job["seed"]}
        checkpoint_ref = lease.job.get("checkpoint")
        checkpoint_raw = await read_verified(port, ArtifactRef.from_dict(checkpoint_ref)) if checkpoint_ref else None
        completed_in_call = 0

        async def retain(role: str, raw: bytes) -> ArtifactRef:
            nonlocal lease
            lease = await port.heartbeat(lease, lease_seconds)
            reference = await port.put_artifact(job_id, role, raw, f"nlm:{job_id}:{lease.epoch}:{role}:{digest(raw)}")
            # Verify references are to the exact bytes uploaded, not just some valid object.
            if reference.sha256 != digest(raw) or reference.byte_length != len(raw):
                raise RetentionUnavailable("Upload receipt does not match produced artifact")
            if await read_verified(port, reference) != raw:
                raise RetentionUnavailable("Uploaded artifact bytes changed")
            lease = await port.heartbeat(lease, lease_seconds)
            return reference

        while True:
            lease = await port.heartbeat(lease, lease_seconds)
            budget = 0 if invocation_epoch_budget is not None and completed_in_call >= invocation_epoch_budget else 1
            result = await asyncio.to_thread(train, manifest, config, checkpoint_bytes=checkpoint_raw, max_epochs=budget)
            checkpoint_raw = result.checkpoint_bytes
            # Heartbeat after computation prevents writes from an expired or cancelled worker.
            lease = await port.heartbeat(lease, lease_seconds)
            checkpoint_ref = await retain("checkpoint", checkpoint_raw)
            await port.checkpoint(lease, checkpoint_ref)
            if result.completed:
                model_ref = await retain("model", result.model_bytes)
                evaluation_ref = await retain("evaluation", result.evaluation_bytes)
                # Reload the retained model and reproduce heldout predictions before completion.
                reloaded = await read_verified(port, model_ref)
                evaluation = decode(await read_verified(port, evaluation_ref))
                by_id = {r.id: r for r in manifest.rows}
                for prediction in evaluation["predictions"]:
                    actual = predict(reloaded, list(by_id[prediction["id"]].values))
                    if actual["label"] != prediction["predicted"] or actual["probabilities"] != prediction["probabilities"]:
                        raise RetentionUnavailable("Reloaded model does not reproduce heldout evaluation")
                lease = await port.heartbeat(lease, lease_seconds)
                receipt = await port.complete(lease, {"model": model_ref, "evaluation": evaluation_ref, "checkpoint": checkpoint_ref})
                # A service ACK or a queued status is insufficient.
                job = receipt.get("job", receipt)
                if job.get("id") != job_id or job.get("state") != "succeeded":
                    raise RetentionUnavailable("MINDEX did not return an authoritative succeeded receipt")
                return receipt
            completed_in_call += budget
            if invocation_epoch_budget is not None and completed_in_call >= invocation_epoch_budget:
                return {"state": "checkpointed", "job_id": job_id, "checkpoint": checkpoint_ref.to_dict(),
                        "note": "Lease remains active until expiry; no success or pause is claimed"}
    except LeaseLost:
        raise
    except ContractError:
        # Only validated permanent input incompatibility marks failed. Transport
        # outages and process crashes remain recoverable through lease expiry.
        await port.fail(lease, "invalid_training_evidence")
        raise
