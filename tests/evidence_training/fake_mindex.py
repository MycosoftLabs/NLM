"""SQLite TEST DOUBLE for the MINDEX worker port, never a production adapter.

Every method opens a fresh connection to a caller-owned temporary database. This
lets tests discard workers and port instances without discarding job fences or
evidence. Passing these tests does not qualify a deployed MINDEX service.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from nlm.evidence_training.contracts import ContractError, decode, digest
from nlm.evidence_training.engine import engine_pin
from nlm.evidence_training.worker import ArtifactRef, JobCancelled, Lease, LeaseLost, RetentionUnavailable


@dataclass(frozen=True)
class Scope:
    issuer: str = "https://identity.fixture.invalid"
    subject: str = "user-a"
    tenant: str = "tenant-a"
    project: str = "project-a"

    @property
    def key(self) -> str:
        return json.dumps([self.issuer, self.subject, self.tenant, self.project])


class SimulatedCrash(BaseException):
    """An abrupt process loss, deliberately outside ordinary exception handlers."""


class SQLiteMindexFake:
    """Scoped durable fixture with atomic leases and immutable evidence receipts."""

    def __init__(self, path: Path, scope: Scope = Scope(), *, hooks: dict[str, Callable] | None = None):
        self.path = Path(path)
        self.scope = scope
        self.hooks = hooks or {}
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS clock (id INTEGER PRIMARY KEY, now REAL NOT NULL);
                INSERT OR IGNORE INTO clock VALUES (1, 1000.0);
                CREATE TABLE IF NOT EXISTS jobs (
                    scope TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                    state TEXT NOT NULL, token TEXT, epoch INTEGER NOT NULL DEFAULT 0,
                    expires REAL, worker TEXT, checkpoint TEXT, outputs TEXT,
                    error_code TEXT, PRIMARY KEY(scope, id));
                CREATE TABLE IF NOT EXISTS artifacts (
                    scope TEXT NOT NULL, id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    length INTEGER NOT NULL, raw BLOB NOT NULL, job_id TEXT,
                    role TEXT, idempotency_key TEXT,
                    PRIMARY KEY(scope, id), UNIQUE(scope, idempotency_key));
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
                    action TEXT NOT NULL, detail TEXT NOT NULL);
            """)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _hook(self, action: str, **detail):
        hook = self.hooks.get(action)
        return hook(self, detail) if hook else None

    def _event(self, db, action: str, **detail):
        db.execute("INSERT INTO events(scope,action,detail) VALUES (?,?,?)",
                   (self.scope.key, action, json.dumps(detail, sort_keys=True)))

    @staticmethod
    def _now(db) -> float:
        return db.execute("SELECT now FROM clock WHERE id=1").fetchone()[0]

    def advance(self, seconds: float):
        with self._connection() as db:
            db.execute("UPDATE clock SET now=now+? WHERE id=1", (seconds,))

    def _job(self, db, job_id: str):
        row = db.execute("SELECT * FROM jobs WHERE scope=? AND id=?", (self.scope.key, job_id)).fetchone()
        if row is None:
            raise PermissionError("Scoped job unavailable")
        return row

    def _guard(self, db, lease: Lease):
        row = self._job(db, lease.job["id"])
        if row["state"] == "cancelled":
            raise JobCancelled("Cancellation invalidated the job lease")
        if (row["state"] != "running" or row["token"] != lease.token
                or row["epoch"] != lease.epoch or row["expires"] <= self._now(db)):
            raise LeaseLost("Expired or superseded lease fence")
        return row

    def _lease(self, row) -> Lease:
        job = json.loads(row["payload"])
        job.update(id=row["id"], state=row["state"])
        if row["checkpoint"]:
            job["checkpoint"] = json.loads(row["checkpoint"])
        return Lease(job, row["token"], row["epoch"],
                     datetime.fromtimestamp(row["expires"], timezone.utc).isoformat())

    def seed_artifact(self, artifact_id: str, raw: bytes) -> ArtifactRef:
        reference = ArtifactRef(artifact_id, digest(raw), len(raw))
        with self._connection() as db:
            db.execute("INSERT INTO artifacts(scope,id,sha256,length,raw) VALUES (?,?,?,?,?)",
                       (self.scope.key, artifact_id, reference.sha256, len(raw), raw))
        return reference

    def seed_job(self, job_id: str, manifest_raw: bytes, captures: dict[str, bytes],
                 config: dict[str, Any]) -> dict[str, Any]:
        for artifact_id, raw in captures.items():
            with self._connection() as db:
                existing = db.execute("SELECT raw FROM artifacts WHERE scope=? AND id=?",
                                      (self.scope.key, artifact_id)).fetchone()
            if existing is None:
                self.seed_artifact(artifact_id, raw)
            elif existing["raw"] != raw:
                raise ContractError("Fixture attempted mutable capture replacement")
        reference = self.seed_artifact(str(uuid.uuid5(uuid.NAMESPACE_URL, "nlm-fixture-dataset:" + job_id)), manifest_raw)
        payload = {"engine_id": "thermal-softmax-v1", "engine_version": "1", "engine_pin": engine_pin(),
                   "seed": config["seed"], "parameters": {k: v for k, v in config.items() if k != "seed"},
                   "dataset": reference.to_dict()}
        with self._connection() as db:
            db.execute("INSERT INTO jobs(scope,id,payload,state) VALUES (?,?,?,'queued')",
                       (self.scope.key, job_id, json.dumps(payload)))
        return self.job(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = self._job(db, job_id)
            value = json.loads(row["payload"])
            value.update(id=row["id"], state=row["state"], lease_epoch=row["epoch"],
                         checkpoint=json.loads(row["checkpoint"]) if row["checkpoint"] else None,
                         outputs=json.loads(row["outputs"]) if row["outputs"] else None,
                         error_code=row["error_code"])
            return value

    def events(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            return [{"action": row["action"], **json.loads(row["detail"])} for row in
                    db.execute("SELECT * FROM events WHERE scope=? ORDER BY sequence", (self.scope.key,))]

    def corrupt_artifact(self, artifact_id: str, replacement: bytes):
        """Failure injection: alter bytes while retaining the original receipt."""
        with self._connection() as db:
            changed = db.execute("UPDATE artifacts SET raw=? WHERE scope=? AND id=?",
                                 (replacement, self.scope.key, artifact_id)).rowcount
            if changed != 1:
                raise PermissionError("Scoped artifact unavailable")

    def replace_job_payload(self, job_id: str, **fields):
        """Failure injection for malformed authoritative input, never public API."""
        with self._connection() as db:
            row = self._job(db, job_id)
            payload = json.loads(row["payload"])
            payload.update(fields)
            db.execute("UPDATE jobs SET payload=? WHERE scope=? AND id=?",
                       (json.dumps(payload), self.scope.key, job_id))

    def cancel(self, job_id: str):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._job(db, job_id)
            if row["state"] in ("succeeded", "failed", "cancelled"):
                return
            db.execute("UPDATE jobs SET state='cancelled',epoch=epoch+1,token=NULL WHERE scope=? AND id=?",
                       (self.scope.key, job_id))
            self._event(db, "cancel", job_id=job_id)

    async def claim(self, job_id: str, worker_id: str, lease_seconds: int) -> Lease | None:
        self._hook("before_claim", job_id=job_id)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._job(db, job_id)
            now = self._now(db)
            if row["state"] in ("succeeded", "failed", "cancelled") or (
                    row["state"] == "running" and row["expires"] > now):
                return None
            db.execute("UPDATE jobs SET state='running',token=?,epoch=epoch+1,expires=?,worker=? WHERE scope=? AND id=?",
                       (uuid.uuid4().hex, now + lease_seconds, worker_id, self.scope.key, job_id))
            lease = self._lease(self._job(db, job_id))
            self._event(db, "claim", job_id=job_id, epoch=lease.epoch)
        self._hook("after_claim", lease=lease)
        return lease

    async def heartbeat(self, lease: Lease, lease_seconds: int) -> Lease:
        self._hook("before_heartbeat", lease=lease)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._guard(db, lease)
            db.execute("UPDATE jobs SET expires=? WHERE scope=? AND id=?",
                       (self._now(db) + lease_seconds, self.scope.key, lease.job["id"]))
            renewed = self._lease(self._job(db, lease.job["id"]))
            self._event(db, "heartbeat", job_id=lease.job["id"], epoch=lease.epoch)
        self._hook("after_heartbeat", lease=renewed)
        return renewed

    def _artifact(self, db, reference: ArtifactRef):
        row = db.execute("SELECT * FROM artifacts WHERE scope=? AND id=?",
                         (self.scope.key, reference.artifact_id)).fetchone()
        if row is None:
            raise PermissionError("Scoped artifact unavailable")
        if row["sha256"] != reference.sha256 or row["length"] != reference.byte_length:
            raise RetentionUnavailable("Immutable artifact reference mismatch")
        return row

    def _verify_bytes(self, db, reference: ArtifactRef, *, role: str):
        row = self._artifact(db, reference)
        raw = bytes(row["raw"])
        if len(raw) != reference.byte_length or digest(raw) != reference.sha256:
            raise RetentionUnavailable("Server artifact readback failed")
        self._event(db, "server_read", artifact_id=reference.artifact_id, role=role)
        return raw

    async def read_artifact(self, reference: ArtifactRef) -> bytes:
        self._hook("before_read", reference=reference)
        with self._connection() as db:
            row = self._artifact(db, reference)
            raw = bytes(row["raw"])
            self._event(db, "read", artifact_id=reference.artifact_id, role=row["role"])
        self._hook("after_read", reference=reference, raw=raw)
        return raw

    async def put_artifact(self, job_id: str, role: str, raw: bytes, idempotency_key: str) -> ArtifactRef:
        self._hook("before_put", job_id=job_id, role=role, raw=raw)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._job(db, job_id)
            existing = db.execute("SELECT * FROM artifacts WHERE scope=? AND idempotency_key=?",
                                  (self.scope.key, idempotency_key)).fetchone()
            if existing is not None:
                if (existing["job_id"] != job_id or existing["role"] != role
                        or existing["sha256"] != digest(raw) or bytes(existing["raw"]) != raw):
                    raise ContractError("Artifact idempotency conflict")
                reference = ArtifactRef(existing["id"], existing["sha256"], existing["length"])
            else:
                reference = ArtifactRef(str(uuid.uuid5(uuid.NAMESPACE_URL, "nlm-fixture-artifact:" + idempotency_key)), digest(raw), len(raw))
                db.execute("INSERT INTO artifacts(scope,id,sha256,length,raw,job_id,role,idempotency_key) VALUES (?,?,?,?,?,?,?,?)",
                           (self.scope.key, reference.artifact_id, reference.sha256, len(raw), raw,
                            job_id, role, idempotency_key))
            self._event(db, "put", job_id=job_id, role=role, artifact_id=reference.artifact_id)
        self._hook("after_put", job_id=job_id, role=role, reference=reference)
        return reference

    async def checkpoint(self, lease: Lease, reference: ArtifactRef):
        self._hook("before_checkpoint", lease=lease, reference=reference)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._guard(db, lease)
            artifact = self._artifact(db, reference)
            if artifact["job_id"] != lease.job["id"] or artifact["role"] != "checkpoint":
                raise ContractError("Checkpoint belongs to another job or role")
            checkpoint = decode(self._verify_bytes(db, reference, role="checkpoint"))
            if checkpoint["manifest_sha256"] != json.loads(row["payload"])["dataset"]["sha256"]:
                raise ContractError("Checkpoint belongs to another dataset")
            db.execute("UPDATE jobs SET checkpoint=? WHERE scope=? AND id=?",
                       (json.dumps(reference.to_dict()), self.scope.key, lease.job["id"]))
            self._event(db, "checkpoint", job_id=lease.job["id"], epoch=lease.epoch,
                        completed_epochs=checkpoint["epoch"])
        self._hook("after_checkpoint", lease=lease, reference=reference)

    async def complete(self, lease: Lease, outputs: dict[str, ArtifactRef]) -> dict[str, Any]:
        self._hook("before_complete", lease=lease, outputs=outputs)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._guard(db, lease)
            if set(outputs) != {"model", "evaluation", "checkpoint"}:
                raise ContractError("All three retained outputs are required")
            decoded = {}
            for role, reference in outputs.items():
                artifact = self._artifact(db, reference)
                if artifact["job_id"] != lease.job["id"] or artifact["role"] != role:
                    raise ContractError("Output belongs to another job or role")
                decoded[role] = decode(self._verify_bytes(db, reference, role=role))
            payload = json.loads(row["payload"])
            if any(value["engine"] != payload["engine_pin"] for value in decoded.values()):
                raise ContractError("Output engine source/runtime pin mismatch")
            if any(value["manifest_sha256"] != payload["dataset"]["sha256"] for value in decoded.values()):
                raise ContractError("Output dataset evidence mismatch")
            if (json.loads(row["checkpoint"]) != outputs["checkpoint"].to_dict()
                    or decoded["model"]["epochs"] != payload["parameters"]["epochs"]
                    or decoded["checkpoint"]["epoch"] != payload["parameters"]["epochs"]
                    or decoded["evaluation"]["model_sha256"] != outputs["model"].sha256
                    or decoded["evaluation"]["checkpoint_sha256"] != outputs["checkpoint"].sha256):
                raise ContractError("Output completion evidence mismatch")
            serialized = {role: ref.to_dict() for role, ref in outputs.items()}
            db.execute("UPDATE jobs SET state='succeeded',outputs=? WHERE scope=? AND id=?",
                       (json.dumps(serialized), self.scope.key, lease.job["id"]))
            self._event(db, "complete", job_id=lease.job["id"], epoch=lease.epoch)
        self._hook("after_complete", lease=lease, outputs=outputs)
        return {"job": self.job(lease.job["id"]), "fixture_only": True}

    async def fail(self, lease: Lease, error_code: str):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._guard(db, lease)
            db.execute("UPDATE jobs SET state='failed',error_code=? WHERE scope=? AND id=?",
                       (error_code, self.scope.key, lease.job["id"]))
            self._event(db, "fail", job_id=lease.job["id"], error_code=error_code)
