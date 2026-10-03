"""Bounded real HTTP transport for MINDEX compute and shared09 private retention.

No redirects, automatic credential fallback, external artifact URLs, or credentials
in error messages. Every request forwards the original user JWT and scope selectors;
worker operations also need a separately reviewed server-side worker policy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

from .contracts import MAX_BYTES, ContractError, decode, digest, encode
from .worker import ArtifactRef, JobCancelled, Lease, LeaseLost, RetentionUnavailable


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpFailure(RetentionUnavailable):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code
        super().__init__(f"MINDEX HTTP {status}: {code}")


class MindexHttpPort:
    """Use base_url like https://mindex.example/api/mindex.

    token_provider must return an original user JWT; renewing it must preserve the
    authenticated issuer/subject/scope. No service key is accepted as user identity.
    Additional worker headers are opaque to this client and never override identity.
    """

    def __init__(self, base_url: str, token_provider: Callable[[], str], tenant_id: str, project_id: str,
                 *, worker_headers: Callable[[], Mapping[str, str]] | None = None,
                 timeout: float = 10, archive_wait_seconds: float = 30):
        parsed = urlsplit(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
            raise ContractError("Invalid MINDEX service URL")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ContractError("MINDEX transport requires HTTPS except explicit loopback fixtures")
        if not 0 < timeout <= 30 or not 0 <= archive_wait_seconds <= 60:
            raise ContractError("Transport timing budget exceeded")
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.tenant_id, self.project_id = str(UUID(tenant_id)), str(UUID(project_id))
        self.worker_headers = worker_headers
        self.timeout, self.archive_wait_seconds = timeout, archive_wait_seconds
        self._opener = build_opener(_NoRedirect())

    def _headers(self, extra=None, *, worker=False):
        token = self.token_provider()
        if not isinstance(token, str) or not token.strip() or any(c in token for c in "\r\n"):
            raise PermissionError("Original user authorization unavailable")
        headers = {"Authorization": f"Bearer {token}", "X-Tenant-Id": self.tenant_id,
                   "X-Project-Id": self.project_id, "Accept": "application/json"}
        if worker and self.worker_headers:
            additional = dict(self.worker_headers())
            allowed = {"x-nlm-worker-authorization", "x-nlm-worker-id"}
            if any(key.lower() not in allowed or not isinstance(value, str) or any(c in value for c in "\r\n")
                   for key, value in additional.items()):
                raise ContractError("Worker headers cannot override original identity or transport headers")
            headers.update(additional)
        headers.update(extra or {})
        return headers

    def _sync_request(self, method, path, raw, headers):
        request = Request(self.base_url + path, data=raw, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = response.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    raise RetentionUnavailable("MINDEX response exceeds byte budget")
                return payload, dict(response.headers)
        except HTTPError as exc:
            # Ignore upstream text that might contain credentials or source data.
            code = "request_failed"
            try:
                obj = decode(exc.read(16385))
                detail = obj.get("detail")
                candidate = detail.get("code") if isinstance(detail, dict) else obj.get("error")
                if isinstance(candidate, str) and len(candidate) <= 80 and all(c.islower() or c == "_" for c in candidate):
                    code = candidate
            except (ValueError, OSError):
                pass
            if exc.code in (401, 403):
                raise PermissionError(f"MINDEX authorization rejected ({exc.code})") from None
            if code in {"cancelled", "job_cancelled"}:
                raise JobCancelled(code) from None
            if code == "stale_or_expired_lease":
                raise LeaseLost(code) from None
            raise HttpFailure(exc.code, code) from None
        except (URLError, TimeoutError, OSError):
            raise RetentionUnavailable("MINDEX transport unavailable") from None

    async def _request(self, method, path, raw=None, *, extra=None, worker=False):
        return await asyncio.to_thread(self._sync_request, method, path, raw, self._headers(extra, worker=worker))

    async def _json(self, method, path, body=None, *, worker=False):
        raw, _ = await self._request(method, path, encode(body) if body is not None else None,
                                     extra={"Content-Type": "application/json"}, worker=worker)
        return decode(raw)

    @staticmethod
    def _lease(payload: dict[str, Any]) -> Lease:
        if payload.get("contract_version") != "nlm-training.v1":
            raise ContractError("Unsupported compute receipt contract")
        lease = payload.get("lease")
        if not isinstance(lease, dict) or set(lease) != {"job", "token", "epoch", "expires_at"}:
            raise ContractError("Malformed worker lease")
        if not isinstance(lease["job"], dict) or not isinstance(lease["token"], str) or len(lease["token"]) != 64 or any(c not in "0123456789abcdef" for c in lease["token"]) or type(lease["epoch"]) is not int or lease["epoch"] < 1 or not isinstance(lease["expires_at"], str):
            raise ContractError("Malformed worker lease fields")
        return Lease(**lease)

    @staticmethod
    def _job_path(job_id):
        return "/nlm-training/v1/worker/jobs/" + str(UUID(job_id))

    @staticmethod
    def _fence(lease):
        return {"token": lease.token, "epoch": lease.epoch}

    async def claim(self, job_id, worker_id, lease_seconds):
        try:
            return self._lease(await self._json("POST", self._job_path(job_id) + "/claim",
                                               {"worker_id": worker_id, "lease_seconds": lease_seconds}, worker=True))
        except HttpFailure as exc:
            if exc.status == 409 and exc.code == "job_not_claimable":
                return None
            raise

    async def heartbeat(self, lease, lease_seconds):
        current = self._lease(await self._json("POST", self._job_path(lease.job["id"]) + "/heartbeat",
                                              {**self._fence(lease), "lease_seconds": lease_seconds}, worker=True))
        if current.token != lease.token or current.epoch != lease.epoch or current.job.get("id") != lease.job["id"]:
            raise LeaseLost("Heartbeat returned a different job or fence")
        return current

    async def read_artifact(self, reference):
        path = "/retention/v1/artifacts/" + str(UUID(reference.artifact_id)) + "/content"
        raw, headers = await self._request("GET", path)
        headers = {key.lower(): value for key, value in headers.items()}
        if headers.get("x-retention-contract") != "retention.v1" or headers.get("x-artifact-id") != reference.artifact_id or headers.get("x-artifact-sha256") != reference.sha256:
            raise RetentionUnavailable("Retained content response identity mismatch")
        return raw

    async def put_artifact(self, job_id, role, raw, idempotency_key):
        if role not in {"model", "evaluation", "checkpoint"} or not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
            raise ContractError("Invalid output artifact")
        payload, _ = await self._request("POST", "/retention/v1/artifacts", raw,
            extra={"X-Artifact-Kind": "artifact", "Content-Type": "application/json",
                   "Idempotency-Key": "nlm-" + digest(idempotency_key.encode("utf-8"))})
        receipt = decode(payload)
        reference = ArtifactRef.from_dict({key: receipt.get(key) for key in ("artifact_id", "sha256", "byte_length")})
        if reference.sha256 != digest(raw) or reference.byte_length != len(raw):
            raise RetentionUnavailable("Retention admission receipt does not match artifact bytes")
        deadline = time.monotonic() + self.archive_wait_seconds
        while True:
            if receipt.get("contract_version") != "retention.v1" or receipt.get("tenant_id") != self.tenant_id or receipt.get("project_id") != self.project_id:
                raise RetentionUnavailable("Retention receipt scope/contract mismatch")
            if any(receipt.get(key) != value for key, value in reference.to_dict().items()):
                raise RetentionUnavailable("Retention receipt changed immutable identity")
            if receipt.get("archive_verified") is True and receipt.get("state") in {"verified", "archived_verified"}:
                return reference
            if receipt.get("state") in {"failed", "cancelled", "deleted", "integrity_blocked"}:
                raise RetentionUnavailable("Retention cannot verify artifact")
            if time.monotonic() >= deadline:
                raise RetentionUnavailable("Artifact admitted but verified archive receipt unavailable")
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
            receipt = await self._json("GET", "/retention/v1/artifacts/" + str(UUID(reference.artifact_id)))

    async def checkpoint(self, lease, reference):
        await self._json("POST", self._job_path(lease.job["id"]) + "/checkpoint",
                         {**self._fence(lease), "artifact": reference.to_dict()}, worker=True)

    async def complete(self, lease, outputs):
        return await self._json("POST", self._job_path(lease.job["id"]) + "/complete",
                               {**self._fence(lease), "outputs": {key: value.to_dict() for key, value in outputs.items()}}, worker=True)

    async def fail(self, lease, error_code):
        await self._json("POST", self._job_path(lease.job["id"]) + "/fail",
                         {**self._fence(lease), "error_code": error_code}, worker=True)
