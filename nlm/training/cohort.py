"""Bounded, offline cohort contract layered on the observed pairing scaffold.

No acquisition, permissions service, normalizer fitting, training or serving occurs here.
Approval references are operator attestations, not independently verified data rights.
Only single-frame, forward-time evaluation is supported; no site-disjoint claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from .pairing import ObservedFrame, PairingResult, PairingSpec, build_observed_pairs

COHORT_VERSION = "nlm.offline-cohort.v1"
PURPOSE = "offline_forecast_evaluation"
MAX_FRAMES = 10_000  # Local scaffold bound, not a measured capacity or commercial allowance.
FLOAT32_MAX = 3.4028234663852886e38


@dataclass(frozen=True)
class RawObservation:
    frame_id: str
    series_id: str
    measured_at: datetime
    source_record_ref: str
    values: Mapping[str, float]
    units: Mapping[str, str]


@dataclass(frozen=True)
class CohortAuthorization:
    version: str
    cohort_id: str
    approved_by: str
    evidence_ref: str
    approved_at: datetime
    expires_at: datetime
    purpose: str
    cohort_sha256: str
    frame_ids: tuple[str, ...]
    series_ids: tuple[str, ...]
    task_id: str


@dataclass(frozen=True)
class TargetContract:
    task_id: str
    channels: tuple[str, ...]
    raw_units: tuple[str, ...]
    model_units: tuple[str, ...]
    horizon_seconds: int
    tolerance_seconds: int


@dataclass(frozen=True)
class SplitContract:
    train_start: datetime
    train_end: datetime
    validation_end: datetime
    test_end: datetime
    policy: str = "forward-time-single-frame-v1"


@dataclass(frozen=True)
class FrozenNormalizer:
    """Externally approved (value - offset) / scale, with explicit train-only lineage.

    Coefficients are supplied, never inferred or fitted here. Lineage validation cannot
    prove that a producer actually fitted these coefficients from the declared rows.
    """
    normalizer_id: str
    approved_by: str
    evidence_ref: str
    channels: tuple[str, ...]
    raw_units: tuple[str, ...]
    model_units: tuple[str, ...]
    offsets: tuple[float, ...]
    scales: tuple[float, ...]
    fit_frame_ids: tuple[str, ...]
    fit_sha256: str
    method: str = "affine-v1"


@dataclass(frozen=True)
class PreparedCohort:
    pairs: PairingResult
    cohort_id: str
    cohort_sha256: str
    contract_sha256: str
    fit_sha256: str
    split_frame_ids: tuple[tuple[str, tuple[str, ...]], ...]


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _iso(value: datetime) -> str:
    if not _aware(value):
        raise ValueError("Timezone-aware measurement/approval/split timestamps required")
    return value.astimezone(timezone.utc).isoformat()


def _strings(value: object, *, unique: bool = True) -> bool:
    return (type(value) is tuple and bool(value) and all(_text(v) for v in value)
            and (not unique or len(set(value)) == len(value)))


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and abs(value) <= FLOAT32_MAX
    except OverflowError:
        return False


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def cohort_digest(rows: Sequence[RawObservation]) -> str:
    """Order-independent binding of the exact bounded raw cohort, including source refs.

    This identifies provided bytes/values; it does not authenticate the measurement source.
    """
    if not 0 < len(rows) <= MAX_FRAMES:
        raise ValueError("Require a bounded nonempty cohort")
    seen: set[str] = set()
    payload = []
    for row in rows:
        if not all(_text(x) for x in (row.frame_id, row.series_id, row.source_record_ref)) or row.frame_id in seen:
            raise ValueError("Unique frame IDs, series identity and source record reference required")
        if not row.values or set(row.values) != set(row.units) or any(not _text(k) for k in row.values):
            raise ValueError("Explicit channel values and matching units required")
        if any(not _finite(v) for v in row.values.values()) or any(not _text(u) for u in row.units.values()):
            raise ValueError("Finite float32-compatible measurements and units required")
        seen.add(row.frame_id)
        payload.append({"id": row.frame_id, "series": row.series_id, "time": _iso(row.measured_at),
                        "source": row.source_record_ref, "values": dict(row.values), "units": dict(row.units)})
    return _digest(sorted(payload, key=lambda item: item["id"]))


def prepare_cohort(rows: Sequence[RawObservation], authorization: CohortAuthorization,
                   target: TargetContract, normalizer: FrozenNormalizer, split: SplitContract,
                   *, now: datetime) -> PreparedCohort:
    """Validate the complete frozen contract before returning any normalized target pairs.

    Caller supplies trusted current UTC time and independently approved local contracts.
    Changed raw data, target semantics, coefficients or split boundaries change the manifest.
    No file/network reads, input tensor preprocessing or model mutations are performed.
    """
    digest = cohort_digest(rows)
    a, t, n, s = authorization, target, normalizer, split
    if (a.version != COHORT_VERSION or a.purpose != PURPOSE
            or not all(_text(x) for x in (a.cohort_id, a.approved_by, a.evidence_ref, a.task_id))
            or not _aware(now) or not _aware(a.approved_at) or not _aware(a.expires_at)
            or not a.approved_at <= now < a.expires_at):
        raise ValueError("Current explicit offline cohort approval required")
    if (not _sha(a.cohort_sha256) or a.cohort_sha256 != digest
            or not _strings(a.frame_ids) or set(a.frame_ids) != {r.frame_id for r in rows}
            or not _strings(a.series_ids) or set(a.series_ids) != {r.series_id for r in rows}):
        raise ValueError("Cohort differs from authorized records/series/digest")
    if (t.task_id != a.task_id or not _strings(t.channels) or not _strings(t.raw_units, unique=False)
            or not _strings(t.model_units, unique=False)
            or len(t.channels) != len(t.raw_units) or len(t.channels) != len(t.model_units)
            or type(t.horizon_seconds) is not int or type(t.tolerance_seconds) is not int
            or not 0 <= t.tolerance_seconds < t.horizon_seconds <= 365 * 86400):
        raise ValueError("Explicit task/channel/unit/horizon contract required")
    boundaries = (s.train_start, s.train_end, s.validation_end, s.test_end)
    if (s.policy != "forward-time-single-frame-v1" or not all(_aware(v) for v in boundaries)
            or not all(x < y for x, y in zip(boundaries, boundaries[1:]))):
        raise ValueError("Ordered bounded single-frame time splits required")
    if any(not s.train_start <= r.measured_at < s.test_end for r in rows):
        raise ValueError("Cohort row outside the frozen split interval")
    for row in rows:
        if set(row.values) != set(t.channels) or tuple(row.units[c] for c in t.channels) != t.raw_units:
            raise ValueError("Raw channel/unit contract mismatch; no implicit conversion")
    if (n.method != "affine-v1" or not all(_text(v) for v in (n.normalizer_id, n.approved_by, n.evidence_ref))
            or n.channels != t.channels or n.raw_units != t.raw_units or n.model_units != t.model_units
            or type(n.offsets) is not tuple or type(n.scales) is not tuple
            or len(n.offsets) != len(t.channels) or len(n.scales) != len(t.channels)
            or any(not _finite(v) for v in (*n.offsets, *n.scales)) or any(v <= 0 for v in n.scales)):
        raise ValueError("Approved frozen affine normalizer with matching ordered channels required")
    train = {r.frame_id: r for r in rows if r.measured_at < s.train_end}
    if not _strings(n.fit_frame_ids) or not set(n.fit_frame_ids) <= set(train):
        raise ValueError("Normalizer fit must use only authorized training records")
    fit_digest = cohort_digest([train[key] for key in n.fit_frame_ids])
    if not _sha(n.fit_sha256) or n.fit_sha256 != fit_digest:
        raise ValueError("Normalizer fit record digest mismatch")
    normalized = []
    for row in rows:
        values = {c: (row.values[c] - offset) / scale
                  for c, offset, scale in zip(t.channels, n.offsets, n.scales)}
        if any(not _finite(v) for v in values.values()):
            raise ValueError("Normalized measurement exceeds float32 range")
        normalized.append(ObservedFrame(row.frame_id, row.series_id, row.measured_at, values,
                                        dict(zip(t.channels, t.model_units)), n.normalizer_id))
    spec = PairingSpec(t.channels, t.model_units, n.normalizer_id, t.horizon_seconds,
                       t.tolerance_seconds, s.train_end, s.validation_end, s.test_end)
    result = build_observed_pairs(normalized, spec)
    split_ids = []
    used: set[str] = set()
    for name in ("train", "validation", "test"):
        ids = {v for p in result.pairs if p.split == name for v in (p.input_id, p.target_id)}
        if not ids:
            raise ValueError(f"No observed pairs in required {name} split")
        if used & ids:
            raise ValueError("Observation reused across splits")
        used.update(ids)
        split_ids.append((name, tuple(sorted(ids))))
    manifest = {
        "version": COHORT_VERSION, "cohort_id": a.cohort_id, "cohort_sha256": digest,
        "approval": {"by": a.approved_by, "evidence": a.evidence_ref, "at": _iso(a.approved_at),
                     "expires": _iso(a.expires_at), "purpose": a.purpose},
        "task": {"id": t.task_id, "channels": t.channels, "raw_units": t.raw_units,
                 "model_units": t.model_units, "horizon": t.horizon_seconds, "tolerance": t.tolerance_seconds},
        "normalizer": {"id": n.normalizer_id, "method": n.method, "by": n.approved_by,
                       "evidence": n.evidence_ref, "offsets": n.offsets, "scales": n.scales,
                       "fit_sha256": fit_digest},
        "split": {"policy": s.policy, "boundaries": [_iso(v) for v in boundaries]},
    }
    return PreparedCohort(result, a.cohort_id, digest, _digest(manifest), fit_digest, tuple(split_ids))
