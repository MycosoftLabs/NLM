"""Observed-target pairing scaffolding; no acquisition, synthetic labels or normalizer fitting.

Caller supplies permitted observations in an explicit channel/unit/normalizer contract.
Timestamps must be measurement times, never ingestion/last-seen fallbacks. Returned IDs
retain lineage. This module does not establish data rights, calibration or model skill.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PairingSpec:
    channels: tuple[str, ...]
    units: tuple[str, ...]
    normalizer_id: str
    horizon_seconds: int
    tolerance_seconds: int
    train_end: datetime
    validation_end: datetime
    test_end: datetime


@dataclass(frozen=True)
class ObservedFrame:
    frame_id: str
    series_id: str  # Same site/device/session; caller must not combine unrelated streams.
    measured_at: datetime
    values: Mapping[str, float]
    units: Mapping[str, str]
    normalizer_id: str


@dataclass(frozen=True)
class ObservedPair:
    input_id: str
    target_id: str
    series_id: str
    input_time: datetime
    target_time: datetime
    split: str
    target: tuple[float, ...]


@dataclass(frozen=True)
class PairingResult:
    pairs: tuple[ObservedPair, ...]
    rejected: tuple[tuple[str, str], ...]
    spec: PairingSpec


def _aware(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def build_observed_pairs(frames: Sequence[ObservedFrame], spec: PairingSpec) -> PairingResult:
    """Preflight whole cohort before pairing; nearest observed future target within tolerance.

    Ambiguous nearest targets and cross-split pairs are rejected, never relabeled.
    The current input is one frame; future windowed inputs need their own boundary purge.
    """
    if (type(spec.channels) is not tuple or type(spec.units) is not tuple
            or not spec.channels or len(set(spec.channels)) != len(spec.channels)
            or len(spec.units) != len(spec.channels)
            or any(not isinstance(s, str) or not s.strip() for s in (*spec.channels, *spec.units, spec.normalizer_id))):
        raise ValueError("Explicit unique channels, units and normalizer required")
    if (type(spec.horizon_seconds) is not int or type(spec.tolerance_seconds) is not int
            or not 0 <= spec.tolerance_seconds < spec.horizon_seconds):
        raise ValueError("Require positive horizon and 0 <= tolerance < horizon")
    boundaries = (spec.train_end, spec.validation_end, spec.test_end)
    if not all(_aware(t) for t in boundaries) or not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError("Ordered timezone-aware split boundaries required")
    by_series: dict[str, list[ObservedFrame]] = {}
    seen_ids: set[str] = set()
    seen_times: set[tuple[str, datetime]] = set()
    for row in frames:
        if not row.frame_id or not row.series_id or row.frame_id in seen_ids or not _aware(row.measured_at):
            raise ValueError("Unique frame IDs, series IDs and aware measurement times required")
        key = (row.series_id, row.measured_at)
        if key in seen_times:
            raise ValueError("Duplicate measurement time in series")
        if row.normalizer_id != spec.normalizer_id:
            raise ValueError("Normalizer mismatch")
        for channel, unit in zip(spec.channels, spec.units):
            value = row.values.get(channel)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3.4028234663852886e38:
                raise ValueError(f"Missing/nonfinite observed channel: {channel}")
            if row.units.get(channel) != unit:
                raise ValueError(f"Unit mismatch: {channel}")
        seen_ids.add(row.frame_id); seen_times.add(key)
        by_series.setdefault(row.series_id, []).append(row)

    def split(time: datetime) -> str | None:
        for boundary, name in zip(boundaries, ("train", "validation", "test")):
            if time < boundary:
                return name
        return None

    pairs: list[ObservedPair] = []
    rejected: list[tuple[str, str]] = []
    horizon = timedelta(seconds=spec.horizon_seconds)
    tolerance = timedelta(seconds=spec.tolerance_seconds)
    for series in sorted(by_series):
        rows = sorted(by_series[series], key=lambda f: f.measured_at)
        times = [f.measured_at for f in rows]
        for row in rows:
            desired = row.measured_at + horizon
            candidates = rows[bisect_left(times, desired-tolerance):bisect_right(times, desired+tolerance)]
            if not candidates:
                rejected.append((row.frame_id, "no_observed_target")); continue
            distance = min(abs(c.measured_at-desired) for c in candidates)
            nearest = [c for c in candidates if abs(c.measured_at-desired) == distance]
            if len(nearest) != 1:
                rejected.append((row.frame_id, "ambiguous_target")); continue
            target = nearest[0]
            partition = split(row.measured_at)
            if partition is None or partition != split(target.measured_at):
                rejected.append((row.frame_id, "split_boundary")); continue
            pairs.append(ObservedPair(row.frame_id, target.frame_id, series, row.measured_at,
                                      target.measured_at, partition,
                                      tuple(float(target.values[c]) for c in spec.channels)))
    return PairingResult(tuple(pairs), tuple(rejected), spec)


class ObservedPairDataset:
    """Attach validated observed targets to an existing input dataset by exact frame ID.

    Adapter keeps training inputs separate from future observations. No target frame is
    ever preconditioned as the input. The caller must supply matching input lineage and
    a train-only-fitted normalizer. RootedFrameDataset's other modality limits remain.
    """
    def __init__(self, inputs, input_frame_ids: Sequence[str], result: PairingResult,
                 split: str, num_env_targets: int):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if len(inputs) != len(input_frame_ids) or len(set(input_frame_ids)) != len(input_frame_ids):
            raise ValueError("Input IDs must be unique and match dataset length")
        if len(result.spec.channels) != num_env_targets:
            raise ValueError("Model output width differs from observed channel contract")
        self.inputs = inputs
        self.indices = {key: i for i, key in enumerate(input_frame_ids)}
        self.pairs = tuple(p for p in result.pairs if p.split == split)
        if any(p.input_id not in self.indices for p in self.pairs):
            raise ValueError("Missing paired input frame")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        import torch
        pair = self.pairs[index]
        sample = dict(self.inputs[self.indices[pair.input_id]])
        if "next_state_target" in sample:
            raise ValueError("Input dataset already contains targets")
        target = torch.tensor(pair.target, dtype=torch.float32)
        if not torch.isfinite(target).all().item():
            raise ValueError("Observed target overflows model dtype")
        sample["next_state_target"] = target
        return sample
