"""Synthetic contract fixtures only: no approved real cohort or quality measurement."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest

from nlm.training.cohort import (
    COHORT_VERSION, PURPOSE, MAX_FRAMES, RawObservation, CohortAuthorization,
    TargetContract, SplitContract, FrozenNormalizer, cohort_digest, prepare_cohort,
)
from nlm.training.pairing import ObservedPairDataset

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fixture():
    rows = [RawObservation(f"a-{i}", "site/device/session-a", T + timedelta(seconds=i * 60),
                           f"fixture://measurement/{i}", {"temperature": 10. + i, "humidity": 50.},
                           {"temperature": "degC", "humidity": "percent"}) for i in range(12)]
    a = CohortAuthorization(COHORT_VERSION, "synthetic-only", "fixture-reviewer", "fixture://approval",
                            T - timedelta(days=1), T + timedelta(days=1), PURPOSE, cohort_digest(rows),
                            tuple(r.frame_id for r in rows), (rows[0].series_id,), "fixture-forecast")
    target = TargetContract(a.task_id, ("temperature", "humidity"), ("degC", "percent"),
                            ("standardized", "standardized"), 60, 0)
    split = SplitContract(T, T + timedelta(seconds=240), T + timedelta(seconds=480), T + timedelta(seconds=720))
    train = rows[:4]
    n = FrozenNormalizer("fixture-affine-v1", "fixture-reviewer", "fixture://normalizer",
                         target.channels, target.raw_units, target.model_units, (10., 50.), (2., 5.),
                         tuple(r.frame_id for r in train), cohort_digest(train))
    return rows, a, target, n, split


def prepare(parts):
    return prepare_cohort(*parts, now=T)


def reauthorize_rows(parts, rows):
    _, a, t, n, s = parts
    a = replace(a, cohort_sha256=cohort_digest(rows), frame_ids=tuple(r.frame_id for r in rows),
                series_ids=tuple(sorted({r.series_id for r in rows})))
    return rows, a, t, n, s


def test_observed_future_targets_normalization_and_split_lineage():
    parts = fixture(); result = prepare(parts)
    assert result.pairs.pairs[0].input_id == "a-0"
    assert result.pairs.pairs[0].target_id == "a-1"
    assert result.pairs.pairs[0].target == (.5, 0.)  # Observed future value; no fallback label.
    assert ("a-3", "split_boundary") in result.pairs.rejected
    assert ("a-7", "split_boundary") in result.pairs.rejected
    assert ("a-11", "no_observed_target") in result.pairs.rejected
    ids = dict(result.split_frame_ids)
    assert not set(ids["train"]) & set(ids["validation"])
    assert not set(ids["validation"]) & set(ids["test"])
    # An input fixture remains the current observation, not the normalized future target.
    inputs = [{"input_identity": r.frame_id} for r in parts[0]]
    ds = ObservedPairDataset(inputs, [r.frame_id for r in parts[0]], result.pairs, "train", 2)
    assert ds[0]["input_identity"] == "a-0"
    assert ds[0]["next_state_target"].tolist() == [.5, 0.]
    assert "next_state_target" not in inputs[0]


def test_manifest_is_order_independent_but_binds_contract_semantics():
    rows, a, t, n, s = fixture(); base = prepare((rows, a, t, n, s))
    assert prepare((list(reversed(rows)), a, t, n, s)) == base
    for target, norm, split in [(replace(t, tolerance_seconds=1), n, s),
                                (t, replace(n, offsets=(9., 50.)), s),
                                (t, n, replace(s, train_end=T + timedelta(seconds=241)))]:
        changed = prepare((rows, a, target, norm, split))
        assert changed.contract_sha256 != base.contract_sha256
        assert changed.cohort_sha256 == base.cohort_sha256


@pytest.mark.parametrize("changes", [
    {"version": "v2"}, {"purpose": "paid_inference"}, {"approved_by": ""}, {"evidence_ref": " "},
    {"approved_at": T + timedelta(seconds=1)}, {"expires_at": T},
    {"approved_at": T.replace(tzinfo=None)}, {"cohort_sha256": "0" * 64},
    {"frame_ids": ("a-0",)}, {"series_ids": ("other",)}, {"task_id": "other"},
    {"frame_ids": [f"a-{i}" for i in range(12)]},
])
def test_rejects_missing_expired_or_mismatched_authorization(changes):
    rows, a, t, n, s = fixture()
    with pytest.raises(ValueError): prepare((rows, replace(a, **changes), t, n, s))


@pytest.mark.parametrize("changes", [
    {"values": {"temperature": 99., "humidity": 50.}},
    {"source_record_ref": "fixture://different-record"}, {"series_id": "another-session"},
    {"measured_at": T + timedelta(seconds=1)},
])
def test_approval_binds_actual_values_source_identity_and_measurement_time(changes):
    rows, *rest = fixture(); rows[0] = replace(rows[0], **changes)
    with pytest.raises(ValueError, match="authorized"): prepare((rows, *rest))


@pytest.mark.parametrize("changes", [
    {"channels": ("humidity", "temperature")}, {"raw_units": ("kelvin", "percent")},
    {"model_units": ("unknown", "standardized")}, {"horizon_seconds": True},
    {"horizon_seconds": 0}, {"tolerance_seconds": 60}, {"tolerance_seconds": -1},
    {"channels": ("temperature", "temperature")}, {"horizon_seconds": 10**100},
])
def test_task_contract_does_not_guess_channels_units_or_horizon(changes):
    rows, a, t, n, s = fixture()
    with pytest.raises(ValueError): prepare((rows, a, replace(t, **changes), n, s))


@pytest.mark.parametrize("changes", [
    {"normalizer_id": ""}, {"approved_by": ""}, {"evidence_ref": ""}, {"method": "fit-on-all-data"},
    {"fit_frame_ids": ("a-4",)}, {"fit_frame_ids": ("a-8",)}, {"fit_frame_ids": ("foreign",)},
    {"fit_frame_ids": ("a-0", "a-0")}, {"fit_frame_ids": ()}, {"fit_sha256": "0" * 64},
    {"scales": (0., 1.)}, {"scales": (-1., 1.)}, {"scales": (True, 1.)},
    {"scales": (1e-320, 1.)}, {"offsets": (float("nan"), 0.)}, {"offsets": (float("inf"), 0.)},
    {"scales": (1.,)}, {"offsets": (10**1000, 0.)},
])
def test_normalizer_rejects_leakage_unbound_provenance_and_invalid_transform(changes):
    rows, a, t, n, s = fixture()
    with pytest.raises(ValueError): prepare((rows, a, t, replace(n, **changes), s))


def test_digest_cannot_launder_holdout_as_a_fit_record():
    rows, a, t, n, s = fixture()
    n = replace(n, fit_frame_ids=(rows[4].frame_id,), fit_sha256=cohort_digest([rows[4]]))
    with pytest.raises(ValueError, match="training records"): prepare((rows, a, t, n, s))


@pytest.mark.parametrize("changes", [
    {"train_start": T + timedelta(seconds=1)}, {"train_end": T},
    {"test_end": T + timedelta(seconds=600)}, {"validation_end": T.replace(tzinfo=None)},
    {"policy": "random"}, {"policy": "site-disjoint"},
])
def test_frozen_split_interval_is_enforced(changes):
    rows, a, t, n, s = fixture()
    with pytest.raises(ValueError): prepare((rows, a, t, n, replace(s, **changes)))


def test_no_completed_prep_with_an_empty_holdout():
    parts = fixture(); rows = parts[0][:8]
    with pytest.raises(ValueError, match="test split"): prepare(reauthorize_rows(parts, rows))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "12", 10**1000])
def test_bad_raw_values_are_rejected_before_hashing_or_normalization(value):
    rows, *rest = fixture(); rows[0] = replace(rows[0], values={"temperature": value, "humidity": 50.})
    with pytest.raises(ValueError): prepare((rows, *rest))


def test_duplicate_ids_timestamps_missing_units_and_bounds():
    parts = fixture(); rows = parts[0]
    with pytest.raises(ValueError): cohort_digest([])
    with pytest.raises(ValueError): cohort_digest([rows[0]] * (MAX_FRAMES + 1))
    with pytest.raises(ValueError): cohort_digest([rows[0], rows[0]])
    with pytest.raises(ValueError): cohort_digest([replace(rows[0], units={})])
    with pytest.raises(ValueError): cohort_digest([replace(rows[0], measured_at=T.replace(tzinfo=None))])
    rows[5] = replace(rows[5], measured_at=rows[4].measured_at)
    with pytest.raises(ValueError, match="Duplicate measurement"): prepare(reauthorize_rows(parts, rows))


def test_wrong_current_time_is_rejected_and_no_source_payload_in_errors():
    with pytest.raises(ValueError): prepare_cohort(*fixture(), now=T.replace(tzinfo=None))
    rows, *rest = fixture(); rows[0] = replace(rows[0], values={"sensitive-source-field": 2.})
    with pytest.raises(ValueError) as error: prepare((rows, *rest))
    assert "sensitive-source-field" not in str(error.value)
