"""Strict thermal feature contract. Ownership is supplied by verified MINDEX policy.

The manifest identifier hashes the exact archived UTF-8 bytes, not a cross-language
JSON reserialization. JSON numbers use finite IEEE-754 values; raw bytes are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

ENGINE_ID = "thermal-softmax-v1"
MANIFEST_SCHEMA = "mycosoft.nlm.training-manifest.v1"
DATASET_SCHEMA = "mycosoft.nlm.modality-dataset.v1"
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 2000
FEATURE_SCHEMA_BYTES = b'{"channels":["temperature","gradient"],"id":"supplied-features","modality":"thermal","units":["degC","degC_per_m"],"version":"1"}'
FEATURE_SCHEMA_SHA256 = hashlib.sha256(FEATURE_SCHEMA_BYTES).hexdigest()
HASH = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Malformed, incompatible or insufficient training evidence."""


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode(value: Any) -> bytes:
    """Deterministic engine-owned JSON; not advertised as RFC8785/JCS."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BYTES:
        raise ContractError("JSON evidence must contain 1..8388608 UTF-8 bytes")

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ContractError("Duplicate JSON property")
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise ContractError("Non-finite JSON number: " + value)

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ContractError("Invalid JSON evidence") from exc
    if not isinstance(value, dict):
        raise ContractError("JSON object required")

    def check(item, depth=0):
        if depth > 16:
            raise ContractError("Evidence nesting exceeds 16")
        if isinstance(item, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in item):
            raise ContractError("Lone Unicode surrogate")
        if isinstance(item, float) and not math.isfinite(item):
            raise ContractError("Non-finite JSON number")
        if isinstance(item, dict):
            for key, val in item.items():
                check(key, depth + 1)
                check(val, depth + 1)
        if isinstance(item, list):
            for val in item:
                check(val, depth + 1)
    check(value)
    return value


def fields(obj: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(obj, dict) or not required <= obj.keys() or obj.keys() - required - (optional or set()):
        raise ContractError("Missing or unknown contract properties")


def text(value: Any, limit: int = 512, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ContractError("Expected bounded text")
    return value


def finite(value: Any, bound: float = 1e6) -> float:
    if type(value) not in (int, float) or abs(value) > bound or not math.isfinite(value):
        raise ContractError("Expected finite numeric value within engine bounds")
    return float(value)


@dataclass(frozen=True)
class Row:
    id: str
    group: str
    label: str | None
    split: str
    values: tuple[float | None, ...]


@dataclass(frozen=True)
class Capture:
    artifact_id: str
    sha256: str
    byte_length: int
    media_type: str
    license: str
    calibration: str
    source_class: str
    row_ids: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    raw: bytes
    sha256: str
    rows: tuple[Row, ...]
    masks: tuple[tuple[bool, ...], ...]
    captures: tuple[Capture, ...]
    source: str

    def require_training_ready(self) -> None:
        if any(row.label is None or any(v is None for v in row.values) for row in self.rows):
            raise ContractError("Labels and observed values required; missing is not zero")
        train = [row for row in self.rows if row.split == "train"]
        test = [row for row in self.rows if row.split == "test"]
        classes = sorted({row.label for row in train})
        if not 2 <= len(classes) <= 32:
            raise ContractError("Need 2..32 training classes")
        if any(sum(row.label == label for row in train) < 2 for label in classes):
            raise ContractError("Every class needs at least two training rows")
        if set(classes) != {row.label for row in test}:
            raise ContractError("Holdout must cover exactly the training labels")


def validate_manifest(raw: bytes) -> Manifest:
    obj = decode(raw)
    fields(obj, {"schema", "dataset", "masks", "captures", "extractor", "timestamp_semantics"})
    if obj["schema"] != MANIFEST_SCHEMA:
        raise ContractError("Unsupported training manifest schema")
    data = obj["dataset"]
    fields(data, {"schema", "modality", "channels", "units", "source", "notes", "rows"}, {"importSources"})
    if data["schema"] != DATASET_SCHEMA or data["modality"] != "thermal":
        raise ContractError("Only the thermal feature engine is bound")
    if data["channels"] != ["temperature", "gradient"] or data["units"] != ["degC", "degC_per_m"]:
        raise ContractError("Thermal channels/units must match pinned schema exactly")
    source = text(data["source"])
    text(data["notes"], 10000, empty=True)
    if "importSources" in data:
        if not isinstance(data["importSources"], list) or len(data["importSources"]) > MAX_ROWS:
            raise ContractError("Invalid import provenance")
        for item in data["importSources"]:
            fields(item, {"reference", "notes"})
            text(item["reference"])
            text(item["notes"], 10000, empty=True)
    text(obj["timestamp_semantics"], 2000)
    fields(obj["extractor"], {"id", "version", "sha256"})
    if obj["extractor"] != {"id": "supplied-features", "version": "1", "sha256": FEATURE_SCHEMA_SHA256}:
        raise ContractError("Extractor feature-schema pin mismatch")
    if not isinstance(data["rows"], list) or not 1 <= len(data["rows"]) <= MAX_ROWS:
        raise ContractError("Expected 1..2000 rows")
    masks = obj["masks"]
    if not isinstance(masks, list) or len(masks) != len(data["rows"]):
        raise ContractError("Presence masks must cover all rows")
    rows = []
    row_ids = set()
    group_splits = {}
    for raw_row, mask in zip(data["rows"], masks):
        fields(raw_row, {"id", "group", "observedAt", "label", "split", "values"})
        row_id, group = text(raw_row["id"]), text(raw_row["group"])
        if row_id in row_ids:
            raise ContractError("Duplicate row id")
        row_ids.add(row_id)
        label = None if raw_row["label"] is None else text(raw_row["label"])
        split = raw_row["split"]
        if split not in ("train", "test"):
            raise ContractError("Invalid row split")
        if group in group_splits and group_splits[group] != split:
            raise ContractError("Capture group overlaps train and holdout")
        group_splits[group] = split
        stamp = raw_row["observedAt"]
        if stamp is not None:
            text(stamp, 64)
            try:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("timezone required")
            except ValueError as exc:
                raise ContractError("Observation timestamp must include timezone") from exc
        values = raw_row["values"]
        if not isinstance(values, list) or len(values) != 2:
            raise ContractError("Two thermal feature values required")
        if not isinstance(mask, list) or len(mask) != 2 or any(type(v) is not bool for v in mask):
            raise ContractError("Presence mask must contain two booleans")
        if mask != [value is not None for value in values]:
            raise ContractError("Presence mask conflicts with missing values")
        converted = tuple(None if value is None else finite(value) for value in values)
        if converted[0] is not None and converted[0] < -273.15:
            raise ContractError("Temperature below absolute zero")
        rows.append(Row(row_id, group, label, split, converted))
    if not isinstance(obj["captures"], list) or not 1 <= len(obj["captures"]) <= 20:
        raise ContractError("Every row needs retained original capture evidence")
    captures = []
    covered, artifacts, capture_hash_splits = set(), set(), {}
    by_id = {row.id: row for row in rows}
    for cap in obj["captures"]:
        fields(cap, {"artifact_id", "sha256", "byte_length", "media_type", "license", "calibration", "source_class", "row_ids"})
        artifact = text(cap["artifact_id"])
        try:
            if str(UUID(artifact)) != artifact:
                raise ValueError("noncanonical")
        except ValueError as exc:
            raise ContractError("Capture artifact id must be canonical UUID") from exc
        if artifact in artifacts or not isinstance(cap["sha256"], str) or not HASH.fullmatch(cap["sha256"]):
            raise ContractError("Duplicate artifact or invalid capture hash")
        artifacts.add(artifact)
        length = cap["byte_length"]
        if type(length) is not int or not 1 <= length <= MAX_BYTES:
            raise ContractError("Capture byte length outside 1..8388608")
        if cap["source_class"] not in ("observed", "synthetic"):
            raise ContractError("Capture source class must be observed or synthetic")
        if not isinstance(cap["row_ids"], list) or not cap["row_ids"] or len(cap["row_ids"]) > MAX_ROWS:
            raise ContractError("Capture must map to rows")
        ids = tuple(text(v) for v in cap["row_ids"])
        if len(set(ids)) != len(ids) or any(v not in row_ids or v in covered for v in ids):
            raise ContractError("Rows must map exactly once to retained captures")
        covered.update(ids)
        splits = {by_id[v].split for v in ids}
        if len(splits) != 1:
            raise ContractError("Original capture overlaps train and holdout")
        split = next(iter(splits))
        if cap["sha256"] in capture_hash_splits and capture_hash_splits[cap["sha256"]] != split:
            raise ContractError("Identical capture bytes appear in both splits")
        capture_hash_splits[cap["sha256"]] = split
        captures.append(Capture(artifact, cap["sha256"], length, text(cap["media_type"], 128), text(cap["license"], 2000), text(cap["calibration"], 2000), cap["source_class"], ids))
    if covered != row_ids:
        raise ContractError("Some rows lack original capture evidence")
    return Manifest(raw, digest(raw), tuple(rows), tuple(tuple(v) for v in masks), tuple(captures), source)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    fields(config, {"seed", "epochs", "learning_rate", "l2"})
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 0xFFFFFFFF:
        raise ContractError("Seed must be uint32")
    if type(config["epochs"]) is not int or not 1 <= config["epochs"] <= 200:
        raise ContractError("Epochs must be 1..200")
    lr, l2 = finite(config["learning_rate"], 1), finite(config["l2"], 1)
    if not 0 < lr <= 1 or not 0 <= l2 <= 1:
        raise ContractError("Invalid optimizer config")
    return {"seed": config["seed"], "epochs": config["epochs"], "learning_rate": lr, "l2": l2}
