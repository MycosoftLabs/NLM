"""Full-batch softmax on two supplied thermal features, using Python stdlib only.

No raw capture decoding or canonical six-stream NLM training occurs here. Synthetic
fixtures qualify algorithm mechanics only. Dataset licenses remain caller evidence.
"""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contracts import ENGINE_ID, FEATURE_SCHEMA_SHA256, ContractError, Manifest, decode, digest, encode, fields, finite, validate_config

MODEL_SCHEMA = "mycosoft.nlm.thermal-softmax-model.v1"
CHECKPOINT_SCHEMA = "mycosoft.nlm.thermal-softmax-checkpoint.v1"
EVALUATION_SCHEMA = "mycosoft.nlm.heldout-evaluation.v1"
REPOSITORY_LICENSE_SHA256_LF = "f0714ff0ff7a0d31e809def32f052372c2a8067af1e3785c3ff4f87e11c89b53"


def engine_pin() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    source = {name: digest((root / name).read_bytes().replace(b"\r\n", b"\n")) for name in ("contracts.py", "engine.py")}
    license_path = root.parents[1] / "LICENSE"
    if license_path.is_file() and digest(license_path.read_bytes().replace(b"\r\n", b"\n")) != REPOSITORY_LICENSE_SHA256_LF:
        raise ContractError("Repository LICENSE differs from reviewed engine license pin")
    return {
        "id": ENGINE_ID, "version": "1", "source_sha256": digest(encode(source)),
        "source_files": source, "source_hash_normalization": "CRLF-to-LF only",
        "license": "Mycosoft proprietary; no new license grant",
        "license_sha256": REPOSITORY_LICENSE_SHA256_LF,
        "runtime": {"implementation": platform.python_implementation(), "python": sys.version,
                    "system": platform.system(), "machine": platform.machine(),
                    "arithmetic": "stdlib-float64-fsum-full-batch-single-thread"},
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "external_model_assets": [], "full_nlm": False,
    }


@dataclass(frozen=True)
class TrainingResult:
    model_bytes: bytes | None
    evaluation_bytes: bytes | None
    checkpoint_bytes: bytes
    completed: bool


def _softmax(scores: list[float]) -> list[float]:
    if not scores or any(not math.isfinite(value) for value in scores):
        raise ContractError("Non-finite model score")
    maximum = max(scores)
    exp = [math.exp(value - maximum) for value in scores]
    total = math.fsum(exp)
    return [value / total for value in exp]


def _probabilities(weights: list[list[float]], vector: list[float]) -> list[float]:
    return _softmax([math.fsum(w * v for w, v in zip(row, vector)) for row in weights])


def _initial_weights(seed: int, classes: int) -> list[list[float]]:
    # Completely specified LCG initialization; no global or platform RNG state.
    state = seed
    weights = []
    for _ in range(classes):
        row = []
        for _ in range(3):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            row.append(((state / 4294967296.0) - 0.5) * 0.02)
        weights.append(row)
    return weights


def _split(manifest: Manifest) -> dict[str, Any]:
    return {
        "training_row_ids": [r.id for r in manifest.rows if r.split == "train"],
        "test_row_ids": [r.id for r in manifest.rows if r.split == "test"],
        "training_groups": sorted({r.group for r in manifest.rows if r.split == "train"}),
        "test_groups": sorted({r.group for r in manifest.rows if r.split == "test"}),
        "capture_sha256": [c.sha256 for c in manifest.captures],
    }


def _scaler(manifest: Manifest) -> tuple[list[float], list[float]]:
    rows = [r for r in manifest.rows if r.split == "train"]
    means = [math.fsum(r.values[c] for r in rows) / len(rows) for c in range(2)]
    scales = [math.sqrt(math.fsum((r.values[c] - means[c]) ** 2 for r in rows) / len(rows)) or 1.0 for c in range(2)]
    return means, scales


def _vector(values, means, scales) -> list[float]:
    return [finite((values[c] - means[c]) / scales[c], 1e12) for c in range(2)] + [1.0]


def _matrix(value: Any, count: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != count:
        raise ContractError("Weight matrix class count mismatch")
    out = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ContractError("Weight matrix feature count mismatch")
        out.append([finite(v, 1e12) for v in row])
    return out


def train(manifest: Manifest, config: dict[str, Any], *, checkpoint_bytes: bytes | None = None,
          max_epochs: int | None = None, on_epoch: Callable[[bytes], None] | None = None,
          before_epoch: Callable[[], None] | None = None) -> TrainingResult:
    """Fit or resume at an epoch boundary; optimizer has no momentum or hidden state.

    max_epochs limits this invocation, while config.epochs fixes the admitted job.
    Callbacks must persist through a fenced durable port; callback errors propagate.
    """
    manifest.require_training_ready()
    config = validate_config(config)
    if max_epochs is not None and (type(max_epochs) is not int or not 0 <= max_epochs <= 200):
        raise ContractError("Invocation epoch budget must be 0..200")
    labels = sorted({row.label for row in manifest.rows if row.split == "train"})
    means, scales = _scaler(manifest)
    pin, split = engine_pin(), _split(manifest)
    fixed = {
        "schema": CHECKPOINT_SCHEMA, "engine": pin, "manifest_sha256": manifest.sha256,
        "config": config, "config_sha256": digest(encode(config)), "split": split,
        "split_sha256": digest(encode(split)), "labels": labels,
        "mean": means, "scale": scales, "optimizer": "full-batch-gradient-descent-no-momentum",
        "cursor_semantics": "completed_epochs; fixed manifest row order; RNG used at initialization only",
    }
    epoch, history = 0, []
    weights = _initial_weights(config["seed"], len(labels))
    if checkpoint_bytes is not None:
        checkpoint = decode(checkpoint_bytes)
        fields(checkpoint, set(fixed) | {"epoch", "weights", "history"})
        if any(checkpoint[key] != value for key, value in fixed.items()):
            raise ContractError("Checkpoint source/runtime/config/manifest/split/scaler mismatch")
        epoch = checkpoint["epoch"]
        if type(epoch) is not int or not 0 <= epoch <= config["epochs"]:
            raise ContractError("Checkpoint epoch outside admitted job")
        weights = _matrix(checkpoint["weights"], len(labels))
        history = checkpoint["history"]
        if not isinstance(history, list) or len(history) != epoch:
            raise ContractError("Checkpoint history/cursor mismatch")
        for index, point in enumerate(history, start=1):
            fields(point, {"epoch", "training_cross_entropy"})
            if type(point["epoch"]) is not int or point["epoch"] != index or finite(point["training_cross_entropy"], 1e12) < 0:
                raise ContractError("Invalid checkpoint history")
    train_rows = [r for r in manifest.rows if r.split == "train"]
    vectors = [_vector(row.values, means, scales) for row in train_rows]
    target_ids = [labels.index(row.label) for row in train_rows]
    end = config["epochs"] if max_epochs is None else min(config["epochs"], epoch + max_epochs)

    def checkpoint_blob():
        return encode({**fixed, "epoch": epoch, "weights": weights, "history": history})

    for _ in range(epoch, end):
        if before_epoch:
            before_epoch()
        probabilities = [_probabilities(weights, vector) for vector in vectors]
        gradient = [[math.fsum((probs[c] - (1.0 if target == c else 0.0)) * vector[f]
                    for probs, target, vector in zip(probabilities, target_ids, vectors)) / len(vectors)
                    + (config["l2"] * weights[c][f] if f < 2 else 0.0)
                    for f in range(3)] for c in range(len(labels))]
        weights = [[finite(weights[c][f] - config["learning_rate"] * gradient[c][f], 1e12)
                    for f in range(3)] for c in range(len(labels))]
        epoch += 1
        loss = -math.fsum(math.log(max(_probabilities(weights, vector)[target], 1e-300))
                          for vector, target in zip(vectors, target_ids)) / len(vectors)
        history.append({"epoch": epoch, "training_cross_entropy": loss})
        if on_epoch:
            on_epoch(checkpoint_blob())
    checkpoint_raw = checkpoint_blob()
    if epoch != config["epochs"]:
        return TrainingResult(None, None, checkpoint_raw, False)
    model = {
        "schema": MODEL_SCHEMA, "engine": pin, "manifest_sha256": manifest.sha256,
        "config": config, "config_sha256": fixed["config_sha256"], "split": split,
        "split_sha256": fixed["split_sha256"], "channels": ["temperature", "gradient"],
        "units": ["degC", "degC_per_m"], "mean": means, "scale": scales,
        "labels": labels, "weights": weights, "epochs": epoch,
    }
    model_raw = encode(model)
    predictions, matrix = [], [[0 for _ in labels] for _ in labels]
    test_rows = [r for r in manifest.rows if r.split == "test"]
    for row in test_rows:
        result = predict(model_raw, list(row.values))
        matrix[labels.index(row.label)][labels.index(result["label"])] += 1
        predictions.append({"id": row.id, "actual": row.label, "predicted": result["label"],
                            "probabilities": result["probabilities"]})
    evaluation = {
        "schema": EVALUATION_SCHEMA, "engine": pin, "manifest_sha256": manifest.sha256,
        "model_sha256": digest(model_raw), "checkpoint_sha256": digest(checkpoint_raw),
        "config_sha256": fixed["config_sha256"], "split_sha256": fixed["split_sha256"],
        "labels": labels, "confusion_matrix": matrix, "predictions": predictions,
        "training_rows": len(train_rows), "heldout_rows": len(test_rows),
        "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(test_rows),
        "source_classes": sorted({c.source_class for c in manifest.captures}),
        "dataset_scope": manifest.source, "history": history,
        "limitations": ["Supplied thermal features only; no raw sensor decoder or calibration validation",
                        "Operator-declared capture independence; this cannot prove physical independence",
                        "Heldout dataset performance only; probabilities are not calibrated",
                        "Synthetic data qualifies fixture mechanics only", "Not full NLM architecture or temporal inference"],
    }
    return TrainingResult(model_raw, encode(evaluation), checkpoint_raw, True)


def predict(model_bytes: bytes, values: list[float]) -> dict[str, Any]:
    """Reload a JSON feature model and return a probability vector, without code loading."""
    model = decode(model_bytes)
    fields(model, {"schema", "engine", "manifest_sha256", "config", "config_sha256", "split", "split_sha256", "channels", "units", "mean", "scale", "labels", "weights", "epochs"})
    if model["schema"] != MODEL_SCHEMA or model["engine"] != engine_pin():
        raise ContractError("Model engine source/runtime pin mismatch")
    if model["channels"] != ["temperature", "gradient"] or model["units"] != ["degC", "degC_per_m"]:
        raise ContractError("Model feature contract mismatch")
    if not isinstance(values, list) or len(values) != 2:
        raise ContractError("Two observed thermal feature values required")
    values = [finite(value) for value in values]
    if values[0] < -273.15:
        raise ContractError("Temperature below absolute zero")
    labels = model["labels"]
    if not isinstance(labels, list) or not 2 <= len(labels) <= 32 or any(not isinstance(x, str) or not x.strip() or len(x) > 512 for x in labels) or len(set(labels)) != len(labels) or labels != sorted(labels):
        raise ContractError("Invalid model labels")
    means, scales = model["mean"], model["scale"]
    if not isinstance(means, list) or not isinstance(scales, list) or len(means) != 2 or len(scales) != 2:
        raise ContractError("Invalid model scaler")
    means, scales = [finite(v) for v in means], [finite(v, 2e6) for v in scales]
    if any(v <= 0 for v in scales):
        raise ContractError("Model scale must be positive")
    config = validate_config(model["config"])
    if model["config_sha256"] != digest(encode(config)) or model["split_sha256"] != digest(encode(model["split"])) or model["epochs"] != config["epochs"]:
        raise ContractError("Model config/split pin mismatch")
    probs = _probabilities(_matrix(model["weights"], len(labels)), _vector(values, means, scales))
    best = max(range(len(labels)), key=lambda index: probs[index])
    return {"label": labels[best], "probabilities": dict(zip(labels, probs)), "model_sha256": digest(model_bytes)}


def validate_checkpoint(manifest_bytes: bytes, config: dict[str, Any], checkpoint_bytes: bytes) -> int:
    """Validate all checkpoint bindings without any gradient update; return cursor."""
    from .contracts import validate_manifest
    result = train(validate_manifest(manifest_bytes), config, checkpoint_bytes=checkpoint_bytes, max_epochs=0)
    return decode(result.checkpoint_bytes)["epoch"]


def validate_outputs(manifest_bytes: bytes, config: dict[str, Any], outputs: dict[str, bytes]) -> None:
    """Server completion boundary: reproduce artifacts from the final checkpoint.

    This proves internal consistency and bound heldout evaluation, not independent
    scientific qualification or trusted execution of the training worker.
    """
    from .contracts import validate_manifest
    if set(outputs) != {"model", "evaluation", "checkpoint"}:
        raise ContractError("Require exactly model, evaluation and checkpoint artifacts")
    result = train(validate_manifest(manifest_bytes), config, checkpoint_bytes=outputs["checkpoint"], max_epochs=0)
    if not result.completed or result.model_bytes != outputs["model"] or result.evaluation_bytes != outputs["evaluation"]:
        raise ContractError("Output bundle does not match admitted manifest/config/final checkpoint")
