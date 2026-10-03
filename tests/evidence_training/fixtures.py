"""Tiny invented fixture values, authored for this task; never field observations.

Repository proprietary terms apply. No license or dataset-use grant is invented.
"""

from nlm.evidence_training.contracts import FEATURE_SCHEMA_SHA256, digest, encode
from uuid import NAMESPACE_URL, uuid5


def fixture():
    groups = [
        ("train-cold", "train", "cold", [[0, -2], [2, -1], [1, -1.5]]),
        ("train-warm", "train", "warm", [[30, 2], [32, 3], [31, 2.5]]),
        ("test-cold", "test", "cold", [[3, -0.8], [4, -0.5]]),
        ("test-warm", "test", "warm", [[28, 1.8], [29, 2.1]]),
    ]
    rows, captures, blobs = [], [], {}
    for group, split, label, values in groups:
        group_rows = [{"id": f"{group}-{index}", "group": group, "split": split,
                       "label": label, "observedAt": None, "values": pair}
                      for index, pair in enumerate(values)]
        rows.extend(group_rows)
        blob = encode({"purpose": "synthetic test-only fixture", "group": group, "values": values})
        artifact_id = str(uuid5(NAMESPACE_URL, f"mycosoft.nlm.synthetic-fixture/{group}"))
        blobs[artifact_id] = blob
        captures.append({"artifact_id": artifact_id, "sha256": digest(blob), "byte_length": len(blob),
                         "media_type": "application/json", "license": "Repository proprietary; synthetic authored test fixture; no new grant",
                         "calibration": "Not calibrated; test-only invented values", "source_class": "synthetic",
                         "row_ids": [row["id"] for row in group_rows]})
    manifest = {
        "schema": "mycosoft.nlm.training-manifest.v1",
        "dataset": {"schema": "mycosoft.nlm.modality-dataset.v1", "modality": "thermal",
                    "channels": ["temperature", "gradient"], "units": ["degC", "degC_per_m"],
                    "source": "Authored synthetic thermal fixture; no physical observations",
                    "notes": "Two invented classes and four declared fixture capture groups", "rows": rows},
        "masks": [[True, True] for _ in rows], "captures": captures,
        "extractor": {"id": "supplied-features", "version": "1", "sha256": FEATURE_SCHEMA_SHA256},
        "timestamp_semantics": "Unknown timestamps on synthetic fixture; no chronology claimed",
    }
    config = {"seed": 42, "epochs": 12, "learning_rate": 0.2, "l2": 0.001}
    return encode(manifest), blobs, config
