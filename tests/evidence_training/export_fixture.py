"""Export actual tiny synthetic training artifacts for offline review only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nlm.evidence_training.contracts import digest, encode, validate_manifest
from nlm.evidence_training.engine import train, validate_outputs
from fixtures import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence-training/fixture-receipts")
    args = parser.parse_args()
    manifest_bytes, captures, config = fixture()
    result = train(validate_manifest(manifest_bytes), config)
    outputs = {"model": result.model_bytes, "evaluation": result.evaluation_bytes, "checkpoint": result.checkpoint_bytes}
    validate_outputs(manifest_bytes, config, outputs)
    artifacts = {"manifest.json": manifest_bytes, "config.json": encode(config),
                 **{f"{key}.json": value for key, value in outputs.items()},
                 **{f"captures/{key}.json": value for key, value in captures.items()}}
    for relative, raw in artifacts.items():
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    receipt = {
        "classification": "synthetic test-only fixture; not observed sensor data",
        "qualification": "local CPU artifact generation and reload; not deployed MINDEX retention",
        "fixture_source_sha256_lf": digest(Path(__file__).with_name("fixtures.py").read_bytes().replace(b"\r\n", b"\n")),
        "files": {name: {"sha256": digest(raw), "byte_length": len(raw)} for name, raw in artifacts.items()},
    }
    (args.output / "receipt.json").write_bytes(encode(receipt))
    evaluation = json.loads(result.evaluation_bytes)
    print(json.dumps({"output": str(args.output.resolve()), "training_rows": evaluation["training_rows"],
                      "heldout_rows": evaluation["heldout_rows"], "confusion_matrix": evaluation["confusion_matrix"],
                      "qualification": receipt["qualification"]}))


if __name__ == "__main__":
    main()
