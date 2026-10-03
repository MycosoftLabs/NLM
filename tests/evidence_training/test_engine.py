import copy
import math
import unittest

from nlm.evidence_training.contracts import ContractError, decode, digest, encode, validate_config, validate_manifest
from nlm.evidence_training.engine import predict, train, validate_checkpoint, validate_outputs
from fixtures import fixture


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.captures, self.config = fixture()
        self.manifest = validate_manifest(self.raw)

    def mutate(self, change):
        value = decode(self.raw)
        change(value)
        return encode(value)

    def test_actual_training_reload_confusion_and_reproducibility(self):
        first, second = train(self.manifest, self.config), train(self.manifest, self.config)
        self.assertEqual(first, second)
        initial = train(self.manifest, self.config, max_epochs=0)
        self.assertNotEqual(decode(initial.checkpoint_bytes)["weights"], decode(first.checkpoint_bytes)["weights"])
        evaluation = decode(first.evaluation_bytes)
        self.assertEqual(evaluation["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(sum(map(sum, evaluation["confusion_matrix"])), 4)
        self.assertEqual(evaluation["source_classes"], ["synthetic"])
        self.assertLess(evaluation["history"][-1]["training_cross_entropy"], evaluation["history"][0]["training_cross_entropy"])
        self.assertEqual(predict(first.model_bytes, [0, -2])["label"], "cold")
        self.assertEqual(predict(first.model_bytes, [31, 3])["label"], "warm")
        self.assertEqual(evaluation["model_sha256"], digest(first.model_bytes))
        validate_outputs(self.raw, self.config, {"model": first.model_bytes, "evaluation": first.evaluation_bytes, "checkpoint": first.checkpoint_bytes})

    def test_exact_resume_from_every_epoch_boundary(self):
        full = train(self.manifest, self.config)
        for epoch in range(self.config["epochs"] + 1):
            partial = train(self.manifest, self.config, max_epochs=epoch)
            self.assertEqual(validate_checkpoint(self.raw, self.config, partial.checkpoint_bytes), epoch)
            resumed = train(self.manifest, self.config, checkpoint_bytes=partial.checkpoint_bytes)
            self.assertEqual(resumed, full, epoch)

    def test_scaling_and_weights_use_training_rows_only(self):
        mutated = self.mutate(lambda obj: obj["dataset"]["rows"][-1].update(values=[900000, -900000]))
        first, changed = train(self.manifest, self.config), train(validate_manifest(mutated), self.config)
        a, b = decode(first.model_bytes), decode(changed.model_bytes)
        for key in ("mean", "scale", "weights"):
            self.assertEqual(a[key], b[key])
        self.assertNotEqual(decode(first.evaluation_bytes)["predictions"], decode(changed.evaluation_bytes)["predictions"])

    def test_mask_retained_and_missing_not_imputed(self):
        obj = decode(self.raw)
        obj["dataset"]["rows"][0]["values"][0] = None
        obj["masks"][0][0] = False
        missing = validate_manifest(encode(obj))
        self.assertIsNone(missing.rows[0].values[0])
        self.assertFalse(missing.masks[0][0])
        with self.assertRaisesRegex(ContractError, "missing is not zero"):
            train(missing, self.config)
        obj["masks"][0][0] = True
        with self.assertRaises(ContractError):
            validate_manifest(encode(obj))

    def test_bad_contract_cases(self):
        edits = [
            lambda o: o["dataset"]["units"].__setitem__(0, "K"),
            lambda o: o["dataset"].update(modality="sine"),
            lambda o: o["extractor"].update(sha256="0" * 64),
            lambda o: o["dataset"]["rows"][0].update(values=[True, 1]),
            lambda o: o["dataset"]["rows"][0].update(values=[10**500, 1]),
            lambda o: o["dataset"]["rows"][0].update(values=[-274, 1]),
            lambda o: o["dataset"]["rows"][0].update(observedAt="2026-10-01"),
            lambda o: o["dataset"]["rows"][6].update(group="train-cold"),
            lambda o: o["captures"][2].update(sha256=o["captures"][0]["sha256"]),
            lambda o: o["captures"][0]["row_ids"].append(o["dataset"]["rows"][6]["id"]),
            lambda o: o["captures"].pop(),
            lambda o: o.update(owner_id="browser-controlled"),
        ]
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(ContractError):
                validate_manifest(self.mutate(edit))

    def test_json_ambiguity_and_numeric_overflow_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":"\\ud800"}', b'[]'):
            with self.assertRaises(ContractError):
                decode(raw)
        self.assertNotEqual(digest(self.raw), digest(b" " + self.raw))
        self.assertEqual(validate_manifest(b" " + self.raw).rows, self.manifest.rows)

    def test_checkpoint_and_completion_reject_foreign_or_corrupt_evidence(self):
        result = train(self.manifest, self.config)
        for key, value in (("manifest_sha256", "0" * 64), ("epoch", 500), ("mean", [0, 0]), ("history", [])):
            checkpoint = decode(result.checkpoint_bytes)
            checkpoint[key] = value
            with self.assertRaises(ContractError):
                train(self.manifest, self.config, checkpoint_bytes=encode(checkpoint))
        with self.assertRaises(ContractError):
            train(self.manifest, {**self.config, "seed": 9}, checkpoint_bytes=result.checkpoint_bytes)
        outputs = {"model": result.model_bytes, "evaluation": result.evaluation_bytes, "checkpoint": result.checkpoint_bytes}
        evaluation = decode(outputs["evaluation"])
        evaluation["accuracy"] = 0.5
        with self.assertRaises(ContractError):
            validate_outputs(self.raw, self.config, {**outputs, "evaluation": encode(evaluation)})

    def test_predict_rejects_bad_model_and_missing_features(self):
        model = train(self.manifest, self.config).model_bytes
        for values in ([None, 1], [float("nan"), 1], [True, 1], [1], [-274, 1]):
            with self.assertRaises(ContractError):
                predict(model, values)
        broken = decode(model)
        broken["scale"][0] = 5e-324
        with self.assertRaises(ContractError):
            predict(encode(broken), [1e6, 1])

    def test_config_bounds(self):
        for key, value in (("seed", True), ("seed", -1), ("epochs", 201), ("epochs", 1.2), ("learning_rate", 0), ("l2", -1)):
            with self.assertRaises(ContractError):
                validate_config({**self.config, key: value})


if __name__ == "__main__":
    unittest.main()
