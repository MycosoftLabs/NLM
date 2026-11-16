"""Tests for NLM training: dataset, losses, checkpoint."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from nlm.model.config import NLMConfig
from nlm.model.nlm_model import NatureLearningModel
from nlm.training.losses import (
    NLMLoss,
    PhysicsConsistencyLoss,
    TemporalCoherenceLoss,
    SpatialCoherenceLoss,
    UncertaintyCalibrationLoss,
)
from nlm.training.dataset import SyntheticNatureDataset, create_dataloader
from nlm.training.checkpoint import save_checkpoint, load_checkpoint
from nlm.training.trainer import NLMTrainer


@pytest.fixture
def small_config():
    return NLMConfig(
        spatial_dim=16,
        temporal_dim=16,
        spectral_sensory_dim=32,
        world_state_dim=32,
        self_state_dim=16,
        action_intent_dim=16,
        hidden_dim=64,
        num_heads=4,
        num_layers=1,
        ff_dim=128,
        num_ssm_layers=1,
        ssm_state_dim=8,
        graph_hidden_dim=32,
        graph_num_heads=2,
        graph_num_layers=1,
        fusion_num_heads=4,
        fusion_num_layers=1,
        max_spectral_bins=32,
        max_acoustic_bins=16,
        max_bioelectric_samples=32,
        max_thermal_grid=4,
        chemical_vector_dim=16,
        max_mechanical_bins=16,
        num_env_targets=8,
        num_anomaly_categories=5,
        num_bio_token_types=20,
        bio_token_embed_dim=16,
        max_bio_tokens=16,
        num_species_classes=10,
        num_compound_classes=10,
    )


class TestLosses:
    def test_physics_consistency_loss(self):
        loss_fn = PhysicsConsistencyLoss()
        # Values in bounds → low loss
        good = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        assert loss_fn(good).item() < 0.01

        # Values out of bounds → higher loss
        bad = torch.tensor([[2.0, -1.0, 0.5, 0.5]])
        assert loss_fn(bad).item() > 0.1

    def test_temporal_coherence_loss(self):
        loss_fn = TemporalCoherenceLoss(max_delta=0.1)
        # Smooth transition → low loss
        t0 = torch.tensor([[0.5, 0.5]])
        t1 = torch.tensor([[0.52, 0.48]])
        assert loss_fn(t0, t1).item() < 0.01

        # Discontinuity → higher loss
        t1_bad = torch.tensor([[1.5, -0.5]])
        assert loss_fn(t0, t1_bad).item() > 0.1

    def test_spatial_coherence_loss(self):
        loss_fn = SpatialCoherenceLoss()
        preds = torch.randn(4, 8)
        coords = torch.randn(4, 3)
        loss = loss_fn(preds, coords)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_nlm_combined_loss(self):
        loss_fn = NLMLoss()
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        anomaly_scores = torch.sigmoid(torch.randn(4, 5))
        losses = loss_fn(
            next_state_pred=pred,
            next_state_target=target,
            anomaly_scores=anomaly_scores,
        )
        assert "total" in losses
        assert "next_state" in losses
        assert "physics" in losses
        assert losses["total"].item() > 0

    def test_uncertainty_calibration(self):
        loss_fn = UncertaintyCalibrationLoss()
        confidence = torch.tensor([0.9, 0.1])
        error = torch.tensor([0.1, 0.9])
        loss = loss_fn(confidence, error)
        assert loss.item() >= 0


class TestDataset:
    def test_synthetic_dataset(self, small_config):
        ds = SyntheticNatureDataset(num_samples=10, config=small_config)
        assert len(ds) == 10
        sample = ds[0]
        assert "spatial_features" in sample
        assert "temporal_features" in sample
        assert "next_state_target" in sample
        assert sample["spatial_features"].shape == (37,)
        assert sample["bio_token_ids"].shape == (small_config.max_bio_tokens,)

    def test_dataloader(self, small_config):
        ds = SyntheticNatureDataset(num_samples=20, config=small_config)
        loader = create_dataloader(ds, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert batch["spatial_features"].shape[0] == 4


class TestCheckpoint:
    def test_save_load_roundtrip(self, small_config):
        model = NatureLearningModel(small_config)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_checkpoint(model, optimizer, epoch=5, loss=0.42, config=small_config, path=path)
            assert path.exists()

            # Load into new model
            model2 = NatureLearningModel(small_config)
            ckpt = load_checkpoint(path, model=model2)
            assert ckpt["epoch"] == 5
            assert abs(ckpt["loss"] - 0.42) < 1e-6

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch at {n1}"


class TestTrainingSmoke:
    """Smoke test: create model, run 2 training steps, verify loss."""

    def test_training_step(self, small_config):
        model = NatureLearningModel(small_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = NLMLoss()

        ds = SyntheticNatureDataset(num_samples=8, config=small_config)
        loader = create_dataloader(ds, batch_size=4, num_workers=0)

        losses = []
        model.train()
        for step, batch in enumerate(loader):
            if step >= 2:
                break

            output = model(
                spatial_features=batch["spatial_features"],
                temporal_features=batch["temporal_features"],
                spectral=batch["spectral"],
                acoustic=batch["acoustic"],
                bioelectric=batch["bioelectric"],
                thermal=batch["thermal"],
                chemical=batch["chemical"],
                mechanical=batch["mechanical"],
                modality_mask=batch.get("modality_mask"),
                env_features=batch.get("env_features"),
                bio_token_ids=batch.get("bio_token_ids"),
                graph_features=batch.get("graph_features"),
                self_state_features=batch.get("self_state_features"),
                recent_actions=batch.get("recent_actions"),
                intended_actions=batch.get("intended_actions"),
            )

            loss_dict = loss_fn(
                next_state_pred=output.next_state,
                next_state_target=batch["next_state_target"],
                anomaly_scores=output.anomaly_scores,
            )

            loss = loss_dict["total"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert len(losses) == 2
        assert all(l > 0 for l in losses)


class _TargetContractModel(torch.nn.Module):
    """Small differentiable stand-in to isolate the trainer's data boundary."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.level = torch.nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0

    def forward(self, spatial_features, **kwargs):
        self.forward_calls += 1
        batch_size = spatial_features.size(0)
        return SimpleNamespace(
            next_state=self.level.expand(batch_size, self.config.num_env_targets),
            anomaly_scores=torch.full((batch_size, self.config.num_anomaly_categories), 0.5),
            ecological_impact=None,
            grounding_confidence=None,
            growth_prediction=None,
        )


class TestTrainerTargetContract:
    @staticmethod
    def _batch():
        # Feature values are immaterial to the stand-in; shape drives the contract.
        return {name: torch.zeros(2, 1) for name in (
            "spatial_features", "temporal_features", "spectral", "acoustic",
            "bioelectric", "thermal", "chemical", "mechanical",
        )}

    @staticmethod
    def _trainer(config, batch, tmp_path):
        model = _TargetContractModel(config)
        return NLMTrainer(
            model, config, [batch], val_loader=[batch], device="cpu",
            output_dir=str(tmp_path),
        )

    @pytest.mark.parametrize("phase", ["train", "validate"])
    @pytest.mark.parametrize("case", [
        "missing", "none", "list", "integer", "bool", "complex",
        "scalar", "broadcast", "wrong_width", "wrong_batch", "nan", "inf", "neg_inf",
    ])
    def test_invalid_targets_fail_before_forward(self, phase, case, small_config, tmp_path):
        batch = self._batch()
        target = torch.zeros(2, small_config.num_env_targets)
        if case == "missing":
            pass
        elif case == "none":
            batch["next_state_target"] = None
        elif case == "list":
            batch["next_state_target"] = target.tolist()
        elif case in ("integer", "bool", "complex"):
            dtype = {"integer": torch.int64, "bool": torch.bool, "complex": torch.complex64}[case]
            batch["next_state_target"] = target.to(dtype)
        elif case == "scalar":
            batch["next_state_target"] = torch.tensor(0.0)
        elif case == "broadcast":
            batch["next_state_target"] = target[:1]
        elif case == "wrong_width":
            batch["next_state_target"] = target[:, :-1]
        elif case == "wrong_batch":
            batch["next_state_target"] = torch.zeros(3, small_config.num_env_targets)
        else:
            target[0, 0] = {"nan": float("nan"), "inf": float("inf"), "neg_inf": -float("inf")}[case]
            batch["next_state_target"] = target
        trainer = self._trainer(small_config, batch, tmp_path)
        before = trainer.model.level.detach().clone()
        with pytest.raises(ValueError, match="next_state_target"):
            if phase == "train":
                trainer.train_epoch(0)
            else:
                trainer.validate()
        assert trainer.model.forward_calls == 0
        assert trainer.global_step == 0
        assert torch.equal(before, trainer.model.level.detach())
        assert not list(tmp_path.glob("*.pt"))

    def test_observed_zero_targets_train_and_validate(self, small_config, tmp_path):
        batch = self._batch()
        batch["next_state_target"] = torch.zeros(2, small_config.num_env_targets)
        trainer = self._trainer(small_config, batch, tmp_path)
        before = trainer.model.level.detach().clone()
        train_losses = trainer.train_epoch(0)
        val_losses = trainer.validate()
        assert trainer.global_step == 1
        assert not torch.equal(before, trainer.model.level.detach())
        assert train_losses["next_state"] == pytest.approx(0.25 ** 2)
        assert val_losses["next_state"] > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in train_losses.values())
        assert all(torch.isfinite(torch.tensor(v)) for v in val_losses.values())

    def test_core_model_trainer_checkpoint_smoke(self, small_config, tmp_path):
        """Synthetic plumbing only: exercise the actual trainer and model together."""
        torch.manual_seed(42)
        dataset = SyntheticNatureDataset(num_samples=4, config=small_config)
        loader = create_dataloader(
            dataset, batch_size=2, num_workers=0, shuffle=False, pin_memory=False,
        )
        trainer = NLMTrainer(
            NatureLearningModel(small_config), small_config, loader,
            val_loader=loader, device="cpu", output_dir=str(tmp_path),
        )
        train_losses = trainer.train_epoch(0)
        val_losses = trainer.validate()
        assert trainer.global_step == 2
        assert all(torch.isfinite(torch.tensor(v)) for v in train_losses.values())
        assert all(torch.isfinite(torch.tensor(v)) for v in val_losses.values())
        path = tmp_path / "synthetic-smoke.pt"
        save_checkpoint(
            trainer.model, trainer.optimizer, 0, val_losses["total"], small_config, path,
            metadata={"synthetic": True},
        )
        restored = NatureLearningModel(small_config)
        checkpoint = load_checkpoint(path)
        # Explicit config is still necessary until full config serialization is fixed.
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert checkpoint["metadata"]["synthetic"] is True
        trainer.model.eval()
        restored.eval()
        batch = next(iter(loader))
        features = {k: v for k, v in batch.items() if not k.endswith("_target")}
        with torch.no_grad():
            expected = trainer.model(**features).next_state
            actual = restored(**features).next_state
        torch.testing.assert_close(actual, expected)
