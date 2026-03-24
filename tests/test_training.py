"""Tests for NLM training: dataset, losses, checkpoint."""

import tempfile
from pathlib import Path

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
