"""
Maritime TAC-O Tests
====================

Tests for all maritime/TAC-O extensions to the NLM:
- Fingerprint types (HydroacousticFingerprint, MagneticAnomalyFingerprint, OceanEnvironmentFingerprint)
- Encoder extensions (hydroacoustic_enc, magnetic_anomaly_enc, ocean_environment_enc)
- Prediction heads (UnderwaterTargetClassification, SonarPerformancePrediction, etc.)
- HydroacousticPreconditioner (Mackenzie, Thorp, ray tracing, magnetometer calibration)
- MarineEcologicalGuard (AVANI marine mammal safety gate)
- Maritime losses
"""

import math

import pytest
import torch

from nlm.model.config import NLMConfig


# --- Fingerprint Tests ---

class TestMaritimeFingerprints:
    def test_hydroacoustic_fingerprint_creation(self):
        from nlm.core.fingerprints import HydroacousticFingerprint, FingerprintType
        fp = HydroacousticFingerprint(
            frequency_bands=[(10, 100), (100, 1000)],
            spectral_energy=[45.2, 62.1],
            harmonics=[120.0, 240.0],
            modulation_rate=7.5,
            broadband_level=85.0,
            narrowband_peaks=[(120.0, 72.5), (240.0, 65.0)],
            ambient_noise_level=60.0,
        )
        assert fp.fingerprint_type == FingerprintType.HYDROACOUSTIC
        assert fp.broadband_level == 85.0
        d = fp.to_dict()
        assert d["type"] == "hydroacoustic"
        assert d["modulation_rate"] == 7.5

    def test_magnetic_anomaly_fingerprint_creation(self):
        from nlm.core.fingerprints import MagneticAnomalyFingerprint, FingerprintType
        fp = MagneticAnomalyFingerprint(
            Bx=25000.0, By=1500.0, Bz=42000.0,
            total_field=49000.0,
            inclination=60.0, declination=-12.0,
            anomaly_magnitude=150.0,
            gradient_x=5.0, gradient_y=3.0,
        )
        assert fp.fingerprint_type == FingerprintType.MAGNETIC_ANOMALY
        assert fp.anomaly_magnitude == 150.0
        d = fp.to_dict()
        assert d["type"] == "magnetic_anomaly"
        assert d["total_field_nT"] == 49000.0

    def test_ocean_environment_fingerprint_creation(self):
        from nlm.core.fingerprints import OceanEnvironmentFingerprint, FingerprintType
        fp = OceanEnvironmentFingerprint(
            sound_speed_profile=[(0, 1500), (100, 1490), (500, 1480)],
            thermocline_depth=75.0,
            sea_surface_temp=18.5,
            salinity=35.2,
            sea_state=3,
            bottom_depth=200.0,
            bottom_type="sand",
        )
        assert fp.fingerprint_type == FingerprintType.OCEAN_ENVIRONMENT
        assert fp.thermocline_depth == 75.0
        d = fp.to_dict()
        assert d["type"] == "ocean_environment"
        assert d["salinity_psu"] == 35.2

    def test_fingerprint_type_enum_extended(self):
        from nlm.core.fingerprints import FingerprintType
        assert FingerprintType.HYDROACOUSTIC == "hydroacoustic"
        assert FingerprintType.MAGNETIC_ANOMALY == "magnetic_anomaly"
        assert FingerprintType.OCEAN_ENVIRONMENT == "ocean_environment"


# --- Prediction Head Tests ---

class TestMaritimeHeads:
    @pytest.fixture
    def config(self):
        return NLMConfig()

    def test_underwater_target_classification_head(self, config):
        from nlm.model.heads import UnderwaterTargetClassificationHead
        head = UnderwaterTargetClassificationHead(config)
        x = torch.randn(4, config.hidden_dim)
        out = head(x)
        assert "logits" in out
        assert "confidence" in out
        assert out["logits"].shape == (4, 12)
        assert out["confidence"].shape == (4,)
        assert (out["confidence"] >= 0).all() and (out["confidence"] <= 1).all()

    def test_sonar_performance_head(self, config):
        from nlm.model.heads import SonarPerformancePredictionHead
        head = SonarPerformancePredictionHead(config)
        x = torch.randn(4, config.hidden_dim)
        out = head(x)
        assert out.shape == (4, 4)

    def test_tactical_recommendation_head(self, config):
        from nlm.model.heads import TacticalRecommendationHead
        head = TacticalRecommendationHead(config)
        x = torch.randn(4, config.hidden_dim)
        out = head(x)
        assert "actions" in out
        assert "urgency" in out
        assert out["actions"].shape == (4, 10)
        assert out["urgency"].shape == (4,)

    def test_marine_mammal_filter_head(self, config):
        from nlm.model.heads import MarineMammalFilterHead
        head = MarineMammalFilterHead(config)
        x = torch.randn(4, config.hidden_dim)
        out = head(x)
        assert out.shape == (4,)
        assert (out >= 0).all() and (out <= 1).all()


# --- Preconditioner Tests ---

class TestHydroacousticPreconditioner:
    def test_mackenzie_sound_speed(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        c = HydroacousticPreconditioner.compute_sound_speed(
            temp_c=15.0, salinity_psu=35.0, depth_m=100.0
        )
        assert 1450 < c < 1550

    def test_sound_speed_varies_with_temp(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        c_cold = HydroacousticPreconditioner.compute_sound_speed(5.0, 35.0, 0.0)
        c_warm = HydroacousticPreconditioner.compute_sound_speed(25.0, 35.0, 0.0)
        assert c_warm > c_cold

    def test_transmission_loss(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        tl = HydroacousticPreconditioner.transmission_loss(1000.0, 1000.0)
        assert tl > 0
        tl_far = HydroacousticPreconditioner.transmission_loss(10000.0, 1000.0)
        assert tl_far > tl

    def test_ray_trace_basic(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        ssp = [(0, 1500), (50, 1495), (100, 1490), (200, 1485), (500, 1480)]
        path = HydroacousticPreconditioner.ray_trace_simple(
            ssp, source_depth=50.0, initial_angle_deg=5.0, max_range=500.0
        )
        assert len(path) > 1
        assert path[0] == (0.0, 50.0)

    def test_magnetometer_calibration(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        bx, by, bz = HydroacousticPreconditioner.calibrate_magnetometer(
            100.0, 200.0, 300.0,
            hard_iron=(10.0, 20.0, 30.0),
        )
        assert bx == 90.0
        assert by == 180.0
        assert bz == 270.0

    def test_magnetometer_with_soft_iron(self):
        from nlm.data.preconditioner import HydroacousticPreconditioner
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        bx, by, bz = HydroacousticPreconditioner.calibrate_magnetometer(
            110.0, 220.0, 330.0,
            hard_iron=(10.0, 20.0, 30.0),
            soft_iron_matrix=identity,
        )
        assert abs(bx - 100.0) < 0.01
        assert abs(by - 200.0) < 0.01
        assert abs(bz - 300.0) < 0.01


# --- AVANI Marine Ecological Guard Tests ---

class TestMarineEcologicalGuard:
    def test_high_mammal_score_gates_classification(self):
        from nlm.guardian.avani import MarineEcologicalGuard
        guard = MarineEcologicalGuard()
        result = guard.evaluate(
            classification_output={"marine_mammal_score": 0.85},
        )
        assert result["action"] == "gate_for_human_review"
        assert result["override_threat"] is True
        assert result["ecological_impact"] == "HIGH"

    def test_low_mammal_score_passes(self):
        from nlm.guardian.avani import MarineEcologicalGuard
        guard = MarineEcologicalGuard()
        result = guard.evaluate(
            classification_output={"marine_mammal_score": 0.1},
        )
        assert result["action"] == "pass"
        assert result["override_threat"] is False

    def test_frequency_overlap_detection(self):
        from nlm.guardian.avani import MarineEcologicalGuard
        guard = MarineEcologicalGuard()
        result = guard.evaluate(
            classification_output={"marine_mammal_score": 0.3},
            acoustic_fingerprint={
                "dominant_frequency": 50.0,
                "narrowband_peaks": [(50.0, 72.0)],
            },
        )
        assert result["action"] == "gate_for_human_review"
        assert result["matched_species"] is not None

    def test_tensor_marine_mammal_score(self):
        from nlm.guardian.avani import MarineEcologicalGuard
        guard = MarineEcologicalGuard()
        result = guard.evaluate(
            classification_output={"marine_mammal_score": torch.tensor(0.9)},
        )
        assert result["action"] == "gate_for_human_review"


# --- Maritime Loss Tests ---

class TestMaritimeLosses:
    def test_classification_loss(self):
        from nlm.training.losses import MaritimeClassificationLoss
        loss_fn = MaritimeClassificationLoss()
        logits = torch.randn(8, 12)
        targets = torch.randint(0, 12, (8,))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_sonar_performance_loss(self):
        from nlm.training.losses import SonarPerformanceLoss
        loss_fn = SonarPerformanceLoss()
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_marine_mammal_loss(self):
        from nlm.training.losses import MarineMammalFilterLoss
        loss_fn = MarineMammalFilterLoss()
        pred = torch.sigmoid(torch.randn(8))
        target = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_combined_loss(self):
        from nlm.training.losses import MaritimeCombinedLoss
        loss_fn = MaritimeCombinedLoss()
        losses = loss_fn(
            classification_logits=torch.randn(4, 12),
            classification_targets=torch.randint(0, 12, (4,)),
            mammal_pred=torch.sigmoid(torch.randn(4)),
            mammal_target=torch.randint(0, 2, (4,)).float(),
        )
        assert "maritime_classification" in losses
        assert "marine_mammal" in losses
        assert "maritime_total" in losses


# --- NLM Model Integration Tests ---

class TestNLMMaritimeIntegration:
    def test_nlm_has_maritime_heads(self):
        from nlm.model.nlm_model import NatureLearningModel
        config = NLMConfig()
        model = NatureLearningModel(config)
        assert hasattr(model, "underwater_target_head")
        assert hasattr(model, "sonar_performance_head")
        assert hasattr(model, "tactical_recommendation_head")
        assert hasattr(model, "marine_mammal_filter")

    def test_nlm_output_has_maritime_fields(self):
        from nlm.model.nlm_model import NatureLearningModel, NLMOutput
        config = NLMConfig()
        model = NatureLearningModel(config)
        batch_size = 2
        spatial = torch.randn(batch_size, 37)
        temporal = torch.randn(batch_size, 12)
        spectral = torch.randn(batch_size, config.max_spectral_bins)
        acoustic = torch.randn(batch_size, config.max_acoustic_bins)
        bioelectric = torch.randn(batch_size, config.max_bioelectric_samples)
        thermal = torch.randn(batch_size, config.max_thermal_grid ** 2)
        chemical = torch.randn(batch_size, config.chemical_vector_dim)
        mechanical = torch.randn(batch_size, config.max_mechanical_bins + 5)

        with torch.no_grad():
            output = model(
                spatial, temporal, spectral, acoustic,
                bioelectric, thermal, chemical, mechanical,
            )

        assert output.underwater_target is not None
        assert output.sonar_performance is not None
        assert output.tactical_recommendation is not None
        assert output.marine_mammal_score is not None
