"""Tests for NLM core ontology: frames, merkle, fingerprints, protocols."""

from datetime import datetime, timezone

from nlm.core.fingerprints import (
    AcousticFingerprint,
    BioelectricFingerprint,
    ChemicalFingerprint,
    MechanicalFingerprint,
    SpectralFingerprint,
    ThermalFingerprint,
)
from nlm.core.frames import (
    ActionContext,
    Observation,
    Provenance,
    RootedNatureFrame,
    SelfState,
    Uncertainty,
    WorldState,
)
from nlm.core.merkle import (
    LineageRecord,
    build_lineage_dag,
    compute_event_root,
    compute_frame_root,
    compute_self_root,
    compute_world_root,
    merkle_hash,
    verify_frame_root,
    verify_lineage,
)
from nlm.core.protocols import DeviceEnvelope, ProtocolHeader, SensorMetadata


class TestFingerprints:
    def test_spectral_fingerprint(self):
        fp = SpectralFingerprint(
            wavelength_bins=[400.0, 500.0, 600.0, 700.0],
            energy_values=[0.1, 0.8, 0.5, 0.2],
            source_type="optical",
        )
        assert fp.num_bins() == 4
        assert fp.peak_wavelength() == 500.0
        assert abs(fp.total_energy() - 1.6) < 1e-10
        d = fp.to_dict()
        assert d["type"] == "spectral"
        assert d["peak_wavelength"] == 500.0

    def test_acoustic_fingerprint(self):
        fp = AcousticFingerprint(
            frequency_bins=[100.0, 200.0, 300.0],
            magnitude=[30.0, 60.0, 45.0],
            duration_ms=100.0,
        )
        assert fp.dominant_frequency() == 200.0
        assert fp.bandwidth() == 200.0

    def test_bioelectric_fingerprint(self):
        fp = BioelectricFingerprint(
            voltage_series=[1.0, 2.0, 3.0, 2.0],
            current_series=[0.1, 0.2],
            impedance=1000.0,
            sample_rate_hz=1000,
        )
        assert fp.mean_voltage() == 2.0
        assert fp.voltage_range() == 2.0
        assert fp.duration_ms() == 4.0

    def test_thermal_fingerprint(self):
        fp = ThermalFingerprint(
            temperature_field=[[20.0, 22.0], [21.0, 25.0]],
            gradient_magnitude=2.5,
        )
        assert fp.mean_temperature() == 22.0
        assert fp.max_temperature() == 25.0
        assert fp.thermal_contrast() == 5.0

    def test_chemical_fingerprint(self):
        fp = ChemicalFingerprint(
            voc_concentrations={"ethanol": 10.0, "acetone": 5.0},
            ph=6.5,
        )
        assert fp.total_voc() == 15.0
        assert fp.is_acidic()

    def test_mechanical_fingerprint(self):
        fp = MechanicalFingerprint(
            pressure_pa=101325.0,
            force_vector=(3.0, 4.0, 0.0),
        )
        assert abs(fp.force_magnitude() - 5.0) < 0.01


class TestMerkle:
    def test_merkle_hash_deterministic(self):
        h1 = merkle_hash("a", "b", "c")
        h2 = merkle_hash("a", "b", "c")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_merkle_hash_different_inputs(self):
        h1 = merkle_hash("a", "b")
        h2 = merkle_hash("a", "c")
        assert h1 != h2

    def test_compute_frame_root(self):
        root = compute_frame_root("self", "world", "event", "parent")
        assert len(root) == 64
        assert verify_frame_root(root, "self", "world", "event", "parent")
        assert not verify_frame_root("wrong", "self", "world", "event", "parent")

    def test_compute_event_root(self):
        root = compute_event_root("2024-01-01", "0,0,0", ["h1", "h2"], ["TEMP_OPT"])
        assert len(root) == 64

    def test_compute_state_roots(self):
        self_root = compute_self_root({"safety_mode": "normal"})
        world_root = compute_world_root({"temperature": 22.0})
        assert len(self_root) == 64
        assert len(world_root) == 64

    def test_lineage_record(self):
        record = LineageRecord(
            frame_root="abc", parent_frame_root="def",
            self_root="s", world_root="w", event_root="e",
            producer="test",
        )
        d = record.to_dict()
        restored = LineageRecord.from_dict(d)
        assert restored.frame_root == "abc"
        assert restored.producer == "test"

    def test_verify_lineage(self):
        # Build a valid chain
        r1_self = compute_self_root({"a": 1})
        r1_world = compute_world_root({"b": 2})
        r1_event = compute_event_root("t1", "loc1", [], [])
        r1_frame = compute_frame_root(r1_self, r1_world, r1_event, "")

        r2_self = compute_self_root({"a": 2})
        r2_world = compute_world_root({"b": 3})
        r2_event = compute_event_root("t2", "loc2", [], [])
        r2_frame = compute_frame_root(r2_self, r2_world, r2_event, r1_frame)

        records = [
            LineageRecord(frame_root=r1_frame, parent_frame_root="",
                         self_root=r1_self, world_root=r1_world, event_root=r1_event),
            LineageRecord(frame_root=r2_frame, parent_frame_root=r1_frame,
                         self_root=r2_self, world_root=r2_world, event_root=r2_event),
        ]
        assert verify_lineage(records)

    def test_build_lineage_dag(self):
        records = [
            LineageRecord(frame_root="b", parent_frame_root="a",
                         self_root="s", world_root="w", event_root="e"),
            LineageRecord(frame_root="c", parent_frame_root="a",
                         self_root="s", world_root="w", event_root="e"),
        ]
        dag = build_lineage_dag(records)
        assert "a" in dag
        assert set(dag["a"]) == {"b", "c"}


class TestProtocols:
    def test_device_envelope_roundtrip(self):
        envelope = DeviceEnvelope(
            device_id="mushroom1-001",
            device_slug="m1-001",
            site_id="site-alpha",
            header=ProtocolHeader(protocol_name="mycorrhizae", protocol_version="2.0"),
            sensors=[SensorMetadata(
                sensor_id="temp-0", sensor_type="temperature", unit="°C",
                accuracy=0.5, range_min=-40, range_max=80,
            )],
            latitude=37.7749,
            longitude=-122.4194,
            altitude=10.0,
            readings={"temp-0": 22.5},
            verified=True,
        )
        d = envelope.to_dict()
        restored = DeviceEnvelope.from_dict(d)
        assert restored.device_id == "mushroom1-001"
        assert restored.header.protocol_name == "mycorrhizae"
        assert len(restored.sensors) == 1
        assert restored.readings["temp-0"] == 22.5
        assert restored.geolocation() == (37.7749, -122.4194, 10.0)

    def test_sensor_metadata(self):
        meta = SensorMetadata(
            sensor_id="bio-0", sensor_type="bioelectric", unit="mV",
            accuracy=0.01, range_min=-100, range_max=100,
            sampling_rate_hz=1000,
        )
        d = meta.to_dict()
        assert d["sensor_type"] == "bioelectric"
        assert d["sampling_rate_hz"] == 1000


class TestFrames:
    def test_rooted_nature_frame_creation(self):
        frame = RootedNatureFrame(
            frame_root="test_root",
            geolocation=(37.7749, -122.4194, 10.0),
            device_ids=["device-001"],
        )
        assert frame.frame_root == "test_root"
        assert frame.geolocation == (37.7749, -122.4194, 10.0)

    def test_observation_fingerprint_count(self):
        obs = Observation(
            spectral=[SpectralFingerprint([], [], "optical")],
            thermal=[ThermalFingerprint([[20.0]])],
        )
        assert obs.fingerprint_count() == 2

    def test_self_state(self):
        state = SelfState(
            safety_mode="cautious",
            active_agents=["agent-1", "agent-2"],
            available_tools=["search", "predict"],
        )
        d = state.to_dict()
        assert d["safety_mode"] == "cautious"
        assert len(d["active_agents"]) == 2

    def test_frame_to_dict(self):
        frame = RootedNatureFrame(
            frame_root="root",
            self_state=SelfState(safety_mode="normal"),
            world_state=WorldState(environmental={"temperature_c": 22.0}),
        )
        d = frame.to_dict()
        assert d["frame_root"] == "root"
        assert d["self_state"]["safety_mode"] == "normal"
        assert d["world_state"]["environmental"]["temperature_c"] == 22.0
