# NLM — Nature Learning Model

**A grounded sensory world model that learns from raw physical reality — wavelengths, waveforms, voltages, gas concentrations, temperature gradients, pressure fields — and predicts what happens next.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## What NLM Is

NLM is not an LLM. It does not start from language. It starts from raw physical reality and builds upward through deterministic scientific transforms, sensory fingerprint extraction, Merkle-rooted state assembly, and a hybrid learned model — before language ever enters the picture.

### The Cognitive Pipeline

```
raw reality
  → deterministic scientific transforms (physics, chemistry, biology)
  → calibration / normalization
  → fingerprint extraction (spectral, acoustic, bioelectric, thermal, chemical, mechanical)
  → state assembly into RootedNatureFrame (Merkle-rooted)
  → 6 learned stream encoders
  → hybrid core model (SSM + graph + sparse attention)
  → prediction heads
  → AVANI guardian layer (grounding, ecological scoring, harm detection, veto)
  → language / agent output
```

### The Six Senses

NLM perceives the physical world through six sensory fingerprint types:

| Sense | Fingerprint | What NLM Perceives |
|-------|------------|-------------------|
| Sight | `SpectralFingerprint` | Wavelengths, spectral power distributions, band indices |
| Hearing | `AcousticFingerprint` | Frequency-energy distributions, harmonics, waveform digests |
| Electroception | `BioelectricFingerprint` | Voltage, current, impedance spectral profiles |
| Thermoception | `ThermalFingerprint` | Temperature gradients, heat flux vectors |
| Smell | `ChemicalFingerprint` | VOC/VSC gas vectors, pH, conductivity, CO₂ |
| Touch | `MechanicalFingerprint` | Pressure, vibration spectra, seismic waveforms |

### Key Architecture Decisions

- **RootedNatureFrame**: Every observation is Merkle-rooted — tamper-evident, verifiable, replayable
- **Deterministic preconditioning**: Physics/chemistry/biology transforms run before any learned model
- **SSM/Mamba backbone**: Not transformer-first. State space models for temporal state evolution (linear complexity)
- **Graph/Hypergraph backbone**: GNN message-passing over entity-relation graphs from MINDEX
- **Sparse attention fusion**: Cross-stream integration only — 6 streams talk to each other
- **AVANI guardian**: Mandatory ecological safety gate on all outputs
- **Multi-Resolution Merkle HyperDAG**: 5-layer graph from raw events to causal lineage

## Quick Start

```python
from nlm.data.rooted_frame_builder import RootedFrameBuilder

builder = RootedFrameBuilder()

# Build a Merkle-rooted frame from raw sensor data
frame = builder.build(
    raw_data={
        "temperature_c": 22.5,
        "humidity_pct": 75.0,
        "co2_ppm": 420,
        "light_lux": 500,
    },
    lat=45.5, lon=-122.6, alt_m=50,
    device_id="fci-001",
    protocol="fci",
)

print(f"Frame root: {frame.frame_root.hex()}")
print(f"Fingerprints: {len(frame.observation.fingerprints)}")
print(f"Derived fields: {list(frame.world_state.derived_fields.keys())}")
```

### Running the Model

```python
import torch
from nlm.model.nlm_model import NLMWorldModel, NLMConfig

model = NLMWorldModel(NLMConfig(d_model=256))

outputs = model(
    lat=torch.tensor([45.5]),
    lon=torch.tensor([-122.6]),
    alt=torch.tensor([50.0]),
    timestamps=torch.tensor([1711100000.0]),
    thermal=torch.tensor([[22.5]]),
    chemical=torch.tensor([[420.0, 75.0]]),
)

print(f"Next state prediction shape: {outputs['next_state']['predicted_state'].shape}")
print(f"Anomaly score: {outputs['anomaly']['anomaly_score'].item():.3f}")
print(f"Grounding confidence: {outputs['grounding_confidence'].item():.3f}")
```

## Project Structure

```
nlm/
  core/
    frames.py              # RootedNatureFrame — central cognitive object
    merkle.py              # Merkle tree, roots, proofs, lineage chain
    fingerprints.py        # 6 sensory fingerprint types
    protocols.py           # Device protocols (FCI, Mushroom1, MycoNode, SporeBase, Petraeus)
  data/
    preconditioner.py      # Deterministic physics/chemistry/biology transforms
    fingerprint_extraction.py  # Raw → fingerprint extraction
    rooted_frame_builder.py    # Full pipeline: raw → RootedNatureFrame
  graph/
    hyperdag.py            # Multi-Resolution Merkle HyperDAG (5 layers)
    retrieval.py           # GraphRAG retrieval
  model/
    ssm_blocks.py          # SSM/Mamba temporal blocks
    graph_encoders.py      # WorldState and SelfState graph encoders
    encoders.py            # Spatial, Temporal, SpectralSensory, ActionIntent encoders
    fusion.py              # Sparse attention cross-stream fusion
    heads.py               # Prediction heads (primary + secondary)
    nlm_model.py           # NLMWorldModel — top-level model class
  guardian/
    avani.py               # AVANI ecological safety guardian
  mindex/
    client.py              # Unified MINDEX client
  telemetry/               # Legacy (bio-tokens demoted to downstream discretization)
  physics/                 # Physics preconditioners (refactored)
  chemistry/               # Chemistry preconditioners
  biology/                 # Biology preconditioners
  search/                  # Universal Earth search
  api/                     # FastAPI endpoints
```

## Service Boundaries

| Service | Owns | Does NOT Own |
|---------|------|-------------|
| **MAS** | Orchestration, agent identity, AVANI governance policies | Model weights, sensor storage |
| **NLM** | Runtime, training, inference, preconditioning, model | MINDEX persistence, AVANI policy |
| **MINDEX** | Persistence, graph/vector store, Merkle lineage, provenance | Model inference, agent behavior |

## Supported Devices

- **FCI** — Fungal Computing Interface
- **Mushroom1** — Sensor boards
- **MycoNode** — Distributed sensors
- **SporeBase** — Data loggers
- **Petraeus** — Environmental monitors

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License — see [LICENSE](./LICENSE) for details.

## Citation

```bibtex
@software{nlm2026,
  title={NLM: Nature Learning Model — Grounded Sensory World Model},
  author={Mycosoft Labs},
  year={2026},
  url={https://github.com/MycosoftLabs/NLM}
}
```
