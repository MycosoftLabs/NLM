"""
NLM Training — Physics-informed training infrastructure.

Training over RootedNatureFrames with:
- Next-state prediction
- Intervention/counterfactual learning
- Physics, temporal, spatial, ecological, causal consistency losses
- Real telemetry + synthetic augmentation
- Provenance-linked training examples via MINDEX
"""

from nlm.training.trainer import NLMTrainer
from nlm.training.losses import NLMLoss
from nlm.training.dataset import RootedFrameDataset
from nlm.training.checkpoint import save_checkpoint, load_checkpoint

__all__ = ["NLMTrainer", "NLMLoss", "RootedFrameDataset", "save_checkpoint", "load_checkpoint"]
