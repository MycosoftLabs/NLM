"""Bounded feature training; no full NLM or production worker qualification."""

from .contracts import ENGINE_ID, FEATURE_SCHEMA_SHA256, ContractError, validate_manifest
from .engine import TrainingResult, predict, train

__all__ = ["ENGINE_ID", "FEATURE_SCHEMA_SHA256", "ContractError", "validate_manifest", "TrainingResult", "predict", "train"]
