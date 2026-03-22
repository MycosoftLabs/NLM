"""
NLM Trainer
============

Single-GPU training loop with:
- Mixed precision (torch.amp)
- Gradient accumulation
- Cyclical learning rate scheduling (matching natural cycles)
- Physics-informed loss
- Replay over RootedNatureFrames
- Early stopping
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from nlm.model.config import NLMConfig
from nlm.model.nlm_model import NatureLearningModel
from nlm.training.losses import NLMLoss


class NLMTrainer:
    """Training loop for the Nature Learning Model.

    Single-GPU, mixed precision, with physics-informed losses.
    """

    def __init__(
        self,
        model: NatureLearningModel,
        config: NLMConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "checkpoints",
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Cyclical LR schedule (matching natural cycles concept)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01,
        )

        # Loss
        self.criterion = NLMLoss()

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, float]] = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_losses: Dict[str, float] = {}
        num_batches = 0
        epoch_start = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                output = self.model(
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

                # Compute losses
                target = batch.get("next_state_target")
                if target is None:
                    target = torch.zeros(
                        batch["spatial_features"].size(0),
                        self.config.num_env_targets,
                        device=self.device,
                    )

                anomaly_target = batch.get("anomaly_target")

                losses = self.criterion(
                    next_state_pred=output.next_state,
                    next_state_target=target,
                    anomaly_scores=output.anomaly_scores,
                    anomaly_targets=anomaly_target,
                    ecological_impact=output.ecological_impact,
                    grounding_confidence=output.grounding_confidence,
                    growth_prediction=output.growth_prediction,
                )

                loss = losses["total"] / self.gradient_accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
                self.global_step += 1

            # Track losses
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                total_losses[key] = total_losses.get(key, 0.0) + val
            num_batches += 1

        # Average losses
        avg_losses = {k: v / max(1, num_batches) for k, v in total_losses.items()}
        avg_losses["epoch_time_s"] = time.time() - epoch_start
        avg_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.history.append(avg_losses)

        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output = self.model(
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

            target = batch.get("next_state_target", torch.zeros(
                batch["spatial_features"].size(0), self.config.num_env_targets, device=self.device,
            ))

            losses = self.criterion(
                next_state_pred=output.next_state,
                next_state_target=target,
                anomaly_scores=output.anomaly_scores,
            )

            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                total_losses[key] = total_losses.get(key, 0.0) + val
            num_batches += 1

        return {k: v / max(1, num_batches) for k, v in total_losses.items()}

    def fit(
        self,
        num_epochs: int = 100,
        patience: int = 10,
        save_every: int = 5,
    ) -> List[Dict[str, float]]:
        """Full training run with early stopping."""
        from nlm.training.checkpoint import save_checkpoint

        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            log_line = f"Epoch {epoch+1}/{num_epochs} | train_loss={train_losses['total']:.4f}"

            # Validate
            val_losses = self.validate()
            if val_losses:
                log_line += f" | val_loss={val_losses.get('total', 0):.4f}"

                # Early stopping
                val_total = val_losses.get("total", float("inf"))
                if val_total < self.best_val_loss:
                    self.best_val_loss = val_total
                    patience_counter = 0
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_total,
                        self.config, self.output_dir / "best.pt",
                    )
                else:
                    patience_counter += 1

            print(log_line)

            # Periodic save
            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    train_losses["total"], self.config,
                    self.output_dir / f"epoch_{epoch+1}.pt",
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return self.history
