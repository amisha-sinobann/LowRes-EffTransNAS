"""
Super-network pre-training with progressive shrinkage and
downstream knowledge-distillation fine-tuning.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models.transformer import LowResTransformer
from nas.search_space import SearchSpace
from training.progressive_shrinkage import ProgressiveShrinkage


def create_optimizer(model: nn.Module, lr: float = 5e-4, weight_decay: float = 0.01):
    return AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def create_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


class SuperNetTrainer:
    """
    Trains the weight-sharing super-network with progressive shrinkage.
    Objective: masked language modeling on a mixture of Wikipedia + Common Crawl.
    """

    def __init__(
        self,
        model: LowResTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        max_epochs: int = 120,
        lr: float = 5e-4,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        log_interval: int = 50,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = create_optimizer(model, lr=lr)
        self.scheduler = create_scheduler(self.optimizer, warmup_epochs, max_epochs)

        self.space = SearchSpace()
        self.shrinkage = ProgressiveShrinkage(
            max_depth=model.encoder.max_depth,
            max_dim=model.encoder.max_dim,
            total_epochs=max_epochs,
            shrink_epochs=80,
        )

    def _mask_tokens(
        self, input_ids: torch.Tensor, mlm_prob: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple random masking for MLM (15 %)."""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
        # Do not mask special tokens (assume 0 = pad, 1 = cls, 2 = sep – adjust as needed)
        special = (input_ids == 0) | (input_ids == 1) | (input_ids == 2)
        probability_matrix.masked_fill_(special, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80 % [MASK], 10 % random, 10 % original
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool()
            & masked_indices
        )
        input_ids = input_ids.clone()
        input_ids[indices_replaced] = 3  # assume 3 is [MASK] id – change for your vocab

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            4, self.model.encoder.vocab_size, labels.shape, device=input_ids.device
        )
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, labels

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Progressive shrinkage: sample a sub-net
            config = self.shrinkage.sample_subnet_config(epoch, self.space)
            self.model.set_config(config)

            # MLM masking
            masked_ids, labels = self._mask_tokens(input_ids)

            self.optimizer.zero_grad()
            outputs = self.model(
                masked_ids,
                attention_mask=attention_mask,
                labels=labels,
                task="mlm",
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % self.log_interval == 0:
                print(
                    f"  Epoch {epoch} step {step+1}/{len(self.train_loader)} "
                    f"loss={loss.item():.4f} config={config}"
                )

        self.scheduler.step()
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        self.model.set_config(None)  # full network
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            masked_ids, labels = self._mask_tokens(input_ids)
            outputs = self.model(
                masked_ids, attention_mask=attention_mask, labels=labels, task="mlm"
            )
            total_loss += outputs["loss"].item()
            n += 1
        return total_loss / max(1, n)

    def train(self):
        best_val = float("inf")
        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()
            print(
                f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )
            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(self.save_dir, f"supernet_epoch{epoch}.pt"))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    ckpt, os.path.join(self.save_dir, "supernet_best.pt")
                )
        return self.model


class DistillationTrainer:
    """
    Knowledge-distillation fine-tuning of a discovered architecture
    using the super-network as teacher (temperature 2.0, 10 epochs).
    """

    def __init__(
        self,
        student: LowResTransformer,
        teacher: LowResTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        epochs: int = 10,
        lr: float = 2e-5,
        temperature: float = 2.0,
        alpha: float = 0.5,  # weight of KD loss vs hard labels
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = create_optimizer(student, lr=lr)

    def train_epoch(self) -> float:
        self.student.train()
        total = 0.0
        n = 0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)

            # Teacher soft labels
            with torch.no_grad():
                t_out = self.teacher(
                    input_ids, attention_mask=attention_mask, task="classification"
                )
                t_logits = t_out["logits"]

            s_out = self.student(
                input_ids, attention_mask=attention_mask, labels=labels, task="classification"
            )
            s_logits = s_out["logits"]

            # KD loss
            soft_loss = F.kl_div(
                F.log_softmax(s_logits / self.temperature, dim=-1),
                F.softmax(t_logits / self.temperature, dim=-1),
                reduction="batchmean",
            ) * (self.temperature ** 2)

            hard_loss = s_out["loss"] if s_out["loss"] is not None else 0.0
            loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(1, n)

    def train(self):
        for ep in range(self.epochs):
            loss = self.train_epoch()
            print(f"Distill epoch {ep}: loss={loss:.4f}")
        return self.student
