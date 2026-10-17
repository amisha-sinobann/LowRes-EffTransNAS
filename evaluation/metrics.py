"""
Evaluation helpers: BLEU, accuracy, latency & memory measurement.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def classification_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            out = model(
                input_ids, attention_mask=attention_mask, task="classification"
            )
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def measure_latency(
    model: nn.Module,
    seq_len: int = 128,
    batch_size: int = 1,
    n_warmup: int = 20,
    n_runs: int = 100,
    device: str = "cpu",
) -> float:
    """
    Measure average inference latency (ms) on the given device.
    Paper uses a Pixel-class ARM CPU, single thread, seq=128, bs=1.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attn = torch.ones_like(dummy)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy, attention_mask=attn, task="classification")

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy, attention_mask=attn, task="classification")
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

    return sum(times) / len(times)


def measure_peak_memory(
    model: nn.Module,
    seq_len: int = 128,
    device: str = "cpu",
) -> float:
    """Return peak memory in MB (approximate for CPU)."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        dummy = torch.randint(0, 1000, (1, seq_len), device=device)
        with torch.no_grad():
            _ = model(dummy, task="classification")
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        # Rough estimate from parameter count
        params = sum(p.numel() for p in model.parameters())
        return params * 4 / (1024 ** 2)  # fp32


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """
    Wrapper around sacrebleu if available, otherwise a very simple
    token-overlap approximation (for environments without the package).
    """
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hypotheses, [references]).score
    except ImportError:
        # Fallback: unigram precision (not a real BLEU)
        scores = []
        for hyp, ref in zip(hypotheses, references):
            h = set(hyp.lower().split())
            r = set(ref.lower().split())
            if not h:
                scores.append(0.0)
            else:
                scores.append(len(h & r) / len(h) * 100)
        return sum(scores) / max(1, len(scores))
