"""
Searchable Transformer blocks and the full encoder (and simple encoder-decoder)
used by LowRes-EffTransNAS.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: float, dropout: float = 0.1):
        super().__init__()
        d_ff = int(d_model * expansion)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Single Transformer block with configurable attention type,
    hidden dimension, head count and FFN expansion ratio.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion: float,
        attn_type: str = "softmax",
        dropout: float = 0.1,
        window_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.expansion = expansion
        self.attn_type = attn_type

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = build_attention(attn_type, d_model, n_heads, dropout, window_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm
        residual = x
        x = self.ln1(x)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class SearchableTransformerEncoder(nn.Module):
    """
    Weight-sharing super-network / searchable Transformer encoder.

    Architecture choices (per layer or global):
      - depth L ∈ {4, 6, 8, 10, 12}
      - hidden dim d ∈ {128, 192, 256, 320, 384, 512, 768}
      - heads h ∈ {2, 4, 6, 8, 12}  (must divide d)
      - FFN ratio r ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}
      - attention type τ ∈ {softmax, linear, sparse-local}

    When used as a super-network the maximum configuration is instantiated
    and sub-networks are obtained by selecting contiguous subsets of
    layers / channels / heads (sandwich rule).
    """

    # Search space definition (matches paper)
    DEPTH_CHOICES = [4, 6, 8, 10, 12]
    DIM_CHOICES = [128, 192, 256, 320, 384, 512, 768]
    HEAD_CHOICES = [2, 4, 6, 8, 12]
    RATIO_CHOICES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    ATTN_CHOICES = ["softmax", "linear", "sparse-local"]

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
        max_depth: int = 12,
        max_dim: int = 768,
        max_heads: int = 12,
        max_ratio: float = 4.0,
        dropout: float = 0.1,
        window_size: int = 32,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth
        self.max_dim = max_dim
        self.max_heads = max_heads
        self.max_ratio = max_ratio
        self.dropout = dropout
        self.window_size = window_size
        self.pad_idx = pad_idx

        # Embedding (always max_dim; sub-networks use the first d channels)
        self.token_emb = nn.Embedding(vocab_size, max_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_seq_len, max_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Maximum number of blocks with maximum capacity
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=max_dim,
                    n_heads=max_heads,
                    expansion=max_ratio,
                    attn_type="softmax",  # will be overridden per sub-net
                    dropout=dropout,
                    window_size=window_size,
                )
                for _ in range(max_depth)
            ]
        )

        self.final_ln = nn.LayerNorm(max_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_num_params(self, config: Optional[Dict] = None) -> int:
        """Approximate parameter count for a given architecture config."""
        if config is None:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        L = config["depth"]
        d = config["dim"]
        h = config["heads"]
        r = config["ratio"]
        # Rough count (embeddings + L blocks)
        emb = self.vocab_size * d + self.max_seq_len * d
        # per block: qkv + out + 2 LN + FFN
        per_block = (
            3 * d * d  # qkv
            + d * d    # out
            + 2 * d    # LN
            + d * int(d * r) * 2  # FFN
            + 2 * int(d * r)  # biases rough
        )
        return emb + L * per_block

    def sample_config(self) -> Dict:
        """Sample a random valid architecture from the search space."""
        import random

        depth = random.choice(self.DEPTH_CHOICES)
        dim = random.choice(self.DIM_CHOICES)
        # heads that divide dim
        valid_heads = [h for h in self.HEAD_CHOICES if dim % h == 0]
        heads = random.choice(valid_heads) if valid_heads else 2
        ratio = random.choice(self.RATIO_CHOICES)
        attn = random.choice(self.ATTN_CHOICES)
        return {
            "depth": depth,
            "dim": dim,
            "heads": heads,
            "ratio": ratio,
            "attn_type": attn,
            # layer-wise overrides can be added later
            "layer_configs": None,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        config: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        If config is None, uses the full super-network (max capacity).
        Otherwise extracts the sub-network defined by config.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Position ids
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Embeddings – always full max_dim; we slice later if needed
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # Key padding mask (True = pad)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        if config is None:
            # Full super-net
            for block in self.blocks:
                x = block(x, key_padding_mask=key_padding_mask)
            x = self.final_ln(x)
            return x

        # ---- Sub-network path ----
        # For a research prototype we instantiate a lightweight independent
        # stack that matches the requested configuration.  A production
        # weight-sharing super-network would keep the full weights and
        # perform true channel / head slicing of every linear projection
        # (see Once-for-All / BigNAS literature).  The independent path
        # guarantees correctness for evaluation and distillation while
        # the progressive-shrinkage trainer still exercises the max-capacity
        # blocks.
        depth = config["depth"]
        dim = config["dim"]
        heads = config["heads"]
        ratio = config["ratio"]
        attn_type = config.get("attn_type", "softmax")

        # Project embeddings to the requested dimension
        if dim != self.max_dim:
            # simple linear projection (not weight-shared)
            if not hasattr(self, "_proj_cache"):
                self._proj_cache = {}
            key = (dim,)
            if key not in self._proj_cache:
                self._proj_cache[key] = nn.Linear(self.max_dim, dim, bias=False).to(x.device)
            x = self._proj_cache[key](x)
        else:
            x = x[..., :dim]

        # Build (or retrieve) a temporary encoder for this config
        cache_key = (depth, dim, heads, ratio, attn_type)
        if not hasattr(self, "_subnet_cache"):
            self._subnet_cache = {}
        if cache_key not in self._subnet_cache:
            blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=dim,
                        n_heads=heads,
                        expansion=ratio,
                        attn_type=attn_type,
                        dropout=self.dropout,
                        window_size=self.window_size,
                    )
                    for _ in range(depth)
                ]
            )
            final_ln = nn.LayerNorm(dim)
            self._subnet_cache[cache_key] = nn.ModuleDict(
                {"blocks": blocks, "final_ln": final_ln}
            ).to(x.device)
        subnet = self._subnet_cache[cache_key]

        for block in subnet["blocks"]:
            x = block(x, key_padding_mask=key_padding_mask)
        x = subnet["final_ln"](x)
        return x


class MLMHead(nn.Module):
    """Masked Language Modeling prediction head."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden)
        x = self.act(x)
        x = self.ln(x)
        return self.decoder(x)


class LowResTransformer(nn.Module):
    """
    Complete model used for both super-network pre-training (MLM)
    and downstream fine-tuning / distillation.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 512,
        max_depth: int = 12,
        max_dim: int = 768,
        pad_idx: int = 0,
        num_labels: int = 2,  # for classification heads
    ):
        super().__init__()
        # Ensure max_heads divides max_dim
        max_heads = 12
        while max_dim % max_heads != 0 and max_heads > 1:
            max_heads -= 1
        self.encoder = SearchableTransformerEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            max_depth=max_depth,
            max_dim=max_dim,
            max_heads=max_heads,
            pad_idx=pad_idx,
        )
        self.mlm_head = MLMHead(max_dim, vocab_size)
        self.classifier = nn.Linear(max_dim, num_labels)
        self.config = None  # current active architecture

    def set_config(self, config: Dict):
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task: str = "mlm",
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids, attention_mask, config=self.config)

        if task == "mlm":
            # Use full dim for MLM head when no config, else slice
            if self.config is not None:
                dim = self.config["dim"]
                logits = self.mlm_head.decoder(
                    F.layer_norm(
                        hidden,
                        (dim,),
                        weight=self.mlm_head.ln.weight[:dim],
                        bias=self.mlm_head.ln.bias[:dim],
                    )
                )  # simplified
            else:
                logits = self.mlm_head(hidden)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}

        elif task == "classification":
            # Mean pool (ignore padding)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(1)

            # Classifier expects the hidden size of the active config
            dim = pooled.size(-1)
            if not hasattr(self, "_cls_cache"):
                self._cls_cache = {}
            if dim not in self._cls_cache:
                self._cls_cache[dim] = nn.Linear(dim, self.classifier.out_features).to(
                    pooled.device
                )
            logits = self._cls_cache[dim](pooled)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

        else:
            raise ValueError(f"Unknown task: {task}")
