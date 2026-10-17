"""
Efficient attention operators used in the LowRes-EffTransNAS search space.
Supports: softmax (standard), linear (kernel-based), sparse-local (window + global).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SoftmaxAttention(nn.Module):
    """Standard multi-head self-attention with scaled dot-product."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """
    Linear attention following Katharopoulos et al. (Transformers are RNNs).
    Uses feature map φ(x) = elu(x) + 1 for positive features.
    Complexity O(T) instead of O(T²).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Apply padding mask to keys/values if present
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True = pad
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(-1).float()  # (B, 1, T, 1)
            k = k * mask
            v = v * mask

        # Linear attention: (Q K^T) V  ≈  Q (K^T V)  with feature maps
        kv = torch.einsum("bhtd,bhte->bhde", k, v)  # (B, H, D, D)
        z = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhtd,bhde,bht->bhte", q, kv, z)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(out)
        return self.out_proj(out)


class SparseLocalAttention(nn.Module):
    """
    Sparse-local attention: fixed local window of size W plus one global token
    (the first token is treated as global).
    Approximate complexity O(T * W).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build sparse attention mask: local window + global (token 0)
        # For simplicity we use a dense mask and mask out non-attended positions.
        # In production one would use a true sparse implementation.
        device = x.device
        mask = torch.full((T, T), float("-inf"), device=device)

        # Local window
        for i in range(T):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0

        # Global token (position 0 attends to all, all attend to position 0)
        mask[:, 0] = 0.0
        mask[0, :] = 0.0

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


def build_attention(
    attn_type: str,
    d_model: int,
    n_heads: int,
    dropout: float = 0.1,
    window_size: int = 32,
) -> nn.Module:
    """Factory for attention modules."""
    attn_type = attn_type.lower()
    if attn_type == "softmax":
        return SoftmaxAttention(d_model, n_heads, dropout)
    elif attn_type == "linear":
        return LinearAttention(d_model, n_heads, dropout)
    elif attn_type in ("sparse-local", "sparse_local", "sparse"):
        return SparseLocalAttention(d_model, n_heads, window_size, dropout)
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")
