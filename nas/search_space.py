"""
Search space definition and architecture encoding for LowRes-EffTransNAS.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import random
import copy


@dataclass
class Architecture:
    """A concrete architecture candidate."""

    depth: int
    dim: int
    heads: int
    ratio: float
    attn_type: str
    # Optional per-layer overrides (list of length `depth`)
    layer_attn: Optional[List[str]] = None
    layer_dim: Optional[List[int]] = None
    layer_heads: Optional[List[int]] = None
    layer_ratio: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Keep only the fields expected by the model
        return {
            "depth": self.depth,
            "dim": self.dim,
            "heads": self.heads,
            "ratio": self.ratio,
            "attn_type": self.attn_type,
            "layer_configs": None,
        }

    def __hash__(self):
        return hash(
            (
                self.depth,
                self.dim,
                self.heads,
                self.ratio,
                self.attn_type,
                tuple(self.layer_attn) if self.layer_attn else None,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Architecture):
            return False
        return (
            self.depth == other.depth
            and self.dim == other.dim
            and self.heads == other.heads
            and self.ratio == other.ratio
            and self.attn_type == other.attn_type
        )


class SearchSpace:
    """Defines the discrete search space of LowRes-EffTransNAS."""

    DEPTH_CHOICES = [4, 6, 8, 10, 12]
    DIM_CHOICES = [128, 192, 256, 320, 384, 512, 768]
    HEAD_CHOICES = [2, 4, 6, 8, 12]
    RATIO_CHOICES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    ATTN_CHOICES = ["softmax", "linear", "sparse-local"]

    def __init__(self):
        self._validate_choices()

    def _validate_choices(self):
        for d in self.DIM_CHOICES:
            valid = [h for h in self.HEAD_CHOICES if d % h == 0]
            if not valid:
                raise ValueError(f"No valid head count for dim={d}")

    def sample(self) -> Architecture:
        """Uniform random sample of a valid architecture."""
        depth = random.choice(self.DEPTH_CHOICES)
        dim = random.choice(self.DIM_CHOICES)
        valid_heads = [h for h in self.HEAD_CHOICES if dim % h == 0]
        heads = random.choice(valid_heads)
        ratio = random.choice(self.RATIO_CHOICES)
        attn = random.choice(self.ATTN_CHOICES)
        return Architecture(
            depth=depth,
            dim=dim,
            heads=heads,
            ratio=ratio,
            attn_type=attn,
        )

    def mutate(self, arch: Architecture, prob: float = 0.15) -> Architecture:
        """
        Mutation operator described in the paper:
        independently redraw each of the four block-level choices
        with probability `prob`.
        """
        new = copy.deepcopy(arch)

        if random.random() < prob:
            new.depth = random.choice(self.DEPTH_CHOICES)
        if random.random() < prob:
            new.dim = random.choice(self.DIM_CHOICES)
            # ensure heads still valid
            valid = [h for h in self.HEAD_CHOICES if new.dim % h == 0]
            if new.heads not in valid:
                new.heads = random.choice(valid)
        if random.random() < prob:
            valid = [h for h in self.HEAD_CHOICES if new.dim % h == 0]
            new.heads = random.choice(valid)
        if random.random() < prob:
            new.ratio = random.choice(self.RATIO_CHOICES)
        if random.random() < prob:
            new.attn_type = random.choice(self.ATTN_CHOICES)

        return new

    def crossover(
        self, parent1: Architecture, parent2: Architecture, prob: float = 0.5
    ) -> Tuple[Architecture, Architecture]:
        """
        Uniform crossover on the layer-wise configuration vectors
        (here simplified to the global choices).
        """
        c1 = copy.deepcopy(parent1)
        c2 = copy.deepcopy(parent2)

        if random.random() < prob:
            c1.depth, c2.depth = c2.depth, c1.depth
        if random.random() < prob:
            c1.dim, c2.dim = c2.dim, c1.dim
            # fix heads
            for c in (c1, c2):
                valid = [h for h in self.HEAD_CHOICES if c.dim % h == 0]
                if c.heads not in valid:
                    c.heads = random.choice(valid)
        if random.random() < prob:
            c1.heads, c2.heads = c2.heads, c1.heads
            for c in (c1, c2):
                valid = [h for h in self.HEAD_CHOICES if c.dim % h == 0]
                if c.heads not in valid:
                    c.heads = random.choice(valid)
        if random.random() < prob:
            c1.ratio, c2.ratio = c2.ratio, c1.ratio
        if random.random() < prob:
            c1.attn_type, c2.attn_type = c2.attn_type, c1.attn_type

        return c1, c2

    def estimate_params(self, arch: Architecture) -> int:
        """Rough parameter count (embeddings + blocks)."""
        d = arch.dim
        L = arch.depth
        r = arch.ratio
        vocab = 32000
        max_len = 512
        emb = vocab * d + max_len * d
        # qkv + out + 2*LN + FFN
        per_block = 4 * d * d + 2 * d + 2 * d * int(d * r)
        return emb + L * per_block

    def estimate_flops(self, arch: Architecture, seq_len: int = 128) -> float:
        """Very rough FLOPs estimate (MACs) for a single forward pass."""
        d = arch.dim
        L = arch.depth
        h = arch.heads
        r = arch.ratio
        T = seq_len

        # Attention: roughly 2 * T * T * d  for softmax, less for linear/sparse
        if arch.attn_type == "softmax":
            attn_flops = 2 * T * T * d
        elif arch.attn_type == "linear":
            attn_flops = 4 * T * d * d  # approximate
        else:  # sparse-local window 32
            W = 32
            attn_flops = 2 * T * W * d

        # FFN
        ffn_flops = 2 * T * d * int(d * r)

        # Total
        return L * (attn_flops + ffn_flops)
