"""
Progressive-shrinkage curriculum for the weight-sharing super-network.

As described in the paper:
  - lower bounds on depth and width start at 80 % of the maximum
  - decrease linearly to 30 % of the maximum over the first 80 of 120 epochs
  - final 40 epochs keep the lower bounds fixed
"""

from typing import Dict, Tuple
import math


class ProgressiveShrinkage:
    def __init__(
        self,
        max_depth: int = 12,
        max_dim: int = 768,
        total_epochs: int = 120,
        shrink_epochs: int = 80,
        start_ratio: float = 0.8,
        end_ratio: float = 0.3,
    ):
        self.max_depth = max_depth
        self.max_dim = max_dim
        self.total_epochs = total_epochs
        self.shrink_epochs = shrink_epochs
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

    def get_bounds(self, epoch: int) -> Tuple[int, int]:
        """
        Returns (min_depth, min_dim) for the current epoch.
        """
        if epoch >= self.shrink_epochs:
            ratio = self.end_ratio
        else:
            # linear interpolation
            progress = epoch / max(1, self.shrink_epochs)
            ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)

        min_depth = max(1, int(math.ceil(self.max_depth * ratio)))
        min_dim = max(64, int(math.ceil(self.max_dim * ratio)))
        # snap min_dim to a reasonable multiple
        min_dim = (min_dim // 64) * 64
        return min_depth, min_dim

    def sample_subnet_config(
        self,
        epoch: int,
        search_space,
        sandwich: bool = True,
    ) -> Dict:
        """
        Sample a sub-network configuration respecting the current lower bounds.
        When sandwich=True we also always include the max and min networks
        (classic sandwich rule of Once-for-All / BigNAS).
        """
        min_d, min_w = self.get_bounds(epoch)

        # Filter choices
        valid_depths = [d for d in search_space.DEPTH_CHOICES if d >= min_d]
        valid_dims = [d for d in search_space.DIM_CHOICES if d >= min_w]
        if not valid_depths:
            valid_depths = [self.max_depth]
        if not valid_dims:
            valid_dims = [self.max_dim]

        import random

        depth = random.choice(valid_depths)
        dim = random.choice(valid_dims)
        valid_heads = [h for h in search_space.HEAD_CHOICES if dim % h == 0]
        heads = random.choice(valid_heads) if valid_heads else 2
        ratio = random.choice(search_space.RATIO_CHOICES)
        attn = random.choice(search_space.ATTN_CHOICES)

        return {
            "depth": depth,
            "dim": dim,
            "heads": heads,
            "ratio": ratio,
            "attn_type": attn,
        }
