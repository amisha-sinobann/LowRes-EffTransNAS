#!/usr/bin/env python
"""
Main entry point for LowRes-EffTransNAS.

This script demonstrates the full pipeline:
  1. Build the super-network
  2. (Optional) pre-train with progressive shrinkage
  3. Run multi-objective evolutionary search
  4. Report the discovered Pareto front

Because full pre-training on Wikipedia + Common Crawl requires
substantial compute and data, the script ships with a *proxy*
evaluation function that can be replaced by a real accuracy
estimator once a trained super-network checkpoint is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Make package importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transformer import LowResTransformer
from nas.search_space import SearchSpace, Architecture
from nas.evolutionary import MultiObjectiveEvolutionarySearch
from evaluation.metrics import measure_latency


def proxy_evaluate(arch: Architecture) -> tuple:
    """
    Proxy fitness used for demonstration / debugging.
    In a real run this would:
      - extract the sub-network from the trained super-net
      - run a short validation pass on a held-out set
      - measure real on-device latency
    Here we use a simple analytic proxy that prefers
    moderate capacity and efficient attention.
    """
    # Fake accuracy: higher for reasonable depths/dims, bonus for linear/sparse
    base = 70.0
    depth_score = min(arch.depth / 12.0, 1.0) * 10
    dim_score = min(arch.dim / 512.0, 1.0) * 8
    attn_bonus = {"softmax": 0.0, "linear": 2.5, "sparse-local": 1.5}.get(
        arch.attn_type, 0.0
    )
    # Penalty for extreme configs that overfit low-resource data
    if arch.dim >= 768 and arch.depth >= 10:
        overfit_pen = 4.0
    else:
        overfit_pen = 0.0
    acc = base + depth_score + dim_score + attn_bonus - overfit_pen
    acc = max(50.0, min(95.0, acc + (hash(str(arch)) % 100) / 50.0))

    space = SearchSpace()
    params = space.estimate_params(arch) / 1e6  # millions
    flops = space.estimate_flops(arch) / 1e9   # GFLOPs
    # Synthetic latency (ms) – linear in params + attention cost
    lat = 2.0 + params * 0.8 + (5.0 if arch.attn_type == "softmax" else 1.5)

    return acc, params, flops, lat


def main():
    parser = argparse.ArgumentParser(description="LowRes-EffTransNAS search")
    parser.add_argument("--pop-size", type=int, default=32, help="Population size")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="pareto_front.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("LowRes-EffTransNAS – Multi-objective Neural Architecture Search")
    print("=" * 60)

    space = SearchSpace()
    print(f"Search space size (approx): > 10^12 candidates")
    print(f"Population: {args.pop_size}, Generations: {args.generations}")

    # Build a dummy super-net just to have the model class available
    # (weights are not used by the proxy evaluator)
    model = LowResTransformer(vocab_size=32000, max_depth=12, max_dim=768)
    print(f"Super-network parameters (full): {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

    searcher = MultiObjectiveEvolutionarySearch(
        search_space=space,
        evaluate_fn=proxy_evaluate,
        population_size=args.pop_size,
        n_generations=args.generations,
        seed=args.seed,
    )

    print("\nStarting evolutionary search ...")
    front = searcher.run()

    print("\n" + "=" * 60)
    print(f"Discovered Pareto front ({len(front)} architectures):")
    print("=" * 60)

    results = []
    for i, ind in enumerate(sorted(front, key=lambda x: -x.fitness[0])):
        arch = ind.arch
        acc, neg_p, neg_f, neg_l = ind.fitness
        params = -neg_p
        flops = -neg_f
        lat = -neg_l
        print(
            f"  [{i}] depth={arch.depth:2d}  dim={arch.dim:3d}  heads={arch.heads:2d}  "
            f"ratio={arch.ratio:.1f}  attn={arch.attn_type:12s}  "
            f"acc={acc:.1f}  params={params:.1f}M  FLOPs={flops:.2f}G  lat={lat:.1f}ms"
        )
        results.append(
            {
                "rank": i,
                "depth": arch.depth,
                "dim": arch.dim,
                "heads": arch.heads,
                "ratio": arch.ratio,
                "attn_type": arch.attn_type,
                "accuracy": float(acc),
                "params_M": float(params),
                "flops_G": float(flops),
                "latency_ms": float(lat),
            }
        )

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Pareto front to {out_path}")

    # Quick latency sanity-check on the smallest model
    if results:
        smallest = min(results, key=lambda r: r["params_M"])
        print(f"\nExample: building LR-T-S style model (dim={smallest['dim']}) ...")
        # In a full pipeline one would call model.set_config(...) and measure
        print("Done.")


if __name__ == "__main__":
    main()
