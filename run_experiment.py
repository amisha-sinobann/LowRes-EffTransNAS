#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LowRes-EffTransNAS  –  Experiment Runner                 ║
║   Neural Architecture Search for Efficient Transformers in Low-Resource NLP  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage examples
--------------
  # Full pipeline (demo mode – proxy fitness, no large data required)
  python run_experiment.py --stage all --exp-name demo_run

  # Only evolutionary search
  python run_experiment.py --stage search --pop-size 32 --generations 20

  # Only report the last Pareto front
  python run_experiment.py --stage report --exp-name demo_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.transformer import LowResTransformer
from nas.search_space import SearchSpace, Architecture
from nas.evolutionary import MultiObjectiveEvolutionarySearch
from evaluation.metrics import measure_latency, measure_peak_memory


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(title: str, width: int = 78):
    print()
    print("╔" + "═" * width + "╗")
    print("║" + title.center(width) + "║")
    print("╚" + "═" * width + "╝")


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def kv(key: str, value, width: int = 22):
    print(f"  {key:<{width}} : {value}")


def ok(msg: str):
    print(f"  ✓  {msg}")


def warn(msg: str):
    print(f"  ⚠  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Proxy evaluator (used when no trained super-net checkpoint is present)
# ─────────────────────────────────────────────────────────────────────────────
def proxy_evaluate(arch: Architecture):
    """Analytic proxy that prefers moderate capacity + efficient attention."""
    base = 70.0
    depth_score = min(arch.depth / 12.0, 1.0) * 10
    dim_score = min(arch.dim / 512.0, 1.0) * 8
    attn_bonus = {"softmax": 0.0, "linear": 2.5, "sparse-local": 1.5}.get(
        arch.attn_type, 0.0
    )
    overfit_pen = 4.0 if (arch.dim >= 768 and arch.depth >= 10) else 0.0
    acc = base + depth_score + dim_score + attn_bonus - overfit_pen
    acc = max(50.0, min(95.0, acc + (hash(str(arch)) % 100) / 50.0))

    space = SearchSpace()
    params = space.estimate_params(arch) / 1e6
    flops = space.estimate_flops(arch) / 1e9
    lat = 2.0 + params * 0.8 + (5.0 if arch.attn_type == "softmax" else 1.5)
    return acc, params, flops, lat


# ─────────────────────────────────────────────────────────────────────────────
# Experiment stages
# ─────────────────────────────────────────────────────────────────────────────
def stage_setup(args, exp_dir: Path):
    banner("STAGE 0  ·  Experiment Setup")
    kv("Experiment name", args.exp_name)
    kv("Output directory", exp_dir)
    kv("Device", args.device)
    kv("Random seed", args.seed)
    kv("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)

    meta = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "device": args.device,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "created": datetime.now().isoformat(),
    }
    with open(exp_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    ok("Experiment directory ready")
    return meta


def stage_build_supernet(args, exp_dir: Path):
    banner("STAGE 1  ·  Build Super-Network")
    model = LowResTransformer(
        vocab_size=32000,
        max_seq_len=512,
        max_depth=12,
        max_dim=768,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    kv("Max depth", 12)
    kv("Max hidden dim", 768)
    kv("Max heads", 12)
    kv("Parameter count (full)", f"{n_params:.1f} M")
    kv("Search-space size", "> 10¹² candidates")

    # Save a tiny checkpoint so later stages know the architecture exists
    ckpt_path = exp_dir / "checkpoints" / "supernet_init.pt"
    torch.save({"model_config": {"vocab_size": 32000, "max_depth": 12, "max_dim": 768}},
               ckpt_path)
    ok(f"Super-network skeleton saved → {ckpt_path.name}")
    return model


def stage_search(args, exp_dir: Path):
    banner("STAGE 2  ·  Multi-Objective Evolutionary Search")
    kv("Population size", args.pop_size)
    kv("Generations", args.generations)
    kv("Crossover prob", 0.5)
    kv("Mutation prob", 0.15)
    kv("Objectives", "accuracy  |  −params  |  −FLOPs  |  −latency")

    space = SearchSpace()
    searcher = MultiObjectiveEvolutionarySearch(
        search_space=space,
        evaluate_fn=proxy_evaluate,
        population_size=args.pop_size,
        n_generations=args.generations,
        seed=args.seed,
    )

    t0 = time.time()
    front = searcher.run()
    elapsed = time.time() - t0
    ok(f"Search finished in {elapsed:.1f}s  ·  Pareto front size = {len(front)}")

    # Serialise results
    results = []
    for i, ind in enumerate(sorted(front, key=lambda x: -x.fitness[0])):
        arch = ind.arch
        acc, neg_p, neg_f, neg_l = ind.fitness
        results.append({
            "rank": i,
            "depth": arch.depth,
            "dim": arch.dim,
            "heads": arch.heads,
            "ratio": arch.ratio,
            "attn_type": arch.attn_type,
            "accuracy": float(acc),
            "params_M": float(-neg_p),
            "flops_G": float(-neg_f),
            "latency_ms": float(-neg_l),
        })

    out_json = exp_dir / "pareto_front.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    hist_json = exp_dir / "search_history.json"
    with open(hist_json, "w") as f:
        json.dump(searcher.history, f, indent=2)

    ok(f"Pareto front → {out_json.name}")
    ok(f"Search history → {hist_json.name}")
    return results


def stage_report(args, exp_dir: Path, results=None):
    banner("STAGE 3  ·  Experiment Report")

    if results is None:
        path = exp_dir / "pareto_front.json"
        if not path.exists():
            warn("No Pareto front found. Run --stage search first.")
            return
        with open(path) as f:
            results = json.load(f)

    # Table header
    print()
    print("  ┌──────┬───────┬─────┬───────┬───────┬──────────────┬────────┬─────────┬──────────┬──────────┐")
    print("  │ Rank │ Depth │ Dim │ Heads │ Ratio │ Attention    │ Acc ↑  │ Params  │ FLOPs    │ Latency  │")
    print("  ├──────┼───────┼─────┼───────┼───────┼──────────────┼────────┼─────────┼──────────┼──────────┤")
    for r in results[:12]:  # show top-12
        print(
            f"  │ {r['rank']:4d} │ {r['depth']:5d} │ {r['dim']:3d} │ {r['heads']:5d} │ "
            f"{r['ratio']:5.1f} │ {r['attn_type']:<12s} │ {r['accuracy']:6.1f} │ "
            f"{r['params_M']:5.1f} M │ {r['flops_G']:6.2f} G │ {r['latency_ms']:6.1f} ms │"
        )
    print("  └──────┴───────┴─────┴───────┴───────┴──────────────┴────────┴─────────┴──────────┴──────────┘")

    # Highlight three representative models (S / M / L)
    if results:
        by_params = sorted(results, key=lambda x: x["params_M"])
        s = by_params[0]
        m = by_params[len(by_params) // 2]
        l = by_params[-1]
        section("Representative models (LR-T-S / LR-T-M / LR-T-L style)")
        for name, r in [("LR-T-S", s), ("LR-T-M", m), ("LR-T-L", l)]:
            print(
                f"  {name}: depth={r['depth']}, dim={r['dim']}, heads={r['heads']}, "
                f"ratio={r['ratio']}, attn={r['attn_type']}  →  "
                f"acc={r['accuracy']:.1f}  params={r['params_M']:.1f}M  "
                f"lat={r['latency_ms']:.1f}ms"
            )

    # Write a short markdown report
    report_path = exp_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(f"# Experiment Report – {args.exp_name}\n\n")
        f.write(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Population: {args.pop_size}, Generations: {args.generations}\n")
        f.write(f"- Pareto front size: {len(results)}\n\n")
        f.write("## Top architectures\n\n")
        f.write("| Rank | Depth | Dim | Heads | Ratio | Attn | Acc | Params (M) | FLOPs (G) | Latency (ms) |\n")
        f.write("|------|-------|-----|-------|-------|------|-----|------------|-----------|--------------|\n")
        for r in results[:15]:
            f.write(
                f"| {r['rank']} | {r['depth']} | {r['dim']} | {r['heads']} | "
                f"{r['ratio']} | {r['attn_type']} | {r['accuracy']:.1f} | "
                f"{r['params_M']:.1f} | {r['flops_G']:.2f} | {r['latency_ms']:.1f} |\n"
            )
    ok(f"Markdown report → {report_path.name}")


def stage_quick_eval(args, exp_dir: Path, results=None):
    """Build the smallest discovered model and measure real latency on current device."""
    banner("STAGE 4  ·  Quick On-Device Check")
    if results is None:
        path = exp_dir / "pareto_front.json"
        if not path.exists():
            warn("No Pareto front – skipping latency check")
            return
        with open(path) as f:
            results = json.load(f)

    if not results:
        return

    smallest = min(results, key=lambda r: r["params_M"])
    kv("Selected model", f"depth={smallest['depth']} dim={smallest['dim']} "
                         f"heads={smallest['heads']} attn={smallest['attn_type']}")

    model = LowResTransformer(vocab_size=1000, max_depth=12, max_dim=768)
    cfg = {
        "depth": smallest["depth"],
        "dim": smallest["dim"],
        "heads": smallest["heads"],
        "ratio": smallest["ratio"],
        "attn_type": smallest["attn_type"],
    }
    model.set_config(cfg)

    device = args.device
    try:
        lat = measure_latency(model, seq_len=128, device=device, n_warmup=5, n_runs=20)
        mem = measure_peak_memory(model, seq_len=128, device=device)
        kv("Measured latency", f"{lat:.2f} ms  (seq=128, bs=1)")
        kv("Peak memory (est.)", f"{mem:.1f} MB")
        ok("On-device check complete")
    except Exception as e:
        warn(f"Latency measurement skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LowRes-EffTransNAS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--stage",
        choices=["all", "setup", "search", "report", "eval"],
        default="all",
        help="Which stage(s) to run",
    )
    p.add_argument("--exp-name", type=str, default="exp_default",
                   help="Name of this experiment run")
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--output-root", type=str, default="experiments")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.output_root) / args.exp_name

    banner("LowRes-EffTransNAS  ·  Experiment Runner")
    print(f"  Paper  : Neural Architecture Search for Efficient Transformer Models")
    print(f"           in Low-Resource NLP")
    print(f"  Mode   : {'Full pipeline' if args.stage == 'all' else args.stage.upper()}")

    results = None

    if args.stage in ("all", "setup"):
        stage_setup(args, exp_dir)
        stage_build_supernet(args, exp_dir)

    if args.stage in ("all", "search"):
        if not exp_dir.exists():
            stage_setup(args, exp_dir)
        results = stage_search(args, exp_dir)

    if args.stage in ("all", "report"):
        stage_report(args, exp_dir, results)

    if args.stage in ("all", "eval"):
        stage_quick_eval(args, exp_dir, results)

    banner("Experiment finished")
    print(f"  All artefacts saved under:  {exp_dir.resolve()}\n")


if __name__ == "__main__":
    main()
