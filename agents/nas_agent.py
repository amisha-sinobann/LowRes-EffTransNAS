#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              LowRes-EffTransNAS  ·  Agentic Research Agent                   ║
║   Autonomous multi-stage NAS experiment agent with goal-driven planning      ║
╚══════════════════════════════════════════════════════════════════════════════╝

The agent:
  1. Observes the current experiment state
  2. Plans the next research actions
  3. Executes stages (search / ablate / analyse / report)
  4. Reflects on results and decides whether to continue or stop
  5. Writes a structured research log

Usage
-----
  python -m agents.nas_agent --goal "find efficient models under 15M params" \\
                             --max-steps 8 --exp-name agentic_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nas.search_space import SearchSpace, Architecture
from nas.evolutionary import MultiObjectiveEvolutionarySearch
from models.transformer import LowResTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Agent state & memory
# ─────────────────────────────────────────────────────────────────────────────
class Stage(str, Enum):
    SETUP = "setup"
    SEARCH = "search"
    ABLATE = "ablate"
    ANALYSE = "analyse"
    REPORT = "report"
    DONE = "done"


@dataclass
class Observation:
    stage: Stage
    pareto_size: int = 0
    best_acc: float = 0.0
    best_params: float = 999.0
    best_latency: float = 999.0
    hypervolume: float = 0.0
    generations_done: int = 0
    notes: str = ""


@dataclass
class Action:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class MemoryEntry:
    step: int
    observation: Dict
    action: Dict
    outcome: str
    timestamp: str


class ResearchMemory:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[MemoryEntry] = []
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
                self.entries = [MemoryEntry(**e) for e in raw]

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def summary(self) -> str:
        if not self.entries:
            return "No prior steps."
        lines = [f"Step {e.step}: {e.action.get('name')} → {e.outcome}" for e in self.entries[-5:]]
        return " | ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Proxy fitness (same as experiment runner)
# ─────────────────────────────────────────────────────────────────────────────
def proxy_evaluate(arch: Architecture):
    base = 70.0
    depth_score = min(arch.depth / 12.0, 1.0) * 10
    dim_score = min(arch.dim / 512.0, 1.0) * 8
    attn_bonus = {"softmax": 0.0, "linear": 2.5, "sparse-local": 1.5}.get(arch.attn_type, 0.0)
    overfit_pen = 4.0 if (arch.dim >= 768 and arch.depth >= 10) else 0.0
    acc = base + depth_score + dim_score + attn_bonus - overfit_pen
    acc = max(50.0, min(95.0, acc + (hash(str(arch)) % 100) / 50.0))
    space = SearchSpace()
    params = space.estimate_params(arch) / 1e6
    flops = space.estimate_flops(arch) / 1e9
    lat = 2.0 + params * 0.8 + (5.0 if arch.attn_type == "softmax" else 1.5)
    return acc, params, flops, lat


# ─────────────────────────────────────────────────────────────────────────────
# Agentic planner
# ─────────────────────────────────────────────────────────────────────────────
class NASAgent:
    """
    Goal-driven research agent for LowRes-EffTransNAS.

    The agent maintains a research goal (e.g. "models under 15M params with
    accuracy > 85"), observes experiment state, plans actions, executes them,
    and reflects until the goal is satisfied or max_steps is reached.
    """

    def __init__(
        self,
        goal: str,
        exp_dir: Path,
        max_steps: int = 10,
        pop_size: int = 24,
        generations: int = 12,
        seed: int = 42,
        device: str = "cpu",
        target_params_M: float = 15.0,
        target_acc: float = 85.0,
    ):
        self.goal = goal
        self.exp_dir = exp_dir
        self.max_steps = max_steps
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.device = device
        self.target_params_M = target_params_M
        self.target_acc = target_acc

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        self.memory = ResearchMemory(self.exp_dir / "agent_memory.json")
        self.results: List[Dict] = []
        self.history: List[Dict] = []
        self.step = 0

    # ── perception ──────────────────────────────────────────────────────────
    def observe(self) -> Observation:
        pareto_path = self.exp_dir / "pareto_front.json"
        if not pareto_path.exists():
            return Observation(stage=Stage.SETUP, notes="No Pareto front yet")

        with open(pareto_path) as f:
            self.results = json.load(f)

        if not self.results:
            return Observation(stage=Stage.SEARCH, notes="Empty front")

        best_acc = max(r["accuracy"] for r in self.results)
        best_params = min(r["params_M"] for r in self.results)
        best_lat = min(r["latency_ms"] for r in self.results)
        # simple hypervolume proxy
        hv = sum(r["accuracy"] * (1.0 / max(0.1, r["params_M"])) for r in self.results)

        # decide stage
        if self.step == 0:
            stage = Stage.SETUP
        elif len(self.results) < 5:
            stage = Stage.SEARCH
        elif not (self.exp_dir / "ablation.json").exists():
            stage = Stage.ABLATE
        elif not (self.exp_dir / "analysis.json").exists():
            stage = Stage.ANALYSE
        else:
            stage = Stage.REPORT

        return Observation(
            stage=stage,
            pareto_size=len(self.results),
            best_acc=best_acc,
            best_params=best_params,
            best_latency=best_lat,
            hypervolume=hv,
            generations_done=self.step * self.generations,
            notes=f"front={len(self.results)} best_acc={best_acc:.1f}",
        )

    # ── planning ────────────────────────────────────────────────────────────
    def plan(self, obs: Observation) -> Action:
        """Simple rule-based planner (can be replaced by an LLM call)."""
        # Goal check
        goal_met = (
            obs.best_acc >= self.target_acc
            and obs.best_params <= self.target_params_M
            and obs.pareto_size >= 5
        )
        if goal_met and obs.stage in (Stage.ANALYSE, Stage.REPORT, Stage.DONE):
            return Action("finish", reason="Research goal satisfied")

        if obs.stage == Stage.SETUP or obs.pareto_size == 0:
            return Action(
                "run_search",
                params={"pop_size": self.pop_size, "generations": self.generations},
                reason="No architectures discovered yet – start evolutionary search",
            )

        if obs.stage == Stage.SEARCH and obs.best_acc < self.target_acc:
            # intensify search
            return Action(
                "run_search",
                params={"pop_size": self.pop_size + 8, "generations": self.generations + 5},
                reason=f"Best acc {obs.best_acc:.1f} < target {self.target_acc} – intensify search",
            )

        if obs.stage == Stage.ABLATE:
            return Action("run_ablation", reason="Collect ablation statistics on attention types & depth")

        if obs.stage == Stage.ANALYSE:
            return Action("run_analysis", reason="Analyse Pareto trade-offs and architectural patterns")

        return Action("write_report", reason="Synthesise findings into final research report")

    # ── execution ───────────────────────────────────────────────────────────
    def execute(self, action: Action) -> str:
        name = action.name
        if name == "run_search":
            return self._exec_search(action.params)
        if name == "run_ablation":
            return self._exec_ablation()
        if name == "run_analysis":
            return self._exec_analysis()
        if name == "write_report":
            return self._exec_report()
        if name == "finish":
            return "GOAL_REACHED"
        return f"Unknown action: {name}"

    def _exec_search(self, params: Dict) -> str:
        pop = params.get("pop_size", self.pop_size)
        gens = params.get("generations", self.generations)
        print(f"  [Agent] Running evolutionary search  pop={pop}  gens={gens}")

        space = SearchSpace()
        searcher = MultiObjectiveEvolutionarySearch(
            search_space=space,
            evaluate_fn=proxy_evaluate,
            population_size=pop,
            n_generations=gens,
            seed=self.seed + self.step,
        )
        front = searcher.run()
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
        with open(self.exp_dir / "pareto_front.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(self.exp_dir / "search_history.json", "w") as f:
            json.dump(searcher.history, f, indent=2)
        self.results = results
        return f"Search done – front size {len(results)}, best_acc={results[0]['accuracy']:.1f}"

    def _exec_ablation(self) -> str:
        print("  [Agent] Running ablation analysis …")
        if not self.results:
            return "No results to ablate"
        # Group by attention type
        by_attn: Dict[str, List] = {}
        by_depth: Dict[int, List] = {}
        for r in self.results:
            by_attn.setdefault(r["attn_type"], []).append(r["accuracy"])
            by_depth.setdefault(r["depth"], []).append(r["accuracy"])

        ablation = {
            "attention_type": {
                k: {"mean_acc": float(np.mean(v)), "count": len(v)} for k, v in by_attn.items()
            },
            "depth": {
                str(k): {"mean_acc": float(np.mean(v)), "count": len(v)} for k, v in sorted(by_depth.items())
            },
        }
        with open(self.exp_dir / "ablation.json", "w") as f:
            json.dump(ablation, f, indent=2)
        return f"Ablation saved – attn types={list(by_attn.keys())}"

    def _exec_analysis(self) -> str:
        print("  [Agent] Analysing Pareto front …")
        if not self.results:
            return "No results"
        efficient = [r for r in self.results if r["params_M"] <= self.target_params_M]
        high_acc = [r for r in self.results if r["accuracy"] >= self.target_acc]
        sweet_spot = [r for r in self.results
                      if r["params_M"] <= self.target_params_M and r["accuracy"] >= self.target_acc]

        # Architectural bias (early vs late capacity – approximated by depth/dim)
        analysis = {
            "n_total": len(self.results),
            "n_under_param_budget": len(efficient),
            "n_above_acc_target": len(high_acc),
            "n_sweet_spot": len(sweet_spot),
            "best_overall": self.results[0],
            "most_efficient": min(self.results, key=lambda r: r["params_M"]),
            "lowest_latency": min(self.results, key=lambda r: r["latency_ms"]),
            "sweet_spot_models": sweet_spot[:5],
            "attn_preference": {},
        }
        from collections import Counter
        analysis["attn_preference"] = dict(Counter(r["attn_type"] for r in self.results))

        with open(self.exp_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        return (f"Analysis done – sweet-spot models={len(sweet_spot)}, "
                f"attn preference={analysis['attn_preference']}")

    def _exec_report(self) -> str:
        print("  [Agent] Writing research report …")
        report_path = self.exp_dir / "AGENT_REPORT.md"
        analysis_path = self.exp_dir / "analysis.json"
        analysis = {}
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)

        lines = [
            f"# Agentic Research Report – LowRes-EffTransNAS",
            f"",
            f"**Goal:** {self.goal}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Steps taken:** {self.step}",
            f"",
            f"## Summary",
            f"",
            f"- Pareto front size: {len(self.results)}",
            f"- Best accuracy: {max(r['accuracy'] for r in self.results):.1f}" if self.results else "",
            f"- Smallest model: {min(r['params_M'] for r in self.results):.1f} M params" if self.results else "",
            f"- Sweet-spot models (≤{self.target_params_M}M & ≥{self.target_acc}%): "
            f"{analysis.get('n_sweet_spot', 0)}",
            f"",
            f"## Top architectures",
            f"",
            f"| Rank | Depth | Dim | Heads | Ratio | Attn | Acc | Params (M) | Latency (ms) |",
            f"|------|-------|-----|-------|-------|------|-----|------------|--------------|",
        ]
        for r in self.results[:10]:
            lines.append(
                f"| {r['rank']} | {r['depth']} | {r['dim']} | {r['heads']} | "
                f"{r['ratio']} | {r['attn_type']} | {r['accuracy']:.1f} | "
                f"{r['params_M']:.1f} | {r['latency_ms']:.1f} |"
            )
        lines += [
            f"",
            f"## Agent memory (last steps)",
            f"",
        ]
        for e in self.memory.entries[-6:]:
            lines.append(f"- Step {e.step}: `{e.action.get('name')}` → {e.outcome}")

        lines += [
            f"",
            f"## Conclusion",
            f"",
            f"The agent autonomously explored the search space and produced a Pareto set "
            f"of efficient Transformer architectures suitable for low-resource NLP deployment.",
        ]
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return f"Report written → {report_path.name}"

    # ── main loop ───────────────────────────────────────────────────────────
    def run(self):
        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + "  LowRes-EffTransNAS  ·  Agentic Research Agent".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print(f"  Goal     : {self.goal}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Exp dir  : {self.exp_dir}")
        print()

        for self.step in range(1, self.max_steps + 1):
            print(f"{'─' * 60}")
            print(f"  AGENT STEP {self.step}/{self.max_steps}")
            print(f"{'─' * 60}")

            obs = self.observe()
            print(f"  Observe  : stage={obs.stage.value}  front={obs.pareto_size}  "
                  f"best_acc={obs.best_acc:.1f}  best_params={obs.best_params:.1f}M")

            action = self.plan(obs)
            print(f"  Plan     : {action.name}  ({action.reason})")

            outcome = self.execute(action)
            print(f"  Outcome  : {outcome}")

            self.memory.add(MemoryEntry(
                step=self.step,
                observation=asdict(obs),
                action=asdict(action),
                outcome=outcome,
                timestamp=datetime.now().isoformat(),
            ))

            if outcome == "GOAL_REACHED":
                print("\n  ✓  Research goal satisfied – agent stopping.")
                break
            time.sleep(0.3)  # polite pause for readability

        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + "  Agent finished".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print(f"  Artefacts → {self.exp_dir.resolve()}")
        print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Agentic NAS research agent for LowRes-EffTransNAS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--goal", type=str,
                   default="Discover efficient Transformers under 15M params with accuracy > 85",
                   help="High-level research goal")
    p.add_argument("--exp-name", type=str, default="agentic_run")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--pop-size", type=int, default=20)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--target-params", type=float, default=15.0)
    p.add_argument("--target-acc", type=float, default=85.0)
    p.add_argument("--output-root", type=str, default="experiments")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.output_root) / args.exp_name
    agent = NASAgent(
        goal=args.goal,
        exp_dir=exp_dir,
        max_steps=args.max_steps,
        pop_size=args.pop_size,
        generations=args.generations,
        seed=args.seed,
        device=args.device,
        target_params_M=args.target_params,
        target_acc=args.target_acc,
    )
    agent.run()


if __name__ == "__main__":
    main()
