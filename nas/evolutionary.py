"""
Multi-objective evolutionary search (NSGA-II style) for LowRes-EffTransNAS.

Fitness vector: (accuracy, -params, -FLOPs, -latency)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .search_space import Architecture, SearchSpace


@dataclass
class Individual:
    arch: Architecture
    fitness: Optional[np.ndarray] = None  # (acc, -params, -flops, -lat)
    rank: int = 0
    crowding: float = 0.0

    def dominates(self, other: Individual) -> bool:
        """True if self dominates other (all objectives better or equal, at least one strictly)."""
        if self.fitness is None or other.fitness is None:
            return False
        better_or_eq = np.all(self.fitness >= other.fitness)
        strictly_better = np.any(self.fitness > other.fitness)
        return better_or_eq and strictly_better


def non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Fast non-dominated sorting (NSGA-II)."""
    fronts: List[List[Individual]] = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_set = []
        for q in population:
            if p.dominates(q):
                p.dominated_set.append(q)
            elif q.dominates(p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_set:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]  # last is empty


def crowding_distance(front: List[Individual]) -> None:
    """Assign crowding distance to individuals in a front."""
    if not front:
        return
    n = len(front)
    for ind in front:
        ind.crowding = 0.0

    n_obj = len(front[0].fitness)
    for m in range(n_obj):
        front.sort(key=lambda x: x.fitness[m])
        front[0].crowding = float("inf")
        front[-1].crowding = float("inf")
        f_min = front[0].fitness[m]
        f_max = front[-1].fitness[m]
        if f_max == f_min:
            continue
        for i in range(1, n - 1):
            front[i].crowding += (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / (
                f_max - f_min
            )


def tournament_selection(
    population: List[Individual], k: int = 2
) -> Individual:
    """Binary tournament based on rank then crowding."""
    candidates = random.sample(population, k)
    candidates.sort(key=lambda x: (x.rank, -x.crowding))
    return candidates[0]


class MultiObjectiveEvolutionarySearch:
    """
    Evolutionary algorithm matching the paper description:
      - population size 64
      - 50 generations
      - uniform crossover (p=0.5)
      - mutation (p=0.15 per gene)
      - non-dominated sorting + crowding distance
      - top 50 % become parents of next generation
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluate_fn: Callable[[Architecture], Tuple[float, float, float, float]],
        population_size: int = 64,
        n_generations: int = 50,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.15,
        early_stop_hv: float = 0.005,
        early_stop_patience: int = 5,
        seed: int = 42,
    ):
        self.space = search_space
        self.evaluate_fn = evaluate_fn
        self.pop_size = population_size
        self.n_gen = n_generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.early_stop_hv = early_stop_hv
        self.early_stop_patience = early_stop_patience
        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []

    def _init_population(self):
        self.population = []
        seen = set()
        while len(self.population) < self.pop_size:
            arch = self.space.sample()
            key = (arch.depth, arch.dim, arch.heads, arch.ratio, arch.attn_type)
            if key in seen:
                continue
            seen.add(key)
            ind = Individual(arch=arch)
            self.population.append(ind)

    def _evaluate_population(self, pop: List[Individual]):
        for ind in pop:
            if ind.fitness is None:
                acc, params, flops, lat = self.evaluate_fn(ind.arch)
                # Fitness: maximize acc, minimize the rest → negate the three
                ind.fitness = np.array(
                    [acc, -params, -flops, -lat], dtype=np.float64
                )

    def _hypervolume_2d(self, front: List[Individual], ref: Tuple[float, float] = (0.0, -1e9)) -> float:
        """Simple 2-D hypervolume on (accuracy, -params) for early stopping."""
        if not front:
            return 0.0
        points = sorted(
            [(ind.fitness[0], ind.fitness[1]) for ind in front],
            key=lambda p: p[0],
            reverse=True,
        )
        hv = 0.0
        prev_p = ref[1]
        for a, p in points:
            if p > prev_p:
                hv += (a - ref[0]) * (p - prev_p)
                prev_p = p
        return hv

    def run(self) -> List[Individual]:
        """Execute the evolutionary search and return the final Pareto front."""
        self._init_population()
        self._evaluate_population(self.population)

        fronts = non_dominated_sort(self.population)
        for f in fronts:
            crowding_distance(f)

        best_hv = self._hypervolume_2d(fronts[0])
        patience = 0

        for gen in range(self.n_gen):
            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = tournament_selection(self.population)
                p2 = tournament_selection(self.population)
                c1, c2 = self.space.crossover(p1.arch, p2.arch, self.cx_prob)
                c1 = self.space.mutate(c1, self.mut_prob)
                c2 = self.space.mutate(c2, self.mut_prob)
                offspring.append(Individual(arch=c1))
                if len(offspring) < self.pop_size:
                    offspring.append(Individual(arch=c2))

            self._evaluate_population(offspring)

            # Combine and select
            combined = self.population + offspring
            fronts = non_dominated_sort(combined)
            for f in fronts:
                crowding_distance(f)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    # fill remaining by crowding distance
                    front.sort(key=lambda x: -x.crowding)
                    new_pop.extend(front[: self.pop_size - len(new_pop)])
                    break
            self.population = new_pop

            # Logging
            cur_front = fronts[0]
            hv = self._hypervolume_2d(cur_front)
            mean_acc = np.mean([ind.fitness[0] for ind in self.population])
            best_acc = max(ind.fitness[0] for ind in self.population)

            self.history.append(
                {
                    "generation": gen,
                    "hypervolume": hv,
                    "mean_acc": mean_acc,
                    "best_acc": best_acc,
                    "front_size": len(cur_front),
                }
            )
            print(
                f"[Gen {gen:03d}] HV={hv:.4f}  best_acc={best_acc:.3f}  "
                f"mean_acc={mean_acc:.3f}  front={len(cur_front)}"
            )

            # Early stopping
            if hv > best_hv * (1.0 + self.early_stop_hv):
                best_hv = hv
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"Early stopping at generation {gen}")
                    break

        # Final Pareto front
        fronts = non_dominated_sort(self.population)
        self.pareto_front = fronts[0]
        return self.pareto_front

    def get_pareto_architectures(self) -> List[Architecture]:
        return [ind.arch for ind in self.pareto_front]
