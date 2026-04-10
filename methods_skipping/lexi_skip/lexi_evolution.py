"""
LExI Stage 2: evolutionary search for per-layer top-k under a global expert budget.

Implements a discrete allocation search minimizing sum_j D[j, k_j-1] subject to
sum_j k_j = B and k_min <= k_j <= k_max (paper Algorithm 2, simplified crossover).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("MoECompressor")


def _fitness(D: np.ndarray, k: np.ndarray) -> float:
    """D: [L, Kmax], k: [L] integer in [1, Kmax]. Index k-1."""
    L = D.shape[0]
    idx = np.clip(k - 1, 0, D.shape[1] - 1)
    return float(D[np.arange(L), idx].sum())


def _random_feasible_vector(
    L: int,
    B: int,
    k_lo: int,
    k_hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if L * k_lo > B or L * k_hi < B:
        raise ValueError(
            f"无可行整数解: L={L}, B={B}, 需要 {L * k_lo} <= B <= {L * k_hi} "
            f"(每层 k in [{k_lo}, {k_hi}])"
        )
    k = np.full(L, k_lo, dtype=np.int64)
    need = B - int(k.sum())
    caps = np.full(L, k_hi - k_lo, dtype=np.int64)
    while need > 0:
        choices = np.where(caps > 0)[0]
        if len(choices) == 0:
            break
        j = int(rng.choice(choices))
        k[j] += 1
        caps[j] -= 1
        need -= 1
    assert int(k.sum()) == B
    return k


def _project_to_budget(k: np.ndarray, B: int, k_lo: int, k_hi: int) -> np.ndarray:
    """Clip to [k_lo, k_hi] then adjust sum to B by greedy moves."""
    k = np.clip(k.astype(np.int64), k_lo, k_hi).copy()
    s = int(k.sum())
    if s == B:
        return k
    # Try to fix by moving mass between layers
    while s != B:
        if s > B:
            # decrease some layer > k_lo
            cand = np.where(k > k_lo)[0]
            if len(cand) == 0:
                break
            j = int(cand[np.argmax(k[cand])])
            k[j] -= 1
            s -= 1
        else:
            cand = np.where(k < k_hi)[0]
            if len(cand) == 0:
                break
            j = int(cand[np.argmin(k[cand])])
            k[j] += 1
            s += 1
    if int(k.sum()) != B:
        raise RuntimeError(f"投影失败: sum={int(k.sum())} != B={B}")
    return k


def _mutate_balance(k: np.ndarray, B: int, k_lo: int, k_hi: int, rng: np.random.Generator) -> np.ndarray:
    """Delta in {-1,0,1} with sum 0: pick i,j and try k[i]-1, k[j]+1."""
    k = k.copy()
    L = len(k)
    for _ in range(L * 2):
        i, j = rng.integers(0, L, size=2)
        if i == j:
            continue
        if k[i] > k_lo and k[j] < k_hi:
            k[i] -= 1
            k[j] += 1
            return k
        if k[j] > k_lo and k[i] < k_hi:
            k[j] -= 1
            k[i] += 1
            return k
    return _project_to_budget(k, B, k_lo, k_hi)


def evolve_topk_allocation(
    D: np.ndarray,
    B: int,
    *,
    k_min: int = 1,
    k_max: int | None = None,
    n_pop: int = 64,
    n_gen: int = 200,
    seed: int | None = 0,
    tournament_k: int = 3,
) -> np.ndarray:
    """
    Args:
        D: Sensitivity matrix, shape [L, Kmax], D[j, k-1] = perturbation loss for layer j with top-k.
        B: Total expert budget (sum of per-layer k).
        k_min, k_max: Per-layer bounds (inclusive). If k_max is None, use D.shape[1].
        n_pop, n_gen: Evolution parameters.
        seed: RNG seed; None for nondeterministic.

    Returns:
        k: shape [L], dtype int64, each in [k_min, k_max], sum(k) == B.
    """
    D = np.asarray(D, dtype=np.float64)
    L, Kmax = D.shape
    if k_max is None:
        k_max = Kmax
    if k_min < 1 or k_max > Kmax:
        raise ValueError(f"k_min/k_max 与 D 列数不一致: k_max={k_max}, Kmax={Kmax}")
    if L * k_min > B or L * k_max < B:
        raise ValueError(
            f"预算 B={B} 与层数 L={L}、边界 [{k_min},{k_max}] 无可行解"
        )

    rng = np.random.default_rng(seed)
    pop: list[np.ndarray] = [
        _random_feasible_vector(L, B, k_min, k_max, rng) for _ in range(n_pop)
    ]

    best_global: np.ndarray | None = None
    best_global_f = float("inf")
    for ind in pop:
        f = _fitness(D, ind)
        if f < best_global_f:
            best_global_f = f
            best_global = ind.copy()

    def tournament() -> np.ndarray:
        cand_idx = rng.integers(0, len(pop), size=min(tournament_k, len(pop)))
        best = pop[int(cand_idx[0])]
        best_f = _fitness(D, best)
        for ci in cand_idx[1:]:
            f = _fitness(D, pop[int(ci)])
            if f < best_f:
                best = pop[int(ci)]
                best_f = f
        return best.copy()

    for g in range(n_gen):
        for ind in pop:
            f = _fitness(D, ind)
            if f < best_global_f:
                best_global_f = f
                best_global = ind.copy()

        new_pop: list[np.ndarray] = []
        # Elitism: keep best
        if best_global is not None:
            new_pop.append(best_global.copy())

        while len(new_pop) < n_pop:
            p1 = tournament()
            p2 = tournament()
            # Uniform crossover (Bernoulli 0.5 per layer)
            mask = rng.random(L) < 0.5
            child = np.where(mask, p1, p2).astype(np.int64)
            child = _project_to_budget(child, B, k_min, k_max)
            if rng.random() < 0.9:
                child = _mutate_balance(child, B, k_min, k_max, rng)
                child = _project_to_budget(child, B, k_min, k_max)
            new_pop.append(child)

        pop = new_pop[:n_pop]

    for ind in pop:
        f = _fitness(D, ind)
        if f < best_global_f:
            best_global_f = f
            best_global = ind.copy()

    assert best_global is not None
    logger.info(
        "[lexi_skip][evolution] best fitness=%.6f sum(k)=%d",
        best_global_f,
        int(best_global.sum()),
    )
    return best_global.astype(np.int64)
