"""
Simulate expert DRAM cache (per decoder layer): capacity C experts resident, count loads (I/O).

Policies:
  - belady: optimal offline — evict the cached expert whose next use is farthest in the future
  - lookahead_lru: evict from cache \\ R_t preferring experts not needed in the next H token steps;
    tie-break by LRU (least recent time step where expert was in R_u)
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Sequence

from utils.expert_trace_io import iter_samples, load_trace_header_only


class _NextUseIndex:
    def __init__(self, seq_sets: Sequence[set[int]]):
        self._pos: dict[int, list[int]] = defaultdict(list)
        for t, S in enumerate(seq_sets):
            for e in S:
                self._pos[int(e)].append(t)

    def next_after(self, e: int, t: int) -> int:
        lst = self._pos[int(e)]
        i = bisect.bisect_right(lst, t)
        return lst[i] if i < len(lst) else 10**18


def _simulate_one_layer(
    seq_sets: list[set[int]],
    capacity: int,
    *,
    policy: str,
    lookahead: int,
) -> int:
    """Return total expert load (swap-in) count for one layer timeline."""
    if capacity <= 0:
        return 0
    T = len(seq_sets)
    if T == 0:
        return 0

    if policy == "belady":
        nxt = _NextUseIndex(seq_sets)
        cache: set[int] = set()
        loads = 0
        for t, R in enumerate(seq_sets):
            Rset = set(R)
            missing = sorted(Rset - cache)
            for e in missing:
                while len(cache) >= capacity and e not in cache:
                    pool = cache - Rset
                    if pool:
                        victim = max(pool, key=lambda x: nxt.next_after(x, t))
                    else:
                        victim = max(cache, key=lambda x: nxt.next_after(x, t))
                    cache.remove(victim)
                if e not in cache:
                    cache.add(e)
                    loads += 1
        return loads

    if policy == "lookahead_lru":
        last_used: dict[int, int] = {}
        cache = set()
        loads = 0
        for t, R in enumerate(seq_sets):
            Rset = set(R)
            for e in Rset:
                last_used[int(e)] = t
            missing = sorted(Rset - cache)
            for e in missing:
                while len(cache) >= capacity and e not in cache:
                    pool = cache - Rset
                    fut_end = min(T, t + 1 + max(0, int(lookahead)))
                    future: set[int] = set()
                    for u in range(t + 1, fut_end):
                        future |= set(seq_sets[u])
                    pref = [x for x in pool if x not in future]
                    if pref:
                        victim = min(pref, key=lambda x: last_used.get(x, -1))
                    else:
                        victim = min(pool, key=lambda x: last_used.get(x, -1))
                    cache.remove(victim)
                if e not in cache:
                    cache.add(e)
                    loads += 1
        return loads

    raise ValueError(f"Unknown policy: {policy!r}")


def layer_timeline_from_sample(
    sample_steps: Sequence[dict[str, object]],
    layer_idx: int,
) -> list[frozenset[int]]:
    """Ordered list of required expert sets (one per token position)."""
    timeline: list[frozenset[int]] = []
    for step in sample_steps:
        layers = step["layers"]
        if not isinstance(layers, dict) or layer_idx not in layers:
            continue
        arr = layers[layer_idx]
        for i in range(int(arr.shape[0])):
            row = arr[i]
            s = {int(x) for x in row if int(x) >= 0}
            timeline.append(frozenset(s))
    return timeline


def timelines_per_layer_from_sample(
    sample_steps: Sequence[dict[str, object]],
) -> dict[int, list[frozenset[int]]]:
    layer_ids: set[int] = set()
    for step in sample_steps:
        layers = step["layers"]
        if isinstance(layers, dict):
            layer_ids |= {int(k) for k in layers.keys()}
    return {lid: layer_timeline_from_sample(sample_steps, lid) for lid in sorted(layer_ids)}


def simulate_sample_loads(
    sample_steps: Sequence[dict[str, object]],
    capacity: int,
    *,
    policy: str = "belady",
    lookahead: int = 64,
) -> tuple[int, int]:
    """
    Returns (total_loads_all_layers, num_token_positions).

    num_token_positions = length of timeline for layer 0 (all layers must match).
    """
    tls = timelines_per_layer_from_sample(sample_steps)
    if not tls:
        return 0, 0
    lengths = [len(v) for v in tls.values()]
    ntok = lengths[0]
    if any(L != ntok for L in lengths):
        raise ValueError(f"Layer timeline length mismatch: {lengths[:8]}...")
    loads = 0
    for timeline in tls.values():
        seq_sets = [set(s) for s in timeline]
        loads += _simulate_one_layer(seq_sets, capacity, policy=policy, lookahead=lookahead)
    return loads, ntok


def simulate_trace_file(
    path: str,
    capacity: int,
    *,
    policy: str = "belady",
    lookahead: int = 64,
) -> tuple[int, int, float]:
    """
    Sum over all samples in trace file.

    Returns (total_loads, total_tokens, avg_loads_per_token).
    """
    total_loads = 0
    total_tokens = 0
    for sample in iter_samples(path):
        L, T = simulate_sample_loads(sample, capacity, policy=policy, lookahead=lookahead)
        total_loads += L
        total_tokens += T
    avg = total_loads / max(total_tokens, 1)
    return total_loads, total_tokens, avg


def capacity_curve(
    path: str,
    capacities: Iterable[int],
    *,
    policy: str = "belady",
    lookahead: int = 64,
) -> list[tuple[int, float]]:
    """List of (capacity, avg_loads_per_token)."""
    caps = sorted({int(c) for c in capacities if int(c) > 0})
    return [(c, simulate_trace_file(path, c, policy=policy, lookahead=lookahead)[2]) for c in caps]


def default_capacity_range(path: str, *, step: int = 1, max_cap: int | None = None) -> list[int]:
    h = load_trace_header_only(path)
    upper = int(h.num_experts) if max_cap is None else min(int(h.num_experts), int(max_cap))
    return list(range(1, upper + 1, int(step)))

