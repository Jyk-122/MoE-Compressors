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
    belady_next: _NextUseIndex | None = None,
) -> int:
    """Return total expert load (swap-in) count for one layer timeline."""
    if capacity <= 0:
        return 0
    T = len(seq_sets)
    if T == 0:
        return 0

    if policy == "belady":
        nxt = belady_next if belady_next is not None else _NextUseIndex(seq_sets)
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


def parse_trace_samples(path: str, *, show_progress: bool = False) -> list[dict[int, list[frozenset[int]]]]:
    """
    Read binary trace once; each item maps layer_idx -> list of expert sets per token.
    Used to avoid re-parsing the file for every capacity point.
    """
    gen = iter_samples(path)
    if show_progress:
        try:
            from pathlib import Path
            from tqdm import tqdm

            gen = tqdm(gen, desc=f"Load trace {Path(path).name}", unit="sample")
        except ImportError:
            pass
    parsed: list[dict[int, list[frozenset[int]]]] = []
    for sample_steps in gen:
        parsed.append(timelines_per_layer_from_sample(sample_steps))
    return parsed


def _belady_layer_caches_for_sample(
    tls: dict[int, list[frozenset[int]]],
) -> dict[int, tuple[list[set[int]], _NextUseIndex]]:
    """Precompute seq_sets + NextUseIndex once per layer (Belady only; reusable across all C)."""
    out: dict[int, tuple[list[set[int]], _NextUseIndex]] = {}
    for lid, timeline in tls.items():
        seq_sets = [set(s) for s in timeline]
        out[lid] = (seq_sets, _NextUseIndex(seq_sets))
    return out


def simulate_parsed_samples(
    parsed: Sequence[dict[int, list[frozenset[int]]]],
    capacity: int,
    *,
    policy: str = "belady",
    lookahead: int = 64,
    belady_caches: list[dict[int, tuple[list[set[int]], _NextUseIndex]]] | None = None,
) -> tuple[int, int, float]:
    """
    Same semantics as simulate_trace_file but on pre-parsed samples.

    If policy is belady and ``belady_caches`` is provided (one entry per sample, from
    :func:`_belady_layer_caches_for_sample`), skips rebuilding :class:`_NextUseIndex` each call.
    """
    total_loads = 0
    total_tokens = 0
    for si, tls in enumerate(parsed):
        if not tls:
            continue
        lengths = [len(v) for v in tls.values()]
        ntok = lengths[0]
        if any(L != ntok for L in lengths):
            raise ValueError(f"Layer timeline length mismatch: {lengths[:8]}...")
        cache = belady_caches[si] if belady_caches is not None else None
        for lid, timeline in tls.items():
            seq_sets = [set(s) for s in timeline]
            bn: _NextUseIndex | None = None
            if policy == "belady" and cache is not None:
                seq_sets, bn = cache[lid]
            total_loads += _simulate_one_layer(
                seq_sets,
                capacity,
                policy=policy,
                lookahead=lookahead,
                belady_next=bn,
            )
        total_tokens += ntok
    avg = total_loads / max(total_tokens, 1)
    return total_loads, total_tokens, avg


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
    return simulate_parsed_samples([tls], capacity, policy=policy, lookahead=lookahead)[:2]


def simulate_trace_file(
    path: str,
    capacity: int,
    *,
    policy: str = "belady",
    lookahead: int = 64,
    parsed: Sequence[dict[int, list[frozenset[int]]]] | None = None,
    belady_caches: list[dict[int, tuple[list[set[int]], _NextUseIndex]]] | None = None,
) -> tuple[int, int, float]:
    """
    Sum over all samples in trace file.

    Returns (total_loads, total_tokens, avg_loads_per_token).
    Pass ``parsed`` / ``belady_caches`` from :func:`parse_trace_samples` to avoid re-reading disk.
    """
    if parsed is None:
        parsed = parse_trace_samples(path, show_progress=False)
    if policy == "belady" and belady_caches is None:
        belady_caches = [_belady_layer_caches_for_sample(tls) for tls in parsed]
    elif policy != "belady":
        belady_caches = None
    return simulate_parsed_samples(
        parsed,
        capacity,
        policy=policy,
        lookahead=lookahead,
        belady_caches=belady_caches,
    )


def capacity_curve(
    path: str,
    capacities: Iterable[int],
    *,
    policy: str = "belady",
    lookahead: int = 64,
    show_progress: bool = True,
) -> list[tuple[int, float]]:
    """List of (capacity, avg_loads_per_token). Parses trace file once; scans capacities with tqdm."""
    caps = sorted({int(c) for c in capacities if int(c) > 0})
    parsed = parse_trace_samples(path, show_progress=show_progress)
    belady_caches: list[dict[int, tuple[list[set[int]], _NextUseIndex]]] | None = None
    if policy == "belady":
        belady_caches = [_belady_layer_caches_for_sample(tls) for tls in parsed]

    iterator: Iterable[int]
    if show_progress:
        try:
            from pathlib import Path
            from tqdm import tqdm

            iterator = tqdm(
                caps,
                desc=f"I/O vs C [{Path(path).name}]",
                unit="cap",
                leave=True,
            )
        except ImportError:
            iterator = caps
    else:
        iterator = caps

    out: list[tuple[int, float]] = []
    for c in iterator:
        _, _, avg = simulate_parsed_samples(
            parsed,
            c,
            policy=policy,
            lookahead=lookahead,
            belady_caches=belady_caches,
        )
        out.append((c, avg))
    return out


def default_capacity_range(path: str, *, step: int = 1, max_cap: int | None = None) -> list[int]:
    h = load_trace_header_only(path)
    upper = int(h.num_experts) if max_cap is None else min(int(h.num_experts), int(max_cap))
    return list(range(1, upper + 1, int(step)))

