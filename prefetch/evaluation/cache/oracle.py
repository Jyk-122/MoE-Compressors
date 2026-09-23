"""Equal-size expert caches, with eviction guided by future true router uses."""
from __future__ import annotations

from collections import deque


def load_required(cache, required, capacity, next_use):
    """Keep the complete requested set resident; clean evictions do not cause writes."""
    required = set(required)
    missing = sorted(required - cache)
    evictions = 0
    for expert in missing:
        if len(cache) == capacity:
            # A current request is pinned. Break equal-next-use ties by expert ID.
            victim = max(cache - required, key=lambda item: (next_use[item], item))
            cache.remove(victim)
            evictions += 1
        cache.add(expert)
    return len(missing), evictions


def simulate_layer(truth, prediction, capacity, experts):
    """Yield one row per decode token. Each invocation starts two empty caches."""
    if capacity < 1:
        raise ValueError("Cache capacity must be positive")
    if prediction is not None and len(prediction) != len(truth):
        raise ValueError("Predictions and truth must contain the same decode tokens")
    for requests in (truth, prediction or []):
        for required in requests:
            if len(set(required)) != len(required) or any(e < 0 or e >= experts for e in required):
                raise ValueError("Expert sets must contain distinct in-range IDs")
            if len(required) > capacity:
                raise ValueError("Cache capacity must hold the complete truth and prediction sets")

    future = [deque() for _ in range(experts)]
    for token, required in enumerate(truth):
        for expert in required:
            future[expert].append(token)
    next_use = [times[0] if times else float("inf") for times in future]
    baseline, prefetched = set(), set()
    for token, required in enumerate(truth):
        base_loads, base_evictions = load_required(baseline, required, capacity, next_use)
        predicted = prediction[token] if prediction is not None else []
        pre_loads, pre_evictions = load_required(prefetched, predicted, capacity, next_use)
        demand_loads, demand_evictions = load_required(prefetched, required, capacity, next_use)
        yield dict(token=token, baseline_loads=base_loads, prefetch_loads=pre_loads,
                   demand_loads=demand_loads, baseline_evictions=base_evictions,
                   prefetch_evictions=pre_evictions, demand_evictions=demand_evictions)
        # At the prefetch event, this token's true demand is still a future use.
        for expert in required:
            future[expert].popleft()
            next_use[expert] = future[expert][0] if future[expert] else float("inf")
