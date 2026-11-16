"""Batched Top-p selection, shared by HF and vLLM."""

import torch


def validate_threshold(topp_threshold):
    if (isinstance(topp_threshold, bool) or not isinstance(topp_threshold, (int, float))
            or not 0 < topp_threshold <= 1):
        raise ValueError("topp_threshold must be a number in (0, 1]")


def select_routes(ids, weights, topp_threshold):
    """Return sorted IDs, renormalized weights and a keep mask, all [tokens, K].

    Inputs are native Top-K routes, with nonnegative weights and positive row
    sums. p=1 explicitly retains all routes, including underflowed weights.
    """
    weights, order = weights.float().sort(dim=-1, descending=True, stable=True)
    ids = ids.gather(-1, order)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    keep = torch.ones_like(weights, dtype=torch.bool)
    if topp_threshold < 1:
        # Keep a rank if the prefix BEFORE it has not yet reached p.
        keep[:, 1:] = weights.cumsum(dim=-1)[:, :-1] < topp_threshold
    weights = weights.masked_fill(~keep, 0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return ids, weights, keep
