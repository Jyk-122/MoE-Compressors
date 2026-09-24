"""Expert selection policy: predicted IDs, weighted by the current native router."""
from __future__ import annotations

import torch

from prefetch.backbone.structure import choice_scores


def select_experts(block, router_logits, original_route, prerouter_logits=None):
    """Return execution (indices, weights), both [S,K].

    original_route is the native (indices, weights) pair, also used as labels.
    prerouter_logits is aligned to the target tokens; None keeps native routing.
    The caller supplies matching token rows for both routers.
    """
    if prerouter_logits is None:
        return original_route

    scores = choice_scores(prerouter_logits.detach(), block)
    # Match the stable expert ordering used by coverage evaluation.
    indices = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :block.top_k]
    weights = router_logits.sigmoid().gather(1, indices)
    if block.norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * block.routed_scaling_factor

    return indices, weights
