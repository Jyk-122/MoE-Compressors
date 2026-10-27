"""Threshold selection within the original, descending top-k routes."""
import math

import torch


def validate_threshold(threshold):
    value = float(threshold)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("threshold must satisfy 0 < threshold <= 1")
    return value


def top_p_mask(weights, threshold):
    threshold = validate_threshold(threshold)
    if threshold == 1:
        return torch.ones_like(weights, dtype=torch.bool)
    probabilities = weights.float() / weights.float().sum(-1, keepdim=True).clamp_min(1e-20)
    prefix = torch.cat((torch.zeros_like(probabilities[:, :1]), probabilities.cumsum(-1)[:, :-1]), -1)
    keep = prefix < threshold
    keep[:, 0] = True
    return keep
