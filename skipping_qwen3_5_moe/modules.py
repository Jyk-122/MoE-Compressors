"""Learnable compensation coefficients and per-expert moment transport.

Every pair uses [source, target]: source supplies a computed expert output,
and target is the expert whose contribution is reconstructed.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CompensateCoefficients(nn.Module):
    """Learn [source, target] logits and normalize over retained sources."""

    def __init__(self, num_experts):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_experts, num_experts, dtype=torch.float32))

    def forward(self, expert_ids, retained):
        """Gather [tokens, K, K] logits; each token retains at least one source."""
        scores = self.logits[expert_ids.unsqueeze(2), expert_ids.unsqueeze(1)]
        return scores.masked_fill(~retained.unsqueeze(-1), -torch.inf).softmax(dim=1)


class MomentCompensation(nn.Module):
    """Train calibrated means and log standard deviations, aggregating in batch."""

    def __init__(self, mean, std, epsilon=1e-5):
        super().__init__()
        if mean.ndim != 2 or std.shape != mean.shape:
            raise ValueError("Expert moments must have matching [experts, hidden] shapes")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Expert moments must be finite, with nonnegative standard deviations")
        self.epsilon = float(epsilon)
        self.mean = nn.Parameter(mean.detach().float().clone())
        self.log_std = nn.Parameter(std.detach().float().clone().clamp_min(epsilon).log())

    @property
    def std(self):
        return self.log_std.exp().clamp_min(self.epsilon)

    def forward(self, outputs, source_ids, target_ids):
        """Apply T(source, target) to corresponding unweighted source outputs."""
        std = self.std
        return self.mean[target_ids] + std[target_ids] * (
            (outputs.float() - self.mean[source_ids]) / std[source_ids])

    def aggregate(self, outputs, expert_ids, retained, routing_weights, coefficients):
        """Recover and sum routed contributions using [T, K, H] intermediates.

        outputs contains computed sources in their original top-k slots; other
        slots are zero. coefficients has shape [T, source_slot, target_slot].
        Only missing targets contribute to the reconstructed mixture.
        """
        mean, std = self.mean[expert_ids], self.std[expert_ids]
        standardized = (outputs.float() - mean) / std
        standardized = standardized * retained.unsqueeze(-1)
        recovered = mean + std * torch.bmm(coefficients.transpose(1, 2), standardized)
        mixture = torch.where(retained.unsqueeze(-1), outputs.float(), recovered)
        return (routing_weights.float().unsqueeze(-1) * mixture).sum(dim=1)
