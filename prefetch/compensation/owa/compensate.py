"""Overlap-based Weight Adjustment adapted to native sigmoid routing."""
from __future__ import annotations

import torch
from torch import nn


class OWA(nn.Module):
    def __init__(self, alpha1=1.0, alpha2=1.0, hit_min=1, hit_max=None):
        super().__init__()
        if alpha1 <= 0 or alpha2 <= 0:
            raise ValueError("OWA alpha1 and alpha2 must be positive")
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.hit_min, self.hit_max = hit_min, hit_max

    def forward(self, native_ids, native_weights, predicted_ids, baseline_weights):
        # match[t, p, a]: predicted slot p is native slot a.
        match = predicted_ids.unsqueeze(2) == native_ids.unsqueeze(1)
        hit = match.any(-1)
        native_on_predicted = (match * native_weights.float().unsqueeze(1)).sum(-1)
        hit_mass = native_on_predicted.sum(-1, keepdim=True)
        total_mass = native_weights.float().sum(-1, keepdim=True)
        missing_mass = total_mass - hit_mass

        adjusted = torch.where(hit, self.alpha1 * native_on_predicted *
                               (1 + missing_mass / hit_mass.clamp_min(1e-20)),
                               baseline_weights.float())
        adjusted = adjusted * (self.alpha2 * total_mass /
                               adjusted.sum(-1, keepdim=True).clamp_min(1e-20))
        count = hit.sum(-1, keepdim=True)
        maximum = native_ids.shape[1] - 1 if self.hit_max is None else self.hit_max
        partial = ((count > 0) & (count < native_ids.shape[1]) &
                   (count >= self.hit_min) & (count <= maximum))
        weights = torch.where(partial, adjusted, baseline_weights.float())
        # A full hit preserves the native route, including its original scaling.
        weights = torch.where(count == native_ids.shape[1], native_on_predicted, weights)
        return weights.to(baseline_weights.dtype)
