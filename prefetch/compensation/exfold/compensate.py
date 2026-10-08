"""Fold each missing native expert onto a calibrated predicted expert."""
from __future__ import annotations

import torch
from torch import nn


class ExFold(nn.Module):
    def __init__(self, coefficients, loss):
        super().__init__()
        if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1] or loss.shape != coefficients.shape:
            raise ValueError("ExFold requires square [experts, experts] coefficient and loss tables")
        self.register_buffer("coefficients", coefficients.float())
        self.register_buffer("loss", loss.float())

    def forward(self, native_ids, native_weights, predicted_ids, baseline_weights):
        # Tables are directed: source native expert -> destination predicted expert.
        match = native_ids.unsqueeze(2) == predicted_ids.unsqueeze(1)
        retained = match.any(-1)
        pair_loss = self.loss[native_ids.unsqueeze(2), predicted_ids.unsqueeze(1)]
        best_loss, destination = pair_loss.min(-1)
        coefficient = self.coefficients[native_ids, predicted_ids.gather(1, destination)]
        covered = torch.isfinite(best_loss) & (best_loss < 1e30)
        contribution = native_weights.float() * coefficient
        contribution = torch.where(~retained & covered, contribution, torch.zeros_like(contribution))

        weights = (match * native_weights.float().unsqueeze(2)).sum(1)
        weights.scatter_add_(1, destination, contribution)
        # Signed regression coefficients already set the output scale.
        return weights.to(baseline_weights.dtype)
