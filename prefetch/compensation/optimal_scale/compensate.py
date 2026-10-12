"""Scale each predicted expert's output channels before the routed sum."""
from __future__ import annotations

import torch
from torch import nn


class OptimalScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", scale.float())  # [experts, hidden]

    def forward(self, experts, hidden, indices, weights, rows):
        """Execute the routed mixture; apply S only on the compensated rows."""
        scaled = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        scaled[rows] = True
        output = torch.zeros_like(hidden)
        # One call per active expert, using the same BF16/NF4 backend as calibration.
        for expert in indices.unique():
            token, slot = torch.where(indices == expert)
            selected = indices[token, slot, None]
            ones = torch.ones_like(selected, dtype=torch.float32)
            values = experts(hidden[token], selected, ones).float()
            channel_scale = torch.where(scaled[token, None], self.scale[expert], 1.0)
            values = values * channel_scale * weights[token, slot, None]
            output.index_add_(0, token, values.to(output.dtype))
        return output
