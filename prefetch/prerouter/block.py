"""Prerouter network: MoE input -> predicted router logits."""
from __future__ import annotations

import torch
from torch import nn


class Prerouter(nn.Module):
    def __init__(self, input_dim, num_experts, head="mlp", hidden_dim=512):
        super().__init__()
        if head == "linear":
            self.net = nn.Linear(input_dim, num_experts, bias=False)
        elif head == "mlp":
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, num_experts))
        else:
            raise ValueError("head must be linear or mlp")

    def forward(self, hidden_states):
        # FP32 master weights; CUDA computes the prediction in BF16.
        with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16,
                            enabled=hidden_states.is_cuda):
            return self.net(hidden_states if hidden_states.is_cuda else hidden_states.float())
