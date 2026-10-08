"""Prerouter network with optional conditioning on the target token embedding."""
from __future__ import annotations

import torch
from torch import nn


class Prerouter(nn.Module):
    def __init__(self, input_dim, num_experts, head="mlp", hidden_dim=512, embedding_dim=None):
        super().__init__()
        if head == "linear":
            self.net = nn.Linear(input_dim, num_experts, bias=False)
        elif head == "mlp":
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, num_experts))
        else:
            raise ValueError("head must be linear or mlp")
        self.token_proj = None
        if embedding_dim is not None:
            output_dim = num_experts if head == "linear" else hidden_dim
            self.token_proj = nn.Linear(embedding_dim, output_dim, bias=False)
            nn.init.zeros_(self.token_proj.weight)

    def forward(self, hidden_states, token_embeddings=None):
        # FP32 master weights; CUDA computes the prediction in BF16.
        with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16,
                            enabled=hidden_states.is_cuda):
            hidden_states = hidden_states if hidden_states.is_cuda else hidden_states.float()
            if token_embeddings is None:
                return self.net(hidden_states)
            tokens = token_embeddings if token_embeddings.is_cuda else token_embeddings.float()
            token_features = self.token_proj(tokens)
            if isinstance(self.net, nn.Linear):
                return self.net(hidden_states) + token_features
            # Fuse context and token identity before the MLP nonlinearity.
            features = self.net[0](hidden_states) + token_features
            return self.net[2](self.net[1](features))
