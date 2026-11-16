"""Reuse native HF experts on compacted retained routes; keep the MoE forward."""

from types import MethodType

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from ..routing import select_routes, validate_threshold
from ...utils import attach_activation_stats


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    if self.dynroute_threshold == 1:
        if self.activation_stats is not None:
            self.activation_stats.record_fixed(
                self.activation_layer_id, top_k_index.shape[0], top_k_index.shape[1])
        return self.dynroute_original_forward(hidden_states, top_k_index, top_k_weights)
    ids, weights, keep = select_routes(top_k_index, top_k_weights, self.dynroute_threshold)
    if self.activation_stats is not None:
        self.activation_stats.record(self.activation_layer_id, keep.sum(-1))
    token, slot = keep.nonzero(as_tuple=True)
    # One native route per retained token/expert pair; dropped pairs never enter GEMMs.
    output = self.dynroute_original_forward(
        hidden_states[token], ids[token, slot, None], weights[token, slot, None])
    return torch.zeros_like(hidden_states).index_add_(0, token, output)


def patch(model, *, topp_threshold):
    """Install on a native HF model before inference; p is in (0, 1]."""
    validate_threshold(topp_threshold)
    blocks = [module for module in model.modules()
              if isinstance(module, Qwen3_5MoeSparseMoeBlock)]
    if not blocks:
        raise ValueError("No Qwen3.5 MoE layers found")
    attach_activation_stats(model, [(index, block.experts, block.gate.top_k, block.gate.weight.device)
                                    for index, block in enumerate(blocks)])
    for block in blocks:
        experts = block.experts
        if not hasattr(experts, "dynroute_original_forward"):
            experts.dynroute_original_forward = experts.forward
        experts.dynroute_threshold = float(topp_threshold)
        experts.forward = MethodType(experts_forward, experts)
    return model
