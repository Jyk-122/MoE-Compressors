"""Qwen3.5 MoE forward with unchanged routing and shared-expert semantics.

This first HF implementation uses eager expert dispatch (unique/where), which
can synchronize on CUDA. The greedy selector itself does not read GPU values
on the CPU. Expert weights must be resident, unquantized packed tensors.
"""

from functools import partial
from types import MethodType

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from ..greedy import greedy_select_topk, greedy_select_topp
from ...utils import attach_activation_stats


def moe_blocks(model):
    """Qwen3.5 has one MoE block per text decoder layer, in traversal order."""
    return dict(enumerate(module for module in model.modules()
                          if isinstance(module, Qwen3_5MoeSparseMoeBlock)))


def expert_output(experts, hidden, expert_id):
    """Compute unweighted expert outputs from the native packed projections."""
    gate, up = F.linear(hidden, experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
    return F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_id])


def substitution_terms(ids, weights, retained, assignment, mean, std):
    """Merge missing targets into one scale vector per retained source.

    Returns scale [tokens, K, hidden] and offset [tokens, hidden].
    A source i contributes w_i*x_i + scale_i*(x_i-mean_i)/std_i;
    offset is the sum of w_j*mean_j over missing targets only.
    """
    missing_weights = weights.float() * ~retained
    target_scale = missing_weights.unsqueeze(-1) * std[ids]
    scale = torch.zeros_like(target_scale)
    scale.scatter_add_(1, assignment.unsqueeze(-1).expand_as(target_scale), target_scale)
    offset = (missing_weights.unsqueeze(-1) * mean[ids]).sum(1)
    return scale, offset


def moe_forward(self, hidden_states):
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    _, weights, ids = self.gate(hidden)
    retained, assignment, _ = self.transfer_select(ids, weights, self.transfer_error)
    if self.activation_stats is not None:
        self.activation_stats.record(self.activation_layer_id, retained.sum(-1))
    scale, offset = substitution_terms(
        ids, weights, retained, assignment, self.transfer_mean, self.transfer_std)

    routed = torch.zeros_like(hidden)
    # Group retained token/slot pairs by expert, as in HF's eager MoE dispatch.
    for expert_id in ids[retained].unique():
        token, slot = torch.where(retained & (ids == expert_id))
        output = expert_output(self.experts, hidden[token], expert_id).float()
        standardized = (output - self.transfer_mean[expert_id]) / self.transfer_std[expert_id]
        contribution = weights[token, slot, None].float() * output
        contribution = contribution + scale[token, slot] * standardized
        routed.index_add_(0, token, contribution.to(hidden.dtype))
    routed = routed + offset.to(hidden.dtype)

    shared = self.shared_expert(hidden) * self.shared_expert_gate(hidden).sigmoid()
    return (routed + shared).reshape_as(hidden_states)


def patch(model, calibration_path, *, topk=None, topp_threshold=None):
    """Load the calibration file and install one TransferMoE selection policy.

    Load the model at its inference dtype/device before patching. Statistics
    remain FP32. Specify exactly one of topk (1..K) or topp_threshold (0..1).
    """
    from ..calibrate import read_calibration

    if (topk is None) == (topp_threshold is None):
        raise ValueError("Specify exactly one of topk and topp_threshold")
    if topk is not None:
        select = partial(greedy_select_topk, topk=topk)
    else:
        if not 0 <= topp_threshold <= 1:
            raise ValueError("topp_threshold must be in [0, 1]")
        select = partial(greedy_select_topp, topp_threshold=topp_threshold)
    states, _ = read_calibration(calibration_path)
    blocks = moe_blocks(model)
    attach_activation_stats(model, [(index, block, block.gate.top_k, block.gate.weight.device)
                                    for index, block in blocks.items()])
    for index, block in blocks.items():
        state = states[index]
        device = block.gate.weight.device
        for name in ("mean", "std", "error"):
            block.register_buffer(f"transfer_{name}", state[name].to(device=device, dtype=torch.float32),
                                  persistent=False)
        block.transfer_select = select
        block.forward = MethodType(moe_forward, block)
    return model
