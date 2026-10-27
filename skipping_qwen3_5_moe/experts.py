"""Unweighted expert outputs for Qwen's packed gate/up and down projections."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def expert_output(experts, hidden, expert_id):
    gate, up = F.linear(hidden, experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
    return F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_id])


def selected_outputs(experts, hidden, expert_ids, active):
    """Execute active (token, slot) pairs and scatter to [tokens, K, hidden]."""
    tokens, slots = expert_ids.shape
    result = hidden.new_zeros(tokens * slots, hidden.shape[-1])
    for expert_id in expert_ids[active].unique():
        token_ids, slot_ids = torch.where(active & (expert_ids == expert_id))
        values = expert_output(experts, hidden[token_ids], expert_id)
        result.index_copy_(0, token_ids * slots + slot_ids, values)
    return result.reshape(tokens, slots, hidden.shape[-1])


@torch.no_grad()
def all_expert_outputs(experts, hidden):
    """Evaluate all samples once per expert; retain FP32 outputs on CPU."""
    count = experts.gate_up_proj.shape[0]
    output = torch.empty(hidden.shape[0], count, hidden.shape[1], dtype=torch.float32)
    weight = experts.gate_up_proj
    hidden = hidden.to(device=weight.device, dtype=weight.dtype)
    for expert_id in range(count):
        output[:, expert_id] = expert_output(experts, hidden, expert_id).float().cpu()
    return output
