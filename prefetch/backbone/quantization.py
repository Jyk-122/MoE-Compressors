"""NF4 conversion of the model's 3D routed-expert Parameters."""
from __future__ import annotations

import gc

import torch
from torch import nn

from prefetch.backbone.structure import find_moe_blocks


class NF4Experts(nn.Module):
    def __init__(self, original, device, blocksize=64, quantize=True):
        super().__init__()
        if blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError("NF4 blocksize must be one of 64, 128, 256, 512, 1024, 2048, 4096")
        import bitsandbytes as bnb
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
        self.gate_up = nn.ModuleList()
        self.down = nn.ModuleList()
        for index in range(self.num_experts):
            for weights, destination in ((original.gate_up_proj, self.gate_up),
                                         (original.down_proj, self.down)):
                linear = bnb.nn.Linear4bit(weights.shape[2], weights.shape[1], bias=False,
                                          compute_dtype=torch.bfloat16, quant_type="nf4",
                                          compress_statistics=True, device="meta")
                if quantize:
                    weight = weights[index].detach().contiguous()
                    linear.weight = bnb.nn.Params4bit(weight, requires_grad=False, blocksize=blocksize,
                                                      quant_type="nf4", compress_statistics=True,
                                                      module=linear).to(device)
                destination.append(linear)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique().tolist():
            token, position = torch.where(top_k_index == expert)
            gate, up = self.gate_up[expert](hidden_states[token]).chunk(2, dim=-1)
            values = self.down[expert](self.act_fn(gate) * up)
            values = values * top_k_weights[token, position, None]
            result.index_add_(0, token, values.to(result.dtype))
        return result


def quantize_experts(model, device, blocksize=64):
    """Load on CPU first; at most one expert is transiently converted on GPU."""
    converted = 0
    for name, block in find_moe_blocks(model):
        original = block.experts
        converted += original.gate_up_proj.numel() + original.down_proj.numel()
        block.experts = NF4Experts(original, device, blocksize)
        del original
        gc.collect()
        print(f"NF4 {name}: cumulative source parameters={converted:,}", flush=True)
    return converted
