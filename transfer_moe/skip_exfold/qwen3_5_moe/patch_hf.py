"""Fold routing weights before native HF expert execution."""

from types import MethodType

from torch.nn import functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from ..routing import fold_routes, validate_topk
from ...utils import attach_activation_stats


def moe_blocks(model):
    return dict(enumerate(module for module in model.modules()
                          if isinstance(module, Qwen3_5MoeSparseMoeBlock)))


def expert_output(experts, hidden, expert_id):
    gate, up = F.linear(hidden, experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
    return F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_id])


def experts_forward(self, hidden_states, top_k_index, top_k_weights):
    ids, weights = fold_routes(
        top_k_index, top_k_weights, self.exfold_scale, self.exfold_loss, self.exfold_norm,
        self.exfold_topk, loss_threshold=self.exfold_loss_threshold,
        renormalize=self.exfold_renormalize)
    if self.activation_stats is not None:
        self.activation_stats.record_fixed(self.activation_layer_id, ids.shape[0], ids.shape[1])
    return self.exfold_original_forward(hidden_states, ids, weights)


def patch(model, calibration_path, *, topk, loss_threshold=0.99, renormalize=True):
    """Use the same per-token budget and reconstruction in prefill and decode."""
    from ..calibrate import read_calibration

    states, _ = read_calibration(calibration_path)
    blocks = moe_blocks(model)
    first = next(iter(blocks.values()))
    validate_topk(topk, first.gate.top_k)
    attach_activation_stats(model, [(index, block.experts, block.gate.top_k, block.gate.weight.device)
                                    for index, block in blocks.items()])
    for index, block in blocks.items():
        experts = block.experts
        if not hasattr(experts, "exfold_original_forward"):
            experts.exfold_original_forward = experts.forward
        for name in ("scale", "loss", "norm"):
            experts.register_buffer(f"exfold_{name}", states[index][name].to(
                device=block.gate.weight.device), persistent=False)
        experts.exfold_topk = topk
        experts.exfold_loss_threshold = loss_threshold
        experts.exfold_renormalize = renormalize
        experts.forward = MethodType(experts_forward, experts)
    return model
