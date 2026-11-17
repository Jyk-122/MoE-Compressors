"""Per-token ExFold for vLLM 0.24.0, unquantized Triton experts and TP.

One routing kernel selects and folds weights. The compact [tokens, topk]
routes go directly to vLLM's fused experts. Native MoE scheduling, the shared
expert and final TP all-reduce remain in the existing runner.
"""

from types import MethodType

import torch
import vllm
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu_worker import Worker

from ..routing import validate_topk
from ...utils import ActivationStatsWorkerMixin, attach_activation_stats


@triton.jit
def _fold_kernel(Ids, Weights, Scale, Loss, Norm, OutIds, OutWeights,
                 E: tl.constexpr, K: tl.constexpr, R: tl.constexpr, B: tl.constexpr,
                 FILTER: tl.constexpr, THRESHOLD: tl.constexpr, RENORMALIZE: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.arange(0, B)
    ids = tl.load(Ids + token * K + slot, mask=slot < K, other=0).to(tl.int32)
    weights = tl.load(Weights + token * K + slot, mask=slot < K, other=0.).to(tl.float32)
    norms = tl.load(Norm + ids, mask=slot < K, other=0.)
    scores = weights * norms
    selected = tl.full((B,), False, tl.int1)
    packed_ids = tl.full((B,), 0, tl.int32)
    packed_weights = tl.full((B,), 0., tl.float32)
    for step in range(R):
        candidates = tl.where((slot < K) & ~selected, scores, -float("inf"))
        chosen = tl.min(tl.where(candidates == tl.max(candidates, 0), slot, B), 0)
        chosen_id = tl.sum(tl.where(slot == chosen, ids, 0), 0)
        chosen_weight = tl.sum(tl.where(slot == chosen, weights, 0.), 0)
        packed_ids = tl.where(slot == step, chosen_id, packed_ids)
        packed_weights = tl.where(slot == step, chosen_weight, packed_weights)
        selected = selected | (slot == chosen)

    valid = (slot[:, None] < K) & (slot[None, :] < R)
    pair = ids[:, None] * E + packed_ids[None, :]
    costs = tl.load(Loss + pair, mask=valid, other=float("inf"))
    best_loss = tl.min(costs, 1)
    owner = tl.min(tl.where(costs == best_loss[:, None], slot[None, :], B), 1)
    scalar = tl.load(Scale + pair, mask=valid, other=0.)
    transfer = valid & ~selected[:, None] & (slot[None, :] == owner[:, None])
    if FILTER:
        transfer = transfer & (best_loss[:, None] <= THRESHOLD)
    folded = packed_weights + tl.sum(tl.where(transfer, weights[:, None] * scalar, 0.), 0)
    if RENORMALIZE:
        folded = folded * tl.div_rn(tl.sum(weights, 0), tl.sum(folded, 0))
    tl.store(OutIds + token * R + slot, packed_ids, mask=slot < R)
    tl.store(OutWeights + token * R + slot, folded, mask=slot < R)


def fold_routes(ids, weights, scale, loss, norm, topk, *,
                loss_threshold=0.99, renormalize=True):
    if topk == ids.shape[1]:
        return ids, weights
    tokens, native_topk = ids.shape
    selected = ids.new_empty((tokens, topk))
    folded = weights.new_empty((tokens, topk))
    _fold_kernel[(tokens,)](
        ids, weights, scale, loss, norm, selected, folded,
        norm.numel(), native_topk, topk, triton.next_power_of_2(native_topk),
        loss_threshold is not None, 0. if loss_threshold is None else loss_threshold,
        renormalize, num_warps=1, enable_fp_fusion=False,
    )
    return selected, folded


def forward_modular(self, x, topk_weights, topk_ids,
                    shared_experts=None, shared_experts_input=None):
    ids, weights = fold_routes(
        topk_ids, topk_weights, self.exfold_scale, self.exfold_loss, self.exfold_norm,
        self.exfold_topk, loss_threshold=self.exfold_loss_threshold,
        renormalize=self.exfold_renormalize)
    if self.activation_stats is not None:
        self.activation_stats.record_fixed(self.activation_layer_id, ids.shape[0], ids.shape[1])
    if self.exfold_topk == topk_ids.shape[1]:
        return self.exfold_original_forward(x, weights, ids, shared_experts, shared_experts_input)
    if shared_experts is not None:
        shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
    return fused_experts(x, self.w13_weight, self.w2_weight, weights, ids)


def patch(model, calibration_path, *, topk, loss_threshold=0.99, renormalize=True):
    """Install after weight loading, before compilation and CUDA graph capture."""
    from ..calibrate import read_calibration

    config = get_current_vllm_config()
    parallel = config.parallel_config
    if vllm.__version__.split("+")[0] != "0.24.0":
        raise RuntimeError("skip_exfold targets vLLM 0.24.0")
    if (config.model_config.hf_text_config.model_type != "qwen3_5_moe_text"
            or config.quant_config is not None or config.lora_config is not None
            or config.kernel_config.moe_backend != "triton"):
        raise ValueError("Use Qwen3.5/3.6 MoE, unquantized weights and --moe-backend triton")
    if (parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("skip_exfold supports TP; EP, EPLB, SP, DP and PCP are unsupported")
    native_topk = config.model_config.hf_text_config.num_experts_per_tok
    validate_topk(topk, native_topk)
    states, _ = read_calibration(calibration_path)
    layers = [(extract_layer_index(name), block) for name, block in model.named_modules()
              if isinstance(block, Qwen3NextSparseMoeBlock)]
    if not layers:
        raise ValueError("No Qwen3.5 MoE layers found")
    attach_activation_stats(model, [(index, block.experts.routed_experts, native_topk,
                                     block.experts.routed_experts.w13_weight.device)
                                    for index, block in layers],
                            max_num_tokens=config.scheduler_config.max_num_batched_tokens)
    for index, block in layers:
        experts = block.experts.routed_experts
        if not hasattr(experts, "exfold_original_forward"):
            experts.exfold_original_forward = experts.forward_modular
        for name in ("scale", "loss", "norm"):
            experts.register_buffer(f"exfold_{name}", states[index][name].to(
                device=experts.w13_weight.device, dtype=torch.float32).contiguous(), persistent=False)
        experts.exfold_topk = topk
        experts.exfold_loss_threshold = loss_threshold
        experts.exfold_renormalize = renormalize
        experts.forward_modular = MethodType(forward_modular, experts)
    return model


class ExFoldWorker(ActivationStatsWorkerMixin, Worker):
    """Enable with --worker-cls and additional_config['skip_exfold']."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        before = torch.cuda.memory_allocated(self.device)
        with set_current_vllm_config(self.vllm_config):
            patch(self.model_runner.get_model(), **self.vllm_config.additional_config["skip_exfold"])
        self.model_runner.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
