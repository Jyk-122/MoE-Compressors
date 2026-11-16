"""TransferMoE for vLLM 0.24.0, CUDA, unquantized Triton experts and TP.

Only RoutedExperts.forward_modular is replaced. vLLM keeps its router,
opaque MoE custom op, shared-expert scheduling and final TP all-reduce.
Selection runs in one Triton kernel; both expert projections use vLLM's
grouped GEMMs. No token/expert loops or GPU-to-CPU decisions run in Python.
"""

from types import MethodType

import torch
import vllm
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu_worker import Worker

from ...utils import ActivationStatsWorkerMixin, attach_activation_stats


logger = init_logger(__name__)


@triton.jit
def _select_kernel(Ids, Weights, Error, Dispatch, Assignment,
                   E: tl.constexpr, K: tl.constexpr, R: tl.constexpr,
                   B: tl.constexpr, FIXED: tl.constexpr, THRESHOLD: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.arange(0, B)
    ids = tl.load(Ids + token * K + slot, mask=slot < K, other=0).to(tl.int32)
    weights = tl.load(Weights + token * K + slot, mask=slot < K, other=0.).to(tl.float32)
    costs = tl.load(Error + ids[:, None] * E + ids[None, :],
                    mask=(slot[:, None] < K) & (slot[None, :] < K), other=0.)

    objective = tl.sum(costs * weights[None, :], 1)
    objective = tl.where(slot < K, objective, float("inf"))
    chosen = tl.min(tl.where(objective == tl.min(objective, 0), slot, B), 0)
    best = tl.sum(tl.where(slot[:, None] == chosen, costs, 0.), 0)
    selected = slot == chosen
    positions = tl.where(selected, 0, -1)
    assignment = tl.full((B,), 0, tl.int32)
    chosen_id = tl.sum(tl.where(selected, ids, 0), 0)
    packed = tl.where(slot == 0, chosen_id, E)
    total = tl.sum(weights * best, 0)

    for step in range(1, R):
        if not FIXED and total <= THRESHOLD:
            break
        gains = tl.sum(tl.maximum(best[None, :] - costs, 0.) * weights[None, :], 1)
        gains = tl.where((slot < K) & ~selected, gains, -float("inf"))
        chosen = tl.min(tl.where(gains == tl.max(gains, 0), slot, B), 0)
        candidate = tl.sum(tl.where(slot[:, None] == chosen, costs, 0.), 0)
        assignment = tl.where(candidate < best, step, assignment)
        best = tl.minimum(best, candidate)
        selected = selected | (slot == chosen)
        positions = tl.where(slot == chosen, step, positions)
        chosen_id = tl.sum(tl.where(slot == chosen, ids, 0), 0)
        packed = tl.where(slot == step, chosen_id, packed)
        total = tl.sum(weights * best, 0)

    # The combine kernel consumes every original target, including retained
    # targets. Those must read their own output, even when D has zero-cost ties.
    assignment = tl.where(selected, positions, assignment)
    tl.store(Dispatch + token * R + slot, packed, mask=slot < R)
    tl.store(Assignment + token * K + slot, assignment, mask=slot < K)


def select_routes(ids, weights, error, *, topk=None, topp_threshold=None):
    """Return packed expert IDs [T, R] and target-to-packed-slot indices [T, K]."""
    tokens, k = ids.shape
    slots = topk if topk is not None else k
    dispatch = torch.empty((tokens, slots), device=ids.device, dtype=torch.int32)
    assignment = torch.empty((tokens, k), device=ids.device, dtype=torch.int32)
    _select_kernel[(tokens,)](
        ids, weights, error, dispatch, assignment, error.shape[0], k, slots,
        triton.next_power_of_2(k), topk is not None,
        0. if topp_threshold is None else topp_threshold,
        num_warps=1, enable_fp_fusion=False,
    )
    return dispatch, assignment


def expert_outputs(hidden, w13, w2, dispatch):
    """Two native grouped GEMMs; unused dynamic-budget slots skip both GEMMs."""
    tokens, slots = dispatch.shape
    experts, gate_up_width, hidden_size = w13.shape
    config_dtype = "float32" if hidden.dtype == torch.float32 else None
    config = try_get_optimal_moe_config(w13.shape, w2.shape, slots, config_dtype, tokens)
    # E is a valid alignment bin. Its blocks become -1 before either GEMM,
    # including in vLLM's low-latency path that bypasses token sorting.
    sorted_ids, block_experts, padded = _prepare_expert_assignment(
        dispatch, config, tokens, slots, experts + 1, expert_map=None)
    block_experts = torch.where(block_experts == experts, -1, block_experts)
    compute_type = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16,
                    torch.float32: tl.float32}[hidden.dtype]

    # Reuse the gate/up buffer for down outputs, as native fused MoE does.
    workspace = hidden.new_empty(tokens * slots * max(gate_up_width, hidden_size))
    gate_up = workspace[:tokens * slots * gate_up_width].view(tokens, slots, gate_up_width)
    outputs = workspace[:tokens * slots * hidden_size].view(tokens, slots, hidden_size)
    activated = hidden.new_empty((tokens * slots, w2.shape[2]))

    def project(x, weight, out, input_topk):
        invoke_fused_moe_triton_kernel(
            x, weight, out, A_scale=None, B_scale=None, topk_weights=None,
            sorted_token_ids=sorted_ids, expert_ids=block_experts, num_tokens_post_padded=padded,
            mul_routed_weight=False, top_k=input_topk, config=config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False,
        )

    project(hidden, w13, gate_up, slots)
    apply_moe_activation(MoEActivation.SILU, activated, gate_up.view(-1, gate_up_width))
    project(activated, w2, outputs, 1)
    return outputs


@triton.jit
def _combine_kernel(Outputs, Ids, Weights, Dispatch, Assignment, Mean, Std, Result,
                    K: tl.constexpr, R: tl.constexpr, H: tl.constexpr, TP: tl.constexpr,
                    BK: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    target = tl.arange(0, BK)
    route = token * K + target
    ids = tl.load(Ids + route, mask=target < K, other=0)
    weights = tl.load(Weights + route, mask=target < K, other=0.).to(tl.float32)
    owner = tl.load(Assignment + route, mask=target < K, other=0)
    source = tl.load(Dispatch + token * R + owner, mask=target < K, other=0)
    valid = (target[:, None] < K) & (h[None, :] < H)
    y = tl.load(Outputs + (token * R + owner[:, None]) * H + h[None, :],
                mask=valid, other=0.).to(tl.float32)
    missing = valid & (source[:, None] != ids[:, None])
    target_mean = tl.load(Mean + ids[:, None] * H + h[None, :], mask=missing, other=0.)
    target_std = tl.load(Std + ids[:, None] * H + h[None, :], mask=missing, other=1.)
    source_mean = tl.load(Mean + source[:, None] * H + h[None, :], mask=missing, other=0.)
    source_std = tl.load(Std + source[:, None] * H + h[None, :], mask=missing, other=1.)
    ratio = tl.div_rn(target_std, source_std)
    # x_i = sum_p x_i^(p). Split the affine bias over TP ranks so the native
    # final [T, H] all-reduce reconstructs a_j*x_i+b_j exactly once.
    value = ratio * y + (target_mean - ratio * source_mean) / TP
    routed = tl.sum(weights[:, None] * value, 0)
    tl.store(Result + token * H + h, routed, mask=h < H)


def combine_outputs(outputs, ids, weights, dispatch, assignment, mean, std, tp_size=1):
    """Fuse compensation and target reduction without [T, K, H] moment buffers."""
    tokens, slots, hidden = outputs.shape
    result = outputs.new_empty((tokens, hidden))
    _combine_kernel[(tokens, triton.cdiv(hidden, 128))](
        outputs, ids, weights, dispatch, assignment, mean, std, result,
        ids.shape[1], slots, hidden, tp_size, triton.next_power_of_2(ids.shape[1]), 128,
        num_warps=4, enable_fp_fusion=False,
    )
    return result


def forward_modular(self, x, topk_weights, topk_ids,
                    shared_experts=None, shared_experts_input=None):
    # Preserve the shared-expert callback expected by the native MoERunner.
    # Its other scheduling modes remain handled by that runner.
    if shared_experts is not None:
        shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
    dispatch, assignment = select_routes(
        topk_ids, topk_weights, self.transfer_error,
        topk=self.transfer_topk, topp_threshold=self.transfer_threshold)
    if self.activation_stats is not None:
        self.activation_stats.record(
            self.activation_layer_id, (dispatch < self.w13_weight.shape[0]).sum(-1))
    outputs = expert_outputs(x, self.w13_weight, self.w2_weight, dispatch)
    return combine_outputs(outputs, topk_ids, topk_weights, dispatch, assignment,
                           self.transfer_mean, self.transfer_std, self.moe_config.tp_size)


def patch(model, calibration_path, *, topk=None, topp_threshold=None):
    """Install after weight loading and before profiling/compilation/graph capture."""
    from ..calibrate import read_calibration

    config = get_current_vllm_config()
    parallel = config.parallel_config
    if vllm.__version__.split("+")[0] != "0.24.0":
        raise RuntimeError("skip_transfer targets vLLM 0.24.0")
    if (config.model_config.hf_text_config.model_type != "qwen3_5_moe_text"
            or config.quant_config is not None or config.lora_config is not None
            or config.kernel_config.moe_backend != "triton"):
        raise ValueError("Use a Qwen3.5/3.6 MoE base model, unquantized weights and --moe-backend triton")
    if (parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("skip_transfer supports tensor parallelism; EP, EPLB, SP, DP and PCP are unsupported")
    if (topk is None) == (topp_threshold is None):
        raise ValueError("Specify exactly one of topk and topp_threshold")
    k = config.model_config.hf_text_config.num_experts_per_tok
    if (topk is not None and not 1 <= topk <= k) or (topp_threshold is not None and topp_threshold < 0):
        raise ValueError(f"Require 1 <= topk <= {k}, or topp_threshold >= 0")

    states, _ = read_calibration(calibration_path)
    layers = [(extract_layer_index(name), block) for name, block in model.named_modules()
              if isinstance(block, Qwen3NextSparseMoeBlock)]
    if not layers:
        raise ValueError("No Qwen3.5 MoE layers found")
    attach_activation_stats(model, [(index, block.experts.routed_experts, k,
                                     block.experts.routed_experts.w13_weight.device)
                                    for index, block in layers], vllm=True)
    for index, block in layers:
        experts = block.experts.routed_experts
        state = states[index]
        for name in ("mean", "std", "error"):
            experts.register_buffer(f"transfer_{name}", state[name].to(
                device=experts.w13_weight.device, dtype=torch.float32).contiguous(), persistent=False)
        experts.transfer_topk = topk
        experts.transfer_threshold = topp_threshold
        experts.forward_modular = MethodType(forward_modular, experts)
    logger.info("TransferMoE installed on %d layers: topk=%s topp_threshold=%s; native MoE runner retained",
                len(layers), topk, topp_threshold)
    return model


class TransferMoEWorker(ActivationStatsWorkerMixin, Worker):
    """Enable with --worker-cls and additional_config['skip_transfer']."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        before = torch.cuda.memory_allocated(self.device)
        with set_current_vllm_config(self.vllm_config):
            patch(self.model_runner.get_model(), **self.vllm_config.additional_config["skip_transfer"])
        self.model_runner.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
