"""CUDA serving kernels for the unquantized Triton MoE in vLLM 0.24.0."""
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _route_kernel(Probs, Ids, Logits, Weights, Dispatch, Alpha,
                  E: tl.constexpr, K: tl.constexpr, B: tl.constexpr, THRESHOLD: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.arange(0, B)
    offset = token * K + slot
    probs = tl.load(Probs + offset, mask=slot < K, other=0.)
    ids = tl.load(Ids + offset, mask=slot < K, other=0).to(tl.int32)
    # Match the router's dtype rounding before computing the top-p prefix.
    weight = tl.div_rn(probs, tl.sum(probs, 0)).to(Weights.dtype.element_ty)
    q = weight.to(tl.float32)
    prefix = tl.cumsum(tl.div_rn(q, tl.sum(q, 0)), 0)
    previous = tl.gather(prefix, tl.maximum(slot - 1, 0), 0)
    keep = (slot < K) & ((THRESHOLD == 1) | (slot == 0) | (previous < THRESHOLD))
    tl.store(Weights + offset, weight, mask=slot < K)
    tl.store(Dispatch + offset, tl.where(keep, ids, E), mask=slot < K)

    # Rows are sources, columns are targets. Padded columns are never stored.
    scores = tl.load(Logits + ids[:, None] * E + ids[None, :],
                     mask=keep[:, None], other=-float("inf"))
    exp = tl.exp(scores - tl.max(scores, 0)[None, :])
    alpha = exp / tl.sum(exp, 0)[None, :]
    pair = token * K * K + slot[:, None] * K + slot[None, :]
    tl.store(Alpha + pair, alpha, mask=(slot[:, None] < K) & (slot[None, :] < K))


def routes(router_logits, logits, top_k, threshold):
    # Keep torch.topk's descending order and tie handling, including for decode.
    probs, ids = torch.softmax(router_logits, dim=-1, dtype=torch.float32).topk(top_k, dim=-1)
    tokens, experts = router_logits.shape
    weights = torch.empty_like(probs, dtype=router_logits.dtype)
    dispatch = torch.empty_like(ids, dtype=torch.int32)
    alpha = probs.new_empty((tokens, top_k, top_k))
    _route_kernel[(tokens,)](probs, ids, logits, weights, dispatch, alpha,
                             experts, top_k, triton.next_power_of_2(top_k), threshold)
    return weights, ids, dispatch, alpha


def expert_outputs(hidden, w13, w2, dispatch, expert_map, outputs):
    """Run both grouped GEMMs, retaining each unweighted result in its top-k slot."""
    tokens, slots = dispatch.shape
    config_dtype = "float32" if hidden.dtype == torch.float32 else None
    config = try_get_optimal_moe_config(w13.shape, w2.shape, slots, config_dtype, tokens)
    # E is a valid bin for the alignment op. Map that bin to -1 *after*
    # alignment so both the small-batch and sorted kernels zero skipped slots.
    if expert_map.numel() > 992:
        # CUDA block alignment requires a warp-rounded bin count below 1024.
        # Direct assignment uses one block per route and has no such limit.
        sorted_ids = None
        block_experts = dispatch.view(-1)
        padded = dispatch.new_full((1,), dispatch.numel() * config["BLOCK_SIZE_M"])
    else:
        sorted_ids, block_experts, padded = _prepare_expert_assignment(
            dispatch, config, tokens, slots, expert_map.numel(), expert_map=None)
    block_experts = expert_map[block_experts]
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
                    torch.float32: tl.float32}[hidden.dtype]

    def project(x, weight, out, input_top_k):
        invoke_fused_moe_triton_kernel(
            x, weight, out, A_scale=None, B_scale=None, topk_weights=None,
            sorted_token_ids=sorted_ids, expert_ids=block_experts, num_tokens_post_padded=padded,
            mul_routed_weight=False, top_k=input_top_k, config=config, compute_type=compute_type,
            use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
            use_int4_w4a16=False, per_channel_quant=False)

    gate_up = hidden.new_empty((tokens, slots, w13.shape[1]))
    project(hidden, w13, gate_up, slots)
    activated = hidden.new_empty((tokens * slots, w2.shape[2]))
    apply_moe_activation(MoEActivation.SILU, activated, gate_up.view(-1, w13.shape[1]))
    project(activated, w2, outputs, 1)


@triton.jit
def _transport_kernel(Outputs, Ids, Dispatch, Weights, Alpha, Mean, Std, Shared, Result,
                      E: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                      BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    slot = tl.arange(0, BLOCK_K)
    route = token * K + slot
    keep = tl.load(Dispatch + route, mask=slot < K, other=E) != E
    source_mask = keep[:, None] & (h[None, :] < H)
    y = tl.load(Outputs + route[:, None] * H + h[None, :], mask=source_mask, other=0.).to(tl.float32)
    q = tl.load(Weights + route, mask=slot < K, other=0.).to(tl.float32)
    routed = tl.sum(q[:, None] * y, 0)
    kept = tl.sum(keep.to(tl.int32), 0)
    if kept < K:
        ids = tl.load(Ids + route, mask=slot < K, other=0)
        moment = ids[:, None] * H + h[None, :]
        mean = tl.load(Mean + moment, mask=source_mask, other=0.)
        std = tl.load(Std + moment, mask=source_mask, other=1.)
        z = tl.div_rn(y - mean, std)
        # Top-p retains a prefix. Reconstruct only the missing targets, in registers.
        for target in range(kept, K):
            expert = tl.load(Ids + token * K + target)
            target_mean = tl.load(Mean + expert * H + h, mask=h < H, other=0.)
            target_std = tl.load(Std + expert * H + h, mask=h < H, other=0.)
            alpha = tl.load(Alpha + route * K + target, mask=keep, other=0.)
            recovered = target_mean + target_std * tl.sum(alpha[:, None] * z, 0)
            target_weight = tl.load(Weights + token * K + target).to(tl.float32)
            routed += target_weight * recovered
    shared = tl.load(Shared + token * H + h, mask=h < H, other=0.)
    # Training casts the routed sum back to activation precision before this add.
    value = routed.to(shared.dtype).to(tl.float32) + shared.to(tl.float32)
    tl.store(Result + token * H + h, value, mask=h < H)


def transport(outputs, ids, dispatch, weights, alpha, mean, std, shared):
    tokens, slots, hidden = outputs.shape
    result = outputs.new_empty((tokens, hidden))
    block_h = 64
    _transport_kernel[(tokens, triton.cdiv(hidden, block_h))](
        outputs, ids, dispatch, weights, alpha, mean, std, shared.contiguous(), result,
        mean.shape[0], slots, hidden, triton.next_power_of_2(slots), block_h)
    return result
