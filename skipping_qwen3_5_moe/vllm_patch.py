"""Top-p compensation for the native Qwen MoE blocks in vLLM 0.24.0."""
from __future__ import annotations

import math
import re
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import read_artifact
from .experts import selected_outputs
from .modules import CompensateCoefficients, aggregate_moments
from .routing import top_p_mask, validate_threshold


def validate_vllm_config(config):
    """Check the execution modes and weight layout used by this adapter."""
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    if text.model_type != "qwen3_5_moe_text" or text.hidden_act != "silu":
        raise ValueError("Expected a Qwen3.5/3.6 MoE text backbone with SiLU experts")
    if not getattr(text, "norm_topk_prob", True):
        raise ValueError("Compensation requires normalized top-k router weights")
    if config.quant_config is not None:
        raise ValueError("Compensation requires unquantized BF16/FP16 expert weights")
    if not model.enforce_eager or int(config.compilation_config.mode) != 0:
        raise ValueError("Compensation requires --enforce-eager and compilation mode 0")
    if config.kernel_config.moe_backend != "triton":
        raise ValueError("Use --moe-backend triton to retain the gate/up/down weight layout")
    if parallel.enable_expert_parallel or parallel.enable_eplb or parallel.use_sequence_parallel_moe:
        raise ValueError("Compensation uses tensor parallelism with replicated expert IDs")
    if parallel.pipeline_parallel_size != 1 or parallel.data_parallel_size != 1:
        raise ValueError("Use one serving replica with --tensor-parallel-size for multiple GPUs")
    if config.speculative_config is not None or config.lora_config is not None:
        raise ValueError("This compensation service expects the base model with ordinary decoding")
    if model.enable_sleep_mode:
        raise ValueError("This compensation worker requires sleep mode to be disabled")
    return text


class VllmCompensation(nn.Module):
    """Reuse trained transport on complete expert outputs after TP reduction."""

    def __init__(self, state, threshold, tp_size, all_reduce):
        super().__init__()
        self.threshold = validate_threshold(threshold)
        self.top_k = int(state["top_k"])
        self.tp_size = tp_size
        self.all_reduce = all_reduce
        # Serving weights are immutable. Keep only the final FP32 scales on device.
        count = state["mean"].shape[0]
        log_std = (state["log_std"].float() if "log_std" in state else
                   state["std"].float().clamp_min(state["epsilon"]).log())
        self.register_buffer("mean", state["mean"].detach().float().contiguous())
        self.register_buffer("std", log_std.detach().exp().clamp_min_(state["epsilon"]).contiguous())
        expert_map = torch.arange(count + 1, dtype=torch.int32)
        expert_map[-1] = -1
        self.register_buffer("expert_map", expert_map)
        self.coefficients = CompensateCoefficients(count)
        if "logits" in state:
            with torch.no_grad():
                self.coefficients.logits.copy_(state["logits"])
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, block, hidden_states):
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden.shape[0] == 0:
            return hidden_states
        router_logits, _ = block.gate(hidden)
        experts = block.experts.routed_experts
        tokens, width = hidden.shape
        rows = tokens * self.top_k
        partials = hidden.new_empty((rows + (tokens if self.tp_size > 1 else 0), width))
        outputs = partials[:rows].view(tokens, self.top_k, width)
        if hidden.is_cuda:
            from . import vllm_kernels

            weights, ids, dispatch, coefficients = vllm_kernels.routes(
                router_logits, self.coefficients.logits, self.top_k, self.threshold)
            vllm_kernels.expert_outputs(hidden, experts.w13_weight, experts.w2_weight,
                                        dispatch, self.expert_map, outputs)
        else:
            weights, ids = router_logits.float().softmax(-1).topk(self.top_k, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).to(router_logits.dtype)
            retained = top_p_mask(weights, self.threshold)
            packed = SimpleNamespace(gate_up_proj=experts.w13_weight,
                                     down_proj=experts.w2_weight, act_fn=F.silu)
            outputs.copy_(selected_outputs(packed, hidden, ids, retained))
            coefficients = self.coefficients(ids, retained)

        # vLLM's shared MLP applies its sigmoid gate inside forward().
        shared = block.shared_expert(hidden)
        if self.tp_size > 1:
            # Moments describe complete expert outputs. Reduce the expert slots
            # and the shared shard in one collective, before applying transport.
            partials[rows:].copy_(shared)
            combined = self.all_reduce(partials)
            outputs = combined[:rows].view_as(outputs)
            shared = combined[rows:]
        if hidden.is_cuda:
            result = vllm_kernels.transport(outputs, ids, dispatch, weights, coefficients,
                                            self.mean, self.std, shared)
        else:
            routed = aggregate_moments(outputs, retained, weights, coefficients,
                                       self.mean[ids], self.std[ids])
            result = routed.to(hidden.dtype) + shared
        return result.reshape_as(hidden_states)


def _compensated_forward(block, hidden_states):
    return block.compensation(block, hidden_states)


def install_compensation(model, path, threshold, text_config, tp_size=1, all_reduce=None):
    """Install in a worker after native weight loading, before memory profiling."""
    states, metadata = read_artifact(path)
    threshold = validate_threshold(metadata["threshold"] if threshold is None else threshold)
    layers = {}
    for name, block in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", name)
        if match and hasattr(block, "experts") and hasattr(block, "shared_expert_gate"):
            index = int(match.group(1))
            if index in layers:
                raise ValueError(f"Ambiguous MoE layer index {index}")
            layers[index] = block
    if not layers or set(layers) != set(states):
        raise ValueError("Compensation checkpoint layers differ from the native vLLM MoE layers")
    if tp_size > 1 and all_reduce is None:
        raise ValueError("Tensor-parallel compensation requires an all-reduce function")

    count, hidden = text_config.num_experts, text_config.hidden_size
    intermediate, remainder = divmod(text_config.moe_intermediate_size, tp_size)
    if remainder:
        raise ValueError("Expert intermediate size must be divisible by the tensor-parallel size")
    # Validate the complete checkpoint and all weights before modifying forwards.
    for index, block in layers.items():
        state = states[index]
        if hasattr(block, "compensation"):
            raise ValueError(f"Layer {index}: compensation is already installed")
        experts = getattr(block.experts, "routed_experts", None)
        if experts is None:
            raise ValueError("Expected vLLM 0.24.0 experts.routed_experts")
        backend = getattr(getattr(experts, "quant_method", None), "unquantized_backend", None)
        if getattr(backend, "name", None) != "TRITON":
            raise ValueError(f"Layer {index}: expected unquantized Triton weight layout")
        if (experts.w13_weight.shape != (count, 2 * intermediate, hidden)
                or experts.w2_weight.shape != (count, hidden, intermediate)
                or experts.w13_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or experts.w13_weight.dtype != experts.w2_weight.dtype):
            raise ValueError(f"Layer {index}: expert weight geometry or dtype is incompatible")
        if (block.shared_expert is None or block.shared_expert.down_proj.reduce_results
                or block.shared_expert.expert_gate is not block.shared_expert_gate):
            raise ValueError(f"Layer {index}: expected a gated, tensor-parallel shared expert")
        std = state["std"] if "std" in state else state["log_std"].exp()
        if (state["mean"].shape != (count, hidden) or std.shape != (count, hidden)
                or state["top_k"] != text_config.num_experts_per_tok):
            raise ValueError(f"Layer {index}: compensation geometry differs from the model")
        if "logits" in state:
            if (state["logits"].shape != (count, count) or not torch.isfinite(state["logits"]).all()
                    or not torch.isfinite(state["log_std"]).all()):
                raise ValueError(f"Layer {index}: invalid trained compensation parameters")
        if (not torch.isfinite(state["mean"]).all() or not torch.isfinite(std).all()
                or (std < 0).any() or not math.isfinite(state["epsilon"]) or state["epsilon"] <= 0):
            raise ValueError(f"Layer {index}: invalid expert moments")

    if threshold == 1 or text_config.num_experts_per_tok == 1:
        # All routes are retained. The native block already includes the gated
        # shared expert and TP reduction; no adapter or device buffers are needed.
        return []

    for index, block in layers.items():
        adapter = VllmCompensation(states[index], threshold, tp_size, all_reduce)
        block.compensation = adapter.to(device=block.gate.weight.device)
        block.forward = MethodType(_compensated_forward, block)
    return sorted(layers)
