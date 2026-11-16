"""DynRoute for vLLM 0.24.0: native Triton experts, unquantized weights and TP."""

from types import MethodType

import torch
import vllm
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu_worker import Worker

from ..routing import select_routes, validate_threshold
from ...utils import ActivationStatsWorkerMixin, attach_activation_stats


def forward_modular(self, x, topk_weights, topk_ids,
                    shared_experts=None, shared_experts_input=None):
    if self.dynroute_threshold == 1:
        if self.activation_stats is not None:
            self.activation_stats.record_fixed(
                self.activation_layer_id, topk_ids.shape[0], topk_ids.shape[1])
        return self.dynroute_original_forward(
            x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    if shared_experts is not None:
        shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
    ids, weights, keep = select_routes(topk_ids, topk_weights, self.dynroute_threshold)
    if self.activation_stats is not None:
        self.activation_stats.record(self.activation_layer_id, keep.sum(-1))
    num_experts = self.w13_weight.shape[0]
    # E is a valid global ID mapped to -1, so alignment skips both expert GEMMs.
    dispatch = ids.masked_fill(~keep, num_experts)
    return fused_experts(
        x, self.w13_weight, self.w2_weight, weights, dispatch,
        global_num_experts=num_experts + 1, expert_map=self.dynroute_expert_map)


def patch(model, *, topp_threshold):
    """Install after loading weights, before profiling/compilation/CUDA graphs."""
    validate_threshold(topp_threshold)
    config = get_current_vllm_config()
    parallel = config.parallel_config
    if vllm.__version__.split("+")[0] != "0.24.0":
        raise RuntimeError("skip_dynroute targets vLLM 0.24.0")
    if (config.model_config.hf_text_config.model_type != "qwen3_5_moe_text"
            or config.quant_config is not None or config.lora_config is not None
            or config.kernel_config.moe_backend != "triton"):
        raise ValueError("Use Qwen3.5/3.6 MoE, unquantized weights and --moe-backend triton")
    if (parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("skip_dynroute supports TP; EP, EPLB, SP, DP and PCP are unsupported")
    blocks = [(extract_layer_index(name), module) for name, module in model.named_modules()
              if isinstance(module, Qwen3NextSparseMoeBlock)]
    if not blocks:
        raise ValueError("No Qwen3.5 MoE layers found")
    attach_activation_stats(model, [(index, block.experts.routed_experts,
                                     config.model_config.hf_text_config.num_experts_per_tok,
                                     block.experts.routed_experts.w13_weight.device)
                                    for index, block in blocks],
                            max_num_tokens=config.scheduler_config.max_num_batched_tokens)
    for _, block in blocks:
        experts = block.experts.routed_experts
        if not hasattr(experts, "dynroute_original_forward"):
            experts.dynroute_original_forward = experts.forward_modular
            expert_map = torch.arange(experts.w13_weight.shape[0] + 1,
                                      dtype=torch.int32, device=experts.w13_weight.device)
            expert_map[-1] = -1
            experts.register_buffer("dynroute_expert_map", expert_map, persistent=False)
        experts.dynroute_threshold = float(topp_threshold)
        experts.forward_modular = MethodType(forward_modular, experts)
    return model


class DynRouteWorker(ActivationStatsWorkerMixin, Worker):
    """Enable with --worker-cls and additional_config['skip_dynroute']."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        before = torch.cuda.memory_allocated(self.device)
        with set_current_vllm_config(self.vllm_config):
            patch(self.model_runner.get_model(), **self.vllm_config.additional_config["skip_dynroute"])
        self.model_runner.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
