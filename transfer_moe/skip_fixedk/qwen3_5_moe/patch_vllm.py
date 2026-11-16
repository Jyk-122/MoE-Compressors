"""Statistics for native vLLM Fixed-K configured through --hf-overrides."""

from types import MethodType

import torch
import vllm
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.worker.gpu_worker import Worker

from ...utils import ActivationStatsWorkerMixin, attach_activation_stats


def forward_modular(self, x, topk_weights, topk_ids,
                    shared_experts=None, shared_experts_input=None):
    self.activation_stats.record_fixed(
        self.activation_layer_id, topk_ids.shape[0], topk_ids.shape[1])
    return self.fixedk_original_forward(
        x, topk_weights, topk_ids, shared_experts, shared_experts_input)


def patch(model, *, native_topk):
    """Record against the original K, before --hf-overrides reduced it."""
    config = get_current_vllm_config()
    parallel = config.parallel_config
    if vllm.__version__.split("+")[0] != "0.24.0":
        raise RuntimeError("skip_fixedk statistics target vLLM 0.24.0")
    if (config.model_config.hf_text_config.model_type != "qwen3_5_moe_text"
            or config.quant_config is not None or config.lora_config is not None
            or config.kernel_config.moe_backend != "triton"):
        raise ValueError("Use Qwen3.5/3.6 MoE, unquantized weights and --moe-backend triton")
    if (parallel.enable_expert_parallel or parallel.enable_eplb
            or parallel.use_sequence_parallel_moe or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1):
        raise ValueError("Fixed-K statistics support TP; EP, EPLB, SP, DP and PCP are unsupported")
    topk = config.model_config.hf_text_config.num_experts_per_tok
    if type(native_topk) is not int or not 1 <= topk <= native_topk:
        raise ValueError("Require 1 <= configured Top-K <= native_topk")
    layers = [(extract_layer_index(name), block) for name, block in model.named_modules()
              if isinstance(block, Qwen3NextSparseMoeBlock)]
    if not layers:
        raise ValueError("No Qwen3.5 MoE layers found")
    attach_activation_stats(model, [(index, block.experts.routed_experts, native_topk,
                                     block.experts.routed_experts.w13_weight.device)
                                    for index, block in layers], vllm=True)
    for _, block in layers:
        experts = block.experts.routed_experts
        if not hasattr(experts, "fixedk_original_forward"):
            experts.fixedk_original_forward = experts.forward_modular
        experts.forward_modular = (MethodType(forward_modular, experts)
                                   if experts.activation_stats is not None
                                   else experts.fixedk_original_forward)
    return model


class FixedKWorker(ActivationStatsWorkerMixin, Worker):
    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        before = torch.cuda.memory_allocated(self.device)
        with set_current_vllm_config(self.vllm_config):
            patch(self.model_runner.get_model(), **self.vllm_config.additional_config["skip_fixedk"])
        self.model_runner.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
