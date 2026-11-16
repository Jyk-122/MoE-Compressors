"""Model loading and shared routed-expert activation statistics."""

import inspect
import os
import re

import torch


def activation_stats_enabled():
    """Read once when installing a patch; unset / 0 disables recording."""
    return os.environ.get("TRANSFER_MOE_RECORD_STATS", "0") == "1"


class ActivationStats:
    """Accumulate [pruned routes, native routes] per layer, on its device.

    layers maps layer IDs to (native Top-K, device). Only report() reads
    counters on the CPU; record() and reset() keep their storage in place.
    """

    @torch.inference_mode(False)
    def __init__(self, layers, *, vllm=False):
        self.native_topk = {index: topk for index, (topk, _) in layers.items()}
        self.counts = {index: torch.zeros(2, dtype=torch.int64, device=device)
                       for index, (_, device) in layers.items()}
        self.attention_mask = None
        devices = {counter.device for counter in self.counts.values()}
        self.token_limits = ({device: torch.full((), torch.iinfo(torch.int64).max,
                                                dtype=torch.int64, device=device)
                              for device in devices} if vllm else None)

    def record(self, layer_id, kept_per_token):
        counter = self.counts[layer_id]
        topk = self.native_topk[layer_id]
        if self.token_limits is not None:
            valid = (torch.arange(kept_per_token.numel(), device=counter.device)
                     < self.token_limits[counter.device])
        elif self.attention_mask is not None:
            valid = self.attention_mask.to(counter.device)
        else:
            valid = torch.ones_like(kept_per_token, dtype=torch.bool)
        pruned = ((topk - kept_per_token) * valid).sum()
        native = valid.sum() * topk
        counter.add_(torch.stack((pruned, native)))

    def record_fixed(self, layer_id, num_tokens, kept):
        self.record(layer_id, torch.full((num_tokens,), kept, dtype=torch.int64,
                                        device=self.counts[layer_id].device))

    def set_num_tokens(self, num_tokens):
        """Called outside CUDA graphs, before each vLLM model execution."""
        for limit in self.token_limits.values():
            limit.fill_(num_tokens)

    def reset(self):
        for counter in self.counts.values():
            counter.zero_()

    def report(self):
        counts = {index: counter.tolist() for index, counter in self.counts.items()}
        pruned = sum(pair[0] for pair in counts.values())
        native = sum(pair[1] for pair in counts.values())
        return {
            "layer_pruning_ratios": {str(index): removed / total if total else None
                                     for index, (removed, total) in counts.items()},
            "overall_pruning_ratio": pruned / native if native else None,
        }


def attach_activation_stats(model, layers, *, vllm=False):
    """layers contains (layer ID, patched module, native Top-K, device)."""
    if hasattr(model, "activation_stats_hook"):
        model.activation_stats_hook.remove()
        del model.activation_stats_hook
    stats = (ActivationStats({index: (topk, device) for index, _, topk, device in layers},
                             vllm=vllm) if activation_stats_enabled() else None)
    model.activation_stats = stats
    for index, module, _, _ in layers:
        module.activation_stats = stats
        module.activation_layer_id = index
    if stats is not None and not vllm:
        signature = inspect.signature(model.forward)

        def set_attention_mask(module, args, kwargs):
            inputs = signature.bind_partial(*args, **kwargs).arguments
            mask = inputs.get("attention_mask")
            if mask is None:
                stats.attention_mask = None
            else:
                tokens = inputs.get("input_ids")
                if tokens is None:
                    tokens = inputs["inputs_embeds"]
                # Cached decoding passes only new tokens, but a full-history mask.
                stats.attention_mask = mask[:, -tokens.shape[1]:].reshape(-1).bool()

        model.activation_stats_hook = model.register_forward_pre_hook(
            set_attention_mask, with_kwargs=True)
    return stats


class ActivationStatsWorkerMixin:
    """vLLM worker RPC access and scheduled-token counts for CUDA graph replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The V1 runner packs real tokens before graph padding. DBO splits that
        # prefix into microbatches; PP would require merging disjoint layer sets.
        if activation_stats_enabled() and (
                self.use_v2_model_runner or self.parallel_config.use_ubatching
                or self.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("Activation statistics support vLLM V1 with single GPU / TP, without DBO / PP")

    def execute_model(self, scheduler_output):
        stats = self.model_runner.get_model().activation_stats
        if stats is not None:
            stats.set_num_tokens(scheduler_output.total_num_scheduled_tokens)
        return super().execute_model(scheduler_output)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.reset_activation_stats()
        return result

    def reset_activation_stats(self):
        stats = self.model_runner.get_model().activation_stats
        if stats is not None:
            stats.reset()

    def get_activation_stats(self):
        stats = self.model_runner.get_model().activation_stats
        return stats.report() if stats is not None else None


def load_model_and_processor(path, device="cuda:0", dtype="auto"):
    from transformers import (
        AutoProcessor, AutoTokenizer, Qwen3MoeForCausalLM, Qwen3_5MoeForConditionalGeneration,
    )

    name = str(path).lower().replace("_", "-").replace(".", "-")
    if re.search(r"qwen3-[56]-(?:moe|\d+b-a\d)", name):
        model_class, processor_class = Qwen3_5MoeForConditionalGeneration, AutoProcessor
    elif re.search(r"qwen3-(?:moe|\d+b-a\d)", name):
        model_class, processor_class = Qwen3MoeForCausalLM, AutoTokenizer
    else:
        raise ValueError(f"Unsupported model path: {path}")

    model = model_class.from_pretrained(
        path, dtype=dtype, device_map=device, attn_implementation="sdpa").eval()
    return model, processor_class.from_pretrained(path)
