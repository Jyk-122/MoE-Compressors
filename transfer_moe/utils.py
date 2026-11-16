"""Model loading and shared routed-expert activation statistics."""

import inspect
import os
import re

import torch


def activation_stats_enabled():
    """Read once when installing a patch; unset / 0 disables recording."""
    return os.environ.get("TRANSFER_MOE_RECORD_STATS", "0") == "1"


class ActivationStats:
    """Accumulate [prefill/decode, pruned/native routes] per layer.

    layers maps layer IDs to (native Top-K, device). Only report() reads
    counters on the CPU; record() and reset() keep their storage in place.
    """

    @torch.inference_mode(False)
    def __init__(self, layers, *, max_num_tokens=None):
        self.native_topk = {index: topk for index, (topk, _) in layers.items()}
        self.counts = {index: torch.zeros((2, 2), dtype=torch.int64, device=device)
                       for index, (_, device) in layers.items()}
        self.phase = 0  # HF: 0 = prefill, 1 = decode.
        self.attention_mask = None
        devices = {counter.device for counter in self.counts.values()}
        # vLLM may mix both phases in one batch. -1 marks graph padding.
        self.token_phases = ({device: torch.full((max_num_tokens,), -1,
                                                dtype=torch.int8, device=device)
                              for device in devices} if max_num_tokens is not None else None)

    def record(self, layer_id, kept_per_token):
        counter = self.counts[layer_id]
        topk = self.native_topk[layer_id]
        if self.token_phases is not None:
            phases = self.token_phases[counter.device][:kept_per_token.numel()]
            valid = torch.stack((phases == 0, phases == 1))
            pruned = ((topk - kept_per_token) * valid).sum(-1)
            native = valid.sum(-1) * topk
            counter.add_(torch.stack((pruned, native), dim=-1))
        else:
            valid = (self.attention_mask.to(counter.device) if self.attention_mask is not None
                     else torch.ones_like(kept_per_token, dtype=torch.bool))
            pruned = ((topk - kept_per_token) * valid).sum()
            native = valid.sum() * topk
            counter[self.phase].add_(torch.stack((pruned, native)))

    def record_fixed(self, layer_id, num_tokens, kept):
        self.record(layer_id, torch.full((num_tokens,), kept, dtype=torch.int64,
                                        device=self.counts[layer_id].device))

    def set_token_phases(self, phases):
        """Update real-token phases in place, outside vLLM CUDA graphs."""
        for buffer in self.token_phases.values():
            buffer[:phases.numel()].copy_(phases)
            buffer[phases.numel():].fill_(-1)

    def reset(self):
        for counter in self.counts.values():
            counter.zero_()

    def report(self):
        counts = {index: counter.tolist() for index, counter in self.counts.items()}

        def ratios(pairs):
            pruned = sum(pair[0] for pair in pairs.values())
            native = sum(pair[1] for pair in pairs.values())
            return {
                "layer_pruning_ratios": {str(index): removed / total if total else None
                                         for index, (removed, total) in pairs.items()},
                "overall_pruning_ratio": pruned / native if native else None,
            }

        report = ratios({index: [prefill[0] + decode[0], prefill[1] + decode[1]]
                         for index, (prefill, decode) in counts.items()})
        for phase, name in enumerate(("prefill", "decode")):
            report[name] = ratios({index: pair[phase] for index, pair in counts.items()})
        return report


def attach_activation_stats(model, layers, *, max_num_tokens=None):
    """layers contains (layer ID, patched module, native Top-K, device)."""
    if hasattr(model, "activation_stats_hook"):
        model.activation_stats_hook.remove()
        del model.activation_stats_hook
    stats = (ActivationStats({index: (topk, device) for index, _, topk, device in layers},
                             max_num_tokens=max_num_tokens) if activation_stats_enabled() else None)
    model.activation_stats = stats
    for index, module, _, _ in layers:
        module.activation_stats = stats
        module.activation_layer_id = index
    if stats is not None and max_num_tokens is None:
        signature = inspect.signature(model.forward)

        def set_forward_metadata(module, args, kwargs):
            inputs = signature.bind_partial(*args, **kwargs).arguments
            cache = inputs.get("past_key_values")
            stats.phase = int(cache is not None and cache.get_seq_length() > 0)
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
            set_forward_metadata, with_kwargs=True)
    return stats


class ActivationStatsWorkerMixin:
    """vLLM worker RPC access and per-token phases for CUDA graph replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The V1 runner packs real tokens before graph padding. DBO splits that
        # prefix into microbatches; PP would require merging disjoint layer sets.
        if activation_stats_enabled() and (
                self.use_v2_model_runner or self.parallel_config.use_ubatching
                or self.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("Activation statistics support vLLM V1 with single GPU / TP, without DBO / PP")

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if activation_stats_enabled():
            runner = self.model_runner
            prepare_inputs = runner._prepare_inputs

            def prepare_inputs_with_stats(scheduler_output, num_scheduled_tokens):
                result = prepare_inputs(scheduler_output, num_scheduled_tokens)
                num_tokens = scheduler_output.total_num_scheduled_tokens
                # Read the final packing order and corrected GPU positions,
                # including prefix cache hits and async speculative decoding.
                req_indices = runner.req_indices.gpu[:num_tokens]
                prompt_lengths = runner.input_batch.num_prompt_tokens_cpu_tensor[
                    :runner.input_batch.num_reqs].to(runner.device, non_blocking=True)
                phases = runner.positions[:num_tokens] >= prompt_lengths[req_indices]
                runner.get_model().activation_stats.set_token_phases(phases)
                return result

            runner._prepare_inputs = prepare_inputs_with_stats
        return result

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
