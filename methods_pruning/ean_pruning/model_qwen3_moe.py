"""
EAN (Expert Activation Norm) 方法 - 基于专家激活范数的剪枝（Qwen3-MoE 实现）

【原理】
在校准集上统计每个专家所输出的激活向量的 L2 范数（Norm）的和。
值越大表示专家越重要，越小越优先被裁剪。

【适配 Qwen3-MoE】
- MoE 层为 Qwen3MoeSparseMoeBlock，内含 gate (Router) 和 experts
- 在校准 forward 时，对每个专家的输出（乘 routing weight 之前）计算 L2 范数，按专家累加求和
- patch 时：与 frequency_pruning 相同，替换为 PrunedQwen3MoeSparseMoeBlock
"""

from __future__ import annotations

import copy
import gc
import logging
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from MoECompressor import MoECompressor
from utils.adapter_calib_config import saved_prune_ratio_from_adapter_dir
from utils.moe_stats import MoEStatsCollector, build_router_prob_hist
from utils.pruning_keep import recompute_keep_indices_from_scores


def _get_moe_layers(model) -> list[tuple[int, Qwen3MoeSparseMoeBlock]]:
    """
    遍历模型，找出所有 MoE 层（Qwen3MoeSparseMoeBlock）。

    Returns:
        [(decoder_layer_idx, mlp_block), ...]
    """
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            moe_layers.append((i, layer.mlp))
    return moe_layers


def _experts_forward_with_norm_collection(
    experts_module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    norm_stats: dict,
    layer_idx: int,
) -> torch.Tensor:
    """
    执行 experts forward，同时收集每个专家输出的 L2 范数统计。

    与 Qwen3MoeExperts.forward 逻辑一致，但在乘 routing weight 之前计算每个 token 输出的 L2 范数，
    按专家累加 sum_norm 和 count。
    """
    num_experts = experts_module.num_experts
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for idx in expert_hit:
        expert_idx = idx[0].item()
        if expert_idx >= num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = experts_module.act_fn(gate) * up
        # 专家输出（乘 routing weight 之前）
        expert_output = F.linear(current_hidden_states, experts_module.down_proj[expert_idx])
        # 计算每个 token 输出的 L2 范数
        norms = torch.norm(expert_output.float(), p=2, dim=-1)
        # 累加统计
        if layer_idx not in norm_stats:
            norm_stats[layer_idx] = {}
        if expert_idx not in norm_stats[layer_idx]:
            norm_stats[layer_idx][expert_idx] = [0.0, 0]  # [sum_norm, count]
        norm_stats[layer_idx][expert_idx][0] += norms.sum().item()
        norm_stats[layer_idx][expert_idx][1] += norms.numel()

        current_hidden_states = expert_output * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


class PrunedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    剪枝后的 SparseMoeBlock（与 frequency_pruning 共用同一实现）。

    1. Router：在原 logits 上将「被剪专家」对应位置置为 -inf
    2. Experts：只保留 keep_indices 对应的专家权重
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        keep_indices: torch.LongTensor,
        old_to_new: torch.LongTensor,
        layer_idx: int,
        layer_stats: dict[str, torch.Tensor] | None = None,
        stats_collector: MoEStatsCollector | None = None,
    ):
        super().__init__()
        # deepcopy gate 避免引用 original_block，防止 GC 无法回收原始 experts 导致显存泄漏
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.keep_indices = keep_indices
        self.old_to_new = old_to_new
        self.num_kept = len(keep_indices)
        self.layer_idx = layer_idx
        self.layer_stats = layer_stats or {}
        self.stats_collector = stats_collector
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj[keep_indices].clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj[keep_indices].clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_logits = router_logits.clone()
        pruned_mask = self.old_to_new == -1
        router_logits[:, pruned_mask] = float("-inf")
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        if "expert_importance" in self.layer_stats:
            importance = self.layer_stats["expert_importance"].to(router_probs.device, dtype=router_probs.dtype)
            if importance.numel() == router_probs.shape[-1]:
                importance = importance.clamp_min(0)
                importance = importance / importance.max().clamp_min(1e-12)
                router_probs = router_probs * importance.unsqueeze(0)
                router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (router_top_value / router_top_value.sum(dim=-1, keepdim=True)).to(router_probs.dtype)
        routing_weights = router_top_value

        selected_experts_new = torch.full_like(router_indices, -1)
        valid_old = router_indices >= 0
        if valid_old.any():
            selected_experts_new[valid_old] = self.old_to_new[router_indices[valid_old]]
        if self.stats_collector is not None:
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=router_indices.detach(),
                default_top_k=self.top_k,
                sequence_length=sequence_length,
            )

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_kept):
            token_idx, top_k_pos = torch.where(selected_experts_new == expert_idx)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states_reshaped[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class EANPruningQwen3Moe(MoECompressor):
    """
    基于 Expert Activation Norm 的专家剪枝，适配 Qwen3-MoE。

    统计每个专家输出激活向量的 L2 范数均值，剪掉范数最小的专家。
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: str | Path | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            adapter_dir=adapter_dir,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上统计每个专家输出的 L2 范数均值，保存 adapter。
        """
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")
        prune_ratio = float(kwargs.get("prune_ratio", 0.5))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[calib] Step 0/4: Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        logger.info("[calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )

        logger.info("[calib] Step 2/4: Forward pass to collect expert activation norms (EAN)")
        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts
        top_k = model.config.num_experts_per_tok

        # norm_stats[layer_idx][expert_idx] = [sum_norm, count]
        norm_stats: dict[int, dict[int, list[float]]] = {}
        expert_counts: dict[int, torch.Tensor] = {i: torch.zeros(num_experts) for i, _ in moe_layers}
        router_hist_bins = 16
        router_hist: dict[int, torch.Tensor] = {
            i: torch.zeros(router_hist_bins, dtype=torch.float32) for i, _ in moe_layers
        }
        layer_tokens: dict[int, int] = {i: 0 for i, _ in moe_layers}

        # 用 MethodType 替换 experts.forward，默认参数在定义时求值，正确捕获当前层的 decoder_layer_idx
        for decoder_layer_idx, block in moe_layers:
            experts = block.experts

            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer=decoder_layer_idx):
                return _experts_forward_with_norm_collection(
                    experts_module=self,
                    hidden_states=hidden_states,
                    top_k_index=top_k_index,
                    top_k_weights=top_k_weights,
                    norm_stats=norm_stats,
                    layer_idx=_layer,
                )

            experts.forward = types.MethodType(_forward, experts)

        # 运行校准 forward
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Calibration forward", unit="batch"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_context_len,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_router_logits=True)
            for router_pos, (decoder_layer_idx, _) in enumerate(moe_layers):
                logits = outputs.router_logits[router_pos]
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                probs = F.softmax(logits.float(), dim=-1)
                _, selected = torch.topk(probs, top_k, dim=-1)
                expert_counts[decoder_layer_idx] += torch.bincount(
                    selected.view(-1), minlength=num_experts
                ).type_as(expert_counts[decoder_layer_idx])
                hist, _ = build_router_prob_hist(probs, bins=router_hist_bins)
                router_hist[decoder_layer_idx] += hist
                layer_tokens[decoder_layer_idx] += int(logits.shape[0])

        logger.info("[calib] Step 3/4: Determining kept experts by EAN (Experts Activation Norm)")
        keep_per_layer = {}
        for decoder_layer_idx, _ in moe_layers:
            layer_stats = norm_stats.get(decoder_layer_idx, {})
            
            sum_norms = torch.full((num_experts,), 0.0, dtype=torch.float64)
            for expert_idx, (sum_norm, count) in layer_stats.items():
                sum_norms[expert_idx] = sum_norm

            num_keep = max(1, int(num_experts * (1 - prune_ratio)))
            _, top_indices = torch.topk(sum_norms, num_keep)
            keep_indices = top_indices.sort().values
            old_to_new = torch.full((num_experts,), -1, dtype=torch.long)
            for new_idx, old_idx in enumerate(keep_indices.tolist()):
                old_to_new[old_idx] = new_idx
            keep_per_layer[str(decoder_layer_idx)] = {
                "keep_indices": keep_indices.cpu(),
                "old_to_new": old_to_new.cpu(),
                "expert_importance": sum_norms.cpu(),
                "expert_activation_count": expert_counts[decoder_layer_idx].cpu(),
                "router_prob_hist": router_hist[decoder_layer_idx].cpu(),
                "router_cdf": torch.cumsum(
                    router_hist[decoder_layer_idx] / router_hist[decoder_layer_idx].sum().clamp_min(1e-12),
                    dim=0,
                ).cpu(),
                "calib_tokens": torch.tensor(layer_tokens[decoder_layer_idx], dtype=torch.long),
            }

        logger.info("[calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "meta.adapter_version": torch.tensor(2, dtype=torch.int32),
            "meta.router_hist_bins": torch.tensor(router_hist_bins, dtype=torch.int32),
        }
        for k, v in keep_per_layer.items():
            state[f"layer_{k}.keep_indices"] = v["keep_indices"]
            state[f"layer_{k}.old_to_new"] = v["old_to_new"]
            state[f"layer_{k}.expert_importance"] = v["expert_importance"]
            state[f"layer_{k}.expert_activation_count"] = v["expert_activation_count"]
            state[f"layer_{k}.router_prob_hist"] = v["router_prob_hist"]
            state[f"layer_{k}.router_cdf"] = v["router_cdf"]
            state[f"layer_{k}.calib_tokens"] = v["calib_tokens"]
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        """
        打补丁：读取 adapter，将给定 model 的每层 MoE 替换为 PrunedQwen3MoeSparseMoeBlock。
        """
        prune_ratio = kwargs.get("prune_ratio")
        if prune_ratio is None:
            raise ValueError("[ean][patch] 需要在 patch_kwargs 中提供 prune_ratio")
        prune_ratio = float(prune_ratio)
        if self.adapter_dir is None:
            raise ValueError("patch 需提供 adapter_dir")

        state = {}
        if self.adapter_dir is not None:
            logger.info("[patch] Loading adapter")
            if not self.adapter_path.exists():
                raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")
            state = load_file(str(self.adapter_path))
        else:
            logger.info("[patch] No adapter provided, using identity expert mapping for acceleration-only mode")

        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        stats_collector.initialize_layers(moe_indices)
        logger.info("[patch] Replacing %d MoE layers", len(moe_indices))

        saved_pr = saved_prune_ratio_from_adapter_dir(self.adapter_dir)

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            key_pre = f"layer_{decoder_layer_idx}"
            keep_key = f"{key_pre}.keep_indices"
            map_key = f"{key_pre}.old_to_new"
            imp_key = f"{key_pre}.expert_importance"
            num_experts = block.gate.num_experts
            recompute = (
                saved_pr is not None
                and abs(prune_ratio - float(saved_pr)) > 1e-5
                and imp_key in state
                and state[imp_key].numel() == num_experts
            )
            if recompute:
                importance = state[imp_key]
                keep_indices, old_to_new = recompute_keep_indices_from_scores(
                    importance, num_experts, prune_ratio
                )
                logger.info(
                    "[ean][patch] layer %d: eval prune_ratio=%s 与 calib(%s) 不同，从 expert_importance 重算 keep",
                    decoder_layer_idx,
                    prune_ratio,
                    saved_pr,
                )
            elif keep_key in state and map_key in state:
                keep_indices = state[keep_key]
                old_to_new = state[map_key]
            else:
                keep_indices = torch.arange(num_experts, dtype=torch.long)
                old_to_new = torch.arange(num_experts, dtype=torch.long)
            layer_stats = {}
            for stat_key in ("expert_importance", "expert_activation_count", "router_prob_hist", "router_cdf", "calib_tokens"):
                full_key = f"{key_pre}.{stat_key}"
                if full_key in state:
                    layer_stats[stat_key] = state[full_key]
            pruned_block = PrunedQwen3MoeSparseMoeBlock(
                block,
                keep_indices.to(block.gate.weight.device),
                old_to_new.to(block.gate.weight.device),
                layer_idx=decoder_layer_idx,
                layer_stats=layer_stats,
                stats_collector=stats_collector,
            )
            layers[decoder_layer_idx].mlp = pruned_block
        self._acceleration_stats_collector = stats_collector

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
