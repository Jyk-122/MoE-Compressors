"""
Top-p skipping：在每个 token 的默认 top-k 路由结果中，按概率从高到低累加，
仅保留累计和首次达到 threshold 的最小专家集合，其余专家跳过计算。

说明：
- 不改变权重形状，仅替换 MoE block 的 forward。
- 为避免“跳过”退化为“增加计算”，仅在 gate.top_k 范围内做动态保留。
"""

from __future__ import annotations

import copy
import gc
import logging
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from MoECompressor import MoECompressor
from utils.moe_stats import MoEStatsCollector

logger = logging.getLogger("MoECompressor")


def _resolve_threshold(kwargs: dict[str, Any]) -> float:
    threshold = kwargs.get("threshold")
    if threshold is None:
        raise ValueError('topp_skip 的 patch 需要 patch_kwargs 中的 threshold，例如 {"threshold": 0.8}')
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold 必须满足 0 < threshold <= 1")
    return threshold


class TopPSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """Full expert weights; routing keeps minimal prefix with cumulative prob >= threshold."""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        threshold: float,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.threshold = float(threshold)
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 在默认 top_k 集合内做归一化后再执行 top-p 判定，避免 top_k 总概率质量不足时
        # threshold 语义失效（始终需要保留全部 top_k 才“最接近”阈值）。
        router_top_value_for_topp = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cumsum_probs = router_top_value_for_topp.cumsum(dim=-1)
        num_keep = (cumsum_probs < self.threshold).sum(dim=-1) + 1
        num_keep = num_keep.clamp(max=self.top_k)

        pos = torch.arange(self.top_k, device=router_indices.device).unsqueeze(0)
        active_mask = pos < num_keep.unsqueeze(1)

        routing_weights = router_top_value * active_mask.to(router_top_value.dtype)
        if self.gate.norm_topk_prob:
            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(router_probs.dtype)

        selected_indices = torch.where(
            active_mask,
            router_indices,
            torch.full_like(router_indices, -1),
        )
        if self.stats_collector is not None:
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=selected_indices.detach(),
                default_top_k=self.top_k,
            )

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(selected_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states_reshaped[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class TopPSkipQwen3Moe(MoECompressor):
    """每层按累计概率阈值 threshold 动态保留专家的 MoE skipping（无 adapter）。"""

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: str | None = None,
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
        logger.info("[topp_skip] calib 未使用（本方法不依赖校准统计量）")

    def patch(self, model, **kwargs) -> Any:
        threshold = _resolve_threshold(kwargs)
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        logger.info(
            "[topp_skip][patch] Replacing %d MoE layers with threshold=%.4f",
            len(moe_indices),
            threshold,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (topp_skip)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            layers[decoder_layer_idx].mlp = TopPSkippedQwen3MoeSparseMoeBlock(
                block,
                threshold=threshold,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
