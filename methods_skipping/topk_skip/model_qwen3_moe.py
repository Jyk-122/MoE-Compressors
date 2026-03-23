"""
TopK skipping：按 router logits 每层仅对 k 个专家做激活计算（k <= num_experts_per_tok）。

不剪枝权重；仅替换 MoE block 的 forward。本方法不依赖校准统计量；其他 skipping
若需统计量，应单独实现 calib()。
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


def _get_moe_layers(model) -> list[tuple[int, Qwen3MoeSparseMoeBlock]]:
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            moe_layers.append((i, layer.mlp))
    return moe_layers


def _resolve_k(kwargs: dict[str, Any]) -> int:
    k = kwargs.get("k")
    if k is None:
        raise ValueError('topk_skip 的 patch 需要 patch_kwargs 中的 k，例如 {"k": 2}')
    return int(k)


class TopKSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """Full expert weights; routing uses top-k (k) instead of gate.top_k."""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        k: int,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.k_skip = int(k)
        if not (1 <= self.k_skip <= self.top_k):
            raise ValueError(f"k 必须满足 1 <= k <= {self.top_k}（gate.top_k / num_experts_per_tok）")
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

        k_eff = self.k_skip
        router_top_value, router_indices = torch.topk(router_probs, k_eff, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        routing_weights = router_top_value

        if self.stats_collector is not None:
            padded = torch.full(
                (router_indices.shape[0], self.top_k),
                -1,
                dtype=torch.long,
                device=router_indices.device,
            )
            padded[:, :k_eff] = router_indices
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=padded.detach(),
                default_top_k=self.top_k,
            )

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(router_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states_reshaped[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class TopKSkipQwen3Moe(MoECompressor):
    """每层只计算 k 个专家的 MoE skipping（无 adapter）。"""

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
        """
        topk_skip 仅依据当前步的 router logits 选专家，**不需要**校准统计量；
        为兼容 `MoECompressor` 抽象接口保留为空实现。

        其他更复杂的 skipping 方法若需依赖校准集上的统计量来裁剪激活，应自行实现
        `calib()`（并可配合 `adapter_dir` 落盘），与剪枝侧类似。
        """
        logger.info("[topk_skip] calib 未使用（本方法不依赖校准统计量）")

    def patch(self, model, **kwargs) -> Any:
        k = _resolve_k(kwargs)
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        logger.info("[topk_skip][patch] Replacing %d MoE layers with k=%d", len(moe_indices), k)

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (topk_skip)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            layers[decoder_layer_idx].mlp = TopKSkippedQwen3MoeSparseMoeBlock(
                block,
                k=k,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
