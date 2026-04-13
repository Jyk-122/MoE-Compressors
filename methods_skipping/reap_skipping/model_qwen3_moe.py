"""
REAP-style skipping（reap_skipping）：在 topp_skip 的路由框架下，用校准得到的每位专家
输出特征平均 L2 模长 m_i，与当前 token 上的路由概率 p_i 组合为 p_i * m_i，
在默认 top_k 内按该分数排序后，对归一化质量做累积阈值选择（与 top-p 同形）。

- calib：参考 reap_pruning，统计每位专家在「乘 routing weight 之前」的专家输出向量的
  L2 范数在 token 上的平均值，得到 m_i，写入 adapter。
- patch：需要 adapter_dir 与 reap calib 产物；patch_kwargs 需 threshold（同 topp_skip）。
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
from safetensors.torch import load_file, save_file
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


def _experts_forward_collect_mean_output_norm(
    experts_module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    norm_stats: dict,
    layer_idx: int,
) -> torch.Tensor:
    """
    与 Qwen3MoeExperts.forward 一致，但在乘 routing weight 之前统计每位专家输出的 L2 范数，
    按专家累加 sum_norm 与 count（用于求平均模长 m_i，不乘 top-k 权重）。
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
        expert_output = F.linear(current_hidden_states, experts_module.down_proj[expert_idx])
        norms = torch.norm(expert_output.float(), p=2, dim=-1)

        if layer_idx not in norm_stats:
            norm_stats[layer_idx] = {}
        if expert_idx not in norm_stats[layer_idx]:
            norm_stats[layer_idx][expert_idx] = [0.0, 0]
        norm_stats[layer_idx][expert_idx][0] += norms.sum().item()
        norm_stats[layer_idx][expert_idx][1] += norms.numel()

        current_hidden_states = expert_output * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _resolve_threshold(kwargs: dict[str, Any]) -> float:
    threshold = kwargs.get("threshold")
    if threshold is None:
        raise ValueError(
            'reap_skipping 的 patch 需要 patch_kwargs 中的 threshold，例如 {"threshold": 0.8}'
        )
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold 必须满足 0 < threshold <= 1")
    return threshold


def _resolve_norm(kwargs: dict[str, Any]) -> bool:
    norm = kwargs.get("norm", True)
    if isinstance(norm, bool):
        return norm
    if isinstance(norm, (int, float)):
        return bool(norm)
    if isinstance(norm, str):
        v = norm.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError('norm 必须是布尔值（true/false），例如 {"threshold": 0.8, "norm": false}')


def _finalize_mean_norms(
    layer_stats: dict[int, list[float]],
    num_experts: int,
) -> torch.Tensor:
    mean_norms = torch.zeros((num_experts,), dtype=torch.float64)
    for expert_idx, (sum_norm, count) in layer_stats.items():
        if count > 0:
            mean_norms[expert_idx] = sum_norm / count
    # 从未被路由到的专家：用非零均值兜底，避免 p*m 全为 0
    nz = mean_norms > 1e-12
    if nz.any():
        fallback = mean_norms[nz].mean()
    else:
        fallback = torch.tensor(1.0, dtype=torch.float64)
    mean_norms = torch.where(nz, mean_norms, fallback)
    return mean_norms


class REAPSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """在 top_k 内按 (p_i * m_i) 的归一化累积质量做阈值裁剪，其余专家跳过。"""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        expert_mean_norm: torch.Tensor,
        threshold: float,
        norm: bool,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.threshold = float(threshold)
        self.norm = bool(norm)  # 与 topp_skip 一致，供 patch_kwargs 透传；路由归一化仍由 gate.norm_topk_prob 控制
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)
        # m_i：与专家索引对齐，避免过小导致数值问题
        m = expert_mean_norm.to(dtype=torch.float32).view(-1).clamp_min(1e-12)
        if m.numel() != self.num_experts:
            raise ValueError(
                f"expert_mean_norm 长度 {m.numel()} 与 num_experts={self.num_experts} 不一致"
            )
        self.register_buffer("expert_mean_norm", m, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        p = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        m_sel = self.expert_mean_norm[router_indices.long()]
        scores = p.to(torch.float32) * m_sel

        sorted_scores, sort_perm = torch.sort(scores, dim=-1, descending=True)
        mass = sorted_scores / sorted_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cum = mass.cumsum(dim=-1)
        num_keep = (cum < self.threshold).sum(dim=-1) + 1
        num_keep = num_keep.clamp(max=self.top_k)

        rank_kept = torch.arange(self.top_k, device=scores.device).unsqueeze(0) < num_keep.unsqueeze(1)
        active_mask = torch.zeros_like(router_indices, dtype=torch.bool)
        active_mask.scatter_(1, sort_perm.long(), rank_kept)

        routing_weights = router_top_value * active_mask.to(router_top_value.dtype)
        if self.gate.norm_topk_prob:
            routing_weights = (
                routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

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
                sequence_length=sequence_length,
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


class REAPSkippingQwen3Moe(MoECompressor):
    """先 calib 统计各层 m_i，再按 p_i*m_i 做类 top-p 的动态 skipping。"""

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
        if self.adapter_dir is None:
            raise ValueError("reap_skipping calib 需提供 adapter_dir")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[reap_skipping][calib] Step 0/4: Loading model and tokenizer")
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

        logger.info("[reap_skipping][calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )

        logger.info("[reap_skipping][calib] Step 2/4: Forward to collect per-expert mean output L2 norm")
        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts

        norm_stats: dict[int, dict[int, list[float]]] = {}

        for decoder_layer_idx, block in moe_layers:
            experts = block.experts

            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer=decoder_layer_idx):
                return _experts_forward_collect_mean_output_norm(
                    experts_module=self,
                    hidden_states=hidden_states,
                    top_k_index=top_k_index,
                    top_k_weights=top_k_weights,
                    norm_stats=norm_stats,
                    layer_idx=_layer,
                )

            experts.forward = types.MethodType(_forward, experts)

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
                model(**inputs)

        logger.info("[reap_skipping][calib] Step 3/4: Building expert_mean_norm tensors")
        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
        }
        for decoder_layer_idx, _ in moe_layers:
            layer_stat = norm_stats.get(decoder_layer_idx, {})
            mean_norms = _finalize_mean_norms(layer_stat, num_experts)
            state[f"layer_{decoder_layer_idx}.expert_mean_norm"] = mean_norms.cpu()

        logger.info("[reap_skipping][calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        threshold = _resolve_threshold(kwargs)
        norm = _resolve_norm(kwargs)
        if self.adapter_dir is None:
            raise ValueError("reap_skipping patch 需提供 adapter_dir")
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib")

        state = load_file(str(self.adapter_path))
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        stats_collector.initialize_layers(moe_indices)
        logger.info(
            "[reap_skipping][patch] Replacing %d MoE layers with threshold=%.4f, norm=%s",
            len(moe_indices),
            threshold,
            norm,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (reap_skipping)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            key = f"layer_{decoder_layer_idx}.expert_mean_norm"
            if key not in state:
                raise KeyError(f"adapter 中缺少 {key}，请使用本方法的 calib 重新生成")
            expert_mean_norm = state[key].type_as(layers[decoder_layer_idx].mlp.gate.weight)
            layers[decoder_layer_idx].mlp = REAPSkippedQwen3MoeSparseMoeBlock(
                block,
                expert_mean_norm=expert_mean_norm,
                threshold=threshold,
                norm=norm,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
