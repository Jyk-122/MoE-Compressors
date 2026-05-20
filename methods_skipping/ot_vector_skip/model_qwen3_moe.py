"""
OT Vector Skip：向量级最优传输，将传输张量 tau[i,j,d] 分解为 T[i,j] * S[i,d]。

核心思路：
- T[i,j]：标量传输矩阵，由标准 OT（Sinkhorn 迭代）在 router 概率分布上求解，
  源分布 = 跳过专家的归一化权重，目标分布 = 保留专家的归一化权重，搬运后分布之和自然为 1
- S[i,d]：逐专家、逐维度的缩放向量，由闭式解（线性回归）在校准集上计算，补偿向量级特征不匹配
- 最终输出：sum_{j in kept} beta_j * S[j] * E_j(x)
  其中 beta 由 Sinkhorn OT 传输计划 P 决定

与 OT Scalar Skip 的对比（消融目标）：
- OT Scalar Skip：仅 T（OT 重分配），无 S → 纯分布传输
- OT Vector Skip：T + S → 分布传输 + 向量级补偿
- 两者对比可量化 S 矩阵的增益

calib 阶段：
- 收集校准集上每层 MoE 的输入 hidden states
- 基于专家在真实输入上的实际输出计算代价矩阵 C
- 支持两种模式指定 k_eff：
  1) layer_topk: 手动指定每层 k_eff
  2) budget: 给定全局平均 k_eff，根据代价矩阵自动分配
- 跑校准前向，收集 A_stats/B_stats，用闭式解求 S 矩阵
- 保存 C 和 S 到 adapter

patch 阶段：
- 加载 C 和 S
- 推理时：Sinkhorn OT 重分配 router 权重 + S 矩阵逐点缩放专家输出
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


def _resolve_layer_topk(kwargs: dict[str, Any], num_layers: int | None = None) -> list[int] | None:
    layer_topk = kwargs.get("layer_topk")
    budget = kwargs.get("budget")
    if layer_topk is None and budget is None:
        raise ValueError(
            'ot_vector_skip 需要 calib_kwargs 或 patch_kwargs 中的 layer_topk 或 budget，'
            '例如 {"layer_topk": [3,4,3,4]} 或 {"budget": 3.0}'
        )
    if layer_topk is not None:
        layer_topk = [int(x) for x in layer_topk]
        if num_layers is not None and len(layer_topk) != num_layers:
            raise ValueError(f"layer_topk 长度必须为 {num_layers}，收到 {len(layer_topk)}")
        return layer_topk
    return None


def _compute_ot_cost_curve(
    cost_matrix: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    top_k: int,
    ot_reg: float,
    sinkhorn_iters: int = 50,
    max_tokens: int = 512,
) -> list[float]:
    T = router_indices.shape[0]
    if T > max_tokens:
        idx = torch.randperm(T, device=router_indices.device)[:max_tokens]
        router_indices = router_indices[idx]
        router_weights = router_weights[idx]
        T = max_tokens

    C_flat = cost_matrix.flatten().to(router_indices.device)
    E = cost_matrix.shape[0]
    costs: list[float] = []

    for k_eff in range(1, top_k + 1):
        if k_eff == top_k:
            costs.append(0.0)
            continue

        k = k_eff
        kept_indices = router_indices[:, :k]
        kept_weights = router_weights[:, :k]
        skipped_indices = router_indices[:, k:]
        skipped_weights = router_weights[:, k:]

        a = skipped_weights / skipped_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        b = kept_weights / kept_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        flat_idx = skipped_indices.unsqueeze(-1) * E + kept_indices.unsqueeze(1)
        C_sub = C_flat[flat_idx]

        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))
        log_K = -C_sub / ot_reg

        log_v = torch.zeros(T, k, device=C_sub.device, dtype=C_sub.dtype)
        for _ in range(sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

        log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
        P = torch.exp(log_P)

        ot_cost = (P * C_sub).sum(dim=(1, 2)).mean().item()
        costs.append(ot_cost)

    return costs


def _allocate_k_eff_from_budget(
    ot_cost_curves: list[list[float]],
    top_k: int,
    budget: float,
    min_k: int = 1,
) -> list[int]:
    num_layers = len(ot_cost_curves)
    total_budget = int(round(budget * num_layers))
    total_budget = max(min_k * num_layers, min(total_budget, top_k * num_layers))

    allocations = [min_k] * num_layers
    current_total = min_k * num_layers

    while current_total < total_budget:
        best_layer = -1
        best_reduction = -1.0

        for i in range(num_layers):
            if allocations[i] < top_k:
                cur_k = allocations[i]
                reduction = ot_cost_curves[i][cur_k - 1] - ot_cost_curves[i][cur_k]
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_layer = i

        if best_layer == -1:
            break

        allocations[best_layer] += 1
        current_total += 1

    return allocations


def _resolve_ot_reg(kwargs: dict[str, Any]) -> float:
    ot_reg = kwargs.get("ot_reg", 0.1)
    ot_reg = float(ot_reg)
    if ot_reg <= 0:
        raise ValueError("ot_reg 必须 > 0")
    return ot_reg


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
    raise ValueError("norm 必须是布尔值")


def _resolve_sinkhorn_iters(kwargs: dict[str, Any]) -> int:
    iters = int(kwargs.get("sinkhorn_iters", 50))
    if iters < 1:
        raise ValueError("sinkhorn_iters 必须 >= 1")
    return iters


def _compute_cost_matrix_from_outputs(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn,
    hidden_states: torch.Tensor,
    max_samples: int = 128,
) -> torch.Tensor:
    E = gate_up_proj.shape[0]
    d = down_proj.shape[1]

    num_tokens = hidden_states.shape[0]
    if num_tokens > max_samples:
        indices = torch.randperm(num_tokens, device=hidden_states.device)[:max_samples]
        hidden_states = hidden_states[indices]

    gate_up = torch.einsum("eod,td->eto", gate_up_proj.float(), hidden_states.float())
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = act_fn(gate) * up

    output = torch.einsum("edo,etf->etd", down_proj.float(), hidden)

    norm2 = (output * output).sum(dim=-1)
    inner = torch.einsum("etd,ftd->eft", output, output)
    C = (norm2.unsqueeze(0) + norm2.unsqueeze(1) - 2 * inner).mean(dim=-1) / d

    C = C.clamp_min(0)
    C_max = C.max()
    if C_max > 0:
        C = C / C_max
    return C


class OTVectorQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    应用 OT 向量级传输（T + S 分解）的 MoE Block。
    T：OT 标量传输重分配 router 权重
    S：逐专家逐维度缩放补偿特征不匹配
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        cost_matrix: torch.Tensor,
        S_matrix: torch.Tensor,
        k_eff: int,
        ot_reg: float,
        sinkhorn_iters: int,
        norm: bool,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.k_eff = int(k_eff)
        if not (1 <= self.k_eff <= self.top_k):
            raise ValueError(f"k_eff 必须满足 1 <= k_eff <= {self.top_k}")
        self.ot_reg = float(ot_reg)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm = bool(norm)
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector
        experts = original_block.experts

        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

        if cost_matrix.shape != (self.num_experts, self.num_experts):
            raise ValueError(
                f"cost_matrix 形状必须为 ({self.num_experts}, {self.num_experts})，"
                f"收到 {cost_matrix.shape}"
            )
        if S_matrix.shape != (self.num_experts, experts.down_proj.shape[1]):
            raise ValueError("S_matrix 形状不匹配")

        self.register_buffer("C", cost_matrix.type_as(experts.gate_up_proj), persistent=False)
        self.register_buffer("S", S_matrix.type_as(experts.gate_up_proj), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        orig_routing_weights = router_top_value.clone()
        if self.gate.norm_topk_prob:
            orig_routing_weights = (
                orig_routing_weights
                / orig_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        router_top_value_kept, router_indices_kept = torch.topk(router_probs, self.k_eff, dim=-1)
        if self.gate.norm_topk_prob:
            kept_routing_weights = (
                router_top_value_kept
                / router_top_value_kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        else:
            kept_routing_weights = router_top_value_kept

        redistributed_weights = kept_routing_weights.clone()

        if self.k_eff < self.top_k:
            K = self.top_k
            k = self.k_eff
            num_tokens = router_indices.shape[0]

            match = (router_indices.unsqueeze(-1) == router_indices_kept.unsqueeze(1))
            is_kept = match.any(dim=-1)
            skipped_mask = ~is_kept

            skipped_indices_flat = router_indices[skipped_mask]
            skipped_indices = skipped_indices_flat.view(num_tokens, K - k)
            skipped_weights_flat = orig_routing_weights[skipped_mask]
            skipped_weights = skipped_weights_flat.view(num_tokens, K - k)

            a = skipped_weights / skipped_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            b = kept_routing_weights

            C_flat = self.C.flatten()
            flat_idx = (
                skipped_indices.unsqueeze(-1) * self.C.shape[1] + router_indices_kept.unsqueeze(1)
            )
            C_sub = C_flat[flat_idx]

            log_a = torch.log(a.clamp_min(1e-12))
            log_b = torch.log(b.clamp_min(1e-12))
            log_K = -C_sub / self.ot_reg

            log_v = torch.zeros(num_tokens, k, device=C_sub.device, dtype=C_sub.dtype)
            for _ in range(self.sinkhorn_iters):
                log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
                log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(2), dim=1)

            log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
            P = torch.exp(log_P)

            total_skipped_mass = skipped_weights.sum(dim=-1, keepdim=True)
            transported_mass = P * total_skipped_mass.unsqueeze(-1)

            redistributed_weights = redistributed_weights + transported_mass.sum(dim=1)

        if self.norm:
            redistributed_weights = (
                redistributed_weights
                / redistributed_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        selected_indices = torch.full(
            (router_indices_kept.shape[0], self.top_k),
            -1,
            dtype=torch.long,
            device=router_indices_kept.device,
        )
        selected_indices[:, : self.k_eff] = router_indices_kept

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

            scale_factor = redistributed_weights[token_idx, top_k_pos, None] * self.S[expert_idx]
            current_hidden_states = current_hidden_states * scale_factor

            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class OTVectorSkipQwen3Moe(MoECompressor):
    def __init__(self, model_name_or_path: str, adapter_dir: str | Path | None = None, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, adapter_dir=adapter_dir, **kwargs)

    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model and tokenizer for OT vector calibration (ot_vector_skip)")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )
        texts = self.load_calibration_data(
            tokenizer, calibration_dataset, max_calib_samples, max_context_len
        )

        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts
        num_moe_layers = len(moe_layers)
        top_k = moe_layers[0][1].gate.top_k
        ot_reg = _resolve_ot_reg(kwargs)
        max_cost_samples = int(kwargs.get("max_cost_samples", 128))
        budget = kwargs.get("budget")
        layer_topk = _resolve_layer_topk(kwargs, num_moe_layers)
        use_budget = layer_topk is None and budget is not None
        if use_budget:
            budget = float(budget)
            logger.info(
                "ot_vector_skip calib: num_moe_layers=%d, budget=%.2f, ot_reg=%.4f, max_cost_samples=%d",
                num_moe_layers, budget, ot_reg, max_cost_samples,
            )
        else:
            logger.info(
                "ot_vector_skip calib: num_moe_layers=%d, layer_topk=%s, ot_reg=%.4f, max_cost_samples=%d",
                num_moe_layers, layer_topk, ot_reg, max_cost_samples,
            )

        A_stats: dict[int, torch.Tensor] = {}
        B_stats: dict[int, torch.Tensor] = {}
        hidden_states_collected: dict[int, list[torch.Tensor]] = {
            decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
        }

        if use_budget:
            router_indices_collected: dict[int, list[torch.Tensor]] = {
                decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
            }
            router_weights_collected: dict[int, list[torch.Tensor]] = {
                decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
            }

            for decoder_layer_idx, block in moe_layers:
                def _collect_hs(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx):
                    bsz, seq_len, hidden_dim = hidden_states.shape
                    hidden_reshaped = hidden_states.view(-1, hidden_dim)
                    hidden_states_collected[_layer].append(hidden_reshaped.detach().cpu())

                    router_logits = F.linear(hidden_reshaped, self_block.gate.weight)
                    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                    router_top_value, router_indices = torch.topk(
                        router_probs, self_block.gate.top_k, dim=-1
                    )
                    if self_block.gate.norm_topk_prob:
                        router_top_value = (
                            router_top_value
                            / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        )
                    router_indices_collected[_layer].append(router_indices.detach().cpu())
                    router_weights_collected[_layer].append(router_top_value.detach().cpu())

                    return self_block.original_forward(hidden_states)

                block.original_forward = block.forward
                block.forward = types.MethodType(_collect_hs, block)

            n_batches = (len(texts) + batch_size - 1) // batch_size
            for start in tqdm(
                range(0, len(texts), batch_size),
                total=n_batches,
                desc="Calibration Pass 1/2: collecting hidden states",
            ):
                batch_texts = texts[start : start + batch_size]
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_context_len,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    model(**inputs)

            for decoder_layer_idx, block in moe_layers:
                block.forward = block.original_forward
                del block.original_forward

            cost_matrices: list[torch.Tensor] = []
            for decoder_layer_idx, block in moe_layers:
                hs_list = hidden_states_collected[decoder_layer_idx]
                if not hs_list:
                    raise RuntimeError(
                        f"Layer {decoder_layer_idx}: no hidden states collected."
                    )
                all_hs = torch.cat([h for h in hs_list], dim=0)
                C = _compute_cost_matrix_from_outputs(
                    gate_up_proj=block.experts.gate_up_proj.data,
                    down_proj=block.experts.down_proj.data,
                    act_fn=block.experts.act_fn,
                    hidden_states=all_hs.to(model.device),
                    max_samples=max_cost_samples,
                ).cpu()
                cost_matrices.append(C)

            sinkhorn_iters_for_curve = int(kwargs.get("sinkhorn_iters", 50))
            ot_cost_curves: list[list[float]] = []
            for decoder_layer_idx, block in moe_layers:
                C = cost_matrices[decoder_layer_idx]
                r_idx_list = router_indices_collected[decoder_layer_idx]
                r_w_list = router_weights_collected[decoder_layer_idx]
                all_indices = torch.cat([t for t in r_idx_list], dim=0)
                all_weights = torch.cat([t for t in r_w_list], dim=0)
                curve = _compute_ot_cost_curve(
                    cost_matrix=C,
                    router_indices=all_indices,
                    router_weights=all_weights,
                    top_k=top_k,
                    ot_reg=ot_reg,
                    sinkhorn_iters=sinkhorn_iters_for_curve,
                )
                ot_cost_curves.append(curve)
                logger.debug(
                    "Layer %d OT cost curve: %s",
                    decoder_layer_idx,
                    ", ".join(f"k={i+1}:{v:.4f}" for i, v in enumerate(curve)),
                )
            layer_topk = _allocate_k_eff_from_budget(ot_cost_curves, top_k, budget)
            logger.info("Budget-based allocation: layer_topk=%s", layer_topk)

            for decoder_layer_idx, curve in enumerate(ot_cost_curves):
                state[f"layer_{decoder_layer_idx}.ot_cost_curve"] = torch.tensor(
                    curve, dtype=torch.float32
                )

            hidden_states_collected = {
                decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
            }

        for (decoder_layer_idx, block), k_eff in zip(moe_layers, layer_topk):
            def _forward(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx, _k=k_eff):
                bsz, seq_len, hidden_dim = hidden_states.shape
                hidden_reshaped = hidden_states.view(-1, hidden_dim)
                hidden_states_collected[_layer].append(hidden_reshaped.detach().cpu())
                num_tokens = hidden_reshaped.shape[0]

                router_logits = F.linear(hidden_reshaped, self_block.gate.weight)
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                router_top_value, router_indices = torch.topk(
                    router_probs, self_block.gate.top_k, dim=-1
                )

                orig_routing_weights = router_top_value.clone()
                if self_block.gate.norm_topk_prob:
                    orig_routing_weights = (
                        orig_routing_weights
                        / orig_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    )

                router_top_value_kept, router_indices_kept = torch.topk(router_probs, _k, dim=-1)
                if self_block.gate.norm_topk_prob:
                    kept_routing_weights = (
                        router_top_value_kept
                        / router_top_value_kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    ).to(router_probs.dtype)
                else:
                    kept_routing_weights = router_top_value_kept

                V = torch.zeros(
                    (num_tokens, num_experts, hidden_dim),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                Y = torch.zeros(
                    (num_tokens, hidden_dim),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )

                final_output = torch.zeros_like(hidden_reshaped)

                for expert_idx in range(num_experts):
                    token_idx_orig, top_k_pos_orig = torch.where(router_indices == expert_idx)
                    if token_idx_orig.numel() > 0:
                        current_state = hidden_reshaped[token_idx_orig]
                        gate, up = F.linear(
                            current_state, self_block.experts.gate_up_proj[expert_idx]
                        ).chunk(2, dim=-1)
                        expert_output = F.linear(
                            self_block.experts.act_fn(gate) * up,
                            self_block.experts.down_proj[expert_idx],
                        )
                        alpha_orig = orig_routing_weights[token_idx_orig, top_k_pos_orig, None]
                        Y.index_add_(0, token_idx_orig, expert_output * alpha_orig)
                        final_output.index_add_(
                            0, token_idx_orig, (expert_output * alpha_orig).to(final_output.dtype)
                        )

                    token_idx_kept, top_k_pos_kept = torch.where(router_indices_kept == expert_idx)
                    if token_idx_kept.numel() > 0:
                        current_state = hidden_reshaped[token_idx_kept]
                        gate, up = F.linear(
                            current_state, self_block.experts.gate_up_proj[expert_idx]
                        ).chunk(2, dim=-1)
                        expert_output = F.linear(
                            self_block.experts.act_fn(gate) * up,
                            self_block.experts.down_proj[expert_idx],
                        )
                        alpha_kept = kept_routing_weights[token_idx_kept, top_k_pos_kept, None]
                        V[token_idx_kept, expert_idx, :] = expert_output * alpha_kept

                if _layer not in A_stats:
                    A_stats[_layer] = torch.zeros(
                        (hidden_dim, num_experts, num_experts),
                        dtype=torch.float64,
                        device="cpu",
                    )
                    B_stats[_layer] = torch.zeros(
                        (hidden_dim, num_experts), dtype=torch.float64, device="cpu"
                    )

                chunk_size = 1024
                for start_d in range(0, hidden_dim, chunk_size):
                    end_d = min(hidden_dim, start_d + chunk_size)
                    V_chunk = V[:, :, start_d:end_d]
                    Y_chunk = Y[:, start_d:end_d]

                    A_stats[_layer][start_d:end_d] += torch.einsum(
                        "ted,tfd->def", V_chunk, V_chunk
                    ).cpu()
                    B_stats[_layer][start_d:end_d] += torch.einsum(
                        "ted,td->de", V_chunk, Y_chunk
                    ).cpu()

                return final_output.reshape(bsz, seq_len, hidden_dim)

            block.forward = types.MethodType(_forward, block)

        n_batches = (len(texts) + batch_size - 1) // batch_size
        pass_desc = "Calibration Pass 2/2 (ot_vector_skip)" if use_budget else "Calibration Forward (ot_vector_skip)"
        for start in tqdm(
            range(0, len(texts), batch_size),
            total=n_batches,
            desc=pass_desc,
        ):
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

        logger.info("Computing cost matrices and solving S matrices (ot_vector_skip)...")
        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
            "meta.ot_reg": torch.tensor(ot_reg, dtype=torch.float32),
        }

        cost_matrices_out: list[torch.Tensor] = []
        lambda_reg = 1e-4

        for decoder_layer_idx, block in moe_layers:
            hs_list = hidden_states_collected[decoder_layer_idx]
            if not hs_list:
                raise RuntimeError(
                    f"Layer {decoder_layer_idx}: no hidden states collected. "
                    "This indicates a bug in the calibration forward pass."
                )
            all_hs = torch.cat([h for h in hs_list], dim=0)
            C = _compute_cost_matrix_from_outputs(
                gate_up_proj=block.experts.gate_up_proj.data,
                down_proj=block.experts.down_proj.data,
                act_fn=block.experts.act_fn,
                hidden_states=all_hs.to(model.device),
                max_samples=max_cost_samples,
            ).cpu()

            state[f"layer_{decoder_layer_idx}.cost_matrix"] = C.float().contiguous()
            cost_matrices_out.append(C)
            logger.debug(
                "Layer %d cost matrix: shape=%s, min=%.6f, max=%.6f, mean=%.6f",
                decoder_layer_idx,
                tuple(C.shape),
                C.min().item(),
                C.max().item(),
                C.mean().item(),
            )

            if decoder_layer_idx in A_stats:
                A = A_stats[decoder_layer_idx]
                B = B_stats[decoder_layer_idx].unsqueeze(-1)

                eye = torch.eye(num_experts, dtype=torch.float64).unsqueeze(0)
                A_reg = A + lambda_reg * eye

                S_raw = torch.linalg.solve(A_reg, B).squeeze(-1)
                S_matrix = S_raw.transpose(0, 1)

                active_expert_mask = A.sum(dim=0).diagonal() > 1e-8
                S_matrix[~active_expert_mask, :] = 1.0

                state[f"layer_{decoder_layer_idx}.expert_S_matrix"] = (
                    S_matrix.float().contiguous()
                )

        state["meta.layer_topk"] = torch.tensor(layer_topk, dtype=torch.int32)

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))
        logger.info("Calibration completed. Adapter saved to %s", self._get_adapter_path())

    def patch(self, model, **kwargs) -> Any:
        norm = _resolve_norm(kwargs)
        sinkhorn_iters = _resolve_sinkhorn_iters(kwargs)
        if self.adapter_dir is None or not self.adapter_path.exists():
            raise FileNotFoundError("ot_vector_skip patch 需提供有效 adapter_dir 且先运行 calib")

        state = load_file(str(self.adapter_path))

        if "layer_topk" in kwargs:
            layer_topk = _resolve_layer_topk(kwargs)
        elif "budget" in kwargs:
            budget = float(kwargs["budget"])
            moe_indices_temp = [
                i for i, layer in enumerate(model.model.layers)
                if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
            ]
            top_k_val = int(model.model.layers[moe_indices_temp[0]].mlp.gate.top_k)
            ot_cost_curves: list[list[float]] = []
            for decoder_layer_idx in moe_indices_temp:
                curve_key = f"layer_{decoder_layer_idx}.ot_cost_curve"
                if curve_key in state:
                    ot_cost_curves.append(state[curve_key].tolist())
                else:
                    raise KeyError(
                        f"adapter 中缺少 {curve_key}，"
                        "请使用 budget 模式重新运行 calib 以生成 OT cost curves"
                    )
            layer_topk = _allocate_k_eff_from_budget(ot_cost_curves, top_k_val, budget)
            logger.info("Budget-based allocation (patch): layer_topk=%s", layer_topk)
        else:
            layer_topk = state["meta.layer_topk"].tolist()
            logger.info("Using saved layer_topk from calib: %s", layer_topk)

        ot_reg = _resolve_ot_reg(kwargs) if "ot_reg" in kwargs else float(state["meta.ot_reg"].item())
        logger.info("ot_vector_skip patch: ot_reg=%.4f, sinkhorn_iters=%d", ot_reg, sinkhorn_iters)

        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        stats_collector.initialize_layers(moe_indices)

        if len(layer_topk) != len(moe_indices):
            raise ValueError(
                f"layer_topk 长度 {len(layer_topk)} 与 MoE 层数 {len(moe_indices)} 不匹配"
            )

        logger.info(
            "Patching %d MoE layers with OT vector transport (T + S), layer_topk=%s",
            len(moe_indices),
            layer_topk,
        )

        for j, decoder_layer_idx in enumerate(
            tqdm(moe_indices, desc="Patching layers (ot_vector_skip)", unit="layer")
        ):
            block = layers[decoder_layer_idx].mlp
            cost_key = f"layer_{decoder_layer_idx}.cost_matrix"
            s_key = f"layer_{decoder_layer_idx}.expert_S_matrix"
            if cost_key not in state:
                raise KeyError(f"adapter 中缺少 {cost_key}，请重新运行 calib")
            if s_key not in state:
                raise KeyError(f"adapter 中缺少 {s_key}，请重新运行 calib")

            cost_matrix = state[cost_key]
            S_matrix = state[s_key]
            k_eff = layer_topk[j]
            layers[decoder_layer_idx].mlp = OTVectorQwen3MoeSparseMoeBlock(
                block,
                cost_matrix=cost_matrix,
                S_matrix=S_matrix,
                k_eff=k_eff,
                ot_reg=ot_reg,
                sinkhorn_iters=sinkhorn_iters,
                norm=norm,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model