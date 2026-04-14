"""
SGC-Skip (Sparse-Group Compensation Skip) for Qwen3-MoE.

核心设计（按讨论确认）：
1) 使用 top-p（在默认 top_k 内）确定保留集 A，被剪集 D 仅做重路由替代；
2) calib 中统计组补偿矩阵 s_{i,j,u}（u 为 hidden 维分组）：
      s = argmin_s E[ ||z_j - s * z_i||^2 ]，闭式解按分组逐一求；
3) 用“补偿后重建误差”构造 replaceability Q(i<-j)，用于 D->A 映射；
4) eval 时被替换专家质量转移到替代专家，前向只执行映射后的专家。
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
    layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            layers.append((i, layer.mlp))
    return layers


def _resolve_threshold(kwargs: dict[str, Any]) -> float:
    threshold = kwargs.get("threshold")
    if threshold is None:
        raise ValueError('sgc_skip 的 patch 需要 patch_kwargs 中的 threshold，例如 {"threshold": 0.8}')
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold 必须满足 0 < threshold <= 1")
    return threshold


def _resolve_replace_threshold(kwargs: dict[str, Any]) -> float:
    v = float(kwargs.get("replace_threshold", 0.0))
    if not (0.0 <= v <= 1.0):
        raise ValueError("replace_threshold 必须满足 0 <= replace_threshold <= 1")
    return v


def _resolve_score_router_power(kwargs: dict[str, Any]) -> float:
    return float(kwargs.get("score_router_power", 0.5))


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


def _expert_forward_from_experts(experts, expert_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(hidden_states, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    out = experts.act_fn(gate) * up
    return F.linear(out, experts.down_proj[expert_idx])


def _expert_forward_from_block(block, expert_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(hidden_states, block.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    out = block.act_fn(gate) * up
    return F.linear(out, block.down_proj[expert_idx])


def _split_groups(vec: torch.Tensor, num_groups: int) -> torch.Tensor:
    # vec: [hidden] or [batch, hidden]
    hidden = vec.shape[-1]
    if hidden % num_groups != 0:
        raise ValueError(f"hidden_dim={hidden} 不能被 num_groups={num_groups} 整除")
    group_size = hidden // num_groups
    if vec.dim() == 1:
        return vec.view(num_groups, group_size)
    return vec.view(vec.shape[0], num_groups, group_size)


def _top_p_mask(router_top_value: torch.Tensor, threshold: float) -> torch.Tensor:
    # 在默认 top_k 集合内归一化后做 top-p 判定
    p = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cumsum_probs = p.cumsum(dim=-1)
    num_keep = (cumsum_probs < float(threshold)).sum(dim=-1) + 1
    num_keep = num_keep.clamp(max=router_top_value.shape[-1])
    pos = torch.arange(router_top_value.shape[-1], device=router_top_value.device).unsqueeze(0)
    return pos < num_keep.unsqueeze(1)


def _sgc_calib_mlp_forward(
    *,
    layer_idx: int,
    threshold: float,
    num_groups: int,
    eps: float,
    lambda_use_router_prob: bool,
    num_store: dict[int, torch.Tensor],
    den_store: dict[int, torch.Tensor],
    tgt_store: dict[int, torch.Tensor],
    pair_weight_store: dict[int, torch.Tensor],
):
    """
    向量化版本：
    - expert forward: [T, k] → batched
    - pair 统计: einsum + mask
    - 聚合: index_add_
    """

    def _forward(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)   # [T, H]
        T = x.shape[0]

        experts = self.experts
        top_k = self.gate.top_k
        E = self.gate.num_experts

        # -------------------------
        # 1. router
        # -------------------------
        router_logits = F.linear(x, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)

        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        routing_weights = router_top_value  # [T, k]

        # -------------------------
        # 2. 正常 top-k forward（保持原行为）
        # -------------------------
        final_hidden_states = torch.zeros_like(x)
        active_experts = torch.unique(router_indices).tolist()

        for expert_idx in active_experts:
            token_idx, top_k_pos = torch.where(router_indices == int(expert_idx))
            if token_idx.numel() == 0:
                continue
            cur = _expert_forward_from_experts(experts, int(expert_idx), x[token_idx])
            cur = cur * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, cur.to(final_hidden_states.dtype))

        # -------------------------
        # 3. batched expert forward（核心优化）
        # -------------------------
        # expand x → [T, k, H]
        x_expand = x.unsqueeze(1).expand(-1, top_k, -1)   # [T, k, H]
        x_flat = x_expand.reshape(T * top_k, hidden_dim)  # [T*k, H]

        expert_ids = router_indices.reshape(-1)           # [T*k]

        # batched forward（按 expert 分组执行）
        Z_flat = torch.zeros_like(x_flat)

        unique_e = torch.unique(expert_ids)
        for e in unique_e.tolist():
            mask = (expert_ids == int(e))
            if mask.sum() == 0:
                continue
            Z_flat[mask] = _expert_forward_from_experts(
                experts, int(e), x_flat[mask]
            )

        Z = Z_flat.view(T, top_k, hidden_dim)   # [T, k, H]

        # -------------------------
        # 4. 分组
        # -------------------------
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 不能被 num_groups={num_groups} 整除")

        group_size = hidden_dim // num_groups
        Zg = Z.view(T, top_k, num_groups, group_size)   # [T, k, G, Dg]

        # -------------------------
        # 5. Gram 计算（pair 内积）
        # -------------------------
        # dot[t,i,j,g] = <z_i, z_j> (group-wise)
        dot = torch.einsum("tigh,tjgh->tijg", Zg, Zg)   # [T, k, k, G]

        norm = (Zg * Zg).sum(dim=-1)                    # [T, k, G]

        # -------------------------
        # 6. top-p mask
        # -------------------------
        active_mask = _top_p_mask(router_top_value, threshold=float(threshold))  # [T, k]
        keep_mask = active_mask
        drop_mask = ~active_mask

        # pair mask: j ∈ drop, i ∈ keep
        pair_mask = drop_mask.unsqueeze(2) & keep_mask.unsqueeze(1)  # [T, k, k]

        if not pair_mask.any():
            return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

        # -------------------------
        # 7. λ 权重
        # -------------------------
        p_top = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if lambda_use_router_prob:
            lam = p_top  # [T, k]
        else:
            lam = torch.ones_like(p_top)

        lam_j = lam.unsqueeze(2)   # [T, k, 1]

        # -------------------------
        # 8. 统计量计算
        # -------------------------
        # num: <zi, zj>
        num_t = dot * lam_j.unsqueeze(-1) * pair_mask.unsqueeze(-1)   # [T,k,k,G]

        # den: ||zi||^2
        norm_i = norm.unsqueeze(2)   # [T,k,1,G]
        den_t = norm_i * lam_j.unsqueeze(-1) * pair_mask.unsqueeze(-1)

        # tgt: ||zj||^2
        norm_j = norm.unsqueeze(1)   # [T,1,k,G]
        tgt_t = norm_j * lam_j.unsqueeze(-1) * pair_mask.unsqueeze(-1)

        # pair weight
        pw_t = lam_j.squeeze(-1) * pair_mask   # [T,k,k]

        # -------------------------
        # 9. scatter-add 到 (E,E,G)
        # -------------------------
        num = num_store[layer_idx]
        den = den_store[layer_idx]
        tgt = tgt_store[layer_idx]
        pair_w = pair_weight_store[layer_idx]

        device = Z.device

        idx_i = router_indices.unsqueeze(2).expand(-1, -1, top_k)  # [T,k,k]
        idx_j = router_indices.unsqueeze(1).expand(-1, top_k, -1)

        idx_i = idx_i.reshape(-1)
        idx_j = idx_j.reshape(-1)

        flat_mask = pair_mask.reshape(-1)

        idx_i = idx_i[flat_mask]
        idx_j = idx_j[flat_mask]

        num_flat = num_t.reshape(-1, num_groups)[flat_mask]
        den_flat = den_t.reshape(-1, num_groups)[flat_mask]
        tgt_flat = tgt_t.reshape(-1, num_groups)[flat_mask]
        pw_flat = pw_t.reshape(-1)[flat_mask]

        # 关键：flatten index
        linear_idx = idx_i * E + idx_j

        num.view(-1, num_groups).index_add_(0, linear_idx, num_flat.to(num.dtype))
        den.view(-1, num_groups).index_add_(0, linear_idx, den_flat.to(den.dtype))
        tgt.view(-1, num_groups).index_add_(0, linear_idx, tgt_flat.to(tgt.dtype))
        pair_w.view(-1).index_add_(0, linear_idx, pw_flat.to(pair_w.dtype))

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    return _forward


def _finalize_sgc_stats(
    *,
    num: torch.Tensor,
    den: torch.Tensor,
    tgt: torch.Tensor,
    pair_w: torch.Tensor,
    eps: float,
    replace_temperature: float,
    scale_clip_min: float,
    scale_clip_max: float,
    shrink_c: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 calib 累积统计量构造：
      comp_scale [E,E,G]
      replaceability [E,E]
    """
    # 闭式解 s = num / den
    scale = num / den.clamp_min(float(eps))

    # 小样本回缩到 1： s_hat = a*s + (1-a)*1, a = n/(n+c)
    if shrink_c > 0.0:
        a = pair_w / (pair_w + float(shrink_c))
        scale = a.unsqueeze(-1) * scale + (1.0 - a.unsqueeze(-1))

    scale = scale.clamp(float(scale_clip_min), float(scale_clip_max))

    # 补偿后重建误差（分组合并）
    residual = tgt - 2.0 * scale * num + (scale * scale) * den
    residual = residual.clamp_min(0.0).sum(dim=-1)
    target = tgt.clamp_min(0.0).sum(dim=-1).clamp_min(float(eps))
    rel = torch.sqrt((residual / target).clamp_min(0.0))
    replaceability = torch.exp(-rel / max(float(replace_temperature), 1e-8))

    # 未观测 pair（pair_w==0）置为默认：scale=1, q=0（对角线单独置1）
    unseen = pair_w <= 0
    if unseen.any():
        scale = torch.where(unseen.unsqueeze(-1), torch.ones_like(scale), scale)
        replaceability = torch.where(unseen, torch.zeros_like(replaceability), replaceability)

    e = scale.shape[0]
    idx = torch.arange(e)
    scale[idx, idx] = 1.0
    replaceability[idx, idx] = 1.0
    replaceability = replaceability.clamp(0.0, 1.0)
    return scale.to(torch.float32), replaceability.to(torch.float32)


class SGCSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """top-p 保留集合 + 组补偿映射的 skipping block。"""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        comp_scale: torch.Tensor,  # [E,E,G]
        replaceability: torch.Tensor,  # [E,E]
        threshold: float,
        replace_threshold: float,
        score_router_power: float,
        norm: bool,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.threshold = float(threshold)
        self.replace_threshold = float(replace_threshold)
        self.score_router_power = float(score_router_power)
        self.norm = bool(norm)

        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

        if comp_scale.dim() != 3:
            raise ValueError(f"comp_scale 维度错误，期望 [E,E,G]，收到 shape={tuple(comp_scale.shape)}")
        if comp_scale.shape[0] != self.num_experts or comp_scale.shape[1] != self.num_experts:
            raise ValueError(
                f"comp_scale 前两维应为 (E,E)=({self.num_experts},{self.num_experts})，收到 {tuple(comp_scale.shape)}"
            )
        self.num_groups = int(comp_scale.shape[-1])

        if replaceability.shape != (self.num_experts, self.num_experts):
            raise ValueError(
                f"replaceability 形状错误: {tuple(replaceability.shape)}，期望 {(self.num_experts, self.num_experts)}"
            )
        self.register_buffer("comp_scale", comp_scale.float().clamp(0.0, 10.0), persistent=False)
        self.register_buffer("replaceability", replaceability.float().clamp(0.0, 1.0), persistent=False)

        self.layer_idx = layer_idx
        self.stats_collector = stats_collector

    def _expand_group_coeff(self, coeff_g: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        # coeff_g: [n_tok, G]
        if self.num_groups == 1:
            return coeff_g
        if hidden_dim % self.num_groups != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 不能被 num_groups={self.num_groups} 整除")
        group_size = hidden_dim // self.num_groups
        return coeff_g.unsqueeze(-1).expand(-1, -1, group_size).reshape(coeff_g.shape[0], hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(x, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # top-p 判定使用 top_k 内归一化概率
        p_top = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        active_mask = _top_p_mask(router_top_value, threshold=self.threshold)

        # 路由权重基准：与 topp_skip 一致
        routing_weights = router_top_value
        if self.gate.norm_topk_prob:
            routing_weights = (
                routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        n_tokens = router_indices.shape[0]
        device = router_indices.device

        # 每个 expert 聚合的 token 与分组系数（避免重复计算同一 expert 前向）
        per_expert_tokens: list[list[int]] = [[] for _ in range(self.num_experts)]
        per_expert_coeffs: list[list[torch.Tensor]] = [[] for _ in range(self.num_experts)]

        # 给 stats 的“实际执行专家”索引（去重后）
        selected_indices = torch.full_like(router_indices, -1)

        for n in range(n_tokens):
            idx_row = router_indices[n]
            w_row = routing_weights[n].to(torch.float32)
            p_row = p_top[n].to(torch.float32)
            keep_row = active_mask[n]

            keep_ids = idx_row[keep_row]
            if keep_ids.numel() == 0:
                # 理论不会发生（_top_p_mask 至少保留一个），兜底保留最大概率项
                best_pos = int(torch.argmax(p_row).item())
                keep_row = torch.zeros_like(keep_row, dtype=torch.bool)
                keep_row[best_pos] = True
                keep_ids = idx_row[keep_row]

            mapped_ids = idx_row.clone()
            for pos in range(self.top_k):
                j = int(idx_row[pos].item())
                if bool(keep_row[pos]):
                    mapped_ids[pos] = j
                    continue

                cand_ids = keep_ids.long()
                q = self.replaceability[cand_ids, j].to(torch.float32)
                if self.score_router_power != 0.0:
                    # 分数融合：replaceability * p_i^beta
                    keep_pos = torch.nonzero(keep_row, as_tuple=True)[0]
                    p_keep = p_row[keep_pos].clamp_min(1e-12)
                    q = q * torch.pow(p_keep, float(self.score_router_power))

                best_q, best_idx = torch.max(q, dim=0)
                if float(best_q.item()) < self.replace_threshold:
                    mapped_ids[pos] = j  # 不替换，保留原专家
                else:
                    mapped_ids[pos] = cand_ids[int(best_idx.item())]

            # 以 mapped_ids 聚合分组系数：coeff(i,g) = sum_{j->i} w_j * s_{i,j,g}
            unique_mapped = torch.unique(mapped_ids)
            coeff_norm = torch.zeros((self.num_experts,), device=device, dtype=torch.float32)
            for i in unique_mapped.tolist():
                i = int(i)
                map_mask = mapped_ids == i
                src_ids = idx_row[map_mask].long()
                src_w = w_row[map_mask].to(torch.float32)
                s = self.comp_scale[i, src_ids, :]  # [m, G]
                coeff_g = (src_w[:, None] * s).sum(dim=0)  # [G]
                if float(coeff_g.abs().sum().item()) <= 0.0:
                    continue
                per_expert_tokens[i].append(n)
                per_expert_coeffs[i].append(coeff_g)
                coeff_norm[i] = float(coeff_g.abs().sum().item())

            # 记录实际执行专家（按 coeff_norm 取前 top_k）
            vals, inds = torch.topk(coeff_norm, k=self.top_k, dim=-1)
            inds = inds.masked_fill(vals <= 0, -1)
            selected_indices[n] = inds

        if self.stats_collector is not None:
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=selected_indices.detach(),
                default_top_k=self.top_k,
                sequence_length=sequence_length,
            )

        final_hidden_states = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            if len(per_expert_tokens[expert_idx]) == 0:
                continue

            token_idx = torch.tensor(per_expert_tokens[expert_idx], device=device, dtype=torch.long)
            coeff_g = torch.stack(per_expert_coeffs[expert_idx], dim=0).to(device=device, dtype=x.dtype)

            z = _expert_forward_from_block(self, expert_idx, x[token_idx])  # [n_tok, hidden]
            if self.num_groups == 1:
                z = z * coeff_g
            else:
                coeff_h = self._expand_group_coeff(coeff_g, hidden_dim=hidden_dim)
                z = z * coeff_h

            final_hidden_states.index_add_(0, token_idx, z.to(final_hidden_states.dtype))

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


class SGCSkipQwen3Moe(MoECompressor):
    """SGC skipping：top-p 选保留集 + 组补偿替代。"""

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
            raise ValueError("sgc_skip calib 需提供 adapter_dir")

        # calib 超参（和 patch 的 top-p 阈值一致可减少分布漂移）
        threshold = float(kwargs.get("threshold", 0.8))
        num_groups = int(kwargs.get("num_groups", 16))
        eps = float(kwargs.get("eps", 1e-8))
        lambda_use_router_prob = bool(kwargs.get("lambda_use_router_prob", True))
        replace_temperature = float(kwargs.get("replace_temperature", 0.15))
        scale_clip_min = float(kwargs.get("scale_clip_min", 0.2))
        scale_clip_max = float(kwargs.get("scale_clip_max", 2.0))
        shrink_c = float(kwargs.get("shrink_c", 32.0))

        if not (0.0 < threshold <= 1.0):
            raise ValueError("calib threshold 必须满足 0 < threshold <= 1")
        if num_groups <= 0:
            raise ValueError("num_groups 必须为正整数")
        if replace_temperature <= 0.0:
            raise ValueError("replace_temperature 必须为正数")
        if scale_clip_min <= 0.0 or scale_clip_max < scale_clip_min:
            raise ValueError("scale clip 范围非法")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[sgc_skip][calib] Step 0/4: Loading model and tokenizer")
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

        logger.info("[sgc_skip][calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )
        if len(texts) == 0:
            raise RuntimeError("校准数据为空")

        moe_layers = _get_moe_layers(model)
        if len(moe_layers) == 0:
            raise RuntimeError("未找到 Qwen3 MoE 层")

        num_experts = int(model.config.num_experts)
        hidden_dim = int(model.config.hidden_size)
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 不能被 num_groups={num_groups} 整除")

        num_store: dict[int, torch.Tensor] = {}
        den_store: dict[int, torch.Tensor] = {}
        tgt_store: dict[int, torch.Tensor] = {}
        pair_weight_store: dict[int, torch.Tensor] = {}
        for layer_idx, _ in moe_layers:
            num_store[layer_idx] = torch.zeros((num_experts, num_experts, num_groups), dtype=torch.float64)
            den_store[layer_idx] = torch.zeros((num_experts, num_experts, num_groups), dtype=torch.float64)
            tgt_store[layer_idx] = torch.zeros((num_experts, num_experts, num_groups), dtype=torch.float64)
            pair_weight_store[layer_idx] = torch.zeros((num_experts, num_experts), dtype=torch.float64)

        logger.info(
            "[sgc_skip][calib] Step 2/4: Patched mlp.forward for compensation stats, "
            "samples=%d, batch_size=%d, threshold=%.4f, num_groups=%d",
            len(texts),
            batch_size,
            threshold,
            num_groups,
        )
        patched: list[tuple[Any, Any]] = []
        for layer_idx, block in moe_layers:
            patched.append((block, block.forward))
            block.forward = types.MethodType(
                _sgc_calib_mlp_forward(
                    layer_idx=layer_idx,
                    threshold=threshold,
                    num_groups=num_groups,
                    eps=eps,
                    lambda_use_router_prob=lambda_use_router_prob,
                    num_store=num_store,
                    den_store=den_store,
                    tgt_store=tgt_store,
                    pair_weight_store=pair_weight_store,
                ),
                block,
            )

        n_batches = (len(texts) + int(batch_size) - 1) // int(batch_size)
        model.eval()
        try:
            with torch.no_grad():
                for start in tqdm(
                    range(0, len(texts), int(batch_size)),
                    total=n_batches,
                    desc="SGC calib forward",
                    unit="batch",
                ):
                    batch_texts = texts[start : start + int(batch_size)]
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_context_len,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    if inputs["input_ids"].numel() == 0:
                        continue
                    _ = model(**inputs)
        finally:
            for block, orig_forward in patched:
                block.forward = orig_forward

        logger.info("[sgc_skip][calib] Step 3/4: Finalizing comp_scale & replaceability")
        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
            "meta.num_groups": torch.tensor(num_groups, dtype=torch.int32),
            "meta.threshold": torch.tensor(threshold, dtype=torch.float32),
            "meta.replace_temperature": torch.tensor(replace_temperature, dtype=torch.float32),
            "meta.scale_clip_min": torch.tensor(scale_clip_min, dtype=torch.float32),
            "meta.scale_clip_max": torch.tensor(scale_clip_max, dtype=torch.float32),
            "meta.shrink_c": torch.tensor(shrink_c, dtype=torch.float32),
        }

        for layer_idx, _ in moe_layers:
            comp_scale, replaceability = _finalize_sgc_stats(
                num=num_store[layer_idx],
                den=den_store[layer_idx],
                tgt=tgt_store[layer_idx],
                pair_w=pair_weight_store[layer_idx],
                eps=eps,
                replace_temperature=replace_temperature,
                scale_clip_min=scale_clip_min,
                scale_clip_max=scale_clip_max,
                shrink_c=shrink_c,
            )
            state[f"layer_{layer_idx}.comp_scale"] = comp_scale.cpu()
            state[f"layer_{layer_idx}.replaceability"] = replaceability.cpu()
            state[f"layer_{layer_idx}.pair_weight"] = pair_weight_store[layer_idx].to(torch.float32).cpu()

        logger.info("[sgc_skip][calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        threshold = _resolve_threshold(kwargs)
        replace_threshold = _resolve_replace_threshold(kwargs)
        score_router_power = _resolve_score_router_power(kwargs)
        norm = _resolve_norm(kwargs)

        if self.adapter_dir is None:
            raise ValueError("sgc_skip patch 需提供 adapter_dir")
        if self.adapter_path is None or not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")

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
            "[sgc_skip][patch] Replacing %d MoE layers with threshold=%.4f, replace_threshold=%.4f, "
            "score_router_power=%.4f, norm=%s",
            len(moe_indices),
            threshold,
            replace_threshold,
            score_router_power,
            norm,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (sgc_skip)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            scale_key = f"layer_{decoder_layer_idx}.comp_scale"
            rep_key = f"layer_{decoder_layer_idx}.replaceability"
            if scale_key not in state or rep_key not in state:
                raise KeyError(
                    f"adapter 缺少 {scale_key} 或 {rep_key}，请确认 calib 产物与当前模型一致"
                )
            layers[decoder_layer_idx].mlp = SGCSkippedQwen3MoeSparseMoeBlock(
                original_block=block,
                comp_scale=state[scale_key].to(block.gate.weight.device),
                replaceability=state[rep_key].to(block.gate.weight.device),
                threshold=threshold,
                replace_threshold=replace_threshold,
                score_router_power=score_router_power,
                norm=norm,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
