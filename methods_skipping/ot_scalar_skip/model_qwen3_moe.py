"""
OT Scalar Skip：仅对 router 概率分布 alpha 做标准最优传输（Optimal Transport）重分配。

核心思路：
- 原始 Top-K 专家的 router 概率分布为 alpha_1,...,alpha_K（sum=1）
- 剪枝后只保留 k_eff 个专家，跳过专家的概率质量需要重新分配给保留专家
- 定义专家间代价矩阵 C[i,j] = mean_t ||E_i(x_t) - E_j(x_t)||^2 / d（基于专家在真实输入上的输出）
- 标准 OT 框架：源分布 a = 跳过专家的归一化权重，目标分布 b = 保留专家的归一化权重
  用 Sinkhorn 迭代求解熵正则化 OT，得到传输计划 P，搬运后分布之和自然为 1
- 不做向量级补偿（无 S 矩阵），纯标量级分布传输

calib 阶段：
- 收集校准集上每层 MoE 的输入 hidden states
- 基于专家在真实输入上的实际输出计算代价矩阵 C
- 支持两种模式指定 k_eff：
  1) layer_topk: 手动指定每层 k_eff，如 [3,4,3,4]
  2) budget: 给定全局平均 k_eff（如 3.0），根据代价矩阵自动分配每层最优 k_eff

patch 阶段：
- 加载代价矩阵 C
- 推理时对每个 token：计算 router probs → 选 top-k_eff → Sinkhorn OT 重分配跳过专家的质量
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
            'ot_scalar_skip 需要 calib_kwargs 或 patch_kwargs 中的 layer_topk 或 budget，'
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
    """
    基于专家在真实输入上的实际输出来计算代价矩阵 C (E x E)。

    C[i,j] = mean_t ||E_i(x_t) - E_j(x_t)||^2 / d

    用恒等式 ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> 避免 O(E^2 * T * d) 显存。
    """
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


class OTScalarQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    应用 OT 标量级分布传输的 MoE Block。
    对每个 token：用熵正则化 OT 将跳过专家的 router 概率质量重分配给保留专家。
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        cost_matrix: torch.Tensor,
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
        self.register_buffer("C", cost_matrix.type_as(experts.gate_up_proj), persistent=False)

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

            scale_factor = redistributed_weights[token_idx, top_k_pos, None]
            current_hidden_states = current_hidden_states * scale_factor

            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class OTScalarSkipQwen3Moe(MoECompressor):
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

        logger.info("Loading model and tokenizer for OT scalar cost matrix calibration (ot_scalar_skip)")
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
                "ot_scalar_skip calib: num_moe_layers=%d, budget=%.2f, ot_reg=%.4f, max_cost_samples=%d",
                num_moe_layers, budget, ot_reg, max_cost_samples,
            )
        else:
            logger.info(
                "ot_scalar_skip calib: num_moe_layers=%d, layer_topk=%s, ot_reg=%.4f, max_cost_samples=%d",
                num_moe_layers, layer_topk, ot_reg, max_cost_samples,
            )

        hidden_states_collected: dict[int, list[torch.Tensor]] = {
            decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
        }
        router_indices_collected: dict[int, list[torch.Tensor]] = {
            decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
        }
        router_weights_collected: dict[int, list[torch.Tensor]] = {
            decoder_layer_idx: [] for decoder_layer_idx, _ in moe_layers
        }

        for decoder_layer_idx, block in moe_layers:
            def _forward(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx):
                bsz, seq_len, hidden_dim = hidden_states.shape
                hidden_reshaped = hidden_states.view(-1, hidden_dim)
                hidden_states_collected[_layer].append(hidden_reshaped.detach().cpu())

                if use_budget:
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

                return self_block.forward_orig(hidden_states)

            block.forward_orig = block.forward
            block.forward = types.MethodType(_forward, block)

        n_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(texts), batch_size),
            total=n_batches,
            desc="Calibration Forward (ot_scalar_skip)",
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

        for decoder_layer_idx, block in moe_layers:
            block.forward = block.forward_orig
            del block.forward_orig

        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
            "meta.ot_reg": torch.tensor(ot_reg, dtype=torch.float32),
        }

        cost_matrices: list[torch.Tensor] = []

        for decoder_layer_idx, block in tqdm(
            moe_layers, desc="Computing OT cost matrices from expert outputs", unit="layer"
        ):
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
            cost_matrices.append(C)
            logger.debug(
                "Layer %d cost matrix: shape=%s, min=%.6f, max=%.6f, mean=%.6f",
                decoder_layer_idx,
                tuple(C.shape),
                C.min().item(),
                C.max().item(),
                C.mean().item(),
            )

        if use_budget:
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

        state["meta.layer_topk"] = torch.tensor(layer_topk, dtype=torch.int32)

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))
        logger.info("Calibration completed. Adapter saved to %s", self._get_adapter_path())

    def patch(self, model, **kwargs) -> Any:
        norm = _resolve_norm(kwargs)
        sinkhorn_iters = _resolve_sinkhorn_iters(kwargs)
        if self.adapter_dir is None or not self.adapter_path.exists():
            raise FileNotFoundError("ot_scalar_skip patch 需提供有效 adapter_dir 且先运行 calib")

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
        logger.info("ot_scalar_skip patch: ot_reg=%.4f, sinkhorn_iters=%d", ot_reg, sinkhorn_iters)

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
            "Patching %d MoE layers with OT scalar transport, layer_topk=%s",
            len(moe_indices),
            layer_topk,
        )

        for j, decoder_layer_idx in enumerate(
            tqdm(moe_indices, desc="Patching layers (ot_scalar_skip)", unit="layer")
        ):
            block = layers[decoder_layer_idx].mlp
            key = f"layer_{decoder_layer_idx}.cost_matrix"
            if key not in state:
                raise KeyError(f"adapter 中缺少 {key}，请重新运行 calib")

            cost_matrix = state[key]
            k_eff = layer_topk[j]
            layers[decoder_layer_idx].mlp = OTScalarQwen3MoeSparseMoeBlock(
                block,
                cost_matrix=cost_matrix,
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