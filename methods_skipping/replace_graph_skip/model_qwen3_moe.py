"""
Replace-Graph skipping for Qwen3-MoE.

核心思路：
1) calib：为每层构建有向可替代性矩阵 Q(i <- j)。
   - 在 expert j 被路由到的 token 子集上，计算用 expert i 近似 j 的相对重建误差；
   - Q = exp(-error / temperature)，可选乘以条件共激活项。
2) patch：每个 token 在默认 top-k 内做贪心子集选择，
   最大化 sum_j p_j * max_{i in A} Q(i <- j)；
   然后把被替代专家的路由权重叠加到替代者（mass transfer / scatter_add）。

说明：
- 不改权重形状，仅替换 MoE block 的 forward。
- 与 top-p 不同：不是概率前缀截断，而是基于有向替代覆盖做子集选择。
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


def _expert_forward(experts_module, expert_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(hidden_states, experts_module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
    out = experts_module.act_fn(gate) * up
    return F.linear(out, experts_module.down_proj[expert_idx])


def _resolve_coverage_threshold(kwargs: dict[str, Any]) -> float:
    v = kwargs.get("coverage_threshold", kwargs.get("threshold"))
    if v is None:
        raise ValueError(
            'replace_graph_skip 的 patch 需要 coverage_threshold，例如 {"coverage_threshold": 0.9}'
        )
    v = float(v)
    if not (0.0 < v <= 1.0):
        raise ValueError("coverage_threshold 必须满足 0 < coverage_threshold <= 1")
    return v


def _resolve_min_keep(kwargs: dict[str, Any]) -> int:
    return int(kwargs.get("min_keep", 1))


def _resolve_max_keep(kwargs: dict[str, Any]) -> int | None:
    v = kwargs.get("max_keep")
    if v is None:
        return None
    return int(v)


def _resolve_replace_threshold(kwargs: dict[str, Any]) -> float:
    v = float(kwargs.get("replace_threshold", 0.0))
    if not (0.0 <= v <= 1.0):
        raise ValueError("replace_threshold 必须满足 0 <= replace_threshold <= 1")
    return v


def _resolve_min_gain(kwargs: dict[str, Any]) -> float:
    v = float(kwargs.get("min_gain", 1e-6))
    if v < 0.0:
        raise ValueError("min_gain 不能为负")
    return v


def _build_candidate_indices(block: Qwen3MoeSparseMoeBlock, candidate_top_r: int) -> torch.Tensor:
    """
    基于 down_proj 权重余弦相似度，为每个 target expert 预选替代候选。
    返回 shape [num_experts, r]，每行包含候选替代 expert 索引（含自身）。
    """
    experts = block.experts
    num_experts = experts.num_experts
    r = max(1, min(int(candidate_top_r), num_experts))

    # [E, H, I] -> [E, H*I]
    w = experts.down_proj.detach().float().reshape(num_experts, -1)
    w = F.normalize(w, p=2, dim=-1, eps=1e-12)
    sim = w @ w.T
    sim.fill_diagonal_(1.0)
    _, top_idx = torch.topk(sim, k=r, dim=-1)
    return top_idx.cpu()


def _finalize_replaceability_matrix(
    *,
    sum_error: torch.Tensor,
    target_count: torch.Tensor,
    coact: torch.Tensor,
    activation: torch.Tensor,
    temperature: float,
    coact_beta: float,
) -> torch.Tensor:
    """
    输入：
      sum_error[i, j] = 累积相对误差和（i 替代 j）
      target_count[j] = target j 的有效 token 数
    输出：
      replaceability[i, j] in [0,1]，越大表示 i 越能替代 j
    """
    num_experts = int(target_count.numel())
    q = torch.zeros((num_experts, num_experts), dtype=torch.float32)

    for j in range(num_experts):
        cnt = float(target_count[j].item())
        if cnt <= 0.0:
            continue
        d_col = (sum_error[:, j] / cnt).to(torch.float32)
        q[:, j] = torch.exp(-d_col / max(float(temperature), 1e-8))

    if coact_beta > 0.0:
        cond = coact.to(torch.float32) / activation.to(torch.float32).unsqueeze(0).clamp_min(1.0)
        cond = cond.clamp(0.0, 1.0)
        q = q * torch.pow(cond, float(coact_beta))

    q = q.clamp(0.0, 1.0)
    q.fill_diagonal_(1.0)
    return q


def _replace_graph_calib_mlp_forward(
    *,
    layer_idx: int,
    candidate_indices_store: dict[int, torch.Tensor],
    sum_error_store: dict[int, torch.Tensor],
    target_count_store: dict[int, torch.Tensor],
    coact_store: dict[int, torch.Tensor],
    activation_store: dict[int, torch.Tensor],
    eps: float,
):
    """
    calib 时临时替换到每层 mlp.forward。
    在保持原始 top-k 聚合输出的同时，统计有向替代误差矩阵。
    """

    def _forward(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(x, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_probs, self.gate.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        routing_weights = router_top_value

        experts = self.experts
        num_experts = experts.num_experts

        # 正常 MoE 输出
        final_hidden_states = torch.zeros_like(x)
        expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        active_experts = torch.unique(router_indices).tolist()
        for expert_idx in active_experts:
            token_idx, top_k_pos = torch.where(router_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            cur_x = x[token_idx]
            expert_out = _expert_forward(experts, int(expert_idx), cur_x)
            expert_cache[int(expert_idx)] = (token_idx, expert_out.detach().float())
            weighted = expert_out * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, weighted.to(final_hidden_states.dtype))

        # 条件共激活统计：coact[i,j] 与 activation[j]
        one_hot = F.one_hot(router_indices.long(), num_classes=num_experts).sum(dim=1).clamp(max=1).to(torch.float32)
        coact_store[layer_idx] += (one_hot.T @ one_hot).cpu().to(torch.float64)
        activation_store[layer_idx] += one_hot.sum(dim=0).cpu().to(torch.float64)

        # 有向替代误差统计
        candidates = candidate_indices_store[layer_idx].to(device=x.device)
        sum_error = sum_error_store[layer_idx]
        target_count = target_count_store[layer_idx]

        for target_idx, (token_idx, z_target) in expert_cache.items():
            # z_target: [n_toks, hidden]
            n_tok = int(token_idx.numel())
            if n_tok == 0:
                continue
            target_count[target_idx] += float(n_tok)

            x_target = x[token_idx]
            denom = torch.norm(z_target, p=2, dim=-1).clamp_min(float(eps))
            cand_row = candidates[target_idx]

            for repl_idx in cand_row.tolist():
                repl_idx = int(repl_idx)
                if repl_idx == target_idx:
                    z_repl = z_target
                else:
                    z_repl = _expert_forward(experts, repl_idx, x_target).float()

                alpha = (z_repl * z_target).sum(dim=-1) / z_repl.pow(2).sum(dim=-1).clamp_min(float(eps))
                residual = z_target - alpha.unsqueeze(-1) * z_repl
                rel_err = torch.norm(residual, p=2, dim=-1) / denom
                sum_error[repl_idx, target_idx] += float(rel_err.sum().item())

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    return _forward


def _greedy_select_subset(
    local_q: torch.Tensor,
    local_w: torch.Tensor,
    *,
    min_keep: int,
    max_keep: int,
    coverage_threshold: float,
    min_gain: float,
) -> torch.Tensor:
    """
    local_q: [m, m], 行=替代者 i，列=目标 j
    local_w: [m], 目标权重（和为 1）
    返回：被选替代者在局部 [0,m) 的索引。
    """
    m = int(local_w.numel())
    if m == 1:
        return torch.zeros((1,), dtype=torch.long, device=local_w.device)

    min_keep_eff = max(1, min(int(min_keep), m))
    max_keep_eff = max(min_keep_eff, min(int(max_keep), m))

    selected = torch.zeros((m,), dtype=torch.bool, device=local_w.device)
    anchor = int(torch.argmax(local_w).item())
    selected[anchor] = True

    covered = local_q[anchor].clone()
    coverage = float((local_w * covered).sum().item())

    while int(selected.sum().item()) < max_keep_eff:
        if coverage >= float(coverage_threshold) and int(selected.sum().item()) >= min_keep_eff:
            break

        best_idx = -1
        best_gain = -1.0
        for c in range(m):
            if bool(selected[c]):
                continue
            new_covered = torch.maximum(covered, local_q[c])
            gain = float((local_w * (new_covered - covered).clamp_min(0.0)).sum().item())
            if gain > best_gain:
                best_gain = gain
                best_idx = c

        if best_idx < 0 or best_gain <= float(min_gain):
            break

        selected[best_idx] = True
        covered = torch.maximum(covered, local_q[best_idx])
        coverage = float((local_w * covered).sum().item())

    if int(selected.sum().item()) < min_keep_eff:
        order = torch.argsort(local_w, descending=True)
        for idx in order.tolist():
            if not bool(selected[idx]):
                selected[idx] = True
            if int(selected.sum().item()) >= min_keep_eff:
                break

    return torch.nonzero(selected, as_tuple=True)[0]


def replace_graph_reroute(
    *,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    replaceability_matrix: torch.Tensor,
    num_experts: int,
    min_keep: int,
    max_keep: int,
    coverage_threshold: float,
    replace_threshold: float,
    min_gain: float,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对每个 token 的 top-k 集合做局部贪心选子集 + 有向替代映射 + 权重叠加。
    """
    out_indices = router_indices.clone()
    out_weights = routing_weights.clone()

    if active_mask is not None:
        token_mask = active_mask.flatten().bool()
        work_indices = router_indices[token_mask]
        work_weights = routing_weights[token_mask]
    else:
        token_mask = None
        work_indices = router_indices
        work_weights = routing_weights

    if work_indices.numel() == 0:
        return out_indices, out_weights

    n_tokens, top_k = work_indices.shape
    device = work_indices.device
    q_global = replaceability_matrix.to(device=device, dtype=torch.float32)

    for n in range(n_tokens):
        idx_row = work_indices[n]
        w_row = work_weights[n]

        valid = idx_row >= 0
        if not bool(valid.any()):
            continue

        idx_valid = idx_row[valid].long()
        w_valid = w_row[valid]
        m = int(idx_valid.numel())
        if m == 0:
            continue

        max_keep_eff = max(1, min(int(max_keep), m))
        min_keep_eff = max(1, min(int(min_keep), max_keep_eff))

        w_prob = w_valid.to(torch.float32)
        w_prob = w_prob / w_prob.sum().clamp_min(1e-12)

        # local_q[local_i, local_j] = Q(global_i <- global_j)
        local_q = q_global.index_select(0, idx_valid).index_select(1, idx_valid).clamp(0.0, 1.0)

        selected_local = _greedy_select_subset(
            local_q,
            w_prob,
            min_keep=min_keep_eff,
            max_keep=max_keep_eff,
            coverage_threshold=float(coverage_threshold),
            min_gain=float(min_gain),
        )

        q_selected = local_q.index_select(0, selected_local)
        best_score, best_pos = torch.max(q_selected, dim=0)
        mapped_local = selected_local[best_pos]

        if replace_threshold > 0.0:
            identity_local = torch.arange(m, device=device, dtype=torch.long)
            mapped_local = torch.where(best_score >= float(replace_threshold), mapped_local, identity_local)

        mapped_global = idx_valid[mapped_local]

        # mass transfer：把被替代专家的权重叠加给替代者
        mass = torch.zeros((num_experts,), device=device, dtype=w_row.dtype)
        mass.scatter_add_(0, mapped_global, w_valid)

        final_w, final_i = torch.topk(mass, k=top_k, dim=-1)
        final_i = final_i.masked_fill(final_w <= 0, -1)

        work_indices[n] = final_i
        work_weights[n] = final_w

    if token_mask is not None:
        out_indices[token_mask] = work_indices
        out_weights[token_mask] = work_weights
    else:
        out_indices = work_indices
        out_weights = work_weights

    return out_indices, out_weights


class ReplaceGraphSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """有向可替代图驱动的 token 内贪心重路由 skipping。"""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        replaceability_matrix: torch.Tensor,
        coverage_threshold: float,
        min_keep: int,
        max_keep: int | None,
        replace_threshold: float,
        min_gain: float,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts

        self.coverage_threshold = float(coverage_threshold)
        self.min_keep = int(min_keep)
        self.max_keep = int(self.top_k if max_keep is None else max_keep)
        self.replace_threshold = float(replace_threshold)
        self.min_gain = float(min_gain)

        if not (1 <= self.min_keep <= self.top_k):
            raise ValueError(f"min_keep 必须满足 1 <= min_keep <= {self.top_k}")
        if not (self.min_keep <= self.max_keep <= self.top_k):
            raise ValueError(f"max_keep 必须满足 min_keep <= max_keep <= {self.top_k}")

        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

        if replaceability_matrix.shape != (self.num_experts, self.num_experts):
            raise ValueError(
                f"layer {layer_idx} replaceability_matrix 形状错误: {tuple(replaceability_matrix.shape)}，"
                f"期望 {(self.num_experts, self.num_experts)}"
            )
        q = replaceability_matrix.float().clone().clamp(0.0, 1.0)
        q.fill_diagonal_(1.0)
        self.register_buffer("replaceability_matrix", q, persistent=False)

        self.layer_idx = layer_idx
        self.stats_collector = stats_collector

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        routing_weights = router_top_value

        active_mask = None
        if self.stats_collector is not None:
            active_mask = self.stats_collector._active_attention_mask

        rerouted_indices, rerouted_weights = replace_graph_reroute(
            router_indices=router_indices,
            routing_weights=routing_weights,
            replaceability_matrix=self.replaceability_matrix,
            num_experts=self.num_experts,
            min_keep=self.min_keep,
            max_keep=self.max_keep,
            coverage_threshold=self.coverage_threshold,
            replace_threshold=self.replace_threshold,
            min_gain=self.min_gain,
            active_mask=active_mask,
        )

        if self.stats_collector is not None:
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=rerouted_indices.detach(),
                default_top_k=self.top_k,
                sequence_length=sequence_length,
            )

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(rerouted_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states_reshaped[token_idx]
            current_hidden_states = _expert_forward(self, expert_idx, current_state)
            current_hidden_states = current_hidden_states * rerouted_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class ReplaceGraphSkipQwen3Moe(MoECompressor):
    """有向可替代图 + 贪心覆盖选择的 skipping。"""

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
            raise ValueError("replace_graph_skip 的 calib 需要提供 --adapter_dir")

        candidate_top_r = int(kwargs.get("candidate_top_r", 8))
        temperature = float(kwargs.get("temperature", 0.15))
        coact_beta = float(kwargs.get("coact_beta", 0.0))
        eps = float(kwargs.get("eps", 1e-8))
        if temperature <= 0.0:
            raise ValueError("temperature 必须为正数")
        if candidate_top_r <= 0:
            raise ValueError("candidate_top_r 必须为正整数")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[replace_graph_skip][calib] Step 0/4: Loading model and tokenizer")
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

        logger.info("[replace_graph_skip][calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )
        if len(texts) == 0:
            raise RuntimeError("校准数据为空，无法计算替代图")

        moe_layers = _get_moe_layers(model)
        if len(moe_layers) == 0:
            raise RuntimeError("未找到 Qwen3 MoE 层")

        layer_indices = [idx for idx, _ in moe_layers]
        num_experts = int(model.config.num_experts)

        candidate_indices: dict[int, torch.Tensor] = {}
        sum_error: dict[int, torch.Tensor] = {}
        target_count: dict[int, torch.Tensor] = {}
        coact: dict[int, torch.Tensor] = {}
        activation: dict[int, torch.Tensor] = {}

        for layer_idx, block in moe_layers:
            candidate_indices[layer_idx] = _build_candidate_indices(block, candidate_top_r)
            sum_error[layer_idx] = torch.zeros((num_experts, num_experts), dtype=torch.float64)
            target_count[layer_idx] = torch.zeros((num_experts,), dtype=torch.float64)
            coact[layer_idx] = torch.zeros((num_experts, num_experts), dtype=torch.float64)
            activation[layer_idx] = torch.zeros((num_experts,), dtype=torch.float64)

        logger.info(
            "[replace_graph_skip][calib] Step 2/4: Patched forward for directed replacement stats, "
            "samples=%d, batch_size=%d, candidate_top_r=%d, temperature=%.4f, coact_beta=%.4f",
            len(texts),
            batch_size,
            candidate_top_r,
            temperature,
            coact_beta,
        )

        patched: list[tuple[Any, Any]] = []
        for layer_idx, block in moe_layers:
            patched.append((block, block.forward))
            block.forward = types.MethodType(
                _replace_graph_calib_mlp_forward(
                    layer_idx=layer_idx,
                    candidate_indices_store=candidate_indices,
                    sum_error_store=sum_error,
                    target_count_store=target_count,
                    coact_store=coact,
                    activation_store=activation,
                    eps=eps,
                ),
                block,
            )

        model.eval()
        n_batches = (len(texts) + int(batch_size) - 1) // int(batch_size)
        try:
            with torch.no_grad():
                for start in tqdm(
                    range(0, len(texts), int(batch_size)),
                    total=n_batches,
                    desc="replace_graph calib forward",
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

        logger.info("[replace_graph_skip][calib] Step 3/4: Finalizing per-layer replaceability matrices")
        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
            "meta.temperature": torch.tensor(float(temperature), dtype=torch.float32),
            "meta.candidate_top_r": torch.tensor(int(candidate_top_r), dtype=torch.int32),
            "meta.coact_beta": torch.tensor(float(coact_beta), dtype=torch.float32),
        }
        for layer_idx in layer_indices:
            q = _finalize_replaceability_matrix(
                sum_error=sum_error[layer_idx],
                target_count=target_count[layer_idx],
                coact=coact[layer_idx],
                activation=activation[layer_idx],
                temperature=temperature,
                coact_beta=coact_beta,
            )
            state[f"layer_{layer_idx}.replaceability_matrix"] = q.cpu()
            state[f"layer_{layer_idx}.target_activation_count"] = target_count[layer_idx].cpu()

        logger.info("[replace_graph_skip][calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        coverage_threshold = _resolve_coverage_threshold(kwargs)
        min_keep = _resolve_min_keep(kwargs)
        max_keep = _resolve_max_keep(kwargs)
        replace_threshold = _resolve_replace_threshold(kwargs)
        min_gain = _resolve_min_gain(kwargs)

        if self.adapter_dir is None:
            raise ValueError("replace_graph_skip 的 patch 需要提供 --adapter_dir")
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
            "[replace_graph_skip][patch] Replacing %d MoE layers with coverage_threshold=%.4f, "
            "min_keep=%d, max_keep=%s, replace_threshold=%.4f, min_gain=%.6f",
            len(moe_indices),
            coverage_threshold,
            min_keep,
            str(max_keep),
            replace_threshold,
            min_gain,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (replace_graph_skip)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            key = f"layer_{decoder_layer_idx}.replaceability_matrix"
            if key not in state:
                raise KeyError(f"adapter 中缺少 {key}，请确认 calib 与当前模型层结构一致")
            layers[decoder_layer_idx].mlp = ReplaceGraphSkippedQwen3MoeSparseMoeBlock(
                original_block=block,
                replaceability_matrix=state[key].to(block.gate.weight.device),
                coverage_threshold=coverage_threshold,
                min_keep=min_keep,
                max_keep=max_keep,
                replace_threshold=replace_threshold,
                min_gain=min_gain,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
