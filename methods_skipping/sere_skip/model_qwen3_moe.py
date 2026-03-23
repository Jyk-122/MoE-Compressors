"""
SERE skipping for Qwen3-MoE.

核心思路：
1) calib: 在校准数据上计算每层 expert similarity matrix，并保存到 adapter.safetensors
2) patch: 推理时保留每个 token 的前 select_top_k(primary) 专家；
          对其余 secondary 专家按 similarity 重路由；若 best_sim < threshold 则保留原专家
"""

from __future__ import annotations

import copy
import gc
import logging
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


def _resolve_select_top_k(kwargs: dict[str, Any]) -> int:
    v = kwargs.get("select_top_k")
    if v is None:
        raise ValueError('sere_skip 的 patch 需要 patch_kwargs 中的 select_top_k，例如 {"select_top_k": 2}')
    return int(v)


def _resolve_threshold(kwargs: dict[str, Any]) -> float:
    v = kwargs.get("threshold")
    if v is None:
        raise ValueError('sere_skip 的 patch 需要 patch_kwargs 中的 threshold，例如 {"threshold": 0.3}')
    v = float(v)
    if not (0.0 <= v <= 1.0):
        raise ValueError("threshold 必须满足 0 <= threshold <= 1")
    return v


def _center_gram(k: torch.Tensor) -> torch.Tensor:
    mean_row = k.mean(dim=1, keepdim=True)
    mean_col = k.mean(dim=0, keepdim=True)
    mean_all = k.mean()
    return k - mean_row - mean_col + mean_all


def _cka_score(x: torch.Tensor, y: torch.Tensor, kernel: str = "linear") -> torch.Tensor:
    # x,y: [n_samples, hidden]
    if kernel == "linear":
        xc = x - x.mean(dim=0, keepdim=True)
        yc = y - y.mean(dim=0, keepdim=True)
        cxy = yc.T @ xc
        cxx = xc.T @ xc
        cyy = yc.T @ yc
        num = (cxy * cxy).sum()
        den = torch.sqrt((cxx * cxx).sum() * (cyy * cyy).sum()).clamp_min(1e-12)
        return torch.clamp(num / den, 0.0, 1.0)

    if kernel == "rbf":
        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True)
        dist_x = x_norm + x_norm.T - 2 * (x @ x.T)
        dist_y = y_norm + y_norm.T - 2 * (y @ y.T)
        var_x = torch.var(dist_x).clamp_min(1e-8)
        var_y = torch.var(dist_y).clamp_min(1e-8)
        kx = torch.exp(-dist_x / (2 * var_x))
        ky = torch.exp(-dist_y / (2 * var_y))
    elif kernel == "polynomial":
        kx = (x @ x.T + 1) ** 2
        ky = (y @ y.T + 1) ** 2
    else:
        raise ValueError(f"不支持的 CKA kernel: {kernel}")

    kxc = _center_gram((kx + kx.T) * 0.5)
    kyc = _center_gram((ky + ky.T) * 0.5)
    num = (kxc * kyc).sum()
    den = torch.sqrt((kxc * kxc).sum() * (kyc * kyc).sum()).clamp_min(1e-12)
    return torch.clamp(num / den, 0.0, 1.0)


def _compute_similarity_matrix(
    expert_outputs: list[torch.Tensor],
    method: str = "frobenius",
    kernel: str = "linear",
) -> torch.Tensor:
    n_experts = len(expert_outputs)
    sim = torch.zeros((n_experts, n_experts), device=expert_outputs[0].device, dtype=torch.float32)

    if method == "frobenius":
        dist = torch.zeros_like(sim)
        for i in range(n_experts):
            dist[i, i] = 0.0
            for j in range(i + 1, n_experts):
                d = torch.norm(expert_outputs[i] - expert_outputs[j], p="fro")
                dist[i, j] = d
                dist[j, i] = d
        max_d = dist.max().clamp_min(1e-12)
        sim = 1.0 - dist / max_d
    elif method == "cosine":
        for i in range(n_experts):
            sim[i, i] = 1.0
            for j in range(i + 1, n_experts):
                s = F.cosine_similarity(expert_outputs[i], expert_outputs[j], dim=1).mean()
                s = (s + 1.0) / 2.0
                sim[i, j] = s
                sim[j, i] = s
    elif method == "cka":
        for i in range(n_experts):
            sim[i, i] = 1.0
            for j in range(i + 1, n_experts):
                s = _cka_score(expert_outputs[i], expert_outputs[j], kernel=kernel)
                sim[i, j] = s
                sim[j, i] = s
    else:
        raise ValueError(f"不支持的 similarity_method: {method}，支持: frobenius/cosine/cka")

    sim.fill_diagonal_(1.0)
    return sim


def _compute_layer_similarity_from_hidden_states(
    block: Qwen3MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    method: str = "frobenius",
    kernel: str = "linear",
    max_tokens: int = 64,
) -> torch.Tensor:
    # hidden_states: [B, S, H]
    hidden_dim = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_dim)
    if max_tokens > 0 and x.shape[0] > max_tokens:
        x = x[:max_tokens]

    # Qwen3MoeExperts: gate_up_proj / down_proj / act_fn
    experts = block.experts
    expert_outputs: list[torch.Tensor] = []
    for expert_idx in range(block.gate.out_features):
        gate, up = F.linear(x, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        out = experts.act_fn(gate) * up
        out = F.linear(out, experts.down_proj[expert_idx])
        # 使用 float32 提升相似度稳定性
        expert_outputs.append(out.float())

    sim = _compute_similarity_matrix(expert_outputs, method=method, kernel=kernel)
    return sim


class SERESkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """SERE re-routing MoE block: reroute secondary experts by similarity."""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        similarity_matrix: torch.Tensor,
        select_top_k: int,
        threshold: float,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.select_top_k = int(select_top_k)
        self.threshold = float(threshold)
        if not (1 <= self.select_top_k <= self.top_k):
            raise ValueError(f"select_top_k 必须满足 1 <= select_top_k <= {self.top_k}")

        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

        # similarity 矩阵按层加载，shape [num_experts, num_experts]
        if similarity_matrix.shape != (self.num_experts, self.num_experts):
            raise ValueError(
                f"layer {layer_idx} similarity_matrix 形状错误: {tuple(similarity_matrix.shape)}，"
                f"期望 {(self.num_experts, self.num_experts)}"
            )
        sim = similarity_matrix.float().clone()
        sim.fill_diagonal_(1.0)
        self.register_buffer("similarity_matrix", sim, persistent=False)

        self.layer_idx = layer_idx
        self.stats_collector = stats_collector

    def _build_expert_mapping(self, router_indices: torch.Tensor) -> torch.Tensor:
        # router_indices: [num_tokens, top_k]
        device = router_indices.device
        mapping = torch.arange(self.num_experts, device=device, dtype=torch.long)
        if self.select_top_k >= self.top_k:
            return mapping

        primary = torch.unique(router_indices[:, : self.select_top_k])
        if primary.numel() == 0:
            return mapping
        primary_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=device)
        primary_mask[primary] = True

        secondary = torch.unique(router_indices[:, self.select_top_k :])
        if secondary.numel() == 0:
            return mapping

        sim = self.similarity_matrix.to(device=device)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=sim.dtype)

        for expert_id in secondary.tolist():
            if primary_mask[expert_id]:
                continue
            cand = sim[expert_id].masked_fill(~primary_mask, neg_inf)
            best_sim, best_primary = torch.max(cand, dim=0)
            if self.threshold > 0.0 and float(best_sim.item()) < self.threshold:
                mapping[expert_id] = expert_id
            else:
                mapping[expert_id] = best_primary.long()
        return mapping

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

        # SERE re-routing: secondary experts -> most similar primary experts
        expert_mapping = self._build_expert_mapping(router_indices)
        rerouted_indices = expert_mapping[router_indices]
        # 保留前 select_top_k 不变
        rerouted_indices[:, : self.select_top_k] = router_indices[:, : self.select_top_k]

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
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class SERESkipQwen3Moe(MoECompressor):
    """SERE skipping（Qwen3-MoE）：相似度驱动 secondary expert 重路由。"""

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
            raise ValueError("sere_skip 的 calib 需要提供 --adapter_dir")

        similarity_method = str(kwargs.get("similarity_method", "frobenius")).lower()
        kernel = str(kwargs.get("kernel", "linear")).lower()
        similarity_batch_size = int(kwargs.get("similarity_batch_size", max(1, batch_size)))
        similarity_max_len = int(kwargs.get("similarity_max_len", min(128, max_context_len)))
        similarity_max_tokens = int(kwargs.get("similarity_max_tokens", 64))

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[sere_skip][calib] Step 0/4: Loading model and tokenizer")
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

        logger.info("[sere_skip][calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )
        if len(texts) == 0:
            raise RuntimeError("校准数据为空，无法计算 similarity matrix")

        use_n = min(similarity_batch_size, len(texts))
        batch_texts = texts[:use_n]
        logger.info(
            "[sere_skip][calib] use %d texts, similarity_method=%s, kernel=%s, max_len=%d, max_tokens=%d",
            use_n,
            similarity_method,
            kernel,
            similarity_max_len,
            similarity_max_tokens,
        )

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=similarity_max_len,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        moe_layers = _get_moe_layers(model)
        if len(moe_layers) == 0:
            raise RuntimeError("未找到 Qwen3 MoE 层")

        logger.info("[sere_skip][calib] Step 2/4: Forward with hooks to compute per-layer similarity")
        similarity_per_layer: dict[int, torch.Tensor] = {}
        hooks = []

        def _make_hook(layer_idx: int):
            def _hook(module: Qwen3MoeSparseMoeBlock, module_inputs):
                if layer_idx in similarity_per_layer:
                    return
                hidden_states = module_inputs[0]
                sim = _compute_layer_similarity_from_hidden_states(
                    block=module,
                    hidden_states=hidden_states,
                    method=similarity_method,
                    kernel=kernel,
                    max_tokens=similarity_max_tokens,
                )
                similarity_per_layer[layer_idx] = sim.detach().cpu()
            return _hook

        for layer_idx, block in moe_layers:
            hooks.append(block.register_forward_pre_hook(_make_hook(layer_idx)))

        model.eval()
        with torch.no_grad():
            _ = model(**inputs)

        for h in hooks:
            h.remove()

        if len(similarity_per_layer) != len(moe_layers):
            missing = sorted({idx for idx, _ in moe_layers} - set(similarity_per_layer.keys()))
            raise RuntimeError(f"以下 MoE 层未收集到 similarity matrix: {missing}")

        logger.info("[sere_skip][calib] Step 3/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
        }
        for layer_idx, sim in similarity_per_layer.items():
            sim = sim.float()
            sim.fill_diagonal_(1.0)
            state[f"layer_{layer_idx}.similarity_matrix"] = sim
        save_file(state, str(self._get_adapter_path()))

        logger.info(
            "[sere_skip][calib] Step 4/4: Done. saved %d layer similarity matrices to %s",
            len(similarity_per_layer),
            self._get_adapter_path(),
        )

    def patch(self, model, **kwargs) -> Any:
        select_top_k = _resolve_select_top_k(kwargs)
        threshold = _resolve_threshold(kwargs)

        if self.adapter_dir is None:
            raise ValueError("sere_skip 的 patch 需要提供 --adapter_dir（用于加载 similarity matrix）")
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
            "[sere_skip][patch] Replacing %d MoE layers with select_top_k=%d, threshold=%.4f",
            len(moe_indices),
            select_top_k,
            threshold,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (sere_skip)", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            sim_key = f"layer_{decoder_layer_idx}.similarity_matrix"
            if sim_key not in state:
                raise KeyError(
                    f"adapter 中缺少 {sim_key}，请确认 calib 与当前模型层结构一致"
                )
            layers[decoder_layer_idx].mlp = SERESkippedQwen3MoeSparseMoeBlock(
                original_block=block,
                similarity_matrix=state[sim_key].to(block.gate.weight.device),
                select_top_k=select_top_k,
                threshold=threshold,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

