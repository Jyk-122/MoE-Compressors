"""
SERE skipping for Qwen3-MoE.

核心思路：
1) calib: 用校准专用 MoE forward（全专家算一次 → 更新 sim → 仅用 top-k 输出聚合），保存 adapter.safetensors
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


class SERECalibSparseMoeBlock(torch.nn.Module):
    """
    校准专用 MoE：单次 forward 内先对全部 expert 算一遍输出并更新相似度累加，
    再用同一批 expert 输出按 top-k 加权聚合，避免 hook 里二次跑全专家导致算力翻倍。
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        layer_idx: int,
        sum_sim_store: dict[int, torch.Tensor | None],
        similarity_method: str,
        kernel: str,
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.sum_sim_store = sum_sim_store
        self.similarity_method = str(similarity_method).lower()
        self.kernel = str(kernel).lower()

        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(x, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        routing_weights = router_top_value

        expert_outputs: list[torch.Tensor] = []
        for expert_idx in range(self.num_experts):
            gate_h, up_h = F.linear(x, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            out = self.act_fn(gate_h) * up_h
            out = F.linear(out, self.down_proj[expert_idx])
            expert_outputs.append(out.float())

        sim_batch = _compute_similarity_matrix(
            expert_outputs, method=self.similarity_method, kernel=self.kernel
        )
        acc = self.sum_sim_store.get(self.layer_idx)
        cpu_sim = sim_batch.detach().cpu()
        if acc is None:
            self.sum_sim_store[self.layer_idx] = cpu_sim.clone()
        else:
            acc += cpu_sim

        final_hidden_states = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(router_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            cur = expert_outputs[expert_idx][token_idx] * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, cur.to(final_hidden_states.dtype))

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


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

        # SERE：每个 token 的 secondary 只在该 token 自己的 top-S primary 里找最相似专家重路由
        rerouted_indices = router_indices.clone()
        S, K = self.select_top_k, self.top_k
        if S < K:
            sim = self.similarity_matrix.to(
                device=router_indices.device, dtype=torch.float32
            )
            prim = router_indices[:, :S].long()
            sec = router_indices[:, S:K].long()
            _, n_sec = sec.shape
            prim_rep = prim.unsqueeze(1).expand(-1, n_sec, -1)
            e_exp = sec.unsqueeze(-1).expand(-1, -1, S)
            sim_cand = sim[e_exp, prim_rep]
            best_sim, best_i = sim_cand.max(dim=-1)
            best_primary = torch.gather(prim_rep, 2, best_i.unsqueeze(-1)).squeeze(-1)
            if self.threshold > 0.0:
                new_sec = torch.where(best_sim < self.threshold, sec, best_primary)
            else:
                new_sec = best_primary
            rerouted_indices[:, S:K] = new_sec

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
        """逐条校准样本 forward；与框架接口兼容的 batch_size 若不为 1 将被忽略。"""
        if self.adapter_dir is None:
            raise ValueError("sere_skip 的 calib 需要提供 --adapter_dir")

        if int(batch_size) != 1:
            logger.warning(
                "[sere_skip][calib] 忽略 batch_size=%s：SERE 校准固定为每条文本单独前向（等价 batch_size=1）",
                batch_size,
            )

        similarity_method = str(kwargs.get("similarity_method", "frobenius")).lower()
        kernel = str(kwargs.get("kernel", "linear")).lower()

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

        moe_layers = _get_moe_layers(model)
        if len(moe_layers) == 0:
            raise RuntimeError("未找到 Qwen3 MoE 层")

        layer_indices = [idx for idx, _ in moe_layers]
        # 每层累加每条校准样本上的 S^{(l)}，最后除以样本数 N（论文式 (1) 后对 N 归一化）
        sum_sim: dict[int, torch.Tensor | None] = {idx: None for idx in layer_indices}
        n_samples_used = 0

        logger.info(
            "[sere_skip][calib] Step 2/4: 替换为校准 MoE（单次 forward 内全专家算 sim + top-k 聚合），"
            "逐条 forward %d 个样本，max_context_len=%d，similarity_method=%s, kernel=%s",
            len(texts),
            max_context_len,
            similarity_method,
            kernel,
        )

        layers = model.model.layers
        for layer_idx, block in moe_layers:
            layers[layer_idx].mlp = SERECalibSparseMoeBlock(
                original_block=block,
                layer_idx=layer_idx,
                sum_sim_store=sum_sim,
                similarity_method=similarity_method,
                kernel=kernel,
            )

        model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="SERE calib forward", unit="sample"):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_context_len,
                    padding=False,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                if inputs["input_ids"].numel() == 0:
                    continue
                _ = model(**inputs)
                n_samples_used += 1

        if n_samples_used == 0:
            raise RuntimeError(
                "校准未产生任何有效样本（请检查文本是否为空或被截断为无 token）"
            )

        missing = [idx for idx in layer_indices if sum_sim[idx] is None]
        if missing:
            raise RuntimeError(
                f"以下 MoE 层在校准中从未得到相似度矩阵（可能全程无有效 token）: {missing}"
            )

        similarity_per_layer: dict[int, torch.Tensor] = {}
        for idx in layer_indices:
            acc = sum_sim[idx]
            if acc is None:
                raise RuntimeError(f"layer {idx} 累加相似度为空（内部错误）")
            sim = (acc / float(n_samples_used)).float()
            sim.fill_diagonal_(1.0)
            similarity_per_layer[idx] = sim

        logger.info("[sere_skip][calib] Step 3/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "meta.adapter_version": torch.tensor(1, dtype=torch.int32),
        }
        for layer_idx, sim in similarity_per_layer.items():
            state[f"layer_{layer_idx}.similarity_matrix"] = sim.float()
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

