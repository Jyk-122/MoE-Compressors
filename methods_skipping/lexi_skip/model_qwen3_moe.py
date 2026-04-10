"""
LExI (Layer-Adaptive Active Experts): data-free per-layer top-k from sensitivity profiling + evolution.

**Calib (Stage 1)**：对每层 MoE 单独用高斯随机 hidden，度量不同 top-k 相对 baseline 的 Frobenius 输出偏差，得到敏感度矩阵 D 并写入 adapter。

**Eval / patch (Stage 2)**：按 `compute_reduction` 或 `target_budget` 确定全局专家预算 B，在 CPU 上进化搜索最小化 sum_j D[j,k_j-1]，再按每层 k_j 替换为与 topk_skip 相同的前向。

三次不同减算比例（如 25% / 40% / 50%）：**一次 calib** 后，仅改 `PATCH_KWARGS` 中 `compute_reduction` 重复 eval，无需重复 Stage1。
"""

from __future__ import annotations

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

from .lexi_evolution import evolve_topk_allocation

logger = logging.getLogger("MoECompressor")

# 复用 topk_skip 的逐层前向实现
from methods_skipping.topk_skip.model_qwen3_moe import TopKSkippedQwen3MoeSparseMoeBlock


def _get_moe_layer_indices(model) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
    ]


def _moe_forward_topk(
    block: Qwen3MoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    k_eff: int,
) -> torch.Tensor:
    """
    与 TopKSkippedQwen3MoeSparseMoeBlock 同构的前向，直接使用 block 上权重（profiling 用，避免每层 deepcopy）。
    """
    gate = block.gate
    experts = block.experts
    num_experts = gate.num_experts
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    router_logits = F.linear(hidden_states_reshaped, gate.weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

    router_top_value, router_indices = torch.topk(router_probs, k_eff, dim=-1)
    if gate.norm_topk_prob:
        router_top_value = (
            router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ).to(router_probs.dtype)
    routing_weights = router_top_value

    final_hidden_states = torch.zeros_like(hidden_states_reshaped)
    for expert_idx in range(num_experts):
        token_idx, top_k_pos = torch.where(router_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        current_state = hidden_states_reshaped[token_idx]
        g, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(g) * up
        current_hidden_states = F.linear(current_hidden_states, experts.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def _profile_layer_sensitivity(
    block: Qwen3MoeSparseMoeBlock,
    dtype: torch.dtype,
    k_base: int,
    mc_iters: int,
    profile_batch: int,
    profile_seq_len: int,
    hidden_dim: int,
) -> torch.Tensor:
    """
    Returns:
        Tensor [k_base]，索引 k-1 对应 top-k 相对 baseline (top-k_base) 的平均 Frobenius 偏差。
    """
    dev = next(block.parameters()).device
    sums = torch.zeros(k_base, device="cpu", dtype=torch.float64)
    block.eval()
    for _ in tqdm(range(mc_iters), desc="lexi MC", leave=False):
        x = torch.randn(profile_batch, profile_seq_len, hidden_dim, device=dev, dtype=dtype)
        with torch.no_grad():
            y_base = _moe_forward_topk(block, x, k_base)
            yb = y_base.detach().float().reshape(-1)
            for k in range(1, k_base + 1):
                if k == k_base:
                    delta = 0.0
                else:
                    y_k = _moe_forward_topk(block, x, k).detach().float().reshape(-1)
                    delta = (y_k - yb).norm(p=2).item()
                sums[k - 1] += delta
    return (sums / float(mc_iters)).to(torch.float32)


class LexiSkipQwen3Moe(MoECompressor):
    """
    LExI：adapter 存敏感度矩阵 D（Stage1）；`patch` 按全局专家预算做进化搜索（Stage2）并替换为逐层 top-k 前向。

    **典型实验流程（减算 25% / 40% / 50%）**

    1. ``calib`` 一次，得到 ``adapter_dir/adapter.safetensors``（含 ``lexi.sensitivity``）。
    2. ``eval`` 三次：同一 ``--adapter_dir``，仅改 ``patch_kwargs`` 中 ``compute_reduction`` 为
       ``0.25``、``0.4``、``0.5``（或改用 ``target_budget`` 直接指定 ``sum_j k_j``）。

    Stage2 在 CPU 上完成，耗时通常可忽略；无需为不同比例重复 calib。
    """

    ADAPTER_KEYS = {
        "sensitivity": "lexi.sensitivity",
        "k_base": "lexi.k_base",
        "layer_indices": "lexi.layer_indices",
    }

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
        Stage1：蒙特卡洛扰动 profiling，写入 adapter.safetensors（不使用校准文本）。

        calib_kwargs 常用字段：
        - mc_iters: int，默认 512
        - profile_batch: int，默认 1
        - profile_seq_len: int，默认 8
        """
        if self.adapter_dir is None:
            raise ValueError("lexi_skip 的 calib 需要提供 --adapter_dir")

        mc_iters = int(kwargs.get("mc_iters", 512))
        profile_batch = int(kwargs.get("profile_batch", 1))
        profile_seq_len = int(kwargs.get("profile_seq_len", 8))

        from transformers import AutoModelForCausalLM

        logger.info("[lexi_skip][calib] Loading model for Stage1 profiling")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device == "cpu":
            model = model.to(self.device)
        model.eval()

        moe_indices = _get_moe_layer_indices(model)
        if not moe_indices:
            raise RuntimeError("模型中未找到 Qwen3MoeSparseMoeBlock")

        first_block = model.model.layers[moe_indices[0]].mlp
        k_base = int(first_block.gate.top_k)
        hidden_dim = model.config.hidden_size

        for decoder_idx in moe_indices:
            kb = int(model.model.layers[decoder_idx].mlp.gate.top_k)
            if kb != k_base:
                raise ValueError(
                    f"[lexi_skip] 要求所有 MoE 层 gate.top_k 一致，层 {decoder_idx} 为 {kb}，期望 {k_base}"
                )

        rows: list[torch.Tensor] = []
        for decoder_idx in tqdm(moe_indices, desc="lexi_skip Stage1 (per layer)", unit="layer"):
            block = model.model.layers[decoder_idx].mlp
            row = _profile_layer_sensitivity(
                block,
                self.torch_dtype,
                k_base,
                mc_iters,
                profile_batch,
                profile_seq_len,
                hidden_dim,
            )
            rows.append(row)

        D = torch.stack(rows, dim=0)
        layer_indices = torch.tensor(moe_indices, dtype=torch.int64)

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {
            self.ADAPTER_KEYS["sensitivity"]: D.cpu(),
            self.ADAPTER_KEYS["k_base"]: torch.tensor(k_base, dtype=torch.int32),
            self.ADAPTER_KEYS["layer_indices"]: layer_indices,
        }
        save_file(state, str(self._get_adapter_path()))
        logger.info(
            "[lexi_skip][calib] Saved sensitivity D shape=%s to %s",
            tuple(D.shape),
            self._get_adapter_path(),
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def patch(self, model, **kwargs) -> Any:
        """
        Stage2 + 替换 MoE。

        patch_kwargs（二选一指定预算）：
        - compute_reduction: float，如 0.25 表示总激活专家数减 25%%，B = round(B0 * (1-r))
        - target_budget: int，直接指定 B = sum_j k_j

        可选：
        - layer_topk: list[int]，若提供则 **跳过进化**，直接使用（长度须等于 MoE 层数）
        - seed: int，进化随机种子
        - n_pop, n_gen: 进化规模
        """
        if self.adapter_dir is None:
            raise ValueError("lexi_skip 的 patch 需要提供 --adapter_dir")
        if self.adapter_path is None or not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")

        state = load_file(str(self.adapter_path))
        D = state[self.ADAPTER_KEYS["sensitivity"]].numpy()
        k_base_meta = int(state[self.ADAPTER_KEYS["k_base"]].item())
        saved_indices = state[self.ADAPTER_KEYS["layer_indices"]].long().tolist()

        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        if moe_indices != saved_indices:
            logger.warning(
                "[lexi_skip] 当前模型 MoE 层索引 %s 与 adapter %s 不一致，仍尝试继续",
                moe_indices,
                saved_indices,
            )

        L = D.shape[0]
        if len(moe_indices) != L:
            raise ValueError(
                f"敏感度矩阵层数 L={L} 与当前模型 MoE 层数 {len(moe_indices)} 不一致"
            )

        layer_topk_list = kwargs.get("layer_topk")
        if layer_topk_list is not None:
            k_list = [int(x) for x in layer_topk_list]
            if len(k_list) != L:
                raise ValueError(f"layer_topk 长度应为 {L}，收到 {len(k_list)}")
        else:
            B0 = L * k_base_meta
            if kwargs.get("target_budget") is not None:
                B = int(kwargs["target_budget"])
            elif kwargs.get("compute_reduction") is not None:
                r = float(kwargs["compute_reduction"])
                B = int(round(B0 * (1.0 - r)))
            else:
                raise ValueError(
                    'lexi_skip 的 patch 需要 patch_kwargs 中的 compute_reduction 或 target_budget，'
                    '或提供 layer_topk，例如 {"compute_reduction": 0.25}'
                )

            k_min = 1
            k_max = k_base_meta
            if L * k_min > B or L * k_max < B:
                raise ValueError(
                    f"预算 B={B} 在 k∈[{k_min},{k_max}] 下无可行解（L={L}）。"
                    f" 请调整 compute_reduction / target_budget。"
                )

            seed = kwargs.get("seed", 0)
            seed = None if seed is None else int(seed)
            n_pop = int(kwargs.get("n_pop", 64))
            n_gen = int(kwargs.get("n_gen", 200))

            k_arr = evolve_topk_allocation(
                D,
                B,
                k_min=k_min,
                k_max=k_max,
                n_pop=n_pop,
                n_gen=n_gen,
                seed=seed,
            )
            k_list = k_arr.tolist()

        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)
        stats_collector.initialize_layers(moe_indices)

        logger.info("[lexi_skip][patch] Per-layer top-k: %s", k_list)
        for j, decoder_layer_idx in enumerate(
            tqdm(moe_indices, desc="Patching layers (lexi_skip)", unit="layer")
        ):
            block = layers[decoder_layer_idx].mlp
            kj = int(k_list[j])
            layers[decoder_layer_idx].mlp = TopKSkippedQwen3MoeSparseMoeBlock(
                block,
                k=kj,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
