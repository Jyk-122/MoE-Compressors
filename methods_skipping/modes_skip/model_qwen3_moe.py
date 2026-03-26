"""
MoDES skipping for Qwen3-MoE (text-only): GMLG — scale top-k router weights by offline
layer importance alpha, then drop experts with alpha·π < tau (re-normalize if gate.norm_topk_prob).

Calib: per MoE layer, ablate by zeroing mlp output vs full forward; KL or MSE on all
non-padding positions; normalize alpha across MoE layers to sum to 1 (same structure as
official per-modality normalization).
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


def _get_moe_layer_indices(model) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
    ]


def _resolve_tau(kwargs: dict[str, Any]) -> float:
    v = kwargs.get("tau")
    if v is None:
        raise ValueError('modes_skip 的 patch 需要 patch_kwargs 中的 tau，例如 {"tau": 0.05}')
    return float(v)


def _zero_mlp_forward(_self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(hidden_states)


def _per_position_loss(
    org_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    loss_type: str,
    temperature: float,
) -> torch.Tensor:
    """Sum loss over token positions (2D logits: [n_pos, vocab])."""
    t = float(temperature)
    if loss_type == "kl":
        log_p = F.log_softmax(ablated_logits / t, dim=-1)
        target = F.softmax(org_logits / t, dim=-1)
        return F.relu(F.kl_div(log_p, target, reduction="sum"))
    if loss_type == "mse":
        return F.mse_loss(ablated_logits, org_logits, reduction="sum")
    raise ValueError(f"不支持的 loss_type: {loss_type}，支持 kl / mse")


class MoDESSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """MoE block: router top-k, scale weights by layer alpha, threshold tau, optional renorm."""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        layer_alpha: torch.Tensor,
        tau: float,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.layer_alpha = layer_alpha.detach().float().view(())
        self.tau = float(tau)
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
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        alpha = self.layer_alpha.to(device=router_top_value.device, dtype=router_top_value.dtype)
        scaled = router_top_value * alpha
        active = scaled >= self.tau
        empty = ~active.any(dim=-1)
        if empty.any():
            pos = scaled[empty].argmax(dim=-1)
            rows = torch.nonzero(empty, as_tuple=False).squeeze(-1)
            active[rows, pos] = True

        routing_weights = router_top_value * active.to(router_top_value.dtype)
        if self.gate.norm_topk_prob:
            routing_weights = (
                routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)

        selected_indices = torch.where(
            active,
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


class MoDESSkipQwen3Moe(MoECompressor):
    """MoDES-style skipping: adapter 存每层 alpha；eval 用 patch_kwargs.tau。"""

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
            raise ValueError("modes_skip 的 calib 需要提供 --adapter_dir")

        if int(batch_size) != 1:
            logger.warning(
                "[modes_skip][calib] 忽略 batch_size=%s：固定为每条文本单独前向（batch_size=1）",
                batch_size,
            )

        loss_type = str(kwargs.get("loss_type", "kl")).lower()
        temperature = float(kwargs.get("temperature", 1.0))
        if loss_type not in ("kl", "mse"):
            raise ValueError(f"loss_type 须为 kl 或 mse，收到: {loss_type}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[modes_skip][calib] Loading model and tokenizer")
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

        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )
        if len(texts) == 0:
            raise RuntimeError("校准数据为空")

        moe_indices = _get_moe_layer_indices(model)
        if not moe_indices:
            raise RuntimeError("未找到 Qwen3 MoE 层")

        loss_sum: dict[int, float] = {idx: 0.0 for idx in moe_indices}
        n_samples_used = 0

        logger.info(
            "[modes_skip][calib] 逐样本 baseline forward + 每层 zero-moe；loss_type=%s T=%s；%d 个 MoE 层",
            loss_type,
            temperature,
            len(moe_indices),
        )

        model.eval()
        layers = model.model.layers

        with torch.no_grad():
            for text in tqdm(texts, desc="MoDES calib", unit="sample"):
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

                attn = inputs.get("attention_mask")
                if attn is None:
                    attn = torch.ones_like(inputs["input_ids"], dtype=torch.bool)
                else:
                    attn = attn.bool()

                org_logits = model(**inputs).logits.float()
                valid = attn[0].bool()
                if not valid.any():
                    continue

                org_slice = org_logits[0, valid]

                for layer_idx in moe_indices:
                    block = layers[layer_idx].mlp
                    orig_forward = block.forward
                    block.forward = types.MethodType(_zero_mlp_forward, block)
                    try:
                        ablated = model(**inputs).logits.float()
                    finally:
                        block.forward = orig_forward

                    abl_slice = ablated[0, valid]
                    loss = _per_position_loss(
                        org_slice, abl_slice, loss_type=loss_type, temperature=temperature
                    )
                    loss_sum[layer_idx] += float(loss.item())

                n_samples_used += 1

        if n_samples_used == 0:
            raise RuntimeError("校准未产生任何有效样本")

        raw = torch.tensor([loss_sum[i] for i in moe_indices], dtype=torch.float64)
        total = raw.sum()
        if total <= 0:
            raise RuntimeError("所有层的累计 loss 非正，无法归一化 alpha；请检查数据与 loss_type")

        alphas = (raw / total).float()
        alpha_by_layer = {moe_indices[j]: alphas[j].view(()) for j in range(len(moe_indices))}

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, torch.Tensor] = {
            "meta.adapter_version": torch.tensor(2, dtype=torch.int32),
        }
        for layer_idx, a in alpha_by_layer.items():
            state[f"layer_{layer_idx}.alpha"] = a.contiguous()

        save_file(state, str(self._get_adapter_path()))
        logger.info(
            "[modes_skip][calib] 完成：%d 样本，alpha 已写入 %s",
            n_samples_used,
            self._get_adapter_path(),
        )

    def patch(self, model, **kwargs) -> Any:
        tau = _resolve_tau(kwargs)

        if self.adapter_dir is None:
            raise ValueError("modes_skip 的 patch 需要提供 --adapter_dir（加载 alpha）")
        if self.adapter_path is None or not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")

        state = load_file(str(self.adapter_path))
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = _get_moe_layer_indices(model)
        stats_collector.initialize_layers(moe_indices)

        logger.info(
            "[modes_skip][patch] 替换 %d 层 MoE，tau=%.6f",
            len(moe_indices),
            tau,
        )

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers (modes_skip)", unit="layer"):
            key = f"layer_{decoder_layer_idx}.alpha"
            if key not in state:
                raise KeyError(f"adapter 缺少 {key}，请对当前模型重新 calib")
            block = layers[decoder_layer_idx].mlp
            layers[decoder_layer_idx].mlp = MoDESSkippedQwen3MoeSparseMoeBlock(
                original_block=block,
                layer_alpha=state[key],
                tau=tau,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
