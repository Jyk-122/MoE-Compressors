"""
REAP (Why Pruning Prevails for One-Shot MoE compression) 方法 - 基于专家激活范数与 top-k weights 的剪枝（Qwen3-MoE 实现）

【原理】
在校准集上统计每个专家所输出的激活向量的 L2 范数（Norm）与 top-k weights 的乘积的均值。
值越大表示专家越重要，越小越优先被裁剪。

【适配 Qwen3-MoE】
- MoE 层为 Qwen3MoeSparseMoeBlock，内含 gate (Router) 和 experts
- 在校准 forward 时，对每个专家的输出（乘 routing weight 之前）计算 L2 范数与 top-k weights 的乘积，按专家累加求均值
- patch 时：与 frequency_pruning 相同，替换为 PrunedQwen3MoeSparseMoeBlock
"""

from __future__ import annotations

import gc
import logging
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from MoECompressor import MoECompressor


def _get_moe_layers(model) -> list[tuple[int, Qwen3MoeSparseMoeBlock]]:
    """
    遍历模型，找出所有 MoE 层（Qwen3MoeSparseMoeBlock）。

    Returns:
        [(decoder_layer_idx, mlp_block), ...]
    """
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            moe_layers.append((i, layer.mlp))
    return moe_layers


def _experts_forward_with_norm_collection(
    experts_module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    norm_stats: dict,
    layer_idx: int,
) -> torch.Tensor:
    """
    执行 experts forward，同时收集每个专家输出的 L2 范数统计。

    与 Qwen3MoeExperts.forward 逻辑一致，但在乘 routing weight 之前计算每个 token 输出的 L2 范数，
    按专家累加 sum_norm 和 count。
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
        # 专家输出（乘 routing weight 之前）
        expert_output = F.linear(current_hidden_states, experts_module.down_proj[expert_idx])
        # 计算每个 token 输出的 L2 范数
        norms = torch.norm(expert_output.float(), p=2, dim=-1)
        # 计算每个 token 输出的 L2 范数与 top-k weights 的乘积
        norms = norms * top_k_weights[token_idx, top_k_pos, None]
        # 累加统计
        if layer_idx not in norm_stats:
            norm_stats[layer_idx] = {}
        if expert_idx not in norm_stats[layer_idx]:
            norm_stats[layer_idx][expert_idx] = [0.0, 0]  # [sum_norm, count]
        norm_stats[layer_idx][expert_idx][0] += norms.sum().item()
        norm_stats[layer_idx][expert_idx][1] += norms.numel()

        current_hidden_states = expert_output * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


class PrunedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    剪枝后的 SparseMoeBlock（与 frequency_pruning 共用同一实现）。

    1. Router：在原 logits 上将「被剪专家」对应位置置为 -inf
    2. Experts：只保留 keep_indices 对应的专家权重
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        keep_indices: torch.LongTensor,
        old_to_new: torch.LongTensor,
    ):
        super().__init__()
        self.gate = original_block.gate
        self.top_k = original_block.gate.top_k
        self.num_experts = original_block.gate.num_experts
        self.keep_indices = keep_indices
        self.old_to_new = old_to_new
        self.num_kept = len(keep_indices)
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj[keep_indices].clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj[keep_indices].clone())
        self.act_fn = experts.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_logits = router_logits.clone()
        pruned_mask = self.old_to_new == -1
        router_logits[:, pruned_mask] = float("-inf")
        router_logits = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (router_top_value / router_top_value.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
        routing_weights = router_top_value

        selected_experts_new = self.old_to_new[router_indices]

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts_new, num_classes=self.num_kept).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for idx in expert_hit:
            expert_idx = idx[0].item()
            if expert_idx >= self.num_kept:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class REAPPruningQwen3Moe(MoECompressor):
    """
    基于 REAP 的专家剪枝，适配 Qwen3-MoE。

    统计每个专家输出激活向量的 L2 范数与 top-k weights 的乘积的均值，剪掉均值最小的专家。
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: str | Path | None = None,
        prune_ratio: float = 0.5,
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
        self.prune_ratio = prune_ratio

    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上统计每个专家输出的 L2 范数均值，保存 adapter。
        """
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[calib] Step 0/4: Loading model and tokenizer")
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

        logger.info("[calib] Step 1/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )

        logger.info("[calib] Step 2/4: Forward pass to collect expert activation norms (REAP)")
        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts
        top_k = model.config.num_experts_per_tok

        # norm_stats[layer_idx][expert_idx] = [sum_norm, count]
        norm_stats: dict[int, dict[int, list[float]]] = {}

        # 用 MethodType 替换 experts.forward，默认参数在定义时求值，正确捕获当前层的 decoder_layer_idx
        for decoder_layer_idx, block in moe_layers:
            experts = block.experts

            def _forward(self, hidden_states, top_k_index, top_k_weights, _layer=decoder_layer_idx):
                return _experts_forward_with_norm_collection(
                    experts_module=self,
                    hidden_states=hidden_states,
                    top_k_index=top_k_index,
                    top_k_weights=top_k_weights,
                    norm_stats=norm_stats,
                    layer_idx=_layer,
                )

            experts.forward = types.MethodType(_forward, experts)

        # 运行校准 forward
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

        logger.info("[calib] Step 3/4: Determining kept experts by REAP")
        keep_per_layer = {}
        for decoder_layer_idx, _ in moe_layers:
            layer_stats = norm_stats.get(decoder_layer_idx, {})
            
            mean_norms = torch.full((num_experts,), 0.0, dtype=torch.float64)
            for expert_idx, (sum_norm, count) in layer_stats.items():
                mean_norms[expert_idx] = sum_norm / count

            num_keep = max(1, int(num_experts * (1 - self.prune_ratio)))
            _, top_indices = torch.topk(mean_norms, num_keep)
            keep_indices = top_indices.sort().values
            old_to_new = torch.full((num_experts,), -1, dtype=torch.long)
            for new_idx, old_idx in enumerate(keep_indices.tolist()):
                old_to_new[old_idx] = new_idx
            keep_per_layer[str(decoder_layer_idx)] = {
                "keep_indices": keep_indices.cpu(),
                "old_to_new": old_to_new.cpu(),
            }

        logger.info("[calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {f"layer_{k}.keep_indices": v["keep_indices"] for k, v in keep_per_layer.items()}
        state.update({f"layer_{k}.old_to_new": v["old_to_new"] for k, v in keep_per_layer.items()})
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        """
        打补丁：读取 adapter，将给定 model 的每层 MoE 替换为 PrunedQwen3MoeSparseMoeBlock。
        """
        if self.adapter_dir is None:
            raise ValueError("patch 需提供 adapter_dir")

        logger.info("[patch] Loading adapter")
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")
        state = load_file(str(self.adapter_path))

        layers = model.model.layers
        moe_indices = [
            i for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        logger.info("[patch] Replacing %d MoE layers", len(moe_indices))

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            key_pre = f"layer_{decoder_layer_idx}"
            keep_indices = state[f"{key_pre}.keep_indices"]
            old_to_new = state[f"{key_pre}.old_to_new"]
            pruned_block = PrunedQwen3MoeSparseMoeBlock(
                block,
                keep_indices.to(block.gate.weight.device),
                old_to_new.to(block.gate.weight.device),
            )
            layers[decoder_layer_idx].mlp = pruned_block

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
