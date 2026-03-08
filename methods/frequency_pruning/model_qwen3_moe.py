"""
Frequency Pruning 方法 - 基于激活频率的专家剪枝（Qwen3-MoE 实现）

【原理】
在校准集上统计每个 token 被路由到哪些专家，累加得到每层各专家的激活次数。
根据 prune_ratio 裁剪每层中激活次数最少的那部分专家。

【适配 Qwen3-MoE】
- MoE 层为 Qwen3MoeSparseMoeBlock，内含 gate (Router) 和 experts
- 通过 output_router_logits 获取每层 router_logits，对 logits 做 topk 得到选中的专家索引
- patch 时：替换为 PrunedQwen3MoeSparseMoeBlock，对 router 做 mask、对 experts 做切片
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
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


class PrunedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    剪枝后的 SparseMoeBlock。

    1. Router：在原 logits 上将「被剪专家」对应位置置为 -inf，再 softmax+topk，
       保证只有保留专家会被选中。
    2. Experts：只保留 keep_indices 对应的专家权重，forward 时用 old_to_new 映射
       将 router 输出的旧索引转为新索引。
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        keep_indices: torch.LongTensor,
        old_to_new: torch.LongTensor,
    ):
        """
        Args:
            original_block: 原始 SparseMoeBlock
            keep_indices: 保留的专家索引，shape (num_kept,)
            old_to_new: 旧索引 -> 新索引，shape (num_experts,)，被剪专家为 -1
        """
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

        # 1. 计算 router logits，将被剪专家的 logits 置为 -inf
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_logits = router_logits.clone()
        pruned_mask = self.old_to_new == -1
        router_logits[:, pruned_mask] = float("-inf")
        router_logits = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (router_top_value / router_top_value.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
        routing_weights = router_top_value

        # 2. 将 router 输出的旧索引映射到新索引（0..num_kept-1）
        selected_experts_new = self.old_to_new[router_indices]

        # 3. 用切片后的专家权重做 forward（与原 Qwen3MoeExperts 逻辑一致）
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


class FrequencyPruningQwen3Moe(MoECompressor):
    """
    基于激活频率的专家剪枝，适配 Qwen3-MoE。

    根据 prune_ratio 在每层剪掉激活次数最少的专家，保留激活频繁的专家。
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
        """
        Args:
            prune_ratio: 剪枝比例 (0~1)，如 0.5 表示剪掉 50% 专家
        """
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
        calibration_data: list[str] | None = None,
        calibration_dataset: str | None = None,
        max_calib_samples: int = 512,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上计算统计量，并保存adapter，需要指定adapter_dir。

        Args:
            calibration_data (list[str] | None): 可选，直接传入的校准文本数据。
            calibration_dataset (str | None): 可选，HuggingFace数据集名，格式如"wikitext:wikitext-2-raw-v1"。
            max_calib_samples (int): 最大校准样本数，默认512。
            batch_size (int): 批量大小，默认1。
            **kwargs: 其它方法相关的参数。
        
        Returns:
            None
        """ 
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")
        texts = self.load_calibration_data(calibration_data, calibration_dataset, max_calib_samples)
        self.model.eval()
        if hasattr(self.model.config, "output_router_logits"):
            self.model.config.output_router_logits = True
        moe_layers = _get_moe_layers(self.model)
        num_experts = self.model.config.num_experts
        top_k = self.model.config.num_experts_per_tok
        expert_counts: dict[int, torch.Tensor] = {i: torch.zeros(num_experts) for i, _ in moe_layers}

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_router_logits=True)
            if outputs.router_logits is None:
                raise RuntimeError(
                    "模型未返回 router_logits，请确保 config.output_router_logits=True"
                )

            # 从 router_logits 推断每 token 选中的 topk 专家，累加计数
            for layer_idx, (decoder_layer_idx, _) in enumerate(moe_layers):
                logits = outputs.router_logits[layer_idx]
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.shape[-1])
                probs = F.softmax(logits.float(), dim=-1)
                _, selected = torch.topk(probs, top_k, dim=-1)
                for tok in range(selected.shape[0]):
                    for k in range(selected.shape[1]):
                        expert_counts[decoder_layer_idx][selected[tok, k].item()] += 1

        keep_per_layer = {}
        for decoder_layer_idx, counts in expert_counts.items():
            num_keep = max(1, int(num_experts * (1 - self.prune_ratio)))
            _, top_indices = torch.topk(counts, num_keep)
            keep_indices = top_indices.sort().values
            old_to_new = torch.full((num_experts,), -1, dtype=torch.long)
            for new_idx, old_idx in enumerate(keep_indices.tolist()):
                old_to_new[old_idx] = new_idx
            keep_per_layer[str(decoder_layer_idx)] = {
                "keep_indices": keep_indices.cpu(),
                "old_to_new": old_to_new.cpu(),
            }
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {f"layer_{k}.keep_indices": v["keep_indices"] for k, v in keep_per_layer.items()}
        state.update({f"layer_{k}.old_to_new": v["old_to_new"] for k, v in keep_per_layer.items()})
        save_file(state, str(self._get_adapter_path()))

    def patch(self, **kwargs) -> Any:
        """
        打补丁：读取 adapter，将每层 MoE 的 SparseMoeBlock 替换为 PrunedQwen3MoeSparseMoeBlock。
        由 run.py 在 eval 时根据 adapter_dir 是否传入自动调用。
        """
        if self.adapter_dir is None:
            raise ValueError("patch 需提供 adapter_dir")
        state = self.adapter
        if state is None:
            if not self.adapter_path.exists():
                raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")
            state = load_file(str(self.adapter_path))
        moe_layers = _get_moe_layers(self.model)
        for decoder_layer_idx, block in moe_layers:
            key_pre = f"layer_{decoder_layer_idx}"
            keep_indices = state[f"{key_pre}.keep_indices"]
            old_to_new = state[f"{key_pre}.old_to_new"]
            pruned_block = PrunedQwen3MoeSparseMoeBlock(
                block,
                keep_indices.to(block.gate.weight.device),
                old_to_new.to(block.gate.weight.device),
            )
            self.model.model.layers[decoder_layer_idx].mlp = pruned_block
        return self.model
