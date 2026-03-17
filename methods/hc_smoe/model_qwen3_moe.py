"""
HC-SMoE (Hierarchical Clustering-based Sparse MoE) 方法 - 基于层次聚类的专家合并（Qwen3-MoE 实现）

【原理】
1. 在校准集上统计专家使用频率和相似度
2. 基于使用频率和相似度对专家进行层次聚类
3. 将聚类结果保存为 adapter，在推理时在线合并专家

【适配 Qwen3-MoE】
- MoE 层为 Qwen3MoeSparseMoeBlock，内含 gate (Router) 和 experts
- 在校准阶段计算专家使用频率和相似度，进行聚类
- 补丁阶段：替换为 HCSMoEQwen3MoeSparseMoeBlock，在线合并专家
"""

from __future__ import annotations

import copy
import gc
import logging
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

from MoECompressor import MoECompressor

# 常量定义
FP32_EPS = 1e-7


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


class HCSMoEQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    HC-SMoE 适配的 SparseMoeBlock。

    1. Router：使用原始路由逻辑
    2. Experts：根据合并后的专家索引在线合并专家权重
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        merge_groups: torch.LongTensor,
        dominant_experts: torch.LongTensor,
    ):
        super().__init__()
        # 保存原始组件
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.merge_groups = merge_groups
        self.dominant_experts = dominant_experts
        self.num_merged_experts = len(dominant_experts)
        
        # 保存原始专家权重
        self.original_experts = original_block.experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 路由逻辑
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_logits = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (router_top_value / router_top_value.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
        routing_weights = router_top_value

        # 在线合并专家
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        
        # 处理每个被激活的专家
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for idx in expert_hit:
            expert_idx = idx[0].item()
            if expert_idx >= self.num_experts:
                continue
            
            # 找到该专家所属的组和主导专家
            group_idx = self.merge_groups[expert_idx].item()
            dominant_idx = self.dominant_experts[group_idx].item()
            
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            
            # 使用主导专家的权重进行前向计算
            gate, up = F.linear(current_state, self.original_experts.gate_up_proj[dominant_idx]).chunk(2, dim=-1)
            current_hidden_states = self.original_experts.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.original_experts.down_proj[dominant_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class HCSMoEQwen3Moe(MoECompressor):
    """
    基于层次聚类的专家合并，适配 Qwen3-MoE。

    1. 校准阶段：计算专家使用频率和相似度，进行层次聚类
    2. 补丁阶段：在线合并专家权重，减少模型大小
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_dir: str | Path | None = None,
        num_merged_experts: int = 4,
        similarity_base: str = "router-logits",
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
        self.num_merged_experts = num_merged_experts
        self.similarity_base = similarity_base

    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上统计专家使用频率和相似度，进行层次聚类，保存 adapter。
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

        logger.info("[calib] Step 2/4: Forward pass to collect expert usage and required data")
        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts
        top_k = model.config.num_experts_per_tok

        # 收集每层的使用频率和所需数据
        usage_frequency = {i: torch.zeros(num_experts, device=model.device) for i, _ in moe_layers}
        router_logits_collector = {i: [] for i, _ in moe_layers}
        expert_outputs_collector = {i: [] for i, _ in moe_layers}
        expert_weights_collector = {i: None for i, _ in moe_layers}

        # 准备收集专家输出的钩子
        hooks = []
        expert_outputs_per_expert = {layer_idx: {i: [] for i in range(num_experts)} for layer_idx, _ in moe_layers}
        
        if self.similarity_base in ["expert-output", "weight+expert-output", "router-logits+expert-output", "router-logits+weight+expert-output"]:
            for layer_idx, block in moe_layers:
                # 保存原始 forward 方法
                block.experts.original_forward = block.experts.forward
                
                # 替换 experts.forward 方法以收集每个专家的输出
                def new_forward(self, hidden_states, top_k_index, top_k_weights, layer=layer_idx):
                    num_experts = self.num_experts
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
                        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                        current_hidden_states = self.act_fn(gate) * up
                        expert_output = F.linear(current_hidden_states, self.down_proj[expert_idx])
                        
                        # 收集专家输出
                        expert_outputs_per_expert[layer][expert_idx].append(expert_output.detach())
                        
                        current_hidden_states = expert_output * top_k_weights[token_idx, top_k_pos, None]
                        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
                    
                    return final_hidden_states
                
                block.experts.forward = types.MethodType(new_forward, block.experts)

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
                outputs = model(**inputs, output_router_logits=True)
            
            # 收集路由 logits 和使用频率
            router_logits = outputs.router_logits
            for i, (layer_idx, _) in enumerate(moe_layers):
                layer_router_logits = router_logits[i]
                router_logits_collector[layer_idx].append(layer_router_logits)
                
                # 计算使用频率
                selected_experts = torch.topk(layer_router_logits, top_k, dim=-1)[1]
                unique, counts = torch.unique(selected_experts, return_counts=True)
                usage_frequency[layer_idx][unique] += counts

        # 恢复原始 forward 方法
        for layer_idx, block in moe_layers:
            if hasattr(block.experts, 'original_forward'):
                block.experts.forward = block.experts.original_forward

        # 收集专家权重
        if self.similarity_base in ["weight", "weight+expert-output", "router-logits+weight", "router-logits+weight+expert-output"]:
            for layer_idx, block in moe_layers:
                experts = block.experts
                weights = []
                for i in range(num_experts):
                    weight_flat = torch.cat(
                        [experts.gate_up_proj[i].flatten(),
                         experts.down_proj[i].flatten()],
                        dim=0
                    )
                    weights.append(weight_flat)
                expert_weights_collector[layer_idx] = torch.stack(weights)

        logger.info("[calib] Step 3/4: Clustering experts")
        merge_info = {}
        
        for layer_idx, _ in moe_layers:
            # 计算使用频率
            freq = usage_frequency[layer_idx]
            freq = freq / freq.sum() if freq.sum() > 0 else freq
            
            # 根据 similarity_base 计算专家相似度
            if self.similarity_base == "router-logits":
                # 基于路由 logits 的相似度
                layer_router_logits = torch.cat(router_logits_collector[layer_idx], dim=0)
                expert_router_logits = layer_router_logits.T  # (num_experts, num_tokens)
                expert_router_logits = expert_router_logits / (expert_router_logits.norm(dim=-1, keepdim=True) + FP32_EPS)
                similarity_matrix = torch.matmul(expert_router_logits, expert_router_logits.T)
            elif self.similarity_base == "weight":
                # 基于专家权重的相似度
                weights = expert_weights_collector[layer_idx]
                weights = weights / (weights.norm(dim=-1, keepdim=True) + FP32_EPS)
                similarity_matrix = torch.matmul(weights, weights.T)
            elif self.similarity_base == "expert-output":
                # 基于专家输出的相似度
                expert_outputs = []
                for i in range(num_experts):
                    if expert_outputs_per_expert[layer_idx][i]:
                        # 合并该专家的所有输出并计算平均值
                        expert_output = torch.cat(expert_outputs_per_expert[layer_idx][i], dim=0).mean(dim=0)
                    else:
                        # 如果该专家没有被激活，使用零向量
                        expert_output = torch.zeros(model.config.hidden_size, device=model.device)
                    expert_outputs.append(expert_output)
                expert_outputs = torch.stack(expert_outputs)
                expert_outputs = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + FP32_EPS)
                similarity_matrix = torch.matmul(expert_outputs, expert_outputs.T)
            elif self.similarity_base == "weight+expert-output":
                # 基于权重和输出的联合相似度
                # 计算权重相似度
                weights = expert_weights_collector[layer_idx]
                weights = weights / (weights.norm(dim=-1, keepdim=True) + FP32_EPS)
                weight_similarity = torch.matmul(weights, weights.T)
                
                # 计算输出相似度
                expert_outputs = []
                for i in range(num_experts):
                    if expert_outputs_per_expert[layer_idx][i]:
                        expert_output = torch.cat(expert_outputs_per_expert[layer_idx][i], dim=0).mean(dim=0)
                    else:
                        expert_output = torch.zeros(model.config.hidden_size, device=model.device)
                    expert_outputs.append(expert_output)
                expert_outputs = torch.stack(expert_outputs)
                expert_outputs = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + FP32_EPS)
                output_similarity = torch.matmul(expert_outputs, expert_outputs.T)
                
                # 合并相似度
                similarity_matrix = (weight_similarity + output_similarity) / 2
            elif self.similarity_base == "router-logits+weight":
                # 基于路由 logits 和权重的联合相似度
                # 计算路由 logits 相似度
                layer_router_logits = torch.cat(router_logits_collector[layer_idx], dim=0)
                expert_router_logits = layer_router_logits.T
                expert_router_logits = expert_router_logits / (expert_router_logits.norm(dim=-1, keepdim=True) + FP32_EPS)
                router_similarity = torch.matmul(expert_router_logits, expert_router_logits.T)
                
                # 计算权重相似度
                weights = expert_weights_collector[layer_idx]
                weights = weights / (weights.norm(dim=-1, keepdim=True) + FP32_EPS)
                weight_similarity = torch.matmul(weights, weights.T)
                
                # 合并相似度
                similarity_matrix = (router_similarity + weight_similarity) / 2
            elif self.similarity_base == "router-logits+expert-output":
                # 基于路由 logits 和输出的联合相似度
                # 计算路由 logits 相似度
                layer_router_logits = torch.cat(router_logits_collector[layer_idx], dim=0)
                expert_router_logits = layer_router_logits.T
                expert_router_logits = expert_router_logits / (expert_router_logits.norm(dim=-1, keepdim=True) + FP32_EPS)
                router_similarity = torch.matmul(expert_router_logits, expert_router_logits.T)
                
                # 计算输出相似度
                expert_outputs = []
                for i in range(num_experts):
                    if expert_outputs_per_expert[layer_idx][i]:
                        expert_output = torch.cat(expert_outputs_per_expert[layer_idx][i], dim=0).mean(dim=0)
                    else:
                        expert_output = torch.zeros(model.config.hidden_size, device=model.device)
                    expert_outputs.append(expert_output)
                expert_outputs = torch.stack(expert_outputs)
                expert_outputs = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + FP32_EPS)
                output_similarity = torch.matmul(expert_outputs, expert_outputs.T)
                
                # 合并相似度
                similarity_matrix = (router_similarity + output_similarity) / 2
            elif self.similarity_base == "router-logits+weight+expert-output":
                # 基于路由 logits、权重和输出的联合相似度
                # 计算路由 logits 相似度
                layer_router_logits = torch.cat(router_logits_collector[layer_idx], dim=0)
                expert_router_logits = layer_router_logits.T
                expert_router_logits = expert_router_logits / (expert_router_logits.norm(dim=-1, keepdim=True) + FP32_EPS)
                router_similarity = torch.matmul(expert_router_logits, expert_router_logits.T)
                
                # 计算权重相似度
                weights = expert_weights_collector[layer_idx]
                weights = weights / (weights.norm(dim=-1, keepdim=True) + FP32_EPS)
                weight_similarity = torch.matmul(weights, weights.T)
                
                # 计算输出相似度
                expert_outputs = []
                for i in range(num_experts):
                    if expert_outputs_per_expert[layer_idx][i]:
                        expert_output = torch.cat(expert_outputs_per_expert[layer_idx][i], dim=0).mean(dim=0)
                    else:
                        expert_output = torch.zeros(model.config.hidden_size, device=model.device)
                    expert_outputs.append(expert_output)
                expert_outputs = torch.stack(expert_outputs)
                expert_outputs = expert_outputs / (expert_outputs.norm(dim=-1, keepdim=True) + FP32_EPS)
                output_similarity = torch.matmul(expert_outputs, expert_outputs.T)
                
                # 合并相似度
                similarity_matrix = (router_similarity + weight_similarity + output_similarity) / 3
            else:
                # 默认使用 router-logits
                layer_router_logits = torch.cat(router_logits_collector[layer_idx], dim=0)
                expert_router_logits = layer_router_logits.T  # (num_experts, num_tokens)
                expert_router_logits = expert_router_logits / (expert_router_logits.norm(dim=-1, keepdim=True) + FP32_EPS)
                similarity_matrix = torch.matmul(expert_router_logits, expert_router_logits.T)
            
            # 基于使用频率和相似度进行聚类
            # 1. 选择使用频率最高的专家作为初始聚类中心
            sorted_freq, sorted_indices = torch.sort(freq, descending=True)
            dominant_experts = sorted_indices[:self.num_merged_experts]
            
            # 2. 将剩余专家分配到最相似的聚类中心
            merge_groups = torch.zeros(num_experts, dtype=torch.long, device=model.device)
            for expert_idx in range(num_experts):
                if expert_idx in dominant_experts:
                    merge_groups[expert_idx] = torch.where(dominant_experts == expert_idx)[0].item()
                else:
                    # 找到最相似的主导专家
                    similarities = similarity_matrix[expert_idx, dominant_experts]
                    closest_idx = torch.argmax(similarities).item()
                    merge_groups[expert_idx] = closest_idx
            
            merge_info[str(layer_idx)] = {
                "merge_groups": merge_groups.cpu(),
                "dominant_experts": dominant_experts.cpu(),
            }

        logger.info("[calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {}
        for layer_idx, info in merge_info.items():
            state[f"layer_{layer_idx}.merge_groups"] = info["merge_groups"]
            state[f"layer_{layer_idx}.dominant_experts"] = info["dominant_experts"]
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        """
        打补丁：读取 adapter，将给定 model 的每层 MoE 替换为 HCSMoEQwen3MoeSparseMoeBlock。
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
            merge_groups = state[f"{key_pre}.merge_groups"]
            dominant_experts = state[f"{key_pre}.dominant_experts"]
            hcsmoe_block = HCSMoEQwen3MoeSparseMoeBlock(
                block,
                merge_groups.to(block.gate.weight.device),
                dominant_experts.to(block.gate.weight.device),
            )
            layers[decoder_layer_idx].mlp = hcsmoe_block

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
