"""
Optimal Scaling Skipping (optimal_scaling_skip)：在 topp_skip 的路由框架下，
通过闭式解（线性回归）在校准集上计算一个 S 矩阵 (num_experts x d)。
使得剪枝后保留专家的特征组合，在欧式距离上无限逼近原始 TopK 的特征组合。

说明：
- calib：获取路由概率，计算 TopP 掩码，分别累加协方差统计量，最终通过 batched 矩阵求逆得到 S 矩阵并写入 adapter。
- patch：加载 S 矩阵，前向计算时对保留专家的输出隐状态应用 alpha * S[i] * E[i] 的逐点缩放。
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


def _resolve_threshold(kwargs: dict[str, Any]) -> float:
    threshold = kwargs.get("threshold")
    if threshold is None:
        raise ValueError('optimal_scaling 需要 patch_kwargs 中的 threshold，例如 {"threshold": 0.8}')
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold 必须满足 0 < threshold <= 1")
    return threshold


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
    raise ValueError('norm 必须是布尔值')


class OptimalScaledQwen3MoeSparseMoeBlock(torch.nn.Module):
    """应用闭式解校准矩阵 S 补偿跳过专家的 MoE Block"""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        expert_S_matrix: torch.Tensor,
        threshold: float,
        norm: bool,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.threshold = float(threshold)
        self.norm = bool(norm)
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector
        experts = original_block.experts
        
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)
        
        # 注册 S 矩阵，形状为 (num_experts, hidden_dim)
        if expert_S_matrix.shape != (self.num_experts, experts.down_proj[0].out_features):
            raise ValueError("expert_S_matrix 形状不匹配")
            
        self.register_buffer("S", expert_S_matrix.to(dtype=experts.gate_up_proj[0].weight.dtype), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Top-p 判定
        router_top_value_for_topp = router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cumsum_probs = router_top_value_for_topp.cumsum(dim=-1)
        num_keep = (cumsum_probs < self.threshold).sum(dim=-1) + 1
        num_keep = num_keep.clamp(max=self.top_k)

        pos = torch.arange(self.top_k, device=router_indices.device).unsqueeze(0)
        active_mask = pos < num_keep.unsqueeze(1)

        routing_weights = router_top_value * active_mask.to(router_top_value.dtype)
        if self.gate.norm_topk_prob:
            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(router_probs.dtype)

        selected_indices = torch.where(
            active_mask,
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
            
            # 【核心修改】：在应用路由权重的同时，逐点乘上校准矩阵 S[i]
            scale_factor = routing_weights[token_idx, top_k_pos, None] * self.S[expert_idx]
            current_hidden_states = current_hidden_states * scale_factor
            
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class OptimalScalingSkippingQwen3Moe(MoECompressor):
    def __init__(self, model_name_or_path: str, adapter_dir: str | Path | None = None, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, adapter_dir=adapter_dir, **kwargs)

    def calib(
        self,
        calibration_dataset: str,
        threshold: float,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model and tokenizer for S-matrix closed-form calibration")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, torch_dtype=self.torch_dtype, device_map=self.device, trust_remote_code=self.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        texts = self.load_calibration_data(tokenizer, calibration_dataset, max_calib_samples, max_context_len)

        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts

        # 用于存储每个层的协方差矩阵 A 和投影矩阵 B
        A_stats: dict[int, torch.Tensor] = {}
        B_stats: dict[int, torch.Tensor] = {}

        for decoder_layer_idx, block in moe_layers:
            # 劫持整个 MoE block 以获取 router_probs 并构建 V 和 Y 统计量
            def _forward(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx):
                bsz, seq_len, hidden_dim = hidden_states.shape
                hidden_reshaped = hidden_states.view(-1, hidden_dim)
                num_tokens = hidden_reshaped.shape[0]

                router_logits = F.linear(hidden_reshaped, self_block.gate.weight)
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                router_top_value, router_indices = torch.topk(router_probs, self_block.gate.top_k, dim=-1)

                # 1. 计算原始的 Normalize 权重 (用于目标 Y)
                orig_routing_weights = router_top_value.clone()
                if self_block.gate.norm_topk_prob:
                    orig_routing_weights = orig_routing_weights / orig_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # 2. 模拟 TopP 逻辑 (用于构建输入特征 V)
                cumsum_probs = (router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)).cumsum(dim=-1)
                num_keep = (cumsum_probs < threshold).sum(dim=-1) + 1
                num_keep = num_keep.clamp(max=self_block.gate.top_k)
                
                pos = torch.arange(self_block.gate.top_k, device=router_indices.device).unsqueeze(0)
                active_mask = pos < num_keep.unsqueeze(1)
                
                kept_routing_weights = router_top_value * active_mask.to(router_top_value.dtype)
                if self_block.gate.norm_topk_prob:
                    kept_routing_weights = kept_routing_weights / kept_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # 构建 V 和 Y
                # 为防止大 SeqLen 导致 OOM，V 在 float32 下构造，累加到 float64 的 A/B
                V = torch.zeros((num_tokens, num_experts, hidden_dim), dtype=torch.float32, device=hidden_states.device)
                Y = torch.zeros((num_tokens, hidden_dim), dtype=torch.float32, device=hidden_states.device)
                
                final_output = torch.zeros_like(hidden_reshaped)

                for expert_idx in range(num_experts):
                    token_idx, top_k_pos = torch.where(router_indices == expert_idx)
                    if token_idx.numel() == 0:
                        continue
                    
                    current_state = hidden_reshaped[token_idx]
                    gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    expert_output = F.linear(self_block.experts.act_fn(gate) * up, self_block.experts.down_proj[expert_idx])

                    alpha_orig = orig_routing_weights[token_idx, top_k_pos, None]
                    alpha_kept = kept_routing_weights[token_idx, top_k_pos, None]

                    Y.index_add_(0, token_idx, expert_output * alpha_orig)
                    V[token_idx, expert_idx, :] = expert_output * alpha_kept
                    
                    # 保持原模型流转路径不变
                    final_output.index_add_(0, token_idx, (expert_output * alpha_orig).to(final_output.dtype))

                # Chunking 维度计算 A 和 B 以极致压缩显存占用
                if _layer not in A_stats:
                    A_stats[_layer] = torch.zeros((hidden_dim, num_experts, num_experts), dtype=torch.float64, device='cpu')
                    B_stats[_layer] = torch.zeros((hidden_dim, num_experts), dtype=torch.float64, device='cpu')

                chunk_size = 1024
                for start_d in range(0, hidden_dim, chunk_size):
                    end_d = min(hidden_dim, start_d + chunk_size)
                    V_chunk = V[:, :, start_d:end_d] # (T, E, D_chunk)
                    Y_chunk = Y[:, start_d:end_d]    # (T, D_chunk)
                    
                    # einsum 沿 T 轴缩并，解耦 D 轴
                    A_stats[_layer][start_d:end_d] += torch.einsum('ted,tfd->def', V_chunk, V_chunk).cpu()
                    B_stats[_layer][start_d:end_d] += torch.einsum('ted,td->de', V_chunk, Y_chunk).cpu()

                return final_output.reshape(bsz, seq_len, hidden_dim)

            block.forward = types.MethodType(_forward, block)

        # 触发前向传播进行统计
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Calibration Forward"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_context_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)

        logger.info("Solving closed-form Normal Equations for S matrices...")
        state: dict[str, torch.Tensor] = {"meta.adapter_version": torch.tensor(1, dtype=torch.int32)}
        
        lambda_reg = 1e-4 # Ridge Regression 的正则化项，保证协方差矩阵非奇异
        
        for layer_idx in A_stats.keys():
            A = A_stats[layer_idx] # (D, E, E)
            B = B_stats[layer_idx].unsqueeze(-1) # (D, E, 1)
            
            # 为 A 加入 L2 惩罚
            eye = torch.eye(num_experts, dtype=torch.float64).unsqueeze(0)
            A_reg = A + lambda_reg * eye
            
            # 使用 PyTorch 优化的 batch 线性求解器
            S_raw = torch.linalg.solve(A_reg, B).squeeze(-1) # 求解结果 (D, E)
            S_matrix = S_raw.transpose(0, 1) # 转置为所需形状 (E, D)
            
            # 如果某个专家在校准集中几乎从未激活，其 S 会退化为0，我们安全兜底至 1.0 (等效不缩放)
            active_expert_mask = A.sum(dim=0).diagonal() > 1e-8
            S_matrix[~active_expert_mask, :] = 1.0
            
            state[f"layer_{layer_idx}.expert_S_matrix"] = S_matrix.float()

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))
        logger.info(f"Calibration completed. Adapters saved to {self._get_adapter_path()}")

    def patch(self, model, **kwargs) -> Any:
        threshold = _resolve_threshold(kwargs)
        norm = _resolve_norm(kwargs)
        if self.adapter_dir is None or not self.adapter_path.exists():
            raise FileNotFoundError("optimal_scaling patch 需提供有效 adapter_dir 且先运行 calib")

        state = load_file(str(self.adapter_path))
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        stats_collector.initialize_layers(moe_indices)
        logger.info("Patching %d MoE layers with closed-form S Matrix, threshold=%.4f", len(moe_indices), threshold)

        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers", unit="layer"):
            block = layers[decoder_layer_idx].mlp
            key = f"layer_{decoder_layer_idx}.expert_S_matrix"
            if key not in state:
                raise KeyError(f"adapter 中缺少 {key}，请重新运行 calib")
            
            expert_S_matrix = state[key]
            layers[decoder_layer_idx].mlp = OptimalScaledQwen3MoeSparseMoeBlock(
                block,
                expert_S_matrix=expert_S_matrix,
                threshold=threshold,
                norm=norm,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model