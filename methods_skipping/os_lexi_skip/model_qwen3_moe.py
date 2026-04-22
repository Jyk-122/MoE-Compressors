"""
Optimal Scaling + LExI (per-layer topK) Skipping：在 lexi_skip 的路由框架下，
通过闭式解（线性回归）在校准集上计算 S 矩阵 (num_experts x d)。
使得剪枝后保留专家的特征组合，在欧式距离上无限逼近原始 TopK 的特征组合。

calib 时需要传入 `layer_topk` 列表来确定每层保留多少个专家，用于计算校准矩阵。

说明：
- calib：获取路由概率，计算基于 layer_topk 的掩码，分别累加协方差统计量，最终通过 batched 矩阵求逆得到 S 矩阵并写入 adapter。
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


def _resolve_layer_topk(kwargs: dict[str, Any], num_layers: int | None = None) -> list[int]:
    layer_topk = kwargs.get("layer_topk")
    if layer_topk is None:
        raise ValueError('os_lexi_skip 需要 calib_kwargs 或 patch_kwargs 中的 layer_topk，例如 {"layer_topk": [3,4,3,4]}')
    layer_topk = [int(x) for x in layer_topk]
    if num_layers is not None and len(layer_topk) != num_layers:
        raise ValueError(f'layer_topk 长度必须为 {num_layers}，收到 {len(layer_topk)}')
    return layer_topk


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


class OptimalScaledLexiQwen3MoeSparseMoeBlock(torch.nn.Module):
    """应用闭式解校准矩阵 S 补偿跳过专家的 MoE Block，支持 per-layer topK"""

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        expert_S_matrix: torch.Tensor,
        k_eff: int,
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
        self.norm = bool(norm)
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector
        experts = original_block.experts
        
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)
        
        if expert_S_matrix.shape != (self.num_experts, experts.down_proj.shape[1]):
            raise ValueError("expert_S_matrix 形状不匹配")
            
        self.register_buffer("S", expert_S_matrix.type_as(experts.gate_up_proj), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)

        router_top_value, router_indices = torch.topk(router_probs, self.k_eff, dim=-1)
        if self.gate.norm_topk_prob:
            router_top_value = (
                router_top_value / router_top_value.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            ).to(router_probs.dtype)
        routing_weights = router_top_value

        selected_indices = torch.full(
            (router_indices.shape[0], self.top_k),
            -1,
            dtype=torch.long,
            device=router_indices.device,
        )
        selected_indices[:, :self.k_eff] = router_indices
        
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
            
            scale_factor = routing_weights[token_idx, top_k_pos, None] * self.S[expert_idx]
            current_hidden_states = current_hidden_states * scale_factor
            
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class OptimalScalingLexiSkipQwen3Moe(MoECompressor):
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

        logger.info("Loading model and tokenizer for S-matrix closed-form calibration (os_lexi_skip)")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, torch_dtype=self.torch_dtype, device_map=self.device, trust_remote_code=self.trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=self.trust_remote_code)
        texts = self.load_calibration_data(tokenizer, calibration_dataset, max_calib_samples, max_context_len)

        model.eval()
        moe_layers = _get_moe_layers(model)
        num_experts = model.config.num_experts
        num_moe_layers = len(moe_layers)
        
        layer_topk = _resolve_layer_topk(kwargs, num_moe_layers)
        logger.info(f"Using per-layer topK: {layer_topk}")

        A_stats: dict[int, torch.Tensor] = {}
        B_stats: dict[int, torch.Tensor] = {}

        for (decoder_layer_idx, block), k_eff in zip(moe_layers, layer_topk):
            def _forward(self_block, hidden_states: torch.Tensor, _layer=decoder_layer_idx, _k=k_eff):
                bsz, seq_len, hidden_dim = hidden_states.shape
                hidden_reshaped = hidden_states.view(-1, hidden_dim)
                num_tokens = hidden_reshaped.shape[0]

                router_logits = F.linear(hidden_reshaped, self_block.gate.weight)
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                router_top_value, router_indices = torch.topk(router_probs, self_block.gate.top_k, dim=-1)

                orig_routing_weights = router_top_value.clone()
                if self_block.gate.norm_topk_prob:
                    orig_routing_weights = orig_routing_weights / orig_routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                router_top_value_kept, router_indices_kept = torch.topk(router_probs, _k, dim=-1)
                if self_block.gate.norm_topk_prob:
                    kept_routing_weights = (
                        router_top_value_kept / router_top_value_kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    ).to(router_probs.dtype)
                else:
                    kept_routing_weights = router_top_value_kept

                V = torch.zeros((num_tokens, num_experts, hidden_dim), dtype=torch.float32, device=hidden_states.device)
                Y = torch.zeros((num_tokens, hidden_dim), dtype=torch.float32, device=hidden_states.device)
                
                final_output = torch.zeros_like(hidden_reshaped)

                for expert_idx in range(num_experts):
                    token_idx_orig, top_k_pos_orig = torch.where(router_indices == expert_idx)
                    if token_idx_orig.numel() > 0:
                        current_state = hidden_reshaped[token_idx_orig]
                        gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                        expert_output = F.linear(self_block.experts.act_fn(gate) * up, self_block.experts.down_proj[expert_idx])
                        alpha_orig = orig_routing_weights[token_idx_orig, top_k_pos_orig, None]
                        Y.index_add_(0, token_idx_orig, expert_output * alpha_orig)
                        final_output.index_add_(0, token_idx_orig, (expert_output * alpha_orig).to(final_output.dtype))
                    
                    token_idx_kept, top_k_pos_kept = torch.where(router_indices_kept == expert_idx)
                    if token_idx_kept.numel() > 0:
                        current_state = hidden_reshaped[token_idx_kept]
                        gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                        expert_output = F.linear(self_block.experts.act_fn(gate) * up, self_block.experts.down_proj[expert_idx])
                        alpha_kept = kept_routing_weights[token_idx_kept, top_k_pos_kept, None]
                        V[token_idx_kept, expert_idx, :] = expert_output * alpha_kept

                if _layer not in A_stats:
                    A_stats[_layer] = torch.zeros((hidden_dim, num_experts, num_experts), dtype=torch.float64, device='cpu')
                    B_stats[_layer] = torch.zeros((hidden_dim, num_experts), dtype=torch.float64, device='cpu')

                chunk_size = 1024
                for start_d in range(0, hidden_dim, chunk_size):
                    end_d = min(hidden_dim, start_d + chunk_size)
                    V_chunk = V[:, :, start_d:end_d]
                    Y_chunk = Y[:, start_d:end_d]
                    
                    A_stats[_layer][start_d:end_d] += torch.einsum('ted,tfd->def', V_chunk, V_chunk).cpu()
                    B_stats[_layer][start_d:end_d] += torch.einsum('ted,td->de', V_chunk, Y_chunk).cpu()

                return final_output.reshape(bsz, seq_len, hidden_dim)

            block.forward = types.MethodType(_forward, block)

        n_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Calibration Forward (os_lexi_skip)"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_context_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)

        logger.info("Solving closed-form Normal Equations for S matrices (os_lexi_skip)...")
        state: dict[str, torch.Tensor] = {"meta.adapter_version": torch.tensor(1, dtype=torch.int32)}
        state["meta.layer_topk"] = torch.tensor(layer_topk, dtype=torch.int32)
        
        lambda_reg = 1e-4
        
        for layer_idx in A_stats.keys():
            A = A_stats[layer_idx]
            B = B_stats[layer_idx].unsqueeze(-1)
            
            eye = torch.eye(num_experts, dtype=torch.float64).unsqueeze(0)
            A_reg = A + lambda_reg * eye
            
            S_raw = torch.linalg.solve(A_reg, B).squeeze(-1)
            S_matrix = S_raw.transpose(0, 1)
            
            active_expert_mask = A.sum(dim=0).diagonal() > 1e-8
            S_matrix[~active_expert_mask, :] = 1.0
            
            state[f"layer_{layer_idx}.expert_S_matrix"] = S_matrix.float().contiguous()

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(self._get_adapter_path()))
        logger.info(f"Calibration completed. Adapters saved to {self._get_adapter_path()}")

    def patch(self, model, **kwargs) -> Any:
        norm = _resolve_norm(kwargs)
        if self.adapter_dir is None or not self.adapter_path.exists():
            raise FileNotFoundError("os_lexi_skip patch 需提供有效 adapter_dir 且先运行 calib")

        state = load_file(str(self.adapter_path))
        
        if "layer_topk" in kwargs:
            layer_topk = _resolve_layer_topk(kwargs)
        else:
            layer_topk = state["meta.layer_topk"].tolist()
            logger.info(f"Using saved layer_topk from calib: {layer_topk}")
        
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)

        layers = model.model.layers
        moe_indices = [
            i for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        stats_collector.initialize_layers(moe_indices)
        
        if len(layer_topk) != len(moe_indices):
            raise ValueError(f"layer_topk 长度 {len(layer_topk)} 与 MoE 层数 {len(moe_indices)} 不匹配")
        
        logger.info("Patching %d MoE layers with closed-form S Matrix, layer_topk=%s", len(moe_indices), layer_topk)

        for j, decoder_layer_idx in enumerate(tqdm(moe_indices, desc="Patching layers (os_lexi_skip)", unit="layer")):
            block = layers[decoder_layer_idx].mlp
            key = f"layer_{decoder_layer_idx}.expert_S_matrix"
            if key not in state:
                raise KeyError(f"adapter 中缺少 {key}，请重新运行 calib")
            
            expert_S_matrix = state[key]
            k_eff = layer_topk[j]
            layers[decoder_layer_idx].mlp = OptimalScaledLexiQwen3MoeSparseMoeBlock(
                block,
                expert_S_matrix=expert_S_matrix,
                k_eff=k_eff,
                norm=norm,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )

        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
