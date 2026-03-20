"""
CAMERA 方法 - 基于专家激活范数的剪枝（Qwen3-MoE 实现）

【原理】
在校准集上统计每个专家所输出的激活向量的 L2 范数（Norm）的和。
值越大表示专家越重要，越小越优先被裁剪。

【适配 Qwen3-MoE】
- MoE 层为 Qwen3MoeSparseMoeBlock，内含 gate (Router) 和 experts
- 在校准 forward 时，对每个专家的输出（乘 routing weight 之前）计算 L2 范数，按专家累加求和
- patch 时：与 frequency_pruning 相同，替换为 PrunedQwen3MoeSparseMoeBlock
"""

from __future__ import annotations

import copy
import gc
import logging
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("MoECompressor")
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)

from MoECompressor import MoECompressor


class PrunedQwen3MoeSparseMoeBlock(nn.Module):
    """
    剪枝后的 SparseMoeBlock。

    使用 keep_mask (num_experts, inter_size) 的 0/1 向量记录每个 micro-expert 是否保留。
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        keep_mask: torch.Tensor,
    ):
        """
        Args:
            original_block: 原始 MoE 块
            keep_mask: (num_experts, inter_size) bool，True 表示保留该 micro-expert
        """
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.act_fn = copy.deepcopy(original_block.experts.act_fn)

        self.gate_up_proj = nn.ParameterList()
        self.down_proj = nn.ParameterList()

        for i in range(self.num_experts):
            mask_i = keep_mask[i]
            keep_idx = mask_i.nonzero(as_tuple=True)[0]
            old_gu = original_block.experts.gate_up_proj[i]
            old_d = original_block.experts.down_proj[i]
            inter_size = old_d.shape[1]

            if keep_idx.numel() > 0:
                # gate_up 在 dim 0 上 concat：前半 gate，后半 up
                gu_indices = torch.cat([keep_idx, keep_idx + inter_size])
                self.gate_up_proj.append(nn.Parameter(old_gu[gu_indices].clone()))
                self.down_proj.append(nn.Parameter(old_d[:, keep_idx].clone()))
            else:
                self.gate_up_proj.append(nn.Parameter(torch.empty(0)))
                self.down_proj.append(nn.Parameter(torch.empty(0)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 1. 路由计算 (无需再做 -inf mask，因为我们改变的是专家内部结构，而不是直接抛弃整只专家)
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.gate.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 2. 遍历选中的专家，进行前向传播
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for idx in expert_hit:
            expert_idx = idx[0].item()
            # 跳过被完全剪空的专家
            if self.gate_up_proj[expert_idx].numel() == 0:
                continue
                
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]

            # 从 ParameterList 中提取瘦身后的权重
            gu_weight = self.gate_up_proj[expert_idx]
            d_weight = self.down_proj[expert_idx]

            gate, up = F.linear(current_state, gu_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, d_weight)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Catcher(nn.Module):
    """用于捕获第一层输入的拦截器"""
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.cached_hidden_states = []
        self.cached_kwargs = []

    def forward(self, hidden_states, **kwargs):
        self.cached_hidden_states.append(hidden_states.cpu())
        # 把 attention_mask, position_ids 等 kwargs 也缓存下来
        self.cached_kwargs.append({
            k: v.cpu() if isinstance(v, torch.Tensor) else v 
            for k, v in kwargs.items()
        })
        raise ValueError("Caught inputs successfully")
    

class CAMERAPruningQwen3Moe(MoECompressor):
    """
    基于 CAMERA 的专家剪枝，适配 Qwen3-MoE。

    统计每个专家输出激活向量的 L2 范数和无穷范数，剪掉范数最小的神经元专家。
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
        alpha: float = 1.0,
        **kwargs,
    ) -> None:
        """
        校准：在校准数据上统计每个专家输出的 L2 范数均值，保存 adapter。
        """
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[calib] Step 1/4: Loading model and tokenizer")
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

        logger.info("[calib] Step 2/4: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )

        logger.info("[calib] Step 3/4: Starting layer-wise CAMERA redundancy analysis...")
        model.eval()
        layers = model.model.layers
        catcher = Catcher(layers[0])
        layers[0] = catcher
        
        # 将输入推入模型，直到触发拦截器
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_context_len)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            try:
                model(**inputs)
            except ValueError:
                pass
                
        layers[0] = catcher.module # 恢复第0层
        cached_hidden_states = catcher.cached_hidden_states
        cached_kwargs = catcher.cached_kwargs
        
        torch.cuda.empty_cache()

        all_keep_masks = {}

        for i in tqdm(range(len(layers)), desc="Layer-wise processing"):
            layer = layers[i].to(self.device)
            
            # 如果是 MoE 层，则计算微专家的冗余度
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                num_experts = layer.mlp.gate.num_experts
                inter_size = layer.mlp.experts.down_proj[0].shape[1]
                expert_l2_norm = {expert_idx: torch.zeros(inter_size, device=self.device) for expert_idx in range(num_experts)}
                expert_inf_norm = {expert_idx: torch.zeros(inter_size, device=self.device) for expert_idx in range(num_experts)}
                
                # 1. 前向推导收集冗余度统计（使用激活幅值作为重要性近似指标）
                for j in range(len(cached_hidden_states)):
                    hs = cached_hidden_states[j].to(self.device)
                    hs_reshaped = hs.view(-1, hs.shape[-1])
                    
                    with torch.no_grad():
                        router_logits = F.linear(hs_reshaped, layer.mlp.gate.weight)
                        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
                        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.gate.top_k, dim=-1)
                        if layer.mlp.gate.norm_topk_prob:
                            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
                        
                        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
                        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                        
                        for idx in expert_hit:
                            expert_idx = idx[0].item()
                            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                            current_state = hs_reshaped[token_idx]
                            
                            gate, up = F.linear(current_state, layer.mlp.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                            gate_up = layer.mlp.experts.act_fn(gate) * up
                            current_hidden_states = gate_up * routing_weights[token_idx, top_k_pos, None]
                            
                            # 【CAMERA 核心】论文 Eq.(11)：‖Φ_{:,j}‖_2^2（按 token 求和）、‖Φ_{:,j}‖_∞^2（按 token 取 max）
                            norm_l2 = current_hidden_states.pow(2).sum(dim=(0, 1))  # (inter_size,)
                            norm_inf = current_hidden_states.abs().pow(2).max(dim=0).values  # 每列 max，shape (inter_size,)
                            expert_l2_norm[expert_idx] += norm_l2
                            expert_inf_norm[expert_idx] = torch.maximum(expert_inf_norm[expert_idx], norm_inf)
                            
                # 2. 【CAMERA 核心】全局排序：本层所有 experts 的 micro-experts 放在一起比较，
                #    取 top (1-λ) 比例的微专家。每个 expert 的保留比例由其在全局排序中的占比决定，各不相同。
                total_micro_experts = num_experts * inter_size
                global_energy = torch.zeros(total_micro_experts, device=self.device)

                for expert_idx in range(num_experts):
                    expert_importance = (1 - alpha) * expert_l2_norm[expert_idx] + alpha * expert_inf_norm[expert_idx]
                    weight_l2_norm = layer.mlp.experts.down_proj[expert_idx].pow(2).sum(dim=0)
                    expert_importance = expert_importance * weight_l2_norm
                    global_base = expert_idx * inter_size
                    global_energy[global_base : global_base + inter_size] = expert_importance

                num_keep_total = max(1, int(total_micro_experts * (1 - self.prune_ratio)))
                _, top_global_indices = torch.topk(global_energy, num_keep_total)

                # 向量化：将 top 索引转为 0/1 mask，按 expert 分段，无 Python 循环
                global_mask = torch.zeros(total_micro_experts, dtype=torch.bool, device=self.device)
                global_mask[top_global_indices] = True
                keep_mask = global_mask.view(num_experts, inter_size)
                all_keep_masks[i] = keep_mask.cpu()

                # 3. 对该层进行物理瘦身 Patch
                layer.mlp = PrunedQwen3MoeSparseMoeBlock(layer.mlp, keep_mask).to(self.device)
            
            # 4. 更新缓存特征：用当前层（如果是 MoE 则已经是剪枝后的结构）推导出下一层的输入
            for j in range(len(cached_hidden_states)):
                hs = cached_hidden_states[j].to(self.device)
                kwargs_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in cached_kwargs[j].items()}
                
                with torch.no_grad():
                    out = layer(hs, **kwargs_device)
                cached_hidden_states[j] = out.cpu() # 存回 CPU
                
            # 处理完该层后释放显存
            layer.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info("[calib] Step 4/4: Saving adapter")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {
            f"layer_{layer_idx}.keep_mask": mask_val
            for layer_idx, mask_val in all_keep_masks.items()
        }
        save_file(state_dict, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        """
        打补丁：读取 adapter，将给定 model 的每层 MoE 替换为 PrunedQwen3MoeSparseMoeBlock。
        """
        accelerate_config = kwargs.get("accelerate_config", {}) or {}
        if self.adapter_dir is None and accelerate_config:
            logger.warning(
                "[camera][patch] 当前方法不支持激活计算加速，且未提供 adapter，保持原模型不变并忽略 accelerate_config=%s",
                accelerate_config,
            )
            return model
        if self.adapter_dir is None:
            raise ValueError("patch 需提供 adapter_dir")
        if accelerate_config:
            logger.warning(
                "[camera][patch] 当前方法不支持激活计算加速，已忽略 accelerate_config=%s",
                accelerate_config,
            )

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
            if hasattr(layers[decoder_layer_idx], "mlp") and isinstance(layers[decoder_layer_idx].mlp, Qwen3MoeSparseMoeBlock):
                block = layers[decoder_layer_idx].mlp
                mask_key = f"layer_{decoder_layer_idx}.keep_mask"
                if mask_key not in state:
                    raise KeyError(f"Adapter 缺少 {mask_key}，请重新运行 calib 生成 adapter")
                keep_mask = state[mask_key].to(block.gate.weight.device).bool()
                layers[decoder_layer_idx].mlp = PrunedQwen3MoeSparseMoeBlock(block, keep_mask)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
