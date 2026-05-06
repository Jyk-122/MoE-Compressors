#!/usr/bin/env python3
"""
OptimalScale 可解释性可视化脚本

实现三个实验：
1. 通道级激活均值散点图 (Micro View)
2. 异常通道范数追踪
3. Token级余弦相似度CDF曲线 (Macro View)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
import types
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

from methods_skipping.os_lexi_skip.model_qwen3_moe import (
    OptimalScalingLexiSkipQwen3Moe,
    OptimalScaledLexiQwen3MoeSparseMoeBlock,
    _get_moe_layers,
    _resolve_layer_topk,
)


def _get_moe_layers_patched(model):
    """获取 MoE 层，同时支持原始和 patch 后的 block"""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(
            layer.mlp, (Qwen3MoeSparseMoeBlock, OptimalScaledLexiQwen3MoeSparseMoeBlock)
        ):
            moe_layers.append((i, layer.mlp))
    return moe_layers


def collect_hidden_states(
    model,
    tokenizer,
    texts,
    device,
    collect_layers=None,
    max_context_len=2048,
    batch_size=1,
):
    """
    收集各层 MoE 输出的 hidden states
    """
    model.eval()
    collected = {}
    
    moe_layers = _get_moe_layers_patched(model)
    moe_indices = [idx for idx, _ in moe_layers]
    
    if collect_layers is None:
        collect_layers = moe_indices
    
    for layer_idx in collect_layers:
        collected[layer_idx] = []
    
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            collected[layer_idx].append(output.detach().cpu())
        return hook
    
    for layer_idx, block in moe_layers:
        if layer_idx in collect_layers:
            hooks.append(block.register_forward_hook(make_hook(layer_idx)))
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Collecting hidden states"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_context_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    for layer_idx in collected:
        collected[layer_idx] = torch.cat(collected[layer_idx], dim=0).view(-1, collected[layer_idx][0].shape[-1])
    
    return collected


def calib_with_model(
    model,
    tokenizer,
    texts,
    layer_topk,
    adapter_dir,
    max_context_len=2048,
    batch_size=1,
):
    """使用已有的模型进行 calib，保存 forward 方法以便恢复"""
    import logging
    logger = logging.getLogger("MoECompressor")
    
    model.eval()
    moe_layers = _get_moe_layers_patched(model)
    num_experts = model.config.num_experts
    num_moe_layers = len(moe_layers)
    
    layer_topk = _resolve_layer_topk({"layer_topk": layer_topk}, num_moe_layers)
    logger.info(f"Using per-layer topK: {layer_topk}")
    
    # 保存原始 forward 方法
    original_forwards = {}
    for layer_idx, block in moe_layers:
        original_forwards[layer_idx] = block.forward

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

    adapter_path = Path(adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_file(state, str(adapter_path / "adapter.safetensors"))
    logger.info(f"Calibration completed. Adapters saved to {adapter_path / 'adapter.safetensors'}")
    
    # 恢复原始 forward 方法
    for layer_idx, block in moe_layers:
        if layer_idx in original_forwards:
            block.forward = original_forwards[layer_idx]


def get_alpha_only_output_with_model(
    model,
    tokenizer,
    texts,
    layer_topk,
    collect_layers=None,
    max_context_len=2048,
    batch_size=1,
):
    """
    使用已有的模型获取仅使用 α 调整（不使用 OptimalScale）的输出
    """
    model.eval()
    
    moe_layers = _get_moe_layers_patched(model)
    num_moe_layers = len(moe_layers)
    assert len(layer_topk) == num_moe_layers
    
    if collect_layers is None:
        collect_layers = [idx for idx, _ in moe_layers]
    
    collected = {}
    for layer_idx in collect_layers:
        collected[layer_idx] = []
    
    # 保存原始 forward 方法
    original_forwards = {}
    for layer_idx, block in moe_layers:
        original_forwards[layer_idx] = block.forward
    
    for (layer_idx, block), k_eff in zip(moe_layers, layer_topk):
        def make_hook(_layer_idx=layer_idx, _k=k_eff):
            def hook(self_block, hidden_states):
                bsz, seq_len, hidden_dim = hidden_states.shape
                hidden_reshaped = hidden_states.view(-1, hidden_dim)
                num_tokens = hidden_reshaped.shape[0]
                
                router_logits = F.linear(hidden_reshaped, self_block.gate.weight)
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                
                router_top_value_kept, router_indices_kept = torch.topk(router_probs, _k, dim=-1)
                if self_block.gate.norm_topk_prob:
                    kept_routing_weights = (
                        router_top_value_kept / router_top_value_kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    ).to(router_probs.dtype)
                else:
                    kept_routing_weights = router_top_value_kept
                
                final_hidden_states = torch.zeros_like(hidden_reshaped)
                selected_indices = torch.full(
                    (router_indices_kept.shape[0], self_block.gate.top_k),
                    -1,
                    dtype=torch.long,
                    device=router_indices_kept.device,
                )
                selected_indices[:, :_k] = router_indices_kept
                
                num_experts = self_block.num_experts
                for expert_idx in range(num_experts):
                    token_idx, top_k_pos = torch.where(selected_indices == expert_idx)
                    if token_idx.numel() == 0:
                        continue
                    current_state = hidden_reshaped[token_idx]
                    gate, up = F.linear(current_state, self_block.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    current_hidden_states = self_block.experts.act_fn(gate) * up
                    current_hidden_states = F.linear(current_hidden_states, self_block.experts.down_proj[expert_idx])
                    
                    scale_factor = kept_routing_weights[token_idx, top_k_pos, None]
                    current_hidden_states = current_hidden_states * scale_factor
                    
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
                
                output = final_hidden_states.reshape(bsz, seq_len, hidden_dim)
                if _layer_idx in collect_layers:
                    collected[_layer_idx].append(output.detach().cpu())
                return output
            return hook
        
        block.forward = types.MethodType(make_hook(), block)
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="Collecting alpha-only outputs"):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_context_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
    
    # 恢复原始 forward 方法
    for layer_idx, block in moe_layers:
        if layer_idx in original_forwards:
            block.forward = original_forwards[layer_idx]
    
    for layer_idx in collected:
        collected[layer_idx] = torch.cat(collected[layer_idx], dim=0).view(-1, collected[layer_idx][0].shape[-1])
    
    return collected


def compute_channel_stats(hidden_states):
    """计算每个通道的均值和方差"""
    mean = hidden_states.mean(dim=0).numpy()
    abs_mean = hidden_states.abs().mean(dim=0).numpy()
    var = hidden_states.var(dim=0).numpy()
    return mean, abs_mean, var


def compute_cosine_similarity(a, b):
    """计算两组向量的余弦相似度"""
    a_np = a.numpy()
    b_np = b.numpy()
    norm_a = np.linalg.norm(a_np, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b_np, axis=1, keepdims=True)
    similarity = (a_np @ b_np.T).diagonal() / (norm_a.flatten() * norm_b.flatten())
    return similarity


def plot_experiment_1(
    orig_abs_mean,
    alpha_abs_mean,
    scaled_abs_mean,
    layer_idx,
    config_name,
    out_dir,
):
    """
    实验1：通道级激活均值散点图
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x_min = min(orig_abs_mean.min(), alpha_abs_mean.min())
    x_max = max(orig_abs_mean.max(), alpha_abs_mean.max())
    ax1.scatter(orig_abs_mean, alpha_abs_mean, s=1, alpha=0.5)
    ax1.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=1)
    ax1.set_xlabel('Original Channel Abs Mean')
    ax1.set_ylabel('Alpha-Only Channel Abs Mean')
    ax1.set_title(f'Layer {layer_idx} ({config_name}): Alpha Only')
    ax1.grid(True, alpha=0.3)
    
    x_min = min(orig_abs_mean.min(), scaled_abs_mean.min())
    x_max = max(orig_abs_mean.max(), scaled_abs_mean.max())
    ax2.scatter(orig_abs_mean, scaled_abs_mean, s=1, alpha=0.5)
    ax2.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=1)
    ax2.set_xlabel('Original Channel Abs Mean')
    ax2.set_ylabel('OptimalScale Channel Abs Mean')
    ax2.set_title(f'Layer {layer_idx} ({config_name}): OptimalScale')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / f'exp1_layer{layer_idx}_{config_name}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved Experiment 1 plot to {out_path}')


def plot_experiment_2(
    orig_hidden,
    alpha_hidden,
    scaled_hidden,
    layer_idx,
    config_name,
    out_dir,
    top_k=10,
):
    """
    实验2：异常通道范数追踪
    """
    import matplotlib.pyplot as plt
    
    orig_norm = orig_hidden.norm(dim=0).numpy()
    top_indices = np.argsort(orig_norm)[::-1][:top_k]
    
    mse_alpha = ((alpha_hidden[:, top_indices] - orig_hidden[:, top_indices]) ** 2).mean(dim=0).numpy()
    mse_scaled = ((scaled_hidden[:, top_indices] - orig_hidden[:, top_indices]) ** 2).mean(dim=0).numpy()
    
    x = np.arange(top_k)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mse_alpha, width, label='Alpha Only')
    bars2 = ax.bar(x + width/2, mse_scaled, width, label='OptimalScale')
    
    ax.set_xlabel('Top Outlier Channel Index')
    ax.set_ylabel('MSE')
    ax.set_title(f'Layer {layer_idx} ({config_name}): Outlier Channel MSE')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch{idx}' for idx in top_indices])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = out_dir / f'exp2_layer{layer_idx}_{config_name}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved Experiment 2 plot to {out_path}')


def plot_experiment_3(
    orig_hidden,
    alpha_hidden,
    scaled_hidden,
    layer_idx,
    config_name,
    out_dir,
):
    """
    实验3：Token级余弦相似度CDF曲线
    """
    import matplotlib.pyplot as plt
    
    sim_alpha = compute_cosine_similarity(orig_hidden, alpha_hidden)
    sim_scaled = compute_cosine_similarity(orig_hidden, scaled_hidden)
    
    sim_alpha_sorted = np.sort(sim_alpha)
    sim_scaled_sorted = np.sort(sim_scaled)
    
    cdf_alpha = np.arange(1, len(sim_alpha_sorted) + 1) / len(sim_alpha_sorted)
    cdf_scaled = np.arange(1, len(sim_scaled_sorted) + 1) / len(sim_scaled_sorted)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sim_alpha_sorted, cdf_alpha, label='Alpha Only', linewidth=2)
    ax.plot(sim_scaled_sorted, cdf_scaled, label='OptimalScale', linewidth=2)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('CDF')
    ax.set_title(f'Layer {layer_idx} ({config_name}): Token-wise Cosine Similarity CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    
    plt.tight_layout()
    out_path = out_dir / f'exp3_layer{layer_idx}_{config_name}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved Experiment 3 plot to {out_path}')


def parse_layer_topk(layer_topk_str):
    """解析 layer_topk 字符串，支持多种格式"""
    # 尝试解析为 JSON
    import json
    try:
        return json.loads(layer_topk_str)
    except:
        pass
    
    # 尝试解析为逗号分隔的整数
    try:
        return [int(x.strip()) for x in layer_topk_str.split(',')]
    except:
        pass
    
    raise ValueError(f"无法解析 layer_topk: {layer_topk_str}")


def main():
    p = argparse.ArgumentParser(
        description="OptimalScale 可解释性可视化",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径或 HuggingFace 模型名",
    )
    p.add_argument(
        "--adapter_base_dir",
        type=str,
        required=True,
        help="adapter 基目录，包含不同配置的子目录",
    )
    p.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="配置列表，格式为 'name:layer_topk_str'，例如 'config1:3,4,3,4' 或 'high_prune:[2,2,2,2]'",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="wikitext:wikitext-2-raw-v1",
        help="校准数据集 (默认: wikitext:wikitext-2-raw-v1)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=128,
        help="最大校准样本数 (默认: 128)",
    )
    p.add_argument(
        "--max_context_len",
        type=int,
        default=2048,
        help="最大上下文长度 (默认: 2048)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="visualizations",
        help="输出目录 (默认: visualizations)",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="要分析的层索引列表 (默认: 所有 MoE 层)",
    )
    args = p.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析配置
    configs = {}
    for cfg in args.configs:
        name, layer_topk_str = cfg.split(':', 1)
        configs[name] = parse_layer_topk(layer_topk_str)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    print("Loading model (this is the only model we will load)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    compressor = OptimalScalingLexiSkipQwen3Moe(args.model, adapter_dir=None)
    texts = compressor.load_calibration_data(
        tokenizer,
        args.dataset,
        max_calib_samples=args.max_samples,
        max_context_len=args.max_context_len,
    )
    
    print("Collecting original hidden states...")
    orig_hiddens = collect_hidden_states(
        model,
        tokenizer,
        texts,
        device,
        collect_layers=args.layers,
        max_context_len=args.max_context_len,
    )
    
    moe_layers = _get_moe_layers_patched(model)
    num_moe_layers = len(moe_layers)
    
    if args.layers is None:
        args.layers = [idx for idx, _ in moe_layers]
    
    for config_name, layer_topk in configs.items():
        print(f"\n=== Processing config: {config_name} ===")
        print(f"Using layer_topk: {layer_topk}")
        
        # 验证 layer_topk 长度
        if len(layer_topk) != num_moe_layers:
            raise ValueError(f"layer_topk 长度 {len(layer_topk)} 与 MoE 层数 {num_moe_layers} 不匹配")
        
        adapter_dir = Path(args.adapter_base_dir) / config_name
        if not adapter_dir.exists():
            print(f"Calibrating for config {config_name} using existing model...")
            calib_with_model(
                model,
                tokenizer,
                texts,
                layer_topk,
                str(adapter_dir),
                max_context_len=args.max_context_len,
                batch_size=1,
            )
        
        print("Collecting alpha-only outputs using existing model...")
        alpha_hiddens = get_alpha_only_output_with_model(
            model,
            tokenizer,
            texts,
            layer_topk,
            collect_layers=args.layers,
            max_context_len=args.max_context_len,
            batch_size=1,
        )
        
        print("Patching model with OptimalScale...")
        scaled_compressor = OptimalScalingLexiSkipQwen3Moe(args.model, adapter_dir=str(adapter_dir))
        # 保存原始 mlp block，因为 patch 会替换整个 mlp 对象
        original_mlps = {}
        for layer_idx, block in moe_layers:
            original_mlps[layer_idx] = block
        
        scaled_model = scaled_compressor.patch(model, layer_topk=layer_topk)
        
        print("Collecting OptimalScale outputs...")
        scaled_hiddens = collect_hidden_states(
            scaled_model,
            tokenizer,
            texts,
            device,
            collect_layers=args.layers,
            max_context_len=args.max_context_len,
        )
        
        # 恢复原始 mlp block，移除 patch
        for layer_idx, block in original_mlps.items():
            model.model.layers[layer_idx].mlp = block
        
        for layer_idx in args.layers:
            print(f"\n--- Layer {layer_idx} ---")
            
            orig_h = orig_hiddens[layer_idx]
            alpha_h = alpha_hiddens[layer_idx]
            scaled_h = scaled_hiddens[layer_idx]
            
            orig_mean, orig_abs_mean, orig_var = compute_channel_stats(orig_h)
            alpha_mean, alpha_abs_mean, alpha_var = compute_channel_stats(alpha_h)
            scaled_mean, scaled_abs_mean, scaled_var = compute_channel_stats(scaled_h)
            
            plot_experiment_1(orig_abs_mean, alpha_abs_mean, scaled_abs_mean, layer_idx, config_name, out_dir)
            plot_experiment_2(orig_h, alpha_h, scaled_h, layer_idx, config_name, out_dir)
            plot_experiment_3(orig_h, alpha_h, scaled_h, layer_idx, config_name, out_dir)
        
        # 释放当前配置的临时数据
        del alpha_hiddens
        del scaled_hiddens
        torch.cuda.empty_cache()
    
    print(f"\nAll visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
