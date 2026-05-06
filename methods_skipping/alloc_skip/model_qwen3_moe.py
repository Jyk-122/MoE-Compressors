"""
Alloc-MoE: Budget-aware expert activation allocation for efficient MoE inference.

**Calib (Stage 1)**: Profile layer-wise sensitivity by constraining deeper layers to
Top-1 while varying current layer's Top-K, compute perplexity change on calibration
data, store sensitivity matrix and layer indices in adapter.

**Eval / patch (Stage 2)**:
1. Use dynamic programming to solve for optimal layer-wise budget allocation under global budget B
2. Optionally use Alloc-T (token-level redistribution) to reallocate activations within layer
   based on routing scores (enabled by default with `enable_alloc_t=True`)

Once calibrated, you can test multiple budgets by just changing `target_budget` or `compute_reduction`
without re-running calibration.
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
from transformers import AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from MoECompressor import MoECompressor
from utils.moe_stats import MoEStatsCollector

logger = logging.getLogger("MoECompressor")


def _resolve_bool_param(kwargs: dict[str, Any], key: str, default: bool) -> bool:
    v = kwargs.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _resolve_k_base(kwargs: dict[str, Any], k_max: int) -> int:
    k_base = int(kwargs.get("k_base", 1))
    if not (1 <= k_base <= k_max):
        raise ValueError(f"k_base 必须满足 1 <= k_base <= {k_max}")
    return k_base


def _dynamic_programming_allocation(
    S: torch.Tensor,
    B: int,
    k_min: int,
    k_max: int,
) -> list[int]:
    """
    Solve optimal layer-wise expert activation allocation using dynamic programming.
    
    Args:
        S: sensitivity matrix of shape [L, K], S[i, k] is cost of allocating (k+1) experts
           to layer i (0-based, since k_min=1 maps to index 0)
        B: total global budget (sum of k_i across layers)
        k_min: minimum experts per layer
        k_max: maximum experts per layer
        
    Returns:
        k_list: list of integers of length L, optimal k_i per layer
    """
    L = S.shape[0]
    K = S.shape[1]  # number of possible k values (from k_min to k_max)
    # Ensure B is within feasible range
    B_min = L * k_min
    B_max = L * k_max
    if B < B_min:
        logger.warning(f"Budget B={B} < min feasible B_min={B_min}, using B_min")
        B = B_min
    if B > B_max:
        logger.warning(f"Budget B={B} > max feasible B_max={B_max}, using B_max")
        B = B_max
    
    # DP table: dp[i, b] = min total cost for first i+1 layers, total budget b
    dp = torch.full((L, B + 1), float('inf'), dtype=torch.float64)
    # Backtrack table: track[i, b] = k allocated to layer i when total budget is b
    track = torch.full((L, B + 1), -1, dtype=torch.int64)
    
    # Initialize first layer
    for k in range(k_min, k_max + 1):
        b = k
        if b <= B:
            s_idx = k - k_min  # map k to 0-based index in S
            dp[0, b] = S[0, s_idx]
            track[0, b] = k
    
    # Fill DP table
    for i in range(1, L):
        # For each possible total budget b up to B
        for b in range(B + 1):
            # Try all possible k for current layer i
            for k in range(k_min, k_max + 1):
                if b >= k:
                    prev_b = b - k
                    s_idx = k - k_min
                    current_cost = dp[i-1, prev_b] + S[i, s_idx]
                    if current_cost < dp[i, b]:
                        dp[i, b] = current_cost
                        track[i, b] = k
    
    # Backtrack to find allocation
    k_list = [0] * L
    remaining_b = B
    for i in range(L-1, -1, -1):
        k_list[i] = track[i, remaining_b].item()
        remaining_b -= k_list[i]
    
    return k_list


def _get_moe_layer_indices(model) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
    ]


def _compute_perplexity_with_config(
    model,
    tokenizer,
    texts: list[str],
    moe_indices: list[int],
    layer_k_config: list[int],
) -> float:
    """
    Compute perplexity with given per-layer Top-K configuration.
    
    Args:
        model: causal LM model
        tokenizer: tokenizer
        texts: calibration texts
        moe_indices: list of MoE layer indices
        layer_k_config: list of integers, length len(moe_indices), k for each MoE layer
        
    Returns:
        perplexity: float
    """
    # Save original blocks (reference only, no copy)
    original_blocks = {}
    for j, layer_idx in enumerate(moe_indices):
        original_blocks[layer_idx] = model.model.layers[layer_idx].mlp
    
    # Patch layers with config
    class TempTopKBlock(torch.nn.Module):
        def __init__(self, orig_block, k_eff):
            super().__init__()
            self.gate = orig_block.gate
            self.k_eff = k_eff
            self.experts = orig_block.experts
        
        def forward(self, hidden_states):
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
            
            final_hidden_states = torch.zeros_like(hidden_states_reshaped)
            num_experts = self.gate.num_experts
            for expert_idx in range(num_experts):
                token_idx, top_k_pos = torch.where(router_indices == expert_idx)
                if token_idx.numel() == 0:
                    continue
                current_state = hidden_states_reshaped[token_idx]
                gate, up = F.linear(current_state, self.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.experts.act_fn(gate) * up
                current_hidden_states = F.linear(current_hidden_states, self.experts.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    
    # Apply patches
    for j, layer_idx in enumerate(moe_indices):
        model.model.layers[layer_idx].mlp = TempTopKBlock(
            original_blocks[layer_idx],
            layer_k_config[j],
        )
    
    # Compute perplexity
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity", leave=False):
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = encodings.input_ids.to(model.device)
            if input_ids.shape[1] <= 1:
                continue
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            num_tokens = input_ids.shape[1] - 1
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Restore original blocks
    for layer_idx, block in original_blocks.items():
        model.model.layers[layer_idx].mlp = block
    
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


class AllocSkippedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    MoE block with Alloc-MoE:
    - Per-layer budget K_l (from Alloc-L)
    - Optional token-level adaptive redistribution (Alloc-T)
    """
    
    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        k_layer: int,
        k_base: int,
        enable_alloc_t: bool,
        layer_idx: int,
        stats_collector: MoEStatsCollector | None,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.k_layer = int(k_layer)
        self.k_base = int(k_base)
        self.enable_alloc_t = bool(enable_alloc_t)
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj.clone())
        self.act_fn = copy.deepcopy(experts.act_fn)
        self.layer_idx = layer_idx
        self.stats_collector = stats_collector
        
        # Validate constraints
        if not (1 <= self.k_base <= self.k_layer <= self.top_k):
            raise ValueError(
                f"k_base={self.k_base} <= k_layer={self.k_layer} <= top_k={self.top_k} not satisfied"
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        T = hidden_states_reshaped.shape[0]  # total tokens
        
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(router_logits.dtype)
        
        # Always get top-k_max candidates for possible redistribution
        router_top_value_full, router_indices_full = torch.topk(router_probs, self.top_k, dim=-1)
        
        if not self.enable_alloc_t:
            # Simple case: fixed k_layer per token
            router_top_value = router_top_value_full[:, :self.k_layer]
            router_indices = router_indices_full[:, :self.k_layer]
            active_mask = torch.ones_like(router_indices, dtype=torch.bool)
        else:
            # Alloc-T: token-level adaptive redistribution
            # 1. Keep k_base for each token
            k_remaining_per_token = self.top_k - self.k_base
            total_extra_budget = (self.k_layer - self.k_base) * T
            
            # 2. Collect all remaining candidate scores
            extra_scores = router_top_value_full[:, self.k_base:].contiguous()
            extra_indices = router_indices_full[:, self.k_base:].contiguous()
            
            # 3. Flatten and select global top total_extra_budget
            flat_scores = extra_scores.view(-1)
            _, flat_indices = torch.topk(flat_scores, total_extra_budget)
            
            # 4. Create selection mask
            active_mask = torch.zeros((T, self.top_k), dtype=torch.bool, device=router_top_value_full.device)
            active_mask[:, :self.k_base] = True  # keep k_base
            
            if total_extra_budget > 0:
                # Convert flat indices back to (token, pos) coordinates
                token_pos = flat_indices // k_remaining_per_token
                pos_in_extra = flat_indices % k_remaining_per_token
                pos_in_full = self.k_base + pos_in_extra
                active_mask[token_pos, pos_in_full] = True
            
            # Use full indices and mask
            router_indices = router_indices_full
            router_top_value = router_top_value_full
        
        # Normalize routing weights
        if self.gate.norm_topk_prob:
            # Only normalize activated positions
            sum_weights = (router_top_value * active_mask).sum(dim=-1, keepdim=True)
            router_top_value = router_top_value * active_mask / sum_weights.clamp_min(1e-12)
        else:
            router_top_value = router_top_value * active_mask
        
        routing_weights = router_top_value
        
        # Update stats collector
        if self.stats_collector is not None:
            # Pad to original top_k with -1 for inactive positions
            padded_indices = torch.where(
                active_mask,
                router_indices,
                torch.full_like(router_indices, -1),
            )
            self.stats_collector.update(
                layer_idx=self.layer_idx,
                selected_indices=padded_indices.detach(),
                default_top_k=self.top_k,
                sequence_length=sequence_length,
            )
        
        # Compute expert outputs
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        for expert_idx in range(self.num_experts):
            token_idx, top_k_pos = torch.where(router_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            # Filter out positions where expert is not active
            is_active = active_mask[token_idx, top_k_pos]
            if not is_active.any():
                continue
            token_idx_active = token_idx[is_active]
            top_k_pos_active = top_k_pos[is_active]
            
            current_state = hidden_states_reshaped[token_idx_active]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx_active, top_k_pos_active, None]
            final_hidden_states.index_add_(0, token_idx_active, current_hidden_states.to(final_hidden_states.dtype))
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class AllocSkipQwen3Moe(MoECompressor):
    """
    Alloc-MoE: budget-aware expert activation allocation.
    
    **Calib**: Layer-wise sensitivity profiling using calibration data, stores sensitivity matrix.
    **Patch**: Dynamic programming for layer-wise allocation + optional token-level redistribution (Alloc-T).
    """
    
    ADAPTER_KEYS = {
        "sensitivity": "alloc.sensitivity",
        "k_max": "alloc.k_max",
        "k_min": "alloc.k_min",
        "layer_indices": "alloc.layer_indices",
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
        Stage1: Layer-wise sensitivity profiling.
        
        Uses allocation-isolating profiling: when profiling layer i, set all deeper layers
        to Top-1 and keep preceding layers at original Top-K to isolate the effect of
        current layer's expert count change.
        
        Args:
            calibration_dataset: HuggingFace dataset for calibration
            max_calib_samples: max calibration samples
            max_context_len: max context length per sample
            batch_size: batch size for profiling
        """
        if self.adapter_dir is None:
            raise ValueError("alloc_skip 的 calib 需要提供 --adapter_dir")
        
        from transformers import AutoModelForCausalLM
        
        logger.info("[alloc_skip][calib] Loading model and tokenizer for Stage1 profiling")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device == "cpu":
            model = model.to(self.device)
        model.eval()
        
        # Load calibration data
        texts = self.load_calibration_data(
            tokenizer,
            calibration_dataset,
            max_calib_samples,
            max_context_len,
        )
        
        moe_indices = _get_moe_layer_indices(model)
        if not moe_indices:
            raise RuntimeError("模型中未找到 Qwen3MoeSparseMoeBlock")
        
        first_block = model.model.layers[moe_indices[0]].mlp
        k_max = int(first_block.gate.top_k)
        k_min = 1
        L = len(moe_indices)
        
        # Verify all layers have same k_max
        for layer_idx in moe_indices:
            kb = int(model.model.layers[layer_idx].mlp.gate.top_k)
            if kb != k_max:
                raise ValueError(
                    f"[alloc_skip] 要求所有 MoE 层 gate.top_k 一致，层 {layer_idx} 为 {kb}，期望 {k_max}"
                )
        
        # Sensitivity matrix: S[i, k_idx] where k_idx = k - k_min
        num_k_values = k_max - k_min + 1
        S = torch.zeros((L, num_k_values), dtype=torch.float64)
        
        for i, layer_idx in enumerate(tqdm(reversed(moe_indices), desc="alloc_skip Stage1 (per layer)", unit="layer")):
            # Process layers from last to first (i in 0..L-1 maps to moe_indices in reverse)
            real_i = L - 1 - i
            real_layer_idx = moe_indices[real_i]
            
            for k in range(k_min, k_max + 1):
                k_idx = k - k_min
                
                # Build config:
                # - layers before real_i: k_max
                # - layer real_i: k
                # - layers after real_i: k_min (Top-1)
                layer_k_config = []
                for j in range(L):
                    if j < real_i:
                        layer_k_config.append(k_max)
                    elif j == real_i:
                        layer_k_config.append(k)
                    else:
                        layer_k_config.append(k_min)
                
                # Compute perplexity with this config
                ppl = _compute_perplexity_with_config(
                    model,
                    tokenizer,
                    texts,
                    moe_indices,
                    layer_k_config,
                )
                S[real_i, k_idx] = ppl
                logger.debug(
                    "[alloc_skip][calib] Layer %d, k=%d: perplexity=%.4f",
                    real_layer_idx,
                    k,
                    ppl,
                )
        
        # Normalize sensitivity matrix by first value (k_min) to make it relative
        for i in range(L):
            S[i] = S[i] - S[i, 0].item()  # S[i, k_min] becomes 0
        
        # Save adapter
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        layer_indices_t = torch.tensor(moe_indices, dtype=torch.int64)
        state = {
            self.ADAPTER_KEYS["sensitivity"]: S.cpu(),
            self.ADAPTER_KEYS["k_max"]: torch.tensor(k_max, dtype=torch.int32),
            self.ADAPTER_KEYS["k_min"]: torch.tensor(k_min, dtype=torch.int32),
            self.ADAPTER_KEYS["layer_indices"]: layer_indices_t,
        }
        save_file(state, str(self._get_adapter_path()))
        logger.info(
            "[alloc_skip][calib] Saved sensitivity S shape=%s, k_max=%d, k_min=%d to %s",
            tuple(S.shape),
            k_max,
            k_min,
            self._get_adapter_path(),
        )
        
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def patch(self, model, **kwargs) -> Any:
        """
        Stage2: Layer-wise allocation via dynamic programming + optional Alloc-T.
        
        patch_kwargs (budget specification, choose one):
        - compute_reduction: float, e.g., 0.25 means 25%% reduction in total activations
        - target_budget: int, directly specify total budget B
        
        optional:
        - layer_k: list[int], if provided, skips DP and uses directly
        - enable_alloc_t: bool, default True, enable token-level redistribution
        - k_base: int, default 1, minimum experts per token for Alloc-T
        """
        if self.adapter_dir is None:
            raise ValueError("alloc_skip 的 patch 需要提供 --adapter_dir")
        if self.adapter_path is None or not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")
        
        state = load_file(str(self.adapter_path))
        S = state[self.ADAPTER_KEYS["sensitivity"]]
        k_max = int(state[self.ADAPTER_KEYS["k_max"]].item())
        k_min = int(state[self.ADAPTER_KEYS["k_min"]].item())
        saved_indices = state[self.ADAPTER_KEYS["layer_indices"]].long().tolist()
        
        layers = model.model.layers
        moe_indices = [
            i
            for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        if moe_indices != saved_indices:
            logger.warning(
                "[alloc_skip] 当前模型 MoE 层索引 %s 与 adapter %s 不一致，仍尝试继续",
                moe_indices,
                saved_indices,
            )
        
        L = S.shape[0]
        if len(moe_indices) != L:
            raise ValueError(
                f"敏感度矩阵层数 L={L} 与当前模型 MoE 层数 {len(moe_indices)} 不一致"
            )
        
        # Determine per-layer k allocation
        layer_k_list = kwargs.get("layer_k")
        if layer_k_list is not None:
            k_list = [int(x) for x in layer_k_list]
            if len(k_list) != L:
                raise ValueError(f"layer_k 长度应为 {L}，收到 {len(k_list)}")
            for kj in k_list:
                if not (k_min <= kj <= k_max):
                    raise ValueError(f"layer_k 元素应在 [{k_min}, {k_max}] 之间，收到 {kj}")
        else:
            B0 = L * k_max
            if kwargs.get("target_budget") is not None:
                B = int(kwargs["target_budget"])
            elif kwargs.get("compute_reduction") is not None:
                r = float(kwargs["compute_reduction"])
                B = int(round(B0 * (1.0 - r)))
            else:
                raise ValueError(
                    'alloc_skip 的 patch 需要 patch_kwargs 中的 compute_reduction 或 target_budget，'
                    '例如 {"compute_reduction": 0.25}'
                )
            
            # Solve via dynamic programming
            logger.info("[alloc_skip][patch] Solving DP for B=%d", B)
            k_list = _dynamic_programming_allocation(S, B, k_min, k_max)
        
        # Alloc-T settings
        enable_alloc_t = _resolve_bool_param(kwargs, "enable_alloc_t", True)
        k_base = _resolve_k_base(kwargs, k_max)
        
        stats_collector = MoEStatsCollector(num_experts=model.config.num_experts)
        stats_collector.initialize_layers(moe_indices)
        
        logger.info(
            "[alloc_skip][patch] Per-layer k: %s, enable_alloc_t=%s, k_base=%d",
            k_list,
            enable_alloc_t,
            k_base,
        )
        for j, decoder_layer_idx in enumerate(
            tqdm(moe_indices, desc="Patching layers (alloc_skip)", unit="layer")
        ):
            block = layers[decoder_layer_idx].mlp
            kj = int(k_list[j])
            # Ensure k_base <= kj
            current_k_base = min(k_base, kj)
            layers[decoder_layer_idx].mlp = AllocSkippedQwen3MoeSparseMoeBlock(
                block,
                k_layer=kj,
                k_base=current_k_base,
                enable_alloc_t=enable_alloc_t,
                layer_idx=decoder_layer_idx,
                stats_collector=stats_collector,
            )
        
        self._acceleration_stats_collector = stats_collector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
