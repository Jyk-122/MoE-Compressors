"""
MoE-I2 Inter-Expert Pruning（Qwen3-MoE 实现）

实现要点：
1) Layer Importance Analysis：单专家移除损失，计算层重要性。
2) Layer-wise Genetic Search：每层搜索待剪专家组合候选。
3) Block-wise KT-Reception Field：按块在候选中选联合最优组合。
4) Structured Expert Pruning：保存 keep_indices/old_to_new，patch 时替换 MoE block。
"""

from __future__ import annotations

import copy
import gc
import itertools
import logging
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from MoECompressor import MoECompressor
from utils.adapter_calib_config import saved_prune_ratio_from_adapter_dir

logger = logging.getLogger("MoECompressor")


def _get_moe_layers(model) -> list[tuple[int, Qwen3MoeSparseMoeBlock]]:
    """返回所有 MoE 层：(decoder_layer_idx, mlp_block)。"""
    layers: list[tuple[int, Qwen3MoeSparseMoeBlock]] = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            layers.append((i, layer.mlp))
    return layers


def _build_old_to_new(num_experts: int, keep_indices: torch.LongTensor) -> torch.LongTensor:
    old_to_new = torch.full((num_experts,), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(keep_indices.tolist()):
        old_to_new[old_idx] = new_idx
    return old_to_new


def _sample_unique_combo(num_experts: int, prune_count: int, rng: random.Random) -> tuple[int, ...]:
    if prune_count <= 0:
        return tuple()
    return tuple(sorted(rng.sample(range(num_experts), k=prune_count)))


class PrunedQwen3MoeSparseMoeBlock(torch.nn.Module):
    """
    剪枝后的 SparseMoeBlock（结构化裁剪专家）。

    - Router：将被剪专家 logits 置为 -inf，再做 softmax+topk。
    - Experts：仅保留 keep_indices 对应的权重。
    """

    def __init__(
        self,
        original_block: Qwen3MoeSparseMoeBlock,
        keep_indices: torch.LongTensor,
        old_to_new: torch.LongTensor,
    ):
        super().__init__()
        self.gate = copy.deepcopy(original_block.gate)
        self.top_k = self.gate.top_k
        self.num_experts = self.gate.num_experts
        self.keep_indices = keep_indices
        self.old_to_new = old_to_new
        self.num_kept = len(keep_indices)
        experts = original_block.experts
        self.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj[keep_indices].clone())
        self.down_proj = torch.nn.Parameter(experts.down_proj[keep_indices].clone())
        self.act_fn = copy.deepcopy(experts.act_fn)

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


class MoEI2PruningQwen3Moe(MoECompressor):
    """
    MoE-I2 第一阶段：Inter-Expert Pruning（Qwen3-MoE）。
    """

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
    def _moe_output_with_pruned_experts(
        self,
        block: Qwen3MoeSparseMoeBlock,
        hidden_states: torch.Tensor,
        pruned_experts: set[int],
    ) -> torch.Tensor:
        """
        使用“动态屏蔽专家”模拟该层剪枝后的输出。
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        gate = block.gate
        experts = block.experts
        num_experts = gate.num_experts

        router_logits = F.linear(hidden_states_reshaped, gate.weight)
        if pruned_experts:
            router_logits = router_logits.clone()
            mask = torch.zeros(num_experts, device=router_logits.device, dtype=torch.bool)
            mask[list(pruned_experts)] = True
            router_logits[:, mask] = float("-inf")

        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_vals, top_idx = torch.topk(router_probs, gate.top_k, dim=-1)
        if gate.norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        routing_weights = top_vals.to(hidden_states_reshaped.dtype)

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = F.one_hot(top_idx, num_classes=num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for idx in expert_hit:
            expert_idx = idx[0].item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            gate_out, up_out = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = experts.act_fn(gate_out) * up_out
            current_hidden_states = F.linear(current_hidden_states, experts.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def _collect_layer_inputs(
        self,
        model,
        tokenizer,
        texts: list[str],
        max_context_len: int,
        batch_size: int,
        search_max_batches: int,
    ) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
        """
        收集每层 MoE 输入和基线输出，后续用于快速评估候选剪枝组合。
        """
        moe_layers = _get_moe_layers(model)
        moe_decoder_indices = [idx for idx, _ in moe_layers]
        layer_inputs: dict[int, list[torch.Tensor]] = {idx: [] for idx in moe_decoder_indices}
        layer_outputs: dict[int, list[torch.Tensor]] = {idx: [] for idx in moe_decoder_indices}

        n_batches = min((len(texts) + batch_size - 1) // batch_size, search_max_batches)
        for b_idx, start in enumerate(
            tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="Collect layer cache", unit="batch")
        ):
            if b_idx >= search_max_batches:
                break
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
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            # hidden_states[l] 是进入 decoder 第 l 层前的激活
            for decoder_layer_idx, block in moe_layers:
                hs = hidden_states[decoder_layer_idx].detach()
                layer_inputs[decoder_layer_idx].append(hs)
                base_out = block(hs).detach()
                layer_outputs[decoder_layer_idx].append(base_out)

            if (b_idx + 1) >= n_batches:
                break
        return layer_inputs, layer_outputs

    def _combo_loss(
        self,
        block: Qwen3MoeSparseMoeBlock,
        layer_inputs: list[torch.Tensor],
        base_outputs: list[torch.Tensor],
        combo: tuple[int, ...],
        cache: dict[tuple[int, ...], float],
    ) -> float:
        """
        评估一个待剪组合在该层的扰动损失（Frobenius norm 累加）。
        """
        if combo in cache:
            return cache[combo]

        pruned = set(combo)
        loss = 0.0
        with torch.no_grad():
            for hs, base_out in zip(layer_inputs, base_outputs):
                pruned_out = self._moe_output_with_pruned_experts(block, hs, pruned)
                diff = (base_out - pruned_out).float()
                loss += torch.norm(diff, p="fro").item()
        cache[combo] = loss
        return loss

    def _allocate_non_uniform_prune_counts(
        self,
        layer_importance: dict[int, float],
        num_experts: int,
        top_k: int,
        num_layers: int,
        prune_ratio: float,
    ) -> dict[int, int]:
        """
        给定全局剪枝率，将待剪专家数分配到各层（重要层少剪）。
        """
        max_prune_per_layer = max(0, num_experts - top_k)
        total_capacity = max_prune_per_layer * num_layers
        target_total_prune = int(round(prune_ratio * num_experts * num_layers))
        target_total_prune = min(max(target_total_prune, 0), total_capacity)

        if target_total_prune == 0:
            return {k: 0 for k in layer_importance}

        # 重要性高 -> 少剪：用 1/(I+eps) 作为可剪权重
        eps = 1e-12
        scores = {k: 1.0 / (v + eps) for k, v in layer_importance.items()}
        score_sum = sum(scores.values()) + eps

        base_alloc = {}
        frac_parts = []
        used = 0
        for layer_idx, score in scores.items():
            raw = target_total_prune * (score / score_sum)
            alloc = int(raw)
            alloc = min(max(alloc, 0), max_prune_per_layer)
            base_alloc[layer_idx] = alloc
            used += alloc
            frac_parts.append((raw - alloc, layer_idx))

        remain = target_total_prune - used
        frac_parts.sort(reverse=True, key=lambda x: x[0])
        ptr = 0
        while remain > 0 and ptr < len(frac_parts):
            _, layer_idx = frac_parts[ptr]
            if base_alloc[layer_idx] < max_prune_per_layer:
                base_alloc[layer_idx] += 1
                remain -= 1
            ptr += 1
            if ptr == len(frac_parts) and remain > 0:
                ptr = 0
        return base_alloc

    def _layerwise_genetic_search(
        self,
        block: Qwen3MoeSparseMoeBlock,
        layer_inputs: list[torch.Tensor],
        base_outputs: list[torch.Tensor],
        prune_count: int,
        topk_candidates: int,
        population_size: int,
        iters: int,
        parent_fraction: float,
        mutation_prob: float,
        mutation_swap: int,
        rng: random.Random,
    ) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], float]]:
        """
        对单层执行 GA，返回 Top-K 待剪组合。
        """
        num_experts = block.gate.num_experts
        if prune_count <= 0:
            return [tuple()], {tuple(): 0.0}

        fitness_cache: dict[tuple[int, ...], float] = {}
        population: list[tuple[int, ...]] = []
        seen = set()
        while len(population) < population_size:
            combo = _sample_unique_combo(num_experts, prune_count, rng)
            if combo not in seen:
                seen.add(combo)
                population.append(combo)

        n_parents = max(2, int(population_size * parent_fraction))
        for _ in range(iters):
            ranked = sorted(
                population,
                key=lambda c: self._combo_loss(block, layer_inputs, base_outputs, c, fitness_cache),
            )
            parents = ranked[:n_parents]
            new_population = parents[:]

            while len(new_population) < population_size:
                p1, p2 = rng.sample(parents, 2)
                union = list(set(p1).union(set(p2)))
                if len(union) >= prune_count:
                    child = tuple(sorted(rng.sample(union, prune_count)))
                else:
                    fill_pool = [x for x in range(num_experts) if x not in union]
                    need = prune_count - len(union)
                    child = tuple(sorted(union + rng.sample(fill_pool, need)))

                # 每个 offspring 都执行 mutation 步骤：
                # 变异位点数量是随机值（few experts），并由 mutation_prob 控制强度。
                child_list = list(child)
                max_swap_num = min(mutation_swap, prune_count)
                if max_swap_num > 0:
                    # 在 [0, max_swap_num] 个候选位点上按 mutation_prob 采样，
                    sampled_swap_num = sum(1 for _ in range(max_swap_num) if rng.random() < mutation_prob)
                    # 为与论文描述对齐，至少执行 1 次替换。
                    swap_num = max(1, min(max_swap_num, sampled_swap_num))
                    mutate_positions = rng.sample(range(prune_count), k=swap_num)
                    for pos in mutate_positions:
                        used = set(child_list)
                        candidates = [x for x in range(num_experts) if x not in used]
                        if candidates:
                            child_list[pos] = rng.choice(candidates)
                    child = tuple(sorted(child_list))
                new_population.append(child)
            population = new_population[:population_size]

        ranked = sorted(
            population,
            key=lambda c: self._combo_loss(block, layer_inputs, base_outputs, c, fitness_cache),
        )
        unique_ranked = []
        used = set()
        for c in ranked:
            if c not in used:
                used.add(c)
                unique_ranked.append(c)
            if len(unique_ranked) >= topk_candidates:
                break
        return unique_ranked, fitness_cache

    def _blockwise_kt_select(
        self,
        moe_layers: list[tuple[int, Qwen3MoeSparseMoeBlock]],
        candidates_per_layer: dict[int, list[tuple[int, ...]]],
        layer_inputs: dict[int, list[torch.Tensor]],
        layer_outputs: dict[int, list[torch.Tensor]],
        kt_t: int,
    ) -> dict[int, tuple[int, ...]]:
        """
        按 block（长度 T）联合选择每层候选组合。

        与逐层独立打分不同，这里对 block 内候选做联合评估：
        给定一个组合分配（每层一个候选），从 block 首层输入开始顺序执行
        “带剪枝专家”的 MoE 前向，最后在 block 末层输出与基线输出之间计算
        reconstruction loss（Frobenius norm）。
        """
        selected: dict[int, tuple[int, ...]] = {}
        layer_to_block = {idx: block for idx, block in moe_layers}
        layer_indices = [idx for idx, _ in moe_layers]
        for b_start in range(0, len(layer_indices), kt_t):
            block_layers = layer_indices[b_start : b_start + kt_t]
            candidate_lists = [candidates_per_layer[i] for i in block_layers]
            first_layer_idx = block_layers[0]
            last_layer_idx = block_layers[-1]
            best_score = float("inf")
            best_assign = None

            for assign in itertools.product(*candidate_lists):
                score = 0.0
                # 对每个缓存样本做 block 联合前向，保留跨层耦合。
                for sample_idx in range(len(layer_inputs[first_layer_idx])):
                    hs = layer_inputs[first_layer_idx][sample_idx]
                    for layer_idx, combo in zip(block_layers, assign):
                        block = layer_to_block[layer_idx]
                        hs = self._moe_output_with_pruned_experts(
                            block=block,
                            hidden_states=hs,
                            pruned_experts=set(combo),
                        )
                    base_out = layer_outputs[last_layer_idx][sample_idx]
                    diff = (base_out - hs).float()
                    score += torch.norm(diff, p="fro").item()
                if score < best_score:
                    best_score = score
                    best_assign = assign

            if best_assign is None:
                for layer_idx in block_layers:
                    selected[layer_idx] = tuple()
            else:
                for layer_idx, combo in zip(block_layers, best_assign):
                    selected[layer_idx] = combo
        return selected

    def calib(
        self,
        calibration_dataset: str,
        max_calib_samples: int = 512,
        max_context_len: int = 2048,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        论文第一阶段复现：Layer Importance + Layer-wise GA + Block-wise KT。

        可通过 --calib_extra 传入：
        - ga_population: int (默认 100)
        - ga_iters: int (默认 50)
        - ga_parent_fraction: float (默认 0.2)
        - ga_mutation_prob: float (默认 0.3)
        - ga_mutation_swap: int (默认 1)
        - kt_k: int (默认 3)
        - kt_t: int (默认 3)
        - search_max_batches: int (默认 16)
        - seed: int (默认 42)
        - max_layers_for_search: int | None (默认 None)

        论文对齐推荐配置（Inter-Expert Pruning）：
        - prune_ratio=0.25
        - max_calib_samples=2048
        - max_context_len=2048
        - ga_population=100
        - ga_iters=50
        - kt_k=3
        - kt_t=3
        - search_max_batches=2048

        可直接通过 run_pruning.sh / run.py 调用（单卡 calib），例如：
        CALIB_KWARGS='{"prune_ratio":0.25,"ga_population":100,"ga_iters":50,"kt_k":3,"kt_t":3,"search_max_batches":2048,"seed":42}' \\
        bash run_pruning.sh calib
        """
        if self.adapter_dir is None:
            raise ValueError("calib 需提供 adapter_dir")

        prune_ratio = float(kwargs.get("prune_ratio", 0.5))
        ga_population = int(kwargs.get("ga_population", 100))
        ga_iters = int(kwargs.get("ga_iters", 50))
        ga_parent_fraction = float(kwargs.get("ga_parent_fraction", 0.2))
        ga_mutation_prob = float(kwargs.get("ga_mutation_prob", 0.3))
        ga_mutation_swap = int(kwargs.get("ga_mutation_swap", 1))
        kt_k = int(kwargs.get("kt_k", 3))
        kt_t = int(kwargs.get("kt_t", 3))
        search_max_batches = int(kwargs.get("search_max_batches", 16))
        seed = int(kwargs.get("seed", 42))
        max_layers_for_search = kwargs.get("max_layers_for_search", None)
        if max_layers_for_search is not None:
            max_layers_for_search = int(max_layers_for_search)

        rng = random.Random(seed)
        torch.manual_seed(seed)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[moei2][calib] Step 0/6: Loading model and tokenizer")
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
        model.eval()

        logger.info("[moei2][calib] Step 1/6: Loading calibration data")
        texts = self.load_calibration_data(
            tokenizer=tokenizer,
            calibration_dataset=calibration_dataset,
            max_calib_samples=max_calib_samples,
            max_context_len=max_context_len,
        )
        
        logger.info("[moei2][calib] Step 2/6: Building layer cache")
        layer_inputs, layer_outputs = self._collect_layer_inputs(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            max_context_len=max_context_len,
            batch_size=batch_size,
            search_max_batches=search_max_batches,
        )

        moe_layers = _get_moe_layers(model)
        if max_layers_for_search is not None and max_layers_for_search > 0:
            moe_layers = moe_layers[:max_layers_for_search]
            layer_inputs = {k: v for k, v in layer_inputs.items() if k in dict(moe_layers)}
            layer_outputs = {k: v for k, v in layer_outputs.items() if k in dict(moe_layers)}

        num_experts = moe_layers[0][1].gate.num_experts
        top_k = moe_layers[0][1].gate.top_k

        logger.info("[moei2][calib] Step 3/6: Layer Importance Analysis")
        layer_importance: dict[int, float] = {}
        for decoder_layer_idx, block in tqdm(moe_layers, desc="Layer importance", unit="layer"):
            fit_cache: dict[tuple[int, ...], float] = {}
            total_importance = 0.0
            for expert_idx in range(num_experts):
                combo = (expert_idx,)
                loss = self._combo_loss(
                    block=block,
                    layer_inputs=layer_inputs[decoder_layer_idx],
                    base_outputs=layer_outputs[decoder_layer_idx],
                    combo=combo,
                    cache=fit_cache,
                )
                total_importance += loss
            layer_importance[decoder_layer_idx] = total_importance

        prune_counts = self._allocate_non_uniform_prune_counts(
            layer_importance=layer_importance,
            num_experts=num_experts,
            top_k=top_k,
            num_layers=len(moe_layers),
            prune_ratio=prune_ratio,
        )

        logger.info("[moei2][calib] Non-uniform prune counts: %s", prune_counts)

        logger.info("[moei2][calib] Step 4/6: Layer-wise Genetic Search")
        candidates_per_layer: dict[int, list[tuple[int, ...]]] = {}
        for decoder_layer_idx, block in tqdm(moe_layers, desc="Genetic search", unit="layer"):
            prune_count = prune_counts.get(decoder_layer_idx, 0)
            # 若该层不剪，直接使用空组合候选。
            if prune_count <= 0:
                candidates_per_layer[decoder_layer_idx] = [tuple()]
                continue

            candidates, fit_cache = self._layerwise_genetic_search(
                block=block,
                layer_inputs=layer_inputs[decoder_layer_idx],
                base_outputs=layer_outputs[decoder_layer_idx],
                prune_count=prune_count,
                topk_candidates=max(1, kt_k),
                population_size=max(4, ga_population),
                iters=max(1, ga_iters),
                parent_fraction=min(max(ga_parent_fraction, 0.05), 0.8),
                mutation_prob=min(max(ga_mutation_prob, 0.0), 1.0),
                mutation_swap=max(1, ga_mutation_swap),
                rng=rng,
            )
            candidates_per_layer[decoder_layer_idx] = candidates

        logger.info("[moei2][calib] Step 5/6: Block-wise KT-Reception Field")
        selected_pruned = self._blockwise_kt_select(
            moe_layers=moe_layers,
            candidates_per_layer=candidates_per_layer,
            layer_inputs=layer_inputs,
            layer_outputs=layer_outputs,
            kt_t=max(1, kt_t),
        )

        logger.info("[moei2][calib] Step 6/6: Save adapter")
        keep_per_layer = {}
        for decoder_layer_idx, _ in moe_layers:
            pruned_combo = set(selected_pruned.get(decoder_layer_idx, tuple()))
            keep_indices = torch.tensor(
                [i for i in range(num_experts) if i not in pruned_combo],
                dtype=torch.long,
            )
            old_to_new = _build_old_to_new(num_experts, keep_indices)
            keep_per_layer[str(decoder_layer_idx)] = {
                "keep_indices": keep_indices.cpu(),
                "old_to_new": old_to_new.cpu(),
            }

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        state = {f"layer_{k}.keep_indices": v["keep_indices"] for k, v in keep_per_layer.items()}
        state.update({f"layer_{k}.old_to_new": v["old_to_new"] for k, v in keep_per_layer.items()})
        save_file(state, str(self._get_adapter_path()))

    def patch(self, model, **kwargs) -> Any:
        """读取 adapter 后替换每层 MoE block。"""
        if self.adapter_dir is None:
            raise ValueError("patch 需提供 adapter_dir")
        prune_ratio = kwargs.get("prune_ratio")
        if prune_ratio is None:
            raise ValueError("[moei2][patch] 需要在 patch_kwargs 中提供 prune_ratio")
        prune_ratio = float(prune_ratio)
        saved_pr = saved_prune_ratio_from_adapter_dir(self.adapter_dir)
        if saved_pr is not None and abs(prune_ratio - float(saved_pr)) > 1e-5:
            raise ValueError(
                f"[moei2_pruning] eval 的 prune_ratio={prune_ratio} 与 adapter 目录 config.json 中的 "
                f"{saved_pr} 不一致；MoE-I² adapter 与校准搜索绑定，请重新 calib 或勿在 patch_kwargs 中改 prune_ratio。"
            )
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"未找到 adapter: {self.adapter_path}，请先运行 calib()")

        state = load_file(str(self.adapter_path))
        layers = model.model.layers
        moe_indices = [
            i for i, layer in enumerate(layers)
            if hasattr(layer, "mlp") and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
        ]
        logger.info("[moei2][patch] Replacing %d MoE layers", len(moe_indices))
        for decoder_layer_idx in tqdm(moe_indices, desc="Patching layers", unit="layer"):
            key_pre = f"layer_{decoder_layer_idx}"
            if f"{key_pre}.keep_indices" not in state:
                # 允许仅搜索部分层：不在 adapter 中的层保持不变
                continue
            block = layers[decoder_layer_idx].mlp
            keep_indices = state[f"{key_pre}.keep_indices"].to(block.gate.weight.device)
            old_to_new = state[f"{key_pre}.old_to_new"].to(block.gate.weight.device)
            layers[decoder_layer_idx].mlp = PrunedQwen3MoeSparseMoeBlock(block, keep_indices, old_to_new)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
