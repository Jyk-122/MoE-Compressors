from __future__ import annotations

from typing import Any

import torch


class MoEStatsCollector:
    """Collect runtime MoE routing stats for eval-time reports (pruning acceleration / skipping)."""
    STAGE_KEYS = ("prefill", "decode")

    def __init__(self, num_experts: int):
        self.num_experts = int(num_experts)
        self._layers: dict[int, dict[str, Any]] = {}
        self._stage_layers: dict[str, dict[int, dict[str, Any]]] = {
            key: {} for key in self.STAGE_KEYS
        }

    def initialize_layers(self, layer_indices: list[int]) -> None:
        for layer_idx in layer_indices:
            self._ensure_layer(self._layers, int(layer_idx))

    def _ensure_layer(self, bucket: dict[int, dict[str, Any]], layer_idx: int):
        if layer_idx not in bucket:
            bucket[layer_idx] = {
                "expert_activation_count": torch.zeros(self.num_experts, dtype=torch.long),
                "total_tokens": 0,
                "total_selected_before": 0,
                "total_selected_after": 0,
            }
        return bucket[layer_idx]

    def _update_bucket(
        self,
        bucket: dict[int, dict[str, Any]],
        *,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        default_top_k: int,
    ) -> None:
        layer = self._ensure_layer(bucket, layer_idx)
        valid = selected_indices[selected_indices >= 0]
        if valid.numel() > 0:
            layer["expert_activation_count"] += torch.bincount(
                valid.cpu(),
                minlength=self.num_experts,
            )

        num_tokens = int(selected_indices.shape[0])
        selected_after = int((selected_indices >= 0).sum().item())
        layer["total_tokens"] += num_tokens
        layer["total_selected_before"] += num_tokens * int(default_top_k)
        layer["total_selected_after"] += selected_after

    def _resolve_stage(self, sequence_length: int | None) -> str | None:
        if sequence_length is None:
            return None
        # 生成时 sequence_length==1 通常表示增量 decode；>1 视作 prefill。
        return "decode" if int(sequence_length) == 1 else "prefill"

    def update(
        self,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        default_top_k: int,
        sequence_length: int | None = None,
    ) -> None:
        self._update_bucket(
            self._layers,
            layer_idx=layer_idx,
            selected_indices=selected_indices,
            default_top_k=default_top_k,
        )
        stage = self._resolve_stage(sequence_length)
        if stage is not None:
            stage_bucket = self._stage_layers[stage]
            self._update_bucket(
                stage_bucket,
                layer_idx=layer_idx,
                selected_indices=selected_indices,
                default_top_k=default_top_k,
            )

    def _summary_from_layers(self, layers_store: dict[int, dict[str, Any]]) -> dict[str, Any]:
        layers = {}
        global_before = 0
        global_after = 0
        for layer_idx, info in layers_store.items():
            before = int(info["total_selected_before"])
            after = int(info["total_selected_after"])
            reduction = 0.0 if before == 0 else 1.0 - (after / before)
            layers[str(layer_idx)] = {
                "total_tokens": int(info["total_tokens"]),
                "total_selected_before": before,
                "total_selected_after": after,
                "activation_reduction_ratio": reduction,
                "expert_activation_count": info["expert_activation_count"].tolist(),
            }
            global_before += before
            global_after += after

        global_reduction = 0.0 if global_before == 0 else 1.0 - (global_after / global_before)
        return {
            "enabled": bool(layers_store),
            "global": {
                "total_selected_before": global_before,
                "total_selected_after": global_after,
                "activation_reduction_ratio": global_reduction,
                "effective_selected_per_token_ratio": 0.0
                if global_before == 0
                else (global_after / global_before),
            },
            "layers": layers,
        }

    def _store_to_tensors(
        self,
        layers_store: dict[int, dict[str, Any]],
        layer_indices: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_layers = len(layer_indices)
        tokens = torch.zeros(num_layers, dtype=torch.long, device=device)
        before = torch.zeros(num_layers, dtype=torch.long, device=device)
        after = torch.zeros(num_layers, dtype=torch.long, device=device)
        activation = torch.zeros((num_layers, self.num_experts), dtype=torch.long, device=device)

        for i, layer_idx in enumerate(layer_indices):
            info = layers_store.get(layer_idx)
            if info is None:
                continue
            tokens[i] = int(info["total_tokens"])
            before[i] = int(info["total_selected_before"])
            after[i] = int(info["total_selected_after"])
            activation[i] = info["expert_activation_count"].to(device=device, dtype=torch.long)
        return tokens, before, after, activation

    def _summary_from_tensors(
        self,
        *,
        layer_indices: list[int],
        tokens: torch.Tensor,
        before: torch.Tensor,
        after: torch.Tensor,
        activation: torch.Tensor,
    ) -> dict[str, Any]:
        layers = {}
        global_before = int(before.sum().item())
        global_after = int(after.sum().item())
        global_tokens = int(tokens.sum().item())

        for i, layer_idx in enumerate(layer_indices):
            layer_before = int(before[i].item())
            layer_after = int(after[i].item())
            reduction = 0.0 if layer_before == 0 else 1.0 - (layer_after / layer_before)
            layers[str(layer_idx)] = {
                "total_tokens": int(tokens[i].item()),
                "total_selected_before": layer_before,
                "total_selected_after": layer_after,
                "activation_reduction_ratio": reduction,
                "expert_activation_count": activation[i].tolist(),
            }

        global_reduction = 0.0 if global_before == 0 else 1.0 - (global_after / global_before)
        return {
            "enabled": bool(global_tokens > 0),
            "global": {
                "total_selected_before": global_before,
                "total_selected_after": global_after,
                "activation_reduction_ratio": global_reduction,
                "effective_selected_per_token_ratio": 0.0
                if global_before == 0
                else (global_after / global_before),
            },
            "layers": layers,
        }

    def distributed_summary(self) -> dict[str, Any]:
        layer_indices = sorted(self._layers.keys())
        if not layer_indices:
            return self.summary()

        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        overall = self._store_to_tensors(self._layers, layer_indices, device)
        stage_tensors = {
            stage: self._store_to_tensors(self._stage_layers[stage], layer_indices, device)
            for stage in self.STAGE_KEYS
        }

        dist = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist = torch.distributed

        if dist is not None:
            for t in overall:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for stage in self.STAGE_KEYS:
                tensors = stage_tensors[stage]
                for t in tensors:
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)

        summary = self._summary_from_tensors(
            layer_indices=layer_indices,
            tokens=overall[0],
            before=overall[1],
            after=overall[2],
            activation=overall[3],
        )
        by_stage = {
            stage: self._summary_from_tensors(
                layer_indices=layer_indices,
                tokens=stage_tensors[stage][0],
                before=stage_tensors[stage][1],
                after=stage_tensors[stage][2],
                activation=stage_tensors[stage][3],
            )
            for stage in self.STAGE_KEYS
        }
        return {
            **summary,
            "by_stage": by_stage,
        }

    def summary(self) -> dict[str, Any]:
        overall = self._summary_from_layers(self._layers)
        by_stage = {
            stage: self._summary_from_layers(self._stage_layers[stage])
            for stage in self.STAGE_KEYS
        }
        return {
            **overall,
            "by_stage": by_stage,
        }


def build_router_prob_hist(router_probs: torch.Tensor, bins: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hist, cdf) on [0,1] for flattened router probabilities."""
    probs = router_probs.detach().float().reshape(-1).cpu()
    hist = torch.histc(probs, bins=bins, min=0.0, max=1.0)
    cdf = torch.cumsum(hist / hist.sum().clamp_min(1.0), dim=0)
    return hist, cdf
