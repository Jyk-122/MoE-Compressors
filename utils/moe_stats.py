from __future__ import annotations

from typing import Any

import torch


class MoEStatsCollector:
    """Collect runtime MoE routing stats for eval-time reports (pruning acceleration / skipping)."""

    def __init__(self, num_experts: int):
        self.num_experts = int(num_experts)
        self._layers: dict[int, dict[str, Any]] = {}

    def _ensure_layer(self, layer_idx: int):
        if layer_idx not in self._layers:
            self._layers[layer_idx] = {
                "expert_activation_count": torch.zeros(self.num_experts, dtype=torch.long),
                "total_tokens": 0,
                "total_selected_before": 0,
                "total_selected_after": 0,
            }
        return self._layers[layer_idx]

    def update(
        self,
        layer_idx: int,
        selected_indices: torch.LongTensor,
        default_top_k: int,
    ) -> None:
        layer = self._ensure_layer(layer_idx)
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

    def summary(self) -> dict[str, Any]:
        layers = {}
        global_before = 0
        global_after = 0
        for layer_idx, info in self._layers.items():
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
            "enabled": bool(self._layers),
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


def build_router_prob_hist(router_probs: torch.Tensor, bins: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (hist, cdf) on [0,1] for flattened router probabilities."""
    probs = router_probs.detach().float().reshape(-1).cpu()
    hist = torch.histc(probs, bins=bins, min=0.0, max=1.0)
    cdf = torch.cumsum(hist / hist.sum().clamp_min(1.0), dim=0)
    return hist, cdf
