from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SelectionOutput:
    selected_indices: torch.LongTensor
    selected_weights: torch.Tensor
    selected_counts: torch.LongTensor


class AccelerationStrategy:
    """Base class for activation acceleration expert selection."""

    def select(
        self,
        router_probs: torch.Tensor,
        default_top_k: int,
        norm_topk_prob: bool,
        kwargs: dict[str, Any] | None = None,
    ) -> SelectionOutput:
        raise NotImplementedError


class TopKStrategy(AccelerationStrategy):
    def select(
        self,
        router_probs: torch.Tensor,
        default_top_k: int,
        norm_topk_prob: bool,
        kwargs: dict[str, Any] | None = None,
    ) -> SelectionOutput:
        kwargs = kwargs or {}
        k = kwargs.get("k", default_top_k)
        if isinstance(k, float):
            if k <= 0:
                raise ValueError("topK.k 必须 > 0")
            selected_k = math.ceil(default_top_k * k) if k <= 1 else int(k)
        else:
            selected_k = int(k)
        selected_k = max(1, min(default_top_k, selected_k))

        values, indices = torch.topk(router_probs, selected_k, dim=-1)
        padded_indices = torch.full(
            (router_probs.shape[0], default_top_k),
            -1,
            dtype=torch.long,
            device=router_probs.device,
        )
        padded_values = torch.zeros(
            (router_probs.shape[0], default_top_k),
            dtype=router_probs.dtype,
            device=router_probs.device,
        )
        padded_indices[:, :selected_k] = indices
        padded_values[:, :selected_k] = values

        if norm_topk_prob:
            denom = padded_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            padded_values = padded_values / denom

        counts = torch.full(
            (router_probs.shape[0],),
            selected_k,
            dtype=torch.long,
            device=router_probs.device,
        )
        return SelectionOutput(padded_indices, padded_values, counts)


class TopPStrategy(AccelerationStrategy):
    def select(
        self,
        router_probs: torch.Tensor,
        default_top_k: int,
        norm_topk_prob: bool,
        kwargs: dict[str, Any] | None = None,
    ) -> SelectionOutput:
        kwargs = kwargs or {}
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise ValueError("topP.kwargs.threshold 为必填字段")
        threshold = float(threshold)
        if threshold <= 0 or threshold > 1:
            raise ValueError("topP.threshold 取值范围为 (0, 1]")

        sorted_probs, sorted_indices = torch.sort(router_probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        k_per_token = torch.sum(cumsum < threshold, dim=-1) + 1
        k_per_token = k_per_token.clamp(min=1, max=default_top_k)

        top_indices = sorted_indices[:, :default_top_k]
        top_values = sorted_probs[:, :default_top_k]
        pos = torch.arange(default_top_k, device=router_probs.device).unsqueeze(0)
        keep_mask = pos < k_per_token.unsqueeze(1)

        padded_indices = torch.where(
            keep_mask,
            top_indices,
            torch.full_like(top_indices, -1),
        )
        padded_values = torch.where(
            keep_mask,
            top_values,
            torch.zeros_like(top_values),
        )

        if norm_topk_prob:
            denom = padded_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            padded_values = padded_values / denom

        return SelectionOutput(padded_indices, padded_values, k_per_token.to(torch.long))


def build_strategy(accelerate_config: dict[str, Any] | None) -> AccelerationStrategy | None:
    if not accelerate_config:
        return None
    method = str(accelerate_config.get("method", "")).strip()
    if not method:
        raise ValueError("accelerate_config.method 不能为空")
    method_l = method.lower()
    if method_l == "topk":
        return TopKStrategy()
    if method_l == "topp":
        return TopPStrategy()
    raise ValueError(f"暂不支持的加速方法: {method}")

