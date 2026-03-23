"""由每层「重要性分数」按 prune_ratio 重算 keep_indices / old_to_new（eval 与 calib 剪枝率不一致时使用）。"""

from __future__ import annotations

import torch


def recompute_keep_indices_from_scores(
    scores: torch.Tensor,
    num_experts: int,
    prune_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        scores: shape (num_experts,)，越大越应保留。
        num_experts: 专家数。
        prune_ratio: [0,1)，剪掉的比例。
    """
    num_keep = max(1, int(num_experts * (1 - float(prune_ratio))))
    scores = scores.to(dtype=torch.float32)
    if scores.numel() != num_experts:
        raise ValueError(f"scores 长度 {scores.numel()} 与 num_experts={num_experts} 不一致")
    _, top_indices = torch.topk(scores, num_keep)
    keep_indices = top_indices.sort().values
    old_to_new = torch.full((num_experts,), -1, dtype=torch.long)
    for new_idx, old_idx in enumerate(keep_indices.tolist()):
        old_to_new[old_idx] = new_idx
    return keep_indices.cpu(), old_to_new.cpu()
