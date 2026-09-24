from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefetchConfig:
    mode: str = "same_token"
    distance: int = 1
    head: str = "mlp"
    hidden_dim: int = 512
    temperature: float = 0.1
    loss: str = "score_kl"
    targets: list[int] | None = None
    ks: list[int] = field(default_factory=lambda: [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384])
    excluded_token_ids: list[int] = field(default_factory=list)  # Teacher-forcing supervision only.
    trace_limit: int = 0
    prerouter_enabled: bool = False  # Decode / teacher-forced response uses predicted experts.


def layer_pairs(num_moe, mode="same_token", distance=1, targets=None):
    if mode not in {"same_token", "previous_token"}:
        raise ValueError(f"Unknown prediction mode: {mode}")
    minimum = 0 if mode == "previous_token" else 1
    if distance < minimum:
        raise ValueError(f"{mode} requires distance >= {minimum}")
    chosen = list(range(distance, num_moe)) if targets is None else list(targets)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("Specify at least one unique target MoE ordinal")
    if any(t < distance or t >= num_moe for t in chosen):
        raise ValueError("Each target must satisfy distance <= target < num_moe")
    return [(t - distance, t) for t in chosen]
