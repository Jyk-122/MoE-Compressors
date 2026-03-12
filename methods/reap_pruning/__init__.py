"""
REAP (Why Pruning Prevails for One-Shot MoE compression) pruning.
Prunes experts with lowest mean L2 norm of activation outputs with top-k weights on calibration data.
"""

from .model_qwen3_moe import REAPPruningQwen3Moe

__all__ = ["REAPPruningQwen3Moe"]
