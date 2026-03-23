"""
MoE-I² Inter-Expert Pruning.
Non-uniform expert pruning via Layer Importance Analysis, Genetic Search, and KT-Reception Field.
"""

from .model_qwen3_moe import MoEI2PruningQwen3Moe

__all__ = ["MoEI2PruningQwen3Moe"]
