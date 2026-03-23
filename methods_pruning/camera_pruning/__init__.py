"""
CAMERA pruning.
Prunes experts with structured micro-expert pruning.
"""

from .model_qwen3_moe import CAMERAPruningQwen3Moe

__all__ = ["CAMERAPruningQwen3Moe"]
