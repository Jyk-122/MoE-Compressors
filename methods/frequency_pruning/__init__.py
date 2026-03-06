"""
Frequency-based expert pruning.
Prunes experts with lowest activation frequency on calibration data.
"""

from .model_qwen3_moe import FrequencyPruningQwen3Moe

__all__ = ["FrequencyPruningQwen3Moe"]
