"""Qwen MoE router prefetch 的最小训练与推理示例。"""

from .patch import load_prefetch, patch, save_prefetch
from .router_cache import RouterCache, RouterResult

__all__ = ["RouterCache", "RouterResult", "patch", "save_prefetch", "load_prefetch"]
