"""Configuration validation and loading of the Qwen checkpoint and processor."""
from __future__ import annotations

import math

import torch

from .routing import validate_threshold


def read_config(path):
    import yaml
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_threshold(config["skipping"]["threshold"])
    training = config["training"]
    for key in ("max_steps", "gradient_accumulation", "log_every", "eval_every", "save_every"):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm", "temperature"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if type(training["warmup_steps"]) is not int or training["warmup_steps"] < 0:
        raise ValueError("training.warmup_steps must be a nonnegative integer")
    for key in ("weight_decay", "feature_weight", "kl_weight"):
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if training["feature_weight"] + training["kl_weight"] == 0:
        raise ValueError("At least one distillation loss must have positive weight")
    epsilon = config["calibration"]["epsilon"]
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("calibration.epsilon must be finite and positive")
    if not config["data"]["train"] or config["data"]["max_length"] <= 1:
        raise ValueError("Training requires data sources and max_length > 1")
    return config


def load_model_and_processor(config, device):
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    options = config["model"]
    dtype = options.get("dtype", "auto")
    if dtype != "auto":
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be auto, float32, float16, or bfloat16")
        dtype = getattr(torch, dtype)
    kwargs = {"dtype": dtype, "device_map": {"": device}}
    if options.get("attn_implementation"):
        kwargs["attn_implementation"] = options["attn_implementation"]
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(options["path"], **kwargs).eval()
    processor = AutoProcessor.from_pretrained(options["path"])
    return model, processor
