#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python -m prefetch_qwen3_5_moe.test \
    --config prefetch_qwen3_5_moe/configs/train.yaml \
    --checkpoint prefetch_qwen3_5_moe/outputs/v1/gate_prefetch.safetensors \
    --max-new-tokens 128

# logits 融合：扫描 prerouter 的权重 lambda
CUDA_VISIBLE_DEVICES=7 python -m prefetch_qwen3_5_moe.test \
    --config prefetch_qwen3_5_moe/configs/train.yaml \
    --checkpoint prefetch_qwen3_5_moe/outputs/v1/gate_prefetch.safetensors \
    --max-new-tokens 128 \
    --fusion-mode logits \
    --fusion-lambdas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 \
    --output prefetch_qwen3_5_moe/outputs/v1/metrics/fusion-logits.json

# probability 融合：温度为 1，扫描相同的 lambda
CUDA_VISIBLE_DEVICES=7 python -m prefetch_qwen3_5_moe.test \
    --config prefetch_qwen3_5_moe/configs/train.yaml \
    --checkpoint prefetch_qwen3_5_moe/outputs/v1/gate_prefetch.safetensors \
    --max-new-tokens 128 \
    --fusion-mode prob \
    --fusion-temperature 1 \
    --fusion-lambdas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 \
    --output prefetch_qwen3_5_moe/outputs/v1/metrics/fusion-prob.json
