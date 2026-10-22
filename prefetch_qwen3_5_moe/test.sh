#!/usr/bin/env bash

python -m prefetch_qwen3_5_moe.test \
    --config prefetch_qwen3_5_moe/configs/train.yaml \
    --checkpoint prefetch_qwen3_5_moe/outputs/gate_prefetch.safetensors \
    --max-new-tokens 128 \
    "$@"
