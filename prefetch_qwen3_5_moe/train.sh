#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
    -m prefetch_qwen3_5_moe.train \
    --config prefetch_qwen3_5_moe/configs/train.yaml
