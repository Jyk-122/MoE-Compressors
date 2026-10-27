#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m skipping_qwen3_5_moe.train --config skipping_qwen3_5_moe/configs/train.yaml "$@"
