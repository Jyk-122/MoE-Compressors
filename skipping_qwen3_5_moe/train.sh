#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m skipping_qwen3_5_moe.train --config skipping_qwen3_5_moe/configs/train.yaml \
  --expected-world-size "$NPROC_PER_NODE" "$@"
