#!/usr/bin/env bash
set -euo pipefail
# Usage: evaluate.sh CHECKPOINT OUTPUT [--mode generation --max-new-tokens 128]
checkpoint="${1:?Checkpoint directory required}"
output="${2:?Output directory required}"
shift 2
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m prefetch.evaluation.evaluate --checkpoint "$checkpoint" --output "$output" "$@"
