#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Extra arguments go to prefetch.training.train.
config="${1:?Usage: train_ddp.sh CONFIG [--resume CHECKPOINT]}"
shift
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m prefetch.training.train --config "$config" "$@"
