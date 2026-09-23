#!/usr/bin/env bash
set -euo pipefail
# Usage: collect_cache_trace.sh CHECKPOINT OUTPUT [--config CONFIG --max-new-tokens 128]
checkpoint="${1:?Checkpoint directory required}"
output="${2:?Trace directory required}"
shift 2
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m prefetch.evaluation.cache.collect --checkpoint "$checkpoint" --output "$output" "$@"
