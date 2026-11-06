#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m skipping_qwen3_5_moe.serve_vllm \
  --model "${MODEL_PATH:-/data1/jiangyikun/models/Qwen3.6-35B-A3B}" \
  --checkpoint "${COMPENSATION_PATH:-skipping_qwen3_5_moe/outputs/compensation.safetensors}" \
  --threshold "${THRESHOLD:-0.5}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  "$@"
