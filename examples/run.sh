#!/bin/bash
# MoE-Compressors 运行脚本：两种模式
#
# 【模式 1】calib：单卡校准，保存 adapter.safetensors
#   bash examples/run.sh calib
#
# 【模式 2】eval：多卡评测（accelerate launch）
#   - ADAPTER_DIR 非空：加载 adapter 做 patch，评测剪枝模型
#   - ADAPTER_DIR 为空：评测原模型（如 EVAL_RAW=1 或 ADAPTER_DIR=""）
#   bash examples/run.sh eval
#   EVAL_RAW=1 bash examples/run.sh eval    # 强制评测原模型

# ========== 参数配置 ==========
METHOD="frequency_pruning"
MODEL="Qwen/Qwen3-MoE-15B-A2B"
# adapter 目录。calib 时保存到此；eval 时若非空则 patch 后评测，空则评测原模型
ADAPTER_DIR="./outputs/frequency_pruning/adapter"
PRUNE_RATIO=0.5
CALIBRATION_DATASET="wikitext:wikitext-2-raw-v1"
MAX_CALIB_SAMPLES=512
OUTPUT_MODEL_PATH="./outputs/frequency_pruning/patched_model"
TASKS="wikitext"
EVAL_LIMIT=0.1
DEVICE="cuda"
DTYPE="float16"
# ==============================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [ -z "$MODE" ] || { [ "$MODE" != "calib" ] && [ "$MODE" != "eval" ]; }; then
  echo "用法: bash examples/run.sh calib | eval"
  echo "  calib: 单卡校准，保存 adapter 到 ADAPTER_DIR"
  echo "  eval:  多卡评测；ADAPTER_DIR 非空则 patch 后评测，否则评测原模型"
  exit 1
fi

BASE_ARGS="--model $MODEL --device $DEVICE --dtype $DTYPE --prune_ratio $PRUNE_RATIO"

if [ "$MODE" = "calib" ]; then
  # 单卡校准
  python run.py $METHOD calib $BASE_ARGS \
    --adapter_dir "$ADAPTER_DIR" \
    --calibration_dataset "$CALIBRATION_DATASET" \
    --max_calib_samples $MAX_CALIB_SAMPLES \
    --batch_size 1

elif [ "$MODE" = "eval" ]; then
  # 多卡评测（accelerate launch）
  EVAL_ARGS="$BASE_ARGS --tasks $TASKS --limit $EVAL_LIMIT"
  if [ -n "$OUTPUT_MODEL_PATH" ]; then
    EVAL_ARGS="$EVAL_ARGS --output_model_path $OUTPUT_MODEL_PATH"
  fi
  # EVAL_RAW=1 或 ADAPTER_DIR 为空 → 不传 adapter_dir，评测原模型
  if [ "${EVAL_RAW:-0}" = "1" ] || [ -z "$ADAPTER_DIR" ]; then
    accelerate launch run.py $METHOD eval $EVAL_ARGS
  else
    accelerate launch run.py $METHOD eval $EVAL_ARGS --adapter_dir "$ADAPTER_DIR"
  fi
fi
