#!/bin/bash
# 调用项目根目录 run.py，支持 calib / patch / eval 任意组合
# 用法: bash examples/run.sh [action1] [action2] ...
#        bash examples/run.sh eval               # 仅评估原模型（不需 adapter）
#        bash examples/run.sh patch eval         # 加载 adapter 剪枝后评估
#        bash examples/run.sh calib patch eval   # 全流程
#        bash examples/run.sh all                # 等同于 calib patch eval

# ========== 参数配置 ==========
METHOD="frequency_pruning"
MODEL="Qwen/Qwen3-MoE-15B-A2B"
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

# 解析参数：空或 all -> calib patch eval
ACTIONS="$*"
if [ -z "$ACTIONS" ] || [ "$ACTIONS" = "all" ]; then
  ACTIONS="calib patch eval"
fi

# 构建参数
BASE_ARGS="--model $MODEL --device $DEVICE --dtype $DTYPE --prune_ratio $PRUNE_RATIO"
if echo "$ACTIONS" | grep -qE "calib|patch"; then
  BASE_ARGS="$BASE_ARGS --adapter_dir $ADAPTER_DIR"
fi

python run.py $METHOD $ACTIONS $BASE_ARGS \
  --calibration_dataset "$CALIBRATION_DATASET" \
  --max_calib_samples $MAX_CALIB_SAMPLES \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --tasks $TASKS \
  --limit $EVAL_LIMIT
