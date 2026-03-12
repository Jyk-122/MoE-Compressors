#!/bin/bash
# MoE-Compressors 运行脚本：两种模式
#
# 【模式 1】calib：单卡校准，保存 adapter.safetensors
#   bash run.sh calib
#
# 【模式 2】eval：多卡评测（accelerate launch）
#   - ADAPTER_DIR 非空：加载 adapter 做 patch，评测剪枝模型
#   - ADAPTER_DIR 为空：评测原模型（如 EVAL_RAW=1）
#   bash run.sh eval
#
# 【外部覆盖】支持通过环境变量覆盖 METHOD、PRUNE_RATIO、MODEL，例如：
#   METHOD=ean_pruning PRUNE_RATIO=0.5 bash run.sh eval
#   MODEL=Qwen/Qwen3-8B bash run.sh calib

# ========== 环境变量 ==========
export HF_ALLOW_CODE_EVAL=1

# ========== 参数配置（支持外部覆盖） ==========
METHOD="${METHOD:-frequency_pruning}"
PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# ========== 路径配置 =========================
DEFAULT_DIR="./outputs"
MODEL_NAME="${MODEL##*/}"
OUTPUT_BASE="${DEFAULT_DIR}/${MODEL_NAME}"
ADAPTER_DIR="${OUTPUT_BASE}/${METHOD}/${PRUNE_RATIO}"
CALIBRATION_DATASET="wikitext:wikitext-2-raw-v1"

# ========== 校准参数配置 =====================
MAX_CALIB_SAMPLES=128
MAX_CONTEXT_LEN=2048

# ========== 评测参数配置 =====================
TASKS=(
  piqa hellaswag winogrande arc_easy arc_challenge mmlu 
  gsm8k hendrycks_math500 mbpp humaneval
)
EVAL_LIMIT=100000
GEN_KWARGS="max_gen_toks=1024"
EVAL_OUTPUT_PATH=""

DEVICE="cuda"
DTYPE="bfloat16"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [ -z "$MODE" ] || { [ "$MODE" != "calib" ] && [ "$MODE" != "eval" ]; }; then
  echo "用法: bash run.sh calib | eval"
  echo "  可选环境变量: METHOD, PRUNE_RATIO, MODEL"
  echo "  示例: METHOD=ean_pruning PRUNE_RATIO=0.5 bash run.sh eval"
  echo ""
  echo "  calib: 单卡校准，保存 adapter 到 ADAPTER_DIR"
  echo "  eval:  多卡评测；ADAPTER_DIR 非空则 patch 后评测，否则评测原模型"
  exit 1
fi

BASE_ARGS="--model $MODEL --device $DEVICE --dtype $DTYPE --prune_ratio $PRUNE_RATIO"

if [ "$MODE" = "calib" ]; then
  if [ -z "$ADAPTER_DIR" ]; then
    echo "ERROR: ADAPTER_DIR 不能为空"
    exit 1
  fi
  # 单卡校准
  python run.py $METHOD calib $BASE_ARGS \
    --adapter_dir "$ADAPTER_DIR" \
    --calibration_dataset "$CALIBRATION_DATASET" \
    --max_calib_samples $MAX_CALIB_SAMPLES \
    --max_context_len $MAX_CONTEXT_LEN \
    --batch_size 1

elif [ "$MODE" = "eval" ]; then
  # 多卡评测（accelerate launch）
  EVAL_ARGS="$BASE_ARGS --tasks ${TASKS[*]} --limit $EVAL_LIMIT --output_base $OUTPUT_BASE"
  EXTRA_ARGS=()
  [ -n "$EVAL_OUTPUT_PATH" ] && EXTRA_ARGS+=(--eval_output_path "$EVAL_OUTPUT_PATH")
  [ -n "$GEN_KWARGS" ] && EXTRA_ARGS+=(--gen_kwargs "$GEN_KWARGS")
  # EVAL_RAW=1 或 ADAPTER_DIR 为空 → 不传 adapter_dir，评测原模型
  if [ "${EVAL_RAW:-0}" = "1" ] || [ -z "$ADAPTER_DIR" ]; then
    accelerate launch run.py $METHOD eval $EVAL_ARGS "${EXTRA_ARGS[@]}"
  else
    accelerate launch run.py $METHOD eval $EVAL_ARGS --adapter_dir "$ADAPTER_DIR" "${EXTRA_ARGS[@]}"
  fi
fi
