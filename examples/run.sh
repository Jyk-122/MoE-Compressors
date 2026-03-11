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

# ========== 环境变量 ==========
export HF_ALLOW_CODE_EVAL=1

# ========== 参数配置 ==========
METHOD="frequency_pruning"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# 输出根目录，与 model 相关的输出统一放在 DEFAULT_DIR/model_name/ 下
# adapter: DEFAULT_DIR/model_name/method/adapter.safetensors
# 原模型 eval 结果: DEFAULT_DIR/model_name/results_xxx.json
# 剪枝模型 eval 结果: DEFAULT_DIR/model_name/method/results_xxx.json
DEFAULT_DIR="./outputs"
MODEL_NAME="${MODEL##*/}"
OUTPUT_BASE="${DEFAULT_DIR}/${MODEL_NAME}"
ADAPTER_DIR="${OUTPUT_BASE}/${METHOD}"
PRUNE_RATIO=0.5
CALIBRATION_DATASET="wikitext:wikitext-2-raw-v1"
MAX_CALIB_SAMPLES=128
MAX_CONTEXT_LEN=2048
TASKS="piqa hellaswag winogrande arc_easy arc_challenge mmlu gsm8k hendrycks_math mbpp humaneval"
EVAL_LIMIT=100000
GEN_KWARGS="max_gen_toks=1024"
EVAL_OUTPUT_PATH=""
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
  EVAL_ARGS="$BASE_ARGS --tasks $TASKS --limit $EVAL_LIMIT --output_base $OUTPUT_BASE"
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
