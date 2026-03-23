#!/bin/bash
# MoE-Compressors（skipping）脚本入口 -> run.py
#
# 用法:
#   bash run_skipping.sh calib   # 单卡校准（按方法定义，topk_skip 为空流程）
#   bash run_skipping.sh eval    # 多卡评测（accelerate）
#
# 关键环境变量:
#   METHOD           skipping 方法名（当前默认 topk_skip）
#   MODEL            模型路径或 HF 名称
#   CALIB_KWARGS     calib 参数 JSON，默认 {}
#   PATCH_KWARGS     eval 参数 JSON，默认 {"k":2}
#   ADAPTER_DIR      calib 输出目录，默认 ./outputs/{MODEL_NAME}/{METHOD}
#   EVAL_ADAPTER_DIR eval 时显式指定 adapter 目录（可选）
#
# 示例:
#   bash run_skipping.sh calib
#   bash run_skipping.sh eval
#   CALIB_KWARGS='{"foo":1}' bash run_skipping.sh calib
#   PATCH_KWARGS='{"k":1}' bash run_skipping.sh eval
#   METHOD=topk_skip MODEL=Qwen/Qwen3-8B PATCH_KWARGS='{"k":1}' bash run_skipping.sh eval

export HF_ALLOW_CODE_EVAL=1

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
METHOD="${METHOD:-topk_skip}"
DEFAULT_DIR="./outputs"
MODEL_NAME="${MODEL##*/}"
OUTPUT_BASE="${OUTPUT_BASE:-$DEFAULT_DIR/$MODEL_NAME}"
ADAPTER_DIR="${ADAPTER_DIR:-$OUTPUT_BASE/$METHOD}"

DEFAULT_PATCH_KWARGS='{"k":2}'
DEFAULT_CALIB_KWARGS='{}'

CALIBRATION_DATASET="${CALIBRATION_DATASET:-wikitext:wikitext-2-raw-v1}"
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-128}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-2048}"

TASKS=(
  piqa hellaswag winogrande arc_easy arc_challenge mmlu
  gsm8k minerva_math500 mbpp humaneval
)
EVAL_LIMIT=100000
GEN_KWARGS="max_gen_toks=1024"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"
EVAL_OUTPUT_CONTENT="metrics"
DEVICE="cuda"
DTYPE="bfloat16"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CK="${CALIB_KWARGS:-$DEFAULT_CALIB_KWARGS}"
PK="${PATCH_KWARGS:-$DEFAULT_PATCH_KWARGS}"

MODE="${1:-}"
if [ -z "$MODE" ] || { [ "$MODE" != "calib" ] && [ "$MODE" != "eval" ]; }; then
  echo "用法: bash run_skipping.sh calib | eval"
  echo "  关键变量: METHOD, MODEL, CALIB_KWARGS, PATCH_KWARGS, ADAPTER_DIR, EVAL_ADAPTER_DIR"
  echo "  示例1: CALIB_KWARGS='{\"foo\":1}' bash run_skipping.sh calib"
  echo "  示例2: PATCH_KWARGS='{\"k\":1}' bash run_skipping.sh eval"
  echo "  示例3: EVAL_ADAPTER_DIR=./outputs/... bash run_skipping.sh eval"
  exit 1
fi

BASE_ARGS=(--model "$MODEL" --device "$DEVICE" --dtype "$DTYPE")

if [ "$MODE" = "calib" ]; then
  python run.py "$METHOD" calib "${BASE_ARGS[@]}" \
    --adapter_dir "$ADAPTER_DIR" \
    --calib_kwargs "$CK" \
    --calibration_dataset "$CALIBRATION_DATASET" \
    --max_calib_samples "$MAX_CALIB_SAMPLES" \
    --max_context_len "$MAX_CONTEXT_LEN" \
    --batch_size 1

elif [ "$MODE" = "eval" ]; then
  EXTRA_ARGS=(--eval_output_content "$EVAL_OUTPUT_CONTENT")
  [ -n "$EVAL_OUTPUT_PATH" ] && EXTRA_ARGS+=(--eval_output_path "$EVAL_OUTPUT_PATH")
  [ -n "$GEN_KWARGS" ] && EXTRA_ARGS+=(--gen_kwargs "$GEN_KWARGS")
  EVAL_ADAPTER_DIR="${EVAL_ADAPTER_DIR:-}"
  if [ -n "$EVAL_ADAPTER_DIR" ]; then
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --adapter_dir "$EVAL_ADAPTER_DIR" \
      --tasks "${TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "$PK" "${EXTRA_ARGS[@]}"
  else
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --tasks "${TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "$PK" "${EXTRA_ARGS[@]}"
  fi
fi
