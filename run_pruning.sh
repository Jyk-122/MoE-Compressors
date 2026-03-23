#!/bin/bash
# MoE-Compressors（剪枝）脚本入口 -> run.py
#
# 用法:
#   bash run_pruning.sh calib   # 单卡校准，生成 adapter
#   bash run_pruning.sh eval    # 多卡评测（accelerate）
#
# 关键环境变量:
#   METHOD           frequency_pruning | ean_pruning | reap_pruning | camera_pruning | moei2_pruning
#   MODEL            模型路径或 HF 名称
#   CALIB_KWARGS     calib 参数 JSON，默认 {"prune_ratio":0.5}
#   PATCH_KWARGS     eval 参数 JSON，默认 {"prune_ratio":0.5}
#   ADAPTER_DIR      adapter 目录，默认 ./outputs/{MODEL_NAME}/{METHOD}
#   EVAL_RAW=1       eval 原模型（不加载 adapter，不传 patch_kwargs）
#
# 示例:
#   bash run_pruning.sh calib
#   METHOD=ean_pruning PATCH_KWARGS='{"prune_ratio":0.5}' bash run_pruning.sh eval
#   PATCH_KWARGS='{"prune_ratio":0.3}' bash run_pruning.sh eval
#   METHOD=camera_pruning CALIB_KWARGS='{"prune_ratio":0.5,"alpha":0.95}' bash run_pruning.sh calib
#   EVAL_RAW=1 bash run_pruning.sh eval

export HF_ALLOW_CODE_EVAL=1

METHOD="${METHOD:-frequency_pruning}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

DEFAULT_DIR="./outputs"
MODEL_NAME="${MODEL##*/}"
OUTPUT_BASE="${DEFAULT_DIR}/${MODEL_NAME}"
ADAPTER_DIR="${ADAPTER_DIR:-$OUTPUT_BASE/${METHOD}}"
DEFAULT_CALIB_KWARGS='{"prune_ratio":0.5}'
DEFAULT_PATCH_KWARGS='{"prune_ratio":0.5}'
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
  echo "用法: bash run_pruning.sh calib | eval"
  echo "  关键变量: METHOD, MODEL, CALIB_KWARGS, PATCH_KWARGS, ADAPTER_DIR, EVAL_RAW"
  echo "  示例1: METHOD=ean_pruning PATCH_KWARGS='{\"prune_ratio\":0.5}' bash run_pruning.sh eval"
  echo "  示例2: PATCH_KWARGS='{\"prune_ratio\":0.3}' bash run_pruning.sh eval"
  echo "  示例3: METHOD=camera_pruning CALIB_KWARGS='{\"prune_ratio\":0.5,\"alpha\":0.95}' bash run_pruning.sh calib"
  echo "  示例4: EVAL_RAW=1 bash run_pruning.sh eval"
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
  if [ "${EVAL_RAW:-0}" = "1" ] || [ -z "$ADAPTER_DIR" ]; then
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --tasks "${TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "{}" "${EXTRA_ARGS[@]}"
  else
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --adapter_dir "$ADAPTER_DIR" \
      --tasks "${TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "$PK" "${EXTRA_ARGS[@]}"
  fi
fi
