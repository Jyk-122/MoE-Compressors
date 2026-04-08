#!/bin/bash
# MoE-Compressors（skipping）脚本入口 -> run.py
#
# 用法:
#   bash run_skipping.sh calib   # 单卡校准（按方法定义，topk_skip/topp_skip 为空流程）
#   bash run_skipping.sh eval    # 多卡评测（accelerate）
#
# 关键环境变量:
#   METHOD           skipping 方法名（topk_skip | topp_skip | sere_skip | modes_skip，必填）
#   MODEL            模型路径或 HF 名称
#   CALIB_KWARGS     calib 参数 JSON，默认 {}
#   PATCH_KWARGS     eval 参数 JSON（topk_skip 默认 {"k":2}；topp_skip 默认 {"threshold":0.8}；sere_skip 默认 {"select_top_k":2,"threshold":0.3}；modes_skip 默认 {"tau":0.05}）
#   ADAPTER_DIR      calib 输出目录，默认 ./outputs/{MODEL_NAME}/{METHOD}
#   EVAL_ADAPTER_DIR eval 时显式指定 adapter 目录（可选）
#   TASKS            eval 时 lm_eval 任务名，空格分隔；未设置则用脚本内默认列表
#   EVAL_LIMIT       eval 时每个任务的 --limit，默认 100000
#   EVAL_BATCH_SIZE  eval 时传给 run.py --eval_batch_size（lm_eval batch），默认 auto；可设为数字如 4
#   EXPERT_TRACE_PATH  若设置则传给 run.py --expert_trace_path（建议单卡 + EVAL_BATCH_SIZE=1）
#
# 示例:
#   METHOD=topk_skip PATCH_KWARGS='{"k":1}' bash run_skipping.sh eval
#   METHOD=topp_skip PATCH_KWARGS='{"threshold":0.8}' bash run_skipping.sh eval
#   METHOD=sere_skip CALIB_KWARGS='{"similarity_method":"frobenius"}' bash run_skipping.sh calib
#   METHOD=sere_skip PATCH_KWARGS='{"select_top_k":2,"threshold":0.3}' bash run_skipping.sh eval
#   METHOD=modes_skip CALIB_KWARGS='{"loss_type":"kl"}' bash run_skipping.sh calib
#   METHOD=modes_skip PATCH_KWARGS='{"tau":0.05}' bash run_skipping.sh eval
#   TASKS="piqa gsm8k" METHOD=topk_skip bash run_skipping.sh eval
#   EVAL_LIMIT=500 METHOD=topp_skip bash run_skipping.sh eval
#   EVAL_BATCH_SIZE=4 METHOD=topk_skip bash run_skipping.sh eval

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
METHOD="${METHOD:-}"
DEFAULT_DIR="./outputs"
MODEL_NAME="${MODEL##*/}"
OUTPUT_BASE="${OUTPUT_BASE:-$DEFAULT_DIR/$MODEL_NAME/$METHOD}"
ADAPTER_DIR="${ADAPTER_DIR:-$OUTPUT_BASE/$METHOD}"

DEFAULT_CALIB_KWARGS='{}'
DEFAULT_PATCH_KWARGS='{}'
if [ "$METHOD" = "topk_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"k":2}'
elif [ "$METHOD" = "topp_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"threshold":0.8}'
elif [ "$METHOD" = "sere_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"select_top_k":2,"threshold":0.3}'
elif [ "$METHOD" = "modes_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"tau":0.05}'
fi

CALIBRATION_DATASET="${CALIBRATION_DATASET:-wikitext:wikitext-2-raw-v1}"
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-128}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-2048}"

DEFAULT_TASKS="piqa hellaswag winogrande arc_easy arc_challenge mmlu gsm8k minerva_math500 mbpp humaneval"
if [ -n "${TASKS:-}" ]; then
  read -ra EVAL_TASKS <<< "$TASKS"
else
  read -ra EVAL_TASKS <<< "$DEFAULT_TASKS"
fi
EVAL_LIMIT="${EVAL_LIMIT:-100000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
GEN_KWARGS="max_gen_toks=1024"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"
EXPERT_TRACE_PATH="${EXPERT_TRACE_PATH:-}"
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
  echo "  关键变量: METHOD, MODEL, CALIB_KWARGS, PATCH_KWARGS, ADAPTER_DIR, EVAL_ADAPTER_DIR, TASKS, EVAL_LIMIT, EVAL_BATCH_SIZE"
  echo "  示例1: METHOD=topk_skip PATCH_KWARGS='{\"k\":1}' bash run_skipping.sh eval"
  echo "  示例2: METHOD=topk_skip PATCH_KWARGS='{\"k\":1}' bash run_skipping.sh eval"
  echo "  示例3: METHOD=topp_skip PATCH_KWARGS='{\"threshold\":0.8}' bash run_skipping.sh eval"
  echo "  示例4: EVAL_ADAPTER_DIR=./outputs/... bash run_skipping.sh eval"
  exit 1
fi

if [ -z "$METHOD" ]; then
  echo "错误: 必须显式设置 METHOD（topk_skip / topp_skip / sere_skip / modes_skip）"
  echo "示例: METHOD=topk_skip bash run_skipping.sh eval"
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
  EXTRA_ARGS=(--eval_output_content "$EVAL_OUTPUT_CONTENT" --eval_batch_size "$EVAL_BATCH_SIZE")
  [ -n "$EVAL_OUTPUT_PATH" ] && EXTRA_ARGS+=(--eval_output_path "$EVAL_OUTPUT_PATH")
  [ -n "$GEN_KWARGS" ] && EXTRA_ARGS+=(--gen_kwargs "$GEN_KWARGS")
  [ -n "$EXPERT_TRACE_PATH" ] && EXTRA_ARGS+=(--expert_trace_path "$EXPERT_TRACE_PATH")
  EVAL_ADAPTER_DIR="${EVAL_ADAPTER_DIR:-}"
  if [ -n "$EVAL_ADAPTER_DIR" ]; then
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --adapter_dir "$EVAL_ADAPTER_DIR" \
      --tasks "${EVAL_TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "$PK" "${EXTRA_ARGS[@]}"
  else
    accelerate launch run.py "$METHOD" eval "${BASE_ARGS[@]}" \
      --tasks "${EVAL_TASKS[@]}" --limit "$EVAL_LIMIT" --output_base "$OUTPUT_BASE" \
      --patch_kwargs "$PK" "${EXTRA_ARGS[@]}"
  fi
fi
