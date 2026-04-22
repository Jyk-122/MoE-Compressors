#!/bin/bash
# MoE-Compressors（skipping）脚本入口 -> run.py
#
# 用法:
#   bash run_skipping.sh calib   # 单卡校准（按方法定义，topk_skip/topp_skip 为空流程）
#   bash run_skipping.sh eval    # 多卡评测（accelerate）
#
# 关键环境变量:
#   METHOD           skipping 方法名（topk_skip | topp_skip | sere_skip | modes_skip | lexi_skip | reap_skipping | replace_graph_skip | sgc_skip | os_skip | os_lexi_skip，必填）
#   MODEL            模型路径或 HF 名称
#   CALIB_KWARGS     calib 参数 JSON，默认 {}
#   PATCH_KWARGS     eval 参数 JSON（topk_skip 默认 {"k":2}；topp_skip / reap_skipping / sgc_skip / os_skip 默认 {"threshold":0.8}；sere_skip 默认 {"select_top_k":2,"threshold":0.3}；modes_skip 默认 {"tau":0.05}；replace_graph_skip 默认 {"coverage_threshold":0.9}；lexi_skip / os_lexi_skip 无默认，lexi_skip 须显式如 {"compute_reduction":0.25}；os_lexi_skip 须显式如 {"layer_topk": [3,4,3,4,...]}）
#   ADAPTER_DIR      calib 输出目录，默认 ./outputs/{MODEL_NAME}/{METHOD}
#   EVAL_ADAPTER_DIR eval 时指定 adapter 目录（lexi_skip 评测 Stage1 产物时必填，须含 adapter.safetensors）
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
#   METHOD=reap_skipping bash run_skipping.sh calib
#   EVAL_ADAPTER_DIR=./outputs/.../reap_skipping METHOD=reap_skipping PATCH_KWARGS='{"threshold":0.8}' bash run_skipping.sh eval
#   METHOD=replace_graph_skip CALIB_KWARGS='{"candidate_top_r":8}' bash run_skipping.sh calib
#   EVAL_ADAPTER_DIR=./outputs/.../replace_graph_skip METHOD=replace_graph_skip PATCH_KWARGS='{"coverage_threshold":0.9}' bash run_skipping.sh eval
#   METHOD=sgc_skip CALIB_KWARGS='{"threshold":0.8,"num_groups":16,"replace_temperature":0.15}' bash run_skipping.sh calib
#   EVAL_ADAPTER_DIR=./outputs/.../sgc_skip METHOD=sgc_skip PATCH_KWARGS='{"threshold":0.8,"replace_threshold":0.1,"score_router_power":0.5}' bash run_skipping.sh eval
#   METHOD=modes_skip CALIB_KWARGS='{"loss_type":"kl"}' bash run_skipping.sh calib
#   METHOD=modes_skip PATCH_KWARGS='{"tau":0.05}' bash run_skipping.sh eval
#   TASKS="piqa gsm8k" METHOD=topk_skip bash run_skipping.sh eval
#   EVAL_LIMIT=500 METHOD=topp_skip bash run_skipping.sh eval
#   EVAL_BATCH_SIZE=4 METHOD=topk_skip bash run_skipping.sh eval
#
#   LExI（lexi_skip）: Stage1 写敏感度矩阵；eval 用 compute_reduction 做 Stage2 并 patch。
#   METHOD=lexi_skip bash run_skipping.sh calib
#   CALIB_KWARGS='{"mc_iters":512,"profile_batch":1,"profile_seq_len":8}' METHOD=lexi_skip bash run_skipping.sh calib
#   EVAL_ADAPTER_DIR=./outputs/.../lexi_skip PATCH_KWARGS='{"compute_reduction":0.25}' METHOD=lexi_skip bash run_skipping.sh eval
#   （25% / 40% / 50%：同一 EVAL_ADAPTER_DIR，仅改 compute_reduction 为 0.25、0.4、0.5 各跑一次 eval）
#
#   Optimal Scaling LExI（os_lexi_skip）：在 lexi_skip 的 per-layer topK 基础上添加 optimal scaling 补偿。
#     CALIB_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=os_lexi_skip bash run_skipping.sh calib
#     EVAL_ADAPTER_DIR=./outputs/.../os_lexi_skip PATCH_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=os_lexi_skip bash run_skipping.sh eval

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
elif [ "$METHOD" = "topp_skip" ] || [ "$METHOD" = "reap_skipping" ] || [ "$METHOD" = "sgc_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"threshold":0.8}'
elif [ "$METHOD" = "sere_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"select_top_k":2,"threshold":0.3}'
elif [ "$METHOD" = "modes_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"tau":0.05}'
elif [ "$METHOD" = "replace_graph_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"coverage_threshold":0.9}'
elif [ "$METHOD" = "os_skip" ]; then
  DEFAULT_PATCH_KWARGS='{"threshold":0.8}'
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
GEN_KWARGS="max_gen_toks=512"
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
  echo "错误: 必须显式设置 METHOD（topk_skip / topp_skip / sere_skip / modes_skip / lexi_skip / reap_skipping / replace_graph_skip / sgc_skip / os_skip / os_lexi_skip）"
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
