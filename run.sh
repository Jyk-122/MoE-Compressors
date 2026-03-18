#!/bin/bash
# MoE-Compressors：
# 用法: 
#   bash run.sh calib     单卡校准
#   bash run.sh eval      多卡评测
#
# 模式:
#   calib  单卡校准 → 保存 adapter 到 outputs/{MODEL}/{METHOD}/{PRUNE_RATIO}/
#   eval   accelerate 多卡评测；ADAPTER_DIR 非空则 patch 后评测，空则评测原模型
#
# 参数:
#   可通过环境变量覆盖：
#     METHOD                剪枝方法 (frequency_pruning|ean_pruning|reap_pruning|camera_pruning|moei2_pruning)
#     PRUNE_RATIO           剪枝比例 (默认 0.5)
#     MODEL                 模型路径
#     CALIB_EXTRA           方法专用 calib 超参，JSON 格式，如 '{"alpha": 0.95}'（camera_pruning）
#   CALIBRATION_DATASET   校准数据集
#   MAX_CALIB_SAMPLES     校准样本数
#   MAX_CONTEXT_LEN       校准最大上下文长度
#   TASKS                 评测任务列表（见脚本内）
#   EVAL_LIMIT            评测 limit
#   GEN_KWARGS            lm_eval 生成参数 (如 max_gen_toks=1024)
#   EVAL_OUTPUT_PATH      评测结果文件路径（空则自动）
#   EVAL_OUTPUT_CONTENT   results 保存内容：metrics=仅数值，full=完整
#   EVAL_RAW              1=评测原模型（忽略 ADAPTER_DIR）

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
CALIBRATION_DATASET="${CALIBRATION_DATASET:-wikitext:wikitext-2-raw-v1}"

# ========== 校准参数配置 =====================
MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-128}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-2048}"

# ========== 评测参数配置 =====================
TASKS=(
  piqa hellaswag winogrande arc_easy arc_challenge mmlu 
  gsm8k minerva_math500 mbpp humaneval
)
EVAL_LIMIT="${EVAL_LIMIT:-100000}"
GEN_KWARGS="${GEN_KWARGS:-max_gen_toks=1024}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"
EVAL_OUTPUT_CONTENT="${EVAL_OUTPUT_CONTENT:-metrics}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [ -z "$MODE" ] || { [ "$MODE" != "calib" ] && [ "$MODE" != "eval" ]; }; then
  echo "用法: bash run.sh calib | eval"
  echo "  可选环境变量: METHOD, PRUNE_RATIO, MODEL, CALIBRATION_DATASET, MAX_CALIB_SAMPLES, MAX_CONTEXT_LEN, CALIB_EXTRA, TASKS, EVAL_LIMIT, GEN_KWARGS, EVAL_OUTPUT_PATH, EVAL_OUTPUT_CONTENT, EVAL_RAW, DEVICE, DTYPE"
  echo "  示例: METHOD=moei2_pruning PRUNE_RATIO=0.25 CALIB_EXTRA='{\"ga_population\":100,\"ga_iters\":50,\"kt_k\":3,\"kt_t\":3}' bash run.sh calib"
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
  # 单卡校准（CALIB_EXTRA 可选，用于方法专用超参如 camera_pruning 的 alpha）
  CALIB_PY_ARGS=(--adapter_dir "$ADAPTER_DIR" \
    --calibration_dataset "$CALIBRATION_DATASET" \
    --max_calib_samples $MAX_CALIB_SAMPLES \
    --max_context_len $MAX_CONTEXT_LEN \
    --batch_size 1)
  [ -n "${CALIB_EXTRA:-}" ] && CALIB_PY_ARGS+=(--calib_extra "$CALIB_EXTRA")
  python run.py $METHOD calib $BASE_ARGS "${CALIB_PY_ARGS[@]}"

elif [ "$MODE" = "eval" ]; then
  # 多卡评测（accelerate launch）
  EVAL_ARGS="$BASE_ARGS --tasks ${TASKS[*]} --limit $EVAL_LIMIT --output_base $OUTPUT_BASE"
  EXTRA_ARGS=()
  [ -n "$EVAL_OUTPUT_PATH" ] && EXTRA_ARGS+=(--eval_output_path "$EVAL_OUTPUT_PATH")
  [ -n "$GEN_KWARGS" ] && EXTRA_ARGS+=(--gen_kwargs "$GEN_KWARGS")
  EXTRA_ARGS+=(--eval_output_content "$EVAL_OUTPUT_CONTENT")
  # EVAL_RAW=1 或 ADAPTER_DIR 为空 → 不传 adapter_dir，评测原模型
  if [ "${EVAL_RAW:-0}" = "1" ] || [ -z "$ADAPTER_DIR" ]; then
    accelerate launch run.py $METHOD eval $EVAL_ARGS "${EXTRA_ARGS[@]}"
  else
    accelerate launch run.py $METHOD eval $EVAL_ARGS --adapter_dir "$ADAPTER_DIR" "${EXTRA_ARGS[@]}"
  fi
fi
