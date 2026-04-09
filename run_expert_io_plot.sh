#!/bin/bash
# 从多条二进制专家轨迹绘制「内存容量–平均换入次数」曲线。
#
# 环境变量:
#   TRACES       轨迹文件路径，空格分隔（必填），例如 "a.bin b.bin c.bin"
#   LABELS       可选，与 TRACES 一一对应的图例，空格分隔；省略则用各文件主名
#   OUT          输出图片路径（默认 ./outputs/expert_io_curve.png）
#   POLICY       belady | lookahead_lru，默认 belady
#   LOOKAHEAD    lookahead_lru 的前瞻 token 数，默认 64
#   CAP_STEP     容量扫描步长，默认 1
#   MAX_CAP      最大容量（默认扫到 num_experts）
#   NO_PROGRESS=1  关闭 tqdm 进度条
#
# 示例:
#   TRACES="./traces/reap.bin ./traces/topp.bin" bash run_expert_io_plot.sh
#   TRACES="a.bin b.bin" LABELS="reap topp" OUT=out.png bash run_expert_io_plot.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TRACES="${TRACES:-}"
if [ -z "$TRACES" ]; then
  echo "用法: 设置 TRACES 为空格分隔的 .bin 轨迹列表"
  echo "示例: TRACES=\"./a.bin ./b.bin\" bash run_expert_io_plot.sh"
  echo "      TRACES=\"./a.bin ./b.bin\" LABELS=\"reap topp\" OUT=./out.png bash run_expert_io_plot.sh"
  exit 1
fi

OUT="${OUT:-./outputs/expert_io_curve.png}"
POLICY="${POLICY:-belady}"
LOOKAHEAD="${LOOKAHEAD:-64}"
CAP_STEP="${CAP_STEP:-1}"
MAX_CAP_ARG=()
[ -n "${MAX_CAP:-}" ] && MAX_CAP_ARG=(--max_cap "$MAX_CAP")
NO_PROGRESS_ARG=()
[ "${NO_PROGRESS:-0}" = "1" ] && NO_PROGRESS_ARG=(--no-progress)

read -ra TRACE_ARR <<< "$TRACES"
TRACE_ARGS=()
for t in "${TRACE_ARR[@]}"; do
  TRACE_ARGS+=(--trace "$t")
done

LABEL_ARGS=()
if [ -n "${LABELS:-}" ]; then
  read -ra LABEL_ARR <<< "$LABELS"
  if [ "${#LABEL_ARR[@]}" -ne "${#TRACE_ARR[@]}" ]; then
    echo "错误: LABELS 条目数 (${#LABEL_ARR[@]}) 必须与 TRACES (${#TRACE_ARR[@]}) 相同"
    exit 1
  fi
  for l in "${LABEL_ARR[@]}"; do
    LABEL_ARGS+=(--label "$l")
  done
fi

python utils/plot_expert_io_curves.py \
  "${TRACE_ARGS[@]}" \
  "${LABEL_ARGS[@]}" \
  --out "$OUT" \
  --policy "$POLICY" \
  --lookahead "$LOOKAHEAD" \
  --cap_step "$CAP_STEP" \
  "${MAX_CAP_ARG[@]}" \
  "${NO_PROGRESS_ARG[@]}"
