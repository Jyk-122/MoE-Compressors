#!/bin/bash
# dataset_analysis.py 脚本入口
#
# 用法:
#   bash run_datasets.sh analyze   # 阶段1：全量数据集推理，收集专家激活频率
#   bash run_datasets.sh select    # 阶段2：从激活数据中选出最优校准子集
#
# 关键环境变量:
#   MODEL                    模型路径或 HF 名称，默认 Qwen/Qwen3-30B-A3B-Instruct-2507
#   DATASET                  HF 数据集名，默认 DKYoon/SlimPajama-200k
#   MAX_SAMPLES              analyze 时最大分析样本数，默认 20000
#   MAX_CONTEXT_LEN          每个样本最大 token 数，默认 2048
#   OUTPUT_DIR               输出根目录，默认 ./outputs/dataset_analysis
#   DEVICE                   推理设备，默认 auto（等同于 cuda）
#   DTYPE                    模型 dtype，默认 bfloat16
#
#   NUM_SAMPLES              select 时子集大小，默认 256
#   METHOD                   select 时选择算法，默认 cluster_stratified
#                           可选: cluster_stratified | greedy_entropy | greedy_coverage
#   N_CLUSTERS               cluster_stratified 时的聚类数，默认 16
#   RANDOM_STATE             随机种子，默认 42
#   ACTIVATION_FILE          select 时指定 analyze 生成的 .npz 文件
#                            （默认自动推导: $OUTPUT_DIR/activation_data.npz）
#
# 示例:
#   # 快速测试（1000 条样本，验证流程）
#   MAX_SAMPLES=1000 bash run_datasets.sh analyze
#
#   # 全量分析（20000 条，生成激活指纹）
#   bash run_datasets.sh analyze
#
#   # 聚类分层采样（推荐默认，选出 256 条校准样本）
#   bash run_datasets.sh select
#
#   # 对比贪心熵增 baseline
#   METHOD=greedy_entropy bash run_datasets.sh select
#
#   # 对比贪心覆盖 baseline
#   METHOD=greedy_coverage bash run_datasets.sh select
#
#   # 使用不同聚类数
#   N_CLUSTERS=32 NUM_SAMPLES=512 bash run_datasets.sh select
#
#   # 换模型 / 换数据集
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 DATASET=wikitext:wikitext-2-raw-v1 bash run_datasets.sh analyze

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${DATASET:-DKYoon/SlimPajama-200k}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/dataset_analysis}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"

NUM_SAMPLES="${NUM_SAMPLES:-256}"
METHOD="${METHOD:-cluster_stratified}"
N_CLUSTERS="${N_CLUSTERS:-16}"
RANDOM_STATE="${RANDOM_STATE:-42}"
ACTIVATION_FILE="${ACTIVATION_FILE:-$OUTPUT_DIR/activation_data.npz}"

MODE="${1:-}"

if [ -z "$MODE" ]; then
  echo "用法: bash run_datasets.sh <analyze|select>"
  echo ""
  echo "  analyze  - 阶段1：对数据集进行 prefill 推理，收集每层专家激活频率，计算测度并可视化"
  echo "  select   - 阶段2：基于专家激活指纹，从候选池中选出最优校准子集"
  echo ""
  echo "环境变量覆盖示例:"
  echo "  MAX_SAMPLES=1000 bash run_datasets.sh analyze"
  echo "  METHOD=greedy_entropy bash run_datasets.sh select"
  exit 1
fi

echo "============================================"
echo "dataset_analysis.py"
echo "  MODE         = $MODE"
echo "  MODEL        = $MODEL"
echo "  DATASET      = $DATASET"
echo "  OUTPUT_DIR   = $OUTPUT_DIR"
if [ "$MODE" = "analyze" ]; then
  echo "  MAX_SAMPLES  = $MAX_SAMPLES"
  echo "  MAX_CONTEXT  = $MAX_CONTEXT_LEN"
  echo "  DEVICE       = $DEVICE"
  echo "  DTYPE        = $DTYPE"
elif [ "$MODE" = "select" ]; then
  echo "  ACTIVATION   = $ACTIVATION_FILE"
  echo "  NUM_SAMPLES  = $NUM_SAMPLES"
  echo "  METHOD       = $METHOD"
  echo "  N_CLUSTERS   = $N_CLUSTERS"
fi
echo "============================================"

if [ "$MODE" = "analyze" ]; then
  python dataset_analysis.py analyze \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --max_samples "$MAX_SAMPLES" \
    --max_context_len "$MAX_CONTEXT_LEN" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output_dir "$OUTPUT_DIR"

elif [ "$MODE" = "select" ]; then
  EXTRA_ARGS=()
  if [ "$METHOD" = "cluster_stratified" ]; then
    EXTRA_ARGS+=(--n_clusters "$N_CLUSTERS" --random_state "$RANDOM_STATE")
  fi

  python dataset_analysis.py select \
    --activation_file "$ACTIVATION_FILE" \
    --num_samples "$NUM_SAMPLES" \
    --method "$METHOD" \
    "${EXTRA_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}/selected_${METHOD}"

else
  echo "错误: MODE 必须是 analyze 或 select，收到 '$MODE'"
  exit 1
fi