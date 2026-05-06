#!/bin/bash
# OptimalScale 可解释性可视化脚本
#
# 用法:
#   bash run_visualize.sh
#
# 关键环境变量:
#   MODEL             模型路径或 HF 名称（必填）
#   ADAPTER_BASE_DIR  adapter 基目录，默认 ./outputs/{MODEL_NAME}/os_lexi_skip_visualize
#   CONFIGS           配置列表，空格分隔，每个配置格式为 'name:layer_topk_str'
#                     例如 'prune50:2,2,2,2' 或 'high_prune:[1,1,1,1]'（必填）
#   DATASET           校准数据集，默认 wikitext:wikitext-2-raw-v1
#   MAX_SAMPLES       最大校准样本数，默认 128
#   MAX_CONTEXT_LEN   最大上下文长度，默认 2048
#   OUT_DIR           输出目录，默认 ./visualizations
#   LAYERS            要分析的层索引，空格分隔（可选，默认所有 MoE 层）
#
# 示例:
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CONFIGS="prune50:2,2,2,2 prune60:1,1,1,1" bash run_visualize.sh
#   MODEL=./my_model CONFIGS="config1:[3,4,3,4] config2:[2,3,2,3]" bash run_visualize.sh

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1

MODEL="${MODEL:-}"
if [ -z "$MODEL" ]; then
  echo "错误: 必须设置 MODEL 环境变量"
  echo "示例: MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CONFIGS=\"prune50:2,2,2,2\" bash run_visualize.sh"
  exit 1
fi

CONFIGS="${CONFIGS:-}"
if [ -z "$CONFIGS" ]; then
  echo "错误: 必须设置 CONFIGS 环境变量"
  echo "示例: MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CONFIGS=\"prune50:2,2,2,2\" bash run_visualize.sh"
  exit 1
fi

MODEL_NAME="${MODEL##*/}"
DEFAULT_ADAPTER_BASE="./outputs/${MODEL_NAME}/os_lexi_skip_visualize"
ADAPTER_BASE_DIR="${ADAPTER_BASE_DIR:-$DEFAULT_ADAPTER_BASE}"
DATASET="${DATASET:-wikitext:wikitext-2-raw-v1}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-2048}"
OUT_DIR="${OUT_DIR:-./visualizations}"
LAYERS="${LAYERS:-}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CMD="python methods_skipping/os_lexi_skip/visualize.py"
CMD="$CMD --model \"$MODEL\""
CMD="$CMD --adapter_base_dir \"$ADAPTER_BASE_DIR\""
CMD="$CMD --configs $CONFIGS"
CMD="$CMD --dataset \"$DATASET\""
CMD="$CMD --max_samples $MAX_SAMPLES"
CMD="$CMD --max_context_len $MAX_CONTEXT_LEN"
CMD="$CMD --out_dir \"$OUT_DIR\""

if [ -n "$LAYERS" ]; then
  CMD="$CMD --layers $LAYERS"
fi

echo "=== 运行 OptimalScale 可解释性可视化 ==="
echo "MODEL: $MODEL"
echo "ADAPTER_BASE_DIR: $ADAPTER_BASE_DIR"
echo "CONFIGS: $CONFIGS"
echo "DATASET: $DATASET"
echo "MAX_SAMPLES: $MAX_SAMPLES"
echo "MAX_CONTEXT_LEN: $MAX_CONTEXT_LEN"
echo "OUT_DIR: $OUT_DIR"
if [ -n "$LAYERS" ]; then
  echo "LAYERS: $LAYERS"
fi
echo ""
echo "命令: $CMD"
echo ""

eval $CMD
