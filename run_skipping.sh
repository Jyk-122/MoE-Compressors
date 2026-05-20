#!/bin/bash
# MoE-Compressors（skipping）脚本入口 -> run.py
#
# 用法:
#   bash run_skipping.sh calib   # 单卡校准（按方法定义，topk_skip/topp_skip 为空流程）
#   bash run_skipping.sh eval    # 多卡评测（accelerate）
#
# 关键环境变量:
#   METHOD           skipping 方法名（topk_skip | topp_skip | sere_skip | modes_skip | lexi_skip | reap_skipping | replace_graph_skip | sgc_skip | os_skip | os_lexi_skip | alloc_skip | ot_scalar_skip | ot_vector_skip，必填）
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
#
#   Alloc-MoE（alloc_skip）：预算感知的层级别 + token级别专家激活分配（基于论文 "Budget-aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference"）。
#     # Stage1：层敏感度画像（只需运行一次）
#     METHOD=alloc_skip bash run_skipping.sh calib
#     # Stage2：使用 compute_reduction 或 target_budget 进行层分配 + 可选 Alloc-T（token级别再分配）
#     #   enable_alloc_t=true：启用token级别自适应再分配
#     #   enable_alloc_t=false：每层每个token固定 k 个专家（仅层分配）
#     EVAL_ADAPTER_DIR=./outputs/.../alloc_skip PATCH_KWARGS='{"compute_reduction":0.5,"enable_alloc_t":true,"k_base":1}' METHOD=alloc_skip bash run_skipping.sh eval
#     # 不同预算可复用同一个 adapter，只改 compute_reduction（0.25/0.4/0.5）各跑一次 eval
#     # 也可直接指定 target_budget 或 layer_k 列表
#
#   OT Scalar Skip（ot_scalar_skip）：仅对 router 概率分布 alpha 做标准最优传输（Sinkhorn）重分配，无向量级补偿。
#     核心：基于专家在真实输入上的输出构建代价矩阵 C，用 Sinkhorn 迭代求解标准 OT 将跳过专家的概率质量重分配给保留专家。
#     参数：
#       layer_topk: 每层保留专家数列表，如 [3,3,3,3,3,3]（与 budget 二选一）
#       budget: 全局平均 k_eff，如 3.0，自动根据代价矩阵分配每层最优 k_eff（与 layer_topk 二选一）
#       ot_reg: OT 熵正则化温度（可选，默认 0.1），越小越接近硬分配，越大越均匀
#       sinkhorn_iters: Sinkhorn 迭代次数（可选，默认 50）
#       max_cost_samples: 计算代价矩阵时采样的 hidden states 数量（可选，默认 128）
#     # calib（layer_topk 模式）：计算代价矩阵 C 并保存 adapter
#     CALIB_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=ot_scalar_skip bash run_skipping.sh calib
#     # calib（budget 模式）：给定全局 budget，自动分配每层 k_eff
#     CALIB_KWARGS='{"budget": 3.0}' METHOD=ot_scalar_skip bash run_skipping.sh calib
#     # eval（layer_topk 模式）：加载 C，推理时 Sinkhorn OT 重分配 router 权重
#     EVAL_ADAPTER_DIR=./outputs/.../ot_scalar_skip PATCH_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=ot_scalar_skip bash run_skipping.sh eval
#     # eval（budget 模式）：patch 时也可重新指定 budget
#     EVAL_ADAPTER_DIR=./outputs/.../ot_scalar_skip PATCH_KWARGS='{"budget": 3.0}' METHOD=ot_scalar_skip bash run_skipping.sh eval
#     # 调 ot_reg / sinkhorn_iters
#     EVAL_ADAPTER_DIR=./outputs/.../ot_scalar_skip PATCH_KWARGS='{"layer_topk": [3,3,3,3,3,3],"ot_reg":0.05,"sinkhorn_iters":100}' METHOD=ot_scalar_skip bash run_skipping.sh eval
#
#   OT Vector Skip（ot_vector_skip）：向量级 OT，tau[i,j,d] = T[i,j] * S[i,d]。
#     T：Sinkhorn OT 标量传输重分配 router 权重（同 ot_scalar_skip）
#     S：逐专家逐维度缩放向量，由闭式解（线性回归）在校准集上计算，补偿向量级特征不匹配
#     参数：
#       layer_topk: 每层保留专家数列表（与 budget 二选一）
#       budget: 全局平均 k_eff（与 layer_topk 二选一）
#       ot_reg: OT 熵正则化温度（可选，默认 0.1）
#       sinkhorn_iters: Sinkhorn 迭代次数（可选，默认 50）
#       max_cost_samples: 计算代价矩阵时采样的 hidden states 数量（可选，默认 128）
#     # calib（layer_topk 模式）：计算 C + 跑校准前向收集 A/B 统计量 → 闭式解求 S
#     CALIB_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=ot_vector_skip bash run_skipping.sh calib
#     # calib（budget 模式）：Pass1 收集 hidden states → 计算 C → 分配 k_eff → Pass2 收集 A/B → 求 S
#     CALIB_KWARGS='{"budget": 3.0}' METHOD=ot_vector_skip bash run_skipping.sh calib
#     # eval：加载 C + S，推理时 Sinkhorn OT 重分配 + S 逐点缩放
#     EVAL_ADAPTER_DIR=./outputs/.../ot_vector_skip PATCH_KWARGS='{"layer_topk": [3,3,3,3,3,3]}' METHOD=ot_vector_skip bash run_skipping.sh eval
#     # 消融对比：ot_scalar_skip（纯 OT，无 S） vs ot_vector_skip（OT + S） vs os_lexi_skip（纯 S，无 OT）
#     # 三者用相同的 layer_topk 或 budget，对比 S 矩阵和 OT 各自的增益

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
  echo "错误: 必须显式设置 METHOD（topk_skip / topp_skip / sere_skip / modes_skip / lexi_skip / reap_skipping / replace_graph_skip / sgc_skip / os_skip / os_lexi_skip / alloc_skip / ot_scalar_skip / ot_vector_skip）"
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
