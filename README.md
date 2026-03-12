# MoE-Compressors

Implementations of training-free MoE pruning/merging algorithms for LLMs with Mixture-of-Experts architecture.

## Requirements

```bash
pip install -r requirements.txt
```


## Overview

```
MoE-Compressors/
├── MoECompressor.py          # 抽象基类（calib / patch / eval）
├── run.py                    # 统一入口，两种模式：calib | eval
├── run.sh                    # 运行脚本：calib 单卡，eval 多卡（accelerate），支持 METHOD/PRUNE_RATIO/MODEL 覆盖
├── methods/
│   ├── frequency_pruning/    # 方法：基于激活频率的专家剪枝
│   │   └── model_qwen3_moe.py
│   ├── ean_pruning/         # 方法：基于 Expert Activation Norm 的专家剪枝
│   │   └── model_qwen3_moe.py
│   └── ...
└── requirements.txt
```

MoE-Compressors支持以下两种运行模式：

| 模式 | 说明 | 运行方式 |
|------|------|----------|
| **calib** | 单卡校准，在校准集上计算统计量，保存 `adapter.safetensors` | `python run.py ...` |
| **eval** | 多卡评测。`adapter_dir` 非空则先 patch 再评测剪枝模型；空则评测原模型 | `accelerate launch run.py ...` |

## Quick Start

### 1. 校准（单卡）

```bash
# adapter 保存在 outputs/model_name/method/
python run.py frequency_pruning calib --model Qwen/Qwen3-30B-A3B-Instruct-2507 --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/frequency_pruning/0.5
```

### 2. 评测（多卡，建议 accelerate launch）

```bash
# 评测剪枝模型（需先完成 calib）
accelerate launch run.py frequency_pruning eval --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/frequency_pruning/0.5 --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507

# 评测原模型（不传 adapter_dir）
accelerate launch run.py frequency_pruning eval --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507
```

### 3. Outputs Organization

通过 `DEFAULT_DIR` 和 `MODEL_NAME` 统一管理输出路径（见 `run.sh`）：

```
outputs/{model_name}/
├── results_20250305_123456.json            # 原模型评测结果
└── {method}/
    └── {prune_ratio}/                      # 如 0.5
        ├── adapter.safetensors             # 剪枝 adapter
        ├── config.json                     # calib 参数（自动保存）
        └── results_20250305_123456.json    # 剪枝模型评测结果
```

### 4. Run Script

使用根目录 `run.sh`，支持通过环境变量覆盖 `METHOD`、`PRUNE_RATIO`、`MODEL`：

```bash
# 默认参数（METHOD=frequency_pruning, PRUNE_RATIO=0.5）
bash run.sh calib           # 单卡校准
bash run.sh eval            # 多卡评测剪枝模型（ADAPTER_DIR 非空）

# 覆盖参数
METHOD=ean_pruning PRUNE_RATIO=0.5 bash run.sh calib
METHOD=ean_pruning PRUNE_RATIO=0.5 bash run.sh eval
MODEL=Qwen/Qwen3-8B bash run.sh eval
EVAL_RAW=1 bash run.sh eval  # 强制评测原模型（不 patch）
```

## Algorithms

| 状态 | 方法 | 论文来源 |
|:----:|------|----------|
| ✓ | **frequency_pruning** | 常见 baseline，见 MoE 剪枝文献 |
| ✓ | **ean_pruning** | [Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations](https://arxiv.org/abs/2504.05586) |
| ☐ | **TODO_EXAMPLE** |   |

## Design Notes

### 仅保存 Adapter

本框架**只保存 adapter**（`adapter.safetensors`），不保存剪枝后的完整模型权重。原因如下：

- base 模型仅留 1 份，仅保存与剪枝相关的adapter，随时可以 `patch` 得到等效的剪枝模型，无需预先保存剪枝后权重，节省存储空间
- MoE的专家剪枝只涉及FFN的修改，非侵入式的`patch`更简洁高效，不提供 `modeling_xxx.py`，且剪枝后模型的 `forward` 逻辑通常与 base 模型不一致，无法用 `from_pretrained()` 直接加载
- 使用剪枝模型的正确方式是：**加载 base 模型 → 加载 adapter → 执行 patch()**，三者缺一不可

**推荐使用流程**：calib 后仅分发 adapter 目录，推理/评测时加载 base 和 adapter_dir 再 patch 即可。

### model_type 与扩展新模型

`--model_type` 未指定时，会从 base 模型的 `config.model_type`（HuggingFace 标准字段）自动推断。

**开发者约定**：新增方法实现时，`model_type` 必须与 HuggingFace 的 `config.model_type` **严格保持一致**，不做二次映射。例如 Qwen3-MoE 的 config 中 `model_type="qwen3_moe"`，则注册表中对应 key 也应为 `"qwen3_moe"`。这样 `run.py` 才能通过 `AutoConfig.from_pretrained(model).model_type` 自动推断并正确查找到实现类。

目录组织示例：为 `ean_pruning` 新增 GPT-OSS 适配时，创建 `methods/ean_pruning/model_gpt_oss.py`，在 `run.py` 的 `METHOD_REGISTRY` 中注册 `"ean_pruning": { ..., "gpt_oss": EANPruningGptOss }`（HF 中该模型的 `config.model_type` 为 `"gpt_oss"`）。

