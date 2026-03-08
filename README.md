# MoE-Compressors

Implementations of training-free MoE pruning/merging algorithms for LLMs with Mixture-of-Experts architecture.

## 结构

```
MoE-Compressors/
├── MoECompressor.py          # 抽象基类（calib / patch / eval）
├── run.py                    # 统一入口，两种模式：calib | eval
├── methods/
│   ├── frequency_pruning/    # 方法：基于激活频率的专家剪枝
│   │   └── model_qwen3_moe.py
│   └── ...
├── examples/
│   └── run.sh                # 示例脚本：calib 单卡，eval 多卡（accelerate）
└── requirements.txt
```

## 两种模式

| 模式 | 说明 | 运行方式 |
|------|------|----------|
| **calib** | 单卡校准，在校准集上计算统计量，保存 `adapter.safetensors` | `python run.py ...` |
| **eval** | 多卡评测。`adapter_dir` 非空则先 patch 再评测剪枝模型；空则评测原模型 | `accelerate launch python run.py ...` |

## 快速开始

### 1. 校准（单卡）

```bash
python run.py frequency_pruning calib --model Qwen/Qwen3-MoE-15B-A2B --adapter_dir ./outputs/adapter
```

### 2. 评测（多卡，建议 accelerate launch）

```bash
# 评测剪枝模型（需先完成 calib）
accelerate launch run.py frequency_pruning eval --model Qwen/Qwen3-MoE-15B-A2B --adapter_dir ./outputs/adapter

# 评测原模型（不传 adapter_dir）
accelerate launch run.py frequency_pruning eval --model Qwen/Qwen3-MoE-15B-A2B
```

### 3. 使用示例脚本

修改 `examples/run.sh` 顶部参数后：

```bash
bash examples/run.sh calib           # 单卡校准
bash examples/run.sh eval            # 多卡评测剪枝模型（ADAPTER_DIR 非空）
EVAL_RAW=1 bash examples/run.sh eval # 强制多卡评测原模型
```

## 依赖

```bash
pip install -r requirements.txt
```

评测需安装 accelerate：`pip install accelerate`
