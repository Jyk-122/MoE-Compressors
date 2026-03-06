# MoE-Compressors

Implementations of training-free MoE pruning/merging algorithms for LLMs with Mixture-of-Experts architecture.

## 结构

```
MoE-Compressors/
├── MoECompressor.py          # 抽象基类（calib / patch / eval）
├── run.py                    # 统一入口，通过 method 参数选择压缩方法
├── methods/
│   ├── frequency_pruning/    # 方法：基于激活频率的专家剪枝
│   │   └── model_qwen3_moe.py
│   └── ...
├── examples/
│   └── run.sh                # 示例脚本：调 run.py，修改顶部参数后运行
└── requirements.txt
```

## 抽象接口

| 接口 | 说明 |
|------|------|
| **calib** | 在校准集上计算专家重要性等统计量，保存到 `adapter.safetensors` |
| **patch** | 读取 adapter，非侵入式修改模型结构/权重/forward，返回可推理的 `ModelForCausalLM` |
| **eval** | 使用 lm_eval 的 HFLM 封装模型，调用 `simple_evaluate` 进行评测 |

## 快速开始

支持三种评估流程：
1. **eval**：直接评估原 MoE 模型（可不传 adapter_dir）
2. **patch eval**：加载已有 adapter，剪枝后评估
3. **calib patch eval**：全流程（校准 → 剪枝 → 评估）

```bash
python run.py frequency_pruning eval --model Qwen/Qwen3-MoE-15B-A2B
python run.py frequency_pruning patch eval --model Qwen/Qwen3-MoE-15B-A2B --adapter_dir ./outputs/adapter
python run.py frequency_pruning calib patch eval --model Qwen/Qwen3-MoE-15B-A2B --adapter_dir ./outputs/adapter
```

或使用示例脚本（修改 `examples/run.sh` 顶部参数后运行）：

```bash
bash examples/run.sh eval              # 仅评估原模型
bash examples/run.sh patch eval        # 剪枝后评估
bash examples/run.sh calib patch eval  # 全流程
bash examples/run.sh all              # 等同于 calib patch eval
```

## 依赖

```bash
pip install -r requirements.txt
```
