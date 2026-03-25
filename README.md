# MoE-Compressors

Implementations of training-free MoE pruning/skipping algorithms for LLMs with Mixture-of-Experts architecture.

## Requirements

```bash
pip install -r requirements.txt
```

## Overview

```
MoE-Compressors/
├── MoECompressor.py          # 抽象基类（calib / patch / eval）
├── run.py                    # 统一入口：剪枝 + skipping，calib | eval
├── run_pruning.sh            # 封装 run.py（剪枝默认 CALIB_KWARGS / PATCH_KWARGS）
├── run_skipping.sh           # 封装 run.py（skipping）
├── utils/
│   ├── moe_stats.py          # 路由统计
│   ├── adapter_calib_config.py
│   └── pruning_keep.py       # 按新 prune_ratio 从 adapter 统计量重算 keep（frequency/ean/reap）
├── methods_pruning/
├── methods_skipping/
└── requirements.txt
```

### 运行模式

| 入口 | 说明 |
|------|------|
| **`python run.py <method> calib\|eval ...`** | 单一 CLI：`method` 为 `*_pruning` 或 skipping（`topk_skip` / `topp_skip` / `sere_skip`） |
| **`run_pruning.sh` / `run_skipping.sh`** | 拼好 `--calib_kwargs` / `--patch_kwargs` 后调用 `run.py` |

**方法参数 JSON**

- **`--calib_kwargs`**：run.py 仅做 JSON 解析后原样透传给 `calib(**kwargs)`；字段含义与合法性由各方法 `calib` 自行处理。
- **`--patch_kwargs`**：eval 阶段唯一的方法参数入口；`run.py` **不做方法语义判断**，只把 JSON 原样传给 `patch(**patch_kwargs)`。剪枝可传 `{"prune_ratio": ...}`，skipping 可传 `{"k": ...}`，合法性在各方法 `patch` 内判断。
- **参数优先级（脚本层）**：显式设置 `CALIB_KWARGS` / `PATCH_KWARGS` 时直接使用；未设置时才使用脚本默认拼接逻辑。
- **JSON 书写建议**：优先用单引号包裹 JSON（如 `PATCH_KWARGS='{"k":2}'`），避免 shell 转义问题。

**eval 与 calib 的 `prune_ratio` 不一致时**：`frequency_pruning` / `ean_pruning` / `reap_pruning` 在 **`patch`** 中若发现与 `config.json` 不同且 adapter 含统计量，会**按当前 `prune_ratio` 重算** keep；**`camera_pruning` / `moei2_pruning`** 会在 **`patch`** 中报错，要求与校准一致。

## Quick Start（`run.py`）

### 剪枝（`run_pruning.sh` 封装）

#### 1. 校准（单卡）

```bash
python run.py frequency_pruning calib --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/frequency_pruning/0.5 \
  --calib_kwargs '{"prune_ratio":0.5}'
```

#### 2. 评测（多卡，建议 accelerate launch）

```bash
accelerate launch run.py frequency_pruning eval --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/frequency_pruning/0.5 \
  --patch_kwargs '{"prune_ratio":0.5}' \
  --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507

# 与 calib 不同剪枝率（仅 frequency / ean / reap 等支持从 adapter 统计量重算）
accelerate launch run.py frequency_pruning eval --model ... --adapter_dir ... \
  --patch_kwargs '{"prune_ratio":0.3}'
```

#### 3. 输出目录

通过 `DEFAULT_DIR` 和 `MODEL_NAME` 统一管理输出路径（见 `run_pruning.sh`）：

```
outputs/{model_name}/
├── results_20250305_123456.json
└── {method}/
    └── {prune_ratio}/
        ├── adapter.safetensors
        ├── config.json
        └── results_20250305_123456.json
```

#### 4. 方法专用 calib 超参

写入 **`--calib_kwargs`**（与 `prune_ratio` 同一 JSON），例如：

```bash
python run.py camera_pruning calib --model ... --adapter_dir ... \
  --calib_kwargs '{"prune_ratio":0.5,"alpha":0.95}'
```

`run_pruning.sh` 默认使用：
- `CALIB_KWARGS='{"prune_ratio":0.5}'`
- `PATCH_KWARGS='{"prune_ratio":0.5}'`

#### 5. run_pruning.sh

```bash
# 最常用
bash run_pruning.sh calib
bash run_pruning.sh eval

# 指定方法（并显式给 patch 参数）
METHOD=ean_pruning PATCH_KWARGS='{"prune_ratio":0.5}' bash run_pruning.sh eval

# 覆盖 eval patch 参数（例如与 calib 不同剪枝率）
PATCH_KWARGS='{"prune_ratio":0.3}' bash run_pruning.sh eval

# 只评测原模型（忽略 adapter）
EVAL_RAW=1 bash run_pruning.sh eval

# 方法专用校准超参（直接显式设置 CALIB_KWARGS）
METHOD=camera_pruning CALIB_KWARGS='{"prune_ratio":0.5,"alpha":0.95}' bash run_pruning.sh calib
```

未设置 `CALIB_KWARGS` / `PATCH_KWARGS` 时，脚本分别使用 `{"prune_ratio":0.5}`。
当 `EVAL_RAW=1` 时，脚本会强制走原模型评测路径（不传 `adapter_dir`，并传空 `patch_kwargs`）。

常用环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `METHOD` | `frequency_pruning` | 剪枝方法名 |
| `MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 模型路径/名称 |
| `CALIB_KWARGS` | `{"prune_ratio":0.5}` | 校准 JSON |
| `PATCH_KWARGS` | `{"prune_ratio":0.5}` | 评测 JSON |
| `ADAPTER_DIR` | `./outputs/{model}/{method}` | calib 输出 / eval 输入目录 |
| `EVAL_RAW` | `0` | `1` 时不加载 adapter |
| `TASKS` / `EVAL_LIMIT` | 脚本内默认 | 评测任务与样本上限 |
| `GEN_KWARGS` | `max_gen_toks=1024` | lm_eval 生成参数 |
| `EVAL_OUTPUT_PATH` / `EVAL_OUTPUT_CONTENT` | 空 / `metrics` | 结果输出路径与内容 |

评测结果中的运行时统计在 **`results["runtime_routing"]`**（含 `patch_kwargs` 与 collector 摘要）。
其中统计结构按轴重排为：
- `runtime_routing["global"]["all|prefill|decode"]`
- `runtime_routing["layers"]["all|prefill|decode"]`
并且仅统计 `attention_mask=1` 的真实 token（过滤 padding 位置）。

### Skipping（`run.py` + `run_skipping.sh`）

与剪枝相同，框架层都是 **`calib` + `eval`**；skipping 一般**不改 checkpoint 形状**。`topk_skip` / `topp_skip` 的 `calib()` 为空，但仍可走 **`run.py … calib`** 写 `config.json`。`sere_skip` 的 `calib()` 会计算并保存每层专家相似度矩阵到 adapter。

#### 1. 校准（单卡）

```bash
python run.py topk_skip calib \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/topk_skip \
  --calib_kwargs '{}'
```

#### 2. 评测（多卡，建议 accelerate launch）

```bash
accelerate launch run.py topk_skip eval \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --patch_kwargs '{"k":2}' \
  --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507

# top-p skipping：保留累计 router prob 首次 >= threshold 的最小专家集合
accelerate launch run.py topp_skip eval \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --patch_kwargs '{"threshold":0.8}' \
  --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507

# SERE skipping：secondary expert 按相似度重路由到 primary expert
python run.py sere_skip calib \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/sere_skip \
  --calib_kwargs '{"similarity_method":"frobenius"}'

accelerate launch run.py sere_skip eval \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --adapter_dir ./outputs/Qwen3-30B-A3B-Instruct-2507/sere_skip \
  --patch_kwargs '{"select_top_k":2,"threshold":0.3}' \
  --output_base ./outputs/Qwen3-30B-A3B-Instruct-2507
```

#### 3. 输出目录

与剪枝类似；可用 `--eval_output_path` 指定路径。

#### 4. `calib_kwargs` / `patch_kwargs`

**路由/跳过相关参数只来自 JSON**：
- `topk_skip`：`patch_kwargs={"k":...}`，要求 `1 <= k <= num_experts_per_tok`。
- `topp_skip`：`patch_kwargs={"threshold":...}`，要求 `0 < threshold <= 1`，并在默认 `top_k` 路由结果中按累计概率动态保留最少专家。
- `sere_skip`：
  - 校准：固定 **每条校准文本单独一次 forward**（`padding=False`，忽略 CLI 的 `--batch_size`，非 1 会 warning）；临时用 `MethodType` **替换各层 `mlp.forward`**，在单次 forward 内先跑全专家写 sim、再按 top-k 用已算输出聚合（避免 hook 双算）；`--max_context_len` 控制截断；每层在每个样本上算相似度矩阵后对样本数 **求平均**。
  - `calib_kwargs={"similarity_method":"frobenius|cosine|cka","kernel":"linear|rbf|polynomial"}`（`kernel` 仅 `cka` 使用）
  - `patch_kwargs={"select_top_k":...,"threshold":...}`，要求 `1 <= select_top_k <= num_experts_per_tok` 且 `0 <= threshold <= 1`。

#### 5. run_skipping.sh

`run_skipping.sh` 要求显式设置 `METHOD`，避免误用默认方法。

```bash
# 最常用
METHOD=topk_skip bash run_skipping.sh calib
METHOD=topk_skip bash run_skipping.sh eval

# 改 k
METHOD=topk_skip MODEL=Qwen/Qwen3-8B PATCH_KWARGS='{"k":1}' bash run_skipping.sh eval

# top-p
METHOD=topp_skip PATCH_KWARGS='{"threshold":0.8}' bash run_skipping.sh eval

# SERE
METHOD=sere_skip CALIB_KWARGS='{"similarity_method":"frobenius"}' bash run_skipping.sh calib
METHOD=sere_skip PATCH_KWARGS='{"select_top_k":2,"threshold":0.3}' bash run_skipping.sh eval

# eval 时显式指定 adapter 目录
METHOD=topk_skip EVAL_ADAPTER_DIR=... bash run_skipping.sh eval
```

常用环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `METHOD` | 无（必填） | skipping 方法名（`topk_skip` / `topp_skip` / `sere_skip`） |
| `MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 模型路径/名称 |
| `CALIB_KWARGS` | `{}` | 校准 JSON（按方法定义） |
| `PATCH_KWARGS` | 按 `METHOD` 自动给默认值 | 评测 JSON（topk: `{"k":2}`；topp: `{"threshold":0.8}`；sere: `{"select_top_k":2,"threshold":0.3}`） |
| `ADAPTER_DIR` | `./outputs/{model}/{method}` | calib 输出目录 |
| `EVAL_ADAPTER_DIR` | 空 | eval 时可覆盖 adapter 目录 |
| `TASKS` / `EVAL_LIMIT` | 脚本内默认 | 评测任务与样本上限 |
| `GEN_KWARGS` | `max_gen_toks=1024` | lm_eval 生成参数 |
| `EVAL_OUTPUT_PATH` / `EVAL_OUTPUT_CONTENT` | 空 / `metrics` | 结果输出路径与内容 |

#### 6. 评测结果

同剪枝，关注 **`results["runtime_routing"]`**。

## Algorithms

### Pruning Methods

| 状态 | 方法 | 论文来源 |
|:----:|------|----------|
| ✓ | **frequency_pruning** | 根据专家被激活的频率剪枝。 |
| ✓ | **ean_pruning** | [Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations](https://arxiv.org/abs/2504.05586) |
| ✓ | **reap_pruning** | [REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](http://arxiv.org/abs/2510.13999) |
| ✓ | **camera_pruning** | [CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](https://arxiv.org/abs/2508.02322) |
| ✓ | **moei2_pruning** | [MoE-I²: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition](https://arxiv.org/abs/2411.01016) |
| ☐ | **TODO** |   |

### Skipping Methods

| 状态 | 方法 | 说明 |
|:----:|------|------|
| ✓ | **topk_skip** | 仅 router logits 的大小判断裁剪专家。 |
| ✓ | **topp_skip** | 在默认 top-k 路由中按累计概率阈值动态保留最少专家。 |
| ✓ | **sere_skip** | [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](https://arxiv.org/abs/2602.07616) |
| ☐ | **TODO** |   |

## Design Notes

### 仅保存 Adapter（剪枝为主）

剪枝方法保存 `adapter.safetensors`；使用方式：**加载 base → 指定 adapter_dir → `patch()`**。skipping（如 `topk_skip` / `topp_skip`）可无权重 adapter，仅 `config.json`（若跑 calib）。

部分剪枝方法会在 adapter 中保存统计字段（如 `layer_{i}.expert_importance`、`router_prob_hist`），用于在 `patch` 阶段按 `patch_kwargs.prune_ratio` 重算 keep。

### model_type 与扩展

`--model_type` 未指定时从 `AutoConfig.model_type` 推断。新增实现时注册表 key 须与 HF `config.model_type` 一致。

- pruning：在 `methods_pruning/<method>/` 实现，并在 **`run.py` 的 `METHOD_REGISTRY`** 注册；建议在 `calib`/`patch` 内统一从 kwargs 读取 `prune_ratio` 等字段。
- skipping：在 `methods_skipping/<method>/` 实现并注册到 **`run.py`**；在 `calib` / `patch` 中直接校验各自 kwargs 字段（如 `k`）。
