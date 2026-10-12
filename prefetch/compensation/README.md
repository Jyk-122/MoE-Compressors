# Prefetch 专家补偿

prerouter 决定执行专家集合，当前真实 router 提供路由分数。OWA/ExFold 调整预取集合上的权重，OptimalScale 在专家输出聚合前应用逐通道 scale。shared experts 保持原生计算。

## 目录结构

每种方法的在线计算和离线校准放在同一子文件夹，共用的数据采集和结果读写留在顶层。

```text
compensation/
├── __init__.py
├── artifacts.py          # 校准结果读写、构建补偿模块
├── calibration.py        # 共用数据循环、专家输出采集
├── cli.py                # 推理与评测共用的命令行参数
├── owa/
│   ├── __init__.py
│   ├── compensate.py     # OWA 权重补偿
│   └── calibrate.py      # alpha1、alpha2 网格搜索及 CLI
├── exfold/
│   ├── __init__.py
│   ├── compensate.py     # ExFold 查表与权重累加
│   └── calibrate.py      # 专家对回归及 CLI
├── optimal_scale/
│   ├── __init__.py
│   ├── compensate.py     # 专家输出通道缩放
│   └── calibrate.py      # 单次遍历采集各层统计量、求解 scale
└── README.md
```

## 配置与调用

命令行可直接指定补偿方法和校准文件，三个入口 `infer_prefetch_demo`、`evaluation.evaluate`、`evaluation.evaluate_base` 使用相同参数。例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.infer_prefetch_demo \
  --checkpoint prefetch/outputs/previous_token_20260925_133256/checkpoint-10000 \
  --prompt '请描述一下这张图。' --image prefetch/datasets/demo.jpg \
  --output prefetch/outputs/demo/metrics.json \
  --execution-mode compensated \
  --method exfold --path prefetch/outputs/exfold.safetensors
```

OWA 可以用 `--method owa --alpha1 1.0 --alpha2 0.9` 直接指定参数，也可以用 `--method owa --path prefetch/outputs/owa.safetensors` 加载校准结果；`--hit-min` 和 `--hit-max` 控制触发区间。

命令行显式指定的字段覆盖运行配置，运行配置未提供补偿设置时使用 checkpoint 中的设置。同一方法保留其他已配置字段；切换方法时，以新方法的命令行参数构建补偿设置。`--execution-mode compensated` 控制是否启用补偿。

也可以在原运行 YAML 的 `prefetch` 下配置：

```yaml
execution_mode: compensated  # native / predicted / compensated
compensation:
  method: owa
  alpha1: 1.0
  alpha2: 1.0
  hit_min: 1
  hit_max: null              # 默认 top_k - 1
```

| 模式 | 专家集合 | 权重 |
| --- | --- | --- |
| native | 当前真实 router top-k | 原生 route 权重 |
| predicted | prerouter top-k | 当前真实 gate 在预测 ID 上的分数，按 block 配置归一化和缩放 |
| compensated | prerouter top-k | OWA/ExFold 补偿权重，或 predicted 权重配合 OptimalScale 通道缩放 |

推理和评测统一通过 `--execution-mode native|predicted|compensated` 选择执行模式。优先级为命令行/Python 显式参数 > 传入配置 > checkpoint 保存值，默认使用 `native`。

支持 `same_token`、`previous_token`、`previous_top`。生成 prefill 使用原生路由，decode 使用所选模式；teacher forcing 仅修改有效 response 输入行，沿用 attention mask、excluded token 和前一 token 对齐规则。路由指标的标签来自当前隐藏状态上的真实 router。

```python
from prefetch.prerouter.patch import patch

state = patch(model, checkpoint="outputs/predictor",
              execution_mode="compensated",
              compensation={"method": "owa", "alpha1": 1.0, "alpha2": 1.0})
```

## 在线核心

- `owa/compensate.py::OWA.forward`：重分配命中专家的权重。
- `exfold/compensate.py::ExFold.forward`：查表，将漏选贡献累加到预取槽位。
- `optimal_scale/compensate.py::OptimalScale.forward`：执行选中专家，逐通道缩放后聚合。
- `../prerouter/patch.py::moe_forward`：选择模式、对齐 token、调用补偿、执行专家。

OWA/ExFold 的权重接口：

```python
weights = compensator(native_ids, native_weights, predicted_ids, baseline_weights)
# 输入和输出均为 [tokens, top_k]，输出槽位顺序与 predicted_ids 相同。
```

`native_weights` 已包含原生 top-k 归一化和 routed scaling，补偿后直接交给 expert backend。新增算法可建立同名子文件夹，在 `compensate.py` 实现此接口、`calibrate.py` 编写校准逻辑，再在 `artifacts.py::build_compensators` 添加分支。OptimalScale 使用输出接口 `compensator(experts, hidden, indices, weights, rows)`，其中 `rows` 指定需要通道缩放的 token 行。当前参数由离线校准获得，prerouter 训练沿用现有损失。

### OWA

记原生集合为 A、预取集合为 P、命中集合 C=A∩P，原生权重为 g，predicted 模式权重为 b，h=Σ(C)g，m=Σ(A\\P)g：

```text
临时权重[j] = alpha1 * g[j] * (1 + m/h),  j 属于 C
            = b[j],                       j 属于 P\\A
最终权重    = alpha2 * 临时权重 * sum(g) / sum(临时权重)
```

全命中时按预取 ID 顺序保留原生权重；零命中或超出触发区间时使用 b。默认区间为 1 到 top_k−1，可用 hit_min/hit_max 调整。alpha1 乘整个命中项，alpha2 决定补偿后的总质量。

用本框架的 b 初始化误取专家权重，并保留模型原生 sigmoid、归一化和缩放语义，是这里对 CommitMoE OWA 的适配约定。阈值及参数需要在目标模型上校准。

### ExFold

每个目标层保存两张有向表 `coefficients[i,j]`、`loss[i,j]`，表示用专家 j 的输出近似专家 i。

```text
weights = 原生命中专家的权重；其他预取槽位从 0 开始
对每个漏选原生专家 i：
    j = argmin(loss[i,j]), j 属于 P
    weights[j] += g[i] * coefficients[i,j]
```

scatter_add 把贡献累加到唯一的预取 ID 槽位。系数可为负，结果直接使用；误取专家仅在被选为替代目标时获得贡献。若某个漏选源专家到整个 P 都没有有效校准记录，其贡献为零。表的覆盖率影响效果，应结合校准报告和 held-out NLL 判断。

### OptimalScale

每层保存 `scale[N,H]`，对预取专家的加权输出进行联合回归。记原生 routed 输出为 Y，predicted 权重为 b：

```text
V[t,e,h] = b[t,e] * E_e(x_t)[h]，e 属于当前预取集合；其余为 0
Y[t,h]   = sum(e 属于原生 Top-K) g[t,e] * E_e(x_t)[h]
output[t,h] = sum(e 属于预取集合) V[t,e,h] * S[e,h]
```

G[h] = V_hᵀ V_h、q[h] = V_hᵀ Y_h 保留专家间的交叉项。默认求解 `(G + ridge*I) S = q + ridge*1`，对应向 predicted 基线 S=1 正则化；`--ridge-center 0` 使用原始零中心 ridge。未观测的专家通道取 S=1，系数允许为负。S=1 对应 predicted 输出。

在线按当前路由中的 active expert 分组，每组通过现有 backend 计算一次未加权输出，乘 S 和路由权重后累加。支持当前 BF16/NF4 接口；此路径使用逐专家调度，延迟需在实际 GPU 后端测量。生成 prefill 使用原生计算，decode 应用 S；teacher forcing 只缩放有效 response 行。全命中行也应用 S，与校准目标一致。

## 独立校准

三个入口共用 `calibration.py` 的数据循环，要求训练好的 predictor checkpoint、原运行配置和过滤后的 JSONL。沿原生轨迹采集 response 输入上的专家输出，冻结 backbone 和 predictor，shared experts 不参与拟合。

采集通过现有 expert backend 的标准接口获得未加权输出，适用于 BF16/NF4 相同调用接口。`--limit` 限制样本数，`--max-tokens` 是全部样本合计每层的 response token 上限，`--chunk-size` 控制暂存激活大小。OWA/ExFold 默认 512 tokens/layer，适合检查流程；OptimalScale 默认最多 32 条样本、65536 tokens/layer。实际 token 数写入校准报告。

### 混合校准数据

使用原 OptimalScale 的五种来源导出一个可直接作为 `--sample-file` 的 JSONL：

| 来源 | 默认数量 | 内容 |
| --- | ---: | --- |
| SlimPajama | 32 | 文本分块，转换为前缀—续写 |
| CodeAlpaca | 32 | 完整问答按上下文长度打包 |
| CogVLM | 64 | 英文标注与对应图片 |
| ChartQA | 32 | 问题、答案与对应图片 |
| 中文 Alpaca | 32 | 完整中文问答按上下文长度打包 |

```bash
python -m prefetch.datasets.prepare_calibration \
  --checkpoint prefetch/outputs/previous_token_20260925_133256/checkpoint-10000 \
  --datasets-root /data1/jiangyikun/datasets \
  --output /data/calib/mixed.jsonl
```

脚本读取 checkpoint 的运行配置，加载相同 processor，使用其中的 `data.max_length` 和 `data.max_image_side`，导出时完成与校准一致的图像处理、长度过滤和 assistant token 对齐校验。也可用 `--config path/to/run.yaml` 指定配置。实际模型权重在校准阶段加载。

默认数据根目录下的相对路径为：

```text
slimpajama-200k/
CodeAlpaca-20k/
CogVLM-SFT-311K/CogVLM-SFT-311K/llava_instruction_single_conversation_formate/
ChartQA/
alpaca-data-gpt4-chinese/
```

CogVLM 的目录可通过 `--cog-root` 单独指定；其中包含 `images/` 和 `labels_en/`。ChartQA 默认使用 `test` split，与原脚本一致，也可指定 `--chart-split train`。

导出结果：

```text
/data/calib/
├── mixed.jsonl
├── mixed.meta.json
└── images/
    └── <sample-id-hash>-0.png
```

每条记录包含 `id/source/group_id/messages/images`，以及 `num_tokens/assistant_text_tokens`。文字消息保持字符串，图文样本的 `images` 保存导出图片的绝对路径，纯文本样本为 `[]`。ChartQA 的 PIL 图片和 CogVLM 的文件图片都转换成 RGB PNG；移动数据目录后需更新图片路径或在目标目录重新导出。

SlimPajama 的原始文本按 token 分块，以最多 64 个 token 为前缀，后续原文作为 assistant 续写；可用 `--text-prefix-tokens` 调整。原始文本由此符合当前 response-input 校准接口。CodeAlpaca 和中文 Alpaca 保留 instruction/input/output 内容并打包完整问答；图文样本分别保留配对关系。

默认导出 192 条有效样本；超长样本会跳过并从同来源继续补足，来源耗尽时报告实际数量。可用 `--counts 32 32 64 32 32` 调整五种来源数量，0 表示停用对应来源。每个来源内部使用固定 seed 打乱，再按比例交错排列，前 32 条同样覆盖五种来源。

校准命令可以直接替换为 `--sample-file /data/calib/mixed.jsonl`。现有 `--limit` 和 `--max-tokens` 继续生效；若要使用全部 192 条、且配置 `max_length=2048`，设置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.compensation.optimal_scale.calibrate \
  --checkpoint prefetch/outputs/previous_token_20260925_133256/checkpoint-10000 \
  --sample-file /data/calib/mixed.jsonl \
  --limit 192 --max-tokens 393216 \
  --output prefetch/outputs/optimal_scale.safetensors
```

### OWA：输出参数或保存 safetensors

```bash
python -m prefetch.compensation.owa.calibrate --checkpoint outputs/predictor --sample-file data/calibration.jsonl --alpha1 0.5 1 2 --alpha2 0.8 1 1.2 --max-tokens 2048 --output outputs/owa.safetensors
```

checkpoint 不含 run_config.json 时，补充 `--config path/to/run.yaml`。`--output` 可省略，stdout JSON 的 `parameters` 直接给出 alpha1、alpha2 和触发区间。

网格搜索最小化 Σ||补偿输出−原生 routed 输出||² / Σ||原生 routed 输出||²，跨目标层汇总选择一组参数。报告包含所有候选误差、未补偿 baseline 误差和部分命中 token 数。触发区间由 `--hit-min/--hit-max` 固定；部分命中样本太少时，参数比较缺乏依据。

### ExFold：保存专家对回归表

```bash
python -m prefetch.compensation.exfold.calibrate --checkpoint outputs/predictor --sample-file data/calibration.jsonl --sampling prefetch --max-tokens 4096 --output outputs/exfold.safetensors
```

- `prefetch`：以 A 中专家为源、P 中专家为目标，在同一隐藏状态上拟合，覆盖实际预测错误。
- `co_routed`：使用 A 内的有向专家对，对应原生共路由校准思路。

对 u=E_i(x)、v=E_j(x)，取 w=||u||₂，进行加权标量回归：

```text
D = sum(w * dot(u,v))
V = sum(w * ||v||²)
U = sum(w * ||u||²)
c = clip(D / (V + ridge), -clip, clip)
loss = (U - 2*c*D + c²*V) / U
```

默认 ridge=1e−8、clip=4，采用 ExFold Qwen 系列的 norm-weighted scalar projection 思路，在线候选约束为逐 token 预取集合。CPU float64 累积统计，保存 FP32 coefficient/loss 和 int64 count。未观测或零能量 pair 保存 c=0、loss=1e30。

artifact 按目标 MoE ordinal 保存，记录层名、专家数、top-k、模型量化配置、LoRA、predictor 路径、采样方法和覆盖统计。加载时校验目标层映射及表尺寸。使用相同 backbone、量化、adapter 和 predictor 进行实验；更换它们后重新校准。

### OptimalScale：单次遍历采集，逐层求解 S

```bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.compensation.optimal_scale.calibrate \
  --checkpoint prefetch/outputs/previous_token_20260925_133256/checkpoint-10000 \
  --sample-file /data/calib/mixed.jsonl \
  --limit 32 --max-tokens 65536 --chunk-size 32 \
  --ridge 1e-4 --ridge-center 1 --channel-chunk-size 64 \
  --output prefetch/outputs/optimal_scale.safetensors
```

模型加载一次，校准数据遍历一次。每个样本沿原生模型轨迹前向时，各目标层分别累积 CPU FP64 G、q，当前分块的专家输出用完即释放。全部采集完成后，逐层 solve 得到 FP32 S，立即释放该层 G、q，并将已完成层的 scale 和元数据写入同一个 safetensors。推理要求所有目标层的 scale 均已完成。

统计时按每个 token 的 K 个预取专家累积 K² 个交叉项。收集期间同时保留全部目标层的 G、q，其 CPU 内存为 `Σ层 8 × H × (N² + N)` 字节，与样本数无关。N=384、H=2560 时，每层约 2.82 GiB，35 个目标层合计约 98.69 GiB，另外需要激活和求解临时内存；程序会在采集前打印统计量内存估算。`--channel-chunk-size` 控制统计和求解时的通道分块及临时内存，G、q 仍覆盖全部通道。最终每层 FP32 S 为 3.75 MiB，文件只含各层 `<target_moe>.scale` 和元数据。

校准日志及 artifact 的 `fit_metrics` 包含每层实际 token 数、观测专家数、predicted 和拟合后的 normalized MSE。它们是校准集上的局部输出误差；使用独立验证集检查 NLL 和任务精度。

```bash
CUDA_VISIBLE_DEVICES=0 python -m prefetch.examples.infer_prefetch_demo \
  --checkpoint prefetch/outputs/previous_token_20260925_133256/checkpoint-10000 \
  --prompt '请描述一下这张图。' --image prefetch/datasets/demo.jpg \
  --execution-mode compensated \
  --method optimal_scale --path prefetch/outputs/optimal_scale.safetensors \
  --output prefetch/outputs/demo/optimal_scale.json
```

评测入口同样接受 `--method optimal_scale --path ...`。已有 predictor checkpoint 可以直接使用，scale 需使用相同基模、量化、adapter 和 predictor 校准。

## 加载与评测

将 YAML 的补偿配置改为：

```yaml
execution_mode: compensated
compensation:
  method: exfold             # owa / exfold / optimal_scale
  path: outputs/exfold.safetensors
```

OWA 同时指定 artifact 和显式参数时，显式参数覆盖校准值。Python 接口同样支持 `compensation={"method": "exfold", "path": "..."}`。native/predicted 执行仅使用各自路由策略，校准表在 compensated 初始化时加载。

```bash
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode native --output outputs/nll-native.json
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode predicted --output outputs/nll-predicted.json
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode compensated --output outputs/nll-compensated.json
```

`prefetch.examples.infer_prefetch_demo` 和 `prefetch.evaluation.evaluate` 同样支持 `--execution-mode`。用独立 held-out 数据比较 NLL/PPL，再比较生成质量；原生轨迹上的局部校准误差不能替代累积误差评测。

当前 cache I/O trace 比较使用 native 执行轨迹。模块负责权重或专家输出补偿，实际异步预取、offloading 调度和延迟收益需要在对应后端测量。完整模型 CUDA、NF4 和优化 expert kernel 路径需在部署环境验证，尤其是 ExFold 的有符号权重。

## 参考

- [CommitMoE](https://ojs.aaai.org/index.php/AAAI/article/view/39454)：Overlap-based Weight Adjustment。
- [ExFold 论文](https://arxiv.org/abs/2608.24938)。
- [ExFold 官方代码](https://github.com/Time-Rune/ExFold-MoE)：参考版本 `f81bf72e57e7fc8126564b20ec5a82e8a128978b` 的 `exfold/qwen3/calibration.py` 和 `kernels.py` 中标量投影及有向误差表语义。
