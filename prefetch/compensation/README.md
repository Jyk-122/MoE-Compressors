# Prefetch 专家补偿

prerouter 决定执行专家集合，当前真实 router 提供原生专家 ID 和权重，compensator 返回预取集合上的新权重。在线路径只执行一次最终 expert mixture，shared experts 保持原生计算。

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
| compensated | prerouter top-k | OWA 或 ExFold 的补偿权重 |

推理和评测入口的 `--execution-mode` 覆盖配置。未指定新字段时，`prerouter_enabled` 继续控制 native/predicted。显式新模式参数优先于旧开关。

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
- `../prerouter/patch.py::moe_forward`：选择模式、对齐 token、调用补偿、执行专家。

统一接口：

```python
weights = compensator(native_ids, native_weights, predicted_ids, baseline_weights)
# 输入和输出均为 [tokens, top_k]，输出槽位顺序与 predicted_ids 相同。
```

`native_weights` 已包含原生 top-k 归一化和 routed scaling，补偿后直接交给 expert backend。新增算法可建立同名子文件夹，在 `compensate.py` 实现此接口、`calibrate.py` 编写校准逻辑，再在 `artifacts.py::build_compensators` 添加分支。当前参数由离线校准获得，prerouter 训练沿用现有损失。

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

## 独立校准

两个入口共用 `calibration.py` 的数据循环，要求训练好的 predictor checkpoint、原运行配置和过滤后的 JSONL。沿原生轨迹采集 response 输入上的专家输出，冻结 backbone 和 predictor，shared experts 不参与拟合。

采集通过现有 expert backend 的标准接口获得未加权输出，适用于 BF16/NF4 相同调用接口。`--limit` 限制样本数，`--max-tokens` 限制每层 token 数，`--chunk-size` 控制暂存激活大小。默认 512 tokens/layer 适合检查流程，专家较多时应增大校准集并检查覆盖率。

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

## 加载与评测

将 YAML 的补偿配置改为：

```yaml
execution_mode: compensated
compensation:
  method: exfold             # 或 owa
  path: outputs/exfold.safetensors
```

OWA 同时指定 artifact 和显式参数时，显式参数覆盖校准值。Python 接口同样支持 `compensation={"method": "exfold", "path": "..."}`。native/predicted 执行仅使用各自路由策略，校准表在 compensated 初始化时加载。

```bash
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode native --output outputs/nll-native.json
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode predicted --output outputs/nll-predicted.json
python -m prefetch.evaluation.evaluate_base --config path/to/run.yaml --checkpoint outputs/predictor --sample-file data/heldout.jsonl --execution-mode compensated --output outputs/nll-compensated.json
```

`prefetch.examples.infer_prefetch_demo` 和 `prefetch.evaluation.evaluate` 同样支持 `--execution-mode`。用独立 held-out 数据比较 NLL/PPL，再比较生成质量；原生轨迹上的局部校准误差不能替代累积误差评测。

当前 cache I/O trace 比较使用 native 执行轨迹。模块负责补偿权重，实际异步预取、offloading 调度和延迟收益需要在对应后端测量。完整模型 CUDA、NF4 和优化 expert kernel 路径需在部署环境验证，尤其是 ExFold 的有符号权重。

## 参考

- [CommitMoE](https://ojs.aaai.org/index.php/AAAI/article/view/39454)：Overlap-based Weight Adjustment。
- [ExFold 论文](https://arxiv.org/abs/2608.24938)。
- [ExFold 官方代码](https://github.com/Time-Rune/ExFold-MoE)：参考版本 `f81bf72e57e7fc8126564b20ec5a82e8a128978b` 的 `exfold/qwen3/calibration.py` 和 `kernels.py` 中标量投影及有向误差表语义。
