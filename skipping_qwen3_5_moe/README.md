# Qwen3.6 MoE top-p 剪枝与输出补偿

每个 token 在原 router 的 top-k 专家内进行 top-p 剪枝，保留累计路由概率首次达到 threshold 的最短前缀。保留专家的输出经过逐通道统计补偿，按学习到的系数组合，恢复被跳过专家的路由贡献。

训练固定使用 `threshold: 0.5`，每层训练两部分参数：系数矩阵 `[N,N]` 和补偿算子的逐专家、逐通道均值与尺度 `[N,H]`。基础模型保持冻结。推理通过 `model(..., threshold=...)` 或 `model.generate(..., threshold=...)` 设置阈值。threshold 表示累计概率；实际剪枝比例由路由分布决定，并单独记录。

## 模块与前向

所有专家对统一使用 `[i, j]`：专家 i 提供计算输出，专家 j 是被替代对象。统计量作用于未乘 router 权重的专家输出。

| 模块 | 实现 | 接口与状态 |
| --- | --- | --- |
| 系数矩阵 | `modules.CompensateCoefficients` | 可训练 `logits: [N,N]`；`forward(expert_ids, retained)` 查表并沿保留专家轴做 softmax |
| 补偿算子 T | `modules.MomentCompensation` | 可训练 `mean/log_std: [N,H]`；`forward` 处理对应专家对，`aggregate` 批量重建并聚合 |
| 模型集成 | `model.CompensatedMoE` | 原模型模块、以上两个组件、局部重建和执行数量统计 |

系数矩阵初始化为零，使各保留专家起初等权参与补偿。均值和标准差由校准集初始化；训练参数为 `mean` 与 `log_std`，计算时使用 `std = exp(log_std).clamp_min(epsilon)`，保持尺度为正。每层可训练参数量为 `N² + 2NH`。

原 top-k 集合记为 S，保留集合为 R，缺失集合为 D。原 router 权重 q 在 S 上归一化。每个专家输出 y_i 的补偿为：

```text
T(i,j,y_i) = mu[j] + std[j] * (y_i - mu[i]) / std[i]
alpha[:,j] = softmax(logits[R,j])
y_hat[j]   = sum(i in R, alpha[i,j] * T(i,j,y_i))
routed     = sum(i in R, q[i]*y_i) + sum(j in D, q[j]*y_hat[j])
output     = routed + native_gated_shared_expert
```

原 top-k 的路由权重用于完整聚合。top-1 专家始终保留。`threshold=1` 保留全部原 top-k 专家。

保留输出按 `[token, top-k slot, H]` 排列，系数按 `[token, source slot, target slot]` 排列，保留掩码支持不同 token 的不同保留数量。利用系数和为 1 的性质，补偿通过下面的批量矩阵乘法实现：

```text
Z_R     = (Y_R - mu_R) / std_R
Y_hat_D = mu_D + std_D * bmm(alpha.transpose(1, 2), Z_R)
```

每层对整个 batch 的全部 token 统一路由，使用 top-k 槽位与 mask 批处理。专家 FFN 按实际激活的专家分组，每个专家一次处理分配给它的全部 token；补偿在专家输出收集完成后并行计算。FP32 用于系数矩阵、补偿算子及聚合；输出转换回模型激活精度。

每个 token 从系数矩阵提取自身 top-k 专家对应的子表，并根据保留集合归一化。

## 全专家校准

[calibrate.py](calibrate.py) 包含统计实现与命令行入口。校准集遍历一次，使用原模型的完整上下文前向。每层通过 forward pre-hook 获取当前输入，统计所有有效 token，覆盖 prompt、assistant 和图像 token，排除 padding。

每个 batch 到达 MoE 层时：

1. 各专家处理相同的有效 token，输出保持未乘 router 权重的形式。
2. 累计每个专家逐通道的输出和、平方和，以及有效 token 数。
3. 原 MoE 按原生路由继续前向，为后续层提供输入。

遍历结束后计算 `mean = sum / count`、`variance = square_sum / count - mean²`，以及 `std = sqrt(max(variance, 0)).clamp_min(epsilon)`。累计量在 CPU 上使用 FP64，最终统计量保存为 FP32。

校准集条数由 `data.calibration_limit` 控制，设为 `null` 时使用配置的数据源提供的全部校准记录。

## 训练目标

`losses.CompensationTrainingModel` 共享一份冻结主模型权重，执行原模型 teacher 前向和补偿 student 前向。

局部重建使用当前 student 层输入 h。保留专家输出用于 student；缺失专家在相同 h 上额外执行一次停止梯度的参考计算。将两部分拼接后，按原 router 权重构造原 routed MoE 目标：

```text
target_l = stop_gradient(sum(j in native_top_k, q[j] * E_j(h_student_l)))
feature_loss = mean(l, valid_token, channel, (routed_student_l - target_l)²)
```

特征损失使用普通 MSE，对各层、有效输入 token 和 hidden 通道求平均。最终 LM KL 使用原模型 teacher 分布作为目标，按下一 token 对齐，覆盖数据中 assistant 标签对应的预测位置：

```text
kl_loss = T² * mean(valid_prediction, KL(softmax(z_teacher/T) || softmax(z_student/T)))
loss = feature_weight * feature_loss + kl_weight * kl_loss
```

teacher 通过 `no_grad` 执行；student 的跨层激活保留计算图。LM head 对所有有效预测位置分别执行一次 teacher 和 student 投影，以 FP32 计算 log-softmax 和 KL，并对预测位置求平均。原生 backbone gradient checkpointing 应保持关闭，因为本实现按一次前向收集局部重建项。

训练日志记录特征损失、KL、实际剪枝比例、平均保留数量及 `coefficient_positions`。后者统计同时存在缺失专家和至少两个保留专家的位置，这些位置可以训练系数矩阵。只保留一个专家时，替代系数恒为 1，补偿算子的均值和尺度仍参与训练；全部保留时，该层补偿参数的直接梯度为零。

`--checkpoint` 可以指定校准产物或已训练的补偿产物作为初始化；每次启动建立新的 optimizer 与学习率调度。

## 数据和运行

在仓库根目录执行。环境需要支持 `Qwen3_5MoeForConditionalGeneration` 的 Transformers、PyTorch、Accelerate（`>=1.2,<2`）、safetensors、PyYAML、datasets 和 Pillow。完整模型使用原生浮点 packed expert 权重。

先修改 [configs/train.yaml](configs/train.yaml) 中的模型路径与数据路径。[data.py](data.py) 包含 JSONL 加载、多源混合、图文处理、assistant 标签对齐和设备搬运：

```json
{"id":"example-1","messages":[{"role":"user","content":"解释 MoE。"},{"role":"assistant","content":"MoE 通过路由选择专家参与计算。"}],"images":[]}
```

每卡 microbatch 为一个完整样本，用梯度累积扩大有效 batch。`data.train` 支持多源及 weight；可单独配置 `data.calibration`，格式与 `data.train` 相同。默认从训练源采样校准数据。`data.validation` 按名称配置独立验证集。

执行校准：

```bash
python -m skipping_qwen3_5_moe.calibrate \
  --config skipping_qwen3_5_moe/configs/train.yaml --device cuda:0
```

训练系数矩阵和补偿算子：

```bash
python -m skipping_qwen3_5_moe.train \
  --config skipping_qwen3_5_moe/configs/train.yaml
```

使用 Accelerate/DDP 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash skipping_qwen3_5_moe/train.sh
```

DDP 每卡保存完整模型与当前样本激活，并同步系数矩阵、均值和对数标准差的梯度。有效 batch 为 `进程数 × gradient_accumulation`。显存预算需要包含主模型权重和 student 的反向激活。

输出目录包含：

- `calibration.safetensors`：各专家的均值、标准差及样本数。
- `run_config.yaml`、`train.jsonl`：运行配置与训练指标。
- `metrics/step-N.json`：验证结果。
- `checkpoint-N.safetensors`、`compensation.safetensors`：各层训练后的系数矩阵、均值与对数标准差；加载时使用同一个基础模型。

产物格式为 `qwen_moe_compensation_v2`。开始实验前使用本目录的校准入口生成初始化文件。

独立评估：

```bash
python -m skipping_qwen3_5_moe.evaluate \
  --config skipping_qwen3_5_moe/configs/train.yaml \
  --checkpoint skipping_qwen3_5_moe/outputs/compensation.safetensors \
  --threshold 0.5 --output skipping_qwen3_5_moe/outputs/evaluation.json
```

生成：

```bash
python -m skipping_qwen3_5_moe.infer \
  --model /data1/jiangyikun/models/Qwen3.6-35B-A3B \
  --checkpoint skipping_qwen3_5_moe/outputs/compensation.safetensors \
  --threshold 0.5 --prompt "请解释混合专家模型。"
```

Python 调用：

```python
from skipping_qwen3_5_moe import load_compensation

controller = load_compensation(model, "skipping_qwen3_5_moe/outputs/compensation.safetensors")
output_ids = model.generate(**inputs, threshold=0.5, max_new_tokens=64)
print(controller.summary())
```

每次调用的 threshold 在调用结束后恢复。生成过程中的 prefill 与 decode 使用本次调用的同一个阈值。控制器按串行请求保存统计；并发请求应使用各自的模型实例或请求级隔离。

## 验证范围

```bash
python -m unittest discover -s skipping_qwen3_5_moe/tests -t . -v
```

CPU 测试覆盖校准统计量与参数初始化、并行补偿的输出和全部参数梯度、实际专家调用数、同输入局部目标、两项损失各自的跨层梯度、单专家保留时的补偿训练、完整补偿参数保存恢复、梯度累积、双进程 Gloo 与单进程更新一致性，以及原生 Transformers `generate` 的 prefill/decode 一致性。

核心 MoE 测试使用 [tests/reference_moe.py](tests/reference_moe.py) 中的 Qwen MoE 参考类；生成测试将这些 MoE 类接入小型 Llama 因果模型。数据测试覆盖文本和图像样本、assistant 标签及多源加载。完整 Qwen3.6-35B-A3B 的 CUDA 训练、混合注意力缓存与实际延迟需要在目标模型和硬件环境验证。当前专家执行采用 PyTorch 分组调用；推理速度取决于专家 kernel、batch 大小与路由分布。
