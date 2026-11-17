# Skip ExFold：逐 token 标量补偿

支持 Qwen3.5 / Qwen3.6 MoE 的 HF 和 vLLM 0.24.0。**Prefill、decode 统一在每个 token 的原生 Top-K 内，按「路由权重 × 校准输出范数」保留 `topk` 个专家。** token 之间不共享专家预算。

这是按本项目实验要求实现的 ExFold 逐 token 变体：选择规则来自论文 Eq. 10 / 20，标量拟合和折叠来自 Eq. 7–9 / 24；decode 同样使用逐 token 规则。论文原版的 decode batch 专家池不在本实现中，因此本方法的 `topk=4` 与论文 `P4+D64` 不同。[ExFold 论文](https://arxiv.org/html/2608.24938v1)

## 原理与参数

校准只收集同一个 token 上共同被 Router 选中的专家对。记 $j$ 为被替代专家，$i$ 为实际执行专家，令 $u_t=x_j(h_t)$、$v_t=x_i(h_t)$、$\omega_t=\|u_t\|_2$，拟合：

$$
c_{ji}=\frac{\sum_t\omega_t\langle u_t,v_t\rangle}
{\sum_t\omega_t\|v_t\|_2^2+\lambda},\qquad
\hat x_j=c_{ji}x_i.
$$

默认 $\lambda=10^{-3}$，系数截断到 $[-4,4]$；用截断后的系数计算相对重构损失。每层保存：

- `scale[j, i]`：用执行专家 $i$ 重构被删专家 $j$ 的标量系数，保留符号。
- `loss[j, i]`：$\sum_t\omega_t\|u_t-c_{ji}v_t\|_2^2/(\sum_t\omega_t\|u_t\|_2^2+10^{-12})$。
- `norm[i]`：专家 $i$ 在被路由选中的校准 token 上的平均输出 L2 范数。
- `pair_counts[j, i]`：共同被路由选中的 token 数，便于检查覆盖情况。

自身系数为 1、损失为 0；未观测到的异专家对系数为 0、损失为 $10^{30}$。**表的行是被删专家，列是执行专家。ExFold 需要单独校准，不能复用 skip_transfer 的校准文件。**

在线先选出 $S=\operatorname{TopK}_{i\in T}(w_i\,\mathrm{norm}_i,k)$，再将每个被删专家 $j$ 映射到 $S$ 内损失最小的专家 $a(j)$，合并路由权重：

$$
\widetilde w_i=w_i+\sum_{j\notin S:\,a(j)=i}w_jc_{ji},\qquad
\hat y=\sum_{i\in S}\widetilde w_i x_i.
$$

默认采用论文 Qwen3.5 附录 C 的设置，并统一应用到 prefill / decode：损失大于 `0.99` 的替代贡献不加入，折叠后将保留权重缩放到原始权重总和。被拒绝的补偿不会增加执行专家数。

| patch 参数 | 默认值 | 含义 |
|---|---|---|
| `calibration_path` | 必填 | 本模型的 ExFold 校准文件 |
| `topk` | 必填 | 每个 token 实际执行的 routed expert 数 |
| `loss_threshold` | `0.99` | 补偿损失上限；`null` 关闭过滤 |
| `renormalize` | `true` | 折叠后恢复原路由权重总和；`false` 使用原始标量折叠权重 |

纯标量折叠消融使用 `"loss_threshold": null, "renormalize": false`。两项在 HF、vLLM 和两个推理阶段均一致。原生 Router 保持 Top-8，`topk=3` 时 prefill / decode 都执行 3 个 routed experts，剪枝比例均为 62.5%；shared expert 按原生路径执行。`topk=8` 直接使用原生路由权重和专家计算。

## 校准

以下命令在仓库根目录执行。把 `CALIB_DATA` 改为自己的 JSONL 路径；数据格式与 `skip_transfer` 相同。为比较算法，可先使用相同的数据、样本数和长度。

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"
CALIB_DATA="/path/to/your/calibration.jsonl"
CALIBRATION="$PWD/transfer_moe/outputs/skip_exfold/calibration.safetensors"
DEVICE="cuda:0"

python -m transfer_moe.skip_exfold.calibrate \
  --model "$MODEL" \
  --data "$CALIB_DATA" \
  --output "$CALIBRATION" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --samples 32 \
  --max-length 2048
```

`--max-length` 默认 4096；示例设为 2048，与之前的 Transfer 实验保持一致。所有非 padding token 参与校准，超长序列由公共数据读取器截断。多卡 HF 校准可设置 `CUDA_VISIBLE_DEVICES=0,1`、`DEVICE=auto`。

校准逐条执行原生路由，只计算被选中的专家输出；临时保存当前层、当前序列的 `[tokens, 原生 K, hidden]` 输出，再累计小型专家对统计表。不会缓存所有层或所有专家在整份数据上的输出。校准完成后恢复原生 expert forward。

## HF 推理

```bash
export TRANSFER_MOE_RECORD_STATS=1

python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_exfold \
  --model-type qwen3_5_moe \
  --patch-kwargs "{\"calibration_path\":\"$CALIBRATION\",\"topk\":3}" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --prompt "请解释混合专家模型如何降低推理计算量。" \
  --max-new-tokens 1024
```

换成 `"topk":4` 即保留 4 个专家。纯标量折叠的参数写法：

```bash
PATCH_KWARGS="{\"calibration_path\":\"$CALIBRATION\",\"topk\":3,\"loss_threshold\":null,\"renormalize\":false}"

python -m transfer_moe.infer \
  --model "$MODEL" --method skip_exfold --model-type qwen3_5_moe \
  --patch-kwargs "$PATCH_KWARGS" \
  --device "$DEVICE" --dtype bfloat16 \
  --prompt "请解释混合专家模型如何降低推理计算量。" --max-new-tokens 1024
```

## vLLM 服务

在安装了仓库指定 vLLM 0.24.0 的环境中运行。示例限制同时执行一个请求，方便做当前的 batch=1 比较。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFER_MOE_RECORD_STATS=0

vllm serve "$MODEL" \
  --served-model-name skip-exfold \
  --dtype bfloat16 \
  --moe-backend triton \
  --worker-cls transfer_moe.skip_exfold.qwen3_5_moe.patch_vllm.ExFoldWorker \
  --additional-config "{\"skip_exfold\":{\"calibration_path\":\"$CALIBRATION\",\"topk\":3}}" \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --max-model-len 32768 \
  --port 8113
```

vLLM 的纯标量消融同样在 `skip_exfold` 内加入 `"loss_threshold":null,"renormalize":false`。评测沿用原有 thinking、生成长度、采样参数与数据集设置。

实现用一个 Triton kernel 完成逐 token 选择、替代映射和权重合并，输出紧凑的 `[tokens, topk]` 路由，交给 vLLM 原生 `fused_experts`。没有逐专家 Python forward、CPU 路由决策或逐通道输出补偿；外层 MoE runner、shared expert 调度和最终 TP all-reduce 保持原生路径。支持未量化的单卡 / TP Triton 路径。

统计开关和读取方式见[公共统计说明](../README.md)。`TRANSFER_MOE_RECORD_STATS=1` 时分别报告 prefill / decode；测时延使用 `0`。

## 验证与速度比较

CPU / HF 检查：

```bash
python -m unittest transfer_moe.tests.test_exfold -v
```

目标 GPU 服务器上检查真实 Triton 路由、fused experts、原生 MoE runner 和 CUDA Graph：

```bash
python -m unittest transfer_moe.tests.test_exfold_vllm \
  transfer_moe.tests.test_vllm.NativeRunnerTests.test_exfold_native_runner_shared_expert_tp_and_graph -v
```

TP=2：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m unittest transfer_moe.tests.test_vllm.NativeRunnerTests.test_exfold_native_runner_shared_expert_tp_and_graph -v
```

比较 routed MoE 的原生 Top-8、相同 k 的 Fixed-K、ExFold（包含选择与补偿开销）：

```bash
python -m transfer_moe.tests.benchmark_vllm \
  --method skip_exfold \
  --tokens 1 128 512 \
  --topk 3 4 8 \
  --dtype bfloat16 \
  --output transfer_moe/outputs/skip_exfold/vllm-benchmark.json
```

这里 `tokens=1` 对应单请求 decode，多个 token 可对应同一请求的 prefill。默认两侧都使用 CUDA Graph；加 `--eager` 可比较非图执行。该脚本使用合成专家权重和校准表，测量 routed MoE 而非完整模型，不含 router、attention、shared expert 或通信，不能代替真实模型的端到端时延与质量评测。

测试覆盖标量拟合方向、加权重构损失、负系数与截断、共同路由计数、按权重乘范数选择、损失过滤、全部保留与原生一致，以及带 KV cache 的生成。CPU 环境会明确跳过 CUDA 测试；实际 GPU 正确性与加速收益以服务器结果为准。
