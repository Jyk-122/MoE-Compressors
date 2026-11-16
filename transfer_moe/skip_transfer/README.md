# TransferMoE

设置 `TRANSFER_MOE_RECORD_STATS=1` 后，HF / vLLM patch 记录每层和总体剪枝比例；默认关闭。读取、清零及 EvalScope 服务评测脚本见[公共统计说明](../README.md)。

## 算法原理

TransferMoE 在原生 Router 给出的 Top-K 专家集合 $T$ 内，为每个 token 选择实际执行的子集 $S$，用保留专家的输出补偿被跳过专家的贡献。选择同时考虑路由权重和专家之间的可替代性。

**离线校准。** 对每层各专家在同一批校准输入上的输出 $x_i$，统计逐通道均值 $\mu_i$ 和标准差 $\sigma_i$（下限为 $\varepsilon$）。用源专家 $i$ 近似目标专家 $j$ 的仿射变换及其平均平方误差为：

$$
T_{i\to j}(x_i)=\mu_j+\sigma_j\odot\frac{x_i-\mu_i}{\sigma_i},
\qquad
D_{ij}=\mathbb E_{h\sim\mathcal D}\left[\left\|x_j(h)-T_{i\to j}(x_i(h))\right\|_2^2\right].
$$

乘除均按通道进行。$D$ 的行是替代源、列是目标，$D_{ii}=0$，一般不对称；误差对输出通道求和、对校准 token 取平均。每层保存 $\mu$、$\sigma$ 和 $D$，供推理复用。

**在线贪心选择。** Router 输出 Top-K 索引及原生权重 $w_j$ 后，读取对应的 $K\times K$ 代价子矩阵。给定保留集合 $S$，每个未保留目标由代价最小的源替代，优化的代理目标为：

$$
a_S(j)=\arg\min_{i\in S}D_{ij},
\qquad
J(S)=\sum_{j\in T}w_j\min_{i\in S}D_{ij}.
$$

首先选择使 $\sum_{j\in T}w_jD_{ij}$ 最小的单个专家；之后每轮加入使 $J(S)$ 下降最多的未选专家，并更新各目标的最佳替代源。这样得到一条嵌套的保留序列，代理误差随保留数量增加而单调不增，全部保留时为零。决策只依赖路由结果与离线统计，无须先执行被跳过的专家。

- `topk=k`：取贪心序列的前 $k$ 个专家。
- `topp_threshold=τ`：取满足 $J(S)\le\tau$ 的最短非空前缀；$\tau$ 是绝对代理平方误差阈值。

**执行与补偿。** 只计算 $S$ 中专家的真实输出，再按原路由权重聚合：

$$
\hat y=\sum_{i\in S}w_i x_i+
\sum_{j\in T\setminus S}w_jT_{a_S(j)\to j}(x_{a_S(j)}).
$$

目标权重 $w_j$ 乘在整个替代输出上，剪枝后不重新归一化；shared expert 按原生路径计算并相加。补偿可融合到聚合过程，无须显式保存每个被跳过专家的重构输出。

这个目标来自归一化路由权重下的加权平方误差上界，但以离线平均代价 $D$ 替代当前 token 的实际误差后，$J$ 只是代理量，阈值不保证逐 token 的真实误差上界。贪心也不保证每个预算下的全局最优子集。完整推导与示例见[算法设计文档](../../docs/research/MoE_Expert_Substitution_Greedy_Selection.md)。

## 运行脚本

当前 `skip_transfer` 支持 Qwen3.5 / Qwen3.6 MoE 的 HF 和 vLLM 后端，模型类型参数统一使用 `qwen3_5_moe`。Qwen3 MoE 已有模型加载分支，其 `skip_transfer` patch 尚未实现。

以下命令均在 **MoE-Compressors 仓库根目录**执行。先修改 `MODEL` 为你的模型路径，再在同一个终端依次运行。示例路径沿用仓库中的 Qwen3.6 配置。

```bash
python -m pip install -r transfer_moe/requirements.txt

export CUDA_VISIBLE_DEVICES=0
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"
CALIB_DATA="transfer_moe/examples/calibration.example.jsonl"
CALIBRATION="transfer_moe/outputs/skip_transfer/calibration.safetensors"
DEVICE="cuda:0"
```

如需使用多张 GPU，将上面的设备设置改为以下内容，再执行后面的校准、推理命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICE="auto"
```

先生成两条文本校准数据，用于跑通流程。正式校准时，把 `CALIB_DATA` 改为自己的代表性数据文件。

```bash
mkdir -p transfer_moe/examples
cat > transfer_moe/examples/calibration.example.jsonl <<'JSONL'
{"messages":[{"role":"user","content":"请解释混合专家模型如何通过路由器选择专家。"},{"role":"assistant","content":"路由器根据当前 token 的隐藏状态计算各个专家的分数，选择分数较高的若干专家，并对它们的输出加权求和。"}]}
{"messages":[{"role":"user","content":"一个盒子里有 12 个球，其中 5 个是红色的，其余是蓝色的。蓝色球有多少个？"},{"role":"assistant","content":"蓝色球有 12 - 5 = 7 个。"}]}
JSONL
```

数据格式为 JSONL，每行一条记录。`messages` 中的 `content` 使用字符串；图文记录额外提供 `images` 路径列表，例如：

```json
{"messages":[{"role":"user","content":"请描述这张图片。"}],"images":["/absolute/path/to/image.jpg"]}
```

图片路径按运行命令时的工作目录解析，也可以直接使用绝对路径。

运行校准，保存每层的专家均值、标准差和有向替代误差矩阵：

```bash
python -m transfer_moe.skip_transfer.calibrate \
  --model "$MODEL" \
  --data "$CALIB_DATA" \
  --output "$CALIBRATION" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --samples 32 \
  --max-length 2048 \
  --max-image-side 672 \
  --epsilon 1e-5
```

`--samples 32` 按文件顺序取前 32 条记录；不足 32 条时使用全部记录，因此上述示例会使用两条。每条记录的全部非 padding token 都参与校准。`--data` 可以依次传入多个 JSONL 文件，记录数量上限对这些文件合计生效。

`--max-length` 是处理后的 token 数上限，超出时会报错；`--max-image-side` 限制图片长边尺寸。生成的校准文件对应当前模型，可以复用于不同的 `topk` 和 `topp_threshold` 设置。

固定保留专家数量进行文本推理。下面每个 token 从原生 router 选中的专家里保留 4 个，其余通过补偿重构：

```bash
python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_transfer \
  --model-type qwen3_5_moe \
  --patch-kwargs "{\"calibration_path\": \"$CALIBRATION\", \"topk\": 4}" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --prompt "请解释混合专家模型如何降低推理计算量。" \
  --max-new-tokens 256
```

`topk` 的取值范围为 1 到原生 `num_experts_per_tok`。对于原生 Top-8 的模型，设置 `topk: 8` 表示全部保留。

按误差阈值动态选择每个 token 的保留数量：

```bash
python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_transfer \
  --model-type qwen3_5_moe \
  --patch-kwargs "{\"calibration_path\": \"$CALIBRATION\", \"topp_threshold\": 0.1}" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --prompt "一个盒子里有 12 个球，其中 5 个是红色的，蓝色球有多少个？" \
  --max-new-tokens 256
```

`topp_threshold` 是校准误差矩阵定义的**绝对代理误差阈值**，算法选择贪心序列中满足阈值的最小保留数量。阈值越大，保留数量越少或不变。这里的 `0.1` 仅展示调用方式，需要根据校准误差尺度和实际任务效果选择，不能解读为 10% 误差或累计路由概率。

`topk` 和 `topp_threshold` 必须二选一。两者以及 `calibration_path` 都放在 `--patch-kwargs` 中，由 `skip_transfer` 的 patch 接收。

图文推理时，修改 `IMAGE` 为实际图片路径后执行：

```bash
IMAGE="/absolute/path/to/image.jpg"

python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_transfer \
  --model-type qwen3_5_moe \
  --patch-kwargs "{\"calibration_path\": \"$CALIBRATION\", \"topk\": 4}" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --images "$IMAGE" \
  --max-image-side 672 \
  --prompt "请描述图片中的主要内容。" \
  --max-new-tokens 256
```

`--images` 支持多个路径；动态阈值图文推理可以将上面的 `topk` 替换为 `topp_threshold`。面向图文任务校准时，在校准 JSONL 中加入代表性的图文记录。

查看完整命令行参数：

```bash
python -m transfer_moe.skip_transfer.calibrate --help
python -m transfer_moe.infer --help
```

vLLM 使用同一份 HF 校准文件。当前实现针对 **vLLM 0.24.0、CUDA、未量化权重、Triton MoE backend**，支持单卡和 Tensor Parallel；不支持量化、LoRA、EP、EPLB、SP、DP 或 Prefill Context Parallel。vLLM 的 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.24.0/requirements/cuda.txt)与上面的 HF 环境不同，请使用独立环境安装：

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r transfer_moe/requirements-vllm.txt

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"
CALIBRATION="$PWD/transfer_moe/outputs/skip_transfer/calibration.safetensors"
```

启动双卡 TP 服务，固定保留 4 个专家。worker 在加载权重后、profiling 和 CUDA Graph capture 前安装 patch；校准统计量占用的显存也计入 KV cache 预算。

```bash
POLICY="{\"skip_transfer\":{\"calibration_path\":\"$CALIBRATION\",\"topk\":4}}"

vllm serve "$MODEL" \
  --served-model-name transfer-moe \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --moe-backend triton \
  --worker-cls transfer_moe.skip_transfer.qwen3_5_moe.patch_vllm.TransferMoEWorker \
  --additional-config "$POLICY" \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --port 8000
```

单卡运行时使用 `CUDA_VISIBLE_DEVICES=0` 和 `--tensor-parallel-size 1`。动态预算时，将 `POLICY` 改为下面这一行，再执行同一条启动命令：

```bash
POLICY="{\"skip_transfer\":{\"calibration_path\":\"$CALIBRATION\",\"topp_threshold\":0.1}}"
```

在另一个终端发送推理请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"transfer-moe","messages":[{"role":"user","content":"请解释混合专家模型如何降低推理计算量。"}],"max_tokens":256,"temperature":0}'
```

这版只替换 `RoutedExperts.forward_modular`，保留原生 [MoERunner](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py) 的 router、MoE custom op、shared expert 调度和最终 TP 归约。两次专家投影仍使用 vLLM 的 [grouped GEMM kernel](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/fused_moe.py)，贪心选择和补偿各用一个 Triton kernel；运行时没有 Python 按专家循环，也不读取 GPU 上的动态保留数量。

固定预算将派发槽位缩至 `topk`；动态预算保持原生 K 个槽位，将未使用槽位标记为跳过，使两次 GEMM 都避开这些专家的乘法。张量形状保持稳定，可以用于 CUDA Graph。TP 下每卡计算局部专家输出，并分摊补偿的常数项，最后仍只归约 `[tokens, hidden]`，不会先归约 `[tokens, K, hidden]`。

先运行 GPU 正确性检查。覆盖全保留与原生输出一致、选择结果与 HF 贪心算法一致、补偿公式、shared expert、动态预算的 CUDA Graph 重放；第二条命令额外运行真实双卡 TP 归约。全保留按 FP16/BF16 数值容差比较，并非逐 bit 一致。

```bash
python -m unittest transfer_moe.tests.test_vllm -v

torchrun --standalone --nproc-per-node=2 -m unittest \
  transfer_moe.tests.test_vllm.NativeRunnerTests -v
```

下面的逐层基准使用合成数据，对比原生 Triton routed experts 与加入选择、补偿后的耗时。**两边默认都启用 CUDA Graph**；`speedup > 1` 才表示该项实测加速，`topk=8` 可观察全保留时的额外开销。

```bash
python -m transfer_moe.tests.benchmark_vllm \
  --experts 256 \
  --hidden-size 2048 \
  --intermediate-size 512 \
  --router-topk 8 \
  --tokens 1 16 128 512 \
  --topk 4 8 \
  --topp-threshold 0.15 \
  --output transfer_moe/outputs/vllm-benchmark.json
```

按实际模型配置修改专家数和维度；有 TP 时，`--intermediate-size` 使用每卡分片后的大小。该基准不包含 router、shared expert、attention 或 TP 通信，不能代替整模型测速。这里 `0.15` 只作用于合成误差矩阵。`--eager` 可用于诊断，两边会一起关闭 CUDA Graph。

整模型对比时，先停止 patch 服务，用以下命令启动原生基线。设备、dtype、TP、Triton backend 和调度参数均与上面的服务相同；两者都保留默认的编译和 CUDA Graph 设置。

```bash
vllm serve "$MODEL" \
  --served-model-name transfer-moe \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --moe-backend triton \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --port 8000
```

在另一个已激活 vLLM 环境的终端运行压测。测完原生服务，再切换为 patch 服务，将 `RUN` 改为 `transfer-topk4` 后运行相同命令。比较输出吞吐量、TTFT 和 TPOT；固定输入/输出长度，避免模型提前结束生成影响速度对比。

```bash
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"
RUN="native-triton"

vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model transfer-moe \
  --tokenizer "$MODEL" \
  --dataset-name random \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 100 \
  --max-concurrency 8 \
  --request-rate inf \
  --ignore-eos \
  --seed 0 \
  --save-result \
  --result-dir "transfer_moe/outputs/bench-$RUN"
```

再用不同并发和输入/输出长度分别观察 decode 与 prefill。若生产配置使用自动选择的 MoE backend，还应将原生服务去掉 `--moe-backend triton` 再测一次：相对原生 Triton 加速，不代表一定快过该 GPU 上的其他 backend。剪枝收益取决于保留数量、batch 和硬件，选择与补偿也有成本；当前开发环境没有 CUDA，以上 GPU 校验和测速尚未执行。
