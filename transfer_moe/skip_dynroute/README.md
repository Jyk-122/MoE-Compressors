# DynRoute：累计到足够的路由权重就停止

设置 `TRANSFER_MOE_RECORD_STATS=1` 后，HF / vLLM patch 记录每层和总体剪枝比例；默认关闭。读取、清零及 EvalScope 服务评测脚本见[公共统计说明](../README.md)。

**已实现 Qwen3.5 / Qwen3.6 MoE 的 HF 与 vLLM patch。** 两个后端共用基于 `cumsum` 的批量 Top-p 筛选，不逐 token 或逐候选循环。本文对应[论文第 17 页附录 A.2、式 (5)](https://arxiv.org/pdf/2609.25809#page=17)中的 DYNROUTE，作者代码中的名字是 `Dynamic_Routing`。建议先阅读 [Fixed-K](../skip_fixedk/README.md) 的共同符号与聚合方式。

## 先用一句话理解

Fixed-K 每次固定请 k 位专家；DynRoute 则从最受 router 信任的专家开始，**请到累计投票权足够为止**。

如果前两位已经拿到大多数权重，就不需要继续增加人数；如果权重分散在很多专家之间，就多保留几位。这里的“信任”只是对路由权重的直观解释，不能视为模型答对问题的概率。

## 先明确 p 的含义

对当前 token，原生 router 选出 K 个专家，权重满足 w₁ ≥ … ≥ w_K。当前实现采用论文主比较使用的 **renormalized 版本**，参数名为 `topp_threshold`，下文记为 p：先在这 K 个候选内部归一化，得到

$$
\widehat w_i=\frac{w_i}{\sum_{j=1}^{K}w_j}.
$$

然后设置 0 < p ≤ 1。p = 0.8 表示至少覆盖“原生 K 个候选总权重的 80%”，不表示保留 80% 的专家，也不表示保留 80% 的模型准确率。

论文还介绍了 raw 版本：直接累加全体 E 个专家 softmax 后、落在 Top-K 中的原始概率。两者的区别是分母不同：

| 版本 | 用于阈值判断的权重 | p 的含义 |
| --- | --- | --- |
| `raw` | 全专家 softmax 后的 Top-K 原始概率 | 全体 E 个专家概率质量中的 p |
| `renormalized` | Top-K 内部重新归一化的权重 | 已选 K 个专家概率质量中的 p |

在 raw 版本中，Top-K 的总概率可能小于 p，此时即使全部保留也达不到目标，只能回退到 K。论文 Table 3 使用 renormalized 版本；作者通用路由代码也支持两种读法。本目录当前只实现 renormalized，raw 在这里仅用于说明区别。[论文式 (5) 的说明](https://arxiv.org/pdf/2609.25809#page=17)

## 在线选择流程

计算累计和 C_i，再找到第一次达到阈值的位置：

$$
C_i=\sum_{j=1}^{i}\widehat w_j,\qquad
k(x,\ell)=\min\{i\in\{1,\ldots,K\}:C_i\ge p\}.
$$

若不存在这样的 i，就令 k = K。实际执行集合是 S = {1, …, k}。**使累计值首次达到或超过 p 的那个专家也要保留**。

选择之后，还要对保留下来的 k 个专家重新分配权重：

$$
\widetilde w_i=\frac{w_i}{\sum_{j=1}^{k}w_j},\qquad
\widehat y=\sum_{i=1}^{k}\widetilde w_i f_{e_i}(x)+y_{\mathrm{shared}}(x).
$$

这里有两次用途不同的归一化：第一次定义阈值的分母，第二次定义剪枝后的输出。第一次归一化不能代替第二次。

```python
weights, order = weights.float().sort(dim=-1, descending=True, stable=True)
ids = ids.gather(-1, order)                         # [tokens, K]
weights = weights / weights.sum(dim=-1, keepdim=True)
keep = torch.ones_like(weights, dtype=torch.bool)
if p < 1:
    keep[:, 1:] = weights.cumsum(dim=-1)[:, :-1] < p
weights = weights.masked_fill(~keep, 0)
weights = weights / weights.sum(dim=-1, keepdim=True)
```

这就是 [routing.py](routing.py) 的筛选核心：第一名始终保留，后面的专家只看“它之前的累计值是否还没达到 p”。因此包含首次越过阈值的专家；若累计值一直未达到目标，会自然保留全部 K 个。所有 token 一起计算，没有循环、`.item()` 或把 GPU 上的数量读回 CPU。

`p = 1` 明确走全保留路径，避免极小权重或浮点舍入提前截断。两个后端此时直接调用原生专家执行函数。

## 手算一个例子

设 K = 5，p = 0.8。下面权重已经在原生 Top-K 内归一化：

| token | 降序权重 | 累计权重 | 保留数 |
| --- | --- | --- | --- |
| A，比较集中 | [0.40, 0.30, 0.15, 0.10, 0.05] | [0.40, 0.70, 0.85, 0.95, 1.00] | 3 |
| B，比较分散 | [0.25, 0.22, 0.20, 0.18, 0.15] | [0.25, 0.47, 0.67, 0.85, 1.00] | 4 |

对 token A，前两位只有 0.70，需要第三位才能达到 0.8；之后将 [0.40, 0.30, 0.15] 除以 0.85，再聚合输出。

再看 raw 与 renormalized 的区别：假设 token A 的全专家 softmax 概率在 Top-K 中是 [0.16, 0.12, 0.06, 0.04, 0.02]，总和只有 0.40。

- raw + p = 0.8：累计最多为 0.40，回退到全部 5 个。
- renormalized + p = 0.8：除以 0.40 后就是表中的 token A，保留 3 个。

因此，与作者实验比较时应同时核对分数来源和 p；本实现的分数来源固定为 renormalized。以上数字均为说明性例子。

## 参数与校准

- **p 越大，同一组路由输入上的保留数越多或不变。** 不同 token 的 k 可以不同，同一 token 在不同层的 k 也可以不同。
- 没有 DIEP/BAN 那样额外规定的 2 或 3 个专家下限；最短非空前缀仍至少包含 1 个专家。
- 本实现 p = 1 明确保留全部 K 个专家，即使部分权重因下溢变成 0。
- 算法运行不需要专家相似度或层敏感度文件；但若要与 Fixed-K 的 k = 4 公平比较，仍需用代表性路由数据选择一个使平均保留数接近 4 的 p。这是预算匹配，与算法本身需要离线统计是两回事。

论文附录 A.4 在记录的 router 输出上扫描 p，再根据预测平均专家数选择参数。最终仍要报告评测中实际的平均专家数；生成任务和似然评分任务可能需要不同的 p。[预算匹配说明](https://arxiv.org/pdf/2609.25809#page=18)

## 两个后端如何执行

[HF patch](qwen3_5_moe/patch_hf.py) 保留原生 MoE `forward`、router 和 shared expert，只包装专家模块：用 mask 提取保留的 `(token, expert)` 对，以每对一个路由槽位调用原生专家模块，再用 `index_add_` 汇总回 token。被删除的路由不会进入专家矩阵乘法，也不需要自行编写专家循环。HF 的动态索引压缩会产生变长张量，在 CUDA 上可能同步；这版 HF 用于 eager 推理，不宣称支持整图 CUDA Graph。

[vLLM patch](qwen3_5_moe/patch_vllm.py) 保持 `[tokens, K]` 的固定形状，将被删除的 ID 改为额外的哨兵 E，并通过专家映射将 E 映射到 -1。原生 Triton 对齐阶段排除这些路由，两次专家 GEMM 都跳过它们，最后由原生 kernel 加权聚合；没有仅清零权重却继续执行专家的问题。[原生映射与对齐实现](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/layers/fused_moe/moe_align_block_size.py)

vLLM 的外层 MoE runner、shared expert 调度和最终 TP 归约保持原生路径。当前适配 **vLLM 0.24.0、CUDA、未量化权重、Triton MoE backend**，支持单卡和 TP；不支持量化、LoRA、EP、EPLB、SP、DP 或 Prefill Context Parallel。固定形状路径提供了 CUDA Graph 检查，但当前开发环境没有 CUDA，GPU 正确性、图重放与加速收益需在目标设备上验证。

安装 patch 时保留模型原生 K，并在新加载的模型上操作；不要先应用 Fixed-K 或其他方法缩小候选集合。`topp_threshold` 必须满足 0 < p ≤ 1，不需要校准文件，可在推理前重复 patch 更换阈值。

注意，[TransferMoE](../skip_transfer/README.md) 的同名 `topp_threshold` 是绝对替代误差阈值，越大越容易少保留专家；DynRoute 是累计路由权重目标，越大越容易多保留专家。参数名一致，含义和调节方向不同。

## HF 运行

在仓库根目录、已有的 HF 环境中执行，先替换模型路径：

```bash
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"

python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_dynroute \
  --model-type qwen3_5_moe \
  --patch-kwargs '{"topp_threshold": 0.8}' \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt "请解释混合专家模型如何降低推理计算量。" \
  --max-new-tokens 256
```

多卡可改为 `--device auto`；图文输入沿用 [infer.py](../infer.py) 的 `--images` 参数。将阈值改为 1 可恢复全保留执行。

## vLLM 运行

在已有的 vLLM 0.24.0 环境中，从仓库根目录启动。worker 会在加载权重后、profiling 和图捕获前安装 patch：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"

vllm serve "$MODEL" \
  --served-model-name dynroute \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --moe-backend triton \
  --worker-cls transfer_moe.skip_dynroute.qwen3_5_moe.patch_vllm.DynRouteWorker \
  --additional-config '{"skip_dynroute": {"topp_threshold": 0.8}}' \
  --max-model-len 4096
```

双卡时设置 `CUDA_VISIBLE_DEVICES=0,1` 和 `--tensor-parallel-size 2`。这里不使用 Fixed-K 的 `--hf-overrides`，原生 K 必须保持不变。

## 验证

```bash
python -m unittest transfer_moe.tests.test_dynroute -v
```

覆盖排序后 ID 与权重配对、阈值恰好命中、至少保留一个专家、全保留、混合 token 预算、实际压缩派发、FP32/BF16 输出和原生生成。CUDA/vLLM 可用时还会检查真实对齐结果是否排除了剪枝路由，并验证原生 fused GEMM 和动态预算的 CUDA Graph 重放；不可用时该项明确跳过。
