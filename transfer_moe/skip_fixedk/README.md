# Fixed-K：每个 token 固定保留 k 个专家

设置 `TRANSFER_MOE_RECORD_STATS=1` 后，HF patch 记录每层和总体剪枝比例；vLLM 可添加统计 worker，默认关闭统计。读取、清零及 EvalScope 服务评测脚本见[公共统计说明](../README.md)。

**当前支持 Qwen3.5 / Qwen3.6 MoE 的 HF 推理；vLLM 使用原生配置覆盖。** 本文以论文[第 4 页的路由定义](https://arxiv.org/pdf/2609.25809#page=4)与[第 17 页 Table 4](https://arxiv.org/pdf/2609.25809#page=17)为依据。四篇说明按 Fixed-K、[DynRoute](../skip_dynroute/README.md)、[DIEP](../skip_diep/README.md)、[BAN](../skip_ban/README.md)的顺序阅读即可。

## 先用一句话理解

原模型每次请 K 个专家一起处理一个 token；Fixed-K 只请其中路由权重最高的 k 个。**固定的是人数，不是专家身份**：不同 token 仍然可以选择不同专家。

例如，模型原来每个 token 激活 8 个专家，现在统一只执行前 4 个。它不判断这个 token 是否困难，也不为某些层额外增加预算，因此适合用作其他方法的基线。

## 四种方法共用的计算流程

对某一层、某一个 token 的输入 x，记：

- E：这一层拥有的 routed expert 总数。
- K：原生 router 为这个 token 选中的专家数。
- e_i 和 w_i：排名第 i 的候选专家 ID 及其权重，i 从 1 到 K，且 w₁ ≥ … ≥ w_K。
- S：剪枝后实际执行的候选排名集合；k = |S|。
- f_e(x)：专家 e 的输出；y_shared(x)：模型原生 shared expert 分支的完整输出，包含它原有的门控。

这四种基线都先选出 S，再把保留权重归一化，计算：

$$
\widetilde w_i=\frac{w_i}{\sum_{j\in S}w_j},\qquad
\widehat y(x)=\sum_{i\in S}\widetilde w_i f_{e_i}(x)+y_{\mathrm{shared}}(x).
$$

归一化的直观含义是：少请几个专家之后，让留下的人重新分配全部 routed 分支的投票权。shared expert 不参与这次筛选，也不计入 k。这是[论文式 (2)](https://arxiv.org/pdf/2609.25809#page=4)与[附录 A.1](https://arxiv.org/pdf/2609.25809#page=16)的比较口径。

这里讨论的是推理时少执行几个专家，E 个专家的模型参数仍然保留。它不会直接缩小 checkpoint，也没有重新训练 router。

## Fixed-K 如何选择

设置一个整数 k，满足 1 ≤ k ≤ K，对每层、每个 token 都取：

$$
S=\{1,2,\ldots,k\}.
$$

执行顺序如下：

1. 运行原生 router，拿到 K 个候选的专家 ID 和权重。
2. 将候选按权重降序排列，同时移动对应的 ID。
3. 保留前 k 个候选，重新归一化它们的权重。
4. 仅执行这些专家，聚合输出，再加上原生 shared expert 分支。

用接近 Python 的伪代码表示，下面的辅助函数只表示算法步骤：

```python
ids, weights = native_router(x)       # 原生 K 个候选
ids, weights = sort_desc(ids, weights)
kept_ids = ids[:k]
kept_weights = weights[:k] / sum(weights[:k])
y = weighted_experts(x, kept_ids, kept_weights) + shared_branch(x)
```

作者仓库实际采用修改 `num_experts_per_tok` 的配置覆盖方式，让模型直接选 k 个专家；对普通的按权重 Top-K 路由，这与先取 K 再截到 k 一致。迁移到有分组、额外选择偏置或其他约束的 router 时，应验证候选集合是否仍然等价。[作者实现说明](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning#two-environments)

## 手算一个例子

以下数字仅用于解释，不是实验结果。假设 K = 5：

| 路由排名 | 专家 ID | 原权重 | k = 3 时的新权重 |
| --- | --- | --- | --- |
| 1 | 12 | 0.40 | 0.40 / 0.85 ≈ 0.4706 |
| 2 | 7 | 0.30 | 0.30 / 0.85 ≈ 0.3529 |
| 3 | 25 | 0.15 | 0.15 / 0.85 ≈ 0.1765 |
| 4 | 3 | 0.10 | 跳过 |
| 5 | 18 | 0.05 | 跳过 |

因此 routed 输出为：

$$
\widehat y_{\mathrm{routed}}
=\frac{8}{17}f_{12}(x)+\frac{6}{17}f_7(x)+\frac{3}{17}f_{25}(x).
$$

下一个 token 的排名可能完全不同，但仍然执行 3 个专家。

## 与其余方法放在一起看

| 方法 | 用什么决定保留集合 | 在线决策是否依赖离线统计 | 主要预算参数 |
| --- | --- | --- | --- |
| Fixed-K | 权重排序后固定取前 k 个 | 否 | k；越大，保留越多 |
| DynRoute | 累计权重首次达到目标 p 的前缀 | 否 | p；越大，保留越多或不变 |
| DIEP | 单个权重相对 Top-1 的比例，再结合专家相似度 | 是，专家相似度 | β；越大，保留越少或不变 |
| BAN | 当前 token 的权重集中度和离线层敏感度 | 是，层敏感度与集中度范围 | λ；越大，保留越多或不变 |

动态方法的预算参数不直接等于专家数量。公平比较时，应记录实际平均 routed 专家数：

$$
\overline k=\frac{\text{所有有效 token-layer 对实际保留的 routed 专家数之和}}
{\text{有效 token-layer 对的数量}}.
$$

例如 Fixed-K 的 k = 4，应该与实测平均约 4 个专家的动态配置比较；不能把某个阈值的数值直接当作预算。以上方向均指在同一组路由输入上改变参数的效果。[论文 §5.1](https://arxiv.org/pdf/2609.25809#page=8)、[附录 A.4](https://arxiv.org/pdf/2609.25809#page=18)

## 当前实现

[HF patch](qwen3_5_moe/patch_hf.py) 沿用 `skip_transfer` 的 `patch(model, **kwargs)` 接口，核心只有一行：

```python
block.gate.top_k = topk
```

在每个 MoE 层设置人数后，[原生 router](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)直接选择 k 个专家并归一化，原生专家模块按这 k 个槽位派发计算。MoE 的 `forward`、专家参数、shared expert 分支都沿用原实现。

`topk` 必须是 1 到原生 K 的整数。patch 只修改运行时 router，保留原始配置中的 K，因此可以再次调用切换预算，包括设回 K。应在新加载的原生模型上、编译和推理之前调用；重新加载模型后也需要重新应用。

[TransferMoE](../skip_transfer/README.md) 的 `topk` 通过贪心替代代价选择专家，并补偿被跳过的输出；Fixed-K 按路由权重选择，随后重新归一化，不需要校准文件。

## 运行

在仓库根目录、已有的 HF 环境中执行，先将 `MODEL` 改为实际路径：

```bash
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"

python -m transfer_moe.infer \
  --model "$MODEL" \
  --method skip_fixedk \
  --model-type qwen3_5_moe \
  --patch-kwargs '{"topk": 4}' \
  --device cuda:0 \
  --dtype bfloat16 \
  --prompt "请解释混合专家模型如何降低推理计算量。" \
  --max-new-tokens 256
```

多卡可改为 `--device auto`；图文输入沿用 [infer.py](../infer.py) 的 `--images` 参数。

vLLM 0.24.0 在模型初始化时就读取专家数量，直接通过 `--hf-overrides` 修改 Qwen3.5 / Qwen3.6 的嵌套文本配置即可。这样 router、专家 kernel 和缓冲区都按 k 初始化。[配置覆盖实现](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/model.py#L441)、[原生 MoE 初始化](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/qwen3_next.py#L162)

需要统一的统计 RPC 时，使用 [patch_vllm.py](qwen3_5_moe/patch_vllm.py) 的 `FixedKWorker`，并通过 `additional_config['skip_fixedk']['native_topk']` 提供覆盖前的 K。这个 wrapper 只计数，专家执行仍调用原生函数；完整命令见[公共统计说明](../README.md)。

在已有的 vLLM 环境中运行：

```bash
MODEL="/data1/jiangyikun/models/Qwen3.6-35B-A3B"

vllm serve "$MODEL" \
  --served-model-name fixedk \
  --hf-overrides '{"text_config": {"num_experts_per_tok": 4}}' \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

这里同样应选择 1 ≤ k ≤ 原生 K；双卡可将 `--tensor-parallel-size` 改为 2。保留各自后端原有的编译与 kernel 设置。当前开发环境没有 CUDA，vLLM 服务及 GPU 性能尚未实测。
