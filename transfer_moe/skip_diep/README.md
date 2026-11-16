# DIEP：结合路由权重与专家相似度决定跳过谁

**当前状态：算法说明，供实现前审核；本目录尚未提供校准程序或推理 patch。** 本文讨论[论文第 17 页式 (6)](https://arxiv.org/pdf/2609.25809#page=17)及附录 A.3、C 中的 DIEP 跳过规则。该论文只复现了原始 DIEP 的跳过判据，并推广到 Top-K；**不包含原始方法的可微剪枝训练目标**。

## 先用一句话理解

router 认为某位专家不太重要，并不一定意味着应该删掉它：如果它与排名第一的专家做的事情很不一样，仍可能值得保留。反过来，如果它与第一名高度相似，就要求它拿到更高的路由权重，才允许继续参加计算。

这里“相似”来自离线收集的专家输出统计。推理时只查表，无须先执行所有候选专家来比较它们的当前输出。

## 离线需要准备什么

对每个 MoE 层 ℓ，准备以下统计量：

| 统计量 | 含义 | 在线用途 |
| --- | --- | --- |
| sim_ℓ[a, b] | 专家 a、b 输出的 linear CKA 相似度，形状 E × E | 查当前 Top-1 与各候选的相似度 |
| mean_sim_ℓ | 该层有效的不同专家对的平均相似度，不含对角线 | 把相似度变成相对尺度 |
| m_ℓ | 校准 token 上 w₂ / w₁ 的中位数 | 原始缩放版本使用；本比较版本默认不额外乘它 |

linear CKA 可以理解为“两个专家输出随样本变化的结构有多接近”。对已经沿样本维度减去均值的输出矩阵 X、Y，作者实现采用：

$$
\operatorname{CKA}(X,Y)=
\frac{\|X^\top Y\|_F^2}
{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

这里每行是一条输出样本，每列是一个隐藏维度，F 表示把矩阵元素平方后求和再开方。结果被限制在 [0, 1]；值大表示结构相似，不代表两个输出逐元素相等，也不保证可以直接互相替代。[作者 CKA 实现](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/diep_calibration.py#L47)

作者的统计构建会对有效的非对角专家对取均值；样本不足的专家对用这一均值填充，对角线设为 1。在线读取时，对平均相似度设置 1e-6 的下限，避免除零。[统计构建](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/diep_calibration.py#L711)、[在线查表](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/expert_pruning/routing.py#L3343)

## 在线如何选择专家

设原生 router 的 K 个候选按权重降序排列，e₁ 是当前 token 的第一名。对候选 e_i 定义：

$$
\gamma_i=\left(\frac{\operatorname{sim}_{\ell}(e_1,e_i)}
{\overline{\operatorname{sim}}_{\ell}}\right)^\alpha,\qquad \gamma_1=1.
$$

本文以**从 1 开始的排名**表示，默认强制保留前 k_min = 2 个专家。其余候选各自判断：

$$
\operatorname{keep}(i)
\iff i\le k_{\min}\quad\text{或}\quad
w_i\ge\beta\gamma_i w_1.
$$

等价地，对非强制保留候选，比较“相对权重 w_i / w₁”与“相似度修正后的门槛 βγ_i”。

- γ_i > 1：比该层通常的专家对更相似，门槛提高，更容易跳过。
- γ_i < 1：相似度较低，门槛降低，较小权重也有机会保留。
- β 越大：同一组输入上门槛越高，保留数越少或不变。

判定完所有候选后，得到集合 S，对其中的权重重新归一化：

$$
\widetilde w_i=\frac{w_i}{\sum_{j\in S}w_j},\qquad
\widehat y=\sum_{i\in S}\widetilde w_i f_{e_i}(x)+y_{\mathrm{shared}}(x).
$$

只计算 S 内的专家，不重构被跳过专家的输出。这里使用的是[作者 `_diep_topk` 的默认 independent 语义](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/expert_pruning/routing.py#L3454)。

## 手算一个例子

设 K = 5，k_min = 2，β = 0.4，α = 1，层内平均相似度为 0.5。以下是说明性数据：

| 排名 | 权重 w_i | 与 Top-1 的相似度 | γ_i | 门槛 βγ_i w₁ | 结果 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.40 | 自身 | 固定为 1 | 无须判断 | 强制保留 |
| 2 | 0.25 | 无须判断 | 无须判断 | 无须判断 | 强制保留 |
| 3 | 0.18 | 0.80 | 1.60 | 0.256 | 跳过 |
| 4 | 0.10 | 0.20 | 0.40 | 0.064 | 保留 |
| 5 | 0.07 | 0.70 | 1.40 | 0.224 | 跳过 |

最终保留排名 {1, 2, 4}，总权重为 0.75，新权重为 [8/15, 1/3, 2/15]。虽然第三名权重高于第四名，但第三名与 Top-1 更相似，所以被跳过；第四名贡献的差异更大，反而留下。

这也说明：**DIEP 的保留集合不一定是一个连续前缀，不能只返回“保留 3 个”然后执行排名前 3 的专家。必须保留具体 mask 或专家 ID。**

## 不要混淆几个版本

| 选择 | 行为 | 本目录拟采用的主版本 |
| --- | --- | --- |
| `independent` / `prefix` | 前者逐专家独立判断；后者在第一个失败处停止，连后面的候选一起丢弃 | `independent` |
| α = 1 / 0 < α < 1 | 前者完整使用相似度比例；后者减弱相似度影响，论文称 DIEP-d | α = 1 |
| 是否额外乘 m_ℓ | 原始缩放包含每层 w₂/w₁ 的中位数；比较版默认用可调 β 设定工作点 | 默认不乘 m_ℓ |
| 是否添加阈值上限 | 作者代码还支持限制 βγ_i 的额外变体 | 不启用额外上限 |

对于上面的例子，prefix 模式在第三名失败后就结束，只保留 {1, 2}，不会继续发现第四名值得保留。α = 0 时，相似度项变为 1，只剩权重比例阈值；因为权重已经降序排列，这时 independent 与 prefix 的选择相同。

论文附录 C 明确以 independent 作为主要解释。原始 DIEP 面向 Top-2，把判据推广到更宽的 Top-K，以及 α < 1 的阻尼形式，都是这篇比较论文做出的选择。m_ℓ 随层变化，因此“不再额外乘 m_ℓ、重新调 β”也不能理解为一个全局 β 在代数上严格替代所有层的 m_ℓ。[论文附录 C](https://arxiv.org/pdf/2609.25809#page=19)

## 核对论文与代码时发现的细节

**下标与最少人数。** 第 17 页式 (6) 写成 i < k_min，但同页文字把 k_min = 2 定义为至少保留两个专家。作者代码使用从 0 开始的 `ranks < k_min`，实际保护前两位。因此上文在从 1 开始的数学排名中写 i ≤ k_min，避免误写成只保护第一位。

**CKA 的样本配对。** 所核对的作者代码按每个专家实际收到的 token 分别收集未乘路由权重的输出，再将两组样本截成相同长度计算 CKA；没有按相同 token ID 配对。这不等价于让所有专家处理同一批输入。后续若改为共同输入、逐样本对齐的 CKA，应标注为校准变体，不能称为与这份作者实现完全一致。[采样代码](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/diep_calibration.py#L380)

**校准数据量。** 论文 A.3 记载使用 512 条 C4 文档；所核对的独立 DIEP 校准脚本默认 `nsamples=128`。若按论文设置复现，需要显式指定 512，并保存实际样本与截断配置，不能把脚本默认值当作论文设置。[校准脚本参数](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/diep_calibration.py#L1055)

## 拟接入本仓库的流程

```python
ids, weights = sort_desc(*native_router(x))
keep = [False] * K
for index in range(K):
    if index < min(k_min, K):
        keep[index] = True
    else:
        similarity = sim[layer, ids[0], ids[index]]
        gamma = (similarity / max(mean_sim[layer], 1e-6)) ** alpha
        keep[index] = weights[index] >= beta * gamma * weights[0]
kept_ids = select_by_mask(ids, keep)
kept_weights = select_by_mask(weights, keep)
kept_weights = kept_weights / sum(kept_weights)
y = weighted_experts(x, kept_ids, kept_weights) + shared_branch(x)
```

这段伪代码描述主版本，不是现成接口。实现时需要保存并校验模型、层号、专家 ID 和相似度矩阵的对应关系；缺少统计时应显式报错，避免悄悄退化成纯权重阈值。

为匹配平均预算，可以在校准路由数据上计算不同 β 的实际保留数。论文 A.4 则采用拟合 Beta 生存函数、结合 γ 分布预测平均保留数再二分求 β 的方案；直接回放是后续实现可选的预算求解方式，需与论文的拟合流程区分。评测时仍需记录实际平均 k。[论文附录 A.4](https://arxiv.org/pdf/2609.25809#page=18)

[TransferMoE](../skip_transfer/README.md) 的 D 是带方向的替代平方误差，DIEP 的 sim 是 CKA 相似度，不能直接互换。DIEP 只让相似度参与筛选；TransferMoE 还用保留专家重构缺失输出。后续接入 [infer.py](../infer.py) 时，应为 DIEP 单独提供校准文件和选择逻辑。
