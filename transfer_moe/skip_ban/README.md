# BAN：结合层敏感度与 token 敏感度分配专家数量

**当前状态：算法说明，供实现前审核；本目录尚未提供校准程序或推理 patch。** BAN 的介绍从[论文第 17 页](https://arxiv.org/pdf/2609.25809#page=17)开始，完整公式在**第 18 页式 (7)、(8)**，校准过程见附录 A.3。

## 先用一句话理解

BAN 把专家预算分配给两种位置：**少算几个专家会明显影响模型输出的层，以及路由权重比较分散的 token**。

它先离线测出哪些层更敏感；推理时再看当前 token 的路由有多集中，把两个信号合起来决定保留多少专家。人数决定以后，仍然只取路由排名最靠前的那些专家。

## 两种敏感度各自表示什么

### 层敏感度：这一层少算专家，会不会影响最终预测

校准时，对同一份输入做以下比较：

1. 跑一次全保留模型，各层都用原生 K 个 routed experts，得到参考预测分布。
2. 只把第 ℓ 层改成保留前 3 个，其余层仍然保持原生 K，再跑完整模型。
3. 比较两次模型最终的下一 token 预测分布，对校准样本求平均，得到该层的变化量 d_ℓ。
4. 每次只干预一层，轮流测所有 MoE 层；再将 d_ℓ 做跨层 min-max 归一化：

$$
s_\ell=\frac{d_\ell-\min_j d_j}{\max_j d_j-\min_j d_j}.
$$

实际实现需要给分母设置正数下限。s_ℓ 越大，说明单独削减这一层对最终预测的影响越明显，在线分配时应倾向于给它更多专家。

这里比较的是**完整模型的预测分布**，不是该层隐藏状态的均方误差，也不是把所有层同时改成 Top-3 后比较一次。[论文 A.3](https://arxiv.org/pdf/2609.25809#page=18)

为明确“分布差异”如何落地，作者的校准代码默认：

- 使用每条校准序列最后一个位置的 logits；也提供使用全部位置的选项。
- 在全保留模型的 logits 中选出概率最高的 1000 个词表项；两次预测都在**同一组词表项**上重新 softmax。
- 计算 KL(P_full || P_only_layer_ℓ_pruned)，再对样本求平均。这里的 Top-1000 是词表项数，与专家数量无关。

因此这是一种截取词表支持集后的 KL，不应描述为完整词表 KL。相关设置都应保存在校准文件中。[作者校准实现](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/ban_calibration.py#L548)

### token 敏感度：前三名已经承担了多少路由权重

对当前 token、当前层的原生 K 个候选，定义：

$$
r=\frac{w_1+w_2+w_3}{\sum_{i=1}^{K}w_i},\qquad
s_{\mathrm{tok}}=\operatorname{clamp}
\left(\frac{r_{\max}-r}{r_{\max}-r_{\min}},0,1\right).
$$

其中 w_i 按降序排列。r_min、r_max 是离线全保留校准时，在所有被统计的 token 和层上收集到的 r 的最小值与最大值，之后固定下来使用。作者代码采用全局范围，不是每个推理 batch 临时计算一遍。[在线计算](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/expert_pruning/routing.py#L3393)

- r 大：前三名已经覆盖较多权重，s_tok 小，倾向少用专家。
- r 小：还有较多权重在后面，s_tok 大，倾向多用专家。

这里的敏感度是一个路由代理信号，不是直接检测 token 的语义难度。由于使用比值，所有候选权重同时乘上相同系数不会改变 r。

## 把两个信号转换成保留数量

有了 s_ℓ、s_tok 后，设置非负参数 λ 与最少保留数量 k_min：

$$
s=\operatorname{clamp}\left(\frac{\lambda}{2}
(s_\ell+s_{\mathrm{tok}}),0,1\right),
$$

$$
k(x,\ell)=\operatorname{clamp}\left(
\operatorname{round}\big(k_{\min}+(K-k_{\min})s\big),
k_{\min},K\right).
$$

可以分三步理解：先把两种敏感度取平均，再用 λ 调节总体预算，最后把 [0, 1] 映射到 [k_min, K] 并取整。这里的 K 一直是原生候选数量。

得到 k 后取 S = {1, …, k}，执行：

$$
\widetilde w_i=\frac{w_i}{\sum_{j=1}^{k}w_j},\qquad
\widehat y=\sum_{i=1}^{k}\widetilde w_i f_{e_i}(x)+y_{\mathrm{shared}}(x).
$$

只对保留专家做真实计算，shared expert 按原生路径执行，不计入 k。BAN 不补偿被跳过专家的输出。

## 手算一个例子

设 K = 8，k_min = 3，λ = 1，校准范围为 [r_min, r_max] = [0.40, 0.90]。以下是说明性数据：

| 情况 | 层敏感度 s_ℓ | Top-3 占比 r | token 敏感度 s_tok | 合并值 s | 取整前人数 | 保留数 |
| --- | --- | --- | --- | --- | --- | --- |
| 较敏感层，路由集中 | 0.60 | 0.85 | 0.10 | 0.35 | 3 + 5 × 0.35 = 4.75 | 5 |
| 同一层，路由分散 | 0.60 | 0.55 | 0.70 | 0.65 | 3 + 5 × 0.65 = 6.25 | 6 |
| 较不敏感层，同样的路由占比 | 0.20 | 0.55 | 0.70 | 0.45 | 3 + 5 × 0.45 = 5.25 | 5 |

前两行展示 token 之间如何分配预算，后两行展示层之间如何分配预算。λ = 1 不代表全保留；它仍然取决于两个敏感度的大小。

## Top-3 与 k_min 是两个独立设置

论文 Table 4 把 BAN 的下限列为 k_min = 3，但不能据此把所有实验写成“永远至少 3 个”：作者的 [Ling-lite 实验脚本](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/scripts/models/ling-lite-1.5-2507/ban.sh#L18)分别使用了下限 2 和 1，以覆盖更低的平均预算。

在作者实现中，以下三个数字可以分别出现：

| 设置 | 用途 | 文中标准设定 |
| --- | --- | --- |
| 集中度分子中的 3 | 计算当前 token 的 Top-3 权重占比 | 固定取前三位；K < 3 时取全部 K 位 |
| 校准干预数量 `k_pruned` | 测量单层减少专家后的预测变化 | 默认 3 |
| 在线保留下限 k_min | 公式 (8) 允许的最少人数 | 默认 3，部分实验设为 1 或 2 |

因此把 k_min 改成 1，不意味着把 r 的分子改成 Top-1，也不自动要求重新做 Top-1 的层干预校准。本目录拟以 **Top-3 信号、Top-3 层校准、默认 k_min = 3** 为主版本，并将下限作为需要明确记录的预算设置。

## 校准与推理如何衔接

离线保存每层 s_ℓ、全局 r_min/r_max，以及对应模型、层映射、原生 K、校准干预数量、词表截断和 logits 位置配置。作者还保存原始层 KL，便于检查归一化前的数值。[校准文件构建](https://github.com/MingZwhy/An-Empirical-Study-on-Expert-Pruning/blob/b85b5a0618ba4be29e18ad57a341f0580aad2e21/calib_utils/ban_calibration.py#L686)

在线只需读这些统计并做少量标量运算，无须再次跑全保留和剪枝两套模型：

```python
ids, weights = sort_desc(*native_router(x))
floor = min(k_min, K)
ratio = sum(weights[:min(3, K)]) / sum(weights)
token_s = clamp((r_max - ratio) / max(r_max - r_min, 1e-6), 0, 1)
layer_s = clamp(layer_sensitivity[layer], 0, 1)
s = clamp(lambda_coef * (layer_s + token_s) / 2, 0, 1)
count = clamp(round_to_even(floor + (K - floor) * s), floor, K)
kept_weights = weights[:count] / sum(weights[:count])
y = weighted_experts(x, ids[:count], kept_weights) + shared_branch(x)
```

这段伪代码只说明计算顺序。作者使用 `torch.round`，恰好遇到半整数时取最近偶数，例如 4.5 → 4、5.5 → 6；复现时不能默默换成统一向上取整。无效派发还需由实际后端跳过，不能仅将权重清零就认定获得了计算加速。

## 参数与预算的边界

- λ 越大，同一组路由输入上的保留数越多或不变；λ = 0 时所有 token 都落到 k_min。
- 仅调 λ 无法让平均保留数低于 k_min。要匹配更低预算，需显式改变下限并在实验中标明。
- 当 r 超过校准范围时，用 clamp 限制 s_tok；范围退化时需要分母保护，不能产生 NaN。
- K ≤ 3 时，前三位占全部候选，r 恒为 1，token 集中度信号失去区分能力；默认下限截到 K 后相当于全保留。
- 动态剪枝后后续层的输入会改变，所以离线统计只是近似依据，最终平均专家数与质量必须实测。

论文 A.4 拟合 token 敏感度的 Beta 分布，并对取整公式数值求解 λ 来命中目标平均预算。后续也可以在校准路由样本上直接回放公式、搜索 λ，但应标明预算求解方式，并在评测中报告实际平均 k。[论文预算求解说明](https://arxiv.org/pdf/2609.25809#page=18)

与 [DynRoute](../skip_dynroute/README.md) 相比，BAN 额外使用层敏感度，而非对每个位置应用同一个累计权重目标；与 [DIEP](../skip_diep/README.md) 相比，它不需要 E × E 的专家相似度矩阵，也不独立挑选不连续的候选。当前 [TransferMoE](../skip_transfer/README.md) 的替代误差统计不能直接作为 BAN 的层敏感度，后续接入 [infer.py](../infer.py) 时需要单独校准。
