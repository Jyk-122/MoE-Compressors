# 基于逐通道统计与替代误差上界的动态 MoE 专家选择算法

> 版本：2026-11-12  
> 目标：在 Router 已确定 Top-K 专家及其权重后，仅执行其中少量专家，使用保留专家经过逐通道仿射变换来近似未执行专家，并以极低在线开销决定**保留集合**与**替代映射**。  
> 示例配置：总专家数 `num_experts=512`，Router `top_k=8`。本文所有公式均允许一般的 `top_k`。

## 1. 问题背景与设计目标

给定当前 token 的隐藏状态 `h`，Router 从 512 个专家中选择 Top-8。设这 8 个专家的全局索引集合为 $T$，每个专家输出 $x_j(h)\in\mathbb R^m$、路由权重 $w_j\ge 0$。原始 MoE 输出为

$$
y(h)=\sum_{j\in T}w_jx_j(h).
$$

我们希望从 Top-8 中选出少量专家 $S\subseteq T$ 实际执行；未保留专家 $j\in T\setminus S$ 由某个已保留专家 $a(j)\in S$ 代替。最终希望得到：

1. 一条嵌套的保留顺序：保留前 1、2、……、8 个专家时对应哪些专家；
2. 每个保留规模下，所有未执行专家分别由哪个保留专家替代；
3. 每个保留规模对应的**代理替代代价**曲线，供后续预算或阈值控制使用；
4. 在线阶段不执行全部 8 个专家、不枚举所有子集、不做 SVD/QR，且不产生 CPU↔GPU/NPU 同步。

**约束**：这里选择对象是 Router 已输出的 Top-8，不能从 Top-8 外增加专家。本文最终选定的方法是 **逐通道仿射替代 + 离线有向替代代价矩阵 + 在线前向贪心（weighted facility-location greedy）**。

## 2. 专家替代算子与离线统计

### 2.1 逐通道统计

对每个专家 $i$，在校准输入分布 $\mathcal D$ 上统计其输出的逐通道均值和标准差：

$$
\mu_i\in\mathbb R^m,\qquad \sigma_i\in\mathbb R_{>0}^m.
$$

其中 $\sigma_i$ 应采用数值稳定处理（例如 $\sigma_{i,c}\leftarrow\max(\sigma_{i,c},\varepsilon)$）。定义标准化输出

$$
z_i(h)=(x_i(h)-\mu_i)\oslash\sigma_i,
$$

$\odot$ 表示逐通道乘法，$\oslash$ 表示逐通道除法。

### 2.2 有向替代算子

规定 $i\to j$ 的含义为：**实际执行专家 $i$，用它的输出去近似专家 $j$**。替代算子为

$$
\boxed{T_{i\to j}(x_i)=\mu_j+\sigma_j\odot\big((x_i-\mu_i)\oslash\sigma_i\big)
=\mu_j+\sigma_j\odot z_i.}
$$

定义真实替代误差**向量**与其平方误差**标量**（明确区分两者）：

$$
\boxed{E_{i\to j}(h)=x_j(h)-T_{i\to j}(x_i(h))
=\sigma_j\odot(z_j(h)-z_i(h)),}
$$

$$
\boxed{e_{i\to j}(h)=\|E_{i\to j}(h)\|_2^2.}
$$

此处 $e$ 是平方 $L_2$ 误差；若要求严格的「每通道 MSE」，需除以输出维度 $m$。只要统一采用同一个常数尺度，专家选择的排序不受影响。

### 2.3 离线有向替代代价矩阵

定义

$$
\boxed{D_{ij}=\mathbb E_{h\sim\mathcal D}\big[e_{i\to j}(h)\big]
=\mathbb E\left[\left\|\sigma_j\odot(z_j(h)-z_i(h))\right\|_2^2\right].}
$$

- $D\in\mathbb R^{512\times512}$；行 $i$ 是**替代源**，列 $j$ 是**被替代目标**。
- $D_{ii}=0$（同一专家替代自身，理论上零误差）。
- 一般 **$D_{ij}\ne D_{ji}$**：目标专家的逐通道 $\sigma_j$ 参与误差加权。
- 离线直接从校准样本累计误差，即可支持逐通道 $\mu,\sigma$；**无需假设单一标量标准差，也无需只用 $512\times512$ 的输出 Gram 矩阵去近似逐通道变换**。
- 离线存储：FP32 的 $512\times512$ 矩阵约 **1 MiB**；FP16 约 0.5 MiB（需检查精度）。

若每个标准化通道具有均值 0、方差 1，记 $\rho_{ij,c}=\mathbb E[z_{i,c}z_{j,c}]$，则

$$
D_{ij}=2\sum_{c=1}^m\sigma_{j,c}^2(1-\rho_{ij,c}).
$$

这是推导关系；工程上直接测量并累计 $\|x_j-T_{i\to j}(x_i)\|^2$ 通常更直观、更稳健。

> **部署前提**：统计的 $\mu_i,\sigma_i$ 必须离线可用，不能要求在当前 token 执行未保留专家后才计算它们。离线估计的 $D$ 反映校准分布的平均表现，不能天然提供每个在线 token 的确定性误差保证。

## 3. 完整 MoE 输出误差与可优化上界

### 3.1 替代后的输出

给定非空保留集合 $S\subseteq T$，对未保留专家指定替代源 $a(j)\in S$。替代输出为

$$
\hat y_S(h)=\sum_{i\in S}w_ix_i(h)+\sum_{j\in T\setminus S}w_jT_{a(j)\to j}(x_{a(j)}(h)).
$$

记 $E_j(h)=E_{a(j)\to j}(h)$（对 $j\in S$ 约定 $E_j(h)=0$），其标量平方误差 $e_j(h)=\|E_j(h)\|_2^2$。于是

$$
y(h)-\hat y_S(h)=\sum_{j\in T}w_jE_j(h).
$$

真实的**输出误差向量范数**为 $\|\sum_jw_jE_j\|_2$，真实的**输出平方误差**为

$$
\boxed{\mathcal L_{\mathrm{true}}(S,a;h)=\left\|\sum_{j\in T}w_jE_j(h)\right\|_2^2.}
$$

注意真实平方误差展开后有 $w_j^2\|E_j\|_2^2$ 和不同误差向量的交叉项，不能直接把它当成单独专家的平方误差之和。

### 3.2 Jensen / 加权柯西不等式

假设 Router 权重非负，且对当前 Top-8 **归一化**：

$$
w_j\ge0,\quad\sum_{j\in T}w_j=1.
$$

由平方范数凸性（Jensen 不等式），也等价于加权柯西–施瓦茨不等式：

$$
\begin{aligned}
\mathcal L_{\mathrm{true}}(S,a;h)
&=\left\|\sum_{j\in T}w_jE_j(h)\right\|_2^2\\
&\le\left(\sum_{j\in T}w_j\right)
\left(\sum_{j\in T}w_j\|E_j(h)\|_2^2\right)\\
&=\sum_{j\in T\setminus S}w_je_j(h).
\end{aligned}
$$

因此定义**当前 token 上的误差上界**：

$$
\boxed{J_{\mathrm{token}}(S,a;h)=\sum_{j\in T\setminus S}w_je_{a(j)\to j}(h),\qquad
\mathcal L_{\mathrm{true}}(S,a;h)\le J_{\mathrm{token}}(S,a;h).}
$$

为何这里是 $w_j$ 而不是 $w_j^2$？因为它是利用归一化权重得到的凸性上界，而非真实平方误差展开式。若仅使用普通柯西，也能导出另一种较松或不同的上界：

$$
\left\|\sum_jw_jE_j\right\|_2^2\le \left(\sum_jw_j^2\right)\left(\sum_j\|E_j\|_2^2\right).
$$

如果 $\sum_j w_j\le1$，上述 $\mathcal L_{\mathrm{true}}\le J_{\mathrm{token}}$ 仍成立（给剩余权重配一个零误差向量）。若权重总和为 $M>1$，则一般只能保证 $\mathcal L_{\mathrm{true}}\le M J_{\mathrm{token}}$，不能直接称 $J_{\mathrm{token}}$ 为真实误差上界。

### 3.3 从 token 误差上界到离线代理目标

由于在线不能计算被裁剪专家的真实输出，使用离线代价 $D_{ij}$ 代替 $e_{i\to j}(h)$。给定 $S$，按独立的最小期望替代误差选择替代源：

$$
\boxed{a_S(j)=\arg\min_{i\in S}D_{ij}.}
$$

最终在线优化的**代理目标**定义为

$$
\boxed{J(S)=\sum_{j\in T}w_j\min_{i\in S}D_{ij}.}
$$

如果专家 $j$ 被保留，$D_{jj}=0$，对应项自然为零。

**这个代理的理论含义必须准确界定**：

- 对每个确定 token，使用实际的 $e_{i\to j}(h)$ 构造的 $J_{\mathrm{token}}$ 是真实平方输出误差的严格上界。
- 离线 $D_{ij}=\mathbb E[e_{i\to j}(h)]$ 给出**期望单专家替代误差**。如果固定权重、保留集合、替代映射，且 $D$ 与评估条件分布一致，则 $J(S)$ 是对应**期望输出平方误差**的上界。
- **实际 MoE 中 Top-8 与 $w_j$ 依赖 token**，因此无条件统计得到的 $D$ 一般不保证当前路由条件下的期望上界，更不能保证每个 token 的误差。可考虑在路由条件、专家共激活、输入簇等条件下校准代价，并实测代理目标与真实输出 MSE 的排序一致性。
- 最小化误差上界是有依据的保守目标，但**不等于最小化真实误差**；上界可能因不同替代误差向量互相抵消而较松。

如需「能量覆盖率」展示，可以在校准集上定义真实覆盖率 $1-\|y-\hat y_S\|_2^2/\|y\|_2^2$ 并独立验证。**不能直接把 $1-J(S)$ 当成能量覆盖率**；$J$ 是带单位的平方误差，且用离线 $D$ 时只是代理量。

## 4. 在线联合决策：前向贪心选择

### 4.1 两个决策合并

给定当前 Top-8 集合 $T$，我们需要决定：

1. **保留哪些专家**：$S\subseteq T$。
2. **谁替代谁**：$a_S(j)=\arg\min_{i\in S}D_{ij}$。

第二个决策由第一个决策及代价矩阵直接确定，所以可将问题写成：

$$
\boxed{\min_{S\subseteq T,\,|S|=k'}J(S)
=\min_{S\subseteq T,\,|S|=k'}\sum_{j\in T}w_j\min_{i\in S}D_{ij}.}
$$

这是**带权设施选址 / 代表点选择**形式的集合优化。精确求解未必有简单的一次排序公式；最终选用逐步增加专家的近似算法，不枚举 $2^8$ 个子集。

### 4.2 状态向量

给当前保留集合 $S$ 维护：

$$
\boxed{e_{\min,j}=\min_{i\in S}D_{ij}}
$$

以及

$$
\boxed{best\_idx[j]=\arg\min_{i\in S}D_{ij}}.
$$

- `e_min[j]`：目标专家 $j$ 的当前最小替代误差。
- `best_idx[j]`：目前由哪个保留专家替代目标专家 $j$。
- `selected_order`：专家实际被选入集合的先后顺序。与 `best_idx` 是两个不同概念。

当前代理误差为 $J(S)=\sum_jw_je_{\min,j}$。

### 4.3 边际收益推导

考虑将尚未选择的专家 $i$ 加入 $S$，新的每目标最小误差为

$$
e_{\min,j}^{\mathrm{new}}=\min(e_{\min,j},D_{ij}).
$$

因此代理目标下降量（定义为**非负收益**）为

$$
\begin{aligned}
\Delta(i\mid S)
&=J(S)-J(S\cup\{i\})\\
&=\sum_{j\in T}w_j\left[e_{\min,j}-\min(e_{\min,j},D_{ij})\right]\\
&=\boxed{\sum_{j\in T}w_j\max(e_{\min,j}-D_{ij},0)}.
\end{aligned}
$$

每轮选择

$$
\boxed{i^*=\arg\max_{i\in T\setminus S}\Delta(i\mid S).}
$$

并同步更新：

$$
e_{\min,j}\leftarrow\min(e_{\min,j},D_{i^*j}),
$$

$$
best\_idx[j]\leftarrow i^*\quad\text{若 }D_{i^*j}<\text{更新前的 }e_{\min,j}.
$$

**索引方向：$D_{ij}$ 的行是候选替代源 $i$，列是被替代目标 $j$。** 若以每行代表一个候选源，在线同时计算全部候选收益可写为 `sum_j weights[j] * max(e_min[j] - D8[i,j], 0)`。

### 4.4 第一轮初始化

没有保留专家时，$e_{\min}=+\infty$，不能直接算 `inf - inf`。第一轮单独计算：

$$
\boxed{i_1=\arg\min_{i\in T}\sum_{j\in T}w_jD_{ij}.}
$$

然后设置：

$$
S=\{i_1\},\qquad e_{\min}=D_{i_1,:},\qquad best\_idx[:]=i_1.
$$

再完成最多 7 轮新增操作即可获得 1～8 的保留序列。

### 4.5 算法性质与边界

- $J(S)$ **单调不增**：新保留源只能降低每个目标的最佳替代误差。
- 收益 $J(S_0)-J(S)$（以任意固定非空基准集合 $S_0$ 表示）具有边际收益递减的**次模结构**；可在合适的有限基准下得到典型 facility-location 收益函数。但**最终贪心不保证每个 $k'$ 的全局最优子集**。
- `selected_order[:k']` 对所有 $k'$ 构成**嵌套子集**；每个阶段的 `best_idx` 可能变化，例如原本由 E2 替代的 E4，加入 E3 后改由 E3 替代。
- `J(S)` 是代理代价曲线，不是严格的真实输出误差曲线。

## 5. 完整数值实例：4 个专家

为了方便手算，将 Top-8 缩小为 Top-4。四个 Router 权重为

$$
(w_1,w_2,w_3,w_4)=(0.4,0.3,0.2,0.1).
$$

假设离线替代代价为（行 = 替代源，列 = 被替代目标）：

| 源 $\downarrow$ / 目标 $\to$ | E1 | E2 | E3 | E4 |
|---|---:|---:|---:|---:|
| **E1** | 0 | 1 | 8 | 9 |
| **E2** | 1 | 0 | 7 | 8 |
| **E3** | 8 | 7 | 0 | 2 |
| **E4** | 9 | 8 | 2 | 0 |

该示例为便于理解使用对称矩阵，实际逐通道替代误差矩阵不要求对称。

### 5.1 第一轮：只能保留 1 个专家

对每个候选源，计算 $\sum_jw_jD_{ij}$：

| 唯一保留的源专家 | 计算 | $J$ |
|---|---|---:|
| E1 | $0.4(0)+0.3(1)+0.2(8)+0.1(9)$ | 2.8 |
| E2 | $0.4(1)+0.3(0)+0.2(7)+0.1(8)$ | **2.6** |
| E3 | $0.4(8)+0.3(7)+0.2(0)+0.1(2)$ | 5.5 |
| E4 | $0.4(9)+0.3(8)+0.2(2)+0.1(0)$ | 6.4 |

选择 **E2**：

- `selected_order = [E2]`
- `e_min = [1, 0, 7, 8]`
- `best_idx = [E2, E2, E2, E2]`
- $J=2.6$。

### 5.2 第二轮：比较新增 E1、E3、E4

当前保留集合为 $S=\{E2\}$。

**新增 E1：** 仅 E1 的误差从 1 降为 0，其他目标没有更好的替代，故 $\Delta(E1\mid S)=0.4(1)=0.4$；新 $J=2.2$。

**新增 E3：** E3 的误差从 7 降为 0，E4 的误差从 8 降为 2：

$$
\Delta(E3\mid S)=0.2(7-0)+0.1(8-2)=\boxed{2.0}.
$$

新 $J=2.6-2.0=\boxed{0.6}$。

**新增 E4：** E3 从 7 降到 2，E4 从 8 降到 0：

$$
\Delta(E4\mid S)=0.2(7-2)+0.1(8-0)=1.8.
$$

因此第二轮新增 **E3**。更新为：

- `selected_order = [E2, E3]`
- `e_min = [1, 0, 0, 2]`
- `best_idx = [E2, E2, E3, E3]`
- $J=0.4(1)+0.1(2)=0.6$。

**注意**：E4 的替代源已从 E2 变成 E3，这就展示了为什么 `best_idx` 必须随保留集合动态更新。

### 5.3 第三、第四轮

剩余 E1 和 E4：

- 新增 E1：消除 E1 的代价 $0.4(1)=0.4$，$J:0.6\to0.2$。
- 新增 E4：消除 E4 的代价 $0.1(2)=0.2$，$J:0.6\to0.4$。

因此第三轮选择 **E1**；第四轮新增 **E4**。完整结果：

| 保留数 $k'$ | 选择顺序前缀 | 每目标最佳替代源 `[E1,E2,E3,E4]` | 代理误差 $J$ |
|---:|---|---|---:|
| 1 | E2 | `[E2,E2,E2,E2]` | 2.6 |
| 2 | E2, E3 | `[E2,E2,E3,E3]` | 0.6 |
| 3 | E2, E3, E1 | `[E1,E2,E3,E3]` | 0.2 |
| 4 | E2, E3, E1, E4 | `[E1,E2,E3,E4]` | 0.0 |

**这个例子里，首选是 E2，而不是路由权重最大的 E1**：选择依据是替代整个 Top-K 的综合能力，而不只是自身 router mass。

### 5.4 正向选择与反向剪枝（对比，不是最终方案）

另一种可选策略是从全体专家出发，每轮删除使 $J$ 增加最少的专家，称为 Backward Greedy。对**上面同一个例子**，结果为：

| 保留数 | Forward Greedy 最终集合 / $J$ | Backward Greedy 最终集合 / $J$ |
|---:|---|---|
| 4 | `{E1,E2,E3,E4}` / 0 | `{E1,E2,E3,E4}` / 0 |
| 3 | `{E1,E2,E3}` / 0.2 | `{E1,E2,E3}` / 0.2 |
| 2 | `{E2,E3}` / 0.6 | `{E1,E3}` / 0.5 |
| 1 | `{E2}` / 2.6 | `{E1}` / 2.8 |

反向过程依次删除 **E4 → E2 → E3**，最后剩下 E1。两种方法的排序可能不同；不是哪个方向对所有预算都更优。当前最终采用 **Forward Greedy**，因为从少量专家开始选择时可以只运行到目标保留数，且直接优化每步新增收益。

## 6. 算法伪代码

以下采用类似 Python 的伪代码；`D8[source, target]` 为方向约定。

```text
Algorithm: Weighted-Greedy Expert Retention with Substitution
Input:
    D            # [512, 512], offline directed substitution costs
    top_ids      # [K], router-selected expert IDs, K = 8
    weights      # [K], nonnegative router masses, normalized if using bound
Output:
    selected_order     # [K], order of retained experts
    cost_curve         # [K], proxy J for 1..K retained experts
    assignment_curve   # [K, K], each target's best replacement source

D8 <- D[top_ids, top_ids]            # shape [K, K]; row = source, col = target

# First retained expert: choose the best single source
first <- argmin_source sum_target(weights[target] * D8[source,target])
selected <- {first}
selected_order <- [top_ids[first]]
e_min[target] <- D8[first,target]    # length K
best_idx[target] <- first            # local index of replacement source
record sum_target(weights[target] * e_min[target])
record top_ids[best_idx]             # global replacement IDs

# Gradually grow the retained set
for step in 2..K:
    for candidate in 0..K-1:
        if candidate in selected:
            gain[candidate] <- -infinity
        else:
            gain[candidate] <- sum_target(
                weights[target] *
                max(e_min[target] - D8[candidate,target], 0)
            )

    chosen <- argmax_candidate gain[candidate]
    selected.add(chosen)
    selected_order.append(top_ids[chosen])

    # Compare against OLD e_min, update errors and replacement source together
    for target in 0..K-1:
        if D8[chosen,target] < e_min[target]:
            e_min[target] <- D8[chosen,target]
            best_idx[target] <- chosen

    cost_curve.append(sum_target(weights[target] * e_min[target]))
    assignment_curve.append(top_ids[best_idx])

return selected_order, cost_curve, assignment_curve
```

**如果目标是固定保留 `keep_k` 个专家**，可以在第 `keep_k` 轮停止，不必计算完整曲线。若路由分数相同、收益相同，建议固定 tie-breaking 规则，以便设备实现可复现。若浮点误差使对角线 `D_ii` 略大于 0，离线应强制设为 0，并将负的代价截断为 0。

### 6.1 在线输出的含义

- `selected_order[:k]`：当前 token 保留 $k$ 个专家时应实际执行哪些专家。
- `assignment_curve[k-1][j]`：第 $j$ 个 Router Top-K 目标专家最终由哪一个保留专家来实现。
- `cost_curve[k-1]`：当前保留集合的离线代理代价 $J(S)$，随保留数量增加保证非增。

注意：不同 token 的 `selected_order` 可以不同，因为 `top_ids` 和 `weights` 动态变化；但离线矩阵 $D$ 不必每个 token 重新统计。

## 7. 工程部署与计算开销

### 7.1 在线流程

```text
Offline (calibration):
    full expert outputs on calibration samples
        -> per-channel mu[512,m], sigma[512,m]
        -> directed D[512,512]

Online (per token):
    Router -> Top-8 expert IDs + 8 weights
           -> gather D8 (8x8)
           -> 1 fused greedy-selection kernel
              -> selected_order / assignment / proxy-cost curve
           -> execute only retained experts
           -> fused reconstruction of substituted MoE output
```

### 7.2 算法计算量

- 离线矩阵 $D$：对每个专家对的误差做校准统计；成本主要在校准阶段。
- 在线从 $512\times512$ 的 $D$ 提取 $8\times8$：64 个元素，即 FP32 **256 bytes** 的逻辑数据量（实际非连续访存、cache line 和访存事务另算）。
- 首轮：$8\times8$ 个 `weight * cost` 项参与约 8 次归约。
- 后续 7 轮：每轮计算最多 $8\times8$ 个边际改善项，归约 8 个候选分数，更新长度为 8 的 `e_min`、`best_idx`。
- 总计算规模可记作 $O(K^3)$，在 $K=8$ 时仅几千次基础标量操作量级；**不需要** SVD、QR、矩阵逆或 256 个保留子集枚举。

### 7.3 单独算子设计建议

推荐将上述 8 轮逻辑融合到**一次设备端 Kernel**，而不是使用 Python 逐轮 `.item()`、反复启动 Kernel：

1. Router、选择算子与 Expert Dispatch 尽量在同一 GPU/NPU 计算流水中，避免为了取 `argmax` 把索引传回 CPU。
2. 将 $D8$、`e_min`、`best_idx` 和候选收益保存在寄存器或其他片上存储。
3. Prefill 阶段按 token 维度并行；batch=1 Decode 阶段的主要风险是 Kernel Launch / 调度延迟，**不是**几千次算术操作本身。
4. 如框架允许，可最终把选择逻辑融合进 Router Top-K 或 Expert Dispatch，降低额外 Kernel Launch 与中间张量写回。
5. 真正的端到端收益还取决于动态专家执行数量、专家加载与调度、替代重建的向量算子及访存，而不只是贪心决策开销。

### 7.4 不显式生成被替代专家输出的重建公式

由 $x_i=\mu_i+\sigma_i\odot z_i$，按替代映射合并后：

$$
\begin{aligned}
\hat y_S
&=\sum_{j\in T}w_j\mu_j
+\sum_{i\in S}\left(w_i\sigma_i+
\sum_{\substack{j\in T\setminus S\\a(j)=i}}w_j\sigma_j\right)\odot z_i.
\end{aligned}
$$

因此只需要真正执行保留专家，按替代映射把其标准化输出加权聚合；不需要单独构造每个未保留专家的 $m$ 维输出。此式中逐通道 $\mu,\sigma$ 仍有带宽和计算成本，可以结合融合算子优化。

## 8. 研究验证与明确限制

1. **代理与真实 MSE 的一致性**：对离线测试 token，仍运行完整 Top-8 获取真实 $y$，测量各个 `keep_k` 的 $\|y-\hat y_S\|^2$，对比 $J(S)$ 的排序/相关性及覆盖率。只有校准结果能说明实际选得好不好。
2. **统计分布偏差**：512 个专家的输出强烈依赖当前 hidden state 及 Router 共激活分布；无条件离线 $D$ 可能不能准确描述特定 Top-8 的替代误差。考虑路由条件或输入簇分桶统计，但会增加存储与标定成本。
3. **标准差过小的通道**：必须采用 $\varepsilon$ 防止除零；逐通道统计可能比标量统计拟合能力更强，但也更依赖校准数据和数值稳定性，应实验验证。
4. **上界不等于覆盖率**：$J(S)$ 是平方误差量纲的代理上界（在所述条件下），并非 $[0,1]$ 的真实能量覆盖率；要选择达到 90% 真实能量保留的最小 `keep_k`，需要在校准集上建立预测或保守阈值。
5. **贪心不是全局最优**：完整 Top-8 的所有保留子集虽然只有 256 个，但最终为了在线部署选择 8 轮贪心，换取更低延迟和天然的嵌套顺序。
6. **权重归一化**：如实际 Router Top-K 权重经过再归一化，总和为 1，则本文 Jensen 上界直接成立；如没有再归一化，必须检查非负性、总 mass，并使用相应的缩放上界。

## 9. 最终确定的算法要点

> **离线**：对 512 个专家的输出统计逐通道 $\mu_i,\sigma_i$，构造有向替代期望平方误差矩阵 $D_{ij}$。  
> **在线**：Router 产生 Top-8 索引和权重；查询 $D8$；第一轮选择加权总代价最小的源专家；接下来每轮选择使 $J(S)$ 下降最多的专家，并维护 `e_min`、`best_idx`；最多 8 轮，得到所有保留规模的顺序、映射和代理代价。  
> **推理**：只执行保留专家，使用逐通道仿射替代公式进行融合重建。  
> **数学依据**：非负且总和不超过 1 的 router 权重下，单 token 的加权替代平方误差和是最终输出**平方误差**的上界；离线 $D$ 则把该思想变为低开销的期望误差代理。  
> **关键边界**：代理的真实有效性最终由数据分布匹配和实测输出 MSE 决定。
