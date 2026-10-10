# MoE 专家补偿模块：一般映射算子下的守恒联合重构

算法设计说明，版本 1.0，2026-10-09。

本文整理一个适用于专家剪枝、预取缺失等场景的补偿框架。主算法保留一般映射算子 $T_{i,j}$，不预设标量、逐通道缩放或其他具体实现；参数默认通过离线校准获得，不要求端到端训练。

核心设计是：给定原始路由和实际可用的专家集合，将原始专家的权重分配给可用专家，通过联合最小化替代残差的平方范数确定分配系数。不同替代操作的误差交叉项是建模重点。

**研究状态：数学形式和求解流程已经明确，尚未通过真实模型实验证明优于 ExFold。一般算子 $T_{i,j}$ 的具体结构，以及与之匹配的紧凑误差统计存储方式，仍是实现选择。**

## 1. 范围与接口

模块解决的问题是：原始 MoE 需要执行一组专家，但因为剪枝、预取失误或其他预算限制，只能执行另一组专家，如何利用可用输出恢复原始 MoE 输出。

本文默认：

1. 原始专家参数和原始 router 固定。
2. 当前层的原始 router 已经运行，因此原始 Top-$k$ 集合及其权重已知。
3. 上游策略给出本层用于补偿的可用专家集合。
4. 在线阶段不计算、加载或查询缺失专家的真实输出。
5. 每个可用专家的输出最多计算一次，然后复用于多条替代边。
6. 分配系数允许为负；不默认施加非负约束。

预取预测器或剪枝策略只负责提供可用集合。它们不属于本文的主算法。预取 router 的概率可以用于选择可用集合，但不会直接覆盖原始 router 权重。

共享专家若存在，按原模型计算并添加到输出。本文仅分析需要替代的 routed experts。

## 2. 符号与索引方向

所有定义先针对单个 MoE 层；多层模型中的算子、校准统计和缓存分别保存，不混用不同层的数据。

| 符号 | 维度或类型 | 定义 |
| --- | --- | --- |
| $N$ | 正整数 | 本层 routed experts 总数 |
| $d$ | 正整数 | 专家输出维度 |
| $x$ | 当前层输入 | 一个 token 的 hidden state |
| $E_a(x)$ | $\mathbb R^d$ | 全局编号为 $a$ 的专家输出 |
| $p(x)$ | $\mathbb R^N$ | 原始 router 概率 |
| $\mathcal O(x)$ | 专家集合 | 原始 router 选择的 Top-$k$ 专家 |
| $\mathcal P(x)$ | 专家集合 | 实际用于补偿的 $k'$ 个可用专家 |
| $k,k'$ | 正整数 | 原始专家数和可用专家数 |
| $\alpha_j(x)$ | 标量 | 原模型实际赋给原始专家 $j$ 的 gate 权重 |
| $\alpha(x)$ | $\mathbb R^k$ | 按固定顺序排列的原始 gate 权重 |
| $y(x)$ | $\mathbb R^d$ | 原始 MoE 输出 |
| $T_{i,j}$ | $\mathbb R^d\to\mathbb R^d$ | 用可用专家 $i$ 的输出近似原始专家 $j$ 的映射算子 |
| $\Theta$ | 参数集合 | 所有映射算子的校准参数 |
| $M_{i,j}(x)$ | 标量 | 从原始专家 $j$ 分配给可用专家 $i$ 的权重 |
| $M(x)$ | $\mathbb R^{k'\times k}$ | 当前两个集合之间的分配矩阵 |
| $r_{i,j}(x)$ | $\mathbb R^d$ | 替代残差 $T_{i,j}(E_i(x))-E_j(x)$ |
| $q=kk'$ | 正整数 | 当前候选替代边数 |
| $m(x)$ | $\mathbb R^q$ | 按列展开的 $M(x)$ |
| $R(x)$ | $\mathbb R^{d\times q}$ | 按相同顺序排列的残差列矩阵 |
| $G(x)$ | $\mathbb R^{q\times q}$ | 真实残差 Gram 矩阵 $R(x)^\top R(x)$ |
| $\widehat G_\sigma$ | $\mathbb R^{q\times q}$ | 当前集合组合的校准 Gram 估计 |
| $\sigma$ | 集合签名 | 当前有序集合组合 $(\mathcal O,\mathcal P)$ |
| $A$ | $\mathbb R^{k\times q}$ | 列守恒约束矩阵 |
| $\ell_{i,j}$ | 标量 | 单条替代边的校准误差分数 |
| $M_0,m_0$ | 与 $M,m$ 相同 | 一个可行的独立替代基线 |
| $\Gamma_\sigma$ | $\mathbb R^{q\times k}$ | 将 $\alpha$ 映射到 $m_0$ 的基线分配矩阵 |
| $\lambda$ | 非负标量 | 分配求解中的稳定正则化系数 |
| $Q_\lambda$ | $\mathbb R^{q\times q}$ | $\widehat G_\sigma+\lambda I_q$ |
| $Z$ | $\mathbb R^{q\times(q-k)}$ | $\ker A$ 的一组正交标准基 |
| $\mathcal D_{\mathrm{cal}}$ | 校准样本集 | 用于估计算子和误差统计的数据 |

**索引方向始终固定：$i$ 是可用的替代者，$j$ 是需要被近似的原始专家。因此 $T_{i,j}(E_i)\approx E_j$。**

矩阵 $M$ 的行对应 $\mathcal P$，列对应 $\mathcal O$。矩阵不是全局 $N\times N$ 表；它只覆盖当前两个集合。

实现时将集合规范排序为

$$
\mathcal O=(o_1,\ldots,o_k),\qquad
\mathcal P=(p_1,\ldots,p_{k'}).
$$

按列展开时，边 $(p_a,o_b)$ 位于第 $a+k'(b-1)$ 个位置。$m$、$R$、$G$、缓存和权重向量必须使用同一顺序。

本文主要关注 $k'\leq k$，特别是预取场景中的 $k'=k$。公式本身也允许更大的候选集合，但必须单独计入实际执行成本。

## 3. 原始输出与一般补偿输出

### 3.1 原始 MoE

原始输出定义为

$$
y(x)=\sum_{j\in\mathcal O(x)}\alpha_j(x)E_j(x).
$$

$\alpha_j$ 必须采用原模型真实执行时的 gate 权重。若原模型在 Top-$k$ 内归一化，则 $\sum_j\alpha_j=1$；若原模型不采用该归一化或存在 routing scale，则保留原规则，不人为改成概率和为一。

### 3.2 映射算子

一般映射为

$$
\widehat E_j^{(i)}(x)=T_{i,j}(E_i(x);\Theta).
$$

主理论只要求算子输出维度为 $d$，且在确定分配 $M$ 时算子参数固定。算子不必是标量或线性矩阵，也不需要可微。

必须保证算子在线计算只使用可获得的信息，不能访问 $E_j(x)$。本文默认 $T_{i,j}$ 的参数经校准后固定；额外输入依赖不是默认设置。

自替代固定为恒等映射：

$$
T_{a,a}(z)=z.
$$

“简单补偿算子”是实现上的成本要求，不是自动成立的性质。若某种 $T_{i,j}$ 的参数或计算量很大，应作为不同实现单独评估。

### 3.3 联合补偿输出

给定当前可用集合，定义

$$
\widehat y(x;M)=
\sum_{i\in\mathcal P(x)}
\sum_{j\in\mathcal O(x)}
M_{i,j}(x)\,T_{i,j}(E_i(x)).
$$

这里 $M_{i,j}$ 已经包含原始 gate 权重的分配份额。输出中不要再乘一次 $\alpha_j$。

例如，若将原始专家 $j$ 的全部贡献分配给可用专家 $i$，则 $M_{i,j}=\alpha_j$，对应贡献为 $\alpha_j T_{i,j}(E_i)$。

## 4. 权重守恒与基线分配

### 4.1 默认约束：每个原始专家的分配权重守恒

对每个 $j\in\mathcal O$，要求

$$
\boxed{\sum_{i\in\mathcal P}M_{i,j}(x)=\alpha_j(x).}
$$

按列展开后：

$$
\boxed{Am=\alpha,\qquad
A=I_k\otimes\mathbf 1_{k'}^\top.}
$$

只要 $k'\geq1$，全连接候选边下 $A$ 满行秩。允许的自由度为

$$
q-k=k(k'-1).
$$

默认不约束每个可用专家接收的总权重，也不强制它等于预取 router 的概率。这会引入额外假设，并可能限制重构能力。

默认 $M$ 是有符号的线性分配矩阵。若 $M_{i,j}<0$，它表示从一个替代贡献中减去一部分输出，因此不能将 $M$ 解释为概率转移矩阵。

### 4.2 独立替代基线

用校准误差表构造一个始终满足列守恒的参考分配。

对于每个原始专家 $j$，定义

$$
\pi_0(j)=
\begin{cases}
j, & j\in\mathcal P,\\
\displaystyle\arg\min_{i\in\mathcal P}\ell_{i,j},
& j\notin\mathcal P.
\end{cases}
$$

并令

$$
(M_0)_{i,j}=
\alpha_j\,\mathbf 1\{i=\pi_0(j)\}.
$$

于是

$$
Am_0=\alpha,\qquad
m_0=\Gamma_\sigma\alpha,\qquad
A\Gamma_\sigma=I_k.
$$

若算子是 ExFold 使用的标量形式，并采用其校准误差表，上述分配对应 ExFold 风格的独立最小误差替代。[1]

若 $T_{i,j}$ 使用其他形式，则 $M_0$ 是“同一算子下的独立替代基线”，不能直接称为原版 ExFold。

### 4.3 守恒约束的含义边界

这里守恒的是从每个原始专家分配出去的系数，不是补偿后输出的范数，也不保证最终有效 router 权重之和不变。

例如，若 $T_{i,j}(z)=\tau_{i,j}z$，最终有效权重为

$$
\beta_i=\sum_j M_{i,j}\tau_{i,j}.
$$

一般而言，$\sum_i\beta_i$ 不等于 $\sum_j\alpha_j$。这是因为算子本身已经改变了输出尺度。

如果还需要有效权重守恒，标量特例可额外加入

$$
\sum_{i,j}M_{i,j}\tau_{i,j}=\sum_j\alpha_j.
$$

这是一条新的线性约束，会改变可行域；原有 $M_0$ 未必满足它，因此必须重新检查可行性和基线比较条件。一般算子下未必存在单一的“有效标量权重”。

## 5. 从替代残差到整个 MoE 的误差

定义每条边的真实替代残差：

$$
r_{i,j}(x)=T_{i,j}(E_i(x))-E_j(x).
$$

由列守恒可得

$$
\begin{aligned}
\widehat y(x;M)-y(x)
&=\sum_{i,j}M_{i,j}T_{i,j}(E_i(x))
-\sum_j\alpha_j E_j(x)\\
&=\sum_{i,j}M_{i,j}
\left[T_{i,j}(E_i(x))-E_j(x)\right]\\
&=\boxed{\sum_{i,j}M_{i,j}r_{i,j}(x)}\\
&=\boxed{R(x)m}.
\end{aligned}
$$

因此真实输出误差平方为

$$
\boxed{
L_x(m)=\|\widehat y-y\|_2^2
=\|R(x)m\|_2^2
=m^\top G(x)m,
\qquad G(x)=R(x)^\top R(x).
}
$$

该恒等式不要求 $T_{i,j}$ 对专家输出是线性的。只要在求解 $m$ 时 $T$ 固定，输出对于分配系数就是线性的。

若删除守恒约束，上面的恒等式不再成立。此时实际误差为

$$
\widehat y-y=
R(x)m+
\sum_j\left[(Am)_j-\alpha_j\right]E_j(x).
$$

所以不能在取消 $Am=\alpha$ 后，仍然将 $m^\top Gm$ 当作完整输出误差；否则 $m=0$ 会虚假地产生“零残差”。

### 5.1 为什么只有一张单边 loss 表不够

用 $e,f$ 简记两条候选边，误差展开为

$$
m^\top Gm
=\sum_e m_e^2\|r_e\|_2^2
+2\sum_{e<f}m_e m_f\langle r_e,r_f\rangle.
$$

第一部分是单边误差，第二部分描述误差相互抵消或放大。

例如，一个缺失专家有两个可用替代者，二者的残差范数都为一。取等量分配：

| 残差关系 | 两条边的范数平方 | 等量分配后的真实误差平方 |
| --- | --- | --- |
| $r_1=r_2$ | 均为 1 | 1 |
| $r_1=-r_2$ | 均为 1 | 0 |

仅保存两个误差标量 1，无法区分这两种情况。交叉项分别是 $+1$ 和 $-1$。

**本文的“误差标量建模”指保存残差内积这些标量统计，而非只保存每条边的误差范数。**映射算子可以是标量、张量或其他简单形式；误差统计的粒度是另一项独立选择。

### 5.2 矩阵维度与正定性

当前有 $q=kk'$ 条候选边，所以

$$
G(x)\in\mathbb R^{q\times q}.
$$

若 $k=k'=8$，则 $G$ 是 $64\times64$，并非当前输入下的 $N\times N$ 专家矩阵。

若试图对全局所有 $N^2$ 条边保存完整的边—边 Gram 矩阵，则维度为 $N^2\times N^2$，共有 $N^4$ 个元素。本文不默认采用这种全局存储。

由于 $G=R^\top R$，它一定半正定，但不一定正定：

$$
G\succeq0,\qquad \operatorname{rank}G\leq\min(d,q).
$$

负的非对角元素是合理的，它们可以表示误差抵消。不要逐元素将 $G$ 截断为非负矩阵。

## 6. 实际使用的误差模型

在线无法获得缺失的 $E_j(x)$，因此真实 $G(x)$ 通常不可计算。部署时使用校准估计

$$
\widehat G_\sigma
\approx
\mathbb E\!\left[G(x)\mid \sigma(x)=\sigma\right].
$$

具体统计后端可以先采用不条件化的共同校准分布，也可以使用集合签名或少量上下文分组。应明确区分它们：

| 模型 | 含义 |
| --- | --- |
| $G(x)$ | 当前输入的真实残差 Gram，仅作离线 Oracle |
| $\widehat G_\sigma$ | 根据校准统计，为当前集合组合构造的 Gram |
| $\widehat G_{\sigma,g(x)}$ | 根据少量可在线获得的上下文分组构造的 Gram，可选扩展 |

静态估计仍然依赖：

1. 当前候选边涉及哪些专家；
2. 校准后的算子 $T_{i,j}$；
3. 校准数据分布及其权重。

它不只是“专家编号之间的固定相似度”。但这种依赖不代表它能够准确刻画每个输入的残差方向。

部署使用的估计应保持对称半正定。共同样本上的 Gram 构造，以及有效的共同二阶矩构造，可以保证这一点。若统计被任意拼接或存在较大估计问题，不能仅凭加入一个任意小的正则系数就假设目标仍然凸。

由于 $\alpha(x)$、集合和输入残差有关，简单使用无条件平均 Gram，不等价于精确优化真实部署分布上的期望误差。这是模型近似，需要由实验评估。

## 7. 联合分配的目标

### 7.1 无正则的原始目标

与前述讨论一致，理想问题为

$$
\boxed{
\min_m \frac12m^\top G(x)m
\quad\text{s.t.}\quad Am=\alpha(x).
}
$$

部署时将不可见的 $G(x)$ 替换为 $\widehat G_\sigma$：

$$
\min_m \frac12m^\top\widehat G_\sigma m
\quad\text{s.t.}\quad Am=\alpha.
$$

这是等式约束凸二次规划。半正定已经足够保证凸性，不要求 $G$ 正定。[2]

### 7.2 推荐的稳定版本：正则化到可行基线

由于系数有符号、误差模型不精确且 Gram 可能奇异，采用

$$
\boxed{
\min_m
F_\lambda(m)=
\frac12m^\top\widehat G_\sigma m
+\frac{\lambda}{2}\|m-m_0\|_2^2
\quad\text{s.t.}\quad Am=\alpha.
}
$$

其中 $\lambda\geq0$。$\lambda=0$ 返回无正则目标；$\lambda>0$ 则抑制对参考替代方案的过大偏移。

这是对原始问题的稳定扩展，而不是声称正则化后的解仍等于无正则解。

选择 $m_0$ 为中心有两个好处：

1. 基线本身可行，且正则化代价为零，便于直接比较。
2. 当没有必要改变基线时，可以保留原来的分配，而不是仅为了减小系数范数重新分配。

$\lambda$ 的尺度应与误差 Gram 的量级匹配，在独立验证集上选择。固定 $\lambda$ 很大时趋近 $m_0$，很小时更接近无正则联合解。

## 8. 拉格朗日法与解析求解

本节将 $\widehat G_\sigma$ 简记为 $\widehat G$，并定义

$$
Q_\lambda=\widehat G+\lambda I_q.
$$

### 8.1 KKT 线性系统

拉格朗日函数为

$$
\mathcal L(m,\nu)=
\frac12m^\top\widehat Gm
+\frac{\lambda}{2}\|m-m_0\|_2^2
+\nu^\top(Am-\alpha),
$$

其中 $\nu\in\mathbb R^k$。

一阶条件为

$$
Q_\lambda m+A^\top\nu=\lambda m_0,\qquad
Am=\alpha.
$$

合并为

$$
\boxed{
\begin{bmatrix}
Q_\lambda&A^\top\\
A&0
\end{bmatrix}
\begin{bmatrix}
m^\star\\
\nu^\star
\end{bmatrix}
=
\begin{bmatrix}
\lambda m_0\\
\alpha
\end{bmatrix}.
}
$$

当 $\widehat G\succeq0$、$\lambda>0$ 且 $A$ 满行秩时，解唯一。

### 8.2 $\lambda>0$ 时的闭式表达

定义

$$
b_\lambda=Q_\lambda^{-1}(\lambda m_0),\qquad
U_\lambda=Q_\lambda^{-1}A^\top,\qquad
S_\lambda=AU_\lambda.
$$

对应维度分别为 $q$、$q\times k$、$k\times k$。此时 $Q_\lambda$ 和 $S_\lambda$ 都正定，得到

$$
\boxed{
m^\star=
b_\lambda+
U_\lambda S_\lambda^{-1}
\left(\alpha-Ab_\lambda\right).
}
$$

这就是每次输入所需的解析分配。它是一组小型线性系统的解，不是逐元素的简单除法。

实现时用 Cholesky 分解和线性求解，不显式形成矩阵逆。

### 8.3 $\lambda=0$ 的情况

如果 $\widehat G$ 正定，可以令上式 $\lambda=0$，得到

$$
m^\star=
\widehat G^{-1}A^\top
\left(A\widehat G^{-1}A^\top\right)^{-1}\alpha.
$$

但一般 Gram 可能奇异，不能直接使用该逆矩阵公式。此时应解 KKT 系统

$$
\begin{bmatrix}
\widehat G&A^\top\\
A&0
\end{bmatrix}
\begin{bmatrix}
m^\star\\
\nu^\star
\end{bmatrix}
=
\begin{bmatrix}
0\\
\alpha
\end{bmatrix}.
$$

若系统奇异，伪逆可给出一个最优解。不要机械地把正定公式里的每个逆都替换成伪逆；那不保证约束和最优性。

唯一解的条件是：$\widehat G$ 在 $\ker A$ 上正定；不必在整个空间正定。[2]

### 8.4 保持守恒的零空间形式

令 $Z$ 为 $\ker A$ 的正交标准基，因此

$$
AZ=0,\qquad Z^\top Z=I_{q-k}.
$$

由于 $m_0$ 可行，任意可行解都可写成

$$
m=m_0+Zt.
$$

代入目标，$\lambda>0$ 时得到

$$
\boxed{
m^\star=
m_0-
Z\left(Z^\top\widehat GZ+\lambda I_{q-k}\right)^{-1}
Z^\top\widehat Gm_0.
}
$$

这一形式将求解维度降低为 $q-k$，并且从构造上保持 $Am^\star=\alpha$。当 $\lambda=0$ 时，可使用该零空间系统的伪逆，得到一个最优分配；正交 $Z$ 下，该选择对应距离 $m_0$ 最近的最优解。

当 $k'=1$ 时，$q-k=0$，没有分配自由度。此时唯一可行分配为 $M_{1,j}=\alpha_j$，在固定算子下不能通过调整 $M$ 进一步改善。

## 9. 能保证什么，不能保证什么

### 9.1 对估计目标，不比参考分配更差

因为 $m_0$ 可行，

$$
F_\lambda(m^\star)\leq F_\lambda(m_0).
$$

所以

$$
\boxed{
(m^\star)^\top\widehat Gm^\star
+\lambda\|m^\star-m_0\|_2^2
\leq m_0^\top\widehat Gm_0.
}
$$

因此校准 Gram 定义的误差目标不会高于基线。对 $\lambda>0$，还得到

$$
\|m^\star-m_0\|_2
\leq
\sqrt{\frac{m_0^\top\widehat Gm_0}{\lambda}}.
$$

若使用真实 $G(x)$，上述不劣于基线的结论适用于当前输入的真实层输出误差。

### 9.2 对真实输入，估计误差会影响结论

记

$$
\Delta G(x)=G(x)-\widehat G.
$$

则

$$
\begin{aligned}
L_x(m^\star)-L_x(m_0)
&\leq
-\lambda\|m^\star-m_0\|_2^2\\
&\quad+
\|\Delta G(x)\|_2
\left(\|m^\star\|_2^2+\|m_0\|_2^2\right).
\end{aligned}
$$

这是一个说明敏感性的上界，不是部署时自动可知的证书。它说明：统计误差较大或系数范数较大时，代理目标的改进未必转化为真实误差改进。

因此不能预先宣称本模块在真实精度上优于 ExFold，也不能将层输出误差的改善直接等同于最终任务得分的改善。

## 10. 只有单边误差标量时的简化版本

若暂时只有各边的原始均方残差统计

$$
g_{i,j}\approx\mathbb E\|r_{i,j}(x)\|_2^2,
$$

可以采用对角近似

$$
\widehat G=\operatorname{diag}(g_e).
$$

注意：ExFold 风格的归一化 loss 表与 $g_{i,j}$ 不一定具有相同尺度。用于联合二次目标时应保留原始二阶统计，或明确记录归一化规则。

### 10.1 无正则解析分配

当所有 $g_{i,j}>0$、$\lambda=0$ 时，各列独立求解，得到

$$
\boxed{
M_{i,j}^\star=
\alpha_j
\frac{1/g_{i,j}}
{\sum_{u\in\mathcal P}1/g_{u,j}}.
}
$$

这将原始权重按逆误差比例分配给多个专家，比“只选择一个最小 loss 目标”更平滑。

如果存在零误差边，可将该列权重放在零误差边上，或使用稳定正则化，不直接计算 $1/0$。

### 10.2 带基线正则化的解析分配

令 $\rho_{i,j}=g_{i,j}+\lambda$，当 $\rho_{i,j}>0$ 时：

$$
\boxed{
M_{i,j}^\star=
\frac{\lambda(M_0)_{i,j}}{\rho_{i,j}}
+
\frac{1/\rho_{i,j}}{\sum_u1/\rho_{u,j}}
\left(
\alpha_j-\sum_u
\frac{\lambda(M_0)_{u,j}}{\rho_{u,j}}
\right).
}
$$

这是完整解析解，而不是计算后再归一化的经验规则。

### 10.3 简化版本的假设

对角模型忽略了所有跨边残差内积；只有这些交叉项为零或足够小时，才接近真实误差模型。

前文两个同方向残差的例子中，对角模型会错误地认为等量分配将误差平方降至 $1/2$，实际误差平方仍然是 1。

因此，对角版本适合作为第一阶段实现和消融基线。完整 Gram 版本是否有额外价值，应通过“同一 $T$、相同集合、对角统计与完整统计”的对比验证。

## 11. Training-free 校准流程

### 11.1 采集共同输入上的专家输出

收集校准 hidden states $x_t$，以及用于拟合和统计的专家输出。对于同一条残差或交叉项，涉及的专家必须处理同一个 $x_t$。

离线阶段可以计算额外专家；在线阶段仍然只执行可用集合。两者的计算预算必须分别报告。

只记录原始 Top-$k$ 输出可能覆盖不足，因为预取集合可能包含原始 Top-$k$ 外的专家。可在少量校准输入上探测全部专家，也可只探测实际候选组合涉及的专家。

应记录每条边的样本数。没有观测到一条边，不等于该边误差为零。

### 11.2 校准一般映射算子

给定一个待选择的低成本算子族 $\mathcal T$，逐边校准：

$$
T_{i,j}^{\mathrm{cal}}
\in\arg\min_{T\in\mathcal T}
\sum_t\omega_{i,j,t}
\|T(E_i(x_t))-E_j(x_t)\|_2^2
+\gamma_T\Omega(T).
$$

其中 $\omega_{i,j,t}\geq0$ 为算子拟合权重，$\gamma_T$ 为算子正则化系数，$\Omega$ 为相应正则项。它们与在线分配的 $\lambda$ 不同。

选择有闭式解的算子族时可以直接通过校准得到参数；没有闭式解时，也可做小规模离线拟合。框架不要求梯度训练，亦不以“可微”为主要贡献。

身份算子 $T_{a,a}=I$ 固定，不参与拟合。

### 11.3 单边 loss 表

使用校准后的算子计算独立替代分数，例如

$$
\ell_{i,j}=
\frac{
\sum_t\omega_{i,j,t}
\|T_{i,j}(E_i(x_t))-E_j(x_t)\|_2^2
}{
\sum_t\omega_{i,j,t}\|E_j(x_t)\|_2^2+\epsilon
}.
$$

这里 $\epsilon>0$ 防止除零。该表主要用于构造 $M_0$。

### 11.4 联合残差统计

对固定集合签名 $\sigma$，以及选定的共同校准样本，计算

$$
\boxed{
\widehat G_\sigma=
\sum_t w_t R_\sigma(x_t)^\top R_\sigma(x_t),
\qquad
w_t\geq0,\quad\sum_t w_t=1.
}
$$

也可以令样本集为该签名出现时的样本，以估计条件分布。算子拟合权重 $\omega_{i,j,t}$ 和 Gram 权重 $w_t$ 可以不同。

**构造同一个 Gram 时，必须对所有交叉项采用一致的共同样本和权重。**把不同边在不同样本上得到的统计任意拼接，可能破坏半正定性和统计含义。

不要仅保存平均残差向量

$$
\mu_{i,j}=\mathbb E[r_{i,j}],
$$

再用 $\mu_{i,j}^\top\mu_{u,v}$ 代替所需的二阶统计。一般有

$$
\mathbb E[r_{i,j}^\top r_{u,v}]
\ne
\mathbb E[r_{i,j}]^\top\mathbb E[r_{u,v}].
$$

### 11.5 验证与冻结

用独立验证数据选择 $\lambda$、统计分组方式和具体算子实现。随后冻结 $T$、误差统计和分配缓存。

如果修改了 $T$，则残差定义随之改变，必须重新计算相关 Gram 或通过有效的结构化二阶统计更新。旧 Gram 不能默认继续使用。

## 12. 误差统计如何存储

主理论允许一般 $T$，但一般算子并不自动带来低成本的全局统计表示。以下是三种实现后端；选择哪一种，要与 $T$ 的形式一起决定。

### 12.1 当前集合组合的局部 Gram

对观测到的签名 $\sigma$ 保存局部 $\widehat G_\sigma$，或直接保存求解后的映射缓存。

每个签名的 Gram 存储为 $O(q^2)$。例如 $k=k'=8$ 时只有 4096 个数，FP32 约 16 KiB。

局部大小小，不代表全部组合都可以枚举。签名数量可能很大，校准覆盖也可能不足。这种后端适合 Oracle、原型和有限签名实验。

对未覆盖签名，需要一个有定义的结构化统计后端，或回退到可行 $M_0$。回退率和数据覆盖必须报告；不能把未观测交叉项直接当零，也不能默认能够在线计算缺失输出来补齐统计。

### 12.2 可选结构：共享基础线性算子

如果最终选择的算子可以写成

$$
T_{i,j}(z)=
\sum_{\mu=0}^{B_T-1}c_{i,j,\mu}D_\mu z,
\qquad D_0=I,
$$

其中 $D_\mu$ 为少量共享基础算子，则可以保存基础输出的二阶统计，而不用保存全局边—边 Gram。

这不是主理论对 $T$ 的默认假设，也不是要求采用低秩补偿。它只是一个可以验证的结构化实现条件。

定义

$$
\mathcal U_a(x)=
[D_0E_a(x),\ldots,D_{B_T-1}E_a(x)]
\in\mathbb R^{d\times B_T},
$$

以及

$$
\mathcal K_{(a,\mu),(b,\nu)}
=
\mathbb E\left[
(D_\mu E_a(x))^\top(D_\nu E_b(x))
\right].
$$

完整 $\mathcal K$ 是 $NB_T\times NB_T$ 的半正定二阶矩矩阵。记其专家块为 $\mathcal K_{a,b}\in\mathbb R^{B_T\times B_T}$，$e_0=(1,0,\ldots,0)^\top$，则

$$
r_{i,j}=\mathcal U_i c_{i,j}-\mathcal U_j e_0,
$$

从而

$$
\begin{aligned}
(\overline G_\sigma)_{(i,j),(u,v)}
={}&c_{i,j}^\top\mathcal K_{i,u}c_{u,v}\\
&-c_{i,j}^\top\mathcal K_{i,v}e_0\\
&-e_0^\top\mathcal K_{j,u}c_{u,v}\\
&+e_0^\top\mathcal K_{j,v}e_0.
\end{aligned}
$$

统计存储为 $O(N^2B_T^2)$。只有 $B_T$ 足够小时才有意义；基础算子和补偿参数的存储也必须计入。

### 12.3 标量算子的实现示例

标量补偿是上面结构的特例，不是本框架的默认映射：

$$
T_{i,j}(z)=\tau_{i,j}z.
$$

固定自替代 $\tau_{a,a}=1$。一种加权最小二乘校准为

$$
\tau_{i,j}=
\frac{
\sum_t\omega_{i,j,t}
\langle E_i(x_t),E_j(x_t)\rangle
}{
\sum_t\omega_{i,j,t}\|E_i(x_t)\|_2^2+\gamma_T
}.
$$

定义共同分布上的未中心化二阶矩

$$
K_{a,b}=
\mathbb E[E_a(x)^\top E_b(x)],
\qquad K\in\mathbb R^{N\times N}.
$$

它是二阶矩，不是中心化协方差。由此精确构造该分布上的平均残差 Gram：

$$
\boxed{
(\overline G_\sigma)_{(i,j),(u,v)}
=
\tau_{i,j}\tau_{u,v}K_{i,u}
-\tau_{i,j}K_{i,v}
-\tau_{u,v}K_{j,u}
+K_{j,v}.
}
$$

这同时说明：

1. 不需要保存 $N^4$ 个统计量，保存 $K$ 和标量参数即可；
2. Gram 会随着补偿系数变化，并非只由专家编号决定；
3. 静态 $K$ 仍对应一个平均分布，不等于当前输入的真实 $G(x)$。

若从不完整的 co-routing 数据估计 $K$，不同条目可能具有不同条件分布；直接将它们拼接，不一定得到有效的共同二阶矩。应使用共同输入探测、明确的统计模型或有标注的近似方案。

### 12.4 可选的 training-free 上下文分组

如果无条件统计存在明显偏差，可以用在线已知量进行少量分组，例如原始 router 熵或缺失权重质量：

$$
\rho_{\mathrm{miss}}(x)=
\frac{
\sum_{j\in\mathcal O\setminus\mathcal P}\alpha_j(x)
}{
\sum_{j\in\mathcal O}\alpha_j(x)
}.
$$

对不同分组分别校准二阶统计。这里没有根据输入预测完整 $q\times q$ 矩阵的神经网络，但存储、样本量和组间泛化问题仍需要验证。

分组扩展不是首个版本的必要部分。先确定无条件或局部统计是否已有收益。

## 13. 在线推理与缓存

### 13.1 在线步骤

~~~text
输入：
    当前 hidden state x
    原始专家集合 O 及 gate 权重 alpha
    当前可用专家集合 P
    已校准的映射算子 T、单边 loss 表和联合统计后端

1. 对 O、P 规范排序，保持 alpha 和边顺序一致。
2. 若 O 完全包含于 P，直接执行原始专家和权重。
3. 用可用集合和 loss 表构造可行参考分配 m0。
4. 获得当前签名的 Gram 估计，或命中预先计算的分配缓存。
5. 求解带列守恒的分配 m_star。
6. 每个实际需要的可用专家只执行一次。
7. 应用 T，将各边的映射输出按 M_star 加权求和。
8. 加回原模型的共享专家输出（若存在）。
~~~

若统计后端没有可靠估计，可以使用 $m_0$ 作为回退分配。该回退不是重新加载缺失专家；其误差仍需统计。

### 13.2 静态统计下的线性缓存

对固定 $\sigma$、$T$、$\widehat G_\sigma$、$\lambda>0$，有

$$
m_0=\Gamma_\sigma\alpha.
$$

因此可以离线求解

$$
\begin{bmatrix}
Q_\lambda&A^\top\\
A&0
\end{bmatrix}
\begin{bmatrix}
W_\sigma\\
V_\sigma
\end{bmatrix}
=
\begin{bmatrix}
\lambda\Gamma_\sigma\\
I_k
\end{bmatrix},
$$

其中 $W_\sigma\in\mathbb R^{q\times k}$，$V_\sigma\in\mathbb R^{k\times k}$。在线直接使用

$$
\boxed{m^\star(x)=W_\sigma\alpha(x).}
$$

缓存命中时，分配本身只需 $O(qk)$ 运算。此结论不包括专家执行和映射算子应用的开销。

首次遇到签名时，只有在统计后端能构造该签名的 $\widehat G_\sigma$ 时才能计算并缓存 $W_\sigma$。缓存不会自动解决未观测组合的统计问题。

修改算子、统计、正则化系数、候选边或约束时，相关缓存失效。采用上下文分组时，缓存键还应包含该统计分组。

### 13.3 实际复杂度

| 部分 | 主要规模 |
| --- | --- |
| 活跃分配变量 | $q=kk'$ |
| 当前 Gram | $q\times q$ |
| 直接 KKT 求解 | $(q+k)\times(q+k)$ |
| 正则版本的 Cholesky 与 Schur 求解 | 一般为 $O(q^3+q^2k+k^3)$ |
| 零空间求解 | $(q-k)\times(q-k)$ |
| 缓存后的分配 | $O(qk)$ |
| 一般映射应用 | 最多 $q$ 条边的 $T$ 计算 |
| 可用专家执行 | 至多 $k'$ 次专家调用 |

固定 $k,k'$ 后，分配求解维度不会因为全局 $N$ 变大而增长。$N$ 仍影响算子参数、校准覆盖及统计存储。

小矩阵并不自动意味着低实际延迟。特别在 batch=1 时，应测量 Gram 构造、分解、同步、映射与聚合的总开销。

一般 $T$ 不一定能融合进 router 权重；也不一定能把多个 $T_{i,j}$ 的计算完全合并。

## 14. 与标量权重重分配的关系

本节仅适用于标量算子 $T_{i,j}(z)=\tau_{i,j}z$。

定义

$$
\beta_i=\sum_jM_{i,j}\tau_{i,j},
$$

则

$$
\widehat y=\sum_{i\in\mathcal P}\beta_iE_i(x).
$$

所以 $q$ 个边变量最终只决定 $k'$ 个有效输出系数；不同 $M$ 可以产生同一输出。单靠输出损失，不能认为学到了唯一的专家替代关系。

为了精确表达这个冗余，定义 $B_\sigma\in\mathbb R^{k'\times q}$，使

$$
\beta=B_\sigma m.
$$

第 $(i,j)$ 条边对应的列，只在可用专家 $i$ 的行上取值 $\tau_{i,j}$。

因为 $m=m_0+Zt$，有效系数的真实可行域为

$$
\boxed{
\beta\in
\beta_0+\operatorname{range}(B_\sigma Z),
\qquad
\beta_0=B_\sigma m_0.
}
$$

直接优化任意 $\beta\in\mathbb R^{k'}$，不自动等价于带守恒的 $M$ 问题。若想严格降维，必须保留上述可行域；有正则化时，还需要保留由 $\|m-m_0\|^2$ 诱导的代价。

在共同校准二阶矩 $K$ 下，输出重构误差可写为

$$
\beta^\top K_{\mathcal P,\mathcal P}\beta
-2\beta^\top K_{\mathcal P,\mathcal O}\alpha
+\alpha^\top K_{\mathcal O,\mathcal O}\alpha.
$$

该表达适合构造直接权重调整的强基线，也可能用于未来的精确降维实现。

在固定标量算子下，各残差都是 $\mathcal O\cup\mathcal P$ 中专家输出的线性组合。瞬时 Gram 和采用同一固定系数的平均 Gram 均满足

$$
\operatorname{rank}G
\leq|\mathcal O\cup\mathcal P|.
$$

这进一步说明不能普遍假设 Gram 正定。引入张量补偿不是解决 $M$ 冗余的必要步骤；可以通过正则化、约束或合法降维处理。

标量实现时，计算 $\beta$ 后直接使用它，不能再经过 softmax、裁剪负值或无说明地重新归一化，否则不再是上述最优解。

## 15. 保留共同专家的可选版本

统一主模型允许重新分配所有原始专家的贡献。若希望已经可用的原始专家保持原权重，令

$$
\mathcal C=\mathcal O\cap\mathcal P,\qquad
\mathcal J=\mathcal O\setminus\mathcal P.
$$

对 $j\in\mathcal C$ 固定

$$
M_{j,j}=\alpha_j,\qquad M_{i,j}=0\quad(i\ne j).
$$

只对 $j\in\mathcal J$ 求解分配：

$$
\widehat y=
\sum_{j\in\mathcal C}\alpha_jE_j(x)
+
\sum_{i\in\mathcal P,j\in\mathcal J}
\widetilde M_{i,j}T_{i,j}(E_i(x)).
$$

由于共同专家的自替代残差为零，

$$
\widehat y-y=
\sum_{i\in\mathcal P,j\in\mathcal J}
\widetilde M_{i,j}r_{i,j}(x).
$$

使用缺失列构造的 $\widetilde R,\widetilde G,\widetilde A$，其余公式不变。变量数变为 $k'|\mathcal J|$，自由度为 $|\mathcal J|(k'-1)$。

该版本更接近独立替代基线的行为，也便于隔离“补偿缺失贡献”的作用。它是额外约束，不保证优于允许共同专家调整的主模型，应单独消融。

当 $\mathcal J=\varnothing$ 时直接返回原始输出，跳过分配求解。

## 16. 其他约束与可微训练的位置

### 16.1 非负分配

若加入 $m\geq0$，问题仍是凸二次规划，但通常不能使用前述等式约束闭式解。需要非负 QP 或活跃集方法。

若等式约束解恰好非负，则它也是非负约束问题的最优解。将负系数裁剪后重新归一化，一般不是非负 QP 的最优解。

本文默认保留有符号版本，并用正则化和实际质量实验判断稳定性。非负版本作为消融，不作为主流程前提。

### 16.2 限制候选替代边

可根据校准覆盖或额外先验屏蔽某些边。但对每个有非零原始权重的专家，至少需要一条可用边，否则守恒约束不可行。

重新构造候选变量和约束矩阵后，KKT 方法仍适用。应检查约束独立性，不能直接沿用未屏蔽图的缓存。

### 16.3 为什么不直接套普通 Sinkhorn

普通熵正则化运输优化针对固定线性边成本和非负质量分配。[4] 本文目标包含二次交叉项，且默认允许负系数，仅要求列守恒。

因此两者不是同一个优化问题。区别并非“一个只能给出 0/1 匹配，一个能给出连续权重”。主算法使用等式约束二次求解，无需 Sinkhorn。

### 16.4 可微扩展不是必要条件

$\lambda>0$ 时，线性求解可以在适当条件下对输入矩阵和参数求导，也可以集成到优化层。[3]

但当前设计先通过离线校准固定 $T$ 与统计，再在线求分配。不需要为满足“可微模块”的形式而引入神经网络。

若未来学习 $T$ 或统计估计器，必须同步更新残差 Gram，并在固定专家与相同信息条件下评估。该扩展不属于首个版本的必要工作。

## 17. 实验设计与停止条件

### 17.1 先测同一算子下的改进上限

固定 $T$、原始 router 和可用集合，在独立样本上比较：

1. 独立最小 loss 分配 $m_0$；
2. 对角误差模型的解析分配；
3. 校准完整 Gram 的联合分配；
4. 使用真实 $G(x)$ 的 Oracle 联合分配。

Oracle 只用于离线诊断，不是可部署方案。比较时保留相同守恒、共同专家固定策略和符号约束；明确报告是否使用正则化。

若无正则 Oracle 相比独立替代也几乎没有改善，应停止扩展固定算子下的这一类分配策略。

若 Oracle 改善明显，但校准 Gram 没有改善，重点问题是统计估计，不宜先增加补偿算子复杂度。

### 17.2 下游质量与实际场景

至少验证：

- 实际预取预测器产生的可用集合；
- 固定预算的剪枝或 token 级专家缩减；
- 不同缺失专家数及原始缺失权重质量；
- 有符号与非负分配；
- 保留共同专家与允许共同专家调整。

随机屏蔽只作为补充，不替代真实预取误差。

层误差可采用

$$
\mathrm{NMSE}(x)=
\frac{\|\widehat y(x)-y(x)\|_2^2}
{\|y(x)\|_2^2+\epsilon},
$$

并报告均值和尾部统计。最终质量必须通过补偿模型自己的自回归生成验证，不能只在原模型生成轨迹上测局部误差。

### 17.3 分离算子收益与分配收益

如果新实现采用与 ExFold 不同的 $T$，需要两类比较：

| 比较 | 回答的问题 |
| --- | --- |
| 原版 ExFold 与完整新模块 | 整体设计是否有收益 |
| 同一 $T$ 下，独立替代与联合分配 | 收益是否来自误差耦合与分配策略 |

标量实现还需比较直接调整 $k'$ 个有效权重的简单方法。训练数据、在线信息和开销应尽量匹配，避免把更大的预算当成建模收益。

### 17.4 必要的开销证据

模块论文可以不实现完整 offloading 调度系统，但仍应报告：

- 映射参数与统计存储，包含所有 MoE 层；
- 校准额外专家调用数、时间和峰值显存；
- 在线统计构造、求解、映射、聚合的总延迟；
- batch=1 与代表性 prefill batch 的开销；
- 实际执行专家数、签名缓存命中率和回退率。

所有权重常驻 GPU 的受限执行实验可验证补偿质量和模块开销，但不能据此宣称真实 CPU—GPU offloading 的端到端加速。

优先目标是在相同质量下允许更少专家或更严重的预取缺失，并保持补偿开销可接受。

## 18. 首个版本的实施顺序

1. 固定一类可校准的简单 $T$，先不增加输入条件网络或端到端训练。
2. 使用真实可用集合，对小规模层回放构造真实残差与 Oracle。
3. 比较独立替代、对角分配、完整 Gram 分配的误差上限。
4. 选择与该 $T$ 一致的可部署统计表示，冻结校准参数。
5. 实现有符号、列守恒、基线中心正则化的解析求解。
6. 验证下游自回归质量与模块总开销。
7. 只有收益稳定后，再扩展算子、条件统计和更多场景。

首个版本不需要枚举所有集合，也不需要训练大型网络预测 $G(x)$。但它必须明确解决已覆盖签名的统计获得方式，并公开未覆盖情况的处理与回退。

## 19. 公式的数值自检

本文公式已用 80 组随机合成数据进行 NumPy/SciPy 自检，采用一般逐通道映射检验主推导，并另外检验标量及共享基础算子特例。另用 40 组数据检验对角闭式解和无正则奇异 Gram 的零空间求解。

检验内容包括：

- 守恒条件下，直接输出差与 $Rm$ 一致；
- KKT 求解、Schur 闭式表达与零空间表达一致；
- 正则版本满足列守恒和估计目标不劣于基线；
- 缓存 $W_\sigma\alpha$ 与逐次求解一致；
- 标量二阶矩公式与直接平均残差 Gram 一致；
- 共享基础算子的二阶矩公式与直接构造一致。

所有一致性检查的最大绝对误差小于 $3\times10^{-13}$，守恒误差小于 $5\times10^{-16}$。

这些检查只验证代数和数值实现的一致性，不构成真实模型精度、性能或优于 ExFold 的证据。

## 20. 尚未确定的实现选择

| 项目 | 本文已经确定的部分 | 待实验或实现选择 |
| --- | --- | --- |
| 映射算子 | 保留一般 $T_{i,j}$，自替代为恒等 | 标量、通道或其他简单算子 |
| 分配变量 | 标量 $M_{i,j}$，可多对多、可有符号 | 是否进一步降维 |
| 原始权重 | 每个原始专家的列分配守恒 | 是否额外要求有效权重守恒 |
| 误差目标 | 完整残差平方范数及交叉内积 | 对角近似是否已足够 |
| 求解 | 等式约束 QP，解析线性系统 | 正则强度与实际计算后端 |
| 参数获得 | 离线校准为主 | 是否需要小规模拟合或训练 |
| 输入依赖 | 不默认预测完整 $G(x)$ | 条件分组是否有价值 |
| 统计存储 | 当前 Gram 小，全局边 Gram 不枚举 | 与最终 $T$ 一致的结构化后端 |
| 共同专家 | 统一模型允许重新分配 | 保留原权重是否更稳 |
| 相对 ExFold 的收益 | 代理目标中的可证明不劣关系 | 真实质量与开销必须验证 |

## 参考资料与来源边界

[1] Wu et al., *ExFold: Unified Expert Folding for Training-Free MoE Prefill-Decode Acceleration*, arXiv:2608.24938v1. [论文](https://arxiv.org/html/2608.24938v1)。

本文引用其逐专家最小 loss 替代与标量投影作为基线。一般 $T$、有符号联合分配、基线中心正则化及本文各实现选择属于本设计，不是 ExFold 已验证的结论。

[2] Boyd and Vandenberghe, *Convex Optimization*, equality-constrained optimization lecture. [官方讲义](https://web.stanford.edu/class/ee364a/lectures/equality.pdf)。

该来源支持等式约束凸二次问题的 KKT 求解与唯一性条件。本文在此基础上推导具体的 MoE 残差分配形式。

[3] Amos and Kolter, *OptNet: Differentiable Optimization as a Layer in Neural Networks*, ICML 2017. [论文](https://proceedings.mlr.press/v70/amos17a.html)。

仅用于说明优化求解可以作为可微层；本文不以可微训练为前提。

[4] Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances*, NeurIPS 2013. [论文](https://arxiv.org/abs/1306.0895)。

用于区分熵正则化线性运输问题与本文的有符号等式约束二次问题。
