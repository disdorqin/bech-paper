# HCH v2 新损失函数：数学研究任务书与可复制提示词

> 文档版本：v0.1  
> 日期：2026-08-11  
> 用途：直接交给一个独立“数学统计 + 时间序列 + 电力市场”研究窗口  
> 当前阶段：**只研究、推导、证伪与设计，不改代码、不跑正式实验**  
> 目标仓库：https://github.com/disdorqin/bech-paper  
> 当前审计基线：`a3770bf813d56b2c597cd7917ee57fd46a8f654f`

---

## 0. 直接给新窗口的角色与总任务

你是一名同时熟悉概率分布、稳健统计、时间序列预测、决策理论与电力市场价格的数学研究者。请为一个**冻结宿主预测器的双向后处理模块**设计有明确生成假设、可完整推导、可稳定优化、能产生向上/向下候选动作并与动作价值路由相接的新目标。

项目不接受“为了好看而把 MSE、tail loss、gate loss 用若干 λ 相加”作为主创新。也不接受“用了 Student-t / skew-t 就是创新”。需要从可证伪的数据假设出发，推导概率模型、正规化密度、负对数似然、方向性部分矩或其他合法决策量，再说明它为何适合尖峰、低价/负价与冻结宿主残差。

请先验证文献和数学，再选择最小可行设计。不要迎合下方候选公式；若其假设、正规化、矩或数值稳定性不成立，请明确推翻并给出更优方案。

---

## 1. 项目背景

### 1.1 研究对象

目标是电力市场价格预测的模型无关后处理。给定已训练并冻结的宿主：

$$
\hat y^{host}_{d,h}=f_{host}(x_{\le d,h}),
$$

HCH v2 不重训宿主，而是利用宿主输出、过去可见价格/残差、日历和可用外生预测，生成：

$$
\hat y^{(0)}=\hat y^{host},\qquad
\hat y^{(-)}=\hat y^{host}+\Delta^{-},\qquad
\hat y^{(+)}=\hat y^{host}+\Delta^{+}.
$$

最终门控从 Identity、Down、Up 三个动作中路由。项目希望同时修正：

- 高价尖峰的漏判与幅值压缩；
- 自适应低价/低谷；
- 有物理负电价的市场中的负价事件；
- 正常期中不值得修正的点。

### 1.2 当前架构

1. **共享连续状态**：用过去可见信息表示低—正常—高的连续位置及强度，不离散切分类别。
2. **Bi-OMC**：Bidirectional Occurrence–Magnitude Correction，同时产生 Down/Up 候选；这是双尾结构增强。
3. **CAGM**：以一天 24 小时为一个 memory episode；当前情景形成 day key，检索过去情景及其 24×3 动作收益。
4. **DVG**：估计执行候选相对 Identity 的风险调整价值；没有正价值时保持冻结宿主。

论文主线是 CAGM-DVG，Bi-OMC 是双尾结构增强，共享连续状态不是第三个独立创新点。

### 1.3 数据与宿主

公开市场包括 NEM、LAGO 系列欧洲市场等；私有山东市场含日前价、实时价以及日前可获得的辅助预测，例如负荷、风电、光伏、联络线、核电、竞价空间、新能源等。山东不可公开，只作为外部真实场景；公开市场承担可复现主证据。

冻结宿主包括 Linear、MLP、LSTM、TCN、PatchTST。同行后处理比较固定为 Identity、Residual-L1、QuantileResidual-LGBM、PIR、δ-Adapter Ada-Y 与 HCH v2。

### 1.4 数据阶段

- S1：宿主训练与历史统计；
- S2：HCH 训练和时间 OOF；
- S3：memory 构建与门控校准；
- S4：完全冻结测试。

任何分布选择、参数拟合或阈值选择都不能查看 S4。

---

## 2. 当前代码状态：不要把 smoke 当成方法证据

代码正在另一条线修复。当前审计已发现：

- Bi-OMC 训练/记忆候选曾错误地相对 latent `z` 而非 `host_pred`；
- 连续状态头没有有效监督，且未真正进入 candidate/key；
- OOF fold 的 key 坐标系混用且 metric projection 重复；
- DVG 的 `k/eta/tau` 硬编码、S3 自检索风险；
- 外生 token 的 mask、类型身份、S1 规范化和 learned-null 不完整；
- memory/bundle 不足以证明冻结；
- 部分 S4 时间戳错位；
- 当前 v2 smoke 不能支持创新或 SOTA 主张。

因此你不能从现有 smoke 的最终 HCH 指标反推损失有效。你可以使用下节的宿主残差统计作为**探索性假设生成**，但必须设计独立的 S2/S3 诊断来证伪。

---

## 3. 已有残差探索：只能当先验线索

已有缓存对 18 个 dataset/channel × host 组合粗略比较了 Normal、Laplace、Student-t、Skew-normal 的拟合倾向。探索性结果为：

- Student-t 在这四个简单候选中 18/18 最优；
- 残差 excess kurtosis 中位数约 10.3；
- 残差绝对 skewness 中位数约 0.63。

代表性 Linear 宿主残差：

| 数据/通道 | skewness | excess kurtosis | 探索性现象 |
|---|---:|---:|---|
| NEM | 12.265 | 178.0 | 极端右偏、极重尾；skew-t 倾向明显 |
| LAGO_DE | -0.231 | 7.94 | 近对称重尾；t 与 skew-t 接近 |
| LAGO_NP | 8.595 | 266.35 | 少量异常造成巨大右偏；skew-t 仅边际改善 |
| DE_EPEX | -0.455 | 7.70 | 左偏重尾；Laplace 曾窄幅领先 |
| Shandong DA | 0.627 | 3.82 | 中度右偏重尾；skew-t 边际改善 |
| Shandong RT | 0.582 | 2.61 | 中度右偏；skew-t 有倾向 |

这些数值尚未经过统一 S2-only 预处理、block bootstrap、按状态条件分析或跨 seed 验证。它们不能证明：

- 所有市场都服从同一种 skew-t；
- 单一全局自由度足够；
- 独立逐小时似然成立；
- skew-t 本身是新方法；
- skewness 来自稳定机制而非少量结构突变。

你的第一项工作是把这些可疑点变成可检验假设。

---

## 4. 必须先纠正的文献前提

用户提出“从分布 PDF 推导 NLL”这个方向是合理的，但以下归属不可直接沿用：

- 原始 TFT 主要使用 quantile loss，不应在未核实前写成“由 Student-t PDF 推导损失”：  
  https://arxiv.org/abs/1912.09363
- 原始 N-BEATS 的主评价/训练叙述不是一个新的 Student-t NLL；请核对原文：  
  https://arxiv.org/abs/1905.10437
- DeepAR 等概率预测工作早已使用参数分布/似然；Student-t likelihood 不是新颖点：  
  https://arxiv.org/abs/1704.04110
- Fernández–Steel skewing 是成熟分布构造，不可冒充本项目原创：  
  https://doi.org/10.1080/01621459.1998.10473708
- generalized asymmetric Student-t 也已有成熟统计文献；请核查其参数化、矩与 identifiability：  
  https://doi.org/10.1016/j.jeconom.2009.11.001

请进一步检索并核实用户提到的“Asymmetric Student-t Loss for TSF”具体论文、版本、公式和任务。所有引用给 DOI/arXiv/官方论文页，区分发表论文、预印本和代码。禁止凭标题印象归因。

---

## 5. 核心研究问题

我们不只是要一个更鲁棒的点预测 loss，而是要同时回答：

1. 冻结宿主残差为何重尾、偏斜、可能随状态改变？
2. 如何从同一概率假设中得到 Down/Up 的 occurrence 与 magnitude，而非手工加两个分类/回归 loss？
3. 如何令两个方向共享知识，同时允许上尾尖峰与下尾负价机制不完全对称？
4. 如何输出候选动作的风险/不确定性，供 DVG 判断“校正是否有价值”？
5. 如何避免分布风险与 CAGM 历史 gain 风险重复计算？
6. 逐小时条件似然是否足够，还是必须建模 24h path dependence / 事件时序偏差？
7. 复杂分布相对简单 Student-t 是否有可辨识的增益，而不是参数冗余？
8. 设计在无物理负价市场是否仍能通过 Adaptive Low 学习，并逐步把共享信息迁移到稀少物理负价？

---

## 6. 待验证的最小数学候选：状态条件 Fernández–Steel skew-t

这只是起点，不是已决定答案。

令冻结宿主残差：

$$
R_{d,h}=y_{d,h}-\hat y^{host}_{d,h}.
$$

令 $Z_{d,h}$ 为预测时可见的连续状态、宿主误差画像、日历与外生上下文。候选假设：

$$
R\mid Z \sim \operatorname{FS\text{-}t}
\big(\mu(Z),\sigma(Z),\nu(Z),\gamma(Z)\big),
$$

参数约束可设：

$$
\sigma=\operatorname{softplus}(a_\sigma)+\epsilon,\qquad
\nu=2+\operatorname{softplus}(a_\nu),\qquad
\gamma=\exp(a_\gamma).
$$

令 $z=(r-\mu)/\sigma$，一个待你严格核验的 Fernández–Steel 形式为：

$$
f(r\mid Z)=
\frac{2}{\sigma(\gamma+\gamma^{-1})}
\begin{cases}
t_\nu(\gamma z), & z<0,\\
t_\nu(z/\gamma), & z\ge 0.
\end{cases}
$$

请从变量替换开始证明：

- 它对 $r$ 的积分是否为 1；
- $\mu$ 是 mode、location 还是 mean；
- $\sigma$ 与 $\gamma$ 的尺度是否混淆；
- $\gamma=1$ 时是否严格退化为对称 Student-t；
- $\nu\to\infty$ 的极限是什么；
- 一阶/二阶矩存在条件；
- 神经网络同时预测 $\sigma,\nu,\gamma$ 时是否可辨识。

不能在证明前直接写代码。

---

## 7. occurrence–magnitude 的解析连接

理想目标不是另加二分类器，而是从条件分布计算：

$$
\pi_-(Z)=P(R<0\mid Z),\qquad
\pi_+(Z)=P(R>0\mid Z),
$$

$$
m_-(Z)=\mathbb E[-R\mid R<0,Z],\qquad
m_+(Z)=\mathbb E[R\mid R>0,Z].
$$

方向性一阶部分矩：

$$
M_-(Z)=\mathbb E[R\mathbf 1(R<0)\mid Z]=-\pi_-m_-\le 0,
$$

$$
M_+(Z)=\mathbb E[R\mathbf 1(R>0)\mid Z]=\pi_+m_+\ge 0.
$$

一个候选动作定义是：

$$
\Delta^-(Z)=M_-(Z),\qquad
\Delta^+(Z)=M_+(Z).
$$

这自然得到“发生概率 × 条件幅值”，并让两方向共享同一条件分布。但它可能因 $\pi$ 相乘而过度收缩，也可能不等价于给定 DVG 动作语义下的 Bayes-optimal correction。

你必须：

1. 对一般 $\mu\ne0$、阈值固定在残差 0 的情形，推导 $\pi_\pm$ 与 $M_\pm$；
2. 给出 Student-t / FS skew-t 下可计算的 CDF 与部分矩；
3. 若无闭式，给稳定可微的数值方案和误差界/验证；
4. 比较 $M_\pm$、条件均值 $\pm m_\pm$、条件中位数和受限 Bayes action；
5. 明确哪一个才符合“三动作中由 gate 选择”的决策问题；
6. 给出每种选择的 shrinkage、极端残差、$\pi\to0/1$ 行为；
7. 说明训练时为何不会让两个候选坍缩为普通条件均值残差。

如果 partial moment 不是合适的动作，请推翻它，但仍需给出从同一生成模型导出的替代，而不是回到任意加权 loss。

---

## 8. 需要比较的分布设计

至少比较下列嵌套层级，不能一开始选最复杂者：

| 方案 | 能力 | 主要风险 |
|---|---|---|
| M0：对称 Student-t | 重尾、参数少 | 无方向非对称 |
| M1：FS skew-t | 共享 df/scale + skew | location/scale/skew 可辨识性 |
| M2：two-piece / generalized asymmetric t | 两侧 scale 或 df 不同 | 参数冗余、数值不稳 |
| M3：状态条件 mixture / regime-continuous t | 多机制 | mixture collapse、失去简洁推导 |
| M4：bulk + EVT tail | 物理极端外推 | threshold 引入、样本稀少、与“少硬阈值”冲突 |

需要给出模型选择原则，而非只比较训练 NLL：

- held-out S3 NLL；
- PIT/coverage；
- tail calibration；
- CRPS 或可计算近似；
- High/Low residual calibration；
- Candidate Oracle Gain；
- 参数稳定性与跨 seed 方差；
- 模型复杂度惩罚或嵌套检验。

若 M0/M1 已足够，优先简单方案。若 M2/M3 才能解决两尾异质性，必须用可证伪结果证明额外参数必要。

---

## 9. 连续状态、共享知识与稀少负价

### 9.1 不做离散 regime classifier

连续状态应进入分布参数，例如：

$$
(\mu,\log\sigma,\nu,\log\gamma)=g_\theta(Z,s),
$$

而不是先用硬阈值把价格切成 Low/Normal/High 再训练三个模型。

请研究：

- 状态是否应是条件 rank、robust scale、宿主置信/偏差画像或其低维组合；
- 如何保证状态在不同市场的尺度可比较；
- 如何让状态对参数的影响单调/平滑，但避免过强硬约束；
- 哪些约束可由 link function 或可识别性自然推出。

### 9.2 Up/Down 共享什么

项目希望共享“重尾强度、日周期、宿主错误画像和情景记忆”，但允许上尾和下尾具有不同偏度/尺度。请明确：

- 共享 backbone、共享 $\nu$、共享日 key 是否有统计依据；
- 哪些参数必须方向特异；
- 是否可用层级先验/partial pooling 把高尾丰富样本的信息传给低尾/负价；
- 这种传递何时会产生 negative transfer。

### 9.3 物理负价的正确定位

负价不是唯一训练标签。无负价市场也有 Adaptive Low，帮助学习“向下残差/低尾语义”；有物理负价市场再用少量事件识别特定幅度与符号。

请提出一个不靠单独 `y<0` 二分类器的数学机制，例如：

- 多市场层级参数；
- shared heavy-tail geometry + market-specific location/scale/skew；
- empirical-Bayes shrinkage；
- conditional distribution transport。

同时给出反例：上尾知识何时不应传给负价。

---

## 10. 24 小时日情节与时间依赖

当前 CAGM 把一天 24h 作为一个 memory episode，但候选分布可能仍逐小时分解：

$$
\mathcal L_{day}=-\sum_{h=1}^{24}\log f(R_{d,h}\mid Z_{d,h}).
$$

请判断这个条件独立近似是否足够。重点是项目关心尖峰召回、幅值与时序偏差，简单逐点 NLL 可能无法区分“幅值对但提前一小时”。

必须比较：

1. **W1 最小方案**：逐小时条件 AST，日内依赖由 encoder/CAGM 上下文吸收；
2. **W2 轻量联合方案**：共享 day-level latent scale/skew 或 multivariate-t low-rank dependence；
3. **W3 路径分布方案**：copula / energy score / path likelihood。

要求：

- 先说明 W1 在何种条件下是合法 composite likelihood；
- 若需要 W2，推导最小新增参数与训练目标；
- 不得仅为“考虑时序”任意再加 timing loss；
- W3 只有在 W1/W2 被 S2/S3 诊断明确推翻时才建议；
- 解释与 CAGM 24h memory 的分工，避免重复建模。

用户曾联想到 Pyraformer 的层级信息聚合。请区分“多尺度网络架构”与“统计知识共享/记忆”。除非数学上必要，不要因为类比卷积或金字塔而增加模块。

---

## 11. 与 CAGM-DVG 的决策接口

候选生成与动作路由必须分工清楚：

- 分布模块：估计残差两侧的发生、幅值与不确定性，生成 candidate；
- CAGM：从相似历史日检索这些 candidate 的 realized action gain；
- DVG：相对 Identity 选择风险调整价值最高的动作。

历史 realized gain：

$$
G^a=\ell(y,\hat y^{host})-\ell(y,\hat y^{(a)}),\qquad G^0=0.
$$

请研究：

1. NLL 产生的尺度/尾风险如何成为 DVG feature，而不是重复当作第二个任意 loss；
2. CAGM 的邻居 gain 方差与分布预测方差是否包含相同风险；
3. 如何得到不重复计算的风险分解（aleatoric / retrieval epistemic / candidate error）；
4. DVG 的 Bayes decision 或效用形式是什么；
5. Identity 动作值为 0 时，soft routing/abstain 如何由模型推出；
6. 若使用 CARA、CVaR 或其他风险效用，必须说明电价修正任务的假设与单位尺度；
7. DVG 是否仍需 `eta/tau`，还是新分布可减少经验门控参数。

不要把“校正动作概率”误称为预测区间的共形覆盖；本任务不自动引入共形方法。

---

## 12. 优化、数值稳定与实现约束

必须给出实现级答案：

- 完整 NLL，不能漏正规化常数；
- 对 $\mu,\sigma,\nu,\gamma$ 的梯度或可验证自动微分表达；
- $\log\Gamma$、`log1p`、tail CDF/partial moment 的稳定计算；
- $\nu\downarrow2$、$\gamma$ 极端、$\sigma\to0$ 时的处理；
- 初始化与参数范围是否需要软限制；
- 32-bit 与 mixed precision 的风险；
- 缺失 horizon mask 下日 loss 如何归一化；
- 不同市场价格单位/尺度下如何保持 loss 可比较；
- 是否需要 stop-gradient；如需要，给决策理论理由；
- 避免单个极端点支配 $\nu/\gamma$ 学习的办法；
- 每项数值公式的单元测试 oracle（高精度积分或 Monte Carlo）。

复杂度不是第一约束。如果神经网络参数化显著提升有效性可以采用；若效果相当，才优先简单、低计算方案。

---

## 13. 必须完成的数学任务

### T1. 假设与证伪

列出分布、条件独立、平稳/非平稳、跨市场共享和缺失机制假设。每个假设给 S2/S3-only 检验及失败后的最小降级方案。

### T2. 正规化与 NLL

从基础对称密度和变量替换推导 FS/two-piece 密度、CDF、log-likelihood；不允许“显然可得”跳步。

### T3. 矩与部分矩

推导：

- mean/variance 存在条件；
- $P(R<0)$ 与 $P(R>0)$；
- $\mathbb E[R\mathbf1(R<0)]$；
- $\mathbb E[R\mathbf1(R>0)]$；
- $\mu\ne0$ 与 threshold=0 的一般式；
- 极限和对称情形。

### T4. Bayes action

针对 MAE、MSE 以及项目的三动作集合，分别推导最优点动作。说明 partial moment candidate 是否合理；如果只是 heuristic，必须明确。

### T5. 可辨识性

分析 location/scale/skew/df 的混淆、不同市场 normalization、神经网络过参数化。给必要约束或重参数化。

### T6. 24h 依赖

在 W1/W2/W3 中做理论和工程取舍，给出推荐及拒绝其他方案的理由。

### T7. 与状态/Bi-OMC 接口

给完整张量输入输出、参数化、candidate 公式、梯度路径，说明 occurrence/magnitude 是否显式输出。

### T8. 与 CAGM/DVG 接口

给 candidate uncertainty、historical gain、risk-adjusted value 的非重复分解。

### T9. 文献碰撞

检索截至当前日期的：

- asymmetric/skew Student-t time-series forecasting loss；
- heavy-tailed probabilistic electricity price forecasting；
- distribution-derived bidirectional residual correction；
- post-hoc frozen forecast correction with probabilistic action routing；
- partial-moment neural decision rules。

逐项说明“已有工作做了什么、本项目还能主张什么”，给真实可点击链接。优先原论文/官方代码。

### T10. 实现与测试草图

只写伪代码、API 和测试，不修改仓库。测试至少包括密度积分、解析矩 vs numerical quadrature、梯度 finite difference、对称/高斯极限、采样恢复、跨尺度等变性、极端参数稳定性。

---

## 14. 强制输出格式

请用一个**全新文件名**返回完整 Markdown，例如：

`hch_v2_derived_loss_math_design_v0.1_YYYY-MM-DD.md`

严格按以下结构：

### 1. Executive verdict

- 推荐最小模型；
- 是否真的需要 asymmetric Student-t；
- 最大数学风险；
- 是否值得进入实现。

### 2. Assumptions and falsification table

| 假设 | 数学作用 | S2/S3 检验 | 失败后的最小方案 |

### 3. Literature and claim boundary

| 工作 | 真实贡献 | 与本项目重叠 | 不构成/构成碰撞 |

### 4. Distribution derivation

逐步推导 normalization、CDF、NLL、limits、moments。

### 5. Directional partial moments

必须覆盖 $\mu\ne0$，并与数值积分互证。

### 6. Bayes decision and candidate semantics

比较 partial moment、conditional mean/median 与三动作最优解。

### 7. State and cross-market parameterization

说明共享与方向特异参数、负价信息迁移及 negative transfer。

### 8. 24h dependence decision

明确选择 W1/W2/W3，不可全部保留为模糊选项。

### 9. CAGM-DVG interface

给无重复风险分解和张量/API。

### 10. Numerical implementation

稳定公式、伪代码、复杂度。

### 11. Unit and falsification tests

给测试输入、oracle、容差与失败解释。

### 12. Ablation and diagnostics

只设计必要比较：M0/M1/推荐模型、旧 loss/new loss、candidate/gate 分解。

### 13. Rejection-risk register

| 风险 | 严重度 | 审稿人会怎么说 | 如何避免 |

### 14. Theorem/lemma/proposition set

至少给：

- 一个正规化/矩命题；
- 一个方向候选或 Bayes action 命题；
- 一个极限/等变性命题。

每个注明假设、结论、证明或证明草图；不能把常识重命名为 theorem。

### 15. Final implementation contract

给唯一推荐设计，不要让代码 AI 自己在多个数学分支中猜。

### 16. Allowed and forbidden claims

给可以写进摘要/方法的严格句子，以及暂不允许的句子。

---

## 15. 暂不允许的主张

在证据完成前，禁止：

- “首次使用 Student-t / skew-t 做时间序列预测”；
- “TFT/N-BEATS 已证明本分布损失”；
- “残差服从 asymmetric Student-t”；
- “解析损失天然优于组合损失”；
- “一个分布统一解决尖峰、低价、负价和正常期”；
- “partial moment 就是 Bayes-optimal correction”；
- “共享上尾知识必然改善负价”；
- “24h token 等价于联合路径概率模型”；
- “模型无关/跨市场/零样本”；
- “正常期不退化”；
- “达到 SOTA”；
- “首次校准校正动作风险”；
- 以私有山东结果作为唯一创新证据。

“数学上可推导”只说明内部一致，不自动说明预测更准或创新成立。

---

## 16. 期望但尚未成立的核心差异化

可研究的严格版本是：

> 现有重尾 likelihood 主要用于直接概率预测；本项目尝试把**状态条件重尾残差分布的方向性决策量**转化为冻结宿主的 Up/Down 候选，再由日情节历史动作收益相对 Identity 路由。创新若成立，来自“分布推导的双向候选 + 情节动作价值路由”的耦合，而不是 Student-t 或 skew-t 分布本身。

你必须通过文献碰撞与 Bayes action 推导判断这句话能否保留；如不能，给最小重构。

---

## 17. 研究窗口的停止条件

完成文档后停止，不改仓库，不实现代码，不运行 S4，不替项目选择最好 seed。请明确标记：

- `READY FOR ARCHITECTURE REVIEW`：数学、文献、实现契约完整；
- `NOT READY`：仍缺正规化、部分矩、可辨识性、碰撞或唯一推荐设计。

如为 `NOT READY`，准确列出阻塞项，而不是提供一个看似完整但无法实现的公式。

---

## 18. 项目后续交接

用户会把你的数学文档带回主架构对话。主架构对话将：

1. 检查数学与 claim boundary；
2. 把唯一推荐损失与已修复的 HCH v2 融合；
3. 再给代码 AI 一份增量实现规范；
4. 通过契约 smoke；
5. 最后才解除 freeze/half-exp 协议的 HOLD。

不要越过这一交接顺序。
