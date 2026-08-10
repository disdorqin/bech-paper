# BECH / GAFE 创新方案六问严格审稿报告

**评审日期：**2026-08-07  
**评审对象：**《18 · 网页端交叉验证提交——创新点评估（第一批，只评创新点）》  
**评审口径：**按 CCF-A 主会资深审稿人标准，优先寻找可导致拒稿的反证，不以“尚未发现完全同名方法”代替创新性证明。  
**证据边界：**本报告评的是当前设计与主张；M1–M9 数学细节尚未提供，因此对保证的结论是“当前表述能否成立”，不是对尚未见到的证明作最终判决。

---

## 0. 总裁决

### 0.1 当前版本的审稿结论

> **CCF-A 口径：Weak Reject，约 4/10；若正文照当前 C1/C2/C3 三创新点提交，我会倾向拒稿。**

拒稿原因不是“问题不重要”，而是：

1. **C1 更像部署契约，非算法创新；**
2. **C2 的“特征作节点 + 图消息传递”已有直接先例，且与 C3 不是不可分割的科学问题；**
3. **C3 包含真正值得研究的内核，但当前把已被占据的 correction selection / action-risk / output adaptation 又写成宽泛 novelty；**
4. **“单点零退化”“共形安全分位”“bootstrap 效能 LCB”之间尚无清楚的风险对象、假设和组合定理；**
5. **数据承诺本身不致命，但当前山东数据的角色、NEM 的任务语义、96 点山东目标字段的业务定义，都可能成为一票否决项。**

### 0.2 六问一句话答案

| 问题 | 严格结论 |
|---|---|
| ① 真创新？ | **三项中只有 C3 的窄交集有潜在真创新；C1 不够，C2 基本不成立，C3 当前表述又过宽。** |
| ② 一票否决风险？ | **有。**首创表述被近邻反证、保证口径过强、选择与认证泄漏、价格事件与残差方向混淆、数据目标语义错误，任何一项都可能直接拒稿。 |
| ③ 如何强化？ | 收敛为**一主创新**：signed-tail occurrence–magnitude 候选生成 + normal/negative/positive 三组 base-relative harm 预算；SCARR 仅作内部认证层。 |
| ④ 主线还是拼盘？ | **当前是有流程顺序的拼盘，不是不可分割的主线。**C1/C2 删除后 C3 仍成立，说明二者不是论文核心。 |
| ⑤ 划界差异成立吗？ | 多数是“应用域不同”或“模块名不同”，**不足以形成方法学划界**；且 δ-Adapter 被重复列为两篇，最危险的 CRC/COSA/action-conditional 工作反而漏列。 |
| ⑥ 数据承诺致命吗？ | **私有山东不进入主表并不致命；但若私有数据决定了结构、公开数据不能复现结构结论，或 96 点目标其实是单位结算/合约价却写成市场日前价，则致命。** |

---

## 1. ① 哪些是真创新，哪些不是

### 1.1 C1 · Adaptive Feature Contract

**裁决：工程价值中等，研究 novelty 很弱；不应列为 CCF-A 的独立算法贡献。**

可用性掩码、缺失协变量处理、变量子集预测、不同输入集合下的统一接口都不是新问题。KDD 2022 已明确提出 [Variable Subset Forecast](https://dl.acm.org/doi/10.1145/3534678.3539394)；ICML 2023 还系统研究了缺失协变量对共形覆盖的影响，并指出不同 mask 下覆盖可能显著变化（[Conformal Prediction with Missing Values](https://proceedings.mlr.press/v202/zaffran23a.html)）。因此，“六篇校正论文没做”不能推出“首个自适应输入契约”。

它可以保留的价值是：

- 定义冻结基座与校正器之间的**接口兼容等级**；
- 区分 common-core 特征轨与 enhanced 特征轨；
- 防止不同市场的特征缺失导致训练/部署行为不一致；
- 公开掩码、信息截止时间和回退规则，提高可复现性。

但这些更像系统协议和实验公平性设计。若没有“不同特征集合下仍保持某种风险有效性”的新定理，或大规模 variable-set post-hoc correction benchmark，不能当主创新。

**建议定位：**从 C1 降为 `Base Forecast / Feature Contract`，写进方法设置或可复现性章节，不再使用“首个”。

### 1.2 C2 · GAFE 特征关系图

**裁决：当前 novelty 基本不成立，并且是最应从主线删除的部分。**

“每个特征为节点、边表示特征交互、GNN 聚合高阶交互”至少已被 [Fi-GNN](https://dl.acm.org/doi/10.1145/3357384.3357951) 明确实现；KDD 2021 的 [FIVES](https://dl.acm.org/doi/10.1145/3447548.3467066) 直接把交互特征生成写成 feature graph 上的 edge search。时序领域里，[MTGNN](https://dl.acm.org/doi/10.1145/3394486.3403118) 和 KDD 2022 的 [ESG](https://dl.acm.org/doi/10.1145/3534678.3539274) 已把多变量/通道作为图节点并学习动态关系。

“节点=特征，不是节点=通道”不是稳固划界：当每个业务特征随时间形成一条序列时，它本身就是一个 variable/channel。审稿人不会因为换了名词就承认新的图学习范式。

更危险的是，2025–2026 的 [CRC](https://arxiv.org/abs/2512.22428) 已把 direction-aware / causality-inspired 编码器、残差校正器和四重安全机制放在同一框架中。即使 CRC 不做电价，C2 的“关系编码 + residual correction + safety”方法组合也不再是空白。

此外还有三项技术风险：

1. 相关 top-k 或 Granger 边只能叫**统计/预测关系**，不能直接叫因果；
2. 若边用全数据或测试期统计量构造，会发生时间泄漏；
3. 特征 mask 改变图拓扑后，原校准证书不自动对新拓扑有效。

**建议定位：**GAFE 只作可选上下文插件和消融。必须与 MLP、cross-attention、Fi-GNN/MTGNN 风格编码器公平对照；若跨市场、跨基座没有稳定增益，直接删除。不要使用“因果边”“唯一位置”或独立 C2 主张。

### 1.3 C3 · BECH + SCARR

**裁决：问题价值高，但四个分主张里没有一个能按当前宽口径单独成立；真正可守住的是更窄的完整交集。**

| C3 当前分主张 | 审稿裁决 | 原因 |
|---|---|---|
| 校正级弃权 | **不能宣称首创** | NeurIPS 2025 的 [PIR](https://openreview.net/forum?id=H7e5RpeIi4) 先识别潜在失败实例再修订；CRC 有选择/裁剪/回退；[COSA](https://openreview.net/forum?id=L7Z5wBMPrW) 用门控控制冻结输出修正。语义上都已在决定“是否/多大程度修改原预测”。 |
| 可预测性驱动分支 | **可能是任务化机制，但当前近似 heuristic** | 用一个全局 AUC/AP 阈值决定结构，容易被认为“依据验证分数选择模块”；没有不可约目标或定理时不构成独立算法创新。 |
| 分尾认证 | **领域实例可保留，通用 novelty 不成立** | 2026 已有 [action-conditional conformal guarantee](https://arxiv.org/abs/2606.05551)、[selection-conditioned routing control](https://arxiv.org/abs/2512.01556)、[CREDO](https://arxiv.org/abs/2505.13243) 和 [Conformal Policy Control](https://arxiv.org/abs/2603.02196)。 |
| 结构不对称 | **仅作为领域结构先验，不足以单独成贡献** | “负尾用 A、正尾用 B”若只是验证集选出来的两种已有模块，是模型选择，不是新方法。 |

截至 2026-08-07 的聚焦核验，尚未看到一篇工作同时占据以下完整交集：

> **冻结异构的绝对电价基座 + 正/负 signed event occurrence–magnitude 分解 + BASE/SHRINK/FULL 三档校正动作 + 相对基座的 normal/negative/positive 分组 harm 预算 + 延迟 rolling-OOS 选择与认证。**

这才是可辩护的创新核。但当前 C3 文本弱化或遗漏了其中两个最重要的元素：

- 没有把 **occurrence–magnitude two-part** 写成明确的候选生成机制；
- 没有把**正常期 hard harm budget**写成不可让步的首要约束。

因此，当前版本不是“已经钉死的真创新”，而是把最有希望的创新核改写得更宽、更容易撞车。

---

## 2. ② 一票否决风险

### 2.1 P0：足以直接拒稿的风险

#### R1 · 文献事实与首创主张不可靠

文档把 “The Forecast After the Forecast” 和 “δ-Adapter”列成两个先例，实际上后者就是前者提出的方法（[ICLR 2026 / arXiv](https://arxiv.org/abs/2601.20280)）。这一重复会让审稿人怀疑碰撞审计是否真正读过原文。

同时，“校正级弃权首次”已被 PIR、CRC、COSA 的选择性修订/门控/回退机制实质反证；“动作而非区间”也已被 2026 action-conditional、decision-risk 和 routing 文献反证。

#### R2 · “单点零退化证书”按当前表述很可能不成立

“单点 harm ≤ ρ×基线”是非常强的条件。有限校准数据通常只能支持明确假设下的边际、分组、动作条件或高概率风险保证，不能自动给每个未来样本确定性零伤害。若 `baseline loss` 接近 0，比例式 harm 还会退化或无定义。

必须把它改成清楚的策略级风险对象，例如：

\[
R_g(\pi)=\mathbb E\!\left[\bigl(\ell(\hat y^{\pi},y)-\ell(\hat y^{base},y)\bigr)_+\mid Z=g\right]\le \varepsilon_g,
\quad g\in\{-,0,+\},
\]

并说明这是期望、分位数、VaR/CVaR 还是 exceedance probability；说明有限样本置信水平、时间依赖假设和失败概率。`zero-degradation` 只能在 \(\varepsilon_g=0\) 且定理确实支持时使用，否则应写 `budgeted harm`。

#### R3 · 结构选择和认证可能复用 S2

当前文字说在 S2 测 AUC/AP决定修正/保护分支，同时又用两层证书决定是否执行。如果同一段数据还用于选阈值、挑分支、拟合校正器、计算 LCB 和风险分位数，最终证书不再是独立验证。

最低要求是五段时间边界：

1. Base train；
2. rolling-OOS residual fit；
3. policy/branch select；
4. independent risk certify；
5. untouched final test。

#### R4 · AUC 与 AP 的跨尾比较无效

不能用负尾 AUC 与正尾 AP直接比较后得出“负尾可修、正尾应保护”。AUC 对极不平衡事件可能很乐观，AP 又受事件基率强烈影响。即使都用 AP，occurrence 可预测也不等于 correction magnitude 可预测，更不等于执行修正会降低损失。

应统一使用：

- 两侧都报告 AUPRC、基率归一化 AP lift、calibration/Brier；
- 真正决定分支的是独立选择集上的 **base-relative action utility–harm frontier**；
- 选择规则预注册，不从测试结果倒推“结构不对称”。

#### R5 · 真实价格尾状态与残差方向可能混淆

负电价不等于基座一定高估，也不等于校正方向一定向下；正尖峰也不等于基座一定低估。必须分开：

- 真实价格事件状态 \(Z_t\in\{-,0,+\}\)；
- 运行时可观测的预测事件概率 \(\hat p_t^-,\hat p_t^+\)；
- 基座残差/候选修正方向 \(D_t\)。

运行时只能使用预测概率和合法上下文；真实 \(Z_t\) 只能在独立认证集和测试后诊断中出现。若路由直接使用真实尾标签，就是 oracle leakage。

#### R6 · 山东 96 点目标语义尚未冻结

结合此前数据盘点，当前已确认的 96 点价格目标是 `epf_unit_data_96.da_cq_price/rt_cq_price` 的**单位级**字段；市场级 96 点 DA/RT 价格尚未确认，且单位 DA 均值与小时市场 DA 的差异很大，不能假定等价。若论文写“山东市场日前电价”而训练标签实为单一机组/单位的合约或结算价格，这是任务定义错误，属于直接拒稿级问题。

### 2.2 P1：不一定单独拒稿，但会显著降分

- GAFE 复杂度高却缺乏不可替代性，容易被判为 graph-for-graph's-sake；
- 正尾 AP 低时，安全机制可能退化为几乎总返回 BASE，形式上安全但无实际效用；
- 分尾 × 三动作 × 24/96 horizon × 多市场会造成认证样本碎片化，证书可能变得空洞；
- NEM_SA1 负价占比 24.6% 时，它更像频繁 regime 而非 rare tail，术语需要调整；
- “模型无关”若只证明不读隐藏层，应写 interface/backbone-agnostic，不能暗示跨未见基座参数零适配；
- enhanced 特征轨给校正器额外业务预报、而基座没有同等信息，会把“校正能力”与“额外信息增益”混在一起。

---

## 3. ③ 如何强化：推荐的最小重构

### 3.1 只保留一个主创新

推荐主问题改为：

> **在不知道真实尾标签、不能改动冻结基座、且正常期伤害必须受控时，如何利用严格 OOS 残差，对负电价与正尖峰产生 occurrence–magnitude 校正候选，并只执行经过分组风险认证的动作？**

论文贡献重新组织为：

1. **任务定义：**signed-tail post-hoc correction for frozen electricity-price forecasters；
2. **核心方法：**BOM-SSC/BECH two-part 候选生成；
3. **安全机制：**SCARR 作为已有 action-/selection-risk 控制思想的领域化实例，不作为第二个通用理论首创；
4. **实证协议：**跨冻结基座、跨语义一致市场、严格 rolling-OOS，公开失败边界。

C1 改为接口协议；C2 降为可选消融。

### 3.2 方法必须恢复的四个结构

#### A. signed occurrence–magnitude 候选生成

分别估计负价事件与正尖峰事件的发生概率，并在相应事件条件下估计基座残差幅值。价格事件状态与残差符号解耦。

#### B. 三档动作

\[
a_t\in\{\mathrm{BASE},\mathrm{SHRINK},\mathrm{FULL}\}.
\]

`BASE` 是安全参考动作；`SHRINK` 不能只是多一个可调 \(\lambda\)，必须通过风险—效用曲线或理论说明其必要性。

#### C. 三组硬预算

正常期 \(\varepsilon_0\) 必须最严格；负尾和正尾分别认证。尾部改善是在安全可行集里优化，不能通过放宽正常期预算换取。

#### D. 独立选择与认证

事件阈值、分支选择、动作强度在 select 段确定；certify 段只能接受/拒绝固定策略，不能继续调参。时序依赖用 block/bootstrap、martingale 或明确的弱依赖假设处理，不能照搬 i.i.d. split conformal 口径。

### 3.3 “可预测性自适应”应怎样改

不要写“AP 高就修正，AP 低就保护”。改为：

> 对每个市场 × 尾侧，在独立选择段上比较固定候选策略的 cross-fitted base-relative utility 和 harm；只有某个非 BASE 策略在后续独立认证段通过三组预算时，才进入测试。

这样，“自适应”是可审计的策略选择过程，而不是看了分数后手工指定网络。

### 3.4 若想冲击 NeurIPS / ICML / ICLR

必须至少补出一种可复用理论：

- signed group × action 的同时风险控制；
- delayed feedback / rolling time-series 下的有效性；
- 策略选择后仍有效的 learn-then-test 或多重检验修正；
- 在 normal harm budget 与 tail utility 之间的可证可行性或 Pareto 性质。

否则更适合把它定位为 KDD/WWW/WSDM/AAAI/IJCAI 的“新任务 + 方法 + 严格多市场实证”，而不是通用共形理论论文。

---

## 4. ④ 整体主线还是拼盘

### 4.1 当前结构：有流水线，不等于有科学主线

当前叙事是：输入契约 → 图特征合成 → 校正与认证。它在工程上有顺序，但三个创新点分别解决：

- C1：特征缺失/可用性；
- C2：高阶特征交互；
- C3：极端电价校正与动作风险。

检验拼盘最简单的方法是做删除测试：

| 删除组件 | 核心科学问题是否仍成立 | 结论 |
|---|---|---|
| 删除 C1 | 是；固定 common-core 特征仍可做双尾安全校正 | C1 是支持协议，不是核心 |
| 删除 C2 | 是；MLP/attention 仍可生成校正候选 | C2 是可替换编码器，不是核心 |
| 删除 SCARR | “只在安全时校正”的核心承诺消失 | SCARR 是核心内部机制，但非独立通用 novelty |
| 删除 occurrence–magnitude | 双尾修复退化为普通 residual corrector | two-part 才是核心结构 |
| 删除 normal budget | 失去与 CRC/COSA 等近邻最重要的任务差异 | normal harm 是核心边界 |

因此当前是**拼盘**。推荐的贯穿主线只有一句：

> **生成有符号双尾修补候选，并在相对冻结基座的分组伤害受控时执行。**

所有模块都必须回答这个问题；不能直接提高这条主张可信度的组件，不进入贡献列表。

---

## 5. ⑤ 现有划界是否成立

### 5.1 当前先例表的逐类裁决

| 先例 | 文中划界是否成立 | 严格审稿意见 |
|---|---|---|
| RevIN / Dish-TS / SAN | **字面成立，但弱相关** | 它们不是最危险的 correction-action 近邻；把篇幅放在这些工作上会像避重就轻。 |
| PIR | **只部分成立** | “无电价双尾”是应用差异；PIR 已做失败实例识别后再修订，直接削弱“校正级弃权首次”。 |
| Post-Training Corrections | **部分成立** | 通用、轻量、模型无关的预测后修正已被占据；剩余差异只能是 signed-tail two-part 与 group harm。 |
| The Forecast After the Forecast / δ-Adapter | **强碰撞，且文档重复计数** | 同一论文。其冻结基座、输入/输出适配、预算 mask、conformal calibrator 已占据宽泛接口创新；只能靠特定风险对象划界。 |
| UEC-STD / Reviving Error Correction | **部分成立** | 架构无关 corrector 与跨 backbone 验证已被占据；它不做双尾分组风险，这是可用差异。 |
| Selective Regression / L2A | **字面成立但不足** | “弃权预测 vs 弃权校正”只是动作语义变化；现代 routing/action-risk 框架已覆盖更一般抽象。 |
| ACI / SPCI / Kath | **对旧文献成立，对当前前沿不成立** | 只与区间共形对比会形成 strawman；必须加入 2026 action-conditional、LEC、CREDO、CPC。 |
| DART spikes 2023 | **目标差异成立** | DART spread 与 absolute price 不同；但还必须加入 2026 [Trading Electrons](https://arxiv.org/abs/2601.05085) 的正/负 DART spike + selective action。 |
| PNN+ELM negative price | **部分成立** | 该文是负价两阶段预测而非冻结基座校正；但它已占据 occurrence/negative-price 专门建模，BECH 不能说负价专门建模首次。 |
| Abramova & Bunn skew-t | **成立但不够贴脸** | 分布不对称与动作不对称不同，但不是主方法近邻。 |
| TrUST 2025 | **当前无法正式裁决** | 文档没有完整标题、链接和方法对象。提交前必须补精确引用；否则不要放入划界表。 |

### 5.2 当前划界表漏掉的最危险先例

必须新增并在主表实现或公平复现：

1. **CRC**：安全残差校正、选择/回退、因果启发编码；同时撞 C2 与 C3；
2. **COSA**：冻结输出端适配 + 门控；
3. **PIR**：失败实例识别 + post-hoc revision；
4. **δ-Adapter**：架构无关 output residual correction + bounded adaptation + conformal calibration；
5. **Action-conditional conformal / LEC / CREDO / CPC**：推翻“动作风险首创”；
6. **Fi-GNN / FIVES / MTGNN / ESG**：推翻“特征作节点的图交互首次”。

### 5.3 可安全使用的差异化声明

> **我们不声称首创模型无关校正、输出端适配、校正选择、动作条件风险控制或特征关系图。我们研究更窄的电价修复问题：对冻结异构绝对电价预测器的 rolling-OOS 残差，分别建模负价与正尖峰的 occurrence 和 conditional magnitude，并在独立认证的 normal/negative/positive base-relative harm budgets 下选择 BASE、SHRINK 或 FULL correction。**

这句话可以守；“首个校正级弃权”“首个特征自适应校正契约”“首个动作共形风险”都不应再用。

---

## 6. ⑥ 数据承诺是否致命

### 6.1 原则判断

**“核心实验公开、山东只作动机与内部外部效度检查”本身不致命，甚至比依赖私有主表更稳妥。**公开 EPF benchmark 确实能提供多个日前市场与现成预测；例如 [EPF Toolbox](https://github.com/jeslago/epftoolbox) 明确包含 EPEX-BE/FR/DE、Nord Pool 和 PJM 的小时日前价格。GEFCom2014 也有公开的 electricity-price track（[IEEE PES 数据页](https://ieee-pes-data-sharing.org/datasets/detail/b1680aa5-a4b8-4423-8760-e509094cacec)）。

但当前承诺有四个潜在致命点。

### 6.2 D0 · 山东私有数据角色自相矛盾

文档一方面说山东只作动机和补充材料，另一方面又用“山东负尾 AP 0.81、正尖峰 AP 0.15”作为 C3 结构不对称的关键证据。若方法结构是看山东结果后设计的，山东就不是普通外部验证，而是设计数据。

解决方式只能二选一：

1. **公开数据主导设计：**在公开市场的 select split 上预注册并复现同样的分支选择结论；山东只作锁定后的外部验证；
2. **山东主导设计：**承认它是 development market，并公开足够的数据字典、时间切分、事件计数、匿名残差或可审计统计；同时不能把论文宣传成完全公开可复现。

### 6.3 D1 · NEM 与日前市场语义不一致

AEMO 官方数据把 NEM 核心价格定义为区域 **5-minute dispatch/spot price**（[AEMO NEM data dashboard](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/data-dashboard-nem)）。它不是与 EPEX/Nord Pool 小时日前竞价价格天然同义的标签。

若论文一句话固定为“任意冻结日前电价预测器”，NEM 不能无说明地并入主表。可选处理：

- 把任务上位化为 short-term wholesale price correction，并分 DA / spot 两轨；或
- 主文只用语义一致的 day-ahead markets，NEM 作为跨任务外部效度；或
- 明确 NEM 的 forecast issuance time、target interval 和合法信息集，不把它叫 day-ahead price。

### 6.4 D2 · 山东 96 点字段业务定义未解决

在目标字段是单位级 `da_cq_price/rt_cq_price`、市场级 DA/RT 尚未核实时，不得承诺“山东市场日前/实时价格修正”。先冻结：

- 价格的市场角色、结算规则、单位与上下限；
- forecast issue time 与每个特征的 availability cutoff；
- 24/96 点目标是否可比；
- 单位价、市场出清价、合约价和结算价之间的关系。

此门未过，96 点实现应保持 STOP；否则再漂亮的模型实验也建立在错误标签上。

### 6.5 D3 · 稀有组认证可能没有统计功效

Lago-DE 报告负价约 1.03%，NEM_SA1 又达 24.6%。这两者不是同一种稀有度；再按尾侧、动作、horizon 和市场细分后，有效样本数可能从数百降到个位数。

必须在建模前公开：

- 每个时间 split 的事件数；
- 每个 group × action 的认证样本数；
- 预注册的最小样本门槛与层级回退规则；
- certificate infeasible / no-action / return-base 的比例；
- 风险上界的宽度，而不只报告是否“通过”。

若最终 95% 以上样本都返回 BASE，安全性不假，但方法实用价值基本消失。

### 6.6 D4 · 公平性与可复现性清单

核心公开实验至少满足：

1. 数据 URL、许可证、版本、哈希、下载日期；
2. 统一或明确分轨的 target semantics、currency/unit、resolution、DST 处理；
3. common-core 与 enhanced 两轨分开，避免额外特征偷赢；
4. 所有基座使用同一信息截止时间；
5. 基座残差来自 rolling-OOS/cross-fit，而非基座训练内预测；
6. select、certify、test 完全隔离；
7. 尾阈值只由训练/选择数据确定；
8. 公开 occurrence、magnitude、normal/tail harm、correction/abstention rate 和最坏基座结果。

### 6.7 数据最终裁决

| 情况 | 裁决 |
|---|---|
| 山东私有，仅锁定后外部验证；公开集完整复现核心结论 | **不致命，合理** |
| 山东决定方法结构，但公开集不能复现且不披露开发过程 | **高风险，接近致命** |
| NEM spot 与欧洲 DA 混为同一任务 | **致命语义错误** |
| 单位 `cq_price` 被写成市场 DA/RT 出清价 | **直接致命** |
| 稀有分组样本不足，策略几乎永远 BASE | **理论上安全、论文价值不足，可能拒稿** |

---

## 7. 建议的最终论文结构

### 7.1 贡献结构

1. **唯一主创新：**Budgeted signed-tail post-hoc correction；
2. **候选生成：**negative/positive occurrence–magnitude decomposition；
3. **内部安全层：**grouped base-relative action-risk certification；
4. **部署协议：**adaptive feature contract，仅作 compatibility/reproducibility；
5. **可选消融：**GAFE，不进标题、不进三大贡献。

### 7.2 推荐标题与 pitch

**推荐标题：**`Budgeted Signed-Tail Correction for Frozen Electricity Price Forecasters`  

**一句话 pitch：**

> 当冻结电价模型漏掉负价或正尖峰时，方法先生成有符号 occurrence–magnitude 修补候选，再只执行那些在独立认证中满足正常期与双尾相对基座伤害预算的动作；否则原样返回基座预测。

### 7.3 提交前六个 Go/No-Go 门

1. 删除 C1/C2 的“首个”与独立主创新身份；
2. 恢复 occurrence–magnitude 和 normal hard budget；
3. 将 SCARR 明确降为内部风险层，补齐 action-risk 近邻；
4. 固定 Z、预测事件概率和残差方向三者的区别；
5. 完成五段式无泄漏协议和依赖时序有效性说明；
6. 先解决山东 96 点目标业务定义，再启动相应实验。

只要任一 P0 门未过，不建议进入大规模 baseline 跑数；否则很可能花大量算力证明了一个审稿定义上仍不成立的方案。

---

## 8. 最终审稿意见

这项工作**不是没有创新**，但创新不在“三个模块各自首次”。它最有价值的地方是一个很窄、很实际、且现有通用 corrector 没有完整解决的问题：

> **冻结绝对电价预测器在负价与正尖峰上的双向失败，如何在正常期伤害受硬约束时被选择性修补。**

当前稿把这个核稀释成了 feature contract、feature graph、predictability branching、abstention、bootstrap、conformal 多个可替换组件，于是显得面面俱到却没有一个点足够深。正确策略不是再加第四个创新点，而是砍掉 C1/C2 的 novelty 身份，把 C3 收窄、形式化并做干净的公开数据验证。

**当前：Weak Reject。**  
**完成上述最小重构且公开市场结果强：KDD/AAAI/IJCAI/WWW 具备可竞争性。**  
**若无新的 signed-group × action × delayed-time-series 理论或公开 benchmark，不建议按 NeurIPS/ICML/ICLR 通用方法论文包装。**
