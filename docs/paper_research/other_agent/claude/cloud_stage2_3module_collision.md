# 三模块架构 Novelty 碰撞审计

> **审计对象**：「图结构 → 极端感应 → 极端修补」三模块串联骨架（模型无关、即插即用、双向极端电价校正）。
> **审计方法**：聚焦碰撞审计（2025–2026 文献，不做广撒网）。每条近邻均给真实可点击链接；对用户/GPT 提供的"已知锚点"做了独立复核，标注了其中无法定位与归属有误者（见附录 A）。批判性优先——直接指出骨架的脆弱点。
> **独立署名**：本报告为阶段二（架构级碰撞）产物，与阶段一报告 `03_交叉验证_顶会检索与创新点发现_调研报告.md` 相互独立，未覆盖。
> **日期**：2026-07-31

---

## 0. 审计结论速览（TL;DR）

| 项 | 结论 |
|---|---|
| 模块 1 图结构 | **部分碰撞**。图引导校正已被 CRC 占据；CoRel/STOIC 已把图建在残差上做后处理。预测列作节点＝实现细节，不是贡献。 |
| 模块 2 感应头 | **组成碰撞**。两段式 occurrence–magnitude 与三态 regime 建模均为成熟范式（统计/能源文献早已做透）。 |
| 模块 3 修补器 | **完整→组成碰撞**。门控校正、非降级/增量损失、harm 预算、反事实损失四项各有直接近邻。 |
| 组合「图→感应→修补」 | **部分未被完整抢占**，但三段中每个不可约元素均被单篇近邻占据；串联本身不是 novelty。 |
| 尚存真正空位 | 「双向极端电价（负价+尖峰）的**绝对日前电价**模型无关后处理点校正」＋「**极端事件条件下**的分布无关、有限样本非降级保证」＋「≥3 市场＋尾部/经济价值指标」。 |

---

## A. 模块级碰撞表

### 模块 1 — 图结构构建（节点含预测列；边＝统计相关 / 领先—滞后 / 因果 / Attention；只服务极端后处理）

| 近邻（真实链接） | 碰撞等级 | 一句判断 |
|---|---|---|
| [CRC: Causality-Inspired Safe Residual Correction](https://arxiv.org/abs/2512.22428)（arXiv 2512.22428） | 部分 | 用因果编码器＋先验邻接矩阵把"图结构"直接用于**校正**（分离自/跨节点动态）——"图只服务后处理、不改预测"已被占据；但其图是变量间因果先验，非特征＋预测列＋时点的混合图。 |
| [CoRel: Relational Conformal Prediction for Correlated Time Series](https://arxiv.org/abs/2502.09443)（ICML 2025） | 部分 | 在**冻结预测器之上**从残差端到端学习稀疏关系图做后处理共形区间——"残差建图＋图后处理"的模型无关组合已存在。 |
| [STOIC: Relational and Sequential Conformal Inference for Energy Time Series over Graphs via Foundation Models](https://arxiv.org/abs/2606.31804)（arXiv 2606.31804） | 部分 | STGNN 点预测 → 残差表格化 → 表格基座模型零样本共形校准，直接命中"图＋能源时序＋残差后处理"交集（但做区间，不做点校正，且不极端特化）。 |
| [STBIM: Spatio-Temporal Backward Inconsistency Learning](https://openreview.net/forum?id=7rOdRAGuBA)（OpenReview） | 部分 | 通用残差学习＋传播模块，可嵌入任意时空模型做预测修正——"后处理残差模块"作为通用件已被提出。 |
| [ASTGCN: GCN＋Attention 短期电价预测](https://arxiv.org/abs/2107.12794)（arXiv 2107.12794）；[MVFSTGNN 多视角融合时空图网络（Applied Energy 2024）](https://www.sciencedirect.com/science/article/abs/pii/S030626192400936X) | 弱 | 图用于**预测**而非后处理——确认"图做 EPF"已做透，反衬"图做校正"是更窄的缝。 |
| [Whiteness testing for graph signals](https://re.public.polimi.it/bitstream/11311/1233899/1/2204.11135.pdf)（Polimi 库） | 弱 | GNN 预测残差是否仍含时空结构——为"残差图结构值得建模"提供动机证据（也暗示：若残差已白，图校正无益）。 |

**模块 1 碰撞判定：部分。** "图服务于校正/后处理"不再是空白（CRC 图编码校正、CoRel 残差图、STOIC 能源残差图均已占据）。"预测列本身作为图节点"未见直接同款，但这是把一列输入变成节点的**实现细节**——审稿人会判 trivial；除非消融证明预测列节点带来不可替代的信息，否则不能作为贡献主张。**脆弱点**：图模块可能只是"图预测换皮"或"注意力机制的图化"。

### 模块 2 — 极端感应头（有符号双尾 occurrence–magnitude 分解 ＋ 三态信号）

| 近邻（真实链接） | 碰撞等级 | 一句判断 |
|---|---|---|
| [A Hybrid Classification-Regression Method for Forecasting Negative Electricity Prices](https://researchnow.flinders.edu.au/en/publications/a-hybrid-classification-regression-method-for-forecasting-negativ/)（IEEE ICCE 2026） | 完整 | 分类器估负价发生概率＋回归器估幅度，捕获 98% 负价事件——两段式 occurrence–magnitude 在负电价上是**直接同款**。 |
| [Loizidis et al.: PNN 分类＋ELM-Bootstrap 区间（Applied Energy 2025）](https://www.sciencedirect.com/science/article/abs/pii/S0306261925007433) | 完整 | 两阶段（负/正价日分类 → 区间预测选择），含市场误分类成本。 |
| [TSEP: Two-Stage Electricity Price Forecasting（EPSR 2021）](https://www.sciencedirect.com/science/article/abs/pii/S0378779621003977) | 完整 | DNN 尖峰发生预测＋ANN 尖峰幅度校准——同一范式用于尖峰。 |
| [Trading Electrons: Predicting DART Spread Spikes（arXiv 2601.05085）](https://arxiv.org/abs/2601.05085) | 部分 | 统一模型**同时处理正、负 DART 尖峰**——"有符号双尾"作为任务已存在（对象是 spread 而非绝对价、用途是交易而非校正）。 |
| [广义有序 Logistic 三态跳变建模（Zakopianska 会议 2020）](http://www.konferencjazakopianska.pl/pliki/proceedings_2020/pdf/2020_Monografia-07.pdf)；[Karakatsani & Bunn 机制切换模型](http://www.agsm.edu.au/bobm/iows/karakatsani_bunn2004.pdf)；[Nord Pool 三态 ordered probit（drop/normal/spike）](https://pure.au.dk/ws/portalfiles/portal/129056239/PhD.pdf) | 完整 | 三态（向上/无/向下跳、正常/尖峰/跌落）建模电价早已存在——**三态信号是教科书概念**。 |

**模块 2 碰撞判定：组成。** 两个核心概念——occurrence–magnitude 两段分解、三态 regime——各自都是成熟范式，且都不是 CCF-A 会议上才有的新东西（能源经济期刊与统计文献早已做透，审稿人大概率知晓）。"有符号双尾"作为目标被 Trading Electrons 处理（但面向 spread 交易、非 post-hoc 校正绝对价）。**唯一仍空的是组合级位置**：把这些组件内嵌到一个**以冻结基座残差为对象、面向绝对日前电价、输出校正门控信号**的模型无关校正头里。**脆弱点**：如果实现上只是把残差分类成三态，会退化为平凡分类器——SPCI 已做残差历史条件分位数，CoRel 已做残差上图。

### 模块 3 — 极端特征修补器（三档动作 ＋ 正常期 harm 预算 ＋ 相对不校正的增量损失）

| 近邻（真实链接） | 碰撞等级 | 一句判断 |
|---|---|---|
| [CRC（arXiv 2512.22428）](https://arxiv.org/abs/2512.22428) | 完整 | 明确以 **Error(Ŷ) ≤ Error(Ŷ_base)** 为目标、用四重安全防火墙（方向门控/分位数截断/逐点选择/shrink-to-base 混合）决定"何时校正、校正多少"——正是"相对不校正的增量损失受控"的完整实现，且保证更严（逐点＋验证级＋概率近似非降级）。 |
| [COSA: Context-aware Output-Space Adapter（ICLR 2026）](https://iclr.cc/virtual/2026/poster/10010061) | 完整/组成 | 上下文向量（近期真值统计＝一种"感应"）＋ **tanh(g) 门控标量**残差校正：ŷ = base + tanh(g)·(W·[base‖C] + b)。**连续门控是"三档动作"的平滑化**——三档只是把门控离散化，不是新机制。 |
| [Conformal Policy Control（ICML 2026）](https://icml.cc/virtual/2026/poster/61296) | 完整 | 安全参考策略作为"概率调节器"＋共形校准决定"新策略能多激进"且可证明满足风险容忍度——**harm 预算思想的有限样本实做**（对非单调有界约束也有保证）。 |
| [Prediction Sets for Counterfactual Decisions / PC-RACP（arXiv 2607.02206）](https://arxiv.org/abs/2607.02206) | 组成 | 结果依赖动作（校正与否改变实现残差）的反事实设定＋ policy-coupled coverage——**"反事实超额损失"概念已被形式化**，且给出有限样本覆盖。 |
| [Conformal Risk-Averse Decision Making with Action Conditional Guarantee（arXiv 2606.05551，ICML 2026）](https://arxiv.org/abs/2606.05551) | 组成 | 逐动作条件保证（pinball 损失最小化）——"校正动作 a 的收益在 a 条件下有保证"已有。 |
| [δ-Adapter（arXiv 2601.20280，ICLR 2026）](https://arxiv.org/abs/2601.20280)；[TAFAS（arXiv 2501.04970，AAAI 2025）](https://arxiv.org/abs/2501.04970)；[RAC: Decision Theoretic Foundations for Conformal Prediction（ICML 2025 Spotlight）](https://icml.cc/virtual/2025/poster/45101)；[UEC / Reviving Error Correction（ICML 2026）](https://arxiv.org/abs/2605.21088) | 组成 | 冻结输出空间编辑/门控适配、风险厌恶 max-min 决策政策——家族级占据。 |

**模块 3 碰撞判定：完整→组成。** 三个子主张逐一被占：
- **三档动作** ≈ CRC 的 shrink-to-base＋逐点选择、COSA/δ-Adapter 的连续 gating——离散化只是简化不是新；
- **相对不校正增量损失** ≈ CRC 的 Corrector's Dilemma 与逐点非降级目标（CRC 的逐点保证比"正常期预算"更强）；
- **正常期 harm 预算** ≈ Conformal Policy Control 的共形调节器＋LTT 风险控制族；
- **反事实表述** ≈ PC-RACP。

**模块 3 是本架构碰撞最重的一环**——每个子组件都有顶会/顶刊直接近邻。此处 GPT 的清单经独立复核基本属实（仅 CREDO 归属需更正，见附录 A），本报告结论不依赖照抄其判断。

---

## B. 组合碰撞结论

**组合交集是否被完整抢占：部分（未被完整抢占，但每个不可约元素均被单篇近邻占据）。**

**逐问推演：**
1. **两段式（预测器＋后处理，家族 D）能否等价覆盖？——基本能。** CRC 与 COSA 都是「编码/上下文 → 门控校正」的两段结构；用户架构的"感应→修补"恰是 COSA 的骨架（其上下文向量就是极简感应）。"三段串联"在结构上只是把 COSA 的上下文向量升级成图 + 三态残差状态。
2. **图预测器（家族 E）能否覆盖"图→修补"？——大半能。** STOIC/CoRel 已把（残差）图用于后处理；CRC 已把图放进校正器。图阶段在概念上是**可折叠**的（见 §C）。
3. **串成三段是否已有逐字同款？——未见。** 面向**绝对日前电价、双向极端（负价＋尖峰）、模型无关后处理点校正**的「图 → 有符号双尾感应 → 带预算修补」链，在本次聚焦检索中未找到单篇完整占用。

**但必须警惕**：组合未被照抄 ≠ 组合是 novelty。若每个组件都可指认近邻，串联本身不构成任务级创新；审稿人视角下，只有**接口与保证**是新的才站得住。

**若"否"——不可替代交集必须包含的元素（收窄后的真空位）：**
1. **对象**：绝对日前电价的**后处理点校正**——Trading Electrons 做 spread＋交易，STOIC/CoRel 做区间，CRC 做通用多变量非极端，UEC/COSA 做通用非极端。**"双向极端＋绝对价＋点校正＋模型无关"四元交叉点仍空**（与阶段一结论一致）。
2. **输入信号**：把 occurrence–magnitude 两段分解作用在**冻结基座的残差**上（而非价格本身），作为校正门控的结构化、**有符号**输入。
3. **保证**：**极端条件（tail-conditional）下的分布无关、有限样本非降级保证**——现有保证是逐点/验证级平均（CRC）、逐动作边缘（action-conditional CP）、正常期预算（CPC），"极端事件发生时校正不伤、且有收益下界"的条件保证仍是方法级空位。
4. **验证**：≥3 市场（山东＋EPEX/NORDPOOL/NEM 等）＋ 尾部指标（极端事件上的条件误差、事件捕获率、负价/尖峰分位误差）＋ 经济价值指标（度电套利、CVaR）。单市场＋传统指标在 A 会语境下必被拒（阶段一结论，此处重申）。

---

## C. 最小重构建议

1. **模块 1 降级、不单独主张。** 把"图结构构建"折叠进校正器的结构编码器（CRC 路线），或做成**可选的关系残差先验**（CoRel 路线）。若坚持保留独立模块，必须给出「预测列节点 vs 无预测列节点」「图 vs 纯 attention 上下文」两组消融，且结果要诚实——拿不出显著增益就把图砍掉或降级为 ablation 材料。
2. **模块 2 重构表述。** 不主张"两段式/三态"。改为：**"校正导向的有符号双尾残差混合（signed two-tailed residual mixture）"**——把 occurrence–magnitude 分解应用于基座残差，残差状态（负极/正常/正极）作为修补动作的门控信号。卖点是**残差状态估计与校正门控的耦合**，不是分解本身。三态信号要承载市场结构性驱动（新能源出力、must-run、负荷的极端条件），避免纯统计标签退化。
3. **模块 3 换保证。** 弃用"正常期 harm 预算 ε₀"（弱于 CRC 逐点保证、与 CPC 重叠），改为 **tail-conditional 非降级保证**：以极端事件集为条件，证明校正后条件误差以高概率不劣于基座；预算在校准集上以**分布无关、有限样本**方式选择（LTT/Conformal Policy Control 风格）。这是相对 CRC/CPC/action-conditional CP 唯一站得住的方法级增量。若坚持预注册正常期预算，则必须说明它如何映射到极端期的条件保证——否则该预算只是超参。
4. **组合主张钉在接口。** 不主张"三段串联即新任务"，主张"**极端事件信号 → 预算化修补动作**"这一接口在**双向极端电价、模型无关**场景的首个完整实例化，并以多市场＋尾部/经济指标立证。论文标题与贡献句应落在交叉点（如 "model-agnostic correction of extreme electricity prices"），而非落在任一组件。

---

## D. 暂不允许的主张（针对本架构新增）

| 禁区主张 | 为什么不允许 | 被占证据 |
|---|---|---|
| 「图节点含预测列即新」 | 实现细节（加一列作节点） | 无同款但无主张价值；审稿人判 trivial |
| 「三段串联即新任务」 | 组成件的拼接；两段式已被结构覆盖 | COSA（ICLR26）、CRC（arXiv 2512.22428） |
| 「occurrence–magnitude 两段式即新」 | 统计与能源文献做透 | Flinders ICCE26、Loizidis（Applied Energy 2025）、TSEP（EPSR 2021） |
| 「三态信号即新」 | 教科书概念（regime/跳变建模） | 三态 ordered logistic/probit（Zakopianska 2020、Nord Pool）、Karakatsani & Bunn |
| 「三档动作即新」 | 连续门控的离散化 | COSA tanh(g)、CRC shrink-to-base、δ-Adapter（ICLR26）、TAFAS（AAAI25） |
| 「相对基座增量损失受控即新」 | CRC 的 Corrector's Dilemma 直接同款 | CRC（arXiv 2512.22428） |
| 「正常期 harm 预算即新」 | CPC 共形调节器＋风险容忍度；且弱于 CRC | Conformal Policy Control（ICML26） |
| 「counterfactual excess loss 即新」 | 反事实决策/结果依赖动作已有形式化 | PC-RACP（arXiv 2607.02206） |
| 「模型无关即插即用校正头」作为主卖点 | 家族 D 已做透 | UEC（ICML26）、δ-Adapter（ICLR26）、TAFAS（AAAI25）、PnP-Corrector（ICML26） |

---

## E. 交叉验证问题回答（5 条，针对我方骨架）

**Q1：图结构作为显式第 1 模块是否必要？**
**否。** CRC 证明图可内嵌校正器，CoRel 证明残差上图可直接做后处理，STOIC 证明"图＋能源时序＋残差后处理"已存在。独立图模块会给审稿人"换皮/拼接"把柄。唯一出路：用消融证明"特征＋预测列＋时点混合图"在**极端残差估计**上显著优于纯 attention/无图上下文，并诚实报告。

**Q2：三模块串联是否真比两段式多 novelty？**
**无内在增益。** COSA 已是"感应（上下文）→门控修补"两段，CRC 是"图编码→校正"两段。三段＝CRC 的图＋COSA 的感应门控，未引入不可约新机制。唯一可辩的多出是接口层：感应信号做成**有符号双尾残差状态**并接入**极端条件保证**。若实验显示图阶段相对两段无显著增益，砍掉它反而更干净。

**Q3：正常期 harm 预算 ε₀ 是否能实做？**
**概念上可以（CPC/LTT 路线），但校准有错位。** 预算在正常期数据上设定，而模块的收益/风险集中在极端期——"正常期预算＋极端期全量修补"无法直接给出极端条件下的保证。可行改法：(a) 预算改成**极端期条件预算**（用极端事件子集校准，Mondrian/tail-conditional 风格）；(b) 预注册只做规范选择，另配事后 tail 非降级检验。若只做 (b)，"预算"就是超参，主张要相应降级。

**Q4：三态信号与既有极端事件检测/regime 模型的实际区别？**
**区别在对象与用途，不在概念。** 现有三态/两段建模的对象是**价格本身**（预测目标）；本架构的对象是**冻结基座的残差**（校正门控输入）。对象转移有意义（残差状态可度量基座系统性极端偏差），但若只把残差分类成三态，会退化为平凡分类器——SPCI 已做残差历史条件分位数，CoRel 已做残差上图。三态信号必须承载**市场结构性驱动**（新能源出力、must-run、负荷的极端条件）才有区分度。

**Q5：与 CRC 的最小差异是否足够？**
**不够。** CRC 已含：因果图编码＋混合校正器＋四重安全防火墙＋逐点/验证级/概率近似非降级保证，且模型无关、即插即用。仅把场景换成"双向极端电价"是领域适配，A 会大概率拒。必须同时叠加方法级差异：(i) **tail-conditional（极端事件条件）非降级保证**——CRC 的保证是逐点/验证级（平均意义），条件保证不等价于平均保证；(ii) **有符号双尾残差混合**（CRC 无符号/极端概念）；(iii) **≥3 市场＋尾部/经济价值指标**（CRC 只用公开 TS benchmark）。三条缺一不可。

---

## 附录 A：锚点独立复核披露（诚实声明）

| 锚点 | 复核结果 |
|---|---|
| CRC / COSA / DEFT / Trading Electrons / action-conditional CP / PC-RACP / CoRel / STOIC / Conformal Policy Control / RAC | 均独立确认，链接见上文各表 |
| **SCoRE、LEC、TAG-EPF** | 按缩写精确检索**未能定位**可核实的对应论文。特别更正：搜索引擎曾把 [arXiv 2505.13243](https://arxiv.org/abs/2505.13243) 误标为 SCoRE，实际该页是 **CREDO**。但这些缩写被引用的用途（selection-/action-conditional conformal；图注意力 EPF）已由本报告独立核实的 CPC、action-conditional CP、CREDO、ASTGCN 等覆盖，碰撞结论不受影响。 |
| **CREDO 归属更正** | CREDO 实为 *Conformalized Decision Risk Assessment*（[ICLR26 poster](https://iclr.cc/virtual/2026/poster/10006644) / [arXiv 2505.13243](https://arxiv.org/abs/2505.13243)），做的是决策风险上界估计（inverse optimization＋生成式共形），**不是** action-conditional conformal 本身；"action-conditional" 由 [Zhu et al.（arXiv 2606.05551）](https://arxiv.org/abs/2606.05551) 承担。 |
| **CPC 对应** | 按 ICML 2026 检索，最吻合的是 *Conformal Policy Control*（[poster 61296](https://icml.cc/virtual/2026/poster/61296)，Prinster 等）——安全参考策略调节器＋共形校准决定攻击性程度。 |
| **LEC** | 未能确认对应论文（ICML26 的 *Questioning the Coverage-Length Metric in Conformal Prediction* 标题不符缩写）。建议按原文出处在阶段三补链。 |

*注：附录所列可核实锚点均为 2025–2026 新证据；2012/2004/2020 等"已知锚点"按提示词规则引用但不计入本轮新增证据。*
