# 架构 v2 · 四问碰撞与改良评审

> **审计对象**：架构 v2（结构上下文〔可选〕＋ 极端感应头〔核心〕＋ 极端修补器〔核心〕），贡献落点="极端事件信号 → 预算化修补动作"接口在双向极端电价、模型无关、绝对日前电价后处理点校正场景的首次完整实例化＋tail-conditional 分布无关非降级保证。
> **方法**：聚焦架构级压力测试（不做广撒网）。对 4 个关键研究问题逐一给出 **覆盖判定（覆盖/部分/缺口）＋真实近邻＋可落地改良**。所有链接均已独立核实；中国现货市场事实均附权威来源。
> **独立署名**：本报告与阶段一（`03_交叉验证_...`）、阶段二（`cloud_stage2_3module_collision.md`）相互独立，不覆盖任何已有报告。本报告沿用阶段二已核实的锚点（CRC/COSA/CoRel/STOIC/CPC/PC-RACP/Trading Electrons 等），并针对 v2 新增 Q1–Q4 证据。
> **日期**：2026-07-31

---

## A. 四问覆盖判定表

| 问题 | 判定 | 最相关近邻（真实链接） | 关键风险 |
|---|---|---|---|
| **Q1 多模型适配** | **部分覆盖**（"模型无关"边界未声明） | [CRC（Transformer＋GNN 基座上评估，arXiv 2512.22428）](https://arxiv.org/abs/2512.22428)、[CoRel（基座已捕获时空结构时图校正冗余，arXiv 2502.09443）](https://arxiv.org/abs/2502.09443)、[COSA（线/MLP/Transformer 基座，ICLR26）](https://iclr.cc/virtual/2026/poster/10010061)、[图信号残差白化检验（Polimi）](https://re.public.polimi.it/bitstream/11311/1233899/1/2204.11135.pdf) | 基座残差近似白噪声/基座自带上下文时模块失效或冗余，多基座消融会当场暴露 |
| **Q2 市场迁移** | **部分覆盖**（缺显式市场异质性适配机制） | [Mondrian CP 组条件覆盖（MAPIE 理论）](http://contrib.scikit-learn.org/MAPIE/stable/theory/mondrian/)、[covariate-shift 加权共形（arXiv 1904.06019）](https://arxiv.org/abs/1904.06019)、[ACI 在线自适应（NeurIPS 2021）](https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)、[LTT 风险控制校准（arXiv 2110.01052）](https://arxiv.org/abs/2110.01052)、[Conformal Policy Control（ICML26）](https://icml.cc/virtual/2026/poster/61296)、[Trading Electrons 结构性吃市场规则（arXiv 2601.05085）](https://arxiv.org/abs/2601.05085)、[STOIC 跨能源基准（arXiv 2606.31804）](https://arxiv.org/abs/2606.31804) | 市场规则（负价下限、价格帽、节点/省间、结算机制）当普通特征消费；预算跨市场不可转移 |
| **Q3 负值独到性** | **部分覆盖**（独到性未钉死、"−态"语义未定义） | [Trading Electrons（价格空间 spread 双尾，arXiv 2601.05085）](https://arxiv.org/abs/2601.05085)、[Hybrid Classification-Regression 负电价（ICCE26）](https://researchnow.flinders.edu.au/en/publications/a-hybrid-classification-regression-method-for-forecasting-negativ/)、[Loizidis（Applied Energy 2025）](https://www.sciencedirect.com/science/article/abs/pii/S0306261925007433)、[TSEP（EPSR 2021）](https://www.sciencedirect.com/science/article/abs/pii/S0378779621003977)、[CQR（NeurIPS 2019）](https://nips.cc/virtual/2019/poster/13524) | 与"价格空间两段式负价预测"只差一步；山西（无负价）直接戳穿"−态"定义 |
| **Q4 双提升** | **部分覆盖**（措辞＋实证边界未定） | [CRC 逐点非降级（arXiv 2512.22428）](https://arxiv.org/abs/2512.22428)、[PC-RACP 反事实（arXiv 2607.02206）](https://arxiv.org/abs/2607.02206)、[action-conditional CP（arXiv 2606.05551）](https://arxiv.org/abs/2606.05551)、[RAC（ICML25）](https://icml.cc/virtual/2025/poster/45101)、[SPO+（arXiv 1710.08005）](https://arxiv.org/abs/1710.08005)、[SPO 时序版（arXiv 2411.12653）](https://arxiv.org/abs/2411.12653) | "提升准确率"措辞夸大：保证是"不劣"非"更好"；正常日与极端日证据组未设计 |

**总判定：v2 的骨架（感应→修补→保证）成立，但四问均停在"部分覆盖"——每问都有一个"不补即被审稿人一击"的结构性缺口。** 关键洞察：四问指向同一根源——**模块对"基座残差里到底有没有结构、市场规则怎么进入模块"缺乏显式、可验证的机制**，这四问的改良恰好能把它补成完整故事。

---

## B. 逐问改良建议（可落地设计，非空话）

### Q1 多模型适配——声明边界＋残差结构门控

**边界声明（v2 必须写进论文的假设清单）**：模块对基座的假设只有三条——(i) 输出**点预测**；(ii) 可回算**滚动 OOS 残差**（用于感应头训练与校准）；(iii) 启用图上下文时需**特征矩阵 X 与基座训练数据对齐**。不依赖基座内部结构。这三条同时就是"模型无关"的**可检验定义**。

**失效/冗余情形（诚实标注）**：
- 基座已捕获全部可纠结构（如 STGNN 自身建模了时空依赖）→ 残差近似白噪声 → 校正无红利甚至加噪。CoRel 已明确观察到这一点（[arXiv 2502.09443](https://arxiv.org/abs/2502.09443)），[图信号白化检验](https://re.public.polimi.it/bitstream/11311/1233899/1/2204.11135.pdf)提供了诊断工具。
- Transformer 基座自带上下文/attention → 模块 2 的残差状态信号与基座内部表示高度相关 → 感应头信息增益低。
- 基座是概率/区间预测器 → 模块忽略其不确定性信息，不"失效"但低效。

**新增设计：残差结构门控（residual-structure gate）**。在感应头之前加一个残差白噪声检验统计量（Ljung-Box 或图信号白化检验），统计量低于阈值时强制 `a=0`（不校正）。作用有二：(1) 把"模型无关"从口号变成**可验证性质**——多基座消融时给出"残差结构强度 × 校正增益"的一致散点（增益∝残差结构）；(2) 审稿人问"哪些基座下模块没用"时有**机制性答案**而非嘴硬。

**多基座消融立证清单（必交证据）**：
- 基座集合：AR-Linear / MLP / iTransformer 类 Transformer / 图基座（STGNN）四族，每族 1–2 代表；
- 指标：整体 MAE、极端条件误差、负价捕获率、**非降级率（校正后不劣于基座的样本占比）**；
- 对照：无图上下文版、无结构门控版、[CRC](https://arxiv.org/abs/2512.22428) 版（同族校正头上限）。

### Q2 市场迁移——新增"市场条件校准层"（MCCL），规则从特征升级为先验

**判定**：当前 v2 只有"多市场测试清单"，缺显式异质性适配机制，迁移性主张不成立。需要新增一个**市场条件校准层（Market-Conditional Calibration Layer, MCCL）**，放在感应头与修补器之间，或内嵌于修补器预算校准程序。五个组成（都可模块内实现）：

1. **分桶组条件校准（Mondrian 风格）**：校准数据按 `市场id × 机制标签` 分组——机制标签 = 是否启用负价、结算机制（日前/实时）、定价机制（节点/系统价）。预算 ε 与修正幅度上限 λ 在每桶内独立校准，获得**桶内条件有效性**（Mondrian 组条件覆盖，见 [MAPIE 理论](http://contrib.scikit-learn.org/MAPIE/stable/theory/mondrian/)）。
2. **层次化收缩（解决稀疏极端样本）**：Mondrian 的已知局限是桶太小（极端事件稀疏）时桶内分位数失稳（<200 样本即高危）。用 **pooled 全局残差分布＋每市场偏移** 的层次模型估计分位数，桶样本不足时向全局收缩——这是对 Mondrian 局限的工程应对，也是可写进方法的贡献点。
3. **市场规则显式消费（rule-aware 约束，不当普通特征）**：
   - **山东**：现货价格下限为**申报/一级 −80 元/MWh、出清/二级 −100 元/MWh**，上限 1500 元/MWh（[中国储能网](https://www.escn.com.cn/news/show-2255103.html)）；节点边际电价出清、日前＋实时双市场（[gzpec 解读](http://www.gzpec.cn/news/scyj/202502/t20250213_33927.html)）。→ 修补器输出 `ŷ⁽⁰⁾+a·δ̂` 需 **clamp 到 [下限−ŷ⁽⁰⁾, 上限−ŷ⁽⁰⁾]**；感应头"−态"阈值应取**市场负价下限**而非通用残差分位数——规则作为**先验进入感应头阈值与修补器可行性约束**。
   - **山西**：全国首个正式运行省份（2023-12-22），96 点/天、现货出清上限 1500 元/MWh，**现货合同价格不低于 0 元/MWh（市场机制不设负价，实际最低 0 元）**（[国网山西](http://www.sx.sgcc.com.cn/articles/202408/a1159881.html)、[山西中长期交易细则](https://www.cei.cn:443//defaultsite/s/article/2024/12/31/4b4ff607-92517a19-0194-1bbe399d-23d7_2024.html?referCode=sxp0a&columnId=4028c7ca-37115425-0137-1156ea68-05f1)）。→ 在山西 **"−态"必须自动退化为"近零/极低残差态"**（模块须能表达"该市场无负价"）——这是检验 MCCL 规则表设计的关键案例。
   - 国外（EPEX/NordPool/NEM）：负价下限与触碰频率差异大，全部进规则表字段。
   - 先例背书：[Trading Electrons](https://arxiv.org/abs/2601.05085) 用日前报价栈构建**结构性价格影响模型**（闭式 INC/DEC 解）——"市场规则结构性消费"在能源金融文献是加分项而非特征工程旁路。
4. **在线再校准（ACI 风格）**：市场规则或供需结构漂移时（山东 2025 年曾出现触发二级 −100 限价的连续低价时段），用 [ACI](https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) 式单参数在线更新 ε/λ，保证长期有效覆盖。MCCL 的参数必须是**随目标市场数据滚动**的，而非一次性静态校准。
5. **规则配置表作为模块输入**：每市场一份 `{price_floor, price_cap, settlement_type, granularity, negative_price_enabled}` 配置，非硬编码——这是跨市场泛化的最小工程，也是可复现性卖点。

**证据设计**：5 市场 × {模块开/关 × 基座×2} 矩阵，报告每市场整体＋极端条件指标，并出示 **MCCL 消融**（去掉 MCCL 版在同一市场上的退化量）。

### Q3 负值独到性——把卖点钉死在"残差空间＋后处理＋tail-conditional 保证"

**与 [Trading Electrons](https://arxiv.org/abs/2601.05085) 的差异句（三要素并排写，缺一即被覆盖）**：
1. **空间**：残差空间（基座输出的系统性偏差）vs 价格空间（直接建模 DART spread）；
2. **用途**：后处理点校正（输出可用的修正电价）vs 交易策略信号；
3. **保证**：tail-conditional 分布无关非降级保证 vs 无保证。

**与"两段式负价文献"的差异句**：本模块的 occurrence–magnitude 作用在**基座残差**上、且只用于**决定校正动作**，不是去预测价格本身（对照 [Flinders ICCE26](https://researchnow.flinders.edu.au/en/publications/a-hybrid-classification-regression-method-for-forecasting-negativ/)、[Loizidis AE25](https://www.sciencedirect.com/science/article/abs/pii/S0306261925007433)、[TSEP](https://www.sciencedirect.com/science/article/abs/pii/S0378779621003977)）。答案三要素：对象（残差vs价格）＋动作（校正vs预测）＋保证（条件非降级vs 无）。

**"−态"通用语义定义（必须写出，支撑跨域泛化）**：
- 方向性定义：`G_t=− ⟺ 残差 r_t 落入下尾极端`，即 `r_t ≤ Q_τ⁻(r)` 或 `ŷ⁽⁰⁾ₜ ≤ 市场负价下限`（该市场启用负价时），其中 `τ⁻` 为市场校准的下尾分位数；
- 语义定义："−态＝该时点经济失配方向为**价格过低/负值**（供过于求或制度性下限触碰）"，"+"态对偶为供不应求/尖峰，"0"态为无极端结构信号；
- 跨域定义：把"负值"泛化为"**下尾极端＋方向性经济失配**"——同一模块迁移到负需求、风电预测负误差、电池 SoC 下限等负值序列时，"−态"语义无需重定义。

**尾部指标目标（预设、可验证、给具体数字模板）**：
- 负价捕获率（漏报/虚警，相对基座）：漏报率降低 X%；
- 负价条件误差（负价小时的 MAE / 低分位 pinball）：相对基座下降 X%；
- 极端事件集非降级率：≥ 90%（tail-conditional 保证的**实证覆盖率**）；
- 尾部误差：最差 1% 小时的平均误差、CVaR；
- 对照基线：仅区间不校正的 [CQR](https://nips.cc/virtual/2019/poster/13524)、两段式分类-回归、CRC。

### Q4 准确率与经济效益双提升——诚实措辞＋两组证据

**措辞规则（红线）**：
- 保证句 = "在极端事件集上，校正后条件误差以 ≥1−δ 概率**不劣于**基座（tail-conditional non-degradation）"。**禁止**写成"提升准确率"或"正常日更好"。
- 贡献句 = "在保证正常期与极端期**不劣于**基座的前提下，于极端期实现尾部误差与经济效益的**实证**改善"——"保证不劣＋实证改善"二分，不合并。
- 因为保证是"≤基座"，经济改善只能是**实证**而非保证：修正价格驱动的套利策略样本外收益优于基座价格驱动策略，必须带显著性（Diebold–Mariano 检验或 bootstrap CI）。

**两组证据（G1＋G2 串联）**：
- **G1 正常期保护**：正常日上 校正 vs 基座 的 MAE 比值（期望≈1，受门控 `a=0` 保护）＋**误报率**（0 态被误判为 ± 态的比例＝不必要校正率）——证明"没为极端日牺牲正常日"；
- **G2 极端期提升**：极端事件集上条件误差、捕获率、非降级率、经济指标——证明"极端期更好"；
- 串联逻辑：G1 证"受保护"、G2 证"极端期更好"，合起来才支撑"兼容提升"主张。

**经济价值指标轴（与尾部指标共同构成评估轴）**：
- 度电套利：储能在日前市场用 校正价 vs 基座价 调度充放电，比 PnL 与 CVaR——decision-focused 验证，可吸收 [SPO+](https://arxiv.org/abs/1710.08005)（及处理依赖数据的时序版 [arXiv 2411.12653](https://arxiv.org/abs/2411.12653)）；
- CVaR：交易组合尾部风险；[action-conditional CP](https://arxiv.org/abs/2606.05551) / [RAC](https://icml.cc/virtual/2025/poster/45101) / [PC-RACP](https://arxiv.org/abs/2607.02206) 提供决策集层面工具，可与校正头做端到端对照；
- 消费者侧（可选）：售电公司采购成本。
- **必须避免**：经济指标当"黑箱加分"——审稿人要求经济收益与校正动作可解释关联（哪个信号/哪个动作带来收益）。收益方差大、跨市场不可比（结算规则不同）→ 统一按**每市场内 校正 vs 基座 的相对收益**报告，并给 bootstrap CI。

---

## C. 必须修改的设计变更清单（consolidated must-fix，按"不改会被审稿人致命一击"排序）

| 优先级 | 变更 | 针对问 | 若不改的致命后果 |
|---|---|---|---|
| **P0** | 声明"模型无关"边界假设清单＋新增**残差结构门控**（白化检验触发 `a=0`） | Q1 | 多基座消融中 Transformer/图基座下模块失效，被 [CoRel](https://arxiv.org/abs/2502.09443) 结论直接反驳 |
| **P0** | 新增**市场条件校准层 MCCL**（Mondrian 分桶＋层次收缩＋规则先验约束＋ACI 在线更新） | Q2 | 迁移主张退化为"测试清单"，审稿人问"为何不按市场条件校准"即崩 |
| **P0** | **山东/山西规则显式消费**（负价下限 clamp、山西"无负价→−态降级"）、规则表作模块输入 | Q2/Q3 | 山西验证直接戳穿"−态"定义；规则当普通特征被认为 naive |
| **P0** | 写出"−态"**通用语义定义＋市场自适应阈值**（负价下限 vs 残差分位数） | Q3 | 跨域泛化与负值独到性无法立论 |
| **P1** | 贡献句改为"**保证不劣＋实证极端期改善**"二分，删除"提升准确率"字样 | Q4 | 与 [CRC](https://arxiv.org/abs/2512.22428) 逐点保证对比时被指夸大 |
| **P1** | 实证清单固定：G1（正常期保护＋误报率）＋G2（极端期提升）＋残差结构散点＋5 市场矩阵；经济指标带显著性检验 | Q4 | "双提升"无证据组支撑 |
| **P2** | 图上下文维持"可选"，启用判据＝残差结构检验显著 | Q1 | "图预测换皮"复审回归（阶段二结论） |

---

## D. 暂不允许的主张（v2＋四问新增禁区）

| 禁区主张 | 为什么不允许 | 被占证据/理由 |
|---|---|---|
| 「模块能适配任意模型算法」 | 存在明确边界：基座残差白/基座已捕获结构时冗余 | [CoRel](https://arxiv.org/abs/2502.09443)、[白化检验](https://re.public.polimi.it/bitstream/11311/1233899/1/2204.11135.pdf) |
| 「市场迁移＝多市场测试清单」作为贡献 | 缺显式适配机制；现成程序已存在 | [Mondrian](http://contrib.scikit-learn.org/MAPIE/stable/theory/mondrian/)、[ACI](https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)、[LTT](https://arxiv.org/abs/2110.01052) |
| 「山东负电价经验可直接迁移山西」 | 山西现货价格下限 0 元、不设负价，"−态"不成立 | [国网山西](http://www.sx.sgcc.com.cn/articles/202408/a1159881.html) |
| 「−态＝负价」作为通用定义 | 需方向性＋市场自适应阈值定义才能跨域 | 本报告 §B-Q3 定义 1–3 |
| 「模块提升准确率」 | 保证是"不劣"，不是"更好" | [CRC](https://arxiv.org/abs/2512.22428) 的 Error(Ŷ)≤Error(Ŷ_base) |
| 「harm 预算覆盖正常期」 | 预算只在极端子集校准，正常期靠门控保护 | 本报告 §B-Q4 措辞红线 |
| 「经济收益是保证的一部分」 | 经济是实证结果，非保证 | — |
| 「有符号双尾＝负价＋尖峰即独到」 | 双尾任务已被 [Trading Electrons](https://arxiv.org/abs/2601.05085) 做（价格空间） | [arXiv 2601.05085](https://arxiv.org/abs/2601.05085) |
| **阶段二禁区仍全部有效**（三段串联即新、两段式即新、三态即新、三档动作即新、图节点含预测列即新、模型无关校正头为主卖点） | 阶段二碰撞审计结论 | `cloud_stage2_3module_collision.md` |

---

## 附：本报告引用的新增验证锚点

| 论文 | 链接 | 用途 |
|---|---|---|
| ACI：Adaptive Conformal Inference Under Distribution Shift（NeurIPS 2021） | [NeurIPS](https://papers.nips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) / [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.00170) | MCCL 在线再校准 |
| EnbPI：Conformal Prediction Interval for Dynamic Time-Series（ICML 2021 oral） | [arXiv 2010.09107](https://arxiv.org/abs/2010.09107) / [GitHub](https://github.com/ConformalPrediction/EnbPI) | 滚动残差共形的滚动窗口机制 |
| Conformal Prediction Under Covariate Shift | [arXiv 1904.06019](https://arxiv.org/abs/1904.06019) | 分布漂移加权共形 |
| CQR：Conformalized Quantile Regression（NeurIPS 2019） | [NeurIPS poster](https://nips.cc/virtual/2019/poster/13524) | 尾部区间基线 |
| LTT：Learn then Test（NeurIPS 2021） | [arXiv 2110.01052](https://arxiv.org/abs/2110.01052) | 预算校准程序（多重假设检验） |
| SPO+（Management Science 2022）／时序版 | [arXiv 1710.08005](https://arxiv.org/abs/1710.08005)／[arXiv 2411.12653](https://arxiv.org/abs/2411.12653) | 经济价值决策聚焦验证 |
| 山东现货市场运行与限价 | [gzpec](http://www.gzpec.cn/news/scyj/202502/t20250213_33927.html)／[山东省发改委](http://fgw.shandong.gov.cn/art/2025/5/16/art_91548_10466800.html)／[中国储能网](https://www.escn.com.cn/news/show-2255103.html)／[新浪财经](https://finance.sina.com.cn/wm/2026-06-02/doc-inhzzcpr1629033.shtml) | 市场规则消费、MCCL 案例 |
| 山西现货市场运行与限价 | [国网山西](http://www.sx.sgcc.com.cn/articles/202408/a1159881.html)／[山西中长期交易细则](https://www.cei.cn:443//defaultsite/s/article/2024/12/31/4b4ff607-92517a19-0194-1bbe399d-23d7_2024.html?referCode=sxp0a&columnId=4028c7ca-37115425-0137-1156ea68-05f1)／[nengyuanquan](http://nengyuanquan.com/index.php/article/chuneng/6711.html) | 市场规则消费、无负价案例 |
