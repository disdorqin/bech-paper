# Stage 2.5 补强报告：BOM-SSC × SCARR 2025–2026 Novelty 碰撞审计

**检索截止：**2026-07-31  
**检索范围：**仅核查 2025–2026 年已接收论文与公开预印本；不重做阶段一/阶段二的广撒网检索。  
**去重规则：**先从阶段一、阶段二正文抽取 89 个既有 URL；下列碰撞表中的 10 篇论文均未在该 URL 集中出现。PIR、δ-Adapter、Post-Training Corrections、UEC-STD、ACI、Conformal PID、KOWCPI、ECI、Conformal Risk Training 等仅作为既有锚点，不冒充本轮新增证据。  
**来源边界：**“主调研方 90 篇底座”的跨家族结论来自 WorkBuddy 提供的汇总；本报告用 2025–2026 原始论文对其中与 BOM-SSC、SCARR 直接相关的主张做增量核验。

> **结论先行**
>
> 1. **BOM-SSC：有实质碰撞，但经收窄后仍可保留。**没有检到一篇工作同时占据“冻结异构电价基座 + 绝对电价正/负双尾 + occurrence–magnitude 分解 + 选择性校正/回退 + 正常期动作诱发损失预算”这一完整交集；然而，通用安全残差校正、冻结输出编辑、门控测试时适配、双向电价尖峰决策等组成部分均已被占据。因此原来的宽泛表述不再安全。
> 2. **SCARR：宽泛的独立 novelty 已不成立。**2026 年已出现 action-conditional conformal guarantee、selection-conditioned conformal routing、conformal decision-risk certificate 和 conformal policy control。SCARR 不能再声称“首次校准动作风险”，应降为 BOM-SSC 内部风险控制层；若要独立成贡献，必须给出超越这些工作的“有符号尾组 × 校正动作 × 延迟时序反馈”新定理。
> 3. **最终结构仍是一篇论文、一条主线：**BOM-SSC（重构版）为任务/方法主贡献，SCARR（收窄版）为校正动作的风险层；跨市场、经济价值和因果图分别作为验证轴、次级效用指标和可选上下文。

---

## A. 正式共识度矩阵（升级版 §5.2）

### A.1 口径

- 这里的“正式”是指：**GPT 阶段二裁决已与主调研方（WorkBuddy）90 篇底座的最终跨家族结论完成逐项对照**。
- 由于仍未收到其他 AA 级 Agent 的逐篇阶段一报告，它**不是**“≥2/3 Agent 投票”的正式多智能体票数。该列准确名称应理解为“当前可审计证据共识度”。
- WorkBuddy 的 C1/C3/C4 是原始候选，不等于最终保留意见。下表同时保留“最初提出”与“90 篇底座综合后如何处置”，避免把候选误写成定论。

| 候选 | GPT 阶段二 | 主调研方（WorkBuddy）90 篇底座 | 正式共识度 | 备注／冲突 |
|---|---|---|---|---|
| 普通“模型无关、冻结基座、输出端残差校正头” | 砍掉独立 novelty | C1 原始候选提出；G3 又指出通用性边界未标定，最终不能仅靠“即插即用”成立 | **高：同意砍宽泛表述** | 无最终冲突；C1 是起点，不是终局。本轮 CRC、COSA、DEFT 进一步封死宽泛 claim |
| **BOM-SSC：双向尾选择性安全校正器** | 保留为主线 | C1 的“检测→触发→校正→回退”与 G1/G2 的“双尾修复层缺失”共同支持 | **高：同向保留，但必须收窄** | 共识核心不是模型无关，而是电价双尾、动作诱发损失与正常期保护的交集 |
| 双尾 occurrence–magnitude 联合建模 | 保留 | C1 含检测与校正闭环，G2 强调负价独立价值，但未显式给出 two-part 统计分解 | **中高：问题同向、方法部分互证** | WorkBuddy 支持“双尾专门处理”，但 occurrence–magnitude 分解仍主要来自 GPT 方案 |
| 条件共形 + 残差闭环 | 改写为校正动作风险层 | C2 原始候选与 G4“极端子集条件覆盖”直接支持 | **高：问题共识；机制需重构** | 不能再写成普通条件覆盖；2026 action-conditional / routing 工作要求 SCARR 改成更窄的校正增量损失对象 |
| 正常期显式退化预算／安全回退 | 保留为关键约束 | G1 与“整体精度优先 vs 尾部事件优先”的深层矛盾直接支持；C1 含严重失效回退 | **高：同向保留** | “非退化”本身已被 CRC 占据；剩余空间是**正常子群的动作诱发正损失预算**，不是全局 NDR |
| 经济价值对齐 loss | 降为评估／约束 | C4 原始候选提出；90 篇底座综合后的 Stage 2.5 处置同意降级 | **高：最终处置同向** | 原始候选与最终处置表面相反，但属于“提出后被证据降级”，不是当前冲突 |
| 跨市场／少样本安全迁移 | 降为验证协议 | 未列为四个核心候选；G3 支持标定通用性边界 | **高：同意不做独立 novelty** | leave-one-market-out、跨基座安全边界仍必须做，否则“通用”无法成立 |
| 动态因果特征图 | 砍主线；可选上下文／消融 | C3 原始候选提出；Stage 2.5 最终处置同意不作为主创新 | **高：最终处置同向** | 原始候选被降级；没有可识别性与干预证据时不能称因果 |
| 统计—事件—安全—经济多目标预算 | 并入约束与评估 | C4 与整体/尾部矛盾部分支持，但未提出同一数学对象 | **中高：用途同向** | 不拆成第三个网络；经济指标只能作为次级效用或 Pareto 轴 |

### A.2 冲突审计

**未发现 GPT 阶段二最终去留与 WorkBuddy 90 篇底座最终处置之间的实质冲突。**需要保留的三处“历史差异”是：WorkBuddy 最初把普通校正头、动态图、经济 loss 列为 C1/C3/C4 候选，后来综合底座证据将其分别收窄、降级或移出主线。报告中必须写成“候选演化”，不能倒叙成“WorkBuddy 从未提出”。

仍待补的是其他 Agent 的原始报告；在它们到达前，不得把本表改写成“多智能体 ≥2/3 正式投票共识”。

---

## B. 2025–2026 碰撞检查

### B.0 判定标准

- **完整碰撞：**论文已同时覆盖候选的任务对象、风险对象、动作机制和保证。
- **直接组成碰撞：**论文占据候选的一项核心 novelty，足以推翻相应宽泛 claim，但没有覆盖整个组合。
- **部分碰撞：**接口或机制相近，必须作为强基线或相关工作，但仍留有明确差异。
- 只判断公开文本可支持的内容；“不同应用领域”不能自动消除方法或理论层面的碰撞。

### B.1 BOM-SSC

| 新论文 | 会议／年份 | 是否构成碰撞 | 一句判断 |
|---|---|---|---|
| [Causality-Inspired Safe Residual Correction for Multivariate Time Series](https://arxiv.org/abs/2512.22428) | arXiv，2025 v1／2026 v2 | **直接组成碰撞：高** | CRC 已提出 plug-and-play 安全残差校正、四重门控/裁剪/选择/回退机制及 non-degradation rate；因此“安全残差校正”“避免过校正”“非退化”均不能再单独宣称新颖，但它未处理电价正负双尾、occurrence–magnitude 分解或正常子群预算 |
| [COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting](https://openreview.net/forum?id=L7Z5wBMPrW) | ICLR 2026 | **直接组成碰撞：高** | COSA 直接修正冻结基座的输出，并用门控控制修正强度以抑制过校正；它封死“冻结输出端 + 门控 + 架构无关”的宽泛 novelty，但没有双尾任务、动作风险证书或显式正常期损失预算 |
| [Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation](https://arxiv.org/abs/2501.04970) | AAAI 2025 | **部分碰撞：中** | TAFAS 已用部分可观测真值与 gated calibration 做模型无关测试时适配；它占据“用滚动反馈门控适配”的接口，差异仍在双尾校正目标、冻结残差对象和安全预算 |
| [Expert-Guided Forecast Editing for Time-Series Foundation Models](https://arxiv.org/html/2607.19659v1) | arXiv，2026-07 | **部分碰撞：中高** | DEFT 已把“冻结 TSFM 的输出轨迹编辑”定义为任务，并在多基座、多数据集上做预算化选择；它依赖专家评分与趋势—季节搜索，不做电价双尾、残差 two-part 建模或无专家的正常期安全约束 |
| [Trading Electrons: Predicting DART Spread Spikes in ISO Electricity Markets](https://arxiv.org/abs/2601.05085) | arXiv，2026 | **直接概念碰撞：高** | 该文已统一处理正/负 DART spread spikes，并把方向信号转成选择性交易与仓位；“双向电价尖峰 + 选择性动作”的大口径已被占据，但它不是对冻结绝对电价预测器的 post-hoc correction，也没有正常期预测退化预算 |

#### BOM-SSC 碰撞结论

**未发现完整碰撞，但发现五条实质组成碰撞。**因此不能写“截至 2026-07 novelty 暂稳”；准确结论是：

> 截至 2026-07-31，未检到抢占 BOM-SSC **完整交集**的工作；但“通用安全残差校正”“冻结输出编辑”“门控适配”“双向电价尖峰选择性决策”均已有直接近邻。BOM-SSC 只有把创新核收窄到“绝对电价有符号双尾的 two-part 后处理 + 基于相对基座增量损失的正常子群预算”后，novelty 才暂时可辩护。

#### BOM-SSC 最小重构

保留名称，但把 **Safe** 从模糊的“总体不退化”重定义为可检验的**动作诱发损失约束**：

\[
\Delta \ell_t(a)=\ell\!\left(y_t,\hat y_t^{(0)}+a\hat\delta_t\right)-\ell\!\left(y_t,\hat y_t^{(0)}\right),
\qquad a\in\{0,\lambda,1\}.
\]

其中 \(a=0\) 为回退基座，\(a=\lambda\) 为保守校正，\(a=1\) 为完整校正。正常期安全对象不是总体 MAE，而是例如

\[
\mathbb E\!\left[(\Delta\ell_t(a_t))_+\mid G_t=\mathrm{normal}\right]\le \varepsilon_0,
\]

并同时报告正常期最坏日／高分位 harm，而不是只给平均 non-degradation rate。

重构后的不可替代交集必须全部出现：

1. **对象限定：**预测绝对电价的冻结异构基座；正尾尖峰与负电价用训练期／校准期预注册阈值分别定义，不把 DART spread spike 偷换成 absolute price tail。
2. **two-part 校正：**分别估计正/负尾 occurrence，以及事件条件下的 residual magnitude；不能只做一个连续残差头。
3. **动作而非幅值门控：**输出 \(\{0,\lambda,1\}\) 三种校正动作，并以相对原预测的增量损失决定是否执行。
4. **正常子群预算：**把正常期 harm 上限作为硬约束或校准目标；总体 NDR 仅是辅助指标。
5. **通用性边界：**至少在 LEAR、LightGBM、PatchTST、Chronos-2 等冻结基座和多个负价/尖峰结构不同的市场上报告何时有效、何时 abstain。

**最小实验闭环：**按市场做严格 rolling-origin train/calibration/test；基座残差必须来自 out-of-fold 或 rolling-OOS 预测；尾阈值、路由阈值和风险预算只在 train/calibration 决定。市场至少包含两个负价丰富市场与一个尖峰主导市场。指标同时覆盖 overall MAE、正/负事件 AUPRC/F1、条件幅值误差、正常期平均/高分位 harm、校正率/回退率和跨基座最差退化；经济收益与 CVaR 作为次级验证，不作为 BOM-SSC 的 novelty 定义。

---

### B.2 SCARR

| 新论文 | 会议／年份 | 是否构成碰撞 | 一句判断 |
|---|---|---|---|
| [Conformal Risk-Averse Decision Making with Action Conditional Guarantee](https://arxiv.org/abs/2606.05551) | ICML 2026 | **直接理论碰撞：极高** | 该文已显式提出 action-conditional conformal prediction，并对每个采取的动作给出条件安全保证；因此“首次对动作做条件共形保证”不可再用 |
| [LEC: Linear Expectation Constraints for Selection-Conditioned Risk Control in Selective Prediction and Routing Systems](https://arxiv.org/abs/2512.01556) | ICML 2026 | **直接路由碰撞：极高** | LEC 已控制被选择样本的条件错误率，并扩展到两模型 routing；因此“首次用共形/校准风险决定接受、拒绝或路由”不可再用 |
| [Conformalized Decision Risk Assessment](https://arxiv.org/abs/2505.13243) | ICLR 2026 | **直接上位概念碰撞：高** | CREDO 已为给定决策提供分布无关的近最优概率证书；它不针对时序校正，但足以推翻“首次校准执行一个决策的风险” |
| [Conformal Policy Control](https://arxiv.org/abs/2603.02196) | ICML 2026 | **直接机制碰撞：高** | CPC 用安全参考策略的校准数据调节新策略能偏离多远，并保证用户风险阈值；这与“从基座到完整校正之间选择强度”的抽象结构高度相似 |
| [Conformal Selective Prediction with General Risk Control](https://arxiv.org/abs/2603.24704) | arXiv，2026 | **直接选择性碰撞：高** | SCoRE 可对任意训练模型和有界连续风险作 trust/abstain 二元选择，并控制被信任样本的有限样本风险；因此“选择性共形 + abstain + 一般风险”已被占据 |

#### SCARR 碰撞结论

**SCARR 的原始宽泛差异化声明已被推翻。**尤其是“校准执行动作的风险，而不是预测区间”已经不足以区分 2026 年 action-conditional conformal、decision-risk、policy-control 与 selection-conditioned routing。

准确结论是：

> SCARR 不能作为独立的第二个通用方法 novelty。它只能作为 BOM-SSC 内部的领域化风险层：控制的是相对冻结基座的**校正动作诱发增量损失**，并按负尾／正常／正尾与校正动作联合分组，在延迟滚动反馈下决定回退、缩减或完整校正。

#### SCARR 最小重构

1. **把风险变量钉死：**

   \[
   L_t^{\mathrm{act}}(a)=
   \left[
   \ell\!\left(y_t,\hat y_t^{(0)}+a\hat\delta_t\right)
   -
   \ell\!\left(y_t,\hat y_t^{(0)}\right)
   \right]_+.
   \]

   这不是预测区间宽度、基础预测误差或交易收益，而是“执行该校正相对不校正新增了多少损失”。

2. **把条件结构钉死：**联合索引为 signed-tail group \(G\in\{- ,0,+\}\) 与 correction action \(A\in\{0,\lambda,1\}\)，而非只有 accept/reject。
3. **把信息时序钉死：**真实 \(y_t\) 只在事后到达；校准必须 rolling、delayed、无未来泄漏。若理论仍假设 exchangeability，必须明确标注；不能把 i.i.d. 保证直接搬到电价依赖序列。
4. **把与现有理论的关系写清：**二元选择时应与 LEC/SCoRE 对照；按动作条件保证时应与 Zhu et al. 对照；安全基座到新策略的插值应与 CPC 对照。只有新的 signed-group × action × delayed-time-series 定理才能支撑 SCARR 独立成为理论贡献。
5. **无新定理时的定位：**称为“domain-specific calibration layer”或“risk-control instantiation”，不称为新的通用 conformal framework。

#### 若要恢复 SCARR 的独立贡献，最低理论门槛

至少证明并验证以下一项，而不是只换损失函数：

- 在预先固定的 signed-tail groups 与多校正动作上给出**同时**有限样本风险控制；
- 在有明确 mixing／weighted exchangeability／online regret 假设的延迟时序下给出有效保证；
- 证明三动作校正路由相较二元 selection 的非平凡可行域或效用改进，并在退化到二元情形时与 LEC/SCoRE 一致；
- 给出 normal-regime harm budget 与 tail-repair utility 的可证 Pareto 可行性，而不只是经验调参。

否则，SCARR 保留为 BOM-SSC 的工程性风险机制即可。

---

### B.3 最小重构后的最终去留

| 候选 | Stage 2.5 去留 | 原因 |
|---|---|---|
| **BOM-SSC（重构版）** | **保留为唯一主创新** | 完整交集未见抢占，但必须删除通用“安全残差校正/冻结输出编辑”首创表述 |
| **SCARR（收窄版）** | **保留为 BOM-SSC 内部风险层** | 广义 action-risk / routing novelty 已被 2026 工作占据；剩余价值在特定风险对象和 signed-tail 时序实例化 |
| 普通模型无关校正头 | 砍掉 | CRC、COSA、δ-Adapter、DEFT 等已使表述拥挤 |
| 跨市场少样本迁移 | 验证轴 | 用于证明通用性边界，不独立宣称方法创新 |
| 动态因果图 | 可选消融／砍主线 | 与核心碰撞无关，且会引入识别性负担 |
| 经济价值对齐 loss | 次级约束／评估 | 不再是独立 novelty；用于验证修正是否产生真实决策价值 |

---

## C. 钉死版差异化声明 + 暂不允许主张（扩充版）

### C.1 BOM-SSC：钉死版差异化声明

> **BOM-SSC 不声称首创安全残差校正、冻结输出编辑或门控适配。它研究的是更窄的电价修复问题：对多个冻结异构基座产生的 rolling-OOS 残差，分别建模负电价与正向尖峰的 occurrence 和 conditional magnitude，并以“相对不校正所新增的损失”为校正动作风险，在预注册的正常期 harm budget 下选择回退、缩减或完整校正。**

与最相关工作的最短差异：

- **vs CRC：**CRC 控制通用多变量残差校正的非退化；BOM-SSC 必须额外解决有符号绝对电价双尾、two-part 事件结构和正常子群动作 harm，而不能把 NDR 改名。
- **vs COSA／δ-Adapter：**它们提供冻结模型的通用输出适配；BOM-SSC 的决策对象是双尾校正动作及其相对基座损失，不是一般漂移下的输出修正。
- **vs DEFT：**DEFT 用有限专家查询编辑冻结 TSFM 轨迹；BOM-SSC 不依赖测试时专家评分，而从严格 OOS 历史残差学习双尾修复与回退。
- **vs Trading Electrons：**后者直接预测正/负 DART spread spikes 并优化交易；BOM-SSC 修正的是冻结 forecaster 的 absolute electricity-price forecast，并显式保护正常期预测。
- **vs PIR：**PIR 的通用 failure identification/revision 不能替代预先定义的 signed-tail occurrence–magnitude 任务与正常子群动作损失约束。

**审稿人 30 秒 pitch：**

> 当冻结电价模型漏掉负价或正向尖峰时，BOM-SSC 先判断“是哪一侧尾部、是否发生、幅值差多少”，再只在校正相对原预测的新增损失受控时执行修正；正常日宁可回退，也不拿整体稳定性换尾部指标。

**会议适配：**当前形态最适合 KDD／WWW／WSDM／AAAI／IJCAI 的“新任务 + 方法 + 严格多市场验证”。若冲击 NeurIPS／ICML／ICLR，需增加可复用的动作 harm 约束理论或公开多市场 benchmark，而不能只靠模块组合。

### C.2 SCARR：钉死版差异化声明

> **SCARR 不声称首创 conformal action-risk、action-conditional guarantee、selective conformal prediction 或 risk-controlled routing。它是 BOM-SSC 的校准层：对 \(\{回退, 缩减, 完整校正\}\) 三种动作相对冻结基座产生的 counterfactual excess loss 做校准，并按负尾／正常／正尾分组，在严格延迟 rolling protocol 下选择校正强度。没有新的 signed-group × action × time-dependence 保证时，SCARR 只是领域化实例，不是独立通用理论。**

与最相关工作的最短差异：

- **vs action-conditional conformal prediction：**现有工作按动作给安全保证；SCARR 只能在风险对象被限定为“post-hoc correction 相对 base 的增量损失”、且增加 signed-tail 与时序延迟结构时形成差异。
- **vs LEC／SCoRE：**现有工作已控制被选择／被信任样本的条件风险；SCARR 需要证明三动作 correction routing 与 signed groups 不只是二元 accept/reject 的重命名。
- **vs CREDO：**CREDO 已提供通用决策风险证书；SCARR 的剩余空间是可观测延迟残差驱动的电价校正动作，不是“决策风险”概念本身。
- **vs CPC：**CPC 已校准安全参考策略与新策略之间的偏离；SCARR 若用 base-to-corrected 插值，必须明确其 signed-tail excess-loss 保证比通用策略插值多了什么。

**审稿人 30 秒 pitch：**

> SCARR 不给预测区间贴一层共形标签，而是估计“这一次把基座预测改成缩减或完整校正，会比保持原样多造成多少损失”，并分别对负尾、正常期和正尾控制这种动作风险。

**会议适配：**无新定理时作为 BOM-SSC 内部机制，适配 KDD／WWW／WSDM／AAAI／IJCAI；若能证明 signed-group × action × delayed-time-series 的同时风险控制，再考虑 ICML／NeurIPS／ICLR 的独立理论贡献。

### C.3 暂不允许的主张（扩充版）

#### 阶段二原清单继续禁止

1. “首次提出模型无关时序校正头”。
2. “首次将共形预测用于电价”。
3. “首次使用 CVaR／套利损失优化储能收益”。
4. “动态图边就是因果关系”。
5. “在山东有效即可证明跨市场通用”。
6. “边际 coverage 达标即可证明极端尾部可靠”。
7. “总体 MAE 提升即可证明校正安全”。

#### 本轮碰撞检查新增禁止

8. **“首次提出安全／非退化的 residual correction。”**CRC 已直接占据该表述。
9. **“首次对冻结时序模型做输出空间编辑／门控修正。”**COSA、δ-Adapter、DEFT 等已覆盖。
10. **“首次把正向和负向电价尖峰统一建模并选择性行动。”**Trading Electrons 已覆盖至少 DART spread 场景；必须限定 absolute price post-hoc correction。
11. **“首次用共形方法校准执行动作的风险。”**CREDO、action-conditional conformal prediction、CPC 已推翻。
12. **“首次提出 selection-conditioned conformal routing／conformal abstention。”**LEC、SCoRE 等已覆盖。
13. **“SCARR 与已有工作不同，因为它不是预测区间而是动作。”**2026 年的直接理论近邻已经也是动作、决策或路由。
14. **“三档动作天然比二元接受／拒绝新。”**必须有不可约的目标、定理或实证，增加一个 \(\lambda\) 不构成 novelty。
15. **“action-conditional guarantee 自动等于 signed-tail group-conditional guarantee。”**动作组与尾部组是不同条件事件，需要分别或联合证明。
16. **“在 i.i.d. 校准集上的保证可原样迁移到电价滚动序列。”**必须声明 exchangeability、mixing、weighted/online 条件或只作经验校准。
17. **“用同一批 calibration 数据选尾阈值、调路由并报告 exact finite-sample guarantee。”**数据自适应选择可能破坏保证，需数据切分、预注册或校正。
18. **“低 abstention rate 证明风险被控制。”**覆盖/风险有效性与效率是两个指标。
19. **“总体 NDR 高证明正常期受保护。”**必须单独报告 normal-group positive harm、最坏日或高分位退化。
20. **“没找到完整组合论文，所以可以写‘首次’。”**本轮只做聚焦碰撞核查，不是可证明穷尽的全球优先权检索。
21. **“冻结基座就等于模型无关。”**必须跨统计、树、深度和 TSFM 基座验证，并报告失败边界。
22. **“DART spread 的正负 spike 等同于绝对电价的负价与正向尖峰。”**两者任务、基准和经济语义不同。
23. **“校正后的交易收益上升证明统计校准有效。”**经济效用不能替代尾部条件风险与校准诊断。
24. **“counterfactual correction loss 在决策时已知。”**它依赖事后真值，只能通过独立历史校准或有效在线机制估计。

### C.4 可安全使用的贡献表述

在完成对应实验／证明后，可使用以下克制版本：

1. “我们定义并系统研究冻结电价预测器的 signed-tail post-hoc correction 问题，同时评估 occurrence、conditional magnitude 与 normal-regime action harm。”
2. “我们提出 BOM-SSC，将双尾 two-part residual modeling 与基于相对基座增量损失的三档校正动作结合。”
3. “我们将 action-/selection-conditioned risk-control 思路实例化到电价校正动作，并在严格 rolling-OOS 协议下分别审计负尾、正常与正尾风险。”
4. “在本次截至 2026-07-31 的聚焦检索中，我们未发现同时覆盖上述完整交集的论文；该陈述不是全球首创权证明。”

---

## 最终裁决

**推荐投稿叙事：一个主创新 + 一个风险层。**

- **主创新：BOM-SSC（重构版）**——保留。
- **风险层：SCARR（收窄版）**——保留，但不单独宣称通用 novelty。
- **最关键的审稿防线：**所有贡献都围绕“相对冻结基座的校正动作诱发损失”展开；正常期安全用独立子群预算验证；双尾阈值与校准严格无泄漏；与 CRC、COSA、Trading Electrons、action-conditional CP、LEC、CREDO、CPC 做正面对照。

若无法补出 SCARR 的新时序分组理论，论文仍可成立，但标题、摘要和贡献列表应只把 BOM-SSC 列为方法创新，把 SCARR 写成风险控制实现。若连 normal-group action harm 和 two-part signed-tail 消融也无法做出稳定增益，则 BOM-SSC 应退回“应用型系统/benchmark”，不再以通用方法投稿。
