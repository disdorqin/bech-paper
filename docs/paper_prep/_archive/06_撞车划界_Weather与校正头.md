# 06 · 撞车划界：Weather 论文与校正头家族（第三轮检索B）

> 生成：2026-08-07 ｜ 执行：freebuff（任务 t20ec0fd2）｜ 验收人：claude_code（导师）
> 目的：确认 BECH 最大撞车风险（Weather 论文 arXiv 2606.19642）并精确划界，同时把校正头家族（CRC / δ-Adapter / UEC / PnP-Corrector / Post-Training Corrections）与 BECH 逐维对比。
> 依据：6 篇论文全文精读（PDF 已存 `D:\AI_Memory\papers\raw\*.pdf`，全文文本 `papers\txt\`），背景见 05_论文准备建议报告_综合.md。

---

## 0. 结论摘要（先看这个）

**BECH 的定位不需要改，但措辞必须改。** Weather 论文确实完整占据了「外挂冻结基座的模型无关在线共形后处理」这个槽位，但它**没有**做：tail-conditional 认证、未认证即弃权、电价域、双尾非对称——这四点恰好是 BECH 的结构支柱，且 Weather 自己把「上下分别共形校正」「针对极端的共形方法」列为 **future work**。校正头家族五篇全部是**通用时序/时空域、无认证、无弃权、无尾部条件化**。

- ✅ **BECH 仍能安全主张（5 点）**：① 负电价专用 + 正尖峰主动弃权的结构性非对称；② 三段 tail-conditional 认证（条件保证而非边际保证）；③ 分布自由的「未认证即弃权」零退化；④ 以经济价值（储能利润/regret/CVaR）为认证目标；⑤ 点预测修正（改预测值而非区间）。
- ❌ **必须放弃（3 点）**：① 「首次模型无关共形后处理」；② 「边际覆盖在极端子集失效」当创新贡献（只能当动机）；③ 「首次安全/零退化残差校正」「首次模型无关校正头」等「首次」措辞。

---

## 1. Weather 论文精读（arXiv 2606.19642，最危险撞车点）

**文献**：Asch, Rossellini, Hassanzadeh & Willett, "Rigorous uncertainty quantification of probabilistic AI weather forecasts with conformal prediction", arXiv:2606.19642v1 (physics.ao-ph, 2026-06-17)。

**方法实质**：对 GenCast / NeuralGCM / AIFS-ENS 三个全球 AI 天气模型的温度、降水集合预报，应用 **在线自适应共形预测**（Gibbs & Candès 2021 式），对原始集合区间加**单一标量 padding** `c_t`：`Ĉ_t = [q̂_lo − c_t, q̂_hi + c_t]`。命中则 `c_{t+τ} = c_{t+τ−1} + η(err_t − α)`。每格点/每变量/每提前期/每分位各拟合一个 `c_t`。

### 1.1 四问逐一回答（任务要求）

| # | 问题 | 答案 | 证据（原文） |
|---|---|---|---|
| 1 | **是否 tail-conditional / 极端子集条件化？** | **否。** 只提供**边际覆盖**保证（Eq.1 时间平均）；极端覆盖仅作**诊断评估**（truth > 气候学 95 分位），并明确承认**极端条件覆盖无理论保证**。 | "The coverage on extremes is a type of conditional coverage, **and the theoretical guarantee does not hold in this setting** (Foygel Barber et al., 2021)."（§2.3）；"improve, but do not perfect, coverage of extreme temperature; for extreme precipitation, improvement is marginal"（结论） |
| 2 | **是否「未认证即弃权」？** | **否。** 总是输出共形化区间，无 reject option / 拒绝选项；唯一特殊处理是不输出空区间。 | "we cannot decrease coverage without sometimes outputting empty prediction intervals …, **which we decide not to do**"（§3） |
| 3 | **是否用于电价？** | **否。** 2m 温度 + 总降水量，3 个全球天气模型（GenCast/NeuralGCM/AIFS-ENS），ERA5/IMERG 真值。 | §2.3 Models and Data；§3 Results |
| 4 | **是否双尾非对称？** | **否。** 单一 `c_t` **对称**作用于上下两侧；作者明确把「上下分别共形校正」列为未来工作。 | "Tracking **separate upper and lower conformal corrections** (Romano et al., 2019) would likely help, **but is left for future work**"（结论） |

### 1.2 与 BECH 的精确划界

| 维度 | Weather 论文 | BECH |
|---|---|---|
| 修正对象 | **区间**（只改区间宽度，不改点预测值） | **点预测**（ŷ = ŷ_base + λ·δ，同时产出事件概率） |
| 保证类型 | 边际覆盖（时间平均 1−α） | 三段 tail-conditional（正常/负尾/正尾**各认证各的预算**，措辞为「相对基座非降级」，受 Barber 定理约束） |
| 极端子集 | 仅诊断评估，无保证 | **认证目标本身**（负尾/正尾是认证段，不是事后诊断） |
| 双尾 | 对称单 padding；非对称留作 future work | 结构性非对称：负尾主动修正、正尖峰主动弃权 |
| 弃权 | 无 | **未认证即弃权**（恒等返回，弃权纳入共形安全论证） |
| 域 | 天气（温度/降水） | 日前电价（Lago 5 市场 / NEM 5 区 / GEFCom2014-P） |
| 价值 | 校准 + CRPS/SSR 不损失 | 经济价值认证（储能利润/regret/CVaR 为认证目标） |

> **关键**：Weather 论文自己把 BECH 的两根支柱（上下非对称、极端专用共形）明确列为 **future work** —— 这既说明这两点真实存在、未被占据，也是 Related Work 里最强的划界引文。同时它引用 Foygel Barber et al. (2021) 说明极端条件覆盖无分布自由保证，这与 BECH 把主张措辞为「相对基座非降级 / 相对风险控制」完全兼容。

---

## 2. 校正头家族精读（家族 D，5 篇）

### 2.1 CRC — Causality-Inspired Safe Residual Correction（arXiv 2512.22428，2025-12-27）

- **是什么**：多变量时序的即插即用安全残差校正。因果启发编码器（方向门控，解耦自/跨变量动态）+ 混合校正器（Ridge 线性 floor + MLP 非线性 delta）。
- **安全机制**：**四重安全防火墙** = 方向门控 / 分位裁剪 / 逐点选择 / shrink-to-base 混合，保证逐点与验证级不退化，用 NDR（Non-Degradation Rate）度量。
- **与 BECH 的差异**：无共形认证、**无弃权**（论文明确说 Safe Learning 的 abstention 是"passive"，CRC 做的是 active correction）、无 tail-conditional、无双尾非对称、**非电价**（ETTh/ETTm/Weather/Electricity/Traffic 基准，Electricity 是负荷非电价）、无经济认证。其四重防火墙可作为 BECH M4 的逐条对照清单（借鉴关系，非撞车）。

### 2.2 δ-Adapter — The Forecast After the Forecast（arXiv 2601.20280，ICLR 2026）

- **是什么**：冻结基座上的轻量架构无关后处理：输入 nudging（软编辑协变量）+ 输出残差校正。理论：局部下降保证（Thm 2/3）、O(δ) 漂移界（Prop 3.1）、组合稳定性。另作特征选择器 + 分布校准器（Quantile Calibrator + **Conformal Corrector**，学习 scale 函数做归一化残差共形，有限样本**边际**覆盖）。
- **与 BECH 的差异**：共形只用于**区间**校准（不确定性），**不用于点预测修正的认证**；无 tail-conditional、无弃权、无双尾非对称、非电价。其 Prop 2.1（二次型下降）正是 BECH M4 幅度头 δ 的理论锚（借鉴关系）。**注意**：其 Conformal Corrector 是「外挂共形」家族成员，但不能替代 BECH 的三段认证 + 弃权主张。

### 2.3 UEC — Universal Error Corrector with Seasonal-Trend Decomposition（arXiv 2605.21088，2026-05-20）

- **是什么**：面向深度时序自回归误差累积的架构无关校正器。把预测与误差各自分解为趋势/季节分量，分别学校正向量，加权损失平衡。4 基座 × 10 数据集。
- **与 BECH 的差异**：**无任何安全/认证机制**（纯性能提升工具）、无共形、无弃权、无 tail-conditional、无双尾非对称、非电价。分解的是**趋势/季节**（时间尺度），BECH 分解的是**正常/负尾/正尾**（事件类型）——结构逻辑不同，不撞车。

### 2.4 PnP-Corrector — Plug-and-Play Correction for Coupled Spatiotemporal Forecasting（arXiv 2605.08935，ICML 2026）

- **是什么**：耦合时空系统（气候/海气耦合 300 天预报）的即插即用校正框架。冻结预训练物理仿真引擎，只训练校正 agent 对抗系统偏差（Reciprocal Error Amplification），自带 DSLCast 骨干，误差 -28%。
- **与 BECH 的差异**：域是**海气耦合物理仿真**（非电价、非通用时序）、无安全/认证/共形/弃权/尾部条件化。仅共享「冻结基座 + 外挂校正」的粗粒度思想，划界清晰。

### 2.5 Post-Training Corrections（arXiv 2505.15354，2025-05-21）

- **是什么**：boosting 式对冻结预测器输出**顺序施加**一组精选校正。理论起点：**仿射校正**——Theorem A：最优仿射校正风险 R⋆ = R₀ − Cov²(Y,Z)/Var(Z)，即 **R₀ − R⋆ ≥ 0，永不恶化**。校正集含简单校正族 + LLM 翻译的自然语言 HITL 安全校正。
- **与 BECH 的差异**：无共形、无弃权、无 tail-conditional、无双尾非对称、非电价、无经济认证。其 Theorem A 是 BECH「弃权恒等返回」的解析锚点（借鉴关系）。

---

## 3. 对比总表（维度 × 各工作）

| 维度 | Weather 2606.19642 | CRC 2512.22428 | δ-Adapter 2601.20280 | UEC 2605.21088 | PnP 2605.08935 | Post-Training 2505.15354 | **BECH（本项目）** |
|---|---|---|---|---|---|---|---|
| 应用域 | 天气（温度/降水） | 通用多变量 TS | 通用单/多变量 TS | 通用 TS | 耦合时空（海气） | 通用 TS | **日前电价**（9 市场） |
| 外挂冻结基座/模型无关 | ✅（任何预报模型） | ✅ | ✅ | ✅ | ✅（冻结物理引擎） | ✅ | ✅ |
| 校正对象 | **区间**（padding） | 点预测（残差） | 点预测 + 区间 | 点预测（残差） | 点预测（系统偏差） | 点预测（仿射/组合） | **点预测**（ŷ_base+λ·δ） |
| 双尾事件分类 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **双路 P(neg\|Z)/P(spike\|Z)** |
| 双尾非对称处理 | ❌（留 future work） | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **负尾修正/正尖峰弃权** |
| tail-conditional 认证 | ❌（仅诊断；明确无保证） | ❌（逐点/验证级实证） | ❌（边际共形区间） | ❌ | ❌ | ❌ | ✅ **三段各认证各的预算** |
| 未认证即弃权 | ❌ | ❌（明确拒绝 abstention 路线） | ❌ | ❌ | ❌ | ❌ | ✅ **恒等返回 + 弃权入共形论证** |
| 不退化/安全保证 | 边际覆盖定理 | 四重防火墙 + NDR | 局部下降 + O(δ) 漂移界 | ❌ | ❌ | Theorem A（仿射永不恶化） | **SCARR 两层证书（效能 LCB + 共形安全分位）** |
| 共形用法 | 在线边际区间校准 | 未用 | 区间校准（Conformal Corrector） | 未用 | 未用 | 未用 | **安全层单点伤害 ≤ ρ×基线**（点修正认证） |
| 经济/价值认证 | ❌（CRPS/SSR 校准指标） | ❌（MSE/NDR） | ❌（MSE/校准） | ❌（MSE） | ❌（RMSE/ACC） | ❌（MSE） | ✅ **储能利润/regret/CVaR 认证目标** |
| 点预测修正的认证式开关 | ❌ | 防火墙（启发式） | 定理（二次型） | ❌ | ❌ | 定理（仿射） | ✅ **认证通过才修正，否则恒等** |

---

## 4. 关键差异一句话说明（每条）

1. **Weather 只校准区间、不修正点预测** —— BECH 修正点预测值并给事件概率，两者修正对象正交。
2. **Weather 只给边际覆盖保证，极端子集无保证且仅诊断** —— BECH 把负尾/正尾做成认证段本身（条件保证，措辞为相对基座非降级）。
3. **Weather 单一对称 padding，上下分开校正留作 future work** —— BECH 的双尾非对称（负尾修正/正尖峰弃权）正是其 future work 的具体化，且叠加了电价域。
4. **Weather 无任何 reject option** —— BECH 的「未认证即弃权」在六篇中独一份。
5. **CRC 明确拒绝 abstention（称 Safe Learning 被动）** —— 恰好把「主动校正 + 认证式弃权」的交叉点留给了 BECH。
6. **CRC 四重防火墙是启发式安全，无分布自由保证** —— BECH 用共形分位给单点伤害上界，更强且可证。
7. **δ-Adapter 的共形只做区间校准** —— 不认证「点修正动作本身是否安全」，BECH 的 SCARR 安全层认证的是修正动作。
8. **δ-Adapter Prop 2.1 / Post-Training Theorem A 是理论锚** —— 分别支撑 BECH 的 δ 有界下降与弃权恒等返回，属借鉴关系。
9. **UEC 分解趋势/季节，无安全机制** —— 与 BECH 按事件类型三段分解 + 认证的结构完全不同。
10. **PnP 面向海气耦合物理仿真** —— 域隔离最远，只共享「冻结基座 + 外挂校正」粗思想。
11. **Post-Training 无共形/无弃权/无尾部** —— 只保证平均意义不恶化，不保证单点伤害、不覆盖尾部。
12. **六篇全部不做经济价值认证** —— 家族 D 做统计、家族 F 只评估，BECH 的「以经济价值为认证目标」是交集空白。
13. **六篇全部非电价域** —— 日前电价 + 负电价专用机制（P2 族 30 篇亦无先例）是 BECH 的域护城河。
14. **六篇全部不做「认证通过才触发」的开关式点修正** —— BECH 的 λ∈[0,1] 认证触发在家族 D 内无先例。

---

## 5. 结论清单

### 5.1 BECH 仍能安全主张（5 点，按强度排序）

1. **负电价专用、正尖峰主动弃权的结构性非对称** —— 六篇无一是电价域、无一按事件类型路由；P2 族独立验证「日前电价负价专用机制」为空白。
2. **三段 tail-conditional 认证（正常/负尾/正尾各认证各的预算）** —— Weather 明确承认极端条件覆盖无分布自由保证并留作 future work；主张措辞须为「相对基座非降级 / 相对风险控制」（Barber 不可能性定理约束）。
3. **分布自由的「未认证即弃权」零退化** —— 弃权本身纳入共形安全论证；CRC 明确走 active correction 不走 abstention，Weather/δ-Adapter/UEC/PnP/Post-Training 均无弃权。
4. **经济价值认证（储能利润/regret/CVaR）作为认证目标** —— 家族 D×F 交集空白，六篇全做统计指标。
5. **认证式开关的点预测修正 + 可叠加性 + 评估方法学** —— 认证通过才触发（λ 证书门），M8/M9 证明可叠加在同行方法/重训基座之上（0/18 退化）；事件数加权漏判率、增益归因审计为方法学贡献。

### 5.2 必须放弃（3 点）

1. **「首次外挂冻结基座的模型无关在线共形后处理」** —— Weather 2606.19642 完整占据；BECH 定位必须收敛到「尾部条件 + 双尾非对称 + 电价日前 + 经济认证」，绝不宣称该定位。
2. **「边际覆盖在极端子集失效」作为创新贡献** —— Weather 已实证（且有 Foygel Barber et al. 2021 理论支撑），只能作为**动机**（motivation），不能作为贡献（contribution）。
3. **「首次模型无关校正头 / 首次安全零退化残差校正」等「首次」措辞** —— 家族 D 六选一已被 CRC（NDR + 防火墙）、δ-Adapter（局部下降）、Post-Training（仿射永不恶化）占据；BECH 的安全主张必须表述为「认证式 + 弃权式 + tail-conditional」的组合而非「首次安全」。

> **附加禁区提醒**（继承 05 报告）：① 不宣称「精确尾部条件覆盖 ≥ 1−α」（Barber 不可能性）；② 不宣称总体 MAE 冠军（M7 avg +0.54%，诚实声明非精度冠军）；③ Trading Electrons 已做双向极端分别建模（DART spread 上），「双向极端分别建模」动作本身不可当贡献，新颖性界定在「日前电价 + 与校正模块耦合 + 双向非对称」。

---

## 6. Related Work 写作要点（供导师/后续直接使用）

- **与 Weather（2606.19642）划界**：motivation 引其「极端覆盖无保证」实证；contribution 强调 BECH 把极端子集从诊断升级为认证目标，并做了其 future work（上下非对称、极端专用共形）在电价域 + 点修正 + 弃权 + 经济认证的具体化。
- **与家族 D 划界**：一张对比表（§3）可直接进论文；强调「认证式开关」「未认证即弃权」「tail-conditional 三段认证」「经济认证」四个无人占据的交点。
- **理论引用**：Barber (2020) 不可能性 → 措辞依据；Foygel Barber et al. (2021) → 极端条件覆盖无保证；δ-Adapter Prop 2.1 → δ 有界下降；Post-Training Theorem A → 弃权恒等返回；CRC 四重防火墙 → M4 对照清单；Romano et al. (2019) CQR → 双尾分开共形的先例（注意它是区间校准，非点修正认证）。
- **引用清单（arXiv ID）**：2606.19642（Weather）、2512.22428（CRC）、2601.20280（δ-Adapter）、2605.21088（UEC）、2605.08935（PnP-Corrector）、2505.15354（Post-Training Corrections）、Barber 2020（Is distribution-free inference possible?）、Foygel Barber et al. 2021、Romano et al. 2019（CQR）。

---

## 附：任务自检（对照验收标准）

| 验收项 | 状态 |
|---|---|
| 对比表（维度×各工作，含 Weather/CRC/δ-Adapter/UEC/PnP/Post-Training/BECH 共 7 列） | ✅ §3 |
| 每条差异一句话说明 | ✅ §4（14 条） |
| 结论清单：BECH 仍能安全主张 3-5 点 | ✅ §5.1（5 点） |
| 结论清单：必须放弃 1-3 点 | ✅ §5.2（3 点） |
| Weather 四问（tail-conditional？弃权？电价？双尾非对称？） | ✅ §1.1 |
| 原文补方法细节（6 篇全文精读，PDF 在 D:\AI_Memory\papers\raw\） | ✅ |
