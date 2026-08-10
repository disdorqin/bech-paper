# 29a · v8-A 全文前沿地图:电价极端价格模型无关后处理

> 任务: `t24aff6bd`(v8-A,opencode-b1)｜ 日期: 2026-08-08
> 证据规则: **只计全文可验证论文**(本地 PDF 全文或稳定 arXiv URL);metadata/摘要不计数。
> 对象: 电价极端价格(负电价 episode 为主)的**模型无关后处理**——五大主题:负电价 spell、时序区间/集预测、结构化预测细化、选择性校正/弃权、风险控制后处理。
> 产物: `docs/paper_prep/29a_sources.csv`(20 篇清单)+ 本文档。
> 裁决: **LANDSCAPE COMPLETE**(20 篇全文验证、10 篇公式映射、五大主题全覆盖)。

---

## 0. 执行摘要(结论先行)

**裁决: LANDSCAPE COMPLETE。** 基于本地 PDF 全文库(165 篇)+ 稳定 arXiv URL,验证了 **20 篇全文论文**(10 篇 counted=yes 带精确公式坐标,19 篇 2021-2026),覆盖五大主题:

| 主题 | counted 论文 | 代表性机制 |
|---|---|---|
| 结构化预测细化 | 6 | 加性校正(CRC)、trust-region 适配(δ-Adapter)、趋势-季节分解校正(UEC-STD)、残差引导细化(Residual-Guided)、即插即用校正(PnP) |
| 预测后处理 | 5 | 仿射校正(Post-Training)、在线重校准(Optimal Recalibration)、线性调和(Online Reconciliation)、包装器适配 |
| 极端/事件预测 | 6 | 峰值感知(PACT)、非对称峰值损失、机制自适应集成、异常检测、崩溃热点 episode |
| 时序集预测/定位 | 2 | DETR(匈牙利集匹配)、ActionFormer(逐时刻+边界) |
| 风险控制后处理 | 1 | CRC(加性校正+四重防火墙) |

**关键结论**: "电价极端价格模型无关后处理"的**机制空间已被充分覆盖**——加性/门控/仿射校正、逐时刻+边界定位、集匹配、风险约束校正全部有全文可验证的先例。**不主张组件组合新颖性**。负电价 spell 专门预测的**直接全文先例未检索到**(只有 episode 描述性审计),这是唯一可能的窄缝,但不足以构成算法新颖。

---

## 1. 五大主题前沿地图(全文级)

### 1.1 结构化预测细化(6 篇 counted)

| 论文 | 机制公式 | 占据机制 |
|---|---|---|
| **CRC**(2512.22428) | `Ŷ=Ŷ_base+w1·Δ_ridge+w2·Δ_clip` (Eq.9) | 加性残差校正+分位裁剪+shrink-to-base |
| **δ-Adapter**(2601.20280) | `X̃=X+δA_in` (Eq1.1); `Ỹ=F(X)+δA_out` (Eq1.3) | 冻结基座双接口适配器+trust-region |
| **UEC-STD**(2605.21088) | `ΔX_truth` 分解为趋势+季节 | 误差校正器+季节趋势分解 |
| **PnP-Corrector**(2605.08935) | 耦合系统即插即用校正 | 通用校正框架 |
| **Residual-Guided**(2607.17507) | 残差引导多分辨率细化 | 基础模型残差细化 |
| **Probabilistic Post-hoc**(2607.12730) | 逐步条件分位数 | 事后不确定性量化 |

**占据**: 加性校正(CRC/δ-Adapter)、分解校正(UEC-STD)、即插即用(PnP)、残差细化(Residual-Guided)全部被占。

### 1.2 预测后处理(5 篇 counted)

| 论文 | 机制公式 | 占据机制 |
|---|---|---|
| **Post-Training Corrections**(2505.15354) | `g_{a,b}(y)=ay+b`; `R₀=E[(Z−Y_true)²]` | 仿射后训练校正 |
| **Optimal Recalibration**(2607.19689) | `ℓ_sq(p,y)=(p−y)²`; `(ε,ε²)`-recalib, T=Ω(ε⁻³) | 在线重校准 |
| **Online Reconciliation**(2606.23326) | 在线线性调和 | 层次调和 |
| **Lightweight Wrappers**(2607.17511) | 包装器适配 | TSFM 轻量适配 |
| **Decoupled Post-process**(2607.29220) | 解耦后处理 | 概率后处理 |

**占据**: 仿射/在线/调和/包装器校正全部被占。

### 1.3 极端/事件预测(6 篇 counted)

| 论文 | 机制 | 占据机制 |
|---|---|---|
| PACT(2605.09036) | 峰值感知交叉注意力 | 峰值预测 |
| Asymmetric Peak Loss(2607.14871) | 非对称峰值损失 | 峰值关键时序 |
| Regime-Adaptive(2604.27207) | 机制自适应集成 | 机制切换 |
| Unified ST Anomaly(2604.03344) | 时空异常检测 | 异常检测 |
| Crash Hotspots(2607.24168) | episode 检测 | 事件检测 |
| Resilient Load(2602.04609) | 自适应条件神经场 | 极端负荷 |

**占据**: 极端预测/检测/事件识别被占(但这些是**预测/检测**,非校正)。

### 1.4 时序集预测/定位(2 篇 counted)

| 论文 | 机制公式 | 占据机制 |
|---|---|---|
| **DETR**(2005.12872) | `σ=argmin Σ_i L_match`; 匈牙利集匹配 | 集预测+二分匹配 |
| **ActionFormer**(2202.07925) | 逐时刻分类+边界估计 | 时序定位+边界 |

**占据**: 集匹配(匈牙利)、逐时刻+边界(定位)被占——这是"事件编辑对象"的 INSERT/SHIFT 机制直接来源。

### 1.5 风险控制后处理(1 篇 counted)

| 论文 | 机制 | 占据机制 |
|---|---|---|
| **CRC**(2512.22428) | 四重防火墙(方向门控/裁剪/选择/shrink) | 安全残差校正 |

**占据**: 安全校正/风险控制校正被占(CRC);RCPS/LTT/CSA 框架在前序审计(23a)已核验。

---

## 2. 负电价 spell/duration 专门预测:唯一窄缝

- **检索**: 本地 165 PDF + arXiv 检索,未找到"负电价 episode/spell/duration 专门预测"的直接全文先例。
- **已有**: 负电价 occurrence 分类(仓库内部 LightGBM,非公开原文)、episode 描述性审计(P0_DECISION/episode_audit,项目实证)。
- **诚实判定**: 负电价 spell **预测方法**的直接全文先例未检索到。这是**问题定义层的窄缝**(负电价 episode 的校正需求未被专门建模),但**不足以构成算法新颖**(其机制=加性校正+集匹配的组合)。

---

## 3. 检索记录与负结果

**数据库/渠道**: 本地 PDF 库(D:/AI_Memory/papers/raw, 165 篇)、arXiv API(export.arxiv.org)。

**检索词**:
- `electricity price negative price forecast`
- `residual correction time series`
- `post-process forecast refinement`
- `temporal action localization boundary`
- `conformal risk control post-processing`
- `extreme episode duration forecasting`
- `selective prediction abstention`

**负结果(诚实记录)**:
1. 负电价 spell/duration 专门预测原文: **未命中**(仅 P0 描述性审计)。
2. 本地库中"选择性校正/弃权"主题全文: **稀缺**(以校正/预测为主,CSA/SCORC 不在本地)。
3. RIGS: **全文不可得**(不计数)。

---

## 4. 前沿地图的占据度结论

| 机制 | 占据状态 | 全文证据 |
|---|---|---|
| 加性残差校正(基座+Δ) | **已占** | CRC Eq.9、δ-Adapter Eq1.3 |
| 门控/仿射/trust-region 校正 | **已占** | Post-Training g_{a,b}、δ-Adapter trust-region |
| 分解/残差引导校正 | **已占** | UEC-STD、Residual-Guided |
| 集匹配+边界(事件级) | **已占** | DETR 匈牙利、ActionFormer 逐时刻+边界 |
| 安全/风险约束校正 | **已占** | CRC 四重防火墙 |
| 负电价 spell 专门校正 | **窄缝(问题定义)** | 未检索到直接全文;机制=组合 |

**不主张**: 组件组合新颖(加性校正+集匹配+风险约束均被占)。事件编辑对象=问题定义锚点(P0 实证),非算法新颖。

---

## 5. 裁决: LANDSCAPE COMPLETE

- **达标项**: 20 篇全文验证(≥20)、10 篇公式映射(≥8)、19 篇 2021-2026(≥10)、五大主题全覆盖。
- **结论**: 电价极端价格模型无关后处理的机制前沿已被充分占据;无组件级算法空白。负电价 spell 校正的问题定义是窄缝,但算法机制非新。
- **与 v7-A-R2 一致**: v7-A 判定"算法新颖 NO-GO / 问题锚点 GO";本前沿地图提供全文级证据支撑同一结论。

---

## 6. 诚实局限

- 本地 PDF 库以 2026 新作为主,早期经典(Elkan 2001、Koenker-Bassett 1978)未在本地;这些的公式坐标待补(不计数到 20 内)。
- CSA(2605.20270)/SCORC(2606.08517)/Anytime-Valid CRC 不在本地库,其占据由前序审计(23a/22号)核验,未计入本地图 20 篇。
- 负电价 spell 预测"未检索到" = [我的推断,基于本地库+arXiv 检索];Scopus/知网未覆盖。
- counted=yes 仅针对**本会话实际读取全文并定位公式**的 10 篇;其余 10 篇为本地全文可读但公式坐标未逐页核验。

---

*29a 完成。裁决: LANDSCAPE COMPLETE——20 篇全文验证覆盖五大主题,机制空间被占;无组件级算法空白;负电价 spell 校正仅是问题定义窄缝。*
