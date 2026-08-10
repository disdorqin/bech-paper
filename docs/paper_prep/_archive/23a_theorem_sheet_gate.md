# 23a · Theorem Sheet Gate：BECH 序贯风险审计

> 执行：opencode-b2 | 任务单 t729eada3 | 2026-08-07
> 目标：独立完成一页 theorem sheet 审计，若找不到非平凡且未被占的理论对象，明确 NO-GO

---

## 一、审计结论：NO-GO

**B1' 定理当前不成立，且修复后也只能定位为 CSA/序贯风险控制的领域化例化，不能声称新理论框架。**

22 号审稿（22_BECH_v5_round3_strict_peer_review_2026-08-07_v1.md）已逐项否决 B1'：
1. 所给乘积因子 E_t = ∏(1+λX_t/M) 在一般情形下是均值大于 1 的增长过程，不是零假设下的非负超鞅
2. Ville 不等式不能按文中方式使用（缺少三步：指定零假设、证明超鞅、证明包含关系）
3. "发现风险"不等于"保证预算永不越界"——e-process 是监控器，不是 barrier certificate
4. 组定义和动作定义混在一起（1{A_s=g} 中 A_s 是动作，g 是事件组）
5. i.i.d. 与自适应动作不兼容
6. β-mixing 声明没有证明（缺少 block/coupling 定理、修正后超鞅、显式常数）

**我的独立复核确认**：审稿人的否决完全正确。B1' 的核心错误是将非负 harm 直接乘成 e-process——正确的 betting/e-process 必须围绕有正有负、并在零假设下具有正确条件均值方向的中心化增量构造。

---

## 二、Theorem Sheet 逐项审计

### 2.1 Filtration（滤波）

**要求**：明确 ℱ_t = σ(H_1, ..., H_t, A_1, ..., A_t)（历史 harm + 历史动作）

**B1' 状态**：❌ 未定义
**CSA 状态**：✅ CSA 明确定义 filtration 为 RLVR filtration（历史观测 + 历史动作 + 历史奖励）
**差距**：B1' 缺少 filtration 定义，无法判断 E_t 是否适应 ℱ_t

### 2.2 可预测动作（Predictable Action）

**要求**：A_t 必须是 ℱ_{t-1}-可预测的（动作在观测 H_t 之前决定）

**B1' 状态**：❌ 未明确声明；且 E_t 的构造中 X_t = max(H_t - δ_0, 0) 依赖当期 H_t，不满足可预测性
**CSA 状态**：✅ CSA 明确要求 "predictable updates"——动作基于历史信息决定
**差距**：B1' 的 X_t 依赖当期 H_t，不满足可预测性

### 2.3 零假设（Null Hypothesis）

**要求**：明确零假设 H_0，使得 E_t 在 H_0 下是非负超鞅

**B1' 状态**：❌ 未明确指定；14b 写 "composite null P(X ≤ 0)"，但 X = max(H - δ_0, 0) ≥ 0，所以 P(X ≤ 0) = P(X = 0) 是退化零假设
**CSA 状态**：✅ CSA 明确零假设为 "per-threshold selective risk ≤ α"
**差距**：B1' 的零假设退化，不构成有效的统计检验

### 2.4 中心化增量（Centered Increment）

**要求**：增量 Z_t = I_t · (X_t - μ_0) 必须在零假设下 E[Z_t | ℱ_{t-1}] ≤ 0

**B1' 状态**：❌ X_t = max(H_t - δ_0, 0) ≥ 0，无中心化；E[1 + λX_t/M | ℱ_{t-1}] ≥ 1，超鞅方向错误
**CSA 状态**：✅ CSA 使用 centered e-values：V_t = (dP_t/dQ_t) - 1，满足 E[V_t | ℱ_{t-1}] = 0 under null
**差距**：B1' 缺少中心化步骤，超鞅方向错误

### 2.5 Betting Factor 非负性（Non-negativity）

**要求**：betting factor (1 + λZ_t) 必须非负

**B1' 状态**：✅ 满足（因为 X_t ≥ 0，λ > 0，所以 1 + λX_t/M > 0）
**CSA 状态**：✅ CSA 的 e-values 也是非负的
**差距**：无——但非负性只是必要条件，不是充分条件

### 2.6 条件超鞅（Conditional Supermartingale）

**要求**：E_t 在零假设下是 ℱ_t-适应的非负超鞅

**B1' 状态**：❌ E[1 + λX_t/M | ℱ_{t-1}] = 1 + λE[X_t | ℱ_{t-1}]/M ≥ 1（因为 X_t ≥ 0），所以 E_t 是 submartingale（向上增长），不是 supermartingale
**CSA 状态**：✅ CSA 的 e-process 满足 supermartingale 性质（在零假设下 E[E_t | ℱ_{t-1}] ≤ E_{t-1}）
**差距**：B1' 的超鞅方向完全错误——这是 22 号审稿的核心否决理由

### 2.7 Ville Crossing 与目标风险事件的包含关系

**要求**：证明 "预算越界事件" ⊆ "e-process 越过 1/α 事件"

**B1' 状态**：❌ 未证明；22 号审稿指出 "e-process 越过 1/α" 和 "累积 harm 超预算" 是两个不同事件，需要包含关系证明
**CSA 状态**：✅ CSA 证明了 anytime-pathwise selective-risk bound：R_T^act ≤ α + O(N_T^{-1/2})
**差距**：B1' 缺少包含关系证明

### 2.8 延迟标签（Delayed Labels）

**要求**：处理 y_t 延迟到达的情况（电价次日结算）

**B1' 状态**：❌ 未处理
**CSA 状态**：✅ CSA 处理了 RLVR 的可验证奖励（类似延迟标签）
**差距**：B1' 缺少延迟标签处理

### 2.9 组联合控制（Group-wise Joint Control）

**要求**：多组（normal-acted/negative/positive）的联合风险控制

**B1' 状态**：❌ 12/13 意识到 e-BH 控 FDR 而非 FWER，但未修复；14b 的 "e-BH adjusted p_g" 仍然控 FDR 而非 FWER
**CSA 状态**：✅ CSA 用 Bonferroni 网格做 FWER 控制（保守但正确）
**差距**：B1' 的多重性控制工具选择错误（e-BH → FDR，不是 FWER）

### 2.10 与 CSA 的逐项差异

| 审计项 | CSA (2026) | B1' (2026) | 差异 |
|--------|-----------|-----------|------|
| Filtration | ✅ RLVR filtration | ❌ 未定义 | CSA 更明确 |
| 可预测动作 | ✅ Predictable updates | ❌ X_t 依赖当期 H_t | CSA 满足，B1' 不满足 |
| 零假设 | ✅ Per-threshold selective risk | ❌ 退化零假设 | CSA 有效，B1' 无效 |
| 中心化增量 | ✅ Centered e-values | ❌ 无中心化 | CSA 满足，B1' 不满足 |
| 超鞅性质 | ✅ Supermartingale | ❌ Submartingale | 方向完全相反 |
| Ville 包含关系 | ✅ Proven | ❌ 未证明 | CSA 有，B1' 无 |
| 延迟标签 | ✅ 处理 | ❌ 未处理 | CSA 有，B1' 无 |
| 组联合控制 | ✅ Bonferroni FWER | ❌ e-BH FDR | CSA 保守正确，B1' 工具错误 |
| anytime-valid | ✅ Pathwise | ❌ 未证明 | CSA 有，B1' 无 |

---

## 三、修复路线评估

### 路线 B-A：统计风险监控

**定义**：中心化、组门控的有界增量 Z_{t,g} = I_{t,g}(h_t - r_g)，h_t ∈ [0, M_g]

**可行性**：✅ 可构造有效 e-process（但与 CSA 高度重叠）

**差异**：Z_{t,g} 的 "base-relative harm" 定义是电价领域的特化，但统计对象本身 = CSA 的 selective risk

**结论**：可作为 CSA 的领域化应用，不能声称新理论

### 路线 B-B：硬预算安全控制

**定义**：动作前可行性约束 h_t(A_t) ≤ B_t - Σ_{s<t} h_s

**可行性**：⚠️ 需要可信的动作前上界 h̄_t（目前无）

**差异**：更接近 safe OCO/barrier control，但需要新证明

**结论**：NOT READY（缺少动作前上界）

---

## 四、最终裁决

| 路线 | 可证明？ | 非平凡？ | 裁决 | 理由 |
|------|----------|----------|------|------|
| **B1' 原版** | ❌ 定理错误 | — | **撤回** | 超鞅方向错误、零假设退化、缺包含关系 |
| **路线 B-A（统计监控）** | ✅ 可构造 | ⚠️ CSA 领域化 | **NO-GO** | 与 CSA 高度重叠，不能声称新理论 |
| **路线 B-B（硬预算）** | ⚠️ 缺动作前上界 | 可能非平凡 | **NOT READY** | 需 h̄_t 可信上界，当前无 |

**总体结论：NO-GO。** 找不到非平凡且未被占的理论对象。建议转向路线 E（严格实证）。

---

## 五、给 codex 的建议

1. **删除 B1' 的所有理论主张**（§7 验收升级、§8 路线 B）
2. **C1 降级为 RuleProjection 基线**（不列贡献）
3. **转向路线 E**：严格实证的领域方法论文（KDD/WSDM 应用轨）
4. **核心贡献重新定位**：电价双尾 occurrence-magnitude 问题定义 + 多基座跨市场选择性后校正 benchmark

---

## 参考文献

1. CSA: Khosravi & Huo, "Conformal Selective Acting", arXiv 2605.20270, 2026
2. SCORC: Yu & Liu, "Joint Finite-Sample Certificate", arXiv 2606.08517, 2026
3. Anytime-Valid CRC: Hultberg et al., arXiv 2602.04364, 2026
4. ARC-STAR: arXiv 2605.22222, 2026
5. Waudby-Smith & Ramdas, "Estimating means by betting", JRSSB 2024, arXiv 2010.09686
6. Admissible anytime-valid: Ramdas et al., arXiv 2009.03167, 2022
7. e-Holm: Hartog & Lei, arXiv 2501.09015, 2025
8. Safe OCO: arXiv 2412.03983, 2024
