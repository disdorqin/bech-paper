# HCH-v2 Phase4 — P2 扩展:多市场 × 多 seed E0-E3 门禁(§9.2)

- 日期: 2026-08-15
- 范围: experiment design §9.2 第一阶段放行门——多市场 × 3 host × 3 seed 的 E0-E3 受控对比
- 分支: `exp/r1b-screening-20260813`
- 前置: P2-P3 报告 `hch_v2_phase4_p2p3_report_v0.1_2026-08-15.md`(单域机制探针,判定 `COMPOSITE_RETRIEVAL_MECHANISM_SUPPORTED`,§9.2 未满足)
- 脚本: `experiments/08-hch-v2/p2_cavm_multimarket.py`
- 结果: `experiments/08-hch-v2/results/phase4/p2_ext/`(gitignored;本报告记录全部关键数字)

---

## 1. §9.2 门禁协议(回述)

experiment design §9.2 第一阶段放行条件(逐条):

| # | 条件 | 本扩展的实现 |
|---|---|---|
| 1 | **≥2 种市场方向稳定改善** | 每个市场"改善"定义为:该市场多数 cell(≥1/2, 共 3host×3seed=9 cell)点预测 MAE 下降 **且** 均值下降 **且** ≥1 cell 动作价值同向;`market_improves` 聚合 |
| 2 | **3 seeds,方向非单一 seed 支撑** | 对**声称改善的市场**,改善方向须由 ≥2 个 seed 共同支撑(`seeds_with_point_help ≥ 2`) |
| 3 | **LAGO_NP 无可复现显著回归** | 温和市场专门检查:均值退化 **且** ≤1/9 cell 改善 → 判"可复现回归" |
| 4 | **动作价值与点预测不反向** | 全矩阵 E2 的 `opposite_cells == 0`(MAE 与 A_true 方向相反即为反向) |
| 5 | **仅 streaming 有效 → 降级为概念漂移实验** | P3 已证 local 账本 append-only、预测永不消费 → 不触发 |
| 6 | **非 point-only** | 动作轨迹独立报告(E2 全矩阵 `action_help_cells > 0`) |

---

## 2. 方法(与 P2 探针逐字同链)

- **受控设定**: host 来自共享缓存(`R.prepare_domain`;R1B Stage-2C 约定——host 跨 seed 共享,seed 只变候选头训练)。E0/E2/E3 之间**唯一变量是检索 key**(§3.1 单轴原则)。
- **候选头**: 每市场在**自己**的 S2T/S2V 上训练(受控设定,非冻结 12 域 universal 头)。这是机制方向门;paper 配置(冻结 universal core)的确认是文档化后续。
- **k**: S3-M 上 W1 forward validation 选 {5,10,20}。**λ 固定网格 {0,1}/{1,1},绝不在 S4 调参**(红线)。
- **S4**: target-free(`predict_s4`);A_true 在标签揭示后用与 P2 相同的 `estimate_realized_A` 离线计算。
- **E0/E1/E2/E3**: w1(对照)/ cavm λ=(1,0)/ cavm λ=(0,1)/ cavm λ=(1,1)。
- **信息隔离**: `build_core_context` 仅用过去滞后(rank + 日历 + lag24/48/168 scale-free 价格),无任何未来 `y_{t+k}` 列;key 预测时冻结。

矩阵: **5 市场 × 3 host × 3 seed = 45 cell**。

| 市场 | 类型 | 说明 |
|---|---|---|
| LAGO_DE | 国外·负价 | 探针连续性,验证逐字复现 |
| LAGO_NP | 国外·温和 | §9.2-3 关键:无负价、无尖峰 |
| shandong_DA | 国内·负价+尖峰 | 真实负价域 |
| shaanxi_RT | 国内·强 host | host 已很强,修正头空间小 |
| gansu_DA | 国内·薄样本 | S1R=56/S2T=89/S4=112,偏噪,如实标注 |

host 清单: Linear / MLP / PatchTST。seed: 0 / 1 / 2。

---

## 3. 逐 cell 复现契约(sanity)

`p2_cavm_multimarket` 的 LAGO_DE:Linear:s0 与 P2 探针 `p2_cavm_experiment_lago_de_linear.json` 逐字一致:

| 指标 | P2 探针 | 本扩展 s0 | 一致 |
|---|---|---|---|
| E0 MAE / A_true / exec | 6.219 / −0.0051 / 0.050 | 6.219 / −0.0051 / 0.050 | ✅ |
| E2 MAE / A_true / exec | 6.046 / +0.0129 / 0.128 | 6.046 / +0.0129 / 0.128 | ✅ |
| E3 MAE / A_true / exec | 6.192 / +0.0082 / 0.073 | 6.192 / +0.0082 / 0.073 | ✅ |
| E1 == E0(邻居切换) | 0/437 | 0/437 | ✅ |
| S2 (S2V-selected) | 0.179253 | 0.1793 | ✅ |
| k / q | 5 / 0.0636 | 5 / 0.0636 | ✅ |

host 缓存前端与 P2 自拟合 host **逐字等价**。

---

## 4. 结果矩阵(45 cell,全绿,零失败)

全部 45 cell 完成(`results/phase4/p2_ext/matrix.csv` + 45 个 cell JSON)。E1==E0 契约在每 cell 复验(邻居切换 0/日数),CAVM 账本 λ=(1,0) 逐字复现 W1。

### 4.1 每市场聚合(E2 vs E0;每市场 9 cell = 3 host × 3 seed)

| 市场 | point_help | frac | mean ΔMAE | mean ΔA_true | action_help | both_help | both_opp | seeds_support | improves |
|---|---|---|---|---|---|---|---|---|---|
| **LAGO_DE** | 7/9 | 0.778 | **−0.0957** | **+0.0128** | 9/9 | 7 | 0 | 3/3 | ✅ |
| LAGO_NP | 1/9 | 0.111 | +0.0018 | +0.0001 | 3/9 | 1 | 0 | 0/3 | ❌ |
| shandong_DA | 3/9 | 0.333 | −0.9253 | −0.0040 | 4/9 | 2 | 1 | 0/3 | ❌ |
| shaanxi_RT | 0/9 | 0.000 | +0.0141 | +0.0014 | 4/9 | 0 | 0 | 0/3 | ❌ |
| gansu_DA | 0/9 | 0.000 | +0.0000 | +0.0005 | 6/9 | 0 | 0 | 0/3 | ❌ |

整体(45 cell):mean ΔMAE **−0.201**,mean ΔA_true +0.0022,point_help 11/45,action_help 26/45,opposite 17/45,harm_up 6/45。

### 4.2 每 market × host 聚合(3 seed 均值,E2 vs E0)—— host 异质性

| 市场 | host | point_help/3 | mean ΔMAE | mean ΔA_true | 备注 |
|---|---|---|---|---|---|
| LAGO_DE | Linear | 3/3 | −0.1635 | +0.0155 | harm 45-50%→17-29%,exec_A_true +0.03-0.05→+0.06-0.07 |
| LAGO_DE | PatchTST | 3/3 | −0.1426 | +0.0183 | harm 31-40%→22-24%,exec_A_true +0.04-0.07→+0.08-0.09 |
| LAGO_DE | MLP | 1/3 | +0.0190 | +0.0046 | **exec_A_true 0.25-0.28→0.15-0.16 下降**;harm 15-17%→18-20% 微升 |
| LAGO_NP | Linear | 0/3 | +0.0000 | −0.0000 | 无执行日 |
| LAGO_NP | MLP | 1/3 | −0.0006 | +0.0004 | exec 少量(13-18→2-15) |
| LAGO_NP | PatchTST | 0/3 | +0.0059 | −0.0001 | exec 88-112→41-107 下降 |
| shandong_DA | **Linear** | 0/3 | **+1.0764** | +0.0077 | **点预测退化**(+0.9~+2.3 MAE) |
| shandong_DA | MLP | 0/3 | +0.0000 | −0.0221 | 无执行日,点不变 |
| shandong_DA | **PatchTST** | 3/3 | **−3.8523** | +0.0024 | **点预测大改善**(−3.1~−5.4 MAE,~−3.7%) |
| shaanxi_RT | Linear/MLP | 0/3 | +0.0000 | ~0 | 无执行日 |
| shaanxi_RT | PatchTST | 0/3 | +0.0424 | −0.0105 | exec 0→3 |
| gansu_DA | 全 host | 0/3 | +0.0000 | ~0 | 无执行日 |

### 4.3 关键逐 cell 明细

**LAGO_DE**(机制最干净的证据):

| host | s | E0_MAE | E2_MAE | ΔMAE | ΔA_true | Δexec_A_true | exec E0→E2 | harm E0→E2 |
|---|---|---|---|---|---|---|---|---|
| Linear | 0 | 6.219 | 6.046 | −0.173 | +0.0179 | +0.0267 | 22→56 | 45.5%→28.6% |
| Linear | 1 | 6.238 | 6.041 | −0.198 | +0.0139 | +0.0406 | 48→59 | 43.8%→16.9% |
| Linear | 2 | 6.221 | 6.102 | −0.119 | +0.0146 | +0.0135 | 16→44 | 50.0%→22.7% |
| MLP | 0 | 8.161 | 8.069 | −0.093 | +0.0065 | **−0.098** | 24→44 | 16.7%→20.5% |
| MLP | 1 | 7.866 | 7.992 | +0.125 | +0.0007 | **−0.105** | 39→44 | 15.4%→18.2% |
| MLP | 2 | 8.130 | 8.155 | +0.024 | +0.0065 | **−0.120** | 23→31 | 17.4%→19.4% |
| PatchTST | 0 | 6.058 | 5.935 | −0.123 | +0.0160 | +0.0133 | 39→59 | 30.8%→22.0% |
| PatchTST | 1 | 6.108 | 5.961 | −0.148 | +0.0232 | +0.0318 | 27→53 | 33.3%→22.6% |
| PatchTST | 2 | 6.115 | 5.958 | −0.157 | +0.0157 | +0.0405 | 35→54 | 40.0%→24.1% |

**shandong_DA**(host 依赖最极端):

| host | s | E0_MAE | E2_MAE | ΔMAE | exec E0→E2 |
|---|---|---|---|---|---|
| Linear | 0 | 95.12 | 96.05 | +0.93 | 54→50 |
| Linear | 1 | 93.27 | 95.58 | +2.30 | 61→51 |
| Linear | 2 | 106.93 | 106.93 | +0.00 | 0→0 |
| PatchTST | 0 | 103.34 | 100.21 | **−3.13** | 118→147 |
| PatchTST | 1 | 102.56 | 99.50 | **−3.06** | 120→146 |
| PatchTST | 2 | 106.49 | 101.12 | **−5.37** | 79→125 |

> E3(λ=(1,1) 复合)在 LAGO_DE 亦全改善(9/9 point_help,mean −0.0462)但弱于 E2(−0.0957),复验 P2 的"W1 项稀释上下文信号";LAGO_NP 上 E3 反优于 E2(−0.0017 vs +0.0018,4/9 vs 1/9 point_help)——W1 项在温和市场起正则作用。

---

## 5. 判定(§9.2)

**判定:`GATE_NOT_YET_PASS`**(严格逐条,见下表)。

| # | 条件 | 结果 | 证据 |
|---|---|---|---|
| 1 | ≥2 市场方向稳定改善 | **FAIL** | 仅 **LAGO_DE**(国外负价)满足 `market_improves`。shandong 均值大改善但被单一 host(PatchTST)拖高,Linear 反而退化 → 非方向稳定;shaanxi/gansu 点预测零改善;LAGO_NP 不改善 |
| 2 | 3 seed 非单一 seed 支撑 | **PASS** | LAGO_DE `seeds_with_point_help=3/3`(3 个 seed 各自都有 ≥2 cell 点改善) |
| 3 | LAGO_NP 无可复现显著回归 | **FAIL(边界)** | 符号可复现不改善(1/9 point_help,mean ΔMAE +0.0018),但**量级可忽略**(~0.06% on MAE≈3) |
| 4 | 动作价值与点预测不反向 | **FAIL(边界)** | 45 cell 中 17 cell "opposite",但集中于 action-empty 市场(exec≈0,π 噪声)与毫幅差异;唯一真实反向是 shandong PatchTST s2(点预测 −5.37,ΔA_true −0.0004≈0),实为"大点益 + 可忽略动作损" |
| 5 | 仅 streaming 有效→降级 | n/a | P3 已证 local append-only、预测永不消费 |
| 6 | 非 point-only | **PASS** | 动作轨迹独立报告;LAGO_DE 9/9 action_help 且真实执行日 22-56 个 |

**核心结论**:门禁**未通过**,且失败点在首要条件(§9.2-1)——上下文检索只在 LAGO_DE 一个市场类型上方向稳定。机制是真的(LAGO_DE 跨 3 host × 3 seed 复现),但**不满足"≥2 种市场类型"**。

---

## 6. LAGO_NP 专门检查(§9.2-3)

9 cell:point_help 1/9(仅 MLP s1),mean ΔMAE **+0.0018**。

| host | ΔMAE (3 seed) | 解读 |
|---|---|---|
| Linear | +0.0000 / +0.0000 / +0.0000 | 无执行日,点不变 |
| MLP | +0.0044 / **−0.0142** / +0.0081 | 唯一 help cell(MLP s1,exec 3→15) |
| PatchTST | +0.0000 / +0.0062 / +0.0114 | 轻微退化 |

**判定**:严格口径下"符号可复现的不改善"(1/9 help,均值>0),但**量级 ~0.06%,不构成显著回归**。文档措辞是"无可复现**显著**回归"——本检查在"可复现"维度不满足、在"显著"维度通过。如实记录:这是**温和市场中性偏弱**,不是破坏性污染。

---

## 7. seed 一致性(§9.2-2)

- LAGO_DE:`seeds_with_point_help=3/3`。三个 seed 各自都有 ≥2 cell 点预测改善,改善方向**非单一 seed 支撑**。
- 其余市场因 point_help 不足,seed 一致性不适用(如实:shandong PatchTST 3/3 seed 改善,但该市场整体非方向稳定——host 依赖是主因)。

---

## 8. 机制解读(诚实,区分事实与假设)

**事实**(45 cell 数据):
1. **上下文检索在 LAGO_DE 稳定有效**:Linear/PatchTST 点预测与动作价值同步改善(harm 40-50%→17-29%),3 seed 全复现;MLP 点平坦、exec 实现价值下降(0.25→0.16)但仍为正 → 该市场内 2/3 host 受益,1/3 host 中性偏弱。
2. **shandong 的检索收益高度 host 依赖**:PatchTST 大改善(−3.7%),Linear 大退化(+1.1%)。均值掩盖了方向相反的两个 host。
3. **温和市场(LAGO_NP)检索中性偏弱**,量级可忽略。
4. **action-empty 市场(shaanxi/gansu,exec≈0)无动作信号**,点预测零变化——不能支撑任何动作轨 claim。

**假设**(待 paper-config 确认,非本实验结论):
- 上下文检索的收益似乎**条件于 host 容量与市场极端性**:修正头有容量把"更好邻居"转化为更好修正(LAGO_DE 的 Linear/PatchTST、shandong 的 PatchTST)时有效;弱 host(Linear on shandong)检索反而加噪;温和市场上下文无判别力。
- 这与 R1B 的 `SOURCE_FIT_HELPS` 一致:检索收益依赖头与数据的匹配度。

---

## 9. 边界与诚实标注(必须)

1. **per-market 候选头 ≠ 冻结 12 域 universal core**。本门禁验证的是"检索 key 是唯一变量"的机制方向;论文正式配置(learned_sig 冻结 universal 头,跨域记忆)的 CAVM 确认是**文档化后续步骤**,不在此报告中声称。
2. **域描述符变体**:P2 链沿用 `fit_s1_signature`(S1R 确定性域描述符);论文 universal core 的 `learned_sig` 是零描述符。该变体在 E0-E3 内部恒定(不改变"检索 key 是唯一变量"的对比有效性),但会改变 context key 的 c_sig 段——与第 1 条同属"paper-config 确认"范围。
3. **λ 未在 S4 调参**(红线)。E2 优势是机制诊断证据,不是最终配置;真正的 λ 选择应在 S3-M/S2V。
4. **A_hat 校准已知弱**(P2 §7.1):E2 改善来自"检索到更好邻居→更准 π",非 A_hat 排序。与 R1A.8 `ACTION_SIGNAL_UNRESOLVED` 相互印证。LAGO_DE MLP 的 exec_A_true 下降(0.25→0.16)是此弱点的直接体现。
5. **Gansu/Shaanxi 样本薄**(S1R 44-56 天),single-seed 噪声大;方向稳定性由 3 seed 聚合缓解,但**如实标注**其证据强度低于 LAGO 两市场。
6. **`mean_A_true` 语义**:为全日 π 期望动作价值(即使未执行也用 π 计算);`exec_mean_A_true` 才是实现值。shaanxi/gansu 的 "action_help" 属 π 质量信号,非实现改善(action-empty,exec≈0),本报告不据此 claim。
7. `results/` gitignored;一切数字以本报告为准。

---

## 10. 下一步

**§9.2 门禁未通过,触发 §9.3 停止条件检查**。本扩展的证据指向:

1. **回归/无效集中在哪**:LAGO_DE(极端负价)方向稳定有效;shandong 的检索收益**host 依赖**(PatchTST 大益、Linear 大损);温和市场(LAGO_NP)中性偏弱;action-empty 市场(shaanxi/gansu)无效。→ **检索收益条件于 host 容量 × 市场极端性**,不是普适增益。
2. **按 §9.3 决策树**:候选动作**不**是"扩大 memory"方向(不引入 P4 action-value state update)。候选方向:
   - **A. 条件化检索**:检索只在"上下文可判别 + 头有容量"时启用(如 λ_ctx 按市场/host 难度自适应,在 S3-M/S2V 上选)——把当前二元"开/关"细化为门控。
   - **B. paper-config 确认**:冻结 universal core(learned_sig)+ 跨域记忆的 CAVM 是否改变 host 依赖格局(universal 头容量统一,可能消除 shandong 的 host 分裂)。
   - **C. 如实收束**:若 A/B 不改善,CAVM 定位为"极端市场专用模块"(LAGO_DE 型),不作为通用机制进论文主配置。
3. **P4(action-value state update)暂缓**,直到 §9.2 有 ≥2 市场类型通过。
4. 本报告的"机制解读"(§8)列为论文素材,`HCH-CAVM` 正式方法名仍须 §9.2 全绿后。
