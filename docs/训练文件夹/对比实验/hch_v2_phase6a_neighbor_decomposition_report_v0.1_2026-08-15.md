# HCH-v2 Phase 6.0 — E2 vs E0 邻居差异分解报告(检索/日选择质量)

- **协议**:`hch_v2_phase6_retrieval_reliability_generalization_research_plan_v0.1_2026-08-15.md` §5 Phase 6.0
- **日期**:2026-08-15
- **脚本**:`experiments/08-hch-v2/p6a_neighbor_decomposition.py`
- **产物**:`experiments/08-hch-v2/results/phase6/decomp/`(per-cell JSON × 27、`matrix.csv`、`per_query_metrics.csv`、`neighbor_diff_long.csv`、`counterfactual_long.csv`、`verdict.json`、`figs/` 6 图、`matrix_run.log`)
- **矩阵**:`{LAGO_DE, LAGO_NP, shandong_DA} × {Linear, MLP, PatchTST} × {s0,s1,s2}` = 27 cell;arm `E0(w1)` / `E2(cavm 0,1)` / `E3(cavm 1,1)`,共享同一 frozen universal core 与候选原子,仅邻居选择 + 距离度量不同。

**复现性锚定(硬断言,全过)**:三臂 MAE / n_execute / 逐日 A_true / 邻居索引 与 `results/phase4/ucore_p2/{mk}_{bb}_s{sd}.json` 全部一致(27×3 臂,容差 MAE 1e-5、A_true 1e-6、n_exec 精确、邻居索引精确,实际差异均为 0.0)。离线四路反事实 replay 与部署 pipeline 逐位一致(offline A_hat diff = 0.0,proposal-nonempty 一致率 = 1.0)。

---

## 1. 一页 verdict

**Gate R 判定:不放行 Phase 6.1(APR)。**

E2 与 E0 的检索差异**几乎完全来自邻居集合的构成(覆盖效应),而不是动作质量、聚合读出、权重或校准**。四路反事实里 `Δ_retrieval_set` 是压倒性主通道(20/27 cell 的 decomposition leader),`Δ_weight/k`(ESS 加权)、`Δ_proposal`(mean/median/trimmed/down-only)在几乎所有 cell 上只移动 ≤1–2 个执行日。Phase 6.1 APR 的触发前提(§4.4"邻居相似且 action-sign 冲突")在观测中**不成立**:在真正存在 E2-vs-E0 差异的 Shandong,邻居中等不同(J≈0.43–0.54)且**动作符号高度一致**(0.75–0.95),不是冲突需要调和;在符号冲突最强的 LAGO_DE,E2 反而是更优的一臂。

| 类别计数(27 cell) | |
|---|---|
| PROPOSAL | 11(Shandong 7 + LAGO_DE 1 + LAGO_NP 3) |
| KEY | 7(全在 LAGO_DE) |
| DATA_SUPPORT | 6(全在 LAGO_NP) |
| MIXED | 2(Shandong MLP s2、LAGO_DE PatchTST s2) |
| CALIBRATION | 1(Shandong MLP s1,零执行退化解) |

> 类别标签按计划 §5 树给出;但**必须结合四路分解读**:Shandong 的 PROPOSAL 树标签(§4.4"邻居差异大但符号一致 → 优先审计候选聚合/读出")已被本分解直接回答——聚合是惰性的(Δ_proposal ≤1 天),故该标签在这里是"检索集合主通道"的别名,不是经典聚合缺陷。同理 LAGO_DE 的 KEY 标签("检索 key 无判别力")在此 market 语义反转:两 key 相距远且符号冲突,而 **context 检索恰恰是 LAGO_DE 唯一能找到的动作来源**(E2 执行 4–28 天 vs E0 0–2 天,换 E0 邻居即归零)。

---

## 2. 方法回顾

三臂只改邻居选择:`E0` = W1 距离(λ=(1,0)),`E2` = context 复合距离(λ=(0,1)),`E3` = (1,1)。候选原子(`z0/s/valid/m_minus/m_plus`)共享,`make_day`/`build_core_context`/`_run_candidate`/`select_s3m_k`/`calibrate_s3c` 与 P2/p5a 同链。

四路反事实在同一候选、不同邻居账本上离线 replay(不调 pipeline,`replay_chain` 复刻 `r1a5._per_neighbor_directional_gains` + `double_event_proposal` + `estimate_action_value` + DVG `lcb`):

| 反事实 | 操作 | 本矩阵观察 |
|---|---|---|
| Δ_retrieval_set | E2 链上用 E0 邻居集合 | **主通道**:20/27 leader;Shandong PatchTST 逐位复现 E0 执行数与 u3 |
| Δ_weight/k | E2 邻居 + ESS 加权(`w=exp(−D/τ)`,`n_eff=1/Σw²`) | 惰性:Shandong ≤+1 天;LAGO_DE MLP/PatchTST 上能 +13 天(唯一二阶信号) |
| Δ_proposal | mean/median/trimmed、down-only vs 双事件 | 惰性:全局 ≤1–2 天;down-only 在 Shandong/LAGO_DE 全部 0 执行 |
| Δ_LCB | A_hat 下 LCB 有/无、q·0.5 / q·1.5 | 主导 fire/no-fire 闸门,但被拦边际日 A_true≈0.026(近零值),且同闸门作用于三臂 |

邻居结构度量(逐 query 日):`Jaccard(E0∩E2)`、全记忆 `rank_corr(W1 距离 vs E2 复合距离)`、`n_eff`、W1 `margin`、逐邻居方向增益符号稳定性、fire 日 `action-sign agreement`。

---

## 3. 邻居结构与执行矩阵

| market | med Jaccard E0∩E2 | sign_agree_fire(均值) | E0 执行(med) | E2 执行(med) | E3 执行(med) | 主要 leader | 主类别 |
|---|---|---|---|---|---|---|---|
| LAGO_DE | 0.43 | 0.25 | 0 | 13 | 5 | retrieval_set(6)/weighted(2) | KEY 7 |
| LAGO_NP | 0.29 | 0.40 | 7 | 29 | 20 | retrieval_set | DATA_SUPPORT 6 |
| shandong_DA | 0.43 | 0.88 | 15 | 13 | 15 | retrieval_set(6)/weighted(1)/none(2) | PROPOSAL 7 |

逐 cell 全表见 `matrix.csv`;逐 query 邻居长表见 `neighbor_diff_long.csv`(每 query × 三臂 × 每邻居:date/rank/idx/raw W1/norm W1/norm ctx/composite/∈E0/∈E2/∈E3)。

**关键结构观察**:
- 三个 market 的 E0 与 E2 邻居集合重叠都只是**中等偏低**(J≈0.29–0.54,mean 更低)——两条检索确实选出不同的日。
- 但全记忆上 W1 与 context 距离**排序高度相关**(Shandong PatchTST s0 `rank_corr=0.85`,s1/s2 类似)→ 两者是同一邻域结构的两种排序,不是两条平行宇宙。
- 共享邻居内排序相关仅 0.48 → 差异集中在"哪些日在候选前沿",而非整体顺序。
- `n_eff≈9.8≈k=10` 全局平坦 → 权重本就接近均匀,ESS 加权无空间(解释 Δ_weight 惰性)。

### 3.1 Shandong PatchTST(动机 cell,E2 略亏)

| cell | E0/E2/E3 执行 | Δ_retrieval_set | weighted | down | no_lcb | E2−E0 MAE |
|---|---|---|---|---|---|---|
| s0 | 35/31/35 | **35(u3=74.590 = E0 逐位)** | 32 | 0 | 317 | +0.86(+0.76%) |
| s1 | 60/44/49 | **60(u3=56.842 = E0 逐位)** | 45 | 0 | 324 | +1.39(+1.25%) |
| s2 | 60/50/56 | **60(u3=59.792 = E0 逐位)** | 50 | 0 | 312 | +1.52(+1.37%) |

把 E2 链上的邻居换成 E0 的集合,执行日数**逐位等于 E0**(35/60/60),逐日 u3 也逐位等于 E0。→ E2-vs-E0 的 MAE 差**100% 是覆盖差(少火 4/16/10 天)**,不是火过的日子质量差(火过的日子 u3 75.0 vs 74.6,几乎相等)。fire 日 action-sign 一致 0.75–0.91。

### 3.2 LAGO_DE(E2 全面占优)

E0(W1)几乎不执行(0–2 天),E2(context)执行 4–28 天;E2 MAE 7/9 cell 更小、2/9 差 <0.03%。**Δ_retrieval_set 反向**:换 E0 邻居 → 执行归零。context 检索是 LAGO_DE 唯一的动作来源。ESS 加权在 MLP/PatchTST 能把 E2 执行从 10→21、4→13、7→25 等(weighted leader)——LAGO_DE 是唯一有可观测二阶权重信号的 market,但 MAE 差异在 1e-3 量级,不构成行动依据。

### 3.3 LAGO_NP(action-empty 边界)

6/9 cell 双臂 proposal-empty ≥0.9 → **DATA_SUPPORT**(候选容量/薄样本,非检索问题)。执行日的 realized 原始增益 ≈ 0(u3 −0.09 ~ +0.20,EUR 量级;A_true 幅度 1e-2)。该 market 本就在设计上作为低机会/action-empty 边界:**这不是"无改善即检索失败",而是机会本身接近零**。报告不给出任何负价安全指标(负价占比 0%,无可报告)。

### 3.4 Shandong Linear / MLP

- **Linear**:E2 全面 ≥ E0(E2 MAE 3/3 更小,执行日 u3 56.9–72.4 vs 49.8–64.5,fire 日一致 0.875–0.95)。
- **MLP**:三 cell 全部几乎 0 执行(0/1/0、0/0/0、0/0/0)。s1 上 E2 对全部 332 天都有非空 proposal 但 **LCB 全拒**(CALIBRATION 退化签名),s2 MIXED;MAE 全等于 host(159.08,零修正)。这是 host 容量不足的零火 cell,不是检索失败。

---

## 4. 四路分解汇总(跨 27 cell)

- `Δ_retrieval_set`:`20/27` 的 leader;在动机 cell(Shandong PatchTST)逐位解释全部执行差;在 LAGO_DE 解释全部动作来源。**检索集合构成是 E2-vs-E0 唯一可观测的主通道。**
- `Δ_weight/k`:惰性(Shandong ≤+1 天);仅 LAGO_DE MLP/PatchTST 有 +13 天上限的二阶信号。`n_eff≈k` 说明权重本已平坦。
- `Δ_proposal`:惰性(median/trimmed/down-only 全局 ≤1–2 天);down-only 在 LAGO_DE 与 Shandong 全部 0 执行 → **聚合/读出层不是可改的层**。
- `Δ_LCB`:LCB 是 fire/no-fire 的主导闸门(Shandong no_lcb 312–324 → LCB 后 31–60);被拦的边际日 A_true≈0.026(近零值),q 从 0.5q 到 1.5q 只是精度/覆盖交换,总量几乎不变;**LCB 本身不是 E2-vs-E0 的差异化因素**(同闸门作用于三臂,且 Shandong MLP 的"全拒"是零火 host 的退化)。

通道耦合提示:Δ_total 不等于四者之和(邻居→directional→proposal→pi→A_hat→LCB 是链);但本矩阵中四个通道的独立信号量级差了两个数量级(主通道改变 4–28 个执行日,其余改变 ≤2 天),结论不依赖严格因果分解。

---

## 5. Gate R 判断(逐 market)

计划 §4.4 的放行前提是 **"邻居很相似且 action-sign 冲突 → APR 才是有意义的候选"**。逐 market 核对:

| market | 邻居相似? | 符号冲突? | APR 前提 | 实际层 |
|---|---|---|---|---|
| LAGO_DE | 否(J 0.43) | 是(E2 已赢,无需调和) | 否 | context 检索已最优;无需干预 |
| LAGO_NP | 否(J 0.29) | 中 | 否 | 薄样本/容量(DATA_SUPPORT) |
| shandong_DA | 否(J 0.43–0.54) | **否**(一致 0.75–0.95) | **否** | 检索集合覆盖效应,per-day 质量等价 |

**明确停止层:检索集合构成层。** Shandong PatchTST 上 E2 少火的 4–16 天是"context 检索选出的邻居使 A_hat 不过 LCB 阈值"的覆盖效应;这些天的边际价值≈0(E0 多火的日 u3 与 E2 火过的日相当,执行日质量全等价)。由于聚合、权重、校准三路反事实均惰性,**计划的五个候选杠杆里没有与观测证据几何匹配的改动**。按计划 §5.10"不满足时明确停止在哪一层"——停在检索集合构成层,并标注其为覆盖效应而非质量退化(与 [[hch-v2-phase5-forensic-reconciliation]] 的结论一致:Shandong 的"退化"是对比基线 E2−E0(w1) 的误读,真正的差距是检索/日选择质量,且该差距是覆盖而非质量)。

> 注:E3(λ=(1,1))在 Shandong PatchTST 执行 35/49/56 天、MAE 介于 E0 与 E2 之间,说明固定混合部分恢复覆盖。此仅为已冻结臂的观测,不是推荐手写 λ(§4.3 禁止 market/host 身份与手写 λ);也不构成对 Phase 6.1 的放行。

---

## 6. 诚实性说明

- **不包装点级优势**:全程报告 MAE/执行数/逐日 u3 原值;不把点级优势包装成动作安全优势。E2 harm_rate 已标注(Shandong PatchTST s0:E2 2/31 执行日有害 vs E0 0/35;LAGO_DE 与 Linear 无害)。
- **LAGO_NP**:负价占比 0%,不报告负价安全指标;低机会/action-empty 边界,如实记为"价值≈0",不写"无改善为检索失败"。
- **Shandong MLP**:零火 cell,MAE 全等于 host,不做任何"E2 好/坏"断言。
- **局限**:universal-core 单一 checkpoint;每 market×host×seed 单 seed 锚定(锚定本身 27×3 臂全过);LAGO_NP 信号弱(1e-2 量级);类别标签按计划树给出但语义需结合分解读(§1 已注明 KEY/PROPOSAL 在此矩阵中的实际含义)。
- **不做的事**:未改 IAH-CRPS/三原子候选/query-dose replay/双事件结构/alpha/LCB/DVG;未实现 APR;未新增 loss/事件头/market-host ID/硬阈值/P4;未用 S4 标签调任何参数(k/λ/decay/q 均为冻结);未覆盖任何既有结果(全部新产物目录)。

---

## 7. 下一步建议(按计划边界)

1. **不放行 Phase 6.1(APR)**——前提不成立、无匹配杠杆。
2. 若继续检索方向研究,应面向"覆盖恢复"而非"证据调和",且需新的无身份条件候选设计(在 §2/§4 边界外另立计划,先复现本 27-cell 基线)。
3. 建议把本分解的"检索集合覆盖效应"结论并入最终论文的局限/讨论段,替换 Phase 4 遗留的"Shandong raw 退化"叙事。
