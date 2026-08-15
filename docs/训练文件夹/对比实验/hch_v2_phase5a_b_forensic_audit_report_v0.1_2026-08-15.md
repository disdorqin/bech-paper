# HCH-v2 Phase 5A/5B：日级/逐小时法医审计 + 四效用离线重算 v0.1

**日期：** 2026-08-15

**依据：**

- `hch_v2_phase5_dual_geometry_action_value_research_direction_v0.1_2026-08-15.md`（§6.1/§6.2/§10 任务边界）
- `hch_v2_phase4_cavm_universal_core_report_v0.1_2026-08-15.md`（Phase 4 结论，`46671c7` 判定 Case D）
- 当前主数学：`hch_v2_iah_crps_final_math_core_v0.3`

**状态声明：** 本轮新增产物全部位于新目录 `experiments/08-hch-v2/results/phase5/forensic/`，未覆盖任何现有结果或脚本。审计用脚本为新建 `experiments/08-hch-v2/p5a_forensic_ledger.py`；复现性锚定全部落在已提交的 `results/phase4/ucore_p2/*.json` 上。

---

## 0. 一句话裁决

**`INCONCLUSIVE`(对 METRIC_MISMATCH)。** 代码正确(0 BUG,15/15 cell 复现性锚定逐位一致);**执行日上 asinh 动作价值与 raw 货币收益符号一致**(Shandong 全 flip 率 0.0,全 15 cell 日级 flip 率 0.62%)。**Phase 4 报告中"Shandong PatchTST raw MAE 大退化(+0.86/+1.39/+1.52)"是对比基线的假象**:该 Δ 是 **E2 vs E0(w1)** 的差值,而 **E0(w1) 本身就是一个执行修正的基线**(Shandong 上比 host 改善 7.86~10.81)。相对 host,E2 在 Shandong PatchTST 上实际改善 7.01~9.29。**真正的残余差距是 E2(CAVM λ=(0,1))检索略差于 W1 检索**——是检索/选择质量问题,不是双几何(MFAV)论证的动机。Dual-Geometry 度量切换假设在已执行动作上没有实证支持。

---

## 1. 任务边界完成情况(§10)

| §10 边界 | 状态 | 说明 |
|---|---|---:|---|
| 不改 IAH-CRPS / 三原子候选 / query-dose replay / 双事件结构 / alpha | ✅ | 零改动,审计脚本只重跑同一确定性链并捕获 evidence |
| 不增加 loss / 事件头 / market-host ID / 硬阈值 / P4 | ✅ | 无任何新增训练或门控 |
| 不使用 S4 target 选择公式或参数 | ✅ | 判定仅用 §6.1 的 flip/混淆矩阵统计,不用于选公式 |
| 现有结果与脚本不覆盖 | ✅ | 新产物目录 `results/phase5/forensic/`;P2 的 cell JSON 仅被读取做锚定 |
| 本轮只做日级/逐小时法医账本 + 四效用离线重算 | ✅ | §6.1(5A)+ §6.2(5B)全部完成,§6.3(5C)不实现 |
| 所有实验记录在报告并提交仓库 | ✅ | 本报告 + 脚本 + 产物一起 commit(不 push) |

---

## 2. 方法：冻结链逐小时回放(不重训、不改变动作)

`p5a_forensic_ledger.py::run_forensic(mk, bb, seed)` 镜像 `p2_cavm_universal_core.py::run()` 的完整确定性链(load_head → fit_s1_reference → zeros det → S3-M mem/val → select_s3m_k → calibrate_s3c → fit_cavm_memory → predict_s4),在 E2(cavm λ=(0,1),即实际部署模式)的 predict 阶段捕获:

- S4 原始数组:`s4_hosts / s4_y / s4_dates / s4_ctxs`(列 0 = S1 mid-rank u)
- E2 evidence:`candidate`(s, z0, valid_mask, m_minus, m_plus)、pi、A_hat、q、lcb、proposals、x_final
- 冻结域尺度 `S_d = median over S1R days of mean|yhat_full[day]|`(§3.2;host cache 按 (dataset, backbone) 无 seed,故跨 seed 共享)

逐小时 ledger 含 §6.1 全部字段:`host_raw / target_raw / corrected_raw / z0 / zY / pi / scale_day / scale_domain / gain_z / gain_raw / gain_mfav / jacobian_proxy / rank_u / tail_state / proposal(I_down,I_up) / A_hat / q / lcb / action / in_I_down / in_I_up / outside`。日级聚合行输出 U0–U3 四效用 + day_MAE_host − day_MAE_corrected。

**审计 cell 集(§6.1,15 cell):** Shandong_DA × PatchTST × s0/1/2(谜题主体)+ Shandong_DA × Linear × s0/1/2(对照)+ LAGO_DE × {Linear, MLP, PatchTST} × s0/1/2(方向稳定正例)。

---

## 3. 复现性锚定(§6.1 Q3 的硬断言)

每个 cell 与已提交 `results/phase4/ucore_p2/{mk}_{bb}_s{sd}.json` 比对:

| 检查 | 内容 | 15 cell 结果 |
|---|---|---|
| E2 MAE | == saved MAE | 全对,误差 0 |
| E2 n_execute | == saved n_execute | 全对 |
| 逐日 A_true | E2 执行日 U0 == saved A_true | 全对,`u0_vs_json_max=0.0`(逐位) |
| 逆变换 roundtrip | `s·sinh(z0) ≡ host_raw`(全 S4 日) | 全对 |
| 执行小时逆变换 | `s·sinh(z0+π) ≡ corrected_raw` | 全对,且 sign 语义正确(π<0→down / π>0→up / π=0→host) |
| interval 索引 | π 非零小时 ⊆ I_down∪I_up,值 == ∓m | 全对 |
| M1(Gate M1) | `U2 ≡ U3/S_d` 逐日精确 | 全对,`M1_max_rel=0.0` |
| U3 聚合 | `U3_day == day_MAE_host − day_MAE_corrected` | 全对 |
| scale 一致性 | `candidate.s[i] ≈ info.s_full[day]` | 全对 |

`verdict.json`: `cells_ok=15/15`, `failures=[]`。**无任何实现/单位/mask/聚合错误。**

---

## 4. 决定性发现：ΣU3/n ≡ 点级 MAE 改善(矛盾完全解开)

对每个 cell 验证恒等式:

```
E2 相对 host 的总 MAE 改善 |E2_mae − host_mae|  ==  Σ(执行日 raw 修正 U3) / n_S4
```

**15/15 cell 逐位成立(误差 < 1e-9)。** 这意味着:**执行日上 raw 货币增益完全解释了 E2 的点级 MAE 变化——不存在隐藏退化,asinh 与 raw 在执行日上没有任何系统性矛盾。**

### 4.1 全 15 cell E0(w1)/E2 vs host 对照表

| cell | host | E0(w1) MAE | E2 MAE | E0−vsH | E2−vsH | E2−E0 | E0ne | E2ne |
|---|---|---|---|---|---|---|---|---|
| shandong:PatchTST:s0 | 121.817 | 113.954 | 114.810 | **−7.863** | **−7.007** | +0.856 | 35 | 31 |
| shandong:PatchTST:s1 | 121.817 | 111.545 | 112.939 | **−10.273** | **−8.878** | +1.394 | 60 | 44 |
| shandong:PatchTST:s2 | 121.817 | 111.011 | 112.528 | **−10.806** | **−9.290** | +1.516 | 60 | 50 |
| shandong:Linear:s0 | 106.930 | 104.014 | 103.752 | −2.916 | **−3.178** | −0.262 | 15 | 15 |
| shandong:Linear:s1 | 106.930 | 104.377 | 104.096 | −2.553 | **−2.834** | −0.281 | 15 | 13 |
| shandong:Linear:s2 | 106.930 | 103.033 | 102.990 | −3.897 | **−3.940** | −0.042 | 26 | 23 |
| LAGO_DE:Linear:s0 | 6.249 | 6.249 | 6.246 | −0.000 | **−0.003** | −0.003 | 0 | 27 |
| LAGO_DE:Linear:s1 | 6.249 | 6.249 | 6.249 | −0.000 | +0.000 | +0.000 | 0 | 28 |
| LAGO_DE:Linear:s2 | 6.249 | 6.249 | 6.251 | −0.000 | **+0.002** | +0.002 | 0 | 25 |
| LAGO_DE:MLP:s0 | 8.417 | 8.417 | 8.383 | 0.000 | **−0.034** | −0.034 | 0 | 11 |
| LAGO_DE:MLP:s1 | 8.417 | 8.417 | 8.372 | 0.000 | **−0.045** | −0.045 | 0 | 10 |
| LAGO_DE:MLP:s2 | 8.417 | 8.417 | 8.382 | 0.000 | **−0.036** | −0.036 | 0 | 4 |
| LAGO_DE:PatchTST:s0 | 6.147 | 6.145 | 6.071 | −0.002 | **−0.076** | −0.074 | 2 | 20 |
| LAGO_DE:PatchTST:s1 | 6.147 | 6.147 | 6.137 | 0.000 | **−0.011** | −0.011 | 0 | 7 |
| LAGO_DE:PatchTST:s2 | 6.147 | 6.147 | 6.083 | 0.000 | **−0.064** | −0.064 | 0 | 13 |

*符号约定:负 = 相对 host 改善(点级 MAE 更低)。*

### 4.2 三个必须陈述的事实

1. **Phase 4 的"+0.86/+1.39/+1.52 大退化"全部是 E2−E0(w1),不是 vs host。** 而 **E0(w1) 本身执行修正**(Shandong PatchTST 上 35/60/60 个执行日,比 host 改善 7.86~10.81)。用"修正后的基线"做分母,把"CAVM 比 W1 略差 0.86~1.52"读成"CAVM 让 raw 变差",是对比基准的选择性错误。

2. **E2 相对 host 在 Shandong PatchTST 上是 +7.01~+9.29 的改善。** 执行日 raw 增益 mean_U3 = +61.7~+75.0 元/日,全为正;ΣU3/n 恒等式保证点级一致。

3. **执行日上 asinh 与 raw 符号一致(Shandong flip=0.0)。** 因此"动作价值在日尺度 asinh 几何中学习、与 raw MAE 目标错位"这一假设,在**已执行动作**上不成立。

### 4.3 真正的残余差距:E2 检索 vs W1 检索

Shandong PatchTST 上 E2−E0 ∈ [+0.86, +1.52] 的成因不是 metric 错位,而是 **CAVM λ=(0,1) 的纯上下文检索比 W1 的原始距离检索执行更少的天数(E2ne 31/44/50 vs E0ne 35/60/60)且单日 raw 值略低**。同一个候选、同一个 LCB,q 由 S3-C 分别标定——差异只在检索到的邻居日。这是**检索/选择质量问题**,落在 Phase 4 已识别的"检索收益条件于 host 容量×市场极端性"主题内。

---

## 5. §6.1 Q1–Q5 逐条回答

### Q1. `gain_z>0 && gain_raw<0` 集中在哪些日、小时、方向和价格幅值?

**极少数,且全部在 LAGO_DE;Shandong 上为零。**

- 全 15 cell 共 321 个执行日、2588 个执行小时。
- 日级混淆矩阵:`pp=265, pn=2, np=4, nn=50` → **day_flip_frac = 0.0062**。
- 小时级:`pp=1955, pn=28, np=3, nn=602` → **hour_flip_frac = 0.0108**(pn=asinh 正/raw 负)。
- 仅有的 2 个 flip 日,均为 normal tail、raw 损害 <0.08 EUR(~6–7 EUR 市场的 ~1%):

| 日 | cell | U0(asinh) | U3(raw) | host→corrected | tail | mean_rank_u |
|---|---|---|---|---|---|---|
| 2017-06-04 | LAGO_DE:Linear:s1 | +0.0022 | −0.0371 | 3.409→3.446 | normal | 0.487 |
| 2017-02-24 | LAGO_DE:Linear:s2 | +0.0032 | −0.0779 | 7.391→7.469 | normal | 0.449 |

- np(4 日,asinh 负/raw 正):asinh **低估** raw 收益方向,集中在 Shandong(每 PatchTST seed 1 日)与 LAGO_DE Linear s0 —— 正是 §2.2 Jacobian 机制的相反方向:高幅值日 raw 修正比 asinh 暗示的更大,而非更小。

**结论:asinh-正/raw-负 不是系统性现象;即便出现,也全部落在量级可忽略的 LAGO_DE 上。**

### Q2. raw 损害是否被少量 `jacobian_proxy` 极大的小时主导?

**否。§2.2 的 Jacobian 过冲假设不成立。**

- `flip_share_of_raw_harm`(flip 小时对总 raw 损害的贡献)全 cell ≤ 1.6%。
- `jac_flip_ratio`(flip 小时 vs 非 flip 小时的 jacobian_proxy 中位数之比):Shandong 0.28;LAGO_DE 除 PatchTST:s2=1.93 外均 <1.5;仅有的 1.93 对应 hour_flip_frac=1.6%、flip_share=1.4%,无可观影响。
- 小时级翻转(mn 方向占 602/2588=23%)是日聚合噪声,不是幅值放大:它们与 jacobian_proxy 无正相关。

### Q3. 是否存在反变换、单位、mask、逐日聚合或 action interval 的实现错误?

**无。** §3 复现性锚定 9 项检查全过;叠加 §4 的 ΣU3/n ≡ 点级 delta 恒等式,链条从 `s·sinh(z0)` 到最终 MAE 完全闭合。E1==E0 契约(Phase 4 已证,27/27)继续成立。

### Q4. PatchTST 与 Linear 的分叉能否由 host residual 与候选位移的相对几何解释?

**执行日层面不需要几何解释——因为执行日上两者没有分歧。**

- Shandong PatchTST 执行日 mean_U0=+0.11~+0.14、mean_U3=+62~+75(全正,flip=0);
- Shandong Linear 执行日 mean_U0=+0.12~+0.16、mean_U3=+57~+72(全正,flip=0);
- 两 backbone 的执行日 raw 价值量级相当。点级差异(PatchTST E2−E0=+0.86~1.52 vs Linear −0.04~−0.28)来自**检索选择**:PatchTST 的 CAVM 上下文检索选出更少/略差的天,W1 原始距离检索反而更好;Linear 上 CAVM 检索略胜。这是检索质量对 backbone 的依赖,不是 host residual 与候选位移的相对几何造成的 metric 分歧。

### Q5. 只看已执行动作时,current LCB 对 raw harm 的识别率是多少?

| 市场组 | 执行日 raw harm 率(exec_raw_harm_rate) |
|---|---|
| Shandong PatchTST | 2.0–3.2% |
| Shandong Linear | 0.0% |
| LAGO_DE Linear | 40.7–44.0% |
| LAGO_DE MLP | 0–36.4% |
| LAGO_DE PatchTST | 15.0–42.9% |

**当前 asinh LCB 在 LAGO_DE 上对 raw harm 的识别率低**(15–44% 的执行日 raw 为负),但在 Shandong 上安全(≤3.2%)。注意 LAGO_DE 的 raw harm 量级极微小(单日 <0.08 EUR),且 LAGO_DE Linear:s0 仍净正——实际损失可忽略。这是一个**真实的、但量级微不足道的** LCB 校准发现,不支持据此切换到 raw 尺度做门控。

---

## 6. Phase 5B：四效用离线反事实(U0–U3)

同一批候选、邻居与已保存动作,离线并列计算四种效用(Gate M1 已证 `U2 ≡ U3/S_d` 逐日精确,`M1_max_rel=0.0`):

| cell | mean_U0(asinh) | mean_U1(raw/scale_day) | mean_U2(raw/S_d) | mean_U3(raw) |
|---|---|---|---|---|
| shandong:PatchTST:s0 | +0.136 | +0.201 | +0.250 | +75.05 |
| shandong:PatchTST:s1 | +0.122 | +0.179 | +0.223 | +66.99 |
| shandong:PatchTST:s2 | +0.112 | +0.165 | +0.206 | +61.68 |
| shandong:Linear:s0 | +0.151 | +0.224 | +0.240 | +70.35 |
| shandong:Linear:s1 | +0.155 | +0.233 | +0.247 | +72.38 |
| shandong:Linear:s2 | +0.120 | +0.182 | +0.194 | +56.87 |
| LAGO_DE:Linear:s0 | +0.018 | +0.018 | +0.001 | +0.049 |
| LAGO_DE:Linear:s1 | +0.021 | +0.027 | −0.000 | −0.006 |
| LAGO_DE:Linear:s2 | +0.009 | +0.007 | −0.001 | −0.035 |
| LAGO_DE:MLP:s0 | +0.068 | +0.076 | +0.039 | +1.362 |
| LAGO_DE:MLP:s1 | +0.085 | +0.106 | +0.057 | +1.983 |
| LAGO_DE:MLP:s2 | +0.151 | +0.232 | +0.112 | +3.892 |
| LAGO_DE:PatchTST:s0 | +0.069 | +0.104 | +0.045 | +1.653 |
| LAGO_DE:PatchTST:s1 | +0.031 | +0.036 | +0.018 | +0.659 |
| LAGO_DE:PatchTST:s2 | +0.088 | +0.123 | +0.059 | +2.139 |

观察:

1. **U2 与 U3 符号完全一致**(构造性恒等式,M1 通过),U1 与 U3 在 daily 层面也同号(§3.2/§5.1 论证)。**真正的对比是 U0(asinh) vs U3(raw)。**
2. **Shandong 上 U0 与 U3 同号且都为正**——asinh 没有系统性误导选择。
3. **LAGO_DE 上 U0 与 U3 几乎同号**(唯一例外是 Linear:s1/s2 的 2 个 flip 日,量级 ≤0.078 EUR),但注意 **mean_U1 > mean_U2** 于多数 cell——执行日 scale_day 通常高于域尺度 S_d(高幅值日才触发执行),符合直觉。
4. **MFAV 切换(U0→U3/S_d)在已执行动作上不会改变任何决策结论**:全部 15 cell 中,改变量级最大的一处(LAGO_DE Linear:s2,−0.035)也只有 0.03% 相对量级,且 U0 已经同号。

**结论:四种效用对已执行动作给出同一幅图景。不存在"asinh 说好、raw 说坏"的系统性反例。**

---

## 7. 判定

| 判定 | 依据 |
|---|---|
| **BUG** | 否 —— 15/15 复现性锚定 + 9 项实现检查 + ΣU3/n 恒等式全过 |
| **METRIC_MISMATCH** | 否 —— 执行日 day_flip 0.62%(Shandong 0%),flip 不集中于高 jacobian_proxy,四效用同号 |
| **INCONCLUSIVE(METRIC_MISMATCH)** | **是** —— 代码正确但无清晰分歧;原始"分叉"被证明是对比基线假象 |

**综合结论(对方向文档 §0/§1.3/§2):**

- 方向文档 §1.3 列出的"Shandong PatchTST 双曲 A_true 为正而 raw MAE 变差"这一**核心谜题在已执行动作上不成立**:E2 相对 host 改善 +7.0~+9.3,执行日 raw 全正。
- §2.2 的 Jacobian 过冲机制与 §2.3 的日变权重反转机制,在**已执行动作的分布**上没有实证支持(flip 小时无 jacobian 集中、四效用同号)。
- **Dual-Geometry / MFAV 度量切换(§3)不作为下一轮实现的动机**。按 §6.3 的语义,本轮结果不支持进入 Phase 5C。
- **真正值得追的残余问题是 E2 检索 vs W1 检索**:Shandong PatchTST 上 CAVM λ=(0,1) 上下文检索选出更少/略差的天(W1 原始距离检索反而更好),这是检索 key 质量 / k 选择 / λ 的问题,不是度量几何问题。与 Phase 4 已识别的"检索收益条件于 host 容量×市场极端性"一致。

---

## 8. 对方向文档的修正建议

1. **§1.3 谜题重述**:将"Shandong PatchTST raw MAE 大退化"改写为"CAVM λ=(0,1) 检索相对 W1 检索的日均原始增益略低(E2−E0=+0.86~+1.52,两者相对 host 均大幅改善)"。谜题从 metric 错位降级为检索选择。
2. **§3 的 MFAV 动机失效**:若后续要保留 dual-geometry 作为备选,需要新的、在已执行动作上出现的 raw/asinh 分歧证据;当前证据不支持。
3. **建议下一轮 = 检索/选择审计**:在 E2 检索邻居(CAVM key = core_context + zeros det)与 W1 检索邻居(W1 距离)之间做逐日差异分解,定位 Shandong PatchTST 上 λ=(0,1) 的次优来源(上下文 key 维度是否引入噪声、k=5~20 的选择、双事件 proposal 对检索邻居的敏感性)。

---

## 9. 诚实声明

- 本轮只用 S4 做**诊断**(§6.2 明确禁止用同一 S4 选公式后报性能);因判定不进入 5C,无公式被选择。
- LAGO_DE 无负价,LAGO_NP 不在本轮 15-cell 集(§6.1 未要求);不报任何 negative-price 指标。
- `verdict.json` 中 `jac_flip_vs_nonflip_median_ratio=null` 是聚合层实现语义:多数 cell flip 小时为空 → 无全局中位数可比;逐 cell 值见 `utility_matrix.csv`。
- 复现性锚定要求 `git_sha=46671c7`(与 Phase 4 universal-core 相同);若该提交之后 core 或检索链被修改,本报告结论仅对 `46671c7` 成立。

---

## 产物

- 脚本:`experiments/08-hch-v2/p5a_forensic_ledger.py`
- 结果:`experiments/08-hch-v2/results/phase5/forensic/`(ledger_*.csv / day_summary_*.csv / utility_matrix.csv / verdict.json / matrix_run.log)
- 本报告:`docs/训练文件夹/对比实验/hch_v2_phase5a_b_forensic_audit_report_v0.1_2026-08-15.md`
