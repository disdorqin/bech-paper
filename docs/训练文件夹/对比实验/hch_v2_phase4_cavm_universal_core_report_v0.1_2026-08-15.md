# HCH-v2 Phase 4：Universal-Core CAVM 确认实验（Paper-config）v0.1

**日期：** 2026-08-15

**依据：**

- `hch_v2_phase4_cavm_p2ext_gate_review_and_next_design_v0.1_2026-08-15.md`（§4/§5/§9 任务边界）
- `hch_v2_phase4_cavm_p2ext_report_v0.1_2026-08-15.md`（P2 扩展 GATE_NOT_YET_PASS）
- 当前主数学：`hch_v2_iah_crps_final_math_core_v0.3`
- 当前架构：v0.4 Universal Core / Data Signature / Local Evidence

**状态声明（§9.1）：** 上一轮提交 `99f75c5`（P2 扩展报告）**未推送**到 `origin`（`git branch -r --contains 99f75c5` 为空，且 99f75c5 即当前 HEAD），因此**无法从公开仓库独立复核**其代码状态；本报告所有结论以本地仓库 `HEAD=99f75c5` 为准。

---

## 1. 任务边界完成情况（§9）

| §9 条目 | 状态 | 说明 |
|---|---|---:|---|
| 1. 核对 99f75c5 真实状态 | ✅ | 未推送，无法公开复核（见上） |
| 2. 不改变现有 E0–E3 结果 | ✅ | 未改动 `p2_cavm_experiment.py`/`p2_cavm_multimarket.py`；本轮新脚本 `p2_cavm_universal_core.py` 独立输出 |
| 3. 实现/确认 paper-config UniversalCoreTrainer | ✅ | 复用现成冻结核心 head_vA（d_model=64, learned_sig, zeros det） |
| 4. candidate/proposal/LCB 分解日志 | ✅ | 每 cell JSON + `matrix.csv` + `summary.json` 全量落盘 |
| 5. 27-cell 快速筛选 | ✅ | 3 target × 3 host × 3 seed，**0 失败**，27/27 通过 E1==E0 契约 |
| 6. 3 seed 收敛、candidate alive、proposal empty、execute-conditioned 指标 | ✅ | 见 §4-§7；**逐 epoch 曲线不可用**（见 §5 诚实声明） |
| 7. 不实现 P4 | ✅ | 未实现 action-value state update |
| 8. 不新增 loss/事件头/market-host-ID gate/硬阈值 | ✅ | 未新增任何 loss 或门控 |
| 9. 返回情况 A/B/C/D | ✅ | 代码判定 B；按设计 §6 分支语义逐市场拆解后为 **D**（见 §8） |

---

## 2. 方法：冻结 universal core 上的 E0–E3 受控链

### 2.1 与 P2 扩展的决定性差异

| 维度 | P2 扩展（99f75c5） | 本实验（paper-config） |
|---|---|---|
| 候选头 | 每个目标市场**自己训练**（S2T/S2V，d_model=32） | **冻结** head_vA（12 国际源域等域采样，d_model=64） |
| 训练 | S2 每 cell 一次 | **无**（candidate 权重永不更新） |
| 确定性 descriptor | `fit_s1_signature` → 目标市场 S1R det（变体 #2） | **zeros(8)**（learned_sig 标准，det 恒零） |
| CAVM key c_sig | 目标市场 det（非零） | **恒 0**（身份无关） |
| 目标域可用信息 | — | 合法 S1 ref + target-local S3-M/S3-C，仅此 |

### 2.2 每 cell 链（`p2_cavm_universal_core.py`）

```
load_head(seed)                       # head_vA_seed{s}.pt, d_model=64, learned_sig
  → HCHV2UniversalPipeline(d_core_context=13, d_model=64, d_value=0,
                           alpha=0.10, k=None, seed, memory_mode="cavm")
  → candidate_head.load_state_dict(head.state_dict()); eval()
  → fit_s1_reference(info.s1_z0, info.s1_hours)        # target-local S1 ref
  → _domain_det = zeros(8); set_domain_descriptors(zeros)  # learned_sig det=0
  → S3-M mem/val (k_val_frac=0.25) → fit_s3_memory → select_s3m_k
  → S3-C → calibrate_s3c (DVG q)
  → fit_cavm_memory(mem_days)          # global ledger, core_context + zeros det
  → predict_s4 E0(w1) / E1(cavm 1,0) / E2(cavm 0,1) / E3(cavm 1,1)
  → A_true offline（同 estimate_realized_A）+ C0-C3 诊断（E0/W1 参考证据）
```

### 2.3 契约检查（全 27 cell 通过）

- **E1(cavm λ=(1,0)) 逐字复现 E0(w1)：** 27/27 cell 邻居切换 0 天、ΔMAE=0.0、ΔA_true=0.0、Δexec_rate=0.0。
- CAVM key dim=55 与 d_model 解耦（布局 [shape8|dyn5|time26|sig8|atom8]，不依赖 32/64）。
- `x_final` 语义：identity 日 = host 预测；execute 日逐小时按 π 覆盖为 x_down/x_up。MAE delta 因此只由 execute 日集合变化驱动。

---

## 3. §4.4 Data-Signature 配置审计表（实证）

head_vA 训练链 = `training_investigation_trainer_cmp.py` Version A =
`r1b_generalization_screen.train_candidate("learned_sig", 12 域, seed)`
→ `UniversalCoreTrainer(head, ...)` 等域采样 + IAH-CRPS + macro S2V checkpoint 选择。

权重实证（head_vA_seed0.pt）：

```
core_encoder.signature.domain_det            shape (8,)    nonzero_frac 0.0000  ← 零描述符
core_encoder.signature.learned_proj.0.weight shape (32,64) nonzero_frac 1.0000 ← learned per-day sig 已训练
core_encoder.signature.learned_proj.0.bias   shape (32,)   nonzero_frac 1.0000
core_encoder.signature.mod_head.weight       shape (128,40) nonzero_frac 0.7750 ← FiLM 已训练
core_encoder.proj.weight / mass_head / shift_head  全满权重
```

| # | 审计项 | 结论 | 证据 |
|---|---|---|---|
| 1 | 哪些 descriptor 参与 universal training | **只有 learned per-day signature**（`learned_proj`→`mod_head` FiLM）；确定性 det=zeros 显式传每 batch | `domain_batches` 用 `det_for_variant("learned_sig")`=zeros |
| 2 | 哪些 descriptor 在 target inference 可用 | 同训练：learned sig（冻结权重）+ det=zeros | 本实验 set_domain_descriptors(zeros) |
| 3 | 是否使用 zero descriptor | **是**，det 恒零（身份无关） | 权重实证 nonzero_frac=0.0 |
| 4 | learned signature 是否真训练 | **是**（learned_proj/mod_head 全满权重） | 权重实证 |
| 5 | 是否某市场 descriptor 被错误冻结到所有 batch | **否**；无市场/宿主 ID 进入核心路径 | det 全零 + market_id 仅审计元数据 |

> **结论：** head_vA 是**真正的 paper-config universal core**（单共享实例、等域采样、learned_sig、zero det），不是每市场 descriptor/candidate head 变体。

---

## 4. 27-cell 快速筛选：E2(cavm 0,1) vs E0(w1) 结果矩阵

**协议：** 3 target（LAGO_DE / LAGO_NP / shandong_DA）× 3 host（Linear/MLP/PatchTST）× 3 seed。每 cell 的 E1==E0 契约 27/27 成立。

### 4.1 E2 vs E0：按 market 聚合（9 cell / market）

| market | point_help | ΔMAE | ΔA_true | action_help | 3-seed | market_improves |
|---|---:|---:|---:|---:|:---:|:---:|
| **LAGO_DE**（极端，neg 1.85%） | **7/9** | **−0.0293** | **+0.0085** | **9/9** | **3/3** | ✅ |
| LAGO_NP（温和，neg 0%） | 4/9 | +0.0012 | −0.0001 | 1/9 | 1/3 | ❌ |
| shandong_DA（极端，neg 11.7%） | 3/9 | +0.3544 | −0.005 | 0/9 | 0/3 | ❌ |

### 4.2 E2 vs E0：按 cell（ΔMAE, raw 电价单位）

| host \ market | LAGO_DE | LAGO_NP | shandong_DA |
|---|---:|---:|---:|
| Linear s0/s1/s2 | −0.003 / +0.000 / +0.002 | −0.000 / +0.000 / −0.000 | **−0.262 / −0.281 / −0.042** |
| MLP s0/s1/s2 | **−0.034 / −0.045 / −0.036** | +0.002 / +0.004 / −0.002 | +0.008 / +0.000 / +0.000 |
| PatchTST s0/s1/s2 | **−0.074 / −0.011 / −0.064** | +0.007 / +0.000 / −0.000 | **+0.856 / +1.394 / +1.516** |

### 4.3 E3(cavm 1,1) vs E0（overall 27 cell）

| 模式 | ΔMAE | ΔA_true | point_help | action_help | opposite(point/action) |
|---|---:|---:|---:|---:|---:|
| E2 (0,1) | +0.1088 | +0.0011 | 14/27 | 10/27 | **8** |
| E3 (1,1) | +0.0055 | +0.0017 | 15/27 | 16/27 | **3** |

E3 的 W1 锚定**稀释了 Shandong PatchTST 的破坏性上下文检索**（opposite 8→3，action_help 10→16），同时 LAGO_DE 收益略减。这与 P2 中"复合检索被 W1 稀释"同源，但在本实验里稀释起到保护作用。

### 4.4 三市场诚实读法

- **LAGO_DE**：唯一方向稳定。MLP/PatchTST 全 seed 改善（Δ≈−0.03/−0.05），Linear 中性；9/9 action_help。
- **LAGO_NP**：Δ 全部 ≤0.007（~0.2%），动作 A_true ≈ 0（~1e-4）→ **无动作价值**（负价 0%，温和市场无修正空间）。
- **shandong_DA**：**host-split 依然存在但方向反转**——Linear 全 seed 改善（−0.26/−0.28/−0.04），PatchTST 全 seed 大退化（+0.86/+1.39/+1.52），MLP 中性。与 P2-ext（Linear 退化 −1.1%、PatchTST 改善 −3.7%）**符号翻转**，说明"host 分裂"不是目标域训练数据导致的 artifact，而是冻结核心候选相对各 host 误差几何的固有属性。

---

## 5. 三 seed 收敛信息（§9.6）

**训练阶段（head_vA, `results/TRAINER_CMP_20260814_seeds012/report.json`）：**

| seed | best_macro_s2v | worst_s2v@best |
|---|---:|---:|
| 0 | 0.2275 | 0.6761 |
| 1 | 0.2288 | 0.6788 |
| 2 | 0.2238 | — |

- **诚实声明：** trainer 只保存 best_macro_s2v，**无逐 epoch 历史**，三 seed 收敛曲线不可重建。本报告只报 best 值 + cell 级 3-seed 一致性。
- **S4 cell 级 3-seed 一致性**（27 cell 全部 3 seed 齐）：LAGO_DE 改善在 3/3 seed 成立；Shandong Linear 改善 3/3、PatchTST 退化 3/3；LAGO_NP 中性 3/3。**无任何方向依赖单一 seed。**

---

## 6. C0-C3 动作容量分解

### 6.0 聚合 vs 按市场（关键！聚合掩盖跨市场异质性）

| 指标 | 聚合(27) | LAGO_DE | LAGO_NP | shandong |
|---|---:|---:|---:|---:|
| C0 alive m−/m+ | 0.973/0.942 | 0.94/0.94 | 1.00/0.99 | 0.98/0.90 |
| C0 oracle_wm_vs_host（负=候选更优） | **−12.27** | −0.26 | −0.19 | **−36.29** |
| C1 proposal_empty | 0.333 | 0.13 | **0.99** | 0.05 |
| C1 A_hat_pos_frac | 0.667 | 0.87 | 0.02 | 0.94 |
| C2 no_lcb_rate / with_lcb_rate | 0.667 / 0.029 | 0.86 / 0.001 | 0.16 / 0.016 | 0.97 / 0.071 |
| C2 **no_lcb_A_true** | +0.0032 | **−0.0086** | +0.0002 | **+0.0184** |
| C2 with_lcb_A_true | — | **+0.0251** | +0.0003 | **+0.12** |
| C3 E0 exec 天数 / E2 exec 天数 | — | 0.2 / 17.2 | 7.0 / 24.2 | 23.4 / 19.7 |

### 6.1 C0 候选支持——**容量充足，设计 §6 的 Case C 恐惧被排除**

- 冻结 universal core 的 weighted_mean 读出在**未见市场**上比 host 平均好 **12.27 MAE 单位**（Shandong 主导：候选 82-88 vs host 107-122）。LAGO_DE −0.26、LAGO_NP −0.19。
- **核心不坍塌**：跨 3 市场 9 host，candidate alive 均 >0.93。
- 但 LAGO_NP 的 oracle 优势在 host 绝对尺度 ~2.6 上量级可忽略（−0.19/2.6 ≈ 7%）。

### 6.2 C1 proposal 支持

- LAGO_DE / Shandong proposal 基本非空（empty 0.13/0.05），A_hat>0 占 0.87/0.94。
- **LAGO_NP proposal 几乎全空（0.99）**，A_hat>0 仅 0.02 → 温和市场不是"检索失败"，是**候选在此无动作容量**。

### 6.3 C2 DVG 门控——**"LCB 过度保守"只在聚合层面成立，逐市场不成立**

- 聚合 no_lcb_A_true=+0.0032（正）→ 代码判 lcb_overconservative=true。**但逐市场拆开：**
  - **LAGO_DE：no_lcb_A_true = −0.0086（负）** → 无 LCB 的"proposal 非空且 A_hat>0"规则本身**无收益**，LCB 拦截是正确的（放行的 0.001 天 A_true=+0.0251）。问题在 **A_hat→LCB 校准**，不是 LCB 过度保守。
  - **Shandong：no_lcb +0.018、with_lcb +0.12** → LCB 放行的天**强正收益**。但 E2 的 raw-yuan MAE 在 PatchTST 仍大退化 → 问题同样不在 LCB 保守，而在**检索→动作的日集合选择**。

### 6.4 C3 实现收益——**scale-free 动作价值与 raw point 结果分叉**

- E0 参考证据下 C3：execute-conditioned A_true 为正（LAGO_DE +0.025、Shandong +0.12），harm 低。
- **但 Shandong PatchTST 上 E2 执行日 scale-free A_true 仍为正（+0.108~+0.136），raw-yuan point MAE 却退化 +0.86..+1.52** —— 动作的 scale-free 价值与 raw 点结果在 PatchTST 上分叉。这不是 A_hat 校准可解释的（执行日实现 A_true 是正的），需要**日级逐小时审计**（下一阶段，超出本轮边界）。
- 已排除混淆：`x_final` 在 identity 日 = host（两种模式相同），MAE delta 完全由 execute 日集合的替换驱动。

---

## 7. 关键机制解释（事实 vs 假设）

| # | 陈述 | 类型 | 依据 |
|---|---|---|---|
| 1 | 冻结 universal core 候选在未见市场容量充足（oracle 优于 host 12.27） | **事实** | C0 oracle_wm_vs_host 逐 cell |
| 2 | 检索收益条件于**市场极端性 × host 误差几何**，非普适 | **事实** | LAGO_DE 稳定 / LAGO_NP 空 / Shandong 分裂 |
| 3 | Shandong host-split 方向在冻结核心下**反转**（Linear 改善、PatchTST 退化） | **事实** | 与 P2-ext 符号对比 |
| 4 | LCB 不是主要瓶颈（LAGO_DE no-LCB 无收益；Shandong with-LCB 强正收益） | **事实** | C2 逐市场 |
| 5 | Shandong PatchTST 的 raw 退化源于执行日集合替换 | **事实** | x_final 语义 + MAE 分解 |
| 6 | 执行日 scale-free A_true 与 raw point 结果分叉的原因 | **假设** | 需日级逐小时审计（下一阶段） |
| 7 | 复合检索 E3 的 W1 锚定起保护作用 | **事实** | opposite 8→3 |

---

## 8. 情况 A/B/C/D 判定

### 8.1 代码判定（`summary.json`）→ **B**

聚合阈值全触发：candidate_capacity ✓、proposal_support ✓、no_lcb_profitable ✓（聚合 +0.0032）、lcb_overconservative ✓（with_lcb 0.029 vs no_lcb 0.667）。**n_market_improving=1（仅 LAGO_DE）。**

### 8.2 按设计 §6 分支语义逐市场拆解 → **D**

| 设计分支 | 判据 | 本实验逐市场 | 结论 |
|---|---|---|---|
| A | ≥2 市场类型方向稳定 | 仅 LAGO_DE | ❌ |
| B | "no-LCB 有收益 + LCB 过度保守" | LAGO_DE no-LCB **负**（−0.0086）；Shandong with-LCB 强正但 point 仍退化 → LCB 不是瓶颈 | ❌ |
| C | 候选支持不足（dose≈0 / proposal 空 / oracle 无法改善 host） | oracle 明确改善 host（−12.27） | ❌ |
| **D** | **仍然只有 LAGO_DE 稳定改善** | **✓** | **✅** |

**判定：情况 D** —— 但带一个决定性正面更新：

> 与 P2-ext 相比，本实验**排除了"候选表示坍塌"这一解释**（设计 Case C）：冻结 universal core 在未见市场有真实容量（oracle −12.27）。瓶颈已定位到**检索→动作链的日选择**，不是候选表示，也不是 LCB 保守度。

### 8.3 按设计 §6 的后续决策（情况 D）

- CAVM **不能**写成跨市场普适主创新。
- 可如实收束为"极端市场条件下、且检索到的执行日集合正确的动作证据路由"。
- 下一步两个候选（均为 §7/§8 允许方向）：
  1. **Shandong PatchTST 日级逐小时审计**：定位 raw/scale-free 分叉（执行日集合质量、x_down/x_up 幅度）。
  2. **A_hat→LCB 校准审计**（LAGO_DE 侧）：no-LCB 规则 A_hat>0 处 A_true<0，需 S3-C 样本量/误差分布/q 稳定性审计 —— 属设计 §6 Case B 的次级动作，但**仅在 D 的收束框架内做故障定位，不宣称 P4**。
- 不做：P4、按 market/host ID 门控、新 loss、硬阈值（§3 红线，均遵守）。

---

## 9. 边界与诚实声明

1. **99f75c5 未推送**：无法从公开仓库独立复核；本报告以本地 HEAD 为准。
2. **LAGO_NP 负价占比 = 0%**：按规则**不报** LAGO_NP 的 negative-price 指标；LAGO_DE（1.85%）与 Shandong（11.7%）可报，本报告未用负价指标做结论。
3. **逐 epoch 收敛曲线不可用**：trainer 仅存 best_macro_s2v；三 seed 一致性以 cell 级 3-seed 全齐为依据。
4. **A_true 语义**：`mean_A_true` 是全 S4 的 π-期望，非实现值；**exec_mean_A_true** 才是 execute-conditioned 实现值。本报告所有收益声明只用后者。
5. **聚合 C2 的误导性**：`no_lcb_profitable`/`lcb_overconservative` 在聚合层为真，但**逐市场拆解后仅 Shandong 的 no-LCB 为正**，LAGO_DE 为负。§8.2 以逐市场为准，代码判定 B 明确标注为聚合伪迹。
6. **Shandong PatchTST raw/scale-free 分叉**：承认尚未解释（需日级审计），不将其包装为任何正/负结论的证据。
7. 本轮所有数字来自 `results/phase4/ucore_p2/`（matrix.csv / summary.json / 27 个 cell JSON），results/ 目录 gitignored，与既有约定一致。

---

## 10. 一句话结论

> Paper-config 冻结 universal core **不坍塌**（未见市场 oracle −12.27，Case C 排除），但 CAVM 动作链只在 LAGO_DE 方向稳定（唯一 Case D 触发），Shandong 的 host-split 在冻结核心下方向反转且 PatchTST 呈 raw-point 大退化，LAGO_NP 无动作容量；瓶颈已从"候选表示"收窄到"检索→动作的日选择"，进入 Day-level 审计或 A_hat 校准审计，P4 维持不放行。
