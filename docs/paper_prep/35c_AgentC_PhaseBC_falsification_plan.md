# Agent C：Phase B/C 方法与最小证伪计划

> 角色：Agent C（方法与证伪者）  
> 日期：2026-08-09  
> 状态：**仅设计文档**。未实现代码、未创建 `experiments/07-*`、未改 `src/`/`data/`、未跑真实 pilot。  
> 权威协议：`docs/paper_prep/3子agent.md` §1/§2/§6；交接：`docs/paper_prep/36_新窗口交接_PhaseBC_2026-08-09.md`  
> 问题锚点：`experiments/05-episode-audit/results/P0_DECISION.md`（problem-anchor GO；算法对象未成立）  
> Hurdle 对齐：`docs/paper_prep/37_跨领域文献调研_双损失_GNN_共享图_2026-08-09.md`

---

## 0. Scope and hard constraints

### 0.1 本轮唯一交付

写入本文件。禁止：

- 实现任何新算法代码或修改 `src/bech.py` / `src/common.py` / `src/backbones.py`
- 创建或运行 `experiments/07-episode-relative-pilot/`
- 修补 `experiments/_archive/retired-methods/06-event-edit-prototype/`
- 改 `data/`、重跑 `01-05` 冻结矩阵、宣称创新已成立

### 0.2 放行串行闸门（硬）

```text
A 未闭合还原  AND  B 有非定义性 T1
        │
        ▼  主窗口交叉验收通过
允许 Agent C 新建 07 并跑合成 DGP → 仅 DGP-4 非等价后才可真实 pilot
        │
A 判 KILL 或 B 无合格 T1
        │
        ▼
禁止一切真实 pilot；本设计仍有效作 Route-E / PIVOT 档案
```

**A/B 未同时通过前，本设计不得被解释为“已授权启动 Phase C”。**

### 0.3 候选一句话

在完整 24h 交付日上，相对**冻结数值基座** `b=f_0(I)`，在**有界非可加 episode 损失** `L_{ep}` 下，从有限非重叠动作菜单中选择预算化 `KEEP/DELETE/REPLACE/INSERT`，并用 S3 对预注册有限策略族做**成对相对风险**高概率选择；无可认证非 identity → 返回基座。

### 0.4 首要生死公式（设计前提，不可静默改写）

\[
H(a;x,Y)=L_{ep}(T(b,a),Y)-L_{ep}(b,Y),\quad
\arg\min_a \mathbb E[H\mid x]=\arg\min_a \mathbb E[L_{ep}(T(b,a),Y)\mid x].
\]

因此“直接学相对 harm”**默认不产生新 Bayes 动作**；必须靠决策/算法/有限样本/保证四类之一的严格分离存活。本计划的第 4 类合成 DGP 与 B3 即直接检验该等价是否在有限菜单上经验/解析成立。

---

## 1. Frozen object and default budget choice

### 1.1 冻结对象（与协议 §2 一致；发现不自洽只报 SPEC INVALID）

| 符号 | 定义 |
|---|---|
| 审计单位 | 完整交付日 `H=24`（清理 DST/缺失/重复后再按日切分） |
| 基座 | `b=f_0(I)\in\mathbb R^{24}`，S1 训练后冻结 |
| 特征 | `Z=z(I,b)`，仅 cutoff 前信息；残差史 ≥24h 滞后 |
| 真值 | `Y\in\mathbb R^{24}`，推理 API **不得**接收 |
| 负价事件 | `S(y)=\{t:y_t<0\}`，`E(y)=CC(S(y))`；0 非负价 |
| 动作词表 | `KEEP/DELETE/REPLACE/INSERT`；`SHIFT/SCALE`⊂`REPLACE` |
| 编辑器 | `T(b,KEEP)=b`；重叠/越界/冲突/tie 规则确定且预注册 |
| 损失 | \(L_{ep}=w_m L_{miss}+w_f L_{fp}+w_b L_{boundary}+w_v L_{value}\in[0,1]\) |
| 匹配 | 带 dummy 的真正线性分配（Hungarian）；未匹配真=miss，未匹配预测=fp |
| 相对风险 | \(R_{rel}(\pi)=\mathbb E[\Delta_\pi]\)，\(\Delta=L_{ep}(T(\cdot),Y)-L_{ep}(b,Y)\) |
| S3 目标 | \(\Pr_{S3}\{R_{rel}(\hat\pi)\le\epsilon\mid S2\}\ge 1-\alpha\)；仅 \(\epsilon=0\) 称高概率总体非退化 |
| 切分 | S1/S2/S3/S4 = 50/20/10/20 rolling-origin |
| 尖峰 | 可作辅助定义/安全弃权，**不**作主创新对象（P0 已否决用当前 p99 拓扑立论） |

跨日 episode：主方案 24h 截断 + left/right censor 标志；若改连续轴须在看 S3/S4 前预注册且全方法共用。

### 1.2 主预算二选一 → **推荐默认：K 个 episode**

| 选项 | 形式 | 利 | 弊 |
|---|---|---|---|
| **K（默认）** | 每日最多编辑 \(K\) 个 episode 级动作（INSERT/DELETE/REPLACE 各计 1） | 与问题锚点（完整漏报/假事件/边界）同构；菜单可穷举；B2/B3 公平；合成 DGP-2/4 可闭式构造 | 不直接限制价格轨迹总偏移；极端 REPLACE 幅度需 value 头+损失 \(L_v\) 约束 |
| L1 幅度 | \(\|T(b,a)-b\|_1\le B_{amp}\)（按日或按 S1 尺度归一） | 贴近连续重建与经济敏感度 | 与“事件结构”叙事弱耦合；连续松弛后菜单爆炸，DGP-4 穷举困难；更易被纯 value 重建还原 |

**默认预注册：**

- 主预算：`K ∈ {1,2}` 网格，在 S2 只学 proposal/score，**S3 在有限策略族上选 (K, 其他超参)**；推荐报告主行 `K=1`（最苛刻、最可证伪），`K=2` 作敏感性。
- L1 幅度：仅作为 Phase C 后敏感性（若获放行），**不**与 K 同时作为主贡献。
- 结构预算 K、风险容忍 \(\epsilon\)、失败概率 \(\alpha\) **三者分开**，禁止混写成一个“安全预算”。

**理由（一句话）：** P0 失败主型是整段漏报/假事件/边界，不是逐点 L1；K 使动作菜单有限、与 B3 risk-table 同支撑，从而 DGP-4 可真正杀新颖性。

### 1.3 与现有 BECH（`src/bech.py`）的接口级差异

现有 BECH 三步（只读、不改）：

| 步骤 | 现接口 | 行为 |
|---|---|---|
| `fit(Z,yhat,y)` | S2 | 双路点级 `P(neg\|Z)/P(spike\|Z)` + 条件幅度；Hurdle 式 occurrence×magnitude |
| `calibrate(Z,yhat,y)` | S3 | SCARR：bootstrap LCB 效能 + 共形单点 harm≤ρ×基线 → 每分支 λ |
| `apply(Z,yhat)` | 推理 | 点级路由；`ŷ+λδ` 或恒等 |

候选（若获实现）应对齐**同一三段生命周期**，但对象从点改为日级 episode 动作：

| 组件 | 现 BECH | 候选（设计） |
|---|---|---|
| 单元 | 小时点 | 24h 日 + episode 集合 |
| 监督 | 点标签+残差 | 有限动作菜单上的 \(L_{ep}\) / cross-fit harm |
| 输出 | 点 δ | 编辑脚本 → `T(b,a)` |
| 预算 | 隐式 τ/λ | 显式 K（默认） |
| 证书 | 分支 λ 共形+LCB | 有限策略族上成对 \(R_{rel}\) 选择；非 identity 不可证 → KEEP |
| 评分 | \(P\times\)幅度（Hurdle） | **Bi-Hurdle 条件 harm**：\( \widehat H(a)\approx \hat P(\text{event struct})\times\hat m(\text{magnitude/boundary}) \) 或直接回归 \(H\)；与 37 文 Hurdle/Tweedie 对齐作实现选择，**非**新理论宣称 |

**明确继承、不宣称创新：** cutoff-safe Z、冻结基座、S3 弃权回基座、CPU/LightGBM 友好实现路径、Hurdle 分解直觉。

---

## 2. Minimal algorithm (pseudocode)

> 仅当主窗口宣布 A/B 双过后方可实现。下列为伪代码级规格。

### 2.0 共享原语（预注册，全方法共用）

```text
EPISODES(y):  CC({t: y_t < 0})  → list of [s,e]
MATCH(Ep, Eq): Hungarian with dummies; costs from (miss, fp, boundary, value)
L_ep(yhat_vec, y):  match + weighted bounded losses → [0,1]
T(b, a): apply KEEP/DELETE/REPLACE/INSERT with frozen conflict/tie rules
BUDGET_OK(a, K): number of non-KEEP episode ops ≤ K
```

损失权重、空集约定、边界饱和常数、value 归一（建议除以 S1 日 MAE 或价格 IQR）在看 S3/S4 前写入配置哈希。

### 2.1 训练与推理

```text
# ========== S1: freeze backbone ==========
f0 ← FitBackbone(S1);  freeze
for each day d in S2∪S3∪S4:
    b_d ← f0(I_d)          # 24-vector
    Z_d ← z(I_d, b_d)      # cutoff-safe

# ========== S2: proposal + value + cross-fit harm labels ==========
# (1) Proposal model q_θ
#     Input: (b,Z). Output: finite non-overlapping action menu A(x)
#     Must always include KEEP.
#     Generators (deterministic post-process):
#       - DELETE each base-negative episode in EPISODES(b)
#       - REPLACE: boundary ±{0,1,2}h × optional value reshape from r_φ
#       - INSERT: top-M locations from occurrence field (see Bi-Hurdle)
#       - prune overlaps by frozen priority: KEEP < DELETE < REPLACE < INSERT
#         then score; tie → longer support → left-most start
#     Enforce |A(x)| ≤ A_max (e.g. 32) including KEEP.

# (2) Value model r_φ
#     For each proposed editable support S, predict residual/trajectory on S.
#     T MUST consume r_φ output (no label-only heads).

# (3) Forward day-level cross-fitting on S2 (FORBIDDEN: same-day self-score)
Split S2 into F chronological folds (F≥5) or expanding-window blocks.
for fold f:
    fit q_θ^{-f}, r_φ^{-f} on S2 \ fold f
    for day d in fold f:
        A_d ← Menu(q_θ^{-f}, r_φ^{-f}, b_d, Z_d, K)
        for a in A_d:
            yhat_a ← T(b_d, a)
            h_d(a) ← L_ep(yhat_a, Y_d) - L_ep(b_d, Y_d)   # relative harm label
            r_d(a) ← L_ep(yhat_a, Y_d)                     # absolute risk label
# Store tables H_cf, R_cf for scoring-head training.

# (4) Conditional harm / risk scorer s_ψ  (Bi-Hurdle style)
#     Option H1 (default, aligned with BECH/Hurdle):
#        s_ψ(a,x) = g( P_struct(a|x) · m_mag(a|x) )   # occurrence × magnitude
#        train P_struct on whether a reduces miss/fp/boundary mass;
#        m_mag on |H| or value/boundary residual when event-active.
#     Option H2:
#        direct regression of h_d(a) or rank-net on pairs (a,a').
#     Train s_ψ on cross-fit labels only (not on same-fit predictions).
#     Also train absolute risk scorer ρ_ψ(a,x) ≈ E[r|x,a] for B3 parity.

# Refit q_θ, r_φ, s_ψ on full S2 for deployment weights (labels already OOF).

# ========== Inference (S3/S4 days) ==========
function Act(x=(b,Z); θ,φ,ψ, K):
    A ← Menu(q_θ, r_φ, b, Z, K)
    a_hat ← argmin_{a in A, BUDGET_OK(a,K)} s_ψ(a, x)
    # tie → frozen dictionary order on action_signature
    return a_hat, A, scores

# ========== S3: valid selection over pre-registered finite policy class ==========
# Policy class Π (frozen before S3): finite grid, e.g.
#   {Identity} ∪ { (K, τ_score, λ_value, head_id) : ... }
# Each π maps x → action via Act with frozen hyperparameters.
# For each π, on each S3 day compute Δ_π(d).
# Build simultaneous UCB / paired RCPS-style bounds (per Agent B T0):
#   select π* = argmax utility among {π : UCB_rel(π) ≤ ε}, else Identity.
# If no non-identity certified → return base (KEEP) for all S4.
# Multiplicity: union bound / Holm over |Π|.

# ========== S4: lock evaluation only ==========
# No refit, no threshold search, no menu expansion.
```

### 2.2 动作签名（审计用，确定性）

```text
action_signature =
  sorted list of (op, start, end, value_hash)
  value_hash = round(T(b,a)[start:end+1], decimals=6) then blake2s-8
KEEP = [("KEEP", 0, 23, hash(b))]
```

---

## 3. Component split (proposal / value / risk scoring / S3)

必须**分开报告**指标与消融，禁止把端到端增益只记在“新方法”名下。

| 组件 | 输入 | 输出 | 成功定义 | 失败模式（证伪） |
|---|---|---|---|---|
| **Proposal** `q_θ` | `(b,Z)` | 有限非重叠菜单 `A(x)∋KEEP` | 菜单覆盖 oracle 动作 support 的召回；INSERT/DELETE/REPLACE 真实改变 24-向量 | 空 INSERT（旧原型坑）；菜单不含 B2 同构动作；重叠未消解 |
| **Value** `r_φ` | `(b,Z,support)` | 轨迹/残差，被 `T` 消费 | 同 support 下降低 \(L_v\)；`max\|T-b\|>0` 当且仅当非 KEEP | 只改分数不改向量；value 与 REPLACE 脱节 |
| **Risk scoring** `s_ψ` | `(a,x)` | 预测 harm（及绝对 r 供 B3） | OOF 排序与真 \(h\) 一致；Bi-Hurdle 校准 | 同拟合自评泄漏；与 `ρ_ψ-r_KEEP` 同序（→等价） |
| **S3 controller** | `{Δ_π}` on S3 | 选 π 或 Identity | 覆盖保证（T0）；非 BASE power 单独报告 | 无证书却上线；事后扩 Π；经验零退化冒充高概率保证 |

报告模板字段：`n_menu`, `oracle_cover@K`, `insert_fire_rate`, `max_l1_edit`, `spearman(s,h_oof)`, `cert_rate`, `fallback_rate`, `n_pi`.

---

## 4. Strong baselines B0–B3

**公平性铁律：** 相同 `(b,Z)`、S1–S4、容量预算、随机种子、`L_{ep}`、K、tie、S3 族规模与多重性校正。B2/B3 不得被弱化为逐点稻草人。

### B0 Identity

```text
π0(x) = KEEP;  T(b,KEEP)=b
```

用途：安全回退下界；证书空洞时的唯一合法部署。

### B1 Point+Value+RC

```text
# 点级 Hurdle（可复用 BECH 思想，但是基线不是贡献）
for t=1..24:
    p_t ← P(y_t<0 | Z_t, b)
    δ_t ← 1[p_t>τ] · m(Z_t,b)     # magnitude / residual
ŷ ← b+δ  (or BECH-style apply)
# 再把 ŷ 解释为 episode 集合，用同一 L_ep 在 S3 做成对 RC
# 不提供显式 INSERT 菜单以外的结构化 op（点拼轨迹）
```

用途：证明“只要点级+RC”是否已够；DGP-1/2 预期 B1 在 \(L_{ep}\) 上可败给结构化菜单。

### B2 ABS-SET+VAL+RC（**主还原基线**）

```text
# 绝对结构化：直接预测事件集合/区间 + value，不显式写 “相对 harm 损失”
q_abs(·|b,Z) → set of intervals (DETR-style matching 或 boundary head 均可)
r_abs → values on those intervals
ŷ = D_K( decode(q_abs,r_abs), b )   # 同一编辑器语义与预算 K，含可回退 b
# 监督：绝对 L_ep(ŷ,Y)（或 Hungarian matching loss + value）
# S3：与候选同一 paired controller、同一 Π 规模
```

用途：检验候选是否只是“绝对 set prediction + value + RC”的换皮。  
**若 B2 在决策与保证上覆盖候选 → 对象/算法层 NO-GO 倾向。**

### B3 Risk-table reduction（**公式级还原基线**）

```text
# 与候选共用同一 Menu(·) 与同一 cross-fit 表
for a in A(x):
    r̂_a ← ρ_ψ(a,x)                 # E[L_ep(T(b,a),Y)|x] 估计
    ĥ_a ← r̂_a - r̂_KEEP            # 或直接用 R 表
a_B3 ← argmin_a r̂_a                # 等价于 argmin ĥ_a
# S3 同控制器
```

用途：在有限菜单上落实 §1 生死公式；与候选并排跑 decision-equivalence。  
**DGP-4 上 B3 与候选同序 ⇒ 解析/经验等价 ⇒ NO-GO。**

### 基线对照表

| ID | 决策核 | 与候选共享 | 预期角色 |
|---|---|---|---|
| B0 | 恒等 | 损失定义 | 安全 |
| B1 | 点 Hurdle+RC | \(L_{ep}\) 评估、S3 思想 | 结构必要性 |
| B2 | 绝对 set+val+RC | 损失、K、S3、容量 | **最强工程还原** |
| B3 | \(\arg\min r_a\) 同菜单 | 菜单、OOF 表、S3 | **最强公式还原** |
| Cand | \(\arg\min s_\psi\approx H\) + S3 | — | 被测对象 |

---

## 5. Four synthetic DGPs + oracles

> 真实数据前**必须**先跑这四类（获 A/B 放行后）。本轮**只设计不跑**。  
> 公共：`H=24`，菜单有限可穷举，种子固定，Bayes oracle 用独立大样本或闭式条件期望。

对每个 \(x\)：

\[
a^\star(x)=\arg\min_{a\in A_B(x)}\mathbb E[L_{ep}(T(b,a),Y)\mid x],\quad A_B\ni\mathrm{KEEP}.
\]

并列时用预注册字典序。记录 \(r_a=\mathbb E[L_{ep}|x,a]\) 与 \(h_a=r_a-r_{\mathrm{KEEP}}\)。

### DGP-1 Fragment/Merge

**生成机制**

- 真事件：单一负价 episode \(E_Y=[6,11]\)，幅度 \(Y_t=-c\)（\(c>0\) 随机）。
- 基座：在 \(t=8\) 处置 \(b_8\ge 0\)（“桥”），使 `EPISODES(b)={[6,7],[9,11]}`。
- \(Z\)：含桥接指示与噪声特征；不含 \(Y\)。
- 菜单：`KEEP`；`MERGE_REPLACE[6,11]`（填桥为负）；`DELETE` 第二段；逐点修补动作（B1 风格，拆成多个 ±δ）。

**Bayes oracle**

- 在足够大的 \(w_f,w_b\) 下，\(a^\star=\mathrm{MERGE}\)；点级分别“修两段”可降点 MAE 但增加 fp/匹配代价。

**预期分离信号**

- B1 / 点排序：点损失↓ 或点召回↑，但 \(L_{ep}\)↑ 或 > MERGE。
- B2/B3/Cand：应选 MERGE（结构损失必要）。

**失败判据（对“结构必要性”）**

- 若 B1 在 \(L_{ep}\) 上已匹配 oracle → 本 DGP 参数太弱，重标定权重（`INVALID` 参数），不得宣称点级已够。
- 本 DGP **成功只说明非可加损失有用，不说明候选新。**

### DGP-2 Budget-K

**生成机制**

- 一日两真事件：A 短高损（如 `[3,4]` 深负），B 长中损（`[12,18]` 浅负）。
- \(K=1\)。
- 构造 \(Z\) 使逐点 \(P(y_t<0)\) 在 B 上更高/更长，但 episode 级 \(\mathbb E[H]\) 在“只修 A”更优（高 \(w_m\) 或 A 的 value 项）。
- 菜单：`KEEP`；`EDIT_A`；`EDIT_B`；禁止同时编辑（预算剪枝）。

**Bayes oracle**

- \(a^\star=\mathrm{EDIT\_A}\)（在预注册权重下）。

**预期分离信号**

- 逐点 top-K / B1 → `EDIT_B`。
- episode 菜单 + \(L_{ep}\)（B2/B3/Cand）→ `EDIT_A`。

**失败判据**

- 若 B2/B3 不能表达 `EDIT_A`（菜单不公）→ `INVALID`。
- 分离成立只证明**预算+episode 损失**相对点排序必要，**非**理论创新。

### DGP-3 Insert/Boundary

**生成机制**

- 子型 3a 完整漏报：`E_Y=[10,12]`，`EPISODES(b)=∅`。
- 子型 3b 边界偏移：`b` 预测 `[9,13]`，真 `[10,12]`。
- 子型 3c 无负价日：`Y≥0`，`b` 可能有假事件。
- 子型 3d 跨日截断：真事件跨午夜，按 24h 截断 + censor 标志。
- 菜单：`KEEP`；`INSERT[10,12]`（必须改 24-向量）；`REPLACE` 边界；`DELETE` 假事件。

**Bayes oracle**

- 3a: INSERT；3b: 收缩边界 REPLACE；3c: DELETE 或 KEEP；3d: 与截断约定一致的动作。

**预期分离信号**

- 动作必须 **真实改变** `final_24_vector`（断言 `max|ŷ-b|>0`）；旧原型式 `pass` 直接 `INVALID`。
- B2/B3 具备同等表达；本 DGP 检验实现完整性与边界损失饱和，不杀/不救新颖性。

**失败判据**

- INSERT 不改输出；无负价日乱 INSERT 且无 S3 可挡 → 实现/证书失败。

### DGP-4 Reduction/Equivalence（**最关键 · 杀新颖性**）

**生成机制**

- 混合 DGP-1..3 的 \((b,Z,Y)\) 与随机菜单（同一生成器对 Cand/B3）。
- 对每个 \(x\)：**穷举**有限菜单上每个 \(a\) 的真条件风险 \(r_a\)（闭式或 \(N_{\mathrm{oracle}}\ge 10^5\) Monte Carlo）。
- 候选决策：\(\arg\min_a \hat h_a\)（或真 \(h_a\) 的 oracle-scored 上界版本）。
- B3：\(\arg\min_a \hat r_a\)。
- 另跑“真表”版本：用真 \(r_a,h_a\) 消除拟合噪声，专测**解析**等价。

**Bayes oracle**

- 由生死公式：\(\arg\min h_a=\arg\min r_a\)（同一 \(A(x)\)）应**逐样本**成立。

**预期分离信号（存活所需）**

- 若仅存在于：**不同菜单、不同信息、不同 S3 族、或不可实现约束** —— 必须在文档中单列，且 A 可攻击。
- **默认预期：同菜单同信息下无决策分离。**

**失败判据 → NO-GO**

```text
解析：每个 x，argsort(h) == argsort(r)（tol 1e-12）且 argmin 相同
  → OBJECT-level decision equivalence → NO-GO

经验（拟合 scorer）：
  action agreement ≥ 99%
  AND max_t |ŷ_cand - ŷ_B3| < 1e-12
  AND accept/certificate agreement ≥ 99%
  AND paired Δ risk CI contains 0
  → empirical equivalence → NO-GO

仅优化噪声的 <1% 差异 → 不得计创新
```

**前三类成功 + 第四类等价 = 结构化工程有用但 A 会对象死亡。**

---

## 6. Decision-equivalence protocol

### 6.1 每样本不可变记录

```text
sample_id
method ∈ {Cand, B0, B1, B2, B3}
action_signature
accept_or_abstain          # S3 后是否部署非 identity
final_24_vector            # float64
risk_ranking               # 菜单内动作的分数序（完整 perm）
certificate_decision       # which π or Identity; UCB values
menu_id                    # hash of A(x)
budget_K
seed
```

### 6.2 汇总统计

| 指标 | 定义 |
|---|---|
| exact action agreement | `signature` 全等比例 |
| acceptance agreement | accept/abstain 一致率 |
| certificate agreement | 所选 π / Identity 一致率 |
| max output diff | \(\max_{d,t}|\hat y^{m1}_{d,t}-\hat y^{m2}_{d,t}|\) |
| L1 output diff | 日均 \(\|\hat y^{m1}-\hat y^{m2}\|_1\) |
| ranking agreement | Kendall-τ / top-1 / full perm match |
| paired risk CI | moving-block bootstrap on day-level \(L_{ep}\) or \(\Delta\) |

主对比对：`Cand vs B3`（公式）、`Cand vs B2`（工程）、参考 `Cand vs B1`、`Cand vs B0`。

### 6.3 裁决规则（Gate 3 前置）

1. **解析等价**（真 \(r,h\) 表）成立 → 立即 **NO-GO**（对象层）。
2. **经验等价**（§5 DGP-4 阈值）→ **NO-GO**。
3. 仅当菜单/信息/证书**不公**造成的表面差异 → `INVALID` 实验，重做公平版。
4. 稳定可复现分离且 A 承认强基线不能保持 → 提交主窗口，**仍不自动 GO**。

---

## 7. Phase C gate (only if A/B pass)

### 7.1 启动条件（全部满足）

- [ ] 主窗口书面验收：Agent A **未**对最强组合闭合 KILL（或仅 OBJECT KILL–ALGORITHM OPEN 且算法点仍开放）
- [ ] 主窗口书面验收：Agent B 存在**非定义性** T1，原子复算通过
- [ ] 本 35c 设计无 SPEC INVALID
- [ ] **禁止**在上述之前创建目录或跑真实数据

### 7.2 若获批的最小范围

| 项 | 规格 |
|---|---|
| 新目录 | `experiments/07-episode-relative-pilot/`（新建；不碰 archive/06） |
| 数据 | `LAGO_DE`、`NEM_SA1` only |
| 基座 | 冻结 **Linear**、**GBDT** → **4 设置** |
| 日 | 完整日历日；清理 23/25/缺失/重复小时后切 S1–S4=50/20/10/20 |
| 主预算 | K=1 主报；K=2 敏感 |
| 基线 | B0–B3 全跑 |
| 合成 | DGP-1..4 先于真实；DGP-4 非 NO-GO 才进真实 |
| 禁止 | 全矩阵 9×5、私有山东主实验、双市场联合、复活 TAP/GAFE/e-process/admin clamp |

### 7.3 成功门槛（Gate 3；仅“允许扩大研究”）

1. 合成 DGP-4：**不**出现解析/经验 decision-equivalence（相对 B3）。
2. 4 设置：**无**安全违例（证书承诺的 \(\epsilon,\alpha\) 在 S3 程序上被遵守；S4 不调参）。
3. 相对 **B2**：至少 **3/4** 设置主损失 \(L_{ep}\) 同向改善，且 **paired** moving-block CI 支持（Holm 后）。
4. 关键动作消融（§8）证明代码路径真实改动作，且去掉后性能/决策显著变差或越界。
5. 泄漏测试 bit-exact 通过。

任一失败：`NO-GO` / `PIVOT` / `INVALID`（按原因），**不得**靠加宽数据挽救。

---

## 8. Ablations and leakage tests

### 8.1 泄漏与不变量（自动化，失败则 INVALID）

```text
L1. 推理 API 签名无 Y / 未来价格
L2. S4 评估时将 Y 置零：action_signature 与 final_24_vector bit-exact 不变
L3. S4 将 Y 置乱（固定种子置换）：同上 bit-exact 不变
L4. Z 列与当日 y 的 max|corr| 审计（阈值预注册，建议 <0.3 同现 BECH 精神）
L5. 残差特征 shift≥24h；assert_no_leakage 建表级
L6. 跨日/菜单/损失权重配置哈希写入结果 JSON
```

### 8.2 消融（先做“路径探针”，再比指标）

| 消融 | 探针（必须先过） | 科学问题 |
|---|---|---|
| 点损失替换 \(L_{ep}\) | 训练目标切换后菜单排序变 | 非可加损失是否必要 |
| 绝对风险评分（→B3） | `s_ψ` 改为 `ρ_ψ` 后决策文件 diff | 相对 harm 是否多馀 |
| 去 INSERT | 菜单无 INSERT；3a 日向量不可达 oracle | INSERT 是否独立贡献 |
| 去 REPLACE | 边界子型动作消失 | 边界编辑是否独立 |
| 去 DELETE | 假事件日无法删 | DELETE 是否独立 |
| 去 value | REPLACE/INSERT 只用常数幅度 | value 头是否被 T 使用 |
| 去 S3 门 | 强制部署 S2 选主策略 | 证书是否约束 accept |

**路径探针：** 每个消融至少 1 个合成日 + 1 个真实校验日（若已放行）上 `action_signature` 或 `final_24_vector` 与全模型不同；否则消融未接线 → `INVALID`。

### 8.3 旧原型坑位检查清单（禁止复现）

- INSERT/SHIFT 不得为 `pass`
- 匹配必须真 Hungarian/线性分配，禁止冒充
- 禁止错误数据集键映射（旧 LAGO_DE/NEM_SA1 事故）
- 日样本必须严格日历日
- S3 必须参与选择/发证
- bootstrap 必须方法间配对（paired-day）

---

## 9. Metrics and statistics

### 9.1 指标

| 层级 | 指标 | 角色 |
|---|---|---|
| 主 | 日级有界 \(L_{ep}\) | Gate 3 主比较 |
| 主辅 | \(\Delta=L_{ep}(\hat y)-L_{ep}(b)\) | 相对基座；证书对象 |
| 结构 | 完整漏报率、假事件率、边界误差、幅值误差 | 与 P0 失败型对齐 |
| 点 | MAE / 正常期 harm | 检测“点好事件差” |
| 部署 | 触发率、fallback 率、cert 率、\|Π\|、有效块数 | 诚实报告 power |
| 等价 | §6 全套 | 新颖性生死 |

负价稀有市场：**禁止**对个位数事件做无加权跨市场算术平均（继承 AGENTS.md）。

### 9.2 统计

- **单位：** 完整日；配对：同日 Cand vs 基线。
- **推断：** moving-block paired-day bootstrap；**块长仅由 S2/S3 自相关预注册**（S4 前锁定）。
- **多重性：** 主比较（4 设置 × 主损失 vs B2）Holm；探索性指标不进 Gate 3。
- **证书：** 报告可计算 \(\epsilon(n,\alpha,|\Pi|)\)、有效样本/块数；无非 BASE 可证时 **fallback 率** 与 **power** 分列。
- **合成阶段：** 以 oracle 与等价审计为主，不把 p 值刷榜当新颖性。

---

## 10. GNN deferred note

依据 `37_跨领域文献调研_...`：GNN 特征交互综合约 5/10，与当前“轻量冻结基座后处理 + episode 对象生死”不匹配；边恢复质量存疑；时序后处理无强先例。

**本轮与 Phase C 试点：不实现 GNN。**  
附录可保留一句：

> “远期可选：在 cutoff-safe 特征上用共享图做特征交互增强；不改变本轮动作对象、损失与证书，且不构成本文主创新。”

禁止把 TAP/GAFE/“一张图三次消费”写回主线。

**Hurdle 对齐（实现可引用、非新定理）：** 评分器采用 occurrence×magnitude（Bi-Hurdle）与降水/保险/生态 Deep Hurdle、ZILN 文献对齐，定位为电价场景的条件 harm 参数化，**不**声称双损失理论首创。

---

## 11. Kill / Pass criteria checklist

### 11.1 设计期（本文件）

- [x] 方法自冻结对象导出
- [x] 默认预算 K 写清及理由
- [x] B0–B3 含 ABS-SET+VAL+RC 与 risk-table reduction
- [x] 四合成 DGP 含第 4 类等价杀伤
- [x] Decision-equivalence 字段与阈值
- [x] 明确 A/B 未过禁止真实 pilot
- [x] 不碰 archive/06；不改 src/data

### 11.2 主窗口串行（设计外）

- [ ] A：最强组合攻击，非点级稻草人
- [ ] B：T0=`KNOWN-FRAMEWORK`；T1 非定义性
- [ ] 公式碰撞：同信息/监督/动作/预算/损失下是否仍有分离

### 11.3 合成（仅 A/B 双过后）

- [ ] DGP-1..3：结构必要性信号（不作为创新证据）
- [ ] DGP-4：非解析/经验等价；否则 **NO-GO**
- [ ] 路径探针：INSERT/REPLACE/DELETE/value/S3 消融真改动作

### 11.4 真实 pilot Gate 3（仅合成通过后）

- [ ] 4 设置泄漏 bit-exact
- [ ] 4 设置无安全违例
- [ ] vs B2：≥3/4 主损失同向 + paired CI + Holm
- [ ] 消融可解释
- [ ] 仍不宣称 A 会创新成立——仅 `GO-to-next-stage`

### 11.5 一票否决 → NO-GO

- 同菜单 argmin 相对 harm ≡ 绝对风险（解析或 §6 经验阈值）
- 差异仅来自不公平菜单/容量/调参/泄漏
- T1 仅为换损失或标准 concentration 例化（B 侧）
- 强组合 B2 完整覆盖目标、推理与保证（A 侧）

### 11.6 INVALID / BLOCKED（非科学否定）

- 实现空动作、错误映射、S3 未发证、非配对 bootstrap、真值泄漏
- 修复后重跑；不得写成 STOP 科学结论（旧 06 覆辙）

---

## 12. What is NOT claimed

本设计**不**声称、且禁止在 35c 或后续草稿中偷渡为贡献：

1. 现有 BECH 主矩阵 / 消融 / 同行对照 = A 会方法创新  
2. P0 episode 审计 = 算法或定理成立  
3. Hungarian、连通分量、exact fallback、`<0` 阈值、预算按定义满足  
4. “直接学相对 harm”单独作为新 Bayes 决策（与 §0.4 冲突）  
5. T0 有限策略族 paired UCB/RCPS 例化 = 新理论  
6. Hurdle/Tweedie/ZILN 对齐 = 电价首创理论（仅定位与实现选择）  
7. GNN / TAP / GAFE / utility edge / e-process / admin clamp / A4 fixed point  
8. 双市场 DA→RT 联合预测  
9. 测试集经验零退化 = 高概率总体非退化  
10. 旧 `06-event-edit-prototype` 任何数值结论  
11. 本文件本身 = Phase C 已放行或创新存活  
12. Agent 投票或自报 COMPLETE 代替主窗口公式级验收  

---

## 13. 交付与下一步

| 项 | 状态 |
|---|---|
| 本文件 `docs/paper_prep/35c_AgentC_PhaseBC_falsification_plan.md` | **本轮完成** |
| 代码 / `experiments/07-*` | **禁止**，除非主窗口基于 35a+35b+35c 书面 `GO` 且 A/B 双过 |
| 总裁决 | 主窗口文档（建议 `37_PhaseB_commander_verdict.md` 或后续编号），非本 Agent |

**给主窗口的三行摘要**

1. **默认预算：** 主用 **K 个 episode**（主报 K=1，K=2 敏感）；L1 幅度仅事后敏感。  
2. **最强证伪点：** **DGP-4 + B3 risk-table**：同有限菜单上 \(\arg\min H\equiv\arg\min L_{ep}\)；解析或 ≥99% 一致且 max\|Δŷ\|<1e-12 且 CI∋0 → **NO-GO**。  
3. **Phase C：** 本设计**描述了**获批后的 07 范围，但**不授权**启动；**仅当 A 还原未闭合且 B 有非定义性 T1 并经主窗口验收**后，才允许新建 `experiments/07-episode-relative-pilot/` 并先跑合成再上 LAGO_DE/NEM_SA1 × Linear/GBDT。

---

*End of Agent C Round-1 deliverable. No code. No pilot.*
