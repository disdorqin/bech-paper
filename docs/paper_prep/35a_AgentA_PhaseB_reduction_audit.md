# 35a · Agent A：Phase B 最强还原红队审计

> 角色：Agent A（最强还原 / 不可约性红队）  
> 日期：2026-08-09  
> 项目：`solar_leak_price_model`  
> 范围：**仅文档审计**；不改 `src/` / `data/` / `experiments/01-05`；不跑真实 pilot；不写实现代码。  
> 权威输入：`36_新窗口交接_PhaseBC_2026-08-09.md`、`3子agent.md` §1/§2/§4、`34_PhaseA_...md`、`P0_DECISION.md`；Hurdle 背景见 `37_跨领域文献调研_...md`（仅对齐，不立创新）。  
> GNN：本轮不作为主创新；一句结论——**不改变下述还原结论**。

---

## 0. Object / Algorithm / Theory verdict

| 层 | 裁决 | 一句话 |
|---|---|---|
| **Object（学习对象）** | **KILL** | 固定 \(x=(b,Z)\) 下，相对 harm 的 Bayes 动作与绝对 \(L_{ep}\) 风险最小化**恒等**；编辑脚本在固定 \(b\) 下与绝对区间/轨迹输出**双射**（至 identity 纤维）。 |
| **Algorithm（可实现算法）** | **KILL（在公平菜单上）** | 同一有限动作菜单 + 同一预算解码 + 同一 S3 控制器时，**ABS-SET+VAL+RC** 与 **Risk-table reduction**（\(r_a-r_{\mathrm{id}}\)）与候选**决策等价**；差异只能来自不公平容量/菜单/调参。 |
| **Theory（保证）** | **KILL as novelty** | 成对相对风险的 RCPS/LTT/simultaneous UCB 是**已知框架例化**（T0）；不产生现有工具推不出的保证。 |
| **综合标签** | 见文末 | **`REDUCTION PASS/KILL`** |

**唯一曾被讨论的“存活性质”及处置：**

1. **“学相对 harm 标签”** → 被生死公式与 risk-table 消去，**不存活**。  
2. **“冻结数值基座 + edit + value + selective RC 四元组组合空白”** → Phase A 已声明仅为小集合组合差，**不是对象级不可约性**，**不存活为 A 会对象**。  
3. **Bi-Hurdle（occurrence×magnitude）** → 与 Tweedie/Hurdle/ZILN **对齐的成熟分解**；写入绝对 structured + value head 后**不改变 argmin**，**不存活**。  
4. **潜在算法开口（诚实降级）**：若强制**无限/连续动作空间**且拒绝有限菜单，候选与强基线都未给出严格复杂度分离——**无证明的 OPEN 不算存活**；本审计将其记为 **未证实、默认不放行**。

**结论预告：** 在协议冻结的有限菜单、同一 \(L_{ep}\)、同一预算与同一 S3 下，候选可被  
`ABS-SET+VAL+RC` 与 `Risk-table reduction` **完整还原**。建议主窗口 **NO-GO（转 Route-E）**；若需 PIVOT，必须**换对象**（而非修补 edit 叙事）。禁止复活 TAP/GAFE、e-process、admin clamp、A4 fixed point、双市场联合、旧 event-edit 原型。

---

## 1. Frozen specification

以下与 `3子agent.md` §2 及交接文档 §9 对齐；发现不自洽只报 `SPEC INVALID`，本审计未发现使还原失败的不自洽。

### 1.1 信息集与切分

| 符号 | 定义 |
|---|---|
| 审计单位 | 完整交付日，\(H=24\) |
| \(b=f_0(I)\in\mathbb R^{24}\) | S1 训练后**冻结**的数值基座输出 |
| \(Z=z(I,b)\) | cutoff-safe 特征；不得含当期/未来 \(Y\)；残差史 ≥24h 滞后 |
| \(Y\in\mathbb R^{24}\) | 真值；仅 S2 监督 / S3 标定 / S4 评估 |
| \(\pi(b,Z)\) | 推理可测；API 无 \(Y\) |
| 切分 | S1/S2/S3/S4 = 50%/20%/10%/20% rolling-origin |

跨日 episode：主方案按 24h **截断**并记 left/right censor；所有方法共用。

### 1.2 事件、动作、编辑器

\[
S(y)=\{t:y_t<0\},\qquad E(y)=\mathrm{CC}(S(y))
\]

- 零值 \(\neq\) 负价；缺失/重复小时先清洗。  
- 动作词表：\(\mathcal A_0=\{\mathrm{KEEP},\mathrm{DELETE},\mathrm{REPLACE},\mathrm{INSERT}\}\)；SHIFT/SCALE ⊂ REPLACE。  
- 编辑器 \(T(b,a)\)：\(T(b,\mathrm{KEEP})=b\)；重叠/越界/冲突/tie **确定**。  
- **主预算（二选一，本审计对两者分别还原）**  
  - (K) 最多编辑 \(K\) 个 episode 槽位；或  
  - (L1) \(\|T(b,a)-b\|_1\le B_{\mathrm{amp}}\)。

记合法动作集 \(\mathcal A(b;B)\subseteq\mathcal A_{\mathrm{menu}}\)（有限菜单 + 预算可行 + 含 identity）。

### 1.3 有界非可加事件损失

\[
L_{ep}(\hat y,Y)=w_m L_{\mathrm{miss}}+w_f L_{\mathrm{fp}}+w_b L_{\mathrm{boundary}}+w_v L_{\mathrm{value}}\in[0,1]
\]

- 真实/预测 episode 集经 **带 dummy 的线性分配（Hungarian）** 匹配。  
- 未匹配真 → miss；未匹配预测 → FP；匹配对 → 边界（+可选 value/轨迹）。  
- 权重、空集约定、tie 在看 S3/S4 前冻结。  
- **Hungarian + 非可加损失 = 问题定义，不是创新。**

### 1.4 相对 harm、风险与目标保证

\[
H(a;x,Y)=L_{ep}(T(b,a),Y)-L_{ep}(b,Y),\quad x=(b,Z)
\]

\[
R_{\mathrm{rel}}(\pi)=\mathbb E[H(\pi(x);x,Y)]
\]

希望：

\[
\Pr_{S3}\big\{R_{\mathrm{rel}}(\hat\pi)\le\epsilon \mid S2\big\}\ge 1-\alpha
\]

仅 \(\epsilon=0\) 可称高概率总体非退化。

### 1.5 冻结候选对象（被攻击句）

> 非可加 episode 损失下，相对冻结数值预测基座的负电价事件校正：  
> 从 \((b,Z)\) 选预算内动作 \(a\)，使 \(T(b,a)\) 的 \(L_{ep}\) 相对 \(b\) 改善，并由 S3 发成对相对风险证书。

### 1.6 生死公式（必须先成立）

因 \(L_{ep}(b,Y)\) 对 \(a\) 为常数，

\[
\boxed{
\arg\min_{a\in\mathcal A(b;B)}\mathbb E[H(a;x,Y)\mid x]
=
\arg\min_{a\in\mathcal A(b;B)}\mathbb E[L_{ep}(T(b,a),Y)\mid x]
}
\]

**推论（决策层）：** “相对 harm 学习”在条件期望最优意义下**不引入新 Bayes 动作**。任何声称“相对性改变最优编辑”的对象级叙事，在此信息集下**直接失败**（除非改变 \(\mathcal A\)、信息或损失——那已不是同一对象）。

### 1.7 Bi-Hurdle 锚点（非自动创新）

将负价写为 occurrence×magnitude（类 Hurdle/Tweedie/ZILN）只说明 **value 头与事件头分解合理**，可嵌入强基线的 \(q\)（集合/区间）+ \(r\)（幅度）。**不改变**上式 argmin，也不构成与 ABS-SET+VAL 的分离。

---

## 2. Strong composite baseline and pseudocode

### 2.1 名称与公平契约

**ABS-SET+VAL+RC**  
\[
h_\eta(b,Z)=D_B\big(q_\theta(\cdot\mid b,Z),\; r_\phi(b,Z,\cdot),\; b\big)
\]

公平性（与候选**逐项相同**）：

| 维度 | 要求 |
|---|---|
| 输入 | 必须含 \((b,Z)\)，同一 cutoff |
| 动作菜单 | 同一 \(\mathcal A_{\mathrm{menu}}\)（KEEP/DELETE/REPLACE/INSERT 参数化） |
| 预算 | 同一 \(K\) 或 \(B_{\mathrm{amp}}\)；identity 必在可行集 |
| 损失 | 同一 \(L_{ep}\) 与匹配规则 |
| 容量/调参 | 同级 proposal 数、同 S2 样本、同搜索预算 |
| S3 | 同一有限策略族、同一 paired selective risk control、同一 \(\epsilon,\alpha\) |
| 不可认证 | 返回 \(b\)（bit-exact identity） |

**不是稻草人：** 禁止用“逐点阈值分类 + 无约束残差”代替本基线。

### 2.2 组件语义

1. **\(q_\theta\)：绝对 interval/set / 动作结构化头**  
   - 直接对“预测负价 episode 集合”（或参数化编辑）打分；  
   - 允许 DETR 式 matching 监督、ActionFormer/Yuan 边界、CRF/HMM duration、一般 structured decoder（Phase A：这些**绝对**结构已成熟）。  
   - 因输入含 \(b\)，可把“相对 \(b\) 的编辑”编成**绝对目标集合**的函数（见 R1）。

2. **\(r_\phi\)：residual/value 重建**  
   - 在给定 support/区间上重建幅度或整段轨迹；  
   - Bi-Hurdle 式 P(event)×magnitude 是 \(q,r\) 的一种参数化，非新对象。

3. **\(D_B\)：constrained decoder**  
   - 投影到预算可行、非重叠、合法边界；  
   - **显式包含 identity**；  
   - 与候选共用 tie-break。

4. **RC：paired selective risk control on S3**  
   - 对预注册策略 \(\pi_j\) 估计成对风险 \(R_{\mathrm{rel}}(\pi_j)\) 的 UCB/RCPS/LTT 上界；  
   - 选 \(\widehat{\mathrm{UCB}}_j\le\epsilon\) 中效用最优者，否则 no-op。

### 2.3 伪代码：ABS-SET+VAL+RC

```text
# ----- S2: fit absolute structured + value on frozen b -----
for day in S2:
    b, Z, Y = load_day(day)          # b frozen from S1
    E_true = CC({t: Y[t] < 0})
    # absolute target: episode set + values (or edit script via R1 map)
    q.fit_step(b, Z, target=E_true)  # set/interval/action scores
    r.fit_step(b, Z, Y, supports=E_true)

# build finite menu generator M(b,Z) -> {a0=id, a1,...,aM} subset A_menu
# train optional scorer s(a;b,Z) ≈ E[L_ep(T(b,a),Y)|b,Z]   # ABSOLUTE risk

# ----- S3: paired selective control (same as candidate) -----
for each pre-registered policy π_j in Pi:   # includes identity
    for day in S3:
        a = π_j(b,Z)                 # uses q,r,D_B
        delta = L_ep(T(b,a),Y) - L_ep(b,Y)
    UCB_j = simultaneous_UCB(deltas_j; alpha, |Pi|)
π_hat = argmin utility among {j: UCB_j <= epsilon}, else identity

# ----- S4 / deploy -----
a = π_hat(b,Z)
return T(b,a)   # or b if identity
```

推理期核心（与候选同一菜单时）：

```text
function Infer_ABS(b, Z):
    menu = M(b, Z)                         # finite, contains id
    for a in menu:
        # absolute conditional risk (estimated)
        r_hat[a] = s_abs(a; b, Z)          # ≈ E[L_ep(T(b,a),Y)|x]
    a* = argmin_{a in menu ∩ feasible(B)} r_hat[a]
    return D_B(a*; b)                      # budget projection
```

### 2.4 Risk-table reduction（第二强还原，协议强制）

同一有限菜单 \(\{a_i\}_{i=0}^{M}\)，\(a_0=\mathrm{identity}\)：

\[
\hat r(a;x)\approx\mathbb E[L_{ep}(T(b,a),Y)\mid x],\qquad
\hat h(a;x)=\hat r(a;x)-\hat r(a_0;x)
\]

决策：

\[
a^\star(x)=\arg\min_a \hat h(a;x)=\arg\min_a \hat r(a;x)
\]

因减去的 \(\hat r(a_0;x)\) **与 \(a\) 无关**。故：

- 若 \(\hat r=\hat{\mathbb E}[L_{ep}\mid x]\) 与候选的 \(\widehat{\mathbb E}[H\mid x]\) 来自**同一回归目标的等价标签**（\(H=L_{ep}\circ T-L_{ep}(b,\cdot)\)），则 **排序、argmin、S3 上的成对均值** 全等价；  
- “相对表”相对“绝对表”**零决策增益**。

伪代码：

```text
function Infer_RiskTable(b, Z):
    menu = M(b, Z)   # SAME menu as candidate
    for a in menu:
        r_abs[a] = score_absolute(a, b, Z)
        # h_rel[a] = r_abs[a] - r_abs[id]   # optional; same argmin
    return argmin_a r_abs[a]
```

### 2.5 弱基线（仅用于反例对照，不作为还原对手）

- Point+Value+RC：逐点 P(neg)+幅度，无 episode 匹配解码。  
- 无预算 top-K 点修正。  
反例若**仅**击败弱基线而 ABS-SET+VAL+RC 成功 → **只证明点级弱，不证明候选新**。

---

## 3. R1–R4 reduction lemmas

约定：\(\mathcal A_{\mathrm{fin}}(x)\) 为协议下有限合法菜单；\(T(b,\cdot)\) 确定。区分：

- **Oracle 可表示**：存在映射使最优值/最优动作一致（杀 **object**）。  
- **已知可实现**：存在标准算法类在多项式/常用 decoder 下实现（杀 **algorithm novelty**）。

---

### R1 · 表示映射（动作输出 ↔ 绝对结构化输出）

**命题 R1（oracle 双射至 identity 纤维）。**  
固定 \(b\)。定义绝对输出空间

\[
\mathcal Y=\mathbb R^{24},\qquad
\mathcal E=\{\text{合法负价 episode 集合}+ \text{段上轨迹}\}
\]

对任意动作 \(a\)，令 \(\Phi(a):=T(b,a)\in\mathcal Y\)。则：

1. \(\Phi(\mathrm{KEEP})=b\)；  
2. 每个 \(a\in\mathcal A_{\mathrm{menu}}\) 产生唯一的 \(\hat y=\Phi(a)\) 及 \(E(\hat y)\)；  
3. 若限定动作语义为“对 \(b\) 的有限 edit 脚本”，则存在逆映射 \(\Psi_b(\hat y)=a\)（在 edit 规范形 / tie rule 下），使 \(T(b,\Psi_b(\hat y))=\hat y\)，且 \(\Psi_b(b)=\mathrm{KEEP}\)。

**证明概要。**  
编辑器按定义是 \(b\) 与离散 edit 的确定函数，故 \(\Phi\) 单值。  
规范形：将 \(\hat y\) 与 \(b\) 的差分解为互不重叠的区间手术（在 \(b_t\ge 0\) 处置负段 → INSERT/REPLACE；删除 \(b\) 上假负段 → DELETE；一致 → KEEP）。协议要求冲突规则确定 ⇒ 分解唯一 ⇒ \(\Psi_b\) 唯一。  
因此候选的“edit 脚本输出”与“绝对 24-向量 / episode 集 + 轨迹”在固定 \(b\) 下**信息等价**。

**与 Phase A 碰撞：** DETR/ActionFormer/Yuan 直接预测绝对 interval/set；LevT 相对初始序列编辑——此处 \(b\) 为数值向量，但 R1 表明 edit 仍是绝对 \(\hat y\) 的再参数化，**不是新输出对象**。

**Oracle：** 杀 object 层“输出对象新”。  
**可实现：** set prediction + value head + 由 \((\hat y,b)\) 提取 edit 均标准；**无算法新对象**。

**反例？** 若 \(T\) 非确定或随机 break tie——则 SPEC INVALID，而非创新。  
**结论：R1 成立 → 表示层可还原。**

---

### R2 · 目标映射（Bayes 决策 / Hurdle 分解）

**命题 R2.1（生死公式 · 决策等价）。**  
对任意固定 \(x=(b,Z)\) 与任意可行集 \(\mathcal A(x)\subseteq\mathcal A_{\mathrm{fin}}(x)\)，

\[
\arg\min_{a\in\mathcal A(x)}\mathbb E[H(a;x,Y)\mid x]
=
\arg\min_{a\in\mathcal A(x)}\mathbb E[L_{ep}(T(b,a),Y)\mid x].
\]

**证明。**  
\(H(a;x,Y)=L_{ep}(T(b,a),Y)-L_{ep}(b,Y)\)。条件期望下第二项 \(\mathbb E[L_{ep}(b,Y)\mid x]\) 与 \(a\) 无关，故不改变 argmin。□

**命题 R2.2（Risk-table 等价）。**  
若绝对风险表 \(r(a;x)=\mathbb E[L_{ep}(T(b,a),Y)\mid x]\)，则相对表 \(h(a;x)=r(a;x)-r(\mathrm{id};x)\) 与 \(r\) **同序**，argmin 相同。

**命题 R2.3（Hurdle / Bi-Hurdle 不改 argmin）。**  
设将 \(L_{ep}\) 或监督分解为 occurrence 损失 + magnitude 损失（权重固定），或训练使用复合似然 \(\ell=\ell_{\mathrm{occ}}+\ell_{\mathrm{mag}}\)，只要**最终决策仍是**在 \(\mathcal A(x)\) 上最小化 \(\mathbb E[L_{ep}(T(b,a),Y)\mid x]\)（或其次优代理的同一 argmin），则最优动作集不变。  
若代理损失改变 argmin，那是**代理偏差**问题，ABS-SET+VAL 可换同一代理——**公平比较下无独有优势**。

**命题 R2.4（S2 标签形式）。**  
候选用 \(H\) 作回归标签；强基线用 \(L_{ep}(T(b,a),Y)\)。因

\[
H(a;x,Y)-H(a';x,Y)=L_{ep}(T(b,a),Y)-L_{ep}(T(b,a'),Y),
\]

成对比较与 listwise 排序学习在同一菜单上**目标等价**。点式回归 \(\hat H\) vs \(\hat L\) 仅差加性函数 \(c(x)=\mathbb E[L_{ep}(b,Y)\mid x]\)；若模型对每 \(x\) 拟合同菜单，线性可辨模型下 argmin 一致。

**Oracle：** object 目标可还原为绝对 structured risk。  
**可实现：** 绝对风险评分器 = 候选相对评分器（减 identity 列）。  
**结论：R2 成立 → 目标层 KILL。**

---

### R3 · 动作/预算映射（constrained decoder）

**命题 R3（支持/边界/轨迹/预算/identity 保持）。**  
设候选推理为：在菜单上按 \(\hat H\) 选 \(a\)，再经预算投影 \(P_B\)。定义强基线 decoder \(D_B\)：

1. 从 \(q,r\) 生成**同一菜单**（或由绝对 \(\hat y\) 经 \(\Psi_b\) 转 edit）；  
2. 按 \(\hat r\)（绝对）排序；  
3. \(P_B\) 与候选相同（非重叠、\(K\) 或 L1、边界合法）；  
4. identity 始终候选。

则对一切 \(x\)，可行输出集相同：

\[
\{T(b,P_B(a)):a\in\mathcal A_{\mathrm{menu}}\}
=
\{D_B(q,r,b)\text{ 可产生的 }\hat y\}.
\]

在 R2 评分一致时，**选出的 \(\hat y\) 相同**（至 tie rule）。

**子情形：**

| 预算 | 候选 | ABS-SET+VAL+RC |
|---|---|---|
| \(K\) episode | 最多 \(K\) 次非 KEEP | 同一 cardinality 约束的 set decoder |
| L1 幅度 | \(\|\hat y-b\|_1\le B\) | 同一 L1 ball 投影 / 罚项 |
| identity | KEEP | \(\hat y=b\) 显式候选 |

**INSERT/DELETE/REPLACE：** 均为对绝对 \(E(\hat y)\) 与轨迹的手术；绝对 set 头直接生成 \(E(\hat y)\) 后填值，**覆盖**同一 support 族。

**Oracle + 可实现：** 在协议“有限菜单”下 **R3 成立**。  
若某人声称“连续无限 edit 空间 + 特殊搜索”才有优势——**双方都未给出严格复杂度定理**，不能记为存活算法贡献。

**结论：R3 成立 → 动作/预算层可还原。**

---

### R4 · 保证映射（S3 选择性风险控制）

**命题 R4（同族同证）。**  
设策略族 \(\Pi=\{\pi_j\}_{j=1}^{J}\) 在 S3 前冻结，日级损失有界 \([0,1]\)，成对

\[
\Delta_j=L_{ep}(T(b,\pi_j),Y)-L_{ep}(b,Y)\in[-1,1].
\]

在 exchangeability（A0）+ 有限类（A2）下，simultaneous UCB / RCPS / LTT 给出

\[
\Pr\big(\forall j:\; R_{\mathrm{rel}}(\pi_j)\le \mathrm{UCB}_j\big)\ge 1-\alpha.
\]

选 \(\hat\pi\in\{\pi_j:\mathrm{UCB}_j\le\epsilon\}\cup\{\mathrm{id}\}\) 的任意数据相关规则（含效用最大），则

\[
\Pr\{R_{\mathrm{rel}}(\hat\pi)\le\epsilon\}\ge 1-\alpha
\]

（identity 的 \(\Delta\equiv 0\)，可行集永非空）。  

**关键：** 该保证**只依赖** \(\{\Delta_j\}\) 的有界与同时效度，**不依赖** \(\pi_j\) 内部是“相对头”还是 “ABS-SET+VAL”。  
故候选的“相对风险证书”= 对同一 \(\Pi\) 的**标准成对风险控制**。

**与“绝对风险控制”：** 若对绝对风险 \(R(\pi)=\mathbb E[L_{ep}(T(b,\pi),Y)]\) 做 UCB，再与 \(R(\mathrm{id})\) 比较，因 \(R_{\mathrm{rel}}=R(\pi)-R(\mathrm{id})\)，在**同一同时界**技术下可平移为相对界；方差可能因配对变小——属 **paired concentration 常识**，标 `KNOWN-FRAMEWORK INSTANCE`，非新理论。

**Oracle：** 保证层无新命题。  
**可实现：** 双方共用同一 controller 代码路径即可 bit 级相同 accept/reject。  

**结论：R4 成立 → theory novelty KILL。**

---

### 3.5 Lemma 总表

| Lemma | Oracle 还原？ | 已知算法还原？ | 杀伤面 |
|---|---|---|---|
| R1 表示 | 是 | 是（set+value+规范 edit） | Object 输出 |
| R2 目标 | 是（恒等） | 是（risk table） | Object 决策 |
| R3 预算 | 是（同菜单） | 是（同 \(D_B\)） | Algorithm 推理 |
| R4 保证 | 是 | 是（RCPS/LTT/UCB） | Theory 证书 |

**四条全通过 ⇒ 公式级碰撞门 KILL。**

---

## 4. Counterexample ledger

记号：\(H=24\) 时点 \(0..23\)；负价用负值；预算默认 \(K=1\)（每例注明）；弱基线 = Point+Value；强 = ABS-SET+VAL+RC（同菜单）。  
**判定列：** 若强基线成功 → 该例**不支持**候选新颖性。

---

### CE1 · 桥接合并（点好、事件差）

| 项 | 值 |
|---|---|
| \(Y\) | 小时 3–5 = −10；7–9 = −10；其余 ≥0。两真实 episode。 |
| \(b\) | 3–9 全为 −1（桥接成单段假合并）或点状漏报 |
| \(Z\) | 可含日历；无 \(Y\) |
| 预算 | \(K=1\) |
| 候选 | REPLACE/DELETE 中间桥或拆段（若菜单含 split⊂REPLACE） |
| 弱基线 | 逐点改 6 附近 → 仍可能保持桥接或碎裂 |
| 强基线 | 绝对预测两个 interval + value；\(L_{ep}\) matching 罚合并 FP/边界 |
| **判定** | **强基线可处理** → 仅说明非可加 \(L_{ep}\) 必要，**对象可被 ABS-SET 吸收** |

---

### CE2 · 零阈值不连续

| 项 | 值 |
|---|---|
| \(Y\) | \(Y_5=-0.01,\; Y_6=+0.01,\; Y_7=-0.01\) → **两** episode（0 非负） |
| \(b\) | 5–7 全 −1 → 一 episode |
| 预算 | \(K=1\) |
| 候选 | 边界 REPLACE 打断 |
| 弱 | 连续阈值平滑易忽略 0 规则 |
| 强 | \(S(y)=\{y<0\}\) 硬编码进损失与解码 |
| **判定** | **规则性 / 定义性**；强基线同样嵌入阈值 → **不新** |

---

### CE3 · 预算 top-K 失败

| 项 | 值 |
|---|---|
| \(Y\) | epA: 2–4 深负；epB: 10–18 浅负长段；epC 假 |
| \(b\) | 漏 A；B 弱；C 假负段 |
| 预算 | \(K=1\) |
| 候选 | 选 INSERT A（\(L_{ep}\) 最优）而非删 C |
| 弱 | 按点置信 top-K 可能先削 C 的点 |
| 强 | 菜单上枚举：INSERT-A / DELETE-C / REPLACE-B，按 \(\mathbb E[L_{ep}]\) 选 |
| **判定** | **强基线同菜单枚举成功** → 预算结构属 structured decoding，**非候选独有** |

---

### CE4 · INSERT / DELETE / 边界

| 项 | 值 |
|---|---|
| \(Y\) | 12–15 = −20（完整事件）；\(b\) 全日 ≥0（**complete miss**） |
| 另例 \(b'\) | 11–16 = −5（边界外扩） |
| 预算 | \(K=1\) |
| 候选 | INSERT 或 REPLACE 边界 |
| 弱 | 点 residual 难“整段插入” |
| 强 | set head 直接提出 [12,15] + value；或 DELETE 假段 |
| **判定** | P0 问题真实，但 **ABS-SET+INSERT 菜单覆盖**；旧原型 INSERT=pass 是**实现失败**非理论缝 |

---

### CE5 · 同 support 不同 value

| 项 | 值 |
|---|---|
| \(Y\) | 4–8 = (−1,−50,−50,−50,−1) |
| \(b\) | 同 support 但是 (−1,−2,−2,−2,−1) |
| 预算 | L1 \(B_{\mathrm{amp}}\) 充足 |
| 候选 | REPLACE 幅度 |
| 弱/强 | value head \(r_\phi\) 足够 |
| **判定** | **纯 value 重建**；无结构新对象 |

---

### CE6 · 无负价日

| 项 | 值 |
|---|---|
| \(Y\) | 全日 ≥0；\(E(Y)=\emptyset\) |
| \(b\) | 可能有假负段 |
| 预算 | \(K=1\) |
| 最优 | DELETE 假段或 KEEP |
| 强 | 空集绝对预测 + FP 项；identity 安全 |
| **判定** | 强基线 + RC 弃权 **同样**；exact fallback **定义性** |

---

### CE7 · 跨日 episode

| 项 | 值 |
|---|---|
| \(Y\) | 日界前 21–23 与次日 0–2 本为连续，截断后两日各一段 + censor 标志 |
| \(b\) | 随意 |
| 预算 | 按日 \(K\) |
| 候选/强 | **同一截断协议** |
| **判定** | 协议层共用；**无分离**。改连续轴须预注册，仍双方共用。 |

---

### CE8 · S3 多策略选择偏差

| 项 | 值 |
|---|---|
| 设置 | \(\Pi\) 含激进 INSERT 与保守 identity |
| 现象 | S3 小样本 UCB 宽 → 常选 identity |
| 候选 | 相对 harm UCB |
| 强 | 同一 simultaneous UCB |
| **判定** | **同一统计现象**；不支持新保证。非 BASE power 不足是 **T0 已知局限**。 |

---

### CE9 · 长块依赖

| 项 | 值 |
|---|---|
| \(Y\) | 负价块长度依赖日前净负荷轨迹（弱依赖） |
| \(b\) | 系统性短预测 |
| 候选 | 声称需新 mixing 界 |
| 强 | 绝对 duration 头（Rabiner HSMM 类）+ 同一 block bootstrap 评估 |
| **判定** | duration 建模 **Phase A 已成熟**；新 weak-dependence 界属 Agent B T1 候选，**不是 edit-relative 对象**给出的免费定理。Agent A：**对象仍可还原**；理论开口若存在也**不绑定** base-relative harm。 |

---

### CE10 · 恒等策略空洞安全

| 项 | 值 |
|---|---|
| 设置 | 所有非 id 策略 S3 UCB > \(\epsilon\) |
| 行为 | 必须返回 \(b\) |
| 候选/强 | 相同 |
| **判定** | no-op 可行集非空 = **定义性安全**，非创新（协议 §5.5 已排除） |

---

### 4.1 反例总判

| 例 | 弱基线是否失败 | 强基线是否失败 | 对候选新颖性 |
|---|---|---|---|
| CE1 桥接 | 可失败 | **否** | 无 |
| CE2 零阈值 | 可失败 | **否** | 无 |
| CE3 top-K | 可失败 | **否** | 无 |
| CE4 INS/DEL/边界 | 常失败 | **否** | 无（问题锚点≠算法） |
| CE5 value | 否 | **否** | 无 |
| CE6 无负价 | 否 | **否** | 无 |
| CE7 跨日 | — | **否** | 无 |
| CE8 S3 偏差 | — | 相同 | 无 |
| CE9 长块 | 点级可失败 | 结构可表达 | 无对象分离 |
| CE10 identity | — | 相同 | 无 |

**Ledger 结论：不存在“强基线失败而候选 Bayes/同菜单最优成功”的协议内反例。**  
前三类成功只证明 **structured \(L_{ep}\) 优于点级**——而 structured 已并入 ABS-SET+VAL+RC。

---

## 5. Guarantee audit

### 5.1 候选声称的保证

\[
\Pr_{S3}\{R_{\mathrm{rel}}(\hat\pi)\le\epsilon\mid S2\}\ge 1-\alpha
\]

### 5.2 强基线可复制性

| 保证要素 | 候选 | ABS-SET+VAL+RC | 差异 |
|---|---|---|---|
| 有界日损失 | 是 | 是 | 无 |
| 成对 \(\Delta\) | 相对 \(b\) | 相对 \(b\)（同一） | 无 |
| 有限 \(\Pi\) | 要 | 要 | 无 |
| simultaneous validity | RCPS/LTT/UCB | 同 | 无 |
| identity fallback | 是 | 是 | 无 |
| \(\epsilon=0\) power | 依赖 S3 与 \(\Pi\) | 同 | 无 |

### 5.3 不能声称的“理论增量”

- exact fallback / bit-exact 弃权  
- Hungarian / CC / \(y<0\)  
- 预算约束按定义满足  
- Hoeffding/Bernstein/union bound / 标准 blocking  
- 换电价损失的 RCPS/LTT  
- 经验测试集零退化  

### 5.4 与绝对证书的关系

控制 \(R(\pi)\) 与 \(R(\mathrm{id})\) 的差 ≡ 控制 \(R_{\mathrm{rel}}\)。配对可减方差：

\[
\mathrm{Var}(\Delta)\le 2\mathrm{Var}(L_{ep}\circ\pi)+2\mathrm{Var}(L_{ep}\circ\mathrm{id})
\]

且常 \({\ll}\) 分列界——**经典配对技巧**，标 **KNOWN**，不可写“相对 harm 理论”。

### 5.5 保证层裁决

**Theory KILL（as novelty）。** 可证明的是 T0 类正确性基线，不是候选的独立理论贡献。

---

## 6. Formula-level collision table

| # | 候选公式/组件 | 强还原对应 | 碰撞结果 |
|---|---|---|---|
| F1 | \(H=L_{ep}(T(b,a),Y)-L_{ep}(b,Y)\) | \(L_{ep}(T(b,a),Y)\) 绝对风险 | **argmin 恒等** |
| F2 | \(\pi\in\arg\min\mathbb E[H\mid x]\) | \(\arg\min\mathbb E[L_{ep}\circ T\mid x]\) | **决策等价** |
| F3 | edit script \(a\) | \(\hat y=T(b,a)\) 绝对输出 / episode set | **R1 双射** |
| F4 | \(E(y)=\mathrm{CC}(\{y<0\})\) | 同一定义嵌入 ABS-SET | **定义共享** |
| F5 | matching + \(L_{\mathrm{miss/fp/b}}\) | Ciliberto 式 structured loss；ActionFormer/Yuan 边界 | **成熟** |
| F6 | INSERT/DELETE/REPLACE | 绝对 set 的增删改；LevT 编辑类比 | **成熟组合** |
| F7 | value/轨迹头 | residual reconstruction；Hurdle magnitude | **成熟** |
| F8 | Bi-Hurdle occ×mag | \(q\) 分类 + \(r\) 回归 | **对齐非创新** |
| F9 | 预算 \(K\) / L1 | constrained decoder \(D_B\) | **共享** |
| F10 | \(T(b,\mathrm{KEEP})=b\) | identity 候选 | **定义性** |
| F11 | \(R_{\mathrm{rel}}\) S3 证书 | paired RCPS/LTT/UCB on \(\Delta\) | **KNOWN 框架** |
| F12 | 模型无关外挂 | 后处理 \(h(b,Z)\)；ABS 同样只吃 \((b,Z)\) | **无分离** |
| F13 | 非可加 episode 损失 | 一般 structured risk（Ciliberto） | **单此不新** |
| F14 | “相对冻结基座”叙事 | 输入含 \(b\) 的绝对预测；risk table 减 \(r_{\mathrm{id}}\) | **叙事≠对象** |

**Formula collision gate：KILL。**  
同信息、监督菜单、动作、预算、损失、假设下，强组合保持 **目标、推理、保证**。

---

## 7. Exact kill proof or one surviving property

### 7.1 Exact kill proof（对象 + 公平算法 + 证书）

**设定（协议内）：**  
有限菜单 \(\mathcal A_{\mathrm{fin}}(x)\)，含 identity；同一 \(T,L_{ep},B\)；评分一致（或等价标签）；同一 \(\Pi\) 与 S3 控制器。

**步骤：**

1. **（决策）** 由 R2.1，  
   \(\arg\min_a\mathbb E[H\mid x]=\arg\min_a\mathbb E[L_{ep}\circ T\mid x]\)。  
2. **（表）** 由 R2.2，相对 risk table 与绝对 risk table 同 argmin。  
3. **（表示）** 由 R1，最优 \(a^\star\) 与最优绝对 \(\hat y^\star=T(b,a^\star)\) 一一对应。  
4. **（实现）** ABS-SET+VAL+RC 在菜单上最小化 \(\hat r(a;x)\) 并经同一 \(D_B\)，故推理输出 \(\hat y\) 与候选一致（至 tie）。  
5. **（证书）** 由 R4，S3 对 \(\Delta\) 的任何同时界对双方策略族给出**相同** \(R_{\mathrm{rel}}\) 保证；accept/reject 可一致。  

**因此：**  
不存在协议内“候选最优 / 候选可证，而 ABS-SET+VAL+RC + Risk-table 不可”的性质。

\[
\boxed{\text{Object KILL + Algorithm KILL（公平有限菜单）+ Theory novelty KILL}}
\]

### 7.2 显式否定的伪存活点

| 伪存活点 | 否决理由 |
|---|---|
| 相对 harm 改变 Bayes 动作 | R2.1 反例：恒不改变 |
| 四元组文献空白 | Phase A：非全球空白；组合差≠不可约对象 |
| episode 非可加损失 | structured loss 成熟 |
| INSERT 完整漏报 | ABS-SET 菜单动作 |
| 高概率零退化 | 标准成对选择 + no-op |
| Hurdle 分解 | 不改 argmin；跨域已有 |
| GNN 特征交互 | **不改变还原结论**；本轮非主创新 |

### 7.3 唯一“开口”为何不计入存活

**开口 O1（算法/连续空间）：** 无有限菜单时，双方都需搜索；无证明显示 base-relative 参数化降低复杂度或样本复杂度。  
**开口 O2（B-T1 弱依赖新界）：** 若存在，属于**通用成对风险 + 时间依赖**，不依赖“edit-relative 对象”；且须 Agent B 独立证明，**不能**救援本对象的 R1–R3 死亡。  

→ **无合格 surviving property 绑定本候选对象。**

### 7.4 与 P0 问题锚点的关系

P0：**problem anchor GO**（负价 episode、完整漏报）仍然成立。  
本文件：**solution object NO-GO**。  
二者不矛盾——真实问题可用 **Route-E：冻结基座 + 绝对结构化后处理 + 标准成对 RC** 诚实书写，而无需声称 base-relative episode edit 为 A 会新对象。

---

## 8. Next action

| 对象 | 建议 |
|---|---|
| 主窗口 | 交叉复算 F1/R2.1 与 R4；验收本文件是否攻击 **ABS-SET+VAL+RC** 与 **risk-table**（是）。 |
| Agent B | 若仍写 T1：须**脱离**“相对 harm 改决策”；仅可能在独立弱依赖/样本复杂度；默认预期 **T0=KNOWN，T1 高风险 KILL**。 |
| Agent C | **禁止**真实 pilot；合成 DGP-4 decision-equivalence 预期 **等价 → NO-GO 确认**。 |
| 实验目录 | **不得**创建 `experiments/07-...` 直至主窗口推翻本还原（预期不推翻）。 |
| 产品线 | **Route-E 应用论文**可继续用现有 BECH 证据；不把 Phase B 候选写进贡献。 |
| PIVOT 方向（若用户要） | 必须**新对象**，例如：制度性事件定义、市场规则约束的可证安全层、或与预测无关的机制问题——**不是**再包装 edit/relative/Hurdle。 |
| 禁止 | TAP/GAFE、e-process、admin clamp、A4、双市场联合、旧 06 原型修补、GNN 主创新叙事。 |

---

## 终局标签

**`REDUCTION PASS/KILL`**

（释义：Agent A 的还原攻击**成功**；候选在 object / 公平 algorithm / theory-novelty 三层被 **KILL**。主窗口侧对应科学裁决倾向：**NO-GO** → Route-E；非 `REDUCTION FAIL/PROCEED`。）

---

### 交付自检

- [x] 结构 0–8 完整  
- [x] 攻击 ABS-SET+VAL+RC + 伪代码  
- [x] Risk-table reduction 决策等价  
- [x] R1–R4 逐条证明/边界  
- [x] 反例账本 ≥10，含强/弱基线  
- [x] 保证审计  
- [x] 公式碰撞表  
- [x] Exact kill；无合格存活性质  
- [x] 单一终局标签  
- [x] 未改代码/数据/旧实验；未跑 pilot  
