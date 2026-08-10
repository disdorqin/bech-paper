# 35b Agent B — Phase B Theorem Sheet

> 角色：Agent B（理论建模者）  
> 日期：2026-08-09  
> 对象：非可加 episode 损失下，相对冻结数值基座的负电价事件校正  
> 约束：只写理论文档；不改代码/数据；不跑 pilot；不复活 e-process/TAP/GAFE/admin clamp；GNN 不进主定理  
> 权威依据：`3子agent.md` §1–2, §5；`36_新窗口交接`；`34_PhaseA`；`37_跨领域文献`；`P0_DECISION.md`

---

## 0. 总裁决：`NOT JUSTIFIED`（T1 严格优势不成立）；T0 单独为 `PROVABLE AS STATED` 且 `KNOWN-FRAMEWORK INSTANCE`

| 层级 | 裁决 | 含义 |
|---|---|---|
| **T0** | **PROVABLE AS STATED** | 有限冻结策略族上 simultaneous paired-risk UCB + no-op 回退，在 A0+A2+A3 下可证 \(\Pr\{R_{rel}(\hat\pi)\le\epsilon\}\ge 1-\alpha\) |
| **T0 创新性** | **KNOWN-FRAMEWORK INSTANCE** | 标准有限类 + Hoeffding/Bernstein + union bound + 选择性接受；换 loss 例化，**不算创新** |
| **T1（唯一尝试方向）** | **NOT JUSTIFIED** | 首选“精确基座锚点下 paired episode harm 相对最强合法 absolute 风险控制的严格有限样本半径/样本复杂度优势”**证不出**；在最强 absolute 组合被允许使用同一 S3 配对日时，相对风险与绝对风险差是 **identity**，估计半径无严格优势 |
| **T1 弱化版** | **AFTER WEAKENING → 仍非创新** | 可陈述“相对未配对 dual-mean 估计，配对 \(\Delta\) 在正相关时 Bernstein 半径更小”，但这是 **经典配对检验/paired concentration 常识**，不是 new lemma |
| **候选理论总体** | **KILL（A 会理论贡献）/ 默认转 Route-E；仅问题锚点保留** | T0 正确但已知；生死公式下 Bayes 动作被 absolute structured risk 还原；无合格非定义性 T1 |
| **PIVOT 出口** | 见 §10 | 若坚持理论线，只能换对象（例如不可实现 absolute oracle 下的计算分离，或预注册生成模型下的 episode 依赖下界），且须新开一轮冻结规格 |

**一句话结论：**  
在冻结对象与 A0+A2+A3 下，**可证明的是已知框架的配对风险选择（T0）**；**证不出**“相对标签”带来超越最强 absolute+paired 控制的严格有限样本优势（T1）；因此 **Phase B 理论门（Gate 2）对 A 会创新应判 FAIL**，诚实出口为 **NO-GO（理论）+ 保留 Route-E 应用**，或 **PIVOT 到明确新对象**（非本 sheet 可硬编的 T1）。

---

## 1. Formal object and information set

### 1.1 审计单位与信息集

- 审计单位：完整交付日，\(H=24\)。  
- 日前/实时分离建模；本 sheet 对单一市场、单一 cutoff 陈述。  
- 输入信息集（推理可测）：
  - 冻结基座输出 \(b=f_0(I)\in\mathbb R^{H}\)，\(f_0\) 在 S1 训练后冻结；
  - 安全特征 \(Z=z(I,b)\)，仅含 cutoff 前信息与 \(b\) 的确定函数；
  - **推理不可见**真值 \(Y\in\mathbb R^{H}\)。
- 协议：S1/S2/S3/S4 = 50%/20%/10%/20% rolling-origin。  
  - S1：训并冻结 \(f_0\)；  
  - S2：学提案/评分（本 sheet 的 T0/T1 **不依赖** S2 内部算法，只要求 S3 前策略族已冻结）；  
  - S3：仅做预注册有限策略选择/发证；  
  - S4：只测试。  
- 概率来源（T0/T1 主 sheet）：**S3 日样本**在条件于 S2（及 S1）后，与目标部署分布 exchangeable（见 A0）。不在本 sheet 混入 weak dependence。

### 1.2 Episode、动作、编辑器

\[
S(y)=\{t\in\{1,\ldots,H\}: y_t < 0\},\qquad
E(y)=\mathrm{CC}(S(y)),
\]

其中 \(\mathrm{CC}\) 返回最大连续区间族（连通分量）。约定：\(y_t=0\) 非负价；缺失/重复小时在切日前已清洗。跨日 episode：主方案按 24h 截断并打 left/right censor 标志（定义性处理，非定理）。

**动作词表**（有限符号，可组合为日级动作）：

\[
\mathcal A_0=\{\mathrm{KEEP},\mathrm{DELETE},\mathrm{REPLACE},\mathrm{INSERT}\}.
\]

\(\mathrm{SHIFT}/\mathrm{SCALE}\) 仅作 \(\mathrm{REPLACE}\) 参数化特例，不另开符号。

**编辑器** \(T(b,a)\)：

- \(T(b,\mathrm{KEEP})=b\)（identity / exact no-action）；  
- 重叠、越界、冲突、tie-break **确定可测**（A3）；  
- 主预算只取一种（本 sheet 默认 **事件数预算**）：日动作 \(a\) 至多编辑 \(K\) 个 episode；  
  备选幅度预算 \(\|T(b,a)-b\|_1\le B_{\mathrm{amp}}\) 不进入主定理，仅作敏感性对象。

记策略 \(\pi\) 为从 \((b,Z)\) 到允许动作的可测映射；有限策略族 \(\Pi=\{\pi_1,\ldots,\pi_M\}\)，**在查看 S3 前冻结**（A0/A2）。

### 1.3 有界非可加事件损失

\[
L_{\mathrm{ep}}(\hat y,Y)
=
w_m L_{\mathrm{miss}}+w_f L_{\mathrm{fp}}+w_b L_{\mathrm{boundary}}+w_v L_{\mathrm{value}}
\in[0,1],
\]

权重与各分量归一化、空集约定、Hungarian/线性分配 matching、tie rule 均在 S3/S4 前固定（A3）。  

**明确排除：** \(L_{\mathrm{ep}}\) 的 matching/Hungarian/连通分量/`<0` 阈值/预算按定义满足 —— **定义，非贡献**。

### 1.4 相对 harm、相对风险、目标保证

对日样本 \(D=(I,Y)\)，\(b=f_0(I)\)，\(Z=z(I,b)\)：

\[
\Delta_\pi(D)
:=
L_{\mathrm{ep}}\big(T(b,\pi(b,Z)),Y\big)
-
L_{\mathrm{ep}}(b,Y),
\]

\[
R_{\mathrm{rel}}(\pi):=\mathbb E[\Delta_\pi(D)].
\]

因 \(L_{\mathrm{ep}}\in[0,1]\)，有 \(\Delta_\pi(D)\in[-1,1]\)。

**目标保证（总体、高概率、容差 \(\epsilon\)）：**

\[
\Pr_{\mathrm{S3}}
\big\{\,
R_{\mathrm{rel}}(\hat\pi)\le \epsilon
\ \big|\ \mathrm{S2}
\,\big\}
\ge 1-\alpha.
\]

仅当 \(\epsilon=0\) 可称“高概率总体非退化”；否则称 \(\epsilon\)-non-degradation。  
**不是：** 逐实例不劣、S4 经验零退化、事后报警。

### 1.5 生死公式（Bayes 层，identity）

对固定 \(x=(b,Z)\) 与动作 \(a\)：

\[
H(a;x,Y)
=
L_{\mathrm{ep}}(T(b,a),Y)-L_{\mathrm{ep}}(b,Y).
\]

第二项对 \(a\) 为常数，故

\begin{align*}
\arg\min_a \mathbb E[H(a;x,Y)\mid x]
&=
\arg\min_a \mathbb E[L_{\mathrm{ep}}(T(b,a),Y)\mid x].
\tag{ID-BAYES}
\end{align*}

**标记：** `identity`（条件期望线性 + 对 \(a\) 加性常数不影响 argmin）。  

**推论：** “直接学习相对 harm”**不**产生新的 Bayes 动作；在相同动作类与信息下，可被 **absolute structured risk minimization** 还原。  
`base-relative` 若有独立价值，只能来自：(i) 决策分离以外的算法/复杂度；(ii) 有限样本证书半径；(iii) 现有框架推不出的保证。本 sheet 的 T1 只攻击 (ii)。

### 1.6 Bi-Hurdle 形式化（架构锚点；非本 sheet 主定理）

> 目的：把 BECH/校正头的 “occurrence × magnitude” 与跨领域 Hurdle/Tweedie/ZILN **对齐为已知分解**，并标明电价双向 episode 的 **潜在 gap**。  
> **禁止声称**“全球首次双分量”。**对齐文献 ≠ 新定理**。

#### 1.6.1 为何 occurrence 与 magnitude 分治

负价日轨迹在“修正需求”坐标上近似 **半连续**：多数小时无需事件级编辑（零修正），一旦发生则幅度/边界连续且重尾。对半连续 DGP，单分量 MSE/MAE 将大量质量放在非事件区，稀释事件条件拟合——这是 Hurdle / zero-inflated / Tweedie（compound Poisson–Gamma）文献的标准动机（见 `37_跨领域文献调研`：降水 Tweedie、保险 ZILN、生态 Deep Hurdle）。

形式化（单尾、点级示意；episode 级是其结构提升而非新似然族）：

\[
Y^\star
=
O\cdot M,
\quad
O\in\{0,1\},
\quad
M\in\mathbb R_{-} \ \text{（负价幅度，条件于 }O=1\text{）}.
\]

双分量学习目标（示意）：

\[
\mathcal L
=
\ell_{\mathrm{occ}}(O,\hat p(Z))
+
O\cdot \ell_{\mathrm{mag}}(M,\hat m(Z)).
\]

**Occurrence** 决定是否触发编辑；**magnitude** 只在事件发生时监督连续修正。这解释 BECH 既有“分类头 + 条件幅度头”，而非宣称新损失公理。

#### 1.6.2 冻结基座下的可校准性

在 \(b\) 冻结、\(Z\) cutoff-safe 时：

- \(\hat p(x)=\widehat{\Pr}(O=1\mid x)\) 可在 S2 用滞后标签训练，在 S3 用 **有界** 日级 \(\Delta_\pi\) 做 **策略级** 校准/选择（T0），而不是对 \(\hat p\) 本身声称分布无关逐点校准定理；  
- 条件幅度头 \(\hat m\) 只影响 \(T(b,a)\) 的数值轨迹，最终证书仍落在 **标量** \(\Delta_\pi\in[-1,1]\) 上——这是可校准性的真正落点（有界 + 有限 \(\Pi\)）。  
- **Known：** 有界损失上的有限类 risk control（RCPS/LTT/CSA 族）可套用；  
- **Not claimed：** 对 Bi-Hurdle 内部 \(\hat p,\hat m\) 的 joint calibration 新定理。

#### 1.6.3 与单分量绝对损失的关系

- 若最终决策准则是 \(\arg\min_a \mathbb E[L_{\mathrm{ep}}(T(b,a),Y)\mid x]\)，则 Bi-Hurdle 只是 **构造 \(a\) / 构造 \(T\)** 的提案机制，不改变 (ID-BAYES)。  
- 若训练用 \(\ell_{\mathrm{occ}}+\ell_{\mathrm{mag}}\) 而评估用 \(L_{\mathrm{ep}}\)，存在 **surrogate–target 间隙**；该间隙是一般 structured prediction 已知问题（Ciliberto 等），**不是**本项目可独占的 T1。  
- 单分量绝对点损（MAE）最优 **不**蕴含 episode \(L_{\mathrm{ep}}\) 最优——这是 **问题性质**（对点级稻草人），一般 absolute interval/set predictor 仍可直接优化 \(L_{\mathrm{ep}}\)。

#### 1.6.4 Known vs 潜在 gap

| 项目 | 归类 | 说明 |
|---|---|---|
| Hurdle / two-part（occ+mag） | **KNOWN** | 经典计量与深度实现（Deep Hurdle、ZILN 双网等） |
| Tweedie deviance 优于 RMSE（半连续） | **KNOWN** | 降水等域多次验证 |
| 零膨胀 + 重尾幅度 | **KNOWN** | 保险 LTV/ZILN 等 |
| 电价 **双向**（负价 + 尖峰）双 hurdle | **应用扩展 / 弱 gap** | 文献以单侧零膨胀为主；双向是自然积，**不等于**新定理 |
| episode 非可加 \(L_{\mathrm{ep}}\) + 冻结基座编辑 | **组合对象** | Phase A：八篇内无完整四元组，**≠ 全球空白** |
| 将 Bi-Hurdle 称为理论创新 | **禁止** | 本 sheet 明确降级为架构锚点与文献定位 |

**双向扩展（仅定义，不进 T0/T1）：**  
\(O^{-}=\mathbf 1\{\exists t:Y_t<0\}\)，\(O^{+}=\mathbf 1\{\exists t:Y_t>\tau_{\mathrm{spike}}\}\)（\(\tau_{\mathrm{spike}}\) 须 cutoff-safe 预注册）。  
尖峰拓扑在 `P0_DECISION` 中因 S1-p99 漂移被警告；**主定理仅绑定负价** \(O^{-}\)。尖峰最多作为独立安全弃权通道，不进入本 sheet 证明。

---

## 2. Assumptions A0–A4

### 2.1 本 sheet 启用（起步集）

**A0（exchangeable 标定）**  
1. 策略/候选网格 \(\Pi\) 在进入 S3 前冻结（可用 S1/S2，但不得用 S3/S4 标签改 \(\Pi\)）。  
2. 日级损失有界：\(L_{\mathrm{ep}}\in[0,1]\) \(\Rightarrow\) \(\Delta_\pi\in[-1,1]\)。  
3. 条件于 S2（及 S1），S3 的日样本 \(D_1,\ldots,D_n\) 与目标部署日 **i.i.d. 或至少 exchangeable**，且 \(\Delta_\pi(D_i)\) 同分布。  
4. 目标风险 \(R_{\mathrm{rel}}\) 的期望与 S3 同分布。

**A2（有限策略类 + 同时有效）**  
1. \(M=|\Pi|<\infty\) 已知。  
2. S3 上任何数据依赖的选择 \(\hat\pi\) 必须通过 **simultaneous** 界（union bound 或等价 multiplicity 控制）以保持 \(1-\alpha\)。

**A3（确定可测）**  
Episode 提取 \(E(\cdot)\)、matching、\(L_{\mathrm{ep}}\)、预算投影、tie rule、编辑器 \(T\) 均为确定可测映射；无额外随机性（若有算法噪声，须并入策略索引使 \(\Pi\) 仍有限预注册）。

### 2.2 本 sheet 明确不用（禁止混写）

**A1（弱依赖）** — **不用。**  
若未来启用，必须 **另开 theorem sheet**，声明 \(\alpha/\beta\)-mixing 系数、block length、coupling error、有效块数 \(n_{\mathrm{eff}}\)，并重写 \(\epsilon(n_{\mathrm{eff}},\alpha,M)\)。  
**禁止** 在 exchangeability 证明里插入 “by blocking” 跳步。

**A4（漂移）** — **不用。**  
无 density ratio / 局部平稳 / drift budget 时，**不声称**跨制度区间安全。制度漂移只作为 Route-E 实证动机，不进 T0。

### 2.3 量词与条件化（全局）

- 概率 \(\Pr_{\mathrm{S3}}(\cdot\mid \mathrm{S2})\)：S2 固定后，仅 S3 样本随机。  
- \(f_0\)、\(\Pi\)、损失权重、\(K\)、\(\epsilon\) 名义值、\(\alpha\) 均在 S3 前固定。  
- 风险分母：按 **日** 期望（每个 S3 日一个 \(\Delta_\pi\)），不是按小时平均冒充日级保证。

---

## 3. T0 statement and proof

### 3.1 元数据

| 项 | 内容 |
|---|---|
| 名称 | T0 — Simultaneous paired-risk UCB selection with no-op |
| 创新标记 | **`KNOWN-FRAMEWORK INSTANCE`** |
| 假设 | A0+A2+A3 |
| 概率来源 | S3 exchangeable/i.i.d. days \| S2 |
| 风险 | \(R_{\mathrm{rel}}(\pi)=\mathbb E[\Delta_\pi]\)，\(\Delta_\pi\in[-1,1]\) |
| 策略类 | 有限 \(\Pi\)，\(M=\|\Pi\|\)，S3 前冻结 |
| 非 BASE power | **不保证**；no-op 只保证可行集非空 |

### 3.2 可计算统计量

对每个 \(\pi\in\Pi\)，S3 样本 \(i=1,\ldots,n\)：

\[
\hat\mu_n(\pi)
:=
\frac1n\sum_{i=1}^n \Delta_\pi(D_i).
\]

**Hoeffding 型同时 UCB**（主陈述；Bernstein 变体见 §3.6，仍为 known）：

\[
\mathrm{UCB}_n(\pi)
:=
\hat\mu_n(\pi)
+
\sqrt{\frac{\log(2M/\alpha)}{2n}}.
\tag{T0-UCB}
\]

（因 \(\Delta\in[-1,1]\)，range \(=2\)，Hoeffding 半径为 \(\sqrt{\frac{2\log(2M/\alpha)}{2n}}=\sqrt{\frac{\log(2M/\alpha)}{2n}}\) 的标准形式：对 \([0,1]\) 变量半径 \(\sqrt{\log/(2n)}\)，对映射 \(\frac{\Delta+1}{2}\in[0,1]\) 后再线性变换，得 **半径 \(r_n=\sqrt{\frac{2\log(2M/\alpha)}{n}}\)** 若直接套 \([-1,1]\) 的 McDiarmid/Hoeffding。为避免实现歧义，本 sheet **固定**以下可计算形式：）

**规范可计算半径（推荐实现）：**

\[
r_n(M,\alpha)
:=
\sqrt{\frac{2\log(2M/\alpha)}{n}},
\qquad
\mathrm{UCB}_n(\pi)=\hat\mu_n(\pi)+r_n(M,\alpha).
\tag{T0-RAD}
\]

推导锚点：Hoeffding 对 \(X\in[a,b]\)，\(\Pr\{\bar X-\mathbb E X\ge t\}\le\exp(-2nt^2/(b-a)^2)\)；取 \([a,b]=[-1,1]\)，\(b-a=2\)，单侧 \(t=r\) 时 \(\exp(-2nr^2/4)=\exp(-nr^2/2)\)；令 \(nr^2/2=\log(2M/\alpha)\) 得 \(r=\sqrt{2\log(2M/\alpha)/n}\)。对双侧/同时控制见证明树。

### 3.3 选择规则

固定名义容差 \(\epsilon\in\mathbb R\)（常取 \(0\) 或业务容许正数）与效用 \(U(\pi)\)（预注册：如 S2 上估计的 \(-\hat\mu\) 或事件召回；**不得**用 S3 先看再定义 \(U\) 而不纳入选择偏差）。

\[
\mathcal F_n
:=
\{\pi\in\Pi:\ \mathrm{UCB}_n(\pi)\le \epsilon\},
\]

\[
\hat\pi
:=
\begin{cases}
\arg\max_{\pi\in\mathcal F_n} U(\pi) & \text{if }\mathcal F_n\neq\emptyset,\\[4pt]
\pi_{\mathrm{BASE}} & \text{if }\mathcal F_n=\emptyset,
\end{cases}
\]

其中 \(\pi_{\mathrm{BASE}}\) 为恒等策略：\(\pi_{\mathrm{BASE}}(b,Z)=\mathrm{KEEP}\)，故 \(\Delta_{\pi_{\mathrm{BASE}}}\equiv 0\)，\(R_{\mathrm{rel}}(\pi_{\mathrm{BASE}})=0\)。

**要求：** \(\pi_{\mathrm{BASE}}\in\Pi\)（预注册进族），以保证 no-op 合法。

### 3.4 定理 T0

**Theorem T0 (paired UCB selection; KNOWN-FRAMEWORK INSTANCE).**  
Assume A0, A2, A3, \(\pi_{\mathrm{BASE}}\in\Pi\), and \(r_n\) as in (T0-RAD). Let \(\hat\pi\) be the selection rule above. Then

\[
\Pr_{\mathrm{S3}}
\big\{\,
R_{\mathrm{rel}}(\hat\pi)\le \epsilon
\ \big|\ \mathrm{S2}
\,\big\}
\ge 1-\alpha.
\]

Furthermore, the guarantee is **non-vacuous for feasibility** because \(R_{\mathrm{rel}}(\pi_{\mathrm{BASE}})=0\le\epsilon\) whenever \(\epsilon\ge 0\); if \(\epsilon<0\), feasibility is not automatic and the procedure may return BASE with a vacuous relative improvement claim.

### 3.5 原子证明

固定 S2。写 \(\Pr\) 为 \(\Pr_{\mathrm{S3}}(\cdot\mid\mathrm{S2})\)。

**Step 1 — 单策略浓度。**  
对固定 \(\pi\)，\(X_i=\Delta_\pi(D_i)\in[-1,1]\) i.i.d.（A0）。  
Hoeffding：

\[
\Pr\big(\hat\mu_n(\pi)-\mathbb E X_1 \ge r_n(M,\alpha)\big)
\le
\exp\big(-n r_n(M,\alpha)^2 / 2\big)
=
\frac{\alpha}{2M}.
\tag{*}
\]

- 标记：`known lemma`（Hoeffding 1963）。  
- 有界常数：range 2 进入指数分母。  
- **不是** new lemma。

**Step 2 — 同时控制（双侧形式用于实现稳健；单侧已够 T0）。**  
对下偏同样 \(\le \alpha/(2M)\)。Union bound over \(M\) policies：

\[
\Pr\big(\exists\pi\in\Pi:\ |\hat\mu_n(\pi)-R_{\mathrm{rel}}(\pi)| > r_n(M,\alpha)\big)
\le
\alpha.
\tag{**}
\]

- 标记：`known lemma`（union bound）。  
- 若只证 UCB 上偏控制，用单侧 \(r_n^{\mathrm{one}}=\sqrt{2\log(M/\alpha)/n}\) 并将 (T0-RAD) 中 \(2M\) 换 \(M\)；实现须与证明一致。本 sheet 采用 (*)(**) 的双侧同时半径。

**Step 3 — 好事件 \(G\)。**  
令 \(G\) 为 (**) 的补事件。在 \(G\) 上，对所有 \(\pi\in\Pi\)：

\[
R_{\mathrm{rel}}(\pi)
\le
\hat\mu_n(\pi)+r_n(M,\alpha)
=
\mathrm{UCB}_n(\pi).
\tag{***}
\]

- 标记：`identity`（代数重排）。

**Step 4 — 选择后覆盖。**  
在 \(G\) 上：

- 若 \(\mathcal F_n\neq\emptyset\)，则 \(\hat\pi\in\mathcal F_n\)，故 \(\mathrm{UCB}_n(\hat\pi)\le\epsilon\)，由 (***) 得 \(R_{\mathrm{rel}}(\hat\pi)\le\epsilon\)。  
- 若 \(\mathcal F_n=\emptyset\)，则 \(\hat\pi=\pi_{\mathrm{BASE}}\)，\(R_{\mathrm{rel}}=0\)。当 \(\epsilon\ge 0\) 时 \(0\le\epsilon\)；当 \(\epsilon<0\) 时 T0 结论在 BASE 分支需要 \(\epsilon\ge 0\) 前提——**实现约束：名义 \(\epsilon\ge 0\)**。  

- 标记：`identity`（分情形）。  
- **S3 选择后覆盖：** 同时界在选择前对所有 \(\pi\) 成立，故数据依赖的 \(\arg\max U\) **不**额外破坏 \(R_{\mathrm{rel}}(\hat\pi)\le\epsilon\)（标准 select-from-safe-set 论证）。  
- 标记：`known lemma`（safe-set selection / FWER 控制下的 selection；RCPS/LTT 同族思想）。**非**新框架。

**Step 5 — 概率。**  
\(\Pr(G)\ge 1-\alpha\)，故 T0 成立。

**Step 6 — no-op 语义（强制降级）。**  
\(\pi_{\mathrm{BASE}}\) 使 \(\{\pi: R_{\mathrm{rel}}(\pi)\le\epsilon\}\) 在 \(\epsilon\ge 0\) 时非空。  
这 **只** 证明证书程序 **不会** 因空可行集而逻辑崩溃。  
**不** 证明：存在 \(\pi\neq\pi_{\mathrm{BASE}}\) 以高概率进入 \(\mathcal F_n\)；**不** 证明 power；**不** 证明 S4 经验改善。

- 标记：`identity` + 明确 **non-power**。

### 3.6 Bernstein 变体（仍为 KNOWN；可选实现）

令 \(\hat\sigma^2_n(\pi)\) 为样本方差，\(v_n(\pi)=\hat\sigma^2_n(\pi)+r_n^{\mathrm{aux}}\)（辅助项按标准经验 Bernstein 处方）。则半径可换成

\[
r_n^{\mathrm{B}}(\pi)
=
\sqrt{\frac{2\hat\sigma^2_n(\pi)\log(3M/\alpha)}{n}}
+
\frac{3\log(3M/\alpha)}{n}
\quad
\text{（示意；实现须固定常数版本）}.
\]

- 标记：`known lemma`（empirical Bernstein）。  
- **禁止** 把 “配对 + Bernstein 在 \(\mathrm{Var}(\Delta)\) 小时半径更小” 写成 T1 创新（见 §4）。

### 3.7 T0 明确不是什么

| 声称 | 裁决 |
|---|---|
| 新风险控制框架 | **否** — KNOWN-FRAMEWORK INSTANCE |
| 高概率总体 \(\epsilon\)-non-degradation（有限 \(\Pi\)） | **是** — 在 A0+A2+A3 |
| 存在可认证非 BASE 策略 | **否** |
| 弱依赖/漂移下仍成立 | **否** — 未假设 A1/A4 |
| episode 结构带来的新浓度 | **否** — 只用有界标量 \(\Delta\) |
| Bi-Hurdle 似然的新校准定理 | **否** |

### 3.8 可计算 \(\epsilon\) 与样本公式

若要求 **检测** 水平为：当真 \(R_{\mathrm{rel}}(\pi)\le 0\) 且希望 \(\mathrm{UCB}\le\epsilon_{\mathrm{nom}}\) 以 \(\epsilon_{\mathrm{nom}}=0\) 接受，则需要 \(\hat\mu_n+r_n\le 0\)。  
**证书侧**（与真值无关的可计算上界容差）：任何被接受的 \(\pi\) 满足“在好事件上 \(R_{\mathrm{rel}}\le\epsilon_{\mathrm{nom}}\)”。  

若设计样本使半径本身 \(\le\epsilon_{\mathrm{des}}\)：

\[
n
\ge
\frac{2\log(2M/\alpha)}{\epsilon_{\mathrm{des}}^2}.
\tag{T0-N}
\]

例：\(M=50\)，\(\alpha=0.1\)，\(\epsilon_{\mathrm{des}}=0.05\) \(\Rightarrow\)  
\(\log(2M/\alpha)=\log(1000)\approx 6.907\)，\(n\ge 2\cdot 6.907 / 0.0025 \approx 5526\) **日**。  
S3=10% 时，多数公开序列 **远小于** 该 \(n\) ⇒ \(\epsilon_{\mathrm{des}}=0\) 的非空泛非 BASE 认证 **power 弱**（见 §9）。

---

## 4. One T1 statement

### 4.0 方向选择与唯一性

按协议 §5.3 **只选一个** T1 方向。本 sheet 选择：

> **首选方向 2：** 精确基座锚点（\(L_{\mathrm{ep}}(b,Y)\) 为 exact observable，非模型估计）下，paired episode harm 相对 **最强合法** absolute 风险控制组合，是否具有 **严格** 有限样本半径或样本复杂度优势。

备选 1（不可约性下界）与备选 3（弱依赖新保证）**本轮不写为贡献**；仅在 §10 PIVOT 中作为出口指针。

### 4.1 最强合法 absolute 对照（必须攻击这个，而非稻草人）

定义 absolute 风险 \(R_{\mathrm{abs}}(\pi):=\mathbb E[L_{\mathrm{ep}}(T(b,\pi(b,Z)),Y)]\)，\(R_{\mathrm{abs}}(\mathrm{BASE}):=\mathbb E[L_{\mathrm{ep}}(b,Y)]\)。

**Identity（关键）：**

\begin{align*}
R_{\mathrm{rel}}(\pi)
&=
R_{\mathrm{abs}}(\pi)-R_{\mathrm{abs}}(\mathrm{BASE}).
\tag{ID-RISK}
\end{align*}

- 标记：`identity`（期望线性）。

**最强合法 absolute 风险控制组合 ABS\*：**  
在 **同一** S3 日样本、同一有限动作/策略菜单 \(\Pi\)、同一 \(L_{\mathrm{ep}}\)、同一 \(\alpha\) multiplicity 下：

1. 对每个 \(\pi\) 形成日损失 \(L_i(\pi)=L_{\mathrm{ep}}(T(b_i,\pi_i),Y_i)\)，\(L_i(\mathrm{BASE})=L_{\mathrm{ep}}(b_i,Y_i)\)；  
2. 使用 **配对差** \(\Delta_i(\pi)=L_i(\pi)-L_i(\mathrm{BASE})\)（因 (ID-RISK)，这是 absolute 差的充分统计）；  
3. 对 \(\{\Delta_i(\pi)\}\) 做与 T0 **相同** 的 simultaneous UCB，并 safe-set 选择。

即：ABS\* 被允许使用配对——因为“absolute 方法不得配对”会人为削弱对照，违反红队公平。

**弱对照（仅用于说明常识，不当 T1 敌手）：**  
UNP：独立两样本或同样本但用 \(\widehat R_{\mathrm{abs}}(\pi)-\widehat R_{\mathrm{abs}}(\mathrm{BASE})\) 的 **未配对** 方差（\(\widehat{\mathrm{Var}}(L_\pi)+\widehat{\mathrm{Var}}(L_b)\)）做双均值检验。

### 4.2 试图陈述的 T1（严格版）— 目标命题

**Claim T1-strict (sought).**  
There exist constants \(c>0\), a nonempty class of laws \(\mathcal P\), and a sequence of problems with size \(n,M,\alpha\), such that any \((1-\alpha)\)-valid certificate for \(R_{\mathrm{rel}}(\pi)\le 0\) that is written in the “relative/paired-harm” language admits a radius \(r_{\mathrm{rel}}\) satisfying

\[
r_{\mathrm{rel}}
\le
(1-c)\, r_{\mathrm{ABS^*}}
\]

with \(r_{\mathrm{ABS^*}}\) the radius of the strongest valid absolute procedure ABS\*, uniformly over \(\mathcal P\), and the inequality is strict on a positive-measure subset of \(\mathcal P\).

### 4.3 裁决：`NOT JUSTIFIED`

**Theorem T1-collapse (negative).**  
Under A0+A2+A3 and the fairness constraint that ABS\* may use the same S3 days and form \(\Delta_i=L_i(\pi)-L_i(\mathrm{BASE})\),

\[
\text{ABS\* and T0 operate on identical scalars }\{\Delta_i(\pi)\}_{\pi\in\Pi}
\]

and therefore induce **identical** simultaneous UCB radii and identical safe sets (for the same concentration inequality). Consequently **no strict finite-sample radius or sample-complexity separation** exists between “paired relative harm control” and “strongest absolute risk-difference control”.

#### 原子论证

**Step A — 风险泛函同一。**  
由 (ID-RISK)，要证 \(R_{\mathrm{rel}}(\pi)\le\epsilon\) 等价于 \(R_{\mathrm{abs}}(\pi)-R_{\mathrm{abs}}(\mathrm{BASE})\le\epsilon\)。  
- `identity`

**Step B — 基座项 exact observable。**  
\(L_{\mathrm{ep}}(b,Y)\) 在 S3 日上可计算，不引入额外模型估计误差。  
- `identity`（评估定义）  
- **但是：** ABS\* 同样可在同一日计算 \(L_{\mathrm{ep}}(b,Y)\)，并不需要估计一个“未知基座风险参数”的第三方。  
- 故 “exact base anchor” **不** 建立单边信息优势。

**Step C — 充分统计。**  
任何仅通过有界损失真值依赖数据、且对 (ID-RISK) 有效的证书，在 i.i.d. 日样本下，对每个 \(\pi\) 的最优无偏风险差估计是 \(\hat\mu_n(\pi)=\overline{\Delta(\pi)}\)（完整性/Rao–Blackwell 直觉；在有限样本 concentration 证明中直接使用 \(\Delta\) 即可达到 Hoeffding/Bernstein 最优型界）。  
- `known lemma`（配对差是风险差的自然估计；非新）  
- 若声称 “relative 语言有更紧的 lemma”：`unproved gap` → 本轮 **不填**，因与 ABS\* 共享 \(\Delta\) 后无对象可填。

**Step D — 半径比较。**  
T0 半径 \(r_n(M,\alpha)\) 由 \(\Delta\in[-1,1]\) 与 \(M,\alpha,n\) 决定。  
ABS\* 使用同一 \(\Delta\) 时半径相同。  
- `identity`

**Step E — 与 UNP 的比较（弱化常识，非 T1）。**  
对 UNP，粗界可用

\[
r_{\mathrm{UNP}}
\asymp
\sqrt{\frac{\mathrm{Var}(L_\pi)\log M'}{n}}
+
\sqrt{\frac{\mathrm{Var}(L_b)\log M'}{n}},
\]

而配对 Bernstein 半径 \(\asymp \sqrt{\mathrm{Var}(\Delta)\log M'/n}\)。  
当 \(\mathrm{Corr}(L_\pi,L_b)>0\) 且接近 1 时 \(\mathrm{Var}(\Delta)\ll \mathrm{Var}(L_\pi)+\mathrm{Var}(L_b)\)。  
- `known lemma`（paired variance reduction）  
- **降级：** 这是统计常识，**不是** “相对 BECH 对象” 的新定理；且敌手不是 UNP，是 ABS\*。

**Step F — “超越 paired Bernstein” 检查。**  
协议要求 T1 必须超越 paired Bernstein 常识并指出 new lemma 或 unproved gap。  
本轮：

| 候选 new lemma | 结果 |
|---|---|
| exact base ⇒ 比 ABS\* 更小半径 | **FALSE** under fairness（Step B–D） |
| episode 非可加 ⇒ 标量 \(\Delta\) 浓度改进 | **FALSE**；非可加已在定义 \(L_{\mathrm{ep}}\) 时折叠为有界标量 |
| Bi-Hurdle 分解 ⇒ 更紧 UCB | **unproved / 无必要真**；证书不经过 hurdle 似然 |
| 有限样本 excess-risk 速率优于 ERM absolute | **unproved gap**；无证明路径不与 (ID-BAYES) 冲突 |

**结论标记：** T1-strict = **`NOT JUSTIFIED`**。  
无 new lemma 成立；唯一相关 gap 是“若限制 absolute 方法不得使用 \(L(b,Y)\) 或不许配对”——该限制 **不合法**（削弱敌手）。

### 4.4 弱化陈述（AFTER WEAKENING）— 诚实可证但非贡献

**Proposition T1-weak (KNOWN, not a contribution).**  
Under A0, for a fixed \(\pi\), let \(\Delta=L_\pi-L_b\). Then

\[
\mathrm{Var}(\Delta)
=
\mathrm{Var}(L_\pi)+\mathrm{Var}(L_b)-2\mathrm{Cov}(L_\pi,L_b)
\le
\mathrm{Var}(L_\pi)+\mathrm{Var}(L_b),
\]

with strict inequality whenever \(\mathrm{Cov}(L_\pi,L_b)>0\). Empirical Bernstein widths using \(\Delta\) are therefore never worse than the naive sum of marginal widths, and are strictly better under positive covariance.

- 全程 `known lemma` / `identity`。  
- **创新审计：排除**（§8）。

### 4.5 对“精确基座锚点”的正确余值（非 T1）

仍值得在论文 **方法叙述**（Route-E）中保留的 **工程语义**（非定理创新）：

1. 证书直接盯 \(R_{\mathrm{rel}}\)，与产品语言“相对冻结基座不劣化”一致；  
2. no-op 与 bit-exact fallback 实现简单；  
3. S3 不需要为基座再训代理模型。  

这些是 **系统性质 / 产品对齐**，在 Phase B 理论门下 **计 0 创新分**。

### 4.6 备选方向为何本轮不升级为 T1

**方向 1（Bayes 预算动作分离 + excess-risk 下界）**  
可构造：同逐点边缘，不同 episode 相关，使点级 top-K 与 episode 预算最优分歧——但协议写明：若一般 structured predictor 可表达，则只算 **问题性质**。  
要升级为 T1，还需对 **明确策略类** 证正 excess-risk 下界且敌手含 ABS-SET+VAL。本轮 **未完成** 该下界的非空洞证明 ⇒ 不写入 T1 以免假贡献。

**方向 3（弱依赖新保证）**  
需 A1 单开 sheet；预期还原为 blocking + 标准浓度 ⇒ 默认 KNOWN 例化。本轮不做。

---

## 5. Proof dependency graph

```text
[A3: measurable T, L_ep, matching]
            |
            v
[Def: Delta_pi in [-1,1]] ----identity----> [ID-RISK: R_rel = R_abs - R_abs(BASE)]
            |                                         |
            v                                         v
[A0: i.i.d./exch S3 days]                    [ID-BAYES: argmin H = argmin L_ep]
            |                                         |
            v                                         v
[Hoeffding/Bernstein]--known-->[union bound]     [Bayes absolute reduction]
            |                         |               |
            v                         v               v
     [simultaneous UCB]      [safe-set selection]   [no decision-level T1]
            \                     /
             \                   /
              v                 v
              [T0 guarantee]  **** KNOWN-FRAMEWORK INSTANCE ****
                      |
                      v
        [compare T0 vs ABS* on same Delta]
                      |
                      v
              [T1-strict collapses] **** NOT JUSTIFIED ****
                      |
                      v
         [T1-weak variance inequality] **** KNOWN, excluded ****
```

**依赖边类型图例：** `identity` / `known lemma` / `new lemma`（本图 **无**）/ `unproved gap`（已拒绝填入主链）。

**禁止跳步检查：** 证明未使用 Ville、betting martingale、e-process、共形关键词替代不等式；T0 仅用 Hoeffding + union bound + 分情形 identity。

---

## 6. Required counterexamples

> 用途：击穿假 T1 / 假决策分离；服务主窗口与 Agent A/C 交叉，不替代合成 DGP 实现。

### 6.1 CE-B1：相对标签不改 Bayes 动作（生死公式实例）

- 设 \(H=3\)，动作 \(\{a_0=\mathrm{KEEP}, a_1, a_2\}\)。  
- 固定 \(x\)，令 \(\mathbb E[L_{\mathrm{ep}}(T(b,a_j),Y)\mid x]=c_j\)，\(c_1<c_2<c_0\)。  
- 则 \(\mathbb E[H(a_j)\mid x]=c_j-c_0\)，argmin 仍为 \(a_1\)。  
- **结论：** 相对 harm 监督若不改动作类，**不能**创造新 Bayes 决策。  
- 击败声称：T1 若建立在“相对学习改变最优动作”。

### 6.2 CE-B2：ABS\* 与 T0 证书重合

- 任意 S3 有限样本 \(\{(b_i,Y_i)\}_{i=1}^n\)，任意 \(\Pi\)。  
- ABS\* 与 T0 均计算 \(\Delta_i(\pi)\) 与同一 \(r_n\)。  
- Safe set 与 \(\hat\pi\) 在同一 \(U\) 下逐样本相同。  
- **结论：** 无 FS 分离。  
- 击败声称：T1-strict。

### 6.3 CE-B3：UNP 更差 ≠ 相对框架创新

- 取 \(L_b=L_\pi+U\) 高度同向（正协方）。  
- \(\mathrm{Var}(\Delta)\ll \mathrm{Var}(L_\pi)+\mathrm{Var}(L_b)\) ⇒ 配对区间更短。  
- **结论：** 只说明应配对，不说明 BECH 对象新。  
- 击败声称：用 UNP 当稻草人证明 T1。

### 6.4 CE-B4：no-op 空洞安全

- \(\Pi=\{\pi_{\mathrm{BASE}},\pi_{\mathrm{bad}}\}\)，\(R_{\mathrm{rel}}(\pi_{\mathrm{bad}})=0.2\)，\(n\) 小使 \(r_n>0.2\)。  
- 常得 \(\mathcal F_n=\{\pi_{\mathrm{BASE}}\}\) 或仅 BASE 通过。  
- T0 成立但 **零非 BASE power**。  
- 击败声称：T0 ⇒ 方法有效。

### 6.5 CE-B5：点级最优 vs episode 预算（问题性质，非 T1）

- 预算 \(K=1\)。点级评分偏好碎裂小时；episode \(L_{\mathrm{ep}}\) 偏好一次 INSERT 完整区间。  
- 点级方法失败；**ABS-SET 直接优化 \(L_{\mathrm{ep}}\) 可成功**。  
- **结论：** 支持 episode 损失作为 **问题锚点**；不支持相对框架不可约。

### 6.6 CE-B6：空事件日与 tie

- \(E(Y)=\emptyset\)，\(E(b)=\emptyset\) ⇒ \(L_{\mathrm{ep}}\) 达约定空-空值；\(\Delta\) 对非 identity 乱 INSERT 严格变差。  
- T0 仍适用（有界性不破）。  
- tie：matching 权重相等时 A3 固定字典序；不同 tie rule 视为不同预注册损失，不得事后挑。

### 6.7 CE-B7：跨日截断

- 真 episode 跨午夜；24h 截断制造边界伪 miss。  
- 所有方法共用截断定义时，相对差仍 well-defined；  
- **不** 产生新浓度 lemma；若隐瞒截断则 SPEC INVALID。

---

## 7. Prior-theorem reduction table

| 本 sheet 对象/步骤 | 可还原到的先验 | 归类 | 残留？ |
|---|---|---|---|
| (ID-BAYES) argmin H = argmin L | 条件期望线性 | identity | 无 |
| (ID-RISK) R_rel = ΔR_abs | 期望线性 | identity | 无 |
| T0 UCB + union bound | Hoeffding；FWER；RCPS/LTT 有限类选择 | KNOWN-FRAMEWORK | 无创新残留 |
| safe no-op | reject option / identity fallback | 定义+标准 | 无 |
| paired Bernstein 更紧 | 经典配对检验 | KNOWN | 无 |
| \(L_{\mathrm{ep}}\) matching | DETR/Ciliberto/Hungarian 结构化损失 | 定义 | 无 |
| 边界/持续期 | ActionFormer/Yuan/Rabiner | 成熟 | 无 |
| 相对初始序列编辑 | LevT（离散 token；非冻结数值+连续幅度+相对风险证） | 部分碰撞 | 仅 **组合叙事** 残留，非定理 |
| Bi-Hurdle occ×mag | Hurdle/Tweedie/ZILN/Deep Hurdle | KNOWN 对齐 | 双向 episode 应用 gap ≠ 定理 |
| 弱依赖 blocking | 标准 time-series concentration | 未启用；启用则 KNOWN | — |
| e-process/Ville 路径 | 已否决旧 B1 | **禁止** | — |

**Phase A 提醒：** 八篇核验集内无“冻结数值基座 + 负价 episode 编辑 + 连续幅度 + 选择性相对风险”完整四元组 ≠ 全球空白 ≠ T1。

---

## 8. Definition-only exclusion audit

| 条目 | 是否在本 sheet 被当贡献？ | 处理 |
|---|---|---|
| exact fallback / \(T(b,\mathrm{KEEP})=b\) | 否 | 定义；T0 可行集工具 |
| no-op 可行 | 否 | 明确 non-power |
| 连通分量 CC / `<0` 阈值 | 否 | 定义 |
| Hungarian / 线性分配 | 否 | 定义 |
| 预算 \(K\) 按定义满足 | 否 | 约束 |
| 投影非扩张 | 否 | 未使用 |
| Hoeffding / Bernstein / union bound | 否 | T0 工具；KNOWN |
| 标准 blocking | 否 | 未启用 A1 |
| RCPS/LTT/CSA 换 \(L_{\mathrm{ep}}\) | 否 | T0 即此类例化 |
| 手工反例本身 | 否 | 仅证伪 |
| 经验零退化 / 消融单调 | 否 | 不进证明 |
| Bi-Hurdle“全球首次” | 否 | 明确禁止 |
| GNN / 共享图 | 否 | 本轮不进主定理 |
| TAP/GAFE/e-process/admin clamp | 否 | 禁止复活 |
| **T1-strict 严格 FS 优势** | **曾尝试** | **`NOT JUSTIFIED`，不计入贡献** |
| **T1-weak 配对方差** | 写出 | **KNOWN，排除出贡献** |

**定义性排除结论：** 本 sheet **零** 非定义性、非标准例化的已证 T1。

---

## 9. Computable epsilon and nontrivial-power check

### 9.1 可计算证书

输入：S3 日数 \(n\)，\(M=|\Pi|\)，\(\alpha\)，预注册 \(\epsilon\ge 0\)。

1. 对每个 \(\pi\in\Pi\) 算 \(\hat\mu_n(\pi)\) 与 \(\mathrm{UCB}_n(\pi)=\hat\mu_n(\pi)+r_n(M,\alpha)\)，  
   \(r_n=\sqrt{2\log(2M/\alpha)/n}\)。  
2. \(\mathcal F_n=\{\pi:\mathrm{UCB}_n(\pi)\le\epsilon\}\)。  
3. 输出 \(\hat\pi\) 与证书布尔值 \(\mathbf 1\{\hat\pi\in\mathcal F_n\}\)（BASE 时若 \(0\le\epsilon\) 则安全）。

**可计算性：** 全为有限和与初等函数；无不可计算的 covering number 或未知常数。

### 9.2 空泛性（vacuity）检查

| 检查 | 结果 |
|---|---|
| \(\epsilon\ge 0\) 时是否永可返回合法策略？ | 是（BASE）⇒ 保证 **非逻辑空** |
| 是否对非 BASE 有 nontrivial power？ | **不保证**；依赖 \(R_{\mathrm{rel}}(\pi)\le -r_n -o(1)\) 的可分间隙 |
| \(r_n\) 在真实 S3 规模是否常 \(>\) 可期改善？ | **经常是**（见下） |

### 9.3 公开数据级 \(n\) 示意（非实验；量级审计）

S3≈10% 日历日：粗算 \(n\sim 0.1\times N_{\mathrm{days}}\)。  
若 \(N_{\mathrm{days}}\sim 1000\)–\(4000\)，则 \(n\sim 100\)–\(400\)。  
取 \(M=30\)，\(\alpha=0.1\)：\(\log(2M/\alpha)=\log(600)\approx 6.40\)，  
\(r_n\approx\sqrt{2\cdot 6.40/n}\approx\sqrt{12.8/n}\)：  
- \(n=100\) ⇒ \(r_n\approx 0.36\)（相对 \([0,1]\) 损失差，**很宽**）  
- \(n=400\) ⇒ \(r_n\approx 0.18\)  
- \(n=2000\) ⇒ \(r_n\approx 0.08\)

故 \(\epsilon=0\) 下认证 **非 BASE** 需要经验 \(\hat\mu\) 显著负且幅度 \(\gtrsim r_n\)。  
Episode 改善若只体现在稀疏子集日，\(\hat\mu\) 接近 0，**T0 将理性弃权**——这是正确行为，不是方法胜利。

### 9.4 非 BASE power 的预注册检验（供 Phase C；本轮不跑）

在 S2 冻结 \(\Pi\) 后、看 S3 前，定义：

\[
\mathrm{PowerProxy}(\pi)
=
\mathbf 1
\big\{
\hat\mu_{S2}^{\mathrm{CF}}(\pi)
\le
-\,r_{n_{S3}}(M,\alpha)-\delta_0
\big\},
\]

交叉拟合估计，\(\delta_0>0\) 预注册。若对所有 \(\pi\neq\mathrm{BASE}\) 为 0，则 **预期** S3 只输出 BASE；此时即使 T0 真，**研究应停在证书工程**，不能声称事件编辑有效。

---

## 10. PASS/PIVOT/KILL recommendation

### 10.1 对 Gate 2（理论诚实）的推荐裁决

| 项 | 推荐 |
|---|---|
| T0 正确性 | **PASS**（证明完整、可计算 \(\epsilon/r_n\)） |
| T0 创新 | **FAIL**（强制 `KNOWN-FRAMEWORK INSTANCE`） |
| T1 非定义性 | **FAIL**（`NOT JUSTIFIED`） |
| **Gate 2 总判** | **KILL（A 会理论创新）** |

### 10.2 对候选对象“相对 episode 校正”理论线

**KILL** 作为 A 会主理论贡献。理由链：

1. (ID-BAYES)：决策层被 absolute structured risk 还原；  
2. (ID-RISK)+ABS\*：证书层与最强 absolute 配对控制 **重合**；  
3. T0 仅剩已知有限类风险选择；  
4. 无 new lemma；T1-strict 失败且不允许用 UNP 稻草人或非法削弱 ABS\* 来挽救。

### 10.3 与 Route-E 的边界

- **保留：** 问题锚点（`P0_DECISION`：负价 episode、完整漏报）；现有 BECH 应用证据链；T0 类 S3 安全选择作为 **工程证书**（需在论文中降级为标准工具）。  
- **Bi-Hurdle：** 仅作与 Hurdle/Tweedie/ZILN 的 **定位引用**，不作新定理。  
- **禁止：** 将 T0 或配对 Bernstein 写成理论贡献；复活 e-process/TAP/GAFE。

### 10.4 PIVOT 出口（仅当用户明确要求继续理论线）

每个出口必须 **重新冻结对象** 并新开 sheet，不得在本对象上硬编 T1：

| 编号 | PIVOT 对象 | 必须交付的真分离 | 主要风险 |
|---|---|---|---|
| P1 | **计算/算法分离**：absolute set oracle 可表示但 poly-time 不可得，而 base-anchored edit 在冻结 \(b\) 的 support 上可近似 | 复杂度下界 + 近似比，敌手含受限 decoder | 易塌缩为启发式，难 A 会 |
| P2 | **信息集分离**：校正头 **禁止** 访问某些对 absolute set 必要、但对相对 edit 不必需的标签结构（须业务可辩护） | 决策分离定理 | 可能不公平 / 不自然 |
| P3 | **预注册生成模型**下 episode 依赖下界（方向 1 强化） | 对含 ABS-SET 的类证 excess-risk \(>0\) | 假设强；用户须接受 |
| P4 | **放弃相对理论**，专做 Route-E：Hurdle correction head 应用论文 | 无 T1 需求 | 与“理论优先”目标冲突但最诚实 |

**不推荐的假 PIVOT：** 弱依赖换 blocking 常数；双市场联合；GNN 特征图；admin clamp；加大网络。

### 10.5 给主窗口的放行建议

```text
Agent B 推荐：
- 理论 Gate 2：KILL（无合格非定义性 T1）
- T0：可采纳为工程证书附录，标记 KNOWN-FRAMEWORK INSTANCE
- 不得因 T0 放行 Phase C 作为“理论验证实验”
- 若 Agent A 对 object 层 REDUCTION PASS/KILL：与本文一致 → 总裁决倾向 NO-GO（A会方法/理论）
- 若用户要应用论文：GO-to-Route-E（非 GO-to-Phase-C-theory）
- 仅当用户选择 P1–P3 之一并重冻规格：PIVOT 新一轮 Phase B
- 真实 pilot experiments/07-...：本 B 票 = 不授权（缺 T1）
```

### 10.6 主要 gap 清单（≤5，供主窗口）

1. **T1-strict 塌缩：** 相对风险与绝对风险差 identity；ABS\* 共享 \(\Delta\) 后无半径优势。  
2. **Bayes 层无分离：** (ID-BAYES) 使“学相对 harm”不产生新最优动作。  
3. **无 new lemma：** 全程为 identity + Hoeffding/union/paired variance 常识。  
4. **Power 空洞：** 公开规模 S3 下 \(r_n\) 常过大，T0 理性退回 BASE，不能证明非 BASE 可认证改善。  
5. **Bi-Hurdle/episode 结构：** 仅问题与架构锚点；不填补定理缺口。

### 10.7 文件自检（协议 §5.6 结构）

- [x] 0 总裁决  
- [x] 1 Formal object + Bi-Hurdle 节  
- [x] 2 A0–A4 分层（启用 A0+A2+A3；A1/A4 单开禁止混写）  
- [x] 3 T0 全证明 + KNOWN 标记 + 可计算 \(r_n\)  
- [x] 4 **唯一** T1（方向 2）及 NOT JUSTIFIED  
- [x] 5 依赖图  
- [x] 6 反例  
- [x] 7 先验还原表  
- [x] 8 定义性排除  
- [x] 9 \(\epsilon\) 与 power  
- [x] 10 PASS/PIVOT/KILL  

---

## 附录 A — 符号表

| 符号 | 含义 |
|---|---|
| \(b,f_0\) | 冻结基座输出/映射 |
| \(Z\) | cutoff-safe 特征 |
| \(Y\) | 真值日轨迹 |
| \(T\) | 编辑器 |
| \(\Pi,M\) | 有限策略族与基数 |
| \(L_{\mathrm{ep}}\) | 有界日级 episode 损失 |
| \(\Delta_\pi,R_{\mathrm{rel}}\) | 配对 harm 与相对风险 |
| \(r_n(M,\alpha)\) | 同时 UCB 半径 |
| \(\pi_{\mathrm{BASE}}\) | 恒等策略 |

## 附录 B — 与 Agent A/C 的接口断言

1. 若 A 证明 object 层 R2 还原闭合：与 §1.5、§4.3 **一致**，B 不反对 object KILL。  
2. 若 A 仅杀死 pointwise 而保留 structured absolute：B 的 T1 仍 FAIL（因 ABS\* 含 structured）。  
3. C 不得把 T0 当“新证书算法”写进贡献列表；合成 equivalence DGP（绝对 vs 相对同菜单）**预期 decision-equivalence**，与 T1-collapse 一致。  
4. 任何 C 侧“经验上相对更好”若动作签名一致，判优化噪声，不翻盘 T1。

---

**End of 35b — Agent B**  
**Signature verdict:** `T0 = PROVABLE AS STATED + KNOWN-FRAMEWORK INSTANCE`; `T1 = NOT JUSTIFIED`; **recommendation = KILL (theory) / Route-E or specified PIVOT**.
