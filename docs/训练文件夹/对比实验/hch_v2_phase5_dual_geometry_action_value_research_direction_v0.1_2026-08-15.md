# HCH-v2 Phase 5：双几何动作价值研究方向与实验门禁 v0.1

**日期：** 2026-08-15  
**状态：** 研究与实验设计；尚未授权修改正式数学核心或生产代码  
**依据：** universal-core CAVM 确认实验汇报（commit `46671c7`，当前公开远端尚不可解析）、IAH-CRPS 数学核心 v0.3、CAVM Phase 4 设计及前序 Phase 3/P2-ext 结果

---

## 0. 一句话裁决

> universal core 没有坍塌；当前最主要的失败不是候选分布学不到，也不是 LCB 一律过严，而是“用宿主中心双曲误差选择动作、再用原始货币 MAE 评价动作”的目标错位。下一步不继续堆检索或 P4，而先审计并验证一个最小数学修订：候选仍在日尺度 asinh 几何中训练，动作价值改在固定市场尺度的原始 MAE 几何中校准。

建议工作名：

- 整体原则：**Dual-Geometry HCH（双几何 HCH）**；
- 被修订部件：**DG-DVG（Dual-Geometry Decision Value Gate）**；
- 核心动作价值：**Metric-Faithful Action Value，MFAV（指标忠实动作价值）**。

这不是正式命名，也不构成新颖性声明。

---

## 1. 对本轮 universal-core 结果的正式复核

### 1.1 Case D 判定应当保留

逐市场拆解后只有 LAGO_DE 方向稳定，因此不能接受由聚合 `no_lcb_A_true=+0.0032` 触发的代码 Case B。Shandong 的大数值贡献掩盖了不同市场、不同 host 的方向冲突；以聚合均值替代市场级一致性会制造伪通过。

因此当前正式状态仍是：

```text
Case D / CAVM universal gate not passed
P4 remains blocked
```

### 1.2 但三条结果同时排除了错误方向

| 事实 | 能排除什么 | 不能推出什么 |
|---|---|---|
| universal core 的 weighted-mean 候选在 Shandong 可比 host 好约 36 MAE | 排除“共享候选核心完全坍塌” | 不代表最终动作链有效 |
| LAGO_DE no-LCB 为负、with-LCB 更安全 | 排除“所有失败都是 LCB 过严” | 不代表当前动作价值与论文指标一致 |
| LAGO_NP proposal-empty≈0.99、负价 0% | 说明该域几乎无可修正动作容量 | 不能把“不改善”写成检索失败 |

### 1.3 两个真正需要解释的反常现象

1. **host-split 反转**：冻结 universal core 下，Shandong Linear 改善而 PatchTST 明显退化；且与 per-market P2-ext 的方向相反。
2. **scale-free/raw 反转**：Shandong PatchTST 执行日的双曲 `A_true` 为正，但原始货币 MAE 明显变差。

第 2 条比第 1 条更基础。只要动作门控优化的量与论文主指标不一致，同一个候选在不同 host 上发生反转是完全可能的；此时增加 context key、market gate 或更复杂网络，只会学习这种错位，而不会消除它。

---

## 2. 为什么双曲 A_true 为正而 raw MAE 可以变差

### 2.1 当前动作价值的对象

当前日尺度坐标为

\[
s_t=\frac1H\sum_h|x^0_{t,h}|,\qquad
z^0_{t,h}=\operatorname{asinh}(x^0_{t,h}/s_t).
\]

查询剂量 \(\pi_{q,h}\) 在历史日 \(j\) 上回放时，现有逐时收益是

\[
g^z_{q\to j,h}
=|z^Y_{j,h}-z^0_{j,h}|
-|z^Y_{j,h}-z^0_{j,h}-\pi_{q,h}|.
\]

它回答的是：

> 这次动作是否减少了 asinh 坐标中的绝对误差？

而论文 raw MAE 回答的是：

> 这次动作是否减少了元/MWh 或 EUR/MWh 中的绝对误差？

两者不是同一个决策问题。v0.3 数学文档也已明确声明，现有 LCB 保证对象不是 raw MAE、sMAPE 或经济收益。

### 2.2 反变换的 Jacobian 会放大高幅值动作

动作回到原始价格为

\[
x^\pi_{j,h}=s_j\sinh(z^0_{j,h}+\pi_{q,h}).
\]

由中值定理，原始动作位移满足

\[
x^\pi_{j,h}-x^0_{j,h}
=s_j\cosh(\xi_{j,h})\,\pi_{q,h},
\]

其中 \(\xi_{j,h}\) 位于 \(z^0_{j,h}\) 与 \(z^0_{j,h}+\pi_{q,h}\) 之间。

因此同样大小的双曲剂量，在 \(|z^0|\) 较大或日尺度 \(s_j\) 较大的时刻，会产生更大的原始货币位移。双曲误差平均把这些时刻压缩了，raw MAE 却会让少数高幅值过冲主导整日结果。这是 Shandong PatchTST 分叉的首要数学假设。

### 2.3 “每天各自归一化”不能保证 raw 排名一致

设每个评价单元的原始 MAE 改善为

\[
\Delta_i=|y_i-x^0_i|-|y_i-x^\pi_i|.
\]

若动作价值使用随日变化的正权重 \(w_i\)，则比较的是

\[
\sum_i w_i\Delta_i,
\]

而 raw MAE 比较的是 \(\sum_i\Delta_i\)。只要存在 \(w_a\ne w_b\)，总可以构造两个改善量使两者符号相反。因此：

> 若要求对任意动作序列都保持与域内 raw MAE 改善同号，乘在每个误差改善上的归一化权重必须在该域内为同一个正常数。

这是后续固定域尺度设计的依据，不是经验调参。

---

## 3. 最小数学修订：Dual-Geometry / MFAV

### 3.1 两种几何只承担各自擅长的职责

| 层 | 几何 | 目的 | 是否改变 |
|---|---|---|---|
| 候选学习 | 日级 host-relative asinh | 跨市场尺度等变、保留零点与负价、训练三原子分布 | 不改 |
| 候选逆变换 | 精确 \(s_t\sinh(\cdot)\) | 得到真实货币单位动作 | 不改 |
| 动作价值 | 固定域尺度下的 raw MAE 改善 | 与论文主点指标同号 | 候选修订 |
| 安全门控 | 对完整整日 MFAV 做现有单侧 LCB | 只在 raw-MAE 改善下界为正时执行 | 公式形式不改，只换校准标量 |

### 3.2 结果前固定域尺度

对部署域 \(d=(market,target/mode,host)\)，仅使用 S1 中的宿主预测定义

\[
S_d
=\operatorname{median}_{t\in S1(d)}
\left(\frac1{|\mathcal H_t|}\sum_h|x^0_{t,h}|\right).
\]

要求：

- \(S_d\) 在 S3-M、S3-C、S4 全程冻结；
- 不使用 S4 target；
- 不输入 market ID 或 host ID；
- `domain` 只表示数据契约和冻结统计所属范围；
- 若 \(S_d\) 数值退化，实验应失败并报告，而不是静默用测试集修复。

### 3.3 指标忠实动作收益

对历史日 \(j\) 回放查询剂量：

\[
x^\pi_{q\to j,h}
=s_j\sinh(z^0_{j,h}+\pi_{q,h}),
\]

\[
g^{M}_{q\to j,h}
=\frac{
|y_{j,h}-x^0_{j,h}|
-|y_{j,h}-x^\pi_{q\to j,h}|
}{S_{d(j)}}.
\]

整日动作价值仍取有效小时平均：

\[
A^{M}_{q\to j}(\pi_q)
=\frac1{|\mathcal H_q\cap\mathcal H_j|}
\sum_h g^{M}_{q\to j,h}.
\]

之后保留原有流程：近邻平均得到 \(\widehat A^M_q\)，S3-C 记录 \(E^M_t=\widehat A^M_t-A^M_t\)，计算单侧分位 \(q^M_{1-\alpha}\)，并使用

\[
LCB^M_q=\widehat A^M_q-q^M_{1-\alpha}>0
\]

决定执行或 Identity。

Phase 5 主实验继续使用既定的 **target-local S3-M memory 与 target-local S3-C calibration**；跨市场共享的是 candidate core，而不是把不同币种的 raw gain 直接混入同一个未校准账本。跨域记忆若以后恢复，只能在独立实验中验证，不能借 MFAV 名义默认放行。

### 3.4 两个直接性质

**正比例尺度不变。** 若域内所有价格及宿主预测乘 \(c>0\)，则原始误差改善和 \(S_d\) 同时乘 \(c\)，所以 \(g^M\) 不变；IAH 候选也保持既有等变性。

**域内 raw-MAE 符号忠实。** 因 \(S_d\) 对该域所有日期和小时固定为正数，在相同有效掩码下：

\[
\operatorname{sign}\left(\overline{A^M}\right)
=\operatorname{sign}
\left(MAE_{host}-MAE_{corrected}\right).
\]

这正是现有日尺度双曲收益不具备的性质。

### 3.5 这不是第二个训练 loss

- IAH 候选仍只使用一个 IAH-CRPS；
- 不增加 MAE 辅助项、事件 BCE、tail loss 或交易 loss；
- MFAV 只定义冻结候选之后的“执行动作价值”；
- 不训练新 failure detector；
- 不改变一个 Down 段、一个 Up 段的结构约束。

因此它是“表示目标”和“决策目标”的职责分离，而不是多损失拼接。

---

## 4. 对现有 CAVM/P4 的重新定位

### 4.1 CAVM 暂不作为 universal 主创新

现有证据只支持：

- LAGO_DE 型极端市场中，context retrieval 可能改善历史证据排序；
- E3 的 W1 锚定比纯 context 更安全；
- 这种收益尚未跨两个市场类型成立。

所以当前论文主线不应写成“CAVM 普适解决跨市场修正”。在 MFAV 验证前，CAVM 只是一个可选证据源：

```text
W1 evidence (core)
+ contextual evidence (optional, anchored)
→ MFAV proposal
→ MF-DVG LCB
```

### 4.2 P4 继续停止

action-value state update 只有在以下条件同时满足后才重新讨论：

1. MFAV 消除 Shandong PatchTST 的 raw/scale-free 反转；
2. 至少 LAGO_DE 与 Shandong 两类市场出现方向稳定收益；
3. LAGO_NP 等无机会域维持低释放和实际非劣；
4. 改善不是单一 host、单一 seed 或单一极端日贡献。

### 4.3 不引入 host/market 身份门控

MFAV 通过真实的候选—宿主反事实误差，自然表达“这个候选相对当前 host 是否值得执行”；无需添加 `host_id` 或 `market_id`。如果它仍不能解决 host-split，再调查候选支持或历史充分性，而不是把实验表格编码成规则。

---

## 5. 文献对照与新颖性边界

| 工作 | 与本轮问题直接相关的做法 | 对 HCH 的启示 | 不能照搬之处 |
|---|---|---|---|
| [PIR, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/331c41353b053683e17f7c88a797701d-Paper-Conference.pdf) | 预测实例级 MSE 作为 failure score；结合局部上下文与全局检索；学习 revision 权重 | 说明“先估计宿主失败，再决定修正”是主流强基线，检索本身已非新颖点 | PIR 使用辅助误差目标和 MSE revision；我们不能再加相同 failure head，否则差异化迅速缩小 |
| [δ-Adapter, 2026](https://arxiv.org/abs/2601.20280) | 有界输入/输出适配器、局部下降与漂移界、量化/共形校准 | 审稿人会要求证明冻结模块不会因动作幅值造成漂移 | 其核心是可学习 adapter，不是双尾候选回放或完整动作价值 LCB |
| [CRC, 2025 preprint](https://arxiv.org/html/2512.22428v1) | ridge+MLP corrector；方向门、分位裁剪、逐点选择、shrink-to-base | 它明确指出方向正确仍可能因 scale explosion 过冲，与本轮 PatchTST 现象高度一致 | 其安全主要是验证集选择和显式防火墙；HCH 应坚持完整日动作价值校准，而非复制四个启发式门 |
| [Proper scoring rules review, 2025](https://arxiv.org/html/2504.01781v1) | 不同 proper score 会对不完美预测产生不同排序；评分规则必须匹配应用 | “候选分布 proper”不自动推出“动作对 raw MAE 最优” | 不能把 IAH-CRPS 的 propriety 扩张成 raw-MAE 安全声明 |
| [Gneiting, Making and Evaluating Point Forecasts](https://www.bundesbank.de/resource/blob/635562/7d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf) | 决策的 Bayes action 必须与预先指定的 loss 对齐 | 支持将分布学习与最终点动作的评价目标严格区分 | 不意味着需要重新训练候选；可以只更换动作效用 |
| [Lago et al., EPF benchmark](https://arxiv.org/abs/2008.08004) | 多年、多市场、强简单基线、适当指标与显著性检验 | 后续不能用跨币种 raw MAE 直接池化，也不能只报总体均值 | 该基准主要评估预测器，不直接解决 post-hoc selective correction |
| [Maciejowska et al., 2025](https://arxiv.org/html/2511.13616v1) | raw MAE/RMSE 与储能利润相关性弱；日曲线形状指标更贴近经济价值 | 论文需把统计准确性与经济价值分表，不能把 MAE 改善等同于套利改善 | 本轮 MFAV 先对齐统计主指标；经济价值仍是独立应用评价，不进入 loss/LCB |

### 5.1 当前最严格差异化声明候选

> PIR/δ-Adapter/CRC 学习额外 revision 或安全规则；HCH 的候选是跨市场冻结的三原子双尾分布，DG-DVG 不再用“检测到极端”或预测误差头决定执行，而是把今天的具体双尾剂量在历史上做宿主相对反事实回放，并对与域内 raw MAE 同号、同时保持跨尺度不变的完整整日动作价值校准下界。

这句话只能在 MFAV 新鲜留出实验通过后使用。

### 5.2 暂不允许的主张

- 不得称 CAVM 为跨市场普适检索机制；
- 不得称现有双曲 LCB 为 raw MAE/sMAPE/利润安全保证；
- 不得称 weighted-mean 候选改善为最终 HCH 改善；
- 不得称 LAGO_NP 无改善为失败；它当前是无动作容量域；
- 不得称 universal core 已达到 SOTA；尚未完成同行后处理基线正式比较；
- 不得称 CRC 的“guaranteed non-degradation”已被我们超越；其假设、实现和公开代码状态需单独审计；
- 不得宣称检索、模型无关修正、abstention 或 conformal correction 本身首次提出。

---

## 6. 下一轮实验：先审计，再改单轴，再新鲜确认

## 6.1 Phase 5A：Shandong PatchTST 日级/逐小时法医审计

**不重训，不改变动作，只重算账本。**

必须覆盖：

- Shandong_DA × PatchTST × seed 0/1/2；
- 对照 Shandong_DA × Linear；
- 正例 LAGO_DE × Linear/MLP/PatchTST。

每个真正执行日、每小时输出：

| 字段 | 含义 |
|---|---|
| `host_raw,target_raw,corrected_raw` | 三条原始价格 |
| `z0,zY,pi` | 双曲坐标与剂量 |
| `scale_day,scale_domain` | 当前日尺度与冻结域尺度 |
| `gain_z` | 当前双曲收益 |
| `gain_raw` | 原始 MAE 收益 |
| `gain_mfav` | `gain_raw/scale_domain` |
| `jacobian_proxy` | `scale_day*cosh(z0)`，仅诊断 |
| `proposal,LCB,action` | 动作链状态 |
| `tail_state` | 由 S1 连续状态定义的 high/low/normal |

必须回答：

1. `gain_z>0 && gain_raw<0` 集中在哪些日、小时、方向和价格幅值？
2. raw 损害是否被少量 `jacobian_proxy` 极大的小时主导？
3. 是否存在反变换、单位、mask、逐日聚合或 action interval 的实现错误？
4. PatchTST 与 Linear 的分叉能否由 host residual 与候选位移的相对几何解释？
5. 只看已执行动作时，current LCB 对 raw harm 的识别率是多少？

任何一项出现实现错误，先修复错误，暂停数学修订。

## 6.2 Phase 5B：同一动作的四效用离线反事实

使用完全相同的候选、邻居和已保存动作，离线并列计算：

| 代号 | 价值定义 | 用途 |
|---|---|---|
| U0 | 当前 asinh-L1 gain | 现网对照 |
| U1 | raw gain / 每日 scale | 验证“日变权重仍会反转” |
| U2 | raw gain / 固定 S1 域 scale | MFAV 候选 |
| U3 | raw gain（不归一化） | 验证 U2 与 raw MAE 符号一致 |

U2 与 U3 在同一域内应有完全一致的动作价值符号和排序比例；若不一致，说明实现或有效掩码有错误。

本阶段使用既有 S4 只能做诊断，不能用来选择正式公式后再报告同一 S4 的性能。正式公式必须在看到新鲜确认段结果之前冻结。

## 6.3 Phase 5C：DG-DVG 单轴筛选

固定以下内容：

- 同一个 universal core checkpoint；
- 同一 Data Signature；
- 同一 split；
- 同一 retrieval key、k 与 λ；
- 同一双事件提案算法；
- 同一 alpha；
- 只替换动作收益坐标和对应 S3-C q。

筛选矩阵：

| 维度 | 设定 |
|---|---|
| 市场 | LAGO_DE、Shandong_DA |
| host | Linear、MLP、PatchTST |
| seed | 0/1/2 |
| retrieval | W1、E3 composite |
| action value | current-asinh、MFAV |
| 规模 | 2×3×3×2×2 = 72 arm-cell |

四个受控臂：

- F0：W1 + current-asinh DVG；
- F1：E3 + current-asinh DVG；
- F2：W1 + MF-DVG；
- F3：E3 + MF-DVG。

先检验 MFAV 是否解决目标错位，再判断 CAVM 是否仍提供额外收益。若 F2 已解决 Shandong，F3 没有额外稳定收益，则 CAVM 不进入主算法。

## 6.4 Phase 5D：无机会域与薄样本安全确认

只有 Phase 5C 通过后才运行：

- LAGO_NP；
- Shaanxi_RT；
- Gansu_DA；
- 后续可扩 Qinghai/Ningxia。

这里的成功标准不是“强行改善”，而是：

- proposal/action 为空时维持 Identity；
- 低释放率与无显著 raw 退化；
- 不因 MFAV 尺度更换制造伪动作；
- 有真实低价/负价或尖峰时才检查 tail gain。

---

## 7. 指标与统计口径

### 7.1 论文主表

1. MAE：每市场原始货币单位，同行最常见主指标；
2. RMSE：显示少数尖峰大误差；
3. sMAPE：保留，不设 floor；近零和负价局限在正文说明；
4. macro relative skill：跨市场只聚合相对 Identity 的百分比改善，不直接平均元与欧元 MAE。

### 7.2 双尾与事件表

- S1 连续状态定义的 high-tail MAE / low-tail MAE；
- 物理负价 MAE 与样本数（仅真实存在时）；
- 尖峰/低谷事件召回、起止偏差、持续长度误差；
- 峰值/谷值幅值误差；
- normal-regime MAE 与退化率。

### 7.3 动作链表

- candidate alive；
- proposal non-empty；
- release rate；
- `exec_mean_gain_raw` 与 `exec_mean_gain_mfav`；
- raw harmful-release rate；
- LCB empirical coverage；
- action-empty 域单独报告，不用 `mean_A_true` 制造改善。

### 7.4 概率质量独立报告

- IAH-CRPS；
- 三原子质量/剂量；
- weighted mean / median 只作候选读出诊断。

不得用 CRPS 改善替代 final point/action 改善。

### 7.5 显著性

- 以日为配对单元；
- 使用适合时间依赖的日损失差检验/块自助法；
- 每市场、每 host 报告区间，不只报 3-seed 均值；
- 多方法比较时控制多重检验；
- 原始序列、split 和超参数必须在全部方法间一致。

---

## 8. 放行门禁与失败分支

### Gate M1：数学/实现一致性

- U2 与 U3 域内符号严格一致；
- 正比例缩放测试保持候选、动作方向和 MFAV 不变；
- bundle round-trip 复现 `S_d/A_hat/q/LCB/final`；
- S4 target-free；
- E1==E0 兼容契约仍成立。

### Gate M2：主要反例被解决

- Shandong PatchTST 不再出现“MFAV LCB 正而 raw MAE 大退化”；
- 3 seed 方向一致；
- 改善不由单一天或单小时贡献；
- normal-regime 退化不越既定 15% 红线。

### Gate M3：跨市场价值

- LAGO_DE 与 Shandong 至少两类市场方向稳定；
- LAGO_NP 等无机会域实质非劣、释放率合理；
- 至少两个 host family 获益，强 host 不被系统性破坏；
- tail 指标和整体 MAE 至少有一条清晰 Pareto 改善路径。

### 失败分支

| 结果 | 结论 | 下一步 |
|---|---|---|
| MFAV 修复 Shandong，LAGO_DE 保持收益 | 双几何成为主线；再判断 CAVM 是否保留 | 扩大市场并进入同行基线 |
| MFAV 修复 raw harm，但动作几乎全拒绝 | 候选有价值，选择证据不足 | 审计提案充分性；仍不放 P4 |
| MFAV 下候选 oracle 好、proposal 仍差 | 双事件/检索选择错误 | 检查事件区间与候选 Bayes action，不改 loss |
| MFAV 仍使 PatchTST raw 大退化 | 不只是指标错位 | 回到逆变换、候选相对 host 几何和实现审计 |
| 仍只有 LAGO_DE 改善 | 无法支撑 universal SOTA | 将 CAVM 降为极端市场专用，重新寻找普适动作机制 |

---

## 9. SOTA 与会议方向

### 9.1 现阶段离 SOTA 还缺什么

当前已经具备：

- 真正共享且冻结的 universal candidate core；
- 多市场、多 host、多 seed 的可证伪实验；
- 明确的 Identity 回退和完整动作价值校准；
- 负价/低价与尖峰统一候选表示。

仍然缺少：

- final action 在两个以上市场类型稳定改善；
- 对 PIR、δ-Adapter 等官方同行后处理模块的正式统一协议比较；
- raw 指标、双尾指标和安全指标同时成立；
- fresh holdout 上的显著性；
- streaming/concept-drift 证据（P4 尚未放行）。

### 9.2 基线处理

既定五个同行基线继续保留：

1. Base / Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR 官方；
5. δ-Adapter Ada-Y 官方。

CRC 是 2025 年末出现的直接安全校正碰撞项。即使当前没有可复现实装，也必须在 Related Work 和 novelty risk 中讨论；若之后出现可靠官方代码，建议把它加入补充对比，而不是假装不存在。

### 9.3 KDD / WWW 适配

在当前问题形态下，**KDD 比 WWW 更自然**：核心是跨域数据挖掘、冻结后处理、选择性决策和多市场工业验证。WWW 缺少直接 Web 场景，除非未来引入公开在线市场信息流、连续更新与大规模部署系统，否则容易被质疑 venue mismatch。

KDD 版本应围绕一条线：

> 一个跨市场冻结候选如何在未见市场中识别“相对当前宿主真正有价值的双尾动作”，并以与主评价指标同号的校准下界安全执行。

不要把论文写成 IAH、CAVM、事件段、LCB、P4 五个独立模块。

---

## 10. 给代码 Agent 的直接执行任务

```text
任务：HCH-v2 Phase 5A/5B 双曲收益—raw MAE 分叉审计。先调查，不修改正式数学默认值。

仓库：https://github.com/disdorqin/bech-paper
最新本地提交：46671c7（若尚未推送，必须如实声明）
数学基线：hch_v2_iah_crps_final_math_core_v0.3

严格边界：
1. 不改 IAH-CRPS、三原子候选、query-dose replay、双事件结构、alpha；
2. 不增加 loss、事件头、market/host ID、硬阈值、P4；
3. 不使用 S4 target 选择公式或参数；
4. 本轮只做日级/逐小时法医账本和四种效用离线重算；
5. 现有结果和脚本不可覆盖，使用全新产物目录和文件名。

执行：
A. 对 Shandong_DA×PatchTST×3 seed、Shandong_DA×Linear×3 seed、LAGO_DE×3 host×3 seed，导出真正执行日的逐小时 ledger：host_raw,target_raw,corrected_raw,z0,zY,pi,scale_day,proposal,A_hat,q,LCB,action。
B. 新增纯诊断字段：gain_z、gain_raw、frozen S1 domain scale、gain_raw/domain_scale、scale_day*cosh(z0)。
C. 验证 inverse transform、单位、mask、24 小时聚合、interval 索引和 action sign。
D. 对完全相同动作计算 U0=current asinh gain、U1=raw/day-scale、U2=raw/fixed-domain-scale、U3=raw；验证同域 U2/U3 符号严格一致。
E. 输出：
   - 每 seed 收敛/执行摘要；
   - gain_z 与 gain_raw 的日级和小时级符号混淆矩阵；
   - raw harm 对 scale、|z0|、jacobian proxy、动作方向、事件状态的分解；
   - Shandong Linear vs PatchTST 对照；
   - 明确判定 BUG / METRIC_MISMATCH / BOTH / INCONCLUSIVE；
   - 若为 METRIC_MISMATCH，生成 Phase 5C 所需最小代码 diff 计划，但暂不切换默认实现。

返回时必须给出：commit、工作树状态、产物路径、运行命令、失败 cell、关键数字、图表路径和最终判定。
```

---

## 11. 当前最终方向

这一轮实验并没有说明原框架应推倒。它给出了一个比“检索参数没调好”更有价值的发现：

> HCH 已经能够学习跨市场可迁移的双尾候选，但其安全决策仍在错误的评价几何中工作。

若 Phase 5A 证实不是代码错误，最值得推进的创新不是增加网络深度，而是建立一个严格分工的双几何系统：asinh 几何负责跨市场学习，固定域尺度的 raw 几何负责动作价值与安全。它同时解释 Shandong host-split、保留负价/尖峰统一表示，并能在不增加训练 loss 或事件头的前提下，把论文主指标和执行门控真正贯穿起来。
