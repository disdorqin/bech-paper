# BECH v5 第三轮严格同行评审

> 审稿日期：2026-08-07  
> 审稿对象：v5 六份候选发现、碰撞与方法设计材料  
> 审稿性质：作者答辩后的方法学复审；重点审查数学有效性、独立创新、统一主线、近邻划界与数据承诺  
> 总体裁决：**Reject / Major Rebuild（不是普通 Major Revision）**

---

## 0. 一页裁决

v5 最大的进步是诚实承认“主方法门槛未达”，并主动淘汰 A1/A2/A3；但它对剩余两个候选的裁决仍然过宽：

1. **B1′ 不是一个有效的 e-process 定理。** 所给乘积因子在一般情形下是均值大于 1 的增长过程，而不是零假设下的非负超鞅；Ville 不等式不能按文中方式使用。它还把“检验/发现预算风险”误写成“保证预算永不越界”。
2. **C1 不是足以承担主贡献的领域机制。** 它的硬核部分只是“把预测投影到已知可行区间”的标准几何事实；而且正文把校正方向写反，并同时声称 `clamp(base)` 与 raw BASE bit-exact，二者不能同时成立。
3. **v5 仍没有独立算法对象、有效新定理或足够厚的领域机制。** 因而“C1+B1′ 作为两项可并入强化”也不能成立；它们最多分别降为规则投影基线与待重新推导的序贯监控探索。

### 六项最终裁决

| 审稿问题 | v5 自裁 | 本轮复审裁决 |
|---|---|---|
| ① 真创新？ | C1 领域机制；B1′ 新控制对象 | **均未过线。当前无新增可接受主创新** |
| ② 一票否决风险？ | 主要是算法对象缺位 | **有三项**：B1′ 定理错误、BASE 恒等性矛盾、数据标签/规则边界语义未锁定 |
| ③ 如何强化？ | 把 C1、B1′ 并入 v4 | **不能直接并入**；先删除错误定理，把 C1 降为规则投影基线，再选择理论路线或实证路线 |
| ④ 主线还是拼盘？ | C1+B1′ 强化 v4 | **仍是拼盘风险**：一个是输出合法化，一个是部署后监控，二者没有共同学习对象或新优化算子 |
| ⑤ 划界成立吗？ | 与 CRC/δ-Adapter 等非碰撞 | **宽划界不成立**；尤其 B1′ 与 CSA/Anytime-Valid CRC/SCORC 高度重叠 |
| ⑥ 数据承诺致命吗？ | 9 市场可做 R1–R3 | **是**；规则边界随市场、产品和时间变化，山东目标字段仍未证明就是受该上下限约束的市场出清价 |

### 编辑建议

> 不建议作者继续给 v5 增加候选模块。下一轮只允许做两件事：其一，重建一个数学上有效、与 CSA 等明确有差异的风险对象；其二，冻结数据标签语义和逐时规则边界。两项任一失败，就应把论文转为严格实证的领域方法论文，而不是 A 类通用 ML 方法论文。

---

## 1. 本轮材料与审稿边界

### 1.1 已读取材料

| 材料 | 状态 | 作用 |
|---|---:|---|
| `14a · v5 Agent A：找一个不可替换的新算法对象` | ✅ | A1–A4 算法候选与显然组合攻击 |
| `14b · AgentB v5 时序安全理论报告` | ✅ | B1–B3 e-process/序贯安全候选 |
| `14c · AgentC · v5 电价特有结构` | ✅ | C1–C4 领域机制候选 |
| `11 · v5 开放式创新候选池` | ✅ | 11 候选汇总与初筛 |
| `12 · v5 候选碰撞矩阵` | ✅ | 组合攻击与 11→2 收敛 |
| `13 · BECH v5 候选方法设计` | ✅ | C1+B1′ 合并建议与 NOT READY 裁决 |

### 1.2 仍缺失的关键正文

六份材料反复引用 `09_BECH_统一方法设计v4.md`，但本轮未提供该正文。因此本报告可以：

- 审核 v5 新增对象的公式是否自洽；
- 审核候选是否真的越过 SOTA；
- 审核 v5 汇总是否忠实于上游 Agent 报告；
- 审核新增模块与既有 BASE、S1–S5、六检验族的接口矛盾。

但仍不能：

- 对 v4 全部符号和数据流做逐行证明；
- 确认 S1–S5 在实现中无泄漏；
- 确认六检验族的完整 family、阈值和分母定义；
- 确认代码中的 BASE 与文档中的 BASE 完全一致。

所以本轮的“Reject”已经由 v5 自身公式足以推出；它不依赖缺失正文。

---

## 2. 相对上一轮：哪些是真修复，哪些是假修复

### 2.1 真修复

1. **诚实承认算法创新缺位。** 14a 对 A1/A2/A3 的淘汰基本正确，没有继续把 OCO、RCPS 或功效学习的换场景例化包装成新算法。
2. **e-BH→FWER 的纠错方向正确。** 12/13 已意识到“任一组超预算”是 FWER 事件，e-BH 只控制 FDR，不能用于该主张。
3. **P3 从硬结论降为待证明。** 13 不再把 fill-to-bound 直接当成已完成定理。
4. **总体标记 NOT READY。** 这是科学上正确的状态标签。

### 2.2 假修复或新增错误

1. 12/13 只修复了 B2 的多重检验工具，却**没有重新验证 B1 的单个 e-process 是否有效**。结果是把错误的原子定理包装成了 FWER 联合定理。
2. 14b 摘要宣称“给出 β-mixing 修正项推导，块长 \(m=\lceil\log n/\lambda\rceil\)、\(\Delta=O(1/n)\)”，但正文没有该推导；结尾反而明确将 mixing 修正列为“需证明”。13 又把这条不存在的推导写成“14b 已给”。这是一次**跨 Agent 事实复制失败**。
3. C1 虽然把 P3 标成待证，但保留了方向反写、BASE 恒等性冲突和“投影不损害任何已证不等式”的过度推论。
4. “新应用对象”被当成保留理由，但 **CSA 已经在 2026 年明确提出逐阈值 e-process、选择性风险、可预测更新、Bonferroni 网格和 act/abstain 部署规则**；这使 B1′ 的应用位置也不再空白。[Conformal Selective Acting](https://arxiv.org/abs/2605.20270)

---

## 3. P0 一票否决：B1′ 定理当前是错的

### 3.1 所给过程不是 e-process

文中定义

\[
X_t=[H_t-\delta_g]_+\in[0,M],\qquad
E_t=\prod_{s\le t}\left(1+\lambda X_s/M\right).
\]

因为 \(X_t\ge 0\)，所以

\[
\mathbb E\!\left[1+\lambda X_t/M\mid\mathcal F_{t-1}\right]
=1+\lambda\mathbb E[X_t\mid\mathcal F_{t-1}]/M\ge 1,
\]

而且只要 \(P(X_t>0)>0\)，条件期望就严格大于 1。该过程一般是向上增长的过程，不是零假设下期望不超过 1 的非负超鞅，因而不能直接调用 Ville 不等式。

14b 所写的“对 composite null \(P(X\le0)\)”也没有修复问题：由于 \(X=[H-\delta]_+\ge0\)，\(X\le0\) 等价于 \(X=0\) a.s.，这是退化零假设，不是通常意义上的风险预算零假设。

正确的 betting/e-process 必须围绕一个**有正有负、并在零假设下具有正确条件均值方向的中心化增量**构造。Waudby-Smith–Ramdas 的工作解决的是有界均值的 betting confidence sequence，并非允许把任意非负 harm 直接乘成 e-process。[原始论文](https://arxiv.org/abs/2010.09686)；正式发表信息为 JRSSB 2024，而非材料写的 “Annals of Statistics 2022”。

### 3.2 Ville 不等式被写成了错误结论

Ville 给出的标准形式是：若 \(E_t\) 在零假设下是从 1 开始的非负超鞅，则

\[
P_{H_0}\!\left(\sup_t E_t\ge 1/\alpha\right)\le\alpha.
\]

它不直接推出材料中的

\[
P\!\left(\exists t:\sum_{s\le t}X_s>B_t\right)
\le 1/E_T\le\alpha.
\]

这里至少缺少三步：

1. 指定明确零假设；
2. 证明 \(E_t\) 在该零假设下为超鞅；
3. 证明“预算越界事件”必然包含于或蕴含“e-process 越过 \(1/\alpha\)”事件。

当前三步均未完成。`1/E_T` 还是随机量，不能这样直接充当固定概率上界。

### 3.3 “发现风险”不等于“保证预算永不越界”

即使重写出一个有效 e-process，它通常也只提供：

- 对某个安全/不安全零假设的序贯证据；或
- 某个均值/风险的 anytime-valid confidence sequence。

它不能自动保证已实现路径上的

\[
\sum_{s\le t}X_s\le B_t\quad\forall t.
\]

原因很直接：\(H_t\) 只有在真实价格 \(y_t\) 到达以后才能观察。若第 \(t\) 次动作已经使预算越界，事后 e-process 报警也不能把该伤害撤销。材料中“越界→整体回退 BASE”至多保护未来时点，是**监控器**，不是越界前的 barrier certificate。

如果作者真正要硬路径预算，动作前必须利用剩余预算和最坏情形动作代价，例如仅在

\[
\overline h_t(A_t)\le B_{t}-\sum_{s<t}h_s

\]

时允许动作；否则回退 BASE。若没有可信的动作前上界 \(\overline h_t\)，就不能承诺 realized-budget 不越界。

### 3.4 组定义和动作定义混在了一起

材料使用

\[
\mathbf1\{A_s=g\},\quad g\in\{0^{acted},-,+\}.

但 \(A_s\) 是动作，\(g\) 却同时被用作真实事件组。`normal-acted` 更不是一种与正/负尾并列的动作。至少需要拆成：

\[
I_{s,0a}=\mathbf1\{G_s=0,A_s\ne BASE\},
\]

\[
I_{s,-}=\mathbf1\{G_s=-,A_s\ne BASE\},\qquad
I_{s,+}=\mathbf1\{G_s=+,A_s\ne BASE\}.
\]

还必须说明 \(G_s\) 是由真值定义还是预测状态定义。若由真值定义，它只能用于事后审计，不能用于当期路由。

另外，14b 的原式把 `1{A_s=g}` 乘在乘积因子外，按字面会在任一非本组样本出现时把整个过程置零；13 则静默改成指数门控。后者才具有“非本组时冻结”的形式，但这种上游公式修复没有被记录，也没有触发对整个定理的重新证明。

组预算还不应无条件按日历时点 \(t/T\) 分配。对于稀疏出现的真实尾组，更自然的索引通常是已观察/已动作计数 \(N_{g,t}\) 或明确的暴露过程；否则一个很久未出现的组会在日历时间中凭空积累预算。

### 3.5 i.i.d. 与自适应动作不兼容

文中既说 \(H_t\) i.i.d.，又允许 e-process 根据历史结果决定未来是否动作。此时经过自适应门控后的 \(I_tH_t\) 一般不再是简单 i.i.d. 序列。正确条件应写成：

- 动作和 betting rate 对 \(\mathcal F_{t-1}\) 可预测；
- 在选定零假设下，中心化增量满足相应的条件均值或条件 mgf 约束；
- 延迟真值只能更新下一时点的控制器。

2026 年的 CSA 正是按 filtration、predictable update 和 gated increment 来构造，而不是把自适应序列重新称为 i.i.d.。[CSA 的 e-process 与定理](https://arxiv.org/html/2605.20270v1)

### 3.6 β-mixing 声明没有证明

14b 的正文没有给出：

- \(\beta(k)\) 的衰减假设；
- block/coupling 定理；
- mixing 修正后的超鞅或近似超鞅；
- \(\Delta_n\) 的显式常数；
- 块长与衰减率参数的关系。

因此 `m=ceil(log n/λ)` 与 `Δ=O(1/n)` 只能视为摘要中的未支撑断言。尤其 `λ` 已被同时用于校正幅度和 betting rate，无法判断该公式中的 \(\lambda\) 指什么。

`Distribution-uniform anytime-valid sequential inference` 讨论的是分布族上一致的渐近序贯推断，并不等价于“为任意 β-mixing 电价序列自动提供 \(O(1/n)\) 修正”。[相关论文](https://arxiv.org/abs/2311.03343)

### 3.7 多重检验虽改对方向，引用仍需修正

12/13 正确指出 e-BH 控 FDR 而非“至少一组出错”的 FWER。但材料把 e-Holm 归为 “Wang & Ramdas 2022” 并不准确。更直接的 FWER e-value 参考是 Hartog–Lei 的闭合检验/e-Holm 工作；默认方案仍建议使用预注册 Bonferroni，因为组数很少且最容易审计。[Family-wise Error Rate Control with E-values](https://arxiv.org/abs/2501.09015)

### 3.8 与 2026 近邻的边界已经非常窄

| 近邻 | 已覆盖内容 | B1′ 剩余差别 |
|---|---|---|
| [CSA](https://arxiv.org/abs/2605.20270) | 每阈值 e-process、选择性风险、可预测更新、Bonferroni 网格、act/abstain、anytime-pathwise | harm 定义与电价 signed groups |
| [Anytime-Valid CRC](https://arxiv.org/abs/2602.04364) | 随增长校准集的 anytime-valid 风险控制 | 校正动作的 base-relative harm |
| [SCORC](https://arxiv.org/abs/2606.08517) | 自适应阈值、selected risk、acceptance floor、utility 的联合有限样本证书 | 时序累计 harm 与电价双尾 |
| [ARC-STAR](https://arxiv.org/abs/2605.22222) | 冻结宿主、post-hoc correction、风险分诊、预算感知路由 | 电价组风险与严格统计对象 |

因此即使 B1′ 被正确重写，也只能先定位为“CSA/序贯风险控制在电价 base-relative harm 上的领域化”，不能再称新的理论框架。

### 3.9 B1′ 的处理决定

**当前必须整项撤回，不允许以‘需补证明’继续保留。** 原因不是证明少一页，而是所定义对象不满足定理的第一步。

允许的两条重构路线只能二选一：

#### 路线 B-A：统计风险监控

定义中心化、组门控的有界增量，例如

\[
Z_{t,g}=I_{t,g}(h_t-r_g),\qquad h_t\in[0,M_g],
\]

并根据要检验“安全”还是“不安全”选择正确方向的 predictable betting factor。该路线给的是均值/选择性风险的序贯证据，必须与 CSA 正面对照，不得声称 realized budget 永不越界。

#### 路线 B-B：硬预算安全控制

把预算写成动作前的可行性约束，控制器根据剩余预算和动作最坏代价决定是否动作。该路线更接近 safe OCO/barrier control，必须证明动作代价上界以及延迟反馈下的可行性；不能再把 e-process 当作硬预算执行器。

---

## 4. P0 一票否决：C1/APB-FC 不是当前声称的领域机制

### 4.1 P1 正确但平凡，而且没有解决负价漏报

若市场标签确实满足 \(y\in[floor,cap]\)，则

\[
\hat y_{base}<floor\Rightarrow y-\hat y_{base}>0

\]

当然成立。但它只处理**基座预测本身落到合法区间外**的情况，不处理真正困难的负价漏报。例如：

\[
floor=-80,quad y=-80,quad \hat y_{base}=200.

\]

此时基座严重漏掉负价，但 \(\hat y_{base}\) 仍位于合法区间内，P1 完全不触发。因此 P1 不能替代 negative-occurrence classifier，也不能支撑原始 BOM 的“负尾 occurrence–magnitude”主张。

如果基座本来就使用 bounded output 或部署前 clip，`base<floor` 子集甚至可能为空。于是 C1 的核心实验只会测量一个不合法输出修复器，而不是双尾校正器。

### 4.2 校正方向在正文中写反

13 §1.2 写：

- `base < floor` → “直接负向校正”；
- `base > cap` → “直接正向”。

但按文中残差定义 \(e=y-\hat y\)：

- `base < floor` 时 \(e>0\)，预测必须**向上**修正；
- `base > cap` 时 \(e<0\)，预测必须**向下**修正。

如果“负向”实际指 negative-price event，而不是 \(\Delta\) 的符号，就必须把动作名拆成 `TAIL_NEG` 与 `DELTA_UP`，不能继续让 `NEG/POS` 同时表示事件组和修正方向。当前写法会直接造成实现反号。

### 4.3 `clamp(base)` 与 bit-exact BASE 不能同时成立

文中方法签名是

\[
\hat y(\pi)=\operatorname{clamp}(\hat y^{BASE}+\lambda\delta,floor,cap).

\]

当 \(\lambda=0\) 时，输出为

\[
\operatorname{clamp}(\hat y^{BASE},floor,cap),

\]

它只有在 raw BASE 本来位于区间内时才等于 raw BASE。恰恰在 C1 想处理的 `base<floor` 或 `base>cap` 子集上，它不可能 bit-exact。

所以必须二选一：

1. **raw BASE fallback**：`A=BASE` 时绕过 clamp，返回 raw BASE；此时模型并不始终满足行政边界；或
2. **projected BASE baseline**：先定义 \(BASE_{proj}=Proj_{C_t}(BASE_{raw})\)，所有策略相对 \(BASE_{proj}\) 比较；此时失去“与原始基座 bit-exact”，但实验归因干净。

审稿建议采用第 2 种，并同时报告 raw BASE，防止把标准投影带来的免费增益算给 BECH。

### 4.4 P2 的正确结论比材料声称的窄

若 \(y\in C=[floor,cap]\)，欧氏投影满足

\[
|y-Proj_C(\tilde y)|\le |y-\tilde y|.

\]

因此对 MAE/MSE 等随绝对误差单调的点预测损失，投影不会比**同一个未投影候选**更差。

它不能推出：

- 不影响所有经济损失或事件指标；
- 不影响所有已证明的置信界；
- 不必重新审计六检验族；
- 不破坏 raw BASE bit-exact；
- 对“投影后基座”仍有额外 novelty。

“1-Lipschitz”本身也不等于“保留任意风险定理”；必须明确比较对象与损失类。

### 4.5 P3 不是 fill-to-bound 新定理

平方损失下，无约束 Bayes 预测是 \(\mu(Z)=E[y\mid Z]\)，区间约束下的 Bayes 预测是

\[
Proj_{[floor,cap]}\{\mu(Z)\}.

\]

写成增量就是

\[
\delta^*_{constrained}=Proj_{[floor-\hat y_b,\;cap-\hat y_b]}\{\mu(Z)-\hat y_b\}.

\]

这只是约束 Bayes act/投影恒等式。最优点通常在条件均值处，并不意味着“填满到边界”；只有条件均值越过边界时最优点才落在边界上，而若真值本身严格受边界约束，真实条件均值原则上也位于闭区间内。

所以 `min(delta*, cap-base)` 既不是新的解析算法，也不应命名为 fill-to-bound 最优。

### 4.6 行政边界不是一个跨年份不变常量

材料把每个市场写成单一 `floor/cap`，但真实规则具有市场、产品和时间索引：

\[
C_{m,p,t}=[floor_{m,p,t},cap_{m,p,t}].

\]

核验结果：

- 欧盟 2026 日前耦合参考边界为 +4000/-500 EUR/MWh，但方法同时规定达到触发条件后自动调整，不能把 -500 当作永久常量。[ACER Decision 02/2026 Annex I](https://www.acer.europa.eu/sites/default/files/documents/Individual%20Decisions_annex/ACER-Decision-02-2026-AnnexI.pdf)
- NEM 在 2025-07-01 至 2026-06-30 的 market price cap 是 20,300 AUD/MWh；AEMO 2026 Q2 报告使用的 market floor 是 -1,000 AUD/MWh。材料中的 15,000 已过时。[AEMC 2026–27 更新](https://www.aemc.gov.au/news-centre/media-releases/aemc-updates-market-price-cap-2026-27)、[AEMO Q2 2026](https://www.aemo.com.au/-/media/files/major-publications/qed/2026/qed-q2-2026.pdf)
- ERCOT 2024 官方报告写明当时 VOLL 与 SWCAP 均为 5,000 USD/MWh；材料同时写 SWCAP=5,000、VOLL=9,000，混用了不同历史时期。[ERCOT ORDC report](https://www.ercot.com/files/docs/2024/10/31/2024-biennial-ercot-report-on-the-ordc-20241031.pdf)
- 山东官方 2023 信息写的是申报价格下限 -0.08 元/kWh、上限 1.3 元/kWh，并说明可适时调整。材料其他位置又出现 -100/1500 等口径，必须按规则版本锁定。[山东省能源局](https://nyj.shandong.gov.cn/art/2023/6/3/art_122995_26321.html)

因此实现需要“规则版本注册表”，不能给每个数据集只配置一个永恒常量。

### 4.7 R1–R3 的当前实验不能证明机制创新

- R1 若只比较 BECH 与 raw BASE，增益可能完全来自免费投影；必须加入 `Projected-BASE`。
- R2 中“分位 clamp 打平 admin clamp → P1/P3 被证伪”逻辑错误。P1 是代数事实，不会被实验打平证伪；打平只能说明该机制没有实用增益。
- R3 在不同市场重复验证 `base<floor ⇒ residual>0` 只是在重复验证同一恒等式，不构成跨市场泛化证据。
- 必须先报告每个 `market × base × year` 的越界样本数。若越界率接近 0，C1 无法承担论文贡献。

### 4.8 C1 的处理决定

**降级为 `RuleProjection` 合法性基线/安全护栏，不列贡献。** 推荐实现：

\[
\hat y^{BASE-proj}_t=Proj_{C_{m,p,t}}(\hat y^{BASE-raw}_t),

\]

BECH 的所有校正增益相对 `BASE-proj` 计算；raw BASE 只作为附加对照。P1/P2 可放附录作为实现正确性说明，不给方法名 APB-FC，不写“领域机制创新”。

---

## 5. 对 A 路候选的复审

### 5.1 A1/A2/A3

14a 的淘汰正确：

- A1 是带预算 OCO 的任务例化；已有工作直接研究未知线性预算约束、部分反馈、零累计约束违反。[Safe and Efficient OCO](https://arxiv.org/abs/2412.03983)
- A2 可拆成 decision-aware conformal/utility routing 与功效学习；[Utility-Directed Conformal Prediction](https://arxiv.org/abs/2410.01767) 和 [Learning Metrics that Maximise Power](https://arxiv.org/abs/2402.03915) 已覆盖两块核心对象。
- A3 是 RCPS/覆盖约束换风险对象。

这些候选不应复活。

### 5.2 A4 不是“不动点”，且两个单调性都不成立

材料把

\[
n_{acted}(\pi(\lambda))\ge n_{min},\qquad
R_0^{acted}(\pi(\lambda))\le\alpha_0

\]

称为“动作—功效自洽不动点”。但这只是两个可行性约束的交集，没有给出映射 \(F(\lambda)\) 和方程 \(\lambda=F(\lambda)\)，不属于不动点问题。

此外：

- 如果 \(\lambda\) 表示校正幅度而非触发阈值，`n_acted` 甚至可能完全不随 \(\lambda\) 变化；
- 即使动作数量增加，新增的是低风险样本时，acted 条件风险可能下降，因此 \(R_0^{acted}\) 不必单调增加；
- 检验功效不仅由样本量决定，还依赖效应量、方差、依赖结构和多重性分配，`n_min(alpha)` 不是充分定义。

因此 A4 不是“差两块证明”，而是对象本身需要重定义。建议改称 **auditability–risk feasibility frontier**，只作为诊断曲线，不作为理论候选。

---

## 6. 对 C2/C3 的复审

### 6.1 C2 稀缺定价感知

定位为实证特征假设是正确的，但必须满足：

1. 储备裕度在真实预测 cutoff 前可得，否则是泄漏；
2. 同一特征也喂给基座，证明收益不是“只给校正器更多信息”；
3. ORDC、NEM 与山东的稀缺形成机制不同，不能用 ERCOT 规则为所有市场背书；
4. 必须区分 reserve forecast、realized reserve 与事后 scarcity adder。

它不能列为算法创新。

### 6.2 C3 双尾成因弃权

“负尾比正尖峰更可预测”可以成为很好的实证假设，但当前不能成为普遍理论：

- AP 0.80–0.84 与 AP 0.15–0.25 是项目数据结果，不是机制证明；
- 不同市场、年份和阈值下可预测性可能反转；
- “成因类严格支配统一阈值”需要很强的条件，通常只能通过 held-out Pareto frontier 做经验裁决；
- 金融、交通和气象同样可能有异质尾部生成机制，“双尾成因不同是电价独有”不成立。

建议把 C3 改写成预注册研究假设：

> 在 matched-coverage 条件下，cause-conditioned routing 是否在负尾/正尾 harm–utility Pareto 前沿上严格改善统一路由？

实验若支持，可称领域发现，不称新定理。

---

## 7. 整体主线审查：v5 并入后反而更像拼盘

### 7.1 当前结构

| 部分 | 实际功能 | 与主学习对象的关系 |
|---|---|---|
| 原 v4 signed-tail corrector | 检测、幅值修正、选择动作 | 主体 |
| C1 admin clamp | 输出合法区间投影 | 前/后处理护栏 |
| B1′ e-process | 部署后序贯监控 | 外部审计器 |
| C2 reserve feature | 特征工程 | 输入增强 |
| C3 cause routing | 分组阈值假设 | 可选路由变体 |

这些组件没有形成新的共同优化对象。把它们都写进 contribution list 会形成典型的：领域先验 + 校正头 + 风险工具 + 特征工程 + 监控器拼装。

### 7.2 推荐唯一主线

论文只保留一个方法主语：

> **对冻结电价预测器进行 signed-tail occurrence–magnitude selective correction，并用正常期、负尾、正尾的 base-relative harm 约束决定是否执行动作。**

其他对象全部降级：

- RuleProjection：合法性预处理基线；
- 风险控制：采用已知 RCPS/LTT/CSA 风格的正确工具，除非真的提出新定理；
- reserve/cause：领域消融；
- S1–S5：实验协议；
- BASE fallback：实现属性。

如果该唯一主线仍无法在“算法对象”上区别于 CRC/δ-Adapter/PIR/CSA，那么论文就必须依赖新的 benchmark、极端事件评估协议和跨基座系统证据，而不是继续制造数学小模块。

---

## 8. SOTA 划界最终裁决

| 宽主张 | 裁决 | 允许的窄表述 |
|---|---|---|
| 首次冻结基座后处理校正 | ❌ | 不允许；PIR、δ-Adapter、CRC、COSA、ARC-STAR 已覆盖 |
| 首次风险感知 act/abstain 校正 | ❌ | 只能说应用于 signed electricity tails |
| 首次逐阈值 e-process 选择性动作 | ❌ | CSA 已直接覆盖 |
| 首次联合风险、接受率和效用证书 | ❌ | SCORC 已覆盖 |
| 首次行政边界约束预测 | ❌/证据不足 | 只称使用 market-rule projection，不称首次 |
| 首次双尾 occurrence–magnitude 电价后校正 | ⚠️ | 只有在完整系统交集查新与实验均成立后可作应用方法主张 |
| 新 e-process 定理 | ❌ | 当前公式错误；重写后也先视为已知序贯工具的例化 |
| 新领域机制 C1 | ❌ | P1/P2 是标准投影事实，作为附录/护栏 |

最危险的近邻不再只是 δ-Adapter/CRC，而是：

- [CSA](https://arxiv.org/abs/2605.20270)：选择性动作 + e-process + anytime pathwise；
- [SCORC](https://arxiv.org/abs/2606.08517)：risk + acceptance + utility 联合证书；
- [Anytime-Valid CRC](https://arxiv.org/abs/2602.04364)：随校准流增长的 anytime-valid 风险控制；
- [ARC-STAR](https://arxiv.org/abs/2605.22222)：冻结宿主、预算感知、post-hoc correction 与 triage。

v5 已列出这些论文，却没有把它们对 B1′ 的直接碰撞反映到最终裁决中。

---

## 9. 数据承诺审查：当前仍是致命 P0

### 9.1 必须锁定的标签语义

每个数据集必须给出：

| 字段 | 必填内容 |
|---|---|
| 市场与产品 | DA/RT、能量/辅助服务、节点/分区/机组 |
| 标签定义 | clearing price、settlement price、unit transaction price 或其他 |
| 时间粒度 | 5/15/30/60 分钟；24/48/96 点 |
| 可用时点 | 每个特征在预测 cutoff 前何时发布 |
| 规则版本 | 每个时间段适用的 floor/cap 与生效日期 |
| 单位 | 元/kWh、元/MWh、AUD/MWh、EUR/MWh 等 |
| 价格修正 | 市场事后 price correction、uplift、admin intervention 是否计入 |
| 缺失/裁剪 | 数据提供方是否已做 clip、四舍五入或地板编码 |

此前已确认的山东 96 点字段是 unit-level `epf_unit_data_96.da_cq_price/rt_cq_price`。在证明它等于受相同 floor/cap 约束的市场级清算标签以前，不能把山东写成 C1 的规则证明数据集。

### 9.2 必须先做的零成本统计

在任何训练前输出：

1. 每个 `market × year × product × base` 的 raw BASE 越界率；
2. true label 超出所配规则边界的计数；
3. 恰等于 floor/cap 的样本数；
4. negative/positive/normal 与 acted-normal 的有效样本量；
5. 每组可实现的最窄置信上界；
6. 规则变更日前后的数据切片；
7. raw BASE 与 Projected-BASE 的免费增益。

若 C1 触发样本不足，立即删除 C1 实验线；不要通过跨市场合并掩盖单市场无功效。

### 9.3 数据一票否决条件

任一成立即停止论文主实验：

- 标签不是所引用规则约束的价格对象；
- 使用了事后 reserve/realized scarcity 特征；
- 规则边界用错年份或产品；
- 负价被数据提供方预先截断但未披露；
- 正/负尾组样本不足以给出非空置信结论；
- 跨市场单位或时间粒度混用；
- 山东私有字段无法公开说明和复现。

---

## 10. 六份文件逐份评分

| 文件 | 严格评分 | 核心意见 |
|---|---:|---|
| 14a 新算法对象 | **6/10** | 最诚实的一份；A1–A3 淘汰合理，但 A4 不是不动点且单调性无一般保证 |
| 14b 时序安全理论 | **1/10** | 核心乘积不是 e-process；Ville 推导错误；mixing 推导仅存在于摘要自述 |
| 14c 领域机制 | **3/10** | 有价值的领域事实池，但 C1 被严重抬高；方向写反、边界过时、与负价漏报问题错位 |
| 11 候选池 | **4/10** | 汇总清楚，但把上游错误 B1 直接标为可证明，未做公式复验 |
| 12 碰撞矩阵 | **5/10** | e-BH/FWER 纠错正确；但只检查候选外部碰撞，没有检查候选自身定理成立性 |
| 13 v5 方法设计 | **3/10** | NOT READY 标签正确；最终保留对象却均不合格，并新增 BASE、方向和 mixing 事实矛盾 |

### 多智能体流程暴露出的系统性问题

当前流程擅长“候选生成→表格淘汰”，但缺少两个强制门：

1. **原子定理复验门**：汇总 Agent 必须从定义重新算一遍条件期望，而不是看到 “Ville + betting” 就接受。
2. **源报告一致性门**：摘要声称有推导时，必须定位正文定理编号和公式；找不到就标红，不得向下游传播。

本轮 B1′ 正是两个门同时失效的结果。

---

## 11. 必须删除、暂停和允许保留的主张

### 11.1 必须删除

- “B1′ 在 i.i.d. 下已给出有限样本 anytime-valid 定理”；
- “\(\prod(1+\lambda[H-\delta]_+/M)\) 是非负鞅/超鞅”；
- “e-process 保证累计 realized harm 从不超过预算路径”；
- “14b 已推导 β-mixing 的 \(O(1/n)\) 修正”；
- “C1 是唯一够格的领域机制创新”；
- “base<floor 时负向校正、base>cap 时正向校正”；
- “clamp 在 λ=0 时对越界 raw BASE 仍 bit-exact”；
- “P3 fill-to-bound 是新的解析最优机制”；
- “跨市场重复验证 P1 可证明机制泛化”。

### 11.2 在重写定理前暂停

- `finite-sample harm-budget certificate`；
- `anytime-valid safe correction`；
- `pathwise budget guarantee`；
- `action-conditional e-process`；
- `certified rollback/abstention`；
- `β-mixing correction`；
- `new theoretical framework`。

### 11.3 可以保留

- “v5 未找到不可替换的新算法对象”；
- A1/A2/A3 的淘汰结论；
- e-BH 不适合 FWER 主张；
- C2 是实证特征假设；
- C3 是待裁决的 cause-conditioned routing 假设；
- 规则边界投影作为合法性基线；
- 总体 `NOT READY`。

---

## 12. 下一版强制重构顺序

### P0-A：撤回错误理论

从方法正文、候选池和碰撞矩阵中删除 B1′ 的“可证明”状态。若重做，先提交一页 theorem sheet：

- filtration；
- 零假设；
- 中心化增量；
- betting factor 的非负性；
- 条件超鞅证明；
- 越界事件与 crossing event 的包含关系；
- 延迟标签；
- 组联合控制；
- 与 CSA 的逐项差异。

任何一项为空都不得恢复理论主张。

### P0-B：修复动作语义与 BASE

统一定义：

- `TAIL_NEG/TAIL_POS/NORMAL`：真实或预测事件状态；
- `DELTA_UP/DELTA_DOWN`：校正方向；
- `BASE_RAW/BASE_PROJ`：两种基线；
- `ACT/ABSTAIN`：是否执行校正。

不得再用 NEG/POS 同时表示事件尾和增量方向。

### P0-C：建立逐时规则注册表

使用 `market × product × effective_start × effective_end × floor × cap × source`，所有数据样本通过时间连接规则。未匹配样本停止运行，不允许用最近年份常量补齐。

### P0-D：完成数据功效普查

在看到每组样本量、越界率和最窄可达 CI 前，不允许承诺 9 市场或任何“零退化”。

### P1：选择论文路线，不再两头占

#### 路线 T：A 类 ML 理论方法

需要真正的新对象，例如同时处理：

- outcome-defined signed groups；
- adaptive correction/abstention；
- delayed labels；
- dependent/nonstationary time series；
- base-relative group harm；
- 非平凡 release/utility。

并给出区别于 CSA/SCORC/RCPS 的定理。工作量显著高于“补一个公式”。

#### 路线 E：严格实证的领域方法

不主张新 e-process 或新投影理论，把贡献放在：

- 电价双尾 occurrence–magnitude 问题定义；
- 多基座、跨市场、严格样本外的选择性后校正 benchmark；
- normal/negative/positive 三组 harm–utility Pareto；
- 可复现事件级指标与失败地图。

这条路线更现实，但投稿语境应转向 KDD/WSDM 应用轨或高水平能源/预测期刊，而不是以通用 ML 理论为卖点。

---

## 13. 最终审稿意见

### 最终票

**Reject / Major Rebuild。**

不是因为作者没有堆出足够多模块，而是因为：

1. 唯一保留的理论候选在第一步就不满足 e-process 定义；
2. 唯一保留的领域候选只修复越界预测，不修复真正的负价漏报，并与 bit-exact BASE 自相矛盾；
3. 2026 近邻已经覆盖选择性 e-process、联合 risk/acceptance/utility 和 anytime-valid CRC，剩余划界比 v5 判断的更窄；
4. 跨市场规则常量和山东标签语义仍未冻结，任何行政边界实验都可能建立在错误标签上。

### 对六个问题的最终回答

1. **真创新？** v5 新增候选中没有。原 v4 的 signed-tail selective correction 仍是唯一可能主线，但尚未证明达到 A 类算法创新门槛。
2. **一票否决风险？** B1′ 错误定理、BASE 恒等性冲突、数据规则语义，任一都足以一票否决。
3. **如何强化？** 先做减法和定义修复；随后只选理论路线或实证路线之一。
4. **贯穿还是拼盘？** C1+B1′ 并入会加重拼盘；应恢复一个主方法，其余均作基线、护栏或消融。
5. **划界成立吗？** 只有“电价 signed-tail base-relative harm 的领域化”这一窄边界可能成立；通用选择性风险与序贯控制边界不成立。
6. **数据承诺致命吗？** 是。未锁定标签、规则版本和有效样本量前，不应启动大规模实验。

> 下一次提交不需要更多候选池。只提交：①修正后的 theorem sheet 或明确放弃理论路线；②规则注册表与标签契约；③BASE_RAW/BASE_PROJ/动作方向的统一定义。三份材料通过后，才值得继续设计实验。
