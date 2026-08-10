# BECH v2 第二轮严格审稿裁决

**评审日期：**2026-08-07  
**评审对象：**《04 · BECH v2 审稿提交包》  
**裁决状态：**设计级审稿完成；逐公式证明审查未完成  
**总体结论：**Major Revision / 当前仍属 Weak Reject 设计，尚不可进入大规模实验

---

## 0. 材料完整性与本轮边界

本轮实际收到的是审稿请求包，而不是其引用的 `03_BECH_统一方法设计v2.md`。后者包含 §3–§10 的数据流、公式、命题、假设、claim-to-evidence 和实验协议，但未随附件提供，具名检索也未找到对应正文。

因此本报告区分两类结论：

- **可以直接裁决：**贡献是否独立、近邻划界、保证口径中的方向性和多重性错误、哪些 claim 必须删除；
- **必须看到全文才能裁决：**每个统计量的精确定义、定理证明是否正确、S1–S5 时间边界的逐变量无泄漏性、代码与公式是否一致。

这不是形式问题。没有 §5.6 和 §6 原文，就不能给“风险保证数学成立”盖章。本轮只能判定：**摘要中所写的保证尚未成立，且至少有四处需要先改定义。**

---

## 1. 相对上一版的变化审计

| 上一轮 P0 | v2 当前状态 | 裁决 |
|---|---|---|
| GAFE 被抬为主创新 | 本包未再把 GAFE 放入三项贡献 | **已明显改善** |
| C1 特征契约被当算法创新 | 改写为 leak-safe evidence / exact fallback | **换了包装，但“协议当 novelty”仍未解决** |
| SCARR 被写成通用动作风险首创 | 已承认 RCPS/LTT 占据风险控制框架 | **改善，但仍需明确只是例化** |
| AP 高/低手工决定分支 | 改成 H-based utility mask 与有限策略族 | **方向更好，但 router 仍是 LOCO 式实现** |
| select/certify 数据复用 | 声明 S3/S4/S5 分离 | **原则上改善，须全文核查冻结边界** |
| 正常期 hard budget 缺失 | 出现 C0 normal 组 harm | **恢复，但风险分母可能掩盖稀疏触发伤害** |
| occurrence–magnitude 核被弱化 | 本包只说 hurdle 已有、未展示完整 two-part 定义 | **待全文确认，不能视为关闭** |
| 时间序列仍使用 i.i.d. 保证 | 改称块近似可交换 / mixing 近似 | **诚实度提高，但不再支持 exact certificate** |
| 山东 96 点目标语义未冻结 | 本包未处理数据语义 | **仍为 P0，未关闭** |

**总体进步：**从“模块命名驱动”转向“统一损失与策略风险驱动”，方向是对的。  
**总体问题：**仍然试图把良好研究规范、特征选择实现和通用风险框架的领域例化拆成三个 novelty。

---

## 2. 裁决①：三项贡献是否真实且非显然组合

### 结论

**不成立为三项独立贡献；作为一个方法的三段实现，可以成立。**  
“由同一个 \(H_t(\pi)\) 和策略族 \(\Pi\) 连接”能提高叙事一致性，但不能自动证明 novelty，也不能证明组合非显然。

### 理由

1. **leak-safe + exact fallback 是必要协议，不是新方法。**无泄漏和固定基座回退属于可信实验与安全设计的基本要求。
2. **utility mask 是状态分层的 removal utility / LOCO 实现。**它可以帮助构造策略，但尚无独立估计目标或新统计理论。
3. **独立风险层是 RCPS/LTT 类风险控制的领域化使用。**改变被认证对象，不自动创造新的风险控制框架。
4. 多个模块都使用同一损失是合理工程设计；大量 pipeline 都可以共享一个任务损失，审稿人不会因此认为模块不可分割。

“未发现同一组合”只说明没有检到完全同构实现，不等于组合非显然。审稿人会问：给定“最大化尾部改善、约束正常期伤害”的目标后，LOCO 选特征、有限策略搜索、held-out 风险检验是否是自然组合？当前答案偏向“是”。

### 最弱、最可裁剪项

- **最应撤销独立 novelty：utility router。**
- **最不应写成贡献：leak-safe / exact fallback。**它们必须保留在协议中，但不能占贡献编号。

### 建议重组

不要写“三项方法创新”，改成：

1. **任务贡献：**冻结电价基座上的 signed-tail, base-relative post-hoc correction；
2. **唯一方法贡献：**在有限动作策略族内联合构造双尾候选并进行 normal/negative/positive 分组风险约束；
3. **实证贡献：**严格 rolling-OOS、跨基座/市场的效用—伤害—回退评估。

leak-safe evidence、utility mask、fallback 和 risk test 都是第二项方法的组成部分。

### 若要补证据

把所有组件从同一个约束优化问题推导出来，而不是事后用 \(H\) 串联：

\[
\max_{\pi\in\Pi} U_-(\pi)+U_+(\pi)
\quad\text{s.t.}\quad
R_0^{\mathrm{harm}}(\pi)\le\varepsilon_0,
\ R_-^{\mathrm{harm}}(\pi)\le\varepsilon_-,
\ R_+^{\mathrm{harm}}(\pi)\le\varepsilon_+.
\]

随后证明或实证说明 mask、动作强度和 fallback 分别改变哪个可行域或目标，而不是仅提高平均指标。

---

## 3. 裁决②：utility router 是否构成独立方法对象

### 结论

**不成立。建议立即撤销独立贡献身份，并入候选策略生成。**

### 最强理由

当前自述已经把它还原为：

> 固定模型移除效用 + 状态分层 + 0/1 mask + 用任务 harm 代替普通预测损失。

这属于 state-conditional LOCO / task-aligned feature selection。跨状态 mask 不同，是分层建模的预期结果；用 \(H\) 作为效用，是把已有 removal importance 换成下游任务损失；0/1 而不是连续权重，是参数化选择。三者均不足以单独支撑方法 novelty。

[CPI](https://arxiv.org/abs/1901.09917) 定义的是依赖有效 knockoff 的条件预测影响，并给出条件独立检验；你们若仅把特征置零/移除后测 \(H\) 差异，即使再用 Holm，也没有解决：

- 相关特征替代性；
- 移除后样本落到训练流形之外；
- removal effect 是否为 conditional importance；
- 时间依赖下 p-value 是否有效。

因此它不是 CPI 的并列条件重要性新方法，更接近**按状态分层的 LOCO heuristic**。Holm 只能控制一组有效 p-value 的多重性，不能把无效检验变有效。

### 若坚持独立，最低门槛

必须同时补齐：

1. 明确定义新的 estimand，例如
   \[
   I_{j,s}(\pi)=
   \mathbb E[H(\pi_{-j})-H(\pi)\mid S=s],
   \]
   并说明它与 LOCO/CPI 的数学差异；
2. 给出相关特征和时间依赖下的一致性、误差控制或 policy-regret 结论；
3. 对照 all-feature、random mask、LOCO、conditional permutation/CPI、learned soft mask；
4. 用独立市场证明跨状态异质性不是选择集噪声；
5. 证明 router 不只提高拟合，还扩大通过风险认证的非平凡策略集合。

即使完成这些，它也更适合作为**次级方法贡献**，不应与 signed-tail safe correction 平级。

---

## 4. 裁决③：风险保证是否数学成立

### 总结论

**当前摘要不足以成立；存在一个明确方向错误、一个联合控制缺口、一个时间依赖口径冲突和一个稀疏触发分母漏洞。经重写后有可能成立，但现在不能称 certificate。**

### 4.1 S3/S4 分离能否排除选择偏差

**结论：有条件成立。**

只有在进入 S4 前，下列对象全部冻结，S3/S4 分离才足够：

- 唯一待认证策略 \(\hat\pi\)；
- 尾阈值、组定义、动作阈值、\(\lambda\)；
- harm loss、裁剪范围、\(\delta,\alpha,\beta\)；
- block 长度与 bootstrap 方法；
- 所有 subgroup 与报告指标。

如果在 S4 上同时认证多个 \(\pi\in\Pi\)，再从通过者中挑最好策略，就重新产生选择问题，必须使用 LTT/多重检验或再留一层独立认证数据。[LTT](https://arxiv.org/abs/2110.01052) 本身就是把候选选择写成多重检验问题。

**最稳方案：**S3 只输出一个 \(\hat\pi\)，S4 只对它做 accept/reject，失败就 BASE，不在 S4 选择替代策略。

### 4.2 效能置信界方向写反或定义不清

若

\[
H_t(\pi)=\ell_t(\pi)-\ell_t(\mathrm{BASE}),
\]

且目标是

\[
\mathbb E[H_t(\hat\pi)\mid G_g]\le-\delta_{\mathrm{efficacy}},
\]

那么应当证明：

\[
\mathrm{UCB}_{1-\beta}\bigl(\mathbb E[H\mid G_g]\bigr)
< -\delta_{\mathrm{efficacy}}.
\]

不是检查 \(H\) 的 LCB。若坚持 LCB 写法，必须改定义为 improvement \(U=-H\)，并证明：

\[
\mathrm{LCB}_{1-\beta}\bigl(\mathbb E[U\mid G_g]\bigr)
> \delta_{\mathrm{efficacy}}.
\]

这是定义级问题，不是符号美观问题。全文必须统一，否则命题方向会错。

### 4.3 安全风险应使用什么统计口径

对固定组、固定策略，令

\[
Z_{t,g}=\mathbf 1\{H_t(\hat\pi)>\delta_{g}^{\mathrm{harm}}\}.
\]

目标是控制

\[
R_g^{\mathrm{exc}}(\hat\pi)
=\Pr(Z_{t,g}=1\mid G_t=g)\le\alpha_g.
\]

这首先是 Bernoulli exceedance-risk 的上置信界问题。若样本独立，可用 exact binomial / Clopper–Pearson 或 Hoeffding–Bentkus UCB。称它为 split-conformal 并不会自动增加有效性，反而混淆“预测分位数覆盖”与“固定策略风险检验”。[RCPS](https://arxiv.org/abs/2101.02703) 与 LTT 已提供更一般的风险控制框架。

### 4.4 块近似可交换不能支撑“精确有限样本”

逐时电价残差具有：日内相关、周周期、季节性、制度变化和突发结构断点。把连续块称为“近似可交换”是工作假设，不是可观察事实。

[SPCI](https://proceedings.mlr.press/v202/xu23r.html) 明确指出很多普通 conformal 方法不适用于非交换时序；其时序结果也是针对专门序贯方法建立。[Exact and Robust Conformal Inference for Time Series](https://proceedings.mlr.press/v75/chernozhukov18a.html) 在交换情形可 exact，在交换性失效时需要近似有效性分析。

因此：

- 若只做 block bootstrap：最多称“在平稳/mixing 与 block 条件下的渐近置信”；
- 若能给显式 mixing coverage/risk gap：可称“带误差项的近似保证”；
- 若没有可计算误差界：只能称“held-out empirical risk audit”；
- 不能同时写“block approximate exchangeability”与“distribution-free exact certificate”。

### 4.5 安全与效能必须联合控制

**需要。**假设每一条约束都以 \(1-\beta\) 单独成立，并不能推出全部同时以 \(1-\beta\) 成立。

若共有 \(m\) 个检验，包括负尾安全、负尾效能、正尾安全、正尾效能、正常组安全，以及多个 horizon/action，则必须：

- 预分配 \(\beta_j\)，满足 \(\sum_j\beta_j\le\beta_{\mathrm{total}}\)；或
- 构造联合置信域；或
- 在 p-value 有效的前提下使用 Holm/LTT 等 family-wise procedure。

安全与效能统计量相关并不会自动消除多重性。最保守的 union bound 已足以给出合法但可能较宽的结果。

### 4.6 C0 稀疏触发存在分母掩盖问题

只控制

\[
\Pr(H>\delta_0\mid G=0)
\]

可能被低触发率轻易“稀释”：若 99% 正常点都返回 BASE，即使少量被校正的正常点伤害很大，按全体 normal 作分母仍可能通过。

至少同时报告或控制：

\[
R_{0}^{\mathrm{population}}
=\Pr(H>\delta_0\mid G=0),
\]

以及

\[
R_{0}^{\mathrm{acted}}
=\Pr(H>\delta_0\mid G=0, A\ne\mathrm{BASE}).
\]

第二项样本不足时不能宣称 acted-normal certificate；应回退到更粗分层、合并 horizon，或只报告带宽置信区间和“证据不足，因此 BASE”。

### 4.7 其他必须写进命题的条件

- \(H\) 是否有界；若用 Hoeffding–Bentkus，需预注册 loss clipping/bounds；
- 尖峰导致 \(H\) 重尾时，均值 bootstrap 是否稳定；
- \(G_g\) 是真实事后组还是运行时预测组；真实组不得用于当前动作路由；
- BASE 是否真的恒等返回同一原预测；任何投影、clip 或缩放都会使 \(H(\mathrm{BASE})\neq0\)；
- group 阈值必须在 S4 前冻结；
- 同一日 24/96 个 horizon 是否作为独立样本，若不是，应以日/周块为统计单元。

### 风险层最终裁决

> **当前允许写“独立 held-out 策略风险评估设计”；暂不允许写“有限样本安全证书”“exact certificate”或“certified abstention”。**

只有修正 UCB/LCB、联合控制、依赖假设和 normal acted-risk 后，才能重新评估“certificate”一词。

---

## 5. 裁决④：整体是否贯穿而非拼盘

### 结论

**比上一版更贯穿，但“同一个 \(H\)”尚不足以摆脱拼盘攻击。**

### 最可能的审稿攻击路径

1. **事后粘合：**先选了 LOCO mask、策略搜索和风险检验，再用同一个 \(H\) 统一符号；
2. **可替换性：**router 换成 MLP attention/all-features，风险命题完全不变，说明它不是核心；
3. **通用 wrapper：**认证层可包在任何候选 policy 外，未体现 BECH 特有理论；
4. **安全的空解：**BASE 永远可行，方法可能靠高 abstention 获得漂亮安全率；
5. **双尾仅是分组：**若负尾/正尾只是两个 group key，而非不同 occurrence–magnitude 机制，结构不对称会被视为标签拆分；
6. **单一损失不足：**大量多阶段模型都用同一 loss，不能据此证明非显然组合。

### 防御方式

- 取消 router 和 evidence 的独立贡献编号；
- 从一个 signed-tail constrained policy problem 推导全部模块；
- 明确负/正尾候选生成为何不是简单 group duplication；
- 把 non-trivial coverage 写成与安全并列的目标，报告 correction rate、无可行策略率和 BASE 退化；
- 做“替换组件”消融：all features、LOCO mask、utility mask、learned mask 在相同风险层下比较；
- 证明或展示 utility mask 增加了可认证的非 BASE 策略，而不只是降低训练 loss；
- 风险层作为方法约束，不另立通用理论首创。

### 最终贯穿主线

> **在冻结电价基座上，构造 signed-tail correction policy，并在 normal/negative/positive base-relative harm 的联合约束下选择是否及多大程度执行。**

只要全文始终围绕这个 constrained policy problem，整体可以成为一篇论文；若继续坚持“evidence、router、certificate 三创新”，仍会被判拼盘。

---

## 6. 裁决⑤：与近邻工作的边界

### 6.1 PIR

**结论：差异足以区分具体问题，不足以宣称 post-hoc selection/revision 新颖。**

[PIR](https://openreview.net/forum?id=H7e5RpeIi4) 已占据模型无关的失败实例识别与事后修订。BECH 可保留的差异必须同时包括：绝对电价正/负 signed events、occurrence–magnitude 候选、base-relative 三组 harm 与独立策略认证。只写“双尾 + 保证”过于抽象；必须在同一冻结基座、同一信息集下实现 PIR 基线。

### 6.2 δ-Adapter

**结论：有可辩护差异，但文档对 δ-Adapter 的描述需要纠正。**

[δ-Adapter](https://arxiv.org/abs/2601.20280) 已占据冻结基座、架构无关输入/输出适配、稀疏 mask、输出残差修正和 conformal calibrator。其 \(\delta\) 是有界编辑尺度，并在平滑性与方向对齐条件下给局部 descent/drift 结论；不宜简化成“逐样本 trust-region 安全保证”。

BECH 的剩余边界是：**不是限制改动幅度，而是对执行 correction policy 相对 BASE 的 signed-group excess harm 作独立风险验收。**这足以区分领域化方法，但不足以声称“首次安全校正”或“首次冻结适配”。必须把 Ada-Y / Ada-X+Y 作为直接基线，并比较相同 correction coverage 下的 tail utility 与 normal harm。

### 6.3 CRC safe residual correction

**结论：碰撞很强；只有完整窄交集能划开。**

[CRC](https://arxiv.org/abs/2512.22428) 已提出 plug-and-play residual correction、局部选择、裁剪/回退和 non-degradation 机制。仅凭“事件分支、弃权、认证”不够，因为 CRC 已在语义上决定何处/如何安全修正。

必须固定差异为：absolute-price signed occurrence–magnitude、normal/negative/positive group-specific action harm、独立时间认证。若 two-part 或正常组硬预算缺失，CRC 会几乎覆盖 BECH 的方法主张。

### 6.4 RCPS / LTT

**结论：当前是框架例化，不是新风险控制方法。**

[RCPS](https://arxiv.org/abs/2101.02703) 面向一般 bounded risk；[LTT](https://arxiv.org/abs/2110.01052) 面向一般预测算法与候选超参数的风险检验。把对象换成动作策略、再拆成两个尾组，仍属于一般风险函数与多约束实例。

可以安全声明“将 LTT/RCPS 风险控制思想实例化到 signed-tail correction policy”；只有给出 signed group × action × dependent time series 的新有限样本结论，才可声称方法学扩展。

### 6.5 CPI / LOCO

**结论：当前 router 是 LOCO 家族的状态化应用，不是 CPI 并列方法，也不应声称 conditional importance。**

CPI 的关键不是用了一个不同检验名，而是通过有效 knockoff 处理条件特征依赖。移除差分 + Holm 在高度相关的电力特征上可能把替代特征判成“不重要”，也可能产生 off-manifold 输入。建议命名为 `state-conditional removal utility mask`，明确它是 policy construction heuristic，而不是条件独立推断。

### 6.6 COSA

**确切出处：**[COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting](https://openreview.net/forum?id=L7Z5wBMPrW)，ICLR 2026。

**结论：与通用 frozen-output correction + gating 直接碰撞；不覆盖 signed-tail group risk。**

COSA 用最近已观测真值的上下文统计构造线性残差修正，并以可学习 gate 控制修正强度，保持基座冻结和架构无关。因此 BECH 不能再宣称 output-space correction、门控或回退接口新颖。剩余差异仍是：事件 occurrence–magnitude、base-relative 三组风险对象与独立 policy certification。

### 边界总裁决

| 近邻 | 是否完全覆盖 BECH | BECH 可保留边界 |
|---|---|---|
| PIR | 否，但占据选择性 revision | signed tails + action harm |
| δ-Adapter | 否，但占据 frozen adapter/mask/conformal | group excess-harm，而非 edit bound |
| CRC | 高度接近 | occurrence–magnitude + 三组 hard budget |
| RCPS/LTT | 覆盖风险控制抽象 | 电价 policy 实例；无新定理不算理论创新 |
| CPI/LOCO | 覆盖 removal importance 家族 | 仅状态化构造，不足独立 novelty |
| COSA | 占据 output gate/adaptation | signed event policy certification |

---

## 7. 裁决⑥：实验前必须删除或收窄的承诺

### 7.1 必须删除

1. **“utility router 是独立方法创新”。**
2. **“leak-safe evidence / exact fallback 是独立创新”。**
3. **“首次 correction-level abstention / certified abstention”。**
4. **“首次模型无关安全校正 / 首次冻结输出安全适配”。**
5. **“使用同一 \(H\) 即证明三部分非拼盘”。**
6. **“未找到完整组合，因此组合非显然/首次”。**
7. **“动作策略风险控制超出 RCPS/LTT 的一般框架”。**
8. **“移除差分 + Holm 等于 conditional importance/CPI”。**
9. **“δ-Adapter 只有逐样本 trust-region，没有风险/校准相关机制”。**
10. **“exact fallback 使整个非 BASE policy 获得确定性安全”。**

### 7.2 在修正数学前必须暂停使用

1. `finite-sample certificate`；
2. `distribution-free safety`；
3. `exact safety under block exchangeability`；
4. `certified abstention`；
5. “安全与效能以 \(1-\beta\) 同时成立”；
6. “块 bootstrap LCB 证明 \(E[H]\le-\delta\)”；
7. “normal group 零退化已认证”；
8. “mixing 假设下仍是 exact guarantee”。

### 7.3 可以保留但必须收窄

| 当前意图 | 安全表述 |
|---|---|
| 策略级风险认证 | “对进入 S4 前冻结的单一策略，在预定义 group/loss 下进行 held-out risk test” |
| 时间依赖保证 | “在明确 mixing/block 条件下的近似或渐近结论”；有显式误差界后再称 guarantee |
| utility router | “state-conditional removal-utility mask，用于构造有限候选策略” |
| exact fallback | “选择 BASE 时预测逐值等于冻结基座，因此动作诱发 harm 为 0” |
| 模型无关 | “接口上不读取基座隐藏层/梯度”；参数跨未见基座需实验 |
| 双尾不对称 | “市场/尾侧条件化候选策略”；不宣称负尾普遍可修、正尾普遍不可修 |
| \(\delta_{efficacy}\) | “预注册最小实质改善阈值”；证据不足时不通过，不降低阈值追结果 |
| C0 normal 风险 | 同时报 population-normal 与 acted-normal harm；后者样本不足就标证据不足 |

### 7.4 必须新增的禁止主张

1. “每个小时点都零伤害”；
2. “true tail group 可在运行时直接用于路由”；
3. “每项 95% 置信即可推出全部约束联合 95%”；
4. “block bootstrap 提供精确有限样本覆盖”；
5. “低触发率下总体 normal harm 小，证明被触发正常点安全”；
6. “Holm 修正解决了相关特征的条件重要性偏差”；
7. “BASE 存在即证明方法非平凡”；
8. “多组/多动作只是增加几个索引，不需要多重性校正”；
9. “同一 S4 可以反复改 block、loss 或阈值直到通过”；
10. “数据未验证前即可将 efficacy 写入方法名或标题”。

---

## 8. 实验前的强制修改顺序

### P0-A · 先改贡献结构

- 一篇论文、一个主方法；
- router 降组件；
- risk control 写成领域化统计层；
- protocol/fallback 不列 novelty。

### P0-B · 再改统计对象

1. 固定 \(H\) 的符号；
2. 把 efficacy 改成 `UCB(H)<-δ` 或 `LCB(-H)>δ`；
3. 定义 population-normal 与 acted-normal 两类风险；
4. 列出所有 group × risk × horizon 检验；
5. 预分配联合 \(\beta\)；
6. 选择唯一依赖假设与相应方法，不混写 split conformal、HB 和 bootstrap。

### P0-C · 再冻结 S1–S5

- S3 输出唯一 \(\hat\pi\)；
- S4 只验收，不调参、不换策略；
- S5 只报告；
- 以日/周块而不是把 24/96 小时无条件当独立样本；
- 所有阈值、clip、block size 在 S4 前固定。

### P0-D · 数据语义门仍未解除

山东 96 点的单位级 `da_cq_price/rt_cq_price` 与市场级 DA/RT 目标仍未证实等价。在业务目标、时间戳和信息截止未冻结前，继续保持对应实现 STOP。

### P1 · 完成后才能跑 G0–G4

最小消融应是：

- G0：BASE；
- G1：普通 residual correction，无 router/认证；
- G2：signed occurrence–magnitude candidates；
- G3：G2 + simple all-feature finite policy + risk layer；
- G4：G3 + utility mask。

这样才能回答 utility router 是否增加**可认证非 BASE 策略**。如果 G4 只比 G3 好一点平均误差、却不提高可行策略率，它应被删除。

---

## 9. 最终裁决与下一份必需材料

### 当前裁决

| 维度 | 结论 |
|---|---|
| 研究问题 | 成立且重要 |
| 三项独立 novelty | 不成立 |
| utility router 独立性 | 不成立 |
| H/Π 贯穿性 | 有帮助，但不足以单独反驳拼盘 |
| 风险保证 | 当前未成立；定义和联合控制需重写 |
| 与 δ-Adapter/CRC 的边界 | 窄边界可成立，宽边界不成立 |
| 是否可以开始大规模实验 | **否** |
| 是否可以继续设计修订 | **是** |

### 最终审稿意见

> v2 的正确进步是：开始把方法写成“有限 correction policy + base-relative risk”，而不是继续堆 GNN、特征契约和通用共形名词。但它仍试图从协议、LOCO 掩码和通用风险测试中拆出三项 novelty。更严重的是，当前安全层混用了不同统计工具，却没有先固定一个一致的风险定理。

最稳妥的下一版不是加公式，而是先完成三次减法：

1. 删除 utility router 的独立创新身份；
2. 删除 exact / distribution-free / certified 等尚未被依赖时序理论支持的词；
3. 删除三创新叙事，恢复“一主方法 + 一内部风险层”。

完成后，再提交 `03_BECH_统一方法设计v2.md` 全文，重点包含：

- \(H\)、组、动作、事件状态和残差方向的逐项定义；
- §5.6 所有置信界公式；
- §6 命题与完整假设；
- S1–S5 每段允许读取的数据；
- S4 同时检验的完整 family；
- C0 population/acted 两种风险；
- BASE 恒等性的实现定义。

在该正文缺失前，本轮不能给“数学保证成立”的最终通过票。
