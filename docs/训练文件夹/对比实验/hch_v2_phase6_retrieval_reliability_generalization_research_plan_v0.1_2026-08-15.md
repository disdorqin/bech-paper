# HCH-v2 Phase 6：检索可靠性、跨市场泛化与概念漂移研究计划 v0.1

**日期：** 2026-08-15  
**状态：** 研究裁决与实验设计；本文件不修改生产代码、不切换默认数学实现  
**适用仓库：** `disdorqin/bech-paper`  
**当前权威法证依据：** [Phase 5A-B 审计提交](https://github.com/disdorqin/bech-paper/commit/a47d9beed1de8686da0b0505987b232af9710fd8)  
**数学依据：** `hch_v2_iah_crps_final_math_core_v0.3`

---

## 0. 先给结论

### 0.1 本轮正式撤回的方向

此前文档提出的 **dual-geometry / MFAV / Phase 5C** 不能继续作为下一步主线。最新法证结果已经证明：

- 15/15 cell 可逐位复现；
- `U2 ≡ U3 / S_d` 恒等式全部成立；
- 执行动作上的 asinh 收益与 raw MAE 方向一致；
- Shandong PatchTST 的“退化”来自把 E2 与 E0(W1) 比较，而不是 E2 与宿主比较；
- `day_flip=0`（Shandong），小时翻转极少，且没有 Jacobian 集中现象。

因此本项目**不改动作价值几何、不新增 MFAV、不引入第二套 LCB**。旧文件 `hch_v2_phase5_dual_geometry_action_value_research_direction_v0.1_2026-08-15.md` 仅保留为被证伪的研究记录，不得作为 AI 的实现依据。

### 0.2 现在真正的问题

当前问题是：

> 在同一 frozen universal core、同一候选与同一 DVG 下，E2 的 context-only 检索为什么有时选不到 E0(W1) 能选到的高价值邻居，尤其在 Shandong-PatchTST 上执行日更少、单日 raw gain 略差？

这属于**检索证据质量/日选择质量**，不是候选分布是否学习到、不是 raw/asinh 度量不一致，也不是“代码没有执行修正”。

### 0.3 下一条可检验主线

暂定名称：**Anchor-Preserving Reliability-Weighted Retrieval（APR）**，中文可称“保锚点的证据可靠性加权检索”。

它不是已经成立的新颖性声明，只是下一轮最小改动假设：

1. W1 作为安全锚点永远保留；
2. context 只作为附加证据，不再默认等权替代 W1；
3. 可靠性由结果前可计算的连续证据统计决定，而不是 market/host ID、硬阈值或事件分类器；
4. IAH-CRPS、三动作、双事件提案、整日动作价值、LCB 门控全部不变。

先做邻居差异分解；只有分解确认“相似但不可行动”是主因，才运行 APR 单轴实验。

---

## 1. 目前证据的统一解释

### 1.1 既有实验应如何合并解读

| 实验 | 已证实事实 | 对下一步的约束 |
|---|---|---|
| Phase 3 D2 | 40 个国内迁移 cell 中 37 个改善，宏平均 MAE `109.3→91.3`（约 +13.9%） | frozen correction core 确实有跨域可迁移信息；不要先重写候选数学 |
| Phase 3 T4 | 两种梯度重分配轴均被 3 seeds 否证 | 等域采样暂定为训练器局部最优；不再盲搜 gradient weighting |
| Phase 3 T5 | `r=0.15` 近中性，`r=0.30` 污染 LAGO_NP | 更多数据不必然更好；新增域先进入证据层，不能直接污染 universal gradient |
| P2 扩展 E0–E3 | context-only 只有 LAGO_DE 稳定；LAGO_NP action-empty；Shandong 依赖 host | context 相似度不等于动作价值；不能把单市场正例写成普适增益 |
| Universal-core 27 cell | shared core 未坍塌，Case D；瓶颈在检索到动作的日选择 | 先做 paper-config 邻居/动作账本审计，P4 继续封锁 |
| Phase 5A-B | E2 对宿主仍有实际改善；所谓 raw/scale-free 分叉是 E2−E0(W1) 比较伪象 | 不改 MFAV、不改 DVG 几何；直接比较 E2 与 E0 的邻居和执行选择 |

### 1.2 目前不能写入论文的结论

- 不能写“CAVM 已经跨市场普适有效”；只有 LAGO_DE 类型稳定支持。
- 不能写“Shandong PatchTST 被 HCH 退化”；最新审计显示 E2 相对宿主仍改善，退化只相对 E0(W1)。
- 不能写“LCB 过严导致失败”；LAGO_DE 的 no-LCB 并没有带来收益，且 LCB 对安全性有作用。
- 不能写“更多国内数据一定提高泛化”；T5 已给出反例。
- 不能写“universal core 已达到 SOTA”；还没有锁定六基线的正式、同数据、同 split 对比。
- 不能写“检索或 abstention 首次提出”；PIR、TS-RAG、RAFT 等已占据一般性 retrieval-enhanced forecasting 叙述。

---

## 2. 数学与代码硬边界（本轮不得动）

以下约束来自 v0.3 数学核心，下一轮 AI 只能在其外部做诊断或单轴变体：

```text
Frozen Host
  → IAH candidate（三原子：Identity / Down / Up）
  → query-dose replay
  → 双事件提案（至多一个 Down 段 + 至多一个 Up 段）
  → 整日动作价值
  → 单侧 LCB 门控
  → final corrected forecast
```

必须保持：

- 训练目标只有 IAH-CRPS；
- 预测器冻结，校正器 universal core 在 S4 冻结；
- `Identity` 是零修正、零收益锚点；
- 结果前信息边界为 `F_d^-`；S4 不得读取目标日标签；
- asinh 只用于候选的尺度与符号保持，精确 inverse 仍为现有实现；
- target-local S1/S3-M/S3-C 可以存在，但不可伪装成 universal 训练参数；
- 不新增 BCE、MAE/SmoothL1 辅助 loss、tail loss、trading loss、事件检测头、market/host ID gate；
- 不增加 MFAV、第二个 LCB、逐小时共形保证、beta-mixing 或 P4 action-value state update；
- 不以价格 floor、硬极值阈值、手写分支替代学习到的证据可靠性。

### 2.1 必须保留的正确性契约

- E1 `cavm(1,0)` 与 E0(W1) 逐 cell 一致；
- query-dose、候选 bundle、inverse、动作区间 round-trip；
- `U2 ≡ U3 / S_d` 与现有 P5 法证一致（仅作诊断，不改正式价值）；
- S4 target-free；
- 正比例缩放下候选、动作方向、LCB 结果不出现非法变化；
- action-empty 域不使用 `mean_A_true` 冒充动作改善。

---

## 3. 文献与工业方法给出的真实启示

### 3.1 检索相似度不能直接当作未来价值

2026 年的 SARAF 明确指出：非平稳时间序列中，“过去片段很像”不保证未来演化相像，单纯 similarity retrieval 会脆弱且冗余，因此需要同时考虑相关性与多样性。[SARAF](https://arxiv.org/html/2606.04135v1)

这与我们的 E2 结果高度吻合：LAGO_DE 中 context key 有效，但温和 LAGO_NP 或 host 分裂下，检索相似度未必能转化为可执行修正。它支持的不是“再造一个更大的 retriever”，而是先量化：

```text
相似度 → 邻居稳定性/行动一致性 → 真实动作收益
```

### 3.2 Retrieval 本身不是新颖点，关键在融合职责

RAFT 通过从训练集检索相似时间片段改善预测，在 10 个基准数据集上报告较高胜率；TS-RAG 则在冻结 backbone 上训练一个 Adaptive Retrieval Mixer，并用多域知识库做 zero-shot 适应。[RAFT](https://arxiv.org/html/2505.04163v1) [TS-RAG](https://arxiv.org/html/2503.07649v4)

对 HCH 的含义是：

- “我们使用历史检索”不能作为核心贡献；
- HCH 的区别必须落在**冻结后处理、双尾候选、动作剂量回放、整日 LCB 决策**的组合；
- 如果做 APR，应强调“可靠性加权的动作证据融合”，而不是泛称 RAG。

### 3.3 异质多域训练确实会产生负迁移

TFPS 以 pattern-specific experts 应对异质模式演化，说明让一个模型用单一函数覆盖所有模式容易在 regime shift 下泛化变差。[TFPS, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/8491a7fcc218946b471b600a915c8b02-Paper-Conference.pdf)

这与 T5 的 `r=0.30` 温和市场污染相呼应。但 HCH 不应因此引入 host/market 专家门控；更稳妥的工程解释是：

- universal core 只学习跨域共同的候选结构；
- 市场特定的状态、邻居、校准留在 local evidence；
- 以 leave-one-market-out 验证“共享部分”和“本地部分”各自承担什么。

### 3.4 概念漂移下，静态组合会失败；在线反馈应先放在证据层

2025 年电力负荷研究报告指出，概念漂移下传统组合方法可能退化到不如单个组件，在线概率组合可缓解这一问题。[Cao et al., Applied Energy 2025](https://ideas.repec.org/a/eee/appene/v399y2025ics0306261925012486.html)

因此“冻结预测器 + 可累积经验”可以成为 HCH 的长期叙述，但当前应严格拆成两层：

1. **冻结层：** universal IAH core 和 host 均不更新；
2. **经验层：** 目标日标签公布后，按合法时间顺序追加 local memory、更新后续检索与校准统计。

在经验层真正放行前，必须用 prequential/rolling-origin 实验验证，不能仅凭封闭测试集宣称抗漂移。

### 3.5 工业标准是 rolling-origin、可审计和漂移监控

StatsForecast 将时间序列交叉验证定义为滑动窗口训练并预测下一段，适合模拟生产中的不断前移；River 的 Page-Hinkley 实现则以累计偏差检测均值改变，可作为监控而不是动作硬门。[StatsForecast CV](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html) [River Page-Hinkley](https://riverml.xyz/dev/api/drift/PageHinkley/)

对本项目的工程化要求：

- 训练/选择参数只能使用过去窗口；
- 每个测试日输出是否更新 memory、更新前后版本和数据截止时间；
- drift detector 只报警、记录和切分分析，第一轮不直接决定 release/abstain；
- 所有指标按日配对并保留逐日长表。

---

## 4. 新的主线：Anchor-Preserving Reliability-Weighted Retrieval

### 4.1 为什么不是简单的 E3 固定混合

已知：

- E0 = W1-only；
- E1 = `cavm(1,0)`，必须等价于 E0；
- E2 = context-only，收益只在部分市场/宿主稳定；
- E3 = context + W1，能减少部分 opposite，但尚未通过跨市场门禁。

固定 `λ=(1,1)` 的 E3 仍把 context 的可靠性当成恒定，不能解释：

- LAGO_DE 的 context 有效；
- LAGO_NP 几乎没有动作容量；
- Shandong PatchTST 与 Linear 对同一检索机制响应不同。

所以只考虑如下**连续证据融合**，不做身份条件：

\[
g_q^{\mathrm{APR}}
=(1-\lambda_q)g_q^{W1}+\lambda_q g_q^{ctx},
\qquad 0\leq\lambda_q\leq1.
\]

这里的 `g` 可以是现有 W1/context 邻居对 Down/Up 原子证据的聚合，不改变候选头、剂量或 DVG；`λ_q` 只决定两类已有证据的相对权重。

### 4.2 可靠性必须只依赖结果前信息

第一版不训练新的神经门控器，先使用可审计的统计量：

- `n_eff`：邻居权重的有效样本数；
- 距离离散度/最近邻间隔：检索是否有清晰的局部邻域；
- 邻居 action-sign agreement：历史回放中同方向修正收益的加权一致性；
- `A_hat` 的邻居方差或 bootstrap 方差；
- W1 与 context 对候选方向/区间的相互支持程度；
- 是否 proposal-empty、是否只有零剂量。

可用一个单调的连续收缩形式作为实验候选（不是最终数学主张）：

\[
\lambda_q
=\frac{R_{ctx,q}}{R_{ctx,q}+cR_{W1,q}+\varepsilon},
\]

其中 `R` 由上述 S3-M 历史证据统计构造，`c` 只在 S2V/S3-M 选择并在 S4 冻结。推荐先以 `n_eff/(variance+ε)` 形式做可解释基线，不引入额外 loss。

### 4.3 三个反对过拟合的设计要求

1. **无 market/host ID：** 不允许根据 Shandong、PatchTST 等身份手写 λ。
2. **无硬 release threshold：** `λ_q` 是连续收缩，最终仍由既有整日 LCB 决定。
3. **无新监督头：** 不能在 S4 用当前目标日结果训练“是否该检索/是否该执行”分类器。

### 4.4 APR 只在证据分解通过后运行

如果邻居审计显示：

- E2 和 E0 邻居高度重叠，差异只来自 `k` 或数值权重，优先做 `k/weight` 分解；
- 邻居差异大但 action-sign 一致，优先做候选聚合/读出审计；
- 邻居很相似但 action-sign 冲突，APR 才是有意义的候选；
- proposal 在两者都为空，问题是候选容量/薄样本，不是检索；
- LCB 前 proposal 有正收益、LCB 后全拒绝，才调查校准，不提前改 LCB。

---

## 5. 实验顺序：先复盘，后单轴改动

### Phase 6.0：E2 vs E0 邻居差异分解（不重训）

**目的：** 找到真正可改的层；不改变任何正式 S4 输出。

**矩阵：**

- 市场：`LAGO_DE`、`shandong_DA`、`LAGO_NP`；
- host：`Linear`、`MLP`、`PatchTST`；
- seed：`0/1/2`；
- arm：E0(W1)、E2(context)、E3(fixed composite)；
- 先使用已有 universal-core checkpoint；若某 cell 实际是 per-market head，必须标注 variant。

**每个 query day 必须落盘：**

| 类别 | 字段 |
|---|---|
| 邻居集合 | indices、距离、权重、rank、E0/E2 overlap/Jaccard |
| 邻居稳定性 | rank correlation、leave-one-neighbor-out 变化、距离 margin、`n_eff` |
| 候选证据 | Down/Up 原子质量、剂量、proposal 是否为空 |
| 动作信号 | `A_hat`、邻居 `A_true` 历史回放均值/方差、方向一致性 |
| 选择链 | proposal → LCB → final action，每一步的拒绝原因 |
| 结果 | executed raw gain、harm、normal/tail 归属、E2−E0 差值 |

**必须增加的分解：**

\[
\Delta_{E2-E0}
=\Delta_{retrieval\ set}
+\Delta_{weight/k}
+\Delta_{proposal}
+\Delta_{LCB\ selection}.
\]

不要求一次给出严格因果分解，但至少通过同一候选在不同邻居账本上的 replay 产生四个可复核反事实。

**输出：**

- `neighbor_diff_long.csv`：逐 query、逐邻居长表；
- `retrieval_decomposition.json`：每 cell 的 overlap、n_eff、sign agreement、proposal-empty、release、raw gain；
- 6 张最小图：距离分布、邻居重叠、n_eff、action-sign agreement、proposal→LCB 漏斗、E2−E0 逐日差值；
- 一页 verdict：`KEY / K_WEIGHT / PROPOSAL / CALIBRATION / DATA_SUPPORT` 五类之一。

### Phase 6.1：APR 最小单轴实验（仅当 6.0 指向 RETRIEVAL/WEIGHT）

**受控变量：** 候选 checkpoint、split、query key、候选、双事件提案、DVG、`q/alpha` 全部固定；只替换 context 与 W1 的连续证据权重。

**arms：**

| arm | 说明 |
|---|---|
| E0 | W1-only 正式控制 |
| E2 | context-only 诊断 |
| E3 | 固定 composite |
| E4 | APR reliability-weighted composite |

**第一轮矩阵：** `2 markets (LAGO_DE/Shandong) × 3 hosts × 3 seeds × 4 arms = 72 cell`；先不加入 LAGO_NP/Shaanxi/Gansu，避免把“没有动作容量”与检索可靠性混为一谈。

**放行条件：**

- 至少 LAGO_DE 与 Shandong 各有 ≥2/3 host、≥2/3 seed 的 raw 非劣；
- macro relative skill 不低于 E0，且不能由一个 PatchTST cell 拖动；
- executed raw harm rate 不恶化，normal-regime MAE 不显著退化；
- LAGO_NP 加入 safety check 后仍无显著回归；
- APR 的 λ 分布能被 `n_eff/variance/agreement` 解释，而不是塌成常数或身份分类器。

**失败分支：**

- E4≈E3：固定 composite 已足够，保留 E3 为简单版本；
- E4≈E0：context 没有稳定额外价值，CAVM 降为探索性模块；
- E4 worse：撤回 APR，回到候选容量或数据契约；
- E4 只改善一个 host：不能加 host ID，转向 host-relative residual/candidate support 审计。

### Phase 6.2：候选容量与低机会域诊断

仅在 6.0/6.1 之后，加入：`LAGO_NP`、`shaanxi_RT`、`gansu_DA`。

按如下漏斗报告：

```text
candidate alive
  → directional proposal non-empty
  → A_hat > 0
  → LCB > 0
  → final action non-empty
  → realized raw gain
```

对于 `proposal-empty≈1` 的市场，不把“无改善”写成检索失败；它应作为低机会/候选支持边界案例。

### Phase 6.3：概念漂移的 shadow-memory 实验

这不是立即放行 P4。它只验证“经验层是否有正向、无泄漏的更新价值”。

**三种 local-memory 方案：**

1. fixed recent window；
2. expanding window；
3. exponentially decayed replay buffer。

所有方案都遵守：

- 当天动作只使用当天开始前已有 memory；
- 当天标签公布后，才追加到下一个 query 的 memory；
- universal core、host、IAH loss 不更新；
- drift detector 只记录报警，不直接做 release gate；
- decay/window 只在 S2V/S3-M 选择，S4 冻结。

**时间协议：** rolling-origin/prequential；至少保留一个明显 regime 变化段（负价渗透、极端尖峰、政策/季节切换）。

**指标：** warm start→after-update 的 cumulative relative skill、每日 raw gain、release/harm、memory age、drift alarm 前后差异。

只有当 shadow-memory 在两个市场类型、三个 seed 上稳定改善且无泄漏，才重新讨论 P4 的状态更新。

---

## 6. 数据扩展：先做数据准入，不做盲目堆数据

### 6.1 现有域的角色

| 域 | 作用 | 当前处理 |
|---|---|---|
| `LAGO_DE` | 国外负价/极端正例 | 机制正例，不等于普适证明 |
| `LAGO_NP` | 温和/低机会负例 | 检验误释放与相似度无判别力 |
| `shandong_DA` | 国内负价+尖峰、特征丰富 | 主真实场景，不能公开则只作 target/transfer |
| `shaanxi_RT` | 强 host/实时角色 | 检验 host 几何与 RT 契约 |
| `gansu_DA` | 薄样本 | 检验低数据量安全边界 |

### 6.2 推荐候选新增数据源

优先选择能补足 regime，而不是只增加样本量：

1. **OPSD / ENTSO-E 欧洲多 bidding zone**：小时价格、负荷、风光等，适合统一 DA price-only 与 covariate-rich 两条契约。[OPSD](https://open-power-system-data.org/) [OPSD time series](https://data.open-power-system-data.org/time_series/) [ENTSO-E Transparency](https://www.entsoe.eu/data/transparency-platform/)
2. **Nord Pool**：北欧 DA 价格与跨区流量，补充负价和高风电 regime。[Nord Pool data portal](https://data.nordpoolgroup.com/)
3. **PJM**：同时有 DA/RT LMP 入口，适合验证国内 RT 契约之外的实时市场迁移。[PJM RT hourly LMP](https://dataminer2.pjm.com/feed/rt_hrl_lmps/definition)
4. **NYISO**：价格、负荷、电网运行数据分栏发布，适合构造外生变量丰富域。[NYISO market data](https://www.nyiso.com/energy-market-operational-data)
5. **ERCOT**：DA/RT settlement point price 与 LMP 数据可作高波动、负价稀疏的补充域。[ERCOT market information](https://www.ercot.com/mktinfo)
6. **NSW-EPNews**（可选）：半小时 NSW spot、天气与新闻；先只取数值 price/weather，不把文本模型引入 HCH。[NSW-EPNews](https://arxiv.org/abs/2506.11050)

上述数据源只能作为候选清单；下载前必须核对许可证、时间戳、时区、DST、修订机制和可复现 API，不承诺全部纳入。

### 6.3 数据准入表（每个新市场先填）

| 字段 | 必须记录 |
|---|---|
| 目标 | market、zone/node、DA/RT、forecast horizon、frequency、timezone/DST |
| 价格 | currency、unit、negative-price rate、zero rate、spike/low-tail rate |
| 特征 | price-only / load / wind / solar / flows / forecast features；可用截止时刻 |
| 质量 | 日期覆盖、缺失率、重复、异常修订、完整 24/48 步比例 |
| 训练角色 | source / target / local-memory-only / holdout |
| 合规 | 来源 URL、许可证、下载脚本、版本 hash |
| 可比性 | 是否可与当前 DA/RT 契约对齐；不满足则单独 domain unit |

### 6.4 多市场训练方式

- 默认继续等域采样；T4 已否证 gradient reweighting，不重新发明 loss weighting。
- 新市场先做 local memory/holdout；只有 leave-one-market-out 后没有稳定负迁移，才进入 universal gradient。
- 训练单位显式使用 `(market, target_mode, host)`，但该元信息只用于数据契约和采样，不输入 S4 gate。
- 对同一市场 DA/RT 可作为两个 domain unit，不能把时间戳和目标角色混成一个序列。
- 先保持数据规模相近的每域上限，再考察新增样本量；不能让最长欧洲市场主导梯度。
- 负价/尖峰稀少时，优先做**连续状态分层的 replay reservoir**（用 S1 可见状态的分位统计，不增加事件头），避免 rare regime 被普通样本完全淹没；这一项必须单独做消融，不能与 APR 同时改。

---

## 7. 训练与适应策略优先级

### P0：不改训练器，先补可观测性（立即）

- 每 epoch 保存 macro S2V CRPS、每域 CRPS、候选 alive、proposal-empty、动作释放和校准统计；
- 保存 best checkpoint 之外的完整曲线；
- 输出 seed 级别曲线和域级别曲线，不只输出一个 best 数值；
- 确认 mixed-domain batch 的 `domain_det` 与数据行没有错位。

### P1：等域采样 + 结果前 local evidence（当前主配置）

这是 T4/D2/T5 共同支持的最稳起点：共享 core 学跨市场共同结构，本地 memory 学域差异。

### P2：连续状态 replay reservoir（单独实验）

不是“尖峰事件二分类器”，而是按 S1 可见连续状态、幅值分位、上下尾符号建立有限容量回放；训练目标仍是 IAH-CRPS。目标是让低频负价/尖峰在 memory 中不被完全淘汰。

### P3：APR 可靠性加权（依赖 Phase 6.0）

只在邻居差异证据指向 retrieval/weight 时实施；否则不做。

### P4：shadow online memory / drift-aware calibration（最后）

先做无泄漏 rolling-origin；不直接更新 universal core，不直接解除数学核心中 P4 门禁。

### 明确不做

- 不再做 macro accumulation 或 gradient difficulty reweighting；
- 不把 r=0.30 国内数据直接混入 universal training；
- 不同时修改检索、候选读出、DVG、loss 和数据集；
- 不用测试标签调 k、λ、decay、q；
- 不把 drift detector 变成硬阈值动作规则。

---

## 8. 指标与论文证据口径

### 8.1 主预测表（与锁定同行基线保持一致）

基线不改：

1. Base / Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR（官方）；
5. δ-Adapter Ada-Y（官方）；
6. HCH（ours）。

每个市场、host、seed 使用同一 split、同一预测器输出和同一可见特征。

主指标：MAE、RMSE、sMAPE；跨市场正文只报告相对 Identity 的 macro relative skill，不直接平均欧元、元、美元的 raw MAE。若 CRC 官方实现可复现，作为补充对比，不替换锁定五个同行后处理基线。[CRC](https://arxiv.org/html/2512.22428v1)

### 8.2 尾部与低价表

- S1 连续 high/normal/low 状态上的 MAE/RMSE；
- 真实 `y<0` 子集的 MAE、样本数、动作释放率（无负价域不填 0 假装有效）；
- 尖峰/低谷的事件召回、起止偏差、持续时长误差、峰谷幅值 MAE；
- normal-regime MAE 与退化率；
- `sMAPE` 近零/负值的局限单独说明，不能拿它替代 raw MAE。

### 8.3 动作与安全表

- candidate alive rate；
- proposal non-empty rate；
- final release rate；
- executed raw gain（元/MWh 或相应单位）；
- executed raw harmful-release rate；
- LCB empirical coverage；
- action-empty 域单独列为“无动作容量”，不把均值 π 或 `A_hat` 叫成真实收益。

### 8.4 漂移表

- rolling window 每段的 MAE/RMSE/sMAPE；
- cumulative relative skill；
- memory 更新次数、memory age、候选/邻居版本；
- drift alarm 前后结果，但报警不能被误写成性能保证。

### 8.5 统计检验

- 以天为配对单位，报告每市场/host 的区间；
- 采用 block bootstrap 或适合时间依赖的日损失差检验；
- 至少 3 seeds；
- 明确所有调参只发生在 S2V/S3-M，S4 target-free；
- 不用单一宏平均掩盖 Shandong host-split 或 LAGO_NP action-empty。

---

## 9. 结果交付规范（AI 执行后必须返回）

本轮若交给本地 AI 实现，必须先完成 Phase 6.0，再决定是否写 APR 代码。返回信息至少包括：

```text
1. git commit / branch / 工作树状态
2. 实际运行命令与配置 hash
3. 训练/验证曲线：每 epoch macro S2V CRPS、每域 CRPS、best epoch
4. 逐 cell 长表：market × host × seed × arm
5. 邻居差异：overlap、rank correlation、n_eff、distance margin、sign agreement
6. 动作漏斗：alive → proposal → A_hat → LCB → execute → realized gain
7. E2-E0 decomposition verdict：KEY / K_WEIGHT / PROPOSAL / CALIBRATION / DATA_SUPPORT
8. raw MAE、RMSE、sMAPE、tail/low-price、normal degradation、release/harm/LCB
9. 失败 cell、异常日期、action-empty 解释
10. 是否满足 Phase 6.1 放行条件；不满足时明确停止在哪一层
```

不得只返回“实验完成/指标提升”，必须提供可复核的逐日、逐 query、逐邻居证据。

---

## 10. 阶段门禁与论文主线

### Gate R：检索证据门禁

只有当 E2/E0 的差异可被邻居集合、权重、proposal 或 LCB 漏斗解释，才允许修改对应层。

### Gate A：APR 门禁

APR 至少在 LAGO_DE 与 Shandong 两类市场、两个以上 host、三个 seed 中稳定非劣，且不能引入 LAGO_NP 回归，才可保留为候选方法。

### Gate D：漂移门禁

shadow-memory 在 rolling-origin 中有稳定累计收益、无时间泄漏、无异常释放，才可讨论 P4 或“经验积累闭环”。

### 当前最连贯的论文主线（暂定）

> 冻结预测器不再被重新训练；HCH 学习一个跨市场可迁移的双尾候选分布，并将历史 query-dose 回放转化为整日动作价值证据。最新结果表明，真正限制跨市场稳定性的不是候选或指标几何，而是“相似历史是否具有可行动价值”的证据选择问题。因此，下一步研究不是堆叠更复杂的事件头或损失，而是用结果前的连续可靠性统计保留 W1 安全锚点、选择性地吸收 context 证据，并在严格 rolling-origin 协议下验证它能否抵抗市场概念漂移。

这条主线比“CAVM 普遍有效”更诚实，也比“新增一个市场/host gate”更有可迁移的研究问题。

---

## 11. 本轮最终决策表

| 项目 | 决定 | 原因 |
|---|---|---|
| MFAV / dual-geometry | **撤回** | P5A-B 未发现 metric mismatch，且恒等式和 raw 方向一致 |
| P4 action-value state update | **继续封锁** | CAVM 尚未通过两类市场门禁 |
| E2 vs E0 邻居差异分解 | **立即做** | 唯一未被法证排除、且可直接定位根因的变量 |
| APR 可靠性加权 | **条件保留** | 仅在分解指向 retrieval/weight 时做；不新增 loss/head |
| 新市场数据 | **准入后加入** | OPSD/ENTSO-E/PJM/NYISO/Nord Pool 等补 regime，不盲目增样本 |
| gradient reweighting | **停止** | T4 三 seed 双向证伪 |
| 国内数据直接高比例混合 | **停止** | T5 `r=0.30` 对 LAGO_NP 产生负迁移 |
| local shadow memory | **后续做** | 先 rolling-origin 验证，再谈概念漂移主张 |
| 六个论文基线 | **保持不变** | 用户已锁定且当前实验能力已具备 |

**下一步唯一建议：** 先让 AI 只实现 Phase 6.0 的邻居/动作长表与分解，不要直接写 APR、不要改数学核心。拿到 verdict 后，再决定是改检索权重、候选容量、校准还是数据支持。

