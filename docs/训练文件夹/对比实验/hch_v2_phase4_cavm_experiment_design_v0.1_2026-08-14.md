# HCH-v2 Phase 4：CAVM 实验设计文档 v0.1

**日期：** 2026-08-14

**目的：** 在不改 v0.3 数学核心的前提下，验证“上下文经验记忆 + 动作价值自校验”是否能解决当前的温和市场脆弱性、概念漂移和动作选择不稳定问题。

**当前控制配置：**

```text
v0.4 Universal Core
12 个国际源域等域采样
IAH-CRPS
W1 atom retrieval
query-dose replay
double-event proposal
S3-C whole-day DVG
LCB gate
weighted_mean point readout
```

**重要：** CAVM 是下一阶段实验变量，不是已经被证明的论文创新。所有实验必须同时保留 `v0.4-core` 控制组。

---

## 1. Phase3 给出的实验事实

| 事实 | 当前解释 | 不允许的误读 |
|---|---|---|
| D2 37/40 国内 cell 改善 | 冻结通用头具有真实跨市场迁移能力 | 不能写成已经击败 PIR/δ-Adapter |
| weighted_mean 显著改善点预测 | 旧 identity readout 是重要问题 | 不能再据此直接否定 IAH-CRPS |
| T4 两种梯度权重均失败 | 等域采样是当前梯度轴的稳健局部最优 | 不能继续无休止搜索域权重 |
| T5 r=0.30 污染 LAGO_NP | 数据直接混入主干会造成负迁移 | 不能写成“更多数据必然更好” |
| Shaanxi RT 只有 1/4 帮助 | 宿主已经强时需要动作级条件化 | 不能宣称所有市场和宿主都改善 |

---

## 2. 实验问题与假设

### Q1：现有 W1 atom key 是否已经足够？

**H1：** 当前 W1 key 只描述候选残差原子，不充分描述宿主预测形状、时间结构和可观测性，因此对温和市场和宿主已强场景不稳定。

### Q2：连续上下文是否比单纯 atom key 更能预测动作收益？

**H2：** 使用宿主形状、动态变化、Data Signature 和时间信息的 context key，可以更好地找到动作价值相似的历史日。

### Q3：global 与 local 经验是否应该分离？

**H3：** global memory 适合冷启动，local memory 适合概念漂移；直接把两者混成一个无标记池，会重现 T5 的分布污染问题。

### Q4：动作价值是否应该作为最终路由依据？

**H4：** 候选质量和动作价值不是同一个问题。只有 context-conditioned action evidence 经现有 LCB 门控后，才能稳定决定 Down/Up/Identity。

### Q5：新增数据的最佳作用是更新参数，还是增加经验？

**H5：** 对分布差异大的新市场，先增加 global/local evidence 比直接增加 universal gradient 更安全。

---

## 3. 实验总原则

### 3.1 只改变一个轴

每组实验只能改变以下一个因素：

- retrieval key；
- global/local memory；
- 是否按揭示标签顺序更新状态；
- optional covariate 是否进入 context key。

不得同时改变：

- IAH-CRPS；
- candidate head；
- host；
- double-event；
- DVG 数学；
- readout；
- 数据切分。

### 3.2 两条评价轨道分开

#### 点预测轨道

weighted_mean 输出作为点预测：

- MAE；
- RMSE；
- sMAPE。

#### 动作轨道

使用现有 action value 和 LCB：

- predicted (\widehat A)；
- realized (A)；
- DVG error；
- LCB；
- execute/identity；
- action gain、harm 和 abstention。

不能用点预测 MAE 替代 action value，也不能只报告 CRPS 就声称动作模块有效。

---

## 4. 数据与市场选择

### 4.1 Universal source

默认继续使用当前 12 个国际源域和等域采样。

T5 的国内混合比例只作为对照：

- `r=0.15`：保留为低比例增量数据对照；
- `r=0.30`：保留为负迁移压力测试；
- 不将二者替换默认主配置。

### 4.2 代表性目标域

首轮不跑全部数据集，选择四类代表：

| 类型 | 数据集/模式 | 作用 |
|---|---|---|
| 真实负价、尖峰 | Shandong DA/RT | 检验下尾和上尾共享机制 |
| 宿主已较强 | Shaanxi RT | 检验是否过度修正 |
| 温和市场 | LAGO_NP | 复现 T5 负迁移脆弱性 |
| 样本偏薄 | Gansu 或 Ningxia | 检验冷启动和小样本鲁棒性 |

如果资源允许，再加入 Qinghai 作为薄数据外部验证；薄数据结果只能报告方向，不能过度解释效应大小。

### 4.3 Host

首轮使用三个互补宿主：

- MLP：弱宿主、修正空间较大；
- Linear：强基线、结构简单；
- PatchTST：现代时序宿主。

LSTM 作为第二轮确认，不作为第一轮所有组合的必要条件。

---

## 5. 严格时间协议

建议继续使用当前 v0.4 的阶段语义：

```text
H0      host fit
S1R     host out-of-sample reference / Data Signature
S2T     IAH-CRPS candidate training
S2V     macro-domain checkpoint selection
S3M     memory and retrieval-k selection
S3C     whole-day DVG calibration
S4      untouched test
```

### 5.1 Frozen transfer track

- S4 预测前不读取 S4 标签；
- S4 不更新 memory；
- 用于严格 zero-shot / frozen paper claim；
- target-domain DVG 必须由允许的 S3C 数据构建。

### 5.2 Streaming state track

- 参数仍然冻结；
- S4 当天预测前不读取当天标签；
- 当天标签完整揭示后，才追加到 local memory；
- 不更新 universal 参数、(k)、检索权重或 q；
- 单独报告为 parameter-frozen state-adaptive 实验。

两个 track 的结果不得合并为一个数字。

---

## 6. 实验矩阵

### E0：当前正式控制组

```text
weighted_mean
W1 atom key
global/local 仍按当前 v0.4 静态方式处理
static DVG + LCB
```

目的：确认 Phase3 和当前 v0.4 的结果仍可复现。

### E1：W1-only + 经验账本诊断

保持 W1 retrieval 不变，只额外记录：

- context key；
- A_hat；
- A_true；
- action error；
- 每日动作结果。

目的：判断当前 W1 邻居是否已经包含足够的动作价值信息。该组不改变预测结果。

### E2：Context-only retrieval

只使用连续 context key 选 top-k，保留原有 query-dose replay、double-event 和 DVG。

目的：检验宿主形状、动态和时间语义是否比 residual atom 本身更有检索价值。

### E3：Context + W1 atom composite retrieval

使用：

\[
d=\lambda_{\mathrm{atom}}d_{W1}
+\lambda_{\mathrm{ctx}}d_{ctx}.
\]

权重只在 S2V/S3-M 选择，S4 冻结。

目的：验证候选形状和宿主状态是否互补。

### E4：Global-only vs Local-only vs Global+Local

在 E3 的检索键下比较：

| 子组 | 经验来源 |
|---|---|
| E4-G | global source memory |
| E4-L | target local memory |
| E4-GL | global + local，显式记录 scope |

目的：验证新数据应当作为全局经验、局部经验，还是二者的收缩组合。

### E5：Streaming local update

在 E4-GL 基础上，按 S4 时间顺序执行：

```text
predict → reveal target → compute A_true → append local experience
```

输出：

- cold-start 曲线；
- steady-state 曲线；
- drift 前后曲线；
- local memory 有效样本数。

目的：验证“冻结参数、经验状态自适应”是否真的有效。

### E6：Optional covariate key

只在 E3 或 E4 取得正向结果后进行。

比较：

- host/history-only key；
- 加入国内可用预测特征；
- optional 全部 mask；
- optional 角色聚合。

目的：确认山东丰富外生变量是否真正增加动作价值信息，而不是仅仅改变 host。

### E7：Action-value state calibration（后续实验）

只有 E3–E5 有稳定收益后才做。

候选方向：

- context-conditioned action error；
- global prior 到 local evidence 的连续收缩；
- adaptive q / online conformal controller。

第一轮不实施 E7，不与 CAVM retrieval 同时改动。

---

## 7. 指标体系

### 7.1 必报主指标

| 轨道 | 指标 | 说明 |
|---|---|---|
| 分布 | CRPS | 与 IAH-CRPS 数学训练目标一致 |
| 点预测 | MAE | 第一版主点指标 |
| 点预测 | RMSE | 反映尖峰大误差 |
| 点预测 | sMAPE | 业务常用，但需注明近零敏感性 |

### 7.2 尾部指标

阈值必须从训练集分布得到，不使用固定金额硬阈值：

- upper-tail MAE：(y>q_{0.9}^{train})；
- lower-tail MAE：(y<q_{0.1}^{train})；
- upper/lower tail RMSE；
- 尾部样本覆盖数量。

真实负价单独报告：

- `negative_count`；
- `negative_price_MAE`；
- `negative_price_bias`。

当市场没有足够负价样本时，必须输出 `not_eligible`，不能填零。

### 7.3 动作与安全指标

- 每日 (A_{true})；
- 每日 (A_{hat})；
- action gain；
- harm rate；
- execution rate；
- abstention rate；
- normal-regime degradation；
- 最差日收益/损失分位点；
- 每市场 macro 与 worst-domain。

### 7.4 漂移指标

- 按时间顺序的 rolling MAE/CRPS；
- cold-start、adaptation、steady-state 三段表现；
- local update 前后差异；
- action error 的滚动均值；
- 经验池有效样本数量；
- 查询距离和 effective neighbor count。

禁止为了“指标更多”增加任意 WIS、多区间覆盖或新的未定义经济 loss。

---

## 8. 收敛、健康和可审计输出

每次训练必须保存：

### 8.1 训练收敛

- train IAH-CRPS 曲线；
- macro S2V CRPS 曲线；
- worst-domain S2V 曲线；
- 每域 S2V 曲线；
- best epoch 和 patience；
- 三个 seed 的均值和标准差。

### 8.2 候选健康

- mean (w^-,w^0,w^+)；
- mass entropy；
- Down/Up alive rate；
- (m^-)、(m^+) median/p95；
- identity collapse 比例；
- invalid-scale day 数量。

### 8.3 动作链健康

- neighbor 数量和距离；
- proposal 的 Down/Up/empty 比例；
- (A_{hat})、(A_{true})、error；
- q、LCB 分布；
- execute/identity 比例；
- fallback 原因。

### 8.4 记忆健康

- global/local memory 天数；
- 经验时间范围；
- context key 版本；
- effective neighbor count；
- 每日新增 local experience；
- memory hash；
- query 前后可见性审计。

### 8.5 必须生成的图

1. train vs S2V CRPS 曲线；
2. 每市场 MAE/CRPS 对比柱状图；
3. upper/lower tail MAE；
4. action gain 与 harm 的累计曲线；
5. cold-start 到 steady-state 曲线；
6. global-only/local-only/global+local 对比；
7. Shandong 负价样本案例图；
8. Shaanxi RT 和 LAGO_NP 失败/改善案例图。

---

## 9. 统计比较和放行标准

### 9.1 统计方法

- 同一 host、同一数据切分、同一 seed 做 paired comparison；
- 按完整日或连续时间块 bootstrap；
- 报告 macro mean、median、worst-domain；
- 不能只报告所有小时拼接后的 micro mean；
- 不能使用 S4 结果反向选择超参数。

### 9.2 第一阶段通过条件

CAVM 不需要每个市场都提升，但必须满足：

1. 至少两个不同类型市场方向稳定改善；
2. 三个 seed 的改善方向不能由单一 seed 支撑；
3. LAGO_NP 不出现可重复的显著退化；
4. action value 与最终点预测改善方向不能完全相反；
5. 如果只在 streaming local update 有效，必须明确降级为概念漂移实验；
6. 如果只改善 weighted mean 而 action 轨道无效，不能宣称 CAVM 主创新成立。

### 9.3 明确停止条件

出现以下任一情况，停止继续加模块：

- CAVM 只在山东负价域有效；
- context key 只改善单一 host；
- global+local 重现 T5 的温和市场污染；
- action harm 没有下降；
- 依赖复杂 learned key 才有收益，固定 key 完全无效；
- gain 主要来自重新调参而不是经验路由；
- 结果只在单 seed 或单一时间段成立。

此时应回到候选表达、时间错位或动作价值定义，而不是继续扩大 memory。

---

## 10. 与对比基线的关系

Phase4 内部诊断使用：

- `v0.4-core/W1`；
- `context-only`；
- `context+atom`；
- `global/local` 变体。

正式论文对比仍保留用户确定的五个同行后处理基线：

1. Base / Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR 官方实现；
5. δ-Adapter Ada-Y 官方实现；
6. HCH（ours）。

CAVM 如果通过内部实验，正式方法写成 HCH-CAVM；否则保留 HCH-v0.4，并把 CAVM 作为失败/探索性结果，不强行写成主贡献。

注意：D2 是“冻结 HCH 对宿主”的迁移证据，不等于已经完成上述五个 peer baseline 的 SOTA 比较。

---

## 11. 数据新增策略

### 第一优先级：不增加主干训练数据

先用现有：

- 12 个国际源域；
- 山东、甘肃、陕西、宁夏、青海国内数据；
- Phase3 已有缓存和结果。

目的：先判断 CAVM 是否能解决已知问题。

### 第二优先级：增加经验数据

如果 E4/E5 方向成立，再增加：

- 具有明确低尾/负价事件的市场；
- 具有丰富 DA/RT 或外生特征的国内市场；
- 不同气候/季节的电力市场。

新增数据优先进入 global/local memory，不能自动加入 universal gradient。

### 第三优先级：新增训练域

只有当新域经过 leave-one-market-out 审核且没有温和市场负迁移，才考虑进入 universal training。比例和采样策略必须另开实验，不能直接改主配置。

---

## 12. 论文写作中的结果边界

可以写：

> Phase3 shows that the frozen correction core transfers to unseen domestic electricity markets, while direct data mixing is conditionally beneficial and can contaminate mild markets. Phase4 therefore studies contextual evidence routing rather than indiscriminate gradient accumulation.

如果 CAVM 通过，可以写：

> CAVM separates transferable correction parameters from mutable contextual action evidence, allowing the frozen module to update its execution preference after labels are revealed without retraining the host or correction core.

不能写：

- 每一个新数据集都必然提升模块；
- 已经实现 universal zero-shot SOTA；
- 首次使用 retrieval memory；
- 首次在线适配；
- 正常期绝不退化；
- 对所有市场负价都有效；
- CAVM 已经具有漂移下的分布无关保证。

---

## 13. 推荐执行顺序

```text
Phase4-0 复现 v0.4-core/W1 基线
    ↓
Phase4-1 只读 CAVM key + 经验账本诊断
    ↓
Phase4-2 context-only / context+atom 检索
    ↓
Phase4-3 global/local 分离
    ↓
Phase4-4 sequential local update（概念漂移）
    ↓
Phase4-5 optional covariate key
    ↓
Phase4-6 action-value state calibration（仅在前面通过后）
    ↓
正式五个后处理基线比较
```

每一步都必须保存独立结果目录和新的文件名，不得覆盖 Phase3 或 v0.4 结果。

***

**本轮实验的真正判据不是“加了多少模块”，而是：**

> 新数据是否能在不污染冻结主干的情况下，增加与当前上下文真正相关的动作证据，并让 Down/Up/Identity 的选择更安全、更稳定。

