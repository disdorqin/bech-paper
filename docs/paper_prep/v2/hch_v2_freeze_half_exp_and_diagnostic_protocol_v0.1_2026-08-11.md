# HCH v2 冻结、Half-Exp 与 v2 诊断协议

> 文档版本：v0.1  
> 日期：2026-08-11  
> 目标仓库：https://github.com/disdorqin/bech-paper  
> 当前状态：**HOLD——现在不要执行**  
> 解除条件：代码修复通过审查 + 新损失数学方案通过审查 + 融合后的代码再次通过契约测试

## 0. 为什么现在先写、但不立即跑

HCH 是“冻结宿主上的后处理模块”，所以 half-exp 之前必须完成正式冻结；否则 S4 上的任何继续训练、记忆更新或门控校准都会破坏实验含义。

但当前实现仍有候选语义、OOF、连续状态、S3 自检索、样本错位和 bundle 持久化问题，而且新损失尚未定型。现在直接跑会得到不可比较且很快作废的结果。因此本协议先锁定实验顺序与证据格式，等待架构融合后执行。

## 1. “冻结模块”的准确含义

### 1.1 第一版不是一个万能 checkpoint

第一轮 within-market half-exp 的冻结单元为：

$$
\text{market/dataset group}\times\text{host}\times\text{seed}.
$$

流程是：

1. S1：训练宿主与拟合只依赖历史的预处理统计，随后冻结宿主；
2. S2：在冻结宿主 OOF 输出上训练 HCH 的状态、Bi-OMC、context/key 等可学习组件，随后冻结；
3. S3：用冻结组件构建日情节 memory，并校准 DVG；随后冻结整个 bundle；
4. S4：只推理，不训练、不校准、不更新 memory。

这能证明“冻结后处理在未见测试期工作”，但还不能证明“一个 HCH checkpoint 对所有宿主/市场零训练通用”。

### 1.2 山东 DA/RT 的处理

- 山东可作为一个 market group，DA 与 RT 是两个 `target_id` channel；
- 日期先划分，两个 channel 共同训练共享模块，target token 区分；
- DA/RT 分别报告结果；
- 禁止同日一个 channel 的真实目标进入另一个 channel 的预测输入；
- 如果融合后代码暂时不支持安全联合训练，可退回两个独立 bundle，但必须在报告中注明，不能宣称已实现共享训练。

### 1.3 后续才做的泛化证据

不放入本轮 half-exp：

- leave-one-host-out：在若干宿主 OOF 上训练 HCH，冻结后应用于未见宿主；
- leave-one-market-out：多市场训练 global HCH，冻结后应用于未见市场；
- Global-Frozen 与 Market-Calibrated 对照；
- 单 checkpoint 跨 DA/RT/海外市场的零样本主张。

这些是下一阶段模型无关性与迁移性验证，不应由当前 within-market 结果代替。

## 2. HOLD 解除门

只有审查方逐项签字后才开始：

| Gate | 必须证据 | 未满足时 |
|---|---|---|
| R：代码修复 | 修复规范全部通过、P0=0、真实契约测试通过 | 返回修复 |
| M：数学方案 | 分布假设、NLL/partial moments、稳定性和 claim boundary 通过 | 返回数学窗口 |
| F：架构融合 | 新损失已接入且未破坏 candidate/gain/freeze 语义 | 返回实现 |
| S：contract smoke | Linear 与 PatchTST 的 S1→S4 freeze/reload 通路通过 | 不得 half-exp |

文档中的 “HOLD” 只能由当前项目审查方解除，执行 AI 不得自行判断。

## 3. Frozen Bundle 规范

### 3.1 Bundle ID

建议 ID：

`{market_or_dataset}__{target_group}__{host}__seed{seed}__{split_hash}__{code_commit}`

不要依赖文件名解析全部元数据；bundle 内部必须有 manifest。

### 3.2 必须封存

| 类别 | 内容 |
|---|---|
| 身份 | dataset/market/target、host、seed、代码 commit、配置 hash |
| 数据 | 数据版本/hash、时区、S1–S4 边界、共同评估 index hash |
| 宿主 | checkpoint、模型配置、预测缓存 hash |
| 预处理 | 仅由 S1 得到的 scaler、状态统计、feature availability |
| HCH | state/context、Bi-OMC、GainKeyEncoder、新损失所需参数 |
| 记忆 | S3 day key、24×3 gain、date id、mask、candidate hash |
| 路由 | `k/eta/tau`、soft-soft/soft-hard 模式、S3 校准日志 |
| 基线 | method implementation label、upstream commit、adapter diff |
| 复现 | Python/依赖版本、设备、确定性配置、训练日志路径 |

### 3.3 冻结不变量

- S4 之前所有参数 `requires_grad=False` 且模型 `eval()`；
- optimizer 不持有将被 S4 调用的参数，或在 freeze 后销毁；
- `predict_s4` 不接收 `y_true`；
- S4 不拟合 scaler、不挑 `k`、不更新 memory、不 early stop；
- S4 前后 bundle 的深 hash 相同；
- 同一方法重载 bundle 后预测一致；
- 每个方法只在共同 keyed S4 index 上评分。

如计划在线更新，必须另立 prequential 协议；本轮禁止把在线更新混入 frozen 结果。

## 4. 实验阶段

### Phase 0：架构融合后的 contract smoke

目的仅是检验冻结链路，不比较 SOTA。

建议最小组合：

| 数据/通道 | 宿主 | 原因 |
|---|---|---|
| NEM_SA1 | Linear、PatchTST | 公开、极端明显、覆盖传统与前沿宿主 |
| LAGO_DE | Linear、PatchTST | 公开欧洲市场、负价/低价重要 |
| Shandong DA | Linear、PatchTST | 多外生真实场景 |

配置：seed 0、极少 epoch/steps。必须完整走 S1→S2 OOF→S3 LODO calibration→freeze→新进程 reload→S4 predict。通过后仍不产生性能主张。

### Phase 1：Half-Exp 主实验

建议代表数据/通道：

1. NEM_SA1；
2. LAGO_DE；
3. LAGO_NP，作为物理负价很少或没有的低尾对照；
4. Shandong DA；
5. Shandong RT。

数据别名以修复后的 manifest 为准，不因命名差异复制数据。划分按每个数据集时间比例执行，不要求固定测试长度；必须日期优先、时间连续，并公布各段日期与样本数。

宿主：

1. Linear；
2. LSTM；
3. PatchTST。

方法：固定六种 Identity、Residual-L1、QuantileResidual-LGBM、PIR、δ-Adapter Ada-Y、HCH v2。PIR/δ-Adapter 若官方实现对某组合确实不支持，保留 `unsupported_official`，禁止换代理方法填数。

运行顺序：

1. 先用 seed 0 完成全部组合并审计异常；
2. 无系统错误后再补 seeds 1、2；
3. 每个组合先冻结 bundle，再启动独立 S4 评估进程；
4. 不因 seed 0 的结果修改 S4、事件定义或方法超参。

计划规模：

$$
5\ \text{channels}\times 3\ \text{hosts}\times 6\ \text{methods}\times 3\ \text{seeds},
$$

其中 unsupported 行保留状态而非伪造数值。

### Phase 2：仅在 half-exp 诊断后决定

不在本协议自动执行：

- 扩展五宿主；
- 增加更多公开市场/国内省份；
- 正式全消融；
- global/market、leave-one-host/market-out；
- 大规模超参搜索。

## 5. 第一版精简指标

原则：指标既要与电价预测文献可对照，又要回答“低/高尾修正、正常期安全与路由有效性”。不堆砌几十个近义指标。

### 5.1 主表：整体预测

每个 dataset/channel × host 报：

- MAE；
- RMSE；
- rMAE（相对 seasonally naïve；naïve 定义固定并公开）。

HCH 与每个同行方法都在相同 timestamp 上计算。主比较以相对 Identity 的配对变化为中心，而不是只看绝对排名。

### 5.2 尾部表：双向事件

所有评估集合由 S1 阈值固定，仅用于评估，不进入路由：

- Adaptive High：S1 条件分布的上 10% 尾；样本足够时另报上 1% stress；
- Adaptive Low：S1 条件分布的下 10% 尾；样本足够时另报下 1% stress；
- Physical Negative：真实 `y<0`，只在该数据确有足够样本时作为独立物理卖点。

核心指标：

| 目标 | 指标 | 解释 |
|---|---|---|
| 高/低尾幅值 | Tail-MAE | 对极端价格幅值的直接误差 |
| 尖峰/低谷发生 | Recall 与 Precision | 宿主或后处理预测是否捕捉到事件 |
| 时序偏差 | Event timing error | 预测事件与真实事件最近匹配的小时偏移 |
| 物理负价 | Negative-price MAE、recall、sign error | 只在样本量足够时报告 |

事件阈值必须来自 S1 同一条件规则；预测事件也使用同一固定价格阈值。匹配窗口和一对一匹配算法写入配置，禁止按结果修改。

若 Physical Negative 在某 S4 少于预先规定的最小计数（建议在执行前锁定，如 50 点），只给计数与描述性结果，不做显著性或“胜出”主张。没有负价的数据仍可通过 Adaptive Low 检验通用低尾能力。

### 5.3 安全与动作表

- Normal-region ΔMAE：HCH 相对 Identity 在 S1 中间 80% 区域的 MAE 变化；
- Harm rate：执行校正后误差大于 Identity 的比例；
- Action rate：非 Identity 的比例；
- Down / Up / Identity 动作占比；
- Realized action gain；
- Candidate Oracle Gain；
- Gate Oracle Gap。

定义：

$$
G^{oracle}_t=\max(0,G_t^{-},G_t^{+}),
\qquad
\text{GateGap}=\mathbb E[G^{oracle}_t-G^{chosen}_t].
$$

Candidate Oracle Gain 判断候选是否有潜力；Gate Oracle Gap 判断损失来自候选还是路由。两者是后续算法改进的核心诊断，不应只报最终 MAE。

## 6. v2 必做诊断，不等同于最终论文消融

正式全消融留到架构稳定后。本轮只做四个低成本、能定位瓶颈的诊断。

### D1. Candidate capacity

按 Overall / High / Low / Physical Negative 报：

- Down/Up candidate 各自 realized gain；
- Oracle 在三动作中可取得的 gain；
- candidate 的方向错误率与幅值偏差。

解释：

- Oracle 也无增益：Bi-OMC/状态/新损失是首要问题；
- Oracle 高、chosen 低：CAGM/DVG 是首要问题。

### D2. Gate decomposition

报：

- 动作混淆矩阵：chosen action vs oracle action；
- 错误 Identity、错误 Down、错误 Up 的比例；
- 每类错误造成的 gain loss；
- estimated gain 与 realized gain 的校准图/分箱表。

不以测试结果调 gate，只用于下一版定位。

### D3. Retrieval quality

比较但不做完整消融：

1. learned memory retrieval；
2. random neighbor；
3. no-memory/local gate。

指标为 neighbor gain-pattern similarity、retrieved gain prediction error 与最终 GateGap。必须输出 neighbor date id，验证没有 S3 self retrieval 或 S4 memory update。

### D4. Continuous state functionality

- 状态头梯度范数；
- 状态与 S1 rank/scale target 的相关/校准；
- 对 state embedding 做小扰动后 candidate 与 day key 的变化量；
- 在 High/Low/Normal 区域的状态分布。

这里只证明共享状态有功能，不把临时 rank/scale 定义包装成创新。

## 7. 统计与汇总

- 每个 seed 保存逐时间戳预测与动作；
- 表中报三 seed 均值±标准差；
- 对 HCH vs Identity、HCH vs 最强同行方法的 MAE、Tail-MAE 差值做按日 block bootstrap 95% CI；
- 同时给每市场结果和市场宏平均，不能让长数据集支配总体结论；
- 山东作为私有外部真实场景，不承担唯一主张；公开数据承担可复现主证据；
- 不允许只汇报 HCH 获胜的数据/宿主。

当前 half-exp 的目标是“得到可信参照和定位模块问题”，不是通过多次尝试挑出最好的一次。

## 8. 机器可读输出

每次 run 必须产生：

### 8.1 `run_manifest.json`

- run_id、method、method status；
- dataset/market/target、host、seed；
- S1–S4 日期和 index hash；
- code/config/data/bundle hashes；
- upstream commit 与 official/limited/unsupported；
- 开始/结束时间、设备、依赖；
- 退出状态与错误。

### 8.2 `predictions.parquet`

至少包含：

- timestamp、date_id、horizon；
- dataset/market/target、host、seed、method；
- y_true、host_pred、final_pred；
- identity/down/up candidate；
- state 输出；
- action/weights、estimated gains/risks；
- neighbor ids/weights；
- bundle hash。

真实标签只允许在离线评分文件生成阶段 join；模型 `predict_s4` 输出本身不接收标签。

### 8.3 `metrics_long.parquet`

- run_id；
- region（overall/high/low/negative/normal）；
- metric；
- value、sample_count；
- threshold source/version；
- CI 信息。

### 8.4 报告

报告只从上述 machine-readable 文件生成，禁止手抄表格。旧结果不覆盖，新实验使用带日期、阶段和 commit 的新目录/文件名。

## 9. Go / Stop 判据

### 可以进入下一轮创新改进

- 所有 freeze invariant 成立；
- 方法共享完全相同 S4 index；
- Candidate Oracle 与 GateGap 可稳定分解问题；
- 结果可由 raw prediction 重算；
- 至少公开数据与山东均完成代表组合；
- 无 official label 违规。

### 必须停止并修代码

- S4 前后 hash 变化；
- `predict_s4` 访问标签；
- memory 含 S4 日期；
- 方法样本不一致；
- 重载 bundle 后预测不一致；
- 同一 run 无法从 manifest 重现；
- candidate gain 与最终执行 candidate 不一致。

### 算法红旗，不是代码错误

- High/Low Candidate Oracle Gain 接近零或为负；
- Oracle 有明显增益但 GateGap 占掉大部分；
- memory 与 random neighbor 无差别；
- state 反事实对输出几乎无影响；
- Normal-region 明显退化；
- 负价效果只来自极少样本且不稳定。

出现这些红旗时保存结果并回到创新设计，不要通过继续调 seed 或改 S4 阈值掩盖。

## 10. 执行 AI 回传格式

解除 HOLD 后，执行 AI 必须新建报告：

```markdown
# HCH v2 Freeze/Half-Exp Handoff
## 1. HOLD 解除依据
## 2. Bundle 清单与 freeze hash
## 3. 数据划分与共同 S4 index
## 4. 方法/宿主/seed 完成矩阵
## 5. 主指标
## 6. 双尾指标
## 7. 安全与动作指标
## 8. 四项诊断
## 9. 官方基线状态与失败
## 10. 异常、缺失与下一版瓶颈
## 11. raw evidence 路径
```

不要在本协议 HOLD 期间生成这个报告；当前只需保留文档等待后续授权。
