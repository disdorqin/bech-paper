# HCH-v2 P1：点读出、训练范式、增量数据与可复现实验设计

**版本**：v0.1 · 2026-08-14  
**用途**：交给本地代码 AI 执行 P1 微调与实验。  
**本阶段目标**：先把候选信息正确读成点预测，建立可复现的冻结协议，再用受控实验判断哪些训练范式和新增市场数据真正产生正向迁移。  
**不改变**：IAH-CRPS 数学核心、query-dose replay、双事件提案、整日动作价值校准、LCB 门控及现有六个固定比较对象。

---

## 0. 执行边界

### 0.1 当前调查结论必须被当作“待复核的实验前提”

上一份调查报告给出的证据是：

1. 候选分布确实学到了可迁移修正信息：S2V 的读出在 6/6 域改善，冻结 S4 在 5/6 域改善，NORD 与 GEFCOM 也有改善。
2. 当前点预测评价路径是错误的：`cand_pred` 使用 `x_identity`，因此候选 MAE 实际仍是宿主原值；尚未真正把候选分布读成点预测。
3. CRPS（候选分布）与 final MAE（宿主/动作输出）被混成一条评价路径。
4. 现网 sequential trainer 在 3 个 seed 上优于 macro accumulation；本 P1 不把 macro accumulation 作为默认修复方向。
5. 硬动作剂量的负增益较多，软读出比硬动作更值得先验证。
6. Shandong 的 rich covariate 接口已有数据契约，但必须单独做 U2-Rich 实验。

这些结论不能直接写成 “SOTA 已成立”。本地 AI 必须重新生成原始表、配置、数据切分哈希和曲线；在新的未触碰 S4 上验证后，才能把结果写成论文证据。

### 0.2 本阶段不做的事

- 不重新发明事件检测头，不加 BCE/辅助分类损失。
- 不加入额外 MAE、尾部 loss、交易 loss 或任意组合 loss。
- 不引入 beta-mixing、查询级 rho、逐小时 conformal sequence、任意多动作区间。
- 不修改 IAH 的 host-relative asinh 几何和三原子结构。
- 不把 market ID 作为预测性 token。
- 不把 S4 标签用于训练、读出选择、超参选择或数据筛选。
- 不把所有网上下载的数据直接拼进训练集；数据必须经过准入门槛和正迁移检验。
- 不覆盖既有结果目录；每次运行建立新 run 目录。

---

## 1. P1 的核心问题与验收标准

本轮只回答四个问题：

1. **候选是否真的能转成点预测？**  
   通过分布到点的预先规定读出函数，在新 S4 上与 Base、Residual-L1、QuantileResidual-LGBM、PIR、delta-Adapter Ada-Y、HCH 比较。
2. **当前训练是否欠收敛、采样失衡或跨市场负迁移？**  
   用训练曲线、S2V 曲线、分域健康量、seed 方差和 leave-one-market-out 证据回答，不能只看一个最终均值。
3. **哪些新增数据能产生正向迁移？**  
   每次只加入一个数据源或一个明确的数据组，比较 source-only、source+new、leave-new-out 和 leave-one-market-out。
4. **哪些指标足以支撑论文的第一版结论？**  
   主表精简为 MAE、RMSE、sMAPE；尾部和安全表单独报告，不用十几个指标稀释主线。

### 1.1 本轮“通过”定义

P1 只有在以下条件全部满足时才算通过：

- `cand_pred` 不再等于 `x_identity`，且有单元测试证明至少一个非 identity 读出可以改变点预测。
- 所有读出选择只使用 S2V，S4 只运行一次作为最终测试。
- 每个候选训练方向使用同一数据切分、总 day exposure、seed 集合和候选损失。
- 每个数据集都有 registry、时间范围、时区、target role、负价/低尾/高尾比例和许可证记录。
- 返回训练/验证曲线、每域曲线、收敛状态、基线对比、迁移矩阵和错误清单。
- 主结论同时给出宏平均、逐域结果和最差域；不只给 micro average。

---

## 2. 严格的数据切分与冻结协议

### 2.1 推荐的阶段流

| 阶段 | 用途 | 是否更新候选参数 | 是否可调读出/超参 |
|---|---|---:|---:|
| S1R | 宿主拟合、输入签名、尺度和时区检查 | 否（宿主按既有流程） | 仅修复数据契约 |
| S2T | IAH 候选训练 | 是 | 只用训练集统计量 |
| S2V | 候选 checkpoint、点读出函数、有限超参选择 | 候选已冻结 | 是 |
| S3M/S3C | 整日动作价值校准与 LCB 门控 | 否 | 只校准动作风险 |
| S4 | 一次性最终测试 | 否 | 否 |

若旧实验曾使用 S4 选择读出或调参，旧 S4 必须在新报告中标为 `development-only`，重新划分一个从未触碰的新 S4。不能把旧结果与新最终结果混为一表。

### 2.2 冻结规则

1. 候选训练完成后保存唯一 checkpoint、配置、git SHA、数据清单和随机种子。
2. 从 checkpoint 加载候选并设置 `eval()`；点读出不反向更新候选。
3. `S2V` 可比较 identity、加权均值、加权中位数、sMAPE Bayes action 和固定 shrink；不得在 S4 选择其中最好者。
4. S3 的动作校准只能使用 S3M/S3C；LCB 阈值保持数学文档规定的 `LCB > 0`。
5. 每个市场/宿主/seed 的结果都保留，不能只保存最优 seed。

### 2.3 无泄漏检查

执行前自动检查：

- 时间戳严格递增，DST 转换后没有重复日或缺小时。
- 同一原始日期不会同时出现在 train/validation/test。
- 归一化、asinh scale、分位数、读出系数均只从允许阶段估计。
- 不通过文件名、市场名或 S4 统计量隐式选择模型。
- 所有临时实验与最终实验都有 manifest；结果文件记录 split hash。

---

## 3. P1 第一优先级：修正分布到点的读出与评价

### 3.1 保持候选几何，不改候选 loss

对每个 horizon 得到三原子支持点：

~~~
z_a = z(host) + delta_a,       a in {down, identity, up}
x_a = s * sinh(z_a)
weights = (w_down, w_identity, w_up)
~~~

其中 `x_identity` 仍然是宿主原值；它只是候选分布的一个支持点，不得再被当作候选点预测的唯一读出。

### 3.2 必须实现并单独评估的读出

先实现无新增训练参数的读出，全部在 S2V 比较：

1. **Identity**：`x_identity`，作为退化/回退基线。
2. **Weighted mean**：`sum_a w_a x_a`，对应平方损失下的 Bayes action。
3. **Weighted median**：三点离散分布的加权中位数，作为 MAE 下的 Bayes action。
4. **sMAPE Bayes action**：在由三原子支持点构成的有限候选区间内，用一维数值搜索最小化加权 sMAPE；不得为每个测试集单独搜索参数。
5. **Global shrink**：`x_read = host + alpha * (x_read_raw - host)`；`alpha` 只能在 S2V 估计一次，作为诊断，不允许逐域逐日调参。

输出中必须记录每个读出在每个 `market × host × seed` 的结果。论文主读出先预注册：

- MAE 主表：weighted median；
- RMSE 辅助表：weighted mean；
- sMAPE 辅助表：sMAPE Bayes action；
- identity 只作为退化基线，不作为 HCH 的最终实现。

如 S2V 证据显示统一读出更稳，可选择一个统一主读出；不能按测试域挑选。

### 3.3 近零和负价处理

定义：

~~~
sMAPE_eps(y, yhat) = 2 * |y - yhat| / (|y| + |yhat| + eps)
~~~

`eps` 只从 S1R 的训练目标尺度计算并冻结，例如 `eps = 1e-6 * median(|y_train|)`；同时报告一个固定敏感性检查。禁止用不同数据集、不同测试日的手工 floor 让结果变好。

当数据集没有物理负价时，不伪造负价指标；将下尾替换为由 S1R 训练分位数定义的低价区间，并在表头明确写成 `low-tail`，不能称为 `negative-price`。

### 3.4 点预测轨与安全动作轨必须分表

**Point track**：候选分布 → 读出点预测，目标是 MAE/RMSE/sMAPE 与同行后处理模块比较。  
**Safe-action track**：query replay → 双事件提案 → 整日动作价值校准 → LCB 门控，目标是 action value、harmful release、normal degradation 和覆盖。  

点读出改善不能自动写成安全动作改善；安全动作不放行也不能被当作候选分布无效。

---

## 4. 数据契约与增量数据准入

### 4.1 现有数据先作为固定锚点

先复现仓库当前的公开/本地数据和既有 holdout，包括 LAGO/EPEX/Nord Pool/PJM、NEM_SA1、GEFCOM14P 及本地 Shandong。Shandong 仅作本地实验，不能在公开报告中泄露原始数据。

### 4.2 Dataset Registry 必须包含的字段

每个数据集一行，保存为 `dataset_registry.csv` 与 JSON：

| 字段 | 说明 |
|---|---|
| `dataset_id` / `market_id` | 稳定 ID，不用路径名代替 |
| `target_name` / `target_role` | DA 或 RT；当前可合并训练，但角色必须记录 |
| `frequency` / `horizon` | 15/30/60 分钟、预测步数 |
| `timezone` / `dst_policy` | 原时区、夏令时处理 |
| `date_start` / `date_end` | 可复现时间范围 |
| `n_days` / `missing_rate` | 日样本数、缺失比例 |
| `negative_rate` / `low_tail_rate` / `high_tail_rate` | 仅描述数据，不作为模型硬门限 |
| `covariate_schema` | 预测时可提前获得的特征列 |
| `license` / `source_url` / `raw_hash` | 许可证、来源、原始文件哈希 |
| `split_hash` | train/S2V/S3/S4 切分哈希 |
| `admission_status` | `candidate`、`accepted`、`rejected`、`holdout-only` |

### 4.3 数据分层

**Tier 1：价格市场主训练/泛化数据**

- ENTSO-E Transparency Platform 的 DA 价格、负荷、发电和跨区流；先选 2–4 个时区一致、缺失可控的区域。
- AEMO NEM 官方区域 spot/dispatch/pre-dispatch 数据；保留 NEM_SA1，再加入一个非 SA 区域做跨市场验证。
- 40-country DA price benchmark（先核验论文、官方数据入口和许可证，再决定是否下载）。
- 现有 LAGO/EPEX/Nord Pool/PJM/GEFCOM 数据。

**Tier 2：接口/协变量/鲁棒性数据**

- Spain hourly price/consumption/generation/weather 数据（第三方仓库只作为可复现候选，必须追溯到原始来源）。
- Hugging Face EDS-lab electricity-demand、ECL 等负荷数据：不作为价格 SOTA 主训练目标，只用于验证 rich covariate 接口和 schema 适配。

优先来源：

- ENTSO-E Transparency Platform：https://transparency.entsoe.eu/
- ENTSO-E 数据说明：https://www.entsoe.eu/data/transparency-platform/
- AEMO aggregated NEM data：https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
- AEMO electricity data：https://www.aemo.com.au/energy-systems/market-it-systems/electricity-system-guides/electricity-data
- HF EDS-lab electricity-demand：https://huggingface.co/datasets/EDS-lab/electricity-demand
- 40-country DA benchmark 线索：https://ieeexplore.ieee.org/iel8/10347231/10841822/11169754.pdf

### 4.4 数据准入流程

每次只接入一个数据集或一个明确的市场组：

1. 下载原始文件，记录来源、许可证、时间范围、哈希。
2. 转换成仓库现有的 24-hour day contract；不在转换器中偷偷填补目标。
3. 输出 schema report：列名、单位、时区、缺失、重复日、负价/低尾/高尾比例。
4. 先做 `holdout-only` 运行，确认输入契约和候选读出可工作。
5. 再做 `source + new`，并保留 `source-only` 同配置对照。
6. 进行 leave-one-market-out：加入新数据后，原有 holdout 和新市场都必须单独报告。
7. 只有经过正迁移判定的数据才进入主训练；未通过者保留在附录的负迁移结果，不删除。

### 4.5 正向迁移判定

不使用一个拍脑袋的绝对提升阈值，使用预注册的三态判定：

- `POSITIVE_TRANSFER`：source 宏平均和未见市场宏平均均不劣，且 paired block bootstrap 的主要指标没有系统性负向域。
- `MIXED_TRANSFER`：宏平均改善但一个或多个市场明显恶化；只能进入消融/附录，不能宣称通用泛化。
- `NEGATIVE_TRANSFER`：未见市场恶化或新增市场无法运行；不纳入主训练。

所有判定都给 95% paired block bootstrap CI、逐域差值和最差域差值。不得只用平均 MAE 选择数据。

---

## 5. 训练范式：受控方向试验，不一次堆叠

### 5.1 固定控制组 T0

T0 完全复现当前 sequential trainer：

- 现有 IAH-CRPS；
- 相同 S1R/S2T/S2V/S3/S4；
- 相同 day exposure、seed `[0, 1, 2]`、checkpoint 规则；
- 当前 `lr=3e-4`、`wd=1e-4`、clip=1.0` 等配置作为控制；
- macro accumulation 只保留为历史 negative control，不作为默认训练器。

### 5.2 候选方向（每次只启用一个方向）

| 编号 | 方向 | 何时运行 | 允许改动 | 不允许改动 |
|---|---|---|---|---|
| T1 | 正确点读出 | 必做 | evaluator/readout | 候选 loss、动作数学 |
| T2 | 市场/宿主均衡采样 | T0 已复现后 | batch sampler、域计数 | 总 day exposure、数据切分 |
| T3 | 学习率 warmup + cosine/one-cycle 二选一 | 仅当曲线显示欠收敛 | scheduler | 其它超参 |
| T4 | 连续模式覆盖采样 | T2 后仍有域失衡时 | 基于 host-only 描述子的软采样权重 | 人工尖峰/负价硬标签 |
| T5 | 数据混合比例/课程 | 仅当新增数据通过准入 | source/new 比例 | S4 选择 |
| T6 | Rich covariate branch | 单独 U2-Rich | role-typed exogenous token、learned-null | 价格-only 主线 |

T2/T4 的描述子只使用预测时可见的 host/输入统计；如果需要使用目标残差计算采样权重，必须明确标记为 training-only，并另做 target-free 对照，不能将其称为无监督泛化。

### 5.3 每个方向的统一实验规则

- 每次只改一个因素，最多同时跑两个候选方向作为平行实验，不把 T2+T3+T5 合成一个“黑箱最佳配置”。
- 三个 seed、相同训练日暴露、相同 checkpoint 选择。
- 每个方向先跑小规模 sanity run，再跑完整源域和 holdout。
- 如果方向改变了输入 schema，必须新建 `schema_version`，不能与旧结果混表。
- 每轮结束都输出 `keep / reject / inconclusive`，下一轮只继承通过项。

### 5.4 训练诊断

每个 epoch/评估周期记录：

- train IAH-CRPS、S2V macro IAH-CRPS、逐域 S2V loss；
- train-S2V gap、best epoch、early-stop 状态；
- 梯度范数、NaN/Inf、scale-invalid；
- `w_down/w_identity/w_up` 的均值、分位数、熵；
- `delta_down/delta_up` 的 alive rate、p50、p95；
- 每个市场/宿主的 batch 次数、day exposure、重复日期；
- 每个 seed 的 wall time 和异常日志。

收敛状态只能从以下标签中选：

`CONVERGED_STABLE`、`CONVERGED_BUT_OVERFIT`、`UNDERTRAINED`、`UNSTABLE`、`DEAD_SHIFT`、`MASS_COLLAPSE`、`NEGATIVE_TRANSFER`、`INCONCLUSIVE`。

标签必须由曲线和诊断共同支持，不能靠单一 loss 阈值。

---

## 6. 指标与对比实验

### 6.1 固定比较对象

不更换已有同行后处理基线：

1. Base / Identity：冻结预测原值。
2. Residual-L1：不分事件，直接学习条件中位残差。
3. QuantileResidual-LGBM：当前 quantile 基线修正。
4. PIR（官方）。
5. delta-Adapter Ada-Y（官方）。
6. HCH（ours）。

所有方法使用相同 host、相同 horizon、相同 S4、相同数据预处理；基线不能因为 HCH 使用了额外的未来信息而获益或受损。

### 6.2 主表：精简但可与同行对齐

**点预测主指标**

- MAE：主指标，按 dataset × host × horizon 报告。
- RMSE：对大误差/尖峰敏感的辅助指标。
- `sMAPE_eps`：解决价格接近零时 MAPE 不稳定；明确 epsilon 公式。

主表同时给：

- dataset macro average；
- market macro average；
- micro average（仅作补充）；
- worst-domain；
- 95% paired block bootstrap CI；
- 相对 Identity 的百分比变化。

不要以 MAPE 作为主指标；接近零或负价时 MAPE 不可比。

### 6.3 尾部表：验证尖峰与低价/负价价值

事件边界只用于评价，不进入 HCH 训练：

- high-tail：S1R 训练目标的 `q90` 或数据集预注册高分位；
- low-tail：S1R 训练目标的 `q10`；
- negative-price：真实 `y < 0`，只有存在负价时报告。

报告：

- high-tail MAE / RMSE；
- low-tail MAE；
- negative-price MAE、负价符号准确率（有负价时）；
- event recall：真实事件被改进的比例；
- timing error：事件首次/峰值位置的小时偏差；
- magnitude error：事件幅值绝对误差。

没有负价的数据集不得填零后计算总体 negative-price 平均；标记 `N/A`，另给 low-tail。

### 6.4 安全动作表

仅对 safe-action track 报告：

- release rate / identity rate；
- harmful release rate；
- net action value（有 DA/RT 或成本定义时）；
- normal-regime degradation；
- whole-day empirical LCB coverage；
- query-dose 分布与双事件动作长度。

这张表不能替代点预测主表，也不能用“放行率低”掩盖候选读出无效。

### 6.5 统计检验

- 时间序列不能随机打乱做独立 t-test；优先使用按天/按周 block bootstrap。
- 对同一 test day 的两个方法使用 paired difference。
- 需要显著性时使用 Diebold–Mariano 或 block bootstrap 的 CI，并报告多市场多 host 的校正方式。
- 每个数据集保留原始日级差值，不能只保存汇总均值。

---

## 7. 实验矩阵与执行顺序

### 7.1 第一轮：修复评价路径

| Run | 内容 | 目的 |
|---|---|---|
| P1-R0 | T0 候选冻结，复现旧曲线 | 固定控制 |
| P1-R1 | 五种点读出，候选 checkpoint 不变 | 证明分布可转点 |
| P1-R2 | 六个固定比较对象的点预测表 | 判断 HCH 是否有点指标优势 |
| P1-R3 | safe-action 轨单独复现 | 与点读出解耦 |

### 7.2 第二轮：训练范式

按以下顺序，一次只做一个：

1. T2 市场/宿主均衡采样；
2. T3 scheduler（只在 T0 曲线欠收敛时）；
3. T4 连续模式覆盖采样；
4. T5 新数据混合比例/课程；
5. T6 Rich covariate U2-Rich。

每个方向都与 T0 配对，不能拿不同方向的最优结果互相比。

### 7.3 第三轮：增量数据

对每个候选数据执行：

1. holdout-only；
2. source-only；
3. source+candidate；
4. leave-candidate-out；
5. leave-one-market-out。

只把 `POSITIVE_TRANSFER` 数据纳入下一轮主训练；`MIXED_TRANSFER` 放附录，`NEGATIVE_TRANSFER` 只保留诊断。

---

## 8. 必须返回的文件、曲线与结论

本地 AI 完成后，不能只返回一句“实验完成”。每个 run 建立独立目录，例如：

`experiments/08-hch-v2/results/P1_<date>_<gitsha>/`

至少返回：

| 文件 | 内容 |
|---|---|
| `run_manifest.json` | git SHA、命令、环境、seed、配置、数据哈希 |
| `dataset_registry.csv/json` | 全部数据集准入信息 |
| `split_manifest.csv/json` | 各阶段日期、时区、split hash |
| `train_loss_curve.csv/png` | train loss |
| `s2v_macro_curve.csv/png` | S2V 宏平均曲线 |
| `s2v_per_domain_curve.csv/png` | 每域曲线 |
| `health_curve.csv/png` | 权重、shift、梯度、NaN/Inf |
| `domain_sampling.csv/png` | 实际 batch/day exposure |
| `readout_matrix.csv` | 读出 × 市场 × host × seed |
| `baseline_comparison.csv` | 六个固定对象的主指标 |
| `tail_metrics.csv` | high/low/negative tail |
| `action_safety.csv` | release、harmful、LCB、action value |
| `transfer_matrix.csv` | source-only vs source+new vs leave-out |
| `seed_summary.csv` | mean ± std、CI、最差域 |
| `VERDICT.md` | 人可读结论、失败项、下一步建议 |

`VERDICT.md` 必须包含以下字段：

~~~
convergence_status:
best_epoch:
best_checkpoint:
candidate_crps:
point_readout_selected:
point_metrics:
tail_metrics:
action_metrics:
transfer_status:
negative_transfer_status:
SOTA_status:
next_recommendation:
~~~

`SOTA_status` 只能使用：

- `NOT_TESTED`
- `CANDIDATE_DISTRIBUTION_ADVANTAGE`
- `POINT_READOUT_ADVANTAGE`
- `ACTION_LIMITED`
- `NEGATIVE_TRANSFER`
- `INCONCLUSIVE`
- `SOTA_SUPPORTED`

只有 HCH 在预先声明的主指标上相对五个同行后处理基线和 Identity 取得稳定、逐域可解释的优势，且没有未报告的安全回退，才允许 `SOTA_SUPPORTED`。

最终返回消息还必须列出：

1. 实际执行的命令；
2. 成功/失败/跳过的数据集；
3. 是否收敛以及依据哪几张曲线；
4. 主表、尾部表、动作表的文件路径；
5. 最佳方向、被拒方向和拒绝理由；
6. 下一轮建议只选 1–2 个方向，不要直接合并所有改动。

---

## 9. 文献与工程依据（用于选择方向，不是强行复制）

本轮要求本地 AI 在报告中核对并引用真实论文/官方数据页，不能把下列线索当成未经核验的结论：

- Bayes point action 与 proper scoring rules：点读出应由损失对应的决策规则解释，而不是手工平均。
  - https://www.bundesbank.de/resource/blob/635562/0d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf
  - https://www.tandfonline.com/doi/abs/10.1198/016214506000001437
- PIR 的冻结预测后处理：比较“冻结宿主 + 后处理”的实验协议。
  - https://papers.neurips.cc/paper_files/paper/2025/file/331c41353b053683e17f7c88a797701d-Paper-Conference.pdf
- delta-Adapter 的黑盒输出校正：
  - https://arxiv.org/html/2601.20280v1
- 多域/跨域训练与不泄漏评估线索：UniTime、BLAST、DAF、SOLID、Fast/Slow Streams。
  - https://arxiv.org/abs/2310.09751
  - https://arxiv.org/html/2505.17871v2
  - https://proceedings.mlr.press/v162/jin22d.html
  - https://arxiv.org/abs/2310.14838
  - https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html
- 电力价格概率后处理与校准：
  - https://arxiv.org/html/2507.15079v1
  - https://arxiv.org/html/2404.02270v2
  - https://arxiv.org/html/2311.07289v2

AI 的任务不是再做一轮广泛综述，而是回答：哪一种训练/采样/数据准入方向最可能解决当前的读出错误、跨域负迁移或曲线欠收敛，并用本项目的受控实验验证。

---

## 10. 最终执行提示词（可直接复制给本地 AI）

~~~
你现在执行 HCH-v2 P1。先读取本提示词、当前仓库代码和训练调查报告。

第一步：创建新的实验 run 目录和 manifest，不覆盖历史结果。检查当前 cand_pred 是否仍等于 x_identity，先实现并测试 IAH 分布到点的五种读出；候选 checkpoint 冻结，读出只在 S2V 选择，S4 只测试一次。

第二步：固定六个比较对象：Identity、Residual-L1、QuantileResidual-LGBM、PIR、delta-Adapter Ada-Y、HCH。主表只用 MAE、RMSE、sMAPE_eps；另做 high-tail、low-tail/negative-price 和 safe-action 表。所有指标按 dataset×host×seed 保存，并给 macro、micro、worst-domain、paired block bootstrap CI。

第三步：完整复现当前 sequential trainer 作为 T0。然后按 T2→T3→T4→T5→T6 顺序逐个做受控方向实验；每次只改一个因素，固定 seed、day exposure、切分和候选 loss。macro accumulation 不作为默认修复方向。

第四步：新增数据只能经过 registry、许可证/来源、时区/DST、target role、缺失和尾部比例检查。一次只加入一个来源，先 holdout-only，再 source+new、leave-new-out、leave-one-market-out。只有 POSITIVE_TRANSFER 才进入主训练；MIXED/NEGATIVE 必须保留结果但不能冒充泛化。

第五步：返回 train/S2V/health/domain-sampling 曲线，readout_matrix、baseline_comparison、tail_metrics、action_safety、transfer_matrix、seed_summary，以及包含 convergence_status、SOTA_status 和 next_recommendation 的 VERDICT.md。不要只返回文字摘要。

严禁：S4 调参或筛数据；新增事件头/额外 loss；任意硬阈值驱动模型；把点读出优势写成动作安全优势；把没有负价的数据填零后报告 negative-price 指标。
~~~

