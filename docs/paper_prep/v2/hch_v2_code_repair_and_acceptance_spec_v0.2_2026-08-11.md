# HCH v2 代码修复与验收规范

> 文档版本：v0.2  
> 日期：2026-08-11  
> 目标仓库：https://github.com/disdorqin/bech-paper  
> 审计基线：`a3770bf813d56b2c597cd7917ee57fd46a8f654f`  
> 当前状态：**立即执行——只修代码与契约，不跑正式实验，不实现新损失，不改论文主张**

## 0. 给执行 AI 的一句话任务

在保持现有仓库结构、五个固定同行后处理基线和 HCH v2 主架构不变的前提下，修复下列数据泄漏、候选语义、连续状态、外生变量、OOF、检索校准、冻结持久化、基线标注和证据输出问题；用自动化契约测试及微型 smoke 证明修复成立，随后停止并等待审查。

## 1. 不可自行改变的项目决策

### 1.1 方法集合

正式比较方法固定为：

1. Base / Identity；
2. Residual-L1；
3. QuantileResidual-LGBM；
4. PIR（只有完整调用官方核心实现时才可标“官方”）；
5. δ-Adapter Ada-Y（同上）；
6. HCH v2（ours）。

不要替换、增加或删除上述方法。Linear、MLP、LSTM、TCN、PatchTST 是冻结宿主，不是同行后处理基线。

### 1.2 HCH v2 主线

- 主创新骨架：CAGM-DVG，利用 24 小时日情节记忆估计候选动作的条件收益，并相对 Identity 路由。
- 结构增强：Bi-OMC，同时生成向下与向上的 occurrence–magnitude 候选。
- 共享上下文：连续低—正常—高状态，必须同时影响 Bi-OMC 与 CAGM，但不作为第三个独立创新点。
- 动作集合：Identity、Down candidate、Up candidate。
- 架构不使用测试标签，不以硬事件阈值决定是否校正。

### 1.3 本轮边界

本轮允许：

- 修复实现错误和数据契约；
- 增加必要的轻量数据结构、序列化、日志与测试；
- 运行编译、单元测试、契约测试和极小规模 1 epoch smoke。

本轮禁止：

- 改仓库顶层目录；
- 重写 v1；
- 引入数学窗口尚未完成的新损失；
- 调参追求指标；
- 跑 half-exp、全量实验或正式消融；
- 用新结果更新论文结论；
- 用简化代理冒充官方 PIR / δ-Adapter；
- 未经当前操作者明确要求直接 push。

## 2. 当前审计结论

当前 `a3770bf` 只能视为工程 bring-up，不能视为科学证据。修复优先级如下。

| 优先级 | 问题 | 当前后果 | 必须达到的结果 |
|---|---|---|---|
| P0 | Bi-OMC 的 `y_down/y_up` 由潜变量 `z` 构造，而非冻结宿主预测 | 记忆中计算的 gain 与最终执行动作不是同一候选 | 所有训练、记忆、路由、输出使用完全相同的 `host_pred + delta` |
| P0 | 不同方法的 S4 有效时间戳不一致 | 方法差值被不同测试样本污染 | 单一评估清单，逐时间戳严格 inner join 后计算所有差值 |
| P0 | `host_cache.py` 有语法错误且 CLI 参数未真正生效 | 缓存流程不可可靠复现 | 全仓编译通过；缓存输出含完整对齐与 manifest |
| P0 | OOF metric projection 重复，fold-specific encoder 的 key 被混用 | 相似性与收益检索没有统一坐标系 | 原始 OOF record 统一训练一次 key encoder，projection 只执行一次 |
| P0 | S3 DVG 参数硬编码且校准时可能检索自身 | 门控结果可被乐观偏差主导 | S3 leave-one-day-out 校准 `k/eta/tau`，S4 前完全冻结 |
| P1 | 连续状态头没有监督、没有进入候选或记忆 | “共享连续状态”是死模块 | 有定义、有 loss、有梯度，并对候选与 key 产生可测影响 |
| P1 | 外生 token 无类型身份、无规范化、mask 未传给注意力、无明确 null | 多特征数据不可泛化，缺失变量处理不成立 | 因果可得、S1-only 规范化、变量身份与 learned-null 均被测试 |
| P1 | memory 非持久化，缺少 bundle/hash | 无法证明冻结与复现实验 | 完整可装载 bundle，round-trip 输出一致 |
| P1 | PIR / δ-Adapter 标签与实际适配范围不符 | 比较可能被审稿人一票否决 | 官方、有限重实现、不支持三种状态严格区分 |
| P1 | 契约清单写 16 条、实际仅 9 条，且部分为随机数组 stub | 测试不能证明模型契约 | 每项均调用真实实现，失败即非零退出 |
| P1 | 报告与 raw evidence 不一致 | 结果不可审计 | 配置、预测、时间戳、checkpoint/hash 和指标可追溯 |

## 3. 修复任务 A：统一时间与样本契约

### A1. 建立唯一的 Episode / Evaluation Manifest

新增一个轻量的统一清单对象，不要求重构目录。每个可评估点至少包含：

- `dataset_id`、`market_id`、`target_id`；
- `date_id`、`horizon`、`timestamp`；
- `split`（S1/S2/S3/S4）；
- `raw_row_index` 与 `is_valid`；
- 时区、DST 处理方式、数据版本或文件 hash；
- 该点可用外生变量及 availability/lag 信息。

四段划分必须在原始时间轴上先完成，再生成所有宿主和后处理样本。沿用仓库已配置的比例即可，不强制统一测试小时数，但必须把比例、边界日期和样本数写入 manifest。

### A2. 修复当前 S4 错位

已观察到示例差异：

- NEM：同行基线约 1720 小时，HCH 约 1752 小时；
- LAGO_DE：同行基线约 10446 小时，HCH 约 10488 小时；
- Shandong DA：同行基线约 7926 小时，HCH 约 7944 小时。

不得通过截取数组尾部解决。所有方法输出必须带 `timestamp + target_id`，由统一评估器按键连接；若任一方法缺点，报告缺失原因与最终共同样本数。

### A3. 24 小时日情节与 DST

- 日情节必须显式含 24 个 horizon。
- 遇到 23/25 小时日期时，只允许两类有记录的策略：先转换到统一无 DST 时间轴，或把该日从所有方法共同清单中排除。
- 禁止只对某个方法静默填充、截断或重复小时。

### A4. Host cache

修复 `experiments/08-hch-v2/host_cache.py`：

- 消除顶层意外缩进并通过 `py_compile`；
- CLI 参数必须实际传入数据、宿主、seed、split 与输出路径；
- 预测保存为“完整时间轴 + valid mask”，或保存显式 keyed rows；禁止无键的 valid-only 数组；
- 每个缓存写入宿主 checkpoint hash、配置 hash、数据 manifest hash；
- 保存真实训练/推理边界，禁止自动重训已标记 frozen 的宿主。

## 4. 修复任务 B：统一 Bi-OMC 候选动作语义

### B1. 唯一合法定义

对每个时刻：

$$
\hat y_t^{(0)}=\hat y_t^{host},\qquad
\hat y_t^{(-)}=\hat y_t^{host}+\Delta_t^{-},\qquad
\hat y_t^{(+)}=\hat y_t^{host}+\Delta_t^{+}.
$$

`BiOMC.forward` 应优先返回 `delta_down`、`delta_up` 及可解释的 occurrence/magnitude 中间量。若返回完整 candidate，也必须显式接收 `host_pred`。禁止以隐变量 `z[...,0]` 代替价格基准。

### B2. 全链路只生成一次候选

建立一个唯一函数，例如：

`build_candidates(host_pred, delta_down, delta_up) -> {identity, down, up}`。

训练 loss、OOF gain、S3 memory gain、DVG 路由、S4 最终输出、evidence 导出都必须调用同一个函数。禁止各文件重复拼 candidate。

### B3. Gain 定义

相对 Identity 的逐点动作收益必须来自最终执行的同一候选：

$$
G_t^{a}=\ell(y_t,\hat y_t^{(0)})-\ell(y_t,\hat y_t^{(a)}),\quad
a\in\{-,+\},\qquad G_t^{0}=0.
$$

当前 v2 可继续使用既有基础误差作为修复期 gain；新数学损失由后续架构融合决定。本轮不得悄悄改 gain 定义。

### B4. 必测不变量

- 输入 `delta_down=delta_up=0` 时，三个候选均等于 host；
- 修改 host 值 $c$，三个 candidate 同步平移 $c$；
- memory 中重新计算的 gain 与导出 prediction 的 gain 逐元素相等；
- Down/Up 的方向语义若采用符号参数化，测试其数值与梯度；不得靠测试集阈值裁剪。

## 5. 修复任务 C：让连续状态成为真实共享上下文

### C1. 修复期最小定义

不要离散标 Low/Normal/High 类。可使用仅由 S1 估计的稳健连续目标，示例：

$$
s_t^{rank}=2\widehat F_{S1,calendar}(y_t)-1,
\qquad
s_t^{scale}=\log\left(1+\frac{|y_t-\operatorname{median}_{S1,calendar}|}
{\operatorname{MAD}_{S1,calendar}+\epsilon}\right).
$$

其中 calendar 条件只能使用预测时已知的小时、工作日/节假日、季节等。若当前实现采用等价的连续 rank/scale 定义，可保留，但须在配置和代码注释中写清楚。

### C2. 状态监督与因果性

- S1 只拟合状态目标的统计量；
- S2 用真实 `y` 形成训练 target，但模型输入不得含当前/未来 `y`；
- S3/S4 只由可见上下文预测状态；
- `state_loss_weight` 必须真正进入总 loss，并记录各 loss 分量；
- 状态 embedding 同时输入 Down/Up candidate head 与 CAGM day key。

### C3. 本轮不替数学窗口作决定

上述 rank/scale 是修复期工作定义，用于证明状态模块不是死代码。后续若数学窗口给出更有依据的分布参数化或状态定义，可在通过本轮验收后替换。本轮不要围绕这一临时目标大规模调参。

### C4. 必测证据

- 至少一个训练 batch 中状态头参数梯度范数非零且有限；
- 固定其他输入，仅扰动 state embedding，Down/Up candidate 至少一项发生变化；
- 固定其他输入，仅扰动 state embedding，day key 发生变化；
- mask 掉 state 后的输出与正常输出不可完全相同；
- S4 前向不接收真实状态 target。

## 6. 修复任务 D：外生变量与多市场输入契约

### D1. 因果可得性

每个变量建立 availability manifest：

| 变量类型 | 预测时允许性 |
|---|---|
| 日前已发布预测（负荷、风、光伏、联络线等） | 可用，记录发布时间/lag |
| 当前目标日真实量 | 禁止 |
| 历史真实量 | 仅在满足 lag 的情况下可用，默认至少 24h |
| 日历、节假日、horizon | 可用 |

山东的日前价和实时价可作为同一市场的两个 target channel；必须使用 `target_id` token，按日期先划分，再生成两个 channel。严禁用同日另一个 target 的真实价格作为当前预测输入。

### D2. 任意数量 token 的正确实现

核心时刻 token 对任意数量外生 token 做交叉注意力：

- 每个外生值先用 S1 统计量规范化；
- 每个变量有稳定 `feature_id/type embedding`，不能只有 `[raw_value,1,0]`；
- 缺失变量使用 mask；
- `exog_mask` 必须传入 attention 的 key padding mask；
- 没有任何外生变量时加入显式 learned-null token，避免 all-masked NaN；
- learned-null 不能携带数据集标签或测试统计量；
- 可加入 `market_id` 和 `target_id` embedding，但不能要求每个公开数据集拥有相同特征集合。

### D3. 必测不变量

- 在 mask 后 token 中写入任意大数，输出不变；
- 无外生输入时输出有限、可反向传播；
- 交换两个变量的数值但保留 type id，与交换 type id 的结果不同；
- 规范化统计只来自 S1；
- 任一 actual 特征的可用行均满足 availability lag；
- DA/RT 同日划分一致，且交叉 target truth leakage 为零。

## 7. 修复任务 E：OOF、Gain Key 与记忆构造

### E1. 不再混用 fold-specific key

当前错误是：不同 fold 的 encoder 产生的 key 被放入同一空间，且 `metric_proj` 被重复调用。改为以下数据流：

1. 在 S2 做按时间前推的 cross-fitting，生成 OOF candidate 与 OOF gain；
2. 每条 `OOFRecord` 保存原始因果上下文、日期、target、host、candidate delta、realized gain、fold 边界和 checkpoint hash；
3. 所有 OOFRecord 生成完后，训练**唯一的** `GainKeyEncoder`；
4. raw context 只经过该统一 encoder 和一次 metric projection；
5. 再用全部 S2 训练最终 Bi-OMC；
6. 冻结最终 Bi-OMC 与 GainKeyEncoder；
7. 在 S3 上用该最终组合生成 candidate、key 与 gain，构建正式 memory。

若因最早一段没有足够历史无法产生合法 OOF，明确丢弃并写入 manifest；禁止用未来 fold 训练后回填过去。

### E2. Projection 只允许一次

选择并固定一种 API：

- `encode_raw(...)` 返回未投影表示，`project_metric(...)` 调用一次；或
- `encode_key(...)` 直接返回最终 key，调用方不得再投影。

增加 hook/counter 测试，保证完整前向路径中 metric projection 恰好一次。

### E3. 版本一致性

每条 memory episode 必须绑定：

- final Bi-OMC checkpoint hash；
- GainKeyEncoder checkpoint hash；
- host checkpoint hash；
- candidate definition/version；
- data/split manifest hash。

checkpoint 不匹配时加载必须失败，禁止 warning 后继续。

## 8. 修复任务 F：CAGM 检索与 DVG 校准

### F1. 24h day episode

每个 S3 记忆情节至少保存：

- `date_id`、`market_id`、`target_id`；
- day key；
- 24×3 的 candidate gain（Identity 为 0）；
- valid horizon mask；
- candidate/checkpoint hashes。

memory 必须进入 bundle/state_dict 或等价可序列化对象，不得只保存在 `persistent=False` 的临时 buffer。

### F2. S3 leave-one-day-out

校准 S3 日期 $d$ 时：

- 检索库中排除 $d$ 本身；
- 同日 DA/RT 是否可互检必须由显式配置控制，默认不使用对方真实 gain；
- 仅用 S3 选择 `k`、`eta`、`tau` 和 soft-soft / soft-hard 模式；
- S4 不得再选参数、更新 memory 或重新估计门槛。

`k/eta/tau` 不得继续硬编码。候选网格和校准目标写入配置，目标必须是相对 Identity 的风险调整动作价值，而非 S4 MAE。

### F3. 输出可审计

每个 S4 点输出：

- 三动作候选值；
- 三动作估计 gain/风险；
- soft 权重与最终 action；
- neighbor date ids、相似度和权重；
- 是否 abstain/Identity；
- 预测前 bundle hash。

这些字段用于诊断，不得包含 S4 真值衍生信息。

## 9. 修复任务 G：冻结与持久化基础

本轮只实现冻结能力，不启动正式冻结实验。

### G1. Bundle 最小内容

- host checkpoint/hash；
- S1 预处理、状态统计和 availability manifest；
- Bi-OMC、state/context encoder 与 GainKeyEncoder checkpoint；
- S3 memory；
- DVG `k/eta/tau`、gate 模式；
- dataset/market/target/seed/split/code commit；
- 完整配置与版本号。

### G2. Freeze API

提供明确的：

- `fit_s1_s2(...)`；
- `build_and_calibrate_s3(...)`；
- `freeze_bundle(...)`；
- `predict_s4(x, metadata)`。

`predict_s4` 的函数签名不得接收 `y_true`。冻结后所有参数 `requires_grad=False`，模型处于 eval，S4 前后 bundle 内容 hash 相同。

### G3. Round-trip

保存 bundle、清空进程、重新加载，在固定 batch 上：

- candidate、key、neighbor、gate、final prediction 在容差内一致；
- hash 与 manifest 一致；
- 缺少任何组件时 fail closed。

## 10. 修复任务 H：同行基线真实性

### H1. 标签规则

| 实际实现 | 结果表标签 |
|---|---|
| 固定 commit 的官方核心模型/训练/推理逻辑，仅做数据接口适配 | `PIR (official)` / `delta-Adapter Ada-Y (official)` |
| 只抽取部分模块、改训练目标或重写检索/归一化 | `limited reimplementation`，不能写 official |
| 官方代码无法在该数据/宿主上合法运行 | `unsupported_official`，保留空值与原因 |

严禁失败后自动回退到代理模型并沿用官方名称。

### H2. 固定来源

- PIR 官方源：  
  https://github.com/ustc-time-series/PIR/blob/fc372bb02090da887d4a20b614a6cfecbfd813d0/models/PIR.py
- δ-Adapter PostY 参考源：  
  https://github.com/Anoise/Adapter/blob/0add06ea7b4d2e0a84c364a8be72eef2676a92f2/AdaIntpX/experiments/exp_decom9_post_y.py

适配层必须记录上游 URL、commit、修改清单和许可证。PIR 的检索若是官方核心，不得删除后仍标官方；δ-Adapter 的样本构造、归一化和训练逻辑若被实质改动，同样不得标官方。

## 11. 修复任务 I：真实自动化测试

把“声明清单”变成实际调用模型/数据管线的测试。至少覆盖：

1. 全仓 Python 编译；
2. S1/S2/S3/S4 时间无交叉；
3. 所有方法的最终 S4 keyed index 完全一致；
4. host cache 全轴对齐与 CLI 生效；
5. candidate 使用 host 基准且全链路一致；
6. Down/Up 方向和零 delta identity；
7. state head 梯度非零；
8. state 反事实扰动改变 candidate 与 key；
9. learned-null 前向有限；
10. masked token 值不影响输出；
11. actual feature 满足 lag；
12. DA/RT date-first 且无真值交叉泄漏；
13. OOF fold 为时间前推、无 future-to-past；
14. 不混合 fold-specific key space；
15. metric projection 恰好一次；
16. S3 leave-one-day-out 不检索自身；
17. freeze 前后 hash 不变；
18. `predict_s4` 不接收/访问 S4 label；
19. memory/bundle round-trip 一致；
20. 官方 baseline 失败不伪装 fallback；
21. 相同 seed/config 输出一致；
22. prediction/evidence 含 timestamp、target、bundle hash。

原有随机数组 sign/identity stub 必须替换为真实 `BiOMC -> candidate builder -> DVG` 路径。测试数量可以多于 22，但报告必须给出每个契约对应的测试名。

## 12. 只允许的验证运行

### 12.1 静态与单元

执行并保存原始日志：

- `python -m compileall src experiments`；
- 项目现有测试；
- 本文新增契约测试；
- import smoke。

环境缺依赖时，应区分“环境未安装”与“代码失败”，给出锁定依赖和复现命令，不得把跳过记为通过。

### 12.2 微型端到端 smoke

仅用于检验通路：

- 至少一个无外生公开数据；
- 至少一个多外生数据（可用山东的小片段）；
- 宿主至少 Linear 与 PatchTST；
- 1 seed、1 epoch 或极小步数；
- 覆盖 S1→S2 OOF→S3 memory/calibration→freeze/reload→S4 predict；
- 不汇报成性能结论，不覆盖既有正式结果文件。

## 13. 验收门槛

只有全部满足，才可向下一轮提交审查：

- [ ] P0 问题全部关闭；
- [ ] 全仓编译通过；
- [ ] 22 项契约均有真实自动测试，非 stub；
- [ ] 微型 Linear 与 PatchTST 通路均完成；
- [ ] 所有方法共享完全相同 S4 keyed index；
- [ ] state 梯度与反事实证据成立；
- [ ] metric projection 一次且 key 空间统一；
- [ ] S3 自检索为零；
- [ ] bundle round-trip 一致，freeze hash 不变；
- [ ] official / limited / unsupported 标签真实；
- [ ] 未运行 half-exp，未更新论文性能主张；
- [ ] 未实现或偷换数学窗口的新损失。

任一项未满足，状态必须是 `NOT READY`，不得用“基本完成”代替。

## 14. 执行 AI 必须回传的报告

请用一个全新文件名生成修复报告，结构严格如下：

```markdown
# HCH v2 Repair Handoff
## 1. 基线
- 起始 commit：
- 结束 commit / diff：
- 环境与依赖：

## 2. 修改清单
| 问题 ID | 文件/符号 | 修改内容 | 对应测试 |

## 3. 契约测试
| 契约 | 测试名 | 结果 | 原始日志路径 |

## 4. 微型 smoke
| 数据 | host | 通路阶段 | 结果 | bundle hash |

## 5. 样本一致性
| dataset/target | S4 起止 | 共同样本数 | 各方法缺失数 |

## 6. 官方基线状态
| 方法 | upstream commit | official/limited/unsupported | 修改点/失败原因 |

## 7. 未解决项
- 必须如实列出，不得留空代替。

## 8. 最终状态
- READY FOR REVIEW / NOT READY
```

同时回传：

- 修改后的文件列表；
- `git diff --stat` 与关键 diff；
- 完整测试日志；
- 微型输出 manifest；
- 一个可加载 bundle 示例；
- 不得只给截图或汇总数字。

## 15. 审查后的下一步

执行 AI 完成后立即停止。由审查方核对代码与证据；与此同时数学窗口可独立研究新损失。只有“代码修复通过 + 数学方案通过 + 二者完成架构融合”三件事都成立，才解除 half-exp 文档的 HOLD。
