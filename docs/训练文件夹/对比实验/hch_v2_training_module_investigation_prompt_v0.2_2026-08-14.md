# HCH-v2 训练与模块问题调查提示词 v0.2

> 用途：交给本地代码 AI。先调查、证伪和排序，再决定是否改代码。
> 本文件替代此前的“直接延长训练/加入学习率调度”的 T0/T1 执行文档。
> 在本调查报告完成并经过人工确认前，不得修改 src/ 生产代码，不得覆盖旧实验结果，不得把 holdout 数据加入训练。

---

## 0. 任务结论与当前优先级

本轮不是让你马上“多训练几轮”，而是回答：

1. 当前 IAH 候选分布是否真的学到了可迁移的修正信息；
2. 候选分布是否被正确转换成点预测；
3. 点预测、动作提案、整日价值和 LCB 门控是否被错误地混成一个评价路径；
4. 当前训练器是否造成跨市场负迁移或域间优化振荡；
5. 哪些修改最可能带来实际收益，哪些修改只是增加复杂度。

当前已知但必须重新核验的事实：

- R1B 候选层 IAH-CRPS 在源域和 holdout 面板上总体改善，说明不能直接假设“模型没有收敛”；
- experiments/08-hch-v2/r1b_stage2d_action_chain.py 的 forecast_metrics() 当前读取 day["host_day"]，需要核对点指标是否仍然只评估冻结宿主；
- experiments/08-hch-v2/r1b_stage2a_panel.py 的 eval_panel_domain() 当前使用 out["x_identity"] 作为 cand_pred，而 x_identity 在 src/iah_candidate.py 中本质上是宿主原值；
- 当前主实验 D_VALUE=0，OptionalCovariateEncoder 没有真正参与主训练；
- UniversalCoreTrainer 使用固定 AdamW、按域逐 batch 更新、K=median(n_batches)，没有 warmup/scheduler，也没有宏域梯度累积；
- R1B action-chain 出现过局部 GATING_HURTS，这不应直接归因于 universal candidate 训练失败。

以上内容只是待核验假设，不得直接当作最终结论。

---

## 1. 代码和协议边界

仓库：https://github.com/disdorqin/bech-paper

开始前：

1. 检查当前 HEAD、工作区状态和最近提交；不要只依据本提示词中的旧 commit。
2. 阅读仓库 AGENTS.md、最新 docs/paper_prep/、docs/训练文件夹/R1B/ 和 experiments/08-hch-v2/。
3. 找出真正被 R1B runner 调用的代码路径，并把 legacy/archive 路径单独列出。
4. 保存当前代码版本、运行环境、配置、数据 split hash 和 runner hash。

数学硬约束必须保持：

~~~
host-relative asinh geometry
→ three-atom IAH candidate
→ IAH-CRPS 唯一候选训练目标
→ query-dose replay
→ 至多一个 Down + 一个 Up 连续事件
→ whole-day action-value calibration
→ LCB > 0 才执行
~~~

禁止在本轮擅自加入：

- BCE/event detection head；
- MAE + λ loss、额外 tail loss、交易 loss 或其他辅助监督；
- beta-mixing、查询级 rho、逐小时 conformal guarantee；
- 任意多区间动作；
- S4 标签参与训练、调参或门控选择；
- 市场 ID 预测 token；
- 把 holdout 市场加入 source training 以修复结果。

可以提出但暂不实现的最小候选：

- 从已有三原子分布导出不同指标对应的点预测；
- 更准确的跨域 batch/gradient 组织；
- 保持单一 IAH-CRPS 的优化器和采样改进；
- 若读出审计失败，才讨论中心位置的最小 location shift；
- Rich covariate branch 的数据契约和隔离实验。

---

## 2. 第一部分：端到端代码审计

请不要只读模型文件。画出实际执行链：

~~~
dataset loader
→ host cache
→ S1/S2/S3/S4 split
→ DomainInfo / DomainBatch
→ candidate training
→ checkpoint selection
→ candidate evaluation
→ query replay
→ double-event proposal
→ local calibration
→ DVG
→ final forecast metrics
~~~

对每一段回答：输入是什么、是否含标签、是否允许更新、输出是什么、在哪个 split 上运行、是否可能泄漏。

### 2.1 候选输出审计

重点检查：

- src/iah_candidate.py 的 x_identity/x_down/x_up 是否被当作 point forecast；
- 三原子分布是否有明确的 raw-space atom：
  x_a = s sinh(z_a), a∈{-,0,+}；
- 是否存在明确的分布到点预测函数；
- 点预测是否仍等于 host identity；
- m_minus/m_plus 是否真正进入最终预测；
- 质量指标是否评估 candidate、action 结果，还是宿主原值；
- CRPS 改善与 MAE/sMAPE 改善是否被错误地当成同一件事。

必须给出一个最小 toy test：

~~~
构造 host=0、target=正向偏移/负向偏移的简单样本；
给定一个非零 m 和非对称 w；
验证 mean、weighted median、identity 是否产生不同预测；
验证最终 evaluator 是否真的读取了 candidate 输出。
~~~

### 2.2 训练链审计

检查 src/universal_trainer.py：

- 域单位是 (market, host) 还是其他单位；
- 每个域实际得到多少 optimizer update；
- K=median(n_batches) 是否截断长域、重复短域；
- 最后一个小 batch 是否被赋予与完整 batch 相同权重；
- 当前 loss 是按小时、按日还是按 batch 加权；
- 是否逐域 sequential update，是否存在域间振荡；
- train/S2V 曲线是否逐域保存；
- checkpoint 是否保存并重新评估 selected checkpoint，而不是读取 last epoch；
- 是否记录 mass entropy、shift alive、shift p50/p95、grad norm、NaN、scale-invalid；
- 是否有任何训练配置在 runner 和 r1a_run.py 中重复定义而可能不一致。

输出一张表：

| 检查项 | 当前实现 | 科学风险 | 是否需要改 | 证据路径 |
|---|---|---|---|---|

### 2.3 Action-chain 审计

确认 R1B runner 使用的是哪一套实现：

- src/hch_v2_pipeline.py 的 canonical pipeline；
- 还是 r1a9_action_calibration.py + r1a11_prequential_calibration_router.py 的实验 action-chain。

如果两者不同，必须说明：

1. 哪个才是当前数学架构对应的正式实现；
2. R1B 结果实际证明了哪一条链；
3. 是否存在“文档声称 v0.3，代码实际跑旧链”的问题；
4. 是否需要先做协议对齐，再做训练改进。

重点审计：

- query dose 是否使用 query day 的 dose 回放到 history day；
- 候选区间是否至多一个 Down 与一个 Up；
- action value 是否按整日而非逐小时授权；
- S3M 与 S3C 是否严格分离；
- DVG q 是否在冻结前完成；
- LCB 是否只在 S4 inference 时执行；
- GATING_HURTS 是 candidate 问题、local calibrator 问题还是证据不足问题。

---

## 3. 第二部分：分布到点预测的独立调查

在不重新训练、不增加 loss 的前提下，使用已有三原子分布生成：

1. Identity；
2. raw-space weighted mean；
3. raw-space weighted median；
4. 由预测分布最小化期望 sMAPE 的数值 Bayes action；
5. 可选的诊断性 shrinkage，不得使用 S4 选择参数。

对每个输出分别评估：

- overall MAE；
- RMSE；
- no-floor sMAPE；
- high-tail MAE；
- low-price/negative-price MAE；
- event recall、timing error、magnitude error。

划分必须明确：

- S2V：可以做读出选择和诊断；
- S3M/S3C：只用于 action calibration；
- S4：只做最终冻结评估。

输出“分布是否包含点定位信息”的三种结论之一：

~~~
READOUT_SUFFICIENT
LOCAL_READOUT_NEEDED
LOCATION_CAPACITY_INSUFFICIENT
~~~

只有第三种结论成立时，才允许提出 z_c=z_0+δ 一类最小结构修改；不能因为点指标暂时不好就直接修改 IAH 几何。

---

## 4. 第三部分：候选剂量和动作方向调查

不要新增事件检测头。使用训练/验证标签只做离线诊断。

### 4.1 候选方向一致性

对每个小时计算：

~~~
residual_z = zY - z0
predicted direction = sign(m_plus - m_minus)
~~~

检查：

- Down/Up 方向准确率；
- m_minus/m_plus 与真实残差幅值的相关性；
- m 与实际 IAH gain 的相关性；
- 高尾、低尾、负价、正常期分别统计；
- 是否出现大量非零剂量但 action gain 为负。

### 4.2 动作层分解

分别报告：

1. candidate distribution quality；
2. directional proposal quality；
3. contiguous interval selection quality；
4. local C0/C3 calibration quality；
5. DVG conservativeness；
6. final realized action value。

如果 candidate CRPS 好但 action value 差，标签必须是：

~~~
LOCAL_ACTION_OR_PROPOSAL_LIMITED
~~~

不能写成 candidate training failed。

---

## 5. 第四部分：跨市场训练方式调查

当前训练的科学目标是：

L_universal = (1/|G|) Σ_g E_{d∼g}[L_IAH-CRPS(d)]

调查以下两个版本，不立即改生产代码。

### Version A：当前 sequential domain update

~~~
domain batch → step → domain batch → step
~~~

### Version B：macro-domain gradient accumulation

~~~
每个市场/宿主各取一个 batch
分别反向传播 loss / |G|
最后统一 optimizer.step()
~~~

比较：

- macro S2V CRPS；
- 最差域 CRPS；
- 域间方差；
- source/holdout gap；
- shift/mass 健康度；
- seed 稳定性；
- 实际日期覆盖率和重复率。

不得同时改变 sampler、optimizer、学习率、epoch 和模型结构。每次只改变一个因素。

### 5.1 学习率和训练时长

只有当曲线显示仍在收敛时，才比较：

- fixed AdamW；
- linear warmup + cosine decay；
- 延长训练但保持总样本曝光可比。

不能把“训练轮数增加后偶然变好”直接写成模块有效性。

### 5.2 域采样和数据扩充

调查：

- 是否每个市场被 4 个宿主重复计权；
- 增加新市场后市场级权重是否改变；
- 是否应该先 market uniform，再 host uniform，再 day sampling；
- 是否存在长域/短域、负价稀少域、尖峰稀少域；
- 是否可以使用宿主可见的连续模式描述进行 coverage sampling，而不是硬事件阈值。

holdout 数据只能用于评估，不能为了修复结果被加入训练。

---

## 6. 第五部分：Rich covariate branch 调查

当前主实验 D_VALUE=0，因此不能声称 HCH 已经利用山东丰富外生变量。

调查但暂不实现：

- 山东 DA/RT 特征如何映射到 KNOWN_FUTURE、OBSERVED_PAST、CALENDAR、STATIC；
- 国外 price-only 数据如何输入 learned-null token；
- 不同市场变量缺失时是否维度和语义一致；
- 变量归一化是否只使用训练段；
- optional branch 是否保持 zero-init residual；
- 是否需要 modality dropout 防止 Rich branch 依赖某个市场特征；
- HCH-Core 与 HCH-Rich 是否应该成为两个正式实验条件。

禁止把山东私有列名硬编码成公开数据集的必需输入。

输出数据契约草案：

~~~
required: host forecast + time index + scale-free history
optional: typed exogenous tokens
missing optional: learned-null token
target: never enters current-day input
~~~

---

## 7. 文献调查要求

本轮必须使用真实可点击的论文链接，优先论文原文、会议页、作者代码仓库和官方预印本。不要用搜索摘要代替论文结论，不要编造链接。

至少覆盖以下方向，并说明“能否迁移到 HCH、迁移代价、是否违反当前数学边界”：

### A. 概率分布到点决策

- proper scoring rules；
- Bayes act from predictive distribution；
- EMOS / distributional post-processing；
- QRA、IDR、conformal probabilistic post-processing；
- electricity price point-to-probabilistic post-processing。

### B. 冻结宿主后处理

- PIR / instance-aware post-hoc revision；
- δ-Adapter / bounded residual correction；
- residual alignment、shrinkage、safe abstention；
- local/global retrieval 但必须审计 leakage。

### C. 跨域/跨市场训练

- time-series domain generalization；
- domain convergence imbalance；
- domain-balanced sampling；
- pattern-balanced sampling；
- shared representation + domain-adaptive interface。

### D. Prequential calibration 和在线适配

- KDD 2024 SOLID；
- ICLR 2025 Fast and Slow Streams；
- time-series test-time adaptation；
- 严格信息不可见和 delayed label protocol。

### E. 电力价格极端事件

- spike-aware probabilistic electricity forecasting；
- negative-price forecasting；
- high/low tail calibration；
- day-ahead/real-time price post-processing；
- statistical metrics 与 economic metrics 的分离。

可从以下真实来源开始核对，但不能只停留在这些论文：

- PIR, NeurIPS 2025: https://papers.neurips.cc/paper_files/paper/2025/file/331c41353b053683e17f7c88a797701d-Paper-Conference.pdf
- δ-Adapter, ICLR 2026 preprint: https://arxiv.org/html/2601.20280v1
- Gneiting, Bayes point forecasts: https://www.bundesbank.de/resource/blob/635562/0d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf
- SOLID, KDD 2024: https://arxiv.org/abs/2310.14838
- Fast and Slow Streams, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html
- UniTime: https://arxiv.org/abs/2310.09751
- BLAST: https://arxiv.org/html/2505.17871v2
- DAF, ICML 2022: https://proceedings.mlr.press/v162/jin22d.html
- Isotonic QRA for electricity price uncertainty: https://arxiv.org/html/2507.15079v1
- Point-to-probabilistic electricity price post-processing: https://arxiv.org/html/2404.02270v2

文献表必须包含：

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|

---

## 8. 最终输出格式

请生成一份新的调查报告，不覆盖旧文件，建议文件名：

~~~
docs/训练文件夹/训练调查/hch_v2_training_module_investigation_report_v0.1_2026-08-14.md
~~~

报告必须包含：

### 8.1 已核验材料和代码路径

列出实际读取的 commit、runner、核心源码、实验结果和文档。

### 8.2 当前问题清单

按 P0/P1/P2/P3 排序，每项包含：

- 问题描述；
- 代码证据；
- 是否影响当前结论；
- 是否是评估 bug、训练问题、模块表达能力问题或 action-chain 问题。

### 8.3 证伪矩阵

| 假设 | 支持证据 | 反证方式 | 最小实验 | 当前结论 |
|---|---|---|---|---|
| H1：点指标差是训练未收敛 |  |  |  |  |
| H2：点指标差是读出错误 |  |  |  |  |
| H3：IAH 分布没有位置修正能力 |  |  |  |  |
| H4：跨域 sequential update 造成负迁移 |  |  |  |  |
| H5：action-chain 而非 candidate 是瓶颈 |  |  |  |  |
| H6：外生变量接口缺失导致山东域失败 |  |  |  |  |

### 8.4 修改候选排序

| 修改候选 | 预期收益 | 实现难度 | 论文风险 | 是否马上做 |
|---|---:|---:|---:|---|
| 修正 point readout/evaluator | 高 | 低 | 低 | 是 |
| 三原子 Bayes readout | 高 | 中 | 低 | 是 |
| macro-domain gradient accumulation | 中 | 中 | 低 | 诊断后 |
| warmup/cosine | 低-中 | 低 | 低 | 诊断后 |
| location shift | 未知 | 中 | 中-高 | 仅在读出失败后 |
| Rich covariate branch | 未知 | 中-高 | 中 | 单独实验 |
| 新事件头/新辅助 loss | 不确定 | 高 | 高 | 否 |

### 8.5 推荐执行顺序

必须给出唯一排序，不得同时建议十个方向并要求全部实现。默认排序应为：

~~~
P0 输出和协议审计
→ P1 三原子分布到点读出审计
→ P2 候选方向/动作分解
→ P3 跨域训练器对照
→ P4 Rich branch 数据契约
→ P5 必要时最小 location capacity 修改
~~~

### 8.6 停止条件

如果发现以下任一情况，必须停止修改生产代码并报告：

- 点指标当前实际上评估的是宿主原值；
- S4 标签参与了 readout、action 或 checkpoint 选择；
- action-chain 与 v0.3 数学链不一致；
- 候选 CRPS 结果无法由当前代码复现；
- holdout 被用于训练或调参；
- 任何文献结论无法提供真实链接。

---

## 9. 本轮明确禁止的行为

- 不要直接把旧 T1 文档的延长训练方案执行一遍；
- 不要先扩大数据集再解释结果；
- 不要把 S4 搬回训练集；
- 不要为了提高 MAE 加第二个监督 loss；
- 不要把 candidate CRPS improved 写成 final point forecast improved；
- 不要把单个 EPEX_FR:PatchTST 的 GATING_HURTS 直接归因于 universal core；
- 不要把零样本、few-shot、target full-shot 三种协议混成一个结论；
- 不要覆盖旧实验结果或复用旧文件名。

最终只提交：

1. 调查报告；
2. 文献核查表；
3. 证伪实验计划；
4. 修改候选排序；
5. 明确说明“下一步是否允许修改代码”。

本轮报告完成后暂停，等待人工确认。

