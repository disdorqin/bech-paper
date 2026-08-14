# HCH v2：T0/T1 训练微调与部件诊断执行文档

**用途**：给本地 AI 执行第一、第二阶段的代码调整与实验。  
**本轮不做**：第三阶段“数据契约修正 + 跨市场训练方案落地”；该部分只保留准备清单，等 T0/T1 结果确认后再决定是否单独出设计文档。  
**原则**：先证据、后改动；先修训练与可观测性，再判断部件是否需要改；不以调参掩盖结构问题。

---

## 0. 唯一基线与当前事实

### 0.1 数学与链路基线

严格以 `hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md` 为数学依据，主链路不能改成另一套：

```text
S1 候选
  → IAH-CRPS 候选分布
  → CAGM 查询剂量回放
  → 双事件提案（up/down）
  → 整日动作价值校准
  → DVG 的整日 LCB 门控
  → 最终动作 / identity fallback
```

本轮不得重新引入：`beta-mixing`、查询级 `rho`、逐小时 CS、任意多区间 WIS、额外训练 loss、独立事件检测头、硬事件阈值。事件统计只能作为离线诊断，不能变成训练目标或新的门控规则。

### 0.2 当前代码与报告的事实

以仓库 `disdorqin/bech-paper` 当前默认分支提交 `5182ef40ad4b2d4981dc8f9a21ff0553b7c440e6` 为起点，不回写旧 R1A/R1B 结果。

最新 R1B 报告已经给出以下事实：

1. `LearnedSig_main` 在 12 个 source domains、3 个 seed 的候选层 CRPS/ΔCRPS 与 LOHO 泛化结果总体稳定，当前不能直接断言“模块完全没有学到”或“只要增加 epoch 就会好”。
2. 当前训练曲线和早停信息显示候选层已经出现收敛迹象；因此 T0/T1 的目标是确认收敛边界、优化器/采样实现是否可靠，以及区分“候选分布学到了”与“后续动作/点读出没有转化为收益”。
3. 现有主链的点输出与 host identity 关系很强；如果 CRPS 改善而 MAE/sMAPE 不变，应先归类为“分布候选与点读出不一致”，不能把它误报成训练失败。
4. NEM-SA1 的 S3M 有效天数很短，C0 fallback 或门控不稳定首先应归类为证据不足；不能用短校准段否定跨市场候选层结论。
5. 当前主实验 `D_VALUE=0`，HCH 核心消费的是 host-relative、scale-free 的固定上下文；任意数量外生变量的分支尚未进入本轮，不能在论文中声称已经充分利用山东丰富特征或所有公开数据的异构特征。
6. 当前 PatchTST 实现是简化的 PatchTST-style host，不要在本轮把它写成官方 PatchTST 的严格复现。

必须对照的现有报告：

- [R1B 最终裁决](https://github.com/disdorqin/bech-paper/blob/5182ef40ad4b2d4981dc8f9a21ff0553b7c440e6/docs/%E8%AE%AD%E7%BB%83%E6%96%87%E4%BB%B6%E5%A4%B9/R1B/hch_v2_r1b_final_verdict_report_2026-08-14.md)
- [R1B Stage2A](https://github.com/disdorqin/bech-paper/blob/5182ef40ad4b2d4981dc8f9a21ff0553b7c440e6/docs/%E8%AE%AD%E7%BB%83%E6%96%87%E4%BB%B6%E5%A4%B9/R1B/hch_v2_r1b_stage2a_report_2026-08-14.md)
- [R1B host / feature schema audit](https://github.com/disdorqin/bech-paper/blob/5182ef40ad4b2d4981dc8f9a21ff0553b7c440e6/docs/paper_prep/v2_final_prep/r1b_domain_feature_schema_audit_v0.1.md)
- [R1B 文献工程侦察](https://github.com/disdorqin/bech-paper/blob/5182ef40ad4b2d4981dc8f9a21ff0553b7c440e6/docs/paper_prep/v2_final_prep/r1b_literature_scout_v0.1.md)

---

## 1. 本轮执行边界

### 1.1 允许做的事

- 修复训练入口、采样器、随机种子、checkpoint 选择、日志和可重复性问题。
- 在不改数学链路的前提下，增加训练轮数/早停耐心值，并提供一个受控的学习率调度实验。
- 为每个阶段输出中间张量、梯度、每域指标和动作链指标，定位部件失效位置。
- 只在证据指向确定的数值/实现错误时做最小代码修复。

### 1.2 本轮禁止做的事

- 不新增损失函数，不把 `IAH-CRPS` 改成 `MSE + λ loss`，不加入事件分类头。
- 不修改 `S1 → IAH → CAGM → 双事件 → 整日价值校准 → LCB` 的顺序。
- 不启用尚未完成的数据契约/外生变量分支，不把山东特征硬塞进当前 13 维核心输入。
- 不引入新的硬阈值、手工尖峰规则、按测试集调参、测试集参与 checkpoint 选择。
- 不删除、覆盖或重命名旧的 R1B 结果；所有新结果必须使用新的时间戳目录和 manifest。
- 不把一次成功的单市场/单 seed 结果写成跨市场 SOTA。

---

## 2. 阶段 T0：训练实现审计与可观测性补齐

T0 不是“调参”，而是先确认当前训练确实在优化正确对象。

### 2.1 入口与唯一执行路径

1. 找到仓库当前 R1B 的正式 runner，以现有文档记录的 runner 为准；先运行其 `--help`/配置打印，不要凭空新建第二套训练入口。
2. 确认正式训练调用 `src/universal_trainer.py`，使用等域采样和 macro-S2V checkpoint 选择。
3. 确认评估调用 `src/hch_v2_pipeline.py` 的权威链路。若仍存在只在 runner 中复制的 analytic action path，必须先标记并统一为同一条默认路径；在统一前不得把两条路径的结果混在一张表里。
4. 检查每个 batch 的 domain descriptor 与实际 market/host 绑定；加入断言，发现错配立即失败，不允许静默填零。

### 2.2 必须记录的训练状态

每个 epoch、每个 seed、每个 domain 至少记录：

- 训练/验证的 IAH-CRPS、有效天数、scale-invalid 天数；
- host 原始 CRPS、候选 CRPS、`ΔCRPS = candidate - host`；
- raw MAE、raw sMAPE（只作点预测诊断，不单独用于选 checkpoint）；
- `S2T/S2V` 样本数、每域更新次数、每域梯度范数、参数更新次数；
- IAH mass entropy、各 atom 的 alive ratio、shift 的 p50/p95、NaN/Inf 计数；
- best epoch、early-stop 计数、学习率、总步数、配置 hash、代码 commit hash；
- 阶段性动作链的 A0/A1/A2、整日价值、LCB 接受/拒绝和 identity fallback 统计。

若现有 logger 缺少上述字段，只补日志和离线读取，不改变模型计算图。

### 2.3 数据切分与泄漏检查

仍使用仓库现有时间切分：`S1 train → S2T tuning → S2V validation → S3M freeze-time calibration → S4 final test`。本轮：

- S1 只用于参数学习；
- S2T 只用于训练过程中的 tuning；
- S2V 只用于 macro checkpoint 选择和 T0/T1 决策；
- S3M 只在冻结后的动作校准/回放阶段使用；
- S4 不得被读取、汇总或用于任何选择。

训练启动时输出并断言每个 domain 的时间边界、时区、DST 处理、有效小时数、负价占比和特征 hash。发现时间交叉或特征 hash 与 manifest 不符时停止实验。

---

## 3. 阶段 T1：训练收敛与优化方法微调

### 3.1 固定控制组

先复跑一个当前控制组，不做任何结构改动：

```text
配置：LearnedSig_main
采样：equal-domain sampling（每域相同更新预算）
优化器：当前 AdamW 配置
训练：当前 epoch/patience 配置
seed：0
```

控制组必须能复现最新 R1B 的 macro-S2V 及 per-domain 方向性；复现不了先修 T0，暂停 T1。

### 3.2 只做两个受控训练臂

不要进行无边界网格搜索。控制组通过后，运行下面两个训练臂：

**T1-A：延长当前训练**

- 保持模型、IAH-CRPS、采样器、batch、LR、weight decay、clip 全部不变；
- 将最大 epoch 从当前短配置提高到 48（若机器预算不足，至少 36），early-stop patience 提高到 10；
- checkpoint 仍按 macro-S2V，而不是平均 S2T 或单一市场选择；
- 保存完整曲线，重点查看最后若干 epoch 是否仍下降、是否出现 train/validation gap。

**T1-B：受控调度器**

- 以 T1-A 为基础，只加入一个可复现的学习率调度策略（推荐 3 个 epoch warm-up 后 cosine decay；若现有框架不便实现，使用验证集 plateau 调度，但只能二选一）；
- 初始 LR、AdamW、weight decay、梯度裁剪、模型初始化不变；
- 不同时修改 epoch、batch、网络宽度、初始化和 loss；
- scheduler 配置写入 manifest，便于逐项归因。

执行顺序：`控制组 → T1-A(seed 0) → T1-B(seed 0)`。只有当两者方向明确且没有泄漏，再对更优者运行 seed 1/2；不要对两个臂都无条件扩展到多 seed。

### 3.3 训练阶段的判定，不用拍脑袋调参

根据曲线和 per-domain 结果给出以下分类之一：

| 现象 | 判定 | 下一步 |
|---|---|---|
| S2T/S2V 在末尾仍有稳定下降，梯度健康，未出现明显 gap | 训练尚未充分 | 采用收敛更好的臂；只增加必要训练预算 |
| S2T 下降但 S2V 反弹，gap 变大 | 过拟合 | 保留 macro-S2V 最优 checkpoint；不要继续加 epoch |
| 候选 CRPS 改善，raw MAE/sMAPE 基本不变 | 候选分布已学到，但点读出未转化 | 转入 T2 点读出/动作链诊断，不把它写成训练失败 |
| source macro 好、单一 unseen domain 崩溃，且更新预算不均 | 域采样/域绑定问题 | 先修采样与 descriptor 断言，不加新数据 |
| 三个 seed、多个域方向稳定，调度器只带来噪声 | 训练层已足够 | 停止 T1，进入 T2 或准备第三阶段 |
| loss/梯度几乎不变、mass/shift 长期死掉 | 计算图或初始化错误 | 只修数值/梯度错误，修复后从控制组重跑 |

---

## 4. 阶段 T2：部件问题的最小诊断与必要修复

T2 不是最终论文消融实验。它是内部故障隔离：先用中间结果判断哪个部件失效，再决定是否需要改一个部件。官方主链默认始终保持 v0.3。

### 4.1 先输出“同一条样本”的链路快照

对固定 seed、固定 batch、固定日期，保存：

```text
host_raw
→ S1 candidate
→ IAH atom mass / shift / scale
→ CAGM dose / replay output
→ up/down proposal
→ whole-day action value
→ LCB
→ accepted action 或 identity fallback
```

每一层同时保存 shape、finite 状态、日期/域 id、以及与上一层的差值。这样可以区分“候选没有改进”“动作价值把改进吃掉”“LCB 过于保守”三类问题。

### 4.2 四类部件诊断

#### B1. IAH 候选分布

- 检查 atom 是否有梯度、mass 是否退化、shift 是否饱和、scale-invalid 是否集中在某一域；
- 只用 `LearnedSig_main` 与已有 `PlainCore`/现有可运行配置作诊断，不新增 head；
- 若候选层 ΔCRPS 已稳定为负，不因点指标没有变化就改 IAH 数学；
- 若候选层在多个域均无改进且健康指标异常，优先修数据归一化、梯度、初始化或 batch 绑定。

#### B2. CAGM 查询剂量回放

- 检查回放样本是否来自训练允许的历史窗口，是否发生未来信息泄漏；
- 检查剂量/检索权重是否有 NaN、全零或只由单一域贡献；
- 对同一候选输出比较“候选层”和“进入 CAGM 后”的 CRPS/动作价值变化；
- 只要确认是实现错误才修，不能为了提高某个域而手工改变剂量或相似度阈值。

#### B3. 双事件提案与整日 LCB 门控

- 分别报告 up/down proposal 的数量、幅值、被 LCB 接受的比例、identity fallback、整日价值；
- 事件统计按训练集分位数做离线分组即可，不增加训练事件标签和硬门槛；
- NEM-SA1、EPEX-FR 等有效 S3M 很短的域，先标记证据不足，不直接改门控；
- 若候选层改善但动作层恶化，先检查动作价值校准与 LCB 实现是否使用了同一整日索引，再决定是否需要最小修复。

#### B4. 点输出与经济指标

- 当前点输出可能接近 host identity；因此同时报告分布 CRPS 和 raw MAE/sMAPE；
- 若 CRPS 改善但 MAE/sMAPE 不改善，这不是“候选没学到”，而是“分布到点动作的读出/门控价值尚未证明”；
- 本轮不凭单一 MAE 设计新的 point loss；把该结果写入 T2 决策，留给后续架构迭代。

### 4.3 允许的最小修复

只允许以下类型的修改，并且每次修改单独建 run、单独报告：

1. 维度、mask、日期索引、market/host descriptor 错配；
2. NaN/Inf、scale-invalid、空 batch、梯度未连通、checkpoint 误选；
3. runner 与 `src/hch_v2_pipeline.py` 不一致造成的重复/旁路实现；
4. 已经存在但没有被调用的日志、验证和 manifest。

如果证据只表明“某部件效果可能不够强”，但没有实现错误，不在 T2 私自重写该部件；先保留证据，等下一轮明确提出结构修改。

---

## 5. 本轮实验范围与指标

### 5.1 数据范围

- 使用当前 R1B 已验证的 12 个 source domains 做训练/验证；保持 market × host 的原始分组。
- 保留 `NORD_DK1` 等 leave-one-market-out / unseen-host 评估作为验证证据，但不把 S4 结果用于选择。
- 本轮不新增 NYISO/ERCOT 等数据；第三阶段准备完成后再决定。
- 国内山东数据如果已接入，必须以独立 domain 写入 manifest，先验证字段、时间、时区、DA/RT 标签，不得无 schema 地拼到国外数据。

### 5.2 精简指标集

本轮只保留三层指标，避免把指标堆成论文拼盘：

1. **主指标**：daily IAH-CRPS、相对 host 的 ΔCRPS、macro 与 per-domain；
2. **点预测诊断**：MAE、sMAPE（说明它们只评价 point readout，不代表分布质量）；
3. **动作/安全诊断**：整日 action value、A0/A1/A2、LCB 接受率、identity fallback、有效天数。

负价、尖峰、时序偏差、幅值误差只作为离线分组诊断，不加入新的训练目标。分组边界从训练集分位数或数据集自然区间产生，并在报告中记录来源，不能看测试集后手工改阈值。

---

## 6. 结果文件与交付要求

每个新 run 建立独立目录，例如：

```text
docs/训练文件夹/T0_T1/<timestamp>_<arm>_<seed>/
```

至少包含：

- `config.json`：完整配置、数据 split、seed、commit hash；
- `training_curve.csv`：epoch/step/域级指标/梯度/学习率；
- `per_domain_health.csv`：更新数、有效天数、NaN、mass、shift、ΔCRPS；
- `chain_snapshot.jsonl`：T2 固定样本的 S1→IAH→CAGM→DVG 快照；
- `checkpoint_hashes.json`：best、last、control 的 hash；
- `repro_check.json`：同配置重复运行的差异；
- `t0_t1_decision.md`：按第 3.3 节分类，明确是训练问题、部件问题还是已足够；
- `README.md`：可复制的实际命令、运行时长、失败重试记录。

禁止把“测试集最高分”“单域最好分数”作为最终结论。报告必须同时给 macro、最差域、按市场/host 分组、有效天数和数据缺失说明。

---

## 7. 第三阶段：只准备，不实施

下面内容是下一轮“数据契约 + 跨市场训练”设计的输入，本轮不要改代码。

### 7.1 已知数据异质性

| 数据族 | 当前已知特征 | 对统一投喂的风险 |
|---|---|---|
| LAGO-DE / LAGO-PJM | DA、小时级、外生 forecast 列少、负价比例低 | 不同货币/价格尺度，不能直接拼接；极端样本稀疏 |
| NEM-SA1 | 高波动、负价占比较高、有效 S3M 天数短、外生列形态不同 | 负价与尖峰分布和欧洲 DA 不同；短校准段会造成 gate 不稳定 |
| NORD-DK1 | DA、无外生变量、负价中等 | 缺失外生变量时不能当成零值特征 |
| EPEX FR/BE/NL | 市场/时区/价格上限不同，负价与尖峰比例不同 | DST、币种、价格上限和尾部定义需要显式记录 |
| GEFCom / 无负价数据 | 负价缺失或几乎不存在 | 只能评价高尾/低位，不应宣称验证了真实负价修正 |
| 山东及其他中国市场 | 预测与实际辅助列丰富，DA/RT 信息可能并存 | 需要区分 D-1 可得 forecast、滞后 actual、目标 price，防止未来信息泄漏 |

当前 HCH 核心只消费固定 scale-free host-relative context；这反而是本轮跨 schema 候选层结果可解释的原因，但不是“已经利用了所有特征”。未来数据契约应至少区分：必需 host 输入、D-1 可得 forecast covariates、滞后 actual covariates、市场元数据、缺失/learned-null 状态。

### 7.2 文献准备方向

第三阶段至少整理以下工程模式，先写证据矩阵，不能直接照搬：

- **后处理校正**：PIR（NeurIPS 2025）的失败识别 + 局部/全局 revision；Post-Training Corrections 的冻结 backbone 顺序校正；δ-Adapter（ICLR 2026）的零初始化小残差、batch/online 训练与校准。  
  [PIR](https://arxiv.org/abs/2505.23583) · [Post-Training Corrections](https://arxiv.org/abs/2505.15354) · [δ-Adapter](https://arxiv.org/abs/2601.20280)
- **电价后处理与时间隔离**：滚动 point forecast + 独立滚动校准窗口，重点核对 OOF、时间泄漏、DST 和实际部署顺序。  
  [Electricity rolling postprocessing](https://arxiv.org/html/2507.15079v1)
- **跨域/跨市场训练**：等域采样、leave-one-market-out、源/目标相似性分析和预训练-only 对照；先吸收 UniTime/GIFT-Eval 与电价迁移学习的实验规范，不马上引入大型 foundation model。  
  [UniTime](https://arxiv.org/abs/2310.09751) · [Electricity transfer learning](https://arxiv.org/abs/2007.03762)
- **TPWRS/TSG/TII/ICDE/IJCAI 工程审查重点**：时间切分、数据可得性、DA/RT 信息边界、域外测试、最差域报告、运行成本和可复现 manifest。先把仓库已有 literature scout 中的候选逐篇标注“可借鉴的训练/评估协议”，不要只抄网络结构。

### 7.3 可能新增的数据（条件式）

只有当 T0/T1 证明当前 source domains 的候选层已稳定、而跨 RT/丰富特征仍是主要空白时，才启动新增数据：

1. **NYISO**：DA 小时 + RT 5-min，优先作为 DA→RT realization 的公开主候选；
2. **ERCOT**：DA 小时 + RT 15-min，作为第二候选；
3. **PJM RT**：先核实归档与再分发许可，再决定是否纳入；
4. 其他数据必须先通过时间分辨率、时区/DST、币种/价格上限、负价比例、DA/RT 可得性和许可证审计。

新增数据应作为新 domain 加入 equal-domain sampler，不能用更多样本量让某一市场支配训练；DA 目标与 RT realization 也应分通道记录，不能混成一个无标签价格序列。

---

## 8. 给本地 AI 的直接执行顺序

```text
1. 阅读本文件、数学 core v0.3、最新 R1B final verdict 和 feature schema audit。
2. 检查当前 runner 是否只有一条权威 HCH v2 pipeline；若不是，先统一/加断言，不先调参。
3. 完成 T0 日志、manifest、数据边界和 descriptor 绑定检查；跑一个小 smoke 验证。
4. 复现当前控制组（LearnedSig_main, seed 0）；复现不了就停在 T0。
5. 依次执行 T1-A、T1-B；只在更优臂上补 seed 1/2；全程不读取 S4。
6. 对控制组与更优臂各抽固定样本，生成 T2 chain snapshot，判断问题位于候选、CAGM、动作价值/LCB 还是点读出。
7. 只有发现实现错误才做最小修复；每个修复单独 run，禁止合并多个改动后归因。
8. 生成第 6 节全部文件和一份简短 decision report，明确：
   - 训练是否充分；
   - 哪个域/host 仍异常；
   - 是否存在候选→动作或分布→点读出断层；
   - 是否需要进入下一轮数据契约/跨市场设计。
9. 不执行第 7 节的新增数据和外生变量改造，等待主调研方确认。
```

**完成标准**：不是“跑出一个更高分”，而是能用同一份 manifest 复现控制组，能解释训练曲线和每个链路部件的变化，并明确下一轮应该改训练、改部件，还是进入数据契约设计。

