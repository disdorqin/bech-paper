# HCH-v2 R1B — 文献侦察 (Literature Scout) v0.1

**日期:** 2026-08-13
**来源任务:** R1B Two-Hour Sprint §21 (Parallel literature scouting — inspiration, not implementation)
**范围:** 一手论文 / 官方仓库;只写文档,不改代码
**预算:** 20–30 分钟

---

## 0. 核心判据

本侦察的唯一判据:

> **哪种评估设计能让我们的 universality claim 更难被伪造?**

据此把文献分成两类价值:

1. **评估设计价值(高优先级)**:LOMO / leave-one-host-out / 零样本跨市场 / seen–unseen 严格分离 / 泄漏审计 / 强基线对照。这些直接决定 R1B 的结论能否被审稿人击穿。
2. **技术价值(低优先级)**:某项训练技巧在论文里报告了收益。**一律不因"论文报告收益"而推荐抄技术**。

---

## 1. 评估设计类(最优先 —— 决定 claim 是否可伪造)

### 1.1 GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation

- **paper:** Aksu et al., Salesforce AI Research / NUS
- **year:** 2024 (arXiv 2410.10393)
- **来源:** https://arxiv.org/abs/2410.10393 · 官方仓库 https://github.com/SalesforceAIResearch/gift-eval
- **确切实验想法:**
  - 把评估拆成 **seen(全量微调)** 与 **unseen(零样本)** 双轨:每个数据集给 train/val/test split,基础模型只允许在 test 上推理。
  - 提供 **non-leaking pretraining pool**(约 230B 点、88 个数据集),与 28 个评估数据集**零重叠**,保证零样本评估没有数据泄漏。
  - 覆盖 7 域 × 10 频段 × 单/多变体 × 短/长 horizon,用 GluonTS 统一评估。
- **解决的问题:** 通用时序模型的 GLUE 式基准缺失;预训练与评估数据重叠导致零样本收益虚高。
- **HCH 能测什么:**
  - 把 R1B 的 2×2 transfer matrix 的"unseen"两格当作**硬纪律**:source 训练池与 DK1 评估必须零重叠(当前协议已满足,但要落成书面审计)。
  - 在论文方法学里加一段"泄漏审计"声明:训练域与未见域无样本重叠、无相关序列时间重叠。
- **HCH 不该抄什么:**
  - 不建 230B 点预训练池(那是 U0 阶段的外部基准,不是 R1B)。
  - 不把 HCH 当 full-shot 模型去跑 GIFT-Eval 排行榜;HCH 是 correction module,不是 end-to-end forecaster,直接对比会错位。
- **优先级:** `now`(评估纪律模板)+ `R1C`(最终 benchmark 参考)

### 1.2 Rethinking Evaluation in the Era of TSFM: (Un)known Information Leakage Challenges

- **paper:** arXiv 2510.13654
- **year:** 2025
- **来源:** https://huggingface.co/papers/2510.13654 · https://ar5iv.labs.arxiv.org/html/2510.13654
- **确切实验想法:**
  - 系统盘点现有 TSFM 评测,识别**两类泄漏**:(a) 数据集被多用途复用导致的 train-test 样本重叠;(b) **相关序列之间的时间重叠**(即使样本不同,两条相关序列共享同一时间窗也算泄漏)。
  - 指出忽略这些会产生过度乐观的零样本估计。
- **解决的问题:** 假阳性 generalization——模型"赢"是因为见过测试分布。
- **HCH 能测什么:**
  - 给 host cache 做泄漏审计:DE/PJM/NEM 与 DK1 之间是否存在共享外生驱动(如同一区域燃料价、天气再分析),导致"市场不同但信息重叠"。
  - 检查 host 自身训练是否严格只用各域 H0(协议已要求,落成 checklist)。
- **HCH 不该抄什么:** 不需要为 R1B 重建整套 benchmark 去污染管线;只取"泄漏二分法"作为审计清单。
- **优先级:** `now`(审计 checklist)

### 1.3 A More Realistic Evaluation of Cross-Frequency Transfer Learning and Foundation Forecasting Models

- **paper:** arXiv 2509.19465(NeurIPS 2025 workshop)
- **year:** 2025
- **来源:** https://arxiv.org/abs/2509.19465
- **确切实验想法:**
  - 只用**私有 + 合成数据**预训练,保证与 15 个大型比赛测试集**零泄漏**;再在 sCRPS / MASE 上对比基础模型 vs 统计模型。
  - 结论:统计模型与集成在无泄漏评估下反超基础模型(sCRPS 优 >8.2%,MASE 优 >20%)。
- **解决的问题:** 预训练/测试重叠 + 统计基线被弱化导致的"基础模型收益虚高"。
- **HCH 能测什么:**
  - R1B 报告"candidate 优于 host baseline"时必须同时报告强统计参照(seasonal-naive 作用于 host 变换误差、Identity),防止收益来自弱基线。
  - 对 HCH 的 host baseline 选择施加"基线强度"要求:host 不能是故意做弱的。
- **HCH 不该抄什么:** 不引入合成数据/私有语料;只取"强基线 + 无泄漏"评估纪律。
- **优先级:** `R1C`(final benchmark 阶段),但"强基线"要求可现在就写进 R1B 报告模板。

### 1.4 TempusBench: An Evaluation Framework for Time-Series Forecasting

- **paper:** NeurIPS 2025(旧基准 M3 等与预训练语料重叠,污染普遍)
- **year:** 2025
- **来源:** https://neurips.cc/virtual/2025/loc/san-diego/130468
- **确切实验想法:** 构建与 TSFM 预训练语料**无重叠的新数据集**;标准化超参调优协议(避免不公平对比);可视化 + 活排行榜。
- **解决的问题:** 经典基准老化 + 污染 + 基线调参不公平。
- **HCH 能测什么:** 提醒我们在 paper benchmark 阶段选用"不在任何公开 TSFM 预训练池内"的电价数据做外部验证时,要核查重叠。
- **HCH 不该抄什么:** 无需现在采用;与 HCH 目前不发布基础模型无关。
- **优先级:** `later`

---

## 2. Cross-domain 训练 / 域平衡采样(与 R1B 训练设计直接相关)

### 2.1 UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting

- **paper:** Liu, Hu, Li, Diao, Liang, Hooi, Zimmermann(WWW 2024)
- **year:** 2024
- **来源:** https://arxiv.org/abs/2310.09751 · https://dl.acm.org/doi/10.1145/3589334.3645434
- **确切实验想法:**
  - 跨域统一训练,**batch 内单域**(保持 channel 数/序列长一致),batch 从全局池随机抽取,**少数样本域过采样**——域平衡采样。
  - 显式建模**域收敛速度不平衡**(简单域先收敛、复杂域欠拟合)。
  - 用自然语言 domain instruction("Exchange rate data with daily sampling rate")喂给 GPT-2 backbone,缓解 domain confusion;mask ratio≈0.5 的 masked 训练。
  - 评估含**对未见域的零样本迁移**。
- **解决的问题:** 跨域训练三难:variates 数差异、域混淆、收敛速度不平衡。
- **HCH 能测什么:**
  - 佐证 R1B 已有 equal-domain sampling 的方向正确。
  - 新增一个**收敛不平衡诊断**:逐域 loss 曲线/逐域收敛 epoch——若某些 source 域先过拟合,说明 equal sampling 还不够。
  - 其"未见域零样本评估"协议与 DK1 冻结设计同构。
- **HCH 不该抄什么:**
  - 不引入语言模型 / 自然语言 domain instruction(R1B 禁加 market_id/host_id 输入,语言指令更重)。
  - 不引入 masked 重建自监督(HCH 的 IAH-CRPS 是判别式校正目标,R1B §10 禁改训练目标)。
- **优先级:** `now`(收敛不平衡诊断)+ 架构部分 `later`

### 2.2 Moirai / Unified Training of Universal Time Series Forecasting Transformers

- **paper:** Woo, Liu, Kumar, Xiong, Savarese, Sahoo(ICML 2024)
- **year:** 2024
- **来源:** https://arxiv.org/abs/2402.02592
- **确切实验想法:**
  - LOTSA 语料:>27B 观测、9 域、全频段统一训练。
  - Masked encoder-only Transformer;binary attention bias 区分 within-series / cross-series;多分布 mixture 头(Student-t / log-normal / NB)输出概率。
  - 评估:Monash 域内 + ETT 等**未见集零样本**,与 full-shot 基线对比。
- **解决的问题:** 任意频率、任意 variate 数、异质分布的统一训练——universality 的"多轴"难点。
- **HCH 能测什么:**
  - 作为参照点:universality 是**多轴**问题(frequency / variate / distribution / market / host),R1B 只证明 market+host 两轴,不能自称完整 universality。
  - 该论文的"评估数据不出现在预训练语料"纪律与 R1B 一致。
- **HCH 不该抄什么:** 跨频段统一训练(HCH 目前 hourly 单频;跨频是独立泛化轴,留给 R1C/U0);不引入 MoE。
- **优先级:** `R1C`(heterogeneity 参照)+ `U0`(LOTSA 式语料若启用)

### 2.3 Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series

- **paper:** Ekambaram, Jati, et al.(IBM Research,NeurIPS 2024)
- **year:** 2024
- **来源:** https://arxiv.org/abs/2401.03955 · 官方 https://github.com/ibm-granite/granite-tsfm
- **确切实验想法:**
  - <1M 参数小模型,在 Monash 等多域预训练;channel-independent 单变量预训练 + 微调时引入 channel mixing。
  - **Diverse Resolution Sampling(DRS)**:对同一序列下采样出多分辨率样本,提升跨分辨率覆盖;**Resolution Prefix Tuning(RPT)**:把分辨率信息嵌入首个 patch。
  - 评估:11 个**未见数据集**上 zero-shot;5% 数据 few-shot 微调即可竞争。
- **解决的问题:** 小参数量下零/少样本跨域迁移。
- **HCH 能测什么:**
  - 其"few-shot 微调只需 5% 数据"提示:若未来允许 target 端极少量证据适配(不是 R1B,是 later),存在轻量适配先例。
  - 域平衡/采样策略的价值在小模型上尤其明显——HCH 候选也是小规模,可类比。
- **HCH 不该抄什么:**
  - 不引入分辨率下采样/前缀(DRS/RPT 针对多频;HCH 单频)。
  - 不引入 channel-mixing 微调(与 R1B 冻结 corrector 的核心测试冲突)。
- **优先级:** `later` / `U0`(小模型 U0 蒸馏设计参考)

---

## 3. Foundation 表示 / 概率预测基础模型(HCH 的外部参照 + U0 相关)

### 3.1 Chronos: Learning the Language of Time Series

- **paper:** Ansari et al.(Amazon)
- **year:** 2024
- **来源:** https://arxiv.org/abs/2403.07815 · 官方 https://github.com/amazon-science/chronos-forecasting
- **确切实验想法:**
  - 把时序量化成 token,T5/GPT-2 架构做"回归即分类";TSMixup + KernelSynth 合成增强。
  - 评估 42 个数据集,严格三组:**Pretraining-only(13)** / **Benchmark I 见过(15)** / **Benchmark II 未见(27)**——seen/unseen 分离是显式设计。
  - 零样本在 Benchmark II 上显著超 AutoETS/AutoARIMA。
- **解决的问题:** 概率预测基础模型的通用化。
- **HCH 能测什么:**
  - 其 seen/unseen 双 benchmark 结构与 R1B 的 2×2 matrix 同构,可作为论文叙述的对照先例。
  - 零样本收益在"未见域"上的呈现方式,HCH 的 DK1 表格可对照其报告格式。
- **HCH 不该抄什么:**
  - 不抄 tokenization / 量化 / T5 backbone / 合成数据。
  - 注意其局限(64 步 horizon 外退化)提醒:基础模型架构未必适配 24h 电价长 horizon,HCH 不需要走这条路。
- **优先级:** `later`(外部对比基线,非方法)

### 3.2 MOMENT: A Family of Open Time-series Foundation Models

- **paper:** Goswami, Saha, et al.(CMU Auton Lab,ICML 2024)
- **year:** 2024
- **来源:** 官方仓库 https://github.com/moment-timeseries-foundation-model/moment · HF: AutonLab/MOMENT-1-*
- **确切实验想法:**
  - Time-series Pile 多域预训练;train/val/test 无重叠,只用 train 预训练。
  - Patch + masked 重建;encoder-only T5。
  - 评估:**limited-supervision only**(零样本 + 线性探针)跨 5 任务(短/长 horizon 预测、AD、分类、插补)。
- **解决的问题:** 多任务通用时序表示;少样本/零样本通用评估。
- **HCH 能测什么:**
  - HCH 路线已规划 MOMENT 作为 U0 representation teacher,本条目确认其评估纪律(无泄漏 split + 线性探针)可复用。
  - 若 U0 启动,线性探针可评估"冻结表示里是否已有 IAH 所需信息"。
- **HCH 不该抄什么:** 不抄多任务头;不用 MOMENT 的 forecast 头直接当 HCH(校正模块 ≠ 生成器)。
- **优先级:** `U0`

### 3.3 Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

- **paper:** Rasul et al.(Mila,NeurIPS 2023 R0-FoMo WS)
- **year:** 2023
- **来源:** https://arxiv.org/abs/2310.08278
- **确切实验想法:**
  - 27 数据集 / 6 域预训练;decoder-only LLaMA 风格;用 lag 特征 + 日期时间特征做自回归概率生成。
  - 零样本在未见数据集上与监督基线可比;小比例微调达 SOTA。
- **解决的问题:** 单变量概率预测基础模型零样本迁移。
- **HCH 能测什么:** 作为概率预测零样本评估的外部参照(CRPS 类指标的报告方式)。
- **HCH 不该抄什么:** 不自回归逐点生成(lags 特征工程与 HCH 的 correction-module 定位不同)。
- **优先级:** `later`

---

## 4. 电价跨市场迁移(领域内直接对照)

### 4.1 Transfer Learning for Electricity Price Forecasting

- **paper:** Gunduz, Ugurlu, Oksuz(Sustainable Energy, Grids and Networks)
- **year:** 2023(arXiv 2020)
- **来源:** https://arxiv.org/abs/2007.03762 · https://www.sciencedirect.com/science/article/abs/pii/S2352467723000048
- **确切实验想法:**
  - 2 层 DNN 先在 source 市场(BE/DE/FR/NP)预训练,再在 target 市场**微调(freeze 除输出层外全部,小 lr,patience=1)**;asinh+Median-MAD 变换;168 个滞后价格 + 温度 + 星期哑变量;预测 24h。
  - DM 检验显著;**相似市场(France↔Belgium)迁移收益更高;target 数据越少收益越大**。
  - 明确对比了 "pre-trained-only(不微调)" 这一档。
- **解决的问题:** target 市场数据稀缺时的跨市场电价预测。
- **HCH 能测什么(三条,全部可落实验):**
  1. **市场相似性解释**:迁移收益 ≈ f(source-target 市场相似度)。HCH 可预先声明一个相似度代理(负电价率、残差波动率、host 误差 regime),把 DK1 结果从"好/坏二元"升级为"符合/不符合相似度假设",这更难被伪造也更有信息量。
  2. **target 数据稀缺 regime**:其"数据越少迁移收益越大"提示 frozen 正确器(不消费 target 标签)正好在最稀缺场景最该有用。
  3. **"pre-trained-only(no fine-tune)"是 HCH frozen 的直接对照档**:该文也承认此档弱于微调档——这正说明 HCH 若 frozen 能改善 DK1,是比"微调收益"更强的 claim。
- **HCH 不该抄什么:**
  - 不抄 target 微调(R1B 核心测试就是 frozen,不微调)。
  - 不引入温度等外生变量(R1B §1 禁加新特征)。
- **优先级:** `now`(领域内直接参照 + 相似度解释模板)

### 4.2 Market-Information-Aware Gated-LoRA of Foundation Models for Transferable Day-Ahead Electricity Price Forecasting

- **paper:** arXiv 2608.11359(**极新,需人工复核**)
- **year:** 2026
- **来源:** https://scirate.com/arxiv/2608.11359
- **确切实验想法:**
  - **明确采用 leave-one-market-out 协议**:4 个中国省级日前市场,逐次留一个市场做零样本评估。
  - 把 Chronos-2 基础模型 + **source 域 gated LoRA**(只更新 ~1% 参数,不需要 target 标签);gate 由 reserve-tightness 等市场状态驱动。
- **解决的问题:** 数据稀缺新市场的零样本迁移。
- **HCH 能测什么:**
  - **领域内 leave-one-market-out 先例**:证明 R1B 的 DK1 设计方向正确,且同行更严谨做法是**多折 LOMO 而非单市场**。
  - 支持把 R1B-B 的 M1(leave-one-source-market-out 诊断)从"可选"升级为"必做"。
- **HCH 不该抄什么:** 不引入基础模型 + LoRA adapter(R1B 禁更新 candidate;HCH 不自带基础模型)。
- **优先级:** `R1B-B`(多折 LOMO 佐证)

---

## 5. Worst-domain / DRO(仅报告,不实现 —— R1B §10 / E3 相关)

### 5.1 GroupDRO: Distributionally Robust Neural Networks for Group Shifts

- **paper:** Sagawa, Koh, Hashimoto, Liang(ICLR 2020)
- **year:** 2020
- **来源:** https://arxiv.org/abs/1911.08731
- **确切实验想法:** min-max:外层最小化模型参数,内层最大化 group 权重(对当前 loss 最高的域升权),迭代求解;直接优化 **worst-domain**。
- **解决的问题:** 分组 shift 下平均指标掩盖最差组崩坏。
- **HCH 能测什么:** 仅当 R1B 出现 worst-domain collapse 时,作为 E3 的候选 trigger。触发证据 = 某 unseen/source 域系统性 DEGRADED。
- **HCH 不该抄什么:** R1B §10 明确禁止在拿到泛化证据前加入 adversarial/GroupDRO loss;避免用 worst-domain 优化掩盖表示不足。
- **优先级:** `later`(触发式,不是默认)

### 5.2 GAS-DRO(碳强度多区域预测的 DRO 变体)

- **paper:** arXiv 2507.09905(2025)
- **year:** 2025
- **来源:** https://browse-export.arxiv.org/pdf/2507.09905
- **确切实验想法:** 多数据集合并训练(加州 BANC_23+24),在多个**独立跨年/跨区域测试集**(Texas/Queensland/UK)上做 worst-domain + Wasserstein 距离量化分布漂移;对比 W-DRO / KL-DRO / DRAGEN 等。
- **解决的问题:** 区域分布漂移下的鲁棒碳强度预测。
- **HCH 能测什么:** 提供"跨区域测试 + 分布漂移量化"报告模板(worst-domain 指标 + 漂移量)。
- **HCH 不该抄什么:** 不引入 diffusion/PPO 内层优化;与 R1B 冻结核心无关。
- **优先级:** `later`

---

## 6. 综合评估设计对比(回答"更难被伪造"的关键表)

| 评估设计 | 谁能造假 | 防伪强度 | R1B 对应 |
|---|---|---|---|
| seen 域 macro 平均 | 调参/挑域拉高平均 | 弱 | 不作为主表 |
| 单未见市场(DK1 单点) | 恰逢相似市场/巧合 | 中 | R1B-A 起步,但不够 |
| **多折 leave-one-market-out** | 难(需所有 fold 都成立) | 强 | **建议 R1B-B 必做 M1 全折** |
| **LOHO(leave-one-host-out)+ 零梯度** | 很难(candidate 从未见过该 host 的梯度/选择信号) | 强 | R1B-A 已设计,保持 |
| 训练/评估池零重叠 + 泄漏审计 | 难(需逐项审计) | 强 | **建议落成书面 checklist** |
| 强基线对照(统计 + Identity + host 自身) | 难(基线越强越难虚报收益) | 强 | **建议纳入报告模板** |
| 预注册 fold / 阈值 / 决策规则 | 极难(post-hoc 解释空间最小) | 极强 | R1B 已有 STOP/CONTINUE,保持冻结 |

结论:**同行文献收敛于同一判据 —— 多折留出 + 零泄漏 + 强基线 + 预注册**,而不是"多喂数据 / 更强的网络"。

---

## 7. 对 R1B 当前 LOHO-PatchTST / DK1 零梯度设计的直接启示

1. **当前设计方向正确,不要削弱。**
   LOHO-PatchTST 的"ZERO S2T gradient + ZERO S2V checkpoint-selection signal"以及 DK1 的 frozen corrector,恰好是文献中最难伪造的两类评估:候选从未接触目标域的梯度、也从未用目标域做过模型选择。GIFT-Eval / Chronos / 泄漏论文都证明"模型见过验证信号"本身就是泄漏通道——R1B 堵住了这条,应保留并在论文里显式强调。

2. **单市场 DK1 不够;把 M1(leave-one-source-market-out)从可选升级为 R1B-B 必做。**
 Gated-LoRA 电价论文已在 4 个市场用多折 LOMO;Gunduz 证明迁移收益依赖市场相似度。单 DK1 折可能是"恰逢相似市场"的巧合,审稿人可一击即破。多折 LOMO(train PJM+NEM→test DE 等)让 claim 需要"所有留出市场都不崩"才成立,是当前成本下**性价比最高的防伪升级**。

3. **补一个泄漏审计 checklist(几小时工作量,防伪收益大)。**
 按 §1.2 的泄漏二分法:审计 DE/PJM/NEM 与 DK1 无样本重叠、无共享外生驱动的时间重叠;审计 host cache 只用各域 H0。把它写进 feature-schema audit 或单独小节,paper 方法学即可宣称"训练/评估零重叠"。

4. **报告模板强制加入"强基线"与"worst 单元格"。**
 按 §1.3,收益要在强基线(host 自身、Identity、作用于 host 变换误差的 seasonal-naive)之上报告;并按 §12 transfer matrix 每格给出 worst-domain 与 failure count,而不是只给 macro。防止"平均改善"掩盖某一格崩坏。

5. **市场相似度解释比二元好/坏更有信息量。**
 按 §4.1,预先声明相似度代理(负电价率、残差波动率、host 误差 regime),把 DK1 结果解释为"符合/不符合相似度假设"。若 DK1 是最不相似市场而失败,这是诚实且难伪造的结论("transfer 收益随相似度衰减"),而不是笼统的"transfer 失败"。

6. **不要抄任何技术。**
 UniTime 语言指令、TTM 分辨率前缀、Chronos tokenization、Moirai 跨频训练、Gated-LoRA 都在各自设置报告了收益,但全部违背 R1B 冻结核心纪律或指向不同泛化轴。**本侦察的净建议是:评估设计升级,零架构变更。**

---

## 8. 来源汇总

- GIFT-Eval:https://arxiv.org/abs/2410.10393 · https://github.com/SalesforceAIResearch/gift-eval
- 泄漏综述:https://arxiv.org/abs/2510.13654 · https://huggingface.co/papers/2510.13654
- 更现实的跨频评估:https://arxiv.org/abs/2509.19465
- TempusBench:https://neurips.cc/virtual/2025/loc/san-diego/130468
- UniTime:https://arxiv.org/abs/2310.09751 · https://dl.acm.org/doi/10.1145/3589334.3645434
- Moirai / Unified Training:https://arxiv.org/abs/2402.02592
- TTM:https://arxiv.org/abs/2401.03955 · https://github.com/ibm-granite/granite-tsfm
- Chronos:https://arxiv.org/abs/2403.07815 · https://github.com/amazon-science/chronos-forecasting
- MOMENT:https://github.com/moment-timeseries-foundation-model/moment
- Lag-Llama:https://arxiv.org/abs/2310.08278
- Gunduz et al.(EPF 迁移):https://arxiv.org/abs/2007.03762 · https://www.sciencedirect.com/science/article/abs/pii/S2352467723000048
- Gated-LoRA 电价(极新):https://scirate.com/arxiv/2608.11359
- GroupDRO:https://arxiv.org/abs/1911.08731
- GAS-DRO:https://browse-export.arxiv.org/pdf/2507.09905
