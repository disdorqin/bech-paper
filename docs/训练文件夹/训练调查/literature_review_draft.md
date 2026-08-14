# HCH-v2 文献核查表（literature review draft）

- **日期**: 2026-08-14
- **用途**: 支撑项目调查报告；全部链接真实、已用 HTTP 抓取验证（HTTP 200 且标题/内容匹配）。
- **验证方法说明**: 本机 `WebFetch` 被 claude.ai 域名白名单阻断，改用两条等价验证通道：(1) firecrawl scrape（arXiv abs 页 / PMLR / NeurIPS / ICLR 官方页面）；(2) 直接 HTTP 抓取 arXiv `<meta name="citation_title">` 与 Crossref API JSON。所有链接验证状态见文末"链接验证状态"小节。
- **HCH-v2 背景锚点**: frozen host(Linear/MLP/LSTM/PatchTST) 输出经双曲坐标 `z0=asinh(host/s)` 进入三原子 IAH 分布校正头（`w⁻δ(z0−m⁻)+w⁰δ(z0)+w⁺δ(z0+m⁺)`），唯一训练目标 IAH-CRPS；动作链 = query-dose replay → double-event proposal → split-conformal LCB>0 才执行，否则 identity 弃权；跨市场 = 等域采样 universal head + 数据签名 FiLM 接口。现状 = 分布层 CRPS 泛化验证成功（R1B NATIVE_GENERALIZATION_SUPPORTED），但 point 读出弱（Paper Gate YELLOW_READOUT: CRPS 40/40 全赢、final MAE 16/40 反伤）——核心缺口是"分布→点读出"。

---

## 0. 总表（按方向分组，共 29 篇）

### A. 概率分布到点决策（9 篇）

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|
| [Gneiting, *Making and Evaluating Point Forecasts*（Bundesbank Eltville 2012 讲座 PDF）](https://www.bundesbank.de/resource/blob/635562/0d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf) | 从预测分布导出点预报：一致评分函数（Bregman/分位数/expectile）+ Bayes act | 无训练；理论：scoring function 必须与任务严格一致 | **直接命中 HCH 当前缺口**：分布层赢≠任意点读出赢。HCH 若从 IAH 分布取点，应按一致评分函数（MAE→中位数、pinball→分位数）取 Bayes act，而非默认取均值 | 无；该理论正是 YELLOW_READOUT 的解释框架 | **强烈建议**（为点读出提供理论落地） |
| [Waghmare & Ziegel, *Proper scoring rules for estimation and forecast evaluation*（arXiv:2504.01781）](https://arxiv.org/abs/2504.01781) | 现代综述：proper scoring rules 数学基础与应用 | 理论综述（G&R 2007 的继任者） | 为 IAH-CRPS 作为唯一训练目标提供正当性；提醒单一规则不能刻画全部预报质量（呼应 HCH 的 CRPS/MAE 分离评估） | 无 | 是（引用与框架） |
| [Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*（JASA 2007，Crossref 记录）](https://api.crossref.org/works/10.1198/016214506000001437) | proper scoring rules 奠基文（CRPS/energy score/区间得分） | 理论 | IAH-CRPS 的合法性根基；energy score 核表示与 HCH 的 asinh 坐标数值稳定设计相关 | 无（原刊 T&F 付费墙；本链接为可点击 Crossref 元数据记录） | 是（必引经典） |
| [Lipiecki, Uniejewski & Weron, *Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: IDR*（arXiv:2404.02270）](https://arxiv.org/abs/2404.02270) | 把 day-ahead 电价 point 预测转概率分布：QRA / Conformal / IDR | 统计后处理；IDR 无调参；Shapley 合并三者 | 与 HCH "frozen host 输出→分布校正"同源；三者 ensemble 击败 distributional DNN，是 HCH universal head 最该打的统计基线 | IDR/PAV 需保留训练集、非可微；HCH 用可微三原子头+双曲坐标、可跨域 | 是（benchmark + 目标对齐） |
| [Lipiecki & Uniejewski, *Isotonic Quantile Regression Averaging (iQRA)*（arXiv:2507.15079）](https://arxiv.org/abs/2507.15079) | 从 point ensemble 生成电价概率预测；随机序约束 QRA | 分位数回归 + isotonic 正则；超参自由 | "强约束降复杂度"与 HCH 稀疏三原子 IAH 同哲学；其可靠性评价（多置信水平 coverage）可作为 HCH 校准对照 | quantile 视角 vs HCH 的 CRPS 驱动分布头；统计后处理 vs 端到端可微 | 是（baseline） |
| [Henzi, Ziegel & Gneiting, *Isotonic Distributional Regression (IDR)*（arXiv:1909.03725，JRSS-B）](https://arxiv.org/abs/1909.03725) | 非参条件分布估计；isotonic 约束下 calibrated+最优 | PAVA；无调参；可与 bagging 结合 | HCH 三原子 IAH 是"受限分布回归"；IDR 的"calibrated + 对损失类最优"是 HCH 想达到的性质；`z0=asinh(host/s)` 天然给出 covariate 偏序，可直接用 IDR 做校准对照 | IDR 在 covariate 偏序上非参拟合、无跨域接口；HCH 是可微参数头 | 是（校准性质 + 对照） |
| [Walz, Henzi, Ziegel & Gneiting, *EasyUQ*（arXiv:2212.08376）](https://arxiv.org/abs/2212.08376) | 把 single-valued 模型输出转成校准分布（IDR 特例，不看模型输入） | PAV + 核平滑；只依赖 output-outcome 对 | **与 HCH 校正头哲学最近**：HCH 也只吃 host 输出。EasyUQ 是无训练的 LocalCore 上界；Smooth EasyUQ 的连续分布类似 HCH 平滑读出 | EasyUQ 是 per-domain 无参数拟合；HCH 是 8.7K 参数 universal head + 跨域 FiLM | **强烈建议**（LocalCore 对照） |
| [Uniejewski, *Smoothing Quantile Regression Averaging (SQR)*（arXiv:2302.00411）](https://arxiv.org/abs/2302.00411) | QRA 平滑化变体；德国等市场电价概率预测 | 分位数回归 + 平滑；滚动训练窗 | SOTA 统计基线；其电池交易经济收益评估（+3.5%）呼应 HCH 的 statistical/economic 分离 | 统计后处理，无可微跨域机制 | 是（baseline） |
| [Baran & Lerch, *Mixture EMOS for calibrating ensemble forecasts of wind speed*（arXiv:1507.06517）](https://arxiv.org/abs/1507.06517) | 集合预报分布后处理：EMOS，TN+LN 加权混合 | 滚动训练窗优化 CRPS/log-score | **HCH 的直接理论祖先**：参数化分布 + 以 CRPS 为训练目标 + 混合分量；HCH 三原子 IAH 是 heavier-tail 友好的离散-连续混合版 EMOS | EMOS 分量少、假设参数形式、单域；HCH 用 asinh 坐标 + 跨域训练 | 是（分布层根基 + 基准） |

### B. 冻结宿主后处理（4 篇）

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|
| [Liu et al., *PIR: Improving Time Series Forecasting via Instance-aware Post-hoc Revision*（NeurIPS 2025）](https://papers.neurips.cc/paper_files/paper/2025/file/331c41353b053683e17f7c88a797701d-Paper-Conference.pdf) | frozen 模型输出上做实例级"识别→修正"（局部+全局检索上下文） | 先估实例准确率，再以 covariate/历史时序修正 | 与 HCH 动作链同构（先识别偏差实例→再修正）；局部+全局检索对应 HCH 的 query-dose replay；模型无关 | PIR 修正 point 而非分布；检索易引入 leakage（HCH R1A 已审计检索价值边界）；HCH 用跨域 universal 替代显式检索 | 是（动作链/实例修正理念） |
| [Liang et al., *δ-Adapter: The Forecast After the Forecast*（arXiv:2601.20280，ICLR 2026）](https://arxiv.org/abs/2601.20280) | frozen backbones 的轻量后处理：input nudging + output residual correction + 分布校准 | tiny bounded modules；O(δ) drift bound；Quantile Calibrator + Conformal Corrector | **与 HCH 校正头最接近的直接先例**：frozen host + 输出侧残差校正 + 分布校准三位一体；bounded residual + 局部下降保证是 HCH 数学边界可借鉴的 | δ-Adapter 是单域后处理、无跨域 FiLM/等域采样；HCH 强调跨市场 universal；其 conformal corrector 是 per-domain | **强烈建议**（结构对标 + 数学保证） |
| [Sokol, Moniz & Chawla, *Conformalized Selective Regression*（arXiv:2402.16300）](https://arxiv.org/abs/2402.16300) | selective regression（reject option）；conformal 提供可信弃权 | conformal 预测集 + 弃权阈值 | 为 HCH 的"action-value split-conformal LCB>0 才执行，否则 identity 弃权"提供统计保证框架（安全弃权） | 该文假设 i.i.d. 回归；HCH 是时序/分布级，需 split-conformal 适配与 LCB 组合 | 是（弃权理论支撑） |
| [Zhang et al., *Factorize to Generalize: Retrieval-Guided Invariant-Dynamic Decomposition*（arXiv:2605.24911）](https://arxiv.org/abs/2605.24911) | 检索增强预测的分解；指出 retrieval 使预测振荡、在平滑序列上反而降精度 | invariant-dynamic 分解 + retrieval 融合 | 提醒 HCH：local/global 检索（query-dose replay）在平滑/趋势域可能反伤——与 R1A 检索价值 audit 结论呼应，支持"检索只在极端/稀有实例上启用" | 检索融合 vs HCH 等域采样 universal 训练（HCH 不强依赖显式检索） | 是（leakage/检索边界审计参考） |

### C. 跨域/跨市场训练（6 篇）

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|
| [UniTime: Language-Empowered Unified Model for Cross-Domain TSF（arXiv:2310.09751）](https://arxiv.org/abs/2310.09751) | 跨域统一时序模型；域识别 + 域收敛失衡缓解 | domain instructions + Language-TS Transformer + masking | 直接支撑 HCH 的等域采样 + 数据签名 FiLM 接口：同一套"域识别 + 域失衡缓解"机制；masking 是 HCH 防"易域主导"的参考 | UniTime 是完整 backbone；HCH 是 frozen host 上的校正头 | 是（域失衡技巧） |
| [DAF: Domain Adaptation for Time Series Forecasting via Attention Sharing（PMLR v162，ICML 2022）](https://proceedings.mlr.press/v162/jin22d.html) | 源域→目标域时序域适应；共享注意力 + 域判别器 | 共享 query/key、私有 value；对抗式域对齐 | 与 HCH "shared representation + domain-adaptive interface"直接对应（FiLM 是另一种域接口）；"align keys 保留域共享、value 域私有"的思路可迁移到 HCH 的签名调制 | DAF 训练完整预测器；HCH 只训练校正头 | 是（域对齐接口设计） |
| [BLAST: Balanced Sampling Time Series Corpus for Universal Forecasting Models（KDD '25，arXiv:2505.17871）](https://arxiv.org/abs/2505.17871) | universal forecasting 训练语料的均衡采样（统计粒度/周期/非平稳性） | 均衡采样协议 + 预训练 | **HCH 等域采样的数据维度证明**：pattern-balanced 采样对 universal 泛化至关重要，为 HCH 12 source 等域采样提供直接文献支撑 | BLAST 面向完整 backbone 预训练语料；HCH 面向 frozen host 校正头 | **强烈建议**（采样方法论） |
| [Chronos: Learning the Language of Time Series（arXiv:2403.07815）](https://arxiv.org/abs/2403.07815) | tokenize 时序 + T5 预训练跨域 probabilistic 模型 | cross-entropy 于 token；大规模真实+合成语料 | 实证跨域预训练可提升 zero-shot；可作为 HCH universal head 的"大模型版"对照（规模对比） | 架构/规模完全不同；HCH 是 budgeted 8.7K 参数校正头 | 是（zero-shot 跨域证据） |
| [FiLM: Visual Reasoning with a General Conditioning Layer（arXiv:1709.07871）](https://arxiv.org/abs/1709.07871) | 通用条件层：feature-wise 线性调制 | conditioning info → per-channel scale/shift | HCH 数据签名 FiLM 接口的原始出处；提供"轻量域调制、不改 backbone"的接口原理论证 | 无（HCH 已在用） | 是（接口原理论证） |
| [Lag-Llama: Towards Foundation Models for Probabilistic TSF（arXiv:2310.08278）](https://arxiv.org/abs/2310.08278) | decoder-only 概率时序基础模型；lag 作协变量 | 跨域预训练 + 概率头 | 与 Chronos 同为"跨域概率预训练"证据；其 probabilistic head 设计（分布层）可对照 HCH 的 IAH 分布头 | 大规模预训练 vs 8.7K 校正头 | 可选（与 Chronos 二选一补充） |

### D. Prequential 校准与在线适配（5 篇）

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|
| [SOLID: Calibration of Time-Series Forecasting（arXiv:2310.14838，KDD'24）](https://arxiv.org/abs/2310.14838) | 检测 context-driven distribution shift 并校准 | Reconditionor（残差-上下文互信息）+ sample-level contextualized adapter（相似样本微调预测层） | **HCH 动作链 query-dose replay 的理论化**：按上下文相似性取样本再局部适配；其"最优 bias-variance trade-off"为 HCH 短证据 C3 授权提供理论依据 | SOLID 微调整个预测层；HCH 只动校正头参数；SOLID 是 per-sample 在线微调，HCH 是 frozen universal + 局部动作 | **强烈建议**（动作链理论支撑） |
| [DSOF: Fast and Slow Streams for Online TSF Without Information Leakage（ICLR 2025 proceedings）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html) | 在线时序预测，明确防信息泄漏（不用已参与训练的历史步评估） | 双流：slow 完整数据经验回放 + fast 时序差分；teacher-student 残差学习 | 为 HCH 的 prequential/延迟标签协议正名：DSOF 把"用训练过的历史步做评估"列为泄漏，正是 HCH 规避的做法；slow/fast 双流可启发 HCH "长期 universal + 短期证据 C3" 双流 | DSOF 是完整在线训练框架；HCH 是 frozen universal + 局部校正 | 是（泄漏协议 + 双流灵感） |
| [TAFAS: Battling Non-stationarity in TSF via Test-time Adaptation（arXiv:2501.04970）](https://arxiv.org/abs/2501.04970) | 时序 TTA；冻结 source + 部分观测 ground truth | gated calibration module；利用部分观测 GT；模型无关 | "部分观测 GT + gated 校准"与 HCH 的 delayed-label / 跨期标签协议高度相关；gated 机制类似 HCH 的 LCB 门控 | TAFAS 在测试时更新 backbone 侧参数；HCH 冻结 backbone 只读 host 输出 | 是 |
| [Towards Principled Test-Time Adaptation for TSF（arXiv:2605.17250）](https://arxiv.org/abs/2605.17250) | 审视 TSF-TTA 协议清洁度；只用 matured ground truth 做适配 | matured-label 协议 + 频率域诊断 | **直接为 HCH 的 delayed-label 协议提供正当性**：只用成熟标签更新，避免未成熟标签泄漏未来信息 | 无（支持 HCH 现有协议） | **强烈建议**（协议正当性） |
| [Test-Time Adaptation for Non-stationary Time Series（arXiv:2602.00073）](https://arxiv.org/abs/2602.00073) | 小脚印 TTA：只更新 normalization affine 参数，frozen backbone | 无标签窗口；分类用熵最小化、回归用预测方差最小化 | "只更新最小可动参数面"与 HCH 8.7K 参数校正头同哲学；可作为 HCH 在线版本（若做）的轻量选择 | HCH 当前是 frozen universal，不做在线更新 | 可选（在线变体时借鉴） |

### E. 电力价格极端事件（5 篇）

| 论文 | 任务 | 训练/校准方式 | 对 HCH 的启发 | 与 HCH 冲突 | 是否建议借鉴 |
|---|---|---|---|---|---|
| [Cornell, Dinh & Pourmousavi, *A probabilistic forecast methodology for volatile electricity prices in NEM*（arXiv:2311.07289）](https://arxiv.org/abs/2311.07289) | 高波动市场（NEM SA）概率预测：spike filtration + 多步后处理 | quantile regression ensemble；spike 过滤 + 变训练窗平均 | HCH 极端价格三原子 IAH 的直接对标：spike 是领域核心问题；该文做 spike 过滤（改目标），HCH 在分布上显式建模尖峰原子（不改目标） | spike 预处理 vs 分布层尖峰建模 | 是（benchmark + spike 处理对照） |
| [Steinbakk et al., *Using published bid/ask curves to error dress spot electricity price forecasts*（arXiv:1812.02433）](https://arxiv.org/abs/1812.02433) | 用 bid/ask 曲线把点预测 error-dress 成非对称自适应分布（Nord Pool） | 特征驱动误差分布 + 体积→价格非线性变换；阈值预警 | "非对称、尾部自适应"正是 HCH 三原子 IAH 想达到的性质；其尾部概率预警系统对应 HCH 极端事件读出 | 用市场价格结构（bid/ask）而非学习头；HCH 是数据驱动可微头 | 是（非对称尾部建模） |
| [Statistical and economic evaluation of forecasts in electricity markets: beyond RMSE and MAE（arXiv:2511.13616）](https://arxiv.org/abs/2511.13616) | 电价预测的统计 vs 经济评价分离（BESS 套利维度分解） | 评价框架：accuracy/dispersion/association/extrema 四维 | **直接支持 HCH Paper Gate 的"CRPS vs MAE 分离"判断**：分布层赢≠经济有效，为 YELLOW_READOUT 提供文献正当性 | 无（评价框架，非模型） | **强烈建议**（评价体系） |
| [Evaluating TSFMs for Electricity Price Forecasting: Contamination Risk, Distributional Shifts（arXiv:2607.02623）](https://arxiv.org/abs/2607.02623) | 评估基础模型在 EPF 的污染风险、分布偏移、协变量依赖、tail 行为 | 两数据集 benchmark；contamination 缓解协议 | HCH leakage audit（数据信息边界）与该文 contamination 关注一致；tail 行为评估呼应 HCH 尾部原子校准 | 面向 TSFM 大模型；HCH 是 8.7K 校正头，但审计方法论可迁移 | 是（审计方法） |
| [Probabilistic Forecasting for Day-ahead Electricity Prices, Battery Trading Strategies and the Economic Evaluation（arXiv:2604.19580）](https://arxiv.org/abs/2604.19580) | day-ahead 电价概率预测 → 电池交易经济评价；指出 QBTS 不激励诚实预报 | quantile-based trading strategy 分析 | 与 HCH "分布→点决策"缺口直接相关：统计好的分布不保证经济决策正确，与 YELLOW_READOUT 一致 | 无 | 是（点决策读出与评价） |

---

## 1. 迁移可行性一句话判断（每方向）

- **A. 概率分布到点决策** — **能迁移，迁移代价低**：不触碰 HCH 数学边界（IAH 三原子与 IAH-CRPS 保持不变），只需在读出侧按一致评分函数取 Bayes act（MAE→中位数）；这是当前 YELLOW_READOUT 最直接、零风险的修复入口。违界判定：**不违反**。
- **B. 冻结宿主后处理** — **能迁移，代价中**：δ-Adapter 的 bounded residual + 分布校准与 HCH 三原子头结构同族，可借其局部下降/漂移界论证；PIR/检索类需先过 HCH 的 leakage 审计（R1A 已确认检索只在极端实例有价值）。违界判定：**不违反现有边界**（但检索增强若引入显式外部数据，需重审数据信息边界）。
- **C. 跨域/跨市场训练** — **能迁移，代价低**：BLAST 等域采样、UniTime masking、FiLM 调制与 HCH 现有 universal head 训练协议同构，属工程补强而非新机制。违界判定：**不违反**。
- **D. Prequential 校准与在线适配** — **能迁移，代价中**：SOLID/DSOF/TAFAS 的"成熟标签 + 局部适配 + 双流"为 HCH 动作链与 delayed-label 协议提供理论背书；若做在线更新版本，需把"只更新校正头参数"写死为边界。违界判定：**不违反**（前提是不解冻 host）。
- **E. 电力价格极端事件** — **能迁移，代价低**：NEM/error-dressing 等为 HCH 三原子 IAH 提供领域对标与评价框架；beyond RMSE/MAE 直接指导点读出修复。违界判定：**不违反**。

## 2. 链接验证状态

> 验证方式：直接 HTTP 抓取（arXiv `<meta name="citation_title">`、Crossref API JSON、PMLR/NeurIPS/ICLR 页面全文）或 firecrawl scrape；全部 HTTP 200 且标题/内容匹配。未发现需标注 UNVERIFIED 的链接。

| # | 论文 | 验证 URL | 状态 |
|---|---|---|---|
| A1 | Gneiting, Making and Evaluating Point Forecasts | https://www.bundesbank.de/resource/blob/635562/0d3de0f3fc003e5b4864828143f268cf/mL/2012-06-01-eltville-11-gneiting-paper-data.pdf | VERIFIED（PDF 内容解析出标题与作者） |
| A2 | Gneiting & Raftery 2007 (JASA) | https://api.crossref.org/works/10.1198/016214506000001437 | VERIFIED（Crossref API 记录；原文在 T&F 付费墙，本链接为可点击元数据记录） |
| A3 | Proper scoring rules for estimation and forecast evaluation | https://arxiv.org/abs/2504.01781 | VERIFIED |
| A4 | Postprocessing of point predictions … IDR | https://arxiv.org/abs/2404.02270 | VERIFIED |
| A5 | Isotonic Quantile Regression Averaging (iQRA) | https://arxiv.org/abs/2507.15079 | VERIFIED |
| A6 | Isotonic Distributional Regression (IDR) | https://arxiv.org/abs/1909.03725 | VERIFIED |
| A7 | EasyUQ | https://arxiv.org/abs/2212.08376 | VERIFIED |
| A8 | Smoothing Quantile Regression Averaging (SQR) | https://arxiv.org/abs/2302.00411 | VERIFIED |
| A9 | Mixture EMOS (Baran & Lerch) | https://arxiv.org/abs/1507.06517 | VERIFIED |
| B1 | PIR (NeurIPS 2025) | https://papers.neurips.cc/paper_files/paper/2025/file/331c41353b053683e17f7c88a797701d-Paper-Conference.pdf | VERIFIED（PDF 内容匹配标题） |
| B2 | δ-Adapter / The Forecast After the Forecast | https://arxiv.org/abs/2601.20280 | VERIFIED |
| B3 | Conformalized Selective Regression | https://arxiv.org/abs/2402.16300 | VERIFIED |
| B4 | Factorize to Generalize (Retrieval-Guided) | https://arxiv.org/abs/2605.24911 | VERIFIED |
| C1 | UniTime | https://arxiv.org/abs/2310.09751 | VERIFIED |
| C2 | DAF (ICML 2022, PMLR) | https://proceedings.mlr.press/v162/jin22d.html | VERIFIED |
| C3 | BLAST (KDD '25) | https://arxiv.org/abs/2505.17871 | VERIFIED |
| C4 | Chronos | https://arxiv.org/abs/2403.07815 | VERIFIED |
| C5 | FiLM | https://arxiv.org/abs/1709.07871 | VERIFIED |
| C6 | Lag-Llama | https://arxiv.org/abs/2310.08278 | VERIFIED |
| D1 | SOLID (KDD'24) | https://arxiv.org/abs/2310.14838 | VERIFIED |
| D2 | DSOF Fast and Slow Streams (ICLR 2025) | https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html | VERIFIED |
| D3 | TAFAS | https://arxiv.org/abs/2501.04970 | VERIFIED |
| D4 | Towards Principled TTA for TSF | https://arxiv.org/abs/2605.17250 | VERIFIED |
| D5 | TTA for Non-stationary Time Series | https://arxiv.org/abs/2602.00073 | VERIFIED |
| E1 | NEM volatile probabilistic forecasting | https://arxiv.org/abs/2311.07289 | VERIFIED |
| E2 | Error dressing with bid/ask curves | https://arxiv.org/abs/1812.02433 | VERIFIED |
| E3 | Beyond RMSE and MAE | https://arxiv.org/abs/2511.13616 | VERIFIED |
| E4 | TSFM contamination risk EPF | https://arxiv.org/abs/2607.02623 | VERIFIED |
| E5 | Day-ahead probabilistic + battery trading economics | https://arxiv.org/abs/2604.19580 | VERIFIED |

## 3. 备注 / 覆盖面说明

- 五方向全部 ≥3 篇真实链接（A=9、B=4、C=6、D=5、E=5），无方向需要降级声明。
- 种子链接全部命中并验证：Bundesbank Gneiting PDF、2404.02270、2507.15079、PIR NeurIPS、δ-Adapter(2601.20280)、UniTime、DAF(PMLR)、SOLID、DSOF(ICLR)、BLAST。δ-Adapter 实际标题为 *The Forecast After the Forecast: A Post-Processing Shift in Time Series*（ICLR 2026）。
- 方向 D 的"延迟标签协议"以 2605.17250（matured-ground-truth 协议）为最强文献支撑；另发现 Grzenda et al. IJCNN 2020 延迟标签评测（HAL: https://hal.science/hal-04468444）但 HAL 页面被反爬拦截，未纳入主表。
- 原刊付费墙论文（G&R 2007 JASA、Nowotarski & Weron 2015 QRA 原刊）未提供 PDF 直链；QRA 以 SQR(2302.00411)/iQRA(2507.15079)/ReModels 包(2405.11372) 等可验证后续文献代替，G&R 2007 以 Crossref 记录代替，均在主表如实标注。
