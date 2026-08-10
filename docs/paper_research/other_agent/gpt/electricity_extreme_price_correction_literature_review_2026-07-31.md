# 模型无关极端电价校正模块：独立文献检索、空白与创新点交叉验证

**检索日期：2026-07-31**  
**研究范围：2021–2026，重点覆盖 CCF-A / 顶会方法论文，并补充与电价任务直接相关的高相关期刊论文**  
**任务边界：只交付调研结论、空白与候选创新点，不撰写论文正文**

---

## 0. 先给结论

### 0.1 核心判断

“模型无关、冻结基座、无需重训、在输出端学习残差或后处理校正”这一宽泛问题，到了 2026 年已经明显拥挤，**不能再独立充当主创新**：

- NeurIPS 2025 的 **PIR** 已提出模型无关的“失败识别 + 后处理修订”；
- ICLR 2026 的 **δ-Adapter** 已直接提出冻结基座、输入微调、输出残差修正、分位数校准和共形校正；
- 2025–2026 年还有 **Post-Training Corrections**、**UEC-STD** 等近邻工作。

因此，原候选题目若仍写成“模型无关校正头”，被审稿人以 *incremental / already covered by post-hoc revision* 拒绝的风险很高。

### 0.2 仍然存在的可辩护空间

本轮保留文献中，没有发现一个成熟方案同时满足以下全部条件：

1. 同时处理负电价与正向尖峰；
2. 把“是否发生极端事件”和“发生后的幅值修正”联合建模；
3. 对任意冻结基座即插即用；
4. 有明确的正常时段退化预算或安全拒绝机制；
5. 跨基座、跨市场验证；
6. 同时报告统计、事件和交易风险价值。

因此，可辩护的问题应改写为：

> 在冻结且异构的电价预测基座之上，设计一个面向双向稀有尾部的选择性校正器：只有在校准后的极端风险足够高时才触发，并在预先规定的正常期退化预算下，联合修正事件发生概率和价格幅值；该模块还需在跨市场漂移下保持尾部校准，并改善真实约束下的交易收益与下行风险。

这比“再做一个 correction head”更接近顶会问题：它包含新的任务定义、选择性风险控制、双尾不对称建模和严格的跨域评估。

---

## 1. 检索统计

```text
检索家族：A / B / C / D / E / F
实际检索来源：
- arXiv
- OpenReview
- NeurIPS Proceedings
- PMLR（ICML/AISTATS/UAI 等）
- ACM Digital Library
- IEEE Xplore
- ScienceDirect / DOI 页面
- RePEc / IDEAS（书目信息交叉核验）
- Semantic Scholar（候选发现与交叉核验）
- GitHub（只核验代码仓库与 MCP 项目说明）

本轮进入可复核候选台账：64 篇（按标题/DOI 去重）
最终保留（强相关 + 中等相关）：38 篇
其中：顶会/CCF-A 方法论文 23 篇；电力/预测领域直接相关期刊或领域会议 13 篇；高相关预印本 2 篇
```

说明：

- “相关性”针对本研究主题，不等于会议等级。
- “顶会”与“电力领域强相关”分开判断；不能把 Applied Energy、Energy Economics、International Journal of Forecasting、IEEE TSG 等期刊说成 CCF-A 会议。
- 2019 年的 Conformalized Quantile Regression 是重要基础，但超出 2021–2026 主范围，因此仅作为理论锚点，不计入 38 篇保留论文。
- 推荐的四个 GitHub 项目均为外部 MCP Server，而非当前会话中已注册的 Skill；本轮核验了仓库说明，但没有把它们虚构成“已调用的数据源”。

---

## 2. A 家族：预测基座（P1）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| A | Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark | Jesus Lago / Applied Energy / 2021 | [DOI/出版社](https://doi.org/10.1016/j.apenergy.2021.116983)；[arXiv](https://arxiv.org/abs/2008.08004) | 提出多市场、长测试期、强简单基线、统计检验和开放工具箱等 EPF 评估规范。 | 强 | 重点是常规日前点预测；没有把双向极端、事件级指标、选择性校正和经济风险统一起来。 |
| A | Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx | Kin G. Olivares / International Journal of Forecasting / 2023 | [DOI/出版社](https://doi.org/10.1016/j.ijforecast.2022.03.001)；[arXiv](https://arxiv.org/abs/2104.05522) | 将外生变量加入 N-BEATS，并在多个电价市场上系统验证。 | 强 | 是专用基座而非模型无关校正；优化整体误差，不专门约束极端时段与正常时段的 Pareto 权衡。 |
| A | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers | Yuqi Nie / ICLR / 2023 | [OpenReview](https://openreview.net/forum?id=Jbdc0vTOcol) | PatchTST 通过 patching 与 channel independence 提升长时序预测。 | 中 | 通用基准与长期预测导向；电价 24/96 点、稀疏极端、市场规则和外生冲击未被专门处理。 |
| A | iTransformer: Inverted Transformers Are Effective for Time Series Forecasting | Yong Liu / ICLR / 2024 | [OpenReview](https://openreview.net/forum?id=JePfAI8fah) | 将变量作为 token，以注意力学习变量间关系。 | 中 | 仍是端到端基座；变量依赖不等于因果或市场结构，对尾部幅值没有专门目标。 |
| A | Unified Training of Universal Time Series Forecasting Transformers | Gerald Woo / ICML / 2024 | [PMLR](https://proceedings.mlr.press/v235/woo24a.html) | 提出 Moirai 和 LOTSA，探索跨频率、跨变量数量的通用零样本预测。 | 中 | 零样本总体指标不保证电价极端表现；对本地市场外生变量、负价边界和交易规则需重新适配。 |
| A | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | Yuxuan Wang / NeurIPS / 2024 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | 明确建模内生目标与多种外生变量的交互。 | 强 | 它回答“外生信息如何进入基座”，不回答冻结黑盒输出如何安全地做双尾校正。 |

### A 家族判断

已较成熟：

- 通用 Transformer、MLP、foundation model 和多尺度基座；
- 外生变量接入；
- 多市场 EPF 评估规范；
- 强简单基线的重要性。

仍未做透：

- 哪类表征在信息截止约束下真正包含极端前兆；
- 基座预测失败与“输入根本没有可预测信息”如何区分；
- 极端信号的双尾、持续时间、提前量和跨市场迁移；
- 对冻结异构基座统一输出什么最小接口，才能让后置模块既可用又不泄漏。

---

## 3. B 家族：极端 / 不平衡预测（P2）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| B | Delving into Deep Imbalanced Regression | Yuzhe Yang / ICML / 2021 | [PMLR](https://proceedings.mlr.press/v139/yang21m.html) | 提出 label distribution smoothing 与 feature distribution smoothing，系统定义连续标签长尾回归。 | 强 | 不是时序方法，实验不含电价；随机重采样/平滑若忽略时间顺序，可能产生泄漏或破坏滚动分布。 |
| B | RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression | Yu Gong / ICML / 2022 | [PMLR](https://proceedings.mlr.press/v162/gong22a.html) | 用标签空间与特征空间的排序一致性改善稀有连续值学习。 | 中 | 没有时间依赖、事件持续性、正负尾不对称和冻结黑盒接口。 |
| B | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection | Yuanyuan Yang / KDD / 2023 | [ACM DL](https://doi.org/10.1145/3580305.3599295) | 用双注意力和对比表示改进多变量时序异常检测。 | 中 | 检测的是已经出现的异常，不等于在合法信息截止前预测未来尖峰；不能直接当作极端电价预测器。 |
| B | Foreseeing the worst: Forecasting electricity DART spikes | Rémi Galarneau-Vincent / Energy Economics / 2023 | [DOI/出版社](https://doi.org/10.1016/j.eneco.2023.106521) | 预测日前与实时价差（DART）尖峰发生概率，直接面向高损失尾部事件。 | 强 | 聚焦 DART 正向尖峰；不是双尾绝对电价校正，也没有冻结多基座通用性。 |
| B | Forecasting the Occurrence of Electricity Price Spikes: A Statistical-Economic Investigation Study | Manuel Zamudio López / Forecasting / 2024 | [DOI/全文](https://doi.org/10.3390/forecast6010007) | 比较统计阈值和经济阈值，采用二分类器评估尖峰发生预测。 | 强 | 主要解决 occurrence，不解决触发后的幅值恢复；二值阈值具有市场依赖性。 |
| B | Integrating PNN classification and ELM-Bootstrap for enhanced Day-Ahead negative price forecasting | Stylianos Loizidis / Applied Energy / 2025 | [DOI/出版社](https://doi.org/10.1016/j.apenergy.2025.126013) | 先分类负价风险，再用 ELM-Bootstrap 做概率式负价预测。 | 强 | 只处理负向尾部；架构专用，未形成适用于任意冻结基座的双向选择性校正。 |

### B 家族判断

已较成熟：

- 连续标签不平衡回归的重加权、分布平滑和表示约束；
- 对高价尖峰发生概率的分类；
- 对负价的分类—回归混合；
- 时序异常检测。

仍未做透：

- “未来极端预测”与“观测后异常检测”经常被混淆；
- 正向尖峰与负价通常被分开建模，没有成熟统一的双尾校正器；
- occurrence 与 magnitude 很少在同一选择性框架中联合优化；
- 正常期退化预算、拒绝校正、阈值迁移和跨市场 tail calibration 缺乏统一规范。

---

## 4. C 家族：不确定性量化（P2）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| C | Adaptive Conformal Inference Under Distribution Shift | Isaac Gibbs / NeurIPS / 2021 | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | 为未知在线分布漂移提供可包裹任意黑盒预测器的自适应共形方法。 | 强 | 主要保证长期边际覆盖；极端条件覆盖、分组样本稀疏和剧烈切换时的区间宽度仍是问题。 |
| C | Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting | Kashif Rasul / ICML / 2021 | [PMLR](https://proceedings.mlr.press/v139/rasul21a.html) | TimeGrad 用扩散式自回归采样学习多变量未来分布。 | 中 | 计算和采样成本高；不是后处理校准器，且总体 CRPS 好不代表双尾稀有事件校准好。 |
| C | Conformal prediction interval estimation and applications to day-ahead and intraday power markets | Christopher Kath / International Journal of Forecasting / 2021 | [DOI/出版社](https://doi.org/10.1016/j.ijforecast.2020.09.006)；[arXiv](https://arxiv.org/abs/1905.07886) | 将共形区间系统应用于日前和日内电力市场，并展示黑盒兼容性。 | 强 | 论文在线版本始于 2019、期刊发表于 2021；重点是区间覆盖，不是双尾事件触发、幅值修正和经济风险联合训练。 |
| C | Sequential Predictive Conformal Inference for Time Series | Chen Xu / ICML / 2023 | [PMLR](https://proceedings.mlr.press/v202/xu23r.html) | SPCI 利用历史残差的序列依赖自适应估计条件分位数。 | 强 | 标量输出与序列反馈设置仍需扩展到 24/96 点联合覆盖；尾部样本不足会导致条件分位数不稳。 |
| C | Conformal Prediction for Time Series with Modern Hopfield Networks | Andreas Auer / NeurIPS / 2023 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html) | HopCPT 检索与当前状态相似的历史误差模式来构造区间。 | 强 | 类似状态检索在概念漂移或从未出现的市场冲击下可能失效；未直接优化极端幅值与交易价值。 |
| C | Distributional neural networks for electricity price forecasting | Grzegorz Marcjasz / Energy Economics / 2023 | [DOI/出版社](https://doi.org/10.1016/j.eneco.2023.106843)；[arXiv](https://arxiv.org/abs/2207.02832) | 让神经网络直接输出正态或 Johnson SU 分布，强调偏度与高阶矩对电价风险的重要性。 | 强 | 参数分布若失配会低估真实双尾；它是预测基座，不是对任意模型输出做后验校准。 |

### C 家族判断

已较成熟：

- 黑盒式边际覆盖；
- 在线分布漂移下的共形更新；
- 用历史相似残差构造局部区间；
- 分位数、分布回归和生成式概率预测。

仍未做透：

- 极端条件覆盖，而不只是全体样本的边际覆盖；
- 24/96 个时点的路径级联合覆盖；
- 正尾与负尾分别校准；
- 置信区间如何决定“校正 / 不校正 / 保守校正”；
- 区间覆盖、点预测修正和交易下行风险如何统一。

---

## 5. D 家族：校正头 / 即插即用模块（P3）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| D | Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift | Taesung Kim / ICLR / 2022 | [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p) | RevIN 以可逆归一化减轻输入实例间和训练—测试分布偏移，可插到多种基座。 | 中 | 是归一化插件，不是冻结模型输出后的事件选择性残差修正；均值/方差处理可能压缩极端信号。 |
| D | Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting | Wenhui Fan / AAAI / 2023 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25914) | 分别学习输入端与输出端的分布系数，缓解时序分布漂移。 | 中 | 仍围绕整体分布漂移；没有双尾事件定义、正常期退化预算或交易价值。 |
| D | Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective | Zhiding Liu / NeurIPS / 2023 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html) | SAN 以时间切片方式预测未来统计量，并作为多基座插件。 | 中 | 校正的是非平稳统计量，不是对冻结黑盒输出做稀有事件触发式幅值校正。 |
| D | Improving Time Series Forecasting via Instance-aware Post-hoc Revision | Zhiding Liu / NeurIPS / 2025 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html)；[arXiv](https://arxiv.org/abs/2505.23583) | PIR 先估计实例是否可能失败，再用局部与全局上下文后处理修订预测，且模型无关。 | **极强** | 与原“触发器 + 校正器”结构高度接近；未专门处理电价双尾、事件级指标、交易约束和显式正常期退化预算。 |
| D | Post-Training Corrections for Improved Time-Series Forecasting | Malik Tiomoko / arXiv / 2025–2026 | [arXiv](https://arxiv.org/abs/2505.15354) | 以类似 boosting 的连续轻量变换，在不重训基座的情况下校正预测；强调模型无关与可解释。 | **极强** | 当前主要是预印本；通用变换可能无法捕捉极端事件机制，但它已占据“模型无关后训练校正”表述。 |
| D | The Forecast After the Forecast: A Post-Processing Shift in Time Series | Daojun Liang / ICLR / 2026 | [OpenReview](https://openreview.net/forum?id=syfWdclGE1)；[arXiv](https://arxiv.org/abs/2601.20280) | δ-Adapter 在冻结基座的输入与输出接口学习有界模块，并提供分位数校准、共形修正和稳定性分析。 | **极强** | 与“轻量、即插即用、模型无关、可解释、UQ 校正头”几乎正面重合；未针对双向电价极端与交易风险。 |
| D | Reviving Error Correction in Modern Deep Time-Series Forecasting | Minh Hoang Nguyen / arXiv / 2026 | [arXiv](https://arxiv.org/abs/2605.21088) | UEC-STD 对趋势与季节分量分别做架构无关误差校正，无需重训基座。 | 强 | 主要针对自回归长预测误差累积；未做电价双尾选择性触发，但进一步压缩了“通用误差校正器”的 novelty。 |

### D 家族判断

已做透或高度拥挤：

- 模型无关后处理；
- 失败实例识别后再修订；
- 冻结基座的输入/输出 adapter；
- 轻量残差修正；
- 与分位数、共形校准结合；
- 多基座通用性验证。

仍然有空间：

- 有业务语义的双尾选择性校正；
- occurrence、signed magnitude、duration 的联合任务；
- 正常期退化上界或风险预算；
- 跨市场、跨基座、跨极端定义的稳健性；
- 合法信息截止下的无泄漏残差库；
- 统计、事件、校准、交易风险同时改进的 Pareto 设计。

**直接结论：不能把“模型无关校正头”本身写成第一创新点。**

---

## 6. E 家族：特征图与因果结构（P1 / P3）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| E | Discrete Graph Structure Learning for Forecasting Multiple Time Series | Chao Shang / ICLR / 2021 | [OpenReview](https://openreview.net/forum?id=WEHSlH5mOk) | 从多变量序列中联合学习离散图结构与预测器。 | 中 | 统计依赖图不等于因果图；变量作为节点的语义与山东市场实际机制需重新定义。 |
| E | Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting | Junchen Ye / KDD / 2022 | [ACM DL](https://doi.org/10.1145/3534678.3539274)；[arXiv](https://arxiv.org/abs/2206.13816) | 学习随时间和尺度演化的多变量图结构。 | 强 | 适合作为动态依赖基线，但不是黑盒后处理；图边的可解释性与可识别性没有自然保证。 |
| E | CUTS: Neural Causal Discovery from Irregular Time-Series Data | Yuxiao Cheng / ICLR / 2023 | [OpenReview](https://openreview.net/forum?id=UG8bQcD3Emv) | 联合插补不规则观测并进行非线性 Granger 因果发现。 | 中 | 因果发现假设较强；电力市场有潜在混杂、政策干预和同步变量，不能把学习到的边直接称为真实因果。 |
| E | FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective | Kun Yi / NeurIPS / 2023 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html) | 在 Fourier 空间将多变量时序统一为图结构进行预测。 | 中 | 频域全局依赖可能弱化短促极端；不提供市场结构或因果解释。 |
| E | CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement | Qihe Huang / NeurIPS / 2023 | [OpenReview](https://openreview.net/forum?id=xOzlW2vUYc) | 通过时间尺度图与变量尺度图处理噪声多变量预测。 | 中 | 主要优化整体基准误差，未证明对电价稀有尾部和外部冲击更有效。 |
| E | Forecasting day-ahead electricity prices with spatial dependence | Yifan Yang / International Journal of Forecasting / 2024 | [DOI/出版社](https://doi.org/10.1016/j.ijforecast.2023.11.006) | 用 R-vine copula 预构建区域价格空间依赖图，再用 STGNN 预测 Nord Pool 多区域价格。 | 强 | 依赖多价格区空间结构；单区域山东任务若无节点语义，不能机械套用。未做极端校正。 |
| E | Day-ahead electricity price prediction in multi-price zones based on multi-view fusion spatio-temporal graph neural network | Anbo Meng / Applied Energy / 2024 | [DOI/出版社](https://doi.org/10.1016/j.apenergy.2024.123553) | 融合距离、价格相关性和分布相似性等多视图构图，预测多价格区电价。 | 强 | 多视图相关图仍不是因果图；适用前提是存在多个有经济或物理联系的价格区。 |

### E 家族判断

已较成熟：

- 静态、离散、动态、多尺度和频域图结构学习；
- 多价格区电价的空间依赖建模；
- 不规则时序的因果发现。

仍未做透：

- “特征图”节点到底是地区、机组、市场变量、时段还是事件；
- 图结构是否能在信息截止前稳定估计；
- 图边是相关、Granger 预测关系还是可干预因果关系；
- 动态图是否真的提高双尾极端，而不是只提高整体平均指标；
- 如何把图作为触发器上下文，而不迫使 P3 依赖某个图基座。

---

## 7. F 家族：价值对齐与经济目标（P3）

| 家族 | 标题 | 第一作者 / 会议或期刊 / 年份 | 链接 | 核心贡献 | 相关性 | 主要局限或未解决问题 |
|---|---|---|---|---|---|---|
| F | Decision-Focused Learning: Through the Lens of Learning to Rank | Jayanta Mandi / ICML / 2022 | [PMLR](https://proceedings.mlr.press/v162/mandi22a.html) | 将决策聚焦学习解释为对可行解的排序学习，直接优化决策质量。 | 强 | 通用组合优化框架；需要将电价交易规则、约束、成本和风险准确形式化。 |
| F | Learning Locally Optimized Decision Losses | Sanket Shah / NeurIPS / 2022 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/file/0904c7edde20d7134a77fc7f9cd86ea2-Paper-Conference.pdf) | 学习局部决策损失以避免对不可微优化器进行端到端求导。 | 中 | 代理损失是否跨市场、跨策略稳定仍是问题；可能牺牲预测可解释性和概率校准。 |
| F | Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-Focused Approach | Linwei Sang / IEEE Transactions on Smart Grid / 2022 | [IEEE Xplore](https://ieeexplore.ieee.org/document/9755891/)；[arXiv](https://arxiv.org/abs/2305.00362) | 用套利 regret 的可导上界与预测误差构造混合损失，直接提升储能套利价值。 | **极强** | 已经证明“把经济目标加入电价预测”并非新点；优化对象依赖具体储能模型，泛化到其他交易策略需验证。 |
| F | Decision-Focused Retraining of Forecast Models for Optimization Problems in Smart Energy Systems | Maximilian Beichter / ACM e-Energy / 2024 | [ACM DL](https://doi.org/10.1145/3632775.3661952)；[开放 PDF](https://publikationen.bibliothek.kit.edu/1000172455/153547971) | 对已有预测器做决策聚焦再训练，改善智能能源系统下游优化价值。 | 强 | 需要领域知识和代理问题；作者也指出它不是纯粹的通用 DFL。 |
| F | Value-oriented price forecasting for arbitrage strategies of Energy Storage Systems through loss function tuning | Ruben Smets / Energy / 2025 | [DOI/出版社](https://doi.org/10.1016/j.energy.2025.137112) | 用下游套利利润选择和调节通用损失函数，服务不同储能系统。 | **极强** | “价值对齐损失”本身已有直接先例；选择策略、市场和成本假设变化时，最优损失可能不稳定。 |
| F | Online Energy Storage Arbitrage under Imperfect Predictions: A Conformal Risk-Aware Approach | Yiqian Wu / ACM e-Energy / 2026 | [ACM DL](https://doi.org/10.1145/3744255.3798116)；[arXiv](https://arxiv.org/abs/2511.01032) | 用共形决策理论和在线校准控制错误价格预测引起的套利下行风险。 | **极强** | “共形 + 套利风险”也已有近邻；仍依赖具体储能控制器，未解决模型无关双尾价格校正。 |

### F 家族判断

已较成熟：

- predict-then-optimize / decision-focused learning；
- 可导 regret 或代理决策损失；
- 对已有预测器做 decision-focused retraining；
- 用收益选择损失；
- 用共形集合控制套利下行风险。

仍未做透：

- 双向极端校正对多种交易者和多种下游策略是否都增值；
- 统计精度、极端事件、利润、CVaR 与正常期稳定性的 Pareto 前沿；
- 当交易规则、手续费、电池退化、容量和报价策略变化时，方法是否仍稳健；
- 经济目标是否会诱导预测器故意产生统计偏差，进而破坏校准或其他用途。

---

## 8. 跨家族空白矩阵

| 模块 | 前人已做透或高度成熟 | 仍然空白 | 最具潜力的创新切入点 |
|---|---|---|---|
| P1 预测基座与特征图结构 | 通用 TS Transformer/MLP/TSFM；外生变量；多变量动态图；多价格区 STGNN；EPF 开放基准 | 合法截止前极端信号的可预测性；机制变量与图节点语义；双尾前兆跨市场稳定性；基座失败与信息不足的区分 | 冻结多类型基座，只输出预测、可用外生上下文和历史 OOS 残差；图结构仅作为可选上下文，不绑死 P3 |
| P2 极端预测与 UQ | 不平衡回归；尖峰分类；负价分类—回归；分位数/分布回归；时序共形与漂移适配 | 双尾 occurrence + magnitude 联合建模；尾部条件覆盖；路径级覆盖；阈值迁移；极端触发的误报成本 | signed rarity + 双尾概率 + 条件共形路由；分别校准正尾、负尾与正常区 |
| P3 校正头与价值对齐 | 通用模型无关后处理、失败识别与修订、δ-Adapter、误差校正、decision-focused / arbitrage-aware 学习 | 显式正常期退化预算；双尾选择性校正；跨基座跨市场安全退化；多种下游策略下的经济稳健性 | “选择性双尾安全校正器”，以风险预算决定触发，并同时评估事件、校准、统计和交易价值 |

---

## 9. 候选创新点（按推荐程度排序）

### 创新点 1：双尾选择性安全校正器（Bidirectional Tail-Selective Safe Corrector）

- **对应空白**：
  - 现有尖峰与负价多被分开处理；
  - PIR/δ-Adapter 是通用后处理，但没有电价双尾、正常期退化预算和事件语义；
  - 现有校正器通常默认“改了更好”，缺少明确的 abstain / no-change 机制。
- **核心设计**：
  - 用连续 signed rarity 表示负尾、正常区、正尾，而不是一个无方向二值标签；
  - 触发器输出 `p_negative`、`p_positive_spike`、`p_normal` 与置信度；
  - 两个幅值专家分别学习负尾与正尾残差，正常专家固定为零修正或极小修正；
  - 以拉格朗日约束或风险控制方式显式规定：正常时段 MAE/sMAPE 退化不超过预算，或正常期错误修正率不超过阈值；
  - 当不确定性过高时拒绝校正，保留原基座输出。
- **可吸收基线**：
  - DIR、RankSim；
  - PIR、δ-Adapter、Post-Training Corrections；
  - 尖峰 occurrence 分类与 PNN-ELM 负价分类—回归。
- **初步验证设想**：
  - 基座：LEAR/LightGBM、LSTM/NBEATSx、PatchTST/iTransformer、TimeMixer、一个 TSFM；
  - 数据：山东 + 山西 + EPEX + Nord Pool + NSW/PJM 中至少三个公开市场；
  - 指标：总体 MAE/RMSE/sMAPE；正尾/负尾 MAE；AUCPR；event hit/miss、提前量、峰值幅度、持续时间；正常期退化；触发覆盖率；
  - 采用严格 rolling-origin，基座训练、校正器训练、门控校准、最终测试四段隔离；
  - 报告每个基座、市场、年份的 Pareto 前沿，而不是只报平均提升。
- **预期贡献类型**：任务创新 + 方法创新 + 评估创新。
- **风险**：若只是两个专家加门控，会被认为是普通 MoE；必须把“选择性风险预算、双尾条件校准和冻结多基座”做成不可替换的技术核心。

### 创新点 2：尾部条件共形路由与幅值闭环（Tail-Conditional Conformal Routing）

- **对应空白**：
  - 普通共形方法主要保证边际覆盖，可能在最关心的极端区间严重欠覆盖；
  - δ-Adapter 已有 Quantile Calibrator 与 Conformal Corrector，因此不能只做“加一个共形层”。
- **核心设计**：
  - 对正常、负尾、正尾建立分组或连续加权非一致性分数；
  - 使用历史 OOS 残差和 regime context 做在线校准；
  - 只有当尾部概率、区间位置和覆盖风险共同满足条件时触发幅值校正；
  - 将“是否校正”视为带覆盖约束的选择性预测问题；
  - 校正后重新计算残差并滚动更新，但所有更新只使用预测时点之后已经公开的真实值。
- **可吸收基线**：
  - ACI、SPCI、HopCPT；
  - Kath & Ziel 的电力市场共形区间；
  - δ-Adapter 的 Quantile/Conformal Corrector。
- **初步验证设想**：
  - 比较 marginal coverage、正尾 conditional coverage、负尾 conditional coverage、平均区间宽度、极端误报率；
  - 在年度漂移、能源危机月份、负价频率突变和不同市场阈值下测试；
  - 对 24/96 点路径同时报告逐点覆盖与路径级覆盖；
  - 做无共形、普通 split conformal、ACI、SPCI、HopCPT 与提出方法的消融。
- **预期贡献类型**：方法创新 + 校准评估创新。
- **风险**：尾部校准集非常小；需避免随意分箱，并给出有限样本或在线风险界限。

### 创新点 3：统计—事件—交易三目标的风险预算校正（Decision-Aligned Tail Correction）

- **对应空白**：
  - Sang、Smets、Beichter 和 Wu 已证明套利/决策目标可进入训练或校准；
  - 但这些工作没有统一研究“只校正双向极端”以及正常期退化。
- **核心设计**：
  - 不把利润简单加权进 loss，而将问题设为约束优化：
    - 最小化正尾/负尾事件损失；
    - 约束总体误差与正常期退化；
    - 最大化净利润或最小化 regret；
    - 控制 CVaR / downside loss；
  - 对多个下游策略共同训练或做 distributionally robust 选择，避免过拟合单一电池参数。
- **可吸收基线**：
  - Sang 2022 的 surrogate regret；
  - Mandi 2022、Shah 2022 的 DFL；
  - Smets 2025 的 value-oriented loss tuning；
  - Wu 2026 的 conformal risk-aware arbitrage。
- **初步验证设想**：
  - 至少两种下游：储能套利 + 报量/报价或价差交易；
  - 多容量、功率、效率、手续费、退化成本与风险偏好；
  - 指标：净收益、regret、CVaR、最大回撤、收益稳定性、约束违反率；
  - 必须使用预测时可知信息，交易模拟不能使用未来实际价调参；
  - 用 block bootstrap 给利润差异置信区间。
- **预期贡献类型**：方法创新 + 决策/应用创新。
- **风险**：经济目标高度依赖仿真假设；若只在一个电池参数上有效，会被认为是策略过拟合。

### 创新点 4：跨市场、跨基座的少样本安全迁移校正（Cross-Market Safe Meta-Calibration）

- **对应空白**：
  - “模型无关”常只表示能接四个模型，不代表能跨市场；
  - 电价阈值、上下限、负价频率和尖峰机制强烈依赖市场。
- **核心设计**：
  - 用市场归一化后的连续稀有度、价格上下限和极端分位位置作为公共语义；
  - 将基座身份、市场身份、预测步长作为条件变量，而不是为每个组合完全重训；
  - 在新市场只用少量、严格历史可得的 OOS 残差校准门控和尺度；
  - 为迁移失败设置 abstention 与回退到原预测的机制。
- **可吸收基线**：
  - Moirai/TSFM 的跨域思想；
  - PIR 的实例失败识别；
  - ACI/SPCI 的在线校准；
  - 多市场 EPF benchmark。
- **初步验证设想**：
  - leave-one-market-out；
  - leave-one-backbone-out；
  - few-shot 校准天数曲线；
  - 市场规则改变或负价频率突变前后测试；
  - 报告最坏市场、最坏基座而不是只报均值。
- **预期贡献类型**：方法创新 + 泛化评估创新。
- **风险**：若没有足够公开市场和统一字段映射，跨市场结论无法成立。

### 不建议作为主创新：动态因果特征图

它可以作为辅助触发器上下文或消融，但不建议与 P3 同时作为两个大模块硬塞进一篇论文：

- 动态图预测本身已有 KDD/NeurIPS/ICLR 大量工作；
- 因果发现需要额外假设、验证与反事实证据；
- 电力市场中存在潜在混杂、政策变化、同步变量和数据不可得；
- 若图只提升少量平均精度，反而分散“选择性双尾校正”的主线。

除非已有多区域、机组/约束或电网拓扑等有明确节点语义的数据，否则先把图作为 P1 可选基座，而非 P3 主创新。

---

## 10. 十个交叉验证问题

### 1. “模型无关的极端电价校正头”是否还有 novelty？

**明确判断：宽泛版本没有足够 novelty；收缩后的双尾选择性安全版本仍有空间。**

几乎正面重合的工作：

- [PIR，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html)：失败识别 + 模型无关后处理修订；
- [δ-Adapter，ICLR 2026](https://openreview.net/forum?id=syfWdclGE1)：冻结基座、输入微调、输出残差、分位数与共形校正；
- [Post-Training Corrections](https://arxiv.org/abs/2505.15354)：无需重训、轻量、模型无关的连续输出变换；
- [UEC-STD](https://arxiv.org/abs/2605.21088)：架构无关误差校正。

所以必须把 novelty 放在双尾任务、选择性风险控制、事件—幅值联合、跨市场泛化和交易下行风险，而不是“有一个 correction head”。

### 2. 同时处理负价和尖峰的成熟校正头是否已有？

**明确判断：本轮保留文献中没有发现满足全部条件的成熟方案。**

已有工作倾向于分开处理：

- 正向尖峰 occurrence：[Zamudio López et al., 2024](https://doi.org/10.3390/forecast6010007)；
- DART 尖峰：[Galarneau-Vincent et al., 2023](https://doi.org/10.1016/j.eneco.2023.106521)；
- 负价分类—回归：[Loizidis et al., 2025](https://doi.org/10.1016/j.apenergy.2025.126013)；
- 通用后处理：PIR、δ-Adapter，但没有电价双尾语义。

难点：

1. 正尾和负尾的生成机制与损失不对称；
2. 阈值和市场价格上下限不同；
3. 稀有样本更少，门控容易过拟合；
4. 误触发会破坏大量正常时段；
5. occurrence、magnitude、duration 与路径一致性相互耦合；
6. 跨市场定义双尾时，绝对阈值不可直接迁移。

### 3. P3 是否必须依赖 P1 和 P2？只靠校正头能否发顶会？

**明确判断：P3 不必修改 P1，但当前形势下必须吸收 P2 或决策约束，单靠通用校正头很难。**

- P1 可以完全冻结，论文不需要再发明一个 SOTA 基座。
- P3 至少需要 P2 提供风险、稀有度或校准信号；否则容易退化成普通残差 MLP。
- 纯校正头曾经可以形成方法论文，但 PIR 与 δ-Adapter 已显著抬高门槛。现在只有在理论、安全选择、尾部条件覆盖或跨域泛化上给出新机制，才可能独立成立。

### 4. 基座应该很 SOTA，还是够用即可？

**明确判断：不需要自创 SOTA 基座，但基座覆盖必须足够强且足够异构。**

建议最少覆盖：

1. 强简单模型：LEAR / LightGBM；
2. 传统深度模型：LSTM 或 NBEATSx；
3. Transformer：PatchTST 或 iTransformer；
4. MLP/多尺度：TimeMixer；
5. TSFM：Moirai、TimesFM 或同类；
6. 可选图模型：当数据确有合理图语义时。

审稿人关心的是：

- 是否只对弱基座有效；
- 是否在强基座上仍有增益；
- 是否不同基座共享同一接口；
- 是否存在“已经很准时不乱改”的安全退化。

### 5. 是否必须加入经济价值指标？

**明确判断：不是所有 ML 顶会论文都强制要求，但你的论文只要宣称“提升交易/套利价值”，经济指标就是必需证据。**

至少报告：

- 净收益（扣手续费、效率损失和电池退化）；
- 相对 oracle 的 regret；
- downside loss / CVaR；
- 最大回撤或收益波动；
- 约束违反率；
- 每 MWh 或每度电净收益；
- 多种容量、功率、风险偏好和策略下的敏感性。

同时保留统计指标，因为只优化利润可能制造系统性偏差。Sang 2022、Smets 2025 和 Wu 2026 已说明 MAE/RMSE 与实际决策价值并不等价。

### 6. 只用山东数据是否足够？

**明确判断：若声称“模型无关、通用、可迁移”，只用山东不够。**

建议：

- 中国：山东主数据 + 山西辅助数据；
- 国际公开市场：至少三个，优先 EPEX（德国/法国/比利时）、Nord Pool、PJM 或澳大利亚 NEM/NSW；
- 使用 Lago et al. 的 EPF benchmark 或兼容的公开划分作为可复现实验；
- 做 leave-one-market-out 或至少跨市场零样本/少样本校准。

如果最终只能用山东，应主动收缩论文主张为“山东电力市场的任务方法”，不要写通用模型无关结论；这更像领域论文而非 A 类 ML 会议论文。

### 7. 是否必须可解释？

**明确判断：不是 A 会形式上的硬要求，但对选择性校正器非常有价值；不需要强行上因果图。**

最低可解释性：

- 何时触发、为何触发；
- 正尾、负尾与正常路由概率；
- 修正方向与幅度；
- 使用了哪些可用上下文；
- 拒绝校正案例；
- 成功和失败事件可视化；
- 门控概率的可靠性图。

若声称“因果”，则必须提供比 attention heatmap 更强的证据。没有干预或可识别假设时，建议称为动态依赖图而非因果图。

### 8. 共形 / 分位数校正是否比点估计更适合极端电价？

**明确判断：概率/共形输出更适合风险识别，但不能替代点值幅度校正；最佳路线是二者联合。**

依据：

- [ACI](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) 支持未知分布漂移下的黑盒在线校准；
- [SPCI](https://proceedings.mlr.press/v202/xu23r.html) 直接利用时序残差依赖；
- [HopCPT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html) 使用相似历史误差状态；
- [Kath & Ziel](https://doi.org/10.1016/j.ijforecast.2020.09.006) 已证明共形区间可用于日前和日内电力市场；
- [Distributional NN for EPF](https://doi.org/10.1016/j.eneco.2023.106843) 显示偏度和高阶矩对电价概率预测重要。

但普通边际覆盖可能掩盖尾部欠覆盖。因此需要：

- 正尾/负尾条件覆盖；
- 分别校准上下尾；
- 将区间/概率用于触发；
- 仍用幅值专家输出最终点修正；
- 报告点值、区间与决策风险三类结果。

### 9. 把套利收益直接加入校正头训练是否有先例？风险是什么？

**明确判断：有明确先例，不能把“加入套利损失”单独当创新。**

先例：

- [Sang et al., 2022](https://ieeexplore.ieee.org/document/9755891/)：用 surrogate regret 训练电价预测器；
- [Beichter et al., 2024](https://doi.org/10.1145/3632775.3661952)：对已有预测器做 decision-focused retraining；
- [Smets et al., 2025](https://doi.org/10.1016/j.energy.2025.137112)：用下游价值调损失；
- [Wu et al., 2026](https://doi.org/10.1145/3744255.3798116)：共形风险控制的在线套利。

主要风险：

1. 过拟合某种电池和交易策略；
2. 交易成本、效率、退化和市场规则设定不真实；
3. 决策优化不可微或梯度不稳定；
4. 收益目标诱导统计偏差，破坏其他用户使用；
5. 用测试期收益选择超参数造成泄漏；
6. 极端少样本导致利润结果方差巨大；
7. 一个偶然事件主导全年收益；
8. 不同市场的结算和报价机制不可直接复用。

### 10. 最可能被拒的风险是什么？

若以“模型无关校正头 + 山东数据 + 传统指标”投稿，最可能收到：

1. **Novelty 不足**：PIR、δ-Adapter 等已做模型无关后处理；
2. **任务定义不清**：异常检测、极端 occurrence 与幅值校正混在一起；
3. **单私有数据集**：无法验证通用性与复现；
4. **基线不强或不完整**：没有 PIR、δ-Adapter、RevIN/SAN、DIR、共形和 DFL 对照；
5. **阈值随意**：尖峰/负价定义对结果敏感；
6. **平均指标掩盖极端**：只报 MAE/RMSE/sMAPE；
7. **正常期退化被隐藏**：总体平均变好但大量正常点被破坏；
8. **泄漏风险**：残差、真实事件标签、滚动校准和信息截止边界未严格隔离；
9. **经济模拟不可信**：忽略手续费、效率、退化、容量与报价约束；
10. **统计显著性不足**：收益由少数极端事件决定却没有 block bootstrap / 事件级置信区间；
11. **“模型无关”证据不足**：只在一两个相似深度模型上测试；
12. **指标口径问题**：任何会在负价或低价区人为裁剪误差的指标都不应作为主要极端指标。例如把真实值与预测值统一 floor 到正数，会直接掩盖负价失败。

---

## 11. 推荐的论文主线与实验最小闭环

### 11.1 推荐主线

> A selective, bidirectional, risk-calibrated post-hoc corrector for extreme electricity prices under a normal-regime degradation budget.

中文：

> 面向双向极端电价、带正常期退化预算的选择性风险校准后处理器。

这条主线保留模型无关优点，但明确避开 δ-Adapter/PIR 已经占据的泛化表述。

### 11.2 最小方法组件

1. **Frozen backbone interface**
   - 基座点预测；
   - 预测时合法可用的上下文；
   - 只由历史滚动 OOS 预测形成的残差；
   - 可选基座身份，不读取基座内部梯度。
2. **Signed rarity / tail-risk estimator**
   - 正尾、负尾、正常区；
   - 连续稀有度优于单一二值阈值；
   - 市场内分位数与绝对业务阈值并行。
3. **Selective gate**
   - 校正、保守校正、拒绝校正；
   - 基于条件共形或校准概率控制误触发。
4. **Bidirectional magnitude experts**
   - 正尾与负尾分开学习；
   - 可共享干层，但输出和约束不同。
5. **Safety constraint**
   - 正常期退化预算；
   - 最坏市场/最坏基座约束；
   - 回退到原预测。
6. **Optional decision layer**
   - 作为训练约束或模型选择目标；
   - 不能只服务单一电池设定。

### 11.3 实验矩阵

| 轴 | 最低要求 |
|---|---|
| 市场 | 山东、山西、至少 3 个公开国际市场 |
| 基座 | 1 个线性/树模型、1 个 RNN/NBEATS、1 个 Transformer、1 个 MLP/多尺度、1 个 TSFM |
| 预测长度 | 24 点；若数据支持再加 96 点 |
| 划分 | 严格 rolling-origin；基座训练 / 校正训练 / 校准 / 最终测试隔离 |
| 极端定义 | 业务绝对阈值 + 市场内分位阈值；阈值敏感性 |
| 统计指标 | MAE、RMSE、sMAPE（不裁剪负价）、DM 检验或配对 block bootstrap |
| 事件指标 | AUCPR、precision/recall、event hit/miss、lead time、peak magnitude、duration |
| 校准指标 | overall/positive-tail/negative-tail coverage、width、reliability、ECE/Brier |
| 安全指标 | 正常期退化、误触发率、拒绝率、最坏市场/基座表现 |
| 经济指标 | 净收益、regret、CVaR/downside、回撤、成本与参数敏感性 |

### 11.4 必做消融

1. 无门控，始终校正；
2. 二值极端标签 vs 连续 signed rarity；
3. 单专家 vs 正负双专家；
4. 无共形 vs 普通共形 vs 尾部条件共形；
5. 无安全预算 vs 不同退化预算；
6. 不含经济目标 vs 混合目标 vs 约束式目标；
7. 不含基座身份 vs 含基座身份；
8. 单市场训练 vs 跨市场训练 vs 少样本校准；
9. 仅点级指标 vs 事件级选择模型；
10. 随机残差划分 vs 严格滚动 OOS 残差，作为泄漏警示对照。

---

## 12. 对原四个候选想法的批判性评估

| 原想法 | 判断 | 原因 | 建议 |
|---|---|---|---|
| 极端感知型模型无关校正头 | **保留但必须重构** | 通用 correction head 已被 PIR、δ-Adapter、post-training corrections 覆盖 | 改成双尾、选择性、风险预算、跨市场；把“模型无关”降为性质而非创新核心 |
| 条件共形预测 + 残差闭环 | **有潜力但不能泛化表述** | 共形时序与 δ-Adapter conformal corrector 已存在 | 做 tail-conditional / signed-tail / pathwise coverage，并让覆盖风险决定是否校正 |
| 动态因果特征图 + 外部冲击传播 | **暂不建议主线** | 动态图和时序因果发现都很拥挤；数据与可识别性要求高 | 先作为可选上下文和消融；有明确多区域/拓扑节点后再独立发展 |
| 经济价值对齐损失 | **必须做评估，但单独不新** | Sang、Smets、Beichter、Wu 已有直接先例 | 用多策略风险预算、稳健 Pareto 与跨市场泛化形成新意，不要只把 profit 加进 loss |

---

## 13. 最终建议

### 推荐保留的论文定位

**主问题**：冻结异构基座在双向极端电价上存在系统性尾部失败；如何只在有把握时修正，并用显式预算保护正常期？

**主方法**：signed-tail risk estimator + conditional conformal selective gate + bidirectional residual experts + safety budget。

**主证据**：

- 多基座；
- 多市场；
- 严格滚动样本外；
- 双尾 occurrence + magnitude；
- 尾部条件覆盖；
- 正常期退化；
- 多策略交易价值与下行风险。

### 不应再使用的主张

- “首次提出模型无关时序校正头”；
- “首次将共形预测用于电价”；
- “首次把套利收益加入电价预测”；
- “attention/图边就是因果解释”；
- “在山东一个数据集上有效，因此对所有市场通用”。

### 最适合的投稿语境

- 若方法包含明确的选择性风险控制、跨市场泛化与充分理论/实验：KDD、AAAI、IJCAI、WSDM/WWW 的方法或应用研究语境更匹配；
- 若主要是电力市场方法与交易仿真：IEEE TSG、Applied Energy、Energy Economics、International Journal of Forecasting 更自然；
- 若目标是 NeurIPS/ICML/ICLR，必须把贡献抽象成一般的“稀有双尾时序选择性后处理”问题，并在非电价长尾时序上补充通用验证，否则容易被认为领域工程。

---

## 附：本轮核验的四个 MCP 项目

| 项目 | 实际类型与状态 | 适用价值 | 本轮注意事项 |
|---|---|---|---|
| [ydzat/literature-review-mcp](https://github.com/ydzat/literature-review-mcp) | Node/TypeScript MCP；README 明示“正在重构中” | DBLP/OpenReview/Papers With Code 搜索、批量分析、综述导出、可配合 Notion | 需要 LLM API Key；Notion 还需单独 MCP；不宜在重构状态下无审计地接入生产工作流 |
| [adamamer20/paper-search-mcp-openai](https://github.com/adamamer20/paper-search-mcp-openai) | Python MCP / OpenAI Deep Research connector | arXiv、PubMed、bioRxiv、medRxiv、Google Scholar、IACR、Semantic Scholar | README 的 TODO 仍列出 IEEE/ACM/ScienceDirect/Springer 未直接实现；不能宣传为完整出版社覆盖 |
| [12458/arxiv-mcp-server](https://github.com/12458/arxiv-mcp-server) | FastMCP v2；arXiv 专用 | 搜索、下载、读取、deep_paper_analysis prompt；STDIO/HTTP/SSE | 只覆盖 arXiv，不能核验正式会议版本与付费出版社元数据 |
| [cktbarking/semanticMCP](https://github.com/cktbarking/semanticMCP) | FastMCP；Semantic Scholar + DOI 下载器 | 论文发现、DOI 解析、开放 PDF 获取 | 出版社下载受开放访问、机构权限和频率限制；“识别 DOI”不等于“必然可下载全文” |

本轮没有将它们表述为“已在当前会话连接并调用”。实际文献证据来自本报告第 1 节列出的公开论文与出版页面。
