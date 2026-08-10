# 模型无关极端电价校正：阶段二交叉验证与创新点收敛

**日期：2026-07-31**  
**阶段：二阶段（对照、分歧补检、收敛）**  
**研究对象：面向双向极端电价、冻结异构基座的选择性后处理校正器**  
**目标语境：KDD / WSDM / WWW / AAAI / IJCAI；若补足通用理论与非电价任务，再考虑 NeurIPS / ICML / ICLR**

---

## 0. 执行结论

本阶段没有重复阶段一的六大家族论文罗列，而是：

1. 对本人阶段一结论与当前可见的 WorkBuddy 候选摘要进行对照；
2. 对三条关键争议各补检 12–13 篇阶段一未出现的 2023–2025 顶会论文；
3. 将原有候选收敛为 **一个主创新 + 一个理论/机制创新**；
4. 把跨市场迁移、经济价值和动态图分别降为 **验证轴、约束/评估、可选上下文**。

当前最稳妥的论文定位是：

> **A selective bidirectional occurrence–magnitude corrector for frozen forecasters, with signed-tail action-risk calibration and an explicit normal-regime degradation budget.**

中文：

> **面向冻结预测基座的双向极端事件—幅值选择性校正器：以符号尾部动作风险校准控制触发，并显式约束正常时段退化。**

### 0.1 最终保留的两项创新

1. **主创新：双向事件—幅值选择性安全校正器**  
   将负价尾和正向尖峰分开建模，把“是否发生极端”与“极端发生后的残差幅值”拆开；校正动作受正常期退化预算约束，风险不够低时回退到原预测。

2. **机制/理论创新：符号尾部分组的共形动作风险路由**  
   共形方法不只输出区间，而是校准“此时执行校正是否安全”；分别控制负尾、正尾和正常区的错误动作风险，并适应时间漂移。

### 0.2 不再独立作为创新

- **普通模型无关 correction head**：PIR、δ-Adapter 等已构成强碰撞，只能作为方法属性。
- **动态因果特征图**：研究拥挤且识别负担大，当前数据条件下只做可选上下文/消融。
- **经济价值 loss**：已有 decision-focused、套利损失、共形风险与电池储能先例，只保留为约束和评估。
- **跨市场少样本迁移**：通用时序迁移已拥挤；保留为通用性证据和压力测试，不单独宣称方法创新。

---

# 5.1 输入材料清单与完整性

```text
读取材料：
- 本人阶段一报告：✅
  文件：electricity_extreme_price_correction_literature_review_2026-07-31.md
- 其他 Agent 报告：❌（待补；当前工作区没有 Agent-X / Agent-Y 的阶段一报告）
- 主调研方候选总览：❌（待补）
  缺少：03_创新点候选_供交叉验证/01_候选创新点总览.md
- 领域全景地图：❌（待补）
  缺少：16_领域全景地图.md
- 主调研方阶段一提示词：✅
  文件：590f41ed-9399-4eb5-a42a-b9e3ccd5b14f.md
- 二阶段提示词：✅
  文件：1d26b322-c90e-4484-91a2-0f80c6b65b30.md
```

### 材料边界

- 当前能确认的“主调研方候选”仅来自阶段一提示词中明确列出的四项摘要：
  1. 极端感知型模型无关校正头；
  2. 条件共形预测 + 残差闭环校正；
  3. 动态因果特征图 + 外部冲击传播；
  4. 经济价值对齐的校正损失函数。
- 这四项在原提示词中被标为“供批判/补充，不要照搬”，因此不能把它们误写成 WorkBuddy 已最终支持的结论。
- 由于缺少其他 Agent 报告，下面的共识度是 **当前可见材料下的临时判断**，不是满足“≥2/3 Agent”的正式多智能体票数。
- 其他 Agent 的主张、证据和冲突项统一标为“待补”，不虚构 Agent 名称或结论。

---

# 5.2 交叉验证对照矩阵

| 候选创新点 | 本人阶段一 | WorkBuddy 候选摘要 | 其他 Agent | 临时共识度 | 备注 |
|---|---:|---:|---:|---|---|
| 普通“模型无关、冻结基座、输出端残差校正头” | ❌ | ✅（初始候选） | 待补 | **冲突，但已有外部证据可裁决** | PIR、δ-Adapter、Post-Training Corrections、UEC-STD 已使宽泛表述拥挤 |
| 双向尾选择性安全校正器 | ✅ | △（可由“极端感知校正头”重构得到） | 待补 | **中高** | 保留为主线；模型无关只是性质，不是 novelty 核心 |
| 双尾 occurrence–magnitude 联合建模 | ✅ | △ | 待补 | **中** | 需要把事件概率与条件幅值分离，并报告事件级证据 |
| 条件共形预测 + 残差闭环 | △（必须改成尾部动作风险路由） | ✅ | 待补 | **中高，但原表述过宽** | “加共形层”不新；需校准“是否执行校正” |
| 正常期显式退化预算 / 安全回退 | ✅ | △ | 待补 | **中** | 是区别于通用 post-hoc revision 的关键约束 |
| 经济价值对齐 loss | △（评估/约束，非独立创新） | ✅ | 待补 | **分歧** | Conformal Risk Training 进一步表明“共形 + CVaR + 电池储能”也已有近邻 |
| 跨市场、跨基座少样本安全迁移 | ✅ | ❌（摘要未提出） | 待补 | **低** | 通用迁移赛道拥挤；并入验证协议，不独立成主创新 |
| 动态因果特征图 + 外部冲击传播 | ❌（主线）/ △（上下文） | ✅（初始候选） | 待补 | **分歧** | 近两年因果时序和动态图方法密集，且需要可识别性与节点语义 |
| 统计—事件—交易三目标风险预算 | ✅ | △（经济价值候选的一种重构） | 待补 | **中** | 并入主创新的约束与评估，不拆成第三个方法模块 |

### 当前裁决

即使后续其他 Agent 多数支持“普通校正头”“动态图”或“经济 loss”，也不能仅凭票数保留。阶段一已发现的直接 SOTA 碰撞和本阶段新增顶会证据，优先级高于多智能体共识。

---

# 5.3 分歧深挖结论

## 争议点 A：跨市场 / 少样本安全迁移能否独立成为主创新？

**支持方：**本人阶段一提出 Cross-Market Safe Meta-Calibration，认为市场归一化稀有度、少样本校准、abstention 和 leave-one-market-out 仍有空间。  
**反对或未支持方：**WorkBuddy 的四项候选摘要没有提出该方向；且通用时序领域的跨域、零样本、少样本和在线适应已经高度活跃。  
**深挖范围：**12 篇阶段一未出现的 2023–2025 顶会论文。

### 新增证据

| 新论文 | 会议 | 对争议的含义 |
|---|---|---|
| [UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting](https://dl.acm.org/doi/10.1145/3589334.3645434) | WWW 2024 | 直接占据“跨域统一时序预测”；仅说 cross-domain 不足以构成新意 |
| [MOMENT: A Family of Open Time-series Foundation Models](https://proceedings.mlr.press/v235/goswami24a.html) | ICML 2024 | 已系统研究多域预训练和有限监督下迁移 |
| [A decoder-only foundation model for time-series forecasting](https://proceedings.mlr.press/v235/das24c.html) | ICML 2024 | TimesFM 展示跨数据域、频率和预测长度的零样本能力 |
| [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://proceedings.mlr.press/v235/liu24cb.html) | ICML 2024 | 以大规模预训练处理少样本与任务泛化 |
| [Large Language Models Are Zero-Shot Time Series Forecasters](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html) | NeurIPS 2023 | 零样本预测本身已有强基线 |
| [Tiny Time Mixers: Fast Pre-trained Models for Enhanced Zero/Few-shot Forecasting](https://proceedings.neurips.cc/paper_files/paper/2024/hash/874a4d89f2d04b4bcf9a2c19545cf040-Abstract-Conference.html) | NeurIPS 2024 | 小模型也能做零/少样本迁移，削弱“轻量迁移”表述的新颖性 |
| [UniTS: A Unified Multi-Task Time Series Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html) | NeurIPS 2024 | 多域、多任务统一表示已经成熟 |
| [OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html) | NeurIPS 2023 | 在线漂移适应已有直接强基线 |
| [DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c44c4afd77d5ee760e7f4bed0c50f878-Abstract-Conference.html) | NeurIPS 2024 | 即插即用的非平稳适应/归一化也已拥挤 |
| [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://openreview.net/forum?id=e1wDDFmlVu) | ICLR 2025 | 跨任务、跨域和稀疏专家均已有大模型路线 |
| [Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage](https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html) | ICLR 2025 | 在严格无泄漏语境下处理在线适应，构成必须比较的基线 |
| [In-context Time Series Predictor](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c699878945bbc2380ea9f421337bea93-Abstract-Conference.html) | ICLR 2025 | 已覆盖 full/few/zero-shot，不更新参数的上下文适应 |

### 结论

**支持本人“跨市场是必要证据”的判断，但推翻“可独立作为第三主创新”的强表述。**

理由：

1. 12 篇新证据证明，跨域、零样本、少样本、漂移适应、插件归一化和在线更新都已是拥挤赛道。
2. 这些工作没有直接解决“冻结异构基座上的双向极端电价 occurrence–magnitude 校正 + 正常期退化预算 + 尾部动作风险控制”。
3. 因此跨市场的价值主要是 **证明主方法不是山东单市场特例**，而不是再发明一个泛化迁移框架。

**阶段二裁决：**不保留为独立创新；并入最小实验闭环，要求 leave-one-market-out、few-shot calibration curve、失败时 abstention/回退。

---

## 争议点 B：动态因果特征图能否作为主创新？

**支持方：**WorkBuddy 阶段一提示词把“动态因果特征图 + 外部冲击传播”列为候选。  
**反对方：**本人阶段一认为它会分散 P3 主线；若没有明确节点语义、干预信息或可识别假设，只能称动态依赖图，不能称因果图。  
**深挖范围：**12 篇阶段一未出现的 2024–2025 ICLR / ICML / NeurIPS 论文。

### 新增证据

| 新论文 | 会议 | 对争议的含义 |
|---|---|---|
| [CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery](https://proceedings.iclr.cc/paper_files/paper/2024/file/0c79d6ed1788653643a1ac67b6ea32a7-Paper-Conference.pdf) | ICLR 2024 | 说明真实时序因果发现缺少 ground-truth，专门需要合成基准评估 |
| [Discovering Mixtures of Structural Causal Models from Time Series Data](https://proceedings.mlr.press/v235/varambally24a.html) | ICML 2024 | 非平稳/异质数据需要混合 SCM 和额外可识别性假设 |
| [Learning Causal Relations from Subsampled Time Series with Two Time-Slices](https://proceedings.mlr.press/v235/wu24p.html) | ICML 2024 | 即便只处理欠采样，也需要专门条件独立结构与理论 |
| [CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series](https://proceedings.mlr.press/v235/lu24i.html) | ICML 2024 | 因果表征与跨域适应的组合已有直接方法，但其任务是分类而非电价校正 |
| [CausalStock: Deep End-to-end Causal Discovery for News-driven Multi-stock Movement Prediction](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54d689d58fe54c92aee2d732fc49fca8-Abstract-Conference.html) | NeurIPS 2024 | 金融预测中“因果图 + 外部文本冲击 + 预测”已有强近邻 |
| [Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series](https://proceedings.neurips.cc/paper_files/paper/2024/hash/68b8d2bc77268facfc75a78782da9559-Abstract-Conference.html) | NeurIPS 2024 | 图结构、连续时间和条件依赖联合建模已很完整 |
| [Causal Deciphering and Inpainting in Spatio-Temporal Dynamics via Diffusion Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2d82a425af4c18a35049899fea5ee82-Abstract-Conference.html) | NeurIPS 2024 | 因果区域识别与时空预测增强已有专门框架 |
| [Improving Generalization of Dynamic Graph Learning via Environment Prompt](https://proceedings.neurips.cc/paper_files/paper/2024/hash/81c565e605161fcf25d08aa230431eba-Abstract-Conference.html) | NeurIPS 2024 | 动态子图、环境变量、OOD 泛化和 SCM 已被共同研究 |
| [ChronoEpilogi: Scalable Time Series Selection with Multiple Solutions](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f24e8cc1c1c06a689850ee766a7357b2-Abstract-Conference.html) | NeurIPS 2024 | 预测等价的多组最小变量集合会使单一“解释图”产生识别歧义 |
| [DyCAST: Learning Dynamic Causal Structure from Time Series](https://openreview.net/forum?id=WjDjem8mWE) | ICLR 2025 | 直接占据“动态因果结构学习”表述 |
| [Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6fde96479648d71e4fd9724374bf76eb-Abstract-Conference.html) | ICLR 2025 | 因果发现与异常根因分析已有联合方案 |
| [On the Identification of Temporal Causal Representation with Instantaneous Dependence](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5726513facc85f802be4a25e77fb9765-Abstract-Conference.html) | ICLR 2025 | 即时依赖下的因果表征需要专门识别理论 |

### 结论

**支持本人阶段一的反对判断：动态图/因果图不应进入当前论文主线。**

新增证据没有找到“动态图直接解决双向极端电价后处理校正”的成熟反例，但这不意味着该方向空白且容易：

1. 近两年顶会已经密集覆盖动态图、动态因果结构、因果表示、异质 SCM、时空因果增强、金融外部冲击和异常根因分析。
2. 若把图接到校正头上，只报预测提升，会被视为模块堆叠。
3. 若声称因果，必须回答节点语义、潜在混杂、即时边/滞后边、干预或识别假设、图的 ground-truth 与稳定性。
4. 当前最核心的问题是“何时安全地修改冻结基座的预测”，动态图不是必要条件。

**阶段二裁决：**砍掉主创新；仅在拥有明确区域/节点/拓扑和预测时合法外生变量时，作为 gate context 的可选消融。没有因果识别证据时统一称 **dynamic dependency graph**。

---

## 争议点 C：条件共形 + 残差闭环是否足以构成创新？

**支持方：**WorkBuddy 候选摘要提出“条件共形预测 + 残差闭环”；本人阶段一也认为尾部条件校准有潜力。  
**反对方：**本人阶段一指出 δ-Adapter 已含 Quantile Calibrator 和 Conformal Corrector；普通共形层或边际覆盖不足以构成新意。  
**深挖范围：**13 篇阶段一未出现的 2023–2025 ICLR / NeurIPS / AAAI 论文。

### 新增证据

| 新论文 | 会议 | 对争议的含义 |
|---|---|---|
| [Conformal PID Control for Time Series Prediction](https://proceedings.neurips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html) | NeurIPS 2023 | 在线时序共形控制和漂移自适应已有强基线 |
| [Class-Conditional Conformal Prediction with Many Classes](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html) | NeurIPS 2023 | 稀少分组下的 class-conditional coverage 已有聚类借力思路 |
| [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | ICLR 2024 | 共形已从覆盖扩展到任意有界单调损失的期望风险控制 |
| [Non-Exchangeable Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html) | ICLR 2024 | 非交换、变点和时间漂移下的 risk control 已被直接处理 |
| [Copula Conformal Prediction for Multi-step Time Series Prediction](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html) | ICLR 2024 | 多步路径依赖与联合覆盖已有专门方案 |
| [Conformalized Time Series with Semantic Features](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dbfb7b1443583fc7ab87e8b1b4f48c9c-Abstract-Conference.html) | NeurIPS 2024 | 语义特征空间加权的时序共形也已存在 |
| [Conformal Prediction via Regression-as-Classification](https://proceedings.iclr.cc/paper_files/paper/2024/file/3a9bbcfde7cbcf2a4b14e3c952e0aee4-Paper-Conference.pdf) | ICLR 2024 | 对异方差、多峰、偏态回归的灵活集合构造已有替代路线 |
| [Neural Conformal Control for Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/34029) | AAAI 2025 | 神经控制式在线共形已经直接面向非平稳时序 |
| [Kernel-based Optimally Weighted Conformal Time-Series Prediction](https://arxiv.org/abs/2405.16828) | ICLR 2025 | 非交换强混合条件下的局部加权和条件覆盖已有理论 |
| [Error-quantified Conformal Inference for Time Series](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html) | ICLR 2025 | 漂移下连续误覆盖反馈比二元反馈更精细 |
| [Sample-Conditional Coverage in Split-Conformal Prediction](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7d83f9e9462c417e208d55e83a5058a8-Abstract-Conference.html) | NeurIPS 2025 | 近似条件覆盖已有新的最优性结果，也提示计算与有限样本仍有难点 |
| [Conformal Risk Training: End-to-End Optimization of Conformal Risk Control](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6559542f75b4452ebaaf82094c7defb7-Abstract-Conference.html) | NeurIPS 2025 | 已覆盖 OCE/CVaR 尾部风险，且包含电池储能利润与尾部损失应用 |
| [Rare event modeling with self-regularized normalizing flows: what can we learn from a single failure?](https://openreview.net/forum?id=gQoBw7sGAu) | ICLR 2025 | 极少失败样本下的生成式稀有事件学习已有专门正则化方案 |

### 结论

**同时支持双方的一部分，并要求重写创新定义。**

- 支持 WorkBuddy：条件/分组/漂移下的风险校准仍是可信切入点。
- 支持本人阶段一：普通“共形区间 + 残差闭环”已经不新。
- 新证据进一步推翻“共形 + CVaR + 电池储能即可创新”的可能性，因为 Conformal Risk Training 已直接覆盖该组合。

真正可保留的差异是：

> **不把共形仅用于输出区间，而是用于校准“是否执行某个后处理校正动作”的风险；动作、风险和误差按负尾、正尾、正常区分组，并同时控制尾部漏校正与正常期误校正。**

必须避免不成立的理论主张：

1. 不声称任意连续条件下的精确 distribution-free conditional coverage。
2. 使用预先定义的 signed-tail groups、近似 group-conditional risk 或受限函数族下的条件风险。
3. 对极小尾部样本报告有效样本量、coverage 置信区间和最小校准样本要求。
4. 当某尾部校准样本不足时，采用层级收缩、相似组聚合或 abstention，而非输出虚假的强保证。

**阶段二裁决：**保留，但从“共形预测层”改为“符号尾部动作风险路由”，并与主创新合并。

---

# 5.4 最终收敛创新点（2 个）

## 创新点 1：双向事件—幅值选择性安全校正器

**英文工作名：Bidirectional Occurrence–Magnitude Selective Safe Corrector（BOM-SSC）**

### 差异化声明（vs 最相关 SOTA）

> **不同于 PIR 的通用失败识别—修订和 δ-Adapter 的通用输入/输出后处理，BOM-SSC 只针对负价与正向尖峰两类有符号稀有尾部，把事件发生与条件幅值分开建模，并在显式正常期退化预算下决定校正、保守校正或回退原预测。**

更精确地说：

- **vs PIR**：PIR 解决通用实例失败识别和 post-hoc revision；本方法解决有符号双尾、occurrence–magnitude 分解、正常期动作安全和电力市场跨域验证。
- **vs δ-Adapter**：δ-Adapter 提供通用输入 nudging、输出 residual、quantile 和 conformal corrector；本方法的核心不是多种 adapter，而是带预算的选择性动作、双尾非对称专家和错误动作风险。
- **vs 普通 hurdle / two-part model**：本方法不只做分类 + 回归，而是对冻结异构基座的 OOS 残差进行后处理，并允许拒绝校正。
- **vs 电价尖峰或负价单任务模型**：统一但不对称地处理负价和正峰，避免把两尾强行共享一个幅值分布。

### 最小方法闭环

1. **输入接口**
   - 冻结基座的点预测或分位数；
   - 预测时刻合法可得的外生变量；
   - 仅由历史 rolling-OOS 预测形成的残差；
   - 可选基座标识和市场标识，不读取基座内部梯度。

2. **连续 signed rarity**
   - 正尾风险 \(r_t^+\)；
   - 负尾风险 \(r_t^-\)；
   - 正常区域概率 \(r_t^0\)；
   - 同时使用市场内分位语义与业务绝对阈值，做阈值敏感性分析。

3. **两阶段输出**
   - occurrence head：预测正尾/负尾/正常事件；
   - magnitude experts：在事件条件下分别预测正尾和负尾残差幅值；
   - 不把“负价”简单定义为回归值小于零后再统一处理。

4. **选择性动作**
   - `correct`：风险低且预期收益明确；
   - `shrink-correct`：证据中等，只做收缩修正；
   - `abstain`：校准不足或可能伤害正常期，直接返回基座预测。

5. **安全约束**
   - 正常期平均退化预算；
   - 正常期上分位/最坏日退化预算；
   - 最坏市场与最坏基座约束；
   - 校正幅值、方向和频率约束。

### 最小实验闭环

| 轴 | 最低要求 |
|---|---|
| 基座集 | LEAR/线性或 LightGBM；LSTM 或 NBEATSx；PatchTST 或 iTransformer；一个 TSFM（TimesFM/Moirai/同类） |
| 市场集 | 山东主数据 + 至少 3 个公开市场；优先 EPEX、Nord Pool、PJM、NEM/NSW |
| 划分 | rolling-origin；基座训练、校正训练、风险校准、最终测试四段隔离 |
| 残差 | 只能用真实 OOS 残差；禁止同一样本先拟合基座再生成“伪 OOS”残差 |
| 统计指标 | MAE、RMSE、sMAPE（不得裁掉负价）、配对 block bootstrap / DM 检验 |
| 事件指标 | 正/负尾 AUCPR、precision、recall、event hit/miss、lead time、duration |
| 幅值指标 | 正尾与负尾条件 MAE、峰值误差、符号错误率、under/over-shoot |
| 安全指标 | 正常期退化、误触发率、拒绝率、最坏日、最坏市场、最坏基座 |
| 经济指标 | 净收益、regret、CVaR/downside、最大回撤；仅作证据和约束，不宣称首次 |
| 泛化 | leave-one-backbone-out；leave-one-market-out；few-shot calibration-day curve |

### 必做消融

1. 始终校正 vs 选择性校正；
2. 单二元 extreme 标签 vs signed rarity；
3. 单幅值专家 vs 正负双专家；
4. occurrence-only、magnitude-only、联合模型；
5. 无安全预算 vs 不同预算；
6. 无 abstention vs abstention；
7. 同分布随机残差 vs 严格 rolling-OOS 残差（泄漏警示对照）；
8. 单市场、跨市场训练、目标市场少样本校准；
9. 不含基座标识 vs 含基座标识；
10. 去掉经济目标后，统计和安全结论是否仍成立。

### 拒稿风险与规避

1. **风险：仍被认为是 PIR / δ-Adapter 的电价应用。**  
   **规避：**把贡献写成新的受约束选择性动作问题；给出形式化风险/退化约束、事件—幅值分解和跨异构基座的必要性实验。

2. **风险：尾部样本太少，门控增益由少数事件驱动。**  
   **规避：**多市场、多个测试年份、事件级 bootstrap、阈值敏感性、最坏事件分析；报告失败市场，不只报均值。

3. **风险：山东私有数据导致不可复现。**  
   **规避：**山东作真实案例，公开市场作主可复现实验；统一数据字典、公开处理代码和严格信息截止表。

### 会议适配

- **KDD / AAAI / IJCAI：**最自然；强调稀有事件、选择性决策、跨域和真实应用。
- **WSDM / WWW：**需强化多源外部事件、跨市场信息或在线系统语境。
- **NeurIPS / ICML / ICLR：**必须抽象为通用的 signed rare-tail post-hoc correction，并补非电价稀有时序任务及理论。

---

## 创新点 2：符号尾部分组的共形动作风险路由

**英文工作名：Signed-Tail Conformal Action-Risk Routing（SCARR）**

### 差异化声明（vs 最相关 SOTA）

> **不同于 δ-Adapter、ACI、Conformal PID、KOWCPI 和 ECI 主要校准预测值或预测区间，SCARR 校准的是“执行校正动作所产生的风险”，并分别约束负尾、正尾和正常区的漏校正、误校正与校正诱发退化。**

该创新应作为 BOM-SSC 的风险控制层，而非另起一个庞大的第二网络。

### 风险变量与动作

令冻结基座预测为 \(\hat y_t^{(0)}\)，候选校正后预测为 \(\hat y_t^{(1)}\)，动作为：

\[
a_t \in \{0,\lambda,1\},
\]

其中 \(0\) 表示回退基座，\(\lambda\in(0,1)\) 表示收缩校正，\(1\) 表示完全校正。

定义动作增量损失：

\[
\Delta \ell_t(a_t)
=
\ell\!\left(y_t,\hat y_t^{(0)} + a_t\Delta_t\right)
-
\ell\!\left(y_t,\hat y_t^{(0)}\right).
\]

目标不是只保证 \(y_t\) 落在区间，而是控制：

\[
\mathbb{E}\!\left[
\phi_g\!\left(\Delta\ell_t(a_t)\right)
\mid g_t=g
\right]
\le \epsilon_g,
\quad
g\in\{\text{negative-tail},\text{normal},\text{positive-tail}\},
\]

其中 \(\phi_g\) 可分别表示：

- 正常区：校正诱发退化；
- 负尾：漏掉负价或修正方向错误；
- 正尾：漏掉尖峰或幅值严重低估；
- 可选：交易 downside / CVaR，但不能把“CVaR + 电池”写成首次。

### 最小实验闭环

| 轴 | 最低要求 |
|---|---|
| 共形/风险基线 | split conformal、ACI、SPCI、HopCPT、Conformal PID、CopulaCPTS、CT-SSF、KOWCPI、ECI、δ-Adapter corrector |
| 分组 | 正尾、负尾、正常；分组规则必须在训练/校准前冻结 |
| 校准 | 只用已揭示真实值的历史窗口；报告延迟标签条件 |
| 指标 | overall/tail coverage、group risk、区间宽度、动作风险、误触发率、拒绝率 |
| 路径 | 24/96 点分别报告 pointwise 与 pathwise/joint 风险 |
| 漂移 | 危机期、负价频率突变、政策/价格上下限变化、跨市场迁移 |
| 小样本 | 每组有效校准样本数、置信区间、不同 calibration window 的稳定性 |

### 理论边界

1. 只在明确写出的交换性、非交换加权、mixing 或在线风险假设下给保证。
2. 不承诺任意 \(X=x\) 的精确条件覆盖；优先做预定义 signed-tail group risk。
3. 当尾部样本不足以支持目标风险水平时，理论上允许 abstain，而不是强行输出窄区间。
4. 若无法给出新的有限样本定理，至少给出：
   - 与现有 CRC/NEX-CRC 的明确归约；
   - 新的动作损失定义；
   - 校正与回退的风险—覆盖—效用 Pareto 曲线；
   - 跨市场下最坏组风险。

### 拒稿风险与规避

1. **风险：审稿人认为只是把 conformal 接到 gate。**  
   **规避：**突出被校准对象是 correction action risk；证明它与输出覆盖不同，并展示同覆盖率下动作风险显著不同。

2. **风险：条件保证不成立或写得过强。**  
   **规避：**限定为预定义组、近似条件风险或受限函数族；明确假设、有限样本项和失效条件。

3. **风险：三组样本极不平衡，负尾保证不稳定。**  
   **规避：**层级/聚类校准、连续 rarity、最小样本门槛、abstention；逐市场报告有效样本量。

### 会议适配

- **NeurIPS / ICML / ICLR：**需要一般动作风险理论，且在电价之外增加至少一个双尾稀有时序任务。
- **KDD / AAAI / IJCAI：**可用严格实验与应用风险闭环支撑，理论要求相对更可控。

---

# 5.5 决策表 + 一句话 Pitch

## 所有候选去留

| 候选 | 去留 | 理由 |
|---|---|---|
| 普通模型无关校正头 | **砍掉独立 novelty** | PIR、δ-Adapter 等已直接覆盖；只保留为 frozen-backbone interface |
| 双向尾选择性安全校正器 | **保留，主创新** | 当前未发现同时覆盖双尾、occurrence–magnitude、动作选择和正常期预算的成熟方案 |
| occurrence–magnitude 联合建模 | **合并进主创新** | 它是区分事件检测与幅值修正的必要机制，不应另起第三个大模块 |
| 条件共形 + 残差闭环 | **重构后保留** | 改为 signed-tail conformal action-risk routing；普通区间校准不新 |
| 正常期退化预算 | **保留，核心约束** | 直接回答“校正何时会伤害大量正常点”，也是与通用 revision 的关键差异 |
| 经济价值对齐 loss | **砍掉独立 novelty，保留评估/约束** | DFL、套利损失、Conformal Risk Training 已覆盖相近思想 |
| 跨市场少样本安全迁移 | **砍掉独立 novelty，保留验证轴** | 通用跨域/零少样本时序方法拥挤；其作用是证明非单市场特例 |
| 动态因果特征图 | **砍掉主线，保留可选消融** | 顶会方法拥挤、识别成本高、与安全校正非必要耦合 |
| 统计—事件—交易三目标 Pareto | **合并为约束和评估** | 避免再造第三个优化模块；用来证明主方法没有以正常期或下游风险为代价 |

## 审稿人 30 秒 Pitch

### Pitch 1：BOM-SSC

> 给任意冻结电价预测器加一个可拒绝的双尾专家：它先判断负价或尖峰是否会发生，再估计条件幅值；只有在风险足够低时才校正，并把正常时段退化显式限制在预算内。

### Pitch 2：SCARR

> 我们不只校准预测区间，而是分别为负尾、正尾和正常区校准“执行校正是否安全”，让不确定性真正决定校正、收缩还是回退。

---

## 6. 最小可投稿方案

### 6.1 一篇论文只讲一条主线

建议把 BOM-SSC 和 SCARR 写成同一篇论文中的：

- **任务/方法主贡献：**BOM-SSC；
- **风险控制机制：**SCARR；
- **实验性贡献：**跨基座、跨市场、双尾事件—幅值—安全—经济四层评估。

不要再加入一个大型动态图模块，也不要把跨市场元学习做成第三套网络。

### 6.2 推荐贡献声明

1. 提出双向极端电价的 post-hoc correction 新任务：同时考虑 occurrence、magnitude 和 normal-regime harm。
2. 提出带 abstention 的双尾选择性校正器，对冻结异构基座统一工作。
3. 提出 signed-tail conformal action-risk routing，在分组风险约束下选择校正强度。
4. 建立跨基座、跨市场、严格 rolling-OOS 的评估协议，并同时报告统计、事件、校准、安全和经济结果。

### 6.3 暂不允许的主张

- “首次提出模型无关时序校正头”；
- “首次将共形预测用于电价”；
- “首次使用 CVaR / 套利损失优化储能收益”；
- “动态图边就是因果关系”；
- “在山东有效即可证明跨市场通用”；
- “边际 coverage 达标即可证明极端尾部可靠”；
- “总体 MAE 提升即可证明校正安全”。

---

## 7. 待补材料到达后的增量动作

收到其他 Agent 报告、候选总览和领域全景地图后，只执行以下增量，不重做本报告的 37 篇补检：

1. 在 5.2 矩阵中增加真实 Agent 列，逐项标注原文证据；
2. 按“≥2/3 同向”重新计算正式共识度；
3. 新发现的候选先与本报告 37 篇新增证据和阶段一 38 篇保留论文去重；
4. 只对新出现的低共识/冲突项再补检 10–20 篇；
5. 检查其他 Agent 是否提出了本报告尚未覆盖的“跨基座共同残差”“事件持续时间/路径一致性”或“选择性预测理论”；
6. 若没有新证据推翻当前裁决，最终创新仍保持 1 个主创新 + 1 个风险机制，不扩成模块堆叠。

---

## 8. 阶段二新增检索统计

```text
阶段一论文清单：不重复
阶段二争议主题：3
新增论文链接：37
  - 跨域/少样本/漂移适应：12
  - 动态图/因果时序：12
  - 条件共形/动作风险/稀有事件：13
与阶段一标题去重：已完成，37 篇均未出现在阶段一报告正文
主要来源：ICLR Proceedings / OpenReview / PMLR / NeurIPS Proceedings / ACM DL / AAAI Proceedings
```

---

## 9. 最终一句话结论

> **不要再发明一个普通 correction head；应研究“对冻结基座的双向极端事件—幅值校正，何时值得执行、何时必须拒绝，以及如何在尾部风险改善的同时把正常期伤害锁进明确预算”。**

