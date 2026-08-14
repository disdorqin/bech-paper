# HCH-v2 Phase 4：CAVM 技术架构设计 v0.1

**文件性质：** 技术设计，不直接授权修改默认生产路径

**日期：** 2026-08-14

**代码基线：**

- v0.4 架构：`disdorqin/bech-paper` 当前 `main@5182ef4`
- Phase3 结果：`exp/r1b-screening-20260813@63b158a`
- 数学依据：`hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md`
- Phase3 权威记录：`docs/训练文件夹/对比实验/hch_v2_phase3_results_master_report_v0.1_2026-08-14.md`

**核心原则：**

1. v0.3 数学核心继续冻结；
2. v0.4 Universal Core / Data-Adaptive Interface / Local Evidence 三层继续保留；
3. 新增内容只解决“经验如何积累、动作如何基于上下文自我校验”；
4. 不新增训练 loss、事件检测头、市场 ID 预测捷径或硬编码价格阈值；
5. `v0.4-core` 必须始终保留，作为所有新实验的控制组。

---

## 1. Phase3 后的正式问题定义

Phase3 已经回答了三个关键问题：

| 结论 | 证据 | 架构影响 |
|---|---|---|
| 国内数据可以作为冻结头迁移评价域 | D2：40 个 cell 中 37 个改善，宏平均 MAE 109.3→91.3，逐 cell 平均 correction +13.9% | 保留国际源域训练的通用核心 |
| 梯度重分配不是当前瓶颈 | T4：两种难度/数据量权重均被 3 个 seed 否证 | 不再继续搜索 gradient weighting |
| 国内数据直接混入主训练有条件 | T5：r=0.15 近似中性，r=0.30 对温和市场产生负迁移 | 新数据优先进入经验层，不直接污染主干 |

因此当前问题不是：

> “怎样让一个更大的校正器拟合所有市场？”

而是：

> “冻结候选模块已经提出 Down / Up / Identity 后，怎样判断当前情景下哪个动作真正值得执行，并让新数据以可验证的方式积累为经验？”

标准的全局训练主要估计：

\[
\mathbb E[\ell_{\mathrm{IAH}}].
\]

最终部署却需要估计：

\[
\mathbb E[\Delta U(a)\mid c_t],
\qquad a\in\{I,D,U\},
\]

其中 (c_t) 是当前预测前可见的连续上下文，(Delta U(a)) 是执行动作相对 Identity 的整日价值。

这两个量不等价。CAVM 的作用是补齐“候选分布学习”和“动作价值选择”之间的条件化缺口，而不是替换 IAH-CRPS。

---

## 2. 总体架构：参数冻结，经验状态可更新

```text
冻结宿主预测器
      │ host forecast + legal history + optional pre-outcome features
      ▼
v0.4 Universal Core
  host-relative asinh → Data Signature/FiLM → IAH-CRPS
      │
      ├── Identity / Down / Up 三原子候选
      │       └── weighted_mean：点预测轨道
      │
      └── 连续上下文键 c_t
              │
              ▼
      CAVM 经验账本
      ├── Global evidence：源域离线经验
      └── Local evidence：目标域已揭示标签的滚动经验
              │
              ▼
      现有 query-dose replay / double-event / whole-day DVG
              │
              ▼
      动作价值估计 + 现有 LCB 门控
              │
              ▼
      执行 Down / Up，或返回 Identity
              │
              ▼
      标签揭示后，写入当日经验；预测前禁止读取当日标签
```

CAVM 不是第四层神经网络，而是 v0.4 Local Evidence 的一个可选增强实现：

```text
v0.4-core：W1 atom key + 原有 replay + 原有 DVG
v0.4-CAVM：连续 context key + 原有 replay + CAVM evidence + 原有 DVG
```

默认配置仍然是 `v0.4-core`。只有当 CAVM 在独立实验中稳定有效，才允许提升为论文主配置。

---

## 3. Universal Core：保持不变的部分

以下数学和训练组件不得因 CAVM 被改写：

- host-relative asinh 坐标；
- Identity / Down / Up 三原子；
- mass 是候选分布质量，不是动作概率；
- Down / Up dose 是双曲坐标中的连续位移；
- 单一 IAH-CRPS 训练目标；
- 查询剂量作用于历史宿主，而不是使用历史样本自己的 dose；
- 至多一个连续 Down 区间和一个连续 Up 区间；
- 整日 action value；
- S3-C 冻结后的 DVG 误差校准；
- LCB 大于零才执行，否则 Identity；
- universal trainer 的等域采样和宏平均 S2V checkpoint 选择。

不得为了 CAVM 增加：

- occurrence BCE；
- 额外 residual loss；
- MAE/CRPS 加权新目标；
- market ID / target ID predictive embedding；
- 固定的负价、低价或尖峰金额阈值；
- 正常/低/高三类硬切分；
- MoE 专家均衡 loss；
- RL、policy gradient 或需要在线反向传播的策略训练。

---

## 4. 连续上下文键 (c_t)

### 4.1 输入边界

上下文键只能使用预测发生前已经合法可见的信息：

- frozen host 的整日预测；
- 合法历史 target；
- 时间和周期信息；
- Data Signature；
- 可选外生变量及 availability mask；
- 当前 IAH 候选的原子摘要。

禁止使用：

- 当前日真实价格；
- 当前日 residual；
- 当前日 action gain；
- 由当前标签计算的事件类别；
- 任何未来回填值。

### 4.2 推荐的固定键结构

第一版不使用大型 Transformer 检索器，使用一个固定维度、可审计的连续键：

\[
c_t=[c_t^{shape},c_t^{dyn},c_t^{time},c_t^{sig},c_t^{atom},c_t^{opt}].
\]

建议组成：

| 子键 | 内容 | 作用 |
|---|---|---|
| `shape` | (z^0) 的均值、标准差、q10/q50/q90、IQR、绝对值均值 | 描述当前宿主价格形状和尺度 |
| `dyn` | 相邻差分均值/标准差、局部波动、变化方向质量 | 区分平稳、尖峰、急降和平台 |
| `time` | 现有 core context 中的日历/周期摘要 | 传递周期语义 |
| `sig` | Data Signature 描述子和 optional availability 摘要 | 处理不同数据可观测性 |
| `atom` | (w^-,w^+,m^-,m^+) 的日级统计 | 表示候选本身提出了什么修正 |
| `opt` | 按 role 聚合的均值/波动/缺失比例 | 兼容不同数量的外生变量 |

外生变量必须先按角色聚合；没有外生变量时使用全零值加 availability mask，不能把缺失数值误当成语义。

### 4.3 负价与低价的统一处理

模型内部不使用 `y<0` 的事件头。

对于有负价的市场，负价样本自然成为下尾经验；对于没有物理负价的市场，低于训练分布 q10 的样本仍可形成下尾经验。这样 Down 分支可以共享：

- 负价修正；
- 非负市场的低价修正；
- 局部剧烈下挫修正。

论文评价时再区分：

- `negative-price MAE`：仅在有足够真实负价样本的市场报告；
- `lower-tail MAE`：所有市场使用训练集经验下分位点报告。

---

## 5. CAVM 经验账本

### 5.1 经验记录

每个已经揭示真实标签的完整日样本保存：

\[
e_i=(c_i, R_i, y_i^{z}, V_i, \tau_i),
\]

其中：

- (c_i)：预测前上下文键；
- (R_i)：已有三原子 residual measure 和宿主 (z^0_i)；
- (y_i^z)：用于现有 query-dose replay 的真实标签坐标；
- (V_i)：动作价值、预测动作价值和校准误差；
- (	au_i)：时间戳及数据版本。

`market_id` 可以作为审计字段保存，但不能参与预测键距离。

### 5.2 全局与局部经验

经验分为两个命名空间：

```text
Global memory：冻结前由源市场 S3-M/S3-C 构建
Local memory ：目标市场标签逐日揭示后追加
```

它们的职责不同：

- Global memory 提供零样本/冷启动先验；
- Local memory 适应目标市场概念漂移；
- Local memory 不修改 universal 参数；
- 新数据不会因为规模大而自动获得更多梯度权重；
- 只有通过 leave-one-market-out 回放的经验，才允许离线晋升到 global memory。

### 5.3 第一版检索

第一版保留原有 top-k replay 结构，仅替换/扩展检索键：

\[
d_i(t)=
\lambda_{\mathrm{atom}}\tilde d_{W1}(R_t,R_i)
+
\lambda_{\mathrm{ctx}}d_c(c_t,c_i).
\]

其中：

- `W1-only` 是正式控制组；
- `context-only` 和 `context+atom` 是实验组；
- (lambda) 必须在 S2V/S3-M 选择并冻结，不得看 S4；
- 第一版仍选择 top-k，历史 replay 仍使用 query dose；
- 不先改为任意加权 replay，以免同时改变 v0.3 的距离、邻居和 replay 定义。

未来如需使用连续权重，可以作为独立扩展验证，不能偷偷混入主实验。

### 5.4 动作价值自校验

对每个已揭示日样本，记录：

\[
E_i=\widehat A_i-A_i.
\]

第一版 CAVM 只把它作为上下文经验统计，继续使用现有 DVG 的 (q) 和 LCB；不马上引入新的动态共形公式。

若后续实验显示静态 (q) 在概念漂移下明显滞后，再单独设计 adaptive conformal controller，并重新证明其适用边界。不能在没有证据时宣称在线分布无关保证。

---

## 6. 冻结语义和部署模式

### 6.1 严格冻结模式

- universal 参数冻结；
- local memory 在 S3 构建，S4 不更新；
- S4 只读历史经验；
- 用于零样本跨市场论文主表。

### 6.2 状态自适应模式

- universal 参数仍冻结；
- S4 每日预测前不读当天标签；
- 当天标签揭示后，才追加到 local memory；
- 用于概念漂移/在线自适应附加实验。

论文必须明确写成：

> parameter-frozen, state-adaptive correction

不能写成“完全冻结且持续学习参数”。

---

## 7. 与近期工作的边界

- RAFT 已经研究从历史样本检索未来模式，因此不能声称“首次使用时间序列记忆”。见 [RAFT, ICML 2025](https://proceedings.mlr.press/v267/han25d.html)。
- ORCA 已研究基于输入和宿主输出的上下文残差在线适配，因此不能声称“首次上下文在线修正”。见 [ORCA, 2026 preprint](https://arxiv.org/html/2606.14222v1)。
- TAFAS 已研究部分标签和门控校准，因此不能声称“首次测试时自适应”。见 [TAFAS, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33965)。
- Post-Training Corrections 已研究冻结预测器后的校正器选择，因此不能声称“首次后训练校正”。见 [Post-Training Corrections, 2026 preprint](https://arxiv.org/html/2505.15354v2)。

可争取的精确主张是：

> 在冻结宿主和冻结校正参数的条件下，利用预测前连续上下文检索历史三原子动作证据，将“校正候选生成”和“整日动作价值安全选择”连接起来，并允许高/低方向共享同一经验状态。

该主张仍需完成专项 novelty collision check，不能现在写成“首次”。

---

## 8. 必须保留的安全回退

以下任一情况必须返回 Identity：

- 没有可用 memory；
- 没有 S3-C DVG 校准；
- 有效相似经验不足；
- 估计动作价值的 LCB 不大于零；
- 输入信息契约不完整；
- bundle 版本或 hash 不匹配。

CAVM 的目标不是强迫更多动作被执行，而是让“该执行时执行，不该执行时 abstain”更可靠。

---

## 9. 技术架构验收标准

只有以下条件全部通过，CAVM 才能进入论文主实验：

1. `v0.4-core` 结果可以完全复现；
2. CAVM 不改变 IAH-CRPS 数学和候选输出；
3. query 之前没有读取 query target；
4. context key 对 target/residual/action gain 具有严格独立性；
5. market ID 只进入审计，不进入 key；
6. global/local memory 可以分别冻结、序列更新和恢复；
7. bundle round-trip 可以复现 key、邻居、proposal、A_hat、q、LCB 和 final output；
8. CAVM 至少在两个不同类型市场和三个 seed 上稳定优于 W1-only 或不产生可重复负迁移；
9. 如果仅在线 local memory 有效，论文主张必须降级为概念漂移适应，而不是零样本通用性。
