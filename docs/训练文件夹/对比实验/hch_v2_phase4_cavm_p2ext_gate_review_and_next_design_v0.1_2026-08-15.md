# HCH-v2 Phase 4：P2 扩展门禁复核与下一步设计 v0.1

**日期：** 2026-08-15

**依据：**

- `hch_v2_phase4_cavm_p2ext_report_v0.1_2026-08-15.md`
- P2 扩展实验：5 市场 × 3 host × 3 seed，共 45 cell
- 当前主数学：`hch_v2_iah_crps_final_math_core_v0.3`
- 当前架构：v0.4 Universal Core / Data Signature / Local Evidence

**重要状态：**

本报告只做科研复核和下一步设计，不修改生产代码，不放行 P4。

---

## 1. 正式判定

### 1.1 P2 扩展的判定有效

本轮应当接受报告给出的：

```text
GATE_NOT_YET_PASS
```

原因不是实验不完整，而是实验已经发现：

- context retrieval 在 LAGO_DE 上有 3 seed 可复现收益；
- LAGO_NP 没有稳定判别力；
- Shandong 的效果高度依赖 host；
- Shaanxi/Gansu 基本没有可执行动作；
- 只有一个市场类型达到“方向稳定改善”。

因此现在不能把 CAVM 写成跨市场普适增益，更不能进入 P4 action-value state update。

### 1.2 但不能把它简单写成“CAVM 失败”

本轮还有一个决定性边界：

> E0–E3 的候选头是每个市场自己训练的，不是论文主配置的冻结 universal core。

所以当前实验实际回答的是：

> “在每个市场已经拥有自己的候选头时，CAVM 检索键是否普遍有益？”

它还没有回答论文真正的核心问题：

> “一个在多源市场上训练后冻结的 universal correction core，能否通过 CAVM 在未见市场上稳定工作？”

因此，下一步首要任务不是继续改检索键，而是补齐 paper-config 实验。

---

## 2. 从结果中分离四种问题

当前报告把多个现象放在同一个“检索是否有效”问题下，需要拆开：

| 现象 | 更可能的根因 | 当前证据强度 | 下一步 |
|---|---|---:|---|
| LAGO_DE context retrieval 有效 | 邻居排序改善，候选有容量，动作链可执行 | 高 | 保留为正例 |
| Shandong PatchTST 改善、Linear 退化 | host capacity / candidate support 与检索发生交互 | 中高 | 做候选容量分解 |
| Shaanxi/Gansu action-empty | 候选没产生有效 proposal，或 DVG 过于保守，或样本太薄 | 中 | 区分三者，不能叫 retrieval 失败 |
| LAGO_NP 无判别力 | 温和市场 context 差异不足，或校正收益本身接近零 | 中高 | 保留为 negative/neutral case |
| E1==E0 | CAVM λ=(1,0) 兼容契约成立 | 高 | 作为代码正确性证据 |
| 17/45 point/action opposite | 部分是 action-empty 或毫幅动作噪声 | 中高 | 使用执行条件指标重算，不重写事实 |

必须新增一个动作链分解：

```text
candidate alive
    → directional proposal non-empty
    → predicted A_hat > 0
    → LCB > 0
    → final action non-empty
    → realized action gain
```

只有最后一步失败，才能称为动作价值/检索方法失败；前面某一步就为空时，应归为候选容量或校准问题。

---

## 3. 当前绝对不做的事

### 3.1 不放行 P4

P4 是 action-value state update。当前只有 LAGO_DE 达到稳定市场级收益，不能把单一市场的成功升级为普适机制。

### 3.2 不立即做“按市场/host ID 门控”

报告提出的 A 方向有价值，但不能直接实现成：

```text
if market == Shandong: lambda_ctx = ...
if host == PatchTST: lambda_ctx = ...
```

这会：

- 违反 universal core 的身份无关原则；
- 把当前 45-cell 结果过拟合成规则；
- 无法泛化到新市场和新宿主；
- 让 CAVM 变成一个隐藏的 market/host classifier。

以后若做条件化，只能使用预测前连续的“证据可靠度”或“上下文相似度”，不能使用市场/宿主身份。

### 3.3 不改 loss 和候选数学

当前证据尚未证明 IAH-CRPS 或三原子候选本身错误。修改 loss 会同时改变候选、检索和 action value，反而无法定位原因。

---

## 4. 下一条主线：Paper-config Universal Core Confirmation

### 4.1 目标

训练真正的共享候选头：

```text
12 international source domains
× host domains
→ one shared universal candidate core
→ freeze
→ unseen domestic / international target evaluation
```

目标市场只能使用：

- 合法 S1 reference；
- target-local S3-M memory；
- target-local S3-C DVG；
- 不更新 universal core 参数。

### 4.2 训练方式

使用现有 `UniversalCoreTrainer`：

- domain unit = `(market, target/mode, host)`；
- 等域采样；
- 只有 IAH-CRPS；
- macro S2V checkpoint selection；
- 每个 batch 显式传递对应 `domain_det`；
- 不使用单域 `fit_s1_signature()` buffer 污染混合域训练。

候选头必须是同一个共享实例，而不是每个目标市场重新训练一份。

### 4.3 Paper-config 小矩阵

先做快速筛选：

| 维度 | 设定 |
|---|---|
| source | 12 国际域 |
| target | LAGO_DE、LAGO_NP、Shandong_DA |
| host | Linear、MLP、PatchTST |
| seed | 0/1/2 |
| retrieval | W1-only、context-only、context+atom |
| 总 cell | 27 个 target cells × retrieval arms |

只有快速筛选显示 universal core 不会完全坍塌，才扩展到：

- Shaanxi_RT；
- Gansu_DA；
- Qinghai/Ningxia；
- 完整 45-cell。

### 4.4 Data Signature 必须显式核对

当前仓库同时存在：

- S1R 确定性 descriptor；
- learned per-day signature；
- 单域 `domain_det` buffer 便利路径。

在 paper-config 实验前，必须生成一份配置审计表，明确：

1. 哪些 descriptor 实际参与 universal training；
2. 哪些 descriptor 在 target inference 中可用；
3. 是否使用 zero descriptor；
4. learned signature 是否真的经过训练；
5. 是否存在某个市场 descriptor 被错误冻结到所有 batch。

如果当前所谓“paper-config”实际上用了每市场 descriptor 或每市场 candidate head，必须明确标为变体，不能称为 universal core。

---

## 5. 必须同步完成的“动作容量分解”

在 universal core 评估中，每个 cell 必须额外输出四个离线诊断，不改变正式 S4 结果。

### C0：候选支持

对每个 query day 记录：

- `frac_m_minus_alive`；
- `frac_m_plus_alive`；
- mean/p95 dose；
- candidate weighted_mean 相对 host 的 oracle 点误差。

如果 candidate 本身没有有效位移，CAVM 不可能产生最终收益。

### C1：proposal 支持

不应用 LCB，记录：

- `proposal_empty_rate`；
- Down/Up interval length；
- directional (\widehat g_{down}, \widehat g_{up})；
- proposal total value。

### C2：DVG 门控

比较：

```text
proposal without LCB
vs
current DVG + LCB
```

这只是离线故障定位，不是正式安全结果。

如果 proposal 有收益但 LCB 大量拒绝，问题是 action calibration；如果 proposal 自身没有收益，问题在候选或检索。

### C3：实现收益

只在真正执行的 action 上报告：

- `exec_count`；
- `exec_mean_A_true`；
- execute-conditioned harm rate；
- final point MAE delta；
- normal-regime degradation。

`mean_A_true` 在 action-empty 大量存在时只能做辅助统计，不能作为动作改善主证据。

---

## 6. Paper-config 之后的分支决策

### 情况 A：Universal core 使至少两类市场稳定改善

条件：

- 至少两个市场类型达到方向稳定；
- LAGO_NP 不出现实质退化；
- 不是单一 host 或单一 seed 贡献；
- action-capacity 分解显示改善确实来自检索/动作链。

决策：

1. 放行 CAVM 进入下一轮；
2. 重新设计一个不依赖 market/host ID 的连续 evidence reliability；
3. 再考虑 P4 action-value state update。

### 情况 B：Universal core 候选有容量，但 LCB 过度保守

表现：

- candidate alive；
- proposal non-empty；
- no-LCB 有收益；
- static LCB 执行率极低。

决策：

- 不立即改 loss；
- 做 S3-C DVG 样本量、误差分布和 q 稳定性审计；
- 设计“状态校准”实验，但暂不宣称正式 P4 通过。

### 情况 C：Universal core 候选支持不足

表现：

- Down/Up dose 长期接近零；
- proposal 几乎为空；
- oracle candidate 也无法改善 host。

决策：

- CAVM 不是主要问题；
- 回到 IAH candidate representation / universal training；
- 重新检查跨域 CRPS 是否让候选平均化；
- 必要时研究 host-relative candidate geometry，而不是继续调检索。

### 情况 D：仍然只有 LAGO_DE 稳定改善

决策：

- CAVM 不能作为 universal 主创新；
- 可以如实收束为“极端市场条件下的动作证据路由”；
- 但这不足以支撑跨市场 SOTA 主张；
- 需要重新寻找能够覆盖温和市场和强宿主的统一机制。

---

## 7. 后续条件化检索的正确形式

只有情况 A 或 B 成立后，才考虑条件化检索。

不使用 market/host ID，而使用连续可靠度：

\[
\lambda_{ctx}(c_t)
=
\sigma\left(
g(\text{context},\text{candidate support},\text{effective evidence},\text{historical error})
\right).
\]

其中：

- `context`：预测前上下文；
- `candidate support`：候选 dose/mass 健康统计；
- `effective evidence`：检索经验数量和距离；
- `historical error`：过去相似上下文的 action-value error。

该量应该是连续的，不是“某市场启用/禁用”。

第一版甚至可以不训练一个新网络，而采用 S3-M 上冻结的收缩估计：

```text
证据弱 → 更依赖 global/W1
证据强 → 才增加 context retrieval 的作用
```

但这属于下一阶段，不得在当前 GATE_NOT_YET_PASS 时直接加入。

---

## 8. 文献对当前判断的支持

近期跨域时间序列研究明确指出，异质数据直接混合会产生 domain conflict 和 negative transfer；ContexTST 用统一表示加上下文锚点缓解这一问题。这与 Phase3 T5 的 LAGO_NP 污染现象一致，但也说明“有 context key”本身并不自动解决跨域语义错位。[ContexTST, 2025](https://arxiv.org/html/2503.01157v1)

变化点条件下的 conformal 研究也显示，普通在线校准在 shift 发生后存在反应延迟，状态感知可以提前调整不确定性；这支持后续研究 state-adaptive DVG，但不能在当前 action signal 尚未稳定时直接引入新的共形控制器。[CPTC, NeurIPS 2025](https://arxiv.org/html/2509.02844v1)

因此当前最科学的顺序是：

```text
先确认 universal core
→ 再分解 candidate / proposal / LCB 的故障位置
→ 再决定 CAVM 是否需要连续可靠度门控
→ 最后才研究 P4 状态校准
```

---

## 9. 交给代码 Agent 的下一阶段任务边界

代码 Agent 下一轮只做以下内容：

1. 核对并报告 `99f75c5` 的真实代码状态；若提交未推送，先说明无法从公开仓库复核；
2. 不改变现有 E0–E3 结果；
3. 实现/确认 paper-config `UniversalCoreTrainer`；
4. 生成 universal core 的 candidate/proposal/LCB 分解日志；
5. 跑 27-cell 快速筛选；
6. 输出三 seed 收敛曲线、candidate alive、proposal empty、execute-conditioned metrics；
7. 不实现 P4；
8. 不新增 loss、事件头、market/host ID gate 或硬阈值；
9. 返回是否进入情况 A/B/C/D。

---

## 10. 当前一句话结论

> P2 扩展证明了 CAVM 在部分极端市场和有足够候选容量的宿主上能改善动作结果，但没有证明跨市场普适性；下一步必须先用真正的冻结 universal core 做确认，并把“检索失败、候选无容量、LCB 过保守”三类问题分离，之后才有资格继续设计 P4。

