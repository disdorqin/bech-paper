# P1 Round-3 · T4 假设文档 — 重定标的软权重采样(先写假设再跑)

- 日期: 2026-08-14
- 方向: T4「连续模式覆盖采样 / 软采样权重」
- 触发: T2 之后"域失衡"轴被数据否决 → 按 §5.3 协议,正式跑之前先写清目标函数与假设。
- 状态: **假设待验(未跑)**

## 1. T2 教训(本假设的锚点)

T2(equal → full_coverage)配对 3/3 seed 一致退化,逐域拆解给出反直觉结论:

- 去掉 NEM 的 5.5× 重复曝光后,DE/PJM 8/8 域只小幅改善,而 **NEM 3/4 域系统性退化**,退化量大于长域改善 → 主导宏平均。
- 含义:**低数据量域(NEM)的重复曝光不是过拟合噪声,而是该域的真实梯度需求**。T0 等域采样(每域 K=22 update/epoch)在短域上的重复是必要正则。

## 2. 假设声明

> **H4**: 在等域采样下,低数据量 / 高难度域(如 NEM)的单步梯度信号被低估——它们值得的梯度步数超过"与其他域等量"的水平。若把每个训练域的软权重设为**预测时可见的宿主侧难度统计**(scale-free 波动率)的单调函数,保持总预算 = 12×K 不变(只改分配,不改总步数),应当同时:
> (a) 提升低难度长域(DE/PJM)的收敛质量(不伤害它们),
> (b) 提升高难度短域(NEM)的 S2V 表现,
> 从而在 S2V 宏平均 CRPS 上优于 T0 等域基线。

**方向判断**: T2 证明"均衡化(每域按其数据量权重)"方向错误;H4 与 T2 相反——**给短/难域更多而非更少的梯度**。这是 T2 的教训不是 T2 的重复。

## 3. 权重定义(`weights_from_host_stats`)

对每个训练域 g,权重 w_g 由三类统计之一计算(全部 **S1R 段**、预测时可见,与 S2V 无关):

| 模式 | 公式 | 目标泄漏类型 | 说明 |
|---|---|---|---|
| `volatility` | w_g ∝ std(z0 over S1R) | **target-free** | 宿主 scale-free 预测的波动率;只用宿主输出(修正头输入层可见),不读任何目标价格 |
| `host_s1r_mae` | w_g ∝ mean\|asinh(y/scale) − z0\| over S1R | **training-only**(§5.2) | 宿主在 S1R 的 scale-free 误差;读取 S1R 目标价格。必须配 target-free 对照并显式标注 |
| `inv_nbatch` | w_g ∝ 1/N_g(S2T batch 数) | target-free | 纯数据量反比(方向 = T2 反方向,验证"短域需更多"这一机制) |

归一化: w ← w / mean(w),使 mean(w)≈1 → 每 epoch 总 update = n_domains×K 不变(预算与 T0 相同,只改分配)。实现走 `UniversalCoreTrainer.train(weights=w, sampling="equal")` 已有的 Case C 温度采样。

## 4. 对照设计(配对)+ 权重行为预检(2026-08-14 已完成)

预检:`weights_from_host_stats` 在 12 源域上计算三类权重,验证它们是否实现"给 NEM 这类短/难域更多梯度"这一机制:

| 模式 | 实测 per-market 权重 | 行为判定 |
|---|---|---|
| `volatility` | DE[0.95–1.47], PJM[0.83–1.05], NEM[0.58–1.72] | **不干净**:NEM PatchTST 仅 0.58,跨 host 方向不一致 |
| `host_s1r_mae` | DE[0.39–0.55], PJM[0.35–0.63], **NEM[1.69–2.86]** | **干净**:NEM 全部 4 host 上调,DE/PJM 下调 → 精准实现 T2 机制 |
| `inv_nbatch` | DE/PJM 0.4×, NEM 2.2× | **干净**:纯数据量反比(T2 反方向),target-free |

→ **主假设 arm 调整为 `host_s1r_mae`**(最能实现"难域更多梯度");target-free 对照用 `inv_nbatch`(机制隔离:是数据量不是难度);`volatility` 降级为可选第三 probe(预检已显示它可能不移动 NEM,跑不跑由第 5 节判定决定)。

| 组 | 采样 | 权重 | 目标泄漏类型 | 用途 |
|---|---|---|---|---|
| **T0(对照)** | equal | 无 | — | 基线(已存在,3 seed) |
| **T4-A** | weighted(host_s1r_mae) | training-only(显式标注) | 主假设 |
| **T4-B** | weighted(inv_nbatch) | target-free | target-free 对照:确认"是数据量不是难度" |
| **T4-C**(可选) | weighted(volatility) | target-free | 难度波动率 probe(预检噪声大,仅当 A/B 有意向时补跑) |

- 每 arm 3 seeds,训练域 = 12 国外源域(与 T0 相同,不动主配置)。
- S2V 宏平均 CRPS 配对 vs T0,协议同 T2(同 trainer、同 epochs、同 seed、同 best_epoch 规则)。
- §5.2 合规:T4-A 因读 S1R 目标价格属 training-only,报告逐条标注;T4-B 为其 target-free 对照。

## 5. 判定阈值

- **KEEP**: 3/3 seed 宏平均 ΔCRPS < 0,且 NEM 至少 2/4 域改善、DE/PJM 不退化(逐域)。
- **INCONCLUSIVE**: ΔCRPS < 0 但 seed 不一致,或只有某 arm 成立。
- **REJECT**: 任一 arm 与 T0 无差异或更差(与 T2 相同:机制证据优先于显著性包装)。

## 6. 红线(全程)

- 不做 S4 调参:S2V 只用于 checkpoint 选择与配对主指标;S4 仅冻结确认(若第三轮 transfer 需要)。
- 权重只用 S1R 统计;`host_s1r_mae` 属 training-only,报告中逐条标注。
- 总预算保持 12×K;不新建损失/头;特征仍只经 host 进入(不复活 T6)。
- 若 mixed/negative,如实保留,不冒充泛化。

## 7. 与 D1/D2 的关系

- D2(国内基准)独立进行:冻结 round-1 head 评估 10 国内模式 × 4 host,与 T4 训练无关。
- T4 不因 D2 结果改变假设;若 D2 揭示国内域特殊性,可作为后续第三轮 transfer 的输入,不改本轮设计。
