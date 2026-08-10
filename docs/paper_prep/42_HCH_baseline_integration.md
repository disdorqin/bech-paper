# 基线方法融入 HCH + 对比实验设计

> 日期：2026-08-09

---

## 一、可融入 HCH 的创新点

### 来自 Vahedi — 已内化，无需改

| Vahedi 机制 | HCH 对应 | 是否已有 |
|---|---|---|
| 二阶段分类→回归 | P(event)×条件幅度 | ✅ 已有 |
| 负价专用训练 | neg head + neg regressor | ✅ 已有 |
| p_neg 加权混合预测 | 已内化为 Hurdle gate + λ | ✅ 已有 |

**HCH 相比 Vahedi 的差异**（论文卖点）：冻结基座外挂 vs 直接预测、选择性触发 vs 总是输出 hybrid、λ 证书弃权 vs 无安全机制、恒等回退 vs 无 fallback。

---

### 来自 CRC — ⭐ 逐点 λ + NDR 指标

| CRC 机制 | 改造后 HCH | 优先级 |
|---|---|---|
| **逐点选择**：每个 (变量, 预测步) 独立选 linear vs hybrid | **逐时段 λ**：当前按天全局标定 1 个 λ → 改为每天 24 小时各自标定 λ_h，在 S3 上 per-hour 选择 | **P0** |
| **NDR 指标**：非退化率 = 校正后不差于基座的比例 | 加入评估体系，报告 HCH 的 NDR | **P0** |

**逐时段 λ 设计**：
```python
# 当前: calibrate() → 1 个 λ 用于全天
# 改为: calibrate() → 24 个 λ_h, h=0..23
# S3 上对每个小时段独立做 bootstrap LCB + safety quantile
# 夜间的 λ 可能更大（需求低、校正空间大），午间 λ 可能更小（光伏峰值、基座已好）
```

---

### 来自 PIR — ⭐ 历史 episode 检索

| PIR 机制 | 改造后 HCH | 优先级 |
|---|---|---|
| **全局检索**：cosine similarity 找相似历史序列 | **历史 episode 检索**：对当前预测日，从训练集中检索 K 个最相似的历史负价 episode → 作为幅度头的额外特征 | **P1** |
| **不确定性估计 δ** | **调制 λ**：δ 高 → λ 放大（更多校正），δ 低 → λ 收缩 | **P1** |

---

### 来自 SpikeReg — ⭐ 尖峰加权训练

| SpikeReg 机制 | 改造后 HCH | 优先级 |
|---|---|---|
| spike penalty：大误差样本加权 | **幅度头加权**：neg 样本 5x 权重、spike 样本 3x 权重 → 幅度头更关注尾部 | **P0** |

---

### 优先级汇总

| 优先级 | 来源 | 改进 | 代码改动 |
|---|---|---|---|
| **P0** | CRC | 逐时段 λ (24 个/天) | `calibrate()` |
| **P0** | CRC | NDR 指标 | `evaluate()` |
| **P0** | SpikeReg | 幅度头 spike-weighted 训练 | `_fit_reg()` 加 sample_weight |
| P1 | PIR | 历史 episode 检索 | `build_corrector_features()` 加 retrieval |

---

## 二、对比实验设计

### 数据集

| 实验 | 数据集 | 基座 | 对比 |
|---|---|---|---|
| **山东主实验** | 山东 hourly | 5 基座 | Base vs HCH vs Vahedi vs Quantile vs PIR |
| **公开复现** | NEM SA1 + LAGO_DE | Linear + GBDT | 同上 |

### 指标矩阵

| 类 | 指标 |
|---|---|
| 点级 | MAE、RMSE、neg_miss_rate、spike_miss_rate |
| Episode 级 | episode_recall、complete_miss、false_events、boundary_mae |
| 安全 | fire_rate、λ_neg、λ_pos、NDR、normal_harm |

### 实验表模板（论文主表）

```
Table: Shandong × 5 backbones

          MAE↓   EP_Recall↑   Neg_Miss↓   Complete_Miss↓   NDR↑   Fire%
Base      xxx    xxx          xxx         xxx              —      —
Vahedi    xxx    xxx          xxx         xxx              xxx    xxx
Quantile  xxx    xxx          xxx         xxx              xxx    xxx
PIR       xxx    xxx          xxx         xxx              xxx    xxx
HCH       xxx    xxx          xxx         xxx              xxx    xxx
```

---

## 三、执行顺序

1. **P0 改进落地**（逐时段 λ + NDR + spike-weight）
2. **山东全基线对比**（5 基座 × 5 方法 = 25 组合）
3. **公开全基线对比**（NEM SA1 + LAGO_DE × 2 基座 × 5 方法 = 20 组合）
4. **图表生成**（主表 + 消融 + episode curve）
