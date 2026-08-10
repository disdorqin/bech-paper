# 调研报告：软门控 / 概率加权混合方法 (Soft Gating)

> 日期：2026-08-09 | 目的：借鉴 Vahedi 软混合范式，探索可融入 HCH 的门控机制

---

## 核心发现

### Vahedi 的 (1-p)·base + p·corrected 是真正的创新

这个公式在 selective prediction / correction 文献中**几乎没有完全一致的先例**。
Selective prediction 社区以硬门控为主流（SelectiveNet, Conformal Selective Regression），
软门控主要来自 **MoE (Mixture of Experts)** 社区。

### 两社区对比

| | Selective Prediction | MoE |
|---|---|---|
| 门控类型 | **硬门控**：阈值判断 abstain/act | **软路由**：连续权重分配 |
| 代表 | SelectiveNet (ICML 2019) | Soft MoE (NeurIPS 2023), SMEAR |
| 适用场景 | 安全关键（宁可弃权也不错） | 精度优先（组合专家提升性能） |

---

## 最相关 5 篇

| # | 论文 | 年份 | 核心方法 | 与 HCH 的关系 |
|---|---|---|---|---|
| 1 | **MoGU** | 2025 | 用预测方差(uncertainty)作为门控信号加权混合专家 | **最接近**：把 Vahedi 的 p(neg) 换成 σ² → 直接对应 HCH 的 SCARR bootstrap LCB |
| 2 | **CSGE** | 2018 | 能源预测领域软门控先驱，多模型竞争+协作软加权 | 能源 + 软门控 lineage |
| 3 | **SMEAR** | 2023 (TMLR 2024) | 软合并专家：所有专家参数加权平均 → 避开离散路由不可微问题 | 软合并 ≈ HCH 的 λ 机制 |
| 4 | **Sigmoid > Softmax** | 2024 | 理论证明 sigmoid gating 样本效率优于 softmax | HCH per-branch sigmoid gate 的理论支撑 |
| 5 | **UGGM** | 2026 | Uncertainty 三级门控：重参数化→传播→生成 | uncertainty→modulate correction 的完整 pipeline |

---

## 可融入 HCH 的具体方向

### 方向 1: 软门控替代硬 τ 阈值 (MoGU 启发)

```
旧: if P(neg|Z) > 0.5 and λ > 0: ŷ = ŷ_base + λ·δ
新: ŷ = (1 - w)·ŷ_base + w·(ŷ_base + δ)
     w = σ(P(neg|Z) - 0.5)  # sigmoid 软化，无硬阈值
     SCARR 调 λ: ŷ = ŷ_base + λ·w·δ
```

**效果预估**：去除 τ 阈值 → 30% 被挡的真正负价事件可以参与校正。
SCARR λ 仍提供安全控制。

### 方向 2: Uncertainty 调制的 λ (UGGM 启发)

```
λ 不仅由 SCARR 全局决定，还由 per-sample uncertainty 调制：
λ(sample) = λ_global · (1 - tanh(u/σ))
```
uncertainty 高 → λ 小（不确定则少改），uncertainty 低 → λ 大（自信则多改）。

---

## 总结

| 方向 | 来源 | 复杂度 | 预期增益 |
|---|---|---|---|
| 软门控替代 τ | MoGU / Vahedi | 低（改一行） | **高**（解除 30% 事件被挡） |
| Uncertainty 调制 λ | UGGM | 中（需 uncertainty estimate） | 中 |
| Sigmoid gating 理论 | Sigmoid > Softmax | 零（已是 sigmoid） | 引用即可 |
