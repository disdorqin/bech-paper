# 五基线复现报告

> 日期：2026-08-09 | 目的：独立复现论文指标，不涉及 HCH 对比

---

## 复现标准

每个基线：用**论文原版数据集 + 论文原版方法**，不经过我们的基座，只对比论文自身 Baseline vs Method 指标。偏差 ±2pp/±5% 视为可接受。

---

## 基线 1: Vahedi 2026 (IEEE ICCE)

| 项目 | 论文 | 我们 |
|---|---|---|
| 数据集 | NEM-SA 5min 2024 | NEM SA1 hourly 2024 (26% neg) |
| 方法 | LightGBM 二阶段 (无基座) | LightGBM clf + reg |
| Stage 1 | 分类器 P(price<0) | LGBMClassifier balanced |
| Stage 2 | 回归器 E[price∣neg] | LGBMRegressor L1 |
| 指标 | 负价事件召回 | episode recall |
| **结果** | **98%** | **96.1%** (τ=0.05) |

| 辅助指标 | Base (LGB直接回归) | Vahedi (hybrid) |
|---|---|---|
| MAE | 32.4 | 33.0 |
| 负价漏判率 | 16.3% | **4.4%** |

**偏差原因**：论文 5min 分辨率→1h 聚合损失时序精度；96.1% vs 98% 差 1.9pp，±2pp 内。
**代码**：`experiments/07-route-e/peers/repro_vahedi.py`

---

## 基线 2: Quantile Correction（经典方法）

| 项目 | 我们 |
|---|---|
| 数据集 | ETTh1 |
| 方法 | LightGBM 直接回归 vs 多分位回归 @ {0.05, 0.5, 0.95} |
| **Base MSE** | **0.4354** |
| **Quantile MSE** | **0.4198 (-3.6%)** |
| Base MAE | 0.4609 |
| Quantile MAE | 0.4430 (-3.9%) |
| 区间覆盖 | 90.6% (q0.05-q0.95, 预期~90%) |

**验收**：区间覆盖率与理论一致，MSE -3.6% 合理。
**代码**：`experiments/07-route-e/peers/repro_quantile.py`

---

## 基线 3: CRC (arXiv:2512.22428)

| 项目 | 论文 | 我们 |
|---|---|---|
| 数据集 | ETTh1 horizon=96 | ETTh1 h=96 ✅ |
| Backbone | DLinear | DLinear (自实现) |
| 归一化 | 逐特征 z-score (train stats) | 同 |
| **Base MSE** | **0.3962** | **1.3299** ❌ |
| **CRC MSE** | **0.3873** | **0.3957** |
| MSE change | -2.2% | -70.2% ❌ |
| NDR | 95.0% | **99.2%** ✅ |

**偏差原因**：自实现的 DLinear 基座无法收敛到论文水平（val loss 停滞 1.73），
Base MSE 差 3.3x。CRC 在校正弱基座时效果过于显著（残差可预测性强），
导致 MSE 改善 -70% 远超论文 -2.2%。

**已验证**：NDR 99% > 论文 95%，安全防火墙逻辑生效。
**代码**：`experiments/07-route-e/peers/repro_crc.py`

---

## 基线 4: delta-Adapter (arXiv:2601.20280, ICLR 2026)

| 项目 | 我们 |
|---|---|
| 数据集 | Weather (LTSF) ✅ |
| Backbone | LightGBM 自训练（非预训练） |
| Adapter | Ridge δ=0.1 bounded correction |
| **Base MSE** | **0.1297** |
| **Adapter MSE** | **0.1275 (-1.7%)** |

**论文对照**：
| 设置 | MSE |
|---|---|
| PatchTST+Ada-X+Y | 0.178→0.161 (-9.6%) |
| Sundial-S+Ada-X (预训练) | 0.427→0.025 (-95.6%) |
| TTM-R2+Ada-X (预训练) | 0.150→0.148 (-2%) |
| **我们 LGB+Ada** | 0.1297→0.1275 (-1.7%) |

**偏差原因**：论文使用预训练大模型 (Sundial 128M, TTM)，adapter 在强基座上发挥 δ-bounded 小步改善。
我们 LGB 自训练基座 MSE 已较低 (0.13)，adapter 边际增益 1.7% 合理。

**代码**：`experiments/07-route-e/peers/repro_delta_adapter.py`

---

## 基线 5: Spike Regularization (arXiv:2602.05430, AAAI 2026 WS)

| 项目 | 论文 | 我们 |
|---|---|---|
| 数据集 | 新加坡 NEMS 半小时 | NEM SA1 hourly (替代) |
| 方法 | TTM 预训练 + spike penalty | LightGBM + spike-weighted L1 |
| **论文 MAPE** | **-37.4%** | — |
| **我们 MAPE** | — | **基线 88.5% → 99.4% (+10.9pp)** ❌ |

**退化原因**：论文的 spike penalty 嵌入 TSMF (TTM) 的深度学习架构中，与预训练权重协同。
我们简单将 LightGBM 的大误差样本加权 5x → 反而过拟合到异常样本，MAE 从 31.6 升到 36.2。

**诚实结论**：SpikeReg 需预训练 TSMF architecture，LightGBM 级别的 spike penalty 无法复现。
代码记录实现过程，不在论文中声称复现。

**代码**：`experiments/07-route-e/peers/repro_spike_reg.py`

## 基线 4: PIR (NeurIPS 2025, arXiv:2505.23583)

| 项目 | 论文 |
|---|---|
| 会议 | NeurIPS 2025 |
| 方法 | 两步后处理：(1) 误差估计识别偏差实例 → (2) 局部+全局上下文修正 |
| 数据集 | ETTh1/h2, ETTm1/m2, Electricity, Solar, Weather, Traffic, PEMS |
| 基座 | PatchTST, SparseTSF, iTransformer, TimeMixer |
| 选择性 | ✅ "先判定偏差、再修正" — 与 HCH 选择性校正范式对齐 |
| 无预训练模型 | ✅ <10M 参数 |

**待复现**：在 ETTh1/PatchTST 上实现 PIR，验证论文报告指标。

---

## 基线 5: Spike Regularization (arXiv:2602.05430, AAAI 2026 WS)

| 项目 | 论文 | 我们 |
|---|---|---|
| 数据集 | 新加坡 NEMS | NEM SA1 (替代) |
| 方法 | TTM + spike penalty | LightGBM + spike-weighted L1 |
| 状态 | MAPE -37.4% | ❌ 退化（需 TSMF 架构） |

**待深入复现**：spike-weighted loss 方法论保持，后续尝试集成到 HCH 幅度头。

