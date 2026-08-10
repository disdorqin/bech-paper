# 同行基线复现记录

> 日期：2026-08-09 | 项目：Route-E 应用论文

---

## 复现标准
每个基线验证：1 数据集 × 1 基座 × 1 指标与论文一致即可。

---

## CRC (arXiv:2512.22428)

| 项 | 论文 | 我们 |
|---|---|---|
| 数据集 | ETTh1 (7集LTSF) | ETTh1 ✅ |
| 基座 | DLinear / PatchTST / TimeXer | Linear |
| 指标 | MSE | MSE |
| **论文值** | DLinear ETTh1 h=96: 0.3962→0.3873 **(-2.2%)** | — |
| **我们** | Linear ETTh1: 0.4181→0.5251 **(+25.6%)** ❌ | — |
| Weather | 论文: PatchTST h=96: 0.1752→0.1616 **(-7.8%)** | Linear Weather: 9.73→12.77 **(+31.3%)** ❌ |

**偏差原因**：简化版无因果编码器(adjacency graph + multi-scale prior + direction gating)。论文的 NDR 95% 完全依赖该 encoder，我们只用 Ridge+LGB → 过拟合。

**判定**：⚠️ 架构简化过多，无法达到论文增益。论文使用因果编码器是关键，将如实记录偏差而非硬凑。

---

## PIR (arXiv:2601.20280, ICLR 2026)

| 项 | 论文 | 我们 |
|---|---|---|
| 数据集 | Weather / ELC / Traffic | ETTh1 / Weather ✅ |
| 基座 | Sundial-S / iTransformer / PatchTST | Linear |
| 指标 | MSE | MSE |
| **论文值** | Weather PatchTST: 0.178→0.161 **(-9.6%)** | — |
| **我们** | ETTh1 Linear: 0.4181→0.4248 **(+1.6%)** ⚠️ | — |
| | Weather Linear: 9.73→9.89 **(+1.7%)** ⚠️ | — |

**偏差原因**：论文用强预训练基座(Sundial 128M)，我们 Linear ≈ sklearn Ridge(1.0)。论文 δ=0.1 输入微调作用于 Transformer/GNN 级别特征空间→效果显著；Linear 无 hidden representation 可利用。

**判定**：⚠️ 基座能力差一个量级，输入端 adapter 对 Linear 无增益属预期。

---

## Spike Regularization (arXiv:2602.05430, AAAI 2026 WS)

| 项 | 论文 | 我们 |
|---|---|---|
| 数据集 | 新加坡 NEMS 半小时 | 不可获取 → ETTh1 |
| 基座 | TTM / MOIRAI / LSTM | Linear |
| 指标 | MAPE | MSE |
| **论文值** | TTM 37.4% MAPE 改善 | 无直接对照 |
| **我们** | ETTh1 Linear MAE: 0.447→0.361 **(-19%)** ✅ 但 MSE +29% | — |

**偏差原因**：新加坡数据不可获取。ETTh1 无真实电价尖峰，spike detection 在无事件场景退化为普通加权 L1 回归 → MAE 改善但 MSE 恶化。

**判定**：⚠️ 数据集不可获取 + 无尖峰场景 → 不纳入正式实验对照。记录方法实现。

---

## Vahedi-style (IEEE ICCE 2026)

| 项 | 论文 | 我们 |
|---|---|---|---|
| 数据集 | NEM-SA 5min (2024) | NEM SA1 小时级 2024 ✅ |
| 基座 | LightGBM (直接预测) | Linear (基座→分类→回归) |
| 指标 | 负价事件召回 98% | 负价 episode 召回率 |
| **论文值** | 98% neg-event recall | — |
| **我们** | — | MAE 69.8→**62.8 (-10.0%)**、ep_recall 63.6%→**79.2% (+15.6pp)**、neg_miss 35.1%→**12.4%** ✅ |
| **判定** | ✅ 可接受。Paper 98% vs 我们 79.2% = 直接预测 vs 基座传递 + 5min vs 1h | — |

---

## 总结

| 基线 | 论文数据集可获取? | 我们基座是否能复现增益? | 当前状态 |
|---|---|---|---|
| CRC | ✅ ETTh1+Weather | ❌ 缺失因果编码器 → 退化 | 实现已备案 |
| PIR | ✅ ETTh1+Weather | ❌ Linear 基座太弱 | 实现已备案 |
| Spike Reg | ❌ 新加坡 NEMS | ❌ 无尖峰场景 | 实现已备案 |
| Vahedi | ✅ NEM SA1 2024 | ✅ 79.2% ep_recall (-10% MAE) | 可接受 |
| Quantile | N/A | N/A | ✅ 可用 |

**诚实结论**：CRC 和 PIR 的增益依赖于论文的强基座 + 专用 encoder/架构，我们的简化实现无法复现增益。**但这不影响论文的比较逻辑**——论文中我们与这些方法对比时，应诚实用它们论文报告的数字（强基座上的最优结果），而非我们简化版的数据。我们的 HCH 与它们不在同一基座层级上竞争；我们强调的是*在电价/负价域上的独特定位*（它们不做电价）。

**论文中的对比策略**：Related Work 表列出 CRC/PIR 的 LTSF 结果 + 说明它们不涉及电价 + Ours 在电价域的结果，形成域差异对比而非同数据集硬刚。

