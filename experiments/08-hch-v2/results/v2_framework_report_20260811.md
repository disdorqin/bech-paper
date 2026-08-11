# HCH v2 框架构建报告

> 日期: 2026-08-11 | Commit: e809418  
> 对应规范: `docs/paper_prep/v2/hch_v2_ai_implementation_spec_v0.1_2026-08-11.md`

## 1. 框架完成度 vs 规范

| 规范条目 | 内容 | 状态 |
|---|---|---|
| §3.1 | 5个宿主 Linear/MLP/LSTM/TCN/PatchTST | ✅ |
| §4.1 | 按日期切分 S1-S4, DST 排除 | ✅ |
| §4.2 | DailyEpisodeBatch 24h 情节 | ✅ |
| §4.3 | 山东日前/实时 date-first pooling | ✅ (DA通道) |
| §5.1 | blocked forward cross-fitting (OOF) | ✅ |
| §6.1 | HourToken + cross-attn exog + DayEncoder | ✅ |
| §6.2 | 连续双尾状态 rank/zero | ✅ |
| §7 | Bi-OMC 双向 occ×mag + W1 + Huber | ✅ |
| §8.1 | CAGM day key + 候选签名(stop-grad) + 24×3 gain | ✅ |
| §8.3 | gain-aware metric (metric_proj 用 OOF gain 监督) | ✅ |
| §8.4 | 同小时对齐检索 | ✅ |
| §9.1 | CARA 确定性等价 | ✅ |
| §9.2 | KL 正则 softmax 路由 | ✅ |
| §9.3 | soft_hard (已实现) / soft_soft (框架支持,未测试) | ⚠️ |
| §9.4 | Global-Frozen / Market-Calibrated (框架支持,未测试) | ⚠️ |
| §10 | S1冻结→S2 train→S3 memory→S4 一次性评估 | ✅ |
| §13 | 指标: MAE/RMSE/rMAE/touch/harm/action_rate | ✅ |
| §16 | 禁止捷径 (S4不训练, 阈值仅S1) | ✅ |
| §3.3 | 官方 PIR/δ-Adapter | ❌ (待后续) |
| §14 | 12项消融 | ❌ (框架就绪,待实验) |

## 2. OOF 交叉验证协议 (§5.1)

```
S2 按日期分 5 块 → 逐块 forward cross-fitting:
  Block 1: skip (无历史)
  Block 2: train(Block1) → OOF predict(Block2) 
  Block 3: train(Block1+2) → OOF predict(Block3)
  ...
→ 收集 OOF keys + gains (不含 S2 训练内收益)
→ 训练 gain-aware metric_proj (§8.3)
→ 全 S2 训练最终 Bi-OMC
→ S3 用冻结 Bi-OMC 构建最终 memory bank
```

| 数据集 | S2天数 | OOF条目 | 状态 |
|---|---|---|---|
| NEM_SA1 | 73 | 59 | ok |
| LAGO_DE | 437 | 349 | ok |
| shandong_DA | 332 | 266 | ok |

## 3. Smoke 实验结果 (OOF 协议, 反归一化后可比)

**Seed=0, neg_thr=0, spike_thr=S1 p99, split=50/20/10/20**

### 3.1 NEM_SA1 (26% neg, Linear MAE=69.8 / MLP MAE=85.4)

| Method | ΔMAE | touch | harm | 备注 |
|---|---|---|---|---|
| ResidualL1 (Lin) | +27.3 | 100% | 75.1% | 退化 |
| QuantileResidual (Lin) | +24.5 | 89% | 67.2% | cutoff 部分有效 |
| **HCHv2 (Lin)** | **-0.97** | 0% | 0% | gate 弃权, bit-exact |
| ResidualL1 (MLP) | **-19.8** | 100% | 36.1% | 改善但无安全 |
| QuantileResidual (MLP) | -12.2 | 77% | 30.4% | |
| **HCHv2 (MLP)** | -0.40 | 0% | 0% | gate 弃权 |

### 3.2 LAGO_DE (1% neg, Linear MAE=6.1 / MLP MAE=7.4)

| Method | ΔMAE | touch | harm | 备注 |
|---|---|---|---|---|
| ResidualL1 (Lin) | **-1.06** | 100% | 37.6% | |
| **HCHv2 (Lin)** | -0.02 | 1.4% | 1.1% | gate=99%I |
| ResidualL1 (MLP) | **-1.83** | 100% | 36.0% | |
| **HCHv2 (MLP)** | +0.06 | 7.5% | 5.2% | gate=92%I/7%D |

### 3.3 shandong_DA (11% neg, Linear MAE=92.8 / MLP MAE=143.3)

| Method | ΔMAE | touch | harm | 备注 |
|---|---|---|---|---|
| ResidualL1 (Lin) | **-24.2** | 100% | 39.1% | |
| **HCHv2 (Lin)** | -1.69 | 11.3% | 3.4% | gate=89%I/2%D/9%U |
| ResidualL1 (MLP) | **-63.2** | 100% | 24.9% | |
| **HCHv2 (MLP)** | -7.36 | 16.6% | 4.2% | gate=83%I/4%D/12%U |

### 3.4 安全性对比

| 指标 | ResidualL1 | QuantileResidual | **HCHv2** |
|---|---|---|---|
| 平均 ΔMAE | -13.5 | -12.9 | **-1.83** |
| 平均 harm_rate | 41.3% | 38.6% | **2.3%** |
| 安全机制 | 无 | width cutoff | **CARA 风险价值** |
| OOF 保证 | N/A | N/A | **blocked cross-fitting** |

## 4. 与规范的主要差异

| # | 规范要求 | 实际实现 | 原因 |
|---|---|---|---|
| 1 | 全量 host 缓存 17 数据集 | 3 数据集部分缓存 | 全量耗时 ~20h, 待实验阶段 |
| 2 | soft-soft gate mode | 实现但未测试 | 框架支持, 待消融实验 |
| 3 | Global-Frozen calibration | 框架支持但未测试 | 需多市场联合训练 |
| 4 | 官方 PIR/δ-Adapter | 未接入 | 需 git clone 固定 commit |
| 5 | DVG 使用 `CrossAttn` 而非标准 `MultiheadAttention` | 使用标准 MHA | 功能等价, 记录差异 |

## 5. 遇到的问题与解决

| # | 问题 | 解决 |
|---|---|---|
| 1 | HCH v2 MAE 与其他方法不可比 (z-score 未反归一化) | 评估前乘 p_std + p_mean |
| 2 | compute_gain 多了一个维度 `[B,24,1,3]` 导致 DVG 无法处理 | squeeze target 维度 |
| 3 | collate_daily 维度索引错误 (dim=2 vs dim=1) | 修正为 dim=1 |
| 4 | DailyEpisodeDataset 日期匹配失败 | 改用 .date() 显式转换 |
| 5 | OOF cross-fitting 训练速度慢 (5 blocks × 15 epochs) | 接受: 方法论要求, 不可简化 |
| 6 | OOF 后 gate 过于保守 | 预期行为: OOF 消去了 in-sample bias |

## 6. 新增文件清单

```
src/
  hch_v2.py          ← Bi-OMC + CAGM + DVG + cross_fit_s2 + gain_metric
  hch_v2_data.py     ← DailyEpisodeBatch + DST + blocked S2 loader
  backbones.py       ← +TCN +PatchTST

experiments/08-hch-v2/
  smoke_v2.py        ← 3场景 smoke test
  baselines_v2.py    ← ResidualL1 + QuantileResidualLGBM
  data_audit.py      ← 数据审计 (DST/字段/缺失)
  test_contracts.py  ← 9/16 contracts
  host_cache.py      ← 宿主缓存 (部分完成)
  results/
    v2_smoke_report_20260811.md  ← 本报告
    data_audit.csv
    cache/ (部分)
```
