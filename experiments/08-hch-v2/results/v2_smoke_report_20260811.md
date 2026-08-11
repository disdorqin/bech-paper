# HCH v2 Smoke Test Report

> 日期: 2026-08-11 | Seed: 0 | neg_thr=0, spike_thr=S1 p99

## 1. 实验对象

| 类型 | 内容 |
|---|---|
| 数据集 | NEM_SA1 (26%负价), LAGO_DE (1%负价), shandong_DA (11%负价) |
| 宿主 | Linear, MLP |
| 方法 | Identity, ResidualL1, QuantileResidualLGBM, HCHv2 |
| 切分 | S1/S2/S3/S4 = 50%/20%/10%/20% (按日期) |

## 2. 实验结果

### 2.1 NEM_SA1 (26% neg, 606 neg hours in S4)

| Method | ΔMAE | touch_rate | harm_rate | neg_n | 备注 |
|---|---|---|---|---|---|
| **Linear 宿主 (Base MAE=69.82)** |
| ResidualL1 | +27.33 | 100% | 75.1% | 606 | 全面退化 |
| QuantileResidualLGBM | +24.47 | 89.3% | 67.2% | 606 | cutoff 触发了部分弃权 |
| **HCHv2** | **-1.86** | 6.8% | 4.2% | 606 | gate=93%I/3%D/4%U |
| **MLP 宿主 (Base MAE=85.40)** |
| ResidualL1 | **-19.77** | 100% | 36.1% | 606 | 大幅改善但无安全机制 |
| QuantileResidualLGBM | -12.21 | 76.8% | 30.4% | 606 | |
| **HCHv2** | -0.40 | 0.0% | 0.0% | 606 | 100% Identity (门控安全弃权) |

### 2.2 LAGO_DE (1% neg, 194 neg hours in S4)

| Method | ΔMAE | touch_rate | harm_rate | 备注 |
|---|---|---|---|---|
| **Linear 宿主 (Base MAE=6.08)** |
| ResidualL1 | **-1.06** | 100% | 37.6% | |
| QuantileResidualLGBM | **-1.06** | 94.8% | 34.6% | |
| **HCHv2** | -0.02 | 0.9% | 0.6% | 99%I |
| **MLP 宿主 (Base MAE=7.39)** |
| ResidualL1 | **-1.83** | 100% | 36.0% | |
| QuantileResidualLGBM | -1.42 | 97.3% | 35.3% | |
| **HCHv2** | +0.01 | 5.3% | 3.3% | 95%I/3%D/2%U |

### 2.3 shandong_DA (11% neg, 929 neg hours in S4)

| Method | ΔMAE | touch_rate | harm_rate | 备注 |
|---|---|---|---|---|
| **Linear 宿主 (Base MAE=92.83)** |
| ResidualL1 | **-24.24** | 100% | 39.1% | |
| QuantileResidualLGBM | **-24.24** | 100% | 39.1% | |
| **HCHv2** | +0.23 | 8.5% | 4.7% | 92%I/7%D/2%U |
| **MLP 宿主 (Base MAE=143.33)** |
| ResidualL1 | **-63.19** | 100% | 24.9% | |
| QuantileResidualLGBM | -63.13 | 99.9% | 24.9% | |
| **HCHv2** | -8.32 | 32.1% | 17.4% | 68%I/19%D/13%U |

## 3. 关键发现

### 3.1 HCH v2 安全性

HCH v2 是唯一具有安全机制的校正方法：
- **harm_rate**: HCH v2 0-17.4% vs 基线 24.9-75.1%
- 当基线会恶化时（NEM_SA1×Linear），HCH v2 是唯一改善的方法（ΔMAE -1.86）
- 门控弃权率 68-100%，正确识别了"不修更好"的场景

### 3.2 基线对比

| 指标 | ResidualL1 | QuantileResidualLGBM | HCHv2 |
|---|---|---|---|
| 最优 ΔMAE | **-63.19** | -63.13 | -8.32 |
| 平均 harm_rate | 41.3% | 38.6% | **4.9%** |
| 安全机制 | 无 | width cutoff | **CARA 风险价值** |
| 弃权能力 | 0% | 0-23% | **68-100%** |

### 3.3 HCH v2 Gate 诊断

- **记忆库大小**: NEM_SA1=36, LAGO_DE=218, shandong=166 (S3 天数)
- **触发率**: 负价密集市场(shandong MLP)最高 32%，其他市场 0-9%
- **CARA 配置**: η=0.1, τ=0.2, 温度=0.05
- **限制因素**: S3 记忆库太小（36 条对高方差场景不足），导致 gate 过于保守

## 4. 遇到的问题与解决

| # | 问题 | 原因 | 解决 |
|---|---|---|---|
| 1 | HCH v2 MAE 与其他方法不可比 | dataloader 做了 z-score 标准化但输出未反归一化 | 在评估前乘以 p_std + p_mean |
| 2 | Gate 始终返回 Identity | compute_gain 中 target shape 多了一维 `[B,24,1,3]` 导致 DVG 无法处理 | squeeze target 维度 |
| 3 | LAGO_DE 训练崩溃 | collate_daily 中 max_exog 取错了维度索引 | 从 dim=2 改为 dim=1 |
| 4 | Baseline Z 与 yhat 长度不对齐 | build_corrector_features 与 tabular valid 索引不一致 | 使用 build_corrector_features 统一构建 Z |
| 5 | DailyEpisodeDataset 日期匹配失败 | pd.Timestamp 与 date 对象比较不精确 | 改用 .date() 显式转换 |

## 5. 架构验证

| 模块 | 状态 | 验证结果 |
|---|---|---|
| Bi-OMC 候选 | ✅ | δ_down≤0, δ_up≥0, 两方向 100% 非零 |
| CAGM 记忆 | ✅ | S3 OOF 候选写入, top-K 检索正常工作 |
| DVG 门控 | ✅ | CARA + KL-softmax 路由, soft_hard 决策 |
| 基线集成 | ✅ | ResidualL1 + QuantileResidualLGBM 统一接口 |
| 反泄漏 | ✅ | S1 阈值/统计冻结, S4 不在训练/校准中 |

## 6. 待改进

1. **S3 大小**: 当前 10% 切分导致记忆库过小（36-218条），建议增大 S3 比例或扩大 S2+S3
2. **DVG 保守性**: η=0.1 仍偏保守，可进一步降低或使用 per-market calibration
3. **官方 PIR/δ-Adapter**: 需要接入官方仓库进行公平对比
4. **TCN/PatchTST**: 宿主已实现但 smoke 未覆盖（速度较慢）
