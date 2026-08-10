# Route-E 同行复现 + 外部验证 + 代码审计 准备文档

> 日期：2026-08-09 | 状态：准备中

---

## 一、同行基线复现目录

### 1.1 目录结构

```
experiments/07-route-e/peers/
├── README.md              # 每个基线的来源、论文DOI/arXiv、复现状态
├── identity/              # B0: 恒等基线（=Base）
├── pointwise_l1/          # B1: 逐点L1残差校正
├── quantile_correction/   # B2: 分位数校正（pinball）
├── crc/                   # B4: CRC (arXiv:2512.22428) — 加性后处理校正
├── delta_adapter/         # B5: δ-Adapter (arXiv:2601.20280) — 冻结基座适配器
├── pnp_corrector/         # B6: PnP-Corrector (arXiv:2605.08935)
└── spike_reg/             # B7: Spike Regularization (arXiv:2602.05430)
```

### 1.2 每个基线的复现协议

| 基线 | 论文 | 复杂 | 来源方案 | 验收标准 |
|---|---|---|---|---|
| B0 Identity | 无 | 0 | `pred = base_pred` | bit-exact = Base |
| B1 Pointwise L1 | 自研 | 低 | LightGBM reg y−yhat | 自实现 |
| B2 Quantile | 自研 | 中 | pinball loss 分位校正 | 自实现 |
| B4 CRC | arXiv:2512.22428 | 中 | 找 GitHub / 按论文复现 | 与原论文指标偏差 <5% |
| B5 δ-Adapter | arXiv:2601.20280 | 高 | GitHub / 论文复现 | 同上 |
| B6 PnP-Corrector | arXiv:2605.08935 | 高 | GitHub / 论文复现 | 同上 |
| B7 Spike Reg | arXiv:2602.05430 | 中 | 论文复现 | 同上 |

### 1.3 复用协议

所有同行基线接收相同的 **冻结基座输出 + 训练/标定/测试窗口划分**。
即：S1 基座冻结后，Baselines 和 HCH 看到的是同一份 Base 预测。

---

## 二、高负价公开数据集调研

### 2.1 已知候选

| 市场 | 区域 | 负价频率 | 公开 | 数据源 |
|---|---|---|---|---|
| NEM SA1 | 南澳 | ~24.6% | 是 | AEMO 公开下载 |
| DK1/DK2 | 丹麦 | ~5-10% | 是 | ENTSO-E 透明平台 / Energi Data Service |
| DE-LU | 德国 | ~2-5% | 是 | ENTSO-E / SMARD |
| SE3/SE4 | 瑞典 | 偶有 | 是 | Nord Pool / ENTSO-E |
| 九州 | 日本 | 2024起出现 | 部分 | JEPX 公开（日文） |
| ERCOT | 德州 | 偶有负价 | 是 | ERCOT 公开 |

### 2.2 调研任务

1. Web search: "negative electricity price frequency 2024 2025 by market"
2. Web search: "ENTSO-E negative price hours Germany Denmark 2024"
3. Web search: "AEMO NEM SA1 negative price statistics 2025"
4. Web search: "Nord Pool negative prices frequency"
5. Web search: "Japanese negative electricity price JEPX 2024"
6. 确定每个数据集的下载 API/URL

### 2.3 拉取优先级

P0: NEM SA1（最高负价频率）  
P1: 丹麦 DK1/DK2（北欧新能源高渗透）  
P2: 德国扩展（ENTSO-E 更长跨度）

---

## 三、源代码审计

### 3.1 对照原始设计文档

| 设计要求 | 实现 | 状态 |
|---|---|---|
| S1/S2/S3/S4 = 50/20/10/20 | `common.py:four_segment_split` | ✅ |
| 防泄漏：resid ≥24h lag | `build_corrector_features: lag 24/48/168` | ✅ |
| y_t 不进特征 | `build_tabular`: price_lag24+ 无当期价格 | ✅ |
| Bi-Hurdle: P(event)×E[Δ∣event] | `HurdleCorrectionHead._raw_delta: m*p` | ✅ |
| SCARR: bootstrap LCB + conformal safety | `calibrate()` | ✅ |
| λ∈[0,1], λ=0→bit-exact identity | `apply()` | ✅ |
| S2 hold-out → S3 calib reuse | `_calib_extra` | ✅ |
| τ=0.5 Bayes gate | 硬编码 0.5 | ✅ |
| Episode 指标 | `episode_metrics()` | ✅ |

### 3.2 公开数据集效果差的调查方向

**问题不是"没有负价就不能工作"**——HCH 有双向头（neg + pos spike）。但公开集上 pos 侧也没显著改善，可能原因：

1. **尖峰阈值问题**：spike_thr 从 S2 的 p99 计算，但 LAGO_DE/FR/BE 的 S4 price range 可能与 S2 不同→标签定义漂移
2. **校正特征 Z 信息不足**：公开集只有 4-5 列特征，Z 仅靠 yhat + 日历 + 残差史
3. **λ grid 太粗**：{0, 0.05, …, 1.0} 共 21 格，可能在精细市场错过最优
4. **base MAE 已经很低**：LAGO_DE Linear 6.08 → 提升空间窄
5. **S3 校准无事件**：负价少→neg branch 在 S3 上无事件可校准→λ=0 正确但不代表方法缺陷

### 3.3 调查方案

- 输出 S4 上每个数据集的 neg/spike 样本数
- 输出 λ_neg 和 λ_pos 分别在 grid 上的可行/不可行分布
- 检查 spike_thr 在不同段的一致性
- 若 spike 侧也弱：实施更细 λ grid + 检查 pos head 训练样本量
