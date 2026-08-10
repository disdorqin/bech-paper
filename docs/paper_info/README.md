# 论文信息登记

> 用途：记录 Route-E 应用论文的所有关键配置信息
> 维护：每次新增数据集/基线/引用时，在此更新

---

## 数据集

| 名称 | 来源 | 特征维度 | 负价频率 | 论文引用 | 状态 |
|---|---|---|---|---|---|
| Shandong | 私有 | 23-35 | 11-13% | 郭鸿业公开统计 | ✅ 主实验 |
| LAGO_DE | Lago 2021 | 4 | ~1% | Lago et al. 2021 | ✅ 已跑 |
| LAGO_BE/FR/NP/PJM | Lago 2021 | 4 | 0-1% | Lago et al. 2021 | ✅ 已跑 |
| GEFCom2014-P | GEFCom | 4 | 0% | Hong et al. 2016 | ✅ 已跑 |
| NEM SA1 | AEMO | 5 | ~15-25% | Gani 2026, Lu 2026 | 待拉取 |
| DK1 | Energi Data Service | 2 | ~1.8% | 无EPF引用 | 待定 |

---

## 基线方法

| # | 名称 | 论文 | 类型 | 状态 |
|---|---|---|---|---|
| B0 | Identity (Base) | — | 恒等 | ✅ |
| B1 | Pointwise L1 Correction | 自研 | 简单后处理 | 待实现 |
| B2 | Quantile Correction | 自研 | 统计后处理 | 待实现 |
| B2 | CRC | arXiv:2512.22428 | 安全残差校正 | 记录方法论 |
| B3 | **PIR** | NeurIPS 2025 (arXiv:2505.23583) | 实例感知后处理修正 | 待复现 |
| B4 | Spike Regularization | AAAI 2026 WS (arXiv:2602.05430) | 尖峰正则化 | 待深入复现 |
| Ours | Hurdle Correction Head | — | Bi-Hurdle选择性校正 | ✅ v1 |

---

## 关键引用

| 文献 | 用途 |
|---|---|
| Hunt 2025 (arXiv:2509.08369) | Tweedie loss 优于 MSE 的降水预测证据 |
| Wang et al. 2019 (arXiv:1912.07753) | ZILN 零膨胀对数正态损失 |
| Kong et al. 2020 (arXiv:2010.16040) | Deep Hurdle Networks |
| Gani et al. 2026 (arXiv:2602.01157) | NEM 负价退化诊断 |
| Lu et al. 2026 (arXiv:2604.23908) | NEM SA1 基准 |
| Lago et al. 2021 (Applied Energy) | Lago 基准数据集 |
| CRC (arXiv:2512.22428) | 同行：安全残差校正 |
| δ-Adapter (arXiv:2601.20280, ICLR 2026) | 同行：冻结基座适配器 |
| Spike Reg (arXiv:2602.05430, AAAI 2026 WS) | 同行：尖峰正则化 |

