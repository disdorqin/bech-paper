# Route-E 同行基线 × 数据集 对照表

> 日期：2026-08-09

---

## 基线数据集对照

| # | 基线 | 论文出处 | 论文使用的数据集 | 数据可获取性 | 我们用什么 |
|---|---|---|---|---|---|
| B1 | **Quantile Correction** | 经典统计方法 (无特定论文) | — | — | 与 Ours 相同的数据即可 |
| B2 | **Vahedi-style** | Vahedi 2026, IEEE ICCE | **NEM-SA 5min** (2024) | AEMO nemweb 公开下载 | 拉 AEMO SA1 数据 |
| B3 | **CRC** | Xie 2025, arXiv:2512.22428 | ETTh1/h2, ETTm1/m2, Weather, ECL, Traffic | ✅ `data/raw/ts_benchmarks/` 已有大部分 | LTSF benchmark 全 7 集 |
| B3 | **PIR** | Liu 2025, NeurIPS 2025 | ETTh1/h2, ETTm1/m2, Electricity, Solar, Weather, Traffic, PEMS | 与 CRC 共用 LTSF | LTSF benchmark |
| B4 | **Spike Reg** | Ponyuenyong 2026, AAAI 2026 WS | 新加坡 NEMS | ⚠️ 不可获取→NEM SA1 替代 | NEM SA1 |

---

## 三组实验 × 数据集映射

### R1 — 山东主实验
```
数据集：山东 hourly (23 features, 11-13% neg)
基座：Linear / MLP / LSTM / Transformer / GBDT
对比：Base vs Ours (HCH)
指标：MAE + episode full
```

### R2 — LTSF 后处理对照
```
数据集：ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic (7集)
基座：Linear / GBDT
对比：Base vs Quantile vs CRC vs δ-Adapter vs Ours
指标：MSE/MAE (跟随 LTSF 社区标准)
注意：LTSF 数据无电价→仅通用时序指标，不报告 episode 指标
```

### R3 — 负电价/尖峰专项
```
数据集：NEM SA1（Vahedi 2026 同款）
基座：Linear / GBDT
对比：Base vs Vahedi-style vs Spike Reg vs CRC vs Ours
指标：MAE + episode full + 负价时段 MAE
```

---

## 数据拉取清单

| 数据集 | 状态 | 操作 |
|---|---|---|
| Shandong | ✅ 已有 | — |
| ETTh1/h2, ETTm1/m2 | ✅ 已有 (`data/raw/ts_benchmarks/`) | — |
| Weather, ECL (Electricity), Traffic | 检查 | 确认文件存在 |
| Exchange, ILI | 不需要（不跑） | — |
| NEM SA1 | ❌ 需拉取 | AEMO nemweb 批量下载 |
| 新加坡 NEMS | ❌ 不可获取 | → 用 NEM SA1 替代 |

---

## 基线实现顺序

1. B1 Quantile (30min, 最简单)
2. B5 Spike Reg (30min, 损失函数改 pinball+spike penalty)
3. B2 Vahedi-style (1h, 两阶段分类→回归)
4. B3 CRC (4h, 需复现因果编码器+四层安全门控)
5. B4 δ-Adapter (4h, 需复现 input nudging+output residual)

每个完成后在 LTSF 或 NEM 上验证与原论文指标偏差 <10%。

