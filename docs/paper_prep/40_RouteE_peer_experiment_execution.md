# Route-E 同行复现 + 数据集 + 负电价实验 执行计划

> 日期：2026-08-09 | 状态：执行中

---

## 一、同行基线复现清单

### 目录: `experiments/07-route-e/peers/`

| # | 基线 | 论文 | 复现方式 | 验收标准 | 预计工作量 |
|---|---|---|---|---|---|
| B0 | Identity | 无 | `return base_pred` | bit-exact=Base | 0h |
| B1 | Pointwise L1 | 自研 | LightGBM reg y-yhat on S2→correct S4 | MAE↓合理 | 0.5h |
| B2 | Quantile Correction | 自研 | Pinball loss multi-quantile on S2→S4 | 区间覆盖合理 | 1h |
| B4 | CRC | arXiv:2512.22428 | 论文复现：因果编码器+四层安全门控 | 与原论文 Lago指标偏差<10% | 4h |
| B5 | δ-Adapter | arXiv:2601.20280 (ICLR'26) | 论文复现：input nudging+output residual | 同上 | 4h |
| B6 | PnP-Corrector | arXiv:2605.08935 | 论文复现：冻结引擎+校正代理 | 同上 | 4h |
| B7 | Spike Reg | arXiv:2602.05430 (AAAI'26 WS) | 论文复现：spike penalty term | 同上 | 2h |

### 复现协议

所有同行基线接收相同的：
- **冻结基座**：S1 训练的 Base 预测
- **数据窗口**：S1/S2/S3/S4 = 50/20/10/20
- **特征集**：校正特征矩阵 Z（与 HCH 相同，公平比较）
- S4 评估：相同指标（MAE、episode recall、complete miss、fire rate）

---

## 二、数据集调研结论

### 已确认

| 数据集 | 负价频率 | 是否被 EPF 论文引用 | 状态 |
|---|---|---|---|
| **NEM SA1** | ~15-25% | **是** — Gani 2026 (arXiv:2602.01157) + Lu 2026 (arXiv:2604.23908) | 可作标准外部验证 |
| **DK1** | ~1.8% (100k样本) | 否 — 仅1篇储能论文提及 | 可选，非标准 benchmark |
| **DE-LU** | 已由 LAGO_DE 覆盖 | — | 不重复 |

### 决定

1. **NEM SA1 作为主外部验证**（有论文引用，可作标准对比）
2. DK1 可选附加；DE-LU 不重复拉取
3. 如无更多标准高负价 benchmark → **必须设计「负电价专项实验组」**

---

## 三、实验矩阵（最终版）

### 主实验 (R1) — 山东 5 基座 Base vs HCH ✅ 已完成

### 外部验证 (R2a) — NEM SA1 + LAGO_DE Base vs HCH

| 数据集 | 重 | 基座 |
|---|---|---|
| NEM SA1 | 高负价 | Linear / GBDT |
| LAGO_DE | 低负价 | Linear / GBDT |

### 消融 (R2b) — 山东 + LAGO_DE

去 occurrence / 去 magnitude / 去 SCARR / 去 λ

### 同行对照 (R2c) — 山东+NEM_SA1+LAGO_DE

B0-B7 全基线 vs HCH，报告 episode 指标

### 负电价专项 (R3) — **新设计**

仅取各数据集中负电价时段，横向对比所有方法在：
- 负价 episode 召回率
- 完整漏报率
- 假事件率
- 边界误差

目的：无论基准市场是否有负价，专门对比各方法**处理负价的能力**

---

## 四、当前执行状态

| 任务 | 状态 |
|---|---|
| DK1 数据拉取 | ✅ 已下载 (100k条，JSON格式，1.8%负价) |
| NEM SA1 数据拉取 | 待执行（AEMO nemweb 多文件下载） |
| 同行 CRC | 待复现 |
| 同行 δ-Adapter | 待复现 |
| 同行 B1/B2 | 待实现 |
| 同行 PnP/SpikeReg | 待复现 |

## 五、立即执行

1. 先实现简单同行 (B0/B1/B2)
2. 再复现论文同行 (CRC/δ-Adapter)
3. PnP/SpikeReg 最后（工作量最大）
4. 所有同行在 LAGO_DE 上验证指标与论文一致性
5. 全部过验后统一跑实验矩阵
