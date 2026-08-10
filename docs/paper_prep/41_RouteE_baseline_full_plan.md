# 同行基线复现 + HCH启发提炼 完整计划

> 日期：2026-08-09 | 目标：完成五基线复现 + 提取可迁移设计 → 论文 Related Work + Method 强化

---

## Phase 1: 严格复现（可独立完成）

### 1.1 Vahedi 2026 (IEEE ICCE) ✅ 96.1% → 收尾

| 项目 | 论文 | 我们 | 状态 |
|---|---|---|---|
| 数据 | NEM-SA 5min 2024 | NEM SA1 hourly 2024 | ✅ |
| 方法 | LightGBM 二阶段（无基座） | 同 | ✅ |
| 指标 | 负价事件召回 98% | **96.1%** | ✅ |
| 负价漏判率 | — | 16.3%→4.4% | ✅ |
| MAE | — | 32.4→33.0 | ✅ |

**差距原因**：论文 5min 分辨率 vs 我们 1h 聚合。±2pp 可接受。

**收尾任务**：
- [ ] 整理 `repro_vahedi.py` 为干净可运行版本
- [ ] 输出最终指标表
- [ ] 写复现小结到报告

### 1.2 Quantile Correction（经典方法）

| 项目 | 做法 |
|---|---|
| 数据 | ETTh1（LTSF 标准集） |
| 方法 | LightGBM quantile regression @ {0.1, 0.5, 0.9} |
| 对比 | 直接回归 vs quantile 校正 |
| 指标 | MSE, MAE, 区间覆盖 |

**收尾任务**：
- [ ] 在 ETTh1 上跑独立脚本（不经过任何基座）
- [ ] 记录 MSE/MAE

---

## Phase 2-4: 论文深读 → 可迁移启发提炼

### 2.1 CRC (arXiv:2512.22428) — 安全校正

**论文核心贡献**：首次将"安全"作为后处理第一目标（不是精度）

**可迁移组件**：

| 论文机制 | HCH 对应 | 可迁移性 |
|---|---|---|
| **方向门控**：仅当校正符号与残差一致时生效 | HCH 已有：P(event)×幅度 → 隐含方向 | ⚠️ 隐式已有 |
| **分位裁剪**：校正量上限 = validation 分位 | HCH 已有：λ∈[0,1] 收缩 | ✅ 已有 |
| **逐点选择**：每个 (node,horizon) 单独选 linear vs hybrid | **HCH 空缺**：当前按天整体选 λ | **⭐ 可迁移** |
| **收缩混合**：validation 改善不显著→回退基座 | HCH 已有：SCARR λ=0 弃权 | ✅ 已有 |
| NDR (非退化率) 作为评估指标 | HCH 空缺 | **⭐ 可加入论文** |

**提取任务**：
- [ ] 深读 CRC §3.2 四层安全防火墙的数学定义
- [ ] 评估"逐点选择"能否作为 HCH 的 λ 细化：从每分支 1 个 λ → 每天每时段 24 个 λ
- [ ] 评估 NDR 指标能否加入论文评估体系
- [ ] 记录"CRC 不碰电价"的 Related Work 分界线

### 2.2 δ-Adapter (arXiv:2601.20280, ICLR 2026) — 即插即用

**论文核心贡献**：O(δ)-bounded 的输入/输出双端 adapter，有理论保证

**可迁移组件**：

| 论文机制 | HCH 对应 | 可迁移性 |
|---|---|---|
| **Input nudging**：可学习 mask 选特征 | HCH 已有：Z = yhat+日历+残差，但不自适应 | **⭐ 可迁移** |
| **Output correction**：加性残差 | 同 HCH | ✅ 已有 |
| **δ-bounded**：|δ|≤δ_max 保证 O(δ) 漂移 | HCH 对应 λ∈[0,1] | ✅ 已有 |
| **Proposition 2.1**：小步校正的局部改善定理 | **HCH 空缺**：无对应理论证明 | **⭐ 启发** |
| **Compositional stability**：input+output 可叠加 | HCH 空缺 | 远期待评估 |

**提取任务**：
- [ ] 深读 δ-Adapter §2.2 小步改善定理 → 能否写出 HCH 版本的命题？
- [ ] 评估 feature selector mask 是否可嵌入 Z 构建 → 自适应特征选择
- [ ] 评估 δ-Adapter 的"预训练基座+Sundial/TTM"与我们的"轻量基座"层次差异 → Related Work 划界
- [ ] 记录 δ-bounded 与 λ-shrinkage 的理论对照

### 2.3 Spike Regularization (arXiv:2602.05430, AAAI 2026 WS) — 尖峰感知

**论文核心贡献**：在 TSFM 训练中加入 spike-aware penalty，改善 MAPE 37.4%

**可迁移组件**：

| 论文机制 | HCH 对应 | 可迁移性 |
|---|---|---|
| **Spike penalty**：|residual|>threshold → 加权 | **HCH 空缺** | **⭐ 可迁移** |
| **TSFM backbones**：TTM/MOIRAI/TimesFM | HCH 用 LightGBM | 不同层级 |
| **MAPE 改善 37.4%**（新加坡市场） | HCH 无 MAPE 指标 | 参考 |

**提取任务**：
- [ ] 深读 spike penalty 的数学形式 → 能否作为 HCH 幅度头的辅助 loss？
- [ ] 评估在 HCH 的 S2 训练中加入 spike-weighted L1 → 负价+尖峰同时加权重
- [ ] 记录 SpikeReg 数据不可获取的限制

---

## Phase 5: 汇总输出

### 5.1 复现报告

写入 `docs/paper_info/peer_reproduction_log.md`（已存在，需更新）：

```
- Vahedi: 96.1% ep_recall（论文 98%, ±2pp 可接受）
- Quantile: ETTh1 MSE/MAE
- CRC: 不可完全复现原因 + 已提取 NDR + 逐点选择
- δ-Adapter: 不可完全复现原因 + 已提取 δ-bounded + 小步改善
- SpikeReg: 不可完全复现原因 + 已提取 spike penalty
```

### 5.2 HCH 启发文档

新建 `docs/paper_info/hch_improvement_ideas.md`：

```markdown
# HCH 改进方向（来自同行基线启发）

## 来源 1: CRC 四层安全防火墙
- **逐点选择**: 当前 λ 按天整体标定 → 可按 (时段, 市场) 逐组标定
- **NDR 指标**: 加入论文评估体系，报告 HCH 的非退化率

## 来源 2: δ-Adapter 小步改善
- **自适应特征选择**: 当前 Z 固定构造 → 可学习 feature mask
- **Proposition 2.1 对应**: 能否写 "Corollary: for τ=0.5, E[|y−ŷ_hch|] ≤ E[|y−ŷ_base|]"

## 来源 3: SpikeReg spike penalty
- **幅度头加权训练**: L1 loss + λ_spike * w(spike) * |δ|
- 在负价和尖峰样本上给额外权重→改善尾部预测

## 来源 4: Vahedi 二阶段
- 确认 HCH 的 occurrence×magnitude vs Vahedi 的 classifier→regressor 的差异
- HCH 优势: 冻结基座 + 选择性触发 + 恒等回退
```

---

## Phase 6: 清理

- [ ] 删除 `run_peers.py` 中错误对照的旧结果
- [ ] 清理 `results/` 中非正式的中间文件
- [ ] 保留 `repro_vahedi.py` 作为独立复现脚本
- [ ] `peer_reproduction_log.md` 最终化

---

## 执行顺序（可部分并行）

```
Phase 1: Vahedi收尾 (0.5h) → Quantile (0.5h)
    ↓
Phase 2-4: CRC/δ-Adapter/SpikeReg 深读 + 提取 (3h, 可交叉)
    ↓
Phase 5: 汇总报告 + HCH启发文档 (1h)
    ↓
Phase 6: 清理 (0.5h)
```
