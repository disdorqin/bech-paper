# Route-E 实验协议 v1

> 日期：2026-08-09 | 目标论文：Route-E 应用论文 | CPU only

---

## 0. 实验矩阵总览

| 编号 | 名称 | 数据 | 基座 | 对比维度 | 核心指标 |
|---|---|---|---|---|---|
| **R1** | 山东主实验 | 山东 hourly/96 | Linear / MLP / LSTM / Transformer / GBDT | Base vs Ours | MAE, RMSE, neg_miss_rate, episode_recall, complete_miss, false_event, boundary_mae |
| **R2a** | 公开基座对比 | LAGO_DE / NEM_SA1 / UniElecPrice 3 国 | 同上 5 基座 | Base vs Ours | 同 R1 |
| **R2b** | 消融 | 山东 + LAGO_DE | Linear / GBDT | -occurrence / -magnitude / -SCARR / -lambda | neg_miss_rate, episode_recall, normal_harm |
| **R2c** | 同行对照 | 山东 + LAGO_DE | Linear / GBDT | Identity / PointCorrection / QuantileCorrection / Ours | episode_recall, worst_harm, degradation_count |

---

## 1. 数据集与切分

### 1.1 山东（主实验）

| 文件 | 时间分辨率 | 特征 | 目标 |
|---|---|---|---|
| `data/raw/provinces/shandong_pmos_hourly.csv` | 1h (23 列) | 日前电价 / 实时电价 / 风电光伏核电地方负荷预测/实际 | 日前(da_cq_price) 和实时(rt_cq_price) 分别建模 |
| `data/raw/provinces/shandong_pmos_96_full_v2.xlsx` | 15min (35 列) | 同上 + 备用容量 + 时段号 | 可选 96 点高分辨率（暂用 hourly） |

**预处理**：
- 清洗 missing/duplicate timestamps
- 分日前(da_price)和实时(rt_price)两条独立建模线
- 特征列：exog_fc = 所有"XX预测值"列；exog_act = 对应"XX实际值"列（lag ≥24h）
- 负价阈值 = 0 元/MWh

### 1.2 公开数据集（复现 + 泛化）

| 优先级 | 数据集 | 负价 | 用途 |
|---|---|---|---|
| P0 | NEM_SA1 | 24.6% | 负价最高频 |
| P0 | LAGO_DE | 1.03% | 欧洲典型 |
| P1 | UniElecPrice 3 负价国 | 有 | 跨市场 |
| P2 | LAGO_BE/FR/PJM/NP | 少/无 | 覆盖多样性 |
| P2 | GEFCom2014-P | 无 | 无事件对照 |

### 1.3 四段切分

```text
S1 (50%)  → 训练+冻结基座
S2 (20%)  → 训练 Hurdle Correction Head
S3 (10%)  → SCARR 标定（选 λ 和 τ）
S4 (20%)  → 锁定测试（唯一报告来源）
```

- 防泄漏：残差历史 ≥24h 滞后；y_t 永不进特征
- 校正特征 Z：yhat + 日内形状 + 残差历史 + 日历 + cutoff-safe exogenous

---

## 2. 方法组件

| 组件 | 函数 | 论文名 |
|---|---|---|
| 基座 | `backbones.make_backbone(name)` | Frozen Numerical Forecaster |
| 校正头 | `selective_hurdle.fit/calibrate/apply` | Hurdle Correction Head (HCH) |
| 双尾分类 | Internally: `P(neg|Z)`, `P(spike|Z)` | Bi-Hurdle occurrence heads |
| 条件幅度 | Internally: `E[y−yhat | tail, Z]` | Conditional magnitude head |
| 安全证书 | SCARR: bootstrap LCB + conformal safety quantile | Paired risk certificate |
| 恒等回退 | λ=0 → output = base (bit-exact) | Exact identity fallback |

---

## 3. 指标定义

### 3.1 点级

| 指标 | 公式 | 说明 |
|---|---|---|
| MAE | mean(|ŷ−y|) | 主点级指标 |
| RMSE | sqrt(mean((ŷ−y)²)) | 惩罚大误差 |
| neg_miss_rate | mean(ŷ[neg] ≥ 0) | 负价漏判率 |
| normal_harm | mean(|ŷ−y|[normal] − |b−y|[normal]) | 正常期退化量 |
| fire_rate | mean(λ>0) | 触发率 |

### 3.2 Episode 级（新核心）

| 指标 | 定义 |
|---|---|
| episode_recall | 真实负价 episode 中被任何预测负价 episode 匹配的比例 |
| complete_miss_rate | 真实负价 episode 中完全未被匹配的比例 |
| false_episode_rate | 预测的虚假负价 episode（无真实对应）率 |
| boundary_mae | 匹配 episode 的 (|start_pred−start_true|+|end_pred−end_true|)/2 |
| episode_count | 真实负价 episode 总数 |

**Episode 提取算法**：price<0 的最大连续区间；Hungarian matching（带 dummy）配预测/真实 episode。

---

## 4. R1：山东主实验

### 4.1 设置

```
数据：山东 hourly (日前价格) × 5 基座
对比：Base (raw) vs Ours (Base + HCH)
报告：点级 + episode 级全指标
```

### 4.2 预期产出

| 表/图 | 内容 |
|---|---|
| Table 1 | 5 基座 × (Base/Ours) × 指标矩阵 |
| Fig 1 | 5 基座 episode_recall 前后对比柱状图 |
| Fig 2 | 代表性负价日：Base vs Ours 24h 曲线对比 |

---

## 5. R2：公开数据集

### 5.1 R2a 基座对比

```
数据：NEM_SA1, LAGO_DE, UniElecPrice×3 × 5 基座
格式：同 R1 Table 1
```

### 5.2 R2b 消融

```
数据：山东 + LAGO_DE × (Linear, GBDT)
消融项：
  1. Full Ours
  2. -occurrence（去掉分类门，直接 regress −> 等价单一幅度头）
  3. -magnitude（仅用 occurrence 二值触发，不做幅度修正）
  4. -SCARR（λ=1 无标定）
  5. -lambda（λ=1，无收缩）
```

### 5.3 R2c 同行对照

```
Baselines against Ours (same data, frozen base):
  B0  Identity (=Base)
  B1  Pointwise residual correction (L1 regression)
  B2  Quantile correction (pinball-based)
  B3  Hurdle Correction Head (Ours)
```

---

## 6. 不做/日后做

| 项目 | 状态 |
|---|---|
| E4 Hurdle 理论诊断（单头 vs 双头等价性证明） | 论文叙述，不做专实验 |
| E5 负价 zoom-in（完整案例图库） | 论文叙述（Fig 2 覆盖基本需求），深度 zoom 日后 |
| GNN 图结构增强 | 不做 |
| e-process / admin clamp / 双市场联合 | 已否决 |

---

## 7. 代码清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `src/backbones.py` | 5 基座工厂 | 保留（不改） |
| `src/common.py` | 数据加载 / 切分 / 评估 + 加山东 loader + episode 指标 | 改造 |
| `src/selective_hurdle.py` | Hurdle Correction Head（原 bech.py 清理重命名） | 重写 |
| `experiments/07-route-e/run_route_e.py` | 统一实验入口 | 新建 |
| `experiments/07-route-e/configs.yaml` | 实验配置 | 新建 |
| `experiments/figures/paper_plot_style.py` | 绘图风格（已写） | 已建 |

---

## 8. 执行顺序

```text
1. 实验协议文档（本文，DONE）
2. 清理 src/ + 重写 selective_hurdle.py
3. 扩展 common.py（山东 loader + episode 指标）
4. 写统一实验 runner
5. R1 山东：跑 5 基座 × Base/Ours
6. R2a 公开：跑选定数据集 × 5 基座
7. R2b 消融：跑
8. R2c 同行：跑
9. 绘图表
10. 论文写作
```
