# 05 — Peer Fidelity Report(WP-3 / B1 smoke)

协议:`hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14.md` §14 peer-fidelity policy
日期:2026-08-14 · git:`5182ef4` · 运行:本地桌面(CPU)

目的:在把 peer 实现(B1 ResidualL1 / B2 QuantileResidual / B3 δ-Adapter / B4 PIR)
放进主矩阵之前,确认其"复现论文声明增益"的能力,以及现有实现是否忠实于官方架构。

---

## 1. δ-Adapter(DELTA_ACCEPT_AS_IS ✅)

### 1.1 测试历史(三个版本,前两个作废)

| 版本 | 基座 | 测试切分 | 结果 | 判定 |
|---|---|---|---|---|
| v1 ridge repro | LGB 自训,Weather | 后 15% 尾段 | 0.1297→0.1275(−1.7%) | **VOID** — 测试段退化 |
| v2 生产 PostY | LGB 自训,Weather | 后 15% 尾段 | 0.1297→0.0025(−98.07%) | **VOID** — 测试段退化 |
| v3 干净重测 | LGB 自训,Weather(剔 −9999) | 同尺度切分 | 0.9376→0.8055(**−14.1%**) | **VALID** |

### 1.2 v2 为何作废(−98% 是假象)

v2 的 −98% 不是适配器能力,是**测试切分退化**:

- 原始 weather.csv train 段含 **50 个 −9999 哨兵值**(缺失填充),把 train std 撑到 **386**;
- val/test 是 2020 年 11–12 月尾段,真实 std ≈ **16**;在 train 归一化尺度下 test 方差仅 ≈0.003;
- 于是 `test corr(yhat, y)=0.079`(LGB 预测与真值几乎无关),`linear(yhat)→y` 系数 ≈0.008(即"预测均值");
- 任何把输入压到均值的适配器在退化段都"完胜"。v1 ridge 因输出被 clip 限制,只到 −1.7%,同样是坏协议的产物。

诊断数据(同一份切分):
```
base test MSE:         0.1297
linear(yhat) test MSE: 0.0030   coef_a=0.0087(≈0 → 预测均值)
test corr(yhat,y) =    0.0791
val residual std:      0.5553, 分桶内 std 0.1609 → 无系统性 yhat 依赖
```

### 1.3 v3 干净重测(−14.1%,可信)

- 剔除 −9999(50 个)→ train std 回到 18.5;
- 同尺度切分,val/test 在 train 归一化下 std ≈0.87–0.97,base 优于 predict-mean;
- **生产版 `DeltaAdapterLimited`(PostY/Ada-Y,实例归一化 + 3 层 MLP + BatchNorm,替换式输出)给出 −14.1%**;
- 论文参考(Weather PatchTST+Ada-X+Y):−9.6%(预训练基座)。

方向一致、幅度同量级、机制真实。**Δ 适配器实现忠实,可入矩阵。**

### 1.4 判定
`DELTA_ACCEPT_AS_IS` — 架构取自 vendor/delta_adapter/AdaIntpX/exp_decom9_post_y.py,
测试版本差异(V1/V2)为数据预处理缺陷,非实现缺陷。B3 入矩阵后按协议内部判定,不再依赖 Weather 参考设定。

---

## 2. PIR(PIR_ACCEPT_AS_IS ✅)

### 2.1 复现结果(ETTh1 + PatchTST,官方 full 机制)

复现脚本:`experiments/07-route-e/peers/repro_pir.py`
官方全机制:FailureID + Ridge local revision + **cosine global retrieval** + α/β 融合。

| 指标 | 复现 | 论文参考 |
|---|---|---|
| 基座 MSE | 0.4272 | 0.466 |
| PIR MSE | 0.3983 | 0.437 |
| **相对改善** | **−6.75%** | **−6.22%** |
| PIR MAE | 0.4998 | — |
| MAE 相对改善 | −5.32% | — |

方向一致、量级几乎相同 → 按 §9.1 政策 **`PIR_ACCEPT_AS_IS`**。

### 2.2 诚实 caveat
- PatchTST 基座训练**发散**(va_mse ep0=0.74 → ep90=3.11),best-state 即近初值;
  复现的 PIR 增益测在**弱基座**上,仍复现论文量级 → 机制本身可靠,但与
  "强基座 + PIR"的论文设定不完全同构。
- 复现走的是**含 retrieval** 的官方路径(验证 PIR 概念);生产 `PIRLimited`
  = QualityEstimator + Refiner,**无 retrieval**(limited_official)。
  B4 入矩阵时按 limited_official 标注;有无 retrieval 的差距由 WP-4/6 矩阵内部判定。

### 2.3 判定
`PIR_ACCEPT_AS_IS` — 复现相对改善 −6.75% ≈ 论文 −6.22%。不补 retrieval 工程(§9 peer-fidelity 政策)。

---

## 3. 综合判定(WP-3/B1)

| 方法 | 判定 | 依据 |
|---|---|---|
| B3 δ-Adapter | `DELTA_ACCEPT_AS_IS` | 剔哨兵后 PostY −14.1% vs 论文 −9.6% |
| B4 PIR | `PIR_ACCEPT_AS_IS` | repro −6.75% vs 论文 −6.22% |

两项 peer 均接受,按现有生产实现(DeltaAdapterLimited / PIRLimited)入矩阵,
B4 标注 limited_official(no_retrieval)。不再为 competitor 投入额外工程时间。

