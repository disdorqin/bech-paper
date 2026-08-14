# 06 — Paper Smoke Report(WP-4 / B2)

协议:`hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14.md` §B2
日期:2026-08-14 · git:`5182ef4` · 运行:本地桌面(CPU) · runner:`runner/run_smoke_matrix.py`

目的(协议 §B2):
- 验证 P0-A 后 **real final point 指标**(x_final = s·sinh(z0+π_eff),π_eff 受 DVG 门控);
- 快速暴露 peer(B1–B4)明显 gap;
- 识别 candidate vs readout vs local-layer failure,为 WP-6 FAILURE_MAP 打标做预检。

**当前状态:** 3 市场(LAGO_DE / LAGO_PJM / NEM_SA1)+ **domestic shandong_DA/RT** × Linear/MLP × B0–B6 = **40 cells 全 OK,0 FAIL**。报告 **FINAL**。
Shandong 用 P0A_RERUN 头(与 3MK 一致);WP-5 public 头落盘后 WP-7 domestic 矩阵正式重跑。

---

## 1. 配置

| 项 | 值 |
|---|---|
| 数据窗 | fit=S3M(train+val)、cal=S3C(B2 width cutoff)、eval=S4(dev) |
| HCH head | B5 = P0A_RERUN 通用头(12 cell 源,seed0);B6 = 单域 S2T/S2V 重训 |
| A2 链 | prequential C0/C3 路由 → DVG gate → final point(P0-A 口径) |
| 指标 | MAE / sMAPE 主;RMSE / rMAE / 负价 / 高尾次(原始空间) |
| 输出 | `results/B2_SMOKE_3MK_20260814/smoke_matrix.csv` |

证据窗与 HCH action chain 对齐(公平比较):baseline 与 HCH 共用同一 S3M/S3C/S4 划分。

---

## 2. 主表(MAE,原始空间)

`Δ` 均为相对该 cell host(B0)的改善,负值=更好。

| cell | B0 host | B1 ResL1 | B2 QuantR | B3 δ-Adapt | B4 PIR | B5 Universal | B6 Local |
|---|---|---|---|---|---|---|---|
| LAGO_DE:Linear | 6.249 | **5.233** (−16.3%) | 5.356 (−14.3%) | 6.532 (+4.5%) | 6.199 (−0.8%) | 6.249 (C3·abstain) | 6.100 (−2.4%) |
| LAGO_DE:MLP | 8.417 | **6.602** (−21.6%) | 7.294 (−13.3%) | 7.968 (−5.3%) | 8.196 (−2.6%) | 8.635 (+2.6% ⚠️) | **6.950** (−17.4%) |
| LAGO_PJM:Linear | 6.106 | **5.292** (−13.3%) | 5.292 (−13.3%) | 6.469 (+6.0%) | 7.293 (+19.4% 🔴) | 6.052 (−0.9%) | 6.101 (−0.1%) |
| LAGO_PJM:MLP | 4.210 | 4.777 (+13.5% 🔴) | 4.778 (+13.5% 🔴) | 4.429 (+5.2%) | 4.211 (+0.02%) | 4.273 (+1.5% ⚠️) | 4.210 (0.0%) |
| NEM_SA1:Linear | 65.108 | 129.906 (+99.5% 🔴) | 106.387 (+63.4% 🔴) | 65.918 (+1.2%) | 119.647 (+83.8% 🔴) | 64.871 (−0.4%) | 74.143 (+13.9% ⚠️) |
| NEM_SA1:MLP | 135.878 | 115.936 (−14.7%) | 134.482 (−1.0%) | 133.905 (−1.5%) | 124.534 (−8.3%) | **97.514** (−28.2% 🔥) | 135.878 (0.0%) |
| **shandong_DA:Linear** | 106.930 | **74.143** (−30.7%) | 74.143 (−30.7%) | 77.381 (−27.6%) | 78.018 (−27.0%) | 108.760 (+1.7% ⚠️ C0) | 97.519 (−8.8%) |
| **shandong_DA:MLP** | 159.081 | 141.535 (−11.0%) | 141.535 (−11.0%) | 132.384 (−16.8%) | 138.314 (−13.1%) | 156.324 (−1.7% C3) | 159.081 (0.0% abstain) |
| **shandong_RT:Linear** | 173.987 | 121.419 (−30.2%) | 121.419 (−30.2%) | 114.545 (−34.2%) | **111.860** (−35.7%) | 177.742 (+2.2% ⚠️ C0) | 123.015 (−29.3%) |
| **shandong_RT:MLP** | 275.330 | 195.827 (−28.9%) | 195.827 (−28.9%) | 193.552 (−29.7%) | 194.123 (−29.5%) | 263.782 (−4.2% C0) | **232.606** (−15.5%) |

> LAGO_DE / LAGO_PJM n=10488h;NEM_SA1 n=1752h。NEM_SA1 的 B1/B2/B4 大伤为 S3-M memory 容量(协议 §16:k candidates > memory 14 被 drop)+ 弱 Linear host 所致。
> shandong n=7968h/域(24 点小时价)。`B2_SMOKE_SHANDONG_20260814/07_METRICS_BY_CELL.csv`(P0A_RERUN 头)。

---

## 3. 关键信号(带进 WP-6)

### S1. B5 的 CRPS 路由与 point readout 背离 → Case A 候选 ⭐
P0-A(P0A_RERUN)**全部 16 cell delta_crps 为负**(CRPS 改善),但 B5 point MAE 在:
- `LAGO_DE:MLP` **+2.6%**(8.417 → 8.635)
- `LAGO_PJM:MLP` **+1.5%**(4.210 → 4.273)

两格均为 **C0(raw IAH)**,CRPS 改善却让 final point 变差 → 路由是 CRPS 驱动、点读出不跟随。
→ **FAILURE_MAP 打标 `POINT_READOUT` 候选**,WP-6 用 Case A 诊断(μ_R = w⁺m⁺ − w⁻m⁻ 零参 point readout)验证。

### S2. B5 最强格:NEM_SA1:MLP −28.2% 🔥
135.9 → 97.5,delta_crps=−0.369 **双指标一致**(CRPS + point 都大幅改善)。
→ headline 故事支柱,WP-6 重点复核(确认非 readout 假象)。

### S3. B5 C3-abstain 格 = host 原值(设计正确性自检)
`LAGO_DE:Linear` B5 mae = 6.248619 **精确等于** B0 host。P0-A 约定:DVG 未 release → π_eff=0 → x_final = s·sinh(z0) = host 原预测。✅ 验证 P0-A 设计在 smoke 层正确落盘。

### S4. 残差方法(B1/B2)在强 host 上反伤
- Linear 域:B1 −13~−16%(强)。
- 强 MLP host(LAGO_PJM:MLP MAE 4.21 全矩阵最低):B1/B2 **+13.5% 伤**。
- 弱 host(NEM_SA1:Linear):B1 +99.5% 爆(残差模型拟合到噪声)。
→ 模式:host 越强,残差校正越可能是噪声注入。WP-6 8×4 矩阵可检验此规律。

### S5. peer 方法(B3/B4)在日尺度电价上整体不占优
- B3 δ-Adapter:−5.3% ~ +6.0%,无大赢。
- B4 PIR:`LAGO_PJM:Linear` **+19.4% 大伤**,其余混合。
→ HCH 的 paper 对比叙事 = **B5 的强 cell(NEM_SA1:MLP 等)+ peer 在电价域普遍偏弱**,而非碾压个别 peer。WP-6 ranks 需保留全表(红线:30% 不叫多数)。

### S6. B6 单域 local 不稳定(与执行计划风险行一致)
仅 `LAGO_DE:MLP` 强(−17.4%),NEM_SA1:Linear +13.9% 伤、其余持平。
单域 S2T/S2V 样本太少 → 业务/local 性质,不作 candidate 借口。

### S7. Shandong(domestic unseen):B5-Universal 点估计弱、B6-Local 强 ⭐
- **B5 Universal CRPS 全负**(−0.047/−0.175/−0.085/−0.142),但 point MAE **1/4 格才改善**:`+1.7%(C0) / −1.7%(C3) / +2.2%(C0) / −4.2%(C0)`。
  山东不参与 public-universal 训练(WP-5 只训 8 headline)→ 典型 **Case A 背离**:分布校正有收益、点 readout 不跟随;3/4 格 B5 不 win。
- **B6 Local 反超**:`−8.8% / 0.0%(abstain) / −29.3% / −15.5%`,RT 域 −29.3%/−15.5% 双指标一致(CRPS −0.130/−0.172)。
  domestic 上单域重训匹配度 > universal 泛化 —— 符合预期(未见市场 local 校准的价值)。
- **B1–B4 在 shandong 全格改善**(−11% ~ −35.7%),无伤格 → 与 3MK 的"强 host 反伤"模式不同(山东 host 均较弱,残差空间大)。
→ WP-7 domestic 矩阵主 metric 预期:B5 赢面弱,B6 local 是 domestic 故事;如实保留全表。

---

## 4. FAILURE_MAP 早期打标(草案,WIP)

| cell | B5 | 初判 |
|---|---|---|
| LAGO_DE:MLP | +2.6% | `POINT_READOUT` 候选(delta_crps<0 但 point 伤) |
| LAGO_PJM:MLP | +1.5% | `POINT_READOUT` 候选 |
| NEM_SA1:MLP | −28.2% | `CANDIDATE`(双指标一致) |
| NEM_SA1:Linear | −0.4% | 接近持平,`HOST`(host 本身弱) |
| LAGO_DE:Linear | 0% | abstain 安全;`CANDIDATE`(B6 local 可补) |
| shandong_DA:Linear | +1.7% | `POINT_READOUT`(CRPS −0.047 <0 但 point 伤) |
| shandong_DA:MLP | −1.7% (C3) | `ABSTAIN_SAFE`(≈host)+ CRPS −0.175 |
| shandong_RT:Linear | +2.2% | `POINT_READOUT`(CRPS −0.085 <0 但 point 伤) |
| shandong_RT:MLP | −4.2% (C0) | `CANDIDATE`(point+CRPS 双负,但幅度小)+ B6 local −15.5% 支柱 |

B1–B4 的弱格(如 NEM_SA1 B1/B4)属 peer 方法自身局限,WP-6 记录为 peer gap,不进入 HCH FAILURE_MAP。
B1–B4 在 shandong 全格改善,记录为 peer 在 domestic 的表现,不进 HCH 判定。

---

## 5. 待办

- [x] Shandong DA / RT × Linear/MLP × B0–B6(`B2_SMOKE_SHANDONG_20260814`,P0A_RERUN 头,40 cells 全 OK)
- [x] 本报告标 FINAL,`#63` 置 done
- [x] `runner/run_smoke_matrix.py` → `run_matrix.py`(8 headline × 4 host,parquet + DM + ranks + FAILURE_MAP),推进 WP-6(已验证)
- [ ] **WP-5 头落地后**:shandong B5/B6 用 WP5_PUBLIC 头重跑 → 并入 WP-7 domestic 矩阵正式表
- [ ] **WP-6**:8 headline × 4 host 全矩阵(B5 = WP5_PUBLIC 头),重点复核 S2(NEM_SA1:MLP −28.2%)与 S1/S7 POINT_READOUT 格
