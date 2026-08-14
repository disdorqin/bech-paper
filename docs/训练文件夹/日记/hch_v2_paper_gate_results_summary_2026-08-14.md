# HCH-v2 Paper Benchmark Gate — 完整实验总结(2026-08-14)

> 阶段终态:**PAPER_GATE_YELLOW_READOUT** · 硬 STOP(协议 §20)· 交人类评审
> 分支:`exp/r1b-screening-20260813` · git `5182ef4`
> 驱动文档:`hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14.md`
> 执行计划:`hch_v2_paper_gate_execution_plan_2026-08-14.md`(WP-0 … WP-8)

---

## 1. 背景与目标

R1B 已证明 HCH-v2 在**CRPS / 分布层**的泛化(7 阶段 100+ 评估、ΔCRPS<0 无例外)。Paper Benchmark Gate 回答更残酷的问题:**它打不打得过同行**——在 8 个 foreign headline 数据集 × 4 host 上,以 final point MAE/sMAPE 对 B0–B6 六类方法做正面比较。内部最低线 Top-1/tied ≥60%(GREEN)、≥70%(STRONG GREEN);山东(domestic)≥75%。规则:30% 不叫多数,完整表格必须留下,调参不碰 sealed test。

## 2. 实验结构

- **方法矩阵**:B0 Identity · B1 ResidualL1 · B2 QuantileResidual · B3 δ-Adapter · B4 PIR · B5 HCH-Universal · B6 HCH-Local
- **8 headline**:LAGO_DE/BE/FR/PJM/NP · NEM_SA1 · GEFCOM14P · NORD_DK1;4 host:Linear/MLP/LSTM/PatchTST
- **证据窗口**:fit=S3M(train+val) · cal=S3C · eval=S4;HCH 输出 = P0-A replay `x_final = s·sinh(z0+π_eff)`
- **Primary cell**:dataset × host × {MAE, sMAPE} = 64(foreign)+ 16(山东)

## 3. 前置资产(WP-0 … WP-4)

| WP | 内容 | 结果 |
|---|---|---|
| WP-0 | 数据/domestic/host manifest | ✅ `02/03/04` 审计就位 |
| WP-1 | P0-A final replay 统一 + 3 回归测试 | ✅ 口径收敛 |
| WP-2 | provenance + PatchTST-style 命名 | ✅ |
| WP-3 | B1 PIR/δ 保真 | ✅ PIR_ACCEPT_AS_IS / DELTA_ACCEPT_AS_IS |
| WP-4 | B2 smoke(3MK + shandong) | ✅ 40 cells 全 OK |

## 4. WP-5 universal 重训与 checkpoint guard(关键决策点)

按协议 §12.1 对 **8 headline × 4 host = 32 域**重训共享 LearnedSig 头(hierarchical balancing, L=L_IAH-CRPS),§12.2 guard 对照 P0A_RERUN 头逐域检查(≤2% / worst ≤3% / anchor ≤2%)。

**三轮全部机器 ROLLBACK:**

| 轮次 | 采样 | macro | worst | 超限域 | 结果 |
|---|---|---|---|---|---|
| 052544 | p=0(等量) | +0.03% | 4.94% | 8 | ROLLBACK |
| **053855** | **p=0.5 温度采样** | **−1.30%** | **4.42%** | **4**(GEFCOM14P:PatchTST +4.42% / MLP +3.27% / NEM_SA1:Linear +2.79% / LAGO_DE:MLP +2.44%) | ROLLBACK → **用户豁免 promote** |
| 054921 | p=1.0 | +0.92% | 10.84% | 8 | ROLLBACK(过头) |

**用户决策(2026-08-14)**:接受 p=0.5 头继续实验,豁免 §12.2 guard 2/3。已在 `WP5_PUBLIC_20260814_053855/guard_report.json` 写入 waiver_record(auto_pass=false, user_waiver=true),完整逐域回归数据保留。**该 caveat 必须在任何论文/verdict 中列出。**

## 5. WP-6 B3 foreign headline 矩阵(seed0)

产物:`experiments/09-paper-gate/results/B3_MATRIX_20260814_105940/`(06–16 + config,32 cells × 7 methods)

| gate 指标 | 值 | §15.1 |
|---|---|---|
| Top-1/tied | **22/64 = 34.4%**(strict 15 + tied 7) | YELLOW(30–60%) |
| Top-2 | 19/64 = 29.7% | GREEN 需 ≥85% |
| host-better | 46.9% | GREEN 需 ≥90% |
| MAE safety failure | 0 | ✅ |

**逐 dataset(B5 赢格/8)**:LAGO_BE 6 · NEM_SA1 3+tied · NORD_DK1 3+2top2 · LAGO_NP 2 · LAGO_PJM 2 · LAGO_FR 1+tied · GEFCOM14P 1+tied · **LAGO_DE 0(B1_ResidualL1 大胜,gap 12–33%)**。

**失败分解**:14 × POINT_READOUT(ΔCRPS<0 但 point 反伤)→ 主导;12 × CANDIDATE;4 × NEUTRAL;2 × ABSTAIN_SAFE。

## 6. WP-7 B4 domestic 矩阵(seed0)

产物:`experiments/09-paper-gate/results/B4_MATRIX_SHANDONG_20260814/`

| gate 指标 | 值 | §15.2 目标 |
|---|---|---|
| Top-1/tied | **0/16 = 0%** | ≥75% → **FAIL** |
| host-better | 50% | |
| worst gap | 53.3%(shandong_RT:Linear MAE) | |
| CRPS | **8/8 ΔCRPS<0**(−0.03…−0.18) | |

16 格全部 selected=C3;best 方法全为 peer(B1/B4/B3);HCH 3 CANDIDATE(−1.4~−1.6%)+ 3 ABSTAIN_SAFE(≡host)+ 2 POINT_READOUT。**不含** CN_ningxia/gansu/shaanxi/qinghai(secondary,headline_eligible=False,无 prep)与 96pt xlsx(用户排除)。

## 7. WP-8 Verdict + 硬 STOP

文档:`experiments/09-paper-gate/results/PAPER_GATE_VERDICT_20260814.md`

```text
PAPER_GATE_YELLOW_READOUT
```

**统一解读**:CRPS 层 **40/40 全赢**(foreign 32 + domestic 8,ΔCRPS<0)——R1B 分布泛化在矩阵尺度完整延续;坏的是 **point 读出层**(`z^point`),16/40 反伤,正是协议 §16 **Case A**。domestic 全 C3 + 大量 abstain 另指向 **Case E**(本地授权过保守)。

**G. 修改建议(≤3,不自动执行)**:
1. Case A:μ_R = w⁺m⁺ − w⁻m⁻ 点读出(0 新增参数)+ λ 收缩网格 {0.25,0.5,0.75,1.0},仅 val 选
2. Case E:本地点读出与安全动作校准分离(identity-anchored/shrunk isotonic)
3. WP-5 豁免固化:今后重训必须全过 §12.2

**硬 STOP(§20)**:不启动 U0 / 大扩数据 / 新 FM / 新架构 / sealed test。交人类评审:freeze | 一个诊断驱动修改 | core review。

## 8. 关键发现(论文层面)

1. **分布层≠点层**:CRPS 全赢但 point 反伤 → IAH probabilistic core 是真的,`distribution→point` 读出太保守。三原子的 expected residual(w⁺m⁺−w⁻m⁻)是现成的第一刀,无需动 CRPS 或加 loss。
2. **陌生难域(山东)**:universal 头选择保守(大量 abstain=C3),简单残差法反而最好 → 本地校准层没有兑现 point 收益。
3. **peer 强弱**:B1_ResidualL1(LAGO_DE 碾压)/ B4_PIR(FR/NP 强)/ B3(山东)都是难缠对手;HCH 赢在分布质量,输在点读出对齐。
4. **方法叙事**:v0.4 早期 CAGM/W1 已被淘汰,留下主线 = host-relative signed distribution · data-signature modulation · evidence-authorized local calibration · structured event correction · abstention。

## 9. 复现路径

```bash
# WP-5 universal 重训(需 host cache)
python experiments/09-paper-gate/runner/retrain_public_universal.py
# WP-6 B3 foreign(8 headline × 4 host × B0–B6)
python experiments/09-paper-gate/runner/run_matrix.py
# WP-7 B4 domestic(山东 DA/RT)
python experiments/09-paper-gate/runner/run_matrix.py --out results/B4_MATRIX_... --datasets shandong_DA shandong_RT
```

产物清单(协议 §18):06_PREDICTIONS.parquet · 07_METRICS · 08_RANKS · 09_WIN_RATE · 12_DM · 13/14 诊断 · 15_FAILURE_MAP · 16_LEDGER · 17_VERDICT · config。

## 10. 决策记录存档

- WP-5 guard 豁免:用户 2026-08-14 接受 p=0.5 头(见 §4 + guard_report.json waiver_record)
- 山东范围:只测 `shandong_pmos_hourly.csv`(24 点);96pt xlsx 排除(用户决定)
- 私有边界:Shandong/private 永不混入 public-universal 训练;山东仅 frozen transfer 展示
- checkpoint:`D:\AI_Memory\checkpoints\checkpoint_20260814_{111500,115000,121500}.json`
