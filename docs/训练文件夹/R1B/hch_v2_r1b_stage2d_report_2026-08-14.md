# R1B Stage-2D — Full action-chain on canonical 16 报告

- **日期**: 2026-08-14(服务器 2026-08-13 16:58 / 17:18:45 完成)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2D_20260813_165807/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §17-§22
- **declared_source_commit**: `979d80e272140ad6d677eb8bf1f6aeecfd2bdd77`
- **parent**: R1B_STAGE2C_20260813_163143(3-seed STABLE → §16 授权)
- **候选**: LearnedSig_main,frozen,seed=0,12 source 域训练(macro_s2v 0.2637→0.2253,与 Stage-2A 逐位一致)
- **评估面板**: canonical 16 = 3 source × 4 hosts + NORD_DK1 × 4 hosts(§18)

---

## 1. 三策略定义(§17)

| 策略 | 内容 |
|---|---|
| A0 | Host Identity(pi=0,不动;net≡0) |
| A1 | Raw IAH 动作:C0 解析效用 → Double Event → 每日 A_hat → S3C DVG → S4 释放 |
| A2 | Evidence-gated 局部校准:R1A.11 prequential C0/C3(gates A/B/C/D,default-to-C0)→ 选定效用映射 → 同一 Double Event 链 |

---

## 2. §20 候选 CRPS + 动作链矩阵(canonical 16)

| 域 | 类别 | host_CRPS | cand_CRPS | ΔCRPS | sel | A1_net | A2_net | A1→A2 release | A1→A2 harm |
|---|---|---|---|---|---|---|---|---|---|
| LAGO_DE:Linear | SRC | 0.250 | 0.162 | **−0.089** | C0 | −0.020 | −0.020 | 0.307 | 0.634 |
| LAGO_DE:MLP | SRC | 0.247 | 0.142 | **−0.104** | C0 | +0.026 | +0.026 | 0.238 | 0.394 |
| LAGO_DE:LSTM | SRC | 0.192 | 0.141 | **−0.051** | **C3** | −0.022 | **0.000** | 0.229→0.0 | 0.680→— |
| LAGO_DE:PatchTST | SRC | 0.268 | 0.153 | **−0.115** | C0 | +0.011 | +0.011 | 0.197 | 0.395 |
| LAGO_PJM:Linear | SRC | 0.116 | 0.078 | **−0.038** | C0 | +0.001 | +0.001 | 0.043 | 0.421 |
| LAGO_PJM:MLP | SRC | 0.059 | 0.047 | **−0.012** | C0 | −0.002 | −0.002 | 0.043 | 0.632 |
| LAGO_PJM:LSTM | SRC | 0.064 | 0.049 | **−0.015** | C0 | −0.003 | −0.003 | 0.037 | 0.875 |
| LAGO_PJM:PatchTST | SRC | 0.082 | 0.056 | **−0.026** | C0 | −0.002 | −0.002 | 0.025 | 0.636 |
| NEM_SA1:Linear | SRC | 0.821 | 0.279 | **−0.542** | C0† | +0.015 | +0.015 | 0.068 | 0.200 |
| NEM_SA1:MLP | SRC | 0.824 | 0.455 | **−0.369** | C0† | +0.277 | +0.277 | 0.411 | 0.133 |
| NEM_SA1:LSTM | SRC | 0.911 | 0.679 | **−0.231** | C0† | +0.020 | +0.020 | 0.027 | 0.0 |
| NEM_SA1:PatchTST | SRC | 0.850 | 0.460 | **−0.390** | C0† | +0.007 | +0.007 | 0.027 | 0.0 |
| NORD_DK1:Linear | **TR** | 0.219 | 0.179 | **−0.040** | C0 | +0.000 | +0.000 | 0.144 | 0.448 |
| NORD_DK1:MLP | **TR** | 0.328 | 0.219 | **−0.109** | C0 | +0.084 | +0.084 | 0.398 | 0.250 |
| NORD_DK1:LSTM | **TR** | 0.195 | 0.143 | **−0.052** | C0 | +0.053 | +0.053 | 0.249 | 0.300 |
| NORD_DK1:PatchTST | **TR** | 0.211 | 0.138 | **−0.073** | **C3** | +0.084 | **+0.078** | 0.413→0.348 | 0.289→0.271 |

† NEM_SA1 4 域:仅 19 个 S3M 日 → Gate A 证据不足,default C0(§22 SAFE_FEW_EVIDENCE_FALLBACK)。

## 3. §20 Selector(逐域)

| 域 | S3M 日 | OOS 日 | 选择 | reason | Gate A/B/C/D | LCB90 |
|---|---|---|---|---|---|---|
| LAGO_DE:Linear | 109 | 79 | C0 | C3_IMPROVEMENT_UNCERTAIN | T/T/T/**F** | −9.4e−5 |
| LAGO_DE:MLP | 109 | 79 | C0 | C3_IMPROVEMENT_UNCERTAIN | T/T/T/**F** | −5.7e−4 |
| LAGO_DE:LSTM | 109 | 79 | **C3** | PREQUENTIAL_AUTHORIZATION_C3 | T/T/T/**T** | **+3.6e−3** |
| LAGO_DE:PatchTST | 109 | 79 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| LAGO_PJM:Linear | 109 | 79 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| LAGO_PJM:MLP | 109 | 79 | C0 | C3_VALUE_NOT_BETTER | T/T/F | — |
| LAGO_PJM:LSTM | 109 | 79 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| LAGO_PJM:PatchTST | 109 | 79 | C0 | C3_VALUE_NOT_BETTER | T/T/F | — |
| NEM_SA1×4 | 19 | 0 | C0 | INSUFFICIENT_PREQUENTIAL_EVIDENCE | F | — |
| NORD_DK1:Linear | 50 | 20 | C0 | C3_IMPROVEMENT_UNCERTAIN | T/T/T/**F** | −0.012 |
| NORD_DK1:MLP | 50 | 20 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| NORD_DK1:LSTM | 50 | 20 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| NORD_DK1:PatchTST | 50 | 20 | **C3** | PREQUENTIAL_AUTHORIZATION_C3 | T/T/T/**T** | **+0.037** |

Gate D(LCB90>0)精确分辨:2/16 授权 C3(DE:LSTM、DK1:PatchTST),LCB 恰为 +;其余 C0 的 Gate-D 失败者 LCB 均 ≤0(最小阈值差 −9.4e−5,恰好挡住)。**无一次越界授权**。

## 4. §21 Safety red line

- 全部 16 域 `mae_rel = 0.0`(candidate point = host,MAE 恒等)→ **SAFETY_FAILURE: False**。
- continuous degradation 全为 0.0,无隐藏退化。

## 5. §20 DVG 关键观察

- **LAGO_DE:LSTM(A2)**:release 0.0 / identity 1.0 / net 0.0 / coverage 1.0 —— C3 局部映射在 S4 完全弃权,**把裸动作的 −0.022/日 亏损直接归零**(门控最干净的 safety win)。
- **NORD_DK1:PatchTST(A2)**:release 0.413→0.348、harm 0.289→0.271、mean_gain|release 0.204→0.225。C3 释放更少但每次释放赚更多,q 0.120→0.246(更保守 LCB)。net 0.084→0.078(略降,因释放少),但转移域上 C3 授权成立。
- **NEM_SA1:MLP**:A1 net **+0.277** 全场最高,release 0.41 / harm 0.13,但 q=0.712(粗动作高度过置信)、S3M 仅 18 日 → 停在 C0。S4 coverage 0.93。
- **coverage caveat**:NEM_SA1:Linear 0.79、NEM_SA1:PatchTST 0.81 低于名义 0.90 —— 与 R1A S4 gate caveat 一致(NEM 样本少),此处仅作记录,不构成 STOP。

## 6. §20 Forecast(点预测 = host)

| 域 | MAE | rMAE | RMSE | no-floor sMAPE | neg-price率 | 高尾率 |
|---|---|---|---|---|---|---|
| LAGO_DE:MLP | 8.42 | 0.235 | 12.56 | 36.1 | 1.9% | 5.0% |
| LAGO_PJM:PatchTST | 4.09 | 0.144 | 6.11 | 15.8 | 0.6% | 5.0% |
| NEM_SA1:MLP | 135.9 | 1.45 | 227.0 | 136.6 | 35.2% | 5.0% |
| NORD_DK1:MLP | 74.3 | 1.14 | 90.0 | 85.7 | 7.0% | 5.0% |

(逐域完整 10 项指标见 `action_chain_matrix.csv`。)

## 7. §22 结果解释 + STOP

| 模式 | 域 | 说明 |
|---|---|---|
| LOCAL_ADAPTATION_LIMITED(candidate 好 + action 差) | LAGO_DE:Linear、LAGO_PJM:MLP/LSTM/PatchTST | ΔCRPS 显著为负,但裸动作净值为负或 ≈0;局部证据不足,门控无法修复(Gate D LCB≤0 或 Gate C 不改善)。**不回改 universal candidate**(§22)。 |
| SAFE_FEW_EVIDENCE_FALLBACK(candidate 好 + 证据不足 default C0) | NEM_SA1×4 | S3M 仅 19 日 → Gate A 默认 C0,raw 动作本身为正 → 安全。 |
| 门控 rescue | LAGO_DE:LSTM | −0.022 → 0.0(纯弃权) |
| 转移域 C3 授权 | NORD_DK1:PatchTST | LCB +0.037,harm 改善 |

**STOP 判定(§21/§22):**

```text
SAFETY_FAILURE:             False
SOURCE_ACTION_UNHEALTHY:    False   (source macro A2 = +0.0276 > 0)
TRANSFER_ACTION_COLLAPSE:   False   (DK1 macro A2 = +0.0539,全正)
GATING_HURTS:               False   (唯一 C3 授权域 A2 ≥ A1 − 0.01)
```

## 8. VERDICT

```text
CONTINUE — canonical-16 action chain healthy
```

- source macro:A1 +0.0257 → A2 +0.0276(门控净增 +0.0018;rescue 贡献来自 DE:LSTM −0.022→0)
- DK1 转移 macro:A1 +0.0554 → A2 +0.0539(转移域动作链同样为正,MLP/LSTM/PatchTST 均 ≥+0.05)
- 训练与 Stage-2A 逐位一致(同 seed0 12 源域),链式评估可复现
- **§19 授权 → action-chain extension:EPEX_FR / PJM_2020 / GEFCOM14P × Linear/PatchTST(6 域)**

## 9. §24 GENERALIZATION_LEDGER v2 条目

```text
experiment_id:            R1B_STAGE2D_20260813_165807
candidate_variant:        LearnedSig_main (frozen, seed=0)
training_market_set:      LAGO_DE, LAGO_PJM, NEM_SA1
training_host_set:        Linear, MLP, LSTM, PatchTST
evaluation_dataset:       canonical 16 (12 source + 4 NORD_DK1)
transfer_category:        SOURCE_SEEN / UNSEEN_MARKET (DK1)
source_macro_delta:       ΔCRPS −0.143 (12 源域均值);A2 macro +0.0276
holdout_delta:            DK1 ΔCRPS −0.069;A2 macro +0.0539
worst_domain_delta:       LAGO_DE:Linear A2 −0.020(动作);ΔCRPS 全负
seed_consistency:         (Stage-2C) 3/3 sign consistent
final_mae_effect:         0.0 (point=host, 恒等)
safety_effect:            SAFETY_FAILURE=False; 2 C3 授权域无越界 (LCB>0 ⇔ 授权)
complexity_added:         prequential C0/C3 选择器 (R1A.11, 不加新超参)
accepted_for_universal_core: PENDING → 取决于 Stage-2E 扩展 + Local-Core
notes:                    LAGO_DE:Linear/PJM×3 动作净值为负但量级小;NEM coverage <0.90 caveat
```

## 10. 下一步

按 §29:→ **Stage-2E action-chain extension**(FR/PJM2020/GEFCOM × Linear/PatchTST,6 域,selector 证据不足必须 default C0,不降 Gate)→ Local-Core upper bound → R1B FINAL VERDICT(§30)。
