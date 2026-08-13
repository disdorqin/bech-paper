# R1B Stage-2E — action-chain extension(FR / PJM2020 / GEFCOM)报告

- **日期**: 2026-08-14(服务器 2026-08-13 17:33→17:47:44)
- **产物**: `experiments/08-hch-v2/results/R1B_STAGE2E_20260813_173357/`
- **协议**: hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1 §19
- **declared_source_commit**: `3cb6ce986216fdaf3cbfb496df4d64d8e35f8795`
- **parent**: R1B_STAGE2D_20260813_165807(§19 扩展授权)
- **候选**: 同一 frozen LearnedSig_main(seed=0,12 源域;训练 macro_s2v 与 2D 一致)
- **评估面板**: EPEX_FR / PJM_2020 / GEFCOM14P × Linear/PatchTST(6 域,全部为未见市场)

---

## 1. 扩展矩阵(§19/§20)

| 域 | sel | ΔCRPS | A1_net | A2_net | A1→A2 release | A1→A2 harm | mae_rel | safety |
|---|---|---|---|---|---|---|---|---|
| EPEX_FR:Linear | C0 | **−0.041** | +0.041 | +0.041 | 0.299 | 0.333 | 0.0 | ok |
| EPEX_FR:PatchTST | **C3** | **−0.043** | **+0.149** | **+0.041** | 0.552→0.090 | 0.189→0.111 | 0.0 | ok |
| PJM_2020:Linear | C0 | **−0.058** | +0.003 | +0.003 | 0.091 | 0.306 | 0.0 | ok |
| PJM_2020:PatchTST | C0 | **−0.037** | +0.007 | +0.007 | 0.071 | 0.286 | 0.0 | ok |
| GEFCOM14P:Linear | C0 | **−0.037** | +0.000 | +0.000 | 0.009 | 0.0 | 0.0 | ok |
| GEFCOM14P:PatchTST | C0 | **−0.063** | +0.000 | +0.000 | 0.014 | 0.0 | 0.0 | ok |

- **候选转移**:6/6 ΔCRPS<0(−0.037~−0.063),MAE 恒等。与 2A→2D 全部面板一致。
- **ext macro**:A1 +0.0335 → A2 +0.0155。全 6 域 A2>0,无一域绝对亏损。

## 2. Selector(§19 规则)

| 域 | S3M 日 | OOS 日 | sel | reason | Gate A/B/C/D | LCB90 |
|---|---|---|---|---|---|---|
| EPEX_FR:Linear | 50 | 20 | C0 | C3_VALUE_NOT_BETTER | T/T/F | — |
| EPEX_FR:PatchTST | 50 | 20 | **C3** | PREQUENTIAL_AUTHORIZATION_C3 | T/T/T/T | +0.0169 |
| PJM_2020:Linear | 98 | 68 | C0 | C3_VALUE_NOT_BETTER | T/T/F | — |
| PJM_2020:PatchTST | 98 | 68 | C0 | RAW_HEALTHY_KEEP_C0 | T/F | — |
| GEFCOM14P:Linear | 54 | 24 | C0 | C3_IMPROVEMENT_UNCERTAIN | T/T/T/F | 0.0 |
| GEFCOM14P:PatchTST | 54 | 24 | C0 | C3_VALUE_NOT_BETTER | T/T/F | — |

**SELECTOR_RESPECTED = True**(无"证据不足还授权"的违规;未降 gate)。

## 3. §21 Safety

- 全 6 域 `mae_rel = 0.0` → **SAFETY_FAILURE: False**。

## 4. STOP 判定

```text
SAFETY_FAILURE:          False
EXT_ACTION_COLLAPSE:     False   (ext macro A2 +0.0155 > 0; 无域 A2 < -0.01)
GATING_HURTS:            True    (EPEX_FR:PatchTST A2=+0.041 < A1=+0.149 - 0.01)
SELECTOR_RESPECTED:      True
```

**内部守卫触发(非协议硬 STOP)。**

## 5. GATING_HURTS 事故链 — EPEX_FR:PatchTST(重点记录)

- 该域仅 **50 个 S3M 日 → 3 个 rolling block → 20 个 OOS 日**。
- prequential 20 天证据:裸动作被判有问题(Gate B=T)、C3 判定更好(LCB90=+0.0169,mean Δ=+0.035)→ 授权 C3。
- **部署的 C3 是全量 S3M 拟合的映射**,而非 rolling-fold 版本。全量拟合 C3 漂移到近乎全弃权(release 0.552→0.090),S4 实测 A2=+0.041,把裸动作 +0.149 的收益丢 0.108。
- **根因**:短证据域(≈50 S3M 日)上,rolling C3 评估 ≠ 部署 C3 行为。这是 §19"local evidence 太短"的精确场景 —— EPEX_FR:PatchTST 的 50 日刚好过 Gate A 门槛(30+14=44),未被 default C0,结果烧掉 0.108。
- **同类现象在 2D 已现(DK1:PatchTST,50 日,A2 0.078 < A1 0.084,差 0.006 未过 0.01 阈值)**,EPEX_FR:PatchTST 是放大版。
- **§22 归类**:LOCAL_ADAPTATION_LIMITED(候选好 + 局部适配在该域不成立;不回改 universal candidate)。候选泛化不受影响。

## 6. 结论与决策

- **候选转移:6/6 全绿**(延续 2A→2E 共 71+ 域评估无一例外)。
- **动作链:net-positive 但有一处门控误授权**(EPEX_FR:PatchTST),记入 caveat。
- **FINAL VERDICT 输入**:该 caveat 纳入 §30 判定;GATING_HURTS 不影响"candidate 泛化方向一致"主结论,但提示 C3 短证据授权需更保守(报告建议:短证据域或需更高 OOS 门槛 / 部署前校验 full-fit map 与 rolling map 的 D_map)。

## 7. §24 GENERALIZATION_LEDGER v2 条目

```text
experiment_id:            R1B_STAGE2E_20260813_173357
candidate_variant:        LearnedSig_main (frozen, seed=0)
training_market_set:      LAGO_DE, LAGO_PJM, NEM_SA1
evaluation_dataset:       6 extension (EPEX_FR/PJM_2020/GEFCOM14P x Linear/PatchTST)
transfer_category:        UNSEEN_MARKET
source_macro_delta:       (见 2D ledger)
holdout_delta:            ext ΔCRPS 6/6 负, macro -0.046
worst_domain_delta:       EPEX_FR:PatchTST A2 +0.041(仍为正;门控误授权丢 0.108)
seed_consistency:         3/3 (Stage-2C)
final_mae_effect:         0.0
safety_effect:            SAFETY_FAILURE=False; GATING_HURTS=True (EPEX_FR:PatchTST)
complexity_added:         无
accepted_for_universal_core: PENDING → 取决于 Local-Core + FINAL VERDICT
notes:                    C3 短证据域误授权;候选 transfer 6/6 健康
```

## 8. 下一步

§23 Local-Core(用户已确认继续):`R1B_STAGE2F_*` 运行中 → 完成后 → §30 R1B FINAL VERDICT。
