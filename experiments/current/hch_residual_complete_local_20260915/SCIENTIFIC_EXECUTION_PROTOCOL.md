# HCH Residual-Complete Local v4.1 — Scientific Execution Protocol

Protocol ID: `HCH_RCL_V4_1_SCIENTIFIC_20260915`

Status: **USER-AUTHORIZED, CONDITIONAL ON FINAL PRE-EXECUTION GATE**

Authority chain:

1. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
2. `docs/current/HCH_RCL_V4_1_FINAL_PREEXECUTION_AUDIT_AND_EXECUTION_DESIGN_20260915.md`
3. this protocol

If this protocol conflicts with the earlier preparation `PROTOCOL.md`, this file governs the actual scientific run.

## 1. Required Phase 0 — bounded final patch

Before any fit, patch and test exactly the A–H items in the final audit:

- Stage-1 scale leakage and `s_b` semantics;
- Stage-2 seed-before-model-construction;
- final Level 3-seed median ensemble matching OOF aggregation;
- B1 q-history warm-start for B2 targets;
- native R2 A+/A− targets/loss;
- R0/R3 matched-control assertions;
- corrected metric arithmetic;
- corrected CAL metric naming/arithmetic.

No scientific fit is legal before all Phase-0 tests pass.

Required gate:

`RCL_V4_1_FINAL_PREEXECUTION_GATE_PASS`

If any check fails:

`RCL_V4_1_FINAL_PREEXECUTION_GATE_FAIL_STOP`

and stop.

## 2. Data boundary

Only `COMMON_BENCHMARK_701020_FULL_V2` TRAIN+VAL frozen host caches.

Cells exactly:

- GANSU_DA/LSTM
- GANSU_DA/TimeMixer
- GANSU_DA/PatchTST
- SHANDONG_DA/TimeMixer
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/PatchTST
- NINGXIA_DA/LSTM

Forbidden:

- TEST
- `host_predictions_joint.npz`
- QINGHAI
- foreign/final
- Host retraining
- baseline rerun
- split changes
- `src/core` edits during scientific execution
- paper edits

The loader must verify the non-joint cache row count equals metadata `fit_n + valid_n` and its segment vocabulary contains no TEST row.

## 3. Stage 1 — Coarse Level

Target:

`b* = mean(r)`.

OOF chronology:

- first 40% TRAIN = warm-up;
- remaining 60% = B1/B2/B3/B4;
- each block predicted only by strictly earlier days;
- inner 15% prefix tail for early stopping;
- seeds 7/17/37.

For every fold:

- history normalization scale uses only fold `fit_ids` raw residual history geometry;
- Level target scale `s_b` uses only `b*` over fold `fit_ids`;
- no future block / inner-early / VAL target may fit these scales.

OOF prediction for each day = coordinate-wise median over seeds 7/17/37.

Final Level:

- train three final refits with seeds 7/17/37;
- retain all three states;
- freeze all parameters;
- inference Level = coordinate-wise median of the three frozen outputs;
- final history scale and `s_b` fit only from final `fit_ids`;
- same architecture fingerprint as OOF models.

Fit count: 96 OOF + 24 final = 120 Stage-1 fits.

## 4. Stage 2 support and history

Targets only on:

`LOCAL_TRAIN = B2 ∪ B3 ∪ B4`.

B1 never enters loss/normalization/early stopping but is legal causal q-history warm-start.

For target day t:

`q_t = r_t - b_tilde_t * 1`.

History is the most recent up to 7 already-defined causal q days, including B1 where applicable.

Rho history is centered q history.

Level conditioning stats are fit only from causal OOF `b_tilde` on LOCAL_TRAIN.

Local q/delta/A stats are fit only from LOCAL_TRAIN q geometry.

## 5. Variants

Seeds: 7/17/37.

### R0 `STACK_ZERO_SUM`

`z = A(S+ - S-)`.

Uses the exact same Shape backbone, magnitude backbone, Level condition and `BalancedAmplitudeBranch` primitive as R1.

### R1 `RCL_ORTHOGONAL` — primary

`z = delta*1 + A(S+ - S-)`.

R1 must import active `core.ResidualCompleteLocalModel` directly.

### R2 `UNTIED_Q`

`z = A+ S+ - A- S-`.

Native targets:

- `A+* = sum(q+)`
- `A-* = sum((-q)+)`
- q-positive/q-negative simplex Shapes.

Loss:

`L = L_rec(q,z) + mean(L_A+, L_A-, L_shape_q)`.

No delta loss and no centered-A loss for R2.

### R3 `DIRECT_Q`

Low-capacity direct H-step q vector.

Must consume:

- rho-history Shape latent;
- q-history magnitude latent;
- normalized coarse Level condition.

No dead Shape-simplex head.

Parameter count <=1.25× R1.

Stage-2 fit count: `8 × 4 × 3 = 96`.

Total scientific fit registry: `216`.

## 6. Training recipe

All Stage-2 variants:

- seed before model construction;
- AdamW;
- lr 1e-3;
- weight decay 1e-4;
- batch size 32;
- max epochs 50;
- patience 8;
- grad clip 1.0;
- same LOCAL_TRAIN support;
- same early-stop tail;
- no hyperparameter sweep.

Engineering parallelism may change scheduling only, never data order, seeds, recipe or reduction rules.

## 7. VAL evaluation

VAL split chronologically 50/50:

- CAL first half;
- EVAL second half.

For every day in order:

1. frozen Level median-ensemble prediction;
2. Local prediction from already-revealed q-history;
3. persist emission;
4. reveal target;
5. compute q;
6. append q history.

Primary verdict = raw output only.

One nonnegative scalar alpha may be fitted on CAL and reported as secondary diagnostic only.

## 8. Correct metric definitions

Let:

- `r = y - host`
- `c = full correction`
- `e_H = r`
- `e_R = r - c`
- `q = r - b_hat*1`
- `z_hat = Local output`.

Report:

- Host Overall MAE = `mean|e_H|`
- Repaired Overall MAE = `mean|e_R|`
- gain% = `(Host-Repaired)/Host*100`
- q reconstruction MAE = `mean|q-z_hat|`
- Stage-1 floor = `mean|mean(r)-b_hat|`
- delta closure error = `mean|mean(r)-b_hat-delta_hat|`
- delta MAE vs zero-delta reference
- A MAE against q-centered target A
- Shape W1 against q-centered signed Shapes.

TRAIN-only threshold masks:

- upper: target >= TRAIN q95
- lower: target <= TRAIN q05
- extreme = upper ∪ lower
- failure-tail day: Host daily MAE >= TRAIN daily-error q90
- normal = not upper/lower.

Subset metrics always score `|e_R|`.

Normal harm:

`mean(max(0, |e_R|-|e_H|))[normal]`.

Secondary CAL score:

`mean|r_CAL-alpha*c_CAL|`.

No metric may compare repaired price directly with residual.

## 9. Aggregation

Per cell metric = median across seeds 7/17/37.

Panel summary = median across 8 cells.

No winner-specific seed selection.

## 10. Primary adjudication

### Residual completeness R1 vs R0

`RCL_RESIDUAL_COMPLETENESS_SUPPORTED` iff:

- >=5/8 positive cells;
- panel median gain >=0.75%;
- worst degradation >=-0.75%;
- delta beats zero-delta >=5/8;
- panel median mean-floor reduction >=25%.

Else MIXED/NOT_SUPPORTED.

### R1 vs R2

- R1 non-worse >=5/8 and panel median >=0 → `ORTHOGONAL_COORDINATES_PREFERRED`;
- R2 better >=5/8 and median >=0.5% → `UNTIED_SIGNED_MASS_PREFERRED`;
- else `LOCAL_PARAMETERIZATION_MIXED`.

### R1 vs R3

- R3 better >=5/8 and panel median >=1% → `STRUCTURED_LOCAL_UNDERFIT_RISK`;
- R1 within 0.5% in >=6/8 and matched capacity → `STRUCTURED_LOCAL_COMPETITIVE`;
- else MIXED.

### R1 vs Host

Development target:

- positive >=7/8;
- panel median gain >=1.0%;
- worst loss >=-0.5%.

Do not interpret this as baseline competitiveness.

## 11. Required evidence

Evidence root:

`experiments/evidence/hch_residual_complete_local_v4_1_20260915/`

Required at minimum:

- `00_protocol/PREEXECUTION_GATE.json`
- `00_protocol/RUN_MANIFEST.json`
- OOF fold/seed records
- final Level ensemble manifests and checkpoint hashes
- Level-condition and Local-normalization manifests
- per variant×cell×seed checkpoints
- per day VAL emission logs
- per fit raw metrics/diagnostics
- cell median table
- R1-vs-R0/R2/R3 tables
- Host comparison table
- tail/extreme table
- parameter/runtime table
- read-set audit
- source/core tree hashes before/after
- `VERIFICATION_REPORT.json`
- `RESULTS.md`

## 12. Independent verifier

Must independently recompute all gate-bearing metrics and verify:

- exact 216-fit registry;
- all causal supports;
- scale-fit supports;
- seed-before-construction;
- OOF/final 3-seed median Level rule;
- B1 history-only role;
- B2-B4 target role;
- R2 native mass supervision;
- raw metric arithmetic and harm sign;
- TEST/joint path absence;
- core unchanged during scientific run;
- all terminal labels.

## 13. Terminal state

After independent verification PASS, output:

`HCH_RCL_V4_1_SCIENTIFIC_EXECUTION_COMPLETE_FOR_ADJUDICATION`

Then stop. No TEST, no foreign/final, no paper update, no new rescue variant.