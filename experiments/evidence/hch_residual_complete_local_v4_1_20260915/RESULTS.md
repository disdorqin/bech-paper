# RCL v4.1 — Residual-Complete Local, scientific execution

Protocol `HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_20260915` · evidence root `experiments/evidence/hch_residual_complete_local_v4_1_20260915/`

Registered neural fits: **216** (96 stage-1 OOF + 24 final-Level refits + 96 Stage-2), against an expected 216.

Independent verification: **PASS** — 4733 checks, 0 failed, 68 supplementary defects (characterised in §8).

## 1. Cell median table (repaired Overall MAE, median across seeds)

| cell | Host | R0 | R1 | R2 | R3 | R1 gain vs Host % | R1 params |
|---|---|---:|---:|---:|---:|---:|---:|
| GANSU_DA | LSTM | 72.82 | 71.83 | 73.54 | 71.26 | +11.503 | 23332 |
| GANSU_DA | TimeMixer | 71.34 | 71.10 | 71.35 | 69.42 | -1.995 | 23332 |
| GANSU_DA | PatchTST | 73.06 | 72.19 | 73.72 | 72.45 | +1.130 | 23332 |
| SHANDONG_DA | TimeMixer | 67.06 | 67.32 | 67.03 | 66.54 | +1.830 | 23332 |
| SHANDONG_DA | iTransformer | 65.67 | 66.36 | 65.68 | 67.65 | -1.304 | 23332 |
| SHAANXI_DA | TimeMixer | 110.76 | 111.19 | 110.76 | 110.74 | -0.413 | 23332 |
| SHAANXI_DA | PatchTST | 116.20 | 117.99 | 116.20 | 117.08 | -1.605 | 23332 |
| NINGXIA_DA | LSTM | 69.93 | 68.84 | 70.36 | 70.84 | -0.124 | 23332 |
| **panel median** | | 72.08 | 71.46 | 72.45 | 71.05 | -0.269 | 23332 |

## 2. Ladder: R1 vs R0 / R2 / R3

### R1 vs R0_STACK_ZERO_SUM

- cells where R1 is non-worse: **4/8**
- cells where R0_STACK_ZERO_SUM is better: **4/8**
- panel median R1 improvement: **-0.0249%**
- panel median R0_STACK_ZERO_SUM improvement: **+0.0235%**
- worst R1 improvement: -1.5418%
- cells with |R1 improvement| <= 0.5%: 3/8; median parameter ratio R0_STACK_ZERO_SUM/R1 = 0.9520

| cell | R1 MAE | R0_STACK_ZERO_SUM MAE | R1 improvement % |
|---|---:|---:|---:|
| GANSU_DA/LSTM | 71.83 | 72.82 | +1.3674 |
| GANSU_DA/TimeMixer | 71.10 | 71.34 | +0.3367 |
| GANSU_DA/PatchTST | 72.19 | 73.06 | +1.1906 |
| SHANDONG_DA/TimeMixer | 67.32 | 67.06 | -0.3928 |
| SHANDONG_DA/iTransformer | 66.36 | 65.67 | -1.0474 |
| SHAANXI_DA/TimeMixer | 111.19 | 110.76 | -0.3864 |
| SHAANXI_DA/PatchTST | 117.99 | 116.20 | -1.5418 |
| NINGXIA_DA/LSTM | 68.84 | 69.93 | +1.5615 |

### R1 vs R2_UNTIED_Q

- cells where R1 is non-worse: **4/8**
- cells where R2_UNTIED_Q is better: **4/8**
- panel median R1 improvement: **-0.0144%**
- panel median R2_UNTIED_Q improvement: **+0.0130%**
- worst R1 improvement: -1.5375%
- cells with |R1 improvement| <= 0.5%: 3/8; median parameter ratio R2_UNTIED_Q/R1 = 0.9534

| cell | R1 MAE | R2_UNTIED_Q MAE | R1 improvement % |
|---|---:|---:|---:|
| GANSU_DA/LSTM | 71.83 | 73.54 | +2.3322 |
| GANSU_DA/TimeMixer | 71.10 | 71.35 | +0.3574 |
| GANSU_DA/PatchTST | 72.19 | 73.72 | +2.0745 |
| SHANDONG_DA/TimeMixer | 67.32 | 67.03 | -0.4389 |
| SHANDONG_DA/iTransformer | 66.36 | 65.68 | -1.0324 |
| SHAANXI_DA/TimeMixer | 111.19 | 110.76 | -0.3862 |
| SHAANXI_DA/PatchTST | 117.99 | 116.20 | -1.5375 |
| NINGXIA_DA/LSTM | 68.84 | 70.36 | +2.1604 |

### R1 vs R3_DIRECT_Q

- cells where R1 is non-worse: **3/8**
- cells where R3_DIRECT_Q is better: **5/8**
- panel median R1 improvement: **-0.5914%**
- panel median R3_DIRECT_Q improvement: **+0.5876%**
- worst R1 improvement: -2.4190%
- cells with |R1 improvement| <= 0.5%: 2/8; median parameter ratio R3_DIRECT_Q/R1 = 0.8565

| cell | R1 MAE | R3_DIRECT_Q MAE | R1 improvement % |
|---|---:|---:|---:|
| GANSU_DA/LSTM | 71.83 | 71.26 | -0.8004 |
| GANSU_DA/TimeMixer | 71.10 | 69.42 | -2.4190 |
| GANSU_DA/PatchTST | 72.19 | 72.45 | +0.3670 |
| SHANDONG_DA/TimeMixer | 67.32 | 66.54 | -1.1704 |
| SHANDONG_DA/iTransformer | 66.36 | 67.65 | +1.9202 |
| SHAANXI_DA/TimeMixer | 111.19 | 110.74 | -0.4056 |
| SHAANXI_DA/PatchTST | 117.99 | 117.08 | -0.7772 |
| NINGXIA_DA/LSTM | 68.84 | 70.84 | +2.8230 |

## 3. Host comparison (R1)

| cell | Host MAE | R1 MAE | gain % |
|---|---:|---:|---:|
| GANSU_DA/LSTM | 81.17 | 71.83 | +11.5029 |
| GANSU_DA/TimeMixer | 69.71 | 71.10 | -1.9951 |
| GANSU_DA/PatchTST | 73.01 | 72.19 | +1.1297 |
| SHANDONG_DA/TimeMixer | 68.58 | 67.32 | +1.8301 |
| SHANDONG_DA/iTransformer | 65.50 | 66.36 | -1.3036 |
| SHAANXI_DA/TimeMixer | 110.73 | 111.19 | -0.4126 |
| SHAANXI_DA/PatchTST | 116.13 | 117.99 | -1.6049 |
| NINGXIA_DA/LSTM | 68.75 | 68.84 | -0.1244 |

positive 3/8 · panel median -0.2685% · worst -1.9951% — protocol section 10: a development target, not a claim of baseline competitiveness.

## 4. Level / delta closure core metrics (R1)

| cell | stage1 floor | delta closure err | floor reduction % | q reconstruction MAE |
|---|---:|---:|---:|---:|
| GANSU_DA/LSTM | 45.32 | 46.66 | -2.969 | 71.83 |
| GANSU_DA/TimeMixer | 50.91 | 50.80 | +0.208 | 71.10 |
| GANSU_DA/PatchTST | 57.29 | 55.43 | +3.254 | 72.19 |
| SHANDONG_DA/TimeMixer | 34.85 | 36.17 | -3.785 | 67.32 |
| SHANDONG_DA/iTransformer | 30.46 | 30.74 | -0.900 | 66.36 |
| SHAANXI_DA/TimeMixer | 48.54 | 50.48 | -4.000 | 111.19 |
| SHAANXI_DA/PatchTST | 57.46 | 63.18 | -9.948 | 117.99 |
| NINGXIA_DA/LSTM | 57.57 | 57.58 | -0.015 | 68.84 |

Three identities hold in every row above, and they are why some of the protocol's columns coincide rather than independent measurements:

- `level_mae = stage1_floor = delta_zero_reference_mae = mean|mean(r) - b_hat|` — the Stage-1 mean-floor error, which is also the reference a zero `delta` would score.
- `delta_mae = delta_closure_error = mean|mean(r) - b_hat - delta_hat|` — `delta_hat` is a per-day scalar, so the delta's own error and the closure error are the same functional.
- `q_reconstruction_mae = repaired_overall_mae` — `q_hat = delta_hat*1 + A_hat(S+ - S-)` and `correction = coarse_level*1 + local` are two routes to the same emitted quantity (`local = q_hat`), so `(r - b_hat) - q_hat = r - correction` identically. The two columns agree to float32-vs-float64 reduction precision, and that agreement is the reconstruction check passing, not a duplicated column.

Panel closure ratio per variant (protocol section 13, `median|b*-b_hat-delta_hat| / median|b*-b_hat|`; below 1 means `delta` recovered part of the coarse Level's mean error, at 1 it carried none):

| variant | closure ratio |
|---|---:|
| R0_STACK_ZERO_SUM | 1.000000 |
| R1_RCL_ORTHOGONAL | 1.049661 |
| R2_UNTIED_Q | 1.001601 |
| R3_DIRECT_Q | 1.024347 |

## 5. Tail / extreme / Normal

| cell | upper | lower | extreme | failure-tail | Normal | Normal Host | Normal harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| GANSU_DA/LSTM | — | 50.06 | 50.06 | — | 73.11 | 81.53 | +6.3531 |
| GANSU_DA/TimeMixer | — | 45.08 | 45.08 | — | 72.63 | 71.17 | +8.2437 |
| GANSU_DA/PatchTST | — | 54.37 | 54.37 | 123.36 | 73.25 | 74.08 | +1.4052 |
| SHANDONG_DA/TimeMixer | 182.61 | 104.63 | 160.51 | — | 63.49 | 65.05 | +2.3675 |
| SHANDONG_DA/iTransformer | 233.41 | 127.54 | 203.64 | — | 60.76 | 60.20 | +5.4733 |
| SHAANXI_DA/TimeMixer | 265.83 | 96.05 | 152.27 | 162.68 | 93.62 | 92.34 | +3.9884 |
| SHAANXI_DA/PatchTST | 261.02 | 87.76 | 145.33 | 145.55 | 106.38 | 104.07 | +7.6206 |
| NINGXIA_DA/LSTM | — | 48.27 | 48.27 | — | 82.02 | 81.96 | +0.2344 |
| **panel median** | 247.22 | 71.06 | 99.85 | 145.55 | 73.18 | 77.81 | +4.7309 |

Panel medians are taken over the cells whose selector matched something: upper 4/8, lower 8/8, extreme 8/8, failure-tail 3/8, Normal 8/8. `—` is an empty subset, not a zero: a null cell means the selector matched no horizon cell in that cell's VAL span, so the metric is undefined rather than zero. The selectors are TRAIN-only thresholds and a frozen Host forecast, so they do not depend on the seed or the variant: a cell is all-empty or all-finite, never a median over fewer than three seeds.

Normal harm is positive in **8/8** cells; the Normal subset improves in 3/8. Thresholds are TRAIN_ONLY: upper = target >= TRAIN q95; lower = target <= TRAIN q05; extreme = upper u lower; failure-tail day = Host daily MAE >= TRAIN daily-error q90; normal = not upper/lower.

Read the 8/8 with care: `normal_harm` is one-sided (`mean(max(0, |e_R| - |e_H|))`), so it is non-negative by construction and any cell with a single worse day reports a positive value. The magnitude is what carries information, and the panel median is +4.7309 in residual-error units against a Normal level of 73.18 — the two subsets that agree in sign are the ones where the mean itself moved.
> section 8 defines harm only for the Normal subset: normal_harm = mean(max(0, |e_R| - |e_H|))[normal]. The upper / lower / extreme / failure-tail columns are therefore repaired-side levels in residual-error space, not harms; the run records no Host counterpart for those subsets, so no harm can be computed for them from this evidence root and none is asserted here.

## 6. Parameters and runtime

| cell | R1 params | R3 params | R3/R1 | R1 best epoch | R1 epochs run |
|---|---:|---:|---:|---:|---:|
| GANSU_DA/LSTM | 23332 | 19984 | 0.8565 | 13 | 21 |
| GANSU_DA/TimeMixer | 23332 | 19984 | 0.8565 | 37 | 45 |
| GANSU_DA/PatchTST | 23332 | 19984 | 0.8565 | 24 | 32 |
| SHANDONG_DA/TimeMixer | 23332 | 19984 | 0.8565 | 7 | 15 |
| SHANDONG_DA/iTransformer | 23332 | 19984 | 0.8565 | 10 | 18 |
| SHAANXI_DA/TimeMixer | 23332 | 19984 | 0.8565 | 13 | 21 |
| SHAANXI_DA/PatchTST | 23332 | 19984 | 0.8565 | 20 | 28 |
| NINGXIA_DA/LSTM | 23332 | 19984 | 0.8565 | 10 | 18 |

median R1 parameters 23332 · max R3/R1 0.8565 (ceiling 1.25) · 248.8s total Stage-2 fit time, median 2.61s per fit.

## 7. Adjudication

- **R1 vs R0 → `MIXED/NOT_SUPPORTED`**
  - FAIL — positive_cells_ge_5_of_8
  - FAIL — panel_median_gain_ge_0p75pct
  - FAIL — worst_degradation_ge_minus_0p75pct
  - FAIL — delta_beats_zero_delta_ge_5_of_8
  - FAIL — panel_median_mean_floor_reduction_ge_25pct
  - measured: {"n_positive_cells": 4, "panel_median_gain_pct": -0.02486154793403947, "worst_gain_pct": -1.5417679601105727, "delta_beats_zero_delta_cells": 2, "panel_median_mean_floor_reduction_pct": -1.9344555462799389}
- **R1 vs R2 → `LOCAL_PARAMETERIZATION_MIXED`**
  - measured: {"n_cells_r1_non_worse": 4, "panel_median_r1_improvement_pct": -0.014407083535289356, "n_cells_r2_better": 4, "panel_median_r2_improvement_pct": 0.013023090455335173}
- **R1 vs R3 → `MIXED`**
  - measured: {"n_cells_r3_better": 5, "panel_median_r3_improvement_pct": 0.5875804225968829, "n_cells_r1_within_0p5pct": 2, "median_parameter_ratio_r3_over_r1": 0.8565060860620607, "matched_capacity": true}
- R1 vs Host → `DEVELOPMENT_TARGET_NOT_MET` (protocol section 10: a development target, not a claim of baseline competitiveness)
  - measured: {"n_positive_cells": 3, "panel_median_gain_pct": -0.2685382153568385, "worst_gain_pct": -1.995109879946896}

## 8. Boundary and verification

- `src/core` unchanged: **True** (digest `68891c77b3974cc3…`, 12 files)
- TEST target reads: **0**; `host_predictions_joint.npz` never opened; no QINGHAI, foreign, final or baseline path read.
- Verifier: 4733 checks, 0 failed.

### Supplementary defects (do not decide the round)

68 defects on the two supplementary centred-target diagnostics, of which 59 are float32 rounding on a supplementary readout and 9 are the amplification described below.

- fields: centered_shape_mae_candidate; fits: 9
- cause: the runner forms the candidate's q as ``r_eval - (r_eval - correction)`` in float32, which cancels: ``r_eval`` and the correction are O(100) while their difference is the Local correction, so the expression returns the correction plus up to one float32 ULP of ``r_eval`` (7.6e-6). That absolute error is harmless until it reaches ``build_residual_complete_targets``, which normalises the shapes by ``max(amplitude, 1e-8)``: when the candidate's own target amplitude is itself O(1e-3) the injected error is a large fraction of it, and the emitted shape targets become noise. The verifier evaluates the same expression in float64, where the cancellation is exact, so the two sides disagree by far more than float32 rounding.
- scope: confined to the two supplementary centred-target diagnostics; no protocol section 8 metric and no ladder comparison reads them
- remedy: replace the cancelling expression with the value it is trying to recover — ``torch.as_tensor(correction, dtype=torch.float32)`` — on both sides; the identity ``r - (r - c) == c`` is exact in reals, and writing ``c`` directly is what makes the float32 and float64 evaluations agree
- not applied: the artifacts under verification are immutable and the round is pre-registered; repairing the runner after seeing the result would be a post-hoc change to the execution. The defect is reported for the adjudicator to act on, not silently fixed.

## 9. Reading

Three section-10 labels: R1-vs-R0 `MIXED/NOT_SUPPORTED`, R1-vs-R2 `LOCAL_PARAMETERIZATION_MIXED`, R1-vs-R3 `MIXED`; the R1-vs-Host development target is `DEVELOPMENT_TARGET_NOT_MET`. The ladder separates the four coordinates on the same emitted quantity (q reconstruction), so the labels differ because the coordinate choice differs, not because the Level or the emission rule changed. All numbers above are the verifier's arithmetic over the raw artifacts; the runner's own metric functions were not imported by the verifier. The 68 supplementary defects are disclosed in section 8 and affect only the centred-target diagnostics.
