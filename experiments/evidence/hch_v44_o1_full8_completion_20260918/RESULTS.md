# HCH v4.4 O1 full-8 completion — TRAIN+VAL only, TEST quarantined

Gate verdict: **AUXILIARY_PERSISTENCE_BROADLY_HARMFUL_ON_RECOVERY_PANEL** (5/5 conditions)

## What changed

Nothing in the method. The frozen O1 objective schedule is extended to the full 8-cell R1
recovery panel: updates 1..400 use `L_rec + (L_b + L_B + L_S)/3`, updates 401..2000 use
`L_rec` alone. The 12 new fits execute the O1 stage's own `train_o1` imported read-only;
architecture, geometry, features, initialization, optimizer, learning rate, batch size,
weight decay, gradient clip, EMA, seeds, data split, history support and the stopping rule
are unchanged and `src/core/**` was not edited.

## Reuse

- the 24 R1 full-loss references (`R1_STEP_BUDGET_2000`) are reused, not retrained, and
  pinned by hash in `REUSE_PROVENANCE.json`;
- the 12 O1 runs of the four original cells are reused **in place** from the prior probe's
  evidence root and are not rerun or copied.

## Eight-cell medians

| cell | source | O1 median VAL MAE | R1 median VAL MAE | median relative gain |
|---|---|---|---|---|
| GANSU_DA__PatchTST | reused_o1 | 75.7672 | 77.1040 | +1.7337% |
| GANSU_DA__LSTM | new_fit | 79.4816 | 79.4853 | +0.0047% |
| SHANDONG_DA__PatchTST | new_fit | 73.7958 | 75.4600 | +2.1831% |
| SHANDONG_DA__iTransformer | reused_o1 | 74.8497 | 74.9808 | +0.1903% |
| SHAANXI_DA__TimeMixer | reused_o1 | 103.4710 | 103.4710 | +0.0000% |
| SHAANXI_DA__PatchTST | new_fit | 106.8355 | 108.5351 | +1.2337% |
| NINGXIA_DA__iTransformer | reused_o1 | 69.2431 | 69.3754 | +0.1907% |
| QINGHAI_DA__TimeMixer | new_fit | 151.1069 | 150.6673 | +1.1678% |

- panel median relative gain (median of the eight cell medians): **+0.6793%**
- cells improving: 7/8 (need 6); seeds improving: 18/24
- worst / best cell: +0.0000% / +2.1831%
- step-0 retreat seeds: 0 (<=4 of 24 proportional reading: PASS; <=2 of 24 absolute reading: PASS)
- median selected correction ratio 0.2244 (rule >= 0.05)
- new-cell sub-panel median +1.2007%; reused-cell sub-panel median +0.1905%

## Gate

| condition | result |
|---|---|
| g1_at_least_6_of_8_cell_medians_improve | PASS |
| g2_panel_median_relative_improvement_ge_0.5pct | PASS |
| g3_worst_cell_degradation_le_0.5pct | PASS |
| g4_not_a_step0_or_near_zero_retreat | PASS |
| g5_no_leakage_anomaly_support_or_code_violation | PASS |

This is a **diagnostic** verdict, not a paper-promotion gate. The thresholds were
preregistered in `PROTOCOL.md` before any new fit.

## Not a SOTA claim

V2 TEST was observed before this line of work and stays quarantined from every decision
here. The stage read no TEST target, prediction or metric. No paper claim follows.

## Evidence

- `REUSE_PROVENANCE.json` — the reused O1 runs and the R1 references, by hash;
- `PER_STEP_CURVES.csv` — the full per-check logging for all 24 runs;
- `PAIRED_PER_SEED.csv`, `CELL_MEDIANS.csv` — the paired comparison and the medians;
- `MECHANISM_SUMMARY.csv` — the post-switch auxiliary directions (interpretation only);
- `GATE.json` — gate arithmetic, compliance audit and the verdict;
- `PLOTS/` — the four required figures as PNG and SVG;
- `PROBE_CONFIG.json`, `ACCESS_AUDIT.json` — the exact config and the access audit;
- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.
