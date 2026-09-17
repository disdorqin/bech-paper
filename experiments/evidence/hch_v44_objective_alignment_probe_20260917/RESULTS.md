# HCH v4.4 objective-alignment probe (O1) — TRAIN+VAL only, TEST quarantined

Gate verdict: **AUXILIARY_PERSISTENCE_NOT_SHOWN_HARMFUL** (4/5 conditions)

## What changed

Exactly one thing: updates 1..400 use the frozen full loss `L_rec + (L_b + L_B + L_S)/3`, updates 401..2000 use `L_rec` alone. Architecture, geometry, features, initialization, optimizer, learning rate, batch size, weight decay, gradient clip, EMA, seeds, data split, history support and the stopping rule are unchanged, and `src/core/**` was not edited.

## Reference

The same-cell, same-seed full-loss step-based R1 runs of the closed recovery stage were **reused, not retrained** (1 distinct reference schedule, all 12 coordinates present). See `REFERENCE_PROVENANCE.json`.

## Four-cell medians

| cell | O1 median VAL MAE | R1 median VAL MAE | median relative gain |
|---|---|---|---|
| GANSU_DA__PatchTST | 75.7672 | 77.1040 | +1.7337% |
| SHANDONG_DA__iTransformer | 74.8497 | 74.9808 | +0.1903% |
| SHAANXI_DA__TimeMixer | 103.4710 | 103.4710 | +0.0000% |
| NINGXIA_DA__iTransformer | 69.2431 | 69.3754 | +0.1907% |

- panel median relative gain (median of the four cell medians): **+0.1905%**
- cells improving: 3/4; seeds improving: 9/12
- worst / best cell: +0.0000% / +1.7337%
- step-0 retreat seeds: 0 (rule <=2); median selected correction ratio 0.2032 (rule >=0.05)

## Gate

| condition | result |
|---|---|
| g1_at_least_3_of_4_cell_medians_improve | PASS |
| g2_panel_median_relative_improvement_ge_0.5pct | FAIL |
| g3_worst_cell_degradation_le_0.5pct | PASS |
| g4_not_a_step0_or_near_zero_retreat | PASS |
| g5_no_leakage_anomaly_or_protocol_violation | PASS |

This is a **diagnostic** verdict, not a paper-promotion gate.

## Not a SOTA claim

V2 TEST was observed before this probe and stays quarantined from every decision here. The probe read no TEST target, prediction or metric. No paper claim follows from it.

## Evidence

- `REFERENCE_PROVENANCE.json` — the reused reference, with per-coordinate hashes;
- `PER_STEP_CURVES.csv` — the full per-check logging;
- `PAIRED_PER_SEED.csv`, `CELL_MEDIANS.csv` — the paired comparison and the medians;
- `GATE.json` — gate arithmetic, compliance audit and the verdict;
- `PLOTS/` — the four required figures as PNG and SVG;
- `PROBE_CONFIG.json`, `ACCESS_AUDIT.json` — the exact config and the access audit;
- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.
