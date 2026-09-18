# HCH v4.4 repairability-spectrum diagnostic -- results

STATUS: COMPLETE_FOR_ADJUDICATION  |  STAGE: HCH_V44_REPAIRABILITY_SPECTRUM_20260918

Diagnostic only: 60 frozen O1 runs reused read-only, 0 new fits, 0 optimizer steps,
0 TEST reads. V2 TEST, protected and final partitions untouched.

## Access
- runs reused: 60 (all present, all hashed, snapshot unchanged: True)
- VAL MAE reconciliation: 60 runs checked, 0 failed, max |delta| = 1.106e-05
- days with fewer than 24 valid hours: 0
- rows written: 21324

## Diagnosis by cell (VAL medians, % of that day's Host MAE)

| cell | O1 gain | ray | b | B | Shape | best | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GANSU_DA__PatchTST | +0.5478 | +3.3013 | +4.0968 | +0.0027 | +17.3427 | 17.3427 (shape) | SHAPE_LIMITED |
| GANSU_DA__TimeMixer | -1.9737 | +5.6016 | +15.4297 | +0.0026 | +20.5947 | 20.5947 (shape) | SHAPE_LIMITED |
| GANSU_DA__iTransformer | -0.2173 | +0.9616 | +16.1742 | +0.0000 | +8.5618 | 16.1742 (level) | LEVEL_LIMITED |
| GANSU_DA__LSTM | +2.0613 | +3.6185 | +21.4194 | +0.0000 | +15.3833 | 21.4194 (level) | LEVEL_LIMITED |
| SHANDONG_DA__PatchTST | +4.4497 | +4.3764 | +3.6152 | -0.5024 | +38.7294 | 38.7294 (shape) | SHAPE_LIMITED |
| SHANDONG_DA__TimeMixer | -1.0315 | +5.3995 | +9.3259 | +0.1182 | +35.8440 | 35.8440 (shape) | SHAPE_LIMITED |
| SHANDONG_DA__iTransformer | +1.0952 | +5.1360 | +3.8726 | +0.0299 | +43.9982 | 43.9982 (shape) | SHAPE_LIMITED |
| SHANDONG_DA__LSTM | +7.3538 | +4.9764 | +3.9471 | -2.7240 | +10.9645 | 10.9645 (shape) | SHAPE_LIMITED |
| SHAANXI_DA__PatchTST | +4.3486 | +1.9406 | +2.6039 | -0.5981 | +35.6868 | 35.6868 (shape) | SHAPE_LIMITED |
| SHAANXI_DA__TimeMixer | +0.0383 | +1.7735 | +1.8532 | +0.0409 | +9.1725 | 9.1725 (shape) | SHAPE_LIMITED |
| SHAANXI_DA__iTransformer | +0.0994 | +4.8004 | +3.5191 | -0.7077 | +30.4839 | 30.4839 (shape) | SHAPE_LIMITED |
| SHAANXI_DA__LSTM | +0.4143 | +3.0120 | +3.1930 | -0.0000 | +8.7247 | 8.7247 (shape) | SHAPE_LIMITED |
| NINGXIA_DA__PatchTST | +0.0002 | +11.4545 | +6.0636 | -0.3595 | +9.8213 | 11.4545 (distance) | DISTANCE_LIMITED |
| NINGXIA_DA__TimeMixer | +4.8535 | +4.7154 | -9.9972 | -0.4463 | +20.5902 | 20.5902 (shape) | SHAPE_LIMITED |
| NINGXIA_DA__iTransformer | +3.8614 | +5.4728 | +10.0812 | -0.9971 | +15.1949 | 15.1949 (shape) | SHAPE_LIMITED |
| NINGXIA_DA__LSTM | +0.3519 | +7.6370 | +23.2871 | -0.0943 | +11.8731 | 23.2871 (level) | LEVEL_LIMITED |
| QINGHAI_DA__PatchTST | +11.5262 | +5.9951 | -1.9271 | -0.1492 | +21.4577 | 21.4577 (shape) | SHAPE_LIMITED |
| QINGHAI_DA__TimeMixer | +2.0892 | +2.7599 | -0.5772 | -0.6909 | +17.9559 | 17.9559 (shape) | SHAPE_LIMITED |
| QINGHAI_DA__iTransformer | +6.7604 | +1.8283 | +7.1518 | -0.9284 | +32.6064 | 32.6064 (shape) | SHAPE_LIMITED |
| QINGHAI_DA__LSTM | +6.5888 | +2.5273 | -10.7582 | -1.9727 | +34.2112 | 34.2112 (shape) | SHAPE_LIMITED |

Diagnosis counts: DISTANCE_LIMITED=1, SHAPE_LIMITED=16, LEVEL_LIMITED=3

## The seven registered low-gain cells

| cell | O1 gain | best headroom % | family | diagnosis | material >=2% |
| --- | --- | --- | --- | --- | --- |
| GANSU_DA__TimeMixer | -1.9737 | 20.5947 | shape | SHAPE_LIMITED | True |
| GANSU_DA__iTransformer | -0.2173 | 16.1742 | level | LEVEL_LIMITED | True |
| SHANDONG_DA__TimeMixer | -1.0315 | 35.8440 | shape | SHAPE_LIMITED | True |
| SHANDONG_DA__iTransformer | +1.0952 | 43.9982 | shape | SHAPE_LIMITED | True |
| SHAANXI_DA__TimeMixer | +0.0383 | 9.1725 | shape | SHAPE_LIMITED | True |
| SHAANXI_DA__iTransformer | +0.0994 | 30.4839 | shape | SHAPE_LIMITED | True |
| SHAANXI_DA__LSTM | +0.4143 | 8.7247 | shape | SHAPE_LIMITED | True |

Dominant-family counts among material-headroom low-gain cells: {'shape': 6, 'level': 1}

## Host-to-residual analogue (no fit, no K search, no learned distance)

See HOST_RESIDUAL_ANALOGUE.csv; the aggregate reports per-day and pooled Spearman between
Host similarity and residual-geometry similarity, and the fraction of oracle-history
B/Shape headroom the legal nearest-Host analogue recovers.

## Leave-one-market-out Ridge probes

sample: 5992 TRAIN cell-days over 20 cells; model: sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True); descriptors: 28

| target | family | median rho | median probe MAE | median const MAE | folds beating const |
| --- | --- | --- | --- | --- | --- |
| log1p_alpha_star | distance | +0.0066 | 0.4705 | 0.4537 | 0/5 |
| ray_headroom_pct | distance | +0.0865 | 7.2706 | 5.6635 | 1/5 |
| B_headroom_pct | balanced_mass | -0.0267 | 6.5685 | 5.7853 | 1/5 |
| shape_headroom_pct | shape | +0.0105 | 25.7623 | 20.2396 | 1/5 |

## Future-design gate A-E

- **A**: PASS -- >= 4 of the 7 registered low-gain cells have material oracle headroom >= 2.0% of Host MAE
- **B**: PASS -- the same dominant headroom family explains >= 3 of those material-headroom low-gain cells
- **C**: FAIL -- at least one fixed legal-history signal for that family generalizes with the same direction in >= 4/5 leave-one-market-out folds and the probe improves its target over the constant baseline in >= 4/5 folds
- **D**: PASS -- not market-ID dependent, and the effect is visible in >= 2 Host families
- **E**: PASS -- TEST reads = 0 and no existing checkpoint/evidence is mutated

All gates passed: **False**.

This stage stops here. No router, similar-day module, amplitude learner, feature branch,
architecture change or foreign experiment is implemented by this stage, and no successor
stage is opened automatically.

## Corrections applied while this stage ran (disclosed)

Three deterministic implementation defects were found and fixed **before any diagnostic
number in this file was interpreted**.  Each was a crash or an unreproducible check, not a
result-conditioned adjustment; no descriptor, target, fold, threshold or model was added,
removed or retuned.

1. `repair_summary` read the per-day headroom columns under names the R0 atlas does not
   use (`ray_pct`/`b_pct`/`B_pct`/`Shape_pct` instead of
   `increment_ray_pct`/`imp_ob_pct`/`imp_oB_pct`/`imp_oS_pct`).  Every cell came out
   `UNDEFINED` and the gate then crashed on an empty numeric field.  An explicit
   `ATLAS_COLUMN` map now carries the mapping.
2. `repair_gate.gate_e_evidence` asked each run record for a per-run `source_tree_digest`,
   which `reuse_provenance` deliberately does not store per run (it stores the collapsed
   distinct set).  The gate now re-reads the digest from each run's own `freeze.json`, so
   all 60 runs are re-digested individually.
3. `repair_r2` normalised the cell-macro-balance weights over the **whole pooled sample**
   and then sliced the training fold, so each fold's weights had a mean slightly off 1.
   With `alpha` fixed at 1.0 by the registration a global weight constant silently rescales
   the penalty.  The training slice is now renormalised to mean weight 1 **within the
   fold**, which is what PROTOCOL Sec. 5 ("macro-balance cells in the training loss/data
   weights") registers.  This independently-surfaced defect was caught by the verifier's
   `lomo_metrics_reproduced` check failing (worst `probe_mae` discrepancy 0.0333) and it
   moved no fold across the `probe_beats_const` line for any target (0,1,1,1 before and
   after), so the gate C outcome is unchanged.

R2 and the gate were re-run after fix 3; `DIAGNOSIS_BY_CELL.csv`, `GATE_A_B_EVIDENCE.json`,
the R0 atlas and the R1 analogues were unaffected by it.
