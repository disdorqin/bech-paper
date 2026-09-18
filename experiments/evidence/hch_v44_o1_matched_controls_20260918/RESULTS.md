# HCH O1 matched-control closure

**Terminal token: `HCH_O1_MATCHED_CONTROL_GEOMETRY_NOT_SUPPORTED`**

- protocol: `HCH_V44_O1_MATCHED_CONTROLS_20260918`
- recipe: `O1_AUX_WARMUP_400_THEN_MAE` — full structured objective through optimizer update 400, `L_rec` only after; selection = min EMA VAL MAE over all checks including step 0
- Phase A verdict: `A2_SHARED_CONTEXT_SUFFICIENT` → selected structured variant `G1_GEOM_SHARED_CONTEXT`
- Phase B verdict: `B2_EXACT_GEOMETRY_NOT_SUPPORTED`
- independent verification: all_pass=**True** (13 scientific checks, 0 failures); hygiene_pass=False

## 1. Source, parity and access

- `src/core` digest reproduces the frozen O1/G2 value: True — `6F18C0E4241A299E32C8D8A617061BAEAA7B8DC98B4EE069DC67199B87531577` (18 files)
- `src/backbones` digest reproduces the frozen O1/G2 value: True — `79B51A56499860D489C1FE4642F111EBB6BABC0281028F57F683A21484C12E7B` (10 files)
- `src/core` modified by this stage: **False**
- reused O1/G2 runs pinned by hash: 24/24; one `probe_code_hash` across all of them: True; every history support registered: True; artifact bytes snapshotted before the fits: 72
- TEST target reads: 0 in the new runs (48 runs), all reused runs zero: True
- forbidden-path guard hits summed over every run: 0; TEST rows dropped before any frame existed, per run: 166

## 2. Variant contract

- G1 and G2 have equal parameter count: True — 14948 trainable parameters, identical named keys and shapes
- the direct control adds no new parameter to the existing `G3`: `G3_SHARED` = 16897 = `G3` = 16897 trainable parameters; only `tied_identity` differs
- G1/G2 bitwise identical at initialization: True; forward output bitwise identical at initialization: True
- only the `tied_identity` flag differs between them: True (`coord_embed` is zero-initialised, so the tie is exact at step 0 and only the released variant can break it)
- direct control inside the registered 1.25x budget: True — ratio 1.130385; `G3_SHARED` adds no parameter to `G3`: True

## 3. Phase A — G2 versus G1, seed-median-first

gain(G2,G1) = 100 * (MAE_G1 - MAE_G2) / MAE_G1, per cell after taking each method's 3-seed median VAL MAE.

| cell | MAE structured (median) | MAE control (median) | gain | structured beats control | structured Host+ |
|---|---|---|---|---|---|
| GANSU_DA__PatchTST | 75.7672 | 75.7710 | +0.0051% | yes | yes |
| GANSU_DA__LSTM | 79.4816 | 79.4652 | -0.0206% | no | yes |
| SHANDONG_DA__PatchTST | 73.7958 | 73.7905 | -0.0072% | no | yes |
| SHANDONG_DA__iTransformer | 74.8497 | 74.8120 | -0.0503% | no | yes |
| SHAANXI_DA__TimeMixer | 103.4710 | 103.4700 | -0.0010% | no | yes |
| SHAANXI_DA__PatchTST | 106.8355 | 106.7567 | -0.0738% | no | yes |
| NINGXIA_DA__iTransformer | 69.2431 | 69.1314 | -0.1615% | no | yes |
| QINGHAI_DA__TimeMixer | 151.1069 | 145.7758 | -3.6570% | no | yes |

- cells where G2 beats G1: 1/8
- panel median gain(G2,G1): **-0.0355%**
- worst cell: -3.6570%; best cell: +0.0051%
- A1 clauses: `{"a1_cells_g2_beats_g1_ge_6of8": false, "a1_panel_median_gain_ge_0.5pct": false, "a1_worst_cell_ge_-0.5pct": false, "a1_g2_host_positive_ge_7of8": true, "a1_no_access_source_numerical_failure": true}`
- A2 clauses: `{"a2_g1_within_0.50pct_of_g2_ge_7of8": true, "a2_panel_median_g1_vs_g2_gain_ge_-0.25pct": true, "a2_worst_g1_vs_g2_cell_ge_-1.00pct": true, "a2_g1_host_positive_ge_7of8": true, "a2_no_access_source_numerical_failure": true}`
- **verdict `A2_SHARED_CONTEXT_SUFFICIENT`**

## 4. Phase B — structured versus direct, seed-median-first

- direct family fitted: `G3_SHARED_DIRECT_CONTEXT_CTRL` — the existing `G3_DIRECT_CONTEXT_CTRL` with `tied_identity=True`, which introduces no new parameter and is byte/forward identical otherwise
- reconstruction-only loss throughout, same optimizer and checkpoint budget

gain(STRUCT,DIRECT) = 100 * (MAE_DIRECT - MAE_STRUCT) / MAE_DIRECT.

| cell | MAE structured (median) | MAE control (median) | gain | structured beats control | structured Host+ |
|---|---|---|---|---|---|
| GANSU_DA__PatchTST | 75.7710 | 76.5858 | +1.0638% | yes | yes |
| GANSU_DA__LSTM | 79.4652 | 80.0726 | +0.7586% | yes | yes |
| SHANDONG_DA__PatchTST | 73.7905 | 73.4503 | -0.4632% | no | yes |
| SHANDONG_DA__iTransformer | 74.8120 | 75.0545 | +0.3231% | yes | yes |
| SHAANXI_DA__TimeMixer | 103.4700 | 103.5229 | +0.0512% | yes | yes |
| SHAANXI_DA__PatchTST | 106.7567 | 108.5481 | +1.6503% | yes | yes |
| NINGXIA_DA__iTransformer | 69.1314 | 67.3026 | -2.7172% | no | yes |
| QINGHAI_DA__TimeMixer | 145.7758 | 155.5435 | +6.2797% | yes | yes |

- cells where structured beats direct: 6/8
- panel median gain(STRUCT,DIRECT): **+0.5408%**
- worst cell: -2.7172%; best cell: +6.2797%
- B1 clauses: `{"b1_cells_structured_beats_direct_ge_6of8": true, "b1_panel_median_gain_ge_0.5pct": true, "b1_worst_cell_ge_-0.5pct": false, "b1_structured_host_positive_ge_7of8": true, "b1_no_access_source_numerical_failure": true}`
- **verdict `B2_EXACT_GEOMETRY_NOT_SUPPORTED`**

## 5. Which structured variant survives for paper use

No structured variant survives for prediction from this stage: exact geometry remains a mathematical representation result only, no predictive advantage may be claimed, and the question returns to human paper-position adjudication rather than another architecture rescue.

## 6. Independent verification

- `verification/verify_matched.py` imports none of `matched_runner`, `matched_train`, `matched_common`, `probe_train`, `probe_common` or `recovery_train`; the panel, seeds, registered schedule and every gate threshold are restated in it, and every number is recomputed from the freeze records, training curves, selected EMA checkpoints and CSV tables
- 13 scientific checks: all_pass=**True** (failures: [])
- hygiene checks: hygiene_pass=False (['runner_cli_is_self_consistent'])
- the reused O1/G2 artifacts were additionally cross-checked against `experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/REUSE_PROVENANCE.json`, a witness written by a *different* stage — the one that closed with its own terminal token `HCH_V44_O1_DOMESTIC20_NOT_READY` and its own independent verification report. It agrees on all three artifact hashes, and independently records the same `selected_val_mae_ema`, `selected_step` and `selected_ema_parameter_hash` for every one of the 24 reused runs — so the baseline MAE that the Phase-A gain divides by is confirmed outside this stage's own records. mtime is not treated as evidence of priority anywhere in this stage; the basis is that stage's own closure artifacts.

### Disclosed defects (none affect a scientific number)

- runner docstring advertises steps that main() does not dispatch — impact: none on any scientific number; the affected artifacts (RESULTS.md, STAGE_TOKEN.json) are written by the executor under the design authority's section 11 file list
- no finalize step exists in matched_runner.main() — impact: RESULTS.md and STAGE_TOKEN.json are executor-written
- `matched_tree_digest` hashes the entire stage directory, so the verifier could not be archived into `verification/` before the fits without changing `matched_code_hash` part-way through the stage. The verifier therefore recomputes the digest over the fit-time file set — the stage tree excluding its own `verification/` subtree — and lists the files archived afterwards. Every run's recorded `matched_code_hash` reproduces exactly under that rule.
- the verifier ran twice: once before the token existed, where its only failure was the absent token, and once after, from its archived location. The final `INDEPENDENT_VERIFICATION_REPORT.json` is the second run.

## 7. Evidence files

- `ACCESS_AUDIT.json`
- `DIRECT_PER_SEED.csv`
- `FIT_LOG_PHASE_A.json`
- `FIT_LOG_PHASE_B.json`
- `G1_PER_SEED.csv`
- `INDEPENDENT_VERIFICATION_REPORT.json`
- `O1_G2_REUSE_PROVENANCE.json`
- `PARAMETER_COUNTS.csv`
- `PHASE_A_CELL_MEDIANS.csv`
- `PHASE_A_GATE.json`
- `PHASE_B_CELL_MEDIANS.csv`
- `PHASE_B_DRIVER_PROVENANCE.json`
- `PHASE_B_GATE.json`
- `RESULTS.md`
- `REUSE_SNAPSHOT_AFTER_PHASE_A.json`
- `REUSE_SNAPSHOT_BEFORE.json`
- `runs/direct/GANSU_DA__LSTM/seed17/freeze.json`
- `runs/direct/GANSU_DA__LSTM/seed17/selected_ema.pt`
- `runs/direct/GANSU_DA__LSTM/seed17/training_curve.json`
- `runs/direct/GANSU_DA__LSTM/seed37/freeze.json`
- `runs/direct/GANSU_DA__LSTM/seed37/selected_ema.pt`
- `runs/direct/GANSU_DA__LSTM/seed37/training_curve.json`
- `runs/direct/GANSU_DA__LSTM/seed7/freeze.json`
- `runs/direct/GANSU_DA__LSTM/seed7/selected_ema.pt`
- `runs/direct/GANSU_DA__LSTM/seed7/training_curve.json`
- `runs/direct/GANSU_DA__PatchTST/seed17/freeze.json`
- `runs/direct/GANSU_DA__PatchTST/seed17/selected_ema.pt`
- `runs/direct/GANSU_DA__PatchTST/seed17/training_curve.json`
- `runs/direct/GANSU_DA__PatchTST/seed37/freeze.json`
- `runs/direct/GANSU_DA__PatchTST/seed37/selected_ema.pt`
- `runs/direct/GANSU_DA__PatchTST/seed37/training_curve.json`
- `runs/direct/GANSU_DA__PatchTST/seed7/freeze.json`
- `runs/direct/GANSU_DA__PatchTST/seed7/selected_ema.pt`
- `runs/direct/GANSU_DA__PatchTST/seed7/training_curve.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed17/freeze.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed17/selected_ema.pt`
- `runs/direct/NINGXIA_DA__iTransformer/seed17/training_curve.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed37/freeze.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed37/selected_ema.pt`
- `runs/direct/NINGXIA_DA__iTransformer/seed37/training_curve.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed7/freeze.json`
- `runs/direct/NINGXIA_DA__iTransformer/seed7/selected_ema.pt`
- `runs/direct/NINGXIA_DA__iTransformer/seed7/training_curve.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed17/freeze.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed17/selected_ema.pt`
- `runs/direct/QINGHAI_DA__TimeMixer/seed17/training_curve.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed37/freeze.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed37/selected_ema.pt`
- `runs/direct/QINGHAI_DA__TimeMixer/seed37/training_curve.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed7/freeze.json`
- `runs/direct/QINGHAI_DA__TimeMixer/seed7/selected_ema.pt`
- `runs/direct/QINGHAI_DA__TimeMixer/seed7/training_curve.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed17/freeze.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed17/selected_ema.pt`
- `runs/direct/SHAANXI_DA__PatchTST/seed17/training_curve.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed37/freeze.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed37/selected_ema.pt`
- `runs/direct/SHAANXI_DA__PatchTST/seed37/training_curve.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed7/freeze.json`
- `runs/direct/SHAANXI_DA__PatchTST/seed7/selected_ema.pt`
- `runs/direct/SHAANXI_DA__PatchTST/seed7/training_curve.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed17/freeze.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed17/selected_ema.pt`
- `runs/direct/SHAANXI_DA__TimeMixer/seed17/training_curve.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed37/freeze.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed37/selected_ema.pt`
- `runs/direct/SHAANXI_DA__TimeMixer/seed37/training_curve.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed7/freeze.json`
- `runs/direct/SHAANXI_DA__TimeMixer/seed7/selected_ema.pt`
- `runs/direct/SHAANXI_DA__TimeMixer/seed7/training_curve.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed17/freeze.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed17/selected_ema.pt`
- `runs/direct/SHANDONG_DA__iTransformer/seed17/training_curve.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed37/freeze.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed37/selected_ema.pt`
- `runs/direct/SHANDONG_DA__iTransformer/seed37/training_curve.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed7/freeze.json`
- `runs/direct/SHANDONG_DA__iTransformer/seed7/selected_ema.pt`
- `runs/direct/SHANDONG_DA__iTransformer/seed7/training_curve.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed17/freeze.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed17/selected_ema.pt`
- `runs/direct/SHANDONG_DA__PatchTST/seed17/training_curve.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed37/freeze.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed37/selected_ema.pt`
- `runs/direct/SHANDONG_DA__PatchTST/seed37/training_curve.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed7/freeze.json`
- `runs/direct/SHANDONG_DA__PatchTST/seed7/selected_ema.pt`
- `runs/direct/SHANDONG_DA__PatchTST/seed7/training_curve.json`
- `runs/g1/GANSU_DA__LSTM/seed17/freeze.json`
- `runs/g1/GANSU_DA__LSTM/seed17/selected_ema.pt`
- `runs/g1/GANSU_DA__LSTM/seed17/training_curve.json`
- `runs/g1/GANSU_DA__LSTM/seed37/freeze.json`
- `runs/g1/GANSU_DA__LSTM/seed37/selected_ema.pt`
- `runs/g1/GANSU_DA__LSTM/seed37/training_curve.json`
- `runs/g1/GANSU_DA__LSTM/seed7/freeze.json`
- `runs/g1/GANSU_DA__LSTM/seed7/selected_ema.pt`
- `runs/g1/GANSU_DA__LSTM/seed7/training_curve.json`
- `runs/g1/GANSU_DA__PatchTST/seed17/freeze.json`
- `runs/g1/GANSU_DA__PatchTST/seed17/selected_ema.pt`
- `runs/g1/GANSU_DA__PatchTST/seed17/training_curve.json`
- `runs/g1/GANSU_DA__PatchTST/seed37/freeze.json`
- `runs/g1/GANSU_DA__PatchTST/seed37/selected_ema.pt`
- `runs/g1/GANSU_DA__PatchTST/seed37/training_curve.json`
- `runs/g1/GANSU_DA__PatchTST/seed7/freeze.json`
- `runs/g1/GANSU_DA__PatchTST/seed7/selected_ema.pt`
- `runs/g1/GANSU_DA__PatchTST/seed7/training_curve.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed17/freeze.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed17/selected_ema.pt`
- `runs/g1/NINGXIA_DA__iTransformer/seed17/training_curve.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed37/freeze.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed37/selected_ema.pt`
- `runs/g1/NINGXIA_DA__iTransformer/seed37/training_curve.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed7/freeze.json`
- `runs/g1/NINGXIA_DA__iTransformer/seed7/selected_ema.pt`
- `runs/g1/NINGXIA_DA__iTransformer/seed7/training_curve.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed17/freeze.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed17/selected_ema.pt`
- `runs/g1/QINGHAI_DA__TimeMixer/seed17/training_curve.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed37/freeze.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed37/selected_ema.pt`
- `runs/g1/QINGHAI_DA__TimeMixer/seed37/training_curve.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed7/freeze.json`
- `runs/g1/QINGHAI_DA__TimeMixer/seed7/selected_ema.pt`
- `runs/g1/QINGHAI_DA__TimeMixer/seed7/training_curve.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed17/freeze.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed17/selected_ema.pt`
- `runs/g1/SHAANXI_DA__PatchTST/seed17/training_curve.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed37/freeze.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed37/selected_ema.pt`
- `runs/g1/SHAANXI_DA__PatchTST/seed37/training_curve.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed7/freeze.json`
- `runs/g1/SHAANXI_DA__PatchTST/seed7/selected_ema.pt`
- `runs/g1/SHAANXI_DA__PatchTST/seed7/training_curve.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed17/freeze.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed17/selected_ema.pt`
- `runs/g1/SHAANXI_DA__TimeMixer/seed17/training_curve.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed37/freeze.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed37/selected_ema.pt`
- `runs/g1/SHAANXI_DA__TimeMixer/seed37/training_curve.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed7/freeze.json`
- `runs/g1/SHAANXI_DA__TimeMixer/seed7/selected_ema.pt`
- `runs/g1/SHAANXI_DA__TimeMixer/seed7/training_curve.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed17/freeze.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed17/selected_ema.pt`
- `runs/g1/SHANDONG_DA__iTransformer/seed17/training_curve.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed37/freeze.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed37/selected_ema.pt`
- `runs/g1/SHANDONG_DA__iTransformer/seed37/training_curve.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed7/freeze.json`
- `runs/g1/SHANDONG_DA__iTransformer/seed7/selected_ema.pt`
- `runs/g1/SHANDONG_DA__iTransformer/seed7/training_curve.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed17/freeze.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed17/selected_ema.pt`
- `runs/g1/SHANDONG_DA__PatchTST/seed17/training_curve.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed37/freeze.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed37/selected_ema.pt`
- `runs/g1/SHANDONG_DA__PatchTST/seed37/training_curve.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed7/freeze.json`
- `runs/g1/SHANDONG_DA__PatchTST/seed7/selected_ema.pt`
- `runs/g1/SHANDONG_DA__PatchTST/seed7/training_curve.json`
- `SOURCE_AUDIT.json`
- `STAGE_TOKEN.json`
- `TRAINING_CURVES.csv`
- `VARIANT_PARITY.json`

