# HCH v4.4 Domestic Baseline Comparison Freeze Protocol

Protocol ID: `HCH_V44_BASELINE_COMPARISON_FREEZE_20260916`

Status: **AUTHORIZED READ-ONLY BASELINE RECONCILIATION / ZERO NEW FITS EXPECTED**

Controlling design: `docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md`.

## 1. Goal

Build a paper-ready domestic baseline registry under `COMMON_BENCHMARK_701020_FULL_V2` so future v4.4 results can be joined cell-by-cell to already verified comparison data. This task does not train v4.4 and should not retrain a baseline whose V2 result already exists.

## 2. Fixed coordinate universe

Markets: `GANSU_DA, SHANDONG_DA, SHAANXI_DA, NINGXIA_DA, QINGHAI_DA`.

Hosts: `PatchTST, TimeMixer, iTransformer, LSTM`.

Methods per cell: `Host, MatchedDirectResidual, delta-Adapter, PIR, COSA, UEC-STD, OMPB`.

Total registered coordinates: `5*4*7 = 140`.

Expected final state:

- numeric = 90;
- blockers = 50;
- new baseline neural fits = 0.

## 3. Split authority

Do not create a new split. Read and verify the frozen V2 split manifests. Required counts:

- GANSU: 291/42/83;
- SHANDONG: 1156/166/330 registered, 329 scored;
- SHAANXI: 284/41/81;
- NINGXIA: 100/16/28;
- QINGHAI: 94/14/27.

Every numeric row must carry `COMMON_BENCHMARK_TEST_WITH_HISTORICAL_EXPOSURE_CAVEAT` or the exact equivalent frozen parent label.

## 4. Parent sources

Read only:

- `experiments/lab/common_benchmark_results_v2/**`;
- `experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913/**`;
- `docs/current/HCH_COMMON_BENCHMARK_701020_FULL_V2_PROTOCOL_20260913.md`;
- frozen qualification/blocker evidence referenced by those sources if needed.

Do not modify parent artifacts.

## 5. Expected method coverage

- Host: 20 numeric;
- MDR: 20 numeric;
- δ-Adapter: 20 numeric;
- COSA: 20 numeric and setting=`ONLINE_TTA`;
- PIR: 10 numeric on PatchTST/TimeMixer, 10 Q-INCOMPATIBLE blockers on iTransformer/LSTM;
- UEC-STD: 20 Q-FIDELITY blockers;
- OMPB: 20 Q-DATA/fidelity blockers.

A blocker is a result. Do not create proxy values or new configs.

## 6. Extraction rules

Use the final missing-target-adjudicated artifacts as numeric authority. For each numeric cell, independently recompute at least:

- Overall MAE from stored prediction and target arrays when arrays are available;
- relative gain vs the same-cell Host;
- scored-day count;
- subset metrics from the frozen result object or raw arrays using the already-frozen threshold/subset contract.

Where final evidence is explicitly metric-only, preserve that disclosure instead of manufacturing a raw array.

## 7. Output tables

Write under `experiments/evidence/hch_v44_baseline_comparison_freeze_20260916/` only.

### `MATRIX_140.csv`

One row for every registered coordinate.

### `STRICT_OFFLINE_TABLE.csv`

Host/MDR/δ/PIR only. Numeric expected=70. Blocked PIR rows may be included with status but must never be ranked.

### `ONLINE_SUPPLEMENTARY_TABLE.csv`

COSA rows, with same-cell Host reference metadata and online/TTA setting.

### `BLOCKED_ROWS.csv`

Exactly 50 blocker coordinates with blocker class, code, reason and frozen source pointer.

### `CELL_COMPARATOR_SNAPSHOT.csv/.json`

Exactly 20 market×Host records. For each record include Host, MDR, δ, legal PIR if present, COSA separately, best numeric strict-offline external post-hoc comparator, and blocker metadata.

## 8. Comparator definition

`best_strict_offline_external_posthoc` may select only between:

- δ-Adapter;
- PIR when numeric.

MDR is an internal project control and is recorded separately. Host is the reference, not an external post-hoc method. COSA is online/TTA and must never participate in the offline selector.

## 9. No rerun / stop rule

Do not fit any model.

If any of the expected 90 frozen numeric coordinates cannot be reconstructed or verified, return `HCH_V44_BASELINE_REGISTRY_BLOCKED_FOR_ADJUDICATION`. A missing frozen result is an evidence-integrity issue; it does not authorize retraining.

Do not reopen UEC-STD, OMPB or PIR-new-Host blockers.

## 10. Independent verifier

The verifier must not import the registry-construction result functions. It must reread parent frozen artifacts and verify:

- coordinate census;
- numeric/blocker counts;
- metric parity;
- setting separation;
- gain arithmetic;
- Shandong 329-day common scoring mask;
- parent source hashes;
- 0 new baseline fits;
- 0 source/paper mutations outside the new evidence root.

Success token:

`HCH_V44_BASELINE_PAPER_REGISTRY_COMPLETE`

Then stop. Do not start v4.4 training.