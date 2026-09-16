# HCH v4.4 Domestic Baseline Comparison Freeze -- Registry

Protocol: `HCH_V44_BASELINE_COMPARISON_FREEZE_20260916`  
Design: `docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md`  
Status: **READ-ONLY BASELINE RECONSTRUCTION / ZERO NEW FITS**

## 1. What was built

A 140-coordinate registry over 5 markets x 4 Hosts x 7 methods: **90 numeric + 50 blocked**. No model was fitted.

Numeric values are recomputed from the frozen V2 TEST arrays by calling the frozen parent metric definition (`experiments/current/china5_posthoc_baseline_panel/panel_metrics.py::cell_metrics`) verbatim, under the thresholds frozen in `05_thresholds/THRESHOLD_FREEZE.json`. No metric formula is reimplemented here.

## 2. Metric authority, and why the sidecars are not it

The numeric authority is `RESULTS_LONG.csv` (`row_kind=SEED_OR_HOST`, `data_adjudication=V2_MISSING_TARGET_SCORABILITY_ADJUDICATION_20260913`), as PROTOCOL section 6 requires. Every one of the 90 numeric MAE values reproduces that authority **exactly** (0 mismatches).

The per-cell JSON sidecars are **not** usable as the numeric authority: they carry `n_test_values = registered*24`, and on SHANDONG_DA they record `Overall_MAE` and `Normal` as `NaN`. That is a property of the frozen `cell_metrics`: `NaN >= q95` is False, so non-finite-target hours fall out of the tail masks, but `normal = ~(up | lo)` is True for them, so they enter the normal complement. The missing-target-adjudicated artifacts exist precisely to resolve this, and they are what this registry uses.

For the 70 cells that have a sidecar, the sidecar was scored from float64 source values while the stored array is float32. Agreement is therefore to ~1e-9 relative, not bit-exact. That deviation is recorded per cell rather than papered over.

## 3. Scoring mask

The mask is derived from the arrays (drop days whose target vector is non-finite) and cross-checked against the declared `SCORABLE_DAY_MANIFEST.json`. The two agree exactly:

| Market | Registered | Scored | Unscorable days |
|---|---:|---:|---|
| GANSU_DA | 83 | 83 | -- |
| SHANDONG_DA | 330 | 329 | 2026-07-17 |
| SHAANXI_DA | 81 | 81 | -- |
| NINGXIA_DA | 28 | 28 | -- |
| QINGHAI_DA | 27 | 27 | -- |

## 4. Method census

| Method | Method setting (canonical) | row `setting` | Numeric | Blocked | Blocker class |
|---|---|---|---:|---:|---|
| Host | HOST_REFERENCE | HOST_REFERENCE | 20 | 0 | -- |
| MatchedDirectResidual | OFFLINE_STATIC_CONTROL | OFFLINE_STATIC_CONTROL | 20 | 0 | -- |
| delta-Adapter | OFFLINE_STATIC_POSTHOC | OFFLINE_STATIC_POSTHOC | 20 | 0 | -- |
| PIR | OFFLINE_STATIC_POSTHOC | OFFLINE_STATIC_POSTHOC / BLOCKED | 10 | 10 | Q-INCOMPATIBLE |
| COSA | ONLINE_TTA | ONLINE_TTA | 20 | 0 | -- |
| UEC-STD | OFFLINE_STATIC_POSTHOC | BLOCKED | 0 | 20 | Q-FIDELITY |
| OMPB | ONLINE_SHIFT | BLOCKED | 0 | 20 | Q-DATA |

A blocked coordinate carries `setting = BLOCKED` (design doc section 5 vocabulary) because it has no evaluation setting. Its method's own setting is preserved in `method_canonical_setting`, taken from the frozen blocker object where that object declares one: UEC-STD declares `OFFLINE_STATIC_POSTHOC` and OMPB declares `ONLINE_SHIFT`. The raw frozen declaration is kept verbatim in `frozen_declared_setting` (the PIR blocker declares none).

## 5. Headline numbers (ABSOLUTE_PRICE, TEST, common benchmark)

| Market | Host | Host MAE | MDR | delta-Adapter | PIR | best offline external | COSA (online) |
|---|---|---:|---:|---:|---:|---|---:|
| GANSU_DA | PatchTST | 69.1333 | 81.1584 | 68.6771 | 66.1709 | PIR | 67.9641 |
| GANSU_DA | TimeMixer | 74.7445 | 82.9857 | 74.3540 | 67.9048 | PIR | 73.2364 |
| GANSU_DA | iTransformer | 73.4233 | 84.2834 | 73.6543 | BLOCKED | delta-Adapter | 73.3541 |
| GANSU_DA | LSTM | 81.1232 | 83.2088 | 81.4406 | BLOCKED | delta-Adapter | 77.2128 |
| SHANDONG_DA | PatchTST | 91.8742 | 94.5720 | 91.0835 | 92.2487 | delta-Adapter | 90.8319 |
| SHANDONG_DA | TimeMixer | 94.1964 | 97.1711 | 94.3810 | 90.4308 | PIR | 93.5426 |
| SHANDONG_DA | iTransformer | 93.6810 | 95.6113 | 92.5824 | BLOCKED | delta-Adapter | 92.8897 |
| SHANDONG_DA | LSTM | 93.6898 | 96.2569 | 93.9562 | BLOCKED | delta-Adapter | 92.9546 |
| SHAANXI_DA | PatchTST | 123.5642 | 172.0496 | 123.4261 | 114.7598 | PIR | 123.2570 |
| SHAANXI_DA | TimeMixer | 121.6104 | 150.9404 | 121.6571 | 113.5060 | PIR | 119.6186 |
| SHAANXI_DA | iTransformer | 127.7551 | 161.6940 | 126.9718 | BLOCKED | delta-Adapter | 127.0890 |
| SHAANXI_DA | LSTM | 119.3908 | 140.8519 | 119.9586 | BLOCKED | delta-Adapter | 117.2492 |
| NINGXIA_DA | PatchTST | 79.7787 | 123.2620 | 79.7389 | 74.1339 | PIR | 79.2852 |
| NINGXIA_DA | TimeMixer | 96.6999 | 112.0954 | 96.2050 | 76.3741 | PIR | 95.8385 |
| NINGXIA_DA | iTransformer | 85.1584 | 121.9261 | 84.5792 | BLOCKED | delta-Adapter | 84.8822 |
| NINGXIA_DA | LSTM | 71.5981 | 122.3823 | 71.4227 | BLOCKED | delta-Adapter | 71.4492 |
| QINGHAI_DA | PatchTST | 100.7470 | 139.2288 | 101.3957 | 101.5347 | delta-Adapter | 99.9693 |
| QINGHAI_DA | TimeMixer | 101.6131 | 136.5170 | 101.2246 | 99.5973 | PIR | 100.4284 |
| QINGHAI_DA | iTransformer | 106.6802 | 147.2014 | 107.3290 | BLOCKED | delta-Adapter | 106.1417 |
| QINGHAI_DA | LSTM | 140.7458 | 150.4949 | 139.4846 | BLOCKED | delta-Adapter | 138.6489 |

`relative_gain_vs_Host_pct` uses the Host-minus-method convention (positive = method better). The two normal-region harm fields use the opposite, method-minus-Host convention (positive = method worse) and are published in separate unit-tokened columns, never ranked together.

## 6. Blocker disposition

All 50 blockers are inherited frozen objects, reopened = false, proxied = false. No proxy value was created for any of them. BLOCKED rows carry the market's registered TEST-day count and leave the scored count empty, because a blocked coordinate has no scored day.

## 7. What this round does NOT claim

- It does not train, retrain, or fine-tune anything. New fits = 0.
- It does not reopen UEC-STD, OMPB or PIR-new-Host blockers.
- It does not modify any parent artifact or any file under `paper/**`.
- It does not treat the frozen V2 TEST as untouched-final evidence; every numeric row carries `COMMON_BENCHMARK_TEST_WITH_HISTORICAL_EXPOSURE_CAVEAT`.
- It does not start v4.4 training.

## 8. Files

- `MATRIX_140.csv`
- `STRICT_OFFLINE_TABLE.csv`
- `ONLINE_SUPPLEMENTARY_TABLE.csv`
- `BLOCKED_ROWS.csv`
- `CELL_COMPARATOR_SNAPSHOT.csv`
- `CELL_COMPARATOR_SNAPSHOT.json`
- `SOURCE_HASHES.json`
- `BASELINE_COMPLETENESS_AUDIT.json`
- `VERIFICATION_REPORT.json`

`VERIFICATION_REPORT.json` is produced by `implementation/verify_registry.py`, which does not import the registry constructor and does not import `panel_metrics` -- it re-implements the frozen region rules and recomputes every value from the parent arrays.

## 9. Superseded artifacts, disclosed

This directory previously held output from a competing instruction file in the same protocol folder. The two documents in `experiments/current/hch_v44_baseline_comparison_freeze_20260916/` prescribe **different** experiments:

- `PROTOCOL.md` -- this registry: read-only, 140 coordinates, 90 numeric, zero new fits.
- `AI_EXECUTION_PROMPT.md` -- 100 coordinates and **50 new training runs**, and it names `experiments/current/hch_v44_five_baseline_supplement_20260916/PROTOCOL.md` as its mandatory first read. **That directory does not exist.**

The controlling design document resolves this in favour of `PROTOCOL.md` and forbids the prompt's training outright: "Any new baseline training merely to fill UEC/OMPB/PIR blocker cells would change method fidelity and is forbidden", and "Expected new baseline neural fit count is `0`". The superseded files are preserved, unmodified, under `superseded_ai_execution_prompt_20260916/` with a README explaining their status. They are **not** part of this registry and were not verified by it.

## 10. Reproducing

```
python experiments/current/hch_v44_baseline_comparison_freeze_20260916/implementation/build_registry.py
python experiments/current/hch_v44_baseline_comparison_freeze_20260916/implementation/verify_registry.py
```
