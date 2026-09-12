# NEW_WINDOW_HANDOFF — China-5 post-hoc baseline panel

Evidence root: `experiments/evidence/china5_posthoc_baseline_panel_20260912`  
Terminal state: `CHINA5_BASELINE_PANEL_PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS`

## What a future new-method experiment must do

1. **Do not retrain any Host and do not retrain any baseline.** Load the frozen Host caches from `02_hosts/<MARKET>/<HOST>/host_predictions.npz` and compare like-for-like against the strict rows of `MAIN_OFFLINE_TABLE.csv`.
2. Add exactly one row per (market, host) in the strict offline table for **Our Method**.
3. Fit every scaler / quantile / calibration statistic on `POST_TRAIN` only; never on `DEV_EVAL`, never on `PROTECTED_FINAL`.
4. Reuse the frozen thresholds in `00_protocol/THRESHOLD_FREEZE.json`. Do not recompute them.
5. Emit `pred`, `y_true` and `residual` plus a per-day 24h loss file, hashed into `RAW_PREDICTION_INDEX.csv`.

## Hard prohibitions carried forward

- no per-province architecture / loss / hyperparameter search
- no seed rescue, threshold rescue, or result-conditioned tuning
- no proxy to stand in for a blocked baseline
- no rewriting of any paper-fidelity adjudication
- `PROTECTED_FINAL` stays sealed until Our Method is fully frozen

## Known coverage limits to disclose, not repair

| cell | n_dev_eval_days | host_mae |
|---|---|---|
| SHANDONG__PatchTST | 82 | 104.5452 |
| SHANDONG__TimeMixer | 82 | 103.7590 |
| SHAANXI__PatchTST | 20 | 101.6389 |
| SHAANXI__TimeMixer | 20 | 99.7407 |
| NINGXIA__PatchTST | 7 | 82.4741 |
| NINGXIA__TimeMixer | 7 | 92.5665 |
| QINGHAI__PatchTST | 6 | 110.8599 |
| QINGHAI__TimeMixer | 6 | 114.1517 |


Markets with very few DEV_EVAL days cannot support a reliable post-processing baseline estimate; their cells are reported with the day count attached rather than dropped.
