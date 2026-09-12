# PROTOCOL_FREEZE — China-5 post-hoc baseline panel

Date frozen: **2026-09-12**
Evidence root: `experiments/evidence/china5_posthoc_baseline_panel_20260912/`
Status: **PRE-REGISTERED BEFORE ANY HOST OR BASELINE RESULT WAS COMPUTED**

This document is written after the P0 dataset-contract audit and **before** any
Host training, any baseline run, and any metric computation. Every threshold,
subset definition and table-placement rule below is fixed here so that no later
choice can be conditioned on a result. Any later change to this file invalidates
the panel and must be recorded as a protocol amendment with its own timestamp and
reason, never as a silent edit.

---

## 1. Task semantics (non-negotiable)

- **day-ahead price → day-ahead price**, next-24-hour horizon, hourly, univariate target.
- `target_column = 日前电价` in every market.
- Forecast origin = **00:00 UTC+8 of target day D**, i.e. the first row of day D
  under the hour-ending label convention. Target = `origin+1h … origin+24h`.
- Timezone `Asia/Shanghai`, no DST branch (China has observed no DST since 1991-09-15).
- **The historical `DA → RT` task, target and cache are forbidden.** No DA→RT
  target column, cache or checkpoint may be read, reused or adapted here.

## 2. Data legality rules

A variable may be used as an input **only if it is provably known at the forecast
origin**. Concretely:

1. Only source columns whose name ends in `预测值` (forecast) are admissible as
   target-day inputs. All `*实际值` (realized) columns are excluded.
2. Excluded unconditionally: `日前电价` (the target), `实时电价` (realized RT),
   `竞价空间预测值` and every `竞价空间实际值`. `竞价空间预测值` is excluded
   project-wide because the GANSU audit could not confirm its pre-day-ahead
   publication time, and every province file uses the same source dialect. This
   is a **conservative exclusion**: it may discard a legal feature, which is
   acceptable; it can never admit an illegal one.
3. **No imputation of any kind.** No `ffill`, `bfill`, interpolation or synthetic
   row anywhere in the panel pipeline. `src/utils/common.py`'s `load_province` /
   `load_shandong` do use `ffill().bfill()` and are therefore **not used** by this
   panel; `panel_contract.py` is a separate, projection-based reader.
4. Gaps are **preserved, never patched**. A day is eligible only if (a) it has 24
   contiguous hours and (b) the 168 hours immediately before its origin are the
   calendar-correct contiguous 168 hours. Days failing either test are excluded
   from the eligible-episode set entirely.
5. Duplicate timestamps and non-finite legal features are hard errors.

## 3. Chronological roles — random split is forbidden

Five chronological roles, identical construction in every market, built from
**timestamps only** before any price value is read:

| Role | Fraction of eligible days | Meaning |
|---|---|---|
| `HOST_TRAIN` | 45% (head of the pre-50% block) | Host fitting; sole source of scalers, tail quantiles and thresholds |
| `HOST_VAL` | 5% (tail 10% of the pre-50% block) | Host early-stopping / sanity only |
| `POST_TRAIN` | 15% | Post-processing baseline fitting |
| `DEV_EVAL` | 5% (tail 25% of the 50–70% block) | Development evaluation; the only cell in the main table |
| `PROTECTED_FINAL` | 30% | **SEALED — not read this round** |

Boundaries are `round(0.50n)`, `round(0.70n)`, `round(0.80n)` on the ordered
eligible-day list — the same deterministic rule already frozen for `GANSU_DA`.
Role names used by the older code (`S1 / DIAG_FIT / DIAG_EVAL / S3 / S4`) map
onto these; the mapping is recorded in `EVIDENCE_PROVENANCE_MAP.csv`.

**`PROTECTED_FINAL` is sealed.** The reader is projection-based (`skiprows`), so
its coordinates are never materialised. Sealing is enforced structurally, not by
convention: `load_target_windows` intersects the requested position set against
the closed-role positions and raises on any overlap. The proof of non-access is
the `protected_final_*_reads = 0` counters recorded in every read audit, plus the
absence of any closed-role position from every read request.

**All scalers, quantiles, tail thresholds, normalisation constants and
calibration statistics are fit on their corresponding training partition only** —
never on `DEV_EVAL`, never on `PROTECTED_FINAL`.

## 4. Host contract (frozen, no per-province search)

The maintained Host contract already accepted by the strict GANSU work is reused
**verbatim**. The dataset adapter and the legal feature mapping may change; the
architecture, loss function and hyperparameters may **not** be re-searched per
province. Any province-specific architecture, loss, learning rate, epoch count,
patch length or seed change is a protocol violation.

- `PatchTST`: `seq_len=168, pred_len=24, channels=1`, Adam, `wd=0`, OneCycleLR
  `pct_start=0.3`, `epochs=100`, `batch=128`, `lr=1e-4`, `patience=100`, `seed=7`,
  `cpu`, official params `patch_len=16, stride=8, padding_patch=end, revin=1,
  affine=0, subtract_last=0, decomposition=0, d_model=512, n_heads=8, e_layers=2,
  d_ff=2048, dropout=0.05, fc_dropout=0.05, head_dropout=0.0`.
- `TimeMixer`: Adam, `wd=0`, OneCycleLR `pct_start=0.3`, `epochs=10`, `batch=128`,
  `lr=0.01`, `patience=10`, `seed=7`, `cpu`, official params `d_model=16, d_ff=32,
  e_layers=2, dropout=0.1, down_sampling_layers=3, down_sampling_window=2,
  down_sampling_method=avg, moving_avg=25, decomp_method=moving_avg, use_norm=1,
  channel_independence=1, use_future_temporal_feature=0, label_len=0, embed=timeF,
  freq=h, top_k=5`.
- Loss `MSELoss` on standardised windows; scaling statistics fit on `HOST_TRAIN`
  only; early stopping on `HOST_VAL`.

Each frozen Host must record: effective config, source commit, checkpoint hash,
split manifest and access manifest, and must dump per-day per-horizon raw
predictions, targets and residuals for `POST_TRAIN` and `DEV_EVAL`.

**Host sanity/fidelity gate.** A Host that fails the gate stops **all**
post-processing baselines under it; the cell is recorded as
`HOST_GATE_FAILED` in `BLOCKED_TRACKS.csv` and no baseline number is reported
for it. Gate criteria are fixed here before any run: the Host must (a) produce
finite predictions of the correct `(n_days, 24)` shape for every evaluated day,
(b) beat the training-partition-mean constant predictor on `DEV_EVAL` MAE for
that market, and (c) have `HOST_VAL` MAE not worse than 3× its `HOST_TRAIN` MAE
(a gross-overfit tripwire, deliberately loose so it cannot be used as a tuning
signal). Failing (b) or (c) is reported, not rescued.

## 5. Baseline fidelity — layered, and never rewritten

Fidelity status is a property of the *implementation*, established earlier and
**immutable** here. A successful run on a Chinese market does **not** upgrade any
paper-fidelity adjudication, and a failed or blocked run does **not** downgrade
one.

| Track | Required status | Placement |
|---|---|---|
| `PIR` (full official QualityEstimator + Refiner + official revision/combination path) | `PAPER_FAITHFUL_EXACT_ACCEPTED` | main offline table |
| `delta-Adapter` (audited official-transfer Ada-Y path) | `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` | main offline table |
| `MatchedDirectResidual` | matched-control implementation | main offline table |
| `COSA` (strict online / TTA chronology: predict, then update) | supplementary track | supplementary table only |
| `UEC-STD` | `TRANSFERRED_IMPLEMENTATION / PAPER_FIDELITY_UNRESOLVED` if a source-faithful transferred run is possible, else `BLOCKED` | supplementary table only |
| `OMPB` | runnable only if instantiable **without fabricating an unverified proxy**, else `BLOCKED` | supplementary table only |

No proxy may be invented to fill a blocked track. A blocked track is reported as
blocked.

## 6. Table separation (no mixed-column "best baseline" claim)

- **`MAIN_OFFLINE_TABLE.csv`** contains only: `Host` (identity), `MatchedDirectResidual`,
  `delta-Adapter`, `PIR`, and — in a future window — `Our Method`.
- **`SUPPLEMENTARY_TABLE.csv`** contains `COSA`, `UEC-STD`, `OMPB`, each carrying an
  explicit `setting` and `fidelity_disclosure` column.
- **Mixing supplementary methods into the main table and then declaring a single
  "best strict baseline" is forbidden.** Any "best" statement must name the table
  it is computed within.

## 7. Metric contract (every legal cell)

Computed per `(market, host)` cell on `DEV_EVAL`, with `POST_TRAIN` values
reported alongside but never as the headline:

`Overall MAE`, `MSE`, `RMSE`, `Tail-MAE`, `Upper-tail MAE`, `Lower-tail MAE`,
`Normal-region MAE`, `Normal harm vs Host`, `relative gain vs Host`,
negative-price subset metrics, per-day 24h loss, and the complete raw
prediction/residual arrays.

### 7.1 Tail and subset definitions — FIXED HERE, BEFORE RESULTS

All thresholds are computed **once per market from `HOST_TRAIN` target values
only**, and then applied unchanged to every cell, every method and every
partition. `HOST_TRAIN` is chosen because it is the earliest purely-Host-training
block: it is disjoint from `POST_TRAIN` (used to fit the baselines) and from
`DEV_EVAL` (the evaluation cell), so no threshold is fit on any day that is
either fitted or scored.

- `q05`, `q95` = 5th and 95th percentiles of the pooled `HOST_TRAIN` target values.
- **Upper-tail region** = target hours with `y ≥ q95`.
- **Lower-tail region** = target hours with `y ≤ q05`.
- **Normal region** = target hours with `q05 < y < q95`.
- `Tail-MAE` = mean absolute error over the union of the upper- and lower-tail hours.
- **Normal harm vs Host** = `MAE_normal(method) − MAE_normal(Host)`, in the same
  units; positive means the method damaged the normal region.
- **Negative-price subset** = all target hours with `y < 0`. Reported as
  `n_hours`, `MAE` when `n_hours > 0`, and `NOT_APPLICABLE_N0` when the market has
  no negative target hour in the evaluated partition. (Shandong is the only
  registered market with negative DA prices; that is a **data fact**, not a
  choice.)
- **Per-day 24h loss** = the 24 absolute errors and the 24 residuals of each
  evaluated day, stored in full, keyed by natural day — not aggregated away.

### 7.2 Day-level subsets — FIXED HERE, BEFORE RESULTS

Three day-level subsets, each fixed now:

1. **Extreme day.** For each eligible day `d`, `spread(d) = max_t y(d,t) − min_t y(d,t)`.
   Let `S90` = the 90th percentile of `spread` over `HOST_TRAIN` days only, fixed
   per market. Day `d` is an **extreme day** iff `spread(d) ≥ S90`. The panel
   reports `Extreme-day MAE` and `n_extreme_days` per cell.
2. **Negative-price day.** A day with at least one target hour `< 0`.
3. **Upper-tail day.** A day with at least one target hour `≥ q95`.

These are descriptive stratifications of the *target*, fixed without reference to
any method's error, so they cannot be selected to flatter a method. Any
additional stratification added later is exploratory and must be labelled as such.

## 8. Prohibitions

- No province-specific architecture / loss / hyperparameter search.
- No seed rescue, threshold rescue, or any tuning conditioned on a result.
- No proxy substituted for a blocked baseline.
- No modification of any baseline fidelity status.
- No read of `PROTECTED_FINAL` by any code path, for any reason, this round.
- No training of any new HCH proposal or new method of any kind.
- No re-training of the existing strict GANSU_DA evidence. It is imported by
  integrity/provenance audit only; **table tidiness is not a reason to retrain**.

## 9. Terminal states

Exactly one of:

- `CHINA5_BASELINE_PANEL_FROZEN_FOR_NEW_METHOD_COMPARISON`
- `CHINA5_BASELINE_PANEL_PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS`

## 10. Roster (recorded at freeze time)

The requested roster named five provinces: 山东, 山西, 陕西, 宁夏, 青海.
The P0 audit found **no Shanxi dataset anywhere in the repository or in
`data/MARKET_INDEX.csv`**. Per the controlling repo rule, Shanxi is **not
invented and not silently substituted**; it is recorded as
`ABSENT_FROM_REGISTRY` in `DATASET_REGISTRY.csv` and `BLOCKED_TRACKS.csv`.
The panel therefore executes on the four markets that actually exist:
`SHANDONG`, `SHAANXI`, `NINGXIA`, `QINGHAI`. This is a coverage limitation to be
disclosed, not repaired. `LIAONING` is additionally recorded as `DATA_BLOCKED`
(registered as a candidate but with no price dataset on disk).

## 11. Appendix — contract-construction defects found during P0 (provenance)

Recorded for honesty; all were found by the fail-closed design of the P0 audit
and fixed **before** any Host was trained:

1. `_day_of` normalised to midnight *before* subtracting the hour, mis-assigning
   every row to the previous natural day. Fixed to subtract first.
2. The projection reader's needed-position set omitted the target-day rows
   themselves. Fixed to seed the set with the target positions.
3. The causal-context eligibility check diffed only positions
   `start−SEQ … start−1`, so a gap ending exactly at `start−1` had its displaced
   step excluded from the check and the day was wrongly admitted with a
   non-calendar context. Fixed to diff `start−SEQ … start` inclusive (`SEQ+1`
   slots). This defect was live for SHAANXI/NINGXIA/QINGHAI and is the reason
   their eligible-day counts are lower than the raw row counts suggest.
