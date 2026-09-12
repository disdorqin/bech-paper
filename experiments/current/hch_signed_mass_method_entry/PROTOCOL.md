# Signed-mass China-5 method harness — execution protocol

Status: **scientific development execution authorized by the user on 2026-09-12, subject to the mandatory pre-run contract patch and verifier PASS described in `AI_SCIENTIFIC_EXECUTION_PROMPT.md`.** No signed-mass `DEV_EVAL` result existed when this authorization was recorded. `PROTECTED_FINAL` remains sealed.

This file is the operational contract. The scientific contract is
`src/core/DESIGN_CONTRACT.md`; the entry contract is
`docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`.

---

## 1. What is being run

The frozen object is the signed-mass decomposition owned by `src/core`:

```
r_h        = y_h − ŷ_h^H
r_h^±      = max(±r_h, 0)                       A^± = Σ_h r_h^±
S_h^±      = r_h^± / A^±                         (Σ_h S^± = 1)
ĉ_h        = Â⁺Ŝ⁺_h − Â⁻Ŝ⁻_h
ŷ_h^R      = ŷ_h^H + α ĉ_h                       α ≥ 0, one scalar
```

Nothing in this package restates those equations. `src/core` computes them; this
package only decides which rows are legal to hand it.

## 2. Coordinates

5 markets × 4 frozen Hosts = **20 market×Host cells**, times 5 registered
configurations, times 3 seeds `{7, 17, 37}`.

| market | Hosts |
|---|---|
| `GANSU_DA`, `SHANDONG_DA`, `SHAANXI_DA`, `NINGXIA_DA`, `QINGHAI_DA` | `PatchTST`, `TimeMixer`, `iTransformer`, `LSTM` |

The only thing that varies across markets is the legal forecast-known feature
width `F ∈ {9, 8, 7, 7, 4}`. The adapter is contract-driven; it never branches on
a market name.

Every Host prediction is **frozen**. This package contains no code that can
train, fine-tune or re-fit a Host or a baseline — asserted statically by
`tests/test_harness.py::test_14` and by the pre-execution verifier.

## 3. Role taxonomy and chronology

Roles are strictly chronological. Boundary ratios are registered once, in
`config.SPLIT_BOUNDARY_RATIOS`, and are identical for every market and Host:

```
s1_end = 0.50   s2_end = 0.70   s3_end = 0.80
host_val_tail_of_s1 = 0.10      dev_eval_tail_of_s2 = 0.25
```

```
|———— HOST_TRAIN ————|—HOST_VAL—|—— POST_TRAIN ——|—— DEV_EVAL ——|—— PROTECTED_FINAL ——|
                     (10% of s1)                   (25% of s2)     sealed, never materialised
```

`GANSU_DA` is served by the frozen *legacy* contract instead of a
`DATASET_CONTRACT.json` panel, with a different native role vocabulary
(`S1 / DIAG_FIT / DIAG_EVAL`). The mapping is the repo's own, disclosed verbatim
in `config.GANSU_LEGACY_ROLE_MAP`, and is the single registered market-literal
branch in the package (a schema dispatch, not a scientific one).

**What each role may be used for:**

| role | scaler / residual scale / mass stats | model fitting | early stopping | α calibration | evaluation |
|---|---|---|---|---|---|
| `HOST_TRAIN` | no | no | no | no | no |
| `HOST_VAL` | no | no | no | no | no |
| `POST_TRAIN` | **fold prefix only** | **fold prefix only** | inner-val suffix only | **OOF only** | no |
| `DEV_EVAL` | no | no | no | no | **yes, once, in a separately authorized round** |
| `PROTECTED_FINAL` | never | never | never | never | never |

`DEV_EVAL` outcomes may not influence any fitted quantity. This is structural,
not disciplinary: `oof.fit_method` cannot see those rows, and
`tests/test_harness.py::test_09` poisons them by ±1e6 and asserts the fitted
method is bit-identical.

`PROTECTED_FINAL` is unreachable because the canonical contracts declare it a
*closed* role, so no frozen artifact's `segment` array ever carries it.
Read count must remain **0**, witnessed three independent ways
(identity / refusal / absence) by `verify_preexecution._sealed_accounting`.

## 4. Fitting order (registered, not tunable per cell)

```
POST_TRAIN  ─┬─ fold 0 .. fold k−1   (expanding, chronological, disjoint)
             │     scaler, residual-history scale, robust Amplitude output scale s_A,
             │     mass normalisation and rare-mass strata fitted on that fold's
             │     inner-train prefix ONLY
             │     early stopping on that fold's inner-val suffix ONLY
             │     → out-of-fold corrections on the fold's holdout block
             ├─ α  = argmin_{α≥0} Σ |r_i − α c_i|   over the concatenated OOF rows
             ├─ epochs = clip(round(median(fold best epochs)), 1, 50)
             └─ final re-fit on ALL of POST_TRAIN, for that many epochs
```

Fold rule (one global rule, a function of data volume only):

```
oof_fold_count(n) = clip((n − 8) // 4, 1, 5)
inner-val size    = max(2, round(0.2 · len(fold train prefix)))
```

α is one scalar per (market, Host, configuration, seed) — never per instance,
per horizon step, or per market. It is the exact weighted median of `r_i / c_i`
with weights `|c_i|`, restricted to `c_i ≠ 0`, and constrained nonnegative. The
solution is `src/core`'s; `calibration.py` contains no fitting mathematics.

## 5. Registered configurations

Exactly four components are removable, and each ablation changes exactly one
switch. `KNN` is **off** in all five and in every future configuration.

| config | `shape_semantic_context` | `use_tcn` | `untied_amplitude_heads` | `rare_mass_sampling` |
|---|---|---|---|---|
| `FULL` | ✔ | ✔ | ✔ | ✔ |
| `NO_SHAPE_CONTEXT` | ✘ | ✔ | ✔ | ✔ |
| `NO_TCN` | ✔ | ✘ | ✔ | ✔ |
| `TIED_AMPLITUDE` | ✔ | ✔ | ✘ | ✔ |
| `NO_RARE_MASS` | ✔ | ✔ | ✔ | ✘ |

`rare_mass_sampling` changes only the training batch order, never the
architecture, and has no test-time effect whatsoever.

## 6. Training defaults (registered; no per-market or per-Host tuning)

`AdamW` · LR `1e-3` · weight decay `1e-4` · batch `32` · max epochs `50` ·
patience `8` · seeds `{7, 17, 37}`.

## 7. Metrics

Reported per (cell, config, seed). q05 / q95 / day-spread-p90 are **market-data
thresholds**, lifted verbatim from the frozen
`00_protocol/THRESHOLD_FREEZE.json` (itself fitted on `HOST_TRAIN` only), never
recomputed per Host or per method. Every metric record carries
`thresholds_used.recomputed_for_this_evaluation = false`, and
`metrics.load_thresholds` fails closed for a market with no frozen record.

Before `DEV_EVAL` is read, each market×Host cell also freezes training-only
high-mass cutoffs `q90(A+)` and `q90(A-)` from `POST_TRAIN`; these are used only
for the A4 rare-mass mechanism metrics.

Mechanism metrics must use the model's direct branch outputs `S+`, `S-`, `A+`,
`A-`. They must not reconstruct branch outputs from the already-cancelled net
correction `c = A+S+ - A-S-`.

Definitions live in `metrics.METRIC_DEFINITIONS` and are emitted alongside every
number, so a reported figure is never separable from what it means.

## 8. No-go list

This package must never contain, and the verifier asserts it contains none of:

* a Gate / TrustGate / RepairabilityGate / BenefitGate / ConfidenceGate;
* a benefit, repairability or routing predictor; a bridge; attention;
  a Transformer; an MoE; a router; a province expert; a market-specific model;
  per-market feature selection;
* any Host or baseline training / fine-tuning / re-fitting path;
* a local re-definition of a core routine (shape loss, amplitude loss, fusion,
  residual decomposition, weighted median, MAE scalar, `TrainFrozenScaler`);
* any write into a frozen Host artifact root or historical evidence root.

## 9. Running it

All commands run from the repository root.

```bash
# 1. input audit → experiments/current/.../HOST_INPUT_READINESS.csv (20 rows)
python experiments/current/hch_signed_mass_method_entry/runner.py readiness

# 2. pre-execution verification → evidence/.../06_audits/PREEXECUTION_AUDIT.md
python experiments/current/hch_signed_mass_method_entry/runner.py verify

# 3. integration smoke (synthetic + one real cell; never a scientific result)
python experiments/current/hch_signed_mass_method_entry/runner.py smoke

# 4. the execution plan (fits nothing)
python experiments/current/hch_signed_mass_method_entry/runner.py run

# 5. the test suite
python -m pytest experiments/current/hch_signed_mass_method_entry/tests/ -q
```

Step 4 executes only with `--execute`, and touching `DEV_EVAL` additionally
requires `--authorization-token SIGNED_MASS_METHOD_EXECUTION_AUTHORIZED`. The user has now explicitly authorized the scientific development round, but the token may be used **only after** the mandatory execution-contract patch, all relevant tests, and the pre-run verifier pass. The scientific run must cover all legal 5 markets × 4 Hosts; no Host may be silently omitted.

### One disclosure about step 5, so the record is exact

Step 5 is not purely inert. `tests/test_harness.py::test_15` and
`tests/test_scaler_prefix.py` exercise the pipeline **on real frozen artifacts**,
because the property they establish — that using this package writes nothing and
reads nothing sealed — is only meaningful against the real inputs. Concretely,
the suite performs a small number of one-epoch fits on `POST_TRAIN` prefixes.

None of that is a scientific result or a scientific execution: no registered
schedule is run (1 epoch, not the fold-derived count), no `DEV_EVAL` row is read,
no metric is computed or reported, no evidence file is written, and no `alpha`
from those fits appears anywhere. The registered 5 × 4 × 5 × 3 grid is still
unrun, and `00_protocol/RUN_MANIFEST.json` — written only by
`run --execute` — does not exist.

## 10. Missing Host artifacts

A cell whose frozen Host artifact is absent is recorded as
`PENDING_EXTERNAL_HOST_ARTIFACT` in `HOST_INPUT_READINESS.csv` and is **not**
trained, **not** substituted with another Host, and **not** back-filled from a
historical cache. There is no code path that could do otherwise: the readiness
table is the only source of truth for which cells may be loaded.
