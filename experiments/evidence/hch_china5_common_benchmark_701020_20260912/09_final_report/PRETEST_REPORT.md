# PRETEST report — `COMMON_BENCHMARK_701020_V1`

**Token: `CHINA5_COMMON_BENCHMARK_PRETEST_VERIFIED`** (20/20 independent checks)
Authority: `experiments/current/hch_china5_common_benchmark_701020/AI_PRETEST_EXECUTION_PROMPT.md`
Stage: `hch_china5_common_benchmark_701020` · Evidence: `experiments/evidence/hch_china5_common_benchmark_701020_20260912/`

PRETEST only. No TEST metric was computed, no method was trained, and the TEST seal held
throughout (`test_label_read_count = 0`, 5 refused probes). Execution stops at this token.

---

## 1. Exact split counts and hashes

Rule: `n_train = (70*M)//100`, `n_test = (20*M)//100`, `n_val = M - n_train - n_test`,
over the currently-OPEN eligible target-day block of size `M`. All five reproduce the
pre-registered counts exactly.

| market | M | TRAIN | VAL | TEST | split-manifest sha256 |
|---|---|---|---|---|---|
| GANSU_DA | 291 | 203 | 30 | 58 | `bc43fd55901749a8afddfaf920031ddbafad7bb47879cead0d71bdf87e90c343` |
| SHANDONG_DA | 1156 | 809 | 116 | 231 | `5fa0158af04ff55eb41287c4f7f9bf01c9965af072661d2a8860be13c6ac38cb` |
| SHAANXI_DA | 284 | 198 | 30 | 56 | `8a510619e7da181a1b9d5267f24b172d2afcaabf47683ddf4d9787229fcc0565` |
| NINGXIA_DA | 101 | 70 | 11 | 20 | `383bd102500c376c10f2ad7f2950a8706b4e4507efd74729f51507ab5d5ba08e` |
| QINGHAI_DA | 94 | 65 | 11 | 18 | `66c8935abe6f80b5c29e17880ec2323b388b5e0d224e371ee18be928f1b24ae9` |

Pre-registered GANSU 203/30/58, SHANDONG 809/116/231, SHAANXI 198/30/56,
NINGXIA 70/11/20, QINGHAI 65/11/18 — **all five match; no count was altered.**

Frozen before any training: `generated_before_any_host_training = true`, `test_seal.lifted = false`.
Per-market `test_inaccessibility_audit.json`: 9/9 checks pass each, `overlap_rows = 0`,
TEST ids are the chronological tail and disjoint from TRAIN ∪ VAL, and the loader reports
`test_target_values_read = 0` / `sealed_final_target_values_read = 0` under projected
parse-time `skiprows` (unrequested rows are never parsed).

TRAIN spans: GANSU 2024-07-13..2025-04-12 · SHANDONG 2022-01-08..2024-03-26 ·
SHAANXI 2025-01-08..2025-08-09 · NINGXIA 2025-10-08..2026-01-13 · QINGHAI 2025-09-11..2025-12-31.

## 2. Twenty-Host readiness

`02_hosts/EXECUTION_HOST_GATE.csv` — **20 rows, `gate_passed = True` for all 20**,
`scaler_fit_partition = TRAIN` (20/20), `selection_partition = VAL` (20/20),
`test_target_values_read = 0` (20/20). `HOST_FREEZE_INDEX.json`: `n_cells = 20`,
`n_gate_pass = 20`, `test_label_read_count = 0`.

All 20 were retrained under the new split; no old low-shot checkpoint was reused
(V07 PASS). QINGHAI's four cells carry `reuse = True` — a **reuse of that same
protocol-scoped smoke run**, not of a low-shot artifact: same split manifest hash, same
source hash, same recipe, and the manifest identity is re-verified by the verifier.

Host sanity gates are TRAIN/VAL only. No TEST Host MAE exists anywhere.

| market | Host | TRAIN | VAL | HOST_VAL_MAE | beats TRAIN-mean constant |
|---|---|---|---|---|---|
| GANSU_DA | LSTM / PatchTST / TimeMixer / iTransformer | 203 | 30 | 93.36 / 73.94 / 79.06 / 81.89 | yes (4/4) |
| SHANDONG_DA | LSTM / PatchTST / TimeMixer / iTransformer | 809 | 116 | 87.33 / 86.70 / 86.37 / 87.78 | yes (4/4) |
| SHAANXI_DA | LSTM / PatchTST / TimeMixer / iTransformer | 198 | 30 | 97.32 / 94.66 / 86.62 / 89.76 | yes (4/4) |
| NINGXIA_DA | LSTM / PatchTST / TimeMixer / iTransformer | 70 | 11 | 69.26 / 84.79 / 102.76 / 91.39 | yes (4/4) |
| QINGHAI_DA | LSTM / PatchTST / TimeMixer / iTransformer | 65 | 11 | 103.87 / 95.85 / 89.99 / 99.16 | yes (4/4) |

## 3. Baseline readiness matrix

`03_baselines/READINESS_MATRIX_20x6.csv` — 20 Hosts × 6 methods = **120 coordinates:
70 numeric-ready, 50 inherited blockers**. This matches the pre-flight authority count
exactly (UEC-STD 20 / OMPB 20 / PIR 10 blocked).

| method | numeric-ready | blocked | note |
|---|---|---|---|
| MatchedDirectResidual | 20 | 0 | `OFFLINE_STATIC_CONTROL` |
| delta-Adapter | 20 | 0 | `OFFLINE_STATIC_POSTHOC` |
| PIR | 10 | 10 | blocked only where the frozen `BACKBONE_CONFIG` has no entry (LSTM, iTransformer) |
| COSA | 20 | 0 | online/prequential; never pooled into an offline ranking |
| UEC-STD | 0 | 20 | inherited |
| OMPB | 0 | 20 | inherited |

Every numeric row declares `fit_partition = TRAIN`,
`scaler_fit_partition = TRAIN`,
`baseline_selection_partition = INTERNAL_FIT_BLOCK_CARVE(frozen panel.VAL_FRACTION)`,
`method_train_semantics = "fit_on_TRAIN; frozen internal VAL_FRACTION carve for early stop"`,
and `test_used_for_fitting_or_selection = false`. Every blocked row declares
`NOT_APPLICABLE_NO_FITTING` for both partition fields and
`NOT_FITTED — inherited blocker; nothing was fitted or selected on any partition` —
a coordinate that was never fitted never claims a fit. All numbers are **VAL** readiness
proofs, not TEST results.

## 4. Inherited blockers

| method | count | blocker code | class | authority |
|---|---|---|---|---|
| UEC-STD | 20 | `UECSTD_TRANSFER_BLOCKED` | `Q-FIDELITY` | `registry/BLOCKERS.csv` |
| OMPB | 20 | `OMPB_DATA_OR_FIDELITY_BLOCKED` | `Q-DATA` | `registry/BLOCKERS.csv` |
| PIR | 10 | `PIR_FROZEN_BACKBONE_CONFIG_ABSENT_FOR_NEW_HOST` | `Q-INCOMPATIBLE` | `pir_transfer.BACKBONE_CONFIG` |

Carried as outcomes, never reopened, never replaced by a proxy: each blocker record has
`inherited = true`, `reopened = false`, `retrained = false`, `proxied = false`,
`numeric_value_recorded = false`, `proxy_object_created = false`. No blocker was changed,
so no separate authority was required or used.

## 5. Core tree digest

`04_core/CORE_TREE_FREEZE.json` — `src/core`, **13 files**:

```
tree_digest_sha256 = 9B278C5A22C8E88D66A0D6AAD8347C55799BF8A4152484DEDF1E0E8EC695D7AB
tree_digest_after_tests = 9B278C5A22C8E88D66A0D6AAD8347C55799BF8A4152484DEDF1E0E8EC695D7AB
unchanged_by_this_stage = true
```

Per-file SHA256 is recorded for all 13. Asserted and passing: `HISTORY_DAYS == 7`
(`HISTORY_DAYS_ASSERTION.json`, 8/8 checks — the bank constructor takes no capacity, the
warm start takes no length, no capacity literal exists, and warm-start provenance is
out-of-fold/prequential); candidate architecture identity
(`c_h = A*(S+_h − S-_h)`, `MinimalSignedRedistributionModel`, 22,723 parameters at the
default config, deterministic multiply-and-subtract fusion, config field names frozen);
safety-formula identity (`R(λ)`, `A_align`, `κ`, `W1` — all identities re-derived and
matched, `module_sha256 = 06D84B5D…`); and deleted components proven absent
(`DELETED_COMPONENTS_ABSENCE.json`, structural tier clean — no forbidden definitions,
flags, imports, module imports, dataset/split/Host literals or baseline names, and GRU is
the only recurrent encoder). Plain-text mentions of deleted component names are reported
separately and are never decisive: the package documents its own absences in prose.

## 6. Method-training manifest

`04_core/METHOD_TRAINING_MANIFEST.json`

```
method_training_manifest_sha256 = B350865FF629D7E736F21E928A9A0100ED0F49D755208A46064E81B034319675
status = FROZEN_PRETEST_NO_TEST_RESULT
```

Frozen: TRAIN/VAL/TEST partition roles; training-pair production; seeds `[7, 17, 37]`
(never selected by seed); optimizer; epochs/early-stop; loss weights; TRAIN-only
normalisation with the amplitude scale; `alpha_0` fitted on VAL with `c_i = 0` coordinates
removed before the ratio and `alpha = 0` on all-zero; the exact seven-record VAL safety
warm-start construction; M0/M1/M2 definitions; and the M1/M2 TEST prequential bank-update
rule (M0 excluded). The hash reproduces by pruning the hash key and `generated_utc` and
re-hashing with `sort_keys=True, ensure_ascii=False`.

`method_training_executed = false`, `method_results_present = false`,
`method_numeric_objects_produced = 0`.

**Blocker carried into JOINT_TEST.** The existing harness cannot express the new split
without changing scientific semantics:
`SIGNED_MASS_HARNESS_ROLE_TAXONOMY_INCOMPATIBLE_WITH_COMMON_BENCHMARK` (class `Q-HARNESS`,
verdict `HARNESS_CANNOT_EXPRESS_TRAIN_VAL_TEST_WITHOUT_CHANGING_SEMANTICS`). It declares
only `POST_TRAIN` / `DEV_EVAL` / `PROTECTED_FINAL`;
`has_no_train_role`, `has_no_val_role`, `has_no_test_role` and
`adapter_raises_on_unmaterialised_role` are all true. Repointing `POST_TRAIN → TRAIN` and
`DEV_EVAL → VAL` would keep the old names and redefine the data contract, which the prompt
forbids. Per §8 the stage therefore **stopped with a concrete blocker rather than
improvising with old POST_TRAIN roles**; TRAIN/VAL method fitting is optional in PRETEST
and was not performed.

## 7. Thresholds

`05_thresholds/` — q05, q95 and the S90 extreme-day threshold frozen from **TRAIN only**,
per market, with exact source row/day ids and hashes persisted. V16 re-derives them
independently from the source and reproduces them bit-exactly.

Target-representation disclosure (amended during PRETEST; **no threshold value changed**):
`cb_contracts.load_windows` builds every window as `np.float32`, so the frozen thresholds
are float32-representation quantiles, not float64 ones. Each market records
`float64_from_source` values, the `max_abs_deviation_float64_vs_float32`, and a rule that
the verifier re-derives. Max absolute deviation: GANSU 2.929687502728484e-05, SHANDONG
5.859375005456968e-05, SHAANXI 2.929687502728484e-05, NINGXIA 1.5136718729991117e-05,
QINGHAI 2.6855468831854523e-05. Metric code and units were frozen before TEST opens, and
normal-region harm is stated in **absolute units** with relative percentages labelled as
such — the historical interpretation bug is not repeated.

The pre-amendment "before" attestation is preserved outside the frozen tree at
`experiments/current/hch_china5_common_benchmark_701020/pretest_verification_history/`
(byte-identical dry-run report + `README.json` quoting all 15 pre-amendment thresholds), so
the claim that no threshold changed rests on an independent re-derivation rather than on
assertion.

## 8. TEST-label read count

**0.** `SPLIT_INDEX.json`: `test_label_read_count = 0`, `test_seal.lifted = false`,
`refused_attempts = 5` (one audit probe per market), each refused with
"TEST targets are sealed during PRETEST". The only legal reader is `TestSeal.reveal()`,
which raises while the seal holds.

## 9. Old strict-results lab — before / after

```
experiments/lab/strict_results
  digest_before = 1d3850d74f225f18a9dd24f9f0a11dea7bcada7d3d97d5a8889334f442492e41  (30 files)
  digest_after  = 1d3850d74f225f18a9dd24f9f0a11dea7bcada7d3d97d5a8889334f442492e41  (30 files)
  file_list_identical = true,  unchanged = true
```

Re-hashed live at report time and still identical. Nothing was written into it; its
interpretation tag remains `HARD_FROZEN_LOW_SHOT_TRANSFER` with
`modification_authorized = false`.

## 10. New common-benchmark lab scaffold

```
experiments/lab/common_benchmark_results
  n_files = 19   digest = 11a164fb952bb983fe670b0e47a889d32aa4525d204c6f6d3de6ae2b7295b920
  registry_rows = 0,  test_numeric_rows = 0
  state = SCAFFOLD_ONLY_NO_NUMERIC_ROWS
```

Created scaffold-only: 4 README/policy files, `registry/` (8 empty CSV skeletons),
`paper_tables/` (4 empty tables + README), `audit/`, `snapshots/`, `implementation/`.
`registry/RESULTS_LONG.csv` carries no numeric row, and no `split == TEST` row can be
written while the seal holds.

## 11. Exact test PASS counts

| suite | result |
|---|---|
| active-core regression (`test_core_candidate`, `test_core_safety`, `test_core_purity`, `test_canonical_core_layout`) | **136 passed, 0 failed** (`136 passed in 7.42s`, returncode 0) |
| independent PRETEST verifier (P6) | **20 / 20 PASS** |
| split inaccessibility audits | 5/5 markets, 9/9 checks each |
| Host gates | 20/20 |
| `HISTORY_DAYS` assertion | 8/8 |
| deleted-components absence | pass (structural tier clean) |
| safety-formula identity | pass (all identities re-derived) |

Full per-check detail: `08_pretest_verification/PRETEST_VERIFICATION.json`
(`verifier_test_rows_used = 0` — the verifier imported nothing from the production
runner; its imports are stdlib plus `numpy`/`openpyxl`/`pandas` only, proved statically
by AST inspection).

## 12. Implementation repairs, with before/after hashes

Three implementation defects were found by cross-checking the verifier against the
artifacts it audits, plus one ordering defect in the lab builder. **Before-hashes are not
recoverable for any of them**: all three stage paths are untracked in git
(`?? experiments/current/hch_china5_common_benchmark_701020/`,
`?? experiments/evidence/hch_china5_common_benchmark_701020_20260912/`,
`?? experiments/lab/common_benchmark_results/`), and the files were edited in place, so no
committed predecessor exists to hash. This is disclosed rather than reconstructed from
displayed text. Only after-hashes are given.

| # | file | after sha256 | what was wrong |
|---|---|---|---|
| 1 | `cb_p6_verify.py` | `FBFF97AF7E055C32A71427C609DD74E7C71540A9EEB05845BD0EBF0BAEC8D01B` | V07 matched its own stage directory by exact path-part equality; the stage dir is dated, so the stage's own artifacts were compared against themselves and read as a substitution |
| 2 | `cb_p6_verify.py` | (same file) | V11 indexed baseline rows by a per-method directory that does not exist; every coordinate would have reported "ready but no numeric object" while silently finding nothing on a *blocked* coordinate even when one existed |
| 3 | `cb_p3_baselines.py` | `9B49103A057596807201F1CAF4B5AB2444920903B8A9FC1FC112E67CAACDBC1B` | V10 required fit/selection semantics the readiness matrix had no columns for; the four semantic columns were added to P3 and V10 was made correct per ready/blocked |
| 4 | `cb_lab.py` | `C0D42280FF54B5D93DE36EFF46A29A4B6743194CCB99BE401C467E01A255AE16` | the lab digest was taken *before* `audit/build_state.json` was written into the lab, so the recorded digest was correct but its coverage was undocumented and not re-derivable |

`cb_p5_thresholds.py` (`A9159BB5CA3746F8F789914A5B4F613141A4546AFD2BA0DE35D1E769482C3F81`)
was amended additively for the float32 disclosure in §7; no threshold value changed.

### 12a. The V19 amendment, and the controls that keep it honest

V19 initially FAILed on `window_counts.test` in the PIR sidecars. This was investigated
rather than waived. The frozen `pir_transfer` entry point splits the cell it is handed
with `panel.role_of_day(i, n_fit, n_val)` and calls the third slice `test`. Deriving that
splitter against the day indices the cell actually holds shows, for all five markets, that
the slice it calls `test` **is the protocol VAL block** — GANSU 30, SHANDONG 116,
SHAANXI 30, NINGXIA 11, QINGHAI 11 days, exactly each market's VAL count — and that
**zero protocol TEST days are present in any cell** (GANSU holds days 0..232; protocol TEST
begins at index 233). It is a name collision, not a TEST read.

The fix does not make V19 ignore the key. P3 now derives
`03_baselines/INTERNAL_ROLE_MAP.json` by calling the baseline's own splitter, and V19
admits a numeric TEST key **only** when the exact key is governed by that map, the map
attributes the role to TRAIN or VAL, the cell holds zero protocol TEST days, and the
derived day count equals the frozen manifest count. Six adversarial controls confirm the
check still bites (each run against the final V19, with byte-exact restoration verified
afterwards):

| control | expected | observed |
|---|---|---|
| untampered | PASS | PASS |
| map removed | FAIL | FAIL |
| `sealed_partition.days_in_cell = 1` | FAIL | FAIL |
| role `n_days = 999` (manifest cross-check) | FAIL | FAIL |
| role attributed to `TEST` | FAIL | FAIL |
| `meta.test_mae` injected into a governed PIR artifact | FAIL | FAIL |
| `meta.test_mae` injected into an ungoverned artifact | FAIL | FAIL |

The map's own attestation fields are named so they do not present as numerics under
TEST-named keys (`sealed_partition`, `numeric_sealed_partition_objects_created`); the
facts are unchanged, only the field names, so the governance document describes the sealed
partition instead of masquerading as a number about it. This naming choice is disclosed
here as a deliberate decision.

## 13. Confirmation that no TEST metric exists

**No TEST metric exists — for any Host, any baseline, or any method.** Specifically:

- `test_label_read_count = 0`; the seal held and refused 5 probes.
- No Host TEST MAE: Host gates are TRAIN/VAL only; `EXECUTION_HOST_GATE.csv` reports
  `test_target_values_read = 0` on all 20 rows.
- No baseline TEST number: every baseline row is a VAL readiness proof
  (`readiness_partition = VAL`, `pretest_note` states it is a readiness proof, not a TEST
  result).
- No method TEST number: `method_training_executed = false`, `method_numeric_objects_produced
  = 0`, `P4_SUMMARY.test_metric_present = false`,
  `METHOD_TRAINING_MANIFEST.method_results_present = false`, and no `.npz` carries a TEST
  segment.
- COSA was never run across TEST and no M0/M1/M2 was run on TEST. No TEST target was
  exposed to any fitting or selection code.
- No protected/final partition was read (`sealed_final_target_values_read = 0`), and no old
  role was repurposed as TEST (V20 PASS).
- The new lab holds no TEST numeric row (V18 PASS).

---

## Stop

Token: **`CHINA5_COMMON_BENCHMARK_PRETEST_VERIFIED`**

Execution stops here. JOINT_TEST is **not** opened automatically. Before it can be, the
`Q-HARNESS` blocker in §6 must be resolved under separate authority: a conforming harness
must implement TRAIN/VAL fitting under the frozen method-training manifest, without
repointing the old `POST_TRAIN`/`DEV_EVAL` role names.
