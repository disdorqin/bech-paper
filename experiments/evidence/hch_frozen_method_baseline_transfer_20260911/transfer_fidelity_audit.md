# Transfer-fidelity audit — frozen HCH-S1 vs admitted offline baselines

Protocol: `docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md`  
Evidence root: `experiments/evidence/hch_frozen_method_baseline_transfer_20260911/`

## 1. Source-commit lineage (the protocol's B0 requirement)

| method | executed code | commit | sha256 |
| --- | --- | --- | --- |
| PIR-paper-protocol | `PIR/models/PIR.py` | `fc372bb02090da887d4a20b614a6cfecbfd813d0` | `5050b9166d08c809be25c65f37a132c7d5102031b94e7a45a7ad9c4ace54e235` |
| PIR-paper-protocol | `PIR/exp/exp_long_term_forecasting_pir.py` | `fc372bb02090da887d4a20b614a6cfecbfd813d0` | `4997552ac8d205f507fc70872ff4605f918ec26831e93946029b2061ffb10737` |
| PIR-paper-protocol | `PIR/data_provider/data_factory.py` | `fc372bb02090da887d4a20b614a6cfecbfd813d0` | `6e086e41672a0e750f25b4ca7ceff5015bea95ec107f1c366773be22691b05ce` |
| PIR-paper-protocol | `PIR/data_provider/data_loader.py` | `fc372bb02090da887d4a20b614a6cfecbfd813d0` | `c5bf4c5a480233401cb19004a20a85092d810793b4b01ec1c0ab4e67cfab900c` |
| delta-Adapter Ada-Y | `experiments\foundation\reference_deps\official_repos\delta-Adapter\Adapter-X+Y\experiments\exp_post_y_add.py` | `0add06ea7b4d2e0a84c364a8be72eef2676a92f2` | `39fab4132d8a0e44e10917ace41d24d0d8addbf5fba30786f33ae51a71e95291` |
| MatchedDirectResidual | `experiments/current/hch_unified_da_shape_upgrade/run_canary.py::fit_direct` | in-repo frozen control | replayed verbatim |

## 2. Proof that no historical proxy path was used

A static AST scan of every module this stage executed found no reference in
executable code to `PIR_PROXY_10epoch_K10`, `src/baselines/pir.py` or any
ridge/refiner substitute, the old 2-epoch delta wrapper, or COSA/UEC-STD/OMPB.
The stage's docstrings *name* those paths in order to record that they were
avoided; the scan strips docstrings before searching, so prose cannot satisfy it.

- forbidden-token hits: `{}`
- files scanned: `panel.py, transfers.py, pir_transfer.py, run_transfer.py`

The PIR row is executed out of the verified official checkout.  `pir_transfer.py`
re-points the top-level `utils` package at `PIR/utils` and puts the PIR checkout and
its vendored runtime shim ahead of the repository on `sys.path`, so `exp`, `models`,
`layers` and `data_provider` resolve to the official repository rather than to any
local namesake.  `verify_transfer.py` re-checks the loaded module path in a fresh
subprocess, so the stage's own path surgery cannot mask the answer.

## 3. Disclosed task-level adaptations (not tuned)

The official PIR semantics are preserved; four task-level bindings differ from the
ETTh1 anchor and are disclosed rather than optimised:

1. `pred_len = 24` (the frozen day-ahead horizon) instead of the anchor's 96.
2. `seq_len = 168` (the frozen panel's registered context width) instead of 96. The
   frozen Host caches record the same adaptation in their own `official_config`
   (`seq_len: 168, pred_len: 24`), so all compared methods see one information set.
3. `enc_in = dec_in = c_out = 1` — the frozen DA task is a single price channel.
4. `load_pretrained_backbone = 0`, so the official code trains its own backbone on
   this data; no pre-trained checkpoint exists for this panel.

**Backbone-width disclosure.** The PIR row runs the audited anchor P-A1 / P-A2
backbone configs (PatchTST `d_model=16, n_heads=4, e_layers=3`; TimeMixer
`d_model=16`) because those widths are part of the paper-native PIR configuration.
The frozen Host family uses the same *architecture* but its own widths (PatchTST
`d_model=512`, TimeMixer `d_model=16`), which are part of the HCH pipeline rather
than of PIR.  Substituting the HCH-trained backbone would change official PIR
semantics, which the protocol forbids.  The asymmetry is therefore disclosed rather
than removed: the Host row is reported separately from the PIR row, and the PIR row
is labelled `PIR-paper-protocol` exactly as the protocol requires.  Parameter counts
for both are in `efficiency.csv` and `metrics_by_seed.csv`.

One shape-contract repair was required and is recorded here: with one channel
`np.corrcoef` collapses to a 0-d array which the official `torch.FloatTensor(...)`
cannot consume.  `get_person_similarity` therefore returns the declared 1x1 shape.
The official experiment assigns that tensor at
`exp_long_term_forecasting_pir.py:168` and never reads it again (verified by grep
over the whole checkout), so no computation depends on the value.

## 4. Data-flow equivalence and leakage

- **Same rows.** Every method consumes the frozen `DIAG_FIT` / `DIAG_EVAL` rows from
  `panel.build_cells()`, hence the same timestamps, the same DA target and the same
  day-level MAE implementation.
- **GANSU non-contiguity.** `GANSU_DA` drops source-unavailable days, so its frozen
  segments are split into maximal runs of strictly 24h-spaced days.  Every
  sliding-window baseline trains and is served inside a single run, and the
  reconstruction is verified to reproduce each frozen target and each frozen 168h
  context to 1e-9 before any window is emitted.
- **Chronology.** `train` / `val` roles are carved from `DIAG_FIT` only; `DIAG_EVAL`
  is the frozen test set for every method.  PIR's retrieval keys are built from the
  *training* windows only, so no retrieval value can carry a test-period label.
- **PIR self-exclusion mask.** The official mask removes keys within `+/- seq_len`
  *ordinals*.  Windows are enumerated chronologically with stride 1, so ordinal
  distance equals hour distance inside a run; across a run boundary the dropped days
  make ordinal distance strictly *shorter* than hour distance, so the mask can only
  ever remove extra keys.  Over-masking is conservative and cannot expose a key the
  paper-native rule would have hidden.
- **Test-time masking.** The official code masks only `if self.training`; at
  inference the mask is off, but keys still come from the training split only.
- **Units.** The official `test()` reports standardised-space metrics.  The frozen
  metric is in raw price units, so PIR predictions are `inverse_transform`-ed before
  scoring; the renormalisation round-trips the frozen targets to 1.4e-14.
- **Delta-Adapter normalisation.** The bounded `tanh` correction is only meaningful
  in the standardised units the official datasets are served in, so the adapter is
  trained and applied in train-span standardised units and de-standardised before
  scoring.  This is the source normalization choice, disclosed.

## 5. No result-conditioned tuning

- tuning-shaped-pattern scan: `{}`
- every hyperparameter is a module-level constant fixed before any DIAG_EVAL metric
  existed; there is no grid, no configuration selection and no
  DIAG_EVAL-conditioned branch in any executed module.
- the fixed 7-day block bootstrap (1000 replicates, seed 20260911) is robustness
  reporting only; no decision in this stage reads it.

## 6. Fidelity labels in force

| method | label |
| --- | --- |
| Host | `FROZEN_HOST` |
| MatchedDirectResidual | `FROZEN_INTERNAL_CONTROL` |
| PIR_paper_protocol | `PAPER_FAITHFUL_EXACT_ACCEPTED` |
| S1_DailyPatch_GRU32 | `FROZEN_METHOD_REPLAY` |
| delta_Adapter_AdaY | `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` |

COSA appears nowhere in the main offline ranking: no already-audited legal transfer
runner exists for it that requires no rescue, so under the protocol it does not
enter the offline best-baseline comparison and no supplementary table is produced.
UEC-STD and OMPB remain excluded.

## 7. Partition access

`{"S3_reads": 0, "S4_reads": 0, "protected_reads": 0, "final_reads": 0, "Shandong_reads": 0, "Shaanxi_reads": 0, "Ningxia_reads": 0, "Qinghai_reads": 0}`  
roles read: `['S1', 'DIAG_FIT', 'DIAG_EVAL']`

SHAANXI / NINGXIA / QINGHAI / SHANDONG / S3 / S4 / protected / final were never
opened; only the `A_DA` track of `GANSU_DA`, `LAGO_DE` and `LAGO_PJM` was read.
