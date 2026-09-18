# implementation/ — HCH final geometry-coupled canary source

STATUS:        ACTIVE
STAGE:         hch_final_geometry_coupled_canary_20260918
KIND:          source
SUPERSEDED_BY: -

Authority: `docs/current/HCH_FINAL_GEOMETRY_COUPLED_METHOD_FREEZE_20260918.md` (scientific)
and `docs/current/HCH_FINAL_GEOMETRY_COUPLED_IMPLEMENTATION_EXPERIMENT_PLAN_20260918.md`
(implementation).

## Direct children

| path | role |
| --- | --- |
| `README.md` | this file (L1 index) |
| `gc_common.py` | stage runtime: evidence root, access-guard reuse, O1 reuse provenance, CSV/aggregation helpers, canary cell registry |
| `geometry_prior.py` | deterministic W=7 revealed-residual geometry prior (no learned parameters) |
| `fused_encoder.py` | `FinalSharedEncoder` — one input affine + one 1-layer BiGRU with total output width 32 |
| `global_local.py` | `DayRepairCore` and `SignedMassQueryShapeHead` |
| `final_model.py` | `HCHFinalCore` and the four registered variants through one forward contract |
| `gc_data.py` | legal TRAIN/VAL data path (reuses the frozen v4.4 pipeline primitives) |
| `trainer.py` | the frozen O1 staged-objective schedule |
| `gc_runner.py` | phases: source audit, unit-test gate, fits, checkpoint redump, gate arithmetic, evidence writing |

## `redump`

`gc_runner.py redump` re-forwards each frozen `selected_ema.pt` and writes a new
per-run `val_predictions_selected_ema.npz` beside the run's own file, which it never
overwrites. It exists because of a defect found while verifying the canary: `trainer.py`
wrote `val_predictions.npz` from the *live* EMA, which keeps moving after the selected
check, so for a run whose selected check was not its last check that file describes the
run's last step instead of the model its freeze record describes. The repair to
`trainer.py` is in the same change; the 48 already-frozen runs were **not** retrained and
their own artifacts were **not** rewritten, because a run's `freeze.json` pins their
hashes. `REDUMP_REPORT.json` records the per-run deviations, and the command is
self-validating: every redump must reproduce the recorded `selected_val_mae_ema`, and
must reproduce the run's own npz exactly wherever that npz already was the selected pass.

## Boundary

No file here imports `core.allocation`, `core.model`, `core.encoders`, `core.bundles` or
`core.rescues`; there is no evidence-source router, coordinate embedding, history GRU,
Shape-history encoder, retrieval, safety gate or market expert on the active path.
V2 TEST is unreachable: all frames come from `recovery_common.pretest_role_frame`, which
refuses any role other than `TRAIN`/`VAL`.
