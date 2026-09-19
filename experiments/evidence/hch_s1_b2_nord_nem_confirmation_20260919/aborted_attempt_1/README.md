# Aborted first sealed attempt — quarantined, disclosed

These files are the complete output of the **first** execution of
`implementation/confirmation.py` against the P0 certificate written at 14:02 on
2026-09-19. That execution terminated in about one second with
`AttributeError: 'NoneType' object has no attribute 'load_state_dict'` on all eight
cells, and wrote `verdict = HCH_S1_B2_NORD_NEM_INVALID` with header-only CSVs.

They are kept, not deleted, and moved out of `confirmation/` so that the live
`confirmation/` directory can only ever contain the outputs of one execution.

## What the failed attempt did and did not do

The stage's own access audit from that process — `confirmation/PROTECTED_ACCESS_AUDIT.json`
in this directory, written by the executor before it died — records:

| field | value |
| --- | --- |
| `cells_accessing_protected` | `{}` |
| `n_cells` | `0` |
| `protected_days_predicted_total` | `0` |
| `sealed_target_rows_parsed_by_stage_reader` | `{NORD_DK1: 0, NEM_SA1: 0}` |
| `fits_on_protected_days` | `0` |
| `tuning_on_protected_days` | `0` |
| `scalar_refits_on_protected_days` | `0` |
| `checkpoint_changes_on_protected_days` | `0` |

So **no protected-final target value was read and no protected day was scored**.
The failure is upstream of the sealed forward: the frozen helper
`intl_shared.load_rederived_host` (dead code the four-market predecessor never
reached) built an adapter and called `load_state_dict` on `model.model`, which is
`None` until `fit()` runs. The sealed span was never touched.

This matters for the one-shot contract: the sealed run was authorised only for a
single execution per P0 certificate, and that execution consumed **zero**
protected information. The re-run is therefore a restoration of the intended
execution, not a result-driven retry — there was no result to look at.

## What changed before the re-run

Three stage-local wiring defects were fixed, none of them in frozen code:

1. `implementation/confirmation.py` — `_protected_host_forward` replaces the
   unusable frozen `load_rederived_host`/`predict_role` pair: it reconstructs the
   Host from the P0 re-derivation blob and refuses to return a protected-span
   forward unless the reconstruction reproduces the frozen Host's **open-role**
   prediction bit-for-bit and its `state_dict_sha` equals the frozen manifest.
2. `implementation/confirmation.py` — the protected prediction is sliced to its
   single channel (`pred_sealed[:, :, 0]`) to match the frozen metrics contract.
3. `implementation/confirmation.py` — `METRIC_KEYS` corrected to the keys the
   frozen `panel_metrics.cell_metrics` actually returns.

A fourth change is prophylactic: `_pin_threads()` re-pins and asserts
`torch.set_num_threads(1)` immediately before every Host build. `src/backbones/backbones.py`
calls `torch.set_num_threads(4)` at import time, so loading a market family
silently undoes the process-level pin; four-thread float32 accumulation is not
bit-exact against the frozen single-thread manifests.

The frozen layer, the frozen Host manifests and the frozen day lists were not
modified. `verification/verify_nn.py` re-checks all of that by content.
