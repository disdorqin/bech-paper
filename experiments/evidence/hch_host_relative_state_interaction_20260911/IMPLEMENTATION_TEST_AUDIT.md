# Implementation and test audit

- Canonical root: `D:\作业\science\solar_leak_price_model`
- Controlling protocol: `docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md`
- Frozen S1 metric replay maximum absolute difference: `1.421e-14` (tolerance `1e-10`)
- beta=0 exact S1 direction replay maximum absolute difference: `1.192e-07` (tolerance `1e-06`)
- Frozen artifact immutability: `True`
- S1 code tree immutability: `True`
- Amplitude code tree immutability: `True`
- HSA code tree immutability: `True`
- Host cache immutability: `True`
- Stacked chronological OOF self-exclusion: `True`
- One global beta per fold/seed applied to all six cells: `True`
- Trainable parameters for the whole six-cell panel: `5`
- Per-market/per-Host beta vectors: `0`
- Forbidden/protected reads: `0`
- Targeted unit suite exit code: `0`
- Targeted unit suite result: `128 checks passed`
- Targeted unit suite output: `tests.txt`
- Row-set-alignment diagnostic worst cell: `-0.7848 pp` (`alpha_rowset_sensitivity.csv`, not a gate input)
- Independently re-verified by `verify_gates.py` (does not import this runner): recomputed R0-R4 and the token from the written evidence alone

## Standardization convention

`beta` is fitted through the frozen `norm_state(train, test)` pattern: for each fit the
statistics come from that fit's own training rows and are applied unchanged to the rows
it predicts. This is the convention the frozen S1 `fit_predict` uses for both its OOF
folds and its final DIAG_EVAL fit. It is stated here because the rejected HSA runner
standardized its final fit on its own OOF rows but then applied `beta` to state
standardized on the full DIAG_FIT block; HRSI does not reproduce that mismatch.
