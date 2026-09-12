# New window handoff — signed-mass method

Written: 2026-09-12T10:44:26Z

## What is now frozen

- The method is one switch vector: `FROZEN_SMALLEST`, {'shape_semantic_context': False, 'use_tcn': False, 'untied_amplitude_heads': False, 'rare_mass_sampling': False}.
- Retained components: none.
- Deleted components: rare_mass_sampling, shape_semantic_context, untied_amplitude_heads, use_tcn (deleted on pre-registered evidence; they are not candidates for rescue).
- Terminal development verdict: `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`.
- Seeds `7/17/37`, aggregated by cell median.  The recipe is identical in all 20 cells; only legal feature width differs between markets.

## Component decisions

- `shape_semantic_context`: **DELETED** — primary mechanism does not improve in a majority of cells; median relevant tail/upper/lower metric does not improve; median Overall-MAE is worse; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact.
- `use_tcn`: **DELETED** — primary mechanism does not improve in a majority of cells; median relevant tail/upper/lower metric does not improve; median Overall-MAE is worse; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact.
- `untied_amplitude_heads`: **DELETED** — median relevant tail/upper/lower metric does not improve; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact.
- `rare_mass_sampling`: **DELETED** — primary mechanism does not improve in a majority of cells; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact.

## Hard constraints this window did not lift

- `PROTECTED_FINAL` was **not** opened.  Read count is 0 and the authorization is still not granted.
- No Host and no baseline comparator was retrained or re-run.  The Host predictions are the frozen artifacts; the strict offline comparison joins the frozen table.
- Every statistic that could touch the evaluation partition — the OOF scalar `alpha`, the robust amplitude scale `s_A`, and the per-cell high-mass threshold — was fitted on `POST_TRAIN` only.

## What a next window may and may not do

- **May**: read this package, re-derive every number from the raw artifacts under `02_crossfit/raw` and `04_frozen_method/raw`, and run the independent verifier in `06_audits/`.
- **May not**: treat this development result as a paper claim.  The gates below are development gates.
- **May not**: re-run comparators to improve the comparison, or re-adjudicate a deleted component on a different metric.

## Gate state at handoff

- `gate_1_host_nonworse_19_of_20`: FAIL
- `gate_2_median_overall_gain_vs_host_at_least_3pct`: PASS
- `gate_3_strict_win_14_of_20`: PASS
- `gate_4_median_tail_gain_vs_host_at_least_5pct`: PASS
- `gate_5_max_normal_harm_at_most_1pct`: FAIL
- `gate_6_one_recipe`: PASS

All gates pass: **no**.  Any claim built on this window must be no stronger than this line.

