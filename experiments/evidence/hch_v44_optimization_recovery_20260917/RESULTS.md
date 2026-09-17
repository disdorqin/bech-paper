# HCH v4.4 optimization recovery — TRAIN+VAL only, TEST quarantined

Terminal token: **HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED**

## What changed

Exactly one thing: the training budget is registered in **optimizer steps** with a burn-in, instead of in epochs. Architecture, geometry, features, loss, initialization, seeds, batch size, weight decay, dropout, gradient clip and EMA are unchanged, and `src/core/**` was not edited.

## R0 — the budget asymmetry that motivated the stage

See `R0_SUMMARY.md` / `R0_DIAGNOSTICS.csv`.

## R1 — 8-cell panel (24 fits)

- cells improving over the frozen recipe: 5/8
- panel median improvement: 0.2913%
- worst improvement: -0.1478%

| gate | result |
|---|---|
| gate_1_at_least_6_of_8_cells_improve | FAIL |
| gate_2_panel_median_improvement_ge_0.5pct | FAIL |
| gate_3_at_least_5_of_8_step_gt0_and_ratio_ge_0.05 | PASS |
| gate_4_worst_degradation_le_1.0pct | PASS |
| gate_5_no_leakage_or_numerical_failure | PASS |

Panel token: `HCH_V44_RECOVERY_R1_PANEL_NOT_PROMOTED`

## Full 20-cell panel

Not run: the 8-cell promotion gate did not pass, so the protocol forbids expanding the recipe.

## What this is not

This is **not** a SOTA or generalization claim. V2 TEST was observed before this stage and directly motivated it, so it is quarantined from every decision here. The recovery process read no TEST target, no TEST prediction and no TEST metric. Any SOTA claim requires a separately authorized untouched confirmation surface.

## Provenance

- `ACCESS_AUDIT.json` — guard state and TEST-read counters;
- `REUSED_SUPPORT_HASHES.json` — every prior-stage artifact reused, with hashes;
- `RECIPE_FREEZE.json` — the frozen recipe and the terminal token;
- `INDEPENDENT_VERIFICATION_REPORT.json` — the independent verifier's verdict.
