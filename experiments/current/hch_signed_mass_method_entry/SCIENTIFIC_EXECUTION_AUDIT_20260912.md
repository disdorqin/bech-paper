# Signed-Mass Scientific Execution Audit — 2026-09-12

Status: `READY_AFTER_MANDATORY_PRE_RUN_PATCH`

This audit asks whether the landed China-5 harness is scientifically faithful to the current signed-mass design and ready to open the first registered `DEV_EVAL` experiment.

## What passes

1. **20/20 frozen Host substrate**: all 5 markets × 4 Hosts resolve to frozen artifacts with no pending/blocked Host input.
2. **Forecast-origin legality**: contract-driven legal feature tensors, no province-specific scientific branch, no target-day actual input.
3. **Chronology**: expanding OOF is inside `POST_TRAIN`; fold statistics fit on fold prefixes; alpha is OOF-only; final model re-fits on all `POST_TRAIN` using a fold-derived epoch rule.
4. **Protected/final sealing**: the adapter does not materialise `PROTECTED_FINAL`; readiness provides identity/refusal/absence witnesses with read count zero.
5. **Architecture controls**: KNN off; no learned Gate/Bridge/attention/Transformer/MoE/router; five configs differ only by the four preregistered switches.
6. **Host/baseline immutability**: harness has no Host/baseline training path and points to frozen artifacts.
7. **Baseline substrate**: the domestic breadth stage is frozen at 140/140 coordinates, with 90 numeric and 50 disclosed blockers. Strict offline numerical evidence exists for Host/MDR/delta on 20/20 cells and PIR on 10/20; COSA stays online supplementary; UEC/OMPB remain blocked.

## Scientific gaps found before the first DEV run

### 1. Branch mechanism metrics were being reconstructed from net correction

Current `metrics.compute_metrics` decomposes `method_pred - host_pred` to obtain predicted Shape/Amplitude. This is not equivalent to the model's direct branch outputs because `A+ S+` and `A- S-` may overlap and cancel at a horizon.

Required correction:
- persist direct `S_hat+`, `S_hat-`, `A_hat+`, `A_hat-`;
- mechanism metrics use those direct outputs;
- final accuracy metrics continue to use the calibrated repaired prediction.

### 2. Amplitude output scale was not wired

The method design specifies `A_hat = s_A softplus(.)`, but the harness currently builds the model with core default `amplitude_scale=1.0`.

Required correction, frozen before DEV:

` s_A = median({A_d^+, A_d^- : A_d^± > 0, d in training prefix}) `.

One shared scale is used by both sign heads. Each OOF fold fits it on its own inner-training prefix; the final model fits it on all `POST_TRAIN`.

### 3. A4 high-mass mechanism metric was not implemented

Freeze `q90(A+)` and `q90(A-)` from `POST_TRAIN` per market×Host and evaluate direct mass L1 on corresponding DEV high-mass subsets.

### 4. Raw branch evidence was missing

The scientific evidence package must retain direct branch outputs and raw repaired predictions so the component decisions and final metrics are independently recomputable.

### 5. D0 and final scientific aggregation were not yet implemented

The harness can fit/evaluate but does not yet produce the registered D0 geometry package, component-decision table, frozen smallest-method confirmation, baseline join or independent final verifier.

## Protocol amendment made before outcome access

The original experiment-entry protocol proposed component screening on PatchTST/TimeMixer only. The user now explicitly requires all four Hosts, and all 20 Host cells are frozen. Before any signed-mass DEV result existed, the protocol was amended so the four component decisions use the complete 20-cell panel.

Initial grid:

`20 cells × 5 configs × 3 seeds = 300 cell/config/seed fits` plus chronological OOF fits.

This is a pre-outcome scope amendment, not result-conditioned expansion.

## Verdict

The harness does **not** need a redesign or a new core. It is structurally sound and may enter scientific execution after the mandatory, bounded pre-run patch above passes tests and the independent pre-execution verifier again reports 20/20 ready with zero protected/final reads.

Controlling execution prompt:

`experiments/current/hch_signed_mass_method_entry/AI_SCIENTIFIC_EXECUTION_PROMPT.md`

The user has explicitly authorized the development run after that patch. No further confirmation is required.

`PROTECTED_FINAL` remains unauthorized.
