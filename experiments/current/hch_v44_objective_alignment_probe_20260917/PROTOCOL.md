# HCH v4.4 objective-alignment diagnostic probe (TRAIN+VAL only)

STATUS:        ACTIVE
STAGE:         hch_v44_objective_alignment_probe_20260917
KIND:          protocol
SUPERSEDED_BY: -

Controlling document for the O1 objective-alignment probe. It fixes the panel, the single
authorized delta, the logging schema, the selection rule, the gate arithmetic and the terminal
tokens. Everything not written here is inherited unchanged from the closed recovery stage
`experiments/current/hch_v44_optimization_recovery_20260917/`.

## 1. Question (and only this question)

The frozen v4.4 loss is `L_full = L_rec + (L_b + L_B + L_S)/3`. A 400-step TRAIN probe showed
TRAIN MAE 74.7107 -> 69.3408 with mean|correction| growing ~0.007 -> 18.73 while `L_rec`, `L_B`
and `L_S` all fell. The probe answers **one** question:

> Are the coordinate auxiliary losses needed only to bootstrap optimization out of the
> near-zero correction initialization, and once the model is in a non-degenerate correction
> regime, does continuing to optimize those auxiliaries harm final MAE?

Explicitly **not** tested: any new architecture, any new feature, router deletion, a new
optimizer, or a loss-weight sweep.

## 2. Data boundary

TRAIN and VAL only. `recovery_common.py` is imported read-only and supplies the identical
quarantine: `pretest_joint_arrays` drops every TEST row before any frame exists,
`pretest_role_frame` refuses `role="TEST"`, and the forbidden-path guard refuses the enumerated
TEST tables at the I/O boundary. No TEST target, TEST prediction or TEST metric may be read, and
TEST may not be opened "to see whether it also works". `src/core/**`, `src/backbones/**`,
`paper/**` and every canonical state file are read-only. The closed recovery stage's directory is
read-only too (its freeze records carry a directory-level `code_hash`).

## 3. Reference (no retraining)

The reference is the **already-executed** full-loss step-based R1 run of the same cell and seed,
reused from `experiments/evidence/hch_v44_optimization_recovery_20260917/r1_runs/`. Reference
availability was verified before any fit: all four required cells
(`GANSU_DA/PatchTST`, `SHANDONG_DA/iTransformer`, `SHAANXI_DA/TimeMixer`,
`NINGXIA_DA/iTransformer`) have seeds 7/17/37 present, 12/12. If they had not, the probe would
have reported `REFERENCE_NOT_AVAILABLE` and stopped without training.

## 4. Panel

Exactly 4 cells x seeds {7, 17, 37} = **12 fits**.

| cell | host |
|---|---|
| GANSU_DA | PatchTST |
| SHANDONG_DA | iTransformer |
| SHAANXI_DA | TimeMixer |
| NINGXIA_DA | iTransformer |

## 5. The single authorized delta

Everything is identical to R1 — architecture, exact `b/B/S+/S-` geometry, Host/history/calendar
evidence, initialization, AdamW, `lr=1e-3`, `batch_size=32`, `weight_decay=1e-4`, `grad_clip=1.0`,
`ema_decay=0.995`, EMA-primary evaluation, `max_steps=2000`, VAL every 50 optimizer steps,
`no_stop_before=800`, `patience_checks=8`, seeds, data split and history support, and no per-cell
tuning. Exactly one thing changes:

```text
updates 1..400    L = L_rec + (L_b + L_B + L_S)/3      (== the frozen L_full)
updates 401..2000 L = L_rec
```

`SWITCH_STEP = 400` is fixed and non-tunable. There is no 200/600/800 alternative, no lambda
sweep, no annealing, no PCGrad/GradNorm/SAM and no rescue optimizer. Step 0 remains a legal VAL
candidate. The registered stopping rule is inherited unchanged, so every run observes at least
step 800 and the whole post-switch window is always measured.

## 6. Required logging (every 50 optimizer steps)

Optimizer step; epoch-equivalent exposure; TRAIN reconstruction MAE on the fixed frozen
diagnostic subset (the same `np.linspace`, cap-256 subset R1 registers); the total optimization
loss actually back-propagated at that step; `L_rec`; `L_b`; `L_B`; `L_S`; EMA VAL MAE; raw-weight
VAL MAE; `mean|correction|/mean|residual|` for VAL and TRAIN; Level-head, B-head, Shape-head and
shared-encoder pre-clip gradient norms; learning rate. The step-400 objective switch is marked
explicitly: `objective_switch_here` is true exactly once, on the check at step 400, and
`used_full_loss_at_this_step` is true for every check at step <= 400 and false for every check at
step > 400.

Loss components are the frozen `structured_loss` terms evaluated on the registered diagnostic
subset with EMA weights, and are asserted against R1's own diagnostic fields on every check:
`L_rec * s_r == diag_reconstruction_mae`, `L_b * s_b == level_mae`, `L_B * s_B == mass_mae`, and
`L_total == L_rec + (L_b + L_B + L_S)/3`. A mismatch aborts the run.

Required curves: (1) TRAIN reconstruction MAE vs step, (2) EMA and raw VAL MAE vs step, (3)
`L_rec`/`L_b`/`L_B`/`L_S` vs step, (4) correction/residual ratio vs step. Each marks step 400.

## 7. Selection

Per seed, the lowest EMA VAL MAE over **all** legal checkpoints including step 0, using the
inherited running-best rule with the registered `1e-7` strict-improvement tolerance. Stopping,
warm-up and every parameter stay exactly as registered; nothing is modified because of this probe.

## 8. Comparison and gate arithmetic

Pair O1 with the same cell/seed R1 reference, take the seed-level paired delta, then the cell
median:

```text
relative_val_gain_O1_vs_R1(seed) = (MAE_R1 - MAE_O1) / MAE_R1 * 100
cell_median(cell)                = median over seeds 7/17/37
panel_median                     = median of the four cell medians
```

Also reported per run: selected step, selected correction/residual ratio, raw-minus-EMA VAL gap,
final TRAIN reconstruction MAE, and the position of the VAL optimum relative to step 400.

This is a **diagnostic** verdict, not a paper-promotion gate. Mark

```text
AUXILIARY_PERSISTENCE_LIKELY_HARMFUL
```

iff **all** of:

1. at least 3 of the 4 cell medians are better (lower VAL MAE) than R1;
2. panel median relative improvement >= 0.5 %;
3. worst cell degradation <= 0.5 % (i.e. `min(cell_medians) >= -0.5`);
4. the improvement is not a retreat to step 0 or a near-zero correction. Operationalized without
   inventing a new threshold: at most 2 of the 12 seeds select step 0 while their R1 reference
   selected a step > 0, **and** the median selected `mean|correction|/mean|residual|` over the 12
   runs is >= 0.05 — the same non-degeneracy threshold the recovery stage's gate 3 already
   registered;
5. no leakage, no numerical anomaly and no protocol violation: `test_target_read_count == 0`,
   no forbidden path opened, no TEST row returned, all metrics finite, the objective switch
   exactly after step 400 and the registered schedule unchanged.

Otherwise mark `AUXILIARY_PERSISTENCE_NOT_SHOWN_HARMFUL`. Whichever way it falls, architecture,
router, initialization and loss must not be modified further, no next architecture is proposed,
and no next stage is started.

## 9. Evidence

A brand-new root `experiments/evidence/hch_v44_objective_alignment_probe_20260917/`; no historical
v4.4 evidence is overwritten. It contains the access audit, the exact config, the per-step curves
CSV, four diagnostic figures as PNG and SVG, the per-seed paired comparison, the cell medians, the
gate arithmetic, the independent verification report and `RESULTS.md`.

The independent verifier must confirm: TEST reads = 0; exactly 4 cells x 3 seeds; only the loss
schedule changed; the objective switch exactly after step 400; no `src/core` scientific mutation;
no hyperparameter sweep; correct paired R1 identities; and correct gate arithmetic.

## 10. Terminal tokens

```text
HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_COMPLETE_FOR_ADJUDICATION
HCH_V44_OBJECTIVE_ALIGNMENT_PROBE_BLOCKED_<REASON>
```

## 11. Return format

Exactly eight items, at most 80 lines: (1) terminal token, (2) 12/12 fit completion status, (3)
TEST read count, (4) the four-cell R1-vs-O1 median VAL comparison table, (5) panel median relative
gain, (6) the core training-curve observations, especially around step 400, (7) the gate verdict,
(8) evidence root and independent-verifier status. At most one markdown table. No process
narrative, no next architecture, no automatic next stage.
