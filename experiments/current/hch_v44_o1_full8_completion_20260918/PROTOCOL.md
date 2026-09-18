# HCH v4.4 O1 full-8 completion (TRAIN+VAL only) — bounded panel completion

STATUS:        ACTIVE
STAGE:         hch_v44_o1_full8_completion_20260918
KIND:          protocol
SUPERSEDED_BY: -

Controlling document for the O1 full-8 completion. It fixes the panel, the reuse set, the single
inherited delta, the logging schema, the selection rule, the gate arithmetic, the mechanism
summary and the terminal tokens. Everything not written here is inherited unchanged from
`experiments/current/hch_v44_objective_alignment_probe_20260917/` (O1) and, through it, from
`experiments/current/hch_v44_optimization_recovery_20260917/` (R1).

This document was written **before any new fit was executed**. The gate thresholds, the
non-degeneracy reading and the mechanism definitions below are preregistered and must not be
adjusted after seeing results.

## 1. Question (and only this question)

> With the already-frozen O1 objective schedule extended to the full 8-cell recovery panel, does
> post-warm-up MAE-only optimization have a stable cross-cell benefit?

Explicitly **not** tested: any new loss, architecture, router, feature or optimizer. No component
attribution (B/Shape/Level removal), no architecture simplification, no TEST evaluation.

## 2. Data boundary

Identical to O1. TRAIN and VAL only. `recovery_common.py` is imported read-only and supplies the
identical quarantine: `pretest_joint_arrays` drops every TEST row before any frame exists,
`pretest_role_frame` refuses `role="TEST"`, and the forbidden-path guard refuses the enumerated
TEST tables at the I/O boundary. No TEST target, TEST prediction or TEST metric may be read, and
TEST may not be opened "to see whether it also works". `src/core/**`, `src/backbones/**`,
`paper/**` and every canonical state file are read-only. The O1 and recovery stage directories are
read-only too.

## 3. Reuse (no retraining of anything that already exists)

* **R1 reference** — all 24 full-loss step-based runs (`R1_STEP_BUDGET_2000`) of the closed
  recovery stage, 8 cells × seeds {7,17,37}, reused read-only and pinned by hash. No R1 run is
  retrained. Presence was verified before any fit; had any been missing the stage would have
  reported `REFERENCE_NOT_AVAILABLE` and stopped without training.
* **Existing O1 runs** — the 12 runs of the four cells `GANSU_DA/PatchTST`,
  `SHANDONG_DA/iTransformer`, `SHAANXI_DA/TimeMixer`, `NINGXIA_DA/iTransformer` are reused
  in place from `experiments/evidence/hch_v44_objective_alignment_probe_20260917/o1_runs/` and
  pinned by hash. They are **not** rerun and **not** copied.

## 4. Panel

Exactly 8 cells × seeds {7, 17, 37} = 24 runs, of which exactly **12 are new fits**. Cell order is
the registered R1 recovery panel order.

| # | cell | host | status |
|---|---|---|---|
| 1 | GANSU_DA | PatchTST | reused O1 |
| 2 | GANSU_DA | LSTM | **new fit** |
| 3 | SHANDONG_DA | PatchTST | **new fit** |
| 4 | SHANDONG_DA | iTransformer | reused O1 |
| 5 | SHAANXI_DA | TimeMixer | reused O1 |
| 6 | SHAANXI_DA | PatchTST | **new fit** |
| 7 | NINGXIA_DA | iTransformer | reused O1 |
| 8 | QINGHAI_DA | TimeMixer | **new fit** |

## 5. The single inherited delta

The O1 schedule is **frozen and unmodified**:

```text
updates 1..400    L = L_rec + (L_b + L_B + L_S)/3      (== the frozen L_full)
updates 401..2000 L = L_rec
```

`SWITCH_STEP = 400` is fixed and non-tunable. There is no 200/600/800 alternative, no lambda sweep,
no annealing, no PCGrad/GradNorm/SAM, no lr sweep, no alternative initialization and no rescue
mechanism. Architecture, `b/B/S+/S-` geometry, inputs, initialization, AdamW, `lr=1e-3`,
`batch_size=32`, `weight_decay=1e-4`, `grad_clip=1.0`, `ema_decay=0.995`, EMA-primary evaluation,
`max_steps=2000`, VAL cadence 50, `no_stop_before=800`, `patience_checks=8`, seeds, data split,
history support and the stopping rule are unchanged, with no per-cell tuning. Step 0 remains a
legal VAL candidate.

**Implementation inheritance is by import, not by copy.** The 12 new fits execute
`probe_train.train_o1` from the O1 stage's `implementation/`, imported read-only, so the new runs
are produced by the identical code bytes as the reused runs. The continuation adds only
orchestration and evidence writing. Consequently every one of the 24 runs must record the same
`probe_code_hash`; a differing hash is a code mutation and fails the gate.

## 6. Required logging and selection

Byte-identical to O1 (inherited by import): every 50 optimizer steps — optimizer step,
epoch-equivalent exposure, TRAIN reconstruction MAE on the frozen diagnostic subset, the
back-propagated optimization loss, `L_rec`/`L_b`/`L_B`/`L_S`, EMA and raw VAL MAE,
`mean|correction|/mean|residual|` for VAL and TRAIN, the four pre-clip gradient norms and the
learning rate; the step-400 switch marked by `objective_switch_here` exactly once and
`used_full_loss_at_this_step` true for every check at step ≤ 400 and false above. Selection is the
inherited running-best rule with the registered `1e-7` strict-improvement tolerance over all legal
checkpoints including step 0.

## 7. Comparison and the 8-cell gate

For each cell, the three seed-level paired gains `(MAE_R1 − MAE_O1) / MAE_R1 × 100`, then the cell
median; the panel statistic is the median of the eight cell medians. Mark

```text
AUXILIARY_PERSISTENCE_BROADLY_HARMFUL_ON_RECOVERY_PANEL
```

iff **all** of:

1. at least **6 of 8** cell medians strictly improve (lower VAL MAE than R1);
2. panel median relative gain **≥ 0.5 %**;
3. worst cell degradation **≤ 0.5 %**, i.e. `min(cell_medians) ≥ −0.5`;
4. the improvement is not a retreat to step 0 or to a near-zero correction:
   * the median selected `mean|correction|/mean|residual|` over the **24** runs is **≥ 0.05** —
     the same non-degeneracy threshold the recovery stage's gate 3 and the O1 protocol registered;
     **and**
   * step-0 retreat count (a seed selecting step 0 while its R1 reference selected step > 0) is at
     most **4 of 24**, the proportional extension of O1's "at most 2 of 12"; **and**
   * the stricter absolute reading of the same O1 rule, at most **2 of 24**, also holds. Both
     readings are preregistered and both must hold. If they disagree the stage reports the
     disagreement and does not claim condition 4. Requiring both is strictly conservative and can
     only under-claim harm, never over-claim it;
5. no leakage, no numerical anomaly, no support mismatch, no code mutation and no protocol
   violation: `test_target_read_count == 0`, no forbidden path opened, no TEST row returned, all
   metrics finite, the objective switch exactly after step 400, the registered schedule unchanged
   and identical across all 24 runs, every run's TRAIN/VAL surface and history support identical to
   its paired R1 reference, and one single `probe_code_hash` across all 24 runs.

Otherwise mark `AUXILIARY_PERSISTENCE_NOT_BROADLY_HARMFUL_ON_RECOVERY_PANEL`. This is a
**diagnostic** verdict, not a paper-promotion gate. Whichever way it falls, no B-loss, Shape-loss or
Level-loss removal, no router deletion, no architecture simplification and no TEST evaluation is
executed, no next architecture is proposed and no next stage is started.

## 8. Mechanism summary (reported, never used for selection)

Not part of the gate. For every run: O1 and R1 selected step, O1 and R1 selected
correction/residual ratio, and the direction of change of `L_rec`/`L_b`/`L_B`/`L_S` after step 400,
measured at two horizons — `selected_step` (only where the selection is post-switch) and the run's
final check. Two cross-over patterns are flagged explicitly because they are what the stage exists
to detect:

* `aux_better_val_worse` — mean coordinate auxiliary error falls while VAL MAE rises;
* `aux_worse_val_better` — mean coordinate auxiliary error rises while VAL MAE falls.

## 9. Evidence

A brand-new root `experiments/evidence/hch_v44_o1_full8_completion_20260918/`. No historical v4.4
evidence is overwritten; the reused O1 and R1 artifacts stay where they are and are referenced by
hash. The root contains the access audit, the exact config, the per-step curves CSV, the four
required diagnostic figures as PNG and SVG, the per-seed paired comparison, the cell medians, the
mechanism summary, the gate arithmetic, the reuse provenance, the independent verification report
and `RESULTS.md`.

The independent verifier must not import the runner. It must independently confirm: the 12 new
fits; the 12 reused O1 identities; the 24 R1 reference identities; the full 8-cell gate arithmetic;
TEST reads = 0; `src/core` unchanged; and that the objective switch occurs exactly after step 400.
It additionally rebuilds the eligible TRAIN/VAL row sets of the four new cells from the raw cache.

## 10. Terminal tokens

```text
HCH_V44_O1_FULL8_COMPLETION_COMPLETE_FOR_ADJUDICATION
HCH_V44_O1_FULL8_COMPLETION_BLOCKED_<REASON>
```

## 11. Return format

At most **90 lines**, reporting only: (1) the terminal token; (2) new-fit completion status; (3)
TEST read count; (4) the 8-cell O1-vs-R1 cell-median table; (5) the panel median; (6) the five gate
conditions; (7) the main curve/mechanism observations; (8) verifier status and the evidence root.
**At most one Markdown table.** No process narrative, no next architecture, no automatic next
experiment.
