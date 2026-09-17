# HANDOFF.md

## Current handoff — v4.4 optimization recovery is CLOSED at `NOT_SUPPORTED`; the step-budget lever is measured and exhausted (2026-09-17)

Stage `experiments/current/hch_v44_optimization_recovery_20260917/` returned
**`HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED`**. Do not reopen it and do not continue tuning v4.4's
optimization path: the preregistered route terminated lawfully.

Route: the R1 8-cell × 3-seed panel gate came in at **3/5** — 5/8 cells improved (gate 1 needs 6) and
the panel median was **+0.2913 %** (gate 2 needs 0.5 %); gates 3, 4 and 5 passed. Because the panel did
not promote, the 20-cell expansion was correctly not run; and because the R2 optimization-limited
trigger was **not** satisfied (no condition held in ≥50 % of the 24 runs; strongest was EMA-lags-raw at
41.7 %), no single-factor arm was authorized. 24/24 fits, independent verification 12/12,
`test_target_read_count = 0`.

**What was learned, in one paragraph.** Fixing the epoch→step budget asymmetry is a **real but partial
and non-uniform** repair. It is large exactly where the frozen recipe was starved — QINGHAI_DA/TimeMixer
+4.381 %, SHAANXI_DA/PatchTST +3.064 %, NINGXIA_DA/iTransformer +2.416 %, all cells whose frozen gain
was ~0.000 % — but it does **not** generalise: on SHANDONG_DA/PatchTST, the frozen recipe's own best
cell at +1.975 %, the longer budget is 0.148 pp *worse*, so epoch-0 selection was not universally
under-trained. And the cells that stay flat fail for a *different* reason than the one that motivated
the stage: GANSU_DA/PatchTST selects step 0 in 3/3 seeds with a monotone divergent curve (TRAIN MAE
74.711 → 72.635 while VAL MAE 77.104 → 77.486 at every check), i.e. the optimizer works and VAL simply
moves the wrong way. **Epoch-budget starvation is therefore necessary but not sufficient**, and "train
longer" is not a general repair for v4.4 G2.

**The next decision belongs to the main window**, and it is an adjudication rather than a run: v4.4 G2
now has a frozen `+0.00 %` TEST result, a measured optimization recovery that failed its own
preregistered bar, and no remaining authorized tuning lever. V2 TEST stays quarantined and the frozen
TEST result stays immutable. G0/G1/G3, foreign/final, optional target-day features, Host/baseline
reruns, broad rescues and `paper/**` remain closed.

## Previous handoff — v4.4 frozen TEST recipe is negative/neutral; tonight's only legal continuation is TRAIN+VAL optimization recovery, then untouched confirmation if it passes (2026-09-17)

For a new conversation window, start from `experiments/current/hch_v44_optimization_recovery_20260917/NEW_WINDOW_HANDOFF_PROMPT.md`. Then read `docs/current/HCH_V44_POSTTEST_REGRESSION_ADJUDICATION_AND_RECOVERY_PLAN_20260917.md` and execute `AI_EXECUTION_PROMPT.md` under the same stage's `PROTOCOL.md`. The first v4.4 G2 TEST result is valid negative evidence: panel median gain vs Host `+0.00%`, only 1/20 wins vs the best same-setting offline selected baseline, and M0/M1/M2 remain clearly stronger under the same V2 TEST support. Do not erase this result.

The reason to continue v4.4 is narrow and evidence-based: the graph learns when given hundreds of optimizer updates, but the original epoch-based recipe selects epoch 0 in 34/60 runs and leaves correction near zero. Therefore the next stage changes training-budget semantics only. R1 keeps the frozen core/geometry/features/loss/initialization/seeds/EMA and uses `max_steps=2000`, VAL every 50 steps, no early stopping before step 800, then patience 8 checks. Step 0 remains a legal candidate. Run the fixed 8-cell × 3-seed TRAIN+VAL recovery panel first; expand to the remaining 12 cells only if the preregistered VAL gate passes. R2 is conditional and may change exactly one factor only when R1 is demonstrably optimization-limited: first `lr=3e-3`, or tiny Level-readout activation if readout activation is directly diagnosed as the bottleneck.

**Critical contamination boundary:** V2 TEST is now exposed development evidence and is quarantined. The recovery executor must not read TEST targets/predictions/metrics, `CELL_MEDIAN_RESULTS.csv`, `PER_SEED_RESULTS.csv`, `HCH_V44_MAIN_20260917.csv`, `RESULTS_LONG.csv`, `BASELINE_COMPLETION_20260917.csv`, or use any TEST number to choose settings. A recovered recipe may only be promoted to `HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION`; an untouched confirmation surface must be separately authorized before any SOTA/generalization claim. G0/G1/G3, foreign/final, optional target-day features, Host/baseline reruns, broad rescues and `paper/**` remain closed.

## Current handoff — v4.4 China-5 full-panel main eval is complete and awaiting adjudication; the frozen G2 recipe is metric-neutral (2026-09-17)

Stage `experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/` returned
`HCH_V44_CHINA5_FULLPANEL_COMPLETE_FOR_ADJUDICATION`. 80/80 shadow-OOF support refits → 20/20 legal
W=7 history + TRAIN-only scales → phase A 10/10 → 60/60 registered G2 fits → freeze → TEST opened once
→ prequential scoring → independent verification 10/10. Evidence:
`experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917/` (start at `RESULTS.md`).

**What it found:** panel median gain vs Host **+0.00 %**; HCH beats Host in 16/20 cells but only
marginally; vs the best same-setting offline baseline HCH wins **1/20** (median gap +2.493 MAE).
Cause is under-training inside the frozen budget, not a broken graph (TRAIN MAE 74.71 → 69.34 in a
400-step probe; VAL improves past epoch 0 in only 26/60 runs; epoch 0 selected in 34/60).

**The one open decision for the main window:** whether to register a larger training budget
(more epochs / higher lr / larger effective step count) for the v4.4 object. Until that is
adjudicated, **no paper claim about v4.4 superiority is authorised**, and the 96-fit G0–G3 ladder,
overseas/final sets, optional target-day features, Host/baseline reruns, rescue/tuning and `paper/**`
writes remain closed. Next-stage candidates: (a) an authorised training-budget stage, or (b) reporting
v4.4 as a negative/neutral result in the paper.

## Prior handoff — resume the blocked v4.4 China-5 stage after D1=A / D2 adjudication (closed 2026-09-17)

The first pre-execution attempt stopped correctly at `HCH_V44_CHINA5_FULLPANEL_BLOCKED_BEFORE_FIT` with 7/10 gates passed, 0/60 primary fits and 0 TEST target reads. Read `experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917/PREEXECUTION_REPORT.md`, then the binding adjudication `docs/current/HCH_V44_CHINA5_FULLPANEL_UNBLOCK_ADJUDICATION_20260917.md`.

D1 is **A**: legal TRAIN history is produced by support-only recipe `V44_SHADOW_OOF_PREFIX_90_10_V1`. For each market×Host split V2 TRAIN into B0..B4; B0 warm-up; for B1..B4 fit the same Host architecture/hyperparameters/registered Host seed on the strictly earlier prefix using `TorchHostAdapter.fit(prefix, valid=None)`, whose internal chronological 90/10 prefix split is now the registered shadow fit/selection rule. Global V2 VAL is forbidden for shadow support. Exact budget is 80 CPU shadow Host refits; record fit/selection/prediction intervals and hashes, mark predictions `oof`, and never create/replace benchmark Host rows.

D2 authorizes the missing production harness only inside `experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/implementation/**` and `/verification/**`, reusing V2 `cb_contracts/cb_cells` read-only. Do not touch `src/core/**`, `src/backbones/**`, frozen Hosts, baseline CSVs, prior evidence roots or `paper/**`. Re-run Phase A with a stage-local non-mutating 12-check source verifier, real production-path synthetic/backward and TRAIN/VAL no-TEST smoke, 20/20 legal W=7 support, TRAIN-only scales, TEST-read=0 and frozen hashes unchanged. Only token `HCH_V44_CHINA5_FULLPANEL_PREEXECUTION_VERIFIED` opens the 60 primary G2 fits. Launcher: `experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/RESUME_AFTER_ADJUDICATION_PROMPT.md`. Final token remains `HCH_V44_CHINA5_FULLPANEL_COMPLETE_FOR_ADJUDICATION`.

## Current handoff — run the authorized v4.4 China-5 primary full-panel experiment next (2026-09-17)

The prerequisites are now closed and accepted. Main-window rerun of the bounded v4.4 source patch gives **125/125 maintained tests PASS** and the independent verifier gives **12/12 PASS**; the selected domestic baseline set has also been independently reviewed and is complete at five methods × 20 cells. The active execution authority is `experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/PROTOCOL.md` with launcher `AI_EXECUTION_PROMPT.md`.

Run only `G2_GEOM_COORD_CONTEXT` on five markets × four frozen Hosts × seeds 7/17/37 = **60 fits** under `COMMON_BENCHMARK_701020_FULL_V2`. Before fitting, Phase A must reproduce the 125 tests, 12-check verifier, Host/split identities, baseline overlay hashes, TRAIN-only scales/caches and legal W=7 history; only token `HCH_V44_CHINA5_FULLPANEL_PREEXECUTION_VERIFIED` opens Phase B. Use the frozen effectiveness profile and EMA primary weights; no per-cell tuning. TEST opens chronologically only after the selected EMA checkpoint/config/scales for that cell/seed are frozen. G0/G1/G3, ablations, foreign/final, optional target-day China features and all rescues remain closed. Final token is `HCH_V44_CHINA5_FULLPANEL_COMPLETE_FOR_ADJUDICATION`, then stop for main-window result review.

Paper motivation work is already landed: `paper/current/FIG1_DOMESTIC_HOST_STRUCTURAL_DIAGNOSTICS.svg`, `FIG2_SHAANXI_REAL_ERROR_ARCHETYPES.svg` and the corresponding Figure-anchored Introduction are current. These VAL-only diagnostics are problem motivation, not v4.4 performance evidence. Do not insert forthcoming G2 results into `paper/current/**` automatically; paper result insertion still requires a fresh explicit user authorization.

## Current handoff — the domestic selected-baseline set is complete (90/90 numeric); the next step is main-window review of the completion evidence (2026-09-17)

The stage under `experiments/current/hch_china5_selected_baseline_completion_20260917/` is **CLOSED** and returned `CHINA5_SELECTED_BASELINE_COMPLETION_VERIFIED`. Read `experiments/evidence/hch_china5_selected_baseline_completion_20260917/FINAL_REPORT.md` first, then `VERIFICATION_REPORT.json`; per-row numbers are in `05_COMPLETION_LONG.csv` and the new overlay is `experiments/lab/common_benchmark_results_v2/BASELINE_COMPLETION_20260917.csv`. All five selected baselines are now 20/20 numeric.

What this authorises: **main-window review of the completion evidence**. Nothing else. The v4.4 96-fit G0/G1/G2/G3 ladder, any TRAIN/VAL scientific fit of the v4.4 method, TEST/QINGHAI access for that method, `foreign`/`final` validation, Host/baseline reruns and all paper edits remain closed until that review passes.

What was done: exactly 50 new numeric cells (PIR five markets × {iTransformer, LSTM}; UEC-STD 20; OMPB 20) under the frozen V2 splits/Hosts, with δ-Adapter/COSA and the existing 10 PIR rows reused by hash identity. Phase A passed 7/7 (interpreter+CUDA, inventory, Host/split identity, scorable-day adjudication, reuse-metric recompute, implementation freeze, TRAIN/VAL-only backend smoke) before the pre-execution token; Phase B froze the backend per method from that smoke (PIR → CUDA, UEC-STD/OMPB → CPU) and ran the GPU lane concurrently with a 6-worker CPU pool (max per-cell 7.1 s, peak RSS 0.98 GB). The independent verifier passed 8/8, including a hard NaN/Inf rejection and a behavioural chronology proof.

Two items must not be lost. The earlier attempt that had already emitted a success token is **rejected**: it scored the frozen-unscorable SHANDONG day `2026-07-17`, so all ten of its SHANDONG rows were `NaN`, and its verifier accepted them because `abs(nan-nan) > 1e-5` is `False`; it also ran on the system interpreter with CPU-only torch. It is archived under `prior_attempt_20260917_nan_shandong/` with a `REJECTION_NOTE.md` and its numbers must never be quoted. And PIR's SHAANXI collapse (−94 % iTransformer, −152 % LSTM) is a genuine result, not an artifact — the CPU re-run reproduces it to 1e-6.

## Current handoff — paper motivation evidence and v4.4 manuscript scaffolding updated while baseline completion continues (2026-09-17)

A VAL-only descriptive audit on existing `COMMON_BENCHMARK_701020` frozen Host artifacts supports the two Introduction phenomena without opening TEST or running any model: across China-5 × four Hosts, the median per-cell fraction of days with ≥2h circular peak-or-valley displacement is **0.900** (range **0.545–1.000**); the exact daily Level-mass ratio `eta=H|b|/||r||1=|b|/MAE_day` has an across-cell median of cell medians **0.492**, with 19/20 cells ≥0.3 and 9/20 ≥0.5. Objectively selected Shaanxi VAL showcases are frozen for the paper: PatchTST 2025-08-24 as timing/placement mismatch (eta≈0.257, peak shift 3h, valley 2h, centered shape corr 0.689→0.836 under diagnostic +2h circular shift) and TimeMixer 2025-08-19 as Level bias (eta≈0.949, peak/valley 1h, centered shape corr≈0.867). Analysis: `experiments/current/hch_paper_figure_diagnostics_20260917/analyze_host_error_archetypes.py`; figure blueprint: `paper/06_figures_tables/MOTIVATION_FIGURES_HOST_ERROR_ARCHETYPES_20260917.md`. These are motivation diagnostics, not final v4.4 performance results.

`paper/current/00_METHOD_OVERVIEW.md` is now the compact Chinese method authority for paper drafting: exact `b/B/S+/S-` geometry, Level MAE lower bound, Host/history/calendar evidence, BiGRU32 + history GRU32 + Shape-history MLP, hour-aligned Shape allocation, exact decoder, fixed effectiveness pack, future manifest-gated market features, motivation figures and ablation plan. `paper/current/02_RELATED_WORK.md` has been rewritten in a PIR-like field-overview style rather than a collision-audit style. `paper/current/04_EXPERIMENTS.md` now records the current selected-baseline completion campaign (δ-Adapter/COSA reuse, 10 PIR + 20 UEC-STD + 20 OMPB missing cells underway) and the component-level ablations: G2-vs-G3 exact geometry, G2-vs-G1 coordinate-specific context, G1-vs-G0 revealed history, plus support ablations for Host mechanism features and coordinate auxiliaries. `paper/current/01_INTRODUCTION.md` contains only a restrained paragraph pointing to the VAL motivation evidence; no v4.4 result is claimed.

PIR is used as a writing/experimental-design reference rather than a method template: its paper first establishes the empirical problem with a per-instance error figure, then introduces the revision framework; its ablations remove Local/Global components and add a capacity-oriented deepened-backbone control. We mirror that logic with population-level problem evidence + real archetypes, then matched component controls. Baseline completion is still running independently; do not insert its numeric outcomes until returned and verified.

## Current handoff — the v4.4 bounded source patch is done and independently verified at zero fit; main-window source review is the only next step (2026-09-17)

The bounded patch under `experiments/current/hch_v4_4_cleanroom_bounded_patch_20260917/PROTOCOL.md` is **CLOSED** and returned `HCH_V4_4_CLEANROOM_BOUNDED_PATCH_READY_FOR_SCIENTIFIC_REVIEW`. Read `experiments/evidence/hch_v4_4_cleanroom_bounded_patch_20260917/RESULTS.md` first, then `src/core/DESIGN_CONTRACT.md`; machine detail is in `PATCH_DIFF_SCOPE.json`, `CALENDAR_CONTRACT_REPORT.json`, `HOUR_ROUTING_REPORT.json`, `SCALE_UNIT_REPORT.json`, `VALID_MASK_REPORT.json`, `FP32_CARVEOUT_REPORT.json`, `VERIFICATION_REPORT.json`.

What this authorises: **main-window source review of the patched core**. Nothing else. The 96-fit G0/G1/G2/G3 ladder, any TRAIN/VAL scientific fit, TEST/QINGHAI access, `foreign`/`final` validation, Host/baseline reruns and all paper edits remain closed until that review passes.

What changed — exactly seven defects, nothing more: hour calendar 6 → 7 channels with annual sine and cosine; Shape allocation moved to same-hour source routing `[B,H,S]` (weights and routed state are now hour-local); `s_r` is the robust scale of the per-day MAE `M1/H_valid`, with the history residual-magnitude channel on the same unit; per-day valid masks are stacked through the history Shape aggregation; structured and G3 reconstruction terms are valid-mask aware and `require_full_daily_support` guards 24/24 hours for the first scientific run; `fp32_region()` disables autocast for the active device type; and the unused trainable `MaskedSoftmaxAllocator.routed` projection was deleted.

Verified state: `geometry.py`, `decoder.py`, `bundles.py`, `heads.py`, `encoders.py`, `feature_engineering.py` and `rescues.py` are byte-identical to the pre-patch tree; no module was added or removed and no new dependency exists; W=7, TOKEN_DIM=32, the BiGRU/GRU/MLP topology, loss weights, EMA, optimizer, difficulty interleaving, optional China features, rescue utilities and the scientific variants are unchanged. 125 maintained tests pass and 12 independent verification checks pass. Parameter counts: G0 = G1 = G2 = 14948, G3 = 16897 (1.1304x G2). Counters: 0 scientific fits, 0 TEST reads, 0 paper mutations, 0 Host/baseline retrains; the previous stage's evidence root is unmodified.

Two items are recorded, not fixed: the Coordinate Headroom Audit's random K-fold and the unconsumed holiday/day-type history channels are future extension follow-ups, and `docs/current/HCH_V4_4_CLEANROOM_CORE_MAIN_WINDOW_AUDIT_20260917.md` is cited by four files but does not exist on disk — the seven-item defect list in `RESEARCH_STATE.md` / `HANDOFF.md` / `experiments/STAGE_INDEX.md` was used instead, and the missing document is reported as a documentation defect.

## Current handoff — patch v4.4 source before fit; run the 50 missing selected-baseline cells in parallel (2026-09-17)

For v4.4 source work, read `docs/current/HCH_V4_4_CLEANROOM_CORE_MAIN_WINDOW_AUDIT_20260917.md` and execute only `experiments/current/hch_v4_4_cleanroom_bounded_patch_20260917/AI_PATCH_PROMPT.md`. The clean-room build passed 105 maintained tests but is not scientific-ready because of bounded implementation mismatches: missing annual-cos calendar channel, cross-hour Shape routing instead of same-hour source allocation, daily-L1/per-hour-MAE scale mismatch, lost valid masks in history Shape aggregation, valid-unaware reconstruction, wrong-device fp32 carve-out helper, and one dead trainable allocator projection. Keep the exact `b/B/S±` geometry and small G0–G3 architecture; this is a fidelity patch only. Required token is `HCH_V4_4_CLEANROOM_BOUNDED_PATCH_READY_FOR_SCIENTIFIC_REVIEW`, then stop for main-window audit.

Baseline completion is independent and may run concurrently. User authority now admits exactly five comparison baselines: delta-Adapter, COSA, PIR, UEC-STD, OMPB; old qualification blockers are historical provenance only and must not gate the new stage. V2 numeric inventory is delta-Adapter 20/20, COSA 20/20, PIR 10/20, UEC-STD 0/20, OMPB 0/20. Execute exactly 50 missing numeric cells: PIR five markets × {iTransformer,LSTM}, plus UEC-STD 20 and OMPB 20. Reuse the exact `COMMON_BENCHMARK_701020_FULL_V2` splits, Hosts and existing numeric artifacts; do not retrain Hosts or rerun complete delta/COSA/PIR cells. Controlling prompt: `experiments/current/hch_china5_selected_baseline_completion_20260917/AI_EXECUTION_PROMPT.md`. It requires a resource-aware CPU process pool and simultaneous GPU lane, a pre-execution verification token, historical-artifact isolation and an independent final verifier. No new numeric baseline result has been returned yet.

## Current handoff — paper v4.4 draft is landed; baseline comparison registry can run in parallel while core receives main-window audit (2026-09-16)

The user explicitly authorized `paper/**` updates. `paper/current/01_INTRODUCTION.md`, `02_RELATED_WORK.md`, `03_METHODOLOGY.md`, `04_EXPERIMENTS.md`, `05_DISCUSSION.md`, `06_CONCLUSION.md` and `README.md` are now aligned to v4.4; supporting current files were added under `paper/01_framework` through `paper/07_reproducibility`. The manuscript no longer uses LGR as the active method. Experiments contain protocol/setup only and the conclusion remains a placeholder. Future result insertion still requires a new explicit paper authorization.

Domestic baseline uncertainty is also resolved. Under `COMMON_BENCHMARK_701020_FULL_V2`, selected runnable comparator coverage is already complete: Host 20/20, MDR 20/20, δ-Adapter 20/20, COSA 20/20 online, PIR 10/20 numeric with 10 inherited new-Host incompatibility blockers; UEC-STD has 20 fidelity blockers and OMPB 20 data/fidelity blockers. The old `90/140` appearance is therefore 90 legitimate numeric rows + 50 legitimate blockers, not 50 forgotten experiments. Do not rerun baselines. In parallel with the v4.4 core review, execute only `experiments/current/hch_v44_baseline_comparison_freeze_20260916/AI_EXECUTION_PROMPT.md` to reconstruct a paper-ready 140-coordinate registry and 20-cell comparator snapshot from frozen V2 artifacts. Expected token: `HCH_V44_BASELINE_PAPER_REGISTRY_COMPLETE`; new baseline fit count must remain 0.

The v4.4 clean-room core source token has already returned and still requires main-window source audit before any 96-fit G0–G3 ladder. Baseline registry freezing and core audit are independent and may proceed in parallel.

## Current handoff — v4.4 clean-room core is built and independently verified at zero fit; main-window source audit is the only next step (2026-09-16)

The clean-room build under `experiments/current/hch_v4_4_cleanroom_core_build_20260916/PROTOCOL.md` is **CLOSED** and returned the source token `HCH_V4_4_CLEANROOM_CORE_READY_FOR_SCIENTIFIC_REVIEW`. Read `experiments/evidence/hch_v4_4_cleanroom_core_build_20260916/RESULTS.md` first, then `src/core/README.md` and `src/core/DESIGN_CONTRACT.md`; the machine detail is in `VERIFICATION_REPORT.json`, `GEOMETRY_ADVERSARIAL_REPORT.json`, `CHRONOLOGY_LEAKAGE_REPORT.json`, `VARIANT_PARITY_REPORT.json`, `FEATURE_EXTENSION_FAILCLOSED_REPORT.json`, `BACKWARD_SMOKE.json` and `PYTEST_REPORT.md`.

What this authorises: **main-window code audit of the rebuilt core**. Nothing else. The 96-fit G0/G1/G2/G3 development ladder, any TRAIN/VAL scientific fit, TEST/QINGHAI access, `foreign`/`final` validation, Host/baseline reruns and all paper edits remain closed until that audit passes. The returned token is a source token, not a scientific one.

What was done: the non-authoritative `src/core` was archived byte-preserving at `src/archive/v44_cleanroom_superseded_20260916/legacy_core/` and replaced by a 16-module rebuild from `docs/current/HCH_V4_4_CLEANROOM_CORE_BUILD_SPEC_20260916.md`. The superseded `targets.py`/`preprocessing.py`/`shape.py`/`fusion.py` left the active path; 13 RCL/action test modules and 10 rejected-preflight v4.4 test modules moved into their own archives with `SUPERSEDED_TESTS.md` mappings; 9 new `test_v44_cleanroom_*.py` modules are the maintained active suite.

Verified state: 105 maintained v4.4 tests pass and 15 independent verification checks pass (the verifier re-derives the algebra in float64 from the frozen equations and does not import the builder). Parameter counts: G0=G1=G2=15972, G3=17987 (1.1220x G2, within the 1.25x budget). G0 is bitwise invariant to arbitrary history mutation; G1's tie survives optimizer steps while G2's coordinates separate; optional features are fail-closed; rescue utilities are disabled and unreachable. Counters: `scientific_neural_fit_count=0`, `test_target_value_read_count=0`, `paper_mutation_count=0`.

Two recorded, non-blocking observations: the frozen zero-initialized Level/balanced-mass readouts make the day-level evidence path connected-but-zero-gradient at step 0 (the history day encoder and the `L`/`B` coordinate embeddings start learning after the first optimizer step), and the scale recipe's median is explicitly the linear-interpolation median. Ten pre-existing foundation-tree failures remain unrelated to `src/core`; they are recorded in `PYTEST_REPORT.md` with an AST check showing none of them imports or exercises the core.

## Current handoff — build the final v4.4 core clean-room from contract; include fixed effectiveness tricks and manifest-gated China feature engineering, but run zero scientific fits (2026-09-16)

The source-design stage is now frozen in `docs/current/HCH_V4_4_CLEANROOM_CORE_BUILD_SPEC_20260916.md`. Do not continue historical source-patch/final-seal chains and do not infer scientific behavior from the current `src/core`; archive/hash the current tree only for provenance, then clean-room rebuild. Execute only `experiments/current/hch_v4_4_cleanroom_core_build_20260916/AI_CORE_BUILD_PROMPT.md` under its `PROTOCOL.md`.

The rebuilt primary core must preserve the final scientific object: exact MAE-aligned `b/B/S+/S-` residual geometry, joint original-residual training, two independent Shape simplexes, and one small masked-softmax evidence allocator over Host/history/calendar, with G1 tied coordinate identity versus G2 released identity. It must also include the now-frozen implementation layers: always-legal Host mechanism features (absolute/relative/ramp/rank/volatility/peak-valley statistics), W=7 calendar-similarity and exact-geometry history features, and a fixed effectiveness profile (near-Host init, exact-once difficulty-interleaved TRAIN batches, EMA=0.995, fixed AdamW recipe, fp32 geometry/loss carve-out, caching/loader acceleration, gradient-conflict logging). These tricks are implementation-only and identical across scientific variants where applicable.

Implement the domestic-literature feature-engineering extension behind `MANIFEST_GATED_ONLY`: load/renewable/wind/PV/hydro/non-market/intertie/bidding-space trajectories and derived renewable penetration, net load, ramps, non-market share, bidding-space ratio, documented net-bidding-space formulas and probabilistic spread may enter only when a frozen issue-time manifest explicitly admits their operands. Current ADMIT=0 means the first primary model still uses none of them. Add a TRAIN-only Coordinate Headroom Audit support utility against `b/B/Shape`; it never becomes a learned gate and never reads TEST. Rescue utilities (PCGrad/GradNorm, SAM/ASAM, scalar calibration, bundle dropout) may exist only disabled and must be unreachable from the primary profile without a new authorization.

This task is source/test only: 0 scientific neural fits, 0 TEST target reads, 0 paper mutations. Required return token is `HCH_V4_4_CLEANROOM_CORE_READY_FOR_SCIENTIFIC_REVIEW`; then stop for main-window code audit before any 96-fit ladder.

## Current handoff — v4.4 design/literature stage is closed; reproduce the final core cleanly before any fit (2026-09-16)

Read `docs/current/HCH_V4_4_FINAL_CONVERGENCE_SYNTHESIS_20260916.md` first. It is the compact authority for the final scientific object. The old source-patch token is historical and does not authorize training because current `src/core` is non-authoritative after later rewrite/deletion and Round-7/8 design changes. Do not continue old patch chains by inference.

Final primary method: frozen Host + absolute/relative Host views + deterministic calendar + exactly W=7 fully revealed residual-geometry history; exact MAE-aligned coordinates `b/B/S+/S-`; joint end-to-end correction training; small Host BiGRU32, causal geometry-history GRU32, deterministic per-hour Shape-history statistics, preserved Host/history/calendar evidence bundles, one tiny masked-softmax geometry-conditioned allocator, day-level Level/B queries and hour-level Shape queries, two independent Shape simplex heads, exact decoder. No target-day unproven exogenous forecast, learned Market Core, exact sparsity, Transformer, MoE, retrieval, safety gate, sequential Level or dual calibration belongs to the first candidate.

Next engineering task is a **clean-room core reproduction** from the design documents, with zero scientific fits and an independent adversarial verifier. Only after that source passes should the fixed 96-fit G0/G1/G2/G3 development ladder run on the existing 8-cell panel × seeds 7/17/37. No fifth rescue arm or post-outcome architecture change is allowed. If G2 fails Host-relative usefulness, stop for adjudication; if usefulness passes but G2-vs-G1/G3 mechanism contrasts fail, do not overclaim the geometry-allocation interpretation.

## Current handoff — Round-8 v4.4 design is converging to a very small clean-room core; continue literature, not source or fits (2026-09-16)

Do **not** continue the old source-final-seal patch chain and do not start the 96-fit G0-G3 ladder. The user reports that current core code was deleted/partially rewritten by a previous AI and explicitly accepts later clean reproduction. Treat `src/core` as non-authoritative. Read `docs/current/HCH_V4_4_PREFREEZE_CLOSURE_20260916.md`, `HCH_V4_4_CLOSEST_WORK_COLLISION_AUDIT_20260916.md`, the literature map now through §§44–59, and `HCH_V4_4_SOURCE_IMPLEMENTATION_PLAN_20260916.md`. No core rebuild or scientific fit is authorized yet.

Round-8 removes rather than adds machinery. The primary legal evidence bank has only Host, W=7 revealed residual-geometry history and calendar. A learned PMA/Perceiver/attention Market Core would duplicate the later geometry-conditioned allocation, so keep feature/bundle-local tokens intact and use deterministic masked-mean+normalization only as query-conditioning context. Exact sparsity is likewise unnecessary for three sources: keep one tiny shared masked-softmax allocator; entmax/Top-K/Hard-Concrete belong only to a future large legal exogenous pool. This preserves the clean G1/G2 contrast: identical information/encoders/parameter count, coordinate identity tied in G1 and released in G2.

The scientific core remains `exact Host residual -> b/B/S+/S- geometry -> geometry-conditioned evidence allocation -> exact reconstruction`. Joint end-to-end training remains preferred because the exact coordinates and reconstruction share the same optimum. Generic multi-task literature is an optimizer warning only: future clean-room preflight should log gradient norms/cosines for reconstruction, b, B and Shape losses on shared parameters. Only if persistent severe negative interference is observed may a preregistered code-only PCGrad/GradNorm rescue be considered; do not reopen sequential/frozen Level by default and do not claim gradient surgery as novelty.

Shape remains one shared trunk plus **two independent 24h simplex heads**. A newly audited uncentered one-field signed parameterization is algebraically expressive but has dead-support ReLU/sign-optimization problems; centered one-field remains mathematically invalid for some one-signed targets. Predicted positive/negative overlap is allowed and logged, not penalized in the first run. Target-mass-weighted W1 remains only a timing/placement auxiliary; reconstruction MAE is primary.

CRAFTER-style future feature engineering remains implementation-only: legal covariates should be screened for incremental Host-failure headroom on exact `b/B/Shape`, not raw-price correlation. Recent Chinese EPF work on net spot bidding capacity and probabilistic load/renewable forecasts strengthens market-semantic derived features, but current target-day `预测值` columns remain HOLD and cannot enter the primary v4.4 candidate without provenance. Continue literature/design; do not edit `paper/**`, open TEST/QINGHAI/foreign/final, or update `EXPERIMENT_LEDGER.md` for literature-only work.

## Current handoff — returned v4.4 source-ready token is not accepted; run the bounded source fidelity patch next (2026-09-16)

Read `docs/current/HCH_V4_4_SOURCE_READINESS_REAUDIT_20260916.md` first. Main-window code review found that the previous source-preflight token overstates readiness. Zero-fit/TEST=0 boundaries are preserved, but the active core is not faithful enough for scientific execution: degenerate signed-Shape scores violate the predicted mass identities; one-signed inactive Shape is still penalized; TRAIN-frozen coordinate scales/valid-aware loss are missing; `host_views()` is unused; Host BiGRU is 64-wide instead of total width 32; exact W=7 day-geometry + hour-Shape history is not built; G2 lacks genuine coordinate-specific allocation; G3 lacks matched hour information; canonical active-core tests are stale/red; verifier independence is insufficient and no real backward smoke was performed.

The only legal next task is source-only patching via `experiments/current/hch_v4_4_source_reaudit_patch_20260916/AI_SOURCE_PATCH_PROMPT.md`. Preserve/archive the returned-but-rejected core/evidence, implement the compact registered G0–G3 substrate correctly, add real v4.4 maintained tests and an independent verifier, and keep all scientific fit/TEST/paper counters at zero. The source patch also freezes two mathematical simplifications: `B=(||r||1-H|b|)/2` as the headline balanced-mass coordinate (with `min(P,N)` cross-check), and target-mass-weighted Shape W1 so inactive signs mask automatically. Do not run the 96 scientific fits until a new token `HCH_V4_4_SOURCE_REAUDIT_PATCH_READY_FOR_SCIENTIFIC_REVIEW` is returned and reviewed.

## Current handoff — v4.4 ADMIT=0 is scientifically safe but audit package is incomplete; repair evidence, then freeze always-legal G0–G3 ladder (2026-09-16)

The first feature-availability run returned the insufficient-evidence stop token with 0 scientific fits/TEST reads/outcome-driven selection. Preserve its conservative scientific consequence: **all target-day provincial forecast columns remain unavailable to the primary method unless later provenance proves otherwise**. But do not accept the package as a completed audit: only the QINGHAI date-mask file exists (`VERIFICATION_REPORT.json` reports `mask_files=1`), the evidence index is placeholder-only, G1/G2/G3 were blanket hard-coded HOLD, derived fields were not numerically checked, and the verifier does not reconstruct five market masks. User-facing month-end coverage dates also disagree with actual source/manifest endpoints. Execute only `experiments/current/hch_v4_4_feature_availability_audit_20260916/BOUNDED_REPAIR_PROMPT.md`; this repair is zero-fit and may legitimately end again with ADMIT=0.

Scientific architecture no longer waits on rich target-day exogenous inputs. The post-audit authority is `docs/current/HCH_V4_4_PREFREEZE_CLOSURE_20260916.md`. Keep exact `b/B/S+/S-` geometry and joint residual training. Primary inputs are only current frozen Host (absolute + within-day-relative), deterministic date/hour channels, and W=7 strictly revealed causal residual history. Encode the current Host with one BiGRU32; encode chronological `(b,B)` history with one causal GRU32; represent Shape history by deterministic active-sign per-hour prototypes/dispersion, not KNN/retrieval. Use a small shared evidence bank and compare shared versus geometry-coordinate-specific softmax/context allocation. Entmax/Market Core/target-day exogenous trajectory encoder are no longer required primary components; they can return only as a later admitted-data extension.

A final mathematical/source-design refinement is now frozen before implementation. The exact geometry is explicitly MAE-aligned: `||r||_1=H|b|+2B`, and any decoded correction obeys daily repair MAE `>=|b*-b_hat|`; this makes Level/B targets more than heuristic auxiliaries. The Shape head is tightened to one centered signed 24h score field whose positive/negative parts are normalized separately. This enforces disjoint signed supports and preserves predicted Jordan-mass semantics instead of allowing independent `S+`/`S-` logits to cancel at the same hour. The refinement is merged into `HCH_V4_4_PREFREEZE_CLOSURE_20260916.md`; source implementation authority is `docs/current/HCH_V4_4_SOURCE_IMPLEMENTATION_PLAN_20260916.md`.

After repaired audit review, the intended first scientific protocol is exactly four variants × 8 exposed cells × seeds 7/17/37 = 96 new postprocessor fits: `G0_GEOM_HOST`, `G1_GEOM_SHARED_CONTEXT`, primary `G2_GEOM_COORD_CONTEXT`, `G3_DIRECT_CONTEXT_CTRL`. No Host/baseline rerun, no TEST/QINGHAI/foreign/final, no paper edit, and no scientific fit until explicit user authorization. Before that experiment, run only the source-only preflight under `experiments/current/hch_v4_4_legal_anchor_source_preflight_20260916/`; required token is `HCH_V4_4_LEGAL_ANCHOR_SOURCE_READY_FOR_SCIENTIFIC_REVIEW`. Do not add zero-sum rescue, PIR retrieval, proposal/safety gate, market expert, deeper capacity ladder or calibration rescue after outcomes are read.

## Current handoff — Round-6 residual design questions are closed except the zero-fit feature-admission manifest; run that audit now (2026-09-16)

The main window has re-read the branch design and completed the missing closest-work audit. Read `docs/current/HCH_V4_4_PREFREEZE_CLOSURE_20260916.md` and `HCH_V4_4_CLOSEST_WORK_COLLISION_AUDIT_20260916.md`. The method architecture is now closed enough for experiment design: exact Level–Balanced-Amplitude–signed-Shape decoder; joint training on original Host residual; one shared 1-layer 24h BiGRU32 trajectory encoder producing day + hour states; deterministic masked-mean Market Core; fp32 entmax-1.5; one Level query, one balanced-amplitude query, 24 Shape queries; inactive-sign Shape loss masking. The collision audit confirms none of the individual primitives is novel; the only paper-facing hypothesis is the combined chain `residual geometry -> repair queries -> coordinate-specific legal-evidence routing -> exact residual reconstruction`.

The only pre-fit blocker is still concrete forecast-origin legality. Primary-source review now gives strong category/timing evidence (national disclosure rule; Gansu pre-bid boundary forecasts; Shaanxi 08:00 renewable forecast submission / 09:00 boundary disclosure / 09:30 bidding cutoff; Ningxia/Qinghai forecast-boundary rules), but workbook provenance and historical interval coverage are not yet sufficient to admit a target-day column. Execute only `experiments/current/hch_v4_4_feature_availability_audit_20260916/AI_EXECUTION_PROMPT.md`; it must remain zero-fit, TEST-target-free and outcome-blind. After its manifest/masks are independently verified, freeze the exact feature pool and the 5-way v4.4 ladder. Do not reopen architecture search unless that audit shows the entire optional feature pool is legally empty.

## Current handoff — main window has taken over v4.4; freeze everything except feature legality, run zero-fit availability audit next (2026-09-16)

The Round-6 v4.4 branch is fully synchronized into canonical state. Read `docs/current/HCH_V4_4_PREFREEZE_CLOSURE_20260916.md` first, then `HCH_V4_4_GEOMETRY_ROUTING_DEEP_DESIGN_20260916.md`. The architecture search is considered closed enough for pre-freeze: exact Level–Balanced-Amplitude–Shape decoder, joint training on original Host residual, always-legal Host/history anchors, one shared compact 24h exogenous trajectory encoder, deterministic masked-mean Market Core, fixed fp32 entmax-1.5 coordinate routing, one day Level query, one day balanced-amplitude query and 24 Shape queries. Shape loss masks inactive signs. No sequential Level, delta/kappa, explicit feature clustering, MoE, full feature Transformer, Gumbel/Top-K selector, market expert, safety gate or dual-calibration headline is in the primary candidate.

The sole blocking task is now forecast-origin legality/provenance. Execute only `experiments/current/hch_v4_4_feature_availability_audit_20260916/{PROTOCOL.md,AI_EXECUTION_PROMPT.md}`. For each China candidate feature/date, require three independent gates: official/provider pre-bid category timing, concrete dataset-column mapping, and date-range coverage. `预测值` alone never passes. Partial evidence produces a frozen date-scoped mask; unsupported dates stay in the benchmark with mask 0. Audit derived relations such as renewable-total and bidding-space without using price/residual outcomes. Required counters: 0 neural fits, 0 TEST target reads, 0 outcome-driven feature selections.

After `HCH_V4_4_FEATURE_AVAILABILITY_AUDIT_COMPLETE`, main-window review should freeze the actual feature pool/masks and only then finalize the 5-way development ladder: `GEOM_NO_EXOG / EXOG_SHARED_ROUTE / V44_COORD_ROUTE / V44_SOFTMAX_ROUTE / DIRECT_DECODER_CTRL`. Do not train in the availability-audit task. No TEST/QINGHAI promotion/foreign/final/paper edit is authorized by this handoff.

## Current handoff — v4.4 is approaching method freeze around exact geometry-induced repair routing; finish issue-time/data-contract details before experiment design (2026-09-16)

Read `docs/current/HCH_V4_4_GEOMETRY_ROUTING_DEEP_DESIGN_20260916.md` first, then the literature map §§30–43 and the USM post-execution audit. No scientific fit is authorized yet.

The strongest new mathematical result is an exact Level–Balanced-Amplitude–Shape residual decoder. For Host residual `r`, use `b=mean(r)`, `B=min(sum(r+),sum((-r)+))`, and signed normalized Shapes. Then `A+=B+[H*b]+`, `A-=B+[-H*b]+`, and `r=A+S+-A-S-` exactly. This restores the project's original Level/Amplitude/Shape semantics while keeping full non-zero-sum expressivity. It removes the v4.0 duplicate-mean problem and the v4.1 frozen-Level error-of-error problem, so the preferred v4.4 training direction is again one jointly trained postprocessor on the original frozen-Host residual, not a sequential Level->Local pipeline. Shape loss is masked when the corresponding target sign mass is zero.

The leading multivariate backbone is now intentionally simple: preserve semantic feature-local tokens; form a deterministic masked-mean Market Core only as a context prior; use fixed 1.5-entmax coordinate routing; one day-level Level/net-mass query, one day-level balanced-amplitude query, and 24 horizon Shape queries; then use the exact decoder above. Avoid explicit feature clustering, full variable self-attention, another generic depth ladder, and preemptive gradient-surgery modules. δ-Adapter (ICLR 2026) already covers sparse horizon-aware Gumbel input masks, and MTAN covers generic global-pool/task-attention; HCH novelty must remain the residual-geometry-induced queries/decoder rather than the sparsifier.

A header audit confirms the public China files already contain substantial candidate forecast trajectories: GANSU 8, NINGXIA 8, SHAANXI 9, QINGHAI 5 `预测值` columns; common across all four are total renewable, hydro+pumped-storage, direct load, non-market-unit output and bidding-space forecasts. These are economically plausible and align with recent Chinese EPF literature, including CSEE 2026 net spot-market bidding capacity. But they are not yet legally admitted to v4.4 merely because the column says forecast: freeze an issue/publication-time manifest for every `(market,target,feature)` before training. Actual-value columns remain target-day forbidden and only legal historically with lag.

Remaining work before v4.4 freeze: (1) formal issue-time availability audit, (2) minimal family-aware 24h trajectory encoder choice, (3) exact one-signed Shape masking convention, (4) minimal ablation/experiment ladder, and (5) final closest-work/claim audit. Net mass remains secondary: initialize the Level head at zero and retain `b=0` as the structural zero-sum reference; do not open another delta/kappa branch. Similar-day/jump/SAM/calibration and other tricks remain implementation-only.

## Current handoff — continue narrowing v4.4 around repair-coordinate sparse cross-routing; no experiment yet (2026-09-16)

Read `docs/current/HCH_MTSF_CAPACITY_FEATURE_NETMASS_LITERATURE_MAP_20260916.md`, especially §§30–37. Round-5 reading further narrows the design: explicit feature clustering/group routers are now secondary because DUET/TimeCAP already occupy that space and generic MMoE/PLE covers task-specific gating. The leading HCH-specific abstraction is **repair-coordinate query routing** over preserved semantic feature tokens, conditioned by one compact market-state core that does not erase local feature identity.

New key precedents: SOFTS/STAR supports centralized core context with redistribution back to local representations; CATS (NeurIPS 2024) treats each future horizon as an independent query and shows cross-attention-only forecasting can outperform heavier self-attention; TAT (ICDE 2026) aligns known future event context directly with difficult future horizons; Adapformer and Li-Net support selective/sparse channel use to avoid irrelevant-variable noise. The provisional query family is one Level query, one Mass query, and 24 horizon-specific Shape queries. Shape hour `h` should be able to access Host state/calendar/future-known exogenous state at hour `h` directly rather than through a pooled daily vector.

Before v4.4 can freeze, resolve only five remaining questions: (1) exact Market-Core pooling operator; (2) sparse routing operator with low tuning burden; (3) Level joint-vs-sequential/frozen training; (4) repository-level audit of which China-5 variables are genuinely forecast-origin/future-known; (5) conservative net-mass treatment. Do not reopen zero-sum mathematics, add explicit clustering/MoE/graph stacks, or start scientific fits. Similar-day/prototype context, jump flags, SHAP, SAM/calibration and other effectiveness tricks remain implementation-only and outside the claimed contribution.

## Current handoff — continue literature deep dive; coordinate-selective channel management now leads explicit learned clustering (2026-09-16)

Read `docs/current/HCH_MTSF_CAPACITY_FEATURE_NETMASS_LITERATURE_MAP_20260916.md` first, especially §§20–29, then `docs/current/HCH_USM_CAPACITY_POSTEXECUTION_CODE_AUDIT_20260916.md`. The user still wants further literature work before the near-final v4.4; no scientific fit is authorized and no final architecture is frozen.

Round-4 reading downgrades explicit K-group learned clustering. DUET, TimeCAP, Adapformer, TSCG/U-Cast-like channel-structure methods and older multi-task feature-learning work make generic automatic grouping/task-specific selection a crowded novelty space. The current leading abstraction is simpler: `semantic legal feature tokens -> compact shared market-state core -> repair-coordinate-specific sparse gates -> Level/Shape/Mass heads`. Feature type remains an embedding/prior, while functional grouping emerges implicitly from the routing patterns. Level/Mass can use day-level feature gates; Shape may use horizon-conditioned gates because domestic EPF evidence shows strong hour-varying relevance.

The scientific selector should be a real information bottleneck (multiplicative gate, sparse normalized mixture or Top-K mask), not a standard attention matrix later interpreted as importance. Visualized routing weights should be described as routing behavior; feature-importance claims require ablation/occlusion/SHAP-style validation. TFT supplies instance-wise variable selection, Adapformer provides target-specific Top-K covariate selection, and SOFTS supports centralized global-core aggregation. Their primitives are prior art; HCH's defensible object is the mapping from heterogeneous forecast-origin evidence to **mathematically derived repair coordinates of a frozen Host**.

DAG (ICML 2026) creates an important exogenous-data rule: historical endogenous, historical exogenous and future-known exogenous trajectories should not be flattened into one undifferentiated tensor. If legal future load/wind/solar/scheduled forecasts exist, encode their trajectories/alignment separately before routing them to repair heads. Do not import DAG's full correlation-discovery architecture into the first v4.4 design unless a simpler selector later fails.

Domestic EPF literature continues to support an implementation hierarchy: strict forecast-origin legality -> minimal TRAIN-only coarse guardrail -> learned relevance/routing -> task-specific correction. XGBoost/MIC/LassoNet-style sparsity, similar-day statistics, holiday/event encoding, exogenous dropout, SAM, calibration, jump flags and caching remain implementation-only effectiveness tools, not contributions.

For zero-sum, do not open another long mechanism branch. `q=A+S+−A−S−` already removes the representation restriction; `D=sum(q)` remains weakly predictable. Preserve the unbalanced signed-mass geometry as a secondary mathematical contribution and consider a zero-default/shrinkage treatment of D, but do not make v4.4 accuracy depend on strong D learnability.

## Previous handoff — USM ladder executed; audit blocks promotion pending bounded calendar/D3 repair decision (2026-09-16)

Read `docs/current/HCH_USM_CAPACITY_POSTEXECUTION_CODE_AUDIT_20260916.md` first, then the executed stage `experiments/current/hch_unbalanced_signed_mass_capacity_20260916/{RESULTS.md,EXECUTION_PLAN.md,PROTOCOL.md}`. Do not launch another architecture and do not open TEST/foreign/final.

The 96 registered new fits completed, but the final token was correctly withheld. Existing D29 affects D3's §11 coordinate reporting only: the code decomposed composed `b_hat+q_hat` and tied RCL Shapes instead of D3's emitted `q_hat` Jordan coordinates. It changes no gate and can be recomputed from existing logs.

Post-execution code audit found a more consequential input-contract defect: `usm_features.calendar_channels` uses `days_since_epoch % 7` directly although epoch day zero is Thursday, then sets `is_weekend = dow>=5`. This flags Tuesday/Wednesday as weekend and misses Saturday/Sunday. U0/U1 do not use this path; U2/U3/D3 do. PROTOCOL §6 also defined U2 as dual Host + raw-summary history, while the executed U2 includes seven calendar channels. Therefore the current positive `CONTEXT_EXPANSION_SUPPORTED` result is informative about the implemented context package but is **not promotable as the registered U2 mechanism** without replay. U3-vs-U2 and U3-vs-D3 share the same wrong calendar substrate, so their internal contrasts remain descriptive, not registered-complete.

Scientific reading that survives: generic network depth is not the leading bottleneck. U1 is ~6.1x larger than frozen R2 (135,524 vs 22,244 params) but panel gain is only +0.0503 pp and D MAE improves 3/8; U3 at 387,428 params adds only +0.0239 pp over U2. The original hypothesis about nonlinear interaction among domestic load/renewable/market covariates was **not tested**, because the authorized boundary contained no legal forecast-known exogenous variables beyond Host/calendar. On zero-sum, U1/U2/U3 correctly implement `q=A+S+−A−S−`, but `D=A+−A−=sum(q)=24*mean(q)` stays essentially as hard as `D_hat=0`; thus representation is fixed but signed net-mass predictability is not.

Do not promote the current B/E statement that smaller B-MAE than E-MAE proves target excess dominance; MAEs are prediction errors, not target composition. If the user authorizes bounded repair, change only calendar semantics + D3 coordinate reporting + related wording, and replay U2/U3/D3 exactly (72 fits, same seeds/split/recipe/gates). U0/U1 need not be replayed for these defects. A publication-grade capacity-only claim would additionally need an optional U1 50/8 replay because frozen U0 used v4.1's 50/8 recipe while U1 used 60/10.

## Previous handoff — unbalanced signed-mass Local capacity ladder is the next authorized design target (2026-09-16)

Read `docs/current/HCH_UNBALANCED_SIGNED_MASS_INTERACTION_DESIGN_20260916.md`, then `experiments/current/hch_unbalanced_signed_mass_capacity_20260916/{PROTOCOL.md,AI_EXECUTION_PROMPT.md}`. The v4.0↔v4.1 reconciliation is complete; do not rerun it. The next development question is now explicitly **network/input capacity under a fixed mathematically complete Local object**, not another delta/kappa redesign.

Stage 1 is reused/frozen from verified v4.1. Stage 2 target is `q=r-b_hat*1` and the Local representation is exact unbalanced signed mass: `q=A+S+−A−S−`. Zero-sum is only the special case `A+=A−`; no explicit `delta` or `kappa` head is used. Interpret `B=min(A+,A−)` as balanced redistribution mass and `|A+−A−|` as unbalanced excess. This is analogous to mass transport plus creation/destruction in unbalanced OT, but no OT/Sinkhorn solver is part of the method.

The capacity ladder isolates one change at a time on the same 8 development cells and seeds 7/17/37: U0 = frozen v4.1 R2 read-only reference; U1 = same inputs/targets/loss with a materially deeper/wider Stage-2 network; U2 = U1 plus absolute+relative Host views and compact raw-residual regime-summary history; U3 = U2 plus a dense feature mixer and 2-layer bidirectional 24h horizon GRU; D3 = matched-capacity direct-q control. New fit budget is exactly 96; Stage-1 is not refit. Raw output is the only promotion surface; shared/dual scalar calibration is secondary diagnostic only and cannot replace headline predictions.

No TEST/joint/QINGHAI/foreign/final, Host/baseline rerun, safety/gate/retrieval/MoE/router/market expert, hyperparameter sweep or paper update. Implement experiment-only first; do not promote/rewrite `src/core` before support. Required preexec token: `HCH_USM_CAPACITY_PREEXECUTION_VERIFIED`; required final token: `HCH_USM_CAPACITY_LADDER_COMPLETE_FOR_ADJUDICATION`, then stop for main-window review.

## Previous handoff — v4.0↔v4.1 reconciliation completed; calibration is MATERIAL, not DOMINANT (2026-09-16)

`HCH_V4_0_V4_1_RECONCILIATION_COMPLETE_FOR_ADJUDICATION` completed with 0 new neural fits. P0 parity passed 8/8; CAL/EVAL IDs and stored residuals match exactly; TEST=0; parent evidence hashes unchanged; independent verifier 24/24 PASS. The missing authority path has now been filled by a post-execution adjudication file at `docs/current/HCH_V4_0_V4_1_CROSS_STAGE_RECONCILIATION_20260916.md`; it must not be retroactively described as a pre-execution document.

The main reconciliation result is protocol comparability: v4.0's historical `+0.984%` C2-RAW headline is already dual-static calibrated, while v4.1 R1 headline is raw. Matched raw gives `gap_raw=-0.8206 MAE` (v4.0 better); matched dual gives `gap_dual=+0.4145` (v4.1 better). `R_cal=0.4948857` sits just below the preregistered 0.50 DOMINANT boundary, so `CALIBRATION_GAP=MATERIAL`. Other labels: Level SMALL, Local SMALL, history information loss NOT_DETECTED, absolute Host-level confound PLAUSIBLE_ONLY, parameterization UNRESOLVED. Differences live mainly in the daily-mean coordinate; centered Shape quality is comparatively stable.

## Current handoff — RCL v4.1 complete; explicit delta closure is NOT supported, stop before another candidate (2026-09-16)

`HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_20260915` is complete and independently verified. Phase-0 gate passed 182/182 tests with 0 fits/0 TEST reads; the scientific campaign then executed exactly 216 fits (96 causal OOF Level + 24 final-Level refits + 96 Stage-2) on the fixed 8-cell TRAIN+VAL panel. Verifier: 4733 checks, 0 failed. TEST target read count remained 0; joint cache, QINGHAI, foreign/final, Host/baseline retraining and `src/core` mutation were absent. Terminal token: `HCH_RCL_V4_1_SCIENTIFIC_EXECUTION_COMPLETE_FOR_ADJUDICATION`. Read `experiments/evidence/hch_residual_complete_local_v4_1_20260915/RESULTS.md` for the compact evidence table.

Primary scientific verdict: **do not promote R1 `RCL_ORTHOGONAL`**. R1 vs matched R0 hard-zero-sum is only 4/8 non-worse with panel median `-0.0249%`; all five support gates fail. The learned delta closes the Stage-1 mean miss in only 2/8 cells and panel closure ratio is `1.049661` (>1), so the explicit Local mean head worsens rather than recovers the typical coarse-Level mean error. R1 vs Host is positive only 3/8, panel median `-0.2685%`, worst `-1.9951%`; development target not met. GANSU/LSTM remains a strong special cell (`+11.503%` vs Host), but most of that gain already exists in R0, so it does not validate delta closure.

R1 vs R2 untied signed mass is exactly mixed (4/8 each; median `-0.0144%` for R1), and R3 Direct-Q beats R1 5/8 with median `+0.5876%` but fails the >=1% dominance gate. Therefore neither orthogonal coordinates, untied signed masses nor Direct-Q has earned promotion. Keep v4.0's structural result `CALIBRATED_ZERO_SUM=MATERIAL`, but reinterpret it correctly: a nonzero-mean remainder **exists**, yet v4.1 shows it is not reliably predictable under the current information/architecture. The bottleneck is now observability / conditional predictability, not merely representation freedom.

Do not rescue RCL with a deeper delta network, new gate/safety, market-specific routing, extra feature blocks or post-hoc calibration from this evidence. Do not open TEST/foreign/final or promote R1 into paper claims. Before any new experiment, perform an existing-artifact-only reconciliation of why v4.0 C2-RAW was broad (7/8 positive vs Host, panel median ≈+0.984%) while the cleaner hierarchical v4.1 R1 is only 3/8 positive. Candidate explanations to adjudicate separately: frozen vs jointly learned Level, raw-residual vs post-Level q-history, target parameterization, calibration dependence, and information observability. No new scientific experiment is authorized by this handoff.

## Current handoff — execute the conditionally authorized RCL v4.1 campaign only through the final patch gate (2026-09-15)

Read `docs/current/HCH_RCL_V4_1_FINAL_PREEXECUTION_AUDIT_AND_EXECUTION_DESIGN_20260915.md` and then `experiments/current/hch_residual_complete_local_20260915/SCIENTIFIC_EXECUTION_PROTOCOL.md`. Active `src/core` is now source-ready, but do not start scientific fits from the current runner without the registered Phase-0 patch: fix Stage-1 scale leakage / separate `s_b`, seed-before-model-construction, 3-seed median final Level ensemble, B1 q-history warm-start, native R2 A+/A− supervision, and the registered metric arithmetic errors. The latest combined source/harness suite before this patch is `148 passed, 1 warning`.

The user has explicitly authorized the scientific experiment in this turn, so no additional human authorization is required **after** `RCL_V4_1_FINAL_PREEXECUTION_GATE_PASS`. Phase 0 must still be fit-free and TEST-free; on any failure return `RCL_V4_1_FINAL_PREEXECUTION_GATE_FAIL_STOP`. If PASS, execute exactly the 8-cell TRAIN+VAL ladder with 96 OOF Level fits + 24 final-Level refits + 96 Stage-2 fits = 216 fits, R0/R1/R2/R3, seeds 7/17/37, raw-output primary evaluation, and independent verification. Local-AI prompt: `experiments/current/hch_residual_complete_local_20260915/AI_SCIENTIFIC_EXECUTION_PROMPT.md`. Final token after verifier PASS: `HCH_RCL_V4_1_SCIENTIFIC_EXECUTION_COMPLETE_FOR_ADJUDICATION`. Stop there; TEST/QINGHAI/foreign/final/baseline rerun/paper edits remain forbidden.

## Current handoff — RCL active core is source-ready; scientific run still blocked on harness/comparator preparation (2026-09-15)

Read `docs/current/HCH_RCL_FINAL_SOURCE_GATE_20260915.md` first. The second independent audit clears the active `src/core`: the previous Shape-history dead path, duplicate amplitude head, feature-axis normalization collapse, masked-statistics defects, double-scaled ramp, missing feature-mask channels, weak freeze check, nonmonotonic q-history, and masked-horizon inconsistencies are fixed. Targeted RCL/current-core tests are `110 passed, 1 warning`; independent probes show nonzero Shape/Local response to rho-history, no dead R1 trainable parameters, F=1 feature variation preserved, correct masked median, and correct single-scale ramp. Archive preservation remains 15/15 SHA256 with canonical digest `bce2e82dc20435db9a119d35eb2629b76eed53b7ee0eb6f5b53b438c4f9ed49e`. Core status is `RCL_V4_1_ACTIVE_CORE_SOURCE_READY`.

Do **not** run the RCL scientific experiment yet. Remaining blockers are experiment-layer: R3 Direct-Q currently ignores the computed Shape/rho-history branch in its q output and therefore has dead Shape parameters; R0 does not reuse the exact R1 `BalancedAmplitudeBranch` primitive; the runner has not frozen TRAIN-only coarse-Level conditioning stats; and the registered Stage-1 OOF/final refit + Stage-2 ladder + VAL prequential evaluator + independent verifier are not implemented yet. Run only `experiments/current/hch_residual_complete_local_20260915/AI_EXECUTION_HARNESS_PREPARATION_PROMPT.md`. It is source/synthetic-smoke only and must return `RCL_V4_1_EXECUTION_HARNESS_READY_FOR_AUTHORIZATION` with scientific fit count 0 and TEST read count 0. After that token, perform one final pre-execution review; only explicit user authorization may start the 8-cell scientific run.

## Current handoff — RCL source alignment reviewed; do NOT execute experiment until post-alignment patch is reviewed (2026-09-15)

Read `docs/current/HCH_RCL_CORE_POSTALIGNMENT_AUDIT_20260915.md` first. The prior source token `RCL_V4_1_CORE_ALIGNMENT_READY_FOR_EXECUTION_REVIEW` is **not** an execution authorization. Broad method/core parity is now in place and the old core archive rechecks 15/15, but adversarial review found implementation defects that existing 90 passing tests missed.

Highest-priority blockers: Shape history is currently dead (`shape_repr.latent` never reaches the Shape head; changing only `shape_history` changes Shape/Local by exactly 0); R1 instantiates a second unused `amplitude_head` alongside the actual `BalancedAmplitudeBranch`; and `build_base_tokens` normalizes forecast features over feature identity rather than horizon, causing any F=1 time-varying feature to collapse to zero. Also fix true masked median/valid-count normalization, the twice-divided Shape ramp, explicit forecast-feature mask channels, strict Stage-1 `requires_grad=False` enforcement, strictly increasing prequential q-history, partial-mask consistency, and stale Shape/Amplitude docstrings.

Run only `experiments/current/hch_residual_complete_local_20260915/AI_POSTALIGNMENT_CORE_PATCH_PROMPT.md`. Required terminal token: `RCL_V4_1_POSTALIGNMENT_PATCH_READY_FOR_REVIEW`. The task is source-only: tests allowed, scientific fits forbidden, TEST/foreign/final reads forbidden, paper edits forbidden. After the token returns, perform another main-window source audit; only then may the user separately authorize the 8-cell RCL scientific experiment.

## Current handoff — RCL v4.1 core alignment refactor DONE (source only); stop for main-window execution review (2026-09-15)

`RCL_V4_1_CORE_ALIGNMENT_READY_FOR_EXECUTION_REVIEW` has been returned. **Stop here. Do not start the RCL scientific run.** This token authorizes nothing beyond source readiness; the experiment still needs separate explicit main-window authorization.

Delivered by this refactor:

1. **Archive** — `src/archive/alignment_safe_core_pre_rcl_20260915/legacy_core/` (15 files, per-file SHA256 all matching, canonical tree digest `bce2e82dc20435db9a119d35eb2629b76eed53b7ee0eb6f5b53b438c4f9ed49e` with a documented recipe). Re-verified after the active rewrite: 0 mismatches. Manifest: `ARCHIVE_MANIFEST.md`; superseded-test mapping: `SUPERSEDED_TESTS.md` + `legacy_tests/`.
2. **Active core** — 12 modules. `CoarseLevelModel` + `ResidualCompleteLocalModel`; `targets.py` is the only place the decomposition is defined; `safety.py`/`history.py` archived out.
3. **Parity** — `build_candidate("R1_RCL_ORTHOGONAL")` returns `core.model.ResidualCompleteLocalModel`; no duplicate R1 class anywhere under `src/mvp`; R0/R2/R3 experiment-only.
4. **Six v4.0 defects** — structural tests, 90 passing, listed in `src/core/README.md`.
5. **Counts** — scientific fit count 0; TEST label read count 0; no `paper/**` change.

Known, disclosed pre-existing failures in the wider foundation suite (not introduced here, not hidden): 10 tests fail with `FileNotFoundError` on V2.5-era evidence paths (`experiments/evidence/v2_5_legacy_results/**`, `experiments/evidence/v2_5_pilots/**`, `experiments/11-hch-v2/manifests/**`). None imports or depends on `src/core`; none of those paths was touched by this refactor.

## Previous handoff — RCL v4.1 design accepted, but active core is not aligned; run source-only refactor before any scientific fit (2026-09-15)

Read `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md` first. A source/math audit of the verified v4.0 ladder found that the registered C2 did not cleanly realize “Level handles coarse bias; Local handles the actual leftover residual.” Its target `q=r-b_tilde*1` depends on an OOF Level prediction that Local never explicitly observes; inference uses a different internal Level head; C2 supervises that head toward `mean(q)` while the `M/kappa/S` branch can already reconstruct full `q`, so coordinate-perfect outputs generally conflict with reconstruction of `r`; and the residual history/normalization geometries are not fully aligned with the q target. Independent verification of v4.0 execution still stands, and the geometric result `CALIBRATED_ZERO_SUM=MATERIAL` remains authoritative, but `PREDICTION_CONDITIONAL_IMBALANCE=MIXED` must not be overread as a terminal rejection of prediction-conditional Local.

The new first-principles candidate is hierarchical. Stage 1 is a standalone coarse Level predictor trained only on `b*=mean(r)` and frozen. TRAIN uses strict causal OOF Level predictions `b_tilde`; Local target is the actual leftover `q=r-b_tilde*1`, and Local explicitly receives `b_tilde` plus post-Level q-history. Decompose `q` uniquely as `delta*1 + rho`, with `delta=mean(q)=b*-b_tilde`, `sum rho=0`, and `rho=A(S+−S−)`. Final correction is `b_hat*1 + delta_hat*1 + A_hat(S_hat+−S_hat−)`. The Local operator is therefore residual-complete and nonzero-sum overall; only its centered Shape component is zero-sum. Frozen Stage-1 Level makes `delta` identifiable, so this is not two jointly free Level heads.

The returned `RCL_PREPARATION_TESTS_PASS` is only a preparation pass. Audit of the actual source shows two blockers: (1) `src/mvp/hch_residual_complete_local_v4_1/` currently contains only target/history helpers and caller-side contract guards, not the Stage-1/Stage-2 networks or runners; (2) active `src/core` still implements the previous exact-zero-sum alignment-safe candidate and its DESIGN_CONTRACT explicitly forbids a daily-level/bias term. Therefore the new method, active core and future experiment are not yet the same implementation object.

Before any scientific fit, execute the source-only authority `docs/current/HCH_RCL_CORE_ALIGNMENT_REFACTOR_PLAN_20260915.md` via `experiments/current/hch_residual_complete_local_20260915/AI_CORE_ALIGNMENT_REFACTOR_PROMPT.md`. The refactor must byte-preserve/archive the current alignment-safe core, rewrite active core around `CoarseLevelModel` + `ResidualCompleteLocalModel`, make R1 import the active core directly, keep R0/R2/R3 as experiment-only controls, and close the six v4.0 semantic defects with structural tests rather than metadata assertions. Required terminal token is `RCL_V4_1_CORE_ALIGNMENT_READY_FOR_EXECUTION_REVIEW`; then stop for main-window audit. No scientific fit, V2 TEST/foreign/final read, paper edit, Host/baseline retraining, safety/gate/retrieval/attention/TCN/Transformer/MoE/market-specific tuning is authorized by the refactor prompt.

## Current handoff — v4.0 mechanism ladder complete; zero-sum restriction is real, but learned imbalance is only mixed (2026-09-15)

Read `RESEARCH_STATE.md` top entry and `experiments/evidence/hch_v4_0_mechanism_ladder_20260915/MAIN_WINDOW_HANDOFF.md` first. The repaired/finalized `HCH_V4_0_MECHANISM_LADDER_20260915` is now complete and independently verified: 96/96 candidate fits, 16 required artifacts, chronological CAL/EVAL, matched `MECH_TRAIN=B2∪B3∪B4`, warm-up+B1 excluded, finalizer clean exit, and TEST target read count 0. Terminal token: `HCH_V4_0_MECHANISM_LADDER_COMPLETE_FOR_ADJUDICATION`.

Final labels: `CALIBRATED_ZERO_SUM=MATERIAL`; `DUAL_STATIC_CALIBRATION=MIXED`; `PREDICTION_CONDITIONAL_IMBALANCE=MIXED`; `CALIBRATION_BEFORE_RESIDUALIZATION=NOT_SUPPORTED`; `REPRESENTATION=MIXED`; `EXTREME_REPAIR=MIXED`.

The strongest durable conclusion is mathematical/structural: Level-only causal calibration does not remove the nonzero-mean remainder. Calibrated median `|kappa|` remains `0.471..0.816` across all 8 cells (panel median ≈0.634), and increases in 7/8 cells versus raw kappa. Therefore strict zero-sum Local is genuinely too restrictive on this panel. However the minimal learned imbalance candidate is not universally successful: C2-CAL beats C1-MATCH in only 4/8 cells. Do not equate “representation bottleneck exists” with “current kappa head is final method”.

Do not force calibrated Level residualization: C2-CAL beats C2-RAW only 3/8 and panel median change is slightly negative, so `CALIBRATION_BEFORE_RESIDUALIZATION=NOT_SUPPORTED`. C2-RAW is actually the broadest Host-relative candidate in this development slice (positive 7/8 vs Host, panel-median ≈+0.984%, worst ≈-0.524%), whereas C2-CAL is positive 4/8 but has the strongest single-cell GANSU/LSTM gain (~+12.44%).

Dual static calibration and component geometry are heterogeneous rather than universal. C1-MATCH vs C0-MATCH is positive 4/8 with panel median only ≈+0.183%; Stage-0 map: GANSU/LSTM and SHANDONG/TimeMixer Level-dominant, GANSU/TimeMixer and SHANDONG/iTransformer complementary, SHAANXI/TimeMixer and NINGXIA/LSTM Local-dominant. Keep dual scalars as a calibration/control tool, not a settled headline contribution.

The same-backbone unrestricted C3-DIRECT does not dominate: verdict MIXED. Direct wins 4/8 only modestly, while structured C2-CAL wins strongly on GANSU/LSTM, SHANDONG/TimeMixer and GANSU/PatchTST. Structured coordinates remain scientifically defensible; unrestricted vector correction is not an obvious replacement.

Extreme repair is MIXED and concentrated: C2-CAL vs C1 combined-extreme gains are about +2.14% GANSU/LSTM, +7.32% GANSU/TimeMixer, +15.34% GANSU/PatchTST, +0.81% SHAANXI/TimeMixer; -3.22% SHANDONG/TimeMixer, -3.33% SHANDONG/iTransformer, and ~0 for NINGXIA/SHAANXI-PatchTST. This is promising for the original difficult/extreme-price story, but not panel-wide.

No like-for-like frozen δ-Adapter/PIR/MDR/COSA artifact was sliceable to this exact EVAL surface, so this experiment cannot support claims of baseline superiority. Do not open V2 TEST, foreign/final, modify `src/core`, update `paper/**`, or automatically launch another experiment from this handoff.

## Current handoff — execute one v4.0 mechanism ladder; standalone component-geometry stage is absorbed, not run separately (2026-09-15)

Controlling design: `docs/current/HCH_V4_0_MECHANISM_LADDER_EXPERIMENT_DESIGN_20260915.md`. Execute exactly `experiments/current/hch_v4_0_mechanism_ladder_20260915/{PROTOCOL.md,AI_EXECUTION_PROMPT.md}`. The user has authorized this single broader TRAIN+VAL campaign. The older unexecuted `hch_component_geometry_adjudication_20260915` must **not** be launched separately: its no-new-fit Tracks A–D are Stage 0 inside v4.0, preserving three chronological calibration cuts, Level/Local/Full MAE direction-vs-scale diagnostics, frozen Tail/Normal geometry, and the static component map.

After Stage 0, complete the full preregistered ladder regardless of intermediate outcomes: causal calibrated-kappa; C0-MATCH shared-alpha reference; C1-MATCH static `alpha_L/alpha_Z`; C2-RAW and C2-CAL with exactly one additional bounded `kappa` scalar (`A±=M(1±kappa)/2`); and C3-DIRECT as a same-backbone low-capacity unrestricted residual control. Fixed cells are the original six v3.9 cells plus GANSU_DA/PatchTST and SHAANXI_DA/PatchTST. The two PatchTST cells were selected using already-exposed development evidence, so the eight-cell panel is development evidence only, never untouched-final evidence.

The executor correctly stopped before fitting because the old wording left warm-up target construction undefined. The adjudication is now frozen: `MECH_TRAIN = B2∪B3∪B4`. Warm-up and B1 are excluded from all primary candidate losses; no in-sample Level prediction, `beta=1`, future calibration, or new expanding-origin proxy may fill them. C0-MATCH, C2-RAW, C2-CAL and C3-DIRECT must use identical target-day IDs, and C1-MATCH derives from C0-MATCH. Old six-cell full-TRAIN v3.9 artifacts remain Stage-0/descriptive only. This matched-support correction is mandatory because otherwise C2 would have less training data than its controls and any failure would be uninterpretable.

All final variant comparisons are on the second chronological half of VAL after CAL-only scalar fitting. TRAIN-only 5/95 target percentiles define upper/lower extreme slots; TRAIN-OOF Host daily-MAE p90 defines the descriptive failure-tail subset. These subsets are analysis only and cannot route predictions. Reuse original six-cell OOF artifacts where legal; only supplement the two PatchTST OOF Level artifacts. All 8 C0-MATCH/C2/C3 candidate fits are fresh on matched `MECH_TRAIN`. GPU/CPU parallelism and artifact reuse are allowed, but scientific settings cannot change.

Hard boundaries: TEST read count 0; no QINGHAI; no foreign/final; no Host/baseline retraining; no `src/core` edits; no safety/gate/Proposal Verification; no retrieval/KNN; no attention/Transformer/TCN/MoE/router; no market-specific tuning or result-conditioned rescue; no `paper/**` edit. Require independent verification and the terminal token `HCH_V4_0_MECHANISM_LADDER_COMPLETE_FOR_ADJUDICATION`, then stop. `EXPERIMENT_LEDGER.md` remains unchanged until verified results return.

## Previous handoff — execute component-geometry adjudication before any v4.0 candidate (2026-09-15)

Controlling design: `docs/current/HCH_COMPONENT_GEOMETRY_ADJUDICATION_DESIGN_20260915.md`. Execution protocol/prompt: `experiments/current/hch_component_geometry_adjudication_20260915/{PROTOCOL.md,AI_EXECUTION_PROMPT.md}`. This is one comprehensive mechanism experiment with **no new neural fits**. It exists because the previous stage simultaneously found raw zero-sum restriction material and shared-alpha calibration defective; directly training untied Local now would confound representation with calibration.

Execute four tracks only: (A) use prior OOF blocks to fit a strictly causal nonnegative L1 `beta_L`, then recompute calibrated `d/kappa/zero-sum floor` on B2..B4; (B) stress-test shared-vs-separate Level/Local calibration over three chronological VAL cuts and compare HOST/LEVEL_ONLY/LOCAL_ONLY/SHARED/SEPARATE; (C) use `mae_alignment_support`, `exact_mae_safe_radius`, and `mae_excess_risk` from `src/core/safety.py` strictly as offline mathematical diagnostics for Level/Local/Full direction-vs-scale geometry, including frozen Tail/Normal masks; (D) produce a majority-vote component map `LEVEL_DOMINANT / LOCAL_DOMINANT / COMPLEMENTARY / ABSTAIN_STATIC` plus Level/Local head evidence.

Do not read V2 TEST, add QINGHAI, rerun Host/baseline, train untied amplitude/M-kappa/dual-alpha neural candidates, enable history safety/gates/retrieval, modify `src/core`, redefine tail thresholds, or edit `paper/**`. Final token must be `HCH_COMPONENT_GEOMETRY_ADJUDICATION_COMPLETE`, then stop for human adjudication. Only if calibrated zero-sum remains material should prediction-conditional imbalance Local be reopened; if calibrated kappa collapses but dual-alpha remains robust, the next candidate should be the minimal zero-sum Level+Local with separate static component calibration.

## Current handoff — Predicted-Level Remainder diagnostic complete; calibrate Level before deciding untied Local (2026-09-15)

Read `docs/current/HCH_PREDICTED_LEVEL_REMAINDER_POSTEXECUTION_ADJUDICATION_20260915.md` first. The diagnostic completed with `HCH_PREDICTED_LEVEL_REMAINDER_DIAGNOSTIC_COMPLETE_FOR_ADJUDICATION`, independent verifier PASS, 72/72 causal OOF fits and TEST target read count 0. Final labels: `SIMPLE_SIGNAL_NOT_DETECTED`, `ZERO_SUM_RESTRICTION_MATERIAL`, `SHARED_ALPHA_BOTTLENECK_SUPPORTED`.

Level remainder `d=b*-b_tilde` has no simple causal signal under the frozen Ridge feature set: Ridge loses to ZERO in all six cells (`-33.77% .. -5.34%`, panel median about `-16.16%`), while panel-median ACF(1)/ACF(7) are near zero. Do not overstate this as impossibility; it only closes the tested simple predictor.

The raw zero-sum floor is large in every cell: median `|kappa|=0.442..0.694`, panel median about `0.600`, and roughly 88–93% of OOF days exceed 0.1. However **do not yet promote untied `A+/A-`**: kappa was computed after raw, uncalibrated OOF Level, while the same stage independently proves that Level and Local need separate calibration. First reuse the persisted OOF Level predictions and fit a strictly causal Level-only scalar on prior OOF blocks, then recompute calibrated `d`, `kappa`, and floor. No new neural fit is required. Only if calibrated kappa remains material should prediction-conditional untied Local become the next architecture candidate.

Shared alpha is genuinely unsupported as the final Level+Local coupling: separate static Level/Local scalars improve 4/6 EVAL cells, panel median `+0.874%`, worst `-0.178%`. Joint optima often switch one component fully off, so this is component-direction heterogeneity, not merely a minor scale mismatch. If Level+Local survives later design, use two static nonnegative component scalars rather than one shared alpha; this is not an instance gate/safety mechanism.

Do not open V2 TEST, foreign/final data, train an untied candidate, add safety/gate/retrieval, or modify `src/core` automatically. `paper/**` remains locked without explicit authorization.

## Previous handoff — run Predicted-Level Remainder diagnostic next; no candidate redesign yet (2026-09-15)

Read `docs/current/HCH_PREDICTED_LEVEL_REMAINDER_DIAGNOSTIC_DESIGN_20260915.md` first, then `experiments/current/hch_predicted_level_remainder_diagnostic_20260915/PROTOCOL.md`. The next stage is diagnostic only. It exists because v3.9 showed that oracle centering makes Local exactly zero-sum, but a learned Level predictor leaves `d=b*-b_tilde`; under strict zero-sum Local this creates an exact irreducible daily-MAE floor `|d|`. For `q=r-b_tilde*1`, define `kappa=(A+−A−)/(A++A−)=H d/||q||_1`; therefore `|kappa|` is exactly the fraction of the current remainder MAE that even an oracle-perfect zero-sum Local cannot remove.

Execute the same six v3.9 cells only. TRAIN is used for four chronological causal OOF blocks after a 40% warm-up; each fold uses seeds 7/17/37 and the unchanged v3.9 Level+Local architecture only to generate raw Level predictions, totaling 72 fits. Then assess whether `d` has simple causal signal (ZERO/LAST/MED7/fixed Ridge), quantify the `kappa` distribution and zero-sum floor, and read the already-frozen v3.9 VAL artifacts to compare one shared MAE-optimal alpha against exact two-parameter nonnegative Level/Local L1 calibration on chronological CAL/EVAL. Final labels are only `level_remainder_signal`, `zero_sum_restriction`, and `shared_alpha`, followed by `HCH_PREDICTED_LEVEL_REMAINDER_DIAGNOSTIC_COMPLETE_FOR_ADJUDICATION`.

Do not train untied amplitude or dual-alpha candidates, do not modify `src/core`, do not reopen baseline/Host training, do not add QINGHAI, and keep V2 TEST read count at 0. AI execution prompt: `experiments/current/hch_predicted_level_remainder_diagnostic_20260915/AI_EXECUTION_PROMPT.md`. Stop after the diagnostic for human adjudication.

## Current handoff — v3.9 verified negative result; next issue is prediction-conditional residualization, not another zero-sum retry (2026-09-15)

`HCH_LEVEL_LOCAL_V3_9_VAL_PILOT_20260915` has completed and independently verified with token `HCH_LEVEL_LOCAL_V3_9_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`. 18/18 TRAIN/VAL fits completed, TEST target read count remained 0, and Local zero-sum invariants passed. Results vs Host / M0: GANSU/LSTM `+4.929/+5.070%`; GANSU/TimeMixer `+0.013/-1.251%`; SHANDONG/TimeMixer `+0.710/+1.178%`; SHANDONG/iTransformer `+0.534/-0.770%`; SHAANXI/TimeMixer `+0.657/-1.877%`; NINGXIA/LSTM `0/+1.446%`. Only 3/6 beat M0 and Level beats zero reference only 2/6. NINGXIA's apparent M0 win is pure abstention (`alpha_0=0` in all seeds); GANSU/TimeMixer is nearly abstaining as well (median `alpha_0≈0.080`).

The central mathematical correction is now frozen in `docs/current/HCH_V3_9_POSTPILOT_MATH_ADJUDICATION_20260915.md`: `r=b*1+A*(S+*-S-*)` is an exact **oracle target decomposition** only when `b*=mean(r)` is known. At inference, hard-zero-sum Local means all daily-mean correction depends on learned `b_hat` (and the shared scalar alpha); `mean(error_after)=b*-alpha b_hat`, so MAE is lower-bounded by `|b*-alpha b_hat|`. Since Level is weak, the zero-sum restriction can become a practical bottleneck. Local amplitude is also not exonerated: centered-amplitude gap improves only 2/6 cells and worsens 3/6.

Do not simply turn on two free amplitude heads and joint-train. The more rigorous candidate is **predicted-Level residualization**: obtain chronological/out-of-fold TRAIN Level predictions `b_tilde`, define `q*=r-b_tilde 1`, then decompose `q*` exactly as `A+*S+*-A-*S-*`. The imbalance obeys `A+*-A-* = H(b*-b_tilde)` exactly, so it is a supervised fallback for actual Level miss rather than an arbitrary duplicate Level. Prefer total mass + imbalance ratio `kappa=(A+-A-)/(A++A-)`, with `kappa=0` recovering v3.9.

No new neural training is authorized yet. First run only read-only diagnostics on already-open TRAIN/VAL artifacts: (1) whether Level prediction residual is observable/structured; (2) distribution of the implied imbalance ratio under out-of-fold/predicted Level, especially in PIR-gap and v3.9-failure cells; (3) whether raw Level, raw Local and full correction want materially different calibration slopes, because the current single `alpha_0` strongly couples them. Do not open V2 TEST, foreign/final, add safety/gate/retrieval, or edit `paper/**` without explicit authorization.

## Current handoff — HCH Level–Local v3.9 prepared; Gain removed, no experiment run by ChatGPT (2026-09-15)

The v3.8 Full-LGR pilot is closed. The next route is now narrowed to **Level + Local**, not because amplitude correction is being abandoned, but because the existing Local signed-simplex operator can represent the entire zero-mean residual exactly once Level removes the daily mean.

Controlling design: `docs/current/HCH_LEVEL_LOCAL_V3_9_DESIGN_20260915.md`. Execution protocol: `experiments/current/hch_level_local_v3_9_pilot_20260915/PROTOCOL.md`. Short executor prompt: `experiments/current/hch_level_local_v3_9_pilot_20260915/AI_EXECUTION_PROMPT.md`.

For `r=y-yhat_H`, define `b*=mean(r)` and `rho*=r-b*1`. Then `sum(rho*)=0`, so `rho*=A*(S+−S−)` exactly with `A=||rho*||_1/2`. Therefore Local is not merely a timing/redistribution head: `A` is the amplitude of the centered correction and `S+/S−` allocate that amplitude over the 24 hours. A Host daily excursion that is too weak/strong is a centered residual pattern and belongs to Local after Level removal.

Do **not** untie `A+/A−`: `A+−A−` would recreate a nonzero-sum degree of freedom and make Local compete with Level. Do **not** add Gain back or rescue it with a larger network. The v3.8 Gain result only closes the explicit Host-aligned scalar Gain coordinate as a supported standalone object.

The v3.9 candidate is exactly `c_hat=b_hat*1+A_hat(S_hat+−S_hat−)`, reusing the v3.8 shared stem/Shape/magnitude branches with one Level scalar, one Local mass, one signed Shape pair and one VAL `alpha_0`. No safety/gate/retrieval/attention/TCN/Transformer and no `src/core` edit. The pilot uses the same six v3.8 cells and seeds 7/17/37, TRAIN/VAL only, 18 fits, with additional read-only amplitude diagnostics.

**Execution ownership changed:** ChatGPT will only prepare/update design and experiment documents. The local AI executes implementation, training and verification. Do not run this pilot from ChatGPT. `EXPERIMENT_LEDGER.md` remains unchanged until a verified local execution returns a scientific result.

## Current handoff — HCH-LGR v3.8 pilot stopped after informative TRAIN/VAL result (2026-09-15)

The isolated LGR implementation is in `src/mvp/hch_lgr_v3_8/`; controlling design/protocol/prompt are `docs/current/HCH_LGR_V3_8_IMPLEMENTATION_DESIGN_20260915.md`, `experiments/current/hch_lgr_v3_8_pilot_20260915/PROTOCOL.md`, and `experiments/current/hch_lgr_v3_8_pilot_20260915/AI_EXECUTION_PROMPT.md`. The pilot ran 18 fits: 6 fixed China-5 cells × seeds 7/17/37, using frozen V2 TRAIN/VAL only and reporting TEST read count 0. No `src/core` edits were intentionally made and QINGHAI was excluded from this pilot.

Results (cell median): GANSU/LSTM `+5.323%` vs Host / `+5.741%` vs M0; GANSU/TimeMixer `0.000%` / `−1.331%`; SHANDONG/TimeMixer `+1.462%` / `+2.094%`; SHANDONG/iTransformer `+0.521%` / `+1.353%`; SHAANXI/TimeMixer `+0.305%` / `−2.276%`; NINGXIA/LSTM `0.000%` / `+1.446%`. Full LGR is non-worse than Host 6/6 and improves M0 4/6, but worst M0 degradation `−2.276%` fails the registered ≤2% gate.

Mechanism: Level coordinate beats its zero reference in 3/6; Gain beats zero reference in **0/6**; local-mass error improves in 6/6. The two problematic cells show non-positive correction/residual alignment; one collapses to `alpha_0=0`. Conclusion: Local remains strongest, Level has partial evidence, current Gain coordinate is unsupported. Stop here. Do not add safety/gate/retrieval/attention, do not expand Gain, do not rerun failed cells, do not open V2 TEST or foreign/final data. A possible future direction is a separately authorized **Level + Local** minimal design, but it is not authorized by this handoff.

Evidence: `experiments/evidence/hch_lgr_v3_8_pilot_20260915/PILOT_REPORT.md`, `RESULTS_MEDIAN.csv`. This was an informative stdout-only feasibility execution without persisted per-seed checkpoints/raw prediction artifacts, so it is not paper-verified evidence and must not be presented as a final paper result. `EXPERIMENT_LEDGER.md` should record this as a development pilot only after the result-entry wording is agreed/verified.

Literature collision is now stricter: Theil-style bias/scale/covariance decomposition and Hodson et al. bias–distribution–sequence decomposition already cover generic forecast-error decomposition and timing/phase error. Do not claim Level/Gain/Local decomposition itself as novel; any future method claim must center on the frozen-Host correction coordinate system and signed-simplex local tail repair, subject to closest-work audit.

## Current action — LGR concrete design frozen; pilot executed and stopped (2026-09-15)

Read `docs/current/HCH_LEVEL_GAIN_REDISTRIBUTION_DESIGN.md` first. The current design is no longer just a three-part idea: it is frozen at the **paper/design level** as `c_hat = b_hat*1 + g_hat*u + A_hat_rho(S_hat_rho+ - S_hat_rho-)`, with `u=yhat_H-mean(yhat_H)1`. Core principle: **sequential target construction, parallel prediction**. TRAIN constructs `b* -> g* -> rho* -> (A_rho*, S_rho+*, S_rho-*)` analytically; inference uses one shared backbone with three conceptual heads so Local/Level/Gain do not form a serial correction chain and cannot propagate stage errors into each other.

Architecture is minimal: keep the existing shared stem, Shape GRU and magnitude-sensitive GRU. Level and Gain are two signed affine scalar readouts from the existing magnitude latent (one 2-output Linear is sufficient); Local reuses the current signed Shape + shared local mass branch, now supervised only on the exact zero-sum Local remainder. Near-flat Host days deterministically mask Gain. No new TCN/Transformer/retrieval/router/gate/market expert is part of the leading design.

Loss is frozen for the first probe as exactly two top-level terms: `L_rec + L_coord`. `L_rec` is TRAIN-scale-normalized full-residual MAE of the synthesized correction. `L_coord` is an equal average of active normalized Level, Gain, local-mass, and signed-Shape-W1 terms. No per-head loss-weight tuning. Retain only one VAL nonnegative scalar `alpha_0` on the complete correction; do not add componentwise calibration or confidence gates.

The user explicitly authorized updates to `paper/**` and `paper/current/**` in this conversation. Manuscript files have been updated so the proposed full LGR is clearly separated from the existing China-5 **Local-only M0 evidence**. Do not state that full LGR already achieved M0's numbers. Figure planning now uses three real error archetypes (Level/Gain/Local) plus residual-energy composition for motivation and a shared-backbone three-head LGR architecture for the method figure. No experiment or `src/core` change has occurred; `EXPERIMENT_LEDGER.md` remains unchanged. Next scientific step, if separately authorized, is a bounded LGR experiment with at minimum `Local only / Level+Local / Gain+Local / Full LGR`; safety is excluded from the main candidate-quality test.

## Current scientific direction — Level + Gain + Local redistribution; diagnose before redesign (2026-09-15)

Read `docs/current/HCH_LEVEL_GAIN_REDISTRIBUTION_DESIGN.md` first for the active discussion direction. The current M0 zero-sum correction is no longer treated as a model of the entire Host residual. Instead, decompose one daily residual exactly as `r=b*1+g*u+rho`, where `u=yhat_H-mean(yhat_H)1`, `b=mean(r)` is whole-day level bias, `g=<r-b1,u>/||u||^2` is Host-aligned daily-curve gain error, and `rho` is the remaining local timing/shape/extreme residual. This cleanly distinguishes: (1) whole-day high/low bias, (2) Host gets direction/timing broadly right but daily excursion/peak-valley magnitude is too weak/strong, and (3) local phase/lag-like or isolated spike/valley error.

The key new theory is that `sum rho=0`, hence the positive and negative masses of `rho` are exactly equal. Therefore the existing local `A(S+−S−)` representation becomes mathematically exact for the **local** target after Level/Gain removal; the shared `A` is no longer just an empirical tied-head survivor. A minimal future LGR model would reuse the existing shared stem + two GRUs and add only two signed scalar outputs (`b`,`g`) from the current scale-sensitive latent; no new encoder, retrieval, learned gate or safety branch is required.

Do not claim the primitive decomposition itself as novel. MOS/EVMOS already cover additive/gain calibration; RevIN/Non-stationary Transformer/Dish-TS cover changing mean/scale; DILATE/functional-data work covers shape/phase; meteorological forecast-error work includes orthogonal positional/structural/residual decompositions; electricity-price work includes spike calibration. The possible novelty is the specific Host-relative, identifiable three-coordinate correction space for frozen day-ahead EPF plus signed-simplex local tail repair. A later closest-work audit is still required before any `first` claim.

A read-only China-5 diagnostic has now been completed on frozen V2 Host prediction/actual arrays. Do not use “electricity-price forecasts are generally lagged” as the top-level paper/problem claim: best centered-curve shift is non-zero on average 41.5% of VAL and 43.4% of TEST days, with median shift zero in most cells; high-error days show more temporal misalignment (about 61.5% VAL / 54.6% TEST non-zero). Lag/phase is therefore one Local failure mode, not the universal explanation.

The three-coordinate residual view is supported much more strongly: per-day L2 energy shares average `29.3% Level / 15.7% Gain / 55.0% Local` on VAL and `29.6% / 17.1% / 53.3%` on TEST. Market-level TEST shares remain non-degenerate across all five markets. Current preferred framing is **structured frozen-Host residual correction**: whole-day Level bias, Host-aligned excursion Gain error, and Local timing/shape/extreme redistribution. The existing `A(S+−S−)` should represent the Local zero-sum remainder only. No neural redesign or `src/core` modification is authorized yet; next discussion is to freeze the smallest coherent architecture/ablation around these three coordinates. Safety remains supplementary. `paper/**` remains locked unless the user explicitly authorizes paper edits.

## Paper workspace governance handoff — stable 01–07 taxonomy; explicit authorization required for all paper writes (2026-09-15)

`paper/` has been normalized to a single stable top-level taxonomy: `_archive/`, `01_framework/`, `02_literature/`, `03_method_math/`, `04_experiments/`, `05_data/`, `06_figures_tables/`, `07_reproducibility/`, `current/`, `README.md`. `current/` now contains only README plus the six manuscript sections. Important moved support paths: claim/evidence -> `paper/01_framework/claims_evidence/`; writing rules -> `paper/01_framework/writing_guidelines/`; math -> `paper/03_method_math/`; literature -> `paper/02_literature/`; obsolete August experiment-prep tree -> `paper/_archive/03_experiments_legacy_20260823/`; former current compatibility files -> `paper/_archive/current_compat_20260914/`.

**Do not proactively edit `paper/**`.** The entire paper workspace is default READ-ONLY and requires explicit user authorization in the current conversation for any create/edit/move/delete operation. `paper/current/**` is stricter: new experiments, literature, or scientific conclusions must not be auto-synchronized into manuscript files. Canonical research state continues to update independently in `RESEARCH_STATE.md`, `EXPERIMENT_LEDGER.md`, and `HANDOFF.md`. When a user asks a scientific question but does not explicitly authorize paper work, leave paper untouched and only recommend possible manuscript changes in chat.

## Paper handoff — manuscript essence now lives in six concise `paper/current/` sections (2026-09-15)

For any new manuscript-writing window, read `paper/current/README.md` after the canonical research-state files. The only manuscript-facing sequence is now: `01_INTRODUCTION.md -> 02_RELATED_WORK.md -> 03_METHODOLOGY.md -> 04_EXPERIMENTS.md -> 05_DISCUSSION.md -> 06_CONCLUSION.md`. These files are intentionally concise and follow a PIR-like flow: problem/motivation first, then closest work, Problem Definition before method, then experimental setup/main results/analysis. Do not use old `*FOUNDATION*` files as section sources; they are now compatibility pointers only.

Detailed evidence lives in the numbered support directories: framework/contributions in `paper/01_framework/CURRENT_PAPER_THESIS_AND_FRAMEWORK_20260915.md`; claim/evidence in `paper/01_framework/claims_evidence/CURRENT_CLAIM_EVIDENCE_MATRIX_20260915.md`; math in `paper/03_method_math/CURRENT_SIGNED_REDISTRIBUTION_DERIVATION_20260915.md`; full 20-cell and prequential experiment tables/training details in `paper/04_experiments/CURRENT_EXPERIMENT_DATA_BOOK_20260915.md`; closest-work notes in `paper/02_literature/CURRENT_RELATED_WORK_NOTES_20260915.md`; data/provenance in `paper/05_data/CURRENT_DATASET_SPLIT_AND_PROVENANCE_20260915.md`; figure/table planning in `paper/06_figures_tables/CURRENT_FIGURE_TABLE_BLUEPRINT_20260915.md`; reproducibility/open items in `paper/07_reproducibility/CURRENT_REPRODUCIBILITY_AND_SUBMISSION_CHECKLIST_20260915.md`. Former current compatibility/foundation files are archived under `paper/_archive/current_compat_20260914/`. `paper/README.md` documents the locked two-layer organization.

Scientific paper content is unchanged: M0 remains primary; M1 supplementary; M2 deleted; current claim boundary is difficult/tail repair rather than broad Overall-SOTA; fresh external confirmation remains pending. This reorganization created no new experiment result and therefore does not update `EXPERIMENT_LEDGER.md`.


## 论文 current foundation 已中文化并扩充实验数据表（2026-09-14）

以后论文写作从 `paper/README.md -> paper/current/README.md -> 00_MASTER_PAPER_BLUEPRINT.md -> 07_CLAIM_EVIDENCE_MATRIX.md` 进入。`paper/current/` 现已统一为中文学术底稿，英文方法名/协议 ID/公式保留。当前主方法仍是 M0 `c=A(S+−S−)` + VAL `alpha_0`；M1 仅 supplementary stabilizer，M2 删除；RRC/LCR 不属于当前 Proposed Method。

`paper/current/04_EXPERIMENTS_FOUNDATION.md` 已成为实验 section 的主数据 authority：含 China-5 split、Host/baseline taxonomy、完整训练流程、20-cell offline MAE+gain 表、20-cell COSA/M1 prequential 表、Tail/Upper/Lower/Negative/Extreme win-count、M0/M1/M2 safety、old low-shot stress、component screen 和 PIR-inspired probe。后续真正写正文时应优先从该文件和其指向的 frozen evidence 自动生成 LaTeX 表，不从聊天记录抄数字。


## Paper-writing handoff — `paper/current/` is now the only current manuscript foundation (2026-09-14)

For any new paper-writing window, read `RESEARCH_STATE.md`, `EXPERIMENT_LEDGER.md`, `HANDOFF.md`, then start at `paper/current/README.md`. The paper workspace has been rigorously restructured without deleting historical assets. Old `_drafts/HCH_FORECAST_REPAIR_PAPER_DRAFT_20260909.md`, `02_claims_evidence/CURRENT_FORECAST_REPAIR_CLAIM_LEDGER_20260909.md`, and `04_literature/CURRENT_CLOSEST_WORKS_POSITIONING_20260909.md` are historical/provenance inputs only and must not be used as current method/result authority.

Current paper-primary method is M0 signed residual redistribution, `c=A(S+−S−)` with VAL-fitted nonnegative `alpha_0`; M1 is supplementary prequential shrinkage only; M2 is deleted. Current thesis is **difficult/tail residual repair for frozen day-ahead electricity-price Hosts**, not universal Overall-MAE SOTA. The V2 evidence boundary must remain explicit: +1.656% median Overall gain, 17/20 positive, 19/20 non-worse; 18/20 Overall wins vs delta-Adapter but only 3/10 vs PIR; Tail wins 15/20 vs delta and 7/10 vs PIR. QINGHAI is not excluded; Shandong is private/internal; V2 carries the historical-exposure caveat.

Use `paper/current/07_CLAIM_EVIDENCE_MATRIX.md` before drafting any claim. Use the corresponding section foundation for Introduction/Related/Method/Experiments/Discussion/Conclusion. `paper/current/09_ICDE2027_WRITING_CONSTRAINTS.md` records venue rules; `10_FIGURE_TABLE_PLAN.md` records visual/evidence sources; `11_OPEN_ITEMS_BEFORE_MANUSCRIPT_FREEZE.md` is the single remaining-work checklist. RRC/LCR remain outside the paper method pending their bounded fidelity replay, and even a passing replay cannot retroactively replace M0 in V2 TEST claims without fresh external evaluation.

No new scientific result was produced by this reorganization; do not add a new `EXPERIMENT_LEDGER.md` experiment entry for it.


## Current action — PIR pilot post-execution audit: RLR closed; RRC/LCR need one fidelity replay before verdict (2026-09-14)

Read `docs/current/HCH_PIR_MINIMAL_PROBE_POSTEXECUTION_AUDIT_20260914.md` first. The executor token `HCH_PIR_MINIMAL_PROBE_VERIFIED` remains valid for evidence integrity (13/13, TEST read count 0, unchanged core/Host/baseline artifacts, 24 fits), but its scientific `NOT_SUPPORTED` conclusion is only final for RLR.

Audit defects not caught by the verifier: RRC was implemented with raw retrieved residual appended after canonicalization even though the frozen protocol requires `tilde_r / TRAIN residual scale`; RRC/LCR instantiate the model before `torch.manual_seed(seed)`, so seeds 7/17/37 do not control initialization; registered retrieval Spearman and signed-Shape-W1 diagnostics are missing. RLR was also implicitly evaluated with the last M0 seed although labelled deterministic/no-seed.

RLR was corrected by a read-only three-seed replay using existing frozen M0 streams. Relative Overall changes are GANSU/TimeMixer `+0.739%`, SHAANXI/PatchTST `-0.034%`, NINGXIA/TimeMixer `-6.053%`, SHANDONG/PatchTST control `-0.109%`; it still fails clearly and is **closed**. Retrieval diagnostics show why: top-10 Host-profile cosine is high (`0.903..0.965`) while retrieved-level correlation is only `-0.437..+0.209` and residual-trajectory correlation `-0.011..+0.277`. Naive Host-shape KNN is not residual-relevant enough to explain PIR.

RRC/LCR verdicts are suspended. LCR shows conditional signal (`-1.755/+2.599/+0.755%` challenge, `-4.982%` control), so level freedom may matter in selected cells but is not transferable as executed. The only legal next execution is one bounded implementation-fidelity replay of RRC/LCR with seed-before-init, registered RRC scaling, and missing diagnostics restored; same four cells/data identities/K/seeds/training recipe/support gates. No V2 TEST, foreign/final data, RLR rerun, combination, gate, retrieval sweep or core edit. If neither passes, close the PIR-inspired minimal-adaptation route and return to frozen M0 for fresh external tail-repair evaluation.

## Previous executor state — PIR-inspired minimal pilot VERIFIED; all three directions rejected by support gate (2026-09-14)

The fixed TRAIN+VAL pilot is complete with independent token **`HCH_PIR_MINIMAL_PROBE_VERIFIED`** (13/13). TEST remained sealed/read count 0; no Host/baseline retraining, no `src/core` edits, and exactly 24 new CUDA fits (RRC 12 + LCR 12) were executed. RLR was deterministic, `K=10`, TRAIN-only.

All variants are `NOT_SUPPORTED_IN_MINIMAL_PILOT`: RLR challenge median Overall `−0.034%`; RRC `−1.186%` with Tail `−7.817%`; LCR `+0.755%` challenge median but `−4.982%` control harm. Do not combine, rescue, promote, or open foreign/final data. Evidence: `experiments/evidence/hch_pir_inspired_minimal_probe_20260914/`.

## Current action — PIR-inspired minimal candidate-quality probe; TRAIN+VAL only, no V2 TEST (2026-09-14)

Read `docs/current/HCH_PIR_MECHANISM_AUDIT_AND_MINIMAL_ADAPTATION_20260914.md` and `experiments/current/hch_pir_inspired_minimal_probe_20260914/PROTOCOL.md`. The V2 evidence says the current bottleneck is candidate quality, not safety: M0 beats δ 18/20 Overall cells but PIR only 3/10 legal overlaps, while M0 remains strong in the intended Tail region. PIR's three useful ideas are failure identification, local contextual revision and global retrieval; only the last two expose information/representation gaps relevant to us. Do **not** reintroduce a learned uncertainty/gate: that collides with the rejected Proposal Verification route.

The new mathematical diagnosis is the current zero-sum invariant: `c=A(S+−S−)` forces `sum_h c_h=0`, so repaired daily mean residual is exactly the Host daily mean residual. This creates an irreducible daily-level MAE floor that PIR is free to correct. The pilot therefore tests exactly three isolated minimal directions outside `src/core`: RLR = deterministic K=10 TRAIN-only residual-level retrieval added to frozen M0; RRC = one retrieved residual reference channel through the existing stem; LCR = one signed level scalar plus Shape/Amplitude on median-centered residuals. No Transformer/router/gate/K sweep/combined variant.

Fixed pilot cells: `GANSU_DA/TimeMixer`, `SHAANXI_DA/PatchTST`, `NINGXIA_DA/TimeMixer`, and `SHANDONG_DA/PatchTST` as non-destruction control. Use V2 TRAIN unchanged and split VAL chronologically into PILOT_CAL/PILOT_EVAL. **TEST read count must stay 0.** Reference M0 reuses existing TRAIN-fitted checkpoints; RRC and LCR each run 12 fits (4 cells × seeds 7/17/37) with the unchanged V2 training recipe; RLR is deterministic. Direct execution prompt: `experiments/current/hch_pir_inspired_minimal_probe_20260914/AI_EXECUTION_PROMPT.md`. Stop after the pilot verdict; do not combine variants or open foreign/final data automatically.

## Current scientific handoff — independent review: tail-repair route is stronger; no safety rescue (2026-09-13)

Read `docs/current/HCH_V2_INDEPENDENT_RESULT_REVIEW_AND_NEXT_DECISION_20260913.md` first. The V2 result is accepted as the standardized development/common benchmark subject to the existing historical-exposure caveat. Independent post-run checks confirm the frozen core digest and `136 passed`; direct recomputation from final exports confirms M0 median Overall gain `+1.656%`, 17/20 positive, M0 > delta 18/20, but M0 > PIR only 3/10 legal overlaps and M0 > per-cell best(delta,PIR) only 13/20. Therefore do not claim broad Overall-MAE superiority.

The stronger evidence is in the intended difficult region: M0 beats PIR on Tail 7/10, Lower-tail 7/10 and Negative-price 2/2 defined overlaps; it beats delta on Tail 15/20, Lower-tail 16/20, Negative-price 3/4 and Extreme-day 10/12 defined cells. This supports a narrower **extreme/tail residual repair** paper framing. QINGHAI is positive 4/4 under V2 and must not be dropped post hoc for table strength.

Safety is not the next rescue lever. M2 remains deleted as non-binding. M1 is only an optional prequential stabilizer: it changes the panel median essentially not at all, improves 6 cells/worsens 10/ties 4, reduces the worst Overall loss `-0.652% -> -0.054%`, and lowers median Tail gain `+7.276% -> +4.429%`. Do not retune safety on V2.

Decision fork: (A) if the paper requires broad Overall superiority over PIR, do not open a foreign final matrix yet; a new candidate-quality idea and a fresh evaluation surface are required. (B) if the paper targets frozen-Host extreme/tail repair, freeze M0 as primary and M1 as setting-separated supplementary stabilizer, then proceed to fresh foreign/public markets with Tail/Upper/Lower/negative/extreme metrics preregistered as primary and Overall-MAE as a guardrail.

## Current scientific handoff — V2 verified; keep M0/M1, delete Shape-weighted M2, do not tune V2 (2026-09-13)

Read `docs/current/HCH_V2_RESULT_ADJUDICATION_20260913.md` first. The final verified V2 result is `CHINA5_COMMON_BENCHMARK_FULL_V2_JOINT_TEST_VERIFIED_WITH_MISSING_TARGET_ADJUDICATION` (12/12). Under the full-eligible standard 70/10/20 benchmark, M0 median Overall-MAE gain is **+1.656%** (17/20 positive, 19/20 non-worse, worst `−0.652%`); M1 median is **+1.652%** and shrinks the worst cell to `−0.054%`, but is not a panel-wide accuracy booster (6 cells improve vs M0, 10 worsen, 4 tie). M2 is numerically identical to M1 in all 20 cell medians and every recorded cell×seed final-lambda summary, so Shape-weighted safety has no supported functional contribution and is deleted from the leading design. Do not retune or rescue `lambda_S` on the same V2 TEST.

The old QINGHAI 4/4 negative-transfer story is superseded as a general market claim: M0 is positive on all four QINGHAI Hosts in V2 (`+0.021/+0.580/+3.468/+2.625%`). The old failure remains useful only as historical motivation for correction stability under low-shot/disjoint deployment; it is not evidence that QINGHAI is intrinsically unlearnable.

Offline: M0 beats delta-Adapter in 18/20 but PIR in only 3/10 legal overlaps; PIR median gain is +5.475% on its legal PatchTST/TimeMixer cells, so no broad PIR-superiority claim. Prequential: COSA is positive on 20/20 with median ~+0.815%; M1 beats COSA in 15/20. Tail gain remains strong (M0 median ~+7.276%, M1 ~+4.429%), but max normal-region relative harm remains ~+18%, so M1 is only an **overall-MAE causal shrinkage stabilizer**, not general no-harm safety.

Leading structure after adjudication: offline main variant M0 = minimal `c=A(S+−S−)` + VAL `alpha_0`; optional prequential variant M1 = `Host + min(alpha_0,lambda_U)c`. M2 / Shape-weighted safety is deleted. No further V2 tuning is authorized. Next legitimate work is paper positioning plus genuinely fresh temporal/external confirmation if needed.

## Current action — V2 missing-target adjudication continuation VERIFIED (2026-09-13)

The authorized continuation is complete. It did not retrain, refit, replay inference, recompute splits, impute targets, or replace a day. The independent verifier returned **`CHINA5_COMMON_BENCHMARK_FULL_V2_JOINT_TEST_VERIFIED_WITH_MISSING_TARGET_ADJUDICATION`** (12/12). Metrics were recomputed from frozen Phase-B raw artifacts using one common finite-target day mask per market.

Final export: 270 numeric rows (20 Host + 70 numeric baselines + 180 M0/M1/M2 seed rows), 60 candidate cell medians, and 50 inherited blockers. Shandong remains `n_registered_test_days=330`, `n_scored_test_days=329`, `n_unscorable_missing_target_days=1`; `2026-07-17` is explicitly retained as `NOT_SCORABLE_MISSING_TARGET / TARGET_VECTOR_NONFINITE_24_OF_24`. All numeric rows carry `data_adjudication=V2_MISSING_TARGET_SCORABILITY_ADJUDICATION_20260913`.

Evidence: `experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913/10_missing_target_adjudication/`; lab: `experiments/lab/common_benchmark_results_v2/`. Historical V1/strict labs and all raw/checkpoint/stream identities are unchanged. Keep the historical-exposure caveat; do not treat this as untouched final confirmation or open a rescue/design route.

## Current action — V2 missing-target adjudication resolved; metric-only resume is the next legal step (2026-09-13)

Read `docs/current/HCH_V2_MISSING_TARGET_ADJUDICATION_20260913.md` first. Full-source audit shows one and only one non-finite V2 target day: `SHANDONG_DA / 2026-07-17`, the final TEST day, with all 24 `日前电价` targets missing. Every other China-5 V2 TRAIN/VAL/TEST day is finite.

Do **not** recompute the 70/10/20 split and do not retrain anything. Keep Shandong TEST membership at 330 registered days, but score only the 329 days whose full 24h target vector is finite. Retain `2026-07-17` explicitly in raw provenance as `NOT_SCORABLE_MISSING_TARGET`; do not impute it, delete it silently, replace it with another day, or let different methods use different masks. All target-dependent metrics/subsets for every Shandong method use the same 329-day mask; other markets exclude zero days.

No prequential replay is needed because the missing day is last: no later COSA/M1/M2 forecast can depend on an unavailable reveal. Preserve all Host/baseline/candidate checkpoints and raw predictions. The next legal execution is **metric/verifier only** using `experiments/current/hch_china5_common_benchmark_701020_full_v2/AI_MISSING_TARGET_ADJUDICATION_CONTINUATION_PROMPT.md`. It may patch only metric aggregation, V2 lab export and independent verification, then recompute from existing raw Phase-B artifacts. It must prove `NO_RETRAIN_NO_REINFERENCE`, a common Shandong mask, and exact split/checkpoint/prediction hash preservation before repopulating the V2 lab.

Any verified final row must carry `data_adjudication = V2_MISSING_TARGET_SCORABILITY_ADJUDICATION_20260913` and registered/scored/unscorable day counts. The V2 historical-exposure caveat remains; this remains a standardized development/common benchmark, not untouched final confirmation.

## Previous stop state — V2 JOINT_TEST stopped for adjudication on missing SHANDONG targets (2026-09-13)

Evidence root: `experiments/evidence/hch_china5_common_benchmark_701020_full_v2_20260913/`. Phase A independently passed **14/14** with `CHINA5_COMMON_BENCHMARK_FULL_V2_PRETEST_VERIFIED`. TEST was then opened once under the registered authorization. The run produced 20 Host TEST caches, 70 numeric baseline artifacts, 50 inherited blockers, and 60 candidate streams / 180 M0-M1-M2 seed rows without changing frozen methods, seeds, losses, or W=7.

The independent final verifier returns **`CHINA5_COMMON_BENCHMARK_FULL_V2_NOT_VERIFIED_STOP_FOR_ADJUDICATION`** (7/8 checks passed). It detected a data-contract failure: `SHANDONG_DA` TEST day `2026-07-17` has 24 missing `日前电价` target values in the source. All four Shandong Host TEST arrays therefore contain non-finite targets and their headline metrics are undefined. This is not a method result. No imputation, deletion, proxy or result-conditioned repair was made.

The V2 lab was reset to its Phase-B-opened-but-unverified scaffold (zero paper-facing numeric rows); old labs remain hash-unchanged. Partial TEST evidence is retained for audit only and must not be ranked or cited. A new continuation requires human adjudication and fresh authorization for a target-blind finite-target eligibility policy before any TEST rerun.

## Current action — FULL-eligible 70/10/20 V2, then one internally gated complete China-5 run (2026-09-13)

Read first: `docs/current/HCH_COMMON_BENCHMARK_701020_FULL_V2_PROTOCOL_20260913.md`, then `experiments/current/hch_china5_common_benchmark_701020_full_v2/PROTOCOL.md`, then `AI_FULL_EXECUTION_PROMPT.md`. The hard scientific standard remains generic/custom chronological **70/10/20**, with official fixed splits retained for datasets such as ETT (~60/20/20). The critical correction is denominator: 70/10/20 must be applied to the **complete eligible target-day sequence**, not merely the old protocol's already-open ~70% block.

The completed `COMMON_BENCHMARK_701020_V1` PRETEST is superseded before TEST because it divided only that old open block. It produced zero TEST metrics and must never be continued to JOINT_TEST. V2 exact China-5 counts are GANSU `416 -> 291/42/83`, SHANDONG `1652 -> 1156/166/330`, SHAANXI `406 -> 284/41/81`, NINGXIA `144 -> 100/16/28`, QINGHAI `135 -> 94/14/27` (full N -> TRAIN/VAL/TEST). A mismatch is a stop condition.

The user has authorized one complete internally gated execution. Phase A keeps V2 TEST sealed, retrains and persists **loadable** checkpoints for all 20 Hosts, freezes baseline objects/replay manifests, creates a TRAIN/VAL/TEST-native method harness against current `src/core`, trains exactly 60 candidate generators (`20 cells × seeds 7/17/37`), fits VAL `alpha_0` and legal seven-record safety warm starts, then requires `CHINA5_COMMON_BENCHMARK_FULL_V2_PRETEST_VERIFIED`. On that exact PASS the same task may automatically open V2 TEST and run all baselines plus M0/M1/M2, finishing with an independent final verifier. No further human authorization is needed between those two phases.

Result surfaces are isolated. `experiments/lab/strict_results/` is lifecycle-archived in place as historical `HARD_FROZEN_LOW_SHOT_TRANSFER`; `experiments/lab/common_benchmark_results/` is lifecycle-archived in place as superseded V1 PRETEST with zero TEST numeric rows; the only active V2 main lab is `experiments/lab/common_benchmark_results_v2/`. Do not physically move old evidence merely to archive it because existing hashes/pointers depend on paths. No old numeric row may enter V2.

V2 TEST carries `COMMON_BENCHMARK_TEST_WITH_HISTORICAL_EXPOSURE_CAVEAT`: historical diagnostics previously materialized some target values now in the complete-series chronological tail. V2 is therefore the standardized like-for-like development/common benchmark, not a claim of untouched final confirmation. A future final confirmation requires a fresh temporal extension or independent external benchmark after method freeze.

Performance guidance is frozen in the execution prompt: the machine has 32 logical CPUs and an RTX 4060 Laptop 8 GB, but the current project Python reports CPU-only PyTorch. Use process-level cell parallelism with one math thread per worker, deterministic read-only cache reuse, one candidate training per cell×seed shared by M0/M1/M2, and one candidate TEST stream per seed. CUDA for the proposed method is optional only if a CUDA runtime is established/tested/frozen before TEST; never reduce seeds/epochs/data or alter numerical semantics merely for speed.

## STOP adding markets: baseline split/data-budget adjudication is now the active bottleneck (2026-09-12)

Read first: `docs/current/BASELINE_SPLIT_PROTOCOL_LITERATURE_AUDIT_20260912.md`. The current strict matrices remain valid but are now explicitly classified as **`HARD_FROZEN_LOW_SHOT_TRANSFER`**, not as neutral paper-native baseline evaluations. The controlling literature/code audit found that δ-Adapter and PIR normally learn their revision modules from the ordinary benchmark training split (~60–70%), COSA adapts over the full held-out test stream, UEC-STD uses 70% backbone training + 10% validation-derived correction support + 20% test, and mainstream EPF uses long out-of-sample periods plus rolling/daily recalibration. Our 45/5/15/5/30 protocol therefore confounds electricity-market transfer with data starvation / short evaluation windows, especially in short domestic markets (POST_TRAIN as low as 20–22 days; DEV_EVAL as low as 6–7 days).

Do **not** expand to additional markets under the current split and do not reinterpret existing negative PIR/δ results as intrinsic method failure. Existing evidence stays immutable. The next scientific step is a small preregistered split/data-budget adjudication with three clearly separated settings: existing hard low-shot transfer (reuse only), common literature-standard benchmark split, and rolling/expanding EPF deployment. No proposed HCH/Signed-Mass method should be evaluated until this confound is resolved.

## GEFCOM14P × LAGO_NP matrix FROZEN — read the stage handoff first, then the registry (2026-09-12)

The third admitted strict baseline stage is closed at **`GEFCOM_NP_BASELINE_MATRIX_FROZEN_WITH_INHERITED_BLOCKERS`**. Evidence root: `experiments/evidence/hch_international_gefcom_np_baseline_matrix_20260912/`. Read, in order: `04_audit/NEW_WINDOW_HANDOFF.md`, `04_audit/BASELINE_MATRIX_SUMMARY.md`, `04_audit/FINAL_AUDIT.md`, `04_audit/RESULT_FREEZE.json`.

What is settled and must not be re-derived. **56/56 coordinates = 36 numeric + 20 inherited blockers** (18/10 per market), row-for-row identical to `MATRIX_TEMPLATE_56.csv`. Both markets `ADMITTED`; source hashes `BA4582BD…D9ED27` (GEFCOM14P) and `A0777631…D0FDEE7` (LAGO_NP). Timezone is `BENCHMARK_LOCAL / SOURCE_UNSPECIFIED` for **both** markets — do not assume CET/UTC and do not introduce a DST normalisation, because neither source's lineage proves a deterministic, target-blind one. GEFCOM14P's irregular block (one duplicated hour label and one absent hour label on `2013-03-10`) was handled by **timestamp-only exclusion**: 8 days dropped (1 target + 7 context), frozen before any outcome was opened, and **all 8 lie inside `PROTECTED_FINAL`, so no scored row was removed**. The two duplicate rows carry **different** prices (48.85 / 43.50); **do not "repair" this from price values**. Splits, thresholds, Host configs, seeds and thread pins are hash-pinned, and the final verifier re-derives every threshold from `HOST_TRAIN` alone. Host gates **8/8 PASS**; `PROTECTED_FINAL` read count is **0** and access is **structural** (unauthorised rows are never parsed), not merely asserted. Both verifiers pass: `GEFCOM_NP_PREEXECUTION_CONTRACT_VERIFIED` (14/14, before any Host was trained) and `GEFCOM_NP_BASELINE_MATRIX_VERIFIED` (45/45).

What a new window may **not** do. Do not reopen UEC-STD (`Q-FIDELITY`) or OMPB (`Q-DATA`), or substitute a proxy for either. Do not re-probe PIR on iTransformer/LSTM or manufacture a backbone config for them — the verified-absent configuration *is* the recorded result. Do not tune any method because its result here looks bad (MDR loses to its own Host in 8/8 cells; that is the frozen outcome and it is recorded as-is). Do not search architecture / seed / LR / epoch / hidden size per market, or tune on `DEV_EVAL`. Do not feed GEFCOM14P `load_system_fc`/`load_zonal_fc` or LAGO_NP `Grid load forecast`/`Wind power forecast` into a Host — they are registered `LEGAL_FORECAST_KNOWN_UNUSED_BY_BASELINE_HOST` for **future method research**, not as a retrofit. Do not rank COSA against offline methods or pool the two settings in any table, figure or claim. Do not read `PROTECTED_FINAL` for any purpose.

Two inherited limits carried forward as **disclosure, never patched**. (1) Both markets' `DEV_EVAL` has `n_negative_hours = 0`, so the negative-price subset is `NOT_APPLICABLE_N0` on all 36 numeric rows — **no negative-price claim can be made from this stage in either direction**; a future claim needs a market and window that actually contains negative hours. (2) LAGO_NP has `n_lower_tail_hours = 0`, so `Lower_tail_MAE` is undefined in all 18 of its rows and the frozen parent contract emits no status field for that subset, leaving those cells empty rather than the literal `NOT_APPLICABLE_N0`; GEFCOM14P has only 2 lower-tail hours. Patching either would mean editing a frozen parent to change how a metric is reported.

Post-freeze human review supersedes one descriptive sentence in the frozen audit; read `docs/current/GEFCOM_NP_POSTFREEZE_REVIEW_20260912.md`. `Normal_harm_vs_Host` is an **absolute price-unit difference**, not a percentage, so the earlier `−0.24 … +4.18` range cannot be compared directly with percentage Overall gains and does **not** establish panel-wide tail-concentrated harm. Like-for-like recomputation gives normal-region gains from `−77.14%` to `+6.91%` and tail gains from `−68.75%` to `+19.82%`; damage location is method/Host dependent. Two additional audit-prose sign slips are corrected only interpretively, not by editing frozen evidence: PIR's four legal gains are negative (`−39.89/−23.63/−2.77/−20.15%` in the relevant cells), and COSA's LAGO_NP/LSTM gain is **`+5.60%`**, not `−5.60%`. The CSVs/raw arrays/verifiers/registry are correct; only the prose was wrong.

The Strict Results Laboratory has been updated. Registry census is now **282 coordinates = 182 active numeric + 90 blocked + 10 superseded/historical**, `RESULTS_LONG.csv` = **1652** rows (1480 numeric, 172 `N0`), 480 evidence pointers with 0 missing, 11 markets, 40 market×Host cells. `registry_verify.py` returns **16 checks / 0 faults → `STRICT_RESULTS_REGISTRY_VERIFIED`**. The new `implementation/registry_verify_mutation_test.py` injects 26 registry defects and the verifier catches **26/26**, restoring the registry to a re-verified state (result at `audit/MUTATION_TEST_RESULT.json`); it caught a `paper_use_scope` narrowing the first verifier pass had missed, and the verifier was extended to close that gap **before** the snapshot was taken. Current snapshot: `snapshots/STRICT_RESULTS_SNAPSHOT_20260912T131712Z.json`; the predecessor `…20260912T105648Z.json` is retained and must never be overwritten. GEFCOM14P now carries the data-handling tag `GEFCOM_IRREGULAR_BLOCK_EXCLUDED_BEFORE_OUTCOME` **without narrowing its paper scope or Shape/WHERE eligibility**; LAGO_NP and NORD_DK1 carry `NONE_DECLARED`.

First commands for a new window:

```powershell
cd D:\作业\science\solar_leak_price_model
python experiments\lab\strict_results\implementation\registry_verify.py          # STRICT_RESULTS_REGISTRY_VERIFIED
python experiments\current\hch_international_gefcom_np_baseline_matrix\implementation\verify_final.py
```

`implementation/i_freeze_results.py` re-derives `04_audit/RESULT_FREEZE.json` from the written
artifacts, but note what it does **not** do: it reads the terminal token from the stored
`04_audit/FINAL_VERIFIER_RESULT.json` rather than re-running the verifier, and it does **not**
refuse to overwrite an existing freeze file. It aborts on the `…_INVALID` token and on an
unrecognised one, and nothing more. Re-run `verify_final.py` yourself before trusting a freeze.

Extending the matrix (new markets, new Hosts, new baselines) or designing a proposed method is allowed only as a **new stage** under its own `PROTOCOL.md` with its own pre-execution contract verifier, and a method stage must state up front how it will be compared against this frozen surface. Nothing in this stage authorises a rescue variant of any method.

## Alignment-safe core history patch done — the core is design-ready, not experiment-authorized (2026-09-12)

The bounded patch `docs/current/HCH_ALIGNMENT_SAFE_CORE_HISTORY_PATCH_PROMPT_20260912.md` is **executed**. It closed the one blocker the independent review left open, and it changed exactly two files: `src/core/history.py` and `experiments/foundation/tests/test_core_safety.py`. Terminal token for the core is now **`ALIGNMENT_SAFE_CORE_READY_FOR_M0_M1_M2_EXPERIMENT_DESIGN`** — the core may be *designed against*. It is **not** authorization to run M0/M1/M2, and the closed `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION` verdict is neither revisited nor rescued.

**What the blocker was.** The online path was strict, but warm start and restore were weaker doors: `from_warm_start()` and `from_state()` could accept a completed record whose `revealed_at` predated its own delivery ordinal, and insertion stored the caller's tensor *references* rather than owned snapshots. Both let evidence be scored that no honest pipeline could have produced.

**What is now true, and must stay true.** One validator judges every record at every door — live append/attach, warm start, restore — so no path is weaker than another; malformed declared-legal evidence raises instead of being quietly dropped or repaired. Completed records need `revealed_at` present and satisfying `candidate_created_at < ordinal <= revealed_at`; pending records must carry neither residual nor reveal time. Warm start additionally requires unique delivery IDs and strictly increasing ordinals; restore additionally requires the seen-ID ledger to cover every active record, strictly chronological active ordinals, and at most seven completed records. Tensors entering the bank are `detach().clone().cpu()` snapshots and every public accessor returns copies, so in-place mutation of a supplied or retrieved tensor cannot rewrite scored evidence, and no autograd graph is retained. `W = 7` remains a structural constant and a short bank is still a blocker, never a shortened window.

**Audit surface for the next window.** (1) `git diff 5a8983c -- src/core/` must show only `history.py`; `safety.py` safe-ray mathematics, the candidate architecture, losses, calibration and mask semantics are byte-unchanged. (2) Active-core subset must be `136 passed`; full foundation suite `10 failed, 160 passed` with the identical ten pre-existing legacy-manifest/evidence failures and nothing new. (3) The archived 20-file core must still re-hash to `0c53faab…bb99` and both frozen evidence roots to `3de082de…8f57` / `7b1f5c04…dcb1`. (4) The sixteen new adversarial tests are non-vacuous: fifteen of them fail against the pre-patch `history.py`, so do not weaken or delete them to make a future change pass. (5) Two things were deliberately **not** done and still need a decision: the variable intra-day mask genericity note is out of scope and untouched (`safety.py` still intersects a multi-record `valid_mask` with `np.all`), and `experiments/current/hch_signed_mass_method_entry/core_bridge.py` still refuses to import the new core — correct behaviour for a closed experiment, not a defect to shim.

## Alignment-safe core audit — one bounded history-contract patch remains before experiment design (2026-09-12)

*Superseded as an action list by the patch note above: the bounded history-contract patch it asks for has since been executed and independently re-verified. The audit findings below are retained unchanged and were accurate when written.*

Read `docs/current/HCH_ALIGNMENT_SAFE_CORE_REVIEW_20260912.md` before trusting the refactor token. Independent review confirms the difficult parts: archive digest matches exactly; active-core subset re-runs **120/120 PASS**; the new minimal candidate is forward-equivalent to the archived closed `FROZEN_SMALLEST` after deleting dead zero-input channels / dead switches (max correction diff `1.57e-09` under direct parameter mapping); and a 1000-random-ray independent MAE oracle finds 0 safe-radius mismatches (worst `5.77e-15`). The design/math does **not** need to be redone.

Do not authorize M0/M1/M2 yet. The only blocker is `src/core/history.py`: `from_warm_start()` / `from_state()` do not re-enforce the full reveal chronology that live `attach_residual()` enforces, and stored Shape/correction/residual tensors are caller-owned aliases rather than immutable evidence snapshots. An adversarial record with legal OOF provenance but `revealed_at=0` for ordinals 10..16 is currently accepted as a ready warm-start bank; mutating an input correction tensor after insertion mutates the bank. Execute only the bounded patch in `docs/current/HCH_ALIGNMENT_SAFE_CORE_HISTORY_PATCH_PROMPT_20260912.md`, then re-audit. Do not touch safe-ray math, W=7, candidate architecture, masks, frozen evidence or protected/final data.

## Alignment-safe core refactor done — handoff is now an audit, not an implementation (2026-09-12)

The source refactor `docs/current/HCH_ALIGNMENT_SAFE_CORE_IMPLEMENTATION_PROMPT_20260912.md` has been executed and stops at source readiness with token `ALIGNMENT_SAFE_CORE_READY_FOR_EXPERIMENT_DESIGN_REVIEW`. That token is **not** authorization for the domestic experiment. No `DEV_EVAL` run was opened, `PROTECTED_FINAL` was read zero times, and the closed signed-mass verdict, frozen evidence digests and paper claims are untouched.

Audit surface, in order. (1) **Archive integrity**: `src/archive/signed_mass_development_core_20260912/ARCHIVE_HASHES.json` records 20 files and tree digest `0c53faab6457ce489cf697543ef498a93e15c99e1098b9d98ffeb2b1e828bb99`; the archive must never be rewritten. (2) **Active tree**: `src/core/` is exactly 15 files — `safety.py` and `history.py` are new; `sampling.py`, `semantic_views.py`, `similarity.py` and four reorganization artifacts are archived and gone, so the de-promotion is verifiable by file presence alone. (3) **W=7 legality**: `HISTORY_DAYS` is a module constant, the bank constructor takes no capacity argument, warm start takes no length argument and refuses below seven rather than shortening, and `candidate_created_at < ordinal` is enforced from timestamps. (4) **Balanced one-scalar Amplitude equivalence**: the new `|A−A+|+|A−A−|` is asserted equal to the *archived* `amplitude_loss` with two identical predictions, and the tied-mass fusion is asserted equal to the archived `fuse` — both against the real archived code, not a restatement. (5) **Exact MAE roots and the safety intersection**: `lambda_U`/`lambda_S` are checked against an independent bisection root and the direct definitional L1 form, the selected `lambda` is checked to be feasible on both risk surfaces and bounded by `alpha_0`, and negative alignment is checked to force a zero radius. (6) **Tests**: run `PYTHONPATH=src python -m pytest experiments/foundation/tests/test_core_candidate.py experiments/foundation/tests/test_core_safety.py experiments/foundation/tests/test_core_purity.py experiments/foundation/tests/test_canonical_core_layout.py -q` (120 pass). The broader `experiments/foundation/tests/` run has 10 pre-existing failures from missing closed-evidence artifacts under `experiments/evidence/**`; those five modules never import `core` and are unrelated to this refactor.

Two things this refactor deliberately did **not** do. It did not add a result to `EXPERIMENT_LEDGER.md`, because no scientific diagnostic occurred — the ledger's own precedent is that it records diagnostics, and a source refactor is not one. And it did not make the safety controller an empirical claim: its correctness is a convex-geometry property on a given evidence set, verified against brute-force references, while whether it is *useful* remains an unexecuted experiment. Any future stage must pass a real pre-execution contract first, and `PROTECTED_FINAL` stays sealed.

## Leading design handoff — minimal repair + dual Residual Alignment Safety; source refactor next, no scientific run yet (2026-09-12)

*Superseded as an action list by the refactor-complete handoff above: the source refactor described at the end of this section has since been executed. The design reasoning below is retained unchanged and remains current.*

Read `docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md` first, then `paper/02_math/RESIDUAL_ALIGNMENT_SAFE_RAY_DERIVATION_20260912.md`, then `docs/current/HCH_ALIGNMENT_SAFE_CORE_REFACTOR_PLAN_20260912.md`. The signed-mass development verdict remains closed/NOT SUPPORTED and this design does not rescue any deleted component. The surviving candidate is written directly as `c=A(S+−S−)`: one shared tiny MLP, Shape GRU32, Amplitude GRU32, two Shape softmax distributions and one nonnegative daily Amplitude scalar. TCN, Shape semantic context, rare-mass sampling and untied amplitude stay deleted.

Safety is a lightweight analytic adjustment. `W=7` is frozen as **minimum complete weekly support**: the shortest contiguous daily window covering one full weekly electricity-market cycle while remaining highly local to deployment drift and matching the generator's history length; it is not claimed optimal, and 7×H coordinates are not iid samples. Use only honest persisted prequential `(S+,S-,c,r)` pairs, with the initial seven seeded from legal chronological OOF POST_TRAIN candidates. Compute a uniform recent exact-MAE safe radius `lambda_U` and a Shape-weighted radius `lambda_S`, then deploy `lambda=min(alpha_0,lambda_U,lambda_S)`. Shape is shared only through final predicted Shapes as a relevance weight; no hidden-state sharing or learned safety Gate. The intersection means Shape may only shrink further and cannot overrule broad recent harm. The property is empirical on the revealed bank only, never a future certificate.

The next user-requested task is **source refactor only**. Prompt: `docs/current/HCH_ALIGNMENT_SAFE_CORE_IMPLEMENTATION_PROMPT_20260912.md`. It must byte-preserving archive the current signed-mass development core before creating the new active `src/core`, and it must stop at source/unit/integration readiness. Do not run M0/M1/M2, do not inspect a successor DEV_EVAL, and do not read `PROTECTED_FINAL`. After the AI returns, audit its archive, W=7 legality, balanced one-scalar Amplitude equivalence, exact MAE roots, uniform∩Shape safety intersection and tests before authorizing any domestic experiment.

## Strict Results Laboratory R0 handoff — bootstrap VERIFIED, GEFCOM/NP may proceed (2026-09-12)

`experiments/lab/strict_results/` is now initialized and independently gated at **`STRICT_RESULTS_REGISTRY_VERIFIED`**. Current pre-stage snapshot: `snapshots/STRICT_RESULTS_SNAPSHOT_20260912T105648Z.json`. Registry census is **226 coordinates = 146 active numeric + 70 blockers + 10 superseded/historical**, `RESULTS_LONG.csv` = **1310** metric rows, spanning 9 admitted markets and 32 active market×Host paper cells. `registry_verify.py` was independently re-run and returns **16/16 checks, 0 faults**. This lab is now the only paper-facing aggregation surface; raw evidence remains in `experiments/evidence/**`.

The China breadth source stage carries a disclosed provenance defect: 86 source-declared aggregate array-content hashes do not reproduce, although the registry's recomputed hashes are stable and all available `pred/y_true` payloads reproduce the frozen MAE exactly. Keep `source_declared_hash` and `hash_agrees_with_source=False`; do not rewrite the frozen stage. Four non-prediction `segment/timestamp` per-array digest anomalies in two Host caches remain disclosed provenance debt, not a metric failure. NEM rows continue to carry the hourly-bin Shape/WHERE exclusion.

The comparison-infrastructure executor may now move from R0 to Q0/Q1/Q2 of `experiments/current/hch_international_gefcom_np_baseline_matrix/`; no Host training is allowed until `GEFCOM_NP_PREEXECUTION_CONTRACT_VERIFIED`. This does not authorize a new HCH method or access to any protected final split.

## Human adjudication after signed-mass closure — discuss correction-alignment stability, do not rescue the closed architecture (2026-09-12)

Read `docs/history/post_signed_mass_adjudication_20260912/README.md` after the canonical signed-mass result below. The terminal verdict remains `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`; all four registered optional components stay deleted. Read-only diagnostics from the frozen artifacts suggest the useful next question is **candidate correction alignment under residual-geometry shift**, not another Shape/Amplitude network variant: QINGHAI is the only market that fails all four Hosts, its frozen method still applies substantial corrections but correction/residual alignment is weak or negative, and its POST_TRAIN→DEV residual-Shape centroid shift is unusually large. NINGXIA is a counterexample to a simple shift threshold, so do not reduce the problem to generic drift detection.

The domestic baselines define an accuracy–negative-transfer spectrum useful for paper/problem formulation: MDR is aggressive and harmful, PIR does not transfer safely, δ-Adapter is tightly bounded around the Host, COSA online is safe but often nearly inert, while signed-mass extracts much larger gains in 16/20 cells but is unsafe in QINGHAI. Recent literature already occupies generic Gate/fallback, uncertainty-conditioned correction, online certificates/TTA and residual retrieval; any future idea must avoid those collisions.

Current theory authority is `docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md`, with the paper math derivation at `paper/02_math/RESIDUAL_ALIGNMENT_SAFE_RAY_DERIVATION_20260912.md`. The current design treats safety as a **lightweight adjustment, not necessarily a contribution**: keep the minimal correction generator unchanged; share only its already-produced `S+/S-` with the final analytic controller; use normalized signed-Shape Wasserstein overlap to weight honest past prequential `(S+,S-,c,r)` outcomes; compute the exact Shape-weighted MAE ray and shrink `lambda` only inside `[0,alpha_0]`. `A_align=2Q+/Q-1` is the loss-derived gate-like statistic; no learned Gate, clipping firewall, KNN retrieval, hidden-state sharing or deployment backprop is proposed. Do not validate current `c_d` by replaying it on the same history that generated it — that is recorded as endogenous replay optimism. CRC full text is now audited and already covers direction gating, clipping, point-wise validation selection and shrink-to-base, so do not claim safety/non-degradation novelty. **No successor experiment is authorized.** Do not read `PROTECTED_FINAL` or create a QINGHAI rescue run.

## Signed-mass 5×4 development experiment is CLOSED — verdict `HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`. There is no authorized successor. (2026-09-12)

Read `experiments/evidence/hch_signed_mass_method_20260912/METHOD_SUMMARY.md` first, then `FINAL_AUDIT.md` and `06_audits/RESULT_AUDIT.json`. The stage is `experiments/current/hch_signed_mass_method_entry/`; the controlling prompt (which carries the user's formal development-experiment authorization) is `AI_SCIENTIFIC_EXECUTION_PROMPT.md`.

```
HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION
```

- **This is the terminal verdict of the only authorized signed-mass method experiment.** The full 20-cell panel was run — `GANSU_DA / SHANDONG_DA / SHAANXI_DA / NINGXIA_DA / QINGHAI_DA` × `PatchTST / TimeMixer / iTransformer / LSTM` — 300 registered screen fits + 60 frozen-method fits, 0 failures, seeds `7/17/37` aggregated by cell median. The verdict token is produced by the preregistered gate rule; do not restate it more weakly or more strongly.
- **All four preregistered components were deleted, and the smallest surviving method is the empty switch vector.** `shape_semantic_context` (1/20 cells improved) and `use_tcn` (2/20) fail all five conditions; `rare_mass_sampling` (9/20) fails conditions 1, 4, 5; `untied_amplitude_heads` — the only one with real signal (`mass_negative_l1` median delta −4.539, 13/20 improved, median Overall-MAE delta −1.1e-5, conditions 1 and 3 TRUE) — still fails conditions 2 (`upper_tail_mae` +1.564), 4 and 5. All four fail condition 4, all driven by `GANSU_DA::LSTM`. **No rescue variant was designed, no per-market exemption applied, no metric substituted, and none is permitted now.** Treat the deletions as the result, not as an obstacle to route around.
- **The numbers to quote, and only these.** 16/20 cells beat their own Host (gains +0.697 % … +15.930 %); 16/20 beat the best setting-matched strict comparator (`delta-Adapter`, `OFFLINE_STATIC_POSTHOC`, in every cell); median Overall-MAE gain **+8.434 %**, median Tail-MAE gain **+7.624 %**. Gates 2/3/4/6 PASS, **gate 1 FAIL** (16/20 host-nonworse, required 19/20) and **gate 5 FAIL** (max cell-median normal-region relative harm **+7.585 %**, cap 1 %). The whole failure sits in **`QINGHAI_DA`**, which loses to its own Host on all four Hosts (−7.392 / −8.235 / −9.826 / −26.689 %) and to `delta-Adapter` on all four. Negative-price MAE is numeric only in the four `SHANDONG_DA` cells; the other 16 report `NOT_APPLICABLE_N0` (an empty subset — never a zero, never averageable across markets).
- **Inference setting: `FROZEN_PARAMETER_CAUSAL_PREQUENTIAL_HISTORY_POSTPROCESSING`.** Parameters are frozen on `POST_TRAIN` before `DEV_EVAL` is opened; no gradient, update or TTA during `DEV_EVAL`; each day uses only historically revealed residual windows from strictly earlier eligible origins. Describe it exactly that way. It is **not** online test-time adaptation and its numbers are never ranked against `ONLINE_TTA` (`COSA`) rows. The `MatchedDirectResidual` control is `OFFLINE_STATIC_CONTROL`, reported beside the rank and never inside it — admitting it would *loosen* the gate.
- **Independent verification PASSED: 752 checks / 0 failed** over 360 artifacts (300 screen + 60 frozen), recomputing every metric from the raw arrays with worst absolute difference `0.0` on every field, 720 identity/hash checks, 0 algebra violations, 0 metric mismatches. `n_excluded_from_rank` is **20** — exactly the twenty `OFFLINE_STATIC_CONTROL` rows, the only rows the setting rule excludes. The 10 `PIR` rows in the frozen table are numeric, setting-matched and *were* ranked candidates (they exist only for PatchTST/TimeMixer, where `delta-Adapter` beat them); the iTransformer/LSTM PIR coordinates are `Q-INCOMPATIBLE` and have no row at all. Do not describe the rank as having excluded blocked rows — it did not, because the frozen domestic table contains none. The run is not the weak link.
- **Disclosed, and it belongs in any future write-up:** three corrections were made to the *audit instrument* after its first pass, and they are stated inside the generated `FINAL_AUDIT.md` ("Instrument changes made after the first audit run") rather than left in someone's memory — the float64 `1e-9` simplex tolerance applied to float32-softmax Shape rows (0/360 artifacts exceed `1e-6`; the tolerance is now `H · eps32` and still rejects an `O(1)` defect), the frozen method's own 60 fits not previously audited, and the frozen label being reconstructed as the placeholder `FROZEN` instead of read from the frozen record. None changes a metric, a component decision or a gate. One implementation defect (`runner._dataset_for` handing a `(dataset, audit)` tuple to two bare-dataset callers) was repaired before it produced any number.
- **Do not, without a new explicit human authorization:** read `PROTECTED_FINAL` (still sealed, read count 0, never authorized this round), retrain or re-run any Host or baseline, reopen any frozen blocker, re-run the component screen hoping for a different seed, or build a rescue variant off `untied_amplitude_heads` or off the QINGHAI cells. The prompt's own rule — *if the preregistered evidence does not support it, delete it; no rescue variant* — has been executed, and the next move is **human adjudication of this evidence**, not another experiment.
- **What a human adjudication could legitimately ask for, if anything:** whether a negative development result on a 20-cell domestic panel is worth a *separately authorized* diagnosis of why the two failures concentrate where they do — not a rerun of this experiment, and not a variant of it. Nothing in this stage is authorized to answer that question.

## Newly authorized international expansion + strict-results lab (2026-09-12)

A new comparison-infrastructure stage is authorized at `experiments/current/hch_international_gefcom_np_baseline_matrix/`. Scope is fixed to `GEFCOM14P / LAGO_NP × PatchTST / TimeMixer / iTransformer / LSTM × {Host, MDR, δ-Adapter, PIR, UEC-STD, COSA, OMPB}` = **56 recorded coordinates**. GEFCOM14P was chosen as an independent GEFCom2014 price-track benchmark; LAGO_NP adds the canonical Nord Pool benchmark. `LAGO_BE/FR` remain historical controls but are not selected in this round because they add less benchmark-family diversity given existing strict German EPEX/LAGO evidence. No result exists yet.

Before any execution, read `PROTOCOL.md`, `AI_EXECUTION_PROMPT.md`, and `MATRIX_TEMPLATE_56.csv` in that stage. Q0 must freeze source hashes/day semantics and handle the known GEFCOM timestamp anomaly (duplicate `2013-03-10 01:00`, absent `02:00`) by target-blind timestamp exclusion of every 24h target or 168h context touching the irregular block; no value-based deduplication/interpolation. LAGO_NP initial census is continuous hourly. Both markets expose forecast-known exogenous columns, but all baseline Hosts remain univariate price-only 168→24. Existing baseline qualification boundaries are inherited: PIR numeric only for PatchTST/TimeMixer, UEC/OMPB blockers remain closed, COSA remains online-only. `PROTECTED_FINAL` stays zero-read.

A new project-wide strict result index is authorized at `experiments/lab/strict_results/`. First bootstrap it from the frozen domestic breadth, NEM/NORD, and active strict LAGO_DE/PJM evidence without retraining; raw evidence remains in `experiments/evidence/**`. The lab stores normalized result/blocker rows, setting/fidelity labels, provenance pointers/hashes, generated paper tables and immutable snapshots. Current NEM rows must retain the hourly-bin caveat (`OVERALL_BASELINE_ONLY_UNTIL_NEM_HOURLY_V2`). A registry update is current only after independent token `STRICT_RESULTS_REGISTRY_VERIFIED`. After the GEFCOM/NP stage passes its own final verifier, append it and write a new registry snapshot. Future Signed-Mass/HCH results also append here only after their own adjudication; never retrain baselines just to update the lab.

## Post-freeze international sanity audit — execution accepted; NORD final-ready; NEM hourly-bin correction required before Shape paper use (2026-09-12)

The frozen NEM_SA1 × NORD_DK1 baseline matrix remains valid comparison infrastructure (`NEM_NORD_BASELINE_MATRIX_FROZEN_WITH_INHERITED_BLOCKERS`): 56/56 coordinates, 36 numeric, 20 inherited blockers, 8/8 Host gates, pre-execution 13/13, final verifier 39/39, `PROTECTED_FINAL` read count 0. Do not reopen its UEC/OMPB/PIR-new-Host blockers and do not rewrite its existing numbers.

A new post-freeze data sanity audit is controlling for paper use: `docs/current/NEM_NORD_POSTFREEZE_DATA_SANITY_AUDIT_20260912.md`. `NORD_DK1` is accepted as `PAPER_BENCHMARK_READY_CURRENT_STRICT`; its local 2023 mean/negative-hour/minimum statistics match the Danish regulator essentially exactly, and all four frozen Hosts beat a read-only previous-day persistence diagnostic on DEV.

`NEM_SA1` raw AEMO source is also externally plausible, but the canonical hourly builder uses default hourly resampling on interval-ending `SETTLEMENTDATE`, so each canonical hour includes the boundary stamp `H:00` rather than the strict wall-clock ending stamp `(H+1):00`. The frozen contract's claim that H:00 represents 11 stamps H:05..H:55 is also incorrect. This is not leakage and barely changes aggregate Host MAE in a read-only corrected-bin sensitivity (`+0.04..+0.17`), but individual spike/hour locations can change materially (max target-bin difference > 1400 AUD/MWh). Therefore the existing NEM matrix remains valid under its actual product, but NEM must not be used as final signed-mass Shape/WHERE evidence until a new versioned corrected hourly product is frozen and **NEM only** is rerun. NORD requires no rerun.

## International NEM_SA1 × NORD_DK1 comparison matrix — COMPLETE / FROZEN (2026-09-12)

The international stage `experiments/current/hch_international_nem_nord_baseline_matrix/` is finished and frozen. Read `experiments/evidence/hch_international_nem_nord_baseline_matrix_20260912/04_audit/NEW_WINDOW_HANDOFF.md` first, then `04_audit/FINAL_AUDIT.md`.

```
NEM_NORD_BASELINE_MATRIX_FROZEN_WITH_INHERITED_BLOCKERS
```

- **56/56** coordinates, **36 numeric, 20 inherited blockers** (18/10 per market). Both markets `ADMITTED` — `NEM_SA1` reconciled to `ADMIT_CURRENT_STRICT_PANEL`, `NORD_DK1` required none. All **8/8** Host gates PASS. `PROTECTED_FINAL` read count **0**.
- Two independent, non-importing verifiers PASS: `NEM_NORD_PREEXECUTION_CONTRACT_VERIFIED` (13/13, obtained before any Host or DEV_EVAL run) and `NEM_NORD_BASELINE_MATRIX_VERIFIED` (39/39). Machine freeze: `04_audit/RESULT_FREEZE.json`.
- Result shape: MDR loses to its own Host on 8/8 cells (−26.9 % to −182.9 %); δ-Adapter beats its Host on 1/8 (NORD_DK1/PatchTST +1.51 %); PIR is positive on 1 of 4 legal cells (NEM_SA1/PatchTST +2.97 %) and −9.2 % to −35.8 % elsewhere; COSA moves the aggregate by < 0.18 % on 7/8 cells. No offline baseline is competitive on this panel.
- **Do not reopen the 20 blocked rows** (UEC-STD `Q-FIDELITY`, OMPB `Q-DATA`, PIR `Q-INCOMPATIBLE`) and do not manufacture a PIR backbone config for iTransformer/LSTM. The thinness *is* the recorded result. Do not merge the COSA `ONLINE_TTA` table into the offline ranking. Do not read `PROTECTED_FINAL`.
- Two `Q-IMPL` evidence-writing defects were found and repaired in place (negated Host residual sign in 8 `.npz`; mislabelled PIR seed in 4 sidecars) with `pred`/`y_true` bit-identical and no metric affected — see `04_audit/HOST_RESIDUAL_SIGN_REPAIR.json`.
- This stage produced a comparison surface, not a hypothesis. Do not design a new method or a post-result rescue variant off it. The signed-mass line below is a separate, still-pending authorization.

## Immediate next action — patch then execute signed-mass 5×4 development experiment (2026-09-12)

The user has explicitly authorized the first real signed-mass method experiment across the complete domestic panel: `GANSU_DA / SHANDONG_DA / SHAANXI_DA / NINGXIA_DA / QINGHAI_DA` × `PatchTST / TimeMixer / iTransformer / LSTM` = 20 cells. Do not ask for another confirmation once the mandatory pre-run patch passes. The execution authority is `experiments/current/hch_signed_mass_method_entry/AI_SCIENTIFIC_EXECUTION_PROMPT.md`; read `SCIENTIFIC_EXECUTION_AUDIT_20260912.md` first.

Do **not** execute the existing runner unchanged. Final audit found bounded gaps before the first DEV result: direct branch outputs must drive Shape/Amplitude mechanism metrics; training-only robust `s_A` must be wired into the Amplitude decoder; q90(A+)/q90(A-) high-mass metrics must be frozen from POST_TRAIN; raw branch/prediction evidence must be persisted; D0/component/frozen-method/baseline aggregation and independent result verification must be added. These are implementation-to-contract corrections, not architecture changes. After patch + tests + verifier PASS, run the full registered 20×5-config×3-seed grid, delete any unsupported one of the four preregistered components, confirm the smallest surviving configuration across all 20 cells, and compare against the already-frozen strict offline baseline table. `PROTECTED_FINAL` remains zero-read and unauthorized.

## International benchmark coverage audit — do not assume PAPER8 is already strict-complete (2026-09-12)

The historically registered international paper panel is `GEFCOM14P / LAGO_BE / LAGO_DE / LAGO_FR / LAGO_NP / LAGO_PJM / NEM_SA1 / NORD_DK1`. Only `LAGO_DE` and `LAGO_PJM` currently have recent unified-DA strict baseline-transfer evidence on PatchTST/TimeMixer with Host, MDR, δ-Adapter and PIR. The other six have historical V1/V2.5 records but are **not** yet interchangeable with the current domestic matrix. `LAGO_NP` has later pilot data-admission evidence; `NEM_SA1` remains conditional after a registry-gate failure. Read `docs/current/INTERNATIONAL_BENCHMARK_RESULT_COVERAGE_AUDIT_20260912.md` before designing any international result expansion.

Keep the consumed `DE_EPEX / NORD_FI / PJM_2020` sentinel panel separate from final-paper benchmark counting: DE/PJM overlap LAGO_DE/LAGO_PJM roles and NORD_FI is development evidence outside PAPER8. Current audit token: `INTERNATIONAL_BENCHMARK_HISTORY_PRESENT_CURRENT_STRICT_MATRIX_INCOMPLETE`.

**Update (2026-09-12, later the same day):** `NEM_SA1` and `NORD_DK1` have since been brought onto the current strict standard by the frozen international matrix stage (see the top section). `NEM_SA1`'s conditional registry status was reconciled to `ADMIT_CURRENT_STRICT_PANEL`, and both markets now carry current-standard Host and baseline evidence on all four Hosts (PatchTST, TimeMixer, iTransformer, LSTM) — with COSA online, and with UEC-STD / OMPB / PIR-on-new-Hosts recorded as inherited blockers rather than numbers. Four of the eight PAPER8 markets remain non-interchangeable with the current matrix: `GEFCOM14P / LAGO_BE / LAGO_FR / LAGO_NP`.

## Immediate next action — land the signed-mass China-5 experiment harness, then run 5×4 (2026-09-12)

Do **not** rewrite `src/core/`: the signed-mass implementation has already passed its source-readiness audit. Before any scientific signed-mass run, execute the integration prompt at `experiments/current/hch_signed_mass_method_entry/AI_IMPLEMENTATION_PROMPT.md`. It must connect the existing core to audited China-5 dataset contracts and frozen PatchTST/TimeMixer/iTransformer/LSTM Host artifacts, implement chronological `POST_TRAIN` OOF training + nonnegative MAE alpha, the five registered FULL/ablation configs, metrics/provenance and pre-execution tests, while keeping `DEV_EVAL` out of fitting and `PROTECTED_FINAL` at zero reads. No scientific result is authorized during this landing step. Once the harness returns READY and the parallel Host breadth artifacts are frozen, the next scientific stage must cover the complete valid **5 domestic markets × 4 Hosts** panel; do not silently omit cells or retrain Hosts/baselines.


## Domestic Host × baseline breadth is FROZEN — read the matrix, do not rerun the blockers (2026-09-12)

The comparison-infrastructure stage `experiments/current/hch_china_host_breadth_expansion/` is complete and frozen:

```
CHINA_DOMESTIC_HOST_BASELINE_MATRIX_PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS
```

Evidence root `experiments/evidence/hch_china_host_breadth_expansion_20260912/`. Start with `NEW_WINDOW_HANDOFF.md`, then `04_audit/FINAL_AUDIT.md`. **140/140** coordinates, **90 numeric, 50 blocked**. Both independent verifiers pass (`CHINA_BREADTH_PREEXECUTION_QUALIFICATION_COMPLETE`; `CHINA_DOMESTIC_HOST_BASELINE_MATRIX_E5_VERIFIED`, 11/11). `PROTECTED_FINAL` read count 0; parent panel 136/136 and imported evidence 45/45 byte-identical.

**Do not rerun the 50 blocked coordinates.** They are recorded blockers, not pending work: UEC-STD 20 (`Q-FIDELITY`, bounded recovery pass consumed, 7.6062% vs a frozen 7.5% gate), OMPB 20 (`Q-DATA`, official data unavailable and no proxy permitted), PIR 10 (`Q-INCOMPATIBLE`, the frozen backbone configuration does not exist for iTransformer/LSTM). Changing a tolerance, seed, split, anchor or proxy to unblock them is forbidden by `PROTOCOL.md` §5 and by the stage's authorization.

Read the results as **two separate tables, never pooled**: `03_results/STRICT_OFFLINE_TABLE.csv` (70 rows: Host, MDR, δ-Adapter, PIR) and `03_results/ONLINE_SUPPLEMENTARY_TABLE.csv` (20 rows, COSA under `ONLINE_TTA`). Raw per-row evidence is at `03_results/RAW_PREDICTION_INDEX.csv` (86 rows with array content hashes) and per-day 24h errors under `03_results/daily/`.

What the matrix says, briefly: Host strength is market-dependent and does not order consistently (LSTM worst in four markets, best in 宁夏); the project-defined MDR control is worse than its own Host in 19/20 cells; δ-Adapter is within ±1.6% of its own Host everywhere; PIR is worse in all 10 cells it can run; COSA's online gain is real but small (+0.01%…+1.71%) and largest where the Host is weakest.

The breadth stage produced **no** HCH / Signed-Mass result and proposes no successor method from its data. It is a substrate: if the signed-mass design line later freezes a method, register it into the breadth registry rather than rerunning any Host or baseline. The signed-mass method line remains design-only and its own handoff follows below, unchanged.

## Next method handoff — signed-mass core source-ready; run only the registered component screen next (2026-09-12)

`src/core` has now been audited and corrected against `docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`. Read `src/core/CORE_READINESS_AUDIT_20260912.md` before implementation work. The next scientific protocol is `docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`; it is pre-registered but no method result exists yet.

Important corrections already landed: the core now takes **all audited forecast-known numeric features** through a generic `(B,H,F)` tensor and one shared MLP instead of hard-coded load/wind/solar; unavailable roles use explicit masks rather than Host substitution; all calendar coordinates survive; Shape history is scale-free while Amplitude history preserves residual magnitude; and pooled OOF MAE calibration restores the historical nonnegative weighted-median constraint. The manual Amplitude severity inventory was removed — Amplitude learns current-day severity from the shared absolute embedding plus its own TCN/GRU history path.

The first method is still `shared MLP -> {Shape small TCN->GRU32, Amplitude small TCN->GRU32} -> (S+,S-,A+,A-) -> exact A+S+−A−S− -> OOF alpha`. No Bridge, no learned proposal selector, no KNN in the first method. Exactly four components may be removed by evidence: Shape semantic context, TCN, untied `A+/A-` heads, rare-mass batch spreading. Their source switches are for leave-one-out ablation only; if one fails, delete it rather than tune/rescue it.

Core validation currently passes the signed-mass/purity/layout suite together with archived V2.5 compatibility tests; `src/core` also compiles cleanly. No new method was trained and `PROTECTED_FINAL` remains untouched. The domestic baseline breadth stage is independent and may continue in parallel. Once it freezes, the smallest surviving signed-mass method should be appended to its registry rather than triggering Host/baseline reruns.

## Current execution handoff — domestic Host × baseline breadth is qualification-first (2026-09-12)

A new comparison-infrastructure stage is explicitly authorized at `experiments/current/hch_china_host_breadth_expansion/`. Read `PROTOCOL.md` and `QUALIFICATION_AND_REPAIR_GATES.md` before its `AI_EXECUTION_PROMPT.md`. It is separate from the signed-mass method-design line: do **not** train/evaluate the new Shape/Amplitude proposal in this stage.

The target matrix is 5 domestic DA markets × 4 Hosts × 7 recorded rows = **140 coordinates**. Hosts are frozen/imported PatchTST and TimeMixer plus new iTransformer and LSTM. GANSU adds only iTransformer/LSTM Host training; its existing PatchTST/TimeMixer evidence is imported. Rows are Host, MDR, δ-Adapter, PIR, UEC-STD, COSA and OMPB. MDR is a project-defined control, not an external reproduced baseline. COSA and OMPB retain separate online settings and must never be merged into an offline winner rank.

Do not begin Chinese DEV_EVAL execution directly. First qualify/repair implementation paths. The executor may correct concrete loader/scaler/checkpoint/source-import/serialization/chronology/wrapper defects, but may not alter scientific hyperparameters, tolerances, splits, seeds or anchors after an outcome. iTransformer must pass its pinned official ETTh1 96→96 numerical anchor; LSTM must pass the common-contract production path. δ/PIR require parent-transfer replay. COSA completes its bounded paper-era/causal-wrapper checks. UEC and OMPB may be reopened only inside the bounded qualification contracts now explicitly authorized by the user. If they remain protocol-correct but blocked, record blockers rather than proxies.

Only an independent `verify_preexecution_qualification.py` PASS with token `CHINA_BREADTH_PREEXECUTION_QUALIFICATION_COMPLETE` allows automatic continuation to the matrix. `PROTECTED_FINAL` remains zero-read. Final tokens are `CHINA_DOMESTIC_HOST_BASELINE_MATRIX_FULLY_FROZEN`, `...PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS`, or `...INVALID`.

## `src/core` now holds the signed-mass core; V2.5 core archived; no experiment opened (2026-09-12)

Read this before touching `src/core/`. The package was repurposed this date from the accepted frozen V2.5 point core to a **signed-mass repair core** implementing the leading discussion design (`Shape` = WHERE, `Amplitude` = HOW MUCH, `ĉ_h = Â⁺Ŝ⁺_h − Â⁻Ŝ⁻_h`, one pooled MAE scalar `α`). This was an **explicitly human-authorized one-time override** of the branch freeze and of the `core`-promotion rule; the rules themselves were not changed. **It is scaffolding only: nothing was trained, no benchmark or scientific diagnostic ran, and it authorizes no experiment.** `PROTECTED_FINAL` stays `SEALED_STRUCTURALLY_NOT_READ`.

The old V2.5 core (`learned_signature`, `iah_candidate`, `iah_crps`, `weighted_mean_readout`, `v25_point_runtime`, `universal_trainer`, plus the package `__init__.py`/`README.md`) moved **byte-identical** to `src/archive/core_pre_extreme_repair_20260912/legacy_core/` (14/14 sha256 verified; archived sources not rewritten). Per `src/archive/README.md` this is **lifecycle metadata, not a new scientific verdict** — V2.5 science is unchanged and its evidence was not rewritten. Callers that still need it bootstrap the out-of-package compat module first:

```python
import legacy_core_compat          # from src/ ; re-registers the archived core.* aliases
from core.v25_point_runtime import HCHV25PointRuntime
```

The compat module is an **opt-in `sys.meta_path` alias**, not a source-level redirect; the trade-off and the known callers are recorded in `ARCHIVE_MANIFEST.md` and `src/archive/README.md`.

**Two boundaries a next window must not cross.** First, `src/mvp/hch_minimal_repair/` — the repository's currently designated ray object — was **not** archived, modified, superseded or deprecated, and no `docs/current` design-document `Status:` line was changed; the two objects coexist. Second, the new core is **not an M0 canary and is not authorized by `docs/current/HCH_MINIMAL_REPAIR_IMPLEMENTATION_SPEC_20260909.md`**; it is a separate preparation step. Inventory and rationale: `src/core/CORE_REORGANIZATION_INVENTORY.md`, `src/core/README.md`, `src/core/DESIGN_CONTRACT.md`. Any future run against the China-5 substrate still fits every scaler, quantile and calibration statistic on `POST_TRAIN` only, never `DEV_EVAL`, never `PROTECTED_FINAL`.

## Current method-design handoff — signed-mass Shape/Amplitude network is the leading discussion candidate (2026-09-12)

Read `docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md` first for the current architecture. The narrow question is extreme/negative-price post-processing under a frozen Host, not generic forecasting: learn horizon-wise positive/negative correction distributions `S+ / S-` (WHERE) and total positive/negative daily correction masses `A+ / A-` (HOW MUCH), then combine them exactly as `c = A+S+ - A-S-` and apply one pooled exact OOF MAE calibration scalar `alpha`.

The current minimal executor is: legal raw information → deterministic train-only canonical coordinates → **one shared tiny MLP stem** → task-directed Shape/Amplitude context → two separate small `TCN -> GRU32` branches → deterministic signed-mass fusion. Shape emphasizes scale-free Host/load/renewable/net-load geometry and ramps; Amplitude emphasizes absolute market/Host stress and residual severity, with one common Amplitude trunk and independent positive/negative heads. No Bridge and no learned Gate are in the default path. Branch interaction already occurs through the shared stem and the differentiable final MAE; gradients through fusion scale Shape learning by predicted mass on extreme days.

Similar-window/KNN is not in the main method; it may only return as auxiliary Shape context if a no-training diagnostic proves similar legal contexts have similar residual Shapes. Amplitude rarity is first handled by training-only stratified batch ordering with no duplicate oversampling by default. `TCN`, `GRU/BiLSTM`, KNN and sampling are implementation tools rather than contribution claims. Before any final method promotion, ablations must justify task-directed context, TCN beyond Daily-Patch GRU32, asymmetric `A+ / A-` heads and rare-mass handling. `PROTECTED_FINAL` stays sealed.

Architecture figure source: `figures/hch-signed-mass-network-architecture.mmd` and markdown preview `figures/hch-signed-mass-network-architecture.md`. Mermaid CLI verification was attempted but the local npm installation returned `ECOMPROMISED / Lock compromised`; the source was retained for later rendering after the environment issue is repaired.

## China-5 frozen comparison substrate is ready — add an `Our Method` row, do not retrain Hosts (2026-09-12)

A frozen Host + baseline panel now exists for four Chinese provincial day-ahead markets (山东, 陕西, 宁夏, 青海) at `experiments/evidence/china5_posthoc_baseline_panel_20260912/`, terminal state `CHINA5_BASELINE_PANEL_PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS`. A future new-method window appends one `Our Method` row per (market, Host) cell; it must **not** retrain a Host or a baseline, must **not** re-fit any scaler, quantile or threshold, and must **not** unseal `PROTECTED_FINAL`. Read `BASELINE_PANEL_SUMMARY.md` and `NEW_WINDOW_HANDOFF.md` in that root first; the authoritative cell values are `MAIN_OFFLINE_TABLE.csv` (32 strict rows), `SUPPLEMENTARY_TABLE.csv` (COSA with its online setting + the two blocked tracks) and `METRICS_BY_CELL.csv` (40 rows).

Everything needed for comparison is frozen: 8 Hosts all passing their gates on the verbatim accepted GANSU_DA recipe, complete raw prediction / target / residual arrays indexed in `RAW_PREDICTION_INDEX.csv` (44 rows, all hashes verified), per-day 24h losses in full, and `DATASET_REGISTRY.csv` / `HOST_REGISTRY.csv` / `BASELINE_FIDELITY_REGISTRY.csv` / `RUN_LEDGER.csv` / `EVIDENCE_PROVENANCE_MAP.csv`. Thresholds were fixed from `HOST_TRAIN` only and are recorded in `00_protocol/THRESHOLD_FREEZE.json`.

Two things a new window must inherit rather than rediscover. First, the **result shape**: only δ-Adapter ever matches or beats a frozen Host, and harm is concentrated in the normal-price region and grows as the training block shrinks — the same signature as `LAGO_PJM/PatchTST`. Any new method should be judged on that structure, not on average MAE alone. Second, the **coverage gaps**: 山西 has no dataset in this repository (`ABSENT_FROM_REGISTRY`) and was neither invented nor substituted; `UEC_STD` and `OMPB` are blocked with no proxy manufactured. These are disclosed limitations, and repairing them requires new data, not a new model. Baseline fidelity statuses are immutable here — COSA stays online/supplementary and must never be merged into the offline column to claim a `best strict baseline`.

## New-window handoff — audited baseline transfer isolates a repairability / harm-region problem (2026-09-12)

The frozen transfer stage terminated with `HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED`. Execution is valid. Frozen `S1 Daily-Patch GRU32 Shape + pooled exact OOF MAE alpha` is strict best among admitted offline methods on both `GANSU_DA` Hosts, both `LAGO_DE` Hosts and `LAGO_PJM/TimeMixer`. The only decisive failure is `LAGO_PJM/PatchTST`, where HCH MAE `3.0458` is `2.3564%` worse than the best admitted delta-Adapter (`2.9756`) and `0.8658%` worse than its own Host (`3.0196`). B1 GANSU competitiveness passes; B2 international parity and B3 overall fail.

This is no longer an average-accuracy or capacity problem. The evidence suggests a **repair applicability / harm region**: HCH is strong where residual geometry is structured, but an already strong Host may leave mostly unrepairable innovation, making deterministic post-hoc correction harmful. Do not locally rescue `PJM/PatchTST` by changing S1, alpha, state features, conditional Amplitude, Verification/Gate, HSA/HRSI, thresholds, markets or baseline selection. The branch is frozen.

The next window is **discussion only**. Read `docs/history/post_baseline_transfer_reset_20260912/README.md` and use `docs/history/post_baseline_transfer_reset_20260912/NEW_WINDOW_PROMPT.md`. The leading hypothesis to interrogate, not assume, is `Forecast Repairability / Residual Predictability`: whether legal forecast-origin information contains a predictable residual component large enough to justify any deterministic repair. This must be mathematically distinguished from the already-rejected per-proposal verifier and from generic selective forecasting / abstention, PIR Host-error scoring and OMPB fallback. No new experiment, code change, new China market, S3/S4, protected or final access is authorized until human discussion explicitly opens a new design.

Baseline-transfer authority: `experiments/evidence/hch_frozen_method_baseline_transfer_20260911/BASELINE_TRANSFER_SUMMARY.md`, `BASELINE_TRANSFER_VERDICT.json`, and `NEW_WINDOW_HANDOFF.md`; executor reports commit `8ff39ec`. Paper writing can continue on stable sections, but final SOTA/contribution/Abstract/Conclusion wording remains open.

## Window handoff — HRSI terminal negative; frozen S1 enters formal baseline transfer (2026-09-11)

`HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1` is valid. R0/R4 pass; R1/R2/R3 fail. HRSI used one global five-parameter beta shared across all six cells, so the negative result is not caused by market-specific fitting. It does not recover GANSU HIGH-state Shape and worsens S1 in 5/6 cell medians. Structural rescue is now closed for this branch.

Freeze the executable method as `S1 Daily-Patch GRU32 Shape + pooled exact OOF MAE alpha`. Do not reopen HSA/HRSI, excursion, conditional Amplitude, learned Verification/Gate, new state transforms, deeper Shape architecture, or post-hoc threshold tuning.

The only authorized next scientific execution is `docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md`, launcher `experiments/current/hch_frozen_method_baseline_transfer/CODEX_GOAL_PROMPT_20260911.md`. Fixed panel remains `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`. Main offline comparison is Host, frozen MatchedDirectResidual, audited δ-Adapter Ada-Y, exact full-official PIR, and frozen HCH-S1. Historical PIR proxy/2-epoch δ wrapper are forbidden. COSA/UEC/OMPB are not part of main offline best-baseline ranking.

Decision rule: PASS (`HCH_BASELINE_TRANSFER_TARGET_MET_FULL_PANEL_ALLOWED`) freezes S1 and authorizes design of the full public-China + international transfer. FAIL (`HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED`) freezes this branch and requires a new-window scientific reset; do not perform another local rescue before the reset.

Paper writing should proceed in parallel. Stable sections can be written now; result-sensitive claims stay gated. See `paper/_drafts/PAPER_WRITING_STAGE_PLAN_20260911.md`.

## Window handoff — Host-Relative State Interaction rejected; S1 + pooled fixed alpha frozen (2026-09-11)

`HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1` is valid. R0 (`PASS`, S1 replay `1.421e-14`, `beta=0` replay `1.192e-07`) and R4 (`PASS`) hold, so the execution is admissible and the negative is scientific rather than procedural: R1, R2 and R3 all fail.

The registered method was executed exactly as specified. Frozen S1 Shape was reused unchanged; `b_hat` was the L2-normalized same-hour mean of the existing seven-day normalized Host-residual history; `Z[h,k]=b_hat[h]*R[h,k]` used the fixed five-role standardized state table; and **one global `beta in R^5` shared by all six cells** was trained on the cross-market stacked chronological OOF pool with a cell-macro-balanced Shape loss. `u_HRSI=normalize(u_S1+Z beta)`; `alpha_HRSI` was refit per cell by the original exact weighted-median rule. The whole panel trains 5 parameters, with 0 per-cell beta vectors.

Where it fails: the shared beta destroys the only mechanism that worked. GANSU Shape cosine changes by only `+0.0002` / `-0.0105` against HSA's `+0.0666` / `+0.1170`, and both GANSU HIGH-state tertile cosines fall (`-0.0109` / `-0.0241`). Overall-MAE gain vs S1 is nonnegative in only 1/6 cell medians and strictly better in only 1/6 (worst cell `-0.6151` pp on LAGO_DE/PatchTST, worst seed `-1.3676` pp); 0/4 international cells are strictly better; only 1/6 cells are non-worse in `>=2/3` seeds; median gain vs Host is `5.61%` with only 3/6 cells `>=5%`; max Normal-MAE harm `1.0227%` exceeds the 1% gate.

The learned beta is stable across seeds (`[0.109, 0.172, 0.000, -0.265, -0.134]`, `[0.048, 0.154, 0.000, -0.313, -0.050]`, `[0.054, 0.144, 0.000, -0.328, -0.047]`) but is set by the four international cells, which contribute most pooled rows; `beta_SUPPLY_MARGIN_FC` is exactly `0.0` for every seed because that role is structurally absent in all three markets. HSA's GANSU gain came from per-market coefficient freedom, so the constraint that makes the interaction universal is exactly the constraint that removes it. The transferability conflict is in the coefficient, not the representation; making the input Host-relative does not resolve it.

Do not rescue this by per-market/per-Host beta, per-hour beta, role selection, conditional Amplitude, a new window or transform, or post hoc threshold relaxation. **`S1 Daily-Patch GRU32 Shape + pooled fixed alpha` is frozen as the final structural candidate.** Read `experiments/evidence/hch_host_relative_state_interaction_20260911/HRSI_SUMMARY.md`, then `HRSI_VERDICT.json`; `verify_gates.py` independently reproduces R0–R4 and the token from the written evidence alone (50/50 integrity checks). No SHAANXI/NINGXIA/QINGHAI, Shandong, full-panel, or protected/final work is authorized by this stage. The next stage is human adjudication, then the full public-China/international comparison against the completed baseline-fidelity suite.

Baseline preparation is complete for that purpose: the selected baselines have all completed strict protocol/official-code/split/training/metric auditing, while their exact admission/fidelity boundaries remain those in the final baseline adjudication. Future HCH-vs-baseline experiments must use identical dataset/task/Host/evaluation contracts.

## Window handoff — Horizon-Aligned Semantic State Residual rejected; S1 + pooled fixed alpha retained (2026-09-11)

`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1` is valid. H0 (`PASS`) and H4 (`PASS`) hold, so the execution is admissible and the negative is scientific rather than procedural: H1, H2 and H3 all fail.

The five-role horizon-aligned semantic state residual was executed exactly as registered. Frozen S1 OOF Shape directions were reused unchanged; the only new trainable object was `beta in R^5` on a deterministic `24x5` role table mean-pooled from audited `primary_channel=True` forecast-time columns, with `u_HSA = normalize(u_S1 + R_d beta)` and `alpha_HSA` refit by the original exact weighted-median rule. `beta=0` replayed S1 at `1.192e-07`; frozen S1 metrics replayed at `1.421e-14`.

Where it works: both GANSU_DA Hosts improve Shape cosine (`+0.0666` PatchTST, `+0.1170` TimeMixer) and the frozen GANSU HIGH-state tertile (`+0.0347` / `+0.1189`) while cutting HIGH-state wrong-hemisphere by 12.5 / 18.2 pp; GANSU Overall-MAE gains `+1.30` / `+4.31` pp vs S1 and `3.55%` / `11.04%` vs Host.

Where it fails: the mechanism does not transfer. Shape cosine rises in only 4/6 cells, wrong-hemisphere is non-worse in only 3/6, Overall-MAE gain vs S1 is nonnegative in only 4/6 cell medians (worst `-0.87` pp on LAGO_DE/PatchTST), median gain vs Host is `6.73%` with only 3/6 cells at `>=5%`, max normal-MAE harm is `1.02%` against a 1% gate, and only 3/6 cells hold `>=` S1 in `>=2/3` seeds. Efficiency was never the constraint (5 parameters, `1.73 s/cell` beta training, `1.22 us/day` overhead).

Do not rescue this by adding China-only roles, selecting roles per market, making `beta` per-hour or conditional, enlarging the step budget, or relaxing H1/H2/H3 after the fact. Keep `S1 Daily-Patch GRU32 Shape + pooled fixed alpha` as the executable method. Read `experiments/evidence/hch_horizon_aligned_state_residual_20260911/HSA_SUMMARY.md`, then `HSA_VERDICT.json`; `verify_gates.py` independently reproduces H0–H4 and the token from the written evidence alone (39/39 integrity checks). No SHAANXI/NINGXIA/QINGHAI, Shandong, full-panel, or protected/final work is authorized by this stage.

## Baseline-fidelity handoff — workstream fully adjudicated and closed (2026-09-11)

The bounded baseline breadth recovery is complete with `HCH_BASELINE_BREADTH_RECOVERY_COMPLETE_FOR_ADJUDICATION`. Read `experiments/evidence/hch_baseline_paper_fidelity_20260911/adjudication/FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md` before using any baseline-fidelity claim.

Final statuses are mixed: δ-Adapter Ada-Y = `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`; PIR full official = `PAPER_FAITHFUL_EXACT_ACCEPTED`; COSA = `ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`; UEC-STD = `HOST_FIDELITY_BLOCKED / NOT_PAPER_FAITHFULLY_ADMITTED`; OMPB = `STANDALONE_DATA_BLOCKED / NOT_REPRODUCED`.

UEC source-native recovery corrected the prior custom-loader/AR mistake. Official ETTh1 loader/fixed borders/train-only scaler/fresh TimeMixer-96/official AR336 gives Host MSE/MAE `0.500369/0.457737` vs paper `0.465/0.449`; MSE error `7.606%` narrowly misses the frozen 7.5% gate. Do not relax the threshold and do not call this a UEC method failure; correction training was never entered.

OMPB exact TCN/96 paper rows are frozen, but official dataset acquisition failed. No source/target data were read and no training/metrics exist. OMPB can reopen only if official or provenance-equivalent data become available.

For HCH final offline comparison, the mandatory admitted layer is sufficient and frozen: Host/identity, direct residual, δ-Adapter (partial fidelity disclosed) and PIR full official (exact). COSA/UEC/OMPB are supplementary and must not block the active HCH method stage. Never write `all baselines were fully reproduced`.


## Window handoff — State-Excursion rejected; Horizon-Aligned Semantic State Residual active (2026-09-11)

`HCH_STATE_EXCURSION_SHAPE_NOT_SUPPORTED_KEEP_S1` is valid and independently adjudicated. S2 signed same-hour excursion improves all four international cells but degrades both `GANSU_DA` cells, so it is not a universal repair representation. Do not rescue it by changing the 7-day window, MAD factor `1.4826`, clipping, or role subsets. `1.4826` is merely the standard Gaussian-consistency MAD rescaling and is now diagnostic-only.

The surviving bottleneck is horizon alignment / representation efficiency: S1 flattens all current-day state trajectories into one token before the GRU readout, and parameter count grows with raw channel count. GANSU is the richest-state/smallest-fit case and is harmed most when additional flattened state channels are added.

Current authorized closure: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`. Freeze S1 entirely. Reuse S1 train-only standardized state levels, map primary channels to the fixed five semantic roles, role-pool to `R in R^{24x5}`, learn exactly `beta in R^5`, and set `u_HSA=normalize(u_S1 + R beta)`. Same five parameters/formula/optimizer in every market; no market-specific tuning. Refit only the pooled exact MAE scalar `alpha` on HSA OOF directions. No conditional Amplitude, Gate, excursion transform, deeper network, extra markets, Shandong, or protected/final access.

Launcher: `experiments/current/hch_horizon_aligned_state_residual/CODEX_GOAL_PROMPT_20260911.md`. Evidence root: `experiments/evidence/hch_horizon_aligned_state_residual_20260911/**`. Even if PASS, stop for human adjudication before public-China expansion.


## Window handoff — Market-State Coupling rejected for Amplitude; State-Excursion Shape closure active (2026-09-10)

The Market-State Coupling diagnostic is complete and valid. Read `experiments/evidence/hch_market_state_coupling_20260910/adjudication/INDEPENDENT_MARKET_STATE_COUPLING_ADJUDICATION.md` before interpreting raw artifacts. The tested scalar state-intensity hypothesis does not stably explain per-day O2 ray distance; chronological one-feature ridge improves the constant-zero distance baseline in only 2/6 cell medians, so `A_state1` was correctly not trained. Keep pooled fixed alpha and keep learned Verification closed.

The actionable signal is Shape-side. Four of six cells are `SHAPE_LIMITED`; both GANSU_DA Hosts degrade in high-state periods (Shape cosine about -0.062/-0.185 high-vs-low, wrong hemisphere +25/+20 pp). Current S1 sees current forecast-time state levels but does not explicitly encode their signed deviation from recent same-hour operating conditions.

The only active scientific execution is now `docs/current/HCH_STATE_EXCURSION_SHAPE_CLOSURE_20260910.md`. Freeze S1 GRU32 topology, Shape target/loss, historical tokens, Host/splits/OOF and fixed-alpha distance. Append only the already-audited signed same-hour seven-day state-excursion channels (`xi/10`) to the S1 current-day Shape patch. Fixed panel: `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds `7,17,37`.

Do not modify Amplitude, reopen Verification, change GRU depth/hidden size, add attention/Transformer/router/experts, select state roles post hoc, add new markets/Shandong, tune after outcomes, or access protected/final roles. Short launcher: `experiments/current/hch_state_excursion_shape/CODEX_GOAL_PROMPT_20260910.md`. Even PASS only permits a later public-China expansion design.

## Window handoff — Market-State Coupling diagnostic authorized (2026-09-10)

The Anchored Compact Amplitude closure is complete and independently adjudicated: the 26-param Shape-only `[u||c]->a` learner is rejected because median O2-over-fixed-alpha capture is 2.01% and oracle-distance Spearman is near zero. Return to `S1 Daily-Patch GRU32 Shape + pooled fixed alpha`; learned Verification remains removed.

The new active scientific question is whether publication-time electricity-market state contains the magnitude information intentionally discarded by normalized Shape. Read `docs/current/HCH_POST_AMPLITUDE_MARKET_STATE_COUPLING_DESIGN_20260910.md` and `docs/current/HCH_MARKET_STATE_COUPLING_EXECUTION_PROTOCOL_20260910.md`. Primary D0 panel remains `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`.

Run C0--C4 first: audit a semantic market-state schema, construct same-hour seven-day robust state excursions, measure coupling to exact O2 ray distance, and separate Shape-limited from distance-limited high-state regimes. Only if all preregistered coupling gates fire may the run train `A_state1`, a one-parameter state-anchored scalar Amplitude initialized exactly at fixed alpha. Otherwise no new HCH learner is allowed.

Do not modify S1 Shape, reopen Verification, add more state features/nonlinear Amplitude, use market experts/router/retrieval, add Shandong, tune after results, or access protected/final roles. Short executor launcher: `experiments/current/hch_market_state_coupling/CODEX_GOAL_PROMPT_20260910.md`. Evidence root: `experiments/evidence/hch_market_state_coupling_20260910/**`.


## Method-design handoff — Shape-only compact Amplitude rejected; market-state coupling diagnostic is next design question (2026-09-10)

The Anchored Compact Amplitude closure is complete and valid. Read `experiments/evidence/hch_anchored_compact_amplitude_20260910/adjudication/INDEPENDENT_ANCHORED_COMPACT_AMPLITUDE_ADJUDICATION.md` before using its artifacts. The tested 26-param `[u_hat || ||v||] -> scalar` learner is rejected: final A2 fails and median capture of O2-over-fixed-alpha headroom is only 2.01%, with cell-median amplitude-oracle Spearman about `-0.044 ... +0.126`. Return the leading executable method to S1 Daily-Patch GRU32 Shape + pooled fixed `alpha`; learned Verification stays removed.

Do not interpret this as evidence that scalar Amplitude is unnecessary. Exact O2 per-day scalar gains remain ~10.8--22.7% in all six cells. The failure is an information bottleneck: normalized Shape supervision intentionally removes residual scale, and `||v||` measures directional concentration rather than residual magnitude, so a deeper network on the same Shape-only input is not justified.

Paper strategy remains: public Chinese provincial DA->DA markets are the intended motivation/stress domain; established international DA benchmarks are the cross-market generality check. However, the present O2 headroom is also large internationally, so no China-specific Amplitude claim follows yet. The next design-only document is `docs/current/HCH_POST_AMPLITUDE_MARKET_STATE_COUPLING_DESIGN_20260910.md` together with `HCH_CHINA_MARKET_STATE_ICDE_DESIGN_20260910.md`.

No new scientific execution is authorized yet. First test whether publication-time-legal market-state excursion predicts exact per-day ray distance and whether high-state-intensity failures are primarily Shape or Amplitude. Only stable coupling may reopen a one-coefficient state-conditioned scalar distance anchored at fixed alpha. If state affects Shape rather than distance, keep fixed alpha and change only the state representation feeding S1. Do not jump to MLP, experts, router, attention, Transformer, graph or retrieval.

ICDE-facing contribution should emphasize the data-engineering object: a leakage-safe horizon-aligned repair table over frozen Host predictions, original-origin residuals and heterogeneous forecast-time market state with masks/provenance; the learned repairer stays lightweight. Read `paper/04_literature/ICDE2027_MARKET_STATE_REPAIR_POSITIONING_20260910.md` for venue/literature positioning. The paper draft has been updated accordingly, with all China-hardness / broad superiority claims still evidence-gated.

## Parallel baseline-fidelity handoff — UEC source-native recovery + standalone OMPB authorized (2026-09-11)

The active HCH scientific execution remains `HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`; this baseline work is auxiliary and must not alter HCH method/data/protected roles.

Important correction: the previous U-A1 runner is now known to be non-source-native for ETTh1. It forced `--data custom` / `Dataset_Custom` and implemented its own AR336 loop, whereas the pinned official UEC source maps `--data ETTh1` to `Dataset_ETT_hour` and exposes long-horizon AR through official modes 2/5/6. Preserve the old `0.596529` Host and `0.588395` UEC MSE as historical custom-runner evidence, but do not use them to conclude that official paper-native UEC is irreproducible.

Read `experiments/evidence/hch_baseline_paper_fidelity_20260910/05_uecstd_ompb_minimal/adjudication/UECSTD_PROTOCOL_CORRECTION_AMENDMENT_20260911.md`, then `docs/current/HCH_UECSTD_HOST_FIDELITY_RECOVERY_PROTOCOL_20260911.md`. UEC now starts from a fresh source-native ETTh1/TimeMixer-96 Host and official AR336 mode-2 gate. Host MSE must be within 7.5% of paper `0.465` before correction training. If the released-source Host fails, stop UEC and disclose the paper-text 70%-train versus released fixed-ETT-border discrepancy; do not auto-run another split. If U-A1 passes fully, only then run Weather U-A2.

OMPB is no longer conditional on UEC. Read `docs/current/HCH_OMPB_STANDALONE_MINIMAL_PAPER_FIDELITY_PROTOCOL_20260911.md`. Run exactly two TCN/96 online-shift anchors (`ETTh1->ETTh2`, `Weather_train->Weather_test_far`) after freezing exact final-paper rows and official dataset hashes. Preserve official Bayesian head/certificate/source-risk/mismatch and predict-before-reveal semantics; perform the registered future-label perturbation audit. No third anchor or tuning rescue.

Combined plan: `docs/current/HCH_UECSTD_OMPB_FIDELITY_RECOVERY_EXECUTION_PLAN_20260911.md`. Launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_UECSTD_HOST_RECOVERY_AND_OMPB_STANDALONE_20260911.md`. Both tracks must reach their own legal terminal status before returning `HCH_BASELINE_BREADTH_RECOVERY_COMPLETE_FOR_ADJUDICATION`; this completion token does not mean both PASS. No HCH transfer follows automatically.

## Baseline-fidelity handoff — mandatory offline anchors sufficient; UEC failed Host fidelity; OMPB still unrun (2026-09-11)

The optional UEC-STD + OMPB breadth stage returned `HCH_UECSTD_OMPB_MINIMAL_FIDELITY_PARTIAL_OR_FAILED`. U-A1 ETTh1/TimeMixer/336 finished, but Host MSE `0.596529` is ~28.3% above paper `0.465`, so Host fidelity fails before UEC can be judged; UEC MSE `0.588395` gives only 1.36% gain vs paper ~3.44%. U-A2 Weather did not complete and its partial checkpoint is excluded. OMPB was never entered, no `OMPB_TARGET_REGISTRY.json` exists, and OMPB/HCH-transfer data access is zero.

Do not write that all baselines are fully reproduced. Current status: δ-Adapter Ada-Y = accepted with disclosed partial exact-anchor fidelity; PIR = exact paper-faithful accepted; COSA = online partial fidelity / paper-native unresolved; UEC-STD = not paper-faithfully reproduced; OMPB = not reproduced. The mandatory offline HCH comparison layer remains sufficient (`Host / direct residual / δ-Adapter / PIR`). If OMPB is still desired, decouple it from UEC and run it independently. Revisit UEC only by first reproducing the paper TimeMixer Host/data/config, not by tuning UEC.

## Baseline-fidelity handoff — mandatory offline anchors closed; minimal UEC-STD + OMPB breadth pass authorized (2026-09-10)

The mandatory offline headline layer remains sufficient and frozen: δ-Adapter Ada-Y is `DELTA_ADAPTER_ANCHOR_ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` (2/3 exact anchors PASS, Electricity borderline FAIL retained), and PIR is `PIR_FULL_OFFICIAL_ANCHOR_ACCEPTED / PAPER_FAITHFUL_EXACT_ACCEPTED` on both mandatory anchors. Do not rerun either. COSA remains `COSA_ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`; do not resume it in this stage.

The user requested one final bounded reproduction pass for UEC-STD and OMPB. The first UEC U0 preflight correctly failed closed as `HCH_UECSTD_PROTOCOL_BLOCKED` before any data read/training because repository helper instructions conflict. Human paper/appendix adjudication on 2026-09-11 now resolves the two anchors without result-conditioned search. Controlling addendum: `docs/current/HCH_UECSTD_PAPER_AUTHORITY_RESOLUTION_20260911.md`; new launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_RESUME_UECSTD_THEN_OMPB_AFTER_PAPER_ADJUDICATION_20260911.md`.

UEC-STD old `src/baselines/uec_std.py` remains pilot-only. Exact anchor protocol is now frozen from the paper: 70/10/20 backbone split; correction data from validation with 70/30 UEC train/tune; 100 steps; batch 64; Huber/SmoothL1; true AR; official two-stage MLP `ecm=linear`; kernel 25. ETTh1/TimeMixer/336 uses seasonal/trend 0.8/0.2 and paper beta 0.3; Weather/TimeMixer/336 uses 0.6/0.4 and paper beta 0.1. Repository auto-beta selection is diagnostic only and must not drive test-result tuning. Exactly two anchors; no rescue/third cell.

After the UEC summary, OMPB may proceed. OMPB must remain an online source→target distribution-shift method. Preferred exact anchors are TCN/96 `ETTh1→ETTh2` and `Weather_train→Weather_test_far`; exact final-paper numeric rows must be extracted/hash-frozen before training. Maximum two anchors per method. No HCH transfer and no δ/PIR/COSA rerun.

Closure: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/BASELINE_FIDELITY_WORKSTREAM_CLOSURE_20260910.md`.

## Parallel handoff — baseline paper-fidelity reproduction opened (2026-09-10)

This is a **parallel baseline workstream**, not a replacement for the current HCH Anchored Compact Amplitude stage. HCH method/task/splits/protected-final access remain unchanged.

Read `docs/current/HCH_BASELINE_PAPER_FIDELITY_AUDIT_AND_REPRODUCTION_DESIGN_20260910.md` and `docs/current/HCH_BASELINE_PAPER_FIDELITY_EXECUTION_PROTOCOL_20260910.md` before baseline work. Historical `src/baselines/delta_adapter.py`, `pir.py`, and `uec_std.py` are pilot/proxy artifacts and must remain replay-compatible; do not overwrite them to create a final baseline. New source-faithful implementations belong under `src/baselines/paper_fidelity/**`.

Major audit findings: current Ada-Y changes official delta scaling/LR/width/training semantics; current PIR replaces the official Transformer refiner/retrieval-value/quality-loss path with ridge/proxy logic; current UEC wrapper does not reproduce its source error-correction training pipeline. Therefore historical baseline results cannot support final SOTA claims.

Fidelity rule: if exact paper dataset+backbone+horizon/protocol is available, validate both Host and baseline raw metrics plus relative gain. If exact anchors are unavailable, derive a same-backbone/closest-horizon paper effect-size interval and freeze it **before** reading unified-DA transfer results. COSA stays in a separately labeled online/TTA track; OMPB requires a distribution-shift online track; NetBurst is Related Work only.

Short launcher: `experiments/current/hch_baseline_paper_fidelity/CODEX_GOAL_PROMPT_20260910.md`. New evidence only under `experiments/evidence/hch_baseline_paper_fidelity_20260910/**`. Stop for human adjudication after the workstream token; do not auto-edit the paper main result table.

## Parallel handoff — baseline paper-fidelity anchor reproduction registered (2026-09-10)

Baseline reproduction is now a parallel paper-admission track; it does not replace the active HCH method closure. Read `docs/current/HCH_BASELINE_PAPER_FIDELITY_REPRODUCTION_AUDIT_20260910.md` before touching external baselines.

Do not treat the current 10-epoch development adapters as faithful SOTA baselines. `src/baselines/delta_adapter.py` is a cache-compatible Ada-Y pilot with material training/protocol changes; `src/baselines/pir.py` is a limited PIR proxy whose official Refiner is replaced by ridge and whose retrieval/revision path is simplified. More epochs alone do not fix either implementation.

Reproduction order is paper-native anchor first, HCH transfer second. δ-Adapter Ada-Y and full PIR are Tier-A offline priorities. COSA/OMPB belong to a separately labeled online/TTA track and must preserve predict-before-update/source-target-shift semantics. Host paper metrics are Gate 0: if the unadapted paper backbone is not within the registered tolerance, do not judge the adapter.

Short launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_GOAL_PROMPT_20260910.md`. New evidence only under `experiments/evidence/hch_baseline_paper_fidelity_20260910/**`. Stop after paper-anchor reproduction for human adjudication; do not automatically run HCH full-transfer/final results.

## Window handoff — Unified-DA S1 passes; Anchored Compact Amplitude is active (2026-09-10)

Main-task correction is canonical: all headline markets, including public Chinese provincial data, forecast next-24h **day-ahead price**. Do not reuse `GANSU_RT` as a main paper target. Public China targets are `GANSU_DA / SHAANXI_DA / NINGXIA_DA / QINGHAI_DA`; target-day `日前电价` is forbidden from input. Existing RT evidence is historical diagnostic only.

The Unified-DA Shape Engineering canary completed validly. Read `experiments/evidence/hch_unified_da_shape_engineering_20260910/adjudication/INDEPENDENT_UNIFIED_DA_SHAPE_ADJUDICATION.md` before interpreting raw artifacts. S1 Daily-Patch GRU32 Shape is the leading Shape candidate: 5/6 Host-nonworse, 4/6 >=3% gain, 3/6 >=5%, median gain 5.85%, maximum 10.74%, max Normal relative harm 0.85%, and 4/6 wins over the fixed development comparator. Gains are stable in five cells. PJM/PatchTST is a real failure: about -1.39%, +0.22%, -0.87% across seeds with consistently worse Shape cosine than S0.

Do **not** add more Shape depth next. The same S1 directions have exact per-day Amplitude O2 gains of ~10.8--22.7%, leaving ~8.5--13.5 pp extra headroom in every cell. This newly satisfies the prior condition for reopening one compact conditional Amplitude after Shape improvement. PJM/PatchTST itself can reach 12.23% with the same S1 direction and oracle scalar distance, so deeper Shape is not the smallest justified repair.

Active design: `docs/current/HCH_ANCHORED_COMPACT_AMPLITUDE_DESIGN_20260910.md`. Freeze S1 Shape/Host/task/splits/inputs/OOF semantics. Test exactly one 26-param Amplitude on `[u || ||v||]`, initialized to reproduce the safe pooled scalar `alpha` and trained only from chronological OOF S1 Shape under direct MAE ray loss. Learned Verification remains removed. Old 648-D Amplitude, raw-state Amplitude, hidden layer, Gate, Bridge/router/retrieval, Transformer/attention and hyperparameter search remain forbidden.

Current development comparators remain Host, matched direct residual, Ada-Y 10 epochs, maintained PIR proxy 10 epochs/K=10. These are development bars only; final paper-faithful baseline reproduction is a separate later stage.

ICDE-facing engineering should emphasize leakage-safe residual provenance, same-task data manifests, forecast-time state joins, day/horizon-aligned repair representation, chronological OOF calibration, lightweight parameter sharing and efficiency reporting—not module count. Supporting note: `paper/04_literature/ICDE_TIME_SERIES_ENGINEERING_POSITIONING_20260910.md`.

Short launcher: `experiments/current/hch_anchored_compact_amplitude/CODEX_GOAL_PROMPT_20260910.md`. New evidence only under `experiments/evidence/hch_anchored_compact_amplitude_20260910/**`. Stop after adjudication token; do not auto-open full China panel, Shandong, final baselines or protected/final roles.

## Window handoff — compact Verification removed; materiality + market-structure audit active (2026-09-09)

The Compact Verification closure completed validly with `HCH_COMPACT_VERIFICATION_CLOSURE_COMPLETE_FOR_ADJUDICATION`; controlling interpretation is `experiments/evidence/hch_compact_verification_closure_20260909/adjudication/INDEPENDENT_COMPACT_VERIFICATION_ADJUDICATION.md`. The 49-param verifier fails against calibrated always-repair (non-worse only 2/6; inverse B4 utility ordering 4/6) and is removed. Do not rescue Verification.

The current leading proposal is still the simple calibrated ray `Delta=s*alpha*u_hat`, but treat it only as a proof of concept. Its six-cell relative Overall-MAE improvements versus Host are approximately `0.11%, 1.12%, 1.30%, 0.49%, 0.76%, 5.34%`; only one cell has strong several-point gain. Current bottleneck is **material gain**, not safety.

Do not claim that GANSU proves Chinese/emerging markets intrinsically break adapters. The canary comparators used 2 training epochs, while vendored official delta-Adapter/PIR defaults are about 10, PIR is a cache-compatible proxy with refiner substitution, and GANSU's DA->RT task has legal current-day DA price/load/renewables/bidding-space information not fully consumed by the frozen univariate Host or pilot adapters. Existing diagnostic tail concentration is not larger in GANSU than DE/PJM, so a generic “more long-tailed” story is specifically unsupported.

The only active next stage is `docs/current/HCH_MATERIAL_GAIN_AND_MARKET_STRUCTURE_AUDIT_20260909.md`. It keeps the same six cells and does not change HCH modules. It audits baseline convergence/fidelity, decomposes current headroom into Verification/per-day-Amplitude/Shape, tests whether legal market-state information explains GANSU residual/correction behavior, and runs only one diagnostic P-State linear Shape with added forecast-time legal covariates. Stop after audit. GRU, conditional Amplitude, learned Gate, full panel, Shandong and protected/final remain closed.

Short executor launcher: `experiments/current/hch_material_gain_market_audit/CODEX_GOAL_PROMPT_20260909.md`.

## Window handoff — Calibrated-Ray passes; conditional Amplitude removed; compact Verification closure active (2026-09-09)

Read `experiments/evidence/hch_calibrated_ray_closure_20260909/adjudication/INDEPENDENT_CALIBRATED_RAY_ADJUDICATION.md` first after canonical state.

The fixed six-cell Calibrated-Ray closure is valid and passes the development gate. The leading proposal is now `Delta=s*alpha*u_hat`: linear 24h Shape + causal scale + one pooled nonnegative MAE calibration scalar from chronological OOF Shape/residual pairs. It improves Host Overall-MAE in 6/6 cell medians, Tail-MAE in 5/6, and Normal-MAE in all six. The 648-D conditional Amplitude network is removed from the leading method. Do not run M0-Compact Amplitude or GRU.

Verification is not retained for architectural symmetry. It receives exactly one final minimal learnability test because calibrated proposals remain harmful on ~29–48% of eval days and oracle selective headroom is positive in every cell (~0.8–4.7% relative MAE). Active controlling document: `docs/current/HCH_COMPACT_VERIFICATION_CLOSURE_20260909.md`.

The only new learner allowed is `q=sigmoid(w^T[v||delta_tilde]+b)` with 48 inputs / 49 parameters, utility-weighted BCE and fixed `q>0.5`, trained only from chronological calibrated OOF proposals. Proposal/Shape/alpha are frozen. If this compact verifier fails V2/V3, remove learned Verification and do not rescue it with a larger verifier. No full-panel, Shandong, extra markets, MLP/GRU/Transformer, threshold search or protected/final execution is authorized.

Short launcher: `experiments/current/hch_compact_verification_closure/CODEX_GOAL_PROMPT_20260909.md`. New evidence only under `experiments/evidence/hch_compact_verification_closure_20260909/**`. Stop after the closure for human adjudication.

## Window handoff — R2 rejects high-dimensional conditional Amplitude; calibrated-ray closure is active (2026-09-09)

R2 completed validly with `HCH_M0_R2_CANARY_COMPLETE_FOR_ADJUDICATION`. Read `experiments/evidence/hch_minimal_repair_m0_20260909_r2/adjudication/INDEPENDENT_M0_R2_ADJUDICATION.md` before interpreting machine artifacts.

R2 confirms the `J>0` filter was harmful but not the full problem. The all-row 648-D conditional Amplitude learner improves several R1 mechanisms yet remains weak: learned OOF utility is positive in only 2/6 cell medians and learned amplitude has near-zero/negative correlation with the exact ray oracle. The decisive diagnostic is the chronological constant-Amplitude probe: it beats learned R2 in all 6/6 cell medians, and its final always-repair version improves Host Overall-MAE in 17/18 seed×cell runs and all 6/6 cell medians.

The next candidate is simpler, not deeper: `Delta=s_d*alpha*u_hat`, where `alpha` is one exact global normalized MAE calibration scalar fitted only from chronological OOF Shape + DIAG_FIT residuals. Use the pooled weighted-median closed form registered in `docs/current/HCH_CALIBRATED_RAY_CLOSURE_EXPERIMENT_20260909.md`. New executor launcher: `experiments/current/hch_calibrated_ray_closure/CODEX_GOAL_PROMPT_20260909.md`.

The closure must evaluate calibrated **always-repair** first and then compute diagnostic-only oracle selective headroom. Do not train a new verifier in this stage. Verification survives only if calibrated proposals still exhibit material/consistent selective headroom; otherwise simplify/remove it. Do not run M0-Compact or GRU-32 yet. The capacity ladder is documented in `docs/current/HCH_POST_R2_CAPACITY_LADDER_20260909.md`.

The closure uses only the same six consumed development cells and immutable Host/baseline evidence. Evidence root is `experiments/evidence/hch_calibrated_ray_closure_20260909/**`. Full panel, Shandong, raw/S3/S4/protected/final remain closed. Even a strong closure result must stop for human adjudication.

## Window handoff — M0 R1 valid failure; revise Amplitude before any capacity increase (2026-09-09)

The six-cell M0 R1 canary is complete and scientifically valid: `HCH_M0_R1_CANARY_NOT_SUPPORTED_FOR_FULL_EXPANSION` with G0 PASS, G1 PASS, G2 FAIL, G3 PASS, G4 FAIL. Read `experiments/evidence/hch_minimal_repair_m0_20260909_r1/adjudication/INDEPENDENT_M0_R1_ADJUDICATION.md` before using the machine summary.

Do not interpret G3 PASS as method effectiveness. M0 is within 1% of the best admitted post-hoc comparator in 5/6 Overall/Tail cells, but those adapters are themselves usually worse than Host. M0 remains worse than Host on Overall-MAE in all 6/6 cells; full expansion is closed.

Shape remains a supported object: cell-median cosine is positive in all cells (`~0.0876–0.5249`) and positive-hemisphere rate >0.5 in all 6/6, but representation quality is modest outside PJM/TimeMixer.

The primary bottleneck is Amplitude learning, not the scalar representation. Using the same OOF predicted Shape, the exact MAE-optimal nonnegative scalar has positive mean utility in every one of 18 market×Host×seed rows, while learned OOF proposal utility is negative in five of six cell medians. The current `J>0`-only training rule leaves ~27–42% of repair-available states unsupervised even though Softplus emits amplitudes on them at inference. R2 design therefore changes only this semantic: train the same scalar head on **all** repair-available OOF Shape rows under the nonnegative ray-MAE objective, allowing zero distance naturally. See `docs/current/HCH_M0_R2_RAY_PROJECTED_AMPLITUDE_DESIGN_20260909.md`.

Do not enlarge Verification first: G4 fails for the current poor proposal distribution, so verifier capacity is not isolated. Do not jump directly to a generic GRU/MLP either. The registered post-R2 ladder is `M0-R2 -> M0-Compact -> M1-GRU32`. M0-Compact reuses the fixed-coordinate 24-D Shape raw vector `v` as the only downstream repair state, reducing Amplitude/Verification to about 25/49 parameters. Only if representation remains limiting may GRU-32 replace `x->v`; downstream heads still consume `v/proposal`, avoiding fold-specific latent-space misalignment. See `docs/current/HCH_POST_R2_CAPACITY_LADDER_20260909.md`.

R2 is now explicitly authorized through `docs/current/HCH_M0_R2_EXECUTION_PROTOCOL_20260909.md`, with short executor prompt `experiments/current/hch_minimal_repair_m0/CODEX_R2_GOAL_PROMPT_20260909.md`. It uses the same six cells/caches/seeds and changes only Amplitude training from `J>0` rows to all repair-available valid OOF Shape rows. It must also run the single registered chronological constant-Amplitude diagnostic probe, write only to `experiments/evidence/hch_minimal_repair_m0_20260909_r2/**`, and stop at `HCH_M0_R2_CANARY_COMPLETE_FOR_ADJUDICATION` or invalid. Historical diagnostic S3/S4 and all protected/final roles remain closed.

## Window handoff — M0 first preflight invalid; R1 global residual-mask repair active (2026-09-09)

The first six-cell M0 canary correctly stopped at G0 with `HCH_M0_CANARY_INVALID` before source implementation or scientific execution. LAGO_DE/LAGO_PJM have complete 168h original-origin historical Host-residual coverage; GANSU_RT does not (`30/62` DIAG_FIT and `4/21` DIAG_EVAL rows incomplete for each Host). The missing blocks are identical across PatchTST/TimeMixer, so no Shape/Amplitude/Verification conclusion exists. Raw/S3/S4/protected/final reads were zero.

The controlling Methodology is unchanged, but the input contract is repaired globally: every market now supplies a 168h residual-availability mask. Partial residual histories are retained with zero-fill only under the mask; causal scale uses available residuals only; zero available residuals cause deterministic exact Keep Host while the row remains in evaluation metrics. This is an input-availability rule, not learned Verification. Complete-history markets carry all-one masks. The revised M0 state is 624-D.

Read, in order, `docs/current/HCH_MINIMAL_INTEGRATED_REPAIR_METHODOLOGY_20260909.md`, `docs/current/HCH_MINIMAL_REPAIR_IMPLEMENTATION_SPEC_20260909.md`, and `docs/current/HCH_M0_INTEGRATED_DEVELOPMENT_EXPERIMENT_20260909.md`. The active code target remains a fresh `src/mvp/hch_minimal_repair/**` package with exactly three tiny heads: Linear `624->24` Shape + normalization; Linear+Softplus one-scalar Amplitude on `[x||u]`; Linear+Sigmoid proposal-aware Verification on `[x||delta||concentration||amplitude]`. Training remains stage-wise chronological stacked OOF.

The fixed R1 canary remains `LAGO_DE / LAGO_PJM / GANSU_RT × PatchTST / TimeMixer`. Preserve the first invalid evidence under `experiments/evidence/hch_minimal_repair_m0_20260909/**`; R1 writes only to `experiments/evidence/hch_minimal_repair_m0_20260909_r1/**`. Even `HCH_M0_R1_CANARY_PASS_FULL_DEVELOPMENT_DESIGN_ALLOWED` only permits a new design discussion; do not auto-run full PAPER8/China, Shandong, M1, new data or protected/final evaluation.

The user has also authorized paper prewriting. Current manuscript skeleton: `paper/_drafts/HCH_FORECAST_REPAIR_PAPER_DRAFT_20260909.md`. The desired story—simple adapters struggle more in emerging Chinese market regimes, the new repairer improves them, and remains competitive on classic international benchmarks—is registered as `PENDING-EVIDENCE`, not an accepted result. Preserve diagnostic-supported Introduction facts, but do not remove pending markers until the corresponding experiment exists.

For Codex use only `experiments/current/hch_minimal_repair_m0/CODEX_GOAL_PROMPT_20260909.md`. Stop after the canary and return G0-G4 plus artifacts for human scientific adjudication.

## Window handoff — integrated minimal Methodology derived; clean M0 development design is next (2026-09-09)

Read `docs/current/HCH_MINIMAL_INTEGRATED_REPAIR_METHODOLOGY_20260909.md` after the canonical state files. The three separate mathematical objects are now integrated into one minimal paper-level method: `Frozen Host -> causal affine-normalized legal state -> normalized 24h Directional Shape -> one scalar MAE Amplitude -> concrete proposal -> utility-weighted hard Verification`.

Core input is intentionally universal: current 24h frozen-Host forecast, previous 168h revealed actual prices/residuals, fixed calendar coordinates and masks. Market-specific DA fundamentals belong only to an enhanced-information setting. Causal coordinates use `m_d=median(past actual)` and `s_d=mean(abs(past residual))`; under `y'=c y+b`, `c>0`, normalized state/Shape and normalized MAE utility are unchanged while physical Amplitude/proposal scale by `c`, giving positive affine equivariance without Market-ID/Host-ID.

Overall MAE is now the primary optimization/model-selection utility; Tail-MAE is a required key secondary metric. MSE remains a Shape-geometry diagnostic. Under MAE, positive fixed-Shape repairability is governed by the one-sided directional derivative `sum_h u_h sign(e_h)>0`; Amplitude training must use chronological OOF predicted Shape and direct directional MAE, not a clipped zero target on bad directions.

Default capacity is **M0**: flatten the fixed normalized legal tensor; linear 24h Shape readout + normalization; linear+Softplus scalar Amplitude conditioned on stop-gradient Shape; linear sigmoid proposal-utility verifier. **M1** is the only allowed capacity escalation: one shared GRU layer, hidden size 32, with the same heads, and only if a clean development experiment shows M0 representation underfit. No Transformer/Bridge/router/retrieval/quantile/Bayesian expansion is authorized.

Training must be chronological stacked OOF: OOF Shape -> Amplitude; OOF Shape+Amplitude -> complete OOF proposal; complete OOF proposal -> Verification utility. Stage-wise training is the default; no arbitrary combined multitask loss. The next task is to design one clean integrated-development experiment for M0 with explicit falsification/upgrade rules. Do not execute it automatically and do not reuse the contaminated diagnostic S3/S4 slices as final truth.

## Window handoff — Shape/Amplitude/Verification mathematics derived; integrated minimal method is next (2026-09-09)

Read the three current design documents in order after the canonical state files: `HCH_DIRECTIONAL_SHAPE_MATHEMATICAL_DESIGN_20260908.md`, `HCH_CONDITIONAL_AMPLITUDE_MATHEMATICAL_DESIGN_20260909.md`, and `HCH_PROPOSAL_UTILITY_VERIFICATION_MATHEMATICAL_DESIGN_20260909.md`.

The surviving paper-level chain is now mathematically explicit: `Frozen Host -> normalized 24h residual Shape -> one identifiable nonnegative scalar Amplitude conditioned on predicted Shape -> concrete proposal Delta -> one proposal-aware utility verifier -> hard Host-or-repair action`. Shape owns signed relative horizon geometry; Amplitude owns only scalar distance along that ray; Verification owns whether the complete proposal should execute. Horizon-wise Amplitude, Bridge/router/CORN/quantile machinery, generic soft shrinkage and market-specific experts remain non-default.

Verification now has a decision-theoretic target. With realized utility `G=L(Host)-L(Host+Delta)`, Bayes execution is `1[E[G|x,Delta]>0]`. The leading verifier objective is utility-weighted BCE: label `1[G>0]`, weight `|G|`, one linear sigmoid head. Its population optimum satisfies `2q*-1=E[G|x,Delta]/E[|G||x,Delta]`, so fixed threshold `q>0.5` is exactly aligned with positive expected utility. This is not ordinary `P(G>0)` and not Host uncertainty.

All downstream supervision must be chronological OOF: Shape OOF predictions define Amplitude supervision; OOF Shape+Amplitude define complete OOF proposals; only those proposals define Verification utility labels. The next task is no longer another component derivation but **integrated minimal Methodology design**: exact legal input tensor, smallest shared temporal encoder, sequential/cross-fitted training algorithm, primary headline metric (MAE vs MSE), and one minimal development experiment that tests linear heads before any nonlinear expansion. No new scientific execution or final implementation is automatically authorized.

## Window handoff — Conditional scalar Amplitude derived; Verification math is next (2026-09-09)

Read `docs/current/HCH_CONDITIONAL_AMPLITUDE_MATHEMATICAL_DESIGN_20260909.md` after the canonical state files and the Shape design. Shape is already fixed conceptually as the normalized 24h frozen-Host residual direction. Given predicted unit Shape `u_hat`, the default Amplitude object is now one **identifiable nonnegative scalar** `a`, with candidate `Delta=a u_hat`. A free horizon-wise Amplitude vector is non-default because it can reallocate relative hour-to-hour magnitude away from Shape and destroys the clean polar decomposition.

For a fixed Shape, define `a*_ell(e,u)=argmin_{a>=0} ell(e,a u)`. Under MSE, `a*=[e^T u]_+`; on positive-alignment rows, `(a-a*)^2` is exactly the fixed-Shape MSE repair regret. Under MAE, the exact scalar is the nonnegative weighted median of `e_h/u_h` with weights `|u_h|`, equivalently the optimizer of `||e-a u||_1`. If Shape is perfect, both targets equal `||e||_2`, so MSE-vs-MAE amplitude disagreement is caused by Shape error rather than by the scalar factorization.

To keep Shape / Amplitude / Verification non-redundant, do not train Amplitude on wrong-direction rows with clipped target zero. Amplitude supervision must be generated from chronological OOF predicted Shape and restricted to a usable-direction condition; wrong-direction/no-op decisions remain for Verification. The minimal Amplitude head is one positive scalar readout, preferably linear + Softplus on `[shared state || stop-gradient predicted Shape]`; physical-unit scaling should reuse causal preprocessing scale, not create a market router.

The unresolved Amplitude choice is only the primary loss instantiation: MSE projection has the clean exact regret identity; MAE projection aligns exactly with MAE via weighted-median/L1 projection. Freeze this choice together with the headline metric before protected evaluation. The next mathematical task is **proposal-level Verification**: derive whether `P(G>0)` or `E[G]` is the cleaner Bayes target and define the smallest Host-vs-repair decision rule from chronological OOF complete proposals. No new scientific execution or final HCH implementation is authorized.

## Window handoff — Directional Shape math derived; Amplitude is next (2026-09-08)

The current design discussion has moved from diagnostic adjudication into Shape mathematics. Read `docs/current/HCH_DIRECTIONAL_SHAPE_MATHEMATICAL_DESIGN_20260908.md` after the canonical state files. Shape is now defined as the signed normalized 24h frozen-Host residual `u*=e/||e||_2`, not independent sign labels. Diagnostics support horizon geometry beyond sign-only repair. For candidate `Delta=a u_hat`, with `kappa=<u*,u_hat>`, exact squared-error gain is `G=2 a ||e|| kappa-a^2`; therefore wrong-hemisphere Shape cannot be repaired by a positive scalar amplitude, the MSE-optimal scalar is `[e^T u_hat]_+`, and maximum recoverable MSE headroom is proportional to `[kappa_+]^2`.

Leading minimal Shape learner: shared legal state -> one horizon-vector readout `v`; train `v` by squared regression to unit target `u*`; normalize only for the candidate `u_hat=v/(||v||+eps)`. The population output `E[u*|z]` yields both a conditional mean direction and a mean-resultant-length-style directional concentration in `||v||`, avoiding extra sign/confidence heads. Do not use this concentration as the final gate. Do not default to residual-norm weighting because Shape should remain magnitude-free.

The next mathematical task is **Amplitude under predicted Shape**, not Shape implementation. Derive both MSE projection amplitude and an MAE-consistent scalar optimum, then decide whether one scalar remains the clean paper object. Any Amplitude labels must be built from chronological OOF Shape predictions, not oracle directions. No new experiment or final-method implementation is authorized.

## Window handoff — repair diagnostics adjudicated; next is mathematical target derivation (2026-09-08)

The D0–D4 Forecast-Repair Diagnostic Evidence Program is complete. Do **not** rerun the original machine verdict blindly. Read `experiments/evidence/hch_repair_diagnostics_20260908/adjudication/INDEPENDENT_DIAGNOSTIC_ADJUDICATION.md` and `adjudication/ADJUDICATED_VERDICT.json` before the machine-generated `verdict.json`; the adjudication supersedes the latter where they differ.

Surviving scientific objects: (1) Host-error heterogeneity is supported; (2) Shape/Alignment is supported and should be formulated as a normalized 24h horizon direction, not merely a repair-occurrence/sign bit; (3) Amplitude is supported as a separate object, but scalar-vs-horizon-wise remains unresolved because the original `scalar capture=2.9670` metric was misdefined; (4) proposal-level Verification/abstention has strong oracle headroom, but learned Verification is not yet validated because the original labels were in-sample rather than OOF. Key adjudicated numbers: D1 21/24 public Host cells, median top-5% MSE concentration 0.4071; D2 median WDR 0.4693, O-SIGN incremental recovery positive 144/144 and O-DIR extra recovery positive 144/144; D3 direction-correct overshoot 4.41%, strong under-correction 83.29%, corrected scalar recovery fraction 32.64% under current simple-candidate directions; D4 harmful rate 60.28%, oracle headroom positive 144/144, always-repair beats Host 53/144 vs oracle selective 130/144, Host-badness AUROC 0.6263.

Introduction evidence currently safe: I1 Host-error heterogeneity SUPPORTED; I2 wrong-direction material SUPPORTED; I3 correct direction insufficient SUPPORTED/qualified; I4 Host-badness differs from proposal utility SUPPORTED. I5 market/regime repair-difficulty spectrum and I6 China-stress extension are UNRESOLVED because the machine runner did not implement the required regime-conditioned D2–D4 attribution. Use `adjudication/INTRO_EVIDENCE_REGISTER_ADJUDICATED.md` for exact allowed/forbidden wording.

Immediate next design sequence, with no new execution automatically authorized: derive the normalized 24h Shape target/loss; obtain chronological OOF predicted Shape; derive Amplitude supervision conditional on that predicted direction (leading scalar target `a*=[e^T u_hat]_+`, but compare against the smallest horizon-wise alternative under the same Shape); construct complete OOF proposals; only then derive/test proposal-utility Verification labels. The final method should remain `Frozen Host -> Shape -> Amplitude -> proposal -> Verification`, with no Bridge/router/CORN/quantile/retrieval/Bayesian machinery unless later evidence specifically requires it.

Governance correction: the diagnostic raw loader materialized full target columns before assigning its new S1/S2/S3/S4 labels. Those newly defined S3/S4 slices are therefore consumed/contaminated under strict project semantics and must not be reused as untouched final truth.

## Window handoff — D0 diagnostics blocked on Host foundation / target semantics (2026-09-08)

The first D0 execution correctly failed closed with `HCH_REPAIR_DIAGNOSTICS_BLOCKED_PREFLIGHT`: `0/26` maintained frozen Host caches exist for `PAPER8 + CHINA5` × `PatchTST/TimeMixer`; raw scientific observations and protected/reserved partitions were not opened; D1–D4 did not run. Treat this as a protocol/infrastructure blocker, not evidence against the repair hypothesis.

Audit conclusion: the current controlling design mistakenly made a **pre-existing cache** a prerequisite. Existing benchmark-foundation code already supports `train a preregistered Host -> freeze it -> emit immutable HostPredictionCache`, so a corrected protocol may authorize a diagnostic-local Host-foundation phase without violating the frozen-Host post-processing identity. Do not overwrite the historical `formal_host_cache_multiyear_v1` tree or tune Host settings from diagnostic outcomes.

Before authorizing that cache build, resolve two scientific-data contracts. (1) Chinese target semantics are not frozen in the new diagnostic design: archived CN-ERB defines day-ahead-information -> next-24h RT price, while PAPER8 is DA-price forecasting; mixing them directly would confound market difficulty with task difficulty. (2) timezone/DST metadata is currently unspecified in the diagnostic registry rows, so episode construction must bind to existing canonical benchmark timestamp semantics or an explicit new contract. Until these are frozen, do not rerun D0 by merely relaxing the cache existence check. `EXPERIMENT_LEDGER.md` remains unchanged.

## Window handoff — diagnostic-evidence branch now controls method necessity + Introduction evidence (2026-09-08)

The diagnostic branch has been refined after the first D0 blocker. The immediate executor roadmap is `experiments/current/hch_repair_diagnostics/DIAGNOSTIC_TODO_AND_DECISION_TREE.md`; for Codex use the short launcher `experiments/current/hch_repair_diagnostics/CODEX_GOAL_PROMPT_20260908.md`. The goal is not to make HCH win but to determine, by counterfactual evidence, whether Shape, Amplitude, and Verification deserve to exist and to generate two reviewable outputs: `INTRO_EVIDENCE_REGISTER.md` and `METHOD_DESIGN_HANDOFF.md`.

D0 now runs `D0-A task/target/time contract -> D0-H fixed Host train/freeze/cache -> D0-C leakage closure`. Missing cache alone no longer blocks execution. Host training is S1-only using fixed maintained PatchTST/TimeMixer configs; after freeze, D1-D4 use only S2 (`DIAG_FIT=first 75%`, `DIAG_EVAL=last 25%`), with S3/S4 closed. PAPER8 DA and CHINA5 CN-ERB DA→RT remain separate task tracks.

Core diagnostic ladder: D1 error heterogeneity; D2 wrong-direction plus O-SIGN/O-DIR oracle recovery; D3 amplitude recovery conditional on positive alignment plus scalar-vs-horizonwise implication; D4 proposal utility, oracle abstention, Host-badness sufficiency falsification, and minimal logistic/tiny-MLP learnability. Negative evidence must simplify the method. No final HCH coding follows automatically after completion; results return to the math/design window first.

## Window handoff — theory-first repair diagnostics registered (2026-09-08)

The active next stage is no longer historical V3/V3.2 repair or another literature pass. The user has authorized one **D0–D4 diagnostic-evidence program** to determine the minimal final Forecast Repairer before any new HCH architecture is implemented. Read `docs/current/HCH_REPAIR_DIAGNOSTIC_EVIDENCE_DESIGN_20260908.md` as the controlling scientific design and use `experiments/current/hch_repair_diagnostics/DIAGNOSTIC_EXECUTOR_PROMPT_20260908.md` only as the short Luna-High launcher.

The current theory-first decomposition starts from `G_MSE = 2 e^T Delta - ||Delta||^2`. With `Delta = a u`, Shape/Alignment determines `e^T u`; conditional Amplitude asks how far to move along that chosen direction, with optimal scalar projection `a* = e^T u` when alignment is positive; Verification, if evidence supports it, must estimate the utility of the **specific proposed correction** relative to keeping the Host. Shape therefore precedes Amplitude conceptually. The final architecture is deliberately not frozen: D2 may remove Shape, D3 may select scalar-projection vs horizon-wise Amplitude, and D4 may remove Verification.

The diagnostic matrix is frozen to the historical PAPER8 international panel (`GEFCOM14P / LAGO_BE / LAGO_DE / LAGO_FR / LAGO_NP / LAGO_PJM / NEM_SA1 / NORD_DK1`) plus currently registered Chinese stress cases (`GANSU_RT / SHAANXI_RT / NINGXIA_RT / QINGHAI_RT / SHANDONG`), with PatchTST/TimeMixer as the first-pass frozen Hosts. Shandong is private supplementary only. Shanxi is **not registered in the current repository** and must not be invented/substituted. The intended `moderate -> extreme repair difficulty` story is only a hypothesis; D1–D4 must determine whether such a spectrum actually appears.

The four scientific diagnostics are: D1 Host-error heterogeneity/tail concentration; D2 wrong-direction/alignment failure for simple residual correctors plus existing δ-Adapter/PIR wrappers; D3 overshoot and optimal-projection amplitude analysis on positively aligned candidates; D4 candidate utility distribution, oracle selective headroom, then minimal logistic/tiny-MLP benefit learnability only if headroom exists. An unsupported object must be removed/simplified, not rescued by architecture growth.

Execution boundary: implementation only under `experiments/current/hch_repair_diagnostics/**`; evidence only under `experiments/evidence/hch_repair_diagnostics_20260908/**`; no edits to `src/**`, `data/**`, historical evidence or `paper/**`; no protected/reserved partition access; stop after `HCH_REPAIR_DIAGNOSTICS_COMPLETE` and return results for design adjudication. `EXPERIMENT_LEDGER.md` remains unchanged until an actual diagnostic execution completes.

Historical V3.2 A+B remains a pre-execution scientific record and is **not** the default next run. Do not finish/execute it merely because its folder remains under `experiments/current/`.

## Window handoff — unified signed-repair redesign discussion (2026-09-08)

New-window priority is **design discussion only; no implementation or experiment authorization**. Read `RESEARCH_STATE.md`, `EXPERIMENT_LEDGER.md`, and this file first. Then read `docs/current/HCH_UNIFIED_SIGNED_REPAIR_METHOD_DESIGN_20260907.md`, `docs/current/HCH_UNIFIED_SIGNED_REPAIR_EXTERNAL_REVIEW_ADJUDICATION_20260907.md`, and the simplified reviewer prompt `docs/current/HCH_ICDE_STRICT_REVIEW_PROMPT_20260907.md` only as needed.

Current paper-first design principles are settled: one complete legal Host-error tensor; one shared encoder; no handcrafted branch-exclusive feature inventories; Shape/Amplitude specialization should be learned from their tasks; shared encoder already provides sharing, so no Bridge by default; avoid architecture inflation. The latest discussion is moving away from the soft candidate `c=p*mu_plus-(1-p)*mu_minus` plus multiplicative shrinkage Gate because of conditional-mean/MAE mismatch and Gate-Amplitude identifiability. The leading **discussion candidate, not yet frozen**, is: Shape predicts repair direction, Amplitude predicts directional magnitude, these define an explicit candidate action `a`, and a proposal-only Counterfactual Utility Gate predicts the MAE benefit `u=|r|-|r-a|` and applies the candidate only when predicted utility is positive (or under a later minimal safety rule if evidence requires it). Gate must not read raw `X`/latent state by default. This idea should be attacked mathematically and against close prior art before any method document rewrite.

Important literature boundary discovered in the latest discussion: frozen-Host lightweight residual post-processing is not itself novel; ICLR 2026 `δ-Adapter` is a strong direct prior for frozen-backbone residual correction/stability, while a recent residual-correction utility-gating work (MURECAST) is a close prior for proposal utility validation. Therefore novelty must come from the **problem formulation and evidence** (e.g. direction–magnitude factorization plus intervention-utility validation under one market-/Host-agnostic recipe), not from calling the module a plugin/gate. ICDE fit should be judged from actual contribution, not by adding database wrappers; recent ICDE accepted work includes direct time-series representation/forecasting methods, so venue fit is plausible only if the paper establishes a general time-series/data-centric insight beyond one electricity dataset.

Before the next reviewer-facing revision, preserve strict separation: author method documents contain only the current design and exact reproducibility specification; reviewer prompts contain evaluation principles only and must not pre-seed expected flaws. Historical V3/V3.2 remains scientific record, not the default paper architecture.

## Paper-first unified signed-repair design ready for strict external review (2026-09-07)

The current architecture discussion has been consolidated into `docs/current/HCH_UNIFIED_SIGNED_REPAIR_METHOD_DESIGN_20260907.md`. It is a design candidate only: one unified causal Host-error tensor, one shared encoder, learned Shape(direction) and Amplitude(sign-conditional magnitude) heads, differentiable candidate repair `c=p*mu_plus-(1-p)*mu_minus`, one minimal proposal-only Benefit Gate, and an end-to-end normalized overall + fixed tail-emphasis task loss. Historical V3 feature inventories/Bridge/CORN/quantile machinery/target-local ladder are not default components of this candidate. No implementation or experiment is authorized by the design.

Use `docs/current/HCH_ICDE_STRICT_REVIEW_PROMPT_20260907.md` in a fresh window for independent scoring before coding. The prompt is now intentionally neutral: it evaluates problem/method novelty, evidence-claim matching, writing/boundaries, scientific insight/completeness/workload, then assigns a contribution-vs-defect ICDE score without pre-seeding expected flaws. Reviewer-facing method material must contain only the current design plus exact reproducibility details; historical alternatives/internal risk lists stay out of the author-facing review document. ICDE 2027 explicitly lists time-series/temporal data but still warns that pure ML without data-engineering relation may be out of scope; recent accepted time-series methods show venue fit is plausible but must be earned by the actual contribution.

The current literature bottleneck is sharper: ICLR 2026 `delta-Adapter` already covers frozen-backbone lightweight residual post-processing, and recent MURECAST-style selective correction makes explicit utility gating a nearby prior. The open design discussion is whether Shape(direction) + Amplitude(magnitude) should produce one or two directional candidate repairs and whether the final Host-preservation mechanism should estimate explicit counterfactual MAE utility rather than act as a generic shrinkage coefficient. Do not implement before this is resolved. `EXPERIMENT_LEDGER.md` remains unchanged.

Strict external review has now returned and is synthesized at `docs/current/HCH_UNIFIED_SIGNED_REPAIR_EXTERNAL_REVIEW_ADJUDICATION_20260907.md`. Verdict: `METHOD_PROMISING_BUT_NEEDS_SMALL_REVISION` (6.2/10) and `ICDE_SCOPE_HIGH_DESK_REJECT_RISK`. The reviewer supports the simplified architecture and says not to restore historical complexity. Pre-implementation revisions are limited to: correct Tail row-weight semantics; weaken exact conditional-mean claims under MAE end-to-end training; define Gate as Host-preserving shrinkage rather than calibrated benefit probability; freeze conservative cross-market vocabulary; and fully specify scaler/mask/timezone/DST/reproducibility rules. No implementation/experiment is authorized yet.

## Active paper-design discussion: simplify HCH around structural cross-market generalization (2026-09-07)

User-level design preference is now explicit: target an A-conference-style method whose novelty is scientifically clear and reproducible from the paper, with code serving the method rather than becoming an engineering system. “Cross-market” should not default to large multi-market pretraining/fine-tuning; prefer a market-agnostic, scale-relative/semantically conditioned repair architecture whose same recipe can be trained/evaluated across markets and Hosts and whose cross-market claim is established by consistent gains. Treat the current 7-D Shape view, 6-D Amplitude view, gated Bridge, and target-local adapter ladder as candidate implementations to be scientifically simplified/reconsidered, not as immutable contributions. Current discussion priority is branch input information and temporal alignment; encoder type is secondary.

Current simplification direction: one unified legal Host-error sequence -> one shared encoder -> simple Shape and Amplitude heads. Do not assign handcrafted branch-exclusive inputs. To keep the final selector non-redundant, the conceptual role split under discussion is Shape=direction/sign proposal, Amplitude=conditional magnitude proposal, Gate=benefit/utility decision against keeping Host. Gate should be driven by the final repair objective rather than duplicate a repair-occurrence classifier. This is a design hypothesis only; no scientific execution has validated it yet.

New discussion hypothesis: use one complete legal base tensor for both branches rather than hand-assigning branch-specific features: `z0=Encoder(X)`, then learned `z_s=f_s(z0)` and `z_a=f_a(z0)`. Default to no Bridge because the shared encoder already shares information. Leading head-level interpretation is a signed hurdle/two-part model: Shape separates repair occurrence from conditional sign; Amplitude learns conditional magnitude only on active directional rows. This is discussion/design state only; do not treat it as an executed result or silently replace the frozen V3/V3.2 implementation.

## Paper-first operating constraint and latest pre-execution submission (2026-09-07)

From this point forward, HCH is to be developed as an A-tier paper method, not as an engineering-heavy system. Scientific novelty and a compact reproducible formulation take priority; code should follow the paper closely enough that another group or a capable AI can reproduce the method without depending on repository-specific orchestration. Distinguish paper method from mature implementation choices (GRU/CORN/quantile heads) and from audit/provenance infrastructure. Prefer the smallest scientifically justified mechanism; do not let stronger local executor capability (`solar-medium`) become a reason to add complexity.

Latest local executor submission reports `HCH_V3_2_AB_REPORT_DIAGNOSTICS_REPAIRED_AWAITING_FINAL_NO_EDIT_CLOSURE`, 45/45 tests, manifest `b5455afaa37364c0617711b201869b23c04d70c70ce7dde6792027d813e53868`, zero real scientific reads. Treat this only as a reported pre-execution implementation state until independently closed; it is not an effectiveness result.

## HCH V3.2 A+B no-edit closure: report-diagnostic semantics only; one final narrow repair next (2026-09-07)

The last-mile repair now independently passes `42/42` tests and compileall; manifest SHA `5666ed8d233589e938340e542b649959d8ccf617f6bc0637dd7e76a04a53df8b` self-verifies with `scientific_execution_authorized=false`. All prior execution/data-access blockers are closed: subset-exact reconstruction, V0 no-confirmation, one-read-per-window caching, factual access provenance, selected-only confirmation fitting, and seed-faithful pooled joint-direction gating are present. No real A+B scientific truth has been read.

Final execution authorization is still withheld because Solar's no-edit closure found four report-only mismatches: Experiment-A conditional direction still uses three-class argmax instead of `p_down` vs `p_up`; Experiment-A AP-Lift uses subtraction rather than the canonical ratio; the 12-row Experiment-A seed-median summary omits several registered nested diagnostics; and conditional Up/Down recall/support are not persisted separately from joint recall/support in development/confirmation evidence. These do not change B capacity gates, but they can change mechanism interpretation and cannot be repaired safely after a one-shot real run.

Read `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_NO_EDIT_CLOSURE_20260907.md`. Next run only `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_REPORT_DIAGNOSTIC_FINAL_REPAIR_AI_PROMPT_20260907.md`; all real reads must remain zero. Then perform one final no-edit closure. Do not authorize real A+B yet. `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.2 A+B final re-audit: last-mile execution semantics only; Solar/Luna closure repair next (superseded by 2026-09-07 no-edit closure above, 2026-09-06)

The latest orchestration-repair package independently passes `35/35` tests and compileall; manifest SHA is `10e2fe0a42a4d80bd7d8ab98aece6062c7e2dedfa6881988a3f77339f08aa9a1` and verifies from the canonical absolute root. Common lifecycle, prediction-derived development metrics, six-key confirmation capability, canonical split binding, and positive-parameter semantics are now present. No real A+B scientific truth has been read.

Final execution readiness remains blocked only by last-mile semantics documented in `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_FINAL_REAUDIT_20260906.md`: subset A+B roles still call whole-role SSI `_episodes()` then filter; V0 still opens confirmation; Experiment-A support/eval and Host granularity are wrong; conditional direction and seed-pooled direction aggregation are not exact; scientific windows are reread per seed; real role access provenance/evidence tables are incomplete; and confirmation trains unselected variants. The controlling design now contains explicit pre-execution clarification for these diagnostic/aggregation semantics without changing thresholds/candidates.

Next action is one implementation-only last-mile closure under `docs/current/HCH_V3_2_AB_LAST_MILE_EXECUTION_CLOSURE_PROTOCOL_20260906.md`, executed with repo-local skill `.agents/skills/solar-research-executor/SKILL.md` and prompt `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_LAST_MILE_EXECUTOR_AI_PROMPT_20260906.md`. Do not authorize real A+B yet. `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.2 A+B implementation audit blocked; protocol repair then re-audit (superseded by 2026-09-06 re-audit above)

The SSI Case-B result has now been converted into a structured V3.2 research program rather than an ad-hoc Local-model rescue. Controlling design: `docs/current/HCH_V3_2_TARGET_LOCAL_SHAPE_RESEARCH_PROGRAM_20260905.md`. The architectural refinement is `Frozen cross-market representation -> Shared ER Shape trunk -> tiny target-market Shape adapter -> Repair Activity/Direction`, with Amplitude and Gate still frozen.

The immediate stage is Experiment A+B only, now frozen at `docs/current/HCH_V3_2_AB_SHIFT_AND_CAPACITY_EXPERIMENT_DESIGN_20260905.md`; paired implementation/preflight prompt: `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_IMPLEMENTATION_PREFLIGHT_AI_PROMPT_20260905.md`. A diagnoses target shift. B uses an identifiable capacity ladder: `B0-H` exact SSI Host-specific 3-scalar reference; `B0-M` matched market-shared 3-scalar baseline; `B1` market-shared four-scalar logit-affine; `B2` B0-M plus a zero-init 2x32 latent residual for 67 total learned target parameters; `B3` 2,277-parameter full local CORN diagnostic ceiling. Capacity gates compare against B0-M; B0-H is reference only.

Critical data rule: TARGET_D2 is consumed for V3.2 design because the SSI D2 pattern directly motivated this program. Do not use it again for variant selection/promotion. Development selection must occur on 12 source-side pseudo-target cells: two fixed chronological 56-day support / 28-day evaluation folds per market/Host carved from SOURCE_GTRAIN. Freeze the candidate first, then use SOURCE_GVAL once for six-cell development confirmation. Only after that may a separate protocol discuss fitting TARGET_D1 and opening S3. S2V/S3/S4/fresh remain closed now.

The first implementation/preflight submission genuinely passes its 23 local tests and compileall, with manifest SHA `c1cce2605914fa9174fa8cc20d488e21c1e41d8c43844c9401992a2756a95f38` and zero real scientific reads, but independent audit blocks execution readiness: `HCH_V3_2_AB_IMPLEMENTATION_AUDIT_BLOCKED_PROTOCOL_MISMATCH`. Main blockers are incomplete future real lifecycle, non-exact SSI calibration parameterization, self-consistent rather than canonical split binding, incomplete development/confirmation adjudication, SOURCE_GVAL lock not integrated into fail-before-reader guard, overly broad causal-history phases, and incomplete production trainers/metrics/Experiment-A diagnostics. Audit: `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_IMPLEMENTATION_AUDIT_20260905.md`.

Next action is implementation-only repair using `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_PROTOCOL_REPAIR_AI_PROMPT_20260905.md`, followed by independent re-audit. Do not authorize or execute real A+B yet. Real SOURCE_GTRAIN/SOURCE_GVAL and TARGET_D1/D2/S2V/S3/S4/fresh remain unopened. `EXPERIMENT_LEDGER.md` remains unchanged because no new scientific experiment occurred.

## HCH V3.1 SSI completed: Case B target-transfer failure; target-local Shape adaptation is next discussion (2026-09-05)

The one authorized real SSI execution completed normally and independent adjudication confirms `HCH_V3_1_SSI_SOURCE_SIGNAL_PRESENT_TARGET_TRANSFER_FAILED`. G0 passes. G1 fails because target D2 B1>A1 occurs in only 2/6 cells and the six-cell median B1/A1 ER-AUPRC ratio is `0.944307 < 1.20`; source G-Val consistency passes 3/3 and NormalDiversion improves 4/6 with median `1.0000 -> 0.937291`. G2 is not eligible for promotion adjudication.

Cell-level target-transfer pattern is diagnostic: DE/PT 1.533 and DE/TM 1.492 strongly favor B1, NORD/PT 0.962 and NORD/TM 0.926 modestly favor A1, and PJM/PT 0.729 and PJM/TM 0.779 materially favor A1. Thus the frozen representation is not signal-null, but the current three-scalar D1 Shape adaptation is insufficiently transferable. Report-only pooled target direction macro-F1 is 0 for B1/B2 for all seeds, further indicating target-side posterior/Identity dominance.

Next discussion should isolate **minimal target-local Shape adaptation / sharing** while keeping the upstream representation frozen and retaining extreme-repair semantics as positive source-level evidence. Do not change Amplitude or Gate, do not add class/focal/delta/threshold rescue, and do not open S2V/S3/S4/fresh panel. Controlling scientific audit: `experiments/evidence/hch_v3_1_shape_semantic_identification/INDEPENDENT_SSI_SCIENTIFIC_ADJUDICATION_20260905.md`. `EXPERIMENT_LEDGER.md` now records this completed experiment.

## HCH V3.1 SSI protocol-conformant; fresh execution authorization was the only remaining gate (superseded by authorized-run state above, 2026-09-05)

Independent closure accepts `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`. The workspace/import mismatch is resolved: repo root `D:\作业\science\solar_leak_price_model`, Git HEAD `9111d0f567ee5b6cd5a8b9f378623aa8586fa34e`, manifest SHA `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`, orchestration SHA `148a71671a5d0b674a2212ac4febae255e78a293f95331774551262cb2eb98be`, and Python import origin all match the frozen fingerprint. Full suite independently passes `46/46`; compileall and manifest self-verification pass; historical HCH representation package is `23/23 MATCH`.

Pretruth/provenance closure also passes: canonical real adapter constructs with `audit_attempts=0`, canonical split/threshold/global/H0 bindings pass, negative probes fail-before-reader, source epochs=20, D1 attempts=64, seeds `[7,17,37]`, G-Val calibration updates=0, eligibility/G2/pooled-direction/G0 contracts pass, and final Gate execution=0. No real SSI scientific truth has been read or executed; S2V/S3/S4/fresh remain `0/0/0/0`.

Read `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PROTOCOL_CONFORMANCE_CLOSURE_20260905.md` before any execution. Do **not** run real SSI until the user newly supplies `HCH_V3_1_SSI_EXECUTION_AUTHORIZED` after this closure. Old literals in history/tests are invalid for authorization. `EXPERIMENT_LEDGER.md` remains unchanged until the real SSI experiment is actually completed.

## HCH V3.1 SSI workspace identity mismatch; prove exact checkout/import before any more closure testing (superseded by protocol-conformance closure above, 2026-09-05)

The second no-edit verifier reported the **old** manifest SHA `334f311bd81c17f444d2bc490a19131cec67b8ffc1ef41b5d82b48446e3552c5` and repeated the old synthetic G0 failure. Independent inspection of the active registered checkout shows the opposite: `D:\作业\science\solar_leak_price_model` currently contains `final_gate_not_executed=True`, `orchestration.py` SHA `148a71671a5d0b674a2212ac4febae255e78a293f95331774551262cb2eb98be`, manifest SHA `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`, and Git HEAD `9111d0f567ee5b6cd5a8b9f378623aa8586fa34e`. `git worktree list` shows this as the single registered worktree.

Therefore do **not** interpret the second 45/46 result as another SSI implementation failure. The immediate blocker is workspace/Python-import identity. Use `experiments/current/hch_v3_1_shape_semantic_identification/WORKSPACE_FINGERPRINT_20260905.json`, `docs/current/HCH_V3_1_SSI_WORKSPACE_IDENTITY_AND_FINAL_RECHECK_PROTOCOL_20260905.md`, and `experiments/current/hch_v3_1_shape_semantic_identification/SSI_WORKSPACE_IDENTITY_FINAL_RECHECK_AI_PROMPT_20260905.md`. The verifier must stop before pytest if repo root, Git HEAD, manifest SHA, orchestration SHA/source marker, or imported module path does not match the fingerprint.

Only after exact identity passes may the prior failing runner regression and the full 46-test suite be rerun. No real SSI scientific execution has occurred; real truth reads remain 0; S2V/S3/S4/fresh = 0/0/0/0; `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.1 SSI first no-edit closure failed 45/46; one-line G0 polarity repair frozen; recheck next (superseded by workspace-identity state above, 2026-09-05)

The first independent no-edit verifier correctly stopped at `HCH_V3_1_SSI_FINAL_CLOSURE_NOT_READY`: 46 collected / 45 passed / 1 failed. The sole failure was `test_authorized_safe_synthetic_path_completes_without_gate`, where synthetic `G0_pass=False`. The verifier made no source/manifest edits, opened no real SSI scientific truth, and reported S2V/S3/S4/fresh = 0/0/0/0.

Root cause is independently confirmed as a single G0 predicate-polarity error: the correct factual field `final_gate_action_executed=False` was treated as a success boolean under `all(bool(v))`. The only implementation correction changes the G0 predicate to `final_gate_not_executed=True`; final evidence still records `final_gate_action_executed=False`. No scientific design or real-data path changed.

New frozen manifest SHA-256: `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`; `verify_manifest=True`; 45/45 non-regression tests PASS with the formerly failing runner test deselected; compileall PASS; historical HCH package `23/23 MATCH`. Run the new no-edit recheck: `docs/current/HCH_V3_1_SSI_FINAL_CLOSURE_RECHECK_PROTOCOL_20260905.md` with `experiments/current/hch_v3_1_shape_semantic_identification/SSI_FINAL_CLOSURE_RECHECK_AI_PROMPT_20260905.md`. Only 46/46 PASS with unchanged manifest may return `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`.

## HCH V3.1 SSI protocol-conformance frozen; final no-edit verification next (superseded by recheck state above, 2026-09-05)

The SSI implementation has now been repaired through the previously identified protocol-semantics blockers without executing real science. Current frozen implementation semantics include: held-out source G-Val with zero calibration updates and common ER truth; eligibility-first D1/D2 metrics and K; six-cell seed-median G2 activity/loss aggregation; true pooled six-cell direction diagnostics; computed fail-closed G0; package-owned canonical real adapter with no arbitrary scientific callback/hash seam; canonical split/threshold/global/H0 binding; historical representation package `23/23 MATCH` against the accepted frozen source bundle; strictly-past S1/S2T causal-history guard; and report-only secondary diagnostics.

Current saved manifest SHA-256: `334f311bd81c17f444d2bc490a19131cec67b8ffc1ef41b5d82b48446e3552c5`. `verify_manifest` currently returns True. Test inventory is exactly 46. After the last corrections, 43 non-manifest/non-runner tests pass, the manifest test passes, and compileall passes. The final step is deliberately **no-edit verification** so the verifier cannot move the audited target while testing it.

Use `docs/current/HCH_V3_1_SSI_FINAL_EXECUTION_CLOSURE_PROTOCOL.md` and `experiments/current/hch_v3_1_shape_semantic_identification/SSI_FINAL_EXECUTION_CLOSURE_VERIFICATION_AI_PROMPT_20260905.md`. The local AI must not edit `.py` or `protocol_manifest.json`, must not run `real=True`, and must not open real SOURCE_GTRAIN/GVAL/D1/D2 truth. Any failure => `HCH_V3_1_SSI_FINAL_CLOSURE_NOT_READY`; full V0-V9 pass => `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`. Return that report for independent review before any fresh authorization literal is consumed.

No real SSI scientific execution has occurred. S2V/S3/S4/fresh panel remain closed; Amplitude/Gate remain outside SSI promotion; `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.1 SSI repaired implementation re-audit still blocked (2026-09-05)

Independent re-audit returns `HCH_V3_1_SSI_REAUDIT_NOT_READY_FOR_EXECUTION`. The repair is materially better: 20 SSI tests pass; compileall passes; A1/B1/B2 source training, correct symmetric B2 D1 calibration, `[B]->[B,H]` eligibility, frozen checkpoint/H0 hash binding and manifest self-verification are now present; the seven protected historical files remain byte-identical to the accepted frozen source bundle.

Do not authorize yet. Two G0 blockers remain: (1) contextual guard bypasses the base allowlist/role-phase contract—fresh `EPEX_FR`, forbidden `S2V`, and wrong-phase `SOURCE_GTRAIN` were independently shown to pass `authorize_context()`; (2) the exact future real SSI orchestration still does not exist behind the authorization gate. Current synthetic execution covers trainer/calibration but not the full registered lifecycle (real guarded adapter, post-Bridge latent/H0 integration, D1-only budget, direction diagnostics, seed aggregation, G0/G1/G2, verdict mapping, evidence writers), while `synthetic=False` remains hard-sealed. A post-audit code change would therefore still be required.

Controlling re-audit: `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PRE_EXECUTION_REAUDIT_20260905.md`. Repair-round-2 prompt: `experiments/current/hch_v3_1_shape_semantic_identification/SSI_PRE_EXECUTION_REPAIR_ROUND2_AI_PROMPT_20260905.md`. Next expected status is `HCH_V3_1_SSI_IMPLEMENTATION_FINALIZED_AWAITING_EXECUTION_REAUDIT`. No scientific SSI execution has occurred; S2V/S3/S4/fresh remain closed and `EXPERIMENT_LEDGER.md` stays unchanged.

## HCH V3.1 SSI pre-execution audit blocked (2026-09-05)

Independent pre-execution review **withholds** `HCH_V3_1_SSI_EXECUTION_AUTHORIZED` for the current implementation. The 12 synthetic tests and reported manifest SHA independently reproduce, and the seven protected historical V3/V3.1 files exactly match the frozen source bundle. But execution is not ready: `runner.run()` still raises `NotImplementedError`; source A1/B1/B2 training is absent; B2 D1 calibration uses the wrong CORN semantics; real frozen checkpoint/H0/data integration is missing; the guard is not bound to a real reader/chronology path; real `[B]` eligibility broadcasting fails; the manifest does not seal the actual code/config/input states; and mandatory integration tests are incomplete. No SSI scientific source/D1/D2 run occurred.

Controlling audit: `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PRE_EXECUTION_AUDIT_20260905.md`. Next step is implementation repair and independent re-audit. Do not consume the authorization literal until that re-audit passes. SSI design remains active; Amplitude/Gate remain frozen/outside promotion; S2V/S3/S4 and the fresh panel remain closed.

## HCH V3.1 learned-component redesign discussion (2026-09-05)

The primary objective is now explicitly reaffirmed as **extreme-price repair**. Keep the V3.1 high-level structure and accepted `SIGNED_MAE_BAYES_GRID_V1`; do not reopen Gate/selector rescue. The first algorithmic object is Shape semantics, because the executed demo trained generic residual-threshold states with no extreme importance and cross-fit actions concentrated in Normal regions. Amplitude stays frozen for the first mechanism test.

Preferred candidate semantics: non-Identity Shape state only when (a) truth lies outside the legally materialized market-relative S1 Q05/Q95 Tail and (b) the signed residual exceeds the unchanged causal directional materiality threshold. For a later coherent Shape redesign, compare the historical CORN parameterization with a two-logit symmetric `Repair? -> Direction?` factorization; do not change both Shape and Amplitude simultaneously because the current Amplitude training masks are derived from Shape state.

The Shape-only protocol is now frozen at the **design level** as `HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_V1` (SSI). It uses the frozen historical post-Bridge Shape token and four rows: historical Shape replay, a fair generic-CORN re-head, an extreme-repair CORN re-head, and an equal-capacity extreme-repair symmetric `Repair -> Direction` head. `B1 vs A1` isolates target semantics; `B2 vs B1` isolates posterior parameterization. Source G-Val tests representation information; target D1->D2 tests whether the current three Shape calibration scalars transfer the ranking. Amplitude and Gate remain outside promotion.

Design: `docs/current/HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_EXPERIMENT_DESIGN_20260905.md`. AI prompt: `docs/current/HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_AI_PROMPT_20260905.md`. No implementation or scientific execution is authorized yet; execution requires the literal `HCH_V3_1_SSI_EXECUTION_AUTHORIZED`. Consumed DE/NORD_FI/PJM S2V stays closed, original S3/S4 stays protected, and `EPEX_FR/NORD_DK1/NORD_NO × PatchTST/TimeMixer` remains untouched.

## HCH V3/V3.1 stage closeout — open a new window next (2026-09-04)

Independent audit accepts `HCH_V3_1_TAIL_FIRST_D2_NOT_SUPPORTED`. The registered 4×7-day cross-fit is valid and independently recomputed: nonzero action availability `5/6`, strictly positive Tail gain `1/6`, Overall harm `2`, Normal harm `2`, zero chronology/provenance failures. Current source/tests are clean at the scientific-contract level: HCH `41/41`, demo `27/27`, compile/static isolation PASS; clean bundle revision `2378a2faf40d12833ca78b6adcac218219d3bbc9`.

The D2 Gate/selector rescue route is now stopped. Keep `SIGNED_MAE_BAYES_GRID_V1` as the accepted V3.1 action readout, but do not add more kappa/selector/tau/tail-threshold/class-weight rules. Cross-fit shows the deeper failure: PJM still acts ~10–12% of hours but zero Tail hours, while DE actions are also entirely Normal. The next scientific object is therefore learned repair ranking / representation semantics, not deployment-margin selection.

Before any coding in the next window, discuss Shape state-target semantics, CORN vs simpler symmetric posterior parameterization, Global-vs-target locality, calendar-only vs richer causal context, and whether Amplitude should initially stay fixed. Any Shape/Amplitude/sharing/D1-adapter change is architecture/core-algorithm work and requires user discussion first.

Read first in the new window: `RESEARCH_STATE.md`, `EXPERIMENT_LEDGER.md`, this file, `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_V3_1_TAIL_FIRST_CROSSFIT_AUDIT_20260904.md`, `docs/current/HCH_V3_V3_1_STAGE_CLOSEOUT_AND_NEXT_WINDOW_HANDOFF_20260904.md`, then `docs/current/NEXT_WINDOW_START_20260904.md`. Consumed DE/NORD_FI/PJM S2V remains closed; S3/S4 remain protected; preregistered `EPEX_FR/NORD_DK1/NORD_NO × PatchTST/TimeMixer` remains untrained and unevaluated.

## HCH V3.1 action policy passed; tail-first D2 selector is next (2026-09-04)

`SIGNED_MAE_BAYES_GRID_V1` passed the preregistered source/G-Val/D1/D2-only
action-policy mechanism validation: nonzero availability 5/6, nonnegative
Overall-MAE gain 6/6, Normal-MAE harm violations 0, and historical fixed-policy
replay preserved. Independent source/test/provenance replay accepts the result.

Fresh-panel execution is nevertheless **not yet authorized**. The current D2
selector remains Overall-MAE-first and the corrected D2 results show Tail-MAE
0/0/+0.10%/0/-0.50%/-0.22% gain approximately across the six cells; both PJM
cells therefore improve Overall-MAE while slightly worsening Tail-MAE. The
active next hypothesis is a single parameter-free selector correction:
`D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1`, minimizing Tail-MAE only inside the set
that is non-harmful on both Overall-MAE and Normal-MAE. Its mechanism-promotion
result must be four-fold contiguous 7-day cross-fit inside D2: select kappa on
21 days and evaluate on the untouched 7 days, then concatenate held-out folds.
This prevents in-sample Tail-MAE minimization from being mistaken for
out-of-sample signal. No architecture or learned branch changes are authorized
in this pass.

Read next:
`experiments/evidence/hch_v3_demo_validation/INDEPENDENT_V3_1_D2_MECHANISM_AUDIT_20260904.md`,
`docs/current/HCH_V3_1_TAIL_FIRST_SAFE_D2_SELECTOR_DESIGN_20260904.md`, and
`docs/current/HCH_V3_1_TAIL_FIRST_SAFE_D2_SELECTOR_AI_PROMPT_20260904.md`.
Consumed DE/NORD/PJM S2V and all S3/S4 remain closed.

A fresh-panel candidate is preregistered from model-independent raw-market
statistics only: `EPEX_FR / NORD_DK1 / NORD_NO × PatchTST / TimeMixer`. Do not
train/evaluate it until the tail-first D2 gate passes independently.

Latest window handoff. Updated: 2026-09-04 (authorized HCH V3 demo independently adjudicated; V3.1 Gate-only refinement active).

## Workspace locator

Actual local repository root: `D:\作业\science\solar_leak_price_model`.

A fresh window must first open that exact directory with DevSpace before reading any scientific state. If `docs/current/NEXT_WINDOW_START_20260902.md` and `docs/current/HCH_DUAL_VIEW_ASYMMETRIC_PROBABILISTIC_EXTREME_REPAIR_DESIGN_v0.1_20260902.md` are not visible, the wrong workspace is open; do not continue from chat memory.

## Current state

The project is now in **HCH V3 AUTHORIZED DEMO CONSUMED / EXECUTED V3 NOT SUPPORTED AS A REPAIR METHOD / V3.1 SOURCE-D1-D2 MECHANISM VALIDATION NEXT**. The authorized six-cell S2V execution itself is accepted, but HCH-V3 full/global-only were pointwise exactly Identity because the Gate took zero nonzero actions. The independent scientific verdict is `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`. Consumed DE/NORD/PJM S2V must not be rerun for V3.1 and S3/S4 remain protected.

The frozen historical V3 formal design is:
`docs/current/HCH_DUAL_VIEW_ASYMMETRIC_PROBABILISTIC_EXTREME_REPAIR_DESIGN_v0.1_20260902.md`

The active narrow V3.1 refinement design is:
`docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_DESIGN_20260904.md`

Local-AI implementation/mechanism-validation prompt:
`docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_AI_PROMPT_20260904.md`

Its preferred pipeline is:

1. `Representation & Sharing`: three semantic information objects `X^c / X^s / X^a`, shared forecast-time covariate context, two private encoders and one low-capacity residual Bridge;
2. `Repair Proposal`: a CORN-style ordered Shape head produces coupled `Down / Identity / Up` probabilities, while the Amplitude branch produces scale-equivariant positive/negative non-crossing severity quantile distributions;
3. Shape and Amplitude define one signed repair distribution rather than collapsing first to `p_+ m^+ - p_- m^-`;
4. `Host-Preserving Expected-Risk Gate`: compare `Keep / Down / Up` actions and execute a repair only if the best nonzero action beats Keep by a scale-normalized Host margin;
5. cross-market pretraining uses one shared repair core; target deployment is exactly **five differentiable D1 scalars** (`b1,b2,T,gamma-,gamma+`) plus **one deterministic D2 `kappa_bar`** selected from model/support-derived risk-advantage breakpoints.

CORN/CORAL, TFT-style covariate handling, Cross-Stitch/MTAN soft sharing, IQF/ISQF quantiles, pretrain→fine-tune and reject-option learning are treated as mature implementation operators, not standalone novelty claims. The current candidate novelty is the structured frozen-Host repair formulation, directional scale-equivariant severity transfer, and risk-based Host-preserving three-action selection.

The earlier day-fixed `alpha_d^+/-` quantile router and scalar-gated mixture-mean correction are no longer preferred core components.

Latest completed experiment is the authorized HCH V3 six-cell S2V demo. Historical machine token: `DEMO_MECHANISM_ONLY`; independent scientific adjudication: `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`. HCH full/global-only were exactly Identity because the Gate never acted. The older CMER verdict remains `CMER_S2V_PILOT_NOT_SUPPORTED`; neither consumed S2V result may be rescued by rerunning a modified method on the same panel.

Historical V5.3 reference experiment:
`experiments/evidence/v5_3_rank_adaptive_tail_junction/RATJ_FINAL_DECISION.md`

Primary verdict:
`V5_3_RATJ_HOST_RANK_NOT_SUPPORTED`

R1 status:
`NOT_AUTHORIZED_R0_FAILED`

Key facts:

- D0 rank replay passed; host/truth ordering is strong in all six original market-host cells.
- IAMR2 passed M1 and improved 5/6 cells vs legal V2.5 endpoint.
- IAMR3 hour-conditioned extension passed 4/6 vs IAMR2.
- When compared directly to the legal V2.5 endpoint, IAMR3 is positive in all six original cells; approximate median gain ~3.70%.
- IAMR3 is not yet tail-safe: PJM UpperTail/Tail-MAE worsens materially vs Identity.
- V5 shared-prior function-space partial pooling failed; do not make shared prior a required headline component.
- IAMR prediction rows duplicated across seeds are deterministic duplicates, not independent calibrator seeds. Future stability must use temporal blocks/bootstraps or truly distinct host seeds.

## Latest TLR / APTC mechanistic facts

- TLR Stage A replay/readjudication passed, but Stage B failed the freeze gate.
- All six TLR cells selected `lambda_h=infinity`; the Fourier hour deviation was fully disabled.
- PJM upper events are temporally clustered, so short chronological validation folds cannot reliably supervise a rare-tail safety selector even when aggregate S2-T event count is not tiny.
- Symmetric high/low tail fallback is not viable because NORD_FI obtains very large IAMR improvement in the lowest host-rank decile.
- V5.2-APTC reused IAMR3 and applied an upper-only raw-space slope-one continuation, but **2026-08-31 source re-audit corrected its junction provenance:** `tlr_runner.load_context()` defines `s1_q95` from S1 truth, and APTC consumes that value. Historical D0/D1/final-report wording that calls it S1 host-q95 is wrong.
- The historical V5.2 verdict remains `APTC_DEVELOPMENT_NOT_SUPPORTED`; stored D2 numbers are reproducible and no historical artifact was overwritten.
- The continuation mechanism itself remains useful: PJM IAMR3 UpperTail relative harm ~+26.4%/+23.8% was reduced by APTC to ~+9.7%/+10.0%, removing about 63%/58% of IAMR3 upper harm.
- Corrected actual-junction drift: DE/NORD actual APTC trigger rate is 0% on S2-V; PJM PatchTST actual threshold 103.33 is ~99.40 percentile with ~0.60% trigger, TimeMixer 103.33 is ~99.23 percentile with ~0.77% trigger. Earlier S1-host-q95 values ~101–103 remain only a post-hoc host-only drift diagnostic.
- Read-only host-rank stability diagnosis rejects fixed S2-T and all-history expanding raw q95 as sole references. Trailing-28 raw q95 adapts better in DE/PJM (~5.8–6.7% trigger) but still over-triggers NORD (~15.6–17.4%) and is block-unstable.
- Formal-cache causality audit confirms the day’s 24 host forecasts are jointly available from the prior 168h context and host inference reads context only. But pure within-day q95 is statistically unsuitable as the sole coordinate: it mechanically triggers 2/24 hours and poorly covers historical upper events.
- Therefore static absolute-price tail junctions and naive moving raw-q95 replacements are rejected as sufficient deployment coordinates. The surviving target is a causal adaptive current-host CDF/PIT/rank coordinate.

## V5.3-RATJ final facts

The registered R0 was executed correctly and then stopped exactly at its hard gate.

- independent contract replay: `7 passed`;
- G0 contract/provenance: true;
- G1 global q95 semantic coherence: **false**;
- G2 block stability: true;
- G3 adaptive value over stale references: true;
- all six cells selected 7-day half-life from `{7,14,28,56}` using S2-T host-only prequential pinball;
- EW-CDF pooled S2-V occupancy: DE ~5.12/5.16%, PJM ~5.28/5.63%, NORD ~12.07/12.33%;
- NORD block occupancy reaches ~28.57/30.36% and individual days reach 18/24 and 17/24 flagged hours; failure is regime/day/hour structured rather than merely a mildly lagged scalar q95;
- R1 was not executed; no correction prediction, S3/S4, q-level search, truth-based R0 selection, protected-source modification, or historical-artifact overwrite occurred;
- retrospective truth diagnosis after the frozen R0 verdict: PJM EW flags cover 86.4% of historical upper events on both hosts; NORD coverage is PatchTST 22.2%, TimeMixer 100%, confirming host-specific tail alignment.

Research interpretation:

> faster forgetting is useful, but a single pooled marginal host-CDF percentile is not a stable universal tail coordinate. The global rank-junction line is stopped. Do not rescue it with 1/3/5-day half-life search, new q levels, host-only DSPOT junctions, or post-hoc PJM-only R1 on the same development panel.

## Current HCH V3 demo-protocol override (2026-09-03)

The generic HCH V3 implementation has passed independent closure audit and is `PROTOCOL_READY` at the implementation layer. The single formal method document has been aligned to the actual audited code: ScaleCarrierV3 uses `s^R=Q(|r|)`, separate `s^P`, MeanAD/tie-aware Shape features, the registered 7D Shape / 6D Amplitude schemas, and 5-trainable + 1-deterministic deployment adaptation.

A small scientific development-demo protocol is now drafted at:

`experiments/evidence/hch_v3_demo_validation/HCH_V3_DEMO_VALIDATION_PROTOCOL_20260903.md`

with execution prompt:

`experiments/evidence/hch_v3_demo_validation/HCH_V3_DEMO_VALIDATION_AI_PROMPT_20260903.md`.

Fixed first panel: `DE_EPEX / NORD_FI / PJM_2020 × PatchTST / TimeMixer`, using existing audited multiyear frozen-host caches. External headline peers are Identity/PIR/delta-Adapter Ada-Y/UEC-STD; CRC remains appendix-only. HCH rows are full cross-market+D1+D2 plus global-only diagnostic.

Preserve outer S1/S2/S3/S4 = 50/20/10/20. Recover S2T/S2V inside S2 = 16/4. For each target deployment use the final 84 S2T days as the entire target parameter-fitting/retrieval/selection budget: 56 days D1 then 28 days D2. HCH may additionally use only its registered immediately preceding 168h revealed residuals as non-parametric causal state. S2V is development-demo evaluation only; S3/S4 remain unread. For each held-out target market, global-core training uses only the other two markets' S2T data over both Hosts, with source-only chronological G-Train/G-Val.

Preparation manifests/support hashes/fidelity state remain accepted. The latest synthetic production integration also passes orchestration/chronology tests, but a fresh independent source audit shows that it still uses handcrafted HCH `make_pair()` tensors, constant-offset peer stand-ins and a synthetic-only execution fixture. It also contains a D2 bug: candidate Normal-MAE is compared against Identity Overall-MAE rather than Identity Normal-MAE under the same mask. Therefore the controlling state is `HCH_V3_PRODUCTION_SYNTHETIC_ORCHESTRATION_PASS / REAL_DATA_TRAINING_FREEZE_REQUIRED`. The next local-AI stage must use legal S1/S2T only to freeze 3 real LOMO Global checkpoints, 6 real HCH D1/D2 deployment states, 24 real peer fitted states, corrected D2 tables and a load-only evaluation adapter. Required next token: `HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`. Do not issue/use `DEMO_EXECUTION_AUTHORIZED` before independent audit of that state. No HCH V3 scientific-effect demo has run yet; `EXPERIMENT_LEDGER.md` remains unchanged.

## Active next research question

Submission target: **ICDE-level extreme electricity-price forecast repair as a model-agnostic post-processing module**. The project remains in architecture redesign and optimizes for SOTA-level effect, novelty, mathematics, robustness and efficiency.

Latest completed paper-candidate run: **CMER**, verdict `CMER_S2V_PILOT_NOT_SUPPORTED`. The one-shot freeze/provenance discipline passed exactly as registered, so the result is scientifically usable and CMER must not be rescued on the consumed DE/NORD/PJM S2-V panel.

CMER failure is effectiveness, not catastrophic safety: median S2-V MAE gain was -0.2527%, UpperTail gain +0.9553%, Normal harm +0.6888%, action rate 6.37%, selected residual-positive precision 48.04%. No cell exceeded 3% MAE harm, but UpperTail/precision gates failed, especially NORD/PatchTST.

New user-confirmed and repository-audited evidence changes the available information boundary: complete Chinese provincial files exist for SHANDONG/GANSU/SHAANXI/NINGXIA/QINGHAI and contain common forecast-time physical/market fundamentals. Direct inspection finds at least four common target-day fields in all five markets: `日前电价`, `新能源总加预测值`, `直调负荷预测值`, `竞价空间预测值`. Shandong also supplies a substantial negative-price regime (~13.4% RT negatives), while Ningxia/Qinghai are natural short-support transfer cases.

The project is now in **STRATEGIC_CONSOLIDATION / NO NEW EXPERIMENT AUTHORIZED**. The previously drafted CN-ERB plan is provisional only and is not the mandatory active run.

The scientific headline remains **cross-market extreme electricity-price repair**. Existing international multi-market evidence remains the historical backbone. Public Chinese provincial markets may later form a related-domain/cross-province extension; they do not automatically replace the international benchmark.

Chinese-data boundary confirmed by the user:

- SHANDONG is private/non-public and may only be a supplementary private stress test;
- GANSU/SHAANXI/NINGXIA/QINGHAI may be used as public/reproducible provincial markets subject to formal provenance/admission checks;
- common forecast-time fundamentals make a future fundamentals-aware/cross-province study plausible, but whether that becomes a headline mechanism is still a human-in-the-loop strategic choice.

Hard success principle now added: on the ultimately selected datasets and headline metrics, the final method must beat the preregistered selected strong baselines/peers; beating Identity or a historical internal endpoint alone is insufficient.

Immediate operational priority before new experiments:

1. summarize the full V2.5 -> IAMR -> V5/V6 -> CMER research arc and current effect level;
2. clean repository navigation so a new Codex can identify canonical state without reading stale V2.1/MCCH entries;
3. classify docs/experiments/src into canonical, high-value history, rejected/read-only, archive, and generated junk;
4. only after that discussion freeze the next paper/benchmark direction.

Repository audit snapshot and migration record: `docs/archive/repository_cleanup_20260901/README.md`. The target layout and AI lifecycle governance are implemented; cleanup remains a repository-only change.

Repository-organization navigation now controlling cleanup:

- `docs/README.md`
- `docs/current/README.md`
- `experiments/STAGE_INDEX.md`
- `src/README.md`
- `data/README.md`

Root `AGENTS.md` has already been rewritten as a short migration-safe canonical navigation contract. The intended long-term lifecycle is: docs current=small rolling synthesis, docs history=stage/version folders, docs archive=cold superseded/verbose material; experiments separate history from cold machine evidence; src uses core -> mvp -> promotion/archive with component-level README updates; paper remains locked.

The older CN-ERB documents remain provisional historical planning material only and are not an authorized run.

Primary data roots:

- `data/CHINA/SHANDONG/source/shandong_pmos_hourly.csv`
- `data/CHINA/GANSU/甘肃24h电价数据集.xlsx`
- `data/CHINA/SHAANXI/陕西24h电价数据集(1).xlsx`
- `data/CHINA/NINGXIA/宁夏24h电价数据集.xlsx`
- `data/CHINA/QINGHAI/青海24h电价数据集.xlsx`

## Research discipline

Do not reopen without new evidence:

- V2.6 Up/loss rescue;
- sample-level selective release/router/controller;
- HCR/HCA/PCC controller line;
- simple online SR2 rescue;
- PLA soft Shared/Local router;
- HPA old-parameter partial pooling;
- DRQ direct residual-center MLP;
- V5 shared-prior pooling as mandatory headline;
- V5.1-TLR Fourier-hour + alpha freeze design;
- symmetric tail fallback/continuation around IAMR;
- V5.2 static S1 raw-q95 APTC freeze design;
- V5.3 single pooled marginal host-CDF/rank junction as a universal six-cell tail coordinate;
- shorter-half-life/q-level/DSPOT-on-host post-hoc rescue of V5.3 on the same panel;
- threshold fishing on the old six-cell S2-V development panel.

CMER is closed after its single S2-V run and must not be tuned/retried. No new experiment/method stage is currently authorized; S3/S4 remain unavailable until the next benchmark/method protocol is explicitly frozen after the strategic review. Repository reorganization/correction is complete; use `docs/current/RESEARCH_SNAPSHOT.md`, `docs/current/NEXT_DECISIONS.md`, `experiments/STAGE_INDEX.md`, and the maintained `src/core/v25_point_runtime/` path for current navigation.

Current framework status (2026-09-03): the integrated HCH V3 candidate remains the same two-branch Shape/Amplitude + deterministic Keep/Down/Up risk-decision architecture with a clean-room implementation under `src/mvp/hch_v3_cross_market_extreme_repair/`. The seventh local binding seal is real: independent replay is 28/28 synthetic tests OK, compile passes, and the standalone sealing probe reproduces optimizer-domain/source/input-state/Target-Loss/D2-candidate/manifest fixes. The controlling **seventh independent audit nevertheless returns `IMPLEMENTATION_CORE_PASS / SEVENTH_REAUDIT_EXECUTION_CLOSURE_INCOMPLETE`**. The remaining bottleneck is no longer a model or isolated config bug; it is the full target-deployment execution closure `E=(C,theta0,S,U,D,R,H,M)`. Exact ordered D1 support data are not yet bound; the bound context/runtime can be bypassed by the unbound D1 runner; target-support information/chronology metadata can remain placeholder/inconsistent; scientific deterministic mode is still warn-only; concrete typed-state enforcement is incomplete; and D2 candidate derivation is verified but support-score/harm computation remains a forthcoming protocol-specific evaluator artifact. The documentation methodology is corrected: Sections 18–20 of `docs/current/HCH_V3_SIXTH_INDEPENDENT_REAUDIT_AND_MINIMAL_SEVENTH_FIX_DESIGN_20260903.md` and Sections 18–21 of `docs/current/HCH_V3_SEVENTH_BINDING_AND_STATE_PROVENANCE_AI_PROMPT_20260903.md` define one final generic closure pass plus a hard stop. If the closure mutation/lifecycle/reproducibility suite passes a fresh independent audit while mechanisms 1–110 remain green, grant `PROTOCOL_READY` and move to data-specific scientific protocol design rather than continue generic hardening. Scientific protocol freeze, benchmark experiments, protected split access, S2-V/S3/S4, method promotion and paper results remain unauthorized until that gate.

## Mandatory per-result research loop

For every future result feedback from the user, follow this order before designing the next stage:

1. read machine artifacts/source and independently recompute the decisive metrics/gates;
2. verify the run followed the registered chronology, provenance, split, and selection boundaries;
3. conduct result-conditioned specialized literature research around the actual newly observed mechanism/failure mode;
4. adjudicate whether the result is implementation failure, estimator failure, hypothesis failure, or narrower positive evidence;
5. only then produce the next plan; if the mathematical/architectural object changes, write the corresponding design/math document, otherwise write code-implementation design + experiment protocol;
6. update canonical state files when the conclusion/stage/bottleneck changes.

## Canonical files

Read in this order on a new window:

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `docs/history/v5_iamr/HCH_V2_6_TO_V5_2_STAGE_SYNTHESIS_AND_HANDOFF_20260831.md`
5. the V5-IAMR final report
6. the V5.1-TLR final decision
7. the V5.2-APTC final decision/audit
8. `docs/history/v5_tail/HCH_V5_2_APTC_POSTHOC_REAUDIT_AND_RANK_JUNCTION_REVIEW_20260831.md`
9. `experiments/evidence/v5_3_rank_adaptive_tail_junction/RATJ_FINAL_DECISION.md`
10. `docs/history/v5_tail/HCH_V5_3_RATJ_R0_AUDIT_AND_CONDITIONAL_TAIL_ADJUDICATION_20260831.md`
11. `docs/archive/v5_tail_details/HCH_V5.4_CONDITIONAL_EVT_TAIL_MATHEMATICAL_ADJUDICATION_v0.1_20260831.md`
12. `docs/archive/v5_tail_details/HCH_V5.4_EVT_TAIL_FEASIBILITY_AUDIT_PROTOCOL_v0.1_20260831.md`
13. `experiments/evidence/v5_4_conditional_evt_tail_feasibility/EVT_FEASIBILITY_FINAL_DECISION.md`
14. `docs/history/v5_tail/HCH_V5_4_E0_REAUDIT_AND_CONDITIONAL_GPD_ADJUDICATION_20260831.md`
15. `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
16. `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
17. `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_PROTOCOL_v0.1_20260831.md`
18. `docs/history/v5_tail/HCH_V5_4_E1_REAUDIT_AND_ICDE_ARCHITECTURE_PIVOT_20260831.md`
19. `docs/archive/v6_extreme_details/HCH_V6_ICDE_EXTREME_REPAIR_ARCHITECTURE_RACE_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
20. `docs/archive/v6_extreme_details/HCH_V6_R0_PARALLEL_ARCHITECTURE_RACE_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
21. `docs/archive/v6_extreme_details/HCH_V6_R0_PARALLEL_ARCHITECTURE_RACE_PROTOCOL_v0.1_20260831.md`
22. `docs/archive/v6_extreme_details/HCH_V6_R0_PARALLEL_ARCHITECTURE_RACE_PROMPT_20260831.md`
23. `experiments/evidence/v6_3_sparse_decision_aware_extreme_repair/reports/SDAER_V6_3_FINAL_RESULT.md`
24. `docs/history/v6_extreme/HCH_V6_3_SDAER_REAUDIT_AND_CMER_PIVOT_20260901.md`
25. `docs/history/cmer/HCH_CMER_CROSS_MARKET_EXTREME_REPAIR_MATHEMATICAL_CONTRACT_v0.1_20260901.md`
26. `docs/history/cmer/HCH_CMER_IMPLEMENTATION_DESIGN_v0.1_20260901.md`
27. `docs/history/cmer/HCH_CMER_EXPERIMENT_PROTOCOL_v0.1_20260901.md`
28. `docs/archive/cmer_details/HCH_CMER_EXECUTION_PROMPT_20260901.md`
29. `docs/archive/v2_5_details/NEXT_WINDOW_START_HCH_20260901.md`

## V6.1 CER execution result

V6.1 internal screen completed with verdict `V6_1_CER_INTERNAL_ONLY_NOT_SUPPORTED`.

- `CER_LB`, `CER_Q`, and `CER_R` were executed only on E0 honest residual folds using F1 fit and chronological F2 calibration/evaluation.
- Harmful-action limits passed, but maximum cell certificate day-miscoverage exceeded `.25` in NORD and all three variants failed usefulness/release gates. No top2 was authorized.
- S2-V, S3, and S4 were not loaded or run. No alpha/q/K/network rescue, learned router, hybrid, negative correction, or GP mean-excess action was introduced.
- Full artifacts and audit: `experiments/evidence/v6_1_certifiable_extreme_repair/`.

## V6.2 HERA H0 execution result

V6.2 HERA completed the preregistered isolated H0 run with `HERA_Q1`, `HERA_Q2`, and `HERA_R2`. The verdict is `V6_2_HERA_INTERNAL_NOT_SUPPORTED`.

- H0-A: day top-quartile event lift >1 in 3/6 F3 cells; G1 precision lift >1 in 6/6; G1+G2 recall > random 4/24 in 6/6; positive day+G1 cells 3/6, so the occurrence stop rule failed.
- H0-B: Q1/Q2/R2 all failed complete point-action promotion. Median F3 `S` was `0.03333`, `0.04169`, and `0.03233`; median action density was `0.7093`, `0.7405`, and `0.6314`, while MAE/Normal-harm, block, and density gates failed.
- E0 F1/F2/F3 labels, OOF scores, rank reliabilities, fallback audit, residual pools, retrieval audit, F2/F3 metrics, weighted-median diagnostics, independent recompute, and final audit are under `experiments/evidence/v6_2_hierarchical_extreme_risk_allocation/`.
- No S2-V truth/prediction was loaded because H0 did not pass; no state hash/freeze was authorized; S3/S4 remain forbidden.

## V6.3 SDAER execution result

V6.3 completed in the isolated root `experiments/evidence/v6_3_sparse_decision_aware_extreme_repair/`.

- Final static verdict: `SDAER_STATIC_NOT_SUPPORTED`.
- Online diagnostic: `SDAER_P2_SEPARATE_ONLINE_DIAGNOSTIC`; it is not mixed with static ranking or claims.
- H0-A passed by independent recomputation. H0-B failed for both `SDAER_S1` and `SDAER_S2`; both failed the mandatory NORD stop condition.
- Static diagnostic ranking only: S2 rank 1 by F3 median S (`0.00534119`), S1 rank 2 (`0.00143156`). Neither is promotable.
- No rank-1 freeze and no S2-V one-shot were authorized. S2-V, S3, and S4 remain unread/unrun.
- P2 used strict predict/store/reveal/update chronology for 366 target days; keep it explicitly outside any static headline claim.
- Read `reports/SDAER_V6_3_FINAL_RESULT.md`, `audits/h0_independent_recompute.json`, `audits/prefix_specific_oof_audit.csv`, `audits/selected_slot_diagnostics_summary.csv`, `audits/sdaer_p2_cell_summary.csv`, and `provenance/final_audit.json` before any next-stage design.

## CMER one-shot S2-V result (2026-09-01)

CMER was executed as the only paper-candidate stage after V6.3. The exact single configuration completed one frozen S2-V pilot and returned `CMER_S2V_PILOT_NOT_SUPPORTED`.

- `15 passed` contract tests; freeze-before-read and independent recomputation passed.
- Median S2-V MAE gain `-0.2527%`, UpperTail gain `0.9553%`, Normal harm `0.6888%`, action rate `6.3738%`, selected residual-positive precision `48.0385%`.
- The pilot failed the registered median MAE, median UpperTail, positive-UpperTail-count, overall precision and NORD precision gates, while no-cell-3%-MAE, Normal-harm, action-rate, NORD-MAE and protected-history checks passed.
- Exactly one S2-V run was performed. No q/K/fusion/source/seed rescue, no second run, no S3/S4, and no paper-scale CMER expansion are authorized.
- Full machine root: `experiments/evidence/cmer_cross_market_extreme_repair/`; report `reports/CMER_FINAL_RESULT.md`; final audit `provenance/final_audit.json`.


## Repository consolidation and correction completion (2026-09-01)

The authorized joint reorganization migration and targeted correction are complete. Use these short navigation files first:

- `docs/current/RESEARCH_SNAPSHOT.md` and `docs/current/NEXT_DECISIONS.md`
- `experiments/STAGE_INDEX.md` (current status: `NO_ACTIVE_EXPERIMENT`)
- `src/README.md` and `src/core/README.md`
- `data/README.md` and `data/MARKET_INDEX.csv`

Physical lifecycle now separates accepted source (`src/core/`), candidates (`src/mvp/`), comparison interfaces (`src/baselines/`, `src/backbones/`), shared utilities (`src/utils/`), and historical source (`src/archive/`). The maintained current HCH entrypoint is `src/core/v25_point_runtime/`; the former full action/retrieval orchestrator is explicit archive-only fidelity code under `src/archive/legacy_hch_runtime/`. Closed machine artifacts are under `experiments/evidence/`; closed-stage navigation is under `experiments/history/`; obsolete support material is under `experiments/archive/`. IAMR remains positive evidence with an explicit pending promotion decision, not a silent core promotion. Independent re-audit confirmed: 0 `archive.*` imports under `src/core`, 6/6 targeted V2.5/core tests passed, current and legacy fidelity bridges import successfully, and the new point runtime is bitwise identical to the old pipeline's point output on matched synthetic state/input (`max_abs_diff=0`). Repository cleanup is therefore closed. No active research experiment is authorized and `paper/` remains locked. Next activity is V2.5 -> V5/IAMR scientific synthesis/discussion only.

## Final implementation sealing handoff (2026-09-02)

The local HCH V3 implementation-sealing task is complete with:

`IMPLEMENTATION_CORE_PASS / SEALING_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The package source, canonical config/manifest closure, official D1 API, deterministic D2 buffer, provenance status, 48-mechanism regression matrix, and standalone probe are ready for a fresh independent audit. Do not treat this as `PROTOCOL_READY`; do not run architecture validation, scientific experiments, protected evaluation, or protocol freeze. No protected data were accessed.

## Fifth Mathematical Execution Contract handoff (2026-09-03)

Local implementation result:

`IMPLEMENTATION_CORE_PASS / EXECUTION_CONTRACT_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The fifth pass closes the execution-contract blockers without changing the HCH V3 architecture. Empty eligible global/D1 updates skip backward and optimizer stepping; official training has no caller quantile-grid override; thresholds, class balance and calibration are typed single-source/executable contracts; D2 uses a typed support-risk table; and provenance tests cover temporary Git clean/modified/untracked/deleted states. Mechanisms 49–68 are explicitly mapped in the regression matrix.

Return this package to a fresh independent fifth audit. Do not grant `PROTOCOL_READY`, do not run architecture validation or scientific experiments, and do not access protected benchmark data. The current real repository admitted-source status remains an audit fact and is printed only by the standalone probe.

## Sixth Execution-State and Semantic Sealing Handoff (2026-09-03)

The sixth local implementation pass is complete without changing the HCH V3
architecture:

`IMPLEMENTATION_CORE_PASS / SIXTH_SEALING_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

Completed locally:

- full stochastic identity for invalid global/D1 batches, including CPU RNG and next-valid trajectory;
- registered threshold-rule semantics and deterministic fixed-threshold materialization checks;
- typed FROZEN optimizer specifications, registered `none` early stopping and exact D1 attempt semantics;
- removal of the public `compute_training_loss(..., class_weights=...)` bypass;
- typed D2 risk-advantage breakpoint candidate provenance;
- FROZEN `HCHV3ExecutionSpec` binding shell;
- permanent regression mechanisms 69--88 and a separate adversarial Python probe.

Required unittest, compileall and pure AST archive-import checks pass locally;
the standalone probe reports zero invalid-step CPU RNG change, zero trajectory
divergence, expected semantic rejections, valid typed D2 selection and a
consistent FROZEN binding. Return to a fresh independent audit. Do not grant
`PROTOCOL_READY`, run scientific experiments, access protected data, or modify
`EXPERIMENT_LEDGER.md`.

## Seventh Binding and State-Provenance Sealing Handoff (2026-09-03)

Local implementation result:

`IMPLEMENTATION_CORE_PASS / SEVENTH_BINDING_SEAL_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The target-deployment binding shell now verifies typed optimizer parameter
domains, current admitted source hashes and tracked-clean status, input model
state, explicit phase scope, manifest-seed runtime initialization,
model/support-derived D2 breakpoints, registered Target/Loss semantics, and a
canonical manifest digest/config payload. Mechanisms 89--110 are mapped; the
required unittest, compileall, pure AST and separate standalone probe pass
locally. Return to independent audit. Do not grant `PROTOCOL_READY`, access
protected data, run experiments, or modify `EXPERIMENT_LEDGER.md`.

## Final Closure Override handoff (2026-09-03)

The final generic closure pass is complete:

`IMPLEMENTATION_CORE_PASS / FINAL_CLOSURE_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The typed ordered D1 support state/hash, bound runtime-initialized D1 runner, strict target-support manifest role and chronology, strict deterministic runtime metadata, concrete registered state checks, seed-bound FROZEN execution hash, and explicit D2 evaluator-provenance boundary are implemented. The dedicated `test_scientific_execution_closure.py` mutation/lifecycle/runtime/D2 suite and standalone final-closure process pass; all mechanisms 1--110 remain green. Return this package to independent audit. Do not grant `PROTOCOL_READY`, run scientific experiments, access protected data, or modify `EXPERIMENT_LEDGER.md`.

## Final Closure independent re-audit handoff (2026-09-03)

Controlling verdict:

`IMPLEMENTATION_CORE_PASS / FINAL_CLOSURE_REAUDIT_INCOMPLETE`

Fresh independent replay confirms 34/34 tests OK and the standalone final-closure probe passes. The architecture remains stable. Exactly two concrete closure properties still fail: (1) a pre-populated AdamW optimizer state can bind under the same D1 optimizer spec and changes the final D1 state even when the model is restored to the exact same frozen input hash (`max parameter diff ~0.00148404`); current D1 must require fresh-empty optimizer state at bind and before the bound run. (2) support-data chronology, calibration support-fold chronology and manifest chronology can be different non-placeholder values and the bound D1 path still executes; the current target-deployment schema must require equality. No other generic hardening is authorized. Follow Section 21 of the existing final-closure design and Section 22 of the existing AI prompt, then return for one fresh independent audit. Both concrete properties now pass with all regressions green. The controlling independent verdict is `PROTOCOL_READY` (implementation-level only). Generic implementation hardening is stopped. Next work is to write and freeze the data-specific chronology-safe scientific-validation protocol; protected benchmark execution remains unauthorized until that protocol is frozen and the user explicitly authorizes execution.

## Final micro-closure independent audit (2026-09-03)

Independent replay: 35/35 tests OK, compileall PASS, AST archive imports `[]`, standalone final-closure PASS. `fresh_empty_v1` rejects prefilled and post-bind-mutated optimizer state; chronology `A/A/A` is accepted while `A/B/A`, `A/A/C`, and `A/B/C` are rejected. Identical frozen tuples reproduce identical D1 output hashes. No new concrete execution-closure counterexample was found. Do not reopen generic execution-contract polishing without a failing property test or a contradiction discovered while freezing the real protocol.

## HCH V3 demo preparation independent audit handoff (2026-09-03)

The preparation artifacts are scientifically coherent and independently replay: 8/8 demo-preparation tests and 35/35 HCH tests pass, chronology/support hashes reconstruct, LOMO excludes the target market, peer wrapper/commit hashes match, and no S2V/S3/S4 truth was opened.

However the controlling verdict is now:

`HCH_V3_DEMO_PREPARATION_PASS / EXECUTION_DRY_BUILD_REQUIRED`

The original preparation prompt did not require the actual scientific demo runner and all execution dependencies to be frozen before user authorization. The independent audit therefore added a mandatory pre-execution dry-build gate to the same protocol/prompt. Before authorization, local AI must implement and synthetic-test the full runner, freeze exact 84-day peer selectors, materialize an auditable S1-only Q05/Q95 threshold artifact, freeze 168h causal-history accounting, file-hash/copy invoked peer dependencies, add runner/metrics/selectors to admitted provenance, and rebuild the clean frozen bundle. It must not read S2V/S3/S4 truth.

Required next token from local AI:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

Only after a fresh independent audit of that state may the user issue `DEMO_EXECUTION_AUTHORIZED`. The previous `be87a7...` clean bundle remains valid preparation evidence but is superseded for future execution because protocol/prompt were updated during the audit. Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_PRE_EXECUTION_AUDIT_20260903.md`. `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3 demo pre-execution dry-build handoff (2026-09-03)

The mandatory updated-protocol dry-build is complete:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

Frozen additions include `scientific_demo_runner.py`, the independent
`standalone_dry_build_probe.py`, runner tests, S1-only threshold artifacts,
synthetic complete output schemas, exact 84-day fitting/selection enforcement,
separate 168-hour causal state, controlled predict/store/reveal boundary,
model-derived typed D2 breakpoint candidates, and file-level hashes for all
invoked peer wrappers/official dependency files. The clean equivalent source
bundle was rebuilt after the final code and protocol hashes were fixed.

Independent fresh-process dry-build probe and all required local suites pass.
No S2V/S3/S4 truth was opened, no Host was retrained, no HCH scientific score
was generated, and `EXPERIMENT_LEDGER.md` was not modified. The root remains a
repository-consolidation dirty worktree; the clean bundle is the future
execution source. Await independent audit, then explicit user authorization.

## HCH V3 demo runner independent re-audit handoff (2026-09-03)

The returned token is not yet accepted. Controlling verdict:

`HCH_V3_DEMO_DRY_BUILD_PARTIAL_PASS / REAL_EXECUTION_RUNNER_INCOMPLETE`

Keep all frozen scientific choices unchanged. The current production `_run_real()` is only an evaluation shell around pre-fitted predictors; it does not construct/train the LOMO HCH core, D1/D2 state or peers. Independent authorized-path synthetic smoke also reproduces `ValueError: prediction already persisted`, and the real path bulk-materializes the full allowed S2V truth dictionary before daily prediction. Tail/Normal metrics and several required artifacts are placeholders on the real path. The 9 runner tests cover helper/synthetic paths but not the future `synthetic=False` control flow.

Next local-AI work is runner-integration only: implement the actual frozen Global->D1->D2->peer pipeline, method-day prediction persistence, day-scoped truth oracle, full registered metrics/mechanism outputs, and an authorization-gated real-path synthetic integration test. No HCH hyperparameter/architecture change, no S2V/S3/S4 scientific run. Rebuild clean provenance after the fixes, then return the same `HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION` token for a fresh audit.

## HCH V3 production-path integration closure handoff (2026-09-04)

The runner integration repair is complete. The exact `synthetic=False` path, under the literal authorization guard, now owns LOMO Global/G-Val checkpoint, bound D1, model-derived D2 candidates/table/kappa, peer states, and the typed seven-method day-wise predict/store/reveal loop. An independent Python process completed the synthetic `DE_EPEX::PatchTST` canary with the full stage log and finite registered metrics; required diagnostics, daily losses, bootstrap inputs, efficiency and provenance artifacts are non-placeholder.

Checks: root preparation+runner 19/19; clean-bundle runner 11/11; HCH 35/35; compileall, pure AST archive-import, independent dry-build probe and final closure probe all pass. The clean frozen bundle and admitted provenance were rebuilt after final code/test/protocol hashes. No real S2V/S3/S4 truth was read, no Host was retrained, no scientific result was generated, and `EXPERIMENT_LEDGER.md` remains unchanged.

Return token:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

Return for a fresh independent pre-execution audit; do not announce V3 validity or issue `DEMO_EXECUTION_AUTHORIZED`.

## HCH V3 real training freeze handoff (2026-09-04)

The latest controlling demo prompt is closed through the S1/S2T-only real training freeze:

`HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`

The exact formal six-cell Host inputs, target-excluded LOMO cores, G-Train/G-Val chronology, actual causal builder/history, 56-day bound D1, corrected 28-day D2, and 24 admitted real peer wrapper states are frozen. Real state reload probe passed in a separate process: 3 Global, 12 HCH and 24 peer states. No real S2V/S3/S4 truth was loaded or scored. No Host retraining, architecture/config/split/support/peer mathematics or verdict-gate changes were made. `EXPERIMENT_LEDGER.md` was not modified.

Final source bundle: revision `bd3afc2b9150cc1129e80bbc895968ebe5faf409`, admitted source tree `94d5a3da116fbe8e9e5d8a55cdff2f620e2f12dd0a4e2c89a91eea1ce7f75d76`, 54 admitted files. Required next step is independent audit; only an explicit `DEMO_EXECUTION_AUTHORIZED` may open real S2V evaluation.

## Independent real-training re-audit override (2026-09-04)

Do **not** authorize S2V from the `bd3afc...` state. Independent source audit found that the real `train_global()` does not implement the frozen equal-deployment sampler: it sequentially exhausts each deployment, producing batch exposure 4/4/8/8 in target-DE and 8/8/4/4 in target-PJM instead of exact 25%/25%/25%/25%. Thus the 3 Global checkpoints and all 6 HCH states derived from them are superseded for scientific evaluation.

Required micro-correction only: use exact 8/8/8/8 source composition in every 32-day Global batch with deterministic cycling/permutation; make G-Val deployment loss invariant to partial chunking; save `full.pt` after D2 kappa is applied; freeze the legal 168h `S2V_START_HISTORY`. Keep every HCH/peer hyperparameter and split unchanged. Recompute Global->D1->D2 once; do not tune if D2 remains `{0}` / zero-action.

Current independent verdict: `HCH_V3_REAL_TRAINING_FREEZE_PARTIAL_PASS / GLOBAL_BALANCING_AND_LOAD_ONLY_STATE_FIX_REQUIRED`.

Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_REAL_TRAINING_FREEZE_AUDIT_20260904.md`. Required next return remains `HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION` after the corrected retrain. S2V/S3/S4 remain unread and `EXPERIMENT_LEDGER.md` unchanged.


## Corrected real-training freeze handoff (2026-09-04)

The independent real-training audit blockers are closed. The corrected freeze uses exact 8/8/8/8 equal-deployment Global batches, chunk-invariant G-Val aggregation, post-D2 full-state persistence, and six `S2V_START_HISTORY` states with 168-hour timestamp/value/mask hashes. The corrected chain was rerun once; all six D2 tables remain `{0}` / zero-action and were not tuned.

Independent fresh-process reload: 3 Global / 12 HCH / 24 peer PASS. Bundle `b9d3d220016e6e5fcb549e42ddcf0b9773cc7695`, admitted tree `7e9aaed368796eaad565795f0ed26d374f114c5da8e4f3cb866761fc49ee7fa1`, 54 files.

Return token remains:
`HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`

S2V/S3/S4 remain unread, no scientific demo was executed, and `EXPERIMENT_LEDGER.md` remains unchanged.

## Final pre-execution audit PASS (2026-09-04)

Independent replay accepts the corrected freeze. Demo suite 25/25 PASS, HCH suite 35/35 PASS, and real-state reload 3 Global / 12 HCH / 24 peer PASS. Every recorded Global epoch has exact equal source-deployment exposure; `full.pt` and D2 kappa agree; all six `S2V_START_HISTORY` states are valid 168-hour histories ending one hour before S2V start. Bundle `b9d3d220016e6e5fcb549e42ddcf0b9773cc7695` is clean and the real-freeze manifest SHA256 is `c97612462880977eb200ff2d01fea03c4db7edf626a781f84b71d3f78261f3e5`.

Controlling state is now `HCH_V3_DEMO_EXECUTION_READY / AWAITING_EXPLICIT_USER_AUTHORIZATION`. No more pre-execution code/training changes are authorized absent a concrete failure. Once the user explicitly supplies `DEMO_EXECUTION_AUTHORIZED`, run only the frozen DE_EPEX × PatchTST canary first; if operationally valid, continue to all six cells regardless of canary effectiveness. All six D2 tables remain `{0}` / zero-action and must not be rescued before S2V.

## Authorized HCH V3 demo closure (2026-09-04)

Explicit authorization `DEMO_EXECUTION_AUTHORIZED` was supplied. P1 `DE_EPEX::PatchTST` passed the operational canary, then P2 ran all six frozen cells. The run used the frozen states only: 3 Global, 12 HCH, 24 peer and 6 S2V-start-history artifacts. It completed 394 S2V days with seven typed predictions per day, durable prediction hashes before each day-scoped reveal, and legal next-day HCH history updates. No S3/S4 truth was read.

Scientific verdict: `DEMO_MECHANISM_ONLY`. HCH full was Identity-equivalent because all six frozen D2 tables were `{0}` with zero Down/Up actions; it was nevertheless best/near-best against the headline PIR/Ada-Y/UEC-STD BestPeer in all six cells. The zero-action mechanism is degenerate, therefore `DEMO_ARCHITECTURE_SIGNAL_POSITIVE` is not awarded. This remains development evidence and does not establish V3 validity. Independent audit is required; no tuning or rescue is authorized.

Artifacts: `experiments/evidence/hch_v3_demo_validation/authorized_real_execution/P2_all_six/`. Bundle revision `f551148a67cdba03fa219e16e0170e5873a5110c`, admitted source tree `802192fcdc2835a4daec651f0d09f135d8a553a34ed6f69aded008b840caa812`, 55 files. Metrics hash `31e22a01278de56282e196806bed1ab3f728b01e45c7632c7f8a9514b4485540`; protocol manifest hash `97717110ce5f319703d6f1f1c00b73909a64303e882bb42408b088c93ce56942`.

Return to independent audit with state: `HCH_V3_DEMO_S2V_EXECUTED_DEMO_MECHANISM_ONLY_AWAITING_INDEPENDENT_AUDIT`.

## Independent S2V result adjudication / V3.1 handoff (2026-09-04)

The authorized execution itself is accepted: independent replay reproduces all stored headline metrics exactly, 394 prediction and reveal events match, S3/S4 remain unread, and execution bundle `f551148a67cdba03fa219e16e0170e5873a5110c` / tree `802192fcdc2835a4daec651f0d09f135d8a553a34ed6f69aded008b840caa812` is clean. But the scientific interpretation is corrected. HCH full/global-only are pointwise exactly Identity over all 394 days. The verdict evaluator wrongly uses the MAE-selected external peer for Tail-MAE too; correct per-metric comparison is MAE near-best `6/6`, Tail near-best `4/6`. The mechanism-only fallback also fires on `diagnostic_status=computed` rather than explicit positive mechanism evidence.

Independent verdict: `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`.

First-principles diagnosis: the learned signed distribution is judged by symmetric expected MAE, but the registered nonzero actions are fixed directional `Q(0.75)` readouts motivated by a different asymmetric loss. A legal D2-only counterfactual keeps all learned distributions fixed and selects the existing directional quantile-grid candidate with minimum expected-MAE risk. It restores safe nonzero actions in 5/6 cells and positive D2 Overall-MAE gain in those five cells, especially PJM (~+2.61%/+1.91%), without violating the existing zero Normal-harm selector. This supports a minimal V3.1 Gate correction, not a network reset.

Read next: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_AUTHORIZED_S2V_RESULT_AUDIT_20260904.md`, then `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_DESIGN_20260904.md` and `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_AI_PROMPT_20260904.md`. V3.1 may be implemented and validated on source/D1/D2 only. Do not rerun the consumed DE/NORD/PJM S2V, do not touch S3/S4, and do not select a fresh validation panel until the D2 mechanism gate passes independently.

## HCH V3.1 tail-first safe D2 selector validation handoff (2026-09-04)

Implemented and independently executed the registered `D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1` using the fixed 4x7-day chronological cross-fit. Learned state equality is exact; historical V3 fixed and V3.1 Overall-first replay both pass. The six-cell cross-fitted promotion gate fails: availability 5/6, positive Tail-MAE gain 1/6, Overall harm 2, Normal harm 2, provenance/numerical failures 0, consumed S2V/S3/S4 access 0. No full-D2 deployment kappa was computed because the gate failed.

Return token:

`HCH_V3_1_TAIL_FIRST_D2_NOT_SUPPORTED`

Do not create or evaluate the preregistered fresh panel. Do not access S3/S4 or rerun consumed S2V. Independent audit is required.
