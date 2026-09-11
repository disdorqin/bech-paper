# HANDOFF.md

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

