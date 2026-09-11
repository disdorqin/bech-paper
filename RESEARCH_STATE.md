# RESEARCH_STATE.md

Canonical research state for the HCH electricity-price post-processing project.
Last updated: 2026-09-11

## Host-Relative State Interaction rejected; S1 + pooled fixed alpha is the frozen structural candidate (2026-09-11)

The valid `HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1` closure rejects the Host-relative state interaction as a universal Shape input. R0 validity and R4 efficiency pass; R1, R2 and R3 fail. This was the last authorized structural closure; no successor structure is open.

Method as executed: frozen S1 Shape reused unchanged; a causal Host-bias direction `b_hat` built from the seven-day same-hour normalized Host-residual history already carried by the frozen repair input (same-hour mean over the existing seven days, then L2-normalized over the 24 horizon hours; no decay, MAD, clipping or alternate window); the fixed five-role standardized state table `R in R^(24x5)` multiplied elementwise into `Z[h,k]=b_hat[h]*R[h,k]`; and **exactly one global `beta in R^5` shared by all six cells**, trained on the cross-market stacked chronological OOF pool with a cell-macro-balanced Shape loss. `u_HRSI = normalize(u_S1 + Z beta)` with `beta=0` replaying S1 at `1.192e-07`; frozen S1 metric replay exact at `1.421e-14`. `alpha_HRSI` refit per cell by the original exact weighted-median rule. Six cells: `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds 7/17/37. The whole six-cell panel trains **5 parameters**, with 0 per-cell beta vectors and 6 cells per beta.

The shared-beta constraint destroys the only mechanism that worked. GANSU_DA Shape cosine changes by only `+0.0002` (PatchTST) and `-0.0105` (TimeMixer) versus HSA's `+0.0666` / `+0.1170`, and both GANSU HIGH-state tertile cosines fall (`-0.0109` / `-0.0241`). Relative Overall-MAE gain vs S1 is nonnegative in only 1/6 cell medians and strictly better in only 1/6; the worst cell is `-0.6151` pp (LAGO_DE/PatchTST) and the worst seed `-1.3676` pp; 0/4 international cells are strictly better and only 1/6 cells are non-worse in `>=2/3` seeds. Median gain vs Host is `5.61%` with only 3/6 cells at `>=5%`, and maximum Normal-MAE harm is `1.0227%`, above the 1% gate. The learned beta is stable in sign and magnitude across seeds (seed7 `[0.109, 0.172, 0.000, -0.265, -0.134]`, seed17 `[0.048, 0.154, 0.000, -0.313, -0.050]`, seed37 `[0.054, 0.144, 0.000, -0.328, -0.047]`), but it is set by the four international cells, which contribute most of the pooled rows; `beta_SUPPLY_MARGIN_FC` is exactly `0.0` for every seed because that role is structurally absent in all three markets, so `Z[:,:,2]` is identically zero panel-wide.

Interpretation: HSA's GANSU gain came from per-market coefficient freedom. The scientific constraint that would make the interaction *universal* — one five-parameter beta for the entire panel — is exactly the constraint that removes it, and the pooled least-squares direction then degrades four cells to buy a `+0.15` pp GANSU/PatchTST change. Making the input Host-relative does not resolve the HSA transferability conflict; the conflict is in the coefficient, not the representation. Do not rescue this route by per-market/per-Host beta, per-hour beta, role selection, conditional Amplitude, a new window or transform, or post hoc threshold relaxation. **Freeze `S1 Daily-Patch GRU32 Shape + pooled fixed alpha` as the final structural candidate.** Evidence: `experiments/evidence/hch_host_relative_state_interaction_20260911/`; independent recomputation by `verify_gates.py` (does not import the runner) reproduces R0–R4 and the token from the written evidence alone at 50/50 integrity checks. This stage does not authorize SHAANXI/NINGXIA/QINGHAI, Shandong, or any full-panel/protected/final experiment.

The alpha row-set confound was measured rather than assumed: HRSI's scalar alpha lives on OOF folds B2–B4 (a beta cannot exist before B2) while frozen S1's pooled alpha uses B1–B4, so the comparison carries a discrete weighted-median shift of up to `26.39%`. Rescoring frozen S1 on HRSI's exact row set makes HRSI look **worse**, not better: non-worse in only 3/6 cell medians, worst cell `-0.7848` pp, worst seed `-1.3998` pp. The failure is therefore not an artifact of the alpha row set. That diagnostic is disclosed in `alpha_rowset_sensitivity.csv` and never feeds a gate.

Baseline comparison infrastructure is considered complete for the current method stage: all selected comparison methods have completed their strict paper-protocol/official-code/split/training/metric consistency audit, while exact admission status and any fidelity limitations remain those recorded in the final baseline adjudication. Later HCH-vs-baseline experiments must use identical dataset/task/Host/evaluation contracts and must not reopen baseline search/tuning.

## Horizon-Aligned Semantic State Residual rejected; S1 + pooled fixed alpha retained (2026-09-11)

The valid `HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1` closure rejects the five-role horizon-aligned semantic state residual as a universal Shape input. H0 validity and H4 efficiency pass; H1, H2 and H3 fail.

Method as executed: frozen S1 OOF Shape directions reused unchanged; a deterministic `R_d in R^(24x5)` role table mean-pooled from audited `primary_channel=True` forecast-time state columns into `DEMAND_FC / RENEWABLE_FC / SUPPLY_MARGIN_FC / INTERCHANGE_FC / MUST_RUN_FC` (structurally absent role = 0, `SUPPLY_MARGIN_FC` absent in all three markets); the only new trainable object is `beta in R^5` with `u_HSA = normalize(u_S1 + R_d beta)`; `alpha_HSA` refit with the original exact weighted-median rule on HSA OOF directions. `beta=0` replays S1 at `1.192e-07`; S1 metric replay is exact at `1.421e-14`. Six cells: `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds 7/17/37.

The mechanism is real but local. Both GANSU_DA Hosts improve Shape cosine (`+0.0666` PatchTST, `+0.1170` TimeMixer) and both improve the frozen GANSU HIGH-state tertile (`+0.0347` / `+0.1189`) while cutting HIGH-state wrong-hemisphere by 12.5 / 18.2 pp. GANSU Overall-MAE beats S1 by `+1.30` / `+4.31` pp and reaches `3.55%` / `11.04%` gain vs Host. But the gain is not universal: cell-median Shape cosine rises in only 4/6 cells and wrong-hemisphere is non-worse in only 3/6; relative Overall-MAE gain vs S1 is nonnegative in only 4/6 cell medians (worst `-0.87` pp on LAGO_DE/PatchTST); median gain vs Host is `6.73%` with only 3/6 cells at `>=5%`; normal-MAE harm peaks at `1.02%`, just over the 1% gate; and only 3/6 cells hold HSA `>=` S1 in `>=2/3` seeds.

`beta` is fitted per market on that market's own chronological OOF rows only, with a single shared five-role vocabulary; no market ID, Host ID, role embedding, per-hour beta, hidden layer, bias, LR/step search, excursion/MAD feature, or market-specific rule is used. Structurally absent roles keep `beta=0` exactly, so LAGO_PJM fits only `DEMAND_FC` and LAGO_DE only `DEMAND_FC`/`RENEWABLE_FC`. Efficiency is not the bottleneck: 5 extra parameters, median beta training `1.73 s/cell`, median inference overhead `1.22 us/day`.

Interpretation: horizon-aligned semantic state supplies usable directional information for GANSU_DA but is not a mechanism that transfers without market-specific behaviour. Do not rescue this route by adding China-only roles, per-market role selection, per-hour or conditional `beta`, larger step budgets, or by relaxing the H1/H2/H3 thresholds post hoc. Keep `S1 Daily-Patch GRU32 Shape + pooled fixed alpha` as the executable method. Evidence: `experiments/evidence/hch_horizon_aligned_state_residual_20260911/`; independent recomputation by `verify_gates.py` (does not import the runner) reproduces H0–H4 and the token from the written evidence alone at 39/39 integrity checks. This stage does not authorize SHAANXI/NINGXIA/QINGHAI, Shandong, or any full-panel/protected/final experiment.

## Baseline fidelity workstream closed with mixed admission status (2026-09-11)

The bounded UEC-STD + OMPB recovery returned `HCH_BASELINE_BREADTH_RECOVERY_COMPLETE_FOR_ADJUDICATION` and is independently closed by `experiments/evidence/hch_baseline_paper_fidelity_20260911/adjudication/FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md`.

The workstream is procedurally complete but not every method is paper-faithfully reproduced. Final statuses: δ-Adapter Ada-Y = `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`; PIR full official = `PAPER_FAITHFUL_EXACT_ACCEPTED`; COSA = `ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`; UEC-STD = `HOST_FIDELITY_BLOCKED / NOT_PAPER_FAITHFULLY_ADMITTED`; OMPB = `STANDALONE_DATA_BLOCKED / NOT_REPRODUCED`.

UEC source-native recovery corrected the prior custom-loader/AR mistake and used official `ETTh1 -> Dataset_ETT_hour`, fixed ETT borders, train-only scaler, fresh TimeMixer-96 and official AR336 inference. The reproduced Host is `0.500369/0.457737` MSE/MAE versus paper `0.465/0.449`; MSE relative error is `7.606%`, narrowly above the frozen `7.5%` gate. The gate is not relaxed; UEC correction was not entered. This is a Host/public-reproducibility block, not a validated UEC method failure.

OMPB exact TCN/96 paper rows were recovered and frozen (`OMPB_TARGET_REGISTRY.sha256 = 29F4C445045C912536714F41BF3A44AEC3979E34D54FA5865CAA0E8AA1714EDB`), but the official dataset bundle was unavailable. No substitute data, training or fabricated metrics were used. OMPB may reopen only if official/provenance-equivalent data become available.

The mandatory offline comparator layer is now sufficient and frozen for later unified-DA transfer: Host/identity, direct residual, δ-Adapter Ada-Y (partial fidelity disclosed) and PIR full official (exact fidelity). COSA/UEC/OMPB are supplementary with explicit fidelity limits and must not block HCH method development. Do not write that all baselines were fully reproduced.

## State-Excursion Shape rejected; Horizon-Aligned Semantic State Residual authorized (2026-09-11)

The valid `HCH_STATE_EXCURSION_SHAPE_NOT_SUPPORTED_KEEP_S1` closure rejects the signed same-hour seven-day excursion representation as a universal Shape input. S2 improves all four international cells relative to S1 but degrades both `GANSU_DA` cells; GANSU Shape cosine falls and wrong-hemisphere does not improve. Do not rescue this route by changing the 7-day window, MAD scaling constant `1.4826`, clipping range, or market-specific state-role selection. The `1.4826` factor is only the standard Gaussian-consistency MAD rescaling and remains diagnostic-only.

The stronger interpretation is a horizon-alignment / representation-efficiency bottleneck. S1 flattens the complete target-day state trajectory into one current-day token before a 32-D projection and 24-D readout; parameter count grows with raw channel count (~14.1k PJM, ~14.9k DE, ~18.7k GANSU) and S2 increases this imbalance further (~23.3k GANSU), while GANSU has only 62 DIAG_FIT days. The next smallest justified change is therefore not another state transform or deeper network.

Authorized closure: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`. Freeze S1 entirely and fit exactly five shared semantic-role coefficients on top of chronological OOF S1 unit Shape: build a fixed `24x5` role-state table from the existing S1 train-only standardized state levels (`DEMAND_FC / RENEWABLE_FC / SUPPLY_MARGIN_FC / INTERCHANGE_FC / MUST_RUN_FC`), set `q=R beta`, and `u_HSA=normalize(u_S1+q)`. `beta=0` initially replays S1; the same five parameters and training rule apply in every market. Refit only the existing pooled exact MAE scalar alpha on HSA OOF directions. No conditional Amplitude, learned Verification, excursion/MAD features, deeper Shape network, market-specific rules, extra markets, Shandong, or protected/final access.

## Parallel baseline breadth recovery re-authorized with UEC source-path correction and standalone OMPB (2026-09-11)

The optional baseline breadth workstream is reopened only to close the two remaining requested gaps, without changing or blocking the active HCH Horizon-Aligned Semantic State Residual closure. Controlling orchestration is `docs/current/HCH_UECSTD_OMPB_FIDELITY_RECOVERY_EXECUTION_PLAN_20260911.md`; UEC and OMPB are scientifically independent and neither authorizes HCH transfer.

A direct source audit found that the previous U-A1 runner was not source-faithful: it forced ETTh1 through `--data custom` / `Dataset_Custom` even though the pinned official UEC source maps `--data ETTh1` to `Dataset_ETT_hour`, and it reimplemented the 336-step AR rollout instead of using official modes 2/5/6. Historical `0.596529/0.588395` Host/UEC MSE values remain immutable evidence for that custom runner but no longer settle official paper-native UEC fidelity. The new UEC recovery first tests a fresh source-native ETTh1 TimeMixer-96 checkpoint and official AR336 Host gate; no UEC corrector is trained unless the Host MSE is within 7.5% of paper. The paper text describes a 70%-train split while the released ETTh1 loader uses fixed ETT borders, so a remaining mismatch is a disclosed paper-code reproducibility discrepancy, not authorization for split search. Controlling amendment: `experiments/evidence/hch_baseline_paper_fidelity_20260910/05_uecstd_ompb_minimal/adjudication/UECSTD_PROTOCOL_CORRECTION_AMENDMENT_20260911.md`; recovery protocol: `docs/current/HCH_UECSTD_HOST_FIDELITY_RECOVERY_PROTOCOL_20260911.md`.

OMPB is now decoupled from UEC and must run to its own terminal state regardless of UEC PASS/FAIL/BLOCKED. Its minimal reproduction remains exactly two official online-shift TCN/96 anchors: `ETTh1->ETTh2` and `Weather_train->Weather_test_far`, with final-paper exact rows hashed before training, official datasets/provenance, fresh source TCNs, frozen-backbone OMPB, and a future-label perturbation causality audit. No static surrogate, extra anchor, hyperparameter rescue, or HCH unified-DA transfer is allowed. Controlling protocol: `docs/current/HCH_OMPB_STANDALONE_MINIMAL_PAPER_FIDELITY_PROTOCOL_20260911.md`. Launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_UECSTD_HOST_RECOVERY_AND_OMPB_STANDALONE_20260911.md`.

## Market-State Coupling rejected for Amplitude; State-Excursion Shape closure authorized (2026-09-10)

The fixed six-cell Market-State Coupling diagnostic completed validly with `HCH_MARKET_STATE_COUPLING_NOT_SUPPORTED`. Independent adjudication is `experiments/evidence/hch_market_state_coupling_20260910/adjudication/INDEPENDENT_MARKET_STATE_COUPLING_ADJUDICATION.md` and controls interpretation. C0 replay/legality passes; state is nondegenerate; however `rho_shape -> O2 distance residual` is weak/mixed across Hosts and the one-feature chronological ridge improves the constant-zero distance baseline in only 2/6 cell medians. `A_state1` was correctly not trained. State-aware learned Amplitude based on this scalar coupling is closed.

The stronger result is upstream: 4/6 cells are classified `SHAPE_LIMITED`, including both GANSU_DA Hosts. On frozen high-state GANSU days, S1 Shape cosine drops by about 0.062/0.185 and wrong-hemisphere rate increases by 25/20 percentage points relative to low-state days. The current S1 sees current forecast-time state levels but has no explicit representation of their deviation from recent same-hour state history. This supports exactly one data-representation closure rather than deeper architecture.

The active experiment is now `docs/current/HCH_STATE_EXCURSION_SHAPE_CLOSURE_20260910.md`. It freezes the S1 Daily-Patch GRU32 topology, Shape target/loss, historical tokens, Host/splits/OOF, pooled fixed-alpha distance and deleted Verification. The only change is to append already-audited signed seven-day same-hour state excursions (fixed `xi/10`) to the current-day Shape patch. Mandatory panel remains `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds 7/17/37. No conditional Amplitude, Gate, deeper GRU/Transformer, market expert, new market, Shandong or protected/final access is allowed. PASS only permits later design of public-China expansion.

## Market-State Coupling diagnostic authorized after Shape-only Amplitude rejection (2026-09-10)

The Anchored Compact Amplitude closure validly rejected the 26-parameter Shape-only conditional distance learner: median O2-over-fixed-alpha capture is only 2.01% and amplitude-oracle ordering is near zero. This rejects `a=f(u,||v||)` as the leading conditional-distance implementation, but does not reject the scalar Amplitude object because exact per-day O2 ray-distance headroom remains large (~10.8--22.7% absolute gain vs Host across the six development cells).

The new hypothesis is information-limited rather than capacity-limited: normalized Shape deliberately removes residual magnitude, so `[u,||v||]` is not guaranteed to contain day-specific distance information. The next stage therefore tests whether legally available forecast-time electricity-market state supplies that missing information. Controlling documents are `docs/current/HCH_POST_AMPLITUDE_MARKET_STATE_COUPLING_DESIGN_20260910.md` and `docs/current/HCH_MARKET_STATE_COUPLING_EXECUTION_PROTOCOL_20260910.md`.

The default executable method remains `S1 Daily-Patch GRU32 Shape + pooled fixed alpha`; learned Verification stays removed. The authorized experiment first performs C0--C4 diagnostics on `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, constructing semantic forecast-time market-state roles and robust same-hour seven-day excursion coordinates. Only if preregistered coupling gates show that state explains O2 distance rather than primarily Shape failure may the same run fit one and only one additional parameter `beta` in a state-anchored scalar Amplitude. No second state feature, nonlinear Amplitude, Shape modification, Gate, router, retrieval, Shandong, hyperparameter search or protected/final access is authorized.

## Anchored Compact Amplitude rejected; market-state coupling becomes the active design question (2026-09-10)

The fixed six-cell Anchored Compact Amplitude closure completed validly with `HCH_ANCHORED_COMPACT_AMPLITUDE_COMPLETE_FOR_ADJUDICATION`. Independent adjudication: `experiments/evidence/hch_anchored_compact_amplitude_20260910/adjudication/INDEPENDENT_ANCHORED_COMPACT_AMPLITUDE_ADJUDICATION.md`.

A0 PASS. The tested 26-parameter Shape-only conditional distance learner on `[u_hat || ||v||]` is rejected. Although its OOF utility is nonnegative in 6/6 cells and improves fixed-alpha OOF utility in 4/6, final A2 fails: Host-nonworse 5/6, improves frozen S1 in only 3/6, median relative Overall-MAE gain 5.79% (<7% bar), PJM/PatchTST remains below Host, and maximum Normal-MAE harm is 1.06% (>1% bar). A3 is decisively unsupported: median capture of O2-over-fixed-alpha headroom is only 2.01%, positive in 4/6, and amplitude-vs-oracle Spearman is approximately -0.044 to +0.126 across cell medians.

The scientific interpretation is not `Amplitude is unnecessary`. Exact per-day scalar O2 remains very large (about 10.78%--22.73% relative gain across the six cells), so conditional ray distance remains a real oracle object. What is rejected is the assumption that normalized Shape output plus directional concentration contains enough information to predict that distance. This is also mathematically expected: Shape supervision `u*=e/||e||` is invariant to residual scale, while `||v||` is a directional concentration statistic rather than a magnitude target. Increasing MLP depth on the same Shape-only input is therefore not justified.

The leading executable method returns to `S1 Daily-Patch GRU32 Shape + pooled OOF fixed alpha`; learned Verification remains removed. The next design question is evidence-first: does publication-time-legal market state contain the missing per-day distance information? Controlling design discussion: `docs/current/HCH_POST_AMPLITUDE_MARKET_STATE_COUPLING_DESIGN_20260910.md`. No state-aware Amplitude experiment is authorized yet. First test state-distance coupling and whether high-state-intensity failure is primarily Shape or Amplitude. If no coupling exists, keep fixed alpha and do not deepen Amplitude. If stable coupling exists, the first allowed candidate is a one-coefficient state-conditioned scalar anchored at fixed alpha, not an MLP/router/expert.

## China market-state / ICDE paper direction registered without interrupting active Amplitude closure (2026-09-10)

The paper-facing application priority is sharpened: public Chinese provincial **DA->DA** markets should be the main motivation/stress domain, while established international DA benchmarks should demonstrate that the same repair formulation generalizes rather than overfits one market family. The desired eventual evidence pattern is that strong output-only adapters may be less reliable under data-rich, rapidly changing market states, HCH converts the available forecast-time operating information into several-percentage-point or larger improvements on multiple public Chinese markets, and remains competitive with or better than admitted paper-faithful post-hoc baselines internationally. This remains a target narrative, not a current result; geography itself must never be used as the definition of difficulty.

A new design-only note is `docs/current/HCH_CHINA_MARKET_STATE_ICDE_DESIGN_20260910.md`. It does **not** modify or interrupt the currently authorized `HCH_ANCHORED_COMPACT_AMPLITUDE_DESIGN_20260910.md` run. After that run is adjudicated, richer Chinese covariates may be used only through an evidence-gated data representation, not a China-specific expert/router.

The leading next representation idea is a small **Forecast-Time Market-State schema** over audited semantic roles such as demand/load, renewable generation, supply/bidding margin and interchange/non-market generation, each with explicit availability and provenance. A deterministic same-hour robust state-excursion coordinate measures how unusual the target-day operating state is relative to the previous seven same-hour observations. No tail expert, market embedding, Transformer, graph, retrieval or feature-attention module is justified by this registration.

If the active 26-parameter Amplitude already succeeds on China and international cells, no extra state-conditioning learner should be added. If it remains specifically weak on China, first run a diagnostic market-state coupling audit. Only if forecast-time state excursion materially predicts O2 scalar distance / amplitude error / proposal utility is one minimal `+1 coefficient` State-Anchored Amplitude candidate justified. If Shape rather than distance is the China-specific bottleneck, enrich the existing S1 current-day patch with the preregistered state-level/excursion channels while keeping the same GRU-32 topology and Shape target/loss.

ICDE 2027 positioning should be explicitly data-centric: heterogeneous temporal-state integration, forecast-publication legality, original-origin residual provenance, horizon-aligned repair tables, availability masks, same-task benchmark manifests, lightweight temporal parameter sharing and parameter/latency reporting. The venue note is `paper/04_literature/ICDE2027_MARKET_STATE_REPAIR_POSITIONING_20260910.md`. The paper draft now contains a market-state-aware EPF Related Work subsection, but all claims that Chinese markets are harder or that state-aware HCH wins broadly remain `PENDING-EVIDENCE` until full public-China + admitted-baseline experiments exist.

## UEC-STD minimal reproduction fails at Host fidelity; OMPB remains unrun (2026-09-11)

The optional breadth-completion stage returned `HCH_UECSTD_OMPB_MINIMAL_FIDELITY_PARTIAL_OR_FAILED`. U-A1 ETTh1/TimeMixer/336 completed under the paper-authority-resolved UEC protocol, but the uncorrected Host itself is far outside the registered paper-fidelity gate: reproduced Host MSE/MAE `0.596529/0.547638` versus paper `0.465/0.449` (MSE error about 28.3%). UEC-STD gives `0.588395/0.544841` with paper beta `0.3`, only 1.36% MSE gain versus the paper's ~3.44%; current official auto-beta diagnostic selects `1.0`, not the paper `0.3`. Because Host fidelity fails first, this run does **not** establish that UEC-STD itself is ineffective; it establishes that the current official TimeMixer/ETTh1 reproduction path does not recreate the paper anchor.

U-A2 Weather/TimeMixer/336 did not complete and its partial checkpoint is excluded. OMPB was never entered because the execution order stopped after the incomplete/failed UEC stage; no `OMPB_TARGET_REGISTRY.json` exists and no OMPB data were read. Therefore neither UEC-STD nor OMPB is paper-faithfully admitted. The overall baseline pool is **not completely reproduced**: δ-Adapter remains accepted with disclosed partial fidelity; PIR remains exact paper-faithful; COSA remains partial/unresolved; UEC-STD is not reproduced; OMPB is not reproduced.

This optional failure does not invalidate the mandatory offline comparison layer (`Host / direct residual / δ-Adapter / PIR`). If OMPB is still desired, it should be decoupled from UEC and run under its own two-anchor online-shift protocol. UEC should only be revisited if there is a specific need for it in the final manuscript, starting from TimeMixer Host/data/config reproduction rather than tuning UEC itself.

## Baseline paper-fidelity workstream closed for current method stage (2026-09-10)

The baseline anchor program is now **closed for the current HCH method-development stage by human adjudication**. This does not mean every candidate baseline was fully reproduced. It means the mandatory headline offline layer is sufficiently established: δ-Adapter Ada-Y is `DELTA_ADAPTER_ANCHOR_ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` (2/3 exact anchors PASS; Electricity remains a disclosed borderline FAIL), while PIR is `PIR_FULL_OFFICIAL_ANCHOR_ACCEPTED / PAPER_FAITHFUL_EXACT_ACCEPTED` on both mandatory PatchTST and TimeMixer anchors. Frozen Host/Identity and matched direct residual remain project-defined controls.

COSA remains a separate online/TTA comparator and is **not fully paper-native reproduced**. Latest-code B3 retains 3/4 method-anchor PASS with C-A2 COSA-P FAIL under the post-paper partial-GT implementation. The one paper-era provenance-resolution run was stopped by the user after C-PE1 Host completed and passed (`0.435901 ± 0.004820` MSE vs paper `0.4312`, 1.09% error); no paper-era COSA-F/P method run or C-PE2 was executed. Current COSA label is `COSA_ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`. OMPB was not run and remains `OPTIONAL_ONLINE_SHIFT_BASELINE_NOT_REPRODUCED`; UEC-STD remains secondary and not paper-faithful. These do not block the main offline HCH comparison.

Controlling closure: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/BASELINE_FIDELITY_WORKSTREAM_CLOSURE_20260910.md`. No additional baseline anchor reproduction is active. When the HCH method is ready for final comparison, the next mandatory baseline step is a frozen unified-DA transfer protocol for the admitted offline implementations, not more paper-anchor tuning. Resume COSA paper-era reproduction only if the final manuscript explicitly needs an online/TTA block.

## Minimal UEC-STD + OMPB reproduction reopened as optional breadth completion (2026-09-10)

After closure of the mandatory offline Tier-A baseline workstream, the user requested one final **minimal** reproduction pass for UEC-STD and OMPB. This does not reopen full-table baseline reproduction and does not block the active HCH method stage.

UEC-STD status: the historical `src/baselines/uec_std.py` remains `LIMITED/PILOT`, because it uses only selected official decomposition/linear-ECM pieces with wrapper-specific 2-epoch training, split/normalization and cache residual construction rather than the paper's native autoregressive correction-data generation, seasonal/trend UEC training, Huber loss, validation coefficient selection and AR evaluation. The new reproduction must run the vendored official repo end-to-end. Minimal anchors are fixed to TimeMixer/336 on ETTh1 (paper Host `0.465/0.449`, UEC-STD `0.449/0.444`) and Weather (Host `0.265/0.292`, UEC-STD `0.256/0.290`).

OMPB status: no accepted HCH implementation exists, and no static surrogate is allowed. OMPB remains an **online source→target shift** method with a frozen backbone, Bayesian residual head, Host-fallback gate, source-risk/posterior-shift/mismatch certificate terms and strict predict-then-update chronology. Minimal anchors are preregistered as TCN/96 on `ETTh1→ETTh2` and `Weather_train→Weather_test_far`, using official paper backbone epochs 20 and 5 respectively. Exact paper metric rows must be extracted from the final UAI/PMLR paper and hashed before any training; if they cannot be unambiguously recovered, fail closed rather than estimate.

Controlling design at that time was `docs/current/HCH_OMPB_UECSTD_MINIMAL_PAPER_FIDELITY_REPRODUCTION_20260910.md`. The first UEC preflight correctly returned `HCH_UECSTD_PROTOCOL_BLOCKED` without reading data/training because repository helper files conflicted. A paper audit then fixed the UEC training/correction details (100 steps, batch 64, Huber/SmoothL1, true AR, official `ecm=linear` two-stage MLP, kernel 25; ETTh1 seasonal/trend `0.8/0.2` with paper beta `0.3`; Weather `0.6/0.4` with beta `0.1`). Its earlier assumption of one common 70/10/20 backbone split is **historical and superseded for ETTh1** by the later source-fidelity amendment: released official source maps ETTh1 to `Dataset_ETT_hour` fixed ETT borders, while Weather remains `Dataset_Custom`. See `UECSTD_PROTOCOL_CORRECTION_AMENDMENT_20260911.md` and the newer source-native recovery protocol. The prior blocker and failed run remain immutable historical evidence.

## Parallel baseline paper-fidelity reproduction workstream opened (2026-09-10)

A separate baseline-fidelity workstream is now authorized alongside the active HCH method stage. It does **not** change HCH, the Unified-DA task, current Host caches, or protected/final evaluation.

The first paper-anchor stage for δ-Adapter completed and stopped correctly for adjudication. D-A1 Weather and D-A2 Traffic pass exact paper-fidelity gates; D-A3 Electricity passes Host and absolute method metric gates but misses the preregistered relative-gain tolerance by only 0.21 percentage points (`3.17%` reproduced vs `5.38%` paper; allowed deviation `2.00 pp`, observed `2.21 pp`). No tolerance is changed and D-A3 remains a formal anchor FAIL. Human adjudication accepts δ-Adapter Ada-Y at the **method level** as `DELTA_ADAPTER_ANCHOR_ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` because the controlling paper-facing rule already requires at least one exact anchor pass and no core substitution; here 2/3 anchors pass. The old 2-epoch cache wrapper remains invalid as a strong baseline. Controlling adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_DELTA_ADAPTER_ANCHOR_ADJUDICATION.md`.

B2 PIR full-official reproduction has now completed and is independently adjudicated as `PIR_FULL_OFFICIAL_ANCHOR_ACCEPTED`: both mandatory ETTh1 horizon-96 anchors pass all Host/Method/gain gates (PatchTST reproduced MSE gain 8.869% vs paper 8.537%; TimeMixer 3.567% vs 3.646%). Full-path evidence confirms official pretrained backbone, QualityEstimator, Transformer Refiner, retrieval index and learned local/global combination with no ridge substitution. Controlling adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_PIR_ANCHOR_ADJUDICATION.md`.

Offline Tier-A anchor status is stable: δ-Adapter Ada-Y = `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`; PIR full official = `PAPER_FAITHFUL_EXACT_ACCEPTED`. COSA B3 then ran 10 seeds on official latest commit `527c0fe`: C-A1 COSA-F/P PASS and C-A2 COSA-F PASS, but C-A2 COSA-P fails severely (`0.589729` vs paper `0.5320`, gain `-4.624%` vs `+4.299%`). This B3 result remains valid for the latest-code / strict delayed-partial-GT protocol and is not rewritten.

Post-B3 provenance audit found that `527c0fe` is dated 2026-08-13 and explicitly adds `POGT adaptation and MLP adapter`; its `tta/cosa.py` introduces partial-ground-truth masks, pending/full-horizon replay and prediction adjustment. The final ICLR 2026 paper/presentation instead describes the current method as relying on full ground truth and lists partial-ground-truth real-time deployment as future work. The paper-era first official commit is `43a8c8d` (2026-03-04). A second discrepancy is that the paper text states 7:1:2 while the paper-era official ETTh1 loader hard-codes `TRAIN_RATIO=0.6`, `TEST_RATIO=0.2`. Therefore overall COSA paper-native fidelity is **UNRESOLVED DUE SOURCE-VERSION / PAPER-CODE PROTOCOL MISMATCH**, not a final method failure.

Exactly one provenance-resolution rerun is authorized under `docs/current/HCH_COSA_PAPER_ERA_CODE_NATIVE_REPRODUCTION_20260910.md`: use commit `43a8c8d`, its native ETTh1 split, its paper-era full-ground-truth batch adaptation, fixed 10 seeds and unchanged K/S/B/LR, and rerun C-A1/C-A2 F/P coherently. No further source/config search is allowed afterward. Controlling adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_COSA_ANCHOR_ADJUDICATION.md`; prompt: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_COSA_PAPER_ERA_PROVENANCE_RESOLUTION_20260910.md`. OMPB, transfer-band derivation and HCH transfer remain closed until this resolves.

The audit establishes that the historical `src/baselines/delta_adapter.py`, `pir.py`, and `uec_std.py` implementations are development pilot/proxy wrappers rather than end-to-end paper reproductions. The main issues are not only epochs: the current Ada-Y wrapper changes official bounded-delta scaling, optimizer/schedule, width and full-batch training; the PIR wrapper replaces the official local Transformer refiner/retrieval-value/quality-loss path with ridge/proxy logic; the UEC wrapper does not reproduce the source correction-data/trend-season/long-horizon training pipeline. Historical evidence remains valid only for its registered pilot scope and must not be used as final SOTA evidence.

Controlling documents are `docs/current/HCH_BASELINE_PAPER_FIDELITY_AUDIT_AND_REPRODUCTION_DESIGN_20260910.md` and `docs/current/HCH_BASELINE_PAPER_FIDELITY_EXECUTION_PROTOCOL_20260910.md`. New implementations must be isolated under `src/baselines/paper_fidelity/**`; historical wrappers/evidence remain replay-only. Exact paper anchor cells are preferred when reference data/backbones are locally available. If not, the executor must derive and freeze a paper-based relative-gain target from same-backbone/closest-horizon paper cells **before** reading unified-DA transfer outcomes.

Baseline track policy: offline headline = Host / direct residual / δ-Adapter Ada-Y / PIR, with UEC-STD secondary if task-compatible; COSA is a separately labeled online/TTA comparator; OMPB is only admissible under an explicit online distribution-shift protocol and must not receive a guessed static-EPF target; NetBurst remains Related Work rather than a forced comparator. Final admission statuses distinguish exact paper reproduction from transfer-estimate validation.

Initial paper-derived transfer expectations are intentionally method/backbone-specific: δ-AdaY central MSE gain ~2–6% (MAE floor ~1%); PIR PatchTST MSE ~2–6%, MAE ~1.5–4%; PIR TimeMixer MSE ~0.5–3.5%, MAE ~0–2%; COSA short-horizon transfer ~1–5% MSE in the online/TTA track pending exact anchors; UEC-STD direct-24h is a modest secondary comparator (~-1–3% MSE, ~0–2% MAE). These ranges are preregistered estimates, not observed project results.

## Parallel baseline paper-fidelity reproduction track authorized (2026-09-10)

A separate baseline-admission/reproduction program is now authorized in parallel with the active HCH method-development closure. Controlling protocol: `docs/current/HCH_BASELINE_PAPER_FIDELITY_REPRODUCTION_AUDIT_20260910.md`; short executor prompt: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_GOAL_PROMPT_20260910.md`. This track must not modify HCH architecture or overwrite old pilot baseline evidence.

Current baseline audit conclusion: existing project δ-Adapter and PIR results are **not paper-faithful reproductions**. `src/baselines/delta_adapter.py` loads the official Ada-Y post network but changes material training semantics (custom residual scaling, LR `1e-3` instead of the paper/official `1e-4` setting, full-batch training, invented hidden-size rule, no validation/early stopping, ignored δ argument). `src/baselines/pir.py` is a `LIMITED_PROXY`: it loads the official QualityEstimator but replaces the official local Refiner with ridge regression and uses a project-specific simplified retrieval/revision path. More epochs alone cannot repair these fidelity gaps.

Baseline reproduction is split into two layers. R1 first reproduces paper-native anchors with official code/data/backbones and requires the unadapted Host itself to be within 5% of the paper metric before adapter fidelity can be judged. Exact adapted metrics must also be within 5%, with relative-gain agreement checked separately. Only after an anchor passes may R2 transfer the verified implementation to the unified HCH DA task. When HCH dataset/backbone/horizon differs, the transfer success range must be frozen from published same-backbone/closest-horizon relative gains before HCH outcomes are run; no outcome-driven tuning is allowed.

Priority Tier-A offline baselines are δ-Adapter Ada-Y and full PIR. δ Ada-X+Y is an optional stronger-access comparator. COSA and OMPB are separate online/TTA baselines and must preserve predict-then-update/source-target-shift semantics rather than being converted into offline methods. Current 10-epoch Ada-Y/PIR-proxy numbers remain development bars only and cannot support a paper SOTA claim.

## Unified DA S1 Shape passes materiality; compact Amplitude is the active next closure (2026-09-10)

The main paper task is now unified across headline international and Chinese markets as **next-24h day-ahead electricity-price forecasting**. Chinese `*_RT` DA-information→RT targets are retired from the paper-facing main task. Public China target IDs are `GANSU_DA / SHAANXI_DA / NINGXIA_DA / QINGHAI_DA`; the target-day `日前电价` is never an input. Historical `GANSU_RT` evidence remains diagnostic-only.

The Unified-DA Shape Engineering canary completed validly with `HCH_UNIFIED_DA_SHAPE_ENGINEERING_CANARY_COMPLETE_FOR_ADJUDICATION`. Independent interpretation is `experiments/evidence/hch_unified_da_shape_engineering_20260910/adjudication/INDEPENDENT_UNIFIED_DA_SHAPE_ADJUDICATION.md`.

S1 Daily-Patch Shape is promoted as the **leading Shape implementation candidate**: 5/6 cells are non-worse than Host, 4/6 reach >=3% relative Overall-MAE gain, 3/6 reach >=5%, median gain is 5.85%, maximum gain 10.74%, maximum Normal-MAE relative harm 0.85%, and S1 beats the fixed development comparator in 4/6 cells while using only ~14k--19k parameters. The gains are seed-stable in five cells. The single real failure is PJM/PatchTST, where S1 consistently lowers Shape cosine and yields about -1.39%, +0.22%, -0.87% across seeds; this is a genuine generalization mismatch, not seed noise.

The decisive post-S1 result is **large conditional ray-distance headroom**. Holding the exact learned S1 direction fixed, the per-day MAE-optimal scalar reaches O2 gains of ~10.8--22.7%. Relative to S1, extra O2 headroom is ~8.5--13.5 percentage points in every cell. This satisfies the project's prior condition for reopening one compact conditional Amplitude after Shape improvement. PJM/PatchTST is especially diagnostic: the same weak S1 direction can reach 12.23% gain with the correct scalar distance, so a deeper Shape model is not the smallest justified next change.

The active design is therefore `docs/current/HCH_ANCHORED_COMPACT_AMPLITUDE_DESIGN_20260910.md`. Freeze S1 Shape, Host, task/data contract and OOF semantics. Test exactly one **26-parameter anchored compact Amplitude** on `[u || ||v||]`, initialized to reproduce the safe pooled scalar `alpha` and trained only from chronological OOF S1 Shape predictions under direct MAE ray loss. Learned Verification/Gate remains deleted; the old 648-D Amplitude, Bridge/router/retrieval, extra Shape depth, Transformer/attention and hyperparameter search remain closed.

ICDE positioning remains data/engineering-aware rather than architecture-heavy: same-task benchmark manifests, leakage-safe original-origin residual provenance, forecast-time covariate joins, day/horizon-aligned repair representation, lightweight temporal parameter sharing, stage-wise OOF calibration, and parameter/latency reporting. See `paper/04_literature/ICDE_TIME_SERIES_ENGINEERING_POSITIONING_20260910.md`. Final paper-faithful baseline reproduction remains a separate later stage; current 10-epoch Ada-Y/PIR-proxy runs are development bars only.

## Compact Verification rejected; material gain + market-structure audit is active (2026-09-09)

The fixed six-cell Compact Verification closure completed validly with `HCH_COMPACT_VERIFICATION_CLOSURE_COMPLETE_FOR_ADJUDICATION`. Independent adjudication is `experiments/evidence/hch_compact_verification_closure_20260909/adjudication/INDEPENDENT_COMPACT_VERIFICATION_ADJUDICATION.md` and controls interpretation.

The 49-parameter verifier is removed from the leading method: selective execution is non-worse than calibrated always-repair in only 2/6 cells, median gain versus always-repair is negative, and B4 utility ordering is inverse in 4/6. No verifier rescue, threshold search or larger gate is allowed. Proposal-level utility remains a diagnostic concept, not a required learned module.

The leading proof-of-concept proposal remains `Delta=s*alpha*u_hat` with linear 24h Shape, causal residual scale and one pooled OOF MAE calibration scalar. It is Host-safe on the six consumed canary cells but is **not yet paper-strength SOTA**: relative Overall-MAE gains are only about `0.11%, 1.12%, 1.30%, 0.49%, 0.76%, 5.34%`. Sub-1% gains are insufficient for the intended framing; the next research target is material gain, with several-percentage-point improvements as the development bar.

Current GANSU comparator harm cannot yet be attributed to an intrinsic Chinese-market property. The canary used `DeltaAdapter(epochs=2)` and `PIR(epochs=2)`, while vendored official defaults are materially longer (delta-Adapter train epochs 10; PIR refine epochs 10), and the maintained PIR cache wrapper contains a registered refiner substitution. GANSU is also a `DA information -> RT price` task with legal current-day DA price, load, renewable and bidding-space forecasts that the fixed univariate Host and canary cache-compatible adapters do not fully consume. The current result supports only a univariate/output-only stress-case observation, not a final market claim.

The only authorized next scientific stage is `docs/current/HCH_MATERIAL_GAIN_AND_MARKET_STRUCTURE_AUDIT_20260909.md`, using the same six consumed cells. It performs: fixed baseline-fidelity convergence checks; GANSU/Track-A market-state signal diagnostics; O0--O4 oracle headroom decomposition for Verification/Amplitude/Shape; and one diagnostic P-State variant that keeps the same linear Shape but appends only forecast-time legal covariates. No GRU, learned Verification, conditional Amplitude, full-panel, Shandong or protected/final execution is authorized. Stop after audit adjudication.

## Calibrated-Ray closure passes; conditional Amplitude removed; one compact Verification closure is active (2026-09-09)

The fixed six-cell calibrated-ray closure completed validly with `HCH_CALIBRATED_RAY_CLOSURE_COMPLETE_FOR_ADJUDICATION`. Independent adjudication is `experiments/evidence/hch_calibrated_ray_closure_20260909/adjudication/INDEPENDENT_CALIBRATED_RAY_ADJUDICATION.md` and controls interpretation.

The leading proposal is now frozen as `Delta_d = s_d * alpha * u_hat_d`, where `u_hat` is the learned linear 24h Directional Shape, `s_d` is the causal residual scale, and `alpha` is one pooled nonnegative MAE calibration scalar fitted only from chronological OOF Shape/residual pairs. The 648-D conditional Amplitude network is removed from the leading method. M0-Compact conditional Amplitude and GRU-32 remain closed.

Calibrated-Ray is Overall-MAE non-worse and strictly better than Host in 6/6 cells; Tail-MAE improves 5/6; Normal-MAE improves in all six cell medians. It remains within 1% of the best admitted post-hoc comparator in 5/6 Overall and 4/6 Tail cells. This is development evidence, not final benchmark superiority.

Verification remains scientifically justified for one final minimal learnability test because calibrated proposals are still harmful on roughly 29–48% of DIAG_EVAL days and oracle selective headroom is positive in all six cells (~0.8–4.7% relative MAE). The only authorized next learner is a 49-parameter linear sigmoid verifier on `[v || delta_tilde]`, trained from chronological calibrated OOF proposals with utility-weighted BCE and fixed `q>0.5`. Controlling design: `docs/current/HCH_COMPACT_VERIFICATION_CLOSURE_20260909.md`. If this verifier does not safely improve calibrated always-repair, learned Verification is removed; no capacity rescue is allowed before full-development design.

## M0 R2 adjudicated; one-parameter calibrated ray is the active next closure (2026-09-09)

The fixed six-cell R2 completed validly with `HCH_M0_R2_CANARY_COMPLETE_FOR_ADJUDICATION`. Independent adjudication is `experiments/evidence/hch_minimal_repair_m0_20260909_r2/adjudication/INDEPENDENT_M0_R2_ADJUDICATION.md` and controls interpretation.

R2 confirms that removing the former `J>0` Amplitude filter improves the high-dimensional conditional learner but does not make it reliable: OOF proposal utility, ray regret, amplitude-oracle gap and always-repair MAE improve in 4/6 cells, yet learned OOF utility is positive in only 2/6. The decisive result is that the preregistered chronological constant-Amplitude probe beats the learned R2 Amplitude in all six cell medians; the learned scalar has near-zero/negative correlation with the exact per-instance ray oracle (`cell-median Spearman ~ -0.267 to +0.073`). Therefore the current 648-D state-conditioned Amplitude head is not supported.

Scalar ray Amplitude itself remains strongly supported. The same learned Shape plus the diagnostic constant normalized amplitude improves DIAG_EVAL Overall-MAE over Host in 17/18 seed×cell runs and in all 6/6 cell medians. This result is diagnostic-only because the constant probe was not the registered R2 method, but it is sufficient to promote a simpler closure hypothesis: after the day-specific causal residual scale `s_d`, normalized repair distance may need only one fitted scalar calibration rather than a conditional network.

The active next candidate is therefore `Delta_d = s_d * alpha * u_hat_d`, where `alpha>=0` is fitted **exactly** from chronological OOF Shape predictions by minimizing aggregate MAE. The exact pooled solution is the nonnegative weighted median over all valid ratios `e_ih/u_ih` with weights `|u_ih|`. Controlling design: `docs/current/HCH_CALIBRATED_RAY_CLOSURE_EXPERIMENT_20260909.md`.

Verification is no longer assumed to survive automatically. R2 learned Verification remains unsupported on R2 proposals, but the safer calibrated-ray proposal changes the decision problem. The closure experiment must first quantify harmful-proposal rate and oracle selective headroom for calibrated always-repair. If headroom is material/consistent, retain the Verification object and later design a compact verifier; if headroom becomes small/inconsistent and calibrated always-repair is Host-safe, simplify/remove learned Verification rather than preserve it for symmetry.

No GRU/MLP/Transformer or conditional Amplitude network is the next step. The post-R2 capacity ladder is now `Calibrated-Ray -> optional M0-Compact (v->scalar / [v||Delta]->q) -> optional M1-GRU32`, with each escalation requiring separate evidence. The calibrated-ray six-cell closure is authorized only on the existing consumed DIAG_FIT/DIAG_EVAL panel; full markets, Shandong and protected/final evaluation remain closed.

## M0 R1 canary valid but not supported; Amplitude training semantics is the active bottleneck (2026-09-09)

The fixed six-cell M0 R1 canary completed validly with `HCH_M0_R1_CANARY_NOT_SUPPORTED_FOR_FULL_EXPANSION`: G0 PASS, G1 PASS, G2 FAIL, G3 PASS, G4 FAIL. Independent adjudication is `experiments/evidence/hch_minimal_repair_m0_20260909_r1/adjudication/INDEPENDENT_M0_R1_ADJUDICATION.md` and controls interpretation.

Directional Shape remains supported as a scientific object: cell-median horizon cosine is positive in 6/6 cells (`~0.0876–0.5249`) and positive-hemisphere rate exceeds 0.5 in 6/6. However, except PJM/TimeMixer, cosine is modest; M0 flattened-linear representation sufficiency is not established.

The complete M0 is not Host-safe/effective enough for expansion: Overall-MAE non-worse/strict-improvement vs Host is 0/6 and 0/6, Tail-MAE improves 3/6, and maximum Normal-MAE relative harm is 7.49%. G3 comparator-relative PASS must not be mistaken for effectiveness because the admitted simple adapters are themselves mostly harmful vs Host on this canary.

The strongest mechanism result localizes the first redesign to Amplitude. For every one of 18 market×Host×seed rows, the exact MAE-optimal nonnegative scalar along the **same OOF predicted Shape** yields positive mean utility; cell-median oracle scalar gain is positive in all six cells, while learned OOF proposal gain is negative in five of six. Thus one-scalar Amplitude is not rejected. The current `J>0`-conditional Amplitude learner is not supported: it trains on only about 58–73% of repair-available OOF states but emits Softplus amplitudes on all repair-available states, creating unsupervised extrapolation. Final Amplitude support is only 115–144 rows on LAGO and 22–23 on GANSU_RT for a 649-parameter conditional head.

The leading R2 mathematical correction is to define/train Amplitude as the full nonnegative MAE ray projection `a*=argmin_{a>=0} ||e-a u_OOF||_1` on **all repair-available OOF Shape rows**, with zero as a valid distance. This does not collapse Verification: zero is a distance along the ray; Verification still estimates expected utility of the learned complete proposal under uncertainty. Design: `docs/current/HCH_M0_R2_RAY_PROJECTED_AMPLITUDE_DESIGN_20260909.md`.

Current linear Verification is not supported for the R1 proposal distribution (full G4 passes only 1/6), but verifier capacity is not yet isolated because upstream proposal mean utility is negative in 5/6 cells. Do not enlarge the verifier or tune `q>0.5` before improving proposal quality.

M0 is shallow but statistically high-dimensional: Shape is ~15k parameters; Amplitude 649; Verification 651, versus 327/56 Shape rows, 115–144/22–23 Amplitude rows, and 147/26 complete verifier labels. M1 GRU-32 therefore remains a plausible later **temporal bottleneck/parameter-sharing** option, not a generic depth escalation, but it is not the next automatic change. First test the one-factor Amplitude semantic correction. The user has now authorized the fixed six-cell R2 development execution through `docs/current/HCH_M0_R2_EXECUTION_PROTOCOL_20260909.md`. R2 changes only Amplitude training eligibility from `repair_available AND J>0` to all repair-available rows with valid chronological OOF Shape. It also runs one diagnostic-only chronological constant-Amplitude probe. Evidence must be isolated under `experiments/evidence/hch_minimal_repair_m0_20260909_r2/**` and execution must stop for human adjudication before any capacity change.

## M0 first canary preflight invalidated by partial historical Host-residual coverage; one global mask-contract repair registered (2026-09-09)

The first fixed six-cell M0 canary stopped correctly at G0 with `HCH_M0_CANARY_INVALID` before any new M0 source package, training, or scientific evaluation. LAGO_DE/LAGO_PJM have complete 168h original-origin historical Host-residual coverage. GANSU_RT does not: for each Host, 30/62 DIAG_FIT rows and 4/21 DIAG_EVAL rows lack at least part of the required 168h historical Host-prediction trail. The missing blocks are identical across PatchTST/TimeMixer, so this is a data/cache-origin coverage property rather than a Host-architecture failure. No raw/S3/S4/protected/final truth was read and no scientific G1-G4 verdict exists.

Independent adjudication treats this as an **input-contract defect**, not method evidence. The mathematical Shape/Amplitude/Verification objects are unchanged. The core data contract now registers one market-agnostic residual-availability mask over the 168h historical residual vector. Partial residual histories are retained with zero-fill only behind the binary mask; the causal residual scale uses only genuinely available residuals. If no historical residual is available at all, the row deterministically returns the frozen Host and remains in evaluation accounting. This availability fallback is not the learned Verification Gate. Complete-history markets carry an all-one mask and retain identical semantics.

The revised M0 state is therefore 624-D: `168 actual + 168 residual + 168 residual mask + 24 future Host + 96 calendar`. Shape remains one linear `624 -> 24` readout; Amplitude remains one linear+Softplus scalar head on `[x || u]`; Verification remains one linear+sigmoid proposal-aware head on `[x || delta || concentration || amplitude]`. No new encoder/module is introduced by the repair. The R1 canary is authorized under the same six cells and must write only to `experiments/evidence/hch_minimal_repair_m0_20260909_r1/**`, preserving the original invalid preflight evidence unchanged.

## Minimal Repair implementation + fixed M0 canary authorized; paper initial draft opened (2026-09-09)

The theory-first Methodology phase is sufficiently closed to start one controlled implementation/development stage. The active method is `Frozen Host -> causal affine-normalized legal state -> normalized 24h Directional Shape -> one nonnegative scalar MAE Amplitude -> concrete proposal -> proposal-level utility-weighted hard Verification`. Historical Bridge/Global/D1/D2/CORN/quantile/router/retrieval machinery is not an implementation compatibility target.

The user explicitly authorizes: (1) a fresh clean implementation in `src/mvp/hch_minimal_repair/**`; (2) implementation tests; (3) one fixed six-cell M0 canary; and (4) an initial paper draft recording the current Introduction/Related Work/Methodology. Controlling files are `docs/current/HCH_MINIMAL_INTEGRATED_REPAIR_METHODOLOGY_20260909.md`, `docs/current/HCH_MINIMAL_REPAIR_IMPLEMENTATION_SPEC_20260909.md`, and `docs/current/HCH_M0_INTEGRATED_DEVELOPMENT_EXPERIMENT_20260909.md`. Short executor launcher: `experiments/current/hch_minimal_repair_m0/CODEX_GOAL_PROMPT_20260909.md`.

The fixed canary is `LAGO_DE / LAGO_PJM / GANSU_RT × PatchTST / TimeMixer`; methods are Frozen Host, matched-capacity Linear direct residual, δ-Adapter Ada-Y, PIR, and M0. It reuses only already-consumed development roles (`DIAG_FIT/DIAG_EVAL`) and compatible frozen Host caches. The contaminated diagnostic S3/S4 roles remain forbidden as untouched final truth. Even a PASS only allows designing the full development expansion; M1/GRU, all-market execution, Shandong, new data admission and protected/final evaluation remain unauthorized.

The desired eventual paper story is registered as a hypothesis, not a result: simple residual adapters may become less reliable in emerging/high-volatility Chinese spot-market stress cases; the proposed direction–distance–utility repairer should improve those cases while remaining competitive on established international benchmarks. This mature-vs-emerging comparison is **PENDING-EVIDENCE** because the prior diagnostic did not establish a market/regime spectrum. The canary only supplies the first integrated method evidence; full regime-conditioned evidence is a later stage.

Paper prewriting is now authorized. Current manuscript skeleton: `paper/_drafts/HCH_FORECAST_REPAIR_PAPER_DRAFT_20260909.md`. Diagnostic-supported Introduction facts (Host-error concentration, material wrong-direction repair, amplitude necessity, and `Host is bad != this proposal is beneficial`) may be recorded; claims about Chinese-market adapter underperformance, proposed-method superiority, international-benchmark competitiveness, Verification learnability, or M0 sufficiency remain marked pending until their registered evidence exists.

## Integrated minimal Methodology promoted for design discussion (2026-09-09)

The separate Shape / Amplitude / Verification derivations are now integrated in `docs/current/HCH_MINIMAL_INTEGRATED_REPAIR_METHODOLOGY_20260909.md`. The leading paper-level method is `Frozen Host -> causal affine-normalized legal state -> normalized 24h Directional Shape -> one identifiable scalar MAE Amplitude -> concrete proposal -> proposal-level utility-weighted hard Verification`. Historical Bridge/Global/router/CORN/quantile/market-expert machinery remains excluded.

The **core information setting** is deliberately universal: current 24h Host forecast, previous 168h revealed actual prices, previous 168h original-origin Host residuals when available, one global residual-availability mask, and fixed calendar coordinates. Market-specific forecast-known fundamentals may only appear in a separately reported enhanced-information setting; they are not required for the core cross-market claim. No Market-ID or Host-ID is required.

A new causal coordinate normalization is registered. For each forecast day use causal location `m_d = median(past actual)` and causal repair scale equal to `mean(abs(past residual))` over **available** residual entries only, with a training-relative numerical floor. Missing residual entries are zero-filled only behind the explicit mask. Rows with zero available residual history Keep Host exactly. Price/history/Host channels are normalized by `(value-m_d)/s_d` or `residual/s_d`, Amplitude is predicted in normalized units and restored by `s_d`, and MAE utility is divided by `s_d`. Under any positive affine monetary transform `y'=c y+b`, `c>0`, the mask and normalized tensor/Shape are unchanged on repair-available rows, physical Amplitude/proposal scale by `c`, normalized utility is unchanged, and the complete final forecast transforms equivariantly. This is intended as a simple cross-market coordinate property rather than a learned adaptation module.

The **primary optimization/model-selection metric is now overall MAE**, with Tail-MAE a required key secondary metric. MSE remains the clean Shape-geometry diagnostic and secondary empirical metric. No default `MAE + lambda TailMAE` composite is used because it would introduce an arbitrary trade-off coefficient. Under MAE, the fixed-Shape optimal scalar remains the nonnegative weighted-median projection. R1 showed that the prior `J>0` eligibility rule creates a train/inference coverage mismatch, so the current design trains on every repair-available OOF Shape row and lets the nonnegative ray objective choose zero distance when appropriate. The old `J=sum_h u_h sign(e_h)` quantity remains a diagnostic of local ray favorability, not a training filter.

The default capacity remains **M0: no learned shared encoder**. R1 additionally shows that feeding the full 624-D state to the 649/651-parameter downstream heads is statistically high-dimensional for the available OOF sample counts. The registered post-R2 capacity ladder is therefore `M0-R2 -> M0-Compact -> M1-GRU32`: M0-Compact first reuses the semantically aligned 24-D raw Shape output `v` as the downstream repair-state bottleneck (`a=f_A(v)`, `q=f_V(v,Delta)`). Only if representation evidence still requires it may a single-layer GRU-32 replace `x->v`. Downstream heads must continue to consume fixed-coordinate `v/proposal`, not arbitrary fold-specific GRU latent states. See `docs/current/HCH_POST_R2_CAPACITY_LADDER_20260909.md`. No MLP/Transformer/hidden-size grid is registered.

Training is stage-wise, not end-to-end by default. Chronological stacked OOF predictions are mandatory: OOF Shape defines Amplitude training; OOF Shape+Amplitude defines complete OOF proposals; only those complete proposals define Verification utility labels. The leading Amplitude training objective under the MAE headline is direct directional MAE `||e-a_hat u_OOF||_1/H` on all repair-available rows with valid OOF Shape, with `u_OOF` frozen. Verification uses normalized proposal MAE utility and utility-weighted BCE with a fixed Bayes-aligned threshold `q>0.5`.

No new scientific execution is authorized by this design. The next action is to register one clean, minimal integrated-development experiment that tests M0 first and allows M1 only if M0 is demonstrably representation-limited. The contaminated diagnostic S3/S4 slices must not be reused as untouched final evaluation.

## Proposal-level Verification mathematical object promoted for design discussion (2026-09-09)

After Shape and scalar Amplitude generate a concrete proposal `Delta`, Verification is now defined by realized utility `G = L(Host)-L(Host+Delta)`. For legal proposal-aware state `x_V`, the exact Bayes Host-vs-repair decision is `A*=1[E[G|x_V]>0]`. This means ordinary `P(G>0)` classification is not the fundamental target: a high probability of tiny gains can still be dominated by a small probability of catastrophic harm.

Two equivalent minimal learning formulations are registered. Direct gain regression with squared loss has population target `E[G|x_V]`. The leading design is **utility-weighted BCE** with label `Y=1[G>0]`, weight `|G|`, and one sigmoid scalar head. Its population optimum is `q*=E[G_+|x_V]/E[|G||x_V]`, so `2q*-1 = E[G|x_V]/E[|G||x_V]` and the fixed threshold `q>0.5` is exactly equivalent to positive expected proposal utility. This gives a bounded normalized utility margin and avoids a tuned market/Host-specific threshold.

The default verifier input is the legal shared state plus the **concrete proposal**, e.g. `[z || Delta]`; it must not reduce to Host-badness alone. Final execution is hard Host preservation versus full proposal execution: `y_final = y_H + 1[q>0.5] Delta`. A generic soft gate is non-default because it would rescale the proposal and duplicate Amplitude. Verification supervision must use chronological OOF **complete** proposals: OOF Shape -> OOF Amplitude -> OOF `Delta` -> utility label `G`. This explicitly repairs the in-sample-label flaw discovered in the first diagnostic runner.

The exact derivation is recorded in `docs/current/HCH_PROPOSAL_UTILITY_VERIFICATION_MATHEMATICAL_DESIGN_20260909.md`. The surviving mathematical method is now `Frozen Host -> normalized 24h Shape -> identifiable scalar conditional Amplitude -> concrete proposal -> utility-weighted hard Verification`. Remaining design questions are limited to the legal input tensor/shared encoder, stage-wise versus joint optimization while preserving OOF downstream supervision, the primary headline metric that fixes Amplitude/Verification targets, and the chronological cross-fitting schedule. No new scientific execution or final HCH implementation is authorized yet.

## Conditional scalar Amplitude mathematical object promoted for design discussion (2026-09-09)

With Shape now defined as the normalized 24h residual direction `u*=e/||e||_2`, the paper-level Amplitude object is promoted as one **identifiable nonnegative scalar distance along the predicted Shape ray**, not a second horizon-wise residual vector. The Euclidean polar decomposition `e = r u*`, `r>=0`, `||u*||_2=1` is unique; allowing a free horizon-wise amplitude vector would let relative hour-to-hour magnitude move arbitrarily between Shape and Amplitude and would collapse the semantic decomposition back toward an unconstrained residual head. Therefore scalar Amplitude is the default mathematical structure even though the old simple-residual D3 proxy did not empirically settle scalar-vs-horizon-wise capacity.

For a fixed predicted unit Shape `u`, the general object is the loss-conditional one-dimensional projection `a*_ell(e,u)=argmin_{a>=0} ell(e,a u)`. Under MSE, `a*_MSE=[e^T u]_+`; on positive-alignment rows the fixed-Shape MSE repair regret is exactly `(a-a*)^2`, so ordinary squared scalar-target regression is already regret-aligned. Under MAE, the exact scalar is the nonnegative weighted median of `e_h/u_h` with weights `|u_h|` (excluding `u_h=0`), equivalently the optimizer of `||e-a u||_1`. If Shape is perfect, both MSE- and MAE-optimal amplitudes collapse to the same scalar `||e||_2`, proving that metric disagreement arises from Shape error rather than from the scalar factorization itself.

To preserve non-redundant `Where / How much / Should we` semantics, the Amplitude learner must **not** be trained on all rows with clipped target `[e^T u]_+`, because zero targets on wrong-direction rows would make Amplitude learn abstention. Amplitude training is conditional on a usable chronological OOF Shape prediction; for MSE, use only rows with `e^T u_OOF>0` and target `a*=e^T u_OOF`. Rows whose Shape is unusable remain part of the later proposal-level Verification problem. The minimal head is one positive scalar readout, preferably linear plus Softplus on `[shared state || stop-gradient predicted Shape]`; any cross-market physical-unit normalization must reuse a causal scale from preprocessing rather than introduce a market router or new module.

The exact derivation is recorded in `docs/current/HCH_CONDITIONAL_AMPLITUDE_MATHEMATICAL_DESIGN_20260909.md`. The remaining method-level amplitude choice is the primary metric instantiation: MSE projection has the cleanest exact regret identity; MAE projection aligns exactly with a headline MAE objective via weighted-median/L1 projection. This choice must be frozen with the headline metric before protected evaluation. The next mathematical task is proposal-level Verification from chronological OOF complete Shape→Amplitude proposals. No new scientific execution or final HCH implementation is authorized yet.

## Repair diagnostics completed; independent adjudication changes the surviving Methodology objects (2026-09-08)

The D0–D4 Forecast-Repair Diagnostic Evidence Program has completed and is independently adjudicated. Raw/machine evidence lives at `experiments/evidence/hch_repair_diagnostics_20260908/`; controlling adjudication is `experiments/evidence/hch_repair_diagnostics_20260908/adjudication/INDEPENDENT_DIAGNOSTIC_ADJUDICATION.md` with machine-readable `ADJUDICATED_VERDICT.json`. These adjudicated artifacts supersede the auto-generated `verdict.json` wherever they differ.

Scientific conclusions now supported by the diagnostic evidence:

1. **Host-error heterogeneity is real and strong.** 21/24 public market×Host cells satisfy the diagnostic tail screen; median top-5% MSE concentration is 0.4071 and median P95/median day loss is 2.2652. This can support the Introduction-level motivation that a small subset of instances carries disproportionate Host error.
2. **Shape/Alignment is necessary, and a horizon-level direction signal remains after hourly sign is corrected.** Across 144 public market×Host×candidate cells, median hourly wrong-direction rate is 0.4693. Correcting only hourly signs improves MSE relative to the original candidate in 144/144 cells; replacing the full 24h direction while preserving candidate L2 norm adds further positive recovery in 144/144. Median sign share of total direction-oracle recovery is about 65.9%, so future Shape should be formulated as a normalized 24h residual direction rather than a single occurrence/sign bit.
3. **Amplitude is a necessary object, but the final scalar-vs-horizon-wise parameterization is not yet empirically frozen.** On direction-correct hours, median overshoot is 4.41% and strong under-correction is 83.29%. The original machine verdict `AMPLITUDE_SUPPORTED_SCALAR_PROJECTION` is invalid because its `scalar capture=2.9670` used the wrong denominator. Corrected scalar projection improves 129/143 public candidate cells but captures only median 32.64% of horizon-wise amplitude-recoverable error under the current simple-candidate directions. Because D2 simultaneously promotes Shape to a normalized horizon direction, a scalar global Amplitude remains the leading minimal mathematical hypothesis, but it must be retested conditionally on out-of-fold predicted Shape rather than inferred from the current direct-residual proxy.
4. **Proposal-level selective execution is necessary as a scientific object.** Median harmful candidate-day rate is 60.28%; positive oracle selective headroom occurs in 144/144 public cells; always-repair beats Host in only 53/144 cells, while oracle selective repair beats Host in 130/144. Even true realized Host-badness has median AUROC only 0.6263 for whether the concrete candidate helps, and the worst-20%-Host subset still has median harmful-repair rate 0.50. This supports `Host is bad != this proposal is beneficial` and distinguishes proposal utility from PIR-style Host uncertainty.
5. **Verification learnability is not yet established.** The original machine verdict used in-sample DIAG_FIT candidate utility labels, violating the intended OOF supervision contract. The next design must form complete out-of-fold Shape→Amplitude proposals before testing or training any learned Verification head.
6. **The market/regime repair-difficulty spectrum is not yet established.** The original runner did not implement the preregistered regime-conditioned D2–D4 attribution and hard-coded its spectrum flag. I5/I6 Introduction claims remain unresolved; Track A and China DA→RT Track B must stay task-qualified.

The surviving minimal research structure is therefore `Frozen Host -> normalized horizon Shape/direction -> Shape-conditioned Amplitude -> concrete repair proposal -> proposal-level Verification/abstention`. Historical Bridge/Global/router/CORN/quantile/market-expert machinery remains unnecessary without new evidence. The immediate next stage is **mathematical target derivation and architecture design discussion**, not another broad experiment or final HCH implementation.

Boundary governance correction: the diagnostic runner loaded complete raw target columns before assigning its new diagnostic S1/S2/S3/S4 labels. Under the project's strict read semantics, those newly created S3/S4 slices are no longer untouched protected truth and may not later be used or described as an unopened final test. This does not invalidate the DIAG_EVAL diagnostics but must constrain future evaluation design.

## Diagnostic-evidence branch refined into a counterfactual decision ladder (2026-09-08)

This branch is now explicitly responsible for two outputs before any new Methodology is coded: (1) falsifiable evidence deciding whether Shape, Amplitude, and Verification are scientifically necessary; and (2) machine-verifiable problem-characterization evidence that may later support the paper Introduction. The controlling design has been revised accordingly. A new executor roadmap is `experiments/current/hch_repair_diagnostics/DIAGNOSTIC_TODO_AND_DECISION_TREE.md`, with a short Codex launcher at `experiments/current/hch_repair_diagnostics/CODEX_GOAL_PROMPT_20260908.md`.

The diagnostic logic is no longer just rate-based. D2 adds evaluation-only `O-SIGN` and `O-DIR` counterfactuals to measure how much simple-residual loss is attributable to wrong sign/full direction; D3 measures recoverable loss from amplitude after conditioning on positive direction and tests scalar projection `a*=e^T u` against horizon-wise magnitude structure; D4 measures oracle selective-execution headroom and additionally tests whether even true Host-error severity can discriminate proposal benefit. The intended conceptual distinction is therefore empirically falsifiable: `Host is bad != this proposal is beneficial`. The stage must produce `INTRO_EVIDENCE_REGISTER.md` and `METHOD_DESIGN_HANDOFF.md`; neither is paper prose or final method code.

D0 is revised into `D0-A task/time contract -> D0-H preregistered Host train/freeze/cache -> D0-C leakage closure`. Missing pre-existing cache is no longer a blocker by itself. D0-H may train PatchTST/TimeMixer on S1 under the fixed maintained Host recipe, freeze them, and write diagnostic-local immutable caches; no result-driven Host tuning is allowed. D1-D4 use only S2 development (`DIAG_FIT=first 75%`, `DIAG_EVAL=last 25%`); S3/S4 remain closed. PAPER8 DA forecasting and CHINA5 prior CN-ERB `DA information -> RT price` are now separate task tracks, so cross-track difficulty statements must remain task-qualified.

Negative evidence has explicit simplification consequences: D2 failure removes Shape; D3 failure removes learned Amplitude; D4 headroom failure removes Verification; headroom without simple learnability forbids rescue by a larger gate. Historical Bridge/Global/D1/D2/CORN/quantile/retrieval machinery remains non-default unless later evidence specifically motivates it.

## D0 repair-diagnostic preflight blocked on Host foundation; blocker is protocol/infrastructure, not a scientific failure (2026-09-08)

The first authorized D0 run stopped correctly with `HCH_REPAIR_DIAGNOSTICS_BLOCKED_PREFLIGHT` before any scientific raw observation read. It found `0/26` maintained frozen Host cache cells for the registered `PAPER8 + CHINA5` × `PatchTST/TimeMixer` matrix, opened no protected/reserved partition, and produced no D1–D4 scientific verdict. The executor's fail-closed behavior is accepted.

Independent audit narrows the cause. The controlling diagnostic protocol over-constrained "frozen Host" into "pre-existing frozen Host cache". Existing foundation code `experiments/foundation/benchmark_foundation/runners/multiyear_host_cache_runner.py` already implements the scientifically legitimate lifecycle `train Host on a preregistered training partition -> freeze Host -> generate immutable HostPredictionCache`; therefore Host generation itself does not violate the frozen-Host post-processing identity. The next protocol revision may authorize a diagnostic-local Host-foundation phase, but it must not overwrite historical foundation caches/provenance or use outcome-driven Host tuning.

Two additional pre-method definitions must be closed before cache generation is authorized. First, the new diagnostic design did not freeze a common target semantics for the Chinese panel: historical CN-ERB uses `day-ahead information boundary -> next-24h real-time price`, while PAPER8 is day-ahead-price forecasting. Mixing these without explicit framing confounds market repair difficulty with task difficulty. Second, current market metadata leaves timezone/DST handling unspecified for all registered diagnostic markets; deterministic episode/timestamp semantics must be bound from existing canonical benchmark episode rules or a new explicit metadata contract before a reproducible 13-market cache build. Consequently the current bottleneck is **Host-foundation + target/time semantics**, not Shape/Amplitude/Verification and not method effectiveness. D1–D4 remain pending; `EXPERIMENT_LEDGER.md` remains unchanged because no scientific diagnostic was executed.

## Directional Shape mathematical object promoted for design discussion (2026-09-08)

Independent diagnostics plus theory now support a more precise Shape object. For frozen-Host residual `e = y - y_host`, define the exact Euclidean polar decomposition `e = r u*`, with `r = ||e||_2` and `u* = e / ||e||_2`. The paper-level Shape object is therefore the **signed normalized 24h residual profile**, not 24 independent sign labels. This choice is scale-free, preserves horizon-relative geometry, and matches the adjudicated D2 result that full O-DIR contains material headroom beyond O-SIGN.

For a unit predicted direction `u_hat`, let `κ = <u*,u_hat>`. A scalar candidate `Delta = a u_hat` has exact squared-error gain `G = 2 a r κ - a^2`; hence `κ <= 0` makes every positive step harmful, `0 < a < 2 r κ` is the beneficial interval, the MSE-optimal scalar amplitude is `a* = [e^T u_hat]_+`, and the maximum recoverable fraction of the perfect residual-oracle MSE gain is `[κ_+]^2`. This formally establishes Shape-before-Amplitude and makes cosine alignment a repair quantity rather than an auxiliary metric.

The leading **minimal Shape learner** is now one horizon-vector readout `v=f_S(z)` trained by unweighted squared regression to the unit target `u*`, followed at inference by `u_hat=v/(||v||+eps)`. In population, the Bayes output is `E[u*|z]`; its normalized vector is the conditional mean direction and its raw norm is a directional-concentration statistic. This avoids a separate sign classifier or confidence head. Do not treat the concentration norm as repair utility or a gate. Residual-magnitude weighting is not the default because it would re-introduce Amplitude information into Shape supervision.

The exact Shape design is recorded in `docs/current/HCH_DIRECTIONAL_SHAPE_MATHEMATICAL_DESIGN_20260908.md`. Future Amplitude supervision must use chronological out-of-fold predicted Shape, not oracle Shape; under MSE the leading target is `[e^T u_hat_OOF]_+`. No Shape implementation or new scientific execution is authorized yet.

## Theory-first repair redesign and D0–D4 diagnostic stage authorized (2026-09-08)

The project has now moved from literature-boundary/design discussion into one **diagnostic-evidence stage**, not into final-method implementation. Historical V3/V3.2 architecture is no longer an implementation compatibility target for the paper method: Bridge/Global/CORN/quantile/D1/D2 machinery remains historical scientific record and may not be restored merely because code already exists. The final Methodology must be derived from the repair mathematics and then implemented directly from the paper equations under the paper-first simplicity rule.

The active mathematical starting point is the additive repair identity. For frozen-Host residual `e = y - y_host` and candidate correction `Delta`, squared-error gain is `G_MSE = ||e||^2 - ||e-Delta||^2 = 2 e^T Delta - ||Delta||^2`. Writing `Delta = a u`, `||u||=1`, gives `G = 2 a (e^T u) - a^2`. Therefore: (1) non-positive alignment `e^T u <= 0` makes every positive step along `u` harmful; (2) conditional on a fixed positively aligned Shape direction, the optimal scalar amplitude is `a* = e^T u`; and (3) positive direction alone is insufficient because overshoot beyond `2a*` is harmful. This establishes the current **theory-first object order**: Shape/Alignment first, then Shape-conditioned Amplitude, then proposal-level Repair Utility/Verification. It does not yet freeze whether Amplitude should be one scalar per 24h Shape or horizon-wise; D3 is explicitly designed to decide that empirically.

The controlling diagnostic design is `docs/current/HCH_REPAIR_DIAGNOSTIC_EVIDENCE_DESIGN_20260908.md`. It registers D0 preflight/leakage audit plus four scientific diagnostics: D1 Host-error heterogeneity/tail concentration; D2 wrong-direction/alignment failures of simple correction; D3 amplitude overshoot and scalar-projection headroom conditional on positive alignment; D4 harmful-repair heterogeneity, oracle selective headroom, and minimal learnability of proposal benefit. Each object is falsifiable: an unsupported diagnostic requires simplifying/removing the corresponding method component rather than adding a rescue network. The stage is authorized only through these diagnostics. Final Shape/Amplitude/Verification HCH implementation remains forbidden until results return for scientific adjudication.

The diagnostic market hypothesis is a **repair-difficulty spectrum**, not a nationality split: established international electricity markets and rapidly changing Chinese provincial markets may span benign/moderate through hard/extreme repair regimes. The international PAPER8 scope is `GEFCOM14P / LAGO_BE / LAGO_DE / LAGO_FR / LAGO_NP / LAGO_PJM / NEM_SA1 / NORD_DK1`. The currently registered Chinese stress panel is `GANSU_RT / SHAANXI_RT / NINGXIA_RT / QINGHAI_RT / SHANDONG`, with Shandong always private supplementary (`public=false`, `private_external_case`) and never sole support for a primary reproducible claim. **Shanxi is not present in the current canonical data registry and must not be invented or silently substituted**; it may enter only after normal data admission if supplied later. The spectrum claim must be earned from D1–D4 raw diagnostics; if Chinese public markets are not empirically harder, the paper story must not force that ordering.

Core first-pass Hosts are `PatchTST` and `TimeMixer`; this is a mechanism diagnosis, not yet the final backbone table. The smallest candidate correctors are a matched linear direct-residual probe, a fixed tiny MLP probe, and the existing read-only `δ-Adapter Ada-Y` and `PIR` wrappers. Protected/consumed historical panels and the newly reserved final diagnostic slice remain closed. Implementation may occur only under `experiments/current/hch_repair_diagnostics/**`, with evidence under `experiments/evidence/hch_repair_diagnostics_20260908/**`; `src/**`, `data/**`, historical evidence, and `paper/**` are not to be modified. The short Luna-High launcher is `experiments/current/hch_repair_diagnostics/DIAGNOSTIC_EXECUTOR_PROMPT_20260908.md`.

Related-work interpretation feeding this stage is now stable: PIR supports instance-level forecast-error heterogeneity; COSA establishes that simple frozen-Host output residual correction can be strong; δ-Adapter provides the alignment-based small-correction theory but does not explicitly guarantee current-instance direction/magnitude or abstention; OMPB demonstrates reliability-controlled gated residual adaptation and Host fallback under shift. Consequently frozen Host, plug-in residual correction, generic gating, or fallback are not sufficient novelty. Any final HCH value must come from a minimal, empirically necessary alignment–amplitude–utility formulation and its behavior across heterogeneous electricity-market regimes.

## Research operating principle — paper-first simplicity and reproducibility (2026-09-07)

For all subsequent HCH design decisions, optimize for an A-tier paper method rather than a high-complexity software system. The scientific contribution must come first; implementation should be a direct, readable realization of the paper equations/architecture that another group can reproduce without repository-specific orchestration knowledge. Treat deep protocol/audit machinery as experiment-control infrastructure, not as part of the proposed method. Avoid adding modules, routers, local experts, hidden market exceptions or engineering abstractions unless a clear scientific result requires them. Prefer the smallest architecture that explains the observed failure and wins the registered metrics/baselines. The local executor is upgraded to `solar-medium` for stronger implementation fidelity, but executor sophistication must not justify method complexity.

The user-facing architecture discussion should therefore distinguish explicitly between: (1) paper-level scientific objects and claimed novelty; (2) mature supporting operators such as GRU/CORN/quantile parameterization; and (3) audit/provenance code that should disappear from the conceptual method. Reproducibility target: the final paper description plus ordinary implementation details should be sufficient for a competent AI/researcher to reconstruct the method; the result must not depend on opaque repo-specific engineering.

Active simplification hypothesis (discussion only): use one unified legal Host-error time-series input and one shared temporal encoder; do not pre-assign handcrafted Shape-only vs Amplitude-only feature inventories. Let task specialization be learned by simple branch heads/losses. To avoid duplicating "repair occurrence" with the final selector, the preferred conceptual role split is now: Shape proposes repair direction/sign, Amplitude proposes conditional magnitude, and a final benefit Gate decides whether the combined candidate repair should be executed relative to keeping the frozen Host. The Gate should be trained against the ultimate repair utility/final forecast loss rather than act as another occurrence classifier. No experiment has yet validated this redesign.

The latest local executor reports `HCH_V3_2_AB_REPORT_DIAGNOSTICS_REPAIRED_AWAITING_FINAL_NO_EDIT_CLOSURE` with 45/45 tests and manifest `b5455afaa37364c0617711b201869b23c04d70c70ce7dde6792027d813e53868`; this remains a pre-execution implementation status pending independent final no-edit closure and is not a new scientific result.

### Paper-first unified signed-repair design candidate registered for external review (2026-09-07)

A new design candidate is now documented at `docs/current/HCH_UNIFIED_SIGNED_REPAIR_METHOD_DESIGN_20260907.md`. It is **not an accepted scientific result and authorizes no experiment**. The candidate intentionally simplifies historical V3 to one complete legal Host-error tensor -> one shared temporal encoder -> learned Shape(direction) and Amplitude(sign-conditional magnitude) specialization -> differentiable candidate residual `c = p*mu_plus - (1-p)*mu_minus` -> a one-layer proposal-only Benefit Gate -> final normalized overall + fixed tail-emphasis task loss. No Bridge, CORN, quantile grid, market/Host router, target-local expert, or branch-specific handcrafted feature inventory is part of the default design. A paired adversarial review prompt is frozen at `docs/current/HCH_ICDE_STRICT_REVIEW_PROMPT_20260907.md`; the next action for this route is strict external design review, not implementation or experiment.

The design explicitly treats ICDE venue fit as a separate risk: ICDE 2027 includes time-series/temporal data and data-centric mining in scope but also warns that pure ML/DM work without a substantive data-engineering relation can be desk-rejected. Recent ICDE precedent nevertheless includes direct time-series methodology (e.g. EnhanceNet 2021, TimeDRL/DeSTR/TFMAE 2024, FOCUS/Auto-TSF 2025), so venue fit should be judged from the actual framing and contribution rather than assumed impossible. A strong forecasting result alone is not automatically sufficient, but a principled general time-series/data contribution can be in-scope.

External-review discipline is now tightened: reviewer-facing author material must contain only the current method and exact reproducibility specification, with no historical alternatives, internal risk lists, or pre-seeded criticisms. `docs/current/HCH_ICDE_STRICT_REVIEW_PROMPT_20260907.md` has been simplified to a neutral reviewer process focused on problem/method novelty, evidence-claim matching, writing/boundaries, scientific insight/completeness/workload, and a final contribution-vs-defect ICDE score. The method document will be regenerated into a pure reproduction specification after the current Shape/Amplitude/Gate objective discussion is resolved, rather than freezing an unsettled loss/Gate design prematurely.

Current literature pressure on the redesign: ICLR 2026 `delta-Adapter` already establishes frozen-forecaster lightweight residual post-processing with bounded edits/stability across backbones, so frozen Host + small residual adapter is not sufficient novelty. A very recent selective residual-correction paper (MURECAST, 2026) also makes correction utility and activation explicit. The active design question is therefore whether HCH can define a simpler and more distinctive signed direction/magnitude proposal plus non-redundant utility-based Host-preservation decision, rather than a generic shrinkage Gate.

### Strict external design review received (2026-09-07)

The independent review is now synthesized at `docs/current/HCH_UNIFIED_SIGNED_REPAIR_EXTERNAL_REVIEW_ADJUDICATION_20260907.md`. Reviewer verdict: `METHOD_PROMISING_BUT_NEEDS_SMALL_REVISION`, method score 6.2/10; ICDE verdict `ICDE_SCOPE_HIGH_DESK_REJECT_RISK`, hypothetical Weak Reject. The review supports the simplification direction and explicitly advises against restoring Bridge/CORN/quantile/router/expert complexity merely to manufacture novelty. Before implementation, the design must correct the Tail-loss row-weight semantics, narrow the conditional-mean claim under end-to-end MAE training, reinterpret the Gate as Host-preserving shrinkage rather than an identified benefit probability, freeze precise market-agnostic claim vocabulary, and complete deterministic preprocessing/timezone/DST/reproducibility semantics. This is a design-review conclusion, not a scientific experiment; no implementation or new data access is authorized.

## HCH V3.2 A+B no-edit closure blocked only on report-diagnostic semantics; no real execution authorized (2026-09-07)

### Active paper-design discussion constraint — structural cross-market generalization, not data-heavy parameter transfer (2026-09-07)

The user has clarified the intended paper claim and design preference. The desired contribution is a **simple, reproducible market-agnostic repair framework** that uses market/deployment data semantics and scale-relative structure so that the same method works across selected markets and frozen Hosts; cross-market validity should be demonstrated primarily by consistent empirical gains across markets/backbones, not by requiring a large multi-market pretraining corpus or extensive source-to-target parameter freezing/fine-tuning. This is a design preference/hypothesis, not yet an accepted scientific result. Future architecture discussion should therefore prioritize the information content and invariance/equivariance of the two repair branches over encoder sophistication, treat MLP/TCN/GRU choice as secondary implementation detail unless experiments show otherwise, and actively simplify hand-crafted feature inventories/Bridge machinery when they do not support a clear paper-level mechanism. The current 7-D Shape view, 6-D Amplitude view, gated Bridge, and V3.2 target-local adapters remain historical/current candidate implementations rather than immutable paper innovations.

Active redesign hypothesis (discussion only): replace branch-specific handcrafted feature inventories with one complete legal base time-series tensor `X` (Host prediction, aligned revealed actual/residual history, universal time/context variables, with only scientifically necessary causal normalization). Learn `z0 = Encoder(X)`, then task specialization internally as `z_s=f_s(z0)` and `z_a=f_a(z0)`. Both branches receive the same information by default; specialization should arise primarily from their targets/losses. A shared encoder is already information sharing, so no Bridge/cross-branch router is assumed unless controlled evidence later demonstrates need. The leading simple probabilistic interpretation is a signed two-part/hurdle factorization: Shape models repair activity and conditional sign, while Amplitude models magnitude conditional on active direction. This is a literature-motivated design hypothesis, not yet a promoted HCH method.


The last-mile repair submission independently replays at `42/42 PASS` (16/16 guard/lifecycle/preflight + 26/26 remaining tests) and compileall passes. Active manifest SHA is `5666ed8d233589e938340e542b649959d8ccf617f6bc0637dd7e76a04a53df8b`, saved-manifest verification is True, and `scientific_execution_authorized=false`. The previously blocking execution/data-access semantics are now closed: exact-subset raw reconstruction no longer calls SSI whole-role `_episodes()`, V0 stops before confirmation, unique scientific windows are cached across seeds, selected confirmation trains only B0-M + the frozen selected candidate, factual role access provenance is present, and main development/confirmation joint-vs-conditional direction plus seed-pooled gate semantics are implemented. No real A+B scientific truth has been read.

Final no-edit closure still withholds execution authorization because four report-only semantics remain non-exact relative to the frozen design: Experiment-A conditional direction still receives three-class argmax rather than the Down-vs-Up conditional readout; Experiment-A future-eval ER-AP-Lift is implemented as `AP - prevalence` instead of the canonical `AP / prevalence`; the 12-row Experiment-A seed-median summary omits several registered nested diagnostics (Down/Up composition, class-conditional centroids, joint/conditional direction); and development/confirmation evidence does not persist explicit conditional Up/Down recall/support separately from joint recall/support. These do not alter B1/B2/B3 gate formulas, but they can alter Experiment-A mechanism interpretation and cannot be safely repaired post hoc from current one-shot evidence.

Controlling closure audit: `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_NO_EDIT_CLOSURE_20260907.md`. The only next action is a final narrow report-diagnostic implementation repair using `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_REPORT_DIAGNOSTIC_FINAL_REPAIR_AI_PROMPT_20260907.md`, with all real scientific reads remaining zero. After that, perform one final independent no-edit closure. Only a passing closure may advance to `HCH_V3_2_AB_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`.

The controlling design remains byte-identical at SHA `21dada9919e9d1c238506a8ad3cca9bccbd94592be246ec2a8063178aba6f603`; `EXPERIMENT_LEDGER.md` remains unchanged at SHA `6ea80a4b9d1ea3ab04a12aec20b6cb1970b1f4cc0b2702fb8ab42a62b55078f4`; SSI manifest/orchestration remain `f242b23f...44acc0` / `148a7167...98be`. TARGET_D1/D2, S2V/S3/S4/fresh and final Gate remain closed.

## HCH V3.2 A+B final re-audit blocked only on last-mile execution semantics; no real execution authorized (superseded by 2026-09-07 no-edit closure above, 2026-09-06)

The orchestration-repair submission is substantially improved and independently replays at `35/35 PASS`; compileall passes; manifest SHA is `10e2fe0a42a4d80bd7d8ab98aece6062c7e2dedfa6881988a3f77339f08aa9a1`; saved-manifest verification is True from the canonical absolute repository root. Common synthetic/future-real orchestration is now shared, development metrics come from model predictions rather than fabricated rows, six-key confirmation locking exists, canonical metadata binding and SSI-compatible positive-parameter semantics remain correct, and no real A+B scientific truth has been read. SSI frozen hashes and `EXPERIMENT_LEDGER.md` remain unchanged.

Independent final re-audit still blocks execution because several last-mile semantics are not yet exact: the real adapter authorizes 56d/28d A+B subsets but internally calls whole-role SSI `_episodes()` and filters afterward; V0 still opens confirmation; Experiment-A future transfer diagnostics use support rather than future eval and pool target Hosts; conditional direction still reuses three-class argmax; pooled direction concatenates seeds instead of per-seed pool -> seed median; scientific windows are reread per seed; real access provenance hard-codes/omits allowed-role counts; evidence lacks all baseline/candidate median rows and actual before/after representation hashes; and confirmation trains unselected variants. Controlling audit: `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_FINAL_REAUDIT_20260906.md`.

Solar has now registered a last-mile implementation-only closure at `docs/current/HCH_V3_2_AB_LAST_MILE_EXECUTION_CLOSURE_PROTOCOL_20260906.md` with paired Luna executor prompt `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_LAST_MILE_EXECUTOR_AI_PROMPT_20260906.md`. The controlling A+B design contains a 2026-09-06 pre-execution clarification for support-vs-eval Experiment-A semantics, conditional direction, seed-faithful pooled direction, and V0 no-confirmation behavior; no candidate/threshold/optimizer/fold changes were made. Repo-local execution skill `.agents/skills/solar-research-executor/SKILL.md` is now the standard commander/executor contract. Real SOURCE_GTRAIN/SOURCE_GVAL and TARGET_D1/D2/S2V/S3/S4/fresh remain unopened; no execution authorization is allowed yet; `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.2 A+B implementation audit blocked; protocol-completion repair required; no real execution authorized (superseded by 2026-09-06 re-audit above)

A new design-only research program is now registered at `docs/current/HCH_V3_2_TARGET_LOCAL_SHAPE_RESEARCH_PROGRAM_20260905.md`. It does not change the accepted SSI Case-B verdict. The refined architectural hypothesis is that the frozen cross-market representation and ER Shape semantics contain usable signal, but the current target adaptation is under-parameterized: three D1 Shape scalars cannot express a market-specific decision-boundary correction. The next architecture therefore inserts a tiny target-market Shape alignment layer between the shared ER Shape trunk and posterior logits, while keeping upstream representation, Amplitude and Gate frozen.

The program contains a staged set of experiments rather than one monolithic redesign: A target-shift decomposition; B a minimal target-local capacity ladder (current 3-scalar baseline vs 4-scalar diagonal affine vs 66-parameter market-shared last-layer residual vs full local-head diagnostic ceiling); C market-vs-Host sharing-axis identification; D conditional direction adaptation with activity ranking frozen; E ER-conditional Amplitude alignment only after Shape transfer is supported. The immediate recommended executable stage is A+B only; C/D/E remain conditional.

Because SSI TARGET_D2 has already been observed and directly informed this design, TARGET_D2 is now consumed for V3.2 method selection and must not be reused as an unbiased promotion gate. V3.2 development selection should use source-side pseudo-target simulations only: for each market/Host, two fixed chronological 56-day support / 28-day evaluation folds carved from that market's SOURCE_GTRAIN, giving 12 pseudo-target development cells. After a candidate is frozen, SOURCE_GVAL may be used once for six-cell development confirmation. TARGET_D1 may only fit the frozen candidate later; S2V/S3/S4/fresh remain closed until a separately registered protected-stage protocol.

The observed SSI market coherence motivates market-local/host-shared adaptation first: both DE Hosts improve strongly, both NORD Hosts regress modestly, both PJM Hosts regress materially. Do not jump to a Local expert/router.

A+B is now frozen precisely at `docs/current/HCH_V3_2_AB_SHIFT_AND_CAPACITY_EXPERIMENT_DESIGN_20260905.md`, with paired implementation/preflight prompt `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_IMPLEMENTATION_PREFLIGHT_AI_PROMPT_20260905.md`. Exact pseudo-target date ranges and hashes are registered. A fairness correction is canonical: `B0-H` remains the exact SSI Host-specific 3-scalar historical reference, while `B0-M` is the matched market-shared 3-scalar baseline used for capacity attribution. The true ladder is `B0-M -> B1 four-scalar affine -> B2 calibrated 67-parameter latent residual -> B3 2,277-parameter full local CORN diagnostic upper bound`; the smallest passing capacity is frozen, and B3 cannot auto-promote.

The first V3.2 A+B implementation/preflight submission reported 23/23 tests PASS, compileall PASS, manifest SHA `c1cce2605914fa9174fa8cc20d488e21c1e41d8c43844c9401992a2756a95f38`, `verify_manifest=True`, and zero real scientific reads. Independent replay confirms the 23 tests genuinely pass, but the implementation is not protocol-complete. Controlling audit verdict: `HCH_V3_2_AB_IMPLEMENTATION_AUDIT_BLOCKED_PROTOCOL_MISMATCH` at `experiments/current/hch_v3_2_target_local_shape_ab/INDEPENDENT_IMPLEMENTATION_AUDIT_20260905.md`.

Blocking implementation mismatches include: future real lifecycle is not implemented (`run(real=True)` intentionally aborts); B0-H/B0-M temperature semantics use `exp` rather than exact SSI softplus positive-temperature parameterization and B1 uses `exp` rather than registered softplus scales; split/hash checks are self-consistent hard-coded reconstruction rather than independent canonical non-truth metadata binding; development adjudication omits/implements incorrectly registered median/NormalDiversion/pooled-direction conditions; SOURCE_GVAL lock is not integrated into fail-before-reader guard; causal-history guard admits D1/D2 phases; production trainers/metrics/Experiment-A diagnostics/confirmation gate and final verdict mapping are incomplete or placeholder-level. Therefore no execution authorization is allowed yet.

Next action is implementation-only repair under `experiments/current/hch_v3_2_target_local_shape_ab/V3_2_AB_PROTOCOL_REPAIR_AI_PROMPT_20260905.md`, then independent re-audit. TARGET_D1/TARGET_D2/S2V/S3/S4/fresh remain forbidden; real SOURCE_GTRAIN/SOURCE_GVAL reads remain 0. `EXPERIMENT_LEDGER.md` remains unchanged because no new scientific experiment occurred.

## HCH V3.1 SSI completed: source signal present, target transfer failed (2026-09-05)

The single authorized real execution of `HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_V1` completed exactly once on the frozen protocol-conformant checkout and has now been independently adjudicated. Final scientific verdict: `HCH_V3_1_SSI_SOURCE_SIGNAL_PRESENT_TARGET_TRANSFER_FAILED` (Case B). G0 passes; G1 fails; G2 is not eligible for promotion adjudication.

Independent recomputation confirms B1 beats A1 on source G-Val in all 3 target-excluded settings, with seed-median ER-AUPRC ratios approximately `1.199 / 1.822 / 1.215`. On target D2, however, B1 beats A1 in only 2/6 cells. The six seed-median B1/A1 ratios are `1.533, 1.492, 0.962, 0.926, 0.729, 0.779` for DE/PT, DE/TM, NORD/PT, NORD/TM, PJM/PT, PJM/TM; six-cell median ratio is `0.944307`, below the registered `1.20` requirement. NormalDiversion improves in 4/6 cells and its six-cell median falls from `1.0000` to `0.937291`, but this is insufficient to rescue G1.

Scientific interpretation: the frozen V3 representation does contain learnable extreme-repair information, so the stronger pessimistic representation-null hypothesis is rejected. But changing Shape supervision to extreme-repair semantics plus the current three D1 Shape calibration scalars is not sufficient for robust cross-market target transfer. Transfer is strongly market-dependent: both DE cells improve substantially, both NORD cells regress modestly, and both PJM cells regress materially. Extreme-repair semantics remain positive source-level evidence but are not promoted as a transferable Shape solution yet.

Report-only pooled target direction diagnostics are also severe: B1 and B2 both have pooled six-cell active-direction macro-F1 `0` for all three seeds (469 active rows per seed; 358 Down / 111 Up), consistent with strong target-side Identity dominance after current calibration. This does not alter the preregistered verdict because G2 was not eligible, but it reinforces the new bottleneck.

The active research bottleneck therefore moves to **minimal target-local Shape adaptation / sharing**, not Amplitude, Gate, class weighting, threshold rescue, or fresh-panel validation. Preserve the frozen upstream representation and extreme-repair target semantics as the candidate learning object while designing the next single-core-object target-local experiment. S2V/S3/S4/fresh remain closed. Independent adjudication: `experiments/evidence/hch_v3_1_shape_semantic_identification/INDEPENDENT_SSI_SCIENTIFIC_ADJUDICATION_20260905.md`.

## HCH V3.1 SSI protocol-conformant and awaiting fresh execution authorization (superseded by authorized-run state above, 2026-09-05)

Independent final closure accepts `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`. Exact checkout/import identity is now proven: repo root `D:\作业\science\solar_leak_price_model`, Git HEAD `9111d0f567ee5b6cd5a8b9f378623aa8586fa34e`, frozen manifest SHA-256 `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`, and `orchestration.py` SHA-256 `148a71671a5d0b674a2212ac4febae255e78a293f95331774551262cb2eb98be`. The full SSI suite independently replays at `46/46 PASS`, compileall passes, and saved-manifest self-verification passes.

The complete historical HCH Python representation package independently matches the accepted frozen bundle `23/23`; canonical split/threshold/global/H0 binding passes; the package-owned real SSI adapter constructs pretruth with `audit_attempts=0` and no verified role reads; negative probes for fresh/S2V/S3/S4/wrong-phase/target-inclusion/date-hash/causal-history all fail-before-reader. Source epochs remain exactly 20, D1 attempts 64, seeds `[7,17,37]`, source G-Val calibration updates 0, eligibility-first metrics/G2 pooled-direction/G0 fail-closed contracts all pass, and final Gate execution remains 0.

No real SSI scientific SOURCE_GTRAIN/GVAL/D1/D2 execution has occurred. Real scientific truth reads remain 0; S2V/S3/S4/fresh successful access remains `0/0/0/0`; `EXPERIMENT_LEDGER.md` remains unchanged because this is protocol readiness, not a scientific experiment result. The controlling closure is `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PROTOCOL_CONFORMANCE_CLOSURE_20260905.md`. The only next scientific action is to wait for a **fresh post-closure user literal** `HCH_V3_1_SSI_EXECUTION_AUTHORIZED`; old occurrences in history/tests must not be consumed.

## HCH V3.1 SSI workspace identity mismatch discovered; verify checkout/import origin before closure (superseded by protocol-conformance closure above, 2026-09-05)

The second independent no-edit verifier again returned `HCH_V3_1_SSI_FINAL_CLOSURE_NOT_READY`, but its result does **not** represent a second failure of the repaired SSI implementation. It reported the old manifest SHA `334f311bd81c17f444d2bc490a19131cec67b8ffc1ef41b5d82b48446e3552c5` and the old synthetic G0 behavior, while the active registered checkout at `D:\作业\science\solar_leak_price_model` currently contains the repaired G0 predicate `final_gate_not_executed=True`, `orchestration.py` SHA `148a71671a5d0b674a2212ac4febae255e78a293f95331774551262cb2eb98be`, and manifest SHA `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`. `git worktree list` shows this checkout as the single registered worktree, at HEAD `9111d0f567ee5b6cd5a8b9f378623aa8586fa34e`.

The active bottleneck is therefore execution-environment identity: the verifier likely ran against another checkout/copy or imported a stale duplicate package. This must be resolved before any further closure conclusion is accepted. A machine-readable fingerprint is now frozen at `experiments/current/hch_v3_1_shape_semantic_identification/WORKSPACE_FINGERPRINT_20260905.json`; the controlling protocol is `docs/current/HCH_V3_1_SSI_WORKSPACE_IDENTITY_AND_FINAL_RECHECK_PROTOCOL_20260905.md`; the local-AI prompt is `experiments/current/hch_v3_1_shape_semantic_identification/SSI_WORKSPACE_IDENTITY_FINAL_RECHECK_AI_PROMPT_20260905.md`.

The next verifier must first prove exact checkout identity and Python import origin before pytest: repo root `D:\作业\science\solar_leak_price_model`, Git HEAD `9111d0f...`, manifest SHA `f242b23f...`, orchestration SHA `148a7167...`, and the source marker `"final_gate_not_executed":True`. Any mismatch stops before tests with `HCH_V3_1_SSI_WORKSPACE_IDENTITY_MISMATCH` or `HCH_V3_1_SSI_PYTHON_IMPORT_ORIGIN_MISMATCH`. Only after identity passes may the 46-test no-edit closure be rerun. No real SSI scientific execution has occurred; real truth reads remain 0; S2V/S3/S4/fresh remain 0/0/0/0; `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.1 SSI first no-edit closure failed 45/46; single G0 polarity bug repaired; recheck pending (superseded by workspace-identity state above, 2026-09-05)

The first independent no-edit closure correctly returned `HCH_V3_1_SSI_FINAL_CLOSURE_NOT_READY`: 46 tests were collected, 45 passed, and only `test_authorized_safe_synthetic_path_completes_without_gate` failed because synthetic `G0_pass=False`. No real SSI science ran; real scientific truth reads were 0; S2V/S3/S4/fresh access was 0/0/0/0; the verifier modified no `.py` or manifest files; `EXPERIMENT_LEDGER.md` remained unchanged.

Independent diagnosis localized the failure to one success-predicate polarity bug, not to SSI science or the real adapter: G0 stored the correct factual audit field `final_gate_action_executed=False` inside a list aggregated by `all(bool(v))`, so a correct no-Gate execution necessarily failed G0. The only runtime correction changes the G0 success predicate to `final_gate_not_executed=True`; final evidence still records the factual field `final_gate_action_executed=False`. No architecture, data split, target, optimizer, seed, threshold, G1/G2 rule, Amplitude, Gate, or real-adapter semantics changed.

Current repaired manifest SHA-256 is `f242b23f528ca351e7329bfdd1a946844d966def0b3630bc524fede96544acc0`; `verify_manifest=True`; test inventory remains exactly 46; 45/45 tests pass when the formerly failing runner test is deselected; compileall passes; historical representation package remains `23/23 MATCH`. The formerly failing synthetic runner test must now be rerun independently under the no-edit recheck protocol `docs/current/HCH_V3_1_SSI_FINAL_CLOSURE_RECHECK_PROTOCOL_20260905.md` and prompt `experiments/current/hch_v3_1_shape_semantic_identification/SSI_FINAL_CLOSURE_RECHECK_AI_PROMPT_20260905.md`. Only a full 46/46 no-edit pass may return `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`.

## HCH V3.1 SSI protocol-conformance patch frozen; final no-edit verification pending (superseded by recheck state above, 2026-09-05)

The previously identified SSI execution/protocol mismatches have now been corrected in the experiment-local implementation without running real SSI science. The frozen common path now enforces held-out source G-Val with `calibration_updates=0` and common ER truth, eligibility-first D1/D2 metrics and budget, six-cell seed-median G2 activity/loss adjudication, true pooled six-cell direction diagnostics, fail-closed computed G0, a package-owned canonical real-data adapter, canonical split/threshold binding, strictly-past causal-history guarding, and secondary non-promotion diagnostics. The actual historical HCH Python package used for representation reconstruction is now checked against the accepted frozen source bundle and currently matches `23/23` files.

Current saved SSI manifest SHA-256 is `334f311bd81c17f444d2bc490a19131cec67b8ffc1ef41b5d82b48446e3552c5`, and `verify_manifest(root, saved_manifest)` independently returns `True`. Current test inventory is exactly `46` tests. After the final corrections, 43 non-manifest/non-runner tests pass, the manifest self-verification test passes, and compileall passes; the final complete no-edit runner/integration closure is intentionally delegated to an independent local-AI verification pass rather than letting the auditor continue changing implementation while testing it.

The controlling next step is **verification only, with no `.py` or manifest edits permitted** under `docs/current/HCH_V3_1_SSI_FINAL_EXECUTION_CLOSURE_PROTOCOL.md` and `experiments/current/hch_v3_1_shape_semantic_identification/SSI_FINAL_EXECUTION_CLOSURE_VERIFICATION_AI_PROMPT_20260905.md`. Any failure must stop at `HCH_V3_1_SSI_FINAL_CLOSURE_NOT_READY`; only a full no-edit V0-V9 pass may return `HCH_V3_1_SSI_PROTOCOL_CONFORMANT_AWAITING_EXECUTION_AUTHORIZATION`. That token is readiness only, not authorization. No real SOURCE_GTRAIN/GVAL/D1/D2 scientific execution has occurred; S2V/S3/S4/fresh remain closed; Amplitude/Gate remain outside SSI promotion; `EXPERIMENT_LEDGER.md` remains unchanged.

The SSI design's S1 boundary is clarified, not broadened scientifically: S1 supervision/metric/threshold semantics remain threshold-artifact-only, while strictly-past S1 residual truth may be used only as unavoidable 168h causal context for exact frozen V3 feature reconstruction on otherwise legal rows, guarded before raw-byte access and forbidden from SSI labels, metrics, calibration targets, selection, or threshold fitting.

## HCH V3.1 SSI final execution re-audit blocked by protocol-semantics mismatch (2026-09-05)

Round-2 materially closes the previous engineering blockers, but independent final execution re-audit still withholds `HCH_V3_1_SSI_EXECUTION_AUTHORIZED`. The controlling verdict is `HCH_V3_1_SSI_FINAL_EXECUTION_REAUDIT_BLOCKED_PROTOCOL_MISMATCH`. Independently verified positives: contextual guard/gate subset `16/16` PASS; target/head/metrics/training/sampler subset `11/11` PASS; frozen/H0/manifest subset `5/5` PASS; compileall PASS; manifest SHA-256 `dc0584d1aefe7f5c6fc5f1e38ed4459309c4d7102ffea5fda8fbeab50fa268e9`; all seven protected historical V3/V3.1 files remain byte-identical to the accepted frozen source bundle. No real SSI scientific execution occurred.

The remaining blockers are scientific-protocol implementation mismatches in the exact common orchestration: SOURCE_GVAL labels are incorrectly used for an extra 64-step Shape calibration instead of held-out evaluation; ineligible rows can enter source/D1/D2 AP, prevalence, K, budget and direction diagnostics; G2 mixes seed rows with the registered `2/6 cells` rule and does not compute true pooled six-cell direction metrics/support eligibility; G0 is hard-coded true instead of fail-closed; and the real scientific input path still accepts arbitrary injected loaders/hash strings rather than one manifest-sealed concrete real-data factory with actual representation/H0 verification. Therefore executing the current manifest could produce a formally complete but scientifically incorrect G1/G2 verdict.

SSI design remains accepted and unchanged. No algorithm/hyperparameter changes are authorized. Next action is protocol-faithfulness repair only under `experiments/current/hch_v3_1_shape_semantic_identification/SSI_PROTOCOL_SEMANTICS_FIX_AI_PROMPT_20260905.md`, then one final execution audit. Controlling audit: `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_FINAL_EXECUTION_REAUDIT_20260905.md`. S2V/S3/S4/fresh panel remain closed; Amplitude/Gate remain frozen/outside promotion; `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3.1 SSI pre-execution re-audit still blocked (2026-09-05)

Independent re-audit of the repaired SSI implementation returns `HCH_V3_1_SSI_REAUDIT_NOT_READY_FOR_EXECUTION`. Important prior blockers are genuinely fixed: the SSI suite independently reproduces at `20 passed`; `compileall` passes; manifest SHA-256 is `a1c3cc060f7ceae5355a0923ecee1cf6e25ccfd321c12bb31137d7c56a7ed848`; A1/B1/B2 source training now exists with the frozen 20-epoch/8+8+8+8/three-seed schedule; B2 D1 calibration now uses repair-activity + conditional-direction semantics; `[B] -> [B,H]` eligibility broadcasting is fixed; and all seven protected V3/V3.1 files still match the accepted frozen source bundle byte-for-byte.

Two G0 blockers remain. First, `PartitionGuard.authorize_context()` does not enforce the base market/role allowlist or exact role-to-phase mapping; direct probes show that contextual access currently allows fresh `EPEX_FR` as `SOURCE_GTRAIN`, forbidden `S2V` under phase `D2`, and `SOURCE_GTRAIN` under phase `D1`. Second, the exact future scientific SSI runner still does not exist behind the authorization gate: `run(..., synthetic=False)` is hard-sealed, while the synthetic path omits real guarded data integration, post-Bridge latent orchestration, H0 lifecycle, D1-derived budget diagnostics, direction metrics, seed aggregation, G0/G1/G2 adjudication, verdict mapping, and evidence/provenance materialization. Authorizing now would therefore require a post-audit code change and invalidate the current manifest.

SSI scientific design remains unchanged and active; no scientific SSI source/D1/D2 execution occurred. S2V/S3/S4/fresh panel remain closed; Amplitude/Gate remain frozen/outside promotion. Required next step is one final implementation closure using `experiments/current/hch_v3_1_shape_semantic_identification/SSI_PRE_EXECUTION_REPAIR_ROUND2_AI_PROMPT_20260905.md`, followed by another independent re-audit. Controlling re-audit: `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PRE_EXECUTION_REAUDIT_20260905.md`.

## HCH V3.1 SSI pre-execution audit blocked (2026-09-05)

Independent review of the first SSI implementation **withholds** scientific execution authorization: `HCH_V3_1_SSI_EXECUTION_AUTHORIZATION_WITHHELD_IMPLEMENTATION_INCOMPLETE`. The reported 12 synthetic tests and manifest SHA are reproducible, and all seven protected historical V3/V3.1 files match the accepted frozen source bundle byte-for-byte. However the current `runner.run()` still raises `NotImplementedError` even with the exact authorization literal; no source-head scientific training loop exists; B2 D1 calibration incorrectly reuses CORN target semantics; real frozen checkpoint/H0 replay and real reader integration are absent; the partition guard is not bound to a real execution path; `[B]` repair-availability broadcasting fails for real `[B,H]` targets; the protocol manifest does not bind the actual SSI code/config/checkpoints/splits/threshold artifacts; and mandatory integration tests are incomplete. No SSI source/D1/D2 scientific execution occurred and no protected data were opened in this audit.

The SSI scientific design itself remains accepted: Shape semantic identification is still the first learned-component object, Amplitude remains frozen, `SIGNED_MAE_BAYES_GRID_V1` remains accepted, Gate/selector rescue stays stopped, consumed S2V/S3/S4 stay closed, and the fresh panel remains untouched. Required next step is implementation repair followed by an independent re-audit **before** the literal may be consumed. Controlling audit: `experiments/current/hch_v3_1_shape_semantic_identification/INDEPENDENT_PRE_EXECUTION_AUDIT_20260905.md`.

## HCH V3.1 learned-component redesign discussion update (2026-09-05)

The user reaffirmed that the primary paper task is **extreme-price repair**, not generic residual correction. The high-level V3.1 architecture remains scientifically attractive and should be preserved unless a minimal mechanism test proves a deeper failure. The first redesign priority is therefore the **Shape learning object**: current demo supervision labels generic normalized Host residual states and used `extreme_importance_mode="none"`, which is misaligned with the observed Normal-heavy action ranking. Amplitude architecture/calibration is not the first object to change and should remain frozen in the first Shape mechanism test because low-quantile V3.1 actions already contain usable severity information.

The preferred discussion direction is an **extreme-conditioned repair state**: an observation is non-Identity only when it lies in the legally materialized market-relative truth Tail (existing S1 Q05/Q95 semantics) and the signed Host residual is materially nonzero under the existing causal directional scale threshold. A future fully coherent Shape redesign may replace ordinal `Down < Identity < Up` CORN with a two-logit symmetric repair factorization `Repair? -> Direction?`, preserving three probabilities and the same tiny target-calibration budget. This is not yet authorized for implementation. Because the current Amplitude loss masks are derived directly from Shape state, the first validation must isolate Shape semantics rather than silently retarget Amplitude at the same time.

A concrete Shape-only mechanism study is now **registered at the design level** as `HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_V1` (SSI). The experiment freezes the historical post-Bridge V3 Shape representation, compares a fair generic-CORN re-head against an extreme-repair CORN re-head, and then compares extreme-repair CORN against an equal-capacity two-logit symmetric `Repair -> Direction` posterior. Source/G-Val identifies whether the frozen representation contains extreme-repair information; target D1->D2 identifies whether the current three Shape calibration scalars preserve that ranking cross-market. Amplitude/Gate are excluded from promotion, and no final MAE claim is produced. Design: `docs/current/HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_EXPERIMENT_DESIGN_20260905.md`; implementation prompt: `docs/current/HCH_V3_1_SHAPE_SEMANTIC_IDENTIFICATION_AI_PROMPT_20260905.md`.

SSI is **design-registered only**. No implementation or scientific execution is authorized yet. Scientific execution requires the literal `HCH_V3_1_SSI_EXECUTION_AUTHORIZED`. The fresh panel and all consumed/protected evaluation data remain closed. Gate/selector rescue remains stopped.

## HCH V3/V3.1 stage closed; learned-component redesign discussion next (2026-09-04)

Independent audit accepts the machine verdict `HCH_V3_1_TAIL_FIRST_D2_NOT_SUPPORTED`. The fixed 4×7-day chronological cross-fit is protocol-faithful: each fold derives CandidateSet/Overall-Tail-Normal support scores/kappa from the other 21 D2 days and evaluates only on the untouched 7-day block. Independent recomputation gives nonzero action availability `5/6`, strictly positive Tail-MAE gain `1/6`, Overall-MAE harm violations `2`, Normal-MAE harm violations `2`, and zero chronology/provenance/numerical failures. HCH tests are `41/41` PASS, demo tests `27/27` PASS, compile/static isolation passes, admitted source tree is `c08fdda02f3f2860ac833fe4d2221c57a23fc7050b7f269736de3db696420a6e`, and clean bundle revision is `2378a2faf40d12833ca78b6adcac218219d3bbc9`.

The Gate/selector rescue line is therefore **stopped**. `SIGNED_MAE_BAYES_GRID_V1` remains accepted because it corrects the old fixed-q=.75 action/loss inconsistency and restores usable low-quantile actions without changing learned weights. However D2 margin selection cannot create an extreme-repair ranking that is absent upstream: DE actions are entirely Normal, and PJM retains ~10-12% action density with zero Tail action under cross-fit. The active scientific bottleneck moves to learned repair signal / representation semantics: Shape state definition and CORN parameterization, target locality, forecast-time context, and only secondarily Amplitude calibration.

No further kappa/selector/tau/tail-threshold/class-weight rescue is authorized. The preregistered fresh panel `EPEX_FR / NORD_DK1 / NORD_NO × PatchTST / TimeMixer` remains completely untrained/unevaluated. Consumed DE/NORD_FI/PJM S2V must not be reused for design selection; original S3/S4 remain protected. Any Shape/Amplitude/sharing/D1-adapter architecture change must first be discussed with the user before implementation.

Independent Tail-first cross-fit audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_V3_1_TAIL_FIRST_CROSSFIT_AUDIT_20260904.md`. Current stage-closeout authority: `docs/current/HCH_V3_V3_1_STAGE_CLOSEOUT_AND_NEXT_WINDOW_HANDOFF_20260904.md`. New-window entry: `docs/current/NEXT_WINDOW_START_20260904.md`.

## HCH V3.1 D2 mechanism validation and independent adjudication (2026-09-04)

The registered `SIGNED_MAE_BAYES_GRID_V1` Gate correction passed the authorized
source/G-Val/D1/D2-only six-cell action-policy mechanism gate. It reused the
frozen V3 learned state dict and changed no architecture, trainable parameter,
Host, split, support, baseline, or optimizer. Historical
`FIXED_DIRECTIONAL_QUANTILE_V1` replay is preserved. Independent replay confirms
36/36 HCH tests and 27/27 demo tests, clean bundle revision
`5155047970597f0ffd9ef58c438ab77f31906591`, and admitted source tree
`0653a7bfc8abdc5ad660cfb7a7cfbeef92b189d2eaa9452460340fb6641be4dd` before
this adjudication update.

V3.1 recovered nonzero candidate/action availability in 5/6 cells,
nonnegative D2 Overall-MAE gain in 6/6, and zero Normal-MAE harm violations.
Selected actions overwhelmingly use the low directional quantiles (.10/.25),
which strongly supports the old fixed-.75 overshoot diagnosis. However the
current D2 selector is still Overall-MAE-first: D2 Tail-MAE gain is 0/0/about
+0.10%/0/about -0.50%/about -0.22% across DE/PT, DE/TM, NORD/PT, NORD/TM,
PJM/PT, PJM/TM. Thus the action policy is supported, but fresh-panel execution
is not yet authorized.

Independent controlling verdict:
`HCH_V3_1_D2_ACTION_POLICY_PASS / TAIL_OBJECTIVE_ALIGNMENT_REQUIRED_BEFORE_FRESH_PANEL`.
The next active algorithmic object is the parameter-free
`D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1` selector: minimize D2 Tail-MAE only among
candidates that do not worsen either Overall-MAE or Normal-MAE versus Identity,
with deterministic lexicographic ties and Identity fallback. This is a D2
selector detail, not an architecture change. Its promotion evidence must use
four fixed contiguous 7-day cross-fit folds inside the legal 28-day D2 support
(select on 21 days, evaluate on the untouched 7 days) to avoid in-sample Tail
selection optimism over many breakpoints. Validate it once on the same legal
D2 supports; consumed S2V and all S3/S4 remain forbidden.

Independent audit:
`experiments/evidence/hch_v3_demo_validation/INDEPENDENT_V3_1_D2_MECHANISM_AUDIT_20260904.md`.
Active selector design/prompt:
`docs/current/HCH_V3_1_TAIL_FIRST_SAFE_D2_SELECTOR_DESIGN_20260904.md` and
`docs/current/HCH_V3_1_TAIL_FIRST_SAFE_D2_SELECTOR_AI_PROMPT_20260904.md`.

## Current main research conclusion

### 2026-09-04 authorized V3 demo independently adjudicated; V3.1 narrow Gate correction is active

The authorized DE_EPEX/NORD_FI/PJM_2020 × PatchTST/TimeMixer S2V demo is now **consumed and independently audited**. Execution/provenance and all stored point metrics are accepted, but the executed HCH-V3 method is not supported as a repair method: `HCH-V3-full` and `HCH-V3-global-only` are pointwise exactly Identity over all 394 S2V days because the Gate never acts. The historical machine token `DEMO_MECHANISM_ONLY` remains immutable provenance, while the independent scientific verdict is `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`.

Independent audit also corrects three evaluator/diagnostic issues without changing historical predictions: metric-specific BestPeer gives MAE near-best `6/6` but Tail-MAE near-best `4/6` (not `6/6`); the machine mechanism-only fallback was too permissive because merely computed diagnostics triggered it; and directional Amplitude coverage/pinball was incorrectly computed over all states rather than true matching Down/Up states. The registered 7-day paired bootstrap CI requirement was also not fully materialized by the executed reporter. These are post-processing/evaluator defects, not prediction changes.

The active scientific hypothesis is a **single minimal V3.1 decision-consistency correction**. Preserve Representation/Shape/Amplitude/signed-distribution learning, Global training, D1 and D2 semantics. Replace only the fixed directional `Q^±(0.75)` action proposal with a parameter-free discrete Bayes action over the already-trained directional quantile grid, minimizing the same expected-MAE risk used by the Gate. A legal D2-only counterfactual with the frozen V3 learned distributions restores nonzero safe actions in 5/6 cells and nonnegative D2 Overall-MAE gain in 5/6, including about +2.61%/+1.91% in PJM. This is development evidence only and does not authorize reuse of consumed S2V.

Controlling audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_AUTHORIZED_S2V_RESULT_AUDIT_20260904.md`. Active design: `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_DESIGN_20260904.md`. Active local-AI prompt: `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_AI_PROMPT_20260904.md`. Authorization is limited to implementation + source/D1/D2 mechanism validation. Original DE/NORD/PJM S2V must not be rerun for V3.1; S3/S4 remain protected; a fresh evaluation panel may be designed only after independent D2 mechanism-gate passage.

### 2026-09-02 integrated framework + reproduction implementation (historical implementation stage)

The current preferred **design hypothesis** for the next extreme-repair framework has become more coherent, but is not yet frozen as a method:

1. preserve the frozen-host post-processing identity;
2. use two semantic input views rather than a generic feature split:
   - market-relative **shape/state view** for abnormality and host-failure state;
   - real-scale **amplitude view** for correction severity;
   - exchange only a small learned shared state between them, preserving private information in both views;
3. reinterpret the old `Down / Identity / Up` atoms as ordered **repair states**, not fixed price candidates;
4. the Shape branch retains **ordered asymmetric repair-state semantics**, with a mature **CORN-style conditional ordinal head** as the preferred default and CORAL retained as an ablation/alternative. For three states, two stable logits produce `q1=P(S in {Identity,Up}|X)` and `q2=P(S=Up | S in {Identity,Up},X)`, hence `p_-=1-q1`, `p_0=q1(1-q2)`, `p_+=q1 q2`. This couples the three states without isolated softmax heads or explicit exponential normalization, while avoiding CORAL's output weight-sharing constraint. Rare-state training should use cross-market pooling, bounded effective-sample/class-prior correction, event/day-aware batching and calibration rather than unbounded inverse-frequency weighting or focal loss as the default;
5. Up/Down severity uses one Amplitude branch with independent directional quantile heads and the preferred **cross-market scale-equivariant severity** formulation: the shared model learns dimensionless positive/negative repair magnitude distributions, while causal direction-specific residual scales restore real price units. A shrunk directional scale carrier is preferred to stabilize few-shot sides. Non-crossing IQF/ISQF-style conditional quantile functions remain the uncertainty representation;
6. the earlier day-fixed `alpha_d^+ / alpha_d^-` weighted-quantile router is removed from the preferred core. Each direction uses one fixed risk quantile `tau_+^* / tau_-^*`, motivated by asymmetric under/over-repair cost, while the complete hourly quantile curve remains available to the final decision. Direction-level and quantile-level loss weights may be asymmetric but should remain fixed/low-complexity rather than becoming a new learned router;
7. Shape and Amplitude no longer collapse first to the mixture-mean correction `p_+ m^+ - p_- m^-`. They jointly define a **signed repair distribution** with probability mass at `Identity=0` and continuous negative/positive severity components. This prevents cancellation from turning strong but opposite repair hypotheses into a misleading small correction;
8. the final Gate is now preferred as a **deterministic Host-preserving expected-risk action selector**, not another neural router. It compares `{Keep Host, Down repair, Up repair}` under the signed repair distribution and executes a nonzero action only when its expected risk beats Keep by a scale-normalized safety margin. Thus repair/no-repair and direction are one decision problem, while Shape remains a state-probability model and Amplitude remains a severity-distribution model;
9. the preferred cross-market training direction is **one shared repair core plus lightweight target-deployment adaptation**. The audited target deployment is exactly **five differentiable D1 scalars** (two state-logit biases, one temperature, two directional severity gains) plus **one deterministic D2 Host-preservation margin**; low-rank/FiLM adapters are not in the registered default. No explicit Market-ID or Host-ID shortcut is required;
10. Stage I is now organized at paper level as three semantic objects `X^c / X^s / X^a`, a shared forecast-time covariate context, two private encoders and one low-capacity residual soft-sharing Bridge. The full integrated data flow, equations, parameter plan, training algorithms and inference algorithm are consolidated in the single current formal design document.
11. **Paper-reproduction implementation is complete at implementation level** under `src/mvp/hch_v3_cross_market_extreme_repair/`. The controlling final independent micro-closure audit grants `PROTOCOL_READY`: 35/35 synthetic tests pass, compile/static isolation passes, the standalone closure process passes, `fresh_empty_v1` optimizer-state semantics are enforced, chronology identity is cross-object closed, and identical frozen tuples reproduce identical D1 output hashes. No new contradiction was found in the architecture/math core. Generic implementation hardening is therefore stopped under the registered hard-stop rule. The next stage is data-specific, chronology-safe scientific-validation protocol design/freezing. Scientific effectiveness remains untested; protected benchmark execution, result-conditioned tuning, method promotion, and paper-result generation remain unauthorized until a protocol is frozen and the user explicitly authorizes execution. The controlling audit is `docs/current/HCH_V3_INDEPENDENT_IMPLEMENTATION_AUDIT_20260902.md`.
11a. **A first small HCH V3 development-demo protocol is now drafted but not executed.** The fixed sentinel panel is `DE_EPEX / NORD_FI / PJM_2020 × PatchTST / TimeMixer`, reusing existing audited multiyear frozen-host caches. The outer date-first `S1/S2/S3/S4=50/20/10/20` boundary is preserved; within S2, `S2T/S2V=16/4` is recovered. For each target deployment, the last 84 S2T days are the complete target-fitting truth budget (`56d D1 + 28d D2`), while S2V is development-demo evaluation only and S3/S4 remain unread. Each target-market-excluded global core trains only on the other two markets' S2T data over both Hosts with source-only chronological G-Train/G-Val. Headline external peers are Identity/PIR/delta-Adapter Ada-Y/UEC-STD; CRC is appendix-only. The demo D2 evaluator uses overall MAE as support score and exact Normal-MAE non-degradation versus Identity as the Host-harm constraint; Tail-MAE does not tune kappa. Full protocol: `experiments/evidence/hch_v3_demo_validation/HCH_V3_DEMO_VALIDATION_PROTOCOL_20260903.md`; local-AI preparation/execution-guard prompt: `experiments/evidence/hch_v3_demo_validation/HCH_V3_DEMO_VALIDATION_AI_PROMPT_20260903.md`. No demo execution is authorized yet; the local AI must stop after preflight/freeze unless the user explicitly supplies `DEMO_EXECUTION_AUTHORIZED`.
12. **2026-09-02 second implementation re-audit found remaining pre-protocol blockers.** The first repair pass successfully removed raw monetary learned-path inputs on the normal non-degenerate domain, added zero-total-history fail-closed behavior, and split adaptation into D1 five differentiable scalars plus D2 deterministic `kappa_bar`. The second audit then found: (a) `repair_available` depends only on history count, so one/two/constant/all-zero residual histories can still produce `pooled_scale=1e-6`, `repair_available=True`, and learned Amplitude inputs up to about `8e7`; (b) flat/step-like Host trajectories can make the within-day Shape scale fall to `eps`, producing `X^s` values around `3e7` and destroying the claimed rebuilt-input scale invariance; (c) unavailable samples are not masked from target/loss construction, so zero-history examples can create normalized severity targets of order `1e6-1e7`; (d) deterministic `kappa_bar=0` is not represented exactly because softplus maps it to about `1e-6`; (e) exact source provenance is not established while the current HCH V3 package is untracked and Git dirty detection ignores untracked files; and (f) masked residual-history NaNs and out-of-grid `tau` semantics still need hardening. These are implementation/protocol blockers, not scientific-effect results.
13. **2026-09-02 literature-conditioned component-hardening plan is now the preferred repair direction.** The high-level architecture is unchanged. The scale system is separated into a monetary price-reference coordinate `s^P`, a pooled absolute repair-magnitude coordinate `s^R = Q_q(|r|)`, and direction-specific severity scales shrunk toward `s^R`; this replaces residual dispersion as the severity carrier and correctly keeps constant non-zero residual histories non-zero. Shape day normalization should use positive-homogeneous MeanAD-around-median with an exact flat-day branch, and within-day ranks must be tie-aware. Training and inference must share one eligibility mask so invalid-scale rows contribute zero Shape/Amplitude loss. Amplitude monetary inputs should normalize by `s^P`, while residual/price scale ratios remain dimensionless. D2 should store exact deterministic `kappa_bar` and may calibrate it from normalized risk-advantage breakpoints rather than an arbitrary dense grid. `tau_*` must lie inside the trained quantile grid unless an explicit continuous IQF/ISQF representation is later promoted. Exact source-tree provenance is required before scientific execution. Full design: `docs/current/HCH_V3_COMPONENT_HARDENING_AND_LITERATURE_REFINEMENT_20260902.md`; strict local-AI prompt: `docs/current/HCH_V3_COMPONENT_HARDENING_AI_PROMPT_20260902.md`.

14. **2026-09-02 component-hardening pass produced real positive implementation evidence, but the third re-audit found remaining contract gaps.** Confirmed positives include `s^R=Q_q(|r|)` with constant-nonzero validity/all-zero invalidity, MeanAD-around-median Shape normalization, tie-aware ranks, standard loss masking by repair eligibility, exact `set_kappa_bar(0)`, in-grid-only quantile readout, D2 risk-advantage breakpoint geometry, untracked-aware Git dirty status, and exact source-tree hashing. The third audit nevertheless finds: (a) valid directional and price scales still contain fixed absolute `eps` clamps, breaking the formal every-positive-rescaling homogeneity contract below the epsilon boundary; (b) target construction still accepts scientific deltas/extreme weights independently of the recorded config, so the same manifest can generate different labels/importance; (c) the public D1 calibration callback can ignore eligibility and update on unavailable rows; (d) `kappa_bar` remains a re-enableable `nn.Parameter` rather than a structurally deterministic scalar/buffer; (e) admitted-source tracked/untracked status is not explicitly recorded; and (f) mandatory hardening regression coverage is incomplete. Controlling verdict: `IMPLEMENTATION_CORE_PASS / THIRD_REAUDIT_HARDENING_INCOMPLETE`.
15. **2026-09-02 final implementation-sealing design remains the controlling narrow repair direction.** The high-level Shape/Amplitude/Gate method is unchanged. The local sealing implementation materially fixes ScaleCarrierV3, official D1 loss ownership, persistent deterministic kappa, qR/qS separation and source tracking metadata, but only an independent audit can adjudicate readiness. Full design: `docs/current/HCH_V3_FINAL_IMPLEMENTATION_SEALING_DESIGN_20260902.md`; strict local-AI prompt: `docs/current/HCH_V3_FINAL_IMPLEMENTATION_SEALING_AI_PROMPT_20260902.md`. No scientific experiment is authorized.
16. **2026-09-02 fourth independent sealing audit found new execution/config-closure blockers.** Positive results: 11/11 tests, compile and archive checks pass; ScaleCarrierV3 is positive-homogeneous across `1e-9..1e9`; mixed-eligibility loss gradients equal eligible-only gradients; kappa is a persistent non-Parameter; Gate Keep/Down/Up signs are correct; source hash/provenance reporting works. Blockers: (a) all-invalid global/D1 steps still call `optimizer.step()`, so AdamW weight decay changes parameters despite exact zero loss (`~1e-3` global, `~5.4e-4` D1 in independent probes); (b) global pretraining still accepts a caller quantile grid independent of `ModelConfig.quantile_levels`, so one manifest can train different pinball objectives; (c) TargetConfig materialized thresholds and manifest `materialized_thresholds` remain duplicate semantic sources, and `rule='none'` can still activate weights; (d) CalibrationProtocolConfig records fields such as D1 steps/objective and D2 safety metadata that official execution does not enforce; (e) effective-number state can be inconsistent with LossConfig and public low-level overrides remain; (f) the 48-row permanent matrix overstates several assertions and its provenance test hard-codes today's untracked state. Historical verdict: `IMPLEMENTATION_CORE_PASS / FOURTH_REAUDIT_SEALING_INCOMPLETE`.
17. **2026-09-03 fifth independent execution-contract audit finds the implementation substantially improved but still not protocol-ready.** Confirmed positives: 19/19 tests, compile pass, pure-AST archive import scan `[]`; ScaleCarrierV3 remains valid; official global/D1 use ModelConfig quantile levels; typed target/class-balance/D2 objects are real; parameter/optimizer-state invalid-batch no-step tests pass; source-tree hash/provenance matches the local handoff. New blockers: (a) global all-invalid steps still execute stochastic model forward before the skip check, changing torch RNG state (2494 state bytes in an independent dropout probe) and causing the next valid update to diverge (`max parameter diff ~0.00438` across 68 tensors); (b) a non-none threshold rule can be paired with an all-null materialized state and silently apply no extreme importance, and arbitrary unregistered rule strings are accepted; (c) a FROZEN CalibrationProtocolConfig can claim an optimizer ID and early-stopping rule not actually enforced by the runner; (d) the public low-level loss still permits manual `class_weights` to alter the objective under the same LossConfig; and (e) typed D2 support rows are not yet bound to model-derived risk-advantage breakpoint provenance. Controlling verdict: `IMPLEMENTATION_CORE_PASS / FIFTH_REAUDIT_EXECUTION_CONTRACT_INCOMPLETE`.

The current brainstorming record is `docs/current/EXTREME_REPAIR_PROBABILISTIC_BRAINSTORM_20260902.md`.

The single current formal candidate design is `docs/current/HCH_DUAL_VIEW_ASYMMETRIC_PROBABILISTIC_EXTREME_REPAIR_DESIGN_v0.1_20260902.md`. It is paired with the reproduction implementation contract `docs/current/HCH_V3_REPRODUCTION_IMPLEMENTATION_CONTRACT_20260902.md` and source package `src/mvp/hch_v3_cross_market_extreme_repair/`. The design document has been aligned to the audited implementation after coding: ScaleCarrierV3 uses `s^R=Q(|r|)`, separate `s^P`, MeanAD/tie-aware Shape coordinates, registered 7D Shape/6D Amplitude features, and 5-trainable + 1-deterministic deployment adaptation. The architecture remains a **paper candidate rather than a scientifically validated method**: generic implementation is `PROTOCOL_READY`, but scientific effectiveness remains untested.

The submission target is an **ICDE-level model-agnostic post-processing module for repairing extreme electricity-price forecasts**, evaluated across multiple electricity markets and strong frozen forecasting backbones. The project is now in **scientific validation protocol design**, not generic implementation hardening.

The strongest surviving body component is target-local IAMR3, but the final paper method is no longer constrained to remain a low-complexity monotone calibrator only. New extreme-repair components may change representation/architecture provided they preserve the frozen-host post-processing interface and are validated honestly.

Headline research objective:

> frozen strong host + legal target support/history -> architecture-agnostic post-hoc correction that improves both overall accuracy and rare extreme-price accuracy, with enough novelty, mathematical structure, robustness evidence, and efficiency to compete at ICDE.

Cross-market deployment and broad multi-market/backbone performance are required. The current preferred internal mechanism is one shared repair core with lightweight target-deployment calibration; alternative parameter-sharing schemes are not part of the active design unless new evidence later requires them.

## Current strongest candidate

HCH V5 IAMR remains the strongest empirical core. V5.1-TLR and V5.2-APTC are both rejected as freeze designs. No freeze-ready V5 module is currently authorized.

The active candidate is coherent at architecture/mathematical level and has a clean-room implementation under `src/mvp/`. The controlling final independent implementation audit grants implementation-level `PROTOCOL_READY`. The architecture remains stable and no new CORN/Amplitude/Gate redesign is indicated. The complete target-deployment execution closure `E=(C,theta0,S,U,D,R,H,M)` now passes the registered mutation/lifecycle/reproducibility tests, including fresh-empty optimizer-state semantics and cross-object chronology identity. Generic implementation polishing is stopped. The next bottleneck is the **data-specific scientific-validation protocol**: Global Core Training, legal target support, concrete D2 score/harm evaluator, baselines, metrics, ablations, protected-panel boundaries, provenance and stop rules must now be written and frozen. Scientific effectiveness remains unknown; S2-V/S3/S4 and protected benchmark execution remain unauthorized until that protocol is frozen and explicitly authorized.

Core evidence on DE_EPEX / NORD_FI / PJM_2020 × PatchTST / TimeMixer:

- host-to-truth rank structure is strong (S2-V Spearman roughly 0.64–0.92);
- IAMR2: 7-knot identity-anchored monotone map, M1 passed, 5/6 cells improved vs legal V2.5 endpoint, no >2% MAE-harm cell, median gain ~1.34%;
- IAMR3: IAMR2 + hour-of-day transformed-space offsets, extension gate passed in 4/6 cells vs IAMR2;
- direct machine-number composition implies IAMR3 improves the legal V2.5 endpoint in all 6 original cells, approximate gains +0.34%, +0.63%, +18.06%, +10.35%, +1.56%, +5.84%; median ~+3.70%;
- IAMR3 overall/normal MAE gains do NOT guarantee tail safety: PJM UpperTail/Tail-MAE materially worsens vs Identity; NORD tail behavior is less problematic;
- IAMR predictions are deterministic conditional on host cache/S2 data; duplicated seed rows in V5 are not independent IAMR training seeds. Future stability must use temporal blocks/bootstraps or genuinely different host seeds.

V5 primary verdict: IAMR_TARGET_LOCAL_ONLY.

V5.1-TLR primary verdict: `TLR_TAIL_SAFE_FREEZE_NOT_SUPPORTED`.

Important TLR findings:

- Stage A replay/statistical readjudication passed;
- all 6 Stage-B cells selected `lambda_h=infinity`, so the new Fourier hour deviation was shut off everywhere; TLR effectively collapsed to IAMR2 + alpha;
- target-S2 tail-aware alpha selection did not satisfy Identity/Legal coverage gates and failed PJM_2020 × PatchTST UpperTail safety;
- PJM S2-T contains many upper events overall but they are temporally clustered, so some chronological validation folds have too little extreme support to reliably supervise a tail-safety selector;
- symmetric tail Identity/slope-one protection is rejected because NORD_FI obtains very large IAMR gains in the lowest host-rank decile;
- development diagnostics indicate the failure is asymmetric: positive spikes are vulnerable to calibration compression, whereas lower/negative-price calibration is often beneficial.

V5.2-APTC primary verdict: `APTC_DEVELOPMENT_NOT_SUPPORTED`.

Important APTC findings:

- historical D0 replay/audit and D1 algebraic contract passed, and the stored D2 headline numbers remain internally reproducible;
- **2026-08-31 post-hoc source/provenance re-audit found a semantic contract error:** the actual APTC junction inherited `cell["s1_q95"]` from TLR, where it is computed from **S1 truth**, not S1 host forecasts. The D1/report wording `S1 host forecasts only` is therefore incorrect; the historical `APTC_FINAL_AUDIT_OK` certified artifact/number consistency but did not verify q95 provenance;
- the historical V5.2 verdict remains `APTC_DEVELOPMENT_NOT_SUPPORTED`; this correction does not rescue APTC and does not authorize S3/S4/freeze;
- upper raw-space slope-one continuation itself remains mechanistically useful: on PJM it reduced IAMR3 UpperTail relative harm from ~+26.42%/+23.83% to ~+9.72%/+9.95% (about 63%/58% of the IAMR3 upper harm removed), while overall gains remained positive;
- corrected actual-junction drift is even more restrictive: DE/NORD actual S1-truth-q95 APTC junction never triggers on S2-V host predictions; PJM PatchTST actual threshold 103.33 is ~99.40 percentile of S2-V host predictions with ~0.60% trigger rate, and TimeMixer 103.33 is ~99.23 percentile with ~0.77% trigger rate;
- the earlier S1-**host**-q95 values (~101–103 in PJM, ~99.27%/~99.23% later percentile) remain valid only as a post-hoc host-only drift diagnostic, not as the threshold actually deployed by V5.2;
- therefore the bottleneck is the forecast-safe coordinate used to define the upper-tail region under host marginal shift, not the slope-one continuation shape alone.

## V5.3-RATJ final result

V5.3-RATJ primary verdict: `V5_3_RATJ_HOST_RANK_NOT_SUPPORTED`.

The registered isolated implementation and hard gate behaved correctly: contract tests replay as `7 passed`; G0/G2/G3 passed; G1 failed; R1 was not executed; no S3/S4, q-level search, truth-based R0 selection, `src/hch_v2/**` modification, or historical-artifact overwrite occurred.

Important R0 findings:

- all six cells selected the shortest preregistered EW half-life, 7 days, by S2-T host-only prequential q95 pinball loss;
- EW-CDF restores near-5% pooled S2-V occupancy in DE and PJM: ~5.12%/~5.16% and ~5.28%/~5.63%;
- NORD remains materially miscalibrated at ~12.07%/~12.33%, so the frozen global q95-semantic gate correctly fails;
- NORD failure is not diffuse: 7-day blocks reach ~28.57%/~30.36% exceedance and individual days reach 18/24 and 17/24 flagged hours; strong hour-of-day concentration is also present;
- S2-T half-life folds already show a large difficult NORD regime block, so the result is not sufficient evidence to search post-hoc for 1/3/5-day half-lives;
- retrospective truth diagnosis performed only after R0 was frozen shows PJM EW-tail flags cover 86.4% of historical S1-truth-q95 upper events for both hosts, while NORD alignment is strongly host-dependent (PatchTST 22.2%, TimeMixer 100%); a single host percentile therefore cannot be claimed as a universal truth-tail proxy;
- adaptive forgetting itself remains useful: EW-CDF improves host-only pinball loss over Fixed-S2T/Expanding and usually over Trailing28. What is rejected is the **single pooled marginal host-CDF junction as a universal six-cell tail coordinate**.

The heuristic global rank-junction route is now stopped according to the preregistered R0 rule. Do not rescue V5.3 with a shorter half-life, DSPOT-on-host junction, new q-level, or PJM-only post-hoc R1 on the same panel.

## Current bottleneck

Preserve IAMR3 as the strongest low-complexity target-local body calibrator while determining whether its unresolved **positive residual tail** has a coherent, low-complexity conditional extreme-value structure under honest chronological evaluation.

The next registered research object is `V5.4 Conditional EVT Tail Feasibility`, but **no V5.4 correction architecture or correction prediction is authorized yet**.

The invariant feasibility object is the honest positive IAMR3 residual tail:

`R = truth - cross-fitted IAMR3 prediction`.

The next audit must:

- construct forward chronological cross-fitted IAMR3 residuals on S2-T only, rather than using in-sample residuals;
- audit positive-tail sample size, distinct event days, clustering, and dependence before fitting any tail correction;
- evaluate stationary POT/GPD stability on a fixed diagnostic threshold ladder, with a daily-declustered companion analysis;
- test whether any nonstationarity is explainable by a small frozen set of forecast-time legal covariates: host level, hour-of-day, current-day 24h host median/IQR, and within-day standardized host position;
- use S2-V only for source-integrity loading if unavoidable by APIs, never for fit/model/threshold/covariate selection and never for V5.4 correction performance;
- stop if honest positive-tail evidence is too sparse, threshold-unstable, or requires a fragmented controller-like model.

V5.4 E0 has completed with verdict `V5_4_CONDITIONAL_EVT_TAIL_FEASIBLE`.
The result is a genuine **existence/modelability** result, not a six-cell universal conditional-model certificate. Honest S2-T cross-fitted IAMR3 residuals retain substantive positive tails in all six cells; all six cells have converged hourly and daily-declustered GP fits on the frozen ladder; q95 support spans 24–71 distinct event days.

Critical post-result interpretation:

- PJM is the strongest conditional-EVT case: q90–q95 hourly `xi` is about 1.03–1.18, while daily-declustered `xi` is about 0.53–0.77;
- because a GPD first moment exists only for `xi < 1`, **GP mean excess is forbidden as a V5.4 correction primitive**; E0 does not justify `tail probability × expected excess` or any direct expected-residual add-back;
- DE has moderate positive tail shape and simple day-regime association;
- NORD has near-zero/weakly negative upper-tail shape above roughly q85 and weak one-dimensional conditional association, so a stationary cell-local GP may be sufficient there;
- the strongest recurring legal occurrence covariates in the historically vulnerable PJM cells are raw host level and current-day host median; no common covariate-dependent shape model is justified.

V5.4-E1 has completed with verdict `V5_4_E1_TAIL_DISTRIBUTION_NOT_IDENTIFIED`. Contract/provenance G0 and selected-model bootstrap G4 passed; G1 failed because both PJM cells were not identified (PJM/PatchTST Fold A retained only 19 validation exceedances), G2 failed on NORD Fold-A occurrence gaps, and G3 failed on NORD severity coverage plus PJM q90 overcoverage. M0 was selected for DE/NORD and M1 for PJM. No S2-V correction prediction/evaluation, S3/S4, mean-excess action, or rescue was run.

Post-hoc source re-audit found one **diagnostic scoring bug** in E1: conditional M1 PIT/q50/q90 coverage was computed with a median validation sigma instead of per-sample `sigma(C_i)`. Correct recomputation does not change the verdict: PJM/PatchTST pooled q90 coverage remains 1.0000 and PJM/TimeMixer becomes ~0.9933, while G1/G2 are unaffected. Do not reuse the E1 calibration scorer.

The decisive scientific failure is broader than that bug: the raw-residual q90 tail domain itself shifts strongly across folds (NORD ~12 -> ~38/46), while conditional likelihood helps mainly in PJM. The fixed-threshold low-dimensional HGP route is therefore stopped rather than rescued.

## Current bottleneck and registered next stage

V6-R0 architecture race is complete with `R0_TOP2_AUTHORIZED=false`; no S2-V was loaded. Independent replay gives `6 passed` and the independent gate recompute matches the machine verdict.

The decisive new mechanism is **dense-action spillover**, not simply lack of proposal signal:

- A_CT_EQ and B_REMR move from near-positive Fold-A median S to negative Fold-B S, confirming shift failure;
- C_BEQA retains positive median F3 UpperTail gain (~+6.99%) but has large overall/normal harm, so it contains useful extreme information with an unsafe release policy;
- B_REMR is the only arm with near-safe median F3 Normal harm (~+0.76%) and exposes a natural pre-truth support-overlap signal through neighbor distances;
- in both NORD hosts, **all three arms have F3 positive 7-day block fraction = 0**, so every 7-day block is harmed;
- across F3 cells/arms, only roughly 17%--56% of dense actions satisfy the retrospective sufficient no-harm geometry `sign(A)=sign(R), |A|<=2|R|`.

Therefore the project pivots to **HCH V6.1 Certifiable Extreme Repair (CER)**. The new invariant is not a learned router. It is a one-sided residual lower-bound certificate used to release a positive bounded action only when no-harm geometry is statistically supported.

For residual `R=Y-B`, forecast-time proposal `U>=0`, lower bound `L`, and robust cap `s`, V6.1 uses

`A = G * 1{L>0} * min(U, 2L, s)`.

If a day-level simultaneous certificate guarantees `R_h >= L_h` for all 24 hours with probability at least `1-alpha`, then every released action is non-harmful on that event; harmful correction probability is therefore bounded by certificate miscoverage under the stated exchangeability assumptions.

Three fixed V6.1 variants are registered in parallel:

1. `CER_LB`: certificate-only conservative action;
2. `CER_Q`: frozen BEQA q75 positive proposal capped by the certificate;
3. `CER_R`: K=16 REMR q75 positive proposal plus a training-reference support-overlap gate, capped by the same certificate.

Frozen certificate constants: `alpha=.20`, residual q20 base lower quantile, day-level max score across 24 hours, positive-only action. No alpha/q/K/network search is authorized.

Internal chronology: fit on E0 F1, split F2 chronologically into calibration half and evaluation half. If at least one CER variant passes the internal screen, refit on F1+F2, calibrate on F3, hash/freeze state, then allow at most top2 a **single** S2-V development evaluation. No S2-V tuning/retry and no S3/S4.

The eventual ICDE candidate must later beat/compete with paper-faithful modern post-processing peers, not merely Identity/IAMR. `delta-Adapter`, COSA and retrieval-based methods such as RAFT define part of the current novelty/performance bar when information boundaries are comparable. The CER differentiation is an extreme-price-specific **certifiable action release** layer rather than another generic residual adapter.

This does not reopen V4-DRQ: V5.4 does not predict the full residual center. IAMR3 remains the body and only the rare positive residual tail distribution is modeled.

Stage synthesis / handoff:
`docs/history/v5_iamr/HCH_V2_6_TO_V5_2_STAGE_SYNTHESIS_AND_HANDOFF_20260831.md`

V5.2 re-audit / rank-junction adjudication:
`docs/history/v5_tail/HCH_V5_2_APTC_POSTHOC_REAUDIT_AND_RANK_JUNCTION_REVIEW_20260831.md`

V5.3 final R0 audit / literature adjudication:
`docs/history/v5_tail/HCH_V5_3_RATJ_R0_AUDIT_AND_CONDITIONAL_TAIL_ADJUDICATION_20260831.md`

V5.4 feasibility documents:

- `docs/archive/v5_tail_details/HCH_V5.4_CONDITIONAL_EVT_TAIL_MATHEMATICAL_ADJUDICATION_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_EVT_TAIL_FEASIBILITY_AUDIT_PROTOCOL_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_EVT_TAIL_FEASIBILITY_PROMPT_20260831.md`

V5.4 E0/E1 documents:

- `docs/history/v5_tail/HCH_V5_4_E0_REAUDIT_AND_CONDITIONAL_GPD_ADJUDICATION_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_PROTOCOL_v0.1_20260831.md`

V5.4-E1 re-audit / ICDE pivot:

- `docs/history/v5_tail/HCH_V5_4_E1_REAUDIT_AND_ICDE_ARCHITECTURE_PIVOT_20260831.md`

V6 architecture race / V6.1 pivot documents:

- `docs/archive/v6_extreme_details/HCH_V6_ICDE_EXTREME_REPAIR_ARCHITECTURE_RACE_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6_R0_PARALLEL_ARCHITECTURE_RACE_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6_R0_PARALLEL_ARCHITECTURE_RACE_PROTOCOL_v0.1_20260831.md`
- `docs/history/v6_extreme/HCH_V6_R0_REAUDIT_AND_CERTIFIED_SELECTIVE_REPAIR_ADJUDICATION_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.1_CERTIFIABLE_EXTREME_REPAIR_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.1_CER_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.1_CER_EXPERIMENT_PROTOCOL_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.1_CER_EXECUTION_PROMPT_20260831.md`

## Important mechanism findings already established

1. V2.5 tri-atom geometry has large oracle headroom; support geometry alone is not the primary bottleneck.
2. Up-head structural repair failed; learning-signal repair can force Up but spills harm into lower/normal regions.
3. HES adds information but is market/host heterogeneous; deterministic action routing fails.
4. sample-level selective release and hierarchical/certifiability controllers fail universal safety certification.
5. cross-fitted audit found severe old in-sample proposal-state optimism, but honest cross-fitting and legal prequential updates still did not recover universal correctionability.
6. Shared and Local experts have real large-margin complementarity, but observable soft arbitration is not reliably learnable.
7. parameter-space Shared-to-Local L2-SP partial pooling failed; 18/18 HPA A1 selections chose lambda=infinity.
8. direct continuous residual-center MLPs (raw-normalized or asinh residual) failed when attached to the frozen old representation.
9. monotone host recalibration succeeded where residual prediction/controller routes repeatedly failed, suggesting the problem is better viewed as a low-complexity target-local calibration problem than a universal residual-prediction problem.

## Rejected / frozen routes

Do not reopen without new evidence:

- V2.6 Up-head repair variants U1-U4;
- V2.6 learning-signal variants L1-L4;
- universal deterministic signed routing;
- universal sample-level selective-release controller;
- hierarchical certifiability release as universal controller;
- host-certifiability classifier/HCS route;
- certified prefix-capacity controller line;
- simple frozen/rolling online SR2 adaptation as universal rescue;
- post-hoc Shared/Local soft router (PLA);
- old-core parameter-space L2-SP partial pooling (HPA);
- direct continuous center MLP on frozen H1 representation (DRQ M1);
- V5 shared-prior function-space partial pooling as headline requirement;
- V5.1-TLR Fourier-hour + alpha tail-safe freeze design;
- symmetric tail Identity fallback or symmetric slope-one continuation for IAMR (destroys NORD lower-tail gains);
- V5.2 static S1 raw-q95 APTC as a freeze design;
- V5.3 single pooled marginal host-CDF/rank junction as a universal six-cell tail coordinate;
- post-hoc V5.3 rescue through shorter EW half-lives, new q levels, host-only DSPOT junctions, or PJM-only R1 on the same panel;
- post-hoc threshold fishing on the old six-cell S2-V development panel.

## Claim boundary

Current defensible claim boundary before the next phase:

> model-agnostic target-local monotone recalibration can improve strong day-ahead electricity-price hosts using a small chronological support set, especially in the body and low-price regions; however, rare positive spikes remain unresolved, and neither static absolute tail junctions nor one pooled adaptive host-percentile junction provide a reliable universal protection coordinate under temporal/regime shift.

Do not currently claim universal zero-shot cross-market correction, universal tail-safe recalibration, or a freeze-ready V5 module.

## V6.1 CER final result

V6.1 Certifiable Extreme Repair completed with verdict `V6_1_CER_INTERNAL_ONLY_NOT_SUPPORTED`.

- Only E0 honest cross-fitted IAMR3 residuals were used: E0 F1 fit, then chronological E0 F2 first-half calibration and second-half evaluation.
- `CER_LB`, `CER_Q`, and `CER_R` all passed finite/provenance, normal-harm, and all-hours harmful-action limits, but all failed usefulness and release gates. The maximum cell day-miscoverage also exceeded `.25` in the NORD cells. Median release rates were `0.0548%`, `0.0548%`, and `0%`; median `S=0` for all three.
- No variant was internally promotable. No top2 was authorized and no S2-V truth/prediction was loaded. S3/S4, rescue search, negative correction, learned router, and GP mean-excess action were not used.
- This is a fixed q20 day-simultaneous certificate conservatism / insufficient-actionability failure on the E0 chronology, not evidence for S2-V support or ICDE/SOTA success.

Machine root: `experiments/evidence/v6_1_certifiable_extreme_repair/`

## V6.1 post-result adjudication and V6.2 registered stage

V6.1 is now **closed as the primary correction route**. Do not rescue it by searching alpha, certificate quantile, calibration-window length, or another day-simultaneous conformal score on the same panel.

New machine/result-conditioned facts:

- CER q20 day-simultaneous calibration is not merely conservative: NORD day coverage collapses to roughly 61%/50%, so the exchangeability/score-stationarity assumption is visibly broken in the hardest market while qhat simultaneously suppresses almost every action;
- `CER_Q` proposal-positive hours have only about 52% positive-residual precision overall; `CER_R` is positive almost everywhere with about chance-level residual-sign precision. A better certificate alone would not rescue these proposals;
- the formal host cache exposes the previous **168h observed target-price context**, which the frozen PatchTST/TimeMixer already use but V6/CER postprocessors largely discarded;
- exploratory context-retrieval diagnostics show the 168h history contains useful UpperTail/regime information, but dense retrieval action still harms overall MAE;
- the correct supervised object is now the **host-specific missed-extreme event** `E=1{truth>=S1_q95 AND IAMR_residual>0}`, not generic residual positivity;
- using legal 168h context + current host/body, simple F1+F2->F3 event diagnostics rank DE/PJM missed-extreme hours strongly (roughly AUROC 0.9+), while NORD improves markedly when modeled hierarchically at the day level;
- NORD F1 contains zero missed-extreme event hours/days, while F2/F3 contain many. This is new evidence that rare-event occurrence can require a cross-market shared fallback even though shared body/IAMR pooling previously failed;
- a sparse exploratory day->hour prototype (selected days, top-risk hours, conservative event severity) yields positive UpperTail behavior with near-zero median overall change, unlike dense V6 actions. Remaining harm is concentrated in NORD false-positive/magnitude control.

The active registered stage is **V6.2 HERA — Hierarchical Extreme Risk Allocation**.

HERA invariants:

1. mandatory legal input includes previous 168h observed prices + current frozen host24 + IAMR body24;
2. supervision is missed-extreme occurrence, factorized as day event plus within-day hour localization;
3. raw shifted classifier probabilities are not used directly; chronological OOF score-rank reliability is used;
4. source sharing is restricted to insufficient-target-support occurrence/ranking fallback;
5. point correction is derived from the Bayes action of the extreme-weighted absolute-risk objective `E[(1+lambda E)|R-A|]`, i.e. an empirical weighted median projected to `[0,s_h]`;
6. exactly three V6.2 variants are registered: `HERA_Q1`, `HERA_Q2`, `HERA_R2`;
7. no CER simultaneous certificate, no mean excess, no fixed q50/q75 add-back, no learned router;
8. S2-V remains unread/locked until H0 S2-T gate passes and full variant state is hashed/frozen. S3/S4 remain forbidden.

V6.2 documents:

- `docs/history/v6_extreme/HCH_V6_1_CER_REAUDIT_AND_HERA_DESIGN_SYNTHESIS_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.2_HERA_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.2_HERA_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.2_HERA_EXPERIMENT_PROTOCOL_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.2_HERA_EXECUTION_PROMPT_20260831.md`

## V6.2 HERA H0 result (2026-08-31)

V6.2 HERA was executed in the isolated root `experiments/evidence/v6_2_hierarchical_extreme_risk_allocation/` using only E0 honest S2-T residual folds and formal legal previous-168h context. The final status is `V6_2_HERA_INTERNAL_NOT_SUPPORTED`.

- All three registered variants (`HERA_Q1`, `HERA_Q2`, `HERA_R2`) completed deterministic F2 and F3 internal evaluation. The missed-extreme target was exactly `truth>=S1_q95 AND IAMR residual>0`.
- H0-A failed: F3 day top-quartile lift exceeded 1 in 3/6 cells, while G1 hour precision lift exceeded 1 in 6/6 and G1+G2 recall exceeded random 4/24 in 6/6. The occurrence stop rule therefore applied (`positive_both_cells=3<4`).
- H0-B also failed for every variant. Q1/Q2/R2 had median `S=0.03333/0.04169/0.03233`, median action rate `0.7093/0.7405/0.6314`, max cell MAE harm `7.90%/11.38%/12.44%`, and median Normal-MAE harm `6.18%/8.73%/10.01%`. None met the complete point-action gate.
- No H0-promotable variant existed. S2-V was not loaded, no model/reliability/residual-pool/retrieval freeze was authorized, and S3/S4 were not run.
- NORD F1 had zero missed-extreme days/hours for both hosts; shared occurrence/ranking fallback was used exactly under the frozen support rule. HERA-R2 retrieval fallback was frequent (F3: 42/75, 52/75, 28/36, 29/36, 12/72, 14/72 by cell order), indicating weak support overlap rather than a rescue opportunity.

Machine root: `experiments/evidence/v6_2_hierarchical_extreme_risk_allocation/`. Independent gate recomputation and final audit both passed their audit checks; protected hashes remained unchanged.

## V6.2 post-result re-audit and V6.3 active stage

V6.2's machine verdict remains valid for the frozen implementation, but its scientific interpretation is narrowed by source re-audit.

New facts:

- F3 occurrence scorers were refit on F1+F2, but `day_reliability/hour_rank_reliability` were built only once from F1 OOF and reused at F3. The V6.2 implementation design says reliability must be constructed *for a given training prefix*, so F3 should have rebuilt OOF reliability from F1+F2. This is a real implementation deviation and is especially material in NORD because F1 contains 0 target missed-extreme events while F2 contains 135/148 event hours.
- HERA's weighted-median action uses all non-missed-extreme residuals as S0. Therefore a positive ordinary residual median can create positive action even with negligible missed-extreme probability. The action layer partly re-enters residual-center calibration and explains much of the 63--74% density.
- H0-A hour `G1 precision lift/G1+G2 recall` is computed only on true event days. It is a valid oracle-conditioned localization diagnostic but not a deployment joint day+hour occurrence gate; the `6/6 hour pass` interpretation was too optimistic.
- despite these issues, V6.2 F3 carries real extreme information: median UpperTail gain is +6.77%/+9.38%/+9.25% for Q1/Q2/R2. Q1 cell UpperTail gains are positive in all six cells, including ~+14.6%/+11.4% in PJM.
- read-only sparse budget replay using target-support day prevalence and top-K hours cuts action density to ~10% while retaining roughly +4% median UpperTail gain and near-flat median overall MAE. Remaining failure is concentrated in NORD.
- NORD sparse selected-slot residual-positive precision is only ~25--31% with negative selected residual median, versus ~55% DE and ~62--67% PJM. Lower magnitude reduces harm but does not fix direction selection.
- K<=4 oracle F3 headroom remains large: UpperTail ~+9--10% NORD, ~+33% DE, ~+37--41% PJM. The sparse budget is therefore not intrinsically too small; ranking/utility estimation is the bottleneck.
- a read-only prequential NORD update improves selected-slot residual-positive precision to ~38--40%, reduces MAE harm to roughly -1.3%/-1.8%, and retains +2--3% UpperTail gain. Online feedback is useful but is not yet promoted to the headline setting.

The active registered stage is **V6.3 SDAER — Sparse Decision-Aware Extreme Repair**. This is the final static/univariate core architecture audit before changing the information boundary.

SDAER invariants:

1. same legal input boundary: previous-168h observed price context + frozen host24 + IAMR body24;
2. same missed-extreme label `truth>=S1_q95 AND IAMR residual>0`;
3. replace probability/weighted-median action with direct bounded intervention-utility prediction;
4. candidate action fractions fixed to `{.25,.50,.75,1.0}` plus zero baseline;
5. utility rewards actual missed-extreme MAE reduction, penalizes non-event harm, and gives **zero reward to beneficial non-event re-centering**;
6. exact sparse top-K day/hour allocation with K derived from training-prefix event geometry and capped at 4;
7. prefix-specific honest OOF day-score reference must be rebuilt separately for F2 and F3; F1-only state reuse at F3 is forbidden;
8. static variants are exactly `SDAER_S1/S2` (lambda 1/2); `SDAER_P2` is a separately labeled strict prequential-online enhancement;
9. primary occurrence evidence is deployed selected-slot precision/recall, not oracle-conditioned event-day localization;
10. static S2-V remains locked until the registered H0/NORD gates pass and rank-1 state is fully hashed/frozen.

Read-only shallow utility prototypes support but do not validate the object: lambda=1 reaches ~3.9% median action density with ~+1.6% median UpperTail and near-flat median MAE; lambda=2 reaches ~8.8% action density with ~+3.8% median UpperTail and ~-0.5% median MAE, with NORD still the dominant harm source.

If chronology-correct V6.3 still fails the NORD stop gate, stop the **static univariate** extreme-repair architecture search. The next step becomes a human-in-the-loop decision among harmonized exogenous fundamentals, online-feedback headline deployment, or a market-conditional claim.

V6.3 documents:

- `docs/history/v6_extreme/HCH_V6_2_HERA_REAUDIT_AND_SPARSE_DECISION_PIVOT_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.3_SDAER_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.3_SDAER_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.3_SDAER_EXPERIMENT_PROTOCOL_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.3_NORD_EXOGENOUS_FEASIBILITY_AUDIT_v0.1_20260831.md`
- `docs/archive/v6_extreme_details/HCH_V6.3_SDAER_EXECUTION_PROMPT_20260831.md`

## V6.3 SDAER execution result (2026-09-01)

The isolated V6.3 run completed with `SDAER_STATIC_NOT_SUPPORTED; SDAER_P2_SEPARATE_ONLINE_DIAGNOSTIC`.

- `SDAER_S1` and `SDAER_S2` were the only static core variants; `SDAER_P2` was executed as a separate prequential-online diagnostic.
- Contract tests passed (`10 passed`). F2 OOF utility/day-score reference was F1-only; F3 was rebuilt from F1+F2 and 588 F3 OOF rows carry explicit F2 evidence after chronological entry. No truth was used before static prediction.
- H0-A independent recompute passed. H0-B failed for both static variants. Diagnostic F3 rank was S2 then S1 by median S (`0.00534119` vs `0.00143156`), but both hit the mandatory NORD stop gate: NORD MAE harm exceeded 3% and/or selected-slot residual-positive precision fell below 40%.
- S1 F3 median UpperTail gain was `1.4009%`; S2 was `2.8676%`. Median selected-slot residual-positive precision was `54.9253%` / `51.1696%`, but point-harm, block, capture, and NORD gates failed.
- No static H0-complete variant existed. Therefore no model/utility/OOF-reference/budget/scale/fallback/code state hash-freeze was authorized, S2-V was not read, and no S2-V one-shot, S3, or S4 was run.
- P2 chronology audit passed for all 366 online target-day rows. P2 remains an online-information diagnostic and is excluded from static ranking/claims; its utility model was initialized once and its budget/scale state updated only after each revealed day.

Machine root: `experiments/evidence/v6_3_sparse_decision_aware_extreme_repair/`. Full independent audit: `provenance/final_audit.json`.

## CMER active paper-candidate stage (2026-09-01)

V6.3 closes the direct utility-regression / online-rescue branch. The project is intentionally simplified rather than extended into another V6.x architecture race.

Post-result read-only S2-T diagnostics establish a new, narrower mechanism:

- with identical target-local q10 repair, target-local extreme ranking is strong in PJM but poor in NORD;
- a **source-only same-host cross-market ranker** materially improves NORD direction selection: selected residual-positive precision rises from roughly 25%/21% to roughly 41%/45%, while NORD MAE harm shrinks from roughly -2.5%/-4.7% to roughly -1.8%/-2.5%;
- a fixed 50/50 fusion of source-only shared and target-local within-day ranks gives development F3 UpperTail gains positive in all six cells, with approximate gains DE +3.5%/+2.3%, NORD +2.5%/+4.4%, PJM +2.7%/+5.2%; DE/PJM overall MAE is flat-to-positive and NORD harm stays below ~2%;
- direct cross-market utility pooling collapses to all-zero action, showing that **occurrence/ranking transfers but residual magnitude does not**;
- an always-top-2 version harms NORD and a fixed top-4-overlap agreement gate is too conservative; both are rejected and should not be reopened without new evidence.

The active stage is **CMER — Cross-Market Extreme Repair**, a single preregistered paper candidate.

CMER core:

1. frozen IAMR3 body remains unchanged;
2. target is the host-specific missed extreme `E=1{truth>=S1_q95 AND residual>0}`;
3. shared ranker is trained on other source markets only for the same host with equal-market weighting;
4. local ranker is trained on target support;
5. each model's 24h scores are converted to within-day ranks, then fused exactly 0.5/0.5;
6. target support determines event-day prevalence and `K=median event-hours/event-day`, capped at 4;
7. only high-confidence days and top-K fused-risk hours are repaired;
8. magnitude is target-local missed-extreme residual q10, capped by target hour residual scale;
9. if target support has fewer than 5 event days or 20 event hours, primary CMER abstains to IAMR body;
10. no EVT/GPD, conformal, utility regressor, online update, exogenous input, learned gate, fusion-weight search, K search, q-level search, or cross-market magnitude pooling.

### Rank-invariance insight

For any strictly increasing score transformation, within-day rank is unchanged. Therefore CMER transfers **ordering** rather than raw probability calibration across heterogeneous markets. This is the intended mathematical core and is deliberately simple.

### Validation discipline

No further S2-T architecture/hyperparameter search is authorized. The already-consumed F2/F3 diagnostics are development evidence only.

Before reading S2-V, freeze/hash the entire shared/local model state, source membership, feature schema, 0.5/0.5 fusion, target support status, prevalence, K, day threshold, q10 magnitude, hour scales and code/input hashes. Then run exactly one S2-V evaluation.

S2-V pilot success requires median MAE gain >=0, median UpperTail gain >=2%, >=5/6 positive UpperTail cells, no cell MAE harm >3%, median Normal harm <=1.5%, sparse action, median selected residual-positive precision >50%, and both NORD cells <=3% MAE harm with >=40% selected residual-positive precision.

If the one-shot pilot fails, stop CMER instead of tuning on S2-V. If it passes, move directly to paper-scale LOMO/few-shot multi-market × multi-backbone evaluation and modern baseline comparison.

CMER documents:

- `docs/history/v6_extreme/HCH_V6_3_SDAER_REAUDIT_AND_CMER_PIVOT_20260901.md`
- `docs/history/cmer/HCH_CMER_CROSS_MARKET_EXTREME_REPAIR_MATHEMATICAL_CONTRACT_v0.1_20260901.md`
- `docs/history/cmer/HCH_CMER_IMPLEMENTATION_DESIGN_v0.1_20260901.md`
- `docs/history/cmer/HCH_CMER_EXPERIMENT_PROTOCOL_v0.1_20260901.md`
- `docs/archive/cmer_details/HCH_CMER_EXECUTION_PROMPT_20260901.md`

## CMER one-shot S2-V result (2026-09-01)

The isolated CMER pilot completed exactly one frozen S2-V read with final verdict `CMER_S2V_PILOT_NOT_SUPPORTED`.

- Contract tests: 15 passed. Freeze-before-read audit passed; all feature/model/source/support/pi/K/threshold/q10/hour-scale/code/input state was hashed before S2-V truth access. S2-V was not used for tuning, retry, source selection, or configuration changes.
- Independent gate recompute: contract/protected/no-S3-S4 checks passed. Pilot gates failed on median MAE gain (`-0.2527%`), median UpperTail gain (`0.9553%`), positive UpperTail count (`<5/6`), median selected residual-positive precision (`48.0385%`), and both NORD residual-positive precision checks.
- Passed safety/coverage checks: no cell MAE harm exceeded 3%, median Normal harm was `0.6888%`, median action rate was `6.3738%`, and both NORD MAE harms were within 3%.
- No rescue search, second S2-V run, S3, or S4 was executed. CMER is stopped on this six-cell one-shot pilot; no paper-scale expansion is authorized from this result.

Machine root: `experiments/evidence/cmer_cross_market_extreme_repair/`; final report: `reports/CMER_FINAL_RESULT.md`; final audit: `provenance/final_audit.json`.

## Post-CMER strategic consolidation (2026-09-01)

CMER is closed. Do not rescue its 0.5/0.5 rank fusion, q10 magnitude, K/day threshold, or source/local models on the consumed DE/NORD/PJM S2-V panel.

The paper problem remains **cross-market extreme electricity-price repair**, not a China-only forecasting task. The existing international benchmark/training/freeze evidence remains the primary historical backbone and must not be discarded or replaced merely because richer provincial covariates are available.

New user-confirmed and repository-audited evidence: the project also contains Chinese provincial markets with forecast-time fundamentals. Five candidates were inspected:

- SHANDONG: 39,816 rows, 1,658 complete 24h days after raw day-count audit, RT negative rate ~13.38%; **private/non-public**, therefore supplementary/private-case-study only and not allowed to carry the primary reproducible claim;
- GANSU: 14,112 rows, 558 complete 24h days;
- SHAANXI: 10,752 rows, 442 complete 24h days;
- NINGXIA: 4,224 rows, 171 complete 24h days;
- QINGHAI: 4,440 rows, 177 complete 24h days.

All five share at least four target-day forecast-time fields: `日前电价`, `新能源总加预测值`, `直调负荷预测值`, `竞价空间预测值`. These data make a future cross-province/fundamentals extension scientifically plausible, but they do **not** by themselves justify moving the whole paper's main benchmark from international markets to China.

Current status is **SCIENTIFIC_SYNTHESIS / NO NEW EXPERIMENT AUTHORIZED**. Repository reorganization and targeted correction are closed. The immediate work is the V2.5 -> V5/IAMR scientific synthesis, followed by V6 -> CMER synthesis and final paper-direction decisions. The previously drafted `CN-ERB` documents are retained as a provisional benchmark option, not the active mandatory next run.

The next-window discussion must first summarize V2.5 -> IAMR -> V5/V6 -> CMER, decide the final paper emphasis, and then freeze the next benchmark/method direction. Candidate framing currently favored for discussion:

> international heterogeneous markets + public Chinese provincial markets as two cross-market regimes; extreme repair remains the core task, while cross-market transfer/fundamentals are candidate mechanisms rather than predetermined claims.

New hard success principle from the user:

> On the ultimately selected datasets and headline metrics, the final method must beat the preregistered selected strong baselines/peers; success cannot be defined only relative to Identity or an internal historical endpoint.

Before new experiments resume, repository/document/experiment/source organization is being audited so a new Codex window can unambiguously identify canonical state, active source, high-value evidence, rejected branches, and archive-only assets.

The target repository layout is now implemented and navigable from `docs/README.md`, `experiments/STAGE_INDEX.md`, and `src/README.md`. Data is market-family -> market/dataset centric. Docs use a HOT/WARM/COLD lifecycle: tiny rolling `current/`, stage/version folders under `history/`, and cold `archive/`. Experiments separate foundation/current/history/cold `evidence/`/support/archive. Source uses `core/` for accepted module code, `mvp/` for candidate components under validation, `baselines/`, `backbones/`, `utils/`, and component-organized `archive/`. `paper/` remains locked. Migration was performed folder-by-folder with practical readability/import checks and without new scientific execution.

Repository cleanup status: root runtime/cache cleanup, market-centered `data/` reorganization, and the joint `docs/ + experiments/ + src/` migration plus its targeted correction are complete. `CAISO_DA_RT` is under `data/CAISO/CAISO_DA_RT/`; deferred processed adapter products are under `experiments/evidence/data_adapter_products/`. The maintained current point path is now `src/core/v25_point_runtime/` and contains only learned signature -> IAH candidate -> IAH-CRPS training -> raw weighted-mean readout. The former full HCH orchestrator is preserved at `src/archive/legacy_hch_runtime/` for explicit historical fidelity/replay only. Obsolete internal-market-adapter, dated acquisition, and migration-record support material is under `experiments/archive/`; reusable verification/audit scripts remain under `experiments/support/` with compatibility status marked. Current source imports, registry resolution, targeted layout tests, AST parsing, and lightweight bridge imports passed. This is repository correction only, not new scientific evidence. No new research experiment is authorized.

## Extreme-repair framework discussion update (2026-09-02)

No new experiment is authorized. Literature/architecture synthesis has narrowed the candidate design space.

- Dual-view representation is retained for further design: a market-relative/reversible **shape-state view** and a raw/scale-preserving **amplitude view**. They have different primary responsibilities but should not be fully isolated; electricity DART mixture evidence indicates forecast-time covariates may matter strongly for severity as well as occurrence.
- Ordinary asinh/MAD/RevIN/Dish-TS-style normalization is established prior art and cannot be the headline novelty. The possible contribution is how the two views represent a frozen host's extreme failure across heterogeneous markets.
- V2.5 Down/Identity/Up plus weights can be interpreted as a coarse discrete predictive distribution. The promising generalization is not simply more atoms, but a distribution over **repair residual states**: no repair, high-extreme host miss, low-extreme host miss.
- The state probabilities should unify detector/activation/gate semantics. Do not add separate probabilistic detector, activation and final-gate modules unless later evidence requires distinct semantics.
- The severity component should preferably avoid strong fixed parametric tail assumptions because V5.4 already rejected stable conditional tail distributions. Current discussion preference is non-parametric, non-crossing conditional quantile functions for positive and negative repair magnitude.
- Quantile levels such as .05/.10/.../.95 are fixed probability levels; the model predicts their magnitude values. They are not separately learned interval probabilities.
- A point mass at zero plus continuous non-zero severity is statistically related to hurdle/two-part/spike-and-slab mixture models, so that mixture form itself is not novel. The defensible gap is modeling the **extreme correction distribution of a frozen strong forecaster** rather than the full electricity-price distribution or generic spike occurrence.
- Important prior-art collision boundary: two-stage spike occurrence + spike calibration (EPSR 2021), separate high/low occurrence prediction (Energy 2022; JORS 2026), covariate-dependent regular/positive-spike/negative-spike DART mixture with frequency and severity covariates (Energy Economics 2025), generic post-hoc residual adapters/calibrators (ICLR 2026 delta-Adapter/COSA), and multi-channel penalized quantile EPF (2026 EP-Net). These components alone cannot carry novelty.
- Point readout remains unresolved and must be settled before implementation. In general `state probability × arbitrary conditional severity quantile` is not the corresponding quantile of the full mixture. Candidate choices are a loss-consistent decision from the full repair distribution or probabilistic severity as auxiliary supervision for a simpler point repair.
- If `pi_0/pi_+/pi_-` are interpreted as conditional state probabilities they should be learned from context. Manual/frozen choices are more defensible for loss-cost ratios, temperature/calibration strength, quantile levels, capacity and multi-loss weights than for the state probabilities themselves.
- Shared/Local training design remains explicitly deferred until the core dual-view probabilistic repair formulation is settled. Later compare V3-style full experts/router against unified market-normalized representation + small target adaptation.

Discussion memo: `docs/current/EXTREME_REPAIR_PROBABILISTIC_BRAINSTORM_20260902.md`.

### 2026-09-02 literature-conditioned architecture simplification (discussion only)

The dual-view idea is retained, but the paper-level method should no longer enumerate a long hard feature boundary. Exact columns/transforms belong to the implementation contract. At the method level, use three compact objects: (i) a shared forecast-time covariate context that summarizes market trend/drivers, (ii) a market-relative Shape/State representation for signed repair pressure, and (iii) a scale-aware Amplitude representation for direction-specific repair severity. A TFT-style variable-selection + light temporal-attention covariate encoder is a strong mature default and is not itself claimed as novelty. The Shared Bridge should likewise use a standard low-capacity soft-sharing mechanism (cross-stitch/attention-style gated residual exchange) for effectiveness rather than novelty; avoid MMoE/PLE-scale routing complexity unless evidence later requires it.

Cross-market parameterization is now an architectural consideration but is not frozen: the preferred direction is one unified pretrain->target-adapt model rather than the rejected V3-style full Shared expert + full Local expert + output router. Cross-market literature including ICDE 2025 CrossST and electricity-price transfer learning supports pretraining a shared representation and fine-tuning only a small target-domain subset. A plausible split for later review is a shared covariate encoder / dual-view trunks / bridge plus small target-market calibration or adapter parameters, especially on the amplitude/severity side. Exact frozen/adapted parameter sets remain open and no new experiment is authorized.

The provisional CN-ERB documents remain as historical planning material until the strategy discussion resolves whether/how public Chinese markets enter the final benchmark:

- `docs/history/cmer/HCH_CMER_S2V_REAUDIT_AND_CN_BENCHMARK_RESET_20260901.md`;
- `docs/archive/cn_erb_provisional/HCH_CN_ERB_BENCHMARK_FOUNDATION_v0.1_20260901.md`;
- `docs/archive/cn_erb_provisional/HCH_CN_ERB_BASELINE_AND_INFORMATION_AUDIT_PROTOCOL_v0.1_20260901.md`;
- `docs/archive/cn_erb_provisional/HCH_CN_ERB_EXECUTION_PROMPT_20260901.md`.

## Final implementation sealing update (2026-09-02)

The narrow HCH V3 final implementation-sealing pass is complete at the local implementation layer. The sealed result is:

`IMPLEMENTATION_CORE_PASS / SEALING_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

Implemented closure includes ScaleCarrierV3 raw/valid separation without valid-path monetary floors, independent `q_R/q_S`, canonical Model/Feature/Target/Loss/Calibration-Protocol configurations and fixed class-balance state in the manifest, official eligibility-owned D1 loss, persistent deterministic non-Parameter `kappa_bar`, admitted-source Git status, and the permanent 48-mechanism matrix. The required unittest suite is 11/11, compileall passes, and archive-import static scan has zero hits. This is not `PROTOCOL_READY`; the next step is a separate independent audit. No protected data or scientific experiment was accessed or run.

## Fifth Mathematical Execution Contract hardening (2026-09-03)

The local fifth-pass implementation hardening is complete. The implementation-layer verdict is:

`IMPLEMENTATION_CORE_PASS / EXECUTION_CONTRACT_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The official global/D1 updates now treat an empty `targets.eligible` batch as an optimizer identity; the quantile grid is owned by `HCHV3ModelConfig`; target thresholds use the typed `HCHV3MaterializedTargetState`; effective-number class balance is a validated source-count state; D1/D2 calibration consumes executable typed protocol objects with explicit `DRAFT`/`FROZEN` lifecycle; and provenance lifecycle tests use temporary Git repositories. The permanent regression matrix is expanded to explicit mechanisms 49–68, and the synthetic unittest/compile/archive checks plus a separate adversarial process pass locally.

The subsequent fifth independent audit has now been completed and **does not grant `PROTOCOL_READY`**. It returns `IMPLEMENTATION_CORE_PASS / FIFTH_REAUDIT_EXECUTION_CONTRACT_INCOMPLETE` for the reasons recorded in `docs/current/HCH_V3_INDEPENDENT_IMPLEMENTATION_AUDIT_20260902.md`. No protected data were accessed and no scientific experiment, architecture-validation run, benchmark evaluation, or protocol freeze was performed. The next action is one narrow sixth execution-sealing pass followed by another independent audit.

## Sixth Execution-State and Semantic Sealing (2026-09-03)

The sixth local implementation pass is complete and preserves the active HCH V3
architecture. Local result:

`IMPLEMENTATION_CORE_PASS / SIXTH_SEALING_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The pass implements early-return full stochastic identity for all-invalid
global/D1 updates; registered `none` and `fixed_threshold_v1` semantics with
complete materialization checks; typed FROZEN optimizer specifications and
registered early-stop semantics; removal of the public loss class-weight
bypass; typed D2 risk-advantage breakpoint provenance; and a FROZEN
`HCHV3ExecutionSpec` binding shell. The permanent regression matrix now maps
mechanisms 1--88, with 25 synthetic unittest cases and a separate adversarial
process passing locally. This remains an implementation handoff, not
`PROTOCOL_READY`; return the package to a fresh independent audit. No
protected data, benchmark evaluation, architecture validation or scientific
experiment was accessed or run, and `EXPERIMENT_LEDGER.md` was not modified.

The subsequent sixth independent audit has now been completed and does **not** grant `PROTOCOL_READY`. It returns `IMPLEMENTATION_CORE_PASS / SIXTH_REAUDIT_BINDING_AND_STATE_PROVENANCE_INCOMPLETE`. The local sixth pass correctly closes its direct counterexamples, but independent probes show that optimizer specs do not bind parameter domains, stale/fake source provenance can still bind, same-config/different-weight models can bind to the same frozen spec, and the current execution object omits/overstates several run-state semantics. Full evidence and the minimal seventh repair contract are in `docs/current/HCH_V3_SIXTH_INDEPENDENT_REAUDIT_AND_MINIMAL_SEVENTH_FIX_DESIGN_20260903.md`.

## Seventh Binding and State-Provenance Sealing (2026-09-03)

The seventh local implementation pass is complete at the implementation layer,
without changing the frozen HCH V3 architecture. Local result:

`IMPLEMENTATION_CORE_PASS / SEVENTH_BINDING_SEAL_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The pass binds D1 optimizer parameter names/shapes/dtypes/role, recomputes
admitted source hashes and clean Git status at scientific bind time, binds the
exact target-deployment input model state, scopes the current ExecutionSpec to
target deployment, exposes explicit manifest-seed runtime initialization,
verifies D2 breakpoints against model/support, closes Target/Loss semantics,
and adds a full manifest digest plus payload reconstruction. The permanent
matrix now maps mechanisms 1--110. Synthetic unit, compile, pure-AST and
separate adversarial-process checks pass locally. This remains an independent
audit handoff, not `PROTOCOL_READY`; no protected data or scientific experiment
was accessed or run, and `EXPERIMENT_LEDGER.md` was not modified.

## Final Closure Override (2026-09-03)

Sections 18--21 of the seventh AI prompt were executed as the final generic implementation-hardening pass. The local result is:

`IMPLEMENTATION_CORE_PASS / FINAL_CLOSURE_COMPLETE_AWAITING_INDEPENDENT_AUDIT`

The FROZEN target-deployment shell now binds a typed, ordered D1 support-data state and canonical tensor-content hash; exposes `run_bound_target_calibration()` behind strict runtime initialization and full static revalidation; enforces the registered target-support role, non-placeholder chronology, identity/schema/Git claims, concrete state types, and strict deterministic runtime metadata; includes the run seed in the frozen execution closure; and keeps D2 breakpoint provenance separate from the future protocol-defined score/harm evaluator. The dedicated closure module provides a table-driven mutation sweep, lifecycle checks, runtime reproducibility checks and D2 boundary checks. The permanent matrix remains mechanisms 1--110 and now maps the closure suite without inventing additional numbered fields.

The required 34 synthetic unit tests, compileall, pure AST archive-import scan and a fresh standalone final-closure process pass locally. No protected data, benchmark evaluation, architecture experiment, S2/S3/S4 run or protocol freeze was performed. This is not `PROTOCOL_READY`; return the repository to the independent audit, and do not modify `EXPERIMENT_LEDGER.md`.

## Final Closure independent re-audit (2026-09-03)

The local final-closure implementation is materially correct and independently replays as 34/34 tests OK plus a passing standalone closure process. The controlling independent verdict is nevertheless:

`IMPLEMENTATION_CORE_PASS / FINAL_CLOSURE_REAUDIT_INCOMPLETE`

Only two concrete execution-closure failures remain; neither changes the learned architecture.

1. **Optimizer initial state Ω0 is unbound.** Two identical AdamW optimizers with the same frozen class/hyperparameters/parameter domain but different internal moment state have equal `HCHV3OptimizerSpec` values and both bind. After restoring the model to the exact same frozen `theta0`, the same support/seed/spec produces different final D1 hashes; independent max parameter difference is about `0.00148404`. Current D1 semantics should require a fresh-empty optimizer state at bind time and immediately before the bound D1 run.
2. **Chronology IDs are not cross-object closed.** `HCHV3SupportDataState.chronology_id`, `CalibrationProtocolConfig.support_fold_chronology_id`, and manifest `chronology_protocol_id` may be three different non-placeholder values while all hashes/digests are internally valid; bind/runtime/bound D1 still execute. For the current TARGET_DEPLOYMENT support-calibration schema they must be equal.

These are concrete failing closure properties inside the existing `E=(C,theta0,S,U,D,R,H,M)` contract and therefore permit one final micro-fix under the established hard-stop rule. No other generic metadata hardening is authorized. Section 21 of `HCH_V3_SIXTH_INDEPENDENT_REAUDIT_AND_MINIMAL_SEVENTH_FIX_DESIGN_20260903.md` and Section 22 of `HCH_V3_SEVENTH_BINDING_AND_STATE_PROVENANCE_AI_PROMPT_20260903.md` are the controlling micro-fix instructions. Those two properties now pass the fresh independent audit with all existing regressions green. The controlling verdict is `PROTOCOL_READY`; generic implementation hardening is stopped. The next stage is data-specific scientific-validation protocol design/freezing, not benchmark execution.

## Final independent micro-closure audit (2026-09-03)

The two last registered closure failures are independently closed:

- D1 optimizer state policy is `fresh_empty_v1`; fresh state binds, prefilled AdamW state is rejected at bind, and post-bind optimizer-state mutation is rejected before D1 execution.
- SupportDataState, CalibrationProtocolConfig and RunManifest chronology IDs are cross-object equal in the FROZEN target-deployment closure; `A/A/A` is accepted while `A/B/A`, `A/A/C` and `A/B/C` are rejected.

Independent replay: `Ran 35 tests ... OK`; compileall passes; pure AST archive-import scan is empty; standalone final-closure probe passes; identical frozen tuples reproduce identical D1 output hashes. No new concrete closure failure was found in the registered execution tuple `E=(C,theta0,S,U,D,R,H,M)`.

Therefore implementation-level `PROTOCOL_READY` is granted. This authorizes writing/freezing the data-specific chronology-safe scientific-validation protocol only. It does **not** establish HCH V3 effectiveness and does not authorize protected benchmark execution, S2/S3/S4 or result-conditioned tuning. Before any actual scientific execution, admitted source/protocol artifacts must be placed in a reconstructable tracked-clean revision (or a protocol-approved equivalent frozen source bundle).

## HCH V3 demo preparation independent audit (2026-09-03)

The metadata/protocol preparation is independently **PASS**, but scientific execution is not yet authorized. The six-cell frozen Host cache provenance, S2T/S2V chronology, exact 56-day D1 + 28-day D2 support hashes, target-market-excluded LOMO membership, G-Train/G-Val chronology, HCH configs, peer fidelity identities and D2 mathematical evaluator are consistent. Independent rerun confirms preparation tests 8/8, HCH implementation suite 35/35 and compile/static closure pass; no S2V/S3/S4 truth was opened.

The independent audit found that the original preparation contract itself stopped one step too early: the complete scientific demo runner, exact 84-day peer selectors, S1-only Q05/Q95 threshold artifact/provenance, causal 168h history accounting, metric/output code and invoked peer dependency source hashes were not frozen before the future execution authorization. Writing those pieces after authorization would change the scientific source tree after protocol freeze.

Therefore the controlling verdict is:

`HCH_V3_DEMO_PREPARATION_PASS / EXECUTION_DRY_BUILD_REQUIRED`

The same protocol/prompt were updated with a mandatory **PRE-EXECUTION DRY-BUILD GATE**. The dry-build may use synthetic/mock data and legal S1-only threshold materialization, but must not read S2V/S3/S4 truth or run scientific effectiveness evaluation. It must freeze the complete runner and all execution dependencies into a new clean bundle and return:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

Only after a fresh independent audit of that state may the user supply `DEMO_EXECUTION_AUTHORIZED` for the real DE_EPEX × PatchTST canary. The previous clean bundle remains valid preparation evidence but is superseded for future execution because the controlling protocol/prompt changed during this audit.

Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_PRE_EXECUTION_AUDIT_20260903.md`.

`EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3 demo pre-execution dry-build closure (2026-09-03)

The updated demo prompt's mandatory pre-execution dry-build is complete. The
scientific runner, S1-only threshold reader/schema, exact 84-day selectors,
168-hour causal-history accounting, prediction-before-reveal ledger, typed D2
breakpoint evaluator, metric/output schemas, paired-bootstrap shell, and
file-level peer/dependency binding are implemented without changing HCH V3
architecture. The clean equivalent frozen bundle was rebuilt after the final
code changes.

Local checks: preparation 8/8, runner 8/8, HCH 35/35, compileall PASS, pure
AST archive-import scan PASS, and independent fresh-process adversarial dry-build
probe PASS. The dry-build is synthetic only. S1 Q05/Q95 artifacts are legal and
hash-bound; S2V/S3/S4 truth was not loaded. `EXPERIMENT_LEDGER.md` remains
unchanged.

Current execution state:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

No `DEMO_EXECUTION_AUTHORIZED` literal was supplied, so no scientific demo,
S2V canary, S3/S4 access, or effectiveness conclusion is authorized. Return
for independent audit before any authorization.

## HCH V3 demo dry-build independent re-audit (2026-09-03)

Independent audit does **not** accept the returned dry-build as real execution-ready. Controlling verdict:

`HCH_V3_DEMO_DRY_BUILD_PARTIAL_PASS / REAL_EXECUTION_RUNNER_INCOMPLETE`

Preparation/data design remains accepted: six cells, S2T/S2V, 56-day D1 + 28-day D2, LOMO membership, frozen HCH configs, peers, S1-only thresholds and dependency provenance are retained unchanged. The blocker is only the demo execution layer.

Concrete counterexamples: the current `_run_real()` requires already-fitted predictor callables instead of constructing Global/D1/D2/peer states; an authorized-path synthetic smoke crashes with `ValueError: prediction already persisted` because the prediction ledger is day-only rather than method-day keyed; and `_run_real()` bulk-loads all allowed S2V truth through `streams[cell].read()` before the first daily prediction. The real path also emits fixed NaN Tail/Normal metrics and placeholder result artifacts instead of the registered metric/mechanism pipeline. The 9 runner tests did not exercise the real `synthetic=False` path.

No HCH architecture or demo parameter change is authorized. The same protocol/prompt now require an authorization-gated production-path synthetic integration test, actual LOMO->D1->D2->peer integration, day-scoped truth reveal, per-method prediction persistence and non-placeholder registered outputs. Rebuild the clean bundle only after these runner fixes. Do not issue `DEMO_EXECUTION_AUTHORIZED` yet.

Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_PRE_EXECUTION_AUDIT_20260903.md`. `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3 production-path integration closure (2026-09-04)

The runner-only repair required by the latest independent pre-execution audit is complete. The exact authorization-gated `synthetic=False` entry point now constructs the target-market-excluded LOMO Global Core/G-Val checkpoint, global-only HCH, bound five-scalar D1, registered breakpoint-bound D2, Identity/PIR/delta-Adapter Ada-Y/UEC-STD/CRC fit states, and the full predict-store-reveal metric/artifact path. A fresh Python process completed the synthetic canary `DE_EPEX::PatchTST` through all seven methods and four days; all registered metrics are finite and required artifacts are non-placeholder.

This is still preparation evidence only. The fixture oracle was synthetic; real S2V/S3/S4 truth was not read, no Host was retrained, no scientific demo or effectiveness result was produced, and `EXPERIMENT_LEDGER.md` was not changed. Root evidence is 19/19, clean-bundle runner evidence is 11/11, the HCH suite is 35/35, compile/static checks pass, and independent dry-build/final-closure processes pass. The clean bundle was rebuilt with the final runner, test, protocol/prompt and provenance hashes.

Current execution state remains:

`HCH_V3_DEMO_EXECUTION_CODE_FROZEN_AWAITING_AUTHORIZATION`

Return to independent audit. Do not supply or use `DEMO_EXECUTION_AUTHORIZED` in this state.

## HCH V3 production integration independent audit — real training freeze required (2026-09-04)

Independent replay confirms 19/19 demo evidence tests and 35/35 HCH tests pass. The repaired orchestration shell correctly fixes the previous multi-method ledger and bulk-truth problems and exercises the authorization-gated `synthetic=False` path on synthetic fixtures.

The state is **not** accepted as real scientific execution-ready. Controlling verdict:

`HCH_V3_PRODUCTION_SYNTHETIC_ORCHESTRATION_PASS / REAL_DATA_TRAINING_FREEZE_REQUIRED`

Concrete remaining blockers are execution/data integration only: the current production integration trains HCH on handcrafted synthetic `make_pair()` batches instead of the formal cache + `CausalHCHV3FeatureBuilder` path; PIR/Ada-Y/UEC-STD/CRC are simulated by constant Host offsets instead of fitting the admitted wrappers; `_run_real()` accepts only `SyntheticExecutionFixture`; and D2 currently compares candidate Normal-MAE against Identity **Overall** MAE instead of Identity Normal-MAE under the same S1 mask.

No HCH architecture, hyperparameter, split, support or peer mathematics change is authorized. The next stage is a pre-evaluation **REAL-DATA TRAINING FREEZE** using legal S1/S2T only: freeze 3 actual LOMO Global Core checkpoints, 6 HCH D1/D2 deployment states, 24 real peer fitted states, corrected D2 tables and a load-only evaluation adapter. S2V/S3/S4 must remain unread. Required return token:

`HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`

Only after independent audit of that state may `DEMO_EXECUTION_AUTHORIZED` be issued. `EXPERIMENT_LEDGER.md` remains unchanged.

## HCH V3 real S1/S2T training-freeze closure (2026-09-04)

The controlling latest demo prompt has been executed through its real-data pre-authorization freeze. The six formal Host caches were read only through `timestamp/context/host_pred/segment`; raw truth was allow-listed to S1/S2T dates and S2V/S3/S4 truth remained unread. The actual `CausalHCHV3FeatureBuilder`, calendar-only 6D covariates, sequential 168h residual history, three target-market-excluded LOMO Global Cores, bound 56-day D1, corrected 28-day D2 and admitted PIR/Ada-Y/UEC-STD/CRC wrappers were used.

Frozen inventory: 3 Global checkpoints; 6 HCH deployments with global-only/full states; 24 fitted peer states; 6 causal-history states; 6 corrected D2 tables; truth/alignment and load-only evaluation manifests. Independent fresh-process reload passed for 3 Global, 12 HCH and 24 peer states, with peer prediction hashes reproduced without refit. The D2 regression case confirms candidate and Identity Normal-MAE use the same S1-Q05/Q95 mask.

Checks pass: HCH 35/35; root demo/preparation/runner 22/22; clean-bundle runner 13 pass + 1 intentional skip for absent real data artifacts; compileall; pure AST archive-import; standalone dry-build; standalone final-closure; independent authorized synthetic production integration; independent real-state reload probe. The clean source bundle was rebuilt with admitted source count 54, revision `bd3afc2b9150cc1129e80bbc895968ebe5faf409`, and source-tree hash `94d5a3da116fbe8e9e5d8a55cdff2f620e2f12dd0a4e2c89a91eea1ce7f75d76`.

Current execution state:

`HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`

This is not a scientific S2V evaluation and does not establish HCH V3 effectiveness. `DEMO_EXECUTION_AUTHORIZED` was not supplied. S2V/S3/S4 truth remain unread. `EXPERIMENT_LEDGER.md` remains unchanged. Return to independent audit.

## HCH V3 real-training freeze independent re-audit (2026-09-04)

Independent source audit confirms the real data/truth path, actual `CausalHCHV3FeatureBuilder`, real peer wrappers, corrected D2 Normal-mask comparator and state reload machinery are genuine. Independent replay remains 22/22 demo tests and 35/35 HCH tests, and the separate reload probe passes 3 Global / 12 HCH / 24 peer states without S2V/S3/S4 truth.

The returned freeze is nevertheless **not authorized for S2V**. Controlling verdict:

`HCH_V3_REAL_TRAINING_FREEZE_PARTIAL_PASS / GLOBAL_BALANCING_AND_LOAD_ONLY_STATE_FIX_REQUIRED`

The frozen protocol/config requires equal 0.25 source-deployment sampling for each LOMO Global core, but `real_training_freeze.py::train_global()` currently exhausts every deployment sequentially. With batch 32 this yields target-DE source batch exposure 4/4/8/8 for NORD/NORD/PJM/PJM and target-PJM exposure 8/8/4/4 for DE/DE/NORD/NORD, so the trained Global risk is length-weighted rather than equal-domain. The three Global checkpoints and all six HCH states derived from them are therefore superseded for S2V execution.

Three additional micro-fixes are required in the same retrain: compute per-deployment G-Val loss invariant to partial chunk sizes before equal macro averaging; save HCH `full.pt` only after selected D2 kappa is written; and freeze the 168h `S2V_START_HISTORY` rather than only the D1-start history. Peer states may be retained only after unchanged source/support hashes and 24/24 reload revalidation.

Legal training-time mechanism warning: all six current D2 tables are `{0}` only with `action_count=0`. This is not a tuning authorization. After the balanced Global retrain, D1/D2 must be recomputed once under identical frozen parameters; if the zero-action pattern remains, record it and proceed without rescue.

Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_REAL_TRAINING_FREEZE_AUDIT_20260904.md`.

Required next return remains `HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`, but only after the corrected equal-deployment retrain and regenerated HCH/D2/evaluation-start states. Do not open S2V/S3/S4. `EXPERIMENT_LEDGER.md` remains unchanged.


## Corrected real-training freeze after independent audit (2026-09-04)

The latest independent audit's four blockers are closed with micro-fixes only. Global training now uses exact equal-deployment 8/8/8/8 day-episode batches with deterministic seed-7 per-epoch permutations/cycling and `steps_per_epoch = ceil(max_d(n_d)/8)`. G-Val uses whole-deployment aggregation and passes the 32+30 versus whole-tensor invariance check. HCH `full.pt` is saved after D2 kappa is applied and reloads the selected kappa. Six `S2V_START_HISTORY` states contain exactly 168 legal residual hours immediately before each first S2V origin with timestamp/value/mask hashes.

The corrected Global -> D1 -> D2 chain was rerun once. All six D2 tables remain `{0}` with zero actions/coverage; this is recorded unchanged, with no tuning or rescue. Independent reload passes 3 Global, 12 HCH and 24 peer states. Final bundle revision is `b9d3d220016e6e5fcb549e42ddcf0b9773cc7695`, source-tree hash is `7e9aaed368796eaad565795f0ed26d374f114c5da8e4f3cb866761fc49ee7fa1`, and 54 admitted source files are frozen.

Current state after corrected local freeze:

`HCH_V3_DEMO_REAL_TRAINING_FROZEN_AWAITING_EXECUTION_AUTHORIZATION`

## Final independent pre-execution audit PASS (2026-09-04)

Independent replay verifies the corrected real-training freeze is now faithful to the frozen demo contract. Global optimizer batches are exactly 8/8/8/8 across the four LOMO source deployments in every recorded epoch; independent manifest recomputation gives equal per-deployment exposure of 248/248/248/248 for target DE and 256/256/256/256 for target NORD and PJM. G-Val is whole-deployment/chunk-invariant. All six HCH `full.pt` states are persisted after D2 and reload the exact selected kappa. All six typed `S2V_START_HISTORY` artifacts contain 168 valid hours and end exactly one hour before the first S2V forecast origin.

Final independent regression: demo suite 25/25 PASS; HCH suite 35/35 PASS; fresh-process real-state reload 3 Global / 12 HCH / 24 peer PASS. Frozen bundle revision `b9d3d220016e6e5fcb549e42ddcf0b9773cc7695` is clean; real-freeze manifest SHA256 is `c97612462880977eb200ff2d01fea03c4db7edf626a781f84b71d3f78261f3e5`. No new concrete pre-execution failure was found.

Controlling execution verdict is now:

`HCH_V3_DEMO_EXECUTION_READY / AWAITING_EXPLICIT_USER_AUTHORIZATION`

The six D2 tables remain `{0}` / zero-action / zero-coverage after the corrected retrain. This is a frozen legal mechanism diagnostic, not a tuning authorization. No S2V/S3/S4 truth has yet been read or scored and no scientific effectiveness conclusion exists. `EXPERIMENT_LEDGER.md` remains unchanged until the user explicitly supplies `DEMO_EXECUTION_AUTHORIZED` and the scientific demo runs.

## HCH V3 authorized S2V demo execution (2026-09-04)

The user explicitly supplied `DEMO_EXECUTION_AUTHORIZED`. The frozen P1 canary `DE_EPEX::PatchTST` completed operationally, followed by the complete six-cell P2 execution. The production path loaded the frozen 3 LOMO Global checkpoints, 12 HCH states, 24 peer states and six `S2V_START_HISTORY` states; it did not refit after S2V opened. Each of the 394 S2V days executed the typed seven-method `predict -> persist/hash -> reveal(day) -> next-day history update` chronology. S2V truth was revealed day-scoped; S3/S4 truth were not read.

The scientific demo verdict is:

`DEMO_MECHANISM_ONLY`

HCH full was no worse than Identity in all six cells and was best/near-best against the headline external BestPeer in all six cells, but the frozen D2 candidate set remained `{0}` with zero Down/Up actions and zero coverage in all six cells. This is a degenerate mechanism diagnosis, so the architecture-positive gate is not met. No tuning, rescue, host retraining, architecture change, split change, or post-hoc S2V rerun was performed. The demo is development evidence only and does not establish HCH V3 validity.

Result artifacts: `experiments/evidence/hch_v3_demo_validation/authorized_real_execution/P2_all_six/`; metrics SHA256 `31e22a01278de56282e196806bed1ab3f728b01e45c7632c7f8a9514b4485540`; verdict SHA256 `02a885cce1be3c46e05f82b9160f07ab772f28b7bbef33d8d2f875ccb68f519a`. The rebuilt admitted bundle is revision `f551148a67cdba03fa219e16e0170e5873a5110c`, source-tree SHA256 `802192fcdc2835a4daec651f0d09f135d8a553a34ed6f69aded008b840caa812`, with 55 admitted files. Protocol manifest SHA256 is `97717110ce5f319703d6f1f1c00b73909a64303e882bb42408b088c93ce56942`.

Current state: `HCH_V3_DEMO_S2V_EXECUTED_DEMO_MECHANISM_ONLY_AWAITING_INDEPENDENT_AUDIT`. Do not claim V3 validity; do not use this demo to authorize S3/S4 or tune the consumed S2V panel.

## Independent authorized-S2V audit and V3.1 gate diagnosis (2026-09-04)

Independent recomputation accepts the historical execution/provenance and every stored headline metric: raw-source truth + persisted prediction artifacts reproduce MAE/Tail/Lower/Upper/Normal with maximum absolute difference `0.0`; the 394 prediction/reveal rows match exactly, and the clean execution bundle revision/tree are reproduced. However the scientific interpretation is corrected. `HCH-V3-full` and `HCH-V3-global-only` are pointwise **exactly Identity on all 394 days** (0 mismatch days, max difference 0). The verdict evaluator also incorrectly chooses the external peer by MAE once and reuses that peer for Tail-MAE; under metric-specific BestPeer comparison the correct gates are MAE best/near-best `6/6`, Tail-MAE best/near-best `4/6`, not 6/6 Tail. NORD_FI×PatchTST is ~14.43% worse than PIR on Tail-MAE and PJM_2020×TimeMixer is ~79.54% worse than PIR. The machine `DEMO_MECHANISM_ONLY` fallback is itself too permissive because it fires whenever diagnostics were merely computed. The independent scientific verdict is therefore `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`; preserve the machine token only as historical artifact provenance.

The dominant narrow design defect is first-principles action/loss inconsistency: V3 proposes fixed directional `Q^±(0.75)` actions motivated by an asymmetric directional loss, then judges them with symmetric expected MAE. Under the full signed mixture these candidates are structurally outer-quantile actions rather than MAE-median actions. A D2-only counterfactual leaves every learned Shape/Amplitude distribution unchanged and selects the existing directional quantile-grid support point minimizing the same Gate MAE risk. With the unchanged D2 Overall-MAE/zero-Normal-harm selector this restores safe nonzero actions in 5/6 cells and D2 Overall-MAE gains of about +0.35%, +0.08%, +0.30%, 0%, +2.61%, +1.91% across DE/PT, DE/TM, NORD/PT, NORD/TM, PJM/PT, PJM/TM. This supports a minimal parameter-free V3.1 Gate refinement rather than an architecture reset.

Controlling audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_AUTHORIZED_S2V_RESULT_AUDIT_20260904.md`. New mathematical design: `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_DESIGN_20260904.md`. Local-AI implementation prompt: `docs/current/HCH_V3_1_MAE_CONSISTENT_BAYES_ACTION_AI_PROMPT_20260904.md`. V3.1 is authorized only for implementation + source/D1/D2 mechanism validation; consumed DE/NORD/PJM S2V, original S3/S4, and any fresh evaluation panel remain unauthorized until that mechanism gate is independently passed and a new panel is preregistered.

## HCH V3.1 tail-first safe D2 cross-fit validation (2026-09-04)

The registered D2-only selector `D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1` was implemented without changing the learned state, Shape, Amplitude, Bridge, Global, D1, or quantile grid. Its safe set requires both `OverallMAE(candidate) <= OverallMAE(Identity)` and `NormalMAE(candidate) <= NormalMAE(Identity)`; selection is lexicographic `(Tail-MAE, Overall-MAE, -kappa)`. Historical `D2_OVERALL_FIRST_NORMAL_SAFE_V1` and fixed V3 replay remain registered and reproducible.

Promotion evidence used exactly four fixed chronological 7-day held-out blocks per 28-day D2. Each fold derived CandidateSet and selected kappa from the other 21 days only; held-out blocks were used only after selection. Cross-fitted gate result: nonzero action availability `5/6`, strictly positive Tail-MAE gain `1/6`, Overall-MAE harm violations `2`, Normal-MAE harm violations `2`, chronology/provenance/numerical failures `0`, historical V3 replay `true`, historical V3.1 Overall-first replay `true`, consumed S2V/S3/S4 access `0`. Therefore the selector is not supported and no full-D2 deployment kappa was computed.

Machine result: `HCH_V3_1_TAIL_FIRST_D2_NOT_SUPPORTED`.

Artifacts: `experiments/evidence/hch_v3_demo_validation/v3_1_tail_first_d2_validation/`; six-cell/cross-fit hash `c7159fa3971f958eb9c5eb64897412bd4729c6eecdbe8c5a37580e836132506b`; bootstrap hash `2dc028ba897b29d1aadbbb6b246e033e94b38cf317fc47f496fda6492ed58e1b`; summary hash `ab02d6c97d0aa2a894937d55dfca30501a347edbb9ab1e3caa8bcb5c57166258`.

Controlling design hashes: MAE-action `6f4503059b63389c0806a58d2e64f5a4267a5c253a3d504a8032d429300ec174`; tail-selector design `3e8ec4178317e16da55e5903adc82df76b5a3014ea6de4e5290e8b09e3b8ddbc`; tail-selector prompt `730987f6be12bda95a91a7c808a7ec2d678b1b7804d09b926ef0d20b25ea6df4`. Admitted source tree `c08fdda02f3f2860ac833fe4d2221c57a23fc7050b7f269736de3db696420a6e`; clean bundle revision `2378a2faf40d12833ca78b6adcac218219d3bbc9`.

No fresh panel was created or run. No S2V/S3/S4 truth was read for this validation. Do not proceed to fresh-panel freeze from this result; return to independent audit.


