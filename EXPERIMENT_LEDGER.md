# EXPERIMENT_LEDGER.md

Canonical experiment history. Last updated: 2026-09-11.

### Horizon-Aligned Semantic State Residual closure — not supported, keep S1 (2026-09-11)

- Token: `HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`. Gates: H0 `PASS`, H1 `FAIL`, H2 `FAIL`, H3 `FAIL`, H4 `PASS`.
- Controlling protocol: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`. Evidence: `experiments/evidence/hch_horizon_aligned_state_residual_20260911/`. Code: `experiments/current/hch_horizon_aligned_state_residual/`.
- Panel: `GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`, seeds 7/17/37. S1 Daily-Patch GRU32, Host, splits, Shape target/loss and S1 OOF chronology were frozen and not retrained; S1 metric replay max abs diff `1.421e-14`, `beta=0` direction replay `1.192e-07`.
- Only new trainable object: `beta in R^5` on a deterministic `R_d in R^(24x5)` role table (mean pool of audited `primary_channel=True` forecast-time columns into the five fixed semantic roles; absent role = structural 0; `SUPPLY_MARGIN_FC` absent in all three markets). `u_HSA = normalize(u_S1 + R_d beta)`; `alpha_HSA` refit by the original exact weighted-median rule on HSA OOF directions.
- Positive mechanism: both GANSU_DA Hosts improve Shape cosine (`+0.0666` / `+0.1170`) and the frozen GANSU HIGH-state tertile (`+0.0347` / `+0.1189`), cutting HIGH-state wrong-hemisphere by 12.5 / 18.2 pp; GANSU Overall-MAE gains vs S1 `+1.30` / `+4.31` pp, vs Host `3.55%` / `11.04%`.
- Failure: the gain is not universal. Shape cosine rises in only 4/6 cells and wrong-hemisphere is non-worse in only 3/6; Overall-MAE gain vs S1 is nonnegative in only 4/6 cell medians (worst `-0.87` pp, LAGO_DE/PatchTST); median gain vs Host `6.73%` with only 3/6 cells `>=5%`; max normal-MAE harm `1.02%` exceeds the 1% gate; only 3/6 cells hold in `>=2/3` seeds.
- `beta` is fitted per market on its own chronological OOF rows with one shared five-role vocabulary. No market/Host ID, role embedding, per-hour beta, bias, hidden layer, optimizer/LR/step/seed search, excursion/MAD/clipping, conditional Amplitude, Gate/Verification, or market-specific rule. Structurally absent roles keep `beta=0` exactly.
- Efficiency: 5 extra parameters; median beta training `1.73 s/cell`; median inference overhead `1.22 us/day`; total `7.06 us/day`. Efficiency is not the limiting factor.
- Independent recomputation: `verify_gates.py` re-derives H0–H4 and the token from the written evidence without importing the runner — 39/39 integrity checks pass and all five gate values plus the token match `HSA_VERDICT.json`.
- Two implementation defects were found and fixed before the frozen run: HSA alpha was initially refit over the S1 row set (B1–B4) instead of the rows carrying an HSA direction (B2–B4), and the per-cell loop passed the whole multi-market schema dict where a single-market role mapping was required. Both are now covered by T12 and by the fail-closed assertions in `role_table`/`calibrate_hsa_oof`.
- No SHAANXI/NINGXIA/QINGHAI, Shandong, full-panel, or protected/final data was touched; no rescue variant is authorized from this result.

### Final baseline-fidelity closure — workstream complete with mixed admission status (2026-09-11)

- Token: `HCH_BASELINE_BREADTH_RECOVERY_COMPLETE_FOR_ADJUDICATION`.
- Independent final adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260911/adjudication/FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md`.
- Final method statuses: δ-Adapter Ada-Y = `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`; PIR full official = `PAPER_FAITHFUL_EXACT_ACCEPTED`; COSA = `ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`; UEC-STD = `HOST_FIDELITY_BLOCKED / NOT_PAPER_FAITHFULLY_ADMITTED`; OMPB = `STANDALONE_DATA_BLOCKED / NOT_REPRODUCED`.
- UEC source-native recovery corrected the prior project-side custom loader/AR path and used official ETTh1 loader/fixed borders/train-only scaler/fresh TimeMixer-96/official AR336. Host MSE/MAE `0.500369/0.457737` vs paper `0.465/0.449`; MSE relative error `7.606%` narrowly fails the frozen `7.5%` gate. UEC correction and Weather rescue were correctly not entered.
- OMPB exact TCN/96 paper rows were frozen before outcomes; registry SHA256 `29F4C445045C912536714F41BF3A44AEC3979E34D54FA5865CAA0E8AA1714EDB`. Official dataset bundle unavailable; source/target reads `0`, training `0`, no fabricated metric.
- Mandatory offline comparison layer is considered sufficient/frozen for later HCH unified-DA transfer: Host, direct residual, δ-Adapter, PIR. Supplementary COSA/UEC/OMPB statuses are disclosed and do not block HCH.
- Future reopen conditions are external-evidence only: UEC official/source artifact resolving Host discrepancy; OMPB official/provenance-equivalent dataset; COSA explicit online/TTA paper requirement. No anchor/tolerance/seed rescue search is authorized.

### UEC source-native Host recovery + standalone OMPB breadth reproduction authorized (2026-09-11)

- This is an auxiliary baseline-fidelity continuation only; it does not replace or modify the active HCH Horizon-Aligned Semantic State Residual closure and does not authorize HCH transfer.
- Source audit amends the previous U-A1 interpretation: `run_uecstd_after_authority.py` forced ETTh1 through `Dataset_Custom` / `--data custom` and implemented a project-side AR336 rollout. Official UEC source instead maps `ETTh1 -> Dataset_ETT_hour`, official ETTh1 scripts pass `--data ETTh1`, and official long-horizon evaluation/training is exposed through modes 2/5/6. Historical Host/UEC MSE `0.596529/0.588395` remain valid custom-runner evidence but are not accepted paper-native anchor metrics.
- New UEC protocol: `docs/current/HCH_UECSTD_HOST_FIDELITY_RECOVERY_PROTOCOL_20260911.md`. First train a fresh official native-96 TimeMixer on source-native ETTh1 and evaluate AR336 only through official mode 2. Host MSE must be within 7.5% of paper `0.465` before any UEC corrector is trained. If Host passes, run the fixed U-A1 mode5/6 anchor; Weather U-A2 is entered only after U-A1 PASS. No split/seed/epoch/beta/kernel rescue is allowed.
- The UEC paper text's generic 70%-train statement conflicts with the released ETTh1 fixed-border loader. This is explicitly registered as a paper-code reproducibility discrepancy; if the released-source Host still fails, do not automatically run an alternative ETTh1 split.
- OMPB is fully decoupled from UEC. It must proceed to its own legal terminal state regardless of UEC PASS/FAIL/BLOCKED. Fixed anchors remain TCN/96 `ETTh1->ETTh2` and `Weather_train->Weather_test_far`; exact final-paper rows and official dataset provenance must be frozen before outcomes, and future-label perturbation must confirm predict-then-update causality. Protocol: `docs/current/HCH_OMPB_STANDALONE_MINIMAL_PAPER_FIDELITY_PROTOCOL_20260911.md`.
- Combined orchestration: `docs/current/HCH_UECSTD_OMPB_FIDELITY_RECOVERY_EXECUTION_PLAN_20260911.md`. Launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_UECSTD_HOST_RECOVERY_AND_OMPB_STANDALONE_20260911.md`. New evidence roots are under `experiments/evidence/hch_baseline_paper_fidelity_20260911/**`; old 2026-09-10 evidence is immutable.
- Terminal token after both independent tracks reach a legal terminal state: `HCH_BASELINE_BREADTH_RECOVERY_COMPLETE_FOR_ADJUDICATION`. This token does not imply both methods PASS.

### UEC-STD + OMPB minimal fidelity execution — partial/failed; UEC Host anchor not reproduced, OMPB unrun (2026-09-11)

- Token: `HCH_UECSTD_OMPB_MINIMAL_FIDELITY_PARTIAL_OR_FAILED`.
- U-A1 ETTh1/TimeMixer/336 completed under the fixed paper-authority protocol. Host MSE/MAE `0.596529/0.547638` vs paper `0.465/0.449`; UEC-STD `0.588395/0.544841` vs paper `0.449/0.444`. Host MSE error is ~28.3%, so the registered Host gate fails before UEC method fidelity can be judged. Reproduced MSE gain is 1.363% vs paper ~3.44%; paper beta `0.3`, current official auto-beta diagnostic `1.0`.
- U-A2 Weather/TimeMixer/336 did not complete; partial checkpoint excluded.
- OMPB was not entered; no target registry was generated, no source/target data were read, and no OMPB anchor exists.
- Scientific consequence: UEC-STD remains `NOT_PAPER_FAITHFULLY_REPRODUCED`; current evidence points first to TimeMixer Host/data/config mismatch, not a validated UEC method failure. OMPB remains `NOT_REPRODUCED`.
- This does not alter mandatory offline admission: δ-Adapter partial-fidelity accepted; PIR exact accepted. COSA remains partial/unresolved.
- Evidence: `experiments/evidence/hch_baseline_paper_fidelity_20260910/05_uecstd_ompb_minimal/FINAL_MINIMAL_BASELINE_FIDELITY_SUMMARY.md` and `UECSTD_FIDELITY_SUMMARY.md`.

### UEC-STD minimal fidelity preflight — valid blocker, then paper-authority resolution (2026-09-11)

- Executor token: `HCH_UECSTD_PROTOCOL_BLOCKED`. U0 failed closed before any data read/training because official helper materials conflicted on seasonal/trend weights, Weather kernel size, a broken `--is_training 9` path, and correction-strength candidate grids. This blocker was scientifically correct and remains recorded.
- Subsequent paper/appendix audit resolved season/trend, beta and correction-training details without result-conditioned search: 100 UEC steps; batch 64; Huber/SmoothL1; true AR; official `ecm=linear` two-stage MLP; kernel `25` for both selected TimeMixer anchors. Its earlier assumption of a common 70/10/20 backbone split is superseded for ETTh1 by the 2026-09-11 source-fidelity amendment: official released ETTh1 path uses `Dataset_ETT_hour` fixed ETT borders; Weather remains `Dataset_Custom`.
- U-A1 ETTh1/TimeMixer/336: `lambda_s/lambda_t=0.8/0.2`; paper-reported STD beta `0.3`; targets unchanged Host `0.465/0.449`, UEC `0.449/0.444`.
- U-A2 Weather/TimeMixer/336: `lambda_s/lambda_t=0.6/0.4`; paper-reported STD beta `0.1`; targets unchanged Host `0.265/0.292`, UEC `0.256/0.290`.
- Candidate-grid inconsistency is not resolved by search: current official auto-beta selection is diagnostic only; exact paper anchor is evaluated at the paper-reported beta and the match/mismatch is disclosed.
- UEC is now unblocked for exactly these two anchors. No third anchor/rescue variant is allowed. OMPB proceeds only after UEC summary under its original source→target online-shift protocol.
- Adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/05_uecstd_ompb_minimal/adjudication/INDEPENDENT_UECSTD_BLOCKER_ADJUDICATION_20260911.md`.
- Paper-authority resolution: `docs/current/HCH_UECSTD_PAPER_AUTHORITY_RESOLUTION_20260911.md`.


### HCH State-Excursion Shape closure — valid rejection; horizon-aligned semantic residual reopened (2026-09-11)

- Token: `HCH_STATE_EXCURSION_SHAPE_NOT_SUPPORTED_KEEP_S1`.
- X0 PASS. X1 FAIL: both GANSU HIGH-state Shape cosines decline (`-0.0307/-0.0906`) and TimeMixer wrong-hemisphere rises `+6.06 pp`; only 2/4 preregistered Shape-limited cells improve.
- X2 FAIL despite international gains: S2 is Host-nonworse 6/6 and improves S1 in 4/6, but median gain is only `5.380%`, max Normal-MAE harm is `3.175%`, and both GANSU cells are worse than S1.
- X3 PASS: all four international cells improve S1; X4 FAIL under the preregistered relative latency ratio, though absolute latency remains microseconds/day.
- Scientific consequence: reject signed same-hour excursion as a universal executable Shape input. Do not tune MAD factor/window/clipping or market-specific role subsets. Independent adjudication: `experiments/evidence/hch_state_excursion_shape_20260910/adjudication/INDEPENDENT_STATE_EXCURSION_SHAPE_ADJUDICATION.md`.
- New authorized single-factor closure: `docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`. It freezes S1 and fits exactly five horizon-aligned semantic-role coefficients on existing S1-standardized state levels; no excursion, conditional Amplitude, Gate, deeper network or market-specific architecture.

### HCH Market-State Coupling diagnostic — valid; state-distance hypothesis rejected, high-state Shape limitation supported (2026-09-10)

- Token: `HCH_MARKET_STATE_COUPLING_NOT_SUPPORTED`.
- Independent adjudication: `experiments/evidence/hch_market_state_coupling_20260910/adjudication/INDEPENDENT_MARKET_STATE_COUPLING_ADJUDICATION.md`.
- C0 PASS and C1 PASS_NONDEGENERATE: S1/fixed-alpha replay is exact; forecast-time state is legal/nondegenerate on the six-cell D0 panel.
- Primary `rho_shape -> a_O2-alpha` coupling is weak/mixed: GANSU `+0.070/-0.055`, DE `+0.150/+0.105`, PJM `-0.095/-0.010`; top-bottom primary bootstrap CIs cross zero. Only DE/PatchTST is classified `DISTANCE_STATE_COUPLED`.
- C4 one-feature chronological ridge improves the constant-zero distance baseline in only 2/6 cell medians; the preregistered state-Amplitude trigger fails and `A_state1` is not trained. Do not reopen nonlinear/state-aware Amplitude from this evidence.
- Stronger upstream result: 4/6 cells are `SHAPE_LIMITED`. Both GANSU Hosts show high-state Shape degradation: cosine high-minus-low about `-0.062/-0.185`, wrong-hemisphere `+25/+20 pp`. PJM cells are also Shape-limited under the registered attribution.
- Scientific consequence: retain pooled fixed alpha and deleted Verification; return to Shape representation. One new closure is authorized under `docs/current/HCH_STATE_EXCURSION_SHAPE_CLOSURE_20260910.md`: same S1 GRU32/target/loss/OOF, append only already-audited signed same-hour seven-day state excursions to the current-day Shape patch. No topology/depth/module expansion.
- Evidence root: `experiments/evidence/hch_market_state_coupling_20260910/`.

### Minimal UEC-STD + OMPB paper-fidelity breadth stage authorized (2026-09-10)

- User requested one final bounded reproduction pass for UEC-STD and OMPB; this does not reopen full-table baseline reproduction.
- UEC-STD old wrapper is not paper-faithful: default 2 epochs, wrapper-specific 70/30 split/target normalization/cache residual construction, and only selected official decomposition/linear ECM components. New anchors must use official AR residual-data generation + STD decomposition + Huber UEC training + validation coefficient selection + AR evaluation.
- Fixed UEC anchors: ETTh1/TimeMixer/336 paper Host `0.465/0.449`, UEC `0.449/0.444`; Weather/TimeMixer/336 Host `0.265/0.292`, UEC `0.256/0.290`. Maximum two anchors.
- OMPB remains online source→target shift only; no static HCH wrapper. Fixed preferred anchors: TCN/96 ETTh1→ETTh2 (20 backbone epochs) and Weather_train→Weather_test_far (5 epochs). Exact final-paper rows must be extracted/hash-frozen before running; unresolved row means fail closed.
- Controlling protocol: `docs/current/HCH_OMPB_UECSTD_MINIMAL_PAPER_FIDELITY_REPRODUCTION_20260910.md`; launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_UECSTD_OMPB_MINIMAL_FIDELITY_20260910.md`.
- No δ/PIR/COSA rerun and no HCH transfer is authorized by this stage.

### HCH Anchored Compact Amplitude closure — valid; Shape-only conditional distance rejected (2026-09-10)

- Token: `HCH_ANCHORED_COMPACT_AMPLITUDE_COMPLETE_FOR_ADJUDICATION`.
- Independent adjudication: `experiments/evidence/hch_anchored_compact_amplitude_20260910/adjudication/INDEPENDENT_ANCHORED_COMPACT_AMPLITUDE_ADJUDICATION.md`.
- A0 PASS: S1 replay max error `1.42e-14`, Shape cosine replay error `8.33e-17`; 26 trainable Amplitude parameters; Host/split/input/OOF contracts frozen.
- A1 SUPPORTED but local: compact OOF utility improves fixed-alpha OOF in 4/6 cells and is nonnegative in 6/6.
- A2 FAIL: Host-nonworse 5/6; improves frozen S1 final Overall-MAE in only 3/6; median relative Overall-MAE gain `5.79% < 7%`; >=5% in 3/6; PJM/PatchTST remains below Host; max Normal-MAE harm `1.06% > 1%`.
- A3 NOT_SUPPORTED: exact O2 per-day scalar oracle remains `10.78%--22.73%` relative gain, but compact median O2-over-fixed-alpha capture is only `2.01%`, positive in 4/6. Cell-median amplitude-oracle Spearman is approximately `-0.044 ... +0.126`, i.e. essentially no stable distance ordering.
- A4: compact beats the best frozen development comparator in 4/6, but this does not rescue A2/A3 and is not a final baseline-fidelity claim.
- Scientific consequence: reject the tested `[u_hat || ||v||] -> scalar` learned Amplitude and return the leading executable method to S1 Shape + pooled fixed `alpha`. Do not deepen the same Shape-only Amplitude input.
- Interpretation: scalar Amplitude as a mathematical object remains strongly supported by O2. The new bottleneck is missing conditional-distance information. Because normalized Shape supervision deliberately removes residual scale and `||v||` measures direction concentration rather than magnitude, forecast-time market state is the leading evidence source to audit next.
- Next design only: `docs/current/HCH_POST_AMPLITUDE_MARKET_STATE_COUPLING_DESIGN_20260910.md`. No execution is authorized by this ledger entry.

### Baseline paper-fidelity workstream closure — mandatory offline anchors sufficient; online/secondary baselines frozen incomplete (2026-09-10)

- Human adjudication stops the long-running baseline anchor program for the current HCH method stage. Closure: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/BASELINE_FIDELITY_WORKSTREAM_CLOSURE_20260910.md`.
- Mandatory offline headline status is sufficient: δ-Adapter Ada-Y = `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY` (Weather/Traffic PASS; Electricity formal borderline FAIL retained); PIR full official = `PAPER_FAITHFUL_EXACT_ACCEPTED` on both mandatory ETTh1/96 anchors.
- COSA is not fully paper-native reproduced. Latest-code B3 retains C-A1 F/P PASS, C-A2 F PASS, C-A2 P FAIL. Paper-era provenance resolution was user-stopped after C-PE1 Host only: 10-seed Host MSE `0.435901 ± 0.004820` vs paper `0.4312`, relative error `1.0902%`, Host gate PASS; no paper-era COSA method or C-PE2 run. Status: `COSA_ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED`.
- OMPB was not run and remains optional online-shift; UEC-STD remains secondary/not paper-faithful. Neither blocks Tier-A offline HCH comparison.
- No more paper-anchor reproduction is active. The next required baseline step, only when HCH is ready for final comparison, is frozen unified-DA transfer of admitted offline δ-Adapter/PIR under the HCH task/split/information boundary. COSA may be resumed only if an online/TTA paper block is explicitly required.

### δ-Adapter paper-anchor reproduction — 2/3 exact anchors pass; method-level fidelity accepted with disclosure (2026-09-10)

- Stage token from executor: `HCH_BASELINE_PAPER_FIDELITY_BLOCKED`; this is procedurally correct because the stop rule requires human adjudication after any anchor failure.
- D-A1 Weather/iTransformer/96: PASS. Paper Host/Method MSE `0.1731/0.1631`; reproduced `0.171949/0.162725`; reproduced gain `5.36%` vs paper `5.78%`.
- D-A2 Traffic/iTransformer/96: PASS. Paper `0.4437/0.4294`; reproduced `0.430100/0.418650`; reproduced gain `2.66%` vs paper `3.22%`.
- D-A3 Electricity/iTransformer/96: formal FAIL on relative-gain agreement only. Host error 4.45% passes; method absolute error 2.22% passes; direction is correct; reproduced gain `3.17%` vs paper `5.38%`, deviation `2.21 pp` vs registered tolerance `2.00 pp`.
- No threshold/tolerance/seed/epoch is changed and D-A3 is not rerun. Independent adjudication accepts δ-Adapter Ada-Y at the method level as `DELTA_ADAPTER_ANCHOR_ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`, consistent with the pre-existing headline-admission rule requiring at least one exact passing anchor and no core substitution.
- The historical 2-epoch cache wrapper remains pilot-only and is superseded for strong-baseline work by this paper-anchored official-style implementation.
- Next authorized baseline stage: B2 full-official PIR anchors only. No HCH transfer, COSA or OMPB yet. Adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_DELTA_ADAPTER_ANCHOR_ADJUDICATION.md`; launcher: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_CONTINUE_PIR_AFTER_DELTA_ADJUDICATION_20260910.md`.

### PIR full-official paper anchors — exact fidelity PASS on PatchTST + TimeMixer (2026-09-10)

- Token: `HCH_PIR_PAPER_FIDELITY_ANCHORS_COMPLETE_FOR_ADJUDICATION`.
- Independent adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_PIR_ANCHOR_ADJUDICATION.md`.
- P-A1 ETTh1/PatchTST/96: Host `0.409700/0.415855` MSE/MAE vs paper `0.410/0.416`; PIR `0.373362/0.399086` vs paper `0.375/0.400`; reproduced MSE gain `8.869%` vs paper `8.537%`. PASS.
- P-A2 ETTh1/TimeMixer/96: Host `0.383801/0.394497` vs paper `0.384/0.399`; PIR `0.370109/0.391756` vs paper `0.370/0.397`; reproduced MSE gain `3.567%` vs paper `3.646%`. PASS.
- Full-path fidelity confirms official PIR commit, pretrained Host loading, QualityEstimator, Transformer Refiner, retrieval-index/global revision and learned local/global combination; no ridge/sklearn proxy path was executed.
- Scientific status: `PIR_FULL_OFFICIAL_ANCHOR_ACCEPTED` / `PAPER_FAITHFUL_EXACT_ACCEPTED`. Historical HCH `src/baselines/pir.py` remains a limited proxy and cannot be retroactively renamed.
- Offline Tier-A anchors are now established: δ-Adapter accepted with disclosed partial fidelity; PIR exact fidelity accepted. Next separate baseline stage is COSA online/TTA, using final ICLR 2026 Table-2 targets and 10-run chronology-matched evaluation. OMPB and HCH transfer remain closed pending adjudication.
- Evidence root: `experiments/evidence/hch_baseline_paper_fidelity_20260910/02_pir_anchor/`.

### COSA latest-code 10-run anchors — partial pass; paper-era provenance mismatch discovered (2026-09-10)

- Token: `HCH_COSA_PAPER_FIDELITY_BLOCKED`; the registered stop is valid and no B3 result is rewritten.
- Latest official commit used: `527c0fe` (2026-08-13). C-A1 COSA-F PASS (`0.423314`), C-A1 COSA-P PASS (`0.424210`), C-A2 COSA-F PASS (`0.561657`), C-A2 COSA-P FAIL (`0.589729` vs paper `0.5320`; gain `-4.624%` vs `+4.299%`). Ten of ten C-A2/P seeds are harmful, so the failure is systematic under this protocol.
- All 40 formal runs passed strict no-leakage checks and preserved Host hashes; no seed/config/tolerance was changed.
- Independent provenance audit found the latest commit is post-paper and explicitly adds `POGT adaptation and MLP adapter`, materially introducing partial-ground-truth masks, delayed/full-horizon replay and prediction adjustment. The final ICLR paper/presentation says the reported method relies on full ground truth and treats partial-ground-truth deployment as future work. Thus latest-code B3 is a deployment-safe/current-code result, not necessarily the Table-2 generation path.
- The paper-era first public commit is `43a8c8d` (2026-03-04). It also hard-codes ETTh1 `TRAIN_RATIO=0.6`, `TEST_RATIO=0.2`, while the paper text states 7:1:2; this is a disclosed paper/code split discrepancy.
- Scientific status: COSA-F latest-code is supported on both anchors; COSA-P latest-code is supported at 96 but not 336. Overall paper-native COSA fidelity remains unresolved due source-version/protocol mismatch rather than being declared irreproducible.
- Exactly one paper-era official-code-native rerun is authorized: commit `43a8c8d`, native ETT split, paper-era full-GT batch semantics, fixed seeds 0..9 and unchanged K/S/B/LR, rerunning both C-A1/C-A2 F/P. No further source/config search afterward.
- Controlling adjudication: `experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_COSA_ANCHOR_ADJUDICATION.md`; protocol: `docs/current/HCH_COSA_PAPER_ERA_CODE_NATIVE_REPRODUCTION_20260910.md`.

### Baseline paper-fidelity audit/design — proxy status corrected; reproduction workstream opened (2026-09-10)

- No new baseline scientific result has been claimed yet; this entry records a source/paper audit and a preregistered reproduction protocol.
- Historical δ-Adapter Ada-Y wrapper is not paper-faithful: it uses residual-std scaling instead of official bounded `delta`, LR `1e-3` instead of paper/source `1e-4`, a clipped small hidden width, full-batch fitting, and historically only 2 epochs. Old results remain pilot evidence.
- Historical PIR wrapper is explicitly a cache proxy: it reuses `QualityEstimator` but substitutes the official local Transformer refiner with ridge residual prediction and does not reproduce official retrieval-value semantics or quality auxiliary training. Old PIR proxy results must not be called PIR paper performance.
- Historical UEC-STD wrapper reuses selected source components but not the full source error-correction data construction/training/AR protocol; it is secondary pilot evidence.
- Exact paper anchors are preferred. Paper-derived transfer ranges are preregistered when exact data/backbones are unavailable and must be frozen before unified-DA evaluation. Exact reproduction checks both Host metric closeness and postprocessor metric/gain closeness.
- Offline headline candidates: Host, direct Linear, δ-AdaY, PIR; UEC-STD secondary if task-compatible. COSA is online/TTA only. OMPB requires a dedicated distribution-shift online track; NetBurst is Related Work only.
- Controlling design: `docs/current/HCH_BASELINE_PAPER_FIDELITY_AUDIT_AND_REPRODUCTION_DESIGN_20260910.md`.
- Execution protocol: `docs/current/HCH_BASELINE_PAPER_FIDELITY_EXECUTION_PROTOCOL_20260910.md`.
- Short launcher: `experiments/current/hch_baseline_paper_fidelity/CODEX_GOAL_PROMPT_20260910.md`.
- New evidence root: `experiments/evidence/hch_baseline_paper_fidelity_20260910/**`.

### Baseline paper-fidelity reproduction program registered (2026-09-10)

- New controlling design: `docs/current/HCH_BASELINE_PAPER_FIDELITY_REPRODUCTION_AUDIT_20260910.md`; executor: `experiments/current/hch_baseline_fidelity_reproduction/CODEX_GOAL_PROMPT_20260910.md`.
- Existing δ-Adapter/PIR numbers are explicitly retained as historical pilot/development evidence, not paper-faithful SOTA reproduction.
- δ audit: current wrapper uses official Ada-Y `PostProcessingNet` but materially changes LR, batching, residual scaling, hidden-size selection and validation/early-stopping semantics; increasing epochs is insufficient.
- PIR audit: current wrapper is a limited proxy because the official Refiner is replaced by ridge and retrieval/revision semantics differ; increasing epochs is insufficient.
- Reproduction doctrine: first pass paper-native Host + method anchors, then transfer a verified implementation to HCH. Host anchor tolerance defaults to 5%; method absolute metric tolerance defaults to 5% plus relative-gain agreement. No unregistered hyperparameter search to match paper numbers.
- Planned exact anchors: δ Ada-Y batch on Weather/Traffic/Electricity + iTransformer; PIR ETTh1 horizon96 + PatchTST/TimeMixer, with optional Electricity96. COSA/OMPB are separate online/TTA anchors, not ordinary offline baselines.
- HCH-transfer estimates must be derived and frozen from published same-backbone/closest-horizon relative gains before HCH outcomes. Current 10-epoch development comparators are not upgraded by this registration.
- New evidence root: `experiments/evidence/hch_baseline_paper_fidelity_20260910/**`; anchor stage must stop for human adjudication before any HCH full-transfer/final matrix.

### HCH Unified-DA Shape Engineering canary — valid material-gain pass; compact Amplitude reopened (2026-09-10)

- Token: `HCH_UNIFIED_DA_SHAPE_ENGINEERING_CANARY_COMPLETE_FOR_ADJUDICATION`.
- Independent adjudication: `experiments/evidence/hch_unified_da_shape_engineering_20260910/adjudication/INDEPENDENT_UNIFIED_DA_SHAPE_ADJUDICATION.md`.
- MG0 PASS: `GANSU_DA` uses a true day-ahead target and new target-compatible Host foundation; target-day DA/RT truth is not used as input; protected/final roles remain closed.
- S0 State-Linear misses the preregistered material bar, so S1 Daily-Patch GRU-32 Shape runs under protocol. S1 passes MG1--MG4: 5/6 Host-nonworse, 4/6 >=3% gain, 3/6 >=5%, median relative Overall-MAE gain 5.85%, maximum 10.74%, maximum Normal relative harm 0.85%, and 4/6 wins over the fixed development comparator. S1 uses only ~14,104--18,712 parameters.
- S1 gains are seed-stable in five cells. PJM/PatchTST is the real counterexample: S1 relative gain is about `-1.39%, +0.22%, -0.87%` across seeds and Shape cosine is consistently worse than S0. Do not hide this cell or explain it away as seed noise.
- The decisive new result is conditional ray-distance headroom. With the **same learned S1 direction**, exact per-day MAE-optimal scalar O2 gain is ~10.78--22.73%; extra O2-over-S1 headroom is ~8.5--13.5 percentage points in every cell. Therefore compact conditional Amplitude is scientifically reopened after having been correctly rejected in its former 648-D form.
- Next authorized learner is exactly one 26-parameter anchored compact Amplitude on `[u || ||v||]`, initialized to reproduce the existing pooled scalar `alpha` and trained only from chronological OOF S1 Shape under direct MAE ray loss. Shape is frozen; learned Verification remains removed; no raw-state Amplitude, hidden layer, Gate, Transformer/router/retrieval or hyperparameter search.
- Controlling design: `docs/current/HCH_ANCHORED_COMPACT_AMPLITUDE_DESIGN_20260910.md`.
- Evidence root: `experiments/evidence/hch_unified_da_shape_engineering_20260910/`.

### HCH Material-Gain + Market-Structure audit — valid; task-confounded China track retired; Shape capacity promoted (2026-09-10)

- Token: `HCH_MATERIAL_GAIN_MARKET_AUDIT_COMPLETE_FOR_ADJUDICATION`.
- Baseline-fidelity convergence: GANSU Ada-Y relative harm drops from ~13.2%/11.5% at 2 epochs to ~4.2%/1.1% at fixed 10 epochs; maintained PIR cache proxy remains ~25.5%/35.6% harmful. Therefore prior `generic GANSU adapter failure` is not supported; the 2-epoch adapter pilot was too weak for final baseline claims.
- The former Chinese task was `DA information -> RT price`. The user has now replaced it with the same main task as international benchmarks: **DA price -> next-24h DA price**. All GANSU_RT material-gain/state results remain historical diagnostics only and must not become main paper evidence.
- Track-A oracle ceilings are still valid and decisive: DE/PJM per-day Amplitude oracle gains are roughly 10–18%; Shape fixed-scale oracle gains roughly 9–31%; Gate headroom is much smaller. The main next bottleneck is proposal Shape quality, with Amplitude secondary.
- State-enriched linear Shape, with unchanged Shape target/loss and unchanged closed-form scalar calibration, reaches about 9.32%/7.37% relative Overall-MAE gain on DE and 1.51%/7.67% on PJM, proving that legal forecast-time data can materially improve Shape without adding scientific modules.
- Current method remains `Shape -> calibrated scalar`, with learned Verification removed. The old 648-D conditional Amplitude remains rejected.
- New authorized development design: `docs/current/HCH_UNIFIED_DA_MATERIAL_REPAIR_ENGINEERING_DESIGN_20260910.md`. Canary is `LAGO_DE / LAGO_PJM / GANSU_DA × PatchTST / TimeMixer`; S0 State-Linear first, followed only if needed by one S1 Daily-Patch GRU-32 Shape encoder. No simultaneous Amplitude change, no Transformer/router/retrieval/Gate.
- Final baseline paper-faithful reproduction is deferred. Current Ada-Y 10 epoch / PIR proxy 10 epoch runs are development comparators only.
- Evidence root retained: `experiments/evidence/hch_material_gain_market_audit_20260909/`.

### HCH Compact Verification closure — valid minimal test; learned Verification removed (2026-09-09)

- Token: `HCH_COMPACT_VERIFICATION_CLOSURE_COMPLETE_FOR_ADJUDICATION`; V0 PASS, V1 descriptive, V2 FAIL, V3 PASS, V4 `REMOVE_LEARNED_VERIFICATION`.
- Independent adjudication: `experiments/evidence/hch_compact_verification_closure_20260909/adjudication/INDEPENDENT_COMPACT_VERIFICATION_ADJUDICATION.md`.
- Proposal replay is exact to `~5.53e-8`; only one 49-parameter linear sigmoid verifier on `[v || delta_tilde]` was tested; no prohibited data/learner was opened.
- Selective execution is non-worse than calibrated always-repair in only 2/6 cell medians; median gain versus always is `-0.01172` native MAE; maximum relative Overall harm versus always is 1.21% and maximum Normal harm 1.38%.
- B4 utility ordering is positive in only 2/6 and inverse in 4/6. Although selected-utility mean is positive in all six cells and selective still beats Host in 6/6, this does not justify keeping a verifier because calibrated always-repair is better in 4/6.
- Scientific consequence: remove learned Verification from the leading implementation. Do not rescue with a larger verifier, threshold search, GRU/MLP/Transformer or extra confidence features. Utility/selective headroom remains diagnostic only.
- Materiality correction: the frozen calibrated-ray proposal improves Host Overall-MAE by only about `0.11%, 1.12%, 1.30%, 0.49%, 0.76%, 5.34%` across the six canary cells. It is proof-of-concept evidence, not yet a paper-strength SOTA result.
- Comparator caveat: current canary Ada-Y/PIR rows are not final SOTA evidence (`epochs=2`; PIR has a registered wrapper substitution; GANSU legal market-state covariates are not fully consumed). The next stage must audit baseline fidelity and market/task structure before any China-specific failure claim.
- Next authorized stage: `docs/current/HCH_MATERIAL_GAIN_AND_MARKET_STRUCTURE_AUDIT_20260909.md`; no new method module is authorized by this verdict.
- Evidence root: `experiments/evidence/hch_compact_verification_closure_20260909/`.

### HCH Calibrated-Ray closure — valid six-cell pass; conditional Amplitude removed (2026-09-09)

- Token: `HCH_CALIBRATED_RAY_CLOSURE_COMPLETE_FOR_ADJUDICATION`.
- Independent adjudication: `experiments/evidence/hch_calibrated_ray_closure_20260909/adjudication/INDEPENDENT_CALIBRATED_RAY_ADJUDICATION.md`.
- C0 PASS: Shape replay max diff `4.44e-16`, R2 constant replay max diff `4.30e-8`, Host/prior evidence hashes unchanged.
- C1 PASS: Calibrated-Ray Overall-MAE non-worse and strictly better than Host in `6/6`; Tail-MAE improves `5/6`; Normal-MAE improves in all six cell medians.
- C2 PASS as a development comparator gate: within 1% of the best admitted comparator in `5/6` Overall and `4/6` Tail cells. PJM/TimeMixer remains the main Overall comparator gap.
- C3 SUPPORTED: remove the 648-D conditional Amplitude network from the leading method. Freeze proposal as `Delta=s*alpha*u_hat`, with one pooled nonnegative MAE calibration scalar fitted from chronological OOF Shape/residual pairs. OOF utility is nonnegative in `5/6` cell medians.
- C4 human adjudication: KEEP Verification as a scientific object for exactly one compact learnability closure. Calibrated proposals remain harmful on roughly `29%–48%` of eval days; oracle selective headroom is positive in all six cells and about `0.8%–4.7%` relative MAE.
- Next authorized learner: one 49-param linear sigmoid verifier on `[v || delta_tilde]`, utility-weighted BCE, fixed `q>0.5`; no raw 624-D state, conditional Amplitude, MLP/GRU/Transformer, alternate verifier, threshold tuning, full-panel, Shandong or protected/final execution. If this compact verifier fails, learned Verification is removed rather than rescued.
- Evidence root: `experiments/evidence/hch_calibrated_ray_closure_20260909/`.

### HCH Minimal Repair M0 R2 — valid Amplitude revision; conditional map rejected, calibrated-ray closure promoted (2026-09-09)

- Machine token: `HCH_M0_R2_CANARY_COMPLETE_FOR_ADJUDICATION`; validity=true. Independent adjudication: `experiments/evidence/hch_minimal_repair_m0_20260909_r2/adjudication/INDEPENDENT_M0_R2_ADJUDICATION.md`.
- R2 is a valid one-factor follow-up: Shape replay differs from R1 by at most `~4.44e-16`, R1 Amplitude replay differs by at most `~2.28e-7`, Host caches are unchanged, and the only scientific change is removal of the `J>0` Amplitude-training filter.
- R2 improves vs R1 in 4/6 cells for OOF utility, ray regret, amplitude-oracle gap and always-repair MAE, but learned OOF proposal utility is positive in only 2/6; full conditional M0 remains not supported.
- The preregistered chronological constant-Amplitude probe is valid and decisive: it uses only earlier OOF blocks for each prediction block and DIAG_FIT-only OOF information for final calibration. Cell-median constant OOF utility exceeds learned R2 utility in 6/6 cells; learned R2 beats constant in 0/6 cell medians.
- Learned Amplitude does not rank the exact per-instance ray oracle: cell-median Spearman is about `-0.267 .. +0.073`; ray-oracle capture is negative in four cells, near zero in one, and only ~0.107 in PJM/TimeMixer. The 648-D state-conditioned linear+Softplus Amplitude map is therefore rejected as the leading implementation.
- Scalar ray Amplitude is **not** rejected. The diagnostic constant normalized amplitude applied to the same final learned Shape improves DIAG_EVAL Overall-MAE vs Host in 17/18 seed×cell runs and all 6/6 cell medians (median gains roughly +0.022 to +1.160 native MAE depending on cell).
- Scientific consequence: promote a one-parameter calibrated ray `Delta=s_d*alpha*u_hat`, with `alpha` fitted exactly from chronological OOF Shape predictions as the pooled nonnegative MAE weighted-median solution. This is tested next in `docs/current/HCH_CALIBRATED_RAY_CLOSURE_EXPERIMENT_20260909.md`.
- Verification remains unresolved rather than automatically retained: R2 verifier ordering is positive in only 1/6 cells and selected utility mean is negative in all six, but proposal quality is confounded. The calibrated-ray closure must quantify oracle selective headroom before deciding whether Verification survives.
- Capacity consequence: do not run GRU next. The registered ladder is calibrated-ray closure -> optional 24-D Shape-state compact conditional heads -> optional GRU-32 only if Shape representation remains limiting.
- Evidence root: `experiments/evidence/hch_minimal_repair_m0_20260909_r2/`.

### HCH Minimal Repair M0 R1 canary — valid run, not supported for full expansion (2026-09-09)

- Machine verdict: `HCH_M0_R1_CANARY_NOT_SUPPORTED_FOR_FULL_EXPANSION`; G0=`PASS`, G1=`PASS`, G2=`FAIL`, G3=`PASS`, G4=`FAIL`.
- Independent adjudication: `experiments/evidence/hch_minimal_repair_m0_20260909_r1/adjudication/INDEPENDENT_M0_R1_ADJUDICATION.md`; it agrees with the no-expansion verdict but narrows the failure attribution.
- G0 implementation/provenance is accepted: 22/22 targeted tests + compileall PASS; clean `src/mvp/hch_minimal_repair/**`; chronological stacked OOF; no historical HCH scientific import; raw/S3/S4/protected/final reads all zero; Host/base source immutable.
- G1 Directional Shape remains supported as an object: cell-median Shape cosine is positive in all 6/6 cells (`~0.0876–0.5249`) and positive-hemisphere rate exceeds 0.5 in all 6/6. Except PJM/TimeMixer, cosine remains modest, so M0 linear representation sufficiency is not established.
- G2 fails decisively: M0 Overall-MAE is non-worse than Host in 0/6 cells and strictly improves in 0/6; Tail-MAE improves 3/6; maximum Normal-MAE relative harm is 7.49%. Full-panel expansion is not justified.
- G3 PASS is comparator-relative only: M0 is within 1% of the best admitted post-hoc comparator on Overall/Tail in 5/6 cells, but the comparators themselves are usually worse than Host. Direct Linear and Ada-Y are harmful vs Host in all six Overall-MAE cells; PIR is harmful in five of six and improves only PJM/TimeMixer. This supports adapter-risk motivation, not M0 effectiveness.
- The strongest mechanism result is Amplitude-specific. For every one of the 18 market×Host×seed rows, the exact MAE-optimal nonnegative scalar along the **same OOF predicted Shape** has positive mean utility. Cell-median oracle scalar gains are positive in all six cells (`~+0.042 to +0.224` normalized), whereas learned OOF proposal gain is negative in five of six. Therefore one-scalar Amplitude is not rejected; the current state→scalar learner/training semantics are the primary failure.
- R1 trained Amplitude only on `J>0` rows but emitted Softplus amplitudes on all repair-available rows. Final Amplitude support is only 115–144 rows on LAGO and 22–23 on GANSU_RT for a 649-parameter head; verifier support is 147/26 rows for a 651-parameter head. Thus “linear” M0 is shallow but statistically high-dimensional.
- G4 learned Verification is not supported for the current proposal distribution: only 1/6 cells passes the full nondegeneracy gate; utility ordering is unstable. Verifier capacity is not yet isolated because five of six upstream OOF proposal distributions have negative mean utility.
- Method consequence: do not add GRU/Transformer or change Verification first. The leading R2 design revises Amplitude to the full nonnegative MAE ray projection `argmin_{a>=0} ||e-a u_OOF||_1` on **all repair-available OOF Shape rows**, allowing zero as a legitimate distance. R2 design: `docs/current/HCH_M0_R2_RAY_PROJECTED_AMPLITUDE_DESIGN_20260909.md`. No R2 execution is automatically authorized.
- Evidence root: `experiments/evidence/hch_minimal_repair_m0_20260909_r1/`.

### HCH Minimal Repair M0 first canary preflight — invalid input contract, no scientific run (2026-09-09)

- Verdict: `HCH_M0_CANARY_INVALID`; G0=`INVALID`, G1-G4=`NOT_RUN`.
- Scope audited: fixed `LAGO_DE / LAGO_PJM / GANSU_RT × PatchTST / TimeMixer` six-cell canary, using existing frozen Host caches only.
- LAGO_DE and LAGO_PJM have complete 168h original-origin historical Host-residual coverage on DIAG_FIT/DIAG_EVAL for both Hosts.
- GANSU_RT has incomplete historical Host-residual coverage for both Hosts: `30/62` DIAG_FIT rows and `4/21` DIAG_EVAL rows are incomplete. Missing periods are identical across PatchTST/TimeMixer, indicating a common historical forecast-origin/cache coverage issue rather than backbone-specific behavior.
- The preflight correctly stopped before creating `src/mvp/hch_minimal_repair/**`, before Shape/Amplitude/Verification training, and before any canary metric evaluation. Raw/S3/S4/protected/final reads = `0/0/0/0/0`; Host retraining = false.
- Scientific interpretation: **no method verdict**. The first implementation spec over-required a fully observed 168h residual history. The protocol is repaired globally, not by market exception: add a 168h residual-availability mask, zero-fill only masked residual coordinates, compute causal scale from available residuals, and exact Keep Host when zero residual history is available. Complete-history cells retain all-one masks.
- Original invalid evidence is immutable at `experiments/evidence/hch_minimal_repair_m0_20260909/`. The repaired R1 canary is authorized to write only under `experiments/evidence/hch_minimal_repair_m0_20260909_r1/`.

### HCH Repair Diagnostic Evidence Program — completed machine run + independent adjudication (2026-09-08)

- Machine completion token: `HCH_REPAIR_DIAGNOSTICS_COMPLETE`; evidence root: `experiments/evidence/hch_repair_diagnostics_20260908/`.
- Scope: PAPER8 public international electricity-price benchmarks plus registered China DA→RT stress track; PatchTST/TimeMixer Hosts; simple Linear/Tiny-MLP probes plus existing δ-Adapter/PIR wrappers. Host fitting used S1; diagnostics used DIAG_FIT/DIAG_EVAL only at the model/evaluation layer.
- Independent scientific adjudication: `experiments/evidence/hch_repair_diagnostics_20260908/adjudication/INDEPENDENT_DIAGNOSTIC_ADJUDICATION.md`; adjudicated machine-readable verdict: `adjudication/ADJUDICATED_VERDICT.json`. These supersede the machine-generated `verdict.json` where they differ.
- D1 Host-error heterogeneity: **SUPPORTED**. 21/24 public market×Host cells satisfy the diagnostic tail screen; median top-5% MSE concentration = 0.4071; median P95/median day-loss ratio = 2.2652.
- D2 Shape/Alignment: **SUPPORTED** with `HORIZON_DIRECTION_SIGNAL`. Across 144 public market×Host×candidate cells, median hourly WDR = 0.4693. Hourly sign correction improves MSE relative to the original candidate in 144/144 cells (median incremental recovery 36.0105 native squared-price units); a full 24h direction oracle adds further positive recovery in 144/144 cells (median +17.5930). Median sign share of total direction-oracle recovery = 65.9%, so sign alone is insufficient.
- D3 Amplitude: **SUPPORTED as a necessary object, final parameterization unresolved**. On direction-correct hours, median overshoot = 4.41% and median strong under-correction = 83.29%. Correctly recomputed scalar projection improves 129/143 public candidate cells but captures only median 32.64% of horizon-wise amplitude-recoverable error under the current simple-candidate directions. The original machine verdict `AMPLITUDE_SUPPORTED_SCALAR_PROJECTION` is rejected because its `scalar capture=2.9670` ratio used the wrong denominator and could exceed 1. Scalar-vs-horizon-wise must be retested after Shape is explicitly modeled as a normalized 24h direction.
- D4 proposal-level Verification: **HEADROOM SUPPORTED; LEARNABILITY UNRESOLVED**. Median harmful candidate-day rate = 60.28%; positive oracle selective headroom occurs in 144/144 public cells; always-repair beats Host in only 53/144 cells whereas oracle selective repair beats Host in 130/144. True realized Host-badness has median AUROC only 0.6263 for candidate benefit, and the worst-20%-Host subset still has median harmful-repair rate 0.50. This supports the scientific distinction `Host is bad != this proposal is beneficial`. The machine `VERIFICATION_LEARNABILITY_SUPPORTED` verdict is invalidated because the verifier was trained on in-sample DIAG_FIT candidate utility rather than OOF candidate utility.
- Introduction evidence: I1 error heterogeneity = SUPPORTED; I2 wrong direction material = SUPPORTED; I3 direction-correct not sufficient = SUPPORTED/qualified; I4 proposal utility differs from Host error = SUPPORTED; I5 market/regime repair-risk spectrum = UNRESOLVED; I6 China stress track extends that spectrum = UNRESOLVED. Exact admissible wording is registered at `adjudication/INTRO_EVIDENCE_REGISTER_ADJUDICATED.md`.
- Boundary correction: the runner read complete raw target columns before assigning the new diagnostic S1/S2/S3/S4 labels. Therefore the newly created diagnostic S3/S4 slices are **consumed/contaminated under strict read semantics** and may not later be presented as untouched protected final truth. This does not invalidate the DIAG_EVAL diagnostic evidence, but it changes future evaluation governance.
- Method consequence: retain a minimal `Frozen Host -> normalized horizon Shape/direction -> Shape-conditioned Amplitude -> concrete proposal -> proposal-level selective decision` research structure. Do not restore historical Bridge/Global/router/CORN/quantile/market-expert complexity. Immediate next stage is mathematical target derivation and Shape-conditioned/OOF utility design discussion; no new experiment is automatically authorized by this verdict.

### HCH Repair Diagnostics D0 preflight — infrastructure blocker, no scientific read (2026-09-08)

- Verdict: `HCH_REPAIR_DIAGNOSTICS_BLOCKED_PREFLIGHT`.
- Scope: registered `PAPER8 + CHINA5` × `PatchTST/TimeMixer` preflight only; D1–D4 were not executed.
- Result: the first executor found `0/26` compatible maintained frozen Host cache cells at the originally required path and therefore stopped before opening scientific raw observations. `scientific_raw_reads=0`; S3/S4/protected/reserved reads = 0; Shanxi was not invented/substituted.
- Scientific validity: **no Shape/Amplitude/Verification verdict**. This is an execution/protocol finding, not evidence against the method hypothesis.
- Independent audit found the protocol had over-constrained `frozen Host` into `pre-existing Host cache`. Maintained foundation code already supports the legitimate lifecycle `train Host on preregistered S1 -> freeze -> immutable cache`, so missing cache alone is no longer a blocker in the revised design.
- Additional preconditions identified before rerun: task/target semantics must separate PAPER8 DA from CHINA5 CN-ERB DA→RT, and timestamp/timezone/DST episode semantics must be bound reproducibly. Revised D0 is `D0-A contract -> D0-H Host foundation -> D0-C leakage closure`; D1–D4 remain pending.
- Evidence: `experiments/evidence/hch_repair_diagnostics_20260908/preflight/`.

### HCH V3.1 Shape Semantic Identification (SSI) — single authorized real execution (2026-09-05)

- Verdict: `HCH_V3_1_SSI_SOURCE_SIGNAL_PRESENT_TARGET_TRANSFER_FAILED` (Case B).
- Validity: `G0=True`; all registered provenance/immutability/data-boundary checks pass. Exactly one real runner call completed normally. Allowed real access counts: SOURCE_GTRAIN 12, SOURCE_GVAL 12, TARGET_D1 6, TARGET_D2 6, CAUSAL_HISTORY_CONTEXT 36. S2V/S3/S4/fresh successful access = 0/0/0/0; final Gate execution = 0.
- Source result: B1 extreme-repair semantics beats A1 generic residual semantics in all 3/3 target-excluded source G-Val settings. Seed-median ER-AUPRC ratios are approximately DE 1.199, NORD 1.822, PJM 1.215.
- Target D2 result: B1 beats A1 in only 2/6 cells. B1/A1 seed-median ER-AUPRC ratios are DE/PT 1.533, DE/TM 1.492, NORD/PT 0.962, NORD/TM 0.926, PJM/PT 0.729, PJM/TM 0.779; six-cell median = 0.944307 < registered 1.20. Therefore G1 fails and G2 is not promotion-evaluated.
- NormalDiversion diagnostic: B1 improves 4/6 cells; six-cell median 1.0000 -> 0.937291. This passes the diversion subcondition but cannot rescue G1.
- Direction diagnostic (report-only): pooled six-cell active-direction macro-F1 is 0 for B1 and B2 for seeds 7/17/37, with 469 active rows per seed (358 Down / 111 Up), indicating unresolved target-side posterior/Identity dominance.
- Scientific interpretation: the frozen V3 representation contains learnable extreme-repair signal, but target semantics + current three D1 Shape scalars do not transfer robustly across markets. The next bottleneck is minimal target-local Shape adaptation/sharing; do not move to Amplitude, Gate, class/focal/threshold rescue, or fresh panel.
- Evidence: `experiments/evidence/hch_v3_1_shape_semantic_identification/`; independent adjudication: `experiments/evidence/hch_v3_1_shape_semantic_identification/INDEPENDENT_SSI_SCIENTIFIC_ADJUDICATION_20260905.md`.

### HCH V3.1 D2 mechanism validation (2026-09-04)

- Verdict: `HCH_V3_1_D2_MECHANISM_VALIDATION_PASS_AWAITING_FRESH_PANEL_DESIGN`.
- Scope: one registered validation on the existing six development cells using
  only legal S1/S2T-derived source/G-Val/D1/D2 data; no consumed S2V/S3/S4
  truth access.
- `SIGNED_MAE_BAYES_GRID_V1` reused the frozen V3 learned distributions and
  changed only the parameter-free Gate action readout. Availability was 5/6,
  nonnegative D2 Overall-MAE gain was 6/6, and Normal-MAE harm violations were
  0. Historical V3 fixed-policy replay was preserved.
- Evidence:
  `experiments/evidence/hch_v3_demo_validation/v3_1_d2_mechanism_validation/`.
  This is a mechanism result only; it does not establish V3.1 effectiveness.
- Independent adjudication accepts the MAE-consistent action policy but withholds fresh-panel execution because the registered D2 selector remains Overall-MAE-first. D2 Tail-MAE gain vs Identity is approximately `0%, 0%, +0.10%, 0%, -0.50%, -0.22%` across DE/PT, DE/TM, NORD/PT, NORD/TM, PJM/PT, PJM/TM; the two PJM cells improve Overall-MAE while slightly worsening Tail-MAE. Independent verdict: `HCH_V3_1_D2_ACTION_POLICY_PASS / TAIL_OBJECTIVE_ALIGNMENT_REQUIRED_BEFORE_FRESH_PANEL`. Next registered development experiment is one D2-only `D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1` selector validation; no S2V/S3/S4 or fresh-panel execution is authorized yet.

| Stage | Primary verdict / status | Main result |
|---|---|---|
| V2.5 Phase D | OPEN_V2_6_UPSIDE_REPAIR | corrected target competitive only in selected cells; transfer LOMO/LOHO viable but double-unseen weak; repeated upper-branch weakness identified |
| V2.6-U | NO_V26_UPSIDE_CANDIDATE | Up structural variants U1-U4 failed; training often actively suppressed Up |
| V2.6-L | NO_V26_LEARNING_SIGNAL_CANDIDATE | L2/L3 increased positive correction and improved UpperTail but harmed Lower/overall; issue became selectivity/representation rather than capacity |
| V2.6 R0-R5 | MIXED_REPRESENTATION_FAILURE | large oracle geometry headroom; HES has incremental information; direct routing fails; safe release preferable but double-unseen weak |
| V2.6-SR | SELECTIVE_RELEASE_REJECTED | SR2 selected; universal S3 certificate failed; later consistency correction established real count 14/18 safe, 4/18 fail |
| V2.6-HCR | HIERARCHICAL_RELEASE_REJECTED | certifiability partially predictable; HCR4 failed universal S3-B certificate |

| V6.2 HERA H0 | INTERNAL_NOT_SUPPORTED | H0-A failed occurrence stop (day lift >1 in 3/6 cells; positive day+G1 cells 3/6); H0-B failed for Q1/Q2/R2; no S2-V/S3/S4 |

V6.2 machine root: `experiments/evidence/v6_2_hierarchical_extreme_risk_allocation/`. All three registered variants used legal previous-168h context, frozen host24, IAMR body24, OOF rank reliability, and empirical weighted-median actions. The NORD F1 zero-support condition was handled by the preregistered same-host fallback. Independent gate recomputation and final audit passed; no rescue or post-gate data access occurred.
| V2.6-HCA | HOST_CERTIFIABILITY_NOT_PREDICTABLE / bounded diagnostic | binary host certifiability weak; capacity target appeared predictable but did not justify host-family adaptation |
| V2.7-PCC | CERTIFIED_PREFIX_CAPACITY_NOT_VIABLE | certificate-aligned prefix capacity failed 4/6 viability requirement; universal controller line stopped |
| CFR | ONLINE_RECOVERABILITY_NOT_SUPPORTED | old S2-T proposal state had severe in-sample optimism; chronological cross-fitting fixed most inflation; expanding/ROLL28 online updates did not recover universal out-of-sample correctionability |
| V3-PLA | PRIOR_LOCAL_ARBITRATION_NOT_PREDICTABLE | Shared/Local complementarity is real and large-margin; soft router cannot capture enough oracle headroom |
| V3-HPA | HPA_NOT_SUPPORTED | A0 oracle margins real; L2-SP target adaptation failed; all 18 A1 runs selected lambda=infinity; old parameter-space partial pooling rejected |
| V4-DRQ | CONTINUOUS_CENTER_NOT_SUPPORTED | direct center residual MLP failed: Center-Z 1/6, median -7.17%; Center-R 1/6, median -4.15% |
| V5-IAMR | IAMR_TARGET_LOCAL_ONLY | monotone target-local calibration succeeds: IAMR2 M1 passed; IAMR3 hour extension passed 4/6 vs IAMR2; shared-prior function-space pooling failed 3/6 |
| V5.1-TLR | TLR_TAIL_SAFE_FREEZE_NOT_SUPPORTED | tail-aware Fourier-hour + alpha freeze attempt failed; all 6 cells selected lambda_h=infinity, coverage gates failed, and PJM PatchTST UpperTail safety failed |
| V5.2-APTC | APTC_DEVELOPMENT_NOT_SUPPORTED | upper slope-one continuation reduced but did not remove PJM tail harm; 2026-08-31 post-hoc source audit corrected a provenance error: actual junction was S1 **truth** q95, not reported S1 host q95; static raw junction was ~99.4%/~99.23% percentile in PJM S2-V and inactive in DE/NORD |
| V5.2 post-hoc junction audit | PROVENANCE_CORRECTED / MOVING_RAW_Q95_NOT_SUFFICIENT | V5.2 final numbers/verdict remain reproducible, but q95-source semantics were wrong; fixed S2-T and expanding host-q95 are too inertial, trailing-28 is responsive but unstable/market-dependent, and pure within-day q95 is legal but statistically misaligned |
| V5.3-RATJ | V5_3_RATJ_HOST_RANK_NOT_SUPPORTED | protocol-compliant R0 only: G0/G2/G3 passed but G1 failed; all cells selected 7d EW half-life; DE/PJM recovered ~5% occupancy, NORD remained ~12.1–12.3% with severe day/block clustering; R1 correctly not executed |
| V5.4 Conditional EVT Tail | V5_4_CONDITIONAL_EVT_TAIL_FEASIBLE | E0 existence/modelability result: honest S2-T cross-fitted IAMR3 residual tails persist in all six cells and all six show converged hourly/daily GP fits; PJM hourly xi reaches ~1.03–1.18 while daily-declustered xi is ~0.53–0.77, so mean-excess correction is forbidden; no S2-V correction prediction, S3 or S4 |
| V5.4-E1 Conditional Tail Distribution | V5_4_E1_TAIL_DISTRIBUTION_NOT_IDENTIFIED | G0/G4 passed; G1 failed on PJM/PatchTST Fold-A count (19<20), G2 failed on NORD Fold-A occurrence gaps, G3 failed on NORD severity coverage and PJM q90 overcoverage; M0 selected for DE/NORD and M1 for PJM; source re-audit found M1 conditional-coverage scoring used median sigma, but corrected recomputation still fails G3 and does not change verdict; no S2-V correction, S3/S4, mean-excess action or rescue |
| V6 ICDE Architecture Race | V6_R0_ARCHITECTURE_RACE_COMPLETE_NO_TOP2 | R0 completed on E0 honest S2-T only. A_CT_EQ and B_REMR were SHIFT_FAILURE (Fold-A positive/near-zero S but Fold-B negative); C_BEQA retained positive median UpperTail gain but large normal/MAE harm. In both NORD hosts every F3 7-day block was harmed by all three dense-action arms. No arm passed promotion, so no S2-V/R1, S3 or S4 was run. |
| V6.1 Certifiable Extreme Repair | V6_1_CER_INTERNAL_ONLY_NOT_SUPPORTED | Internal E0 screen completed: all three fixed variants have median release <1% and median MAE/UpperTail gain <=0; NORD cell day-miscoverage also exceeds the .25 certificate screen. No S2-V one-shot, S3 or S4 was authorized. |
| V6.2 HERA | V6_2_HERA_INTERNAL_NOT_SUPPORTED / REAUDIT_NARROWED | Strong F3 UpperTail gains but dense action causes MAE/Normal harm. Re-audit found F3 reused F1-only reliability state despite prefix-specific design; weighted-median non-event pool also leaks into ordinary residual-center correction. NORD sparse selected-slot sign precision remains poor. No S2-V/S3/S4. |
| V6.3 SDAER | SDAER_STATIC_NOT_SUPPORTED / P2_DIAGNOSTIC_ONLY | Prefix-correct sparse utility execution passed chronology but both static variants failed H0-B/NORD stop. S2-V remained unread. Direct utility and online rescue are closed as headline routes. |
| CMER | CMER_S2V_PILOT_NOT_SUPPORTED | Exactly one frozen S2-V one-shot executed; safety/action-density checks passed but median MAE gain -0.2527%, median UpperTail gain +0.9553%, <5/6 positive UpperTail cells, median selected residual-positive precision 48.04%, and NORD precision failures. No rescue/S3/S4. |
| HCH V3 authorized demo | HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2 | Authorized 3-market × 2-Host S2V demo executed once with frozen states. HCH full/global-only were pointwise exactly Identity across all 394 S2V days because the Gate never acted. Independent metric replay is exact; metric-specific external-peer comparison is MAE near-best 6/6 and Tail-MAE near-best 4/6. A D2-only no-retrain counterfactual supports one narrow V3.1 action-risk consistency correction; consumed S2V cannot be reused for V3.1. |
| CN-ERB | PROVISIONAL_PLAN / NOT_ACTIVE | Drafted China Provincial Extreme-Repair benchmark option after CMER; later strategic review clarified that the paper remains cross-market rather than China-only. SHANDONG is private/supplementary-only; public provincial markets remain candidate related-domain extensions. No CN-ERB run is authorized yet. |

## V5-IAMR corrected interpretation

- IAMR2 vs legal V2.5 endpoint: 5/6 positive, no >2% MAE harm, median gain ~1.34%.
- IAMR3 vs IAMR2: positive 4/6.
- Directly combining recorded M1/M2 machine gains implies IAMR3 vs legal V2.5 endpoint is positive in all six cells; approximate gains:
  - DE_EPEX/PatchTST +0.34%
  - DE_EPEX/TimeMixer +0.63%
  - NORD_FI/PatchTST +18.06%
  - NORD_FI/TimeMixer +10.35%
  - PJM_2020/PatchTST +1.56%
  - PJM_2020/TimeMixer +5.84%
  - median ~+3.70%.
- Tail caveat: PJM IAMR2/IAMR3 materially worsens UpperTail/Tail-MAE vs Identity; overall improvement is not yet a tail-safe headline result.
- IAMR is deterministic given host cache/S2; repeated seed rows in V5 do not represent independent calibrator training seeds.

## V5.1-TLR mechanistic readout

- Stage A replay: `V5_IAMR_REPLAY_AND_STATISTICAL_READJUDICATION_OK`.
- TLR3 selected `lambda_h=infinity` in all six original cells; the Fourier hour component was never deployed.
- Selected alpha values were heterogeneous and included exact Identity fallbacks, but PJM_2020 × PatchTST still selected full correction and failed UpperTail safety on S2-V.
- PJM S2-T upper events are temporally clustered: total counts can look adequate while individual chronological validation folds are too sparse for a reliable rare-tail eligibility rule.
- IAMR gain-by-host-rank diagnosis shows NORD_FI’s dominant gain occurs in the lowest host decile; symmetric tail fallback/continuation is therefore rejected.
- Upper-only raw slope-one continuation is the current mechanistic hypothesis because it can preserve high-host anomaly magnitude without deleting lower-tail calibration.

## V5.2-APTC mechanistic readout

- Historical D0: `V5_2_APTC_D0_AUDIT_OK`; D1 algebraic status: `D1_MATHEMATICAL_CONTRACT_OK`; D2 active-cell overall MAE remained attractive: DE PatchTST +2.18%, NORD PatchTST +18.87%, NORD TimeMixer +9.78%, PJM PatchTST +2.75%, PJM TimeMixer +3.78% vs Identity; DE TimeMixer fell back to Identity.
- PJM Tail/UpperTail remained unsafe: Upper relative harm about +9.72% / +9.95% for PatchTST / TimeMixer.
- **Post-hoc provenance correction (2026-08-31):** `tlr_runner.load_context()` computes `s1_q95` from S1 `y_true`; APTC consumes this value directly. Therefore the actual deployed junction was S1 truth q95. Historical D1/D0/report strings claiming `S1 host forecasts only` are semantically incorrect, and the final audit did not inspect upstream q95 provenance.
- Corrected actual-junction drift: DE/NORD APTC trigger rate is 0% on S2-V host predictions; PJM PatchTST threshold 103.33 is ~99.40 percentile with ~0.60% trigger, TimeMixer 103.33 is ~99.23 percentile with ~0.77% trigger. The earlier S1-host-q95 drift numbers remain a valid post-hoc host-only diagnostic, not the actual V5.2 junction.
- Continuation remains useful: relative to Identity, PJM IAMR3 UpperTail harm was ~+26.42%/+23.83%; APTC reduced this to ~+9.72%/+9.95%, removing about 63%/58% of that harm.
- Read-only host-rank diagnosis: fixed S2-T q95 triggers 0% in DE, ~17–24% in NORD and ~1.5% in PJM; expanding history is similarly inertial; trailing-28 improves DE/PJM to ~5.8–6.7% but still triggers ~15.6–17.4% in NORD and has large block-level swings. Thus a moving raw q95 is not equivalent to a stable rank coordinate.
- Formal-cache causality audit confirms each daily episode uses the prior 168h context to jointly produce the next 24 host forecasts and host `predict()` reads context only. The current-day 24h vector is therefore legally available for post-processing, but pure within-day q95 mechanically triggers 2/24 hours and poorly covers historical truth-upper events, so it is rejected as the sole coordinate.
- Hence static absolute-price tail definition remains rejected; the surviving object is a causal adaptive current-host CDF/PIT/rank coordinate coupled to the useful slope-one continuation.

## V5.3-RATJ mechanistic readout

- Independent contract replay: `7 passed`; final audit and protected-source checks are consistent with the user report.
- R0 gates: G0=true, G1=false, G2=true, G3=true. R1 was correctly blocked and no RATJ correction prediction was generated.
- All six cells selected 7-day EW half-life. This establishes strong evidence that stale-history forgetting matters, but does not justify an unregistered shorter-half-life search.
- EW-CDF pooled S2-V exceedance is near nominal in DE (~5.12/5.16%) and PJM (~5.28/5.63%), but remains ~12.07/12.33% in NORD.
- NORD excess occupancy is regime clustered: one 7-day block reaches ~28.57/30.36%; single days reach 18/24 and 17/24 flagged hours, with strong daytime/evening concentration. Hence pooled marginal host CDF conflates day-level shift and hour structure.
- Retrospective truth-only diagnosis after the frozen R0 verdict: selected EW flags cover 86.4% of historical upper events in both PJM hosts, versus 22.2%/100% in NORD PatchTST/TimeMixer. Host-tail alignment is not universal even when global host/truth rank is strong.
- Scientific verdict: adaptive forgetting is useful, but a **single pooled marginal host-percentile junction is rejected as a universal tail coordinate**. Do not rescue V5.3 with new q levels, shorter half-lives, host-only DSPOT junctions, or post-hoc PJM-only R1 on this panel.

## Current next research question

CMER completed exactly one frozen S2-V evaluation and received `CMER_S2V_PILOT_NOT_SUPPORTED`. The result is not a safety collapse: no cell exceeded 3% MAE harm, median Normal harm was only 0.69%, and median action rate was 6.37%. The failure is effectiveness/ordering: median MAE gain -0.25%, median UpperTail gain +0.96%, fewer than 5/6 positive UpperTail cells, median selected residual-positive precision 48.0%, with severe NORD precision failure. CMER is stopped; no rescue on its consumed S2-V panel is allowed.

New repository evidence broadens, but does not replace, the research setting: the provincial datasets contain a coherent set of forecast-time market fundamentals. SHANDONG is private/non-public and can only be supplementary; GANSU/SHAANXI/NINGXIA/QINGHAI are candidate public related-domain markets subject to provenance/admission review. Their existence motivates a possible cross-province/fundamentals extension, not an automatic China-only reset.

Current status is **STRATEGIC_CONSOLIDATION / NO NEW EXPERIMENT AUTHORIZED**. The next discussion must summarize the full V2.5→IAMR→V5/V6→CMER arc, clarify final ICDE framing, and decide how international heterogeneous markets and public Chinese provincial markets should be combined in the final cross-market benchmark.

The user added a hard paper-success requirement: on the ultimately selected datasets and headline metrics, the final method must beat the preregistered selected strong baselines/peers; beating Identity or an internal historical endpoint alone is not sufficient. No S3/S4 consumption or new architecture is authorized until that benchmark/method direction is explicitly frozen.

Stage synthesis:
`docs/history/v5_iamr/HCH_V2_6_TO_V5_2_STAGE_SYNTHESIS_AND_HANDOFF_20260831.md`

V5.2 post-hoc audit/review:
`docs/history/v5_tail/HCH_V5_2_APTC_POSTHOC_REAUDIT_AND_RANK_JUNCTION_REVIEW_20260831.md`

V5.3 R0 audit / next-direction adjudication:
`docs/history/v5_tail/HCH_V5_3_RATJ_R0_AUDIT_AND_CONDITIONAL_TAIL_ADJUDICATION_20260831.md`

V5.4 feasibility protocol:
`docs/archive/v5_tail_details/HCH_V5.4_EVT_TAIL_FEASIBILITY_AUDIT_PROTOCOL_v0.1_20260831.md`

V5.4 E0 re-audit / E1 registration:

- `docs/history/v5_tail/HCH_V5_4_E0_REAUDIT_AND_CONDITIONAL_GPD_ADJUDICATION_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_MATHEMATICAL_CONTRACT_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_IMPLEMENTATION_DESIGN_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_PROTOCOL_v0.1_20260831.md`
- `docs/archive/v5_tail_details/HCH_V5.4_E1_CONDITIONAL_GPD_TAIL_MODEL_PROMPT_20260831.md`

## V6.3 SDAER result (2026-09-01)

- Isolated root: `experiments/evidence/v6_3_sparse_decision_aware_extreme_repair/`.
- Static core variants were exactly `SDAER_S1(lambda=1)` and `SDAER_S2(lambda=2)`; `SDAER_P2` was kept as a separate online information setting.
- Contract tests: `10 passed`. F2 used F1-only OOF utility/day-score reference; F3 rebuilt its reference from F1+F2, with 588 explicit F2-evidence rows entering the F3 reference after chronological availability.
- H0-A independent recompute passed. Static H0-B failed for both variants. F3 diagnostic rank: S2 first (`median S=0.00534119`), S1 second (`0.00143156`); this is not a promotion because both failed H0-B and the NORD stop gate.
- F3 median UpperTail gain: S1 `1.4009%`, S2 `2.8676%`; median selected-slot residual-positive precision: S1 `54.9253%`, S2 `51.1696%`. NORD harm/precision failures, overall point-harm, 7-day block, and captured-benefit gates prevent static promotion.
- No S2-V one-shot was authorized; no S2-V/S3/S4 artifacts were generated. No historical artifacts or `src/hch_v2/**` were modified.
- P2 strict state audit passed on all 366 target-day rows. P2 is diagnostic-only, excluded from static gate/ranking, and must not be used to rescue the static result.
- Full machine result: `experiments/evidence/v6_3_sparse_decision_aware_extreme_repair/reports/SDAER_V6_3_FINAL_RESULT.md`; final audit: `provenance/final_audit.json`.

## CMER one-shot S2-V result (2026-09-01)

- Isolated root: `experiments/evidence/cmer_cross_market_extreme_repair/`.
- Exactly one CMER configuration was run on DE_EPEX/NORD_FI/PJM_2020 × PatchTST/TimeMixer. Shared rankers used same-host target-excluded source markets with equal-market weighting; local rankers used target S2-T; all six cells met the 5-event-day/20-event-hour local-support requirement.
- Contract tests passed (`15 passed`). The freeze barrier, input/code hashes, model states, rank fusion, support state, prevalence, K, active threshold, target q10 magnitude and hourly scales were sealed before the S2-V truth load. No S2-V search/retry, S3, or S4 occurred.
- Median S2-V MAE gain was `-0.2527%`; median UpperTail gain `0.9553%`; median Normal harm `0.6888%`; median action rate `6.3738%`; median selected residual-positive precision `48.0385%`.
- The strict pilot gate failed on point-gain, UpperTail gain/coverage, overall residual-positive precision, and NORD residual-positive precision. Safety checks for 3% cell MAE harm, Normal harm, action density, NORD MAE harm, freeze and protected history passed.
- Verdict: `CMER_S2V_PILOT_NOT_SUPPORTED`. No CMER rescue or paper-scale expansion is authorized from this pilot.
- Full report: `experiments/evidence/cmer_cross_market_extreme_repair/reports/CMER_FINAL_RESULT.md`; independent gate: `audits/independent_gate_recompute.json`; final audit: `provenance/final_audit.json`.


## HCH V3 authorized S2V demo (2026-09-04)

- Authorization: explicit `DEMO_EXECUTION_AUTHORIZED`.
- Scope: frozen DE_EPEX/NORD_FI/PJM_2020 × PatchTST/TimeMixer; P1 `DE_EPEX::PatchTST` operational canary followed by P2 all six cells.
- Chronology: 394 S2V days × 7 typed methods; predictions persisted and hashed before each day-scoped truth reveal, followed only by legal next-day HCH history update.
- Frozen artifacts: 3 LOMO Global checkpoints, 12 HCH states, 24 fitted peer states, 6 S2V-start histories, six D2 tables. All D2 tables remained `{0}` with zero Down/Up actions and zero coverage.
- Result: HCH full was no worse than Identity in all six cells and best/near-best versus headline external BestPeer (`PIR`/`delta-Adapter Ada-Y`/`UEC-STD`) in all six cells, but the all-Keep mechanism is degenerate.
- Verdict: `DEMO_MECHANISM_ONLY` (architecture-positive gate withheld because of degenerate mechanism; CRC remained `LIMITED_VARIANT_APPENDIX_ONLY`).
- No HCH/peer tuning, rescue, Host retraining, architecture/split/support change, or post-hoc S2V rerun was performed. S3/S4 truth were not read. This development demo does not establish HCH V3 validity.
- Artifacts: `experiments/evidence/hch_v3_demo_validation/authorized_real_execution/P2_all_six/`.
- Provenance: bundle `f551148a67cdba03fa219e16e0170e5873a5110c`, admitted source tree `802192fcdc2835a4daec651f0d09f135d8a553a34ed6f69aded008b840caa812`, protocol manifest `97717110ce5f319703d6f1f1c00b73909a64303e882bb42408b088c93ce56942`, metrics `31e22a01278de56282e196806bed1ab3f728b01e45c7632c7f8a9514b4485540`.

### Independent HCH V3 demo correction (2026-09-04)

- Historical machine token `DEMO_MECHANISM_ONLY` is preserved, but its fallback logic is not protocol-faithful: the implementation assigns mechanism-only whenever diagnostics were merely computed rather than requiring explicit positive mechanism evidence.
- Independent metric replay from raw truth + persisted predictions is exact (`max abs diff = 0`). `HCH-V3-full` and `HCH-V3-global-only` are pointwise exactly Identity on every one of the 394 S2V day artifacts, so the executed method has no realized repair effect.
- The verdict evaluator also has a BestPeer bug: it selects the peer by MAE once and reuses the same peer for Tail-MAE. Correct metric-specific external comparison gives MAE best/near-best `6/6` but Tail-MAE best/near-best only `4/6`. NORD_FI/PatchTST is ~14.43% worse than PIR on Tail-MAE; PJM_2020/TimeMixer is ~79.54% worse than PIR.
- Independent scientific adjudication: `HCH_V3_DEMO_NOT_SUPPORTED_AS_EXECUTED / GATE_ACTION_RISK_MISMATCH_SUPPORTED_ON_D2`.
- First-principles diagnosis: fixed directional `Q(0.75)` point actions are optimized for a different asymmetric loss but are judged by symmetric expected-MAE Gate risk. A legal D2-only, no-retraining counterfactual selecting the existing quantile-grid candidate minimizing the same MAE risk restores safe nonzero actions in 5/6 cells and positive D2 Overall-MAE gain in those five cells (largest ~+2.61%/+1.91% on PJM PatchTST/TimeMixer) with no Normal-MAE harm under the existing selector.
- This does **not** authorize rescue on consumed S2V. The next object is a minimal V3.1 parameter-free Gate action readout correction, validated first on source/D1/D2 only. Detailed audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_AUTHORIZED_S2V_RESULT_AUDIT_20260904.md`.

## HCH V3.1 tail-first safe D2 selector development validation (2026-09-04)

- Registered exactly one new selector: `D2_TAIL_FIRST_SAFE_LEXICOGRAPHIC_V1`; action policy remains `SIGNED_MAE_BAYES_GRID_V1` and learned states are unchanged.
- The selector uses the fixed four chronological 7-day cross-fit on each existing 28-day D2 cell. CandidateSet, Overall/Tail/Normal support scores and kappa are fit on other 21 days; held-out 7-day truth is evaluation-only.
- Gate: nonzero availability `5/6`; strictly positive cross-fitted Tail-MAE gain `1/6`; Overall-MAE harm violations `2`; Normal-MAE harm violations `2`; chronology/provenance/numerical failures `0`; historical V3 fixed replay `true`; historical V3.1 Overall-first replay `true`; consumed S2V/S3/S4 access `0`.
- Verdict: `HCH_V3_1_TAIL_FIRST_D2_NOT_SUPPORTED`. Full-D2 deployment kappa was not computed because the cross-fitted promotion gate failed. No fresh panel was created/run and no consumed S2V/S3/S4 truth was accessed.
- Independent audit accepts the failure as scientific, not implementation-related: 41/41 HCH tests and 27/27 demo tests pass; source tree and clean-bundle revision match the manifest. DE actions are entirely Normal; PJM retains ~10-12% action density but zero Tail action under cross-fit. Therefore D2 selector/margin rescue is frozen and the bottleneck moves upstream to learned repair ranking / Shape-state semantics / target locality / causal context. `SIGNED_MAE_BAYES_GRID_V1` remains accepted; no more selector/tau/threshold rescue is authorized. Independent audit: `experiments/evidence/hch_v3_demo_validation/INDEPENDENT_V3_1_TAIL_FIRST_CROSSFIT_AUDIT_20260904.md`.
- Artifacts: `experiments/evidence/hch_v3_demo_validation/v3_1_tail_first_d2_validation/`; cross-fit result SHA256 `c7159fa3971f958eb9c5eb64897412bd4729c6eecdbe8c5a37580e836132506b`; bootstrap SHA256 `2dc028ba897b29d1aadbbb6b246e033e94b38cf317fc47f496fda6492ed58e1b`.
- Controlling hashes: MAE-action design `6f4503059b63389c0806a58d2e64f5a4267a5c253a3d504a8032d429300ec174`; tail design `3e8ec4178317e16da55e5903adc82df76b5a3014ea6de4e5290e8b09e3b8ddbc`; tail prompt `730987f6be12bda95a91a7c808a7ec2d678b1b7804d09b926ef0d20b25ea6df4`.


