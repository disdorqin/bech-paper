# AI SCIENTIFIC EXECUTION PROMPT — Signed-Mass China-5 × 4-Host Development Experiment

Date: 2026-09-12
Repository: `D:\作业\science\solar_leak_price_model`
Authorization: **explicitly granted by the user for the development experiment**
Authorization token: `SIGNED_MASS_METHOD_EXECUTION_AUTHORIZED`
Protected/final authorization: **NOT granted**

## 0. Mission

Run the first real scientific experiment for the current signed-mass Shape/Amplitude repair method on the complete domestic development panel:

- 5 markets:
  - `GANSU_DA`
  - `SHANDONG_DA`
  - `SHAANXI_DA`
  - `NINGXIA_DA`
  - `QINGHAI_DA`
- 4 frozen Hosts:
  - `PatchTST`
  - `TimeMixer`
  - `iTransformer`
  - `LSTM`

Total scientific panel: **20 market×Host cells**.

Baseline experiments are already frozen. Do not retrain any Host or baseline. The purpose of this run is to test the signed-mass module, adjudicate the four preregistered removable components, freeze the smallest supported method, and compare that frozen method against the already-frozen strict offline baseline table.

This prompt authorizes `DEV_EVAL` use for this development experiment only. `PROTECTED_FINAL` remains structurally sealed with read count exactly zero.

---

## 1. Mandatory read order

Before editing code or executing the scientific run, read:

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `experiments/AGENTS.md`
5. `experiments/STAGE_INDEX.md`
6. `src/AGENTS.md`
7. `src/core/CORE_READINESS_AUDIT_20260912.md`
8. `src/core/DESIGN_CONTRACT.md`
9. `src/core/README.md`
10. `docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`
11. `docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`
12. `experiments/current/hch_signed_mass_method_entry/PROTOCOL.md`
13. `experiments/current/hch_signed_mass_method_entry/PREEXECUTION_AUDIT.md`
14. `experiments/current/hch_signed_mass_method_entry/PREEXECUTION_READINESS.json`
15. `experiments/evidence/hch_china_host_breadth_expansion_20260912/NEW_WINDOW_HANDOFF.md`
16. `experiments/evidence/hch_china_host_breadth_expansion_20260912/04_audit/FINAL_AUDIT.md`

The latest controlling docs override older discussion text.

---

# PART A — Mandatory scientific-execution contract patch BEFORE DEV_EVAL

The harness landing is structurally sound, but the final audit found several scientific-evidence gaps that must be repaired before the first registered `DEV_EVAL` result is opened.

These are implementation-to-contract corrections, not outcome-driven method changes. No `DEV_EVAL` result exists yet.

## A1. Preserve and evaluate DIRECT branch outputs

Current final correction is:

\[
\hat c_h=\hat A^+\hat S_h^+-\hat A^-\hat S_h^-.
\]

Positive and negative predicted Shapes can overlap at the same horizon and cancel in `c`. Therefore it is scientifically wrong to infer branch mechanism metrics by decomposing the already-cancelled net correction.

Modify the harness inference path so every prediction returns and can persist:

- `shape_positive` = direct model `S_hat+`, shape `(N,H)`;
- `shape_negative` = direct model `S_hat-`, shape `(N,H)`;
- `mass_positive` = direct model `A_hat+`, shape `(N,)`;
- `mass_negative` = direct model `A_hat-`, shape `(N,)`;
- `correction` = raw signed-mass correction;
- `prediction` = calibrated Host + alpha×correction;
- `host`;
- `target`;
- `valid_mask`;
- stable episode/day/timestamp identity.

Update `training.predict_corrections`, `oof.FittedMethod.predict`, and `metrics.compute_metrics` or an equivalent clean interface.

Mechanism metrics MUST compare realised signed-mass targets against these direct branch outputs.

Do not change the mathematical fusion.

## A2. Implement the already-designed robust Amplitude output scale `s_A`

The design authority specifies:

\[
\hat A^+=s_A\,softplus(f_+(z)),\qquad
\hat A^-=s_A\,softplus(f_-(z)).
\]

The harness currently leaves `amplitude_scale=1.0`, which does not implement the design for daily residual masses that can be hundreds or thousands in price×hour units.

For each legal fitting prefix, compute exactly one shared training-only robust output scale:

\[
\boxed{
s_A=median\{A_d^+,A_d^-:A_d^\pm>0,\ d\in\mathcal T\}
}
\]

where `T` is that fold's inner-training prefix. Use a fixed numerical floor only if the set is empty/degenerate.

Rules:

- each OOF fold fits `s_A` on its own inner-training prefix;
- final re-fit fits `s_A` once on all `POST_TRAIN`;
- both positive and negative heads use the SAME `s_A`;
- `DEV_EVAL` never contributes;
- it is not tuned and not an ablation switch;
- persist it in `FittedStats` and evidence.

Pass this scale into the existing core `amplitude_scale` configuration. Do not redesign `src/core` unless a tiny interface repair is strictly required; prefer harness wiring because core already exposes the scale.

## A3. Freeze high-mass mechanism thresholds from POST_TRAIN only

A4 rare-mass adjudication requires direct high-mass Amplitude metrics.

For every market×Host cell define once, before any `DEV_EVAL` metric is computed:

\[
q^{A+}_{0.90}=Q_{0.90}(A^+\mid POST\_TRAIN),
\qquad
q^{A-}_{0.90}=Q_{0.90}(A^-\mid POST\_TRAIN).
\]

These thresholds are Host-specific because the residual is Host-specific, but the rule `q90` is global and identical everywhere.

Persist both values and their fit-row hash.

On `DEV_EVAL`, report:

- `high_mass_positive_l1` using direct `A_hat+` on days where realised `A+ >= q90(A+)`;
- `high_mass_negative_l1` using direct `A_hat-` on days where realised `A- >= q90(A-)`;
- subset day counts.

If a DEV subset is empty, record `NOT_APPLICABLE_N0`; never invent a value.

## A4. Save raw prediction/branch evidence

The scientific protocol requires raw evidence, not metrics-only output.

For every `(market, host, config, seed)` evaluated on `DEV_EVAL`, persist an immutable raw artifact containing at minimum:

- episode/day identity;
- target timestamps if available;
- target;
- frozen Host prediction;
- direct `S_hat+`, `S_hat-`;
- direct `A_hat+`, `A_hat-`;
- raw correction;
- alpha;
- final repaired prediction;
- valid mask;
- config switches;
- seed;
- frozen Host artifact hash;
- dataset-contract hash;
- source hash;
- core provenance hash / git state.

Create/update `RAW_PREDICTION_INDEX.csv` with content hashes.

Never overwrite frozen Host/baseline evidence.

## A5. Implement D0 report generation

Add a pure no-training D0 analysis using **POST_TRAIN only** for all 20 cells.

Required outputs under `01_d0_geometry/`:

- `D0_BY_CELL.csv`
- `D0_SUMMARY.md`
- any machine-readable JSON needed for verification

Per cell report at least:

- `A+`, `A-` mean/median/IQR/std;
- zero/near-zero mass fractions;
- positive/negative asymmetry;
- defined Shape-day counts;
- Shape entropy/concentration;
- dominant peak positions;
- adjacent-day W1 vs deterministic random-day W1 as a descriptive similarity diagnostic only;
- relation between residual mass and frozen extreme/tail/negative-price subsets;
- max exact reconstruction error of `r = A+S+ - A-S-`.

D0 does NOT enable KNN.

D0 is allowed to stop execution only for a structural/provenance/algebra failure (e.g. reconstruction inconsistency, wrong roles, illegal reads, no usable residual object). Do not invent a post-hoc numerical novelty gate from D0.

## A6. Add post-run aggregation and independent verification

Implement a scientific-results aggregator and an independent read-only verifier.

Required root artifacts after execution:

```text
experiments/evidence/hch_signed_mass_method_20260912/
├── 00_protocol/
├── 01_d0_geometry/
├── 02_crossfit/
├── 03_component_screen/
├── 04_frozen_method/
├── 05_baseline_comparison/
├── 06_audits/
├── METHOD_CONFIG.json
├── COMPONENT_DECISIONS.csv
├── CELL_METRICS.csv
├── RAW_PREDICTION_INDEX.csv
├── METHOD_SUMMARY.md
├── FINAL_AUDIT.md
├── VERDICT.json
└── NEW_WINDOW_HANDOFF.md
```

The independent verifier must recompute from raw artifacts:

- all primary metrics;
- all direct branch mechanism metrics;
- component ON/OFF deltas;
- final paper-candidate gates;
- baseline joins;
- 20-cell coverage;
- zero protected/final reads.

It must not import a precomputed verdict and merely echo it.

---

# PART B — Re-run pre-execution validation after the patch

Before any registered scientific run:

1. extend tests for A1–A6 above;
2. rerun harness tests;
3. rerun relevant signed-mass core purity/layout tests;
4. rerun `runner.py verify`;
5. verify all 20 cells remain `READY_FROZEN`;
6. verify all five registered configurations differ exactly as intended;
7. verify KNN is OFF;
8. verify no learned Gate/Bridge/attention/Transformer/MoE/router appears;
9. verify `PROTECTED_FINAL` read count = 0;
10. verify no Host/baseline artifact changed hash.

Known unrelated historical foundation collection/layout failures disclosed in the landing report are not a reason to modify historical evidence during this task. Do not repair unrelated archived-layout drift here.

If this mandatory patch or verifier fails, STOP with:

`HCH_SIGNED_MASS_EXECUTION_INVALID`

Do not open `DEV_EVAL`.

If it passes, scientific execution is authorized below. No further user confirmation is required.

---

# PART C — D0 no-training analysis on all 20 cells

Run D0 for:

\[
5\ markets \times 4\ Hosts = 20\ cells.
\]

Use `POST_TRAIN` only.

Write all D0 evidence before the network run begins.

Confirm again after D0:

- `PROTECTED_FINAL` read count = 0;
- `DEV_EVAL` has not yet been used by D0;
- no Host/baseline was retrained.

Then proceed unless D0 uncovered a structural invalidity.

---

# PART D — Full 20-cell component experiment

The user explicitly requires all four Hosts in the scientific experiment. The controlling method protocol was amended **before any signed-mass DEV result existed** so component adjudication uses the full 20-cell panel.

Run all five registered configurations on all 20 cells and all three seeds:

- `FULL`
- `NO_SHAPE_CONTEXT`
- `NO_TCN`
- `TIED_AMPLITUDE`
- `NO_RARE_MASS`

Seeds:

- 7
- 17
- 37

Thus the initial registered grid contains:

\[
20\ cells \times 5\ configs \times 3\ seeds = 300\ final\ cell/config/seed\ fits,
\]

plus the required chronological OOF fold fits inside each run.

Use the full `DEV_EVAL` partition of every cell exactly once for this registered configuration grid. A clean `--dev-eval-all` interface may be added; otherwise use the existing global cap only if it provably includes every cell's complete DEV partition and record the actual per-cell counts.

Authorization token for this development execution:

`SIGNED_MASS_METHOD_EXECUTION_AUTHORIZED`

The token does **not** authorize `PROTECTED_FINAL`.

---

# PART E — Component adjudication: fail => delete

Aggregate seeds by **cell median** first. Do not choose a best seed.

Each ablation is compared with `FULL` and changes exactly one factor.

## E1 Shape semantic context

ON = `FULL`
OFF = `NO_SHAPE_CONTEXT`

Primary mechanism:
- direct branch Shape W1, positive and negative reported separately;
- use the mean of available positive/negative W1 values only as a compact cell summary.

Retain only if the controlling protocol's five conditions pass across the 20 cells.

## E2 TCN

ON = `FULL`
OFF = `NO_TCN`

Primary difficult-region metric:
- Tail-MAE.

Diagnostics:
- direct Shape W1;
- direct positive/negative mass L1;
- Overall-MAE.

TCN is an executor, not a contribution. If evidence is not consistently favorable, delete it and use MLP→GRU.

## E3 Untied positive/negative Amplitude heads

ON = `FULL`
OFF = `TIED_AMPLITUDE`

Primary mechanism:
- direct `mass_positive_l1`;
- direct `mass_negative_l1`.

Also report upper-tail and lower-tail MAE separately.

If unsupported, use the tied head. Do not replace it with two expert networks.

## E4 Rare-mass spread batching

ON = `FULL`
OFF = `NO_RARE_MASS`

Primary mechanism:
- direct `high_mass_positive_l1`;
- direct `high_mass_negative_l1`;

Also report Tail-MAE, negative-price MAE where applicable and Normal-region harm.

If unsupported, remove it. Do NOT proceed to duplicate oversampling in this stage.

## Common retention rule

Use the controlling protocol exactly:

1. targeted mechanism improves in a majority of the 20 cell medians;
2. median relevant Tail/upper/lower metric improves;
3. median Overall-MAE is non-worse;
4. no new cell incurs >1% Normal-region relative harm solely from that component;
5. effect is not a one-cell/one-seed artifact.

Because cell medians are taken over the fixed three seeds and a strict majority is >10/20 cells, do not add a new ad-hoc significance gate.

Write `COMPONENT_DECISIONS.csv` with evidence columns, not just KEEP/DROP.

**If a component fails, delete it. No rescue variant.**

---

# PART F — Freeze and confirm the smallest surviving method

Construct exactly one final switch vector containing only retained components.

Examples:

- if all pass -> `FULL`;
- if TCN fails only -> `shape_semantic_context=1, use_tcn=0, untied_amplitude_heads=1, rare_mass_sampling=1`;
- if several fail -> turn all failed switches OFF simultaneously.

Do not add any new component.

Run the smallest surviving configuration on all 20 cells × seeds `{7,17,37}` as the frozen-method confirmation.

If it is byte/config-identical to an already executed registered configuration, you may reuse the already persisted raw predictions only if an independent verifier confirms exact identity; do not retrain merely for table cosmetics. If several deletions produce a new switch combination, train that one confirmation configuration normally under the same frozen recipe.

Persist it under `04_frozen_method/`.

---

# PART G — Compare against the already-frozen baselines

Do NOT rerun any comparator.

Authoritative strict offline table:

`experiments/evidence/hch_china_host_breadth_expansion_20260912/03_results/STRICT_OFFLINE_TABLE.csv`

Authoritative supplementary online table:

`experiments/evidence/hch_china_host_breadth_expansion_20260912/03_results/ONLINE_SUPPLEMENTARY_TABLE.csv`

Strict offline numeric rows include:

- Host: 20/20;
- MatchedDirectResidual: 20/20 (project control);
- δ-Adapter: 20/20, with its existing fidelity disclosure;
- PIR: 10/20 only; the iTransformer/LSTM coordinates remain `Q-INCOMPATIBLE` and must stay blocked.

UEC-STD and OMPB remain blocked and receive no invented values.
COSA is `ONLINE_TTA` supplementary and must never enter the strict offline winner rank.

For each of the 20 cells, compare the cell-median frozen signed-mass method against the best **numeric, setting-matched strict offline** comparator available in that cell. Preserve fidelity labels and blockers.

Required table:

`05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.csv`

Also create a supplementary view that places Our Method beside COSA without pooling the rankings, clearly labeled as different execution settings.

Important inference-setting disclosure for Our Method:

- model parameters remain frozen on DEV;
- no gradient/update/TTA occurs during DEV;
- each day may use only historically revealed residual windows from strictly earlier eligible origins;
- therefore describe it as **frozen-parameter causal/prequential-history post-processing**, not online TTA.

Do not mislabel it as COSA-style online adaptation.

---

# PART H — Development paper-candidate gate

Use the frozen smallest method, aggregated by cell median over seeds.

Across all 20 eligible cells require:

1. Host-nonworse in at least 19/20 cells;
2. median Overall-MAE gain vs Host >= 3%;
3. strict win vs best available setting-matched strict offline comparator in at least 14/20 cells;
4. median Tail-MAE gain vs Host >= 5% among cells with nonempty frozen tail subsets;
5. maximum cell-median Normal-region relative harm <= 1%;
6. identical scientific recipe across all markets/Hosts; only legal feature width may differ by dataset contract.

These are development gates, not final claims.

If the smallest surviving method fails this gate:

`HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`

STOP. Do not add modules, change losses, tune per market, relax thresholds or open protected/final.

If it passes:

`HCH_SIGNED_MASS_DEVELOPMENT_CANDIDATE_SUPPORTED_READY_FOR_HUMAN_FREEZE`

STOP. Passing does not authorize protected/final.

Any information-boundary/provenance failure:

`HCH_SIGNED_MASS_EXECUTION_INVALID`

---

# PART I — Evidence and verification requirements

Final package must contain at least:

- complete D0 report;
- 300 registered component-grid results or explicit invalid blockers;
- direct branch raw outputs;
- OOF alpha provenance;
- training-prefix `s_A` provenance;
- high-mass q90 provenance;
- per-seed metrics;
- cell-median metrics;
- component decisions;
- smallest-method config;
- strict baseline comparison;
- raw prediction index;
- parameter count;
- training/inference runtime/overhead;
- independent recomputation audit;
- zero protected/final reads;
- no Host/baseline retraining proof.

Do not report only a headline average.

The summary must explicitly show:

1. all 20 market×Host cells;
2. Overall MAE and gain vs Host;
3. Tail / upper / lower / normal metrics;
4. negative-price metrics where applicable;
5. best strict comparator and margin to it;
6. which of the four components survived and why;
7. failures by market and by Host;
8. whether the development paper-candidate gate passed.

---

# PART J — State files

This is an important scientific experiment. Before stopping, update:

- `EXPERIMENT_LEDGER.md` with the experiment and exact verdict;
- `RESEARCH_STATE.md` with supported/rejected components and the current bottleneck;
- `HANDOFF.md` with evidence paths, frozen smallest configuration and next allowed action.

Do not write a paper claim stronger than the evidence.

---

## Final instruction

This task is authorized to perform the complete signed-mass **development** experiment after the mandatory pre-run patch and verifier PASS.

Do not ask the user for another confirmation.
Do not stop after PatchTST/TimeMixer.
Do not omit iTransformer/LSTM.
Do not open `PROTECTED_FINAL`.
Do not rescue failed components.

The scientific unit of evidence is the full **5-market × 4-Host = 20-cell** panel.
