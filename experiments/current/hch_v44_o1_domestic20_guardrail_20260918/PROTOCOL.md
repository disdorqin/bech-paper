# HCH v4.4 O1 Domestic-20 + International Guardrail Protocol

STATUS:        ACTIVE
STAGE:         hch_v44_o1_domestic20_guardrail_20260918
KIND:          protocol
SUPERSEDED_BY: -

Scientific authority:
`docs/current/HCH_V44_O1_DOMESTIC20_INTERNATIONAL_GUARDRAIL_DESIGN_20260918.md`.

This protocol is executable only as written. It contains three serial phases:
D = China-5 × 4-Host O1 breadth, M = zero-fit gradient audit, I = conditional international guardrail.
Phase I is forbidden unless Phase D passes its preregistered gate.

## 1. Immutable candidate recipe

Recipe ID: `O1_AUX_WARMUP_400_THEN_MAE`.

- architecture: current v4.4 G2 exact geometry candidate;
- inputs: frozen Host + deterministic calendar + W=7 strictly revealed residual-history evidence only;
- optimizer: AdamW;
- lr: 1e-3;
- batch size: 32;
- weight decay: 1e-4;
- grad clip: 1.0;
- EMA: 0.995, EMA-primary evaluation;
- max optimizer steps: 2000;
- VAL every 50 steps;
- no stop before 800;
- patience: 8 VAL checks;
- step 0 is legal;
- updates 1..400: `L_rec + (L_b+L_B+L_S)/3`;
- updates 401..2000: `L_rec`;
- seeds: 7,17,37.

Forbidden: switch-step tuning, loss-weight tuning, lr tuning, initialization changes, architecture/router edits, feature additions, component-loss ablations, PCGrad/GradNorm/SAM, per-market recipes, Host/baseline retraining except a support-only history mechanism explicitly required by a legal preflight.

## 2. Global access boundary

V2 TEST is contaminated development history and remains quarantined.

Never read:
- V2 TEST targets;
- V2 TEST HCH predictions;
- V2 TEST metrics;
- V2 TEST baseline comparison tables used for model selection;
- protected/final targets in international markets.

The executor may read canonical state, controlling design/protocol, TRAIN/VAL or DIAG_FIT/DIAG_EVAL support, frozen Host predictions needed for those legal roles, and frozen baseline point estimates only in Phase I after Phase D passes.

All access must be audited. A forbidden read is terminal.

## 3. Phase D — domestic full 20-cell breadth

### 3.1 Reuse

Reuse by hash, never retrain, all 24 O1 runs from these eight cells:
- GANSU_DA/PatchTST
- GANSU_DA/LSTM
- SHANDONG_DA/PatchTST
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/PatchTST
- NINGXIA_DA/iTransformer
- QINGHAI_DA/TimeMixer

Expected source roots:
- `experiments/evidence/hch_v44_objective_alignment_probe_20260917/`
- `experiments/evidence/hch_v44_o1_full8_completion_20260918/`

### 3.2 New fits

Exactly these 12 missing cells × 3 seeds = 36 fits:
- GANSU_DA/TimeMixer
- GANSU_DA/iTransformer
- SHANDONG_DA/TimeMixer
- SHANDONG_DA/LSTM
- SHAANXI_DA/iTransformer
- SHAANXI_DA/LSTM
- NINGXIA_DA/PatchTST
- NINGXIA_DA/TimeMixer
- NINGXIA_DA/LSTM
- QINGHAI_DA/PatchTST
- QINGHAI_DA/iTransformer
- QINGHAI_DA/LSTM

The new runs must execute the same O1 trainer bytes by import if possible. If a compatibility wrapper is required, it must be stage-local and must not change scientific computation.

### 3.3 Domestic aggregation

Primary cell statistic:
`cell_o1_mae = median(seed7, seed17, seed37 O1 VAL MAE)`.

Host is deterministic per cell:
`gain_host_pct = 100 * (host_mae - cell_o1_mae) / host_mae`.

Primary panel statistics are formed from these 20 cell-level values.

Paired seed-level gains are supplementary only.

### 3.4 Domestic gate

D0 validity PASS iff:
- exactly 60 O1 coordinates = 24 reused + 36 new;
- no existing O1 run retrained;
- one scientific code state;
- same split/Host/history/scales support as frozen v4.4 substrate;
- all metrics finite;
- no retry or setting chosen from outcomes;
- TEST reads = 0.

D1 breadth PASS iff:
- strict Host-positive cells >=16/20;
- cells with gain >=-0.5% >=19/20;
- worst gain >=-1.0%.

D2 materiality PASS iff:
- panel median gain >=+1.5%;
- cells with gain >=+2.0% >=8/20.

D3 market coverage PASS iff:
- 5/5 market medians >0;
- >=4/5 market medians >=+1.0%.

D4 seed stability PASS iff:
- >=15/20 cells have at least 2/3 seeds beating Host;
- 0 cells have all three seeds worse than Host by >1%.

Domestic PASS requires D0-D4 all PASS.

On Domestic FAIL:
- still run Phase M;
- do not run Phase I;
- stop after verification with `HCH_V44_O1_DOMESTIC20_NOT_READY`.

## 4. Phase M — zero-fit shared-gradient audit

Run after D on all 20 domestic selected O1 checkpoints. It is descriptive and cannot modify D0-D4.

Use the same fixed TRAIN diagnostic subset as the O1 training evidence.

For each checkpoint:
1. load EMA-selected weights;
2. forward once;
3. independently backprop each of `L_rec, L_b, L_B, L_S` without optimizer.step;
4. vectorize gradients only on parameters shared by the coordinate tasks (shared Host/history/calendar encoder, shared allocator/context representation, shared Shape trunk where mathematically shared; exclude parameters belonging only to a single output head);
5. compute:
   - cos(rec,b)
   - cos(rec,B)
   - cos(rec,S)
   - cos(rec,aux_sum), where aux_sum=(g_b+g_B+g_S)/3;
   - norm ratios ||g_b||/||g_rec||, ||g_B||/||g_rec||, ||g_S||/||g_rec||.

Record zero optimizer steps and unchanged checkpoint/source hashes.

Report fractions of negative cosines overall, by Host family and by market. No gate.

## 5. Phase I — conditional international guardrail

Only legal when D0-D4 all PASS.

### 5.1 Fixed cells

Exactly:
- LAGO_DE/PatchTST
- LAGO_DE/TimeMixer
- LAGO_PJM/PatchTST
- LAGO_PJM/TimeMixer

O1 seeds 7,17,37 = 12 fits.

### 5.2 Preflight before any fit

Verify:
- exact historical DIAG_FIT/DIAG_EVAL identities and Host hashes;
- O1-compatible legal W=7 history support;
- no protected/final outcome access;
- existing admitted baseline rows and fidelity labels from the frozen 2026-09-11 transfer substrate, using only `experiments/evidence/hch_frozen_method_baseline_transfer_20260911/{baseline_admission_matrix.csv,best_baseline_gap_by_cell.csv,metrics_by_cell.csv,provenance.json,transfer_fidelity_audit.md}` as the comparison authority;
- no proxy baseline is substituted and no baseline is retrained.

If legal O1 history support is absent and cannot be reconstructed from already-authorized pre-eval support without changing the information setting, stop Phase I with an explicit blocker. Do not invent a workaround.

### 5.3 International metrics

HCH/O1 cell MAE = median of its three seed MAEs.

BestBaselineMAE = minimum frozen admitted offline baseline MAE under the accepted transfer protocol.

`gap_best_pct = 100*(o1_mae-best_baseline_mae)/best_baseline_mae`.

Also report Host-relative Overall-MAE gain and Normal-MAE harm vs Host.

### 5.4 International gate

I0 validity:
- exact four cells and 12 fits;
- O1 identical to domestic candidate;
- accepted Host/split/baseline provenance;
- no DIAG_EVAL setting choice;
- protected/final reads = 0.

I1 best-baseline proximity:
- gap <=1.0% in >=3/4;
- gap <=2.0% in 4/4.

I2 Host safety:
- Host-positive in >=3/4;
- every cell Host-relative gain >=-1.0%.

I3 normal safety:
- Normal-MAE harm vs Host <=1.0% in 4/4.

I4 no leakage/numerical/protocol violation.

International PASS requires I0-I4 all PASS.

No international result authorizes foreign-specific tuning.

## 6. Decision tokens

Use exactly one:

- `HCH_V44_O1_DOMESTIC20_NOT_READY`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_BLOCKED_<REASON>`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_NOT_MET`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_PASS`
- `HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_<REASON>`

The PASS token authorizes no TEST opening. It means only that the frozen O1 recipe is ready for a separately designed untouched confirmation stage.

## 7. Evidence root

Write only new evidence to:
`experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/`.

Stage-local code may be written only under:
`experiments/current/hch_v44_o1_domestic20_guardrail_20260918/implementation/**`
and
`experiments/current/hch_v44_o1_domestic20_guardrail_20260918/verification/**`.

Do not edit `src/core/**`, `src/backbones/**`, frozen prior evidence, baseline result tables, or `paper/**`.

Minimum artifacts:
- ACCESS_AUDIT.json
- REUSE_PROVENANCE.json
- DOMESTIC_PER_SEED.csv
- DOMESTIC_CELL_MEDIANS.csv
- DOMESTIC_MARKET_SUMMARY.csv
- DOMESTIC_GATE.json
- GRADIENT_CONFLICT_AUDIT.csv
- GRADIENT_CONFLICT_SUMMARY.json
- INTERNATIONAL_PREFLIGHT.json if reached
- INTERNATIONAL_CELL_COMPARISON.csv if reached
- INTERNATIONAL_GATE.json if reached
- INDEPENDENT_VERIFICATION_REPORT.json
- RESULTS.md
- STAGE_TOKEN.json

The verifier must not import the runner and must recompute all gate arithmetic from raw written rows.

## 8. Return format

Return at most 110 lines and only:
1. terminal token;
2. Phase-D new/reused fit counts;
3. TEST/protected/final read counts;
4. one domestic 20-cell table with O1 median MAE, Host MAE, gain, seed wins;
5. D0-D4 results plus panel/market summaries;
6. Phase-M gradient-conflict summary;
7. if Phase I ran, one four-cell table with O1, Host, best baseline, gap and I0-I4; otherwise exact skip/block reason;
8. evidence root and independent-verifier token;
9. one final sentence stating only whether the frozen O1 recipe is ready for separately authorized untouched confirmation.

Line budget: items 1-3 one line each; item 4 may use up to 24 lines including header; item 5 up to 8 lines; item 6 up to 8 lines; item 7 up to 18 lines including its optional table; items 8-9 up to 3 lines each. Total hard cap: 110 lines.

At most two Markdown tables. No process narrative. No architecture proposal. No hyperparameter proposal. No paper claim.

Terminal token must be exactly one of:
- `HCH_V44_O1_DOMESTIC20_NOT_READY`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_BLOCKED_<REASON>`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_NOT_MET`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_PASS`
- `HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_<REASON>`
