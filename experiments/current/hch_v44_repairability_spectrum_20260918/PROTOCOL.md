# HCH v4.4 Repairability Spectrum Diagnostic Protocol

STATUS:        ACTIVE
STAGE:         hch_v44_repairability_spectrum_20260918
KIND:          protocol
SUPERSEDED_BY: -

Authority:
docs/current/HCH_V44_REPAIRABILITY_SPECTRUM_DIAGNOSTIC_20260918.md

## 1. Scope

Diagnostic only. No new neural candidate, no optimizer step, no core edit, no O1 retraining, no TEST access.

Use the completed domestic O1 evidence:
experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/

Reuse all 60 selected O1 checkpoints/read-only run records and their frozen TRAIN/VAL support.

## 2. Access contract

Allowed:
- domestic TRAIN and VAL labels;
- frozen Host TRAIN/VAL predictions;
- legal W=7 revealed residual-history support;
- selected O1 EMA checkpoints;
- current stage design/protocol and canonical state.

Forbidden:
- V2 TEST target/prediction/metric;
- protected/final targets;
- paper-facing TEST baseline tables for selection;
- target-day realized covariates;
- market/province IDs as probe features;
- mutation of src/core, src/backbones, paper, prior evidence or checkpoints.

All reads must be counted.

## 3. R0 exact headroom

For every eligible TRAIN/VAL day:
- reconstruct exact b*, B*, S+*, S-* from Host residual;
- reconstruct O1 predicted coordinates and correction;
- compute residual geometry descriptors from the design;
- compute O_b, O_B, O_S, O_bB, O_bS, O_BS oracle swaps;
- compute exact nonnegative per-day MAE-optimal ray scalar alpha* for the frozen O1 correction.

Ray minimization must be deterministic and independently verified against brute-force or weighted-median characterization on a deterministic sample.

Do not clip negative oracle-swap improvements.

## 4. R1 legal-history evidence

Use exactly the descriptor schema preregistered in the design.

No feature may be added after seeing associations.

For every target day with W=7 support:
- compare current Host absolute/relative views to each legal historical Host trajectory;
- identify the nearest historical day independently under absolute and relative Host similarity;
- compare that historical day's residual b/B/Shape with the target;
- compute the oracle best history analogue only as a target-aware upper bound;
- compare against the current W=7 summary.

No K search, no learned distance, no temperature, no retrieval model.

## 5. R2 fixed low-capacity probes

Five leave-one-market-out folds.

Features:
- only the fixed legal descriptors;
- no market ID;
- no Host-family ID;
- scaling fit on each training fold only.

Model:
- Ridge(alpha=1.0) only;
- no hyperparameter search;
- macro-balance cells in the training loss/data weights.

Targets:
1. log1p(alpha*);
2. ray-oracle headroom / Host MAE;
3. B-swap headroom / Host MAE;
4. Shape-swap headroom / Host MAE.

Metrics:
- held-out-market Spearman rho;
- MAE;
- constant-training-median MAE;
- R2;
- top-vs-bottom quartile target difference.

Separately evaluate frozen O1 correction ratio as a one-scalar predictor of realized O1 utility. It is descriptive only.

## 6. Diagnosis

Use the exact taxonomy in the design:
LOW_HEADROOM / DISTANCE_LIMITED / SHAPE_LIMITED / BALANCED_MASS_LIMITED / LEVEL_LIMITED / MIXED_HEADROOM.

The seven current low-gain cells are fixed from the already-completed domestic result:
- GANSU_DA/TimeMixer
- GANSU_DA/iTransformer
- SHANDONG_DA/TimeMixer
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/iTransformer
- SHAANXI_DA/LSTM

Do not change this set.

## 7. Future-design gate

Recompute A-E exactly from the design.

The gate can only say whether a future main-window discussion of one bounded cross-market conditioning mechanism is evidence-supported. It cannot create or train that mechanism.

## 8. International handling

No foreign fit and no foreign preflight in this stage.

The earlier out-of-order LAGO preflight is historical evidence only. Do not extend it.

## 9. Evidence and verification

New root:
experiments/evidence/hch_v44_repairability_spectrum_20260918/

Stage-local implementation/verification only:
experiments/current/hch_v44_repairability_spectrum_20260918/implementation/
experiments/current/hch_v44_repairability_spectrum_20260918/verification/

Required artifacts are those listed in the design.

The independent verifier must not import the runner. It must:
- recompute a deterministic sample of exact geometry and decoder identity;
- recompute all oracle swaps;
- verify ray minimizer;
- verify descriptor legality;
- verify leave-one-market-out exclusion;
- verify no market/Host ID probe features;
- recompute diagnosis and gate A-E;
- verify TEST/protected/final reads = 0;
- verify all source/checkpoint/prior evidence hashes unchanged.

## 10. Terminal tokens

Exactly one:
HCH_V44_REPAIRABILITY_SPECTRUM_COMPLETE_FOR_ADJUDICATION
HCH_V44_REPAIRABILITY_SPECTRUM_BLOCKED_<REASON>

## 11. Return format

At most 110 lines:
1. terminal token;
2. reuse/access/zero-fit counts;
3. one 20-cell table: current O1 gain, ray/b/B/Shape oracle headroom, diagnosis;
4. summary of the seven low-gain cells;
5. Host-to-residual analogue results overall + by market/Host;
6. LOMO probe table for four targets across five held-out markets;
7. future-design gate A-E;
8. one concise mechanism interpretation;
9. verifier token and evidence root.

At most two Markdown tables. No new architecture, no successor experiment, no TEST claim.
