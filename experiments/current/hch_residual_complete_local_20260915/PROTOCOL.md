# HCH Residual-Complete Local Mechanism Protocol

Protocol ID: `HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_20260915`

Status: **PREPARED / NOT YET EXECUTION-AUTHORIZED / TRAIN+VAL DEVELOPMENT ONLY**

## 1. Purpose

Test the first-principles successor to v4.0:

> Freeze a coarse Level predictor first; then make Local learn **all residual actually left by that predictor**, rather than an oracle-centered residual or a hidden prediction-conditioned target.

The primary candidate is:

\[
\hat c=\hat b\mathbf1+
\hat\delta\mathbf1+
\hat A(\hat S^+-\hat S^-).
\]

No TEST/foreign/final access is permitted under this prepared protocol.

## 2. Scientific hypotheses

### H1 — matched prediction-conditioning

TRAIN Local target must be defined by the same causal Level predictor family that is explicitly observed by Local and later used at inference.

### H2 — residual completeness

For

\[
q=r-\tilde b\mathbf1,
\]

Local must represent arbitrary `q`, not enforce `sum q=0`.

### H3 — orthogonal closure is better conditioned than M-kappa

Use

\[
\delta=mean(q),
\qquad
\rho=q-\delta\mathbf1,
\]

\[
q=\delta\mathbf1+A(S^+-S^-),
\]

so remainder mean, centered magnitude and signed location are separately identified.

## 3. Fixed cells

Use exactly the same 8-cell v4.0 development panel:

1. `GANSU_DA / LSTM`
2. `GANSU_DA / TimeMixer`
3. `GANSU_DA / PatchTST`
4. `SHANDONG_DA / TimeMixer`
5. `SHANDONG_DA / iTransformer`
6. `SHAANXI_DA / TimeMixer`
7. `SHAANXI_DA / PatchTST`
8. `NINGXIA_DA / LSTM`

Seeds: `7,17,37`.

QINGHAI is excluded from this mechanism stage.

## 4. Data boundary

Use `COMMON_BENCHMARK_701020_FULL_V2` only.

- TRAIN: Stage-1 cross-fitting + final fit + Local training support.
- VAL: chronological 50/50 CAL/EVAL.
- TEST: sealed, mandatory `test_label_read_count=0`.
- No Host retraining.
- No baseline rerun.
- No `src/core` modification.
- No paper modification.

## 5. Stage 1 — Coarse Level

### 5.1 Architecture

Minimal Level-only model:

- deterministic base feature stem;
- existing magnitude-sensitive GRU32;
- one signed scalar readout.

No Shape encoder/head.

### 5.2 Target

\[
b^*=mean(r).
\]

Loss:

\[
L_{level}=|\hat b-b^*|/s_b.
\]

`s_b` is TRAIN-only robust scale.

### 5.3 Causal OOF predictions

TRAIN chronology:

- first 40% = warm-up;
- remaining 60% split into B1/B2/B3/B4;
- each block prediction uses only earlier target days;
- seeds 7/17/37;
- persist `b_tilde`, target IDs, prefix IDs, checkpoint hashes.

`LOCAL_TRAIN = B2∪B3∪B4`.

B1 may be used only as causal Local-history warm-start support.

### 5.4 Final Level predictor

After OOF objects are frozen, refit the exact same Level-only architecture on legal TRAIN support with an inner chronological early-stop tail. Freeze it before any VAL candidate evaluation.

The final Level model is never updated during Local training or VAL inference.

## 6. Stage 2 residual space

For each legal Local TRAIN day:

\[
q_t=r_t-\tilde b_t\mathbf1.
\]

Define:

\[
\delta_t^*=mean(q_t),
\]

\[
\rho_t^*=q_t-\delta_t^*\mathbf1,
\]

\[
A_t^*=\frac12\|\rho_t^*\|_1,
\]

and signed simplex targets from `rho*`.

Exact identity:

\[
q_t=\delta_t^*\mathbf1+A_t^*(S_t^{+*}-S_t^{-*}).
\]

## 7. Local input contract

### 7.1 Current-day conditioning

Local receives the current coarse Level prediction `b_hat` as an explicit scalar condition.

TRAIN: use causal OOF `b_tilde`.

VAL: use final frozen Level prediction.

The scalar must be normalized with TRAIN-only statistics.

### 7.2 History space

Past Local history must be built in post-Level residual space:

\[
q_{t-j}=r_{t-j}-\hat b_{t-j}\mathbf1.
\]

Shape history uses centered:

\[
\rho_{t-j}=q_{t-j}-mean(q_{t-j})\mathbf1.
\]

Magnitude history uses raw `q` channels:

- scaled q;
- scaled |q|;
- scaled q+;
- scaled q-.

No raw-Host residual history may silently substitute for q-history in the proposed RCL candidate.

## 8. Candidate ladder

All four candidates use the same frozen Stage-1 Level predictions, same `LOCAL_TRAIN`, same inputs, same seeds and same optimizer schedule.

### R0 — `STACK_ZERO_SUM`

Local target is centered `rho` only:

\[
\hat z=\hat A(\hat S^+-\hat S^-).
\]

Final:

\[
\hat c=\hat b\mathbf1+\hat z.
\]

This is the matched hard-zero-sum control.

### R1 — `RCL_ORTHOGONAL` (primary)

\[
\hat z=\hat\delta\mathbf1+
\hat A(\hat S^+-\hat S^-).
\]

Final:

\[
\hat c=\hat b\mathbf1+\hat z.
\]

### R2 — `UNTIED_Q`

Direct exact signed-mass parameterization of q:

\[
\hat z=\hat A^+\hat S^+-\hat A^-\hat S^-.
\]

Level remains frozen and external, so this is a legitimate comparator rather than the historical joint untied-head route.

R2 must use the same scalar-degree budget as R1.

### R3 — `DIRECT_Q`

Same Local backbone/conditioning/history, low-capacity direct H-step q residual head.

Parameter count must be <=1.25× R1.

## 9. Local normalization

Fit from legal `LOCAL_TRAIN` q targets only:

- `s_q` from q;
- `delta_stats` from delta;
- `s_A` from centered local mass A;
- for R2, `s_A+` and `s_A-` from q signed masses.

Do not reuse v3.9/v4.0 mass/Level scales when their target geometry differs.

## 10. Local losses

### R0

\[
L=L_{rec}(\rho,\hat z)+mean(L_A,L_S).
\]

### R1

\[
L=L_{rec}(q,\hat z)+mean(L_\delta,L_A,L_S).
\]

where:

\[
L_\delta=|\hat\delta-\delta^*|/s_\delta.
\]

### R2

\[
L=L_{rec}(q,\hat z)+mean(L_{A+},L_{A-},L_S).
\]

### R3

Only normalized q reconstruction MAE.

No tunable loss-weight sweep.

## 11. Training recipe

Unless a correctness issue requires an implementation-only repair, inherit:

- AdamW;
- lr `1e-3`;
- weight decay `1e-4`;
- batch size `32`;
- max epochs `50`;
- patience `8`;
- gradient clip `1.0`;
- seed before model construction;
- one math thread per fit where applicable.

Early stopping for Local uses q reconstruction MAE on an inner chronological tail of `LOCAL_TRAIN`.

## 12. VAL chronology

VAL split chronologically:

- CAL = first 50%;
- EVAL = second 50%.

Primary verdict uses **raw model outputs without dual-alpha rescue**.

A single nonnegative final scalar may be fitted on CAL and reported as a secondary calibration diagnostic only; it cannot turn a failing raw candidate into a primary PASS.

For VAL prequential history:

- initialize with last legal causal q-history from TRAIN/B4;
- for each VAL day: predict Level -> predict Local -> persist -> reveal target -> compute actual q -> append history;
- no future target may enter history.

## 13. Required diagnostics

Per cell×seed:

- Stage-1 Level MAE and zero reference;
- Stage-1 daily-mean residual floor;
- `delta` target/prediction and zero-delta reference;
- `A`, Shape W1;
- q reconstruction MAE;
- full residual MAE;
- Tail/Upper/Lower/Extreme MAE where defined;
- Normal harm;
- raw and optional calibrated metrics;
- parameter count and runtime;
- content/checkpoint hashes.

Aggregate additionally:

\[
closure\_ratio=
\frac{median|b^*-\hat b-\hat\delta|}
{median|b^*-\hat b|}.
\]

## 14. Primary adjudication

### A. Residual-complete benefit: R1 vs R0

Support `RCL_RESIDUAL_COMPLETENESS_SUPPORTED` iff all:

- positive Overall cells >=5/8;
- panel median Overall gain >=0.75%;
- worst degradation >=-0.75%;
- delta beats zero-delta >=5/8 cells;
- panel-median mean-floor reduction >=25%.

Else MIXED / NOT_SUPPORTED.

### B. Coordinate choice: R1 vs R2

If R1 is non-worse in >=5/8 and panel median >=0, prefer orthogonal `delta+A`.

If R2 is better in >=5/8 and median >=0.5%, prefer direct untied signed masses.

Otherwise `LOCAL_PARAMETERIZATION_MIXED`.

### C. Structure: R1 vs R3

If DIRECT_Q wins >=5/8 and panel median >=1%, mark `STRUCTURED_LOCAL_UNDERFIT_RISK`.

If R1 is within 0.5% in >=6/8 and has <= parameter count, mark `STRUCTURED_LOCAL_COMPETITIVE`.

Otherwise MIXED.

### D. Host-relative quality

R1 candidate-quality target:

- positive vs Host >=7/8;
- panel median gain >=1.0%;
- worst loss >=-0.5%.

This is development evidence only.

## 15. Falsification / stop rules

- If R1 does not beat R0: do not add more Local degrees of freedom.
- If R2 beats R1: revisit orthogonal coordinate choice, not network size.
- If R3 clearly beats both: structured signed-simplex Local may underfit; stop shape-parameter polishing.
- If all variants fail: bottleneck is likely post-Level residual observability/information, not zero-sum expressivity.
- No result-conditioned rescue variant in the same run.

## 16. Forbidden additions

No:

- safety controller;
- learned gate/confidence selector;
- retrieval/KNN;
- attention/Transformer/TCN;
- MoE/router;
- market-specific expert;
- extra loss-weight sweep;
- per-market hyperparameter search;
- TEST/foreign/final access;
- `src/core` edit;
- paper update.

## 17. Required terminal state

If later explicitly authorized and executed, stop after independent verification with one terminal token:

- `HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_COMPLETE_FOR_ADJUDICATION`

This token does not authorize TEST, foreign/final, paper promotion or core promotion.
