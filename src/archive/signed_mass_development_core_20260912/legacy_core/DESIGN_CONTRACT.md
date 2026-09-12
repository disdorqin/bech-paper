# Signed-Mass Core Design Contract

Date: 2026-09-12  
Authority: `docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`

This file defines implementation invariants. It is not an experiment result.

## 1. Exact residual object

For residual `r = y - y_host`:

\[
r^+=\max(r,0),\qquad r^-=\max(-r,0),
\]

\[
A^+=\sum_h r_h^+,\qquad A^-=\sum_h r_h^-,
\]

\[
S_h^+=r_h^+/A^+,\qquad S_h^-=r_h^-/A^-.
\]

For nonzero mass directions:

\[
S_h^\pm\ge0,\qquad \sum_hS_h^\pm=1.
\]

Exact reconstruction:

\[
\boxed{r_h=A^+S_h^+-A^-S_h^-}.
\]

Prediction fusion is only

\[
\boxed{c_h=\hat A^+\hat S_h^+-\hat A^-\hat S_h^-}.
\]

No trainable fusion layer may replace this identity.

## 2. Information invariants

### I1 — forecast-origin legality

Current-day inputs may contain only:

- frozen Host forecast;
- audited forecast-known numeric covariates;
- deterministic calendar information;
- availability masks.

Historical input may contain already-revealed original-origin residuals. Target-day actuals/future residuals are forbidden.

### I2 — complete generic feature tensor

All admitted target-day numeric covariates enter through one generic `(B,H,F)` tensor. `src/core` contains no province-specific feature names or feature-count branches.

### I3 — explicit availability

Missing semantic roles are represented by a feature mask. They must never be replaced by Host price or another physical variable.

### I4 — one learned shared stem

All current-day absolute facts pass through exactly one `UnifiedFeatureStem`. There is no per-feature, per-market or per-Host MLP.

### I5 — Shape semantics

Shape receives the shared embedding plus scale-free within-day profiles/ramps. Under positive residual scaling, its exact target remains invariant.

### I6 — Amplitude semantics

Amplitude receives the shared absolute embedding and magnitude-preserving historical residual representation. Under positive residual scaling, its exact target scales linearly.

### I7 — separate historical views

The Shape history and Amplitude history are different tensors. A normalised Shape tensor must never be reused as an Amplitude magnitude proxy.

## 3. Architecture invariants

### I8 — small local + cross-day executor

The default branch pattern is:

```text
1x1 channel projection -> small local TCN -> daily pooling -> GRU32
```

The Shape branch additionally preserves target-day per-horizon local states for its decoder.

### I9 — no hidden-state Bridge

Shape and Amplitude interact through the shared MLP and final differentiable reconstruction, not a cross-branch recurrent/attention block.

### I10 — asymmetric amplitude is minimal

Default Amplitude uses one common trunk plus two untied nonnegative scalar heads. It does not instantiate positive/negative expert encoders.

### I11 — no instance-level proposal selector

The current core contains no learned third network that predicts whether the proposed correction should be applied.

### I12 — optional retrieval is not correction

Similarity/KNN remains disabled by default. If later authorized, it may supply Shape decoder context only and must use strictly revealed permitted history.

## 4. Loss invariants

Shape auxiliary:

\[
W_1(S,\hat S)=\sum_{h=1}^{H-1}|F_S(h)-F_{\hat S}(h)|.
\]

Amplitude auxiliary:

\[
L_A=|A^+-\hat A^+|+|A^--\hat A^-|
\]

(up to one train-frozen numerical normalization).

Repair objective:

\[
L_R=H^{-1}\sum_h|r_h-c_h|.
\]

Default combined objective:

\[
L=L_R+\lambda_SL_S+\lambda_AL_A.
\]

No market-specific loss weights are allowed in the first experiment.

## 5. Gradient coupling

The deterministic fusion gives

\[
\partial c_h/\partial \hat A^+=\hat S_h^+,
\qquad
\partial c_h/\partial \hat S_h^+=\hat A^+,
\]

and analogous negative terms. Therefore large predicted correction mass naturally increases the final-loss sensitivity to Shape placement error. This coupling is sufficient by default; no learned Bridge is justified without new evidence.

## 6. Calibration invariant

For **one fitted market×Host repair model**, chronological OOF candidate/residual coordinates determine

\[
\alpha^*=\arg\min_{\alpha\ge0}\sum_i|r_i-\alpha c_i|.
\]

Implementation:

\[
\alpha^*=\max\left(0,\operatorname{WeightedMedian}(r_i/c_i;|c_i|)\right)
\]

for nonzero candidate coordinates.

The scalar is:

- pooled within that fitted repair model's legal calibration rows;
- never fitted from evaluation/protected labels;
- never per test instance;
- not required to be numerically identical across markets/Hosts.

## 7. Four controlled removal switches

Exactly four first-stage method hypotheses have explicit source switches:

| Hypothesis | Switch | Removal meaning |
|---|---|---|
| Shape privileged task context | `shape_semantic_context` | zero privileged Shape geometry; both tasks rely on shared factual embedding |
| local TCN | `use_tcn` | keep 1x1 channel projection + GRU, remove temporal convolution |
| positive/negative Amplitude asymmetry | `untied_amplitude_heads` | tie both mass readouts to one scalar head |
| rare-mass batch organization | `rare_mass_sampling` | use ordinary training ordering |

These switches are **not** a search grid. The future experiment uses one-factor leave-one-out tests. If a component fails, it is removed instead of tuned/rescued.

KNN is not part of this four-component screen.

## 8. Core purity

`src/core/**` must not contain:

- dataset paths;
- province/market branches;
- Host-name branches;
- baseline implementations;
- experiment registry/evaluation code;
- access to `experiments/`, `paper/`, `docs/` or archived runtime as imports.

Historical V2.5 source remains outside promoted candidate code under `src/archive/core_pre_extreme_repair_20260912/`.

## 9. Scientific status

Passing source/unit tests means the implementation respects the contract. It does **not** mean:

- signed-mass geometry is empirically useful;
- the two semantic views are necessary;
- TCN is necessary;
- asymmetric heads are necessary;
- rare-mass batch spreading is necessary;
- the method beats admitted baselines.

Those questions belong to the next registered development experiment.
