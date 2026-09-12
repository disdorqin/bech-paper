# Alignment-Safe Core Design Contract

Date: 2026-09-12
Authority: `docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md`
Math authority: `paper/02_math/RESIDUAL_ALIGNMENT_SAFE_RAY_DERIVATION_20260912.md`
Refactor authority: `docs/current/HCH_ALIGNMENT_SAFE_CORE_IMPLEMENTATION_PROMPT_20260912.md`

This file defines implementation invariants. **It is not an experiment result, and
nothing in it is an empirical claim.**

## 0. Standing

The closed adjudication
`HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION` is unchanged by
this contract. Gates 1 and 5 failed; all four optional development components were
deleted on their own registered rules; the smallest surviving method was the empty
switch vector. This refactor implements that survivor plus an analytic deployment
safety layer. It does not reinstate, retune or reinterpret anything the
adjudication deleted.

## 1. The candidate object

For a realised residual `r = y − y_host` the two exact mass directions are

\[
r^+ = \max(r,0), \qquad r^- = \max(-r,0),
\]
\[
A^+ = \sum_h r_h^+, \qquad A^- = \sum_h r_h^-,
\]
\[
S_h^+ = r_h^+ / A^+, \qquad S_h^- = r_h^- / A^-,
\]

with `S± ∈ Δ^{H−1}` whenever the corresponding mass is positive, and the exact
reconstruction

\[
r_h = A^+ S_h^+ - A^- S_h^- .
\]

The **candidate** is the tied-mass factorization

\[
\boxed{c_h = A\,(S_h^+ - S_h^-)}, \qquad A \ge 0, \qquad S^\pm \in \Delta^{H-1}.
\]

One nonnegative Amplitude scalar and two horizon Shapes. No trainable fusion layer
may replace this identity: fusion is `fuse(amplitude, shape)`, multiplication and
subtraction, and nothing else.

## 2. Invariants

### C1 — exact zero-sum correction

\[
\sum_h c_h = A \Big(\sum_h S_h^+ - \sum_h S_h^-\Big) = A(1 - 1) = 0 .
\]

The candidate is a within-day **redistribution**, not a level correction. There is
no daily-level term, no bias branch and no additive offset, and adding one would
break this invariant.

### C2 — Shape is a simplex

`S+` and `S-` are produced by a masked softmax over the horizon axis: entries are
non-negative and the row sums to 1 over the **valid** horizon. A row with no valid
position returns exact zeros rather than a uniform guess over positions the caller
declared absent, and no `-inf` is ever produced, so no `NaN` can appear.

### C3 — Amplitude is one nonnegative scalar

`A = s_A · softplus(f(z_A)) ≥ 0` for a single scalar readout `f`. There is exactly
one amplitude head. Two heads (a `A+`/`A-` pair, tied or untied) would reintroduce
the degree of freedom the closed adjudication deleted, so the tied-mass form is not
offered as a configurable option either.

### C4 — Shape scale invariance, Amplitude scale equivariance

Shape consumes scale-free coordinates (`z`-normalised residual, its first
difference, historical `S±`), so under a positive rescaling of the residual its
target is invariant. Amplitude consumes magnitude-preserving coordinates, so its
target scales linearly. The two historical views are different tensors and must
never be interchanged.

### C5 — one shared stem

All current-day absolute facts pass through exactly one `UnifiedFeatureStem`
(`Linear(D,64) → GELU → LayerNorm → Linear(64,32)`). There is no per-feature,
per-market or per-Host MLP. Only the input width tracks the admitted schema.

### C6 — forecast-origin legality

Current-day inputs may contain only the frozen Host forecast, audited
forecast-known numeric covariates, deterministic calendar information and
availability masks. Historical input may contain only already-revealed
original-origin residuals. The target-day actual and its residual are never inputs:
the batch type the model consumes carries no residual column at all.

### C7 — explicit availability

A missing semantic role is represented by a mask. It is never replaced by Host
price or by another physical variable.

### C8 — no learned safety component

There is no learned gate, trust score, reliability network, uncertainty head,
proposal selector or second network deciding whether the correction applies. The
controller is a closed-form computation on a bounded number of persisted records.

## 3. Deployment: the exact safe ray

A frozen Host gives `y_H`; the frozen candidate gives `c` before the target is
known; the realised residual is `r = y − y_H`. Deployment is restricted to the ray

\[
y(\lambda) = y_H + \lambda c, \qquad \lambda \ge 0 ,
\]

so the only question is how much of an already-generated correction to keep.

### 3.1 Exact ray geometry

Writing `z_i = r_i / c_i` and `w_i = |c_i|` for coordinates with `c_i ≠ 0`,

\[
L(\lambda) = \lVert r - \lambda c \rVert_1
           = \sum_{c_i \ne 0} w_i\,|z_i - \lambda| + \sum_{c_i = 0} |r_i| ,
\]

which is piecewise linear in `λ` with breakpoints exactly at the ratios `r_i/c_i`.
The excess risk over the Host,

\[
R(\lambda) = \sum_i q_i\big(|z_i - \lambda| - |z_i|\big),
\]

is convex with `R(0) = 0`, so its no-harm set is an interval beginning at zero and
the safe radius is its right endpoint. It is computed by scanning sorted positive
breakpoints; there is no grid, no search and no tolerance to tune.

`q_i = |c_i|` gives the uniform radius `lambda_U`; `q_i = |c_i| · kappa_{s(i)}` gives
the Shape-weighted radius `lambda_S`, where the weight is the record's relevance.

### 3.2 Alignment support

With `Q = Σ q_i` and `Q_+ = Σ_{z_i > 0} q_i`,

\[
A_{\text{align}} = \frac{2Q_+}{Q} - 1, \qquad R'(0^+) = Q - 2Q_+ = -Q\,A_{\text{align}} .
\]

`A_align < 0` means the right derivative at zero is positive, so by convexity every
positive step is harmful and the safe radius is exactly zero. Coordinates with
`r_i = 0` sit on the non-supporting side; they are not free.

### 3.3 Shape relevance

\[
\kappa_{d,s} = 1 - \frac{W_1(S_d^+, S_s^+) + W_1(S_d^-, S_s^-)}{2\,(H-1)},
\qquad
W_1(P,Q) = \sum_{h=1}^{H-1} \big|F_P(h) - F_Q(h)\big| .
\]

For distributions on the simplex `W₁ ≤ H−1`, so `kappa ∈ [0,1]` is a normalised
disagreement rather than a free score. With fewer than two valid horizon positions
the Shape carries no discriminative information and `kappa = 1` exactly.

Shape is used as **relevance**, never as prediction. No similarity model, neighbour
count, bandwidth, temperature or retrieval bank participates.

### 3.4 Final scale

\[
\boxed{\lambda = \min(\alpha_0,\ \lambda_U,\ \lambda_S)}
\]

Taking the minimum makes Shape **one-way conservative**: it can shorten the
permitted ray, never lengthen it beyond what the broad recent evidence already
allows. When the Shape-weighted evidence has zero total weight, `lambda_S` is not a
constraint and the channel keeps `alpha_0`; the same holds for a radius of `+inf`,
which means "this evidence imposes no restriction" rather than "any strength is
safe". The result is therefore always finite and never exceeds `alpha_0`: the
controller can only shrink a registered correction, never amplify it, and never
manufactures a correction for a candidate whose mass is zero.

## 4. The prequential evidence bank

The controller consumes the last `HISTORY_DAYS = 7` **completed honest** records
`(S+_s, S-_s, c_s, r_s)`, oldest first.

**H1 — `W = 7` is structural.** Not a configuration knob, not a window search, not
tunable. The bank constructor takes no capacity argument and the warm start takes
no length argument.

**H2 — honesty is enforced from timestamps.** `candidate_created_at < ordinal` is
required; a candidate created no earlier than its own outcome is refused with
`InSampleCandidateError`. A residual dated before its own slot, or not strictly
after its candidate, is refused. A completed record is never rewritten.

**H3 — a pending candidate is not evidence.** Only records that have received their
residual are exposed. `completed()` and `as_arrays()` refuse a short bank rather
than letting the controller run on fewer observations than the design specifies;
`IncompleteHistoryError` is raised instead. A pending candidate never displaces a
completed one through eviction.

**H4 — no replay.** A duplicate delivery identifier is refused, including one
already evicted from the window; `ordinals` must strictly increase; a residual
cannot be attached to a candidate that was never persisted.

**H5 — warm start is out-of-fold or prequential only.** In-sample predictions are
excluded by provenance. Fewer than seven legal completed records raise
`InsufficientWarmStartError`; **W is not shortened** and the caller is expected to
stop with a blocker.

**H6 — the bank is data, not model state.** It is not a buffer, a parameter, a
module attribute or part of `state_dict`. It is passed to `plan_deployment` as an
argument, so the controller's evidence is a function of what actually happened
rather than of which checkpoint happened to be loaded. The package is also
file-system agnostic: the experiment harness owns disk persistence and split
provenance.

## 5. Calibration invariant

For one fitted market×Host candidate generator, one pooled scalar is fitted from
that model's legal chronological out-of-fold rows:

\[
\alpha^* = \arg\min_{\alpha \ge 0}\sum_i |r_i - \alpha c_i|
        = \max\Big(0,\ \operatorname{WeightedMedian}(r_i/c_i;\ |c_i|)\Big)
\]

over nonzero candidate coordinates (`c_i = 0` is removed *before* forming ratios,
so no `inf`/`nan` can arise; an all-zero candidate returns `0`).

The scalar is pooled within the fitted model, never per test instance, never fitted
from evaluation or protected labels, and not required to be numerically identical
across markets or Hosts. It is fitted on `POST_TRAIN` only.

## 6. Loss invariants

Repair objective, Shape Wasserstein-1 auxiliary, and the balanced Amplitude
auxiliary:

\[
L_R = H^{-1}\sum_h |r_h - c_h|, \qquad
W_1(S,\hat S) = \sum_{h=1}^{H-1}|F_S(h) - F_{\hat S}(h)|,
\]
\[
L_A^{\text{bal}} = |A^+ - A| + |A^- - A| .
\]

The balanced objective is **mathematically identical** to the archived tied-head
objective `amplitude_loss(A, A+, A, A-)`, which fed the old function two equal
predictions; the new form writes the same number with one prediction. This identity
is asserted in the source tests against the archived code itself, not restated.

Combined objective `L = L_R + λ_S L_S + λ_A L_A` with weights that are **not
scientifically frozen**. No market-specific loss weight is permitted.

## 7. Gradient coupling

The deterministic fusion gives

\[
\frac{\partial c_h}{\partial A} = S_h^+ - S_h^-, \qquad
\frac{\partial c_h}{\partial S_h^+} = A, \qquad
\frac{\partial c_h}{\partial S_h^-} = -A ,
\]

so a large predicted correction mass naturally increases the final-loss sensitivity
to Shape placement error. This coupling is sufficient; no learned Bridge is
justified without new evidence.

## 8. Core purity

`src/core/**` must not contain dataset paths, province or market branches, Host-name
branches, baseline implementations, experiment registry or evaluation code, result
aggregation, paper-table logic, or a trainable Gate/selector/reliability network. It
must not import `experiments/`, `paper/`, `docs/`, `archive/`, or the retired V2.5
subpackages. It must not reference `PROTECTED_FINAL` or any evaluation split.

Deleted development components must not reappear under any name, including a
configurable one: no TCN, no semantic Shape-context routing, no untied amplitude
heads, no rare-mass sampling, no KNN/similarity retrieval, no generic
LSTM/bidirectional architecture switch, no daily bias branch.

## 9. Scientific status

Respecting this contract means the implementation is what the design says. It does
**not** mean:

- the signed-mass candidate is empirically useful;
- the safety controller improves anything in practice — it is **unvalidated
  empirically**;
- Shape relevance weighting is necessary, or that `W = 7` is optimal rather than
  structural;
- the two branches, or the shared stem, are necessary;
- the method beats any admitted baseline.

Gates 1 and 5 of the closed adjudication remain failed. Those questions belong to a
future registered experiment, and this refactor authorizes none.
