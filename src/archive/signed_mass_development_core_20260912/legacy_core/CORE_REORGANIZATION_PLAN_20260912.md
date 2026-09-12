# `src/core` Reorganization & Architecture Preparation Plan
Date: 2026-09-12
Project: solar-model科研
Target repository: `D:\作业\science\solar_leak_price_model`
Status: PREPARATION ONLY — no new scientific experiment authorization

## 0. Purpose

`src/core` should become the unique, minimal, paper-facing implementation space for the current extreme-price post-processing line.

The current folder contains historical residue from earlier V2/V2.5/V3 routes. The next operation should:
1. audit and archive obsolete core code without destroying historical reproducibility;
2. create a clean `src/core` structure;
3. integrate only the currently justified mathematical/data abstractions;
4. prepare reusable model components for the next research phase;
5. NOT train or evaluate the new method yet.

The guiding principle remains:

> Solve one narrow post-processing data problem, not forecasting as a whole.

Current candidate object:
- Shape = where positive/negative correction mass should appear across the horizon;
- Amplitude = how much total positive/negative correction mass is needed;
- no learned Gate by default.

---

# 1. Scientific architecture to prepare

The architecture should remain small and coherent:

```text
legal raw inputs
    ↓
deterministic canonicalization
    ↓
ONE shared tiny MLP stem
    ↓
semantic routing
    ├── Shape view
    └── Amplitude view
    ↓
branch-local dual-axis encoders
    ├── Shape: local TCN + temporal BiLSTM/GRU
    └── Amplitude: local TCN + temporal BiLSTM/GRU
    ↓
Shape outputs S+, S-
Amplitude outputs A+, A-
    ↓
deterministic signed-mass fusion
    c_h = A+ S+_h - A- S-_h
    ↓
optional global pooled OOF MAE calibration alpha
    ↓
Host + alpha * c
```

No learned Gate.
No Bridge.
No CORN.
No quantile grid.
No market/Host routers.
No per-market special cases.

---

# 2. Why ONE shared MLP stem

The user explicitly prefers one unified MLP instead of a separate network per feature.

The first learned stage should therefore be one shared feature-interaction stem.

Before that stem, only deterministic operations are allowed:
- masks / legality encoding;
- training-only scaling;
- within-day robust normalization;
- first differences / ramps;
- fixed calendar encodings;
- safe daily summary statistics.

For a token/window feature vector `x`:

\[
e = \phi_{\text{stem}}(x)
\]

where `phi_stem` is one small MLP, shared across the entire model.

The goal is:
- allow low-cost cross-feature interaction;
- avoid feature-specific MLP proliferation;
- keep preprocessing auditable.

Recommended initial stem:
- Linear(D_in, 64)
- GELU
- LayerNorm
- Linear(64, 32)

Exact dimensions remain configurable and are not a scientific claim.

---

# 3. Semantic routing AFTER the shared stem

The shared MLP does not mean Shape and Amplitude receive identical information.

Semantic routing should be simple:
- deterministic channel definitions;
- masks / selected derived coordinates;
- one small branch projection if dimension alignment is needed.

No per-feature experts.

## 3.1 Shape semantics

Question:
> WHERE should correction mass appear?

Shape should emphasize:
- within-day normalized Host profile;
- normalized load/wind/solar/net-load profiles;
- profile ramps;
- historical residual Shape;
- horizon/calendar position.

Absolute scale should be de-emphasized.

## 3.2 Amplitude semantics

Question:
> HOW MUCH total correction is required?

Amplitude should emphasize:
- absolute Host level/range;
- load level / peak;
- renewable total / ratio;
- net-load extrema / ramps;
- rolling residual severity;
- past A+ / A-;
- extreme-frequency summaries.

Absolute scale must remain.

---

# 4. Branch-local encoder: simplest useful BiLSTM/TCN design

The phrase “BiLSTM for temporal dependencies, TCN for feature structure” should be implemented carefully.

Do NOT create a large EPformer clone.

Use the same lightweight branch block class twice with separate weights:

```text
BranchEncoder
    input embedding
        ↓
    local TCN over within-day horizon
        ↓
    one daily/window representation
        ↓
    BiLSTM/GRU over historical windows
        ↓
    branch latent
```

Interpretation:
- TCN = within-window / within-day local multi-scale pattern extraction;
- BiLSTM/GRU = evolution across historical windows/days.

This is cleaner than making TCN and BiLSTM two unrelated large parallel towers.

Recommended first implementation:
- 2-layer lightweight TCN or depthwise temporal conv stack;
- kernel sizes small (e.g. 3, optionally 5);
- hidden width 32;
- one-layer BiLSTM or BiGRU hidden 32 per direction OR current Daily-Patch GRU32 compatible mode;
- dropout small/configurable;
- no attention by default.

The implementation should expose encoder choice as configuration:
`temporal_rnn = "gru" | "lstm"`
but default to the project’s validated GRU-compatible path until evidence justifies BiLSTM.

This preserves the recent GRU work while allowing a BiLSTM ablation later.

---

# 5. Similar-window / KNN context

Similarity learning is useful, but retrieval must not become the main correction mechanism.

Recent electricity-price literature repeatedly uses similar-day selection, while recent time-series post-processing work already occupies residual-retrieval territory.

Therefore:

## Default role
KNN/similar-window is an OPTIONAL Shape context constructor.

It may:
1. embed each historical legal window using the shared stem / Shape representation;
2. find K nearest revealed historical windows;
3. compute a weighted historical Shape prototype:

\[
\bar S^\pm
=
\sum_{j\in N_K(d)}
w_j S_j^\pm,
\qquad
w_j
=
\frac{\exp(-d_j/\tau)}
{\sum_\ell\exp(-d_\ell/\tau)}.
\]

The prototype is fed as context to the Shape decoder.

It must NOT directly output the final residual correction.

Default implementation state:
`enabled = false`

Reason:
- prepare the interface now;
- activate only after D0 or ablation proves it adds value;
- avoid turning the method into RATL-style retrieval correction.

Required safeguards:
- search bank contains only historically revealed training/post-training records;
- no DEV_EVAL labels in the bank;
- no target-day actuals;
- deterministic tie handling;
- configurable K and temperature;
- all similarity statistics fitted only on permitted partitions.

---

# 6. Rare extreme samples in Amplitude

Amplitude has a different sparsity problem:
large A+ / A- events are rare and can be washed out by normal samples.

Do NOT solve this by building another KNN correction system.

Prepare a training-only `RareMassSampler` / stratified sampler.

Concept:
- compute training-only empirical quantile strata for A+ and A-;
- construct batches with both ordinary and high-mass samples;
- use one fixed global rule across markets.

Example initial contract:
- uniform base samples;
- ensure presence of upper A+ stratum;
- ensure presence of upper A- stratum.

This is a data-sampling mechanism, not a third predictive head.

The exact proportions must be configurable and frozen before method evaluation.

If later evidence shows reweighting is unnecessary, the sampler can be disabled.

---

# 7. Positive / negative Amplitude asymmetry

Use one Amplitude encoder trunk but two independent heads:

\[
\hat A^+
=
s_A\operatorname{softplus}(f_+(z_A)),
\]

\[
\hat A^-
=
s_A\operatorname{softplus}(f_-(z_A)).
\]

Do not force the last-layer parameters to be shared.

Do not yet create:
- positive GRU;
- negative GRU;
- upper/lower expert routers.

If later diagnostics show stable feature-relevance differences, a small group selector may be evaluated, but it is not part of the preparation architecture.

---

# 8. Shape outputs and location-aware loss

Shape outputs horizon distributions:

\[
\hat S^\pm
=
\operatorname{softmax}(q^\pm).
\]

Thus:
- nonnegative;
- sum to 1.

Prepare a 1-D Wasserstein loss helper:

\[
W_1(S,\hat S)
=
\sum_{h=1}^{H-1}
|F_S(h)-F_{\hat S}(h)|.
\]

This is the default Shape auxiliary objective candidate.

Also implement simple L1 distribution loss as an ablation option.

Do NOT hardcode tail thresholds into Shape.

---

# 9. Deterministic fusion

The two branches must connect through the mathematical factorization, not through a learned fusion MLP:

\[
\hat c_h
=
\hat A^+\hat S_h^+
-
\hat A^-\hat S_h^-.
\]

This exact fusion is mandatory.

No black-box fusion layer after Shape/Amplitude.

---

# 10. Gate policy

No learned Gate in the default core.

Historical proposal verification already failed.

Do not implement a class named:
- Gate
- TrustGate
- RepairabilityGate
- BenefitGate
- ConfidenceGate

in the promoted current core.

Historical Gate code may be archived for fidelity.

Host preservation is handled by:
- constrained factorization;
- MAE training;
- optional global alpha calibration;
- normal-region validation.

---

# 11. Global MAE calibration

Prepare a small pure module for the current validated pooled calibration:

\[
\alpha^*
=
\arg\min_\alpha
\sum_i|r_i-\alpha c_i|.
\]

For nonzero candidate corrections, alpha is the weighted median of:

\[
r_i/c_i
\]

with weights:

\[
|c_i|.
\]

This module:
- is fitted only on the legally allowed calibration/OOF partition;
- is scalar/global, not per-instance;
- does not read test labels.

Expose:
- `fit(candidate, residual) -> alpha`
- `apply(candidate, alpha)`

No hidden adaptation.

---

# 12. Proposed clean `src/core` layout

Target structure:

```text
src/core/
├── README.md
├── DESIGN_CONTRACT.md
├── __init__.py
├── contracts.py
├── preprocessing.py
├── stem.py
├── semantic_views.py
├── similarity.py
├── encoders.py
├── shape.py
├── amplitude.py
├── fusion.py
├── calibration.py
├── losses.py
├── sampling.py
└── model.py
```

Responsibilities:

## `contracts.py`
Pure dataclasses / typed structures:
- `ForecastOriginBatch`
- `SemanticViews`
- `ShapeOutput`
- `AmplitudeOutput`
- `RepairOutput`
- `ModelConfig`

No dataset-specific logic.

## `preprocessing.py`
Deterministic transformations:
- masks;
- train-frozen scalers;
- within-day robust normalization;
- ramps/differences;
- daily summaries.

## `stem.py`
ONE shared tiny MLP:
- `UnifiedFeatureStem`

## `semantic_views.py`
Construct:
- Shape view;
- Amplitude view;
using fixed semantic routing.

No neural expert routing.

## `similarity.py`
Optional:
- `KNNShapeContext`
- memory bank interface
- weighted Shape prototype
Default disabled.

## `encoders.py`
Reusable lightweight:
- `LocalTCN`
- `WindowRNN`
- `BranchEncoder`

No market-specific subclasses.

## `shape.py`
- Shape branch;
- simplex outputs;
- optional similar-window context input.

## `amplitude.py`
- one trunk;
- positive/negative independent heads.

## `fusion.py`
Pure:
`A+ * S+ - A- * S-`

## `calibration.py`
Pooled exact MAE scalar alpha.

## `losses.py`
- repair MAE;
- W1 Shape;
- Shape L1 ablation;
- amplitude L1;
- total loss composition.

## `sampling.py`
Training-only rare-mass stratification.

## `model.py`
Thin orchestration:
- stem
- semantic views
- branches
- deterministic fusion

No dataset loading.
No experiment logic.
No evaluation registry logic.

---

# 13. Archive plan

Never delete historical core code directly.

Create:

`src/archive/core_pre_extreme_repair_20260912/`

Before moving anything:
1. inventory `src/core`;
2. grep repository-wide imports/usages;
3. map every file to:
   - CURRENT_REUSE
   - HISTORICAL_FIDELITY
   - OBSOLETE_UNREFERENCED
   - UNCERTAIN
4. hash/archive manifest.

Move only files whose dependency status is understood.

Historical-fidelity code must remain importable for old experiments.

If a historical experiment imports an old core symbol:
- preserve it under archive;
- use an explicit compatibility import only if needed;
- do not rewrite evidence artifacts.

Create:
`src/archive/core_pre_extreme_repair_20260912/ARCHIVE_MANIFEST.md`

Include:
- old path;
- new path;
- reason;
- repository usage sites;
- status;
- hash.

---

# 14. Core purity rules

The promoted `src/core` must NOT import from:
- `experiments/`
- `paper/`
- `docs/history/`
- `src/archive/`

except an explicitly documented compatibility layer outside promoted core.

`src/core` must not contain:
- dataset-specific file paths;
- SHANDONG/GANSU/etc. conditionals;
- Host-name conditionals;
- result thresholds selected per market;
- baseline code;
- experiment registry code.

Core is scientific method only.

---

# 15. Tests to prepare

The reorganization phase may run unit/integration tests, but not scientific model experiments.

Required tests:

1. Signed-mass reconstruction exactness.
2. S+ / S- simplex invariants.
3. Scale invariance:
   `S(c*r) == S(r)` for c>0.
4. Scale equivariance:
   `A(c*r) == c*A(r)`.
5. Zero-mass direction handling.
6. Wasserstein loss sanity.
7. Pooled MAE alpha vs brute-force grid on toy data.
8. No future-value leakage through masks.
9. Similarity bank uses only allowed historical indices.
10. Same input dimension works for multiple datasets without market branches.
11. Model forward shape for generic H, not hardcoded to 24.
12. No Gate module in promoted core.
13. Repository import test after archive movement.
14. Historical reproduction import smoke test.

---

# 16. What NOT to implement yet

Preparation AI must NOT:
- train the proposed model;
- choose final K;
- choose final TCN kernel set;
- choose GRU vs BiLSTM based on results;
- tune rare-sample ratios;
- activate KNN by default;
- create a learned Gate;
- access PROTECTED_FINAL;
- add market-specific feature lists to core;
- claim the method is frozen.

The goal is clean scientific infrastructure, not method promotion.

---

# 17. Terminal condition

Success means:

`CORE_REORGANIZATION_READY_FOR_SIGNED_MASS_METHOD_DEVELOPMENT`

Required final report:
- old core inventory;
- archive manifest;
- clean new core tree;
- implemented mathematical primitives;
- implemented architecture scaffolding;
- tests and results;
- remaining TODOs that require scientific evidence;
- confirmation that no new scientific experiment ran;
- confirmation that protected data were untouched;
- canonical state-file updates.
