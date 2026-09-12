# `src/core` — signed-mass repair candidate core

Status: **user-authorized current candidate scaffolding; source-ready, not yet scientifically promoted**.

Scientific design authority:
`docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`.

The package implements one narrow post-processing hypothesis for a frozen Host:

\[
r_h=A^+S_h^+-A^-S_h^-,
\qquad
\hat c_h=\hat A^+\hat S_h^+-\hat A^-\hat S_h^-.
\]

Shape answers **WHERE** correction belongs. Amplitude answers **HOW MUCH** positive/negative correction mass is needed.

## Current pipeline

```text
Host + ALL audited forecast-known target-day numeric features + masks/calendar
+ revealed historical residuals
    -> deterministic canonical coordinates
    -> ONE shared tiny MLP
    -> Shape: shared embedding + scale-free geometry
         -> small TCN -> GRU32 over Shape history -> S+, S-
    -> Amplitude: shared absolute embedding
         -> small TCN -> GRU32 over magnitude history -> A+, A-
    -> exact A+S+ - A-S-
    -> chronological OOF nonnegative MAE scalar alpha
    -> Host + alpha * correction
```

There is no hidden-state Bridge, learned fusion, market expert or instance-level proposal selector in the current path.

## Data contract

The core is deliberately dataset-agnostic.

`RawInputs.forecast_features` has shape `(B,H,F)` and contains **all numeric target-day forecast-known covariates admitted by the dataset contract**. The experiment/data adapter owns their names and provenance. The core receives a same-shape availability mask so a missing role is not confused with a physical zero.

This means a data-rich Chinese market can expose more legal columns without adding a market-specific network branch. Only the input width changes; the method recipe does not.

The core never reads files or dataset names.

## Shared MLP

All absolute current-day legal facts pass through one `UnifiedFeatureStem`:

```text
Linear(D,64) -> GELU -> LayerNorm -> Linear(64,32)
```

No per-feature MLP exists.

## Shape representation

Shape additionally receives deterministic within-day profile coordinates for every legal forecast feature:

\[
N(x)_h=(x_h-\mathrm{median}(x))/(1.4826\,\mathrm{MAD}(x)+\epsilon)
\]

and first differences. Historical residual input for Shape is also scale-free:

```text
N(residual), Delta N(residual), historical S+, historical S-
```

The output uses horizon softmax, so each predicted `S+`/`S-` is a simplex. Ordered error is measured by one-dimensional Wasserstein-1.

## Amplitude representation

Amplitude does **not** receive a large hand-crafted severity inventory. It receives the same shared absolute MLP embedding and learns current-day severity with its small TCN. Historical residual input is separately magnitude-preserving:

```text
scaled residual, abs residual, positive residual, negative residual
```

One Amplitude trunk feeds two untied nonnegative heads for `A+` and `A-`.

## TCN / GRU roles

- small TCN: within-day local pattern;
- GRU32: cross-day evolution.

They are executors, not contribution claims. LSTM is a configuration-only future encoder ablation.

## Calibration

For each fitted market×Host repair model, one scalar is fitted from that model's legal chronological OOF candidate rows:

\[
\alpha^*=\arg\min_{\alpha\ge0}\sum_i|r_i-\alpha c_i|.
\]

The implementation uses the exact weighted-median solution followed by the registered nonnegative constraint. It is not per-day adaptation and not one universal cross-market scalar.

## Four removable hypotheses

The source exposes exactly four first-stage ablation switches:

- `shape_semantic_context`;
- `use_tcn`;
- `untied_amplitude_heads`;
- `rare_mass_sampling`.

They exist only because the research contract says: **if one fails its registered development ablation, delete it instead of rescuing it.** They are not an architecture search grid.

KNN/similar-window context remains disabled and is not part of the first method experiment.

## File map

| File | Responsibility |
|---|---|
| `contracts.py` | generic tensor/config contracts |
| `preprocessing.py` | deterministic coordinates and separate Shape/Amplitude history views |
| `stem.py` | one shared MLP |
| `semantic_views.py` | minimal task routing |
| `encoders.py` | local TCN + window GRU/LSTM |
| `shape.py` | `S+`,`S-` heads |
| `amplitude.py` | `A+`,`A-` heads |
| `fusion.py` | exact target decomposition and deterministic fusion |
| `losses.py` | repair MAE, Shape W1, Amplitude L1 |
| `calibration.py` | exact nonnegative pooled MAE scalar |
| `sampling.py` | optional training-only rare-mass batch organization |
| `similarity.py` | disabled evidence-gated Shape context interface |
| `model.py` | thin composition |

## Historical source

The former V2.5 core was archived byte-preserving under:

`src/archive/core_pre_extreme_repair_20260912/legacy_core/`

with compatibility support at `src/legacy_core_compat.py`. Archive placement is lifecycle metadata, not a change to historical scientific verdicts.

## Validation

Current synthetic core tests live at:

- `experiments/foundation/tests/test_core_signed_mass.py`
- `experiments/foundation/tests/test_core_purity.py`
- `experiments/foundation/tests/test_canonical_core_layout.py`

Source readiness does **not** imply scientific success. The next scientific stage is controlled separately by the signed-mass method experiment protocol.
