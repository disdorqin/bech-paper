# `src/core` — minimal repair + residual alignment safety

Status: **source refactor complete; not a scientific promotion.**

Scientific design authority:
`docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md`.
Math authority: `paper/02_math/RESIDUAL_ALIGNMENT_SAFE_RAY_DERIVATION_20260912.md`.
Refactor authority:
`docs/current/HCH_ALIGNMENT_SAFE_CORE_IMPLEMENTATION_PROMPT_20260912.md`.

This package is a **source implementation**. Nothing in it has been run against a
development or evaluation split, and no empirical claim is made for it here.

## What this is, and what it is not

The closed adjudication
`HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION` stands. It failed
gates 1 and 5 (16/20 Host-non-worse against 19 required; max cell-median
normal-region relative harm +7.585% against a 1% cap), passed gates 2/3/4/6, and
deleted all four optional development components on their own registered rules. The
smallest surviving method was the **empty switch vector**, and the failure was
concentrated entirely in `QINGHAI_DA`.

This refactor is **not** a rescue of the deleted components. Semantic Shape-context
routing, the local TCN, rare-mass sampling, the untied `A+`/`A-` amplitude option,
the KNN/similarity retrieval interface and the generic LSTM/bidirectional switches
are **absent from this package by name and by construction** — not disabled behind
flags. A configurable copy of a deleted component is the same component.

What survives is the empty switch vector plus a new, purely analytic deployment
safety layer.

## The candidate object

\[
c_h = A\,(S^+_h - S^-_h), \qquad A \ge 0, \qquad S^\pm \in \Delta^{H-1}
\]

Shape answers **WHERE** correction mass belongs across the horizon. Amplitude
answers **HOW MUCH** total correction mass there is — one nonnegative scalar. The
correction is exactly zero-sum over the horizon: it is a within-day residual
*redistribution*, never a level shift.

## Pipeline — complete

```text
legal raw information
  -> deterministic preprocessing
  -> ONE shared tiny MLP stem
  -> Shape:      GRU32 over Shape history      -> S+, S-
  -> Amplitude:  GRU32 over magnitude history  -> one scalar A >= 0
  -> c = A * (S+ - S-)
  -> global OOF calibration scalar alpha_0 = max(0, WM(r/c; |c|))
  -> deployment safety layer: lambda = min(alpha_0, lambda_U, lambda_S)
  -> Host + lambda * c
```

There is no hidden-state Bridge, no learned fusion, no market or province expert,
no instance-level selector, no learned gate, no trust score and no trainable safety
network. The trainable surface is the shared stem and the two branches.

## Deployment safety layer (analytic, not learned)

The controller reads the last **seven completed honest** persisted prequential
records `(S+_s, S-_s, c_s, r_s)` for this market and Host and computes, in closed
form:

| Quantity | Definition |
|---|---|
| uniform risk | `R^U(λ) = Σ_s [‖r_s − λ c_s‖₁ − ‖r_s‖₁]` |
| Shape relevance | `kappa_{d,s} = 1 − (W₁(S+_d,S+_s) + W₁(S-_d,S-_s)) / (2(H−1))` |
| Shape-weighted risk | `R^S(λ) = Σ_s kappa_{d,s} [‖r_s − λ c_s‖₁ − ‖r_s‖₁]` |
| exact safe radii | `lambda_U`, `lambda_S`: right endpoints of the no-harm intervals, found from sorted `r/c` breakpoints |
| final scale | `lambda = min(alpha_0, lambda_U, lambda_S)` |

Shape enters as **relevance**, never as prediction: it says which past outcomes of
the same frozen generator resemble the current candidate's geometry, and it
introduces no similarity model, neighbour count, bandwidth or temperature. Taking
the minimum makes Shape **one-way conservative** — it can shorten the permitted
ray, never lengthen it.

If the Shape-weighted evidence carries zero total weight, `lambda_S` is not a
constraint and the channel keeps `alpha_0`. There is no threshold, no grid search,
no tolerance and no learned gate anywhere in the layer. `src/core/safety.py` is
pure functions only and can be audited by reading it.

## Why `W = 7`

`HISTORY_DAYS = 7` is a single structural constant, not a configuration knob.
Seven is the minimum complete weekly support, it is the same temporal scale the
candidate generator's own history branch consumes, and it keeps the bank local
enough to describe the current regime. It is a structural default, not a learned
optimum. A tunable window would be a window search under another name, so the bank
constructor takes no capacity argument and the warm start has no length argument.

**Honesty is enforced from timestamps, not trusted from the caller.** A record is
evidence only if its candidate was generated *and persisted* before its own
realised residual became available (`candidate_created_at < ordinal`). An in-sample
candidate — one produced with the target day inside its fitting data — is refused
outright, because it would make every radius optimistic and the controller would
look safest exactly where it had cheated. `POST_TRAIN` warm start must use
chronological out-of-fold or prequential outputs, never final-model in-sample
predictions, and it refuses rather than shortening the window.

## Data contract

The core is deliberately dataset-agnostic. `RawInputs.forecast_features` has shape
`(B,H,F)` and holds **every numeric target-day forecast-known covariate admitted by
the dataset contract**; the experiment adapter owns names and provenance, and an
availability mask distinguishes a missing role from a physical zero. A data-rich
market can expose more legal columns without adding a market-specific branch: only
the input width changes, the recipe does not.

The core never reads a file, a dataset path or a market name, and it knows nothing
about splits, thresholds, baselines or evaluation.

## Shared MLP and the two branches

All current-day legal facts pass through exactly one `UnifiedFeatureStem`
(`Linear(D,64) -> GELU -> LayerNorm -> Linear(64,32)`). No per-feature, per-market
or per-Host MLP exists.

The two branches keep deliberately different views of history:

- **Shape** history is scale-free (`normalised residual`, `Δ normalised residual`,
  historical `S+`, historical `S-`), because Shape is a normalised geometry and its
  target is invariant under positive residual scaling.
- **Amplitude** history is magnitude-preserving (`scaled residual`, `abs residual`,
  positive part, negative part), because Amplitude is a scale and its target scales
  linearly.

Reusing the normalised Shape tensor as a magnitude proxy would destroy the second
property, so the two tensors are never interchanged.

## Calibration

One scalar per fitted market×Host candidate generator, from that model's legal
chronological out-of-fold rows:

\[
\alpha^* = \max\left(0,\ \operatorname{WeightedMedian}(r_i/c_i;\ |c_i|)\right)
\]

computed over nonzero candidate coordinates. It is pooled, never per-day and never
per-instance, and it is fitted on `POST_TRAIN` only — never on an evaluation or
protected split. `calibration.py` is carried over byte-identically from the
pre-refactor core; the refactor had no reason to touch it and did not.

## File map

| File | Responsibility |
|---|---|
| `contracts.py` | dataclasses and provenance constants; no dataset logic |
| `preprocessing.py` | deterministic coordinates and the two separate history views |
| `stem.py` | the one shared MLP |
| `encoders.py` | window GRU encoder (GRU32, mean / mean+max pooling) |
| `shape.py` | `S+`, `S-` heads and the masked horizon softmax |
| `amplitude.py` | the single balanced nonnegative Amplitude head |
| `fusion.py` | exact target decomposition and deterministic `A(S+ − S-)` |
| `losses.py` | repair MAE, Shape W₁, balanced Amplitude L1, combined objective |
| `calibration.py` | exact nonnegative pooled MAE scalar (unchanged from the archive) |
| `safety.py` | the analytic safe-ray layer — pure functions only |
| `history.py` | the seven-record honest prequential bank |
| `model.py` | thin composition, forward pass and `plan_deployment` |

## Historical source

The pre-refactor signed-mass development core was archived byte-preserving under
`src/archive/signed_mass_development_core_20260912/legacy_core/`, with
`ARCHIVE_HASHES.json` recording a per-file SHA256 and the canonical tree digest, and
`SUPERSEDED_TESTS.md` recording the superseded test file verbatim. The earlier V2.5
core remains under `src/archive/core_pre_extreme_repair_20260912/legacy_core/` with
compatibility support at `src/legacy_core_compat.py`.

Archive placement is lifecycle metadata. It is not, and must not be read as, a
change to any historical scientific verdict — including the closed signed-mass
verdict, which this refactor leaves exactly as it was.

## Validation

Source tests:

- `experiments/foundation/tests/test_core_candidate.py` — candidate-generator
  invariants (simplex, one nonnegative Amplitude, exact zero-sum correction,
  balanced-loss equivalence to the archived tied head, generic horizon, no deleted
  component on the active path);
- `experiments/foundation/tests/test_core_safety.py` — safety mathematics,
  prequential legality and historical-preservation proofs;
- `experiments/foundation/tests/test_core_purity.py` — purity and post-archive
  import resolution;
- `experiments/foundation/tests/test_canonical_core_layout.py` — the active file
  set is exactly the declared tree.

Run them from the repository root with `src` on the import path:

```bash
PYTHONPATH=src python -m pytest \
  experiments/foundation/tests/test_core_candidate.py \
  experiments/foundation/tests/test_core_safety.py \
  experiments/foundation/tests/test_core_purity.py \
  experiments/foundation/tests/test_canonical_core_layout.py -q
```

## What is not established

Passing these tests means the implementation respects the design contract. It does
**not** mean the candidate generator is empirically useful, that the safety layer
improves anything in practice, that the Shape relevance weighting is necessary, or
that the method beats any admitted baseline. Gates 1 and 5 of the closed
adjudication are still failed and nothing here changes that.

The safety controller is **unvalidated empirically**. Its correctness is a
mathematical property of convex piecewise-linear geometry on a given evidence set,
verified against brute-force references; whether it is *useful* is an experiment,
and no such experiment is authorized by this refactor.
