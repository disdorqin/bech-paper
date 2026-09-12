# Signed-Mass Core Readiness Audit — 2026-09-12

Status: `CORE_SOURCE_READY_FOR_REGISTERED_METHOD_EXPERIMENT`

This is an implementation audit, not a scientific experiment result.

## 1. What was reviewed

- archived-core migration/inventory;
- current `src/core` architecture against the signed-mass method design;
- input semantics for data-rich Chinese provincial markets;
- Shape/Amplitude history separation;
- calibration contract;
- optional rare-mass/KNN code paths;
- current-core purity;
- historical V2.5 compatibility.

## 2. Material defects found and corrected

### A. Missing load was incorrectly replaced by Host price

Prior scaffold used Host price as load when load was absent. This created a false market variable and violated semantic missingness.

Correction:
- removed hard-coded load/wind/solar input contract;
- all legal target-day covariates now enter through generic `forecast_features (B,H,F)`;
- unavailable values are zero only together with an explicit `forecast_feature_mask`;
- no physical role is fabricated.

### B. Rich provincial legal features were silently discarded

The old scaffold supported only a small fixed role list while audited provincial contracts expose more legal forecast columns.

Correction:
- the core accepts the complete audited legal numeric feature tensor;
- dataset adapters own names/provenance;
- one shared MLP handles arbitrary legal feature count;
- no province branch was added.

### C. Historical Amplitude magnitude had been normalized away

The old scaffold computed historical positive/negative mass from already-normalized Shape channels, making supposed mass summaries approximately constant rather than true historical magnitude.

Correction:
- `shape_history`: scale-free residual profile/ramp and historical sign Shapes;
- `amplitude_history`: training-scale residual / absolute / positive / negative magnitude;
- separate branch encoders consume the correct history view.

### D. Calendar information was truncated

The old builder used only the first calendar coordinate.

Correction:
- all supplied legal calendar coordinates are preserved.

### E. MAE calibration lost the registered nonnegative constraint

Correction:

\[
alpha^*=max(0,WeightedMedian(r/c;|c|)).
\]

The documentation now correctly states that alpha is one pooled scalar per fitted market×Host repair model/run, not one universal cross-market scalar and not per test instance.

### F. Rare positive oversampling pool was wrong

Correction:
- positive pool = POSITIVE or BOTH;
- negative pool = NEGATIVE or BOTH.

Duplicate oversampling remains disabled in the first method design.

## 3. Simplification made during audit

Removed the manual Amplitude severity feature inventory from the core.

Current design:
- all absolute legal target-day facts interact once in the shared MLP;
- Amplitude TCN/GRU learns severity from that absolute embedding plus magnitude-preserving residual history;
- Shape alone receives extra deterministic scale-free profile/ramp coordinates.

This is simpler, uses more of the audited domestic information, and is less market-specific.

## 4. Four experiment switches

Source now exposes only the four preregistered removable hypotheses:
- `shape_semantic_context`;
- `use_tcn`;
- `untied_amplitude_heads`;
- `rare_mass_sampling`.

KNN remains disabled and outside the first screen.

## 5. Validation

Relevant source, purity, layout and historical-compatibility tests were run together.

Result:

```text
53 passed
```

Additional compile check:

```text
python -m compileall -q src/core
```

PASS.

Tests now cover exact factorization, generic legal feature counts, explicit missingness, masked scaling, complete calendar preservation, separate Shape/Amplitude history semantics, nonnegative MAE calibration, horizon-general forward passes, optional-context legality, rare-mass sampling legality, controlled removal switches, source purity and old V2.5 compatibility.

## 6. What PASS does not mean

This audit does not establish that signed-mass repair, task-directed context, TCN, asymmetric heads or rare-mass batching improve forecasting. It does not establish superiority to any baseline.

Those scientific questions belong to:

`docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`.

`PROTECTED_FINAL` remains unopened by this source audit.
