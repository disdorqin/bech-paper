# HCH-v2 v0.4 Universal Adaptive Architecture Design

> Date: 2026-08-12  
> Repository baseline: `disdorqin/bech-paper@97449234294686a2ca78d8fb59d76b54e9e0eb15`  
> Status: **ARCHITECTURE CANDIDATE FOR IMPLEMENTATION / PRE-EXPERIMENT**  
> Mathematical core: `hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md` remains authoritative.  
> Principle: **simple and effective, distinctive, mathematically grounded**.

---

## 0. Architecture decision

HCH-v2 is no longer defined as “one shared correction network + one local memory”.

The intended system is:

\[
\boxed{
\text{Universal Correction Core}
+
\text{Data-Adaptive Interface}
+
\text{Local Evidence Layer}
}
\]

The three layers answer three different questions:

1. **Universal Correction Core** — what correction knowledge can be learned across datasets and host models?
2. **Data-Adaptive Interface** — how should the same core behave when observability, scale, dynamics and covariates differ?
3. **Local Evidence Layer** — is the proposed correction historically supported and safe enough to execute in the current target domain?

The mathematical decision chain remains:

\[
\boxed{
\text{Candidate}
\rightarrow
\text{Historical Evidence}
\rightarrow
\text{Positive-Value Decision}
}
\]

The new architecture changes **how candidate parameters are represented, trained, frozen and adapted**, not the v0.3 definitions of IAH-CRPS, query-dose replay, double-event proposal or split-conformal DVG.

---

# 1. Scope and claim boundary

The paper is evaluated primarily on **electricity forecasting**, with electricity price as the main target and other electricity/energy series considered only as generalization extensions.

The architecture itself should not require:

- a particular market name;
- a particular currency;
- DA/RT identity as a predictive token;
- a fixed number of covariates;
- rich exogenous features;
- a particular host backbone.

A valid HCH deployment must therefore include a complete **core-only path** that works when the user has only:

\[
\{\text{frozen host forecast},\ \text{legal target history},\ \text{time information}\}.
\]

Rich covariates may improve HCH, but their absence must not make the module structurally incomplete.

---

# 2. Layer I — Universal Correction Core

## 2.1 Goal

Learn a small transferable parameter set

\[
\theta_{\rm core}
\]

that maps host-relative, scale-free, pre-outcome context to the IAH three-atom candidate measure:

\[
F_{d,h}
=
w^-_{d,h}\delta_{z^-_{d,h}}
+
w^0_{d,h}\delta_{z^0_{d,h}}
+
w^+_{d,h}\delta_{z^+_{d,h}}.
\]

The core should learn **correction geometry**, not dataset identity.

## 2.2 Minimal common input

All datasets must support the core path.

Recommended common input:

\[
Z^{\rm core}_{d,h}
=
\{
z^0_{d,h},
u_{d,h},
T_{d,h},
L^{\rm sf}_{d,h}
\},
\]

where:

- \(z^0\): host-anchored hyperbolic coordinate from the v0.3 math;
- \(u\): local S1 continuous rank context;
- \(T\): calendar/time representation;
- \(L^{\rm sf}\): legal lag/history features expressed in scale-free coordinates.

The core must **not** require dataset-specific z-score price coordinates as its only price representation.

## 2.3 Host-relative scale-free lag context

Current mixed raw/normalized lag channels should be replaced in the formal core by scale-free quantities.

Preferred family:

- lagged host hyperbolic coordinate;
- lagged realized target hyperbolic coordinate when legally available;
- lagged hyperbolic residual \(z^Y-z^0\);
- lagged rank/state values;
- deterministic time gaps / availability masks.

The exact minimal set may be adjusted during implementation, but every core price-like channel must have a documented scale transformation.

## 2.4 Core output

The IAH head retains v0.3 semantics:

\[
(w^-,w^0,w^+)=\operatorname{softmax}(\ell^-,0,\ell^+),
\]

\[
m^-=\operatorname{ReLU}(r^-),\qquad
m^+=\operatorname{ReLU}(r^+),
\]

\[
z^-=z^0-m^-,
\qquad
z^+=z^0+m^+.
\]

The weights are predictive distribution masses, not action probabilities.

The core does not decide whether to execute the action.

---

# 3. Layer II — Data-Adaptive Interface

## 3.1 Design objective

HCH should not classify “this is Shandong / Germany / NEM”.

It should infer:

> “Given the information that is observable here, and the scale/dynamics of this series, how should the shared correction core operate?”

Therefore dataset identity is replaced by a **Data Signature**.

\[
e_{g,d}
=
\Phi
\left(
e^{\rm obs},
e^{\rm dist},
e^{\rm dyn},
e^{\rm learned}
\right).
\]

## 3.2 Stable human-designed geometry

Only a small set of deterministic descriptors should be engineered.

### Observability signature

Describes what is available:

- context/history length;
- number of optional covariates;
- missingness / availability ratio;
- sampling interval or time-gap representation;
- count of known-future vs observed-past optional inputs.

### Distribution signature

Prefer scale-free summaries derived from host/history geometry:

- robust quantiles in hyperbolic coordinates;
- IQR / robust dispersion;
- zero-crossing rate;
- robust tail asymmetry;
- local volatility.

These are not labels and must not use the current outcome.

### Dynamics signature

Keep the deterministic part minimal:

- a small number of lag correlations / periodicity descriptors;
- recent local change / volatility summaries;
- time-resolution indicator.

The purpose is to provide stable anchors, not to manually encode every domain property.

## 3.3 Learned signature

The remainder is learned:

\[
e^{\rm learned}
=
\operatorname{Pool}(E_{\rm core}(Z^{\rm core})).
\]

The intended principle is:

\[
\boxed{\text{a little stable geometry + learned representation}}
\]

rather than a hand-built dataset classifier.

## 3.4 Conditioning mechanism

First implementation should use lightweight continuous modulation, not MoE:

\[
(\gamma,\beta)=H_{\omega}(e_{g,d}),
\]

\[
h'=\gamma\odot h+\beta.
\]

Acceptable first implementations:

- FiLM-style affine modulation;
- AdaLN-style modulation;
- zero-initialized residual conditioning.

Do **not** add:

- expert routing;
- expert balancing loss;
- dataset-specific candidate heads;
- a separate domain-classification loss.

## 3.5 market_id / target_id policy

`market_id` and `target_id` remain useful **audit metadata** for:

- data splitting;
- reporting;
- local S1/CAGM ownership;
- information-cutoff contracts;
- DA/RT analysis.

They should **not be mandatory predictive embeddings in the universal candidate core**.

A local S1 rank reference should normally be instantiated per deployment series/domain, so a neural market/target token is not required merely to obtain a local rank.

---

# 4. Optional Covariate Enrichment

## 4.1 Core principle

Price-only or univariate input is a complete mode, not a damaged rich-feature mode.

The architecture should satisfy:

\[
h_{\rm final}
=
h_{\rm core}
+
g(e,M)\odot h_{\rm optional}.
\]

When no optional covariate is available:

\[
h_{\rm optional}=0
\quad\Rightarrow\quad
h_{\rm final}=h_{\rm core}.
\]

## 4.2 Generic information roles

The lowest-level optional interface should avoid electricity-market-specific feature names.

Minimum generic roles:

- `KNOWN_FUTURE_COVARIATE`
- `OBSERVED_PAST_COVARIATE`
- `STATIC_COVARIATE`
- `CALENDAR`
- `OTHER_LEGAL_PRE_OUTCOME`

Optional semantic tags such as `wind`, `load`, `solar`, `weather` may be provided, but the module must remain valid without them.

## 4.3 Homogenization

Different covariates are first projected into a common token space:

\[
v_j
\rightarrow
E_{\rm role}(v_j,\text{role},\text{mask},\text{time relation})
\]

and then pooled/attended into \(h_{\rm optional}\).

The optional branch must preserve explicit masks and must never infer information availability from numerical values alone.

## 4.4 Non-destructive injection

The optional branch should be residual and preferably zero-initialized.

Goal:

\[
\text{HCH-Rich at initialization}
\approx
\text{HCH-Core}.
\]

This makes extra information an enhancement rather than a rewrite of the universal core.

---

# 5. Layer III — Target-Local Evidence Layer

The local layer remains nonparametric / calibration-oriented.

It contains:

\[
\mathcal R_{\rm S1},
\quad
\mathcal M_{\rm CAGM},
\quad
k,
\quad
\mathcal P_{\rm event},
\quad
q_{\rm DVG}.
\]

## 5.1 Local rank

Build from target-domain S1 host predictions without current-day target leakage.

## 5.2 CAGM

Store the v0.3 residual three-atom measure and outcome-separated replay fields.

Retrieval uses exact 1D W1 over shared valid hours.

## 5.3 Query-dose replay

History evaluates the **query day’s dose**.

No history-own-dose substitution is allowed.

## 5.4 Double event

At most:

- one contiguous Down interval;
- one contiguous Up interval;
- non-overlapping;
- either may be empty.

## 5.5 DVG

S3-C calibrates only the already frozen whole-day policy:

\[
E_t=\widehat A_t-A_t,
\]

\[
LCB_q=\widehat A_q-q_{1-\alpha}.
\]

Execute only if:

\[
LCB_q>0.
\]

No target-domain calibration means no target-domain certified safety claim.

---

# 6. Three deployment modes

## Mode A — Frozen Core / zero-label candidate transfer

Inputs:
- frozen host forecasts;
- unlabeled target-domain host/history context.

Use:
- frozen \(\theta_{\rm core}\);
- data signature;
- local rank if constructible from host predictions.

Claims:
- candidate transfer may be evaluated;
- target-domain DVG certification is unavailable without target outcomes.

## Mode B — Few-shot local adaptation

Freeze \(\theta_{\rm core}\).

Train only a small local adapter:

\[
\phi_g
\]

with the same IAH-CRPS.

Then build local memory/calibration from available target outcomes.

Candidate parameters remain largely shared; local adaptation is parameter-efficient.

## Mode C — Full local evidence

For a domain with sufficient history:

- frozen universal core;
- optional branch if available;
- local S1;
- local CAGM;
- frozen \(k\);
- S3-C DVG quantile.

This is the main safe-deployment protocol.

---

# 7. Separation between architecture and training

The architecture defines **what components exist**.

Training protocol defines **how their parameters are obtained**.

The architecture must not hard-code U0/U1/U2 into the mathematical loss.

Specifically:

- IAH candidate semantics remain unchanged;
- U0 may initialize the representation;
- U1 trains the core with IAH-CRPS;
- U2 trains optional/context adaptation with IAH-CRPS while core is frozen;
- local adapters, if used, also use IAH-CRPS;
- CAGM/DVG remain evidence/calibration components, not neural auxiliary losses.

---

# 8. Frozen package

A formal frozen HCH package should contain two parts.

## 8.1 Universal package

- architecture version;
- \(\theta_{\rm core}\);
- Data Signature specification/version;
- optional-context parameters if present;
- core/optional feature-role contract;
- training provenance;
- source domain list;
- source host list;
- optional U0 teacher provenance.

## 8.2 Local package

- target-domain S1 rank reference;
- CAGM atom memory;
- memory timestamps and hashes;
- frozen \(k\);
- proposal/tie-break version;
- DVG \(\alpha\), errors and/or \(q\);
- information-cutoff contract hash;
- fallback reasons.

Universal package and local package must be separable.

---

# 9. Required invariances / degradation behavior

The final architecture must explicitly test:

1. positive price-unit scaling preserves \(z^0,w,m,W_1,\pi\) and scales raw actions accordingly;
2. removing all optional covariates reproduces the core path;
3. changing `market_id`/`target_id` audit labels alone does not change the universal candidate output;
4. a price-only dataset is a first-class valid input;
5. unseen optional feature counts are handled by set/token aggregation rather than fixed-column assumptions;
6. missing optional data degrades toward HCH-Core, not to arbitrary learned-null behavior;
7. target-free S4 does not access target/residual/action gain.

---

# 10. Paper-level framing

Recommended conceptual statement:

> **HCH separates transferable host-relative correction knowledge from data-adaptive observability modeling and target-local historical evidence calibration.**

The architectural novelty should not be written as “we identify datasets automatically”.

A more defensible description is:

> HCH uses a lightweight data-conditioned interface to modulate a shared host-relative correction prior according to pre-outcome observability and scale-free time-series dynamics, while keeping historical action evidence and value calibration local to the target domain.

The MoE comparison is conceptual:

- MoE: choose/specialize experts;
- HCH: continuously modulate one compact shared correction law.

Do not claim superiority to MoE without direct experiments.

---

# 11. Scope for the first experiment cycle

Do not implement every possible generalization before the first training run.

First experiment cycle should require only:

1. valid HCH-Core path;
2. lightweight Data Signature;
3. optional residual branch with masking;
4. v0.3 local evidence chain;
5. universal/local bundle separation.

Defer unless evidence demands them:

- full MoE;
- multimodal text/image covariates;
- large local adapters;
- online continual learning;
- multi-teacher routing;
- non-electricity main-table claims.

---

# 12. Architecture acceptance gate

Architecture is ready for training only when:

- the formal runner uses the new IAH path and never the legacy HCH path;
- core-only input works on price-only public data;
- optional branch can be disabled exactly;
- S3-M and S3-C are distinct;
- final event proposal is replayed as final \(\pi_q\);
- bundle round-trip reproduces candidate, neighbors, proposal, \(\widehat A\), \(q\), LCB and final action;
- audit metadata IDs are not silently used as predictive shortcuts;
- all P0 items in the code-correction document are closed.
