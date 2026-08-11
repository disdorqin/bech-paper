# HCH v2 Mathematical Evidence Audit: Shandong and Public Cross-Market Data

> Version: v0.1
> Date: 2026-08-11
> Target repository: `https://github.com/disdorqin/bech-paper`
> Reference audit commit: `a3770bf813d56b2c597cd7917ee57fd46a8f654f`
> Purpose: a directly executable contract for a local coding/research AI
> Scope: data inspection, leakage-safe diagnostics, statistical evidence export, and blocker reporting only
> Out of scope: production-model modification, new loss implementation, S4 evaluation, formal winner selection, and paper claims

---

## 0. Role and governing objective

Act as a rigorous data auditor and statistical research engineer for a frozen-host electricity-price post-processing project.

Your job is **not** to prove that HCH v2, BOM-SSC, CAGM-DVG, Bi-OMC, Student-t, or skew-t is effective. Your job is to produce the leakage-safe empirical evidence needed by a separate mathematical research window to decide:

1. what residual-generating assumptions are defensible;
2. whether heavy tails and asymmetry are stable mechanisms or artifacts of a few events;
3. whether directional occurrence and magnitude can be estimated coherently;
4. whether Up/Down candidates should use partial moments, conditional actions, or another distribution-derived decision quantity;
5. whether 24-hour dependence requires a joint model;
6. what can be shared across markets, channels, and frozen hosts;
7. whether a normal-regime harm budget versus tail-repair gain is empirically feasible;
8. which facts are known, unknown, blocked, or invalid because of data semantics.

Do not optimize for a positive conclusion. A clean negative result is more valuable than an unsupported favorable result.

---

## 1. Scientific positioning that must be preserved

### 1.1 Different roles of private and public data

The Shandong dataset is:

- the main problem-origin dataset;
- the main industrial/application dataset because the supervisor needs practical value from the work;
- private and non-releasable;
- allowed to motivate the research question and inform method design;
- **not automatically an untouched external-validation dataset** if it has already influenced architecture, thresholds, hypotheses, or hyperparameters.

The public datasets are:

- the primary reproducible evidence for the paper;
- the basis for fair public comparisons;
- the proper setting for temporal frozen tests and any claimed cross-market generalization;
- not substitutes for the Shandong use case, but an independently auditable evidence base.

Never blur these roles. In every output, label evidence as one of:

- `PRIVATE_DESIGN_ORIGIN`
- `PRIVATE_APPLICATION_EVIDENCE`
- `PUBLIC_DEVELOPMENT_EVIDENCE`
- `PUBLIC_FROZEN_TEST_EVIDENCE` — prohibited in this task because S4 values must not be used
- `CROSS_MARKET_TRANSFER_EVIDENCE`

### 1.2 Three different meanings of “frozen”

Audit and report these separately:

1. **Host freeze**: the base forecaster is trained first and cannot be updated by HCH.
2. **Temporal freeze**: all HCH parameters, distribution choices, state transforms, thresholds, memory, routing parameters, and model-selection decisions are fixed before S4.
3. **Market-transfer freeze**: a corrector or shared parameterization trained on source markets is applied to an unseen target market with zero-shot or explicitly bounded few-shot adaptation.

Do not call an experiment “cross-market frozen” merely because its host is frozen. Do not call a method “zero-shot” if any target-market labels, residuals, scalers, distribution parameters, thresholds, memory gains, or early-stopping decisions were used.

---

## 2. Hard safety, privacy, and integrity rules

### 2.1 Repository safety

- Begin by recording the actual repository path, current branch, `git status`, and `HEAD` commit.
- Compare `HEAD` with the reference audit commit above, but do not checkout, reset, merge, pull, or rewrite history.
- Do not modify production code, model definitions, training logic, data loaders, split definitions, or existing artifacts.
- Place all new scripts and outputs in one new isolated audit directory. Suggested name:

  `research_outputs/math_evidence_audit_2026-08-11/`

- Do not commit or push unless separately authorized.
- Preserve all unrelated user changes.

### 2.2 S4 prohibition

No statistical routine in this task may read S4 `y_true`, residuals, action gains, loss values, or distributional values.

S4 may be inspected only for:

- schema names and dtypes;
- split-boundary metadata;
- row counts;
- key uniqueness;
- timestamp/index presence;
- whether a file exists;
- whether code paths appear capable of fitting or updating on S4.

If the repository cannot prevent S4 values from being loaded, stop the affected computation and mark it `BLOCKED_BY_S4_ISOLATION`.

### 2.3 Private-data handling

- Do not copy raw Shandong rows into reports.
- Do not export proprietary full time series, individual bid records, credentials, tokens, or operationally sensitive free-text fields.
- Aggregate statistics, schemas, units, date spans, event counts, quantiles, model residual summaries, and non-identifying plots are allowed.
- If a table would contain exact timestamps paired with private prices or proprietary covariates, aggregate by hour/month/state or redact the timestamp.
- Record private source paths only in a local-only manifest if needed; in the shareable report use a stable alias such as `SHANDONG_DA_SOURCE_01`.

### 2.4 Honesty rules

For every requested item, return exactly one status:

- `COMPUTED`
- `NOT_AVAILABLE`
- `BLOCKED`
- `INVALID_SEMANTICS`
- `INSUFFICIENT_SAMPLE`
- `NOT_APPLICABLE`

Never:

- invent a field definition, time boundary, unit, market rule, sample count, or result;
- silently replace a missing source with another dataset;
- treat an in-sample residual as OOF evidence;
- repair timestamps by guesswork;
- silently drop hard cases, missing days, DST days, price-cap events, or negative prices;
- select the most favorable threshold, host, seed, market, or metric;
- report a p-value without the effect size, sample unit, and dependence treatment;
- call an exploratory fit a validated distribution;
- call an oracle result an implementable result.

When a value is inferred rather than explicitly documented, label it `INFERRED` and state the evidence and uncertainty.

---

## 3. Canonical data partitions for this audit

Use the repository’s authoritative S1/S2/S3/S4 definitions if they exist. Do not create new boundaries simply to make analysis possible.

The intended evidence roles are:

| Stage | Permitted role in this audit |
|---|---|
| S1 | raw-data and host-training inventory; residual evidence only if predictions are genuinely cross-fitted |
| S2 | fit exploratory statistics, transforms, bins, thresholds, distribution parameters, and nonparametric candidate rules using time-OOF host predictions |
| S3 | held-out development evaluation, calibration diagnostics, candidate comparison, and safety feasibility analysis |
| S4 | metadata-only audit; no target-dependent computation |

The preferred residual evidence is:

$$
R_{d,h}^{(m,c,j)}=Y_{d,h}^{(m,c)}-\widehat Y_{d,h}^{host,(m,c,j)},
$$

where `m` is market/dataset, `c` is channel, and `j` is frozen host identity.

Every residual row used must have provenance proving that the host prediction was available without training on that target row.

If valid S2 time-OOF and S3 held-out host predictions already exist, use them. If they do not exist:

- you may regenerate S2/S3 predictions only through an existing authoritative pipeline with unchanged split and host hyperparameters;
- do not silently write replacement training code;
- do not tune the host;
- do not run S4;
- if regeneration requires semantic or production-code changes, report the exact blocker and command plan rather than proceeding.

---

## 4. Phase A — repository, dataset, and information-boundary audit

Complete Phase A before any distribution fitting.

### A1. Dataset inventory

For every private and public dataset/channel, report:

- stable `dataset_id`, `market_id`, and `channel_id`;
- private/public evidence role;
- source type and public official URL/license when applicable;
- raw and processed file aliases;
- date span by split;
- timezone and DST handling;
- market-day definition;
- temporal resolution and expected horizons per day;
- currency and price unit;
- whether prices can be negative;
- official or observed price floor/cap;
- price tick/rounding behavior and repeated boundary mass;
- target definition;
- forecast origin, issuance time, and horizon;
- number of complete and incomplete days;
- duplicate-key count;
- missing-target count;
- missing-feature counts;
- known regulatory, market-rule, or data-generation changes.

For Shandong DA and RT, verify rather than assume:

- the exact information cutoff;
- whether every auxiliary series is a forecast, plan, realized value, revision, or unknown;
- whether RT uses any information unavailable at the declared forecast origin;
- the actual negative-price floor/cap rules represented in the data;
- whether DA and RT timestamps refer to delivery hour, publication time, or ingestion time;
- whether the same delivery day is aligned identically across targets, features, and host predictions.

### A2. Feature availability and leakage matrix

Create one row for every feature in every dataset configuration with:

- feature name or private-safe alias;
- semantic description;
- unit;
- source;
- raw timestamp meaning;
- first time it becomes available;
- revision behavior;
- value type: forecast / plan / realized / calendar / lag / target-derived / unknown;
- usable for DA at issuance: yes/no/unknown;
- usable for RT at issuance: yes/no/unknown;
- available in public markets: all/some/none;
- common cross-market representation, if one exists;
- leakage risk and evidence.

Explicitly detect:

- future realized load, wind, solar, generation, interconnector, or price;
- backfilled values presented as forecasts;
- target-derived residual/error columns;
- scalers fitted after the relevant cutoff;
- centered rolling windows;
- full-series ranks or quantiles;
- target-day aggregates unavailable at issuance;
- timestamp shifts that accidentally expose the target.

Create a separate `common_information_set.csv` that distinguishes:

1. features available in all public and private markets;
2. public-only or dataset-specific features;
3. Shandong-only operational features;
4. features that must never enter a reproducible common configuration.

### A3. Split and freeze ledger

For every dataset/channel/host, report:

- exact S1/S2/S3/S4 boundaries;
- split unit: timestamp/day/week/etc.;
- purge or gap;
- OOF fold definitions;
- host fit data;
- HCH fit data;
- state/scaler fit data;
- distribution-choice data;
- memory-build data;
- gate-calibration data;
- hyperparameter-selection data;
- whether S3 can self-retrieve;
- whether any fit/update path is reachable during S4;
- freeze artifact or bundle proving parameters are fixed.

Audit the three freeze meanings separately. If a market-transfer experiment exists, report exactly which target-market information is consumed.

### A4. Canonical keyed residual table audit

The conceptual key is:

`dataset_id, market_id, channel_id, host_id, split_id, delivery_day, horizon, timestamp`

Verify:

- uniqueness;
- consistent timezone;
- common S2/S3 index across hosts and methods;
- target/prediction alignment;
- 23/24/25-hour days;
- incomplete horizons and masks;
- OOF provenance;
- no projection or coordinate transform applied twice;
- residual sign convention `y_true - host_pred`.

Do not export private raw rows. Export only audit counts and keyed-table fingerprints/checksums.

### Phase A stop condition

Do not continue to Phase B for an affected dataset/channel/host if:

- residual sign is ambiguous;
- predictions are in-sample;
- timestamps are misaligned;
- split provenance is missing;
- S2/S3 cannot be separated from S4;
- target semantics differ across files without a verified mapping.

Report unaffected combinations normally.

---

## 5. Phase B — leakage-safe empirical residual geometry

Run all computations per `dataset × channel × host` before any pooling. Also produce carefully defined macro-aggregates that weight markets equally, not only sample-weighted totals.

Use day-level or market-day block bootstrap confidence intervals. Use at least 2,000 bootstrap replicates if computationally feasible; otherwise report the number used and why. Preserve temporal order within sampled blocks.

### B1. Basic and robust summaries

On valid S2-OOF and S3 residuals separately, compute:

- sample count and effective number of market days;
- mean, median, standard deviation;
- MAD with its stated consistency scaling, IQR, and one additional robust scale if available;
- minimum and maximum;
- quantiles at `0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999` when sample size permits;
- Fisher skewness and excess kurtosis with uncertainty;
- Bowley/quantile skewness and medcouple if a reliable implementation is available;
- trimmed/winsorized sensitivity versions, clearly labeled and never substituted for raw results;
- `P(R<0)`, `P(R=0)`, `P(R>0)`;
- `E[R 1(R<0)]`, `E[R 1(R>0)]`;
- `E[-R | R<0]`, `median(-R | R<0)`;
- `E[R | R>0]`, `median(R | R>0)`;
- positive and negative tail event counts;
- contribution of the largest 1, 5, 10, and top 0.1% absolute residuals to total squared error and absolute error.

For the target price itself, separately compute:

- `P(Y<0)`, `P(Y=0)`, and counts at known price bounds;
- positive/negative price quantiles;
- frequency of exact repeated prices or boundary values;
- physical negative-price episodes by day and consecutive duration.

Never conflate `Y<0` with `R<0`.

### B2. Tail stability and “few-event domination”

Assess whether apparent heavy tails/asymmetry are stable across:

- S2 versus S3;
- year/season/month;
- hour of day;
- weekday/weekend if meaningful;
- host identity;
- rolling windows with fixed, predeclared lengths;
- removal of the largest 1, 5, and 10 events, reported only as sensitivity analysis;
- market-rule or known structural-break periods.

Produce:

- empirical survival curves for positive and negative residuals on comparable standardized axes;
- QQ plots against Normal, Laplace, and fitted symmetric Student-t;
- Hill or another tail-index stability plot only when the number of exceedances is sufficient;
- block-bootstrap intervals for tail-index estimates;
- change-point or distribution-drift diagnostics with effect sizes;
- rolling robust location, scale, sign probability, and tail mass.

Do not conclude “power law” or “Student-t” from a straight-looking plot. Mark tail-index results unreliable when thresholds are unstable.

### B3. Outcome slices versus predictive states

Every conditional analysis must label its conditioning variable as either:

- `PREDICTIVE_STATE`: computable from information available at forecast origin; or
- `OUTCOME_DEFINED_SLICE`: uses `Y` or `R` and is allowed only for evaluation/description.

Never present an outcome-defined slice as an input state.

For outcome-defined sensitivity, define high and low thresholds on S2 only and freeze them for S3. Include at least:

- residual upper/lower 95%, 97.5%, and 99% tails;
- price upper/lower 95%, 97.5%, and 99% tails;
- physical negative prices `Y<0` as a separate slice;
- market cap/floor boundary events where applicable.

Do not select one threshold as “best.” Report the grid.

For predictive states, use only verified available variables. Candidate state families to inspect include:

- host prediction robust rank based on past/fitting data only;
- past-visible rolling price/residual location and scale;
- hour/day/season;
- host uncertainty proxy if genuinely available at issuance;
- private exogenous forecasts after their availability is verified;
- a common-information configuration shared by public markets;
- a Shandong-enriched configuration reported separately.

Fit all bin edges or smoothers on S2 and freeze them for S3. Report cell counts and uncertainty. Do not interpret sparse cells; use `INSUFFICIENT_SAMPLE` rather than silent merging unless the merge rule was predeclared.

### B4. Occurrence–magnitude decomposition

For every valid predictive-state bin or predeclared smooth state value, estimate on S2 and evaluate stability on S3:

$$
\pi_-(Z)=P(R<0\mid Z),\quad \pi_+(Z)=P(R>0\mid Z),
$$

$$
m_-(Z)=E[-R\mid R<0,Z],\quad m_+(Z)=E[R\mid R>0,Z],
$$

and

$$
M_-(Z)=-\pi_-(Z)m_-(Z),\quad M_+(Z)=\pi_+(Z)m_+(Z).
$$

Also estimate side-conditional medians and robust magnitudes.

Report:

- calibration/reliability of `pi_-` and `pi_+` from S2 to S3;
- stability of conditional magnitudes;
- correlation and mutual dependence between occurrence and magnitude estimates;
- how much `M_±` shrinks relative to side-conditional mean/median actions;
- behavior when `pi` is small, moderate, or close to one;
- sample sizes and block-bootstrap intervals.

The purpose is to test whether one coherent conditional distribution could plausibly share occurrence and magnitude—not to assume that partial moments are the correct actions.

### B5. Minimal held-out distribution diagnostics

Compare only the smallest justified nesting initially:

- empirical/nonparametric reference;
- Normal;
- Laplace;
- symmetric Student-t (`M0`);
- Fernández–Steel skew-t (`M1`) only if its implementation passes normalization, symmetry, and sampling tests.

Rules:

- Fit parameters on S2 only and evaluate on S3.
- Fit per dataset/channel/host first.
- Record optimizer, initialization, convergence status, boundary hits, Hessian/profile behavior, and random seed.
- Do not use a custom skew-t implementation unless numerical integration confirms density normalization across a declared parameter grid.
- Do not proceed to generalized asymmetric t, mixture, copula, or EVT merely because M1 wins by a small NLL margin.
- If a reliable implementation is unavailable, mark M1 `NOT_AVAILABLE`; do not substitute skew-normal and label it skew-t.

Evaluate with:

- held-out mean NLL and day-block bootstrap difference intervals;
- PIT histogram and uniformity diagnostics, emphasizing effect size;
- central and signed-tail coverage/calibration;
- CRPS or a documented numerical approximation;
- positive/negative log-score contributions;
- parameter stability across time blocks and seeds;
- likelihood profiles or practical identifiability diagnostics for location, scale, skew, and degrees of freedom;
- frequency with which fitted `nu` approaches moment-existence boundaries;
- AIC/BIC only as secondary descriptive quantities, never as the sole selector.

Explicitly test whether skew improvement remains after removing or downweighting a few largest events as sensitivity analysis. Report raw and sensitivity results together.

---

## 6. Phase C — candidate-action semantics without implementing HCH v2

This phase is a diagnostic comparison, not a new production method.

### C1. Identify the actual loss/utility semantics

Find and report the exact implemented definitions of:

- MAE;
- MSE/RMSE if used;
- sMAPE and every floor/clipping convention;
- pinball or probabilistic scores if used;
- action gain used by CAGM/DVG;
- any business/economic loss, including units and asymmetric costs.

Do not assume the paper metric is the same as the action utility. If no economic utility is formally defined, say so.

### C2. Leakage-safe candidate comparison

Using only S2-fitted predictive-state bins/smoothers and S3 evaluation, compare the following residual corrections where estimable:

1. Identity: `delta_0 = 0`;
2. unconditional/conditional mean residual;
3. unconditional/conditional median residual;
4. sign-constrained mean action;
5. sign-constrained median action;
6. Down side-conditional mean/median;
7. Up side-conditional mean/median;
8. directional partial moments `M_-` and `M_+`.

Use the exact correction sign convention `host_pred + delta`.

For each candidate dictionary, report S3 gains relative to Identity under each legitimate loss:

$$
G^a=\ell(Y,\widehat Y^{host})-\ell(Y,\widehat Y^{host}+\Delta^a).
$$

Report:

- overall mean/median gain;
- high-tail, low-tail, physical-negative, and normal-slice gain;
- probability and expected magnitude of harm;
- day-block bootstrap intervals;
- candidate magnitude distribution;
- oracle-versus-implementable gap;
- frequency each action is pointwise oracle best;
- margin between best and second-best action;
- whether partial-moment candidates show double-shrinkage relative to side-conditional actions.

All pointwise oracle selections must be labeled `POST_OUTCOME_ORACLE_UPPER_BOUND`. They are never deployable policies.

### C3. Empirical tail-repair versus normal-harm frontier

For each valid candidate dictionary, construct an empirical S2/S3 diagnostic frontier:

$$
\max_{\pi}\;E[G_{tail}(\pi)]
\quad\text{subject to}\quad
E[(-G_{normal}(\pi))_+]\le\epsilon.
$$

Use a predeclared grid of `epsilon` values, including zero and scale-normalized values. Produce two distinct frontiers:

1. pointwise oracle upper bound using outcomes, clearly labeled;
2. implementable S2-trained/S3-evaluated policy using only predictive-state features, if an existing valid policy or a predeclared simple diagnostic policy is available.

Do not tune `epsilon` on S3 and then report the same S3 result as validation. Do not claim a finite-sample guarantee; this phase only tests whether a nontrivial feasible region appears to exist.

Perform sensitivity over multiple outcome-defined definitions of `tail` and `normal`; do not choose the most favorable one.

---

## 7. Phase D — 24-hour dependence and event-path evidence

The goal is to decide whether hourly conditional likelihood (`W1`) is an adequate composite likelihood or whether a minimal joint component (`W2`) is empirically necessary.

For valid S2 and S3 residual days, compute per dataset/channel/host:

- 24×24 residual correlation and rank-correlation matrices;
- 24×24 absolute-residual dependence matrices;
- dependence after robust per-hour standardization;
- within-day residual ACF by lag;
- consecutive sign-run and tail-event run-length distributions;
- probability that one extreme hour is followed by another within the day;
- day-level robust scale and skew proxies;
- variance explained by a shared day-level scale factor;
- eigenvalue spectrum, effective rank, and variance explained by the first 1–5 factors;
- stability of low-rank structure from S2 to S3;
- dependence conditional on available day-level context when feasible.

For event paths, use S2-frozen thresholds and report:

- true versus host-predicted daily peak/trough hour offset;
- magnitude error conditional on timing offset;
- event duration and integrated event mass/area;
- recall with ±0, ±1, and ±2-hour timing tolerance;
- whether apparent large pointwise residuals are mainly amplitude failures, timing shifts, or both.

Do not introduce a timing loss. Return evidence relevant to choosing W1 or W2.

Suggested decision evidence:

- support W1 when most remaining dependence is weak after conditioning/standardization and a joint layer offers little stable held-out gain;
- investigate W2 when a stable low-rank/day-scale dependence remains across S2/S3 and materially affects tail calibration or candidate decisions;
- do not recommend W3 copula/path modeling in this audit.

---

## 8. Phase E — cross-market sharing and transfer geometry

### E1. Comparable normalization

For every market/channel/host, define price and residual normalizations using S1/S2 information only. Report at least:

- robust location/scale normalization;
- unit/currency normalization where economically valid;
- behavior under positive affine rescaling;
- whether negative-price semantics survive the transformation;
- whether sMAPE or floor-based metrics break affine/scale comparability.

Never pool raw-price residuals across currencies or markets without an explicit transformation.

### E2. Shared versus market-specific residual geometry

After valid normalization, compare markets using:

- quantile-function distances;
- Wasserstein distance;
- signed-tail mass and magnitude differences;
- Student-t/FS parameter differences with uncertainty;
- distribution-distance heatmaps and clustering as descriptive evidence;
- within-market versus between-market variance of location, scale, skew, and tail parameters;
- host-conditioned differences—never erase host identity while claiming model independence.

Report whether evidence favors:

- one fully shared residual distribution;
- shared tail geometry with market-specific location/scale/skew;
- market/channel-specific distributions with only a shared backbone;
- no defensible pooling for certain pairs.

### E3. Negative-price knowledge transfer

Separate:

- downward residual semantics `R<0`;
- low-price states;
- physical negative prices `Y<0`;
- price-floor/censoring events.

Quantify per market:

- event counts and days;
- conditional downward magnitude in adaptive-low states;
- how physical-negative events differ from ordinary downward residuals in available predictors and path structure;
- whether upper-tail and lower-tail standardized geometry are similar enough to justify shared parameters;
- counterexamples where sharing upper-tail knowledge would create negative transfer.

Do not train a new `Y<0` classifier.

### E4. Transfer protocols audit

If repository artifacts support cross-market experiments, classify each as:

- in-market temporal generalization;
- pooled multi-market training;
- leave-one-market-out zero-shot;
- target-scaler-only adaptation;
- few-shot target adaptation;
- full target retraining.

For every protocol, enumerate every target-market quantity consumed. A target scaler, residual quantile, state bin, memory episode, or gate threshold is adaptation and must be disclosed.

If feasible without new model development, perform a minimal diagnostic comparison of:

- market-specific M0/M1 fits;
- pooled normalized fit;
- leave-one-market-out normalized fit;
- optional explicitly labeled target-scaler adaptation.

Fit on source S2 only and evaluate on target S3 only. This is a distribution-transfer diagnostic, not an HCH performance claim.

---

## 9. Phase F — CAGM/DVG evidence availability and risk overlap

Do not trust current candidate-gain artifacts unless candidate anchoring, OOF coordinates, and retrieval semantics have been verified as corrected.

First determine whether valid S2/S3 artifacts exist for:

- `host_pred`;
- Down/Up candidate values relative to `host_pred`;
- candidate uncertainty;
- day key;
- retrieved neighbor IDs and distances;
- realized per-action gain;
- neighbor gain mean/variance;
- DVG score/value;
- selected action;
- pointwise oracle action.

If any semantic defect remains, output the required schema and mark downstream items `BLOCKED`; do not analyze corrupted gains.

If valid artifacts exist, report on S2/S3 only:

- Identity/Down/Up action-gain distributions;
- identity-optimal frequency;
- tail and normal action confusion;
- neighbor-distance versus gain-prediction quality;
- effective number and diversity of retrieved days;
- self-retrieval and temporal-overlap checks;
- variance of gains within retrieved neighborhoods;
- relationship between candidate predictive uncertainty and retrieval gain variance;
- correlation/redundancy among aleatoric proxy, retrieval uncertainty, candidate error, and DVG value;
- whether risk terms appear to double-count the same failure mode.

Do not claim an aleatoric/epistemic decomposition merely from naming two variances. Report observable relationships and unresolved identification limits.

---

## 10. Required machine-readable outputs

Produce, at minimum:

1. `00_EXECUTIVE_EVIDENCE_VERDICT.md`
2. `01_DATASET_AND_INFORMATION_BOUNDARY.md`
3. `02_RESIDUAL_GEOMETRY.md`
4. `03_OCCURRENCE_MAGNITUDE_AND_ACTIONS.md`
5. `04_DAY_DEPENDENCE_AND_EVENT_PATHS.md`
6. `05_CROSS_MARKET_TRANSFER_GEOMETRY.md`
7. `06_CAGM_DVG_EVIDENCE_STATUS.md`
8. `07_BLOCKERS_AND_NEXT_MEASUREMENTS.md`
9. `dataset_manifest.csv`
10. `feature_availability.csv`
11. `common_information_set.csv`
12. `split_freeze_ledger.csv`
13. `host_prediction_provenance.csv`
14. `residual_summary.csv`
15. `conditional_residual_summary.csv`
16. `tail_threshold_sensitivity.csv`
17. `distribution_fit_holdout.csv`
18. `candidate_action_diagnostics.csv`
19. `safety_pareto.csv`
20. `day_dependence_summary.csv`
21. `event_path_summary.csv`
22. `cross_market_distance.csv`
23. `transfer_protocol_ledger.csv`
24. `cagm_dvg_risk_overlap.csv` or a schema-only blocked version
25. `blockers.csv`
26. `evidence_bundle.json`
27. `MANIFEST.json`
28. analysis scripts and a reproducible command log

If one logical table is too large, partition it by dataset while preserving a schema manifest.

Every CSV must include:

- `evidence_status`;
- `dataset_id` where applicable;
- `channel_id` and `host_id` where applicable;
- `split_id`;
- `n_observations`;
- `n_market_days`;
- uncertainty method;
- source artifact alias;
- code/script version or hash.

`evidence_bundle.json` must be concise and designed for the mathematical research window. It should contain:

- verified dataset semantics;
- verified forecast information boundaries;
- valid residual combinations;
- robust distribution summaries and intervals;
- occurrence/magnitude summaries;
- distribution-fit comparisons;
- candidate-action comparison summaries;
- 24-hour dependence summaries;
- cross-market pooling/transfer summaries;
- safety-frontier summaries;
- CAGM/DVG evidence readiness;
- all blockers and invalid combinations;
- exact paths to detailed local outputs.

Do not embed raw private data in the JSON.

---

## 11. Required figures

Create aggregate, non-identifying figures with clear dataset/channel/host labels:

- positive and negative residual survival plots;
- QQ plots;
- PIT/calibration plots for valid fitted distributions;
- rolling robust-statistics plots;
- hour-of-day residual and tail heatmaps;
- occurrence-versus-magnitude state plots;
- candidate shrinkage comparison plots;
- candidate gain and harm distributions;
- tail-repair versus normal-harm Pareto curves;
- 24×24 residual/absolute-residual dependence heatmaps;
- day-dependence eigenvalue spectra;
- peak/trough timing-error plots;
- cross-market distance heatmap;
- market-specific versus pooled parameter interval plots;
- negative-price versus adaptive-low comparison plots.

All axes must state units or normalization. Do not truncate tails without explicitly marking the truncation.

---

## 12. Questions the final audit must answer directly

Return a one-paragraph evidence answer, with confidence and blockers, for each question:

1. Are residual heavy tails stable across time, host, and market, or dominated by a few episodes?
2. Is asymmetry stable enough to justify M1 over M0?
3. Are location, scale, skew, and degrees of freedom practically identifiable with available sample sizes?
4. Do price caps/floors, censoring, discreteness, or point masses make a continuous skew-t structurally inadequate?
5. Does residual geometry vary smoothly with forecast-time observable state?
6. Can occurrence and magnitude share one conditional model without erasing direction-specific mechanisms?
7. Do partial-moment candidates exhibit material double-shrinkage relative to side-conditional actions?
8. Which loss/utility actually defines a rational Up/Down action in the project?
9. Is there a nontrivial empirical tail-gain/normal-harm feasible region?
10. Does stable 24-hour dependence remain after conditioning, enough to justify W2 over W1?
11. What can be pooled across markets and hosts, and what must remain market/channel/host-specific?
12. Does adaptive-low information transfer to physical negative prices, and where does negative transfer appear?
13. Is any current cross-market experiment truly zero-shot or frozen under a precise definition?
14. Are candidate predictive risk and CAGM retrieval risk empirically distinct or redundant?
15. Which mathematical assumptions are ready to formalize, which are falsified, and which remain untestable with current artifacts?

---

## 13. Reproducibility and statistical reporting

- Record Python/environment versions and package versions.
- Record all random seeds.
- Record exact commands and wall-clock failures.
- Use deterministic algorithms where practical.
- Unit-test timestamp keys, residual signs, split exclusion, masks, and scale transforms before analysis.
- Include day-block bootstrap code and verify it resamples the declared independent unit.
- Report confidence intervals and denominators, not only point estimates.
- For multiple markets/hosts/thresholds, present the full family; do not hide multiplicity.
- Favor effect sizes and stability over binary significance declarations.
- Report market macro-averages as well as pooled micro-averages.
- Preserve failed runs and convergence failures in the manifest.
- Hash or fingerprint all source artifacts used, without exposing private bytes.

---

## 14. Completion status

End `00_EXECUTIVE_EVIDENCE_VERDICT.md` with exactly one status:

### `COMPLETE_FOR_MATH_REVIEW`

Use only if Phase A semantics are valid and the core S2/S3 residual, occurrence–magnitude, candidate-action, dependence, and cross-market outputs are complete.

### `PARTIAL_BLOCKED`

Use if useful evidence was produced but one or more material combinations or phases are blocked. List the precise blockers, affected combinations, and the smallest legitimate next action.

### `INVALID_EVIDENCE`

Use if leakage, in-sample residuals, timestamp mismatch, target-semantic ambiguity, or S4 contamination makes the main results unusable.

Do not use optimistic wording to compensate for a blocked status.

---

## 15. Stop conditions and handoff

Stop after producing and validating the audit bundle.

Do not:

- implement the new probabilistic loss;
- modify HCH v2;
- select M0/M1 as the final model;
- tune DVG/CAGM;
- unfreeze half-exp;
- run S4;
- choose a best seed;
- write novelty claims;
- claim safety, model independence, cross-market generalization, or SOTA.

The mathematical research window will use this evidence to derive:

1. the residual-generating model;
2. directional Bayes-action semantics;
3. the candidate dictionary;
4. a non-duplicative CAGM-DVG risk decomposition;
5. a normal-harm-constrained selective routing result;
6. the unique implementation contract.

Return the full output directory path, the completion status, the valid evidence combinations, and the blocker list. Do not summarize unavailable computations as completed.
