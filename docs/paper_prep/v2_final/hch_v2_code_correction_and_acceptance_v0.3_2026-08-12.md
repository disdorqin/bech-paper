# HCH-v2 Code Correction and Acceptance Specification v0.3

> Date: 2026-08-12  
> Audit baseline: `main@97449234294686a2ca78d8fb59d76b54e9e0eb15`  
> Target architecture: v0.4 Universal Adaptive Architecture  
> Mathematical authority: v0.3 IAH-CRPS math core  
> Decision: **FORMAL TRAINING BLOCKED until all P0 items pass**.

---

# 0. Executive verdict

The repository currently contains two different realities:

1. new v0.3 mathematical modules (`iah_candidate.py`, `iah_crps_loss.py`, `s1_rank.py`, `w1_retrieval.py`, `query_replay.py`, `double_event.py`, `dvg_calibrate.py`);
2. the active `smoke_v2.py` and `hch_v2.py` path still executes the legacy HCH formulation.

Therefore:

\[
\boxed{
\text{module-level v0.3 implementation exists}
\neq
\text{formal v0.3 HCH system exists}
}
\]

The current 45/45 report is useful module evidence, but it is not an end-to-end acceptance test.

Do not start formal training from the current `smoke_v2.py`.

---

# 1. P0 — Blocking defects

All P0 items must be fixed before any training result is considered HCH-v2 v0.4 evidence.

---

## P0-1. Replace the formal legacy runner

### Current problem

`experiments/08-hch-v2/smoke_v2.py` imports:

- `HCHV2`;
- `candidate_loss_fn`;
- `state_loss_fn`;
- `compute_state_targets`.

It trains:

\[
L_{\rm candidate}
+
\lambda L_{\rm state}.
\]

It then calls legacy memory and `calibrate_s3()` searching `k/eta/tau`.

This contradicts the v0.3 mathematical contract.

### Required correction

The formal smoke/training runner must call only the new path:

1. S1 local rank;
2. IAH candidate;
3. IAH-CRPS;
4. atom memory;
5. W1 retrieval;
6. query-dose directional replay;
7. double-event proposal;
8. final \(\pi_q\) replay;
9. whole-action \(\widehat A\);
10. S3-C split-conformal DVG;
11. target-free S4.

### Acceptance

A grep/import audit of the formal runner must show no import/use of:

- `candidate_loss_fn`;
- `state_loss_fn`;
- `ContinuousStateHead`;
- `BiOMC`;
- legacy `DVG`;
- CARA/KL/temperature calibration.

---

## P0-2. Legacy fail-closed must be real, not declarative

### Current problem

`src/hch_v2.py` defines `require_not_legacy()`, but the current formal smoke does not invoke it before using legacy HCH.

### Required correction

Choose one:

A. move legacy implementation to an explicitly named legacy module and remove it from formal imports; or  
B. preserve the file but make every formal entry point reject it.

Recommended:

- keep old code for historical reproducibility;
- formal runner imports a new v0.4 orchestrator;
- any formal config resolving to legacy returns a hard error.

### Acceptance

A test intentionally selecting legacy HCH from the formal runner must fail before training starts.

---

## P0-3. Implement one end-to-end orchestrator

### Current problem

The new mathematical pieces are isolated utilities.

No authoritative class/function currently owns the full sequence.

### Required component

Introduce one formal orchestrator, e.g.

`HCHV2UniversalPipeline`

or equivalent.

It must provide explicit methods such as:

- `fit_s1_reference(...)`;
- `train_candidate_s2(...)`;
- `fit_s3_memory(...)`;
- `select_s3m_k(...)`;
- `calibrate_s3c(...)`;
- `freeze_bundle(...)`;
- `predict_s4(...)`.

Exact names may differ, but stage ownership must be explicit.

### Acceptance

One synthetic and one real-data smoke must trace a single day from host to final raw-price action with evidence artifacts.

---

## P0-4. Split S3-M and S3-C in the manifest

### Current problem

`ExperimentManifest` exposes only S1/S2/S3/S4.

v0.3 requires:

\[
S3\text{-M}
\cap
S3\text{-C}
=
\varnothing.
\]

### Required correction

Do not necessarily change the global top-level 50/20/10/20 split yet.

Inside the current S3 allocation, create a chronological nested split:

- memory prefix;
- k/proposal validation if required;
- calibration suffix.

The exact fraction belongs to the training protocol and may later change.

Manifest must store these dates and include them in the split hash.

### Acceptance

Automated assertion:

- S1/S2/S3-M/S3-C/S4 have no overlap;
- S3-C cannot mutate candidate, memory, k or proposal.

---

## P0-5. Final \(\pi_q\) replay is mandatory

### Current problem

`query_replay.py` correctly replays a supplied query dose, and `double_event.py` correctly proposes intervals independently, but no formal path proves:

\[
\text{directional replay}
\rightarrow
\text{proposal}
\rightarrow
\text{final }\pi_q
\rightarrow
\text{replay again}
\rightarrow
\widehat A_q.
\]

### Required correction

For every query:

1. replay full Down dose;
2. replay full Up dose;
3. estimate hourly directional gain;
4. select disjoint Down/Up events;
5. form final sparse \(\pi_q\);
6. replay this final \(\pi_q\) on the same frozen neighbors;
7. define DVG \(\widehat A_q\) from the final replay.

Do not estimate DVG value from the pre-proposal directional gains.

### Acceptance

Unit test where proposal removes hours must show final \(\widehat A\) changes accordingly.

---

## P0-6. Rebuild the freeze bundle for the new semantics

### Current problem

Legacy `HCHV2Bundle` stores legacy neural memory keys/gains and legacy calibration state.

### Required universal bundle

- core model state;
- optional adapter state if present;
- Data Signature version/spec;
- IAH coordinate/version;
- training provenance;
- source dataset/host metadata.

### Required local bundle

- S1 rank reference;
- atom memory;
- memory dates/timestamps;
- W1 version;
- frozen k;
- event proposal/tie-break version;
- DVG alpha/errors/q;
- local data/split/cutoff hashes;
- fallback codes.

### Acceptance

Reload must reproduce, for fixed queries:

- scale;
- \(u\);
- atoms;
- W1 distances;
- neighbor IDs;
- final \(\pi\);
- \(\widehat A\);
- q;
- LCB;
- final raw prediction.

Parameter hash alone is insufficient.

---

## P0-7. Fix scale-valid semantics in candidate and loss

### Current problem

`IAHCandidateHead._compute_scale()` says it uses valid hours but actually uses only finite host values; the passed `valid_mask` is ignored.

Both candidate and loss clamp:

\[
s\leftarrow\max(s,10^{-12})
\]

despite the mathematical rule:

- no mathematical epsilon;
- unidentifiable scale => Identity.

The CRPS loss also does not explicitly exclude `scale_valid == 0` days.

### Required correction

- compute \(s\) over **valid AND finite** hours;
- if no valid hour or \(s=0\), set `SCALE_UNIDENTIFIED`;
- invalid-scale days must not be treated as ordinary CRPS training examples;
- computational branches may use a dummy safe denominator internally only if it cannot change mathematical output and is explicitly documented as implementation masking, not a modified scale definition.

### Acceptance

Tests:

- invalid hours do not affect \(s\);
- all-zero valid host => Identity;
- all-invalid day => Identity;
- scale-invalid day contributes no misleading CRPS gradient;
- no tiny positive scale is silently replaced by \(10^{-12}\) as the mathematical scale.

---

# 2. P1 — Semantic correctness / generalization defects

P1 may not crash the system, but it can invalidate cross-market conclusions.

---

## P1-1. Replace current “mid-rank” implementation

### Current problem

`S1RankReference._interpolate_rank()` uses `searchsorted(..., side='right')` and `(idx + frac)/(n+1)`.

For tied pools this is not a true empirical mid-rank.

A pool of identical values can map the identical query far from 0.5.

### Required correction

Define and test one exact rank convention.

Recommended tie-aware empirical mid-rank based on left/right counts, with a predeclared boundary interpolation rule for off-grid values.

Do not silently use epsilon to create fake order within ties.

### New architecture note

Do **not** “fix” this by putting `market_id` and `target_id` into the neural core.

Instead, build a local S1 rank-reference object per deployment series/domain. IDs remain lookup/audit keys.

---

## P1-2. Fix self-exclusion in CAGM retrieval

### Current problem

`get_neighbors()` treats:

\[
D<10^{-14}
\]

as “self”.

Two different days may legitimately have identical atom measures and W1=0.

They are valid perfect neighbors and must not be removed.

### Required correction

Self-exclusion must be based on:

- memory index;
- date ID;
- timestamp key;

not distance.

### Acceptance

Two distinct identical days with W1=0 remain neighbors, while the actual query record is excluded.

---

## P1-3. Rebuild core lag context in scale-free coordinates

### Current problem

`hch_v2_data.py` mixes:

- normalized lagged prices;
- raw host lags;
- raw residual lags;
- raw hour number.

This reintroduces dataset/currency scale into a core intended to transfer across markets.

### Required correction

Create a formal `core_lag_context` with documented scale-free price-like channels.

Recommended candidates:

- lagged host \(z^0\);
- lagged target \(z^Y\) where legal;
- lagged hyperbolic residual \(z^Y-z^0\);
- lagged local rank/state;
- legal availability masks.

Time-of-day should use existing cyclic time features rather than raw hour as an unexplained price-context component.

---

## P1-4. Core encoder must not depend on dataset z-score price coordinates

### Current problem

Legacy `HourTokenEncoder` starts from `host_model`, which is dataset S1 mean/std normalized.

For the new universal core, the primary host geometry should be:

\[
z^0=\operatorname{asinh}(x^0/s).
\]

### Required correction

Formal `CoreContextEncoder` consumes host-relative scale-free core input.

`host_model` may remain in the batch for legacy/baseline compatibility but must not be the only core price channel.

---

## P1-5. Split core and optional feature paths

### Current problem

Legacy `HourTokenEncoder` consumes exogenous features directly inside the same encoder.

This makes rich-feature behavior part of the base representation.

### Required correction

Create structurally separate:

- `CoreContextEncoder`;
- `OptionalCovariateEncoder`;
- lightweight `DataAdaptiveConditioner`.

The formal invariant must be:

\[
\text{optional disabled}
\Rightarrow
\text{pure core path}.
\]

Core parameters are frozen during U2.

---

## P1-6. Replace ordinal `exog_type=j+1`

### Current problem

Current `exog_type` is a local column index.

Column 1 has no stable meaning across datasets.

### Required correction

For the first version, use generic information roles only:

- known future;
- observed past;
- static;
- calendar;
- other legal pre-outcome.

Do not require detailed business semantics yet.

Store local column name only as audit metadata.

---

## P1-7. Add Data Signature without target leakage

### Required component

Implement a small `DataSignature` object with:

### deterministic
- observability descriptors;
- scale-free distribution summaries;
- small dynamics summary.

### learned
- pooled core representation.

The signature must be computed only from pre-outcome information / frozen S1 statistics.

### Conditioning

Use lightweight FiLM/AdaLN/zero-init residual modulation.

Do not add domain-classification loss.

---

## P1-8. Treat `market_id` / `target_id` as metadata by default

Batch fields may remain.

Formal universal candidate path should not depend on them unless a later explicit ablation enables them.

Acceptance test:

Changing audit IDs while holding all actual input information fixed does not change candidate output.

---

## P1-9. 24-hour restriction must be explicit

### Current problem

`ExperimentManifest` drops 23/25-hour dates and the dataset/collate path hardcodes 24.

This is acceptable for the first experiment cycle **only if declared**.

### Required correction

For now choose:

`DAY_LENGTH_PROTOCOL = COMPLETE_24H_ONLY`

and store:

- excluded dates;
- reason;
- counts;
- hash effect.

Do not claim generic DST support in the current code.

A later version may support 23/24/25h.

---

## P1-10. Enforce one-day semantics in atom memory

`CAGMAtomMemory.add_day()` should either:

- assert batch size = 1; or
- explicitly iterate individual days.

Do not rely on `.squeeze()` to decide whether a batch is one day.

Memory records must keep date/hour/timestamp identity.

---

# 3. P2 — Engineering enhancements after the core is valid

These improve training but do not block the first v0.4 smoke.

---

## P2-1. U0 teacher feature bank

Add an offline artifact format for representation teachers.

Do not couple teacher inference to HCH training loop.

Fields:

- teacher model/version;
- teacher layer;
- window ID;
- representation;
- preprocessing hash;
- possible corpus-overlap warning.

---

## P2-2. Domain-balanced sampler

Formal training loader should sample over:

\[
g=(dataset,host)
\]

rather than raw concatenated hours.

Do not implement outcome-tail oversampling in the default sampler.

---

## P2-3. U2 masking scheduler

Provide reproducible masks:

- all features;
- group drop;
- random feature drop;
- full optional drop.

Seed and mask policy must be logged.

---

## P2-4. Parameter groups for progressive unfreezing

If U0 is used, support:

- frozen encoder / train head;
- low-LR partial encoder;
- final frozen core.

This is a training utility, not an architecture dependency.

---

# 4. File-level action map

## `src/hch_v2.py`

Status: **legacy only**.

Required:
- remove from formal import path;
- retain explicit legacy warning/guard;
- do not extend this legacy class with new features.

Preferred: new formal orchestrator rather than mixing old/new semantics in one class.

---

## `src/iah_candidate.py`

Keep:
- three-atom semantics;
- softmax center logit;
- ReLU shifts;
- raw inverse transform.

Fix:
- valid-mask-aware scale;
- no mathematical epsilon semantics;
- scale-invalid training behavior;
- integration with universal core representation.

---

## `src/iah_crps_loss.py`

Keep exact v0.3 CRPS.

Fix:
- scale-invalid exclusion;
- finite target/valid mask contract;
- explicit per-day valid count;
- fail loudly when a batch contains no valid training day.

Do not add auxiliary loss.

---

## `src/s1_rank.py`

Rewrite rank interpolation.

Do not make market/target neural conditioning mandatory.

Make rank-reference ownership local and auditable.

---

## `src/w1_retrieval.py`

Keep exact W1.

Fix:
- ID-based self exclusion;
- timestamp/hour identity;
- one-day record semantics;
- shared-valid-hour alignment contract.

---

## `src/query_replay.py`

Keep mathematical function.

Add:
- final proposal replay integration test;
- explicit valid-hour intersection;
- evidence record schema.

---

## `src/double_event.py`

Current result is usable for H=24, but:

- implement/document deterministic tie-break exactly;
- if code claims O(H²), replace current nested U_L/U_R precomputation with actual linear prefix/suffix max-subarray preprocessing or change complexity claim;
- preserve brute-force oracle in tests.

This is not a P0 blocker for H=24 correctness, but the paper/code complexity claim must match implementation.

---

## `src/dvg_calibrate.py`

Keep split-conformal logic.

Minor:
- rename `DGVSplitConformal` to `DVGSplitConformal` or provide compatibility alias;
- freeze only after quantile computed;
- include version/hash;
- integrate with final-policy replay, not directional pre-proposal gains.

---

## `src/hch_v2_data.py`

Major rewrite areas:
- explicit core vs optional fields;
- scale-free core lag context;
- generic optional information roles;
- Data Signature inputs/statistics;
- 24h-only protocol logging;
- audit IDs separated from predictive core.

Preserve:
- raw/model dual channel if needed for baselines;
- target-free S4;
- timestamp/date evidence.

---

## `src/eval_manifest.py`

Keep date-first authority.

Add:
- explicit S3-M/S3-C nested dates;
- excluded non-24h dates and reasons;
- split hash includes nested split;
- optional host/domain identifiers for audit.

---

## `experiments/08-hch-v2/host_cache.py`

Good idea; keep offline cache strategy.

Fix/extend later:
- record host model commit/checkpoint hash;
- distinguish OOF/frozen semantics;
- support external pretrained hosts as additional cached providers;
- never train HCH through host gradients.

---

## `experiments/08-hch-v2/smoke_v2.py`

Replace completely as formal v0.4 smoke.

Do not patch legacy loss incrementally.

---

# 5. New minimal formal components

Avoid architecture bloat.

Recommended maximum new core files:

1. `hch_v2_pipeline.py` — authoritative end-to-end orchestrator;
2. `hch_v2_context.py` — CoreContextEncoder + DataSignature + OptionalCovariateEncoder/conditioner;
3. `hch_v2_bundle.py` — universal/local frozen package if legacy bundle cannot be cleanly reused.

Existing mathematical files remain separate.

Do not create a large package hierarchy until training evidence demands it.

---

# 6. Mandatory test suite before training

## Mathematical

1. valid-mask scale;
2. zero-scale fallback;
3. scale equivariance;
4. CRPS manual equality;
5. mass sum;
6. exact zero shifts;
7. raw candidate ordering;
8. W1 manual examples;
9. query-dose gain bound;
10. final-\(\pi\) replay.

## Rank / domain

11. tied rank pool maps equal value to exact mid-rank convention;
12. audit ID change does not alter candidate;
13. price-only core path works;
14. optional full-mask reproduces core;
15. two identical non-self days with W1=0 are valid neighbors.

## Stage separation

16. S1/S2/S3-M/S3-C/S4 non-overlap;
17. S3-C cannot change candidate hash;
18. S3-C cannot change memory hash;
19. S3-C cannot change k/proposal;
20. S4 batch target is absent.

## End-to-end

21. directional replay -> proposal -> final replay trace;
22. q=inf => Identity;
23. bundle round trip;
24. timestamp-order perturbation does not change keyed evaluation;
25. legacy runner selection fails closed.

## Universal/optional

26. core-only on at least one price-only dataset;
27. rich dataset with optional branch off equals core checkpoint;
28. Data Signature uses no current target;
29. core parameter hash unchanged during U2;
30. optional branch zero/near-zero initialization preserves core output.

---

# 7. First smoke after P0 closure

Only after all P0 tests:

### Data
One public price-only market first.

### Host
Linear first.

### Procedure
- build S1 rank;
- train IAH core on S2;
- S3-M memory/k;
- S3-C DVG;
- S4 target-free prediction;
- evaluate externally.

### Required evidence JSON per day

- dataset;
- host;
- date/timestamps;
- scale;
- rank;
- masses;
- shifts;
- W1 neighbor dates/distances;
- directional gain estimate;
- proposed intervals;
- final \(\pi\);
- final replay \(\widehat A\);
- q;
- LCB;
- action/fallback;
- final raw prediction.

Do not start U0 or U2 until this smoke is correct.

---

# 8. Second smoke — universal core

After single-domain smoke:

- 2–3 datasets;
- 2 hosts;
- core-only inputs;
- domain-balanced sampler;
- IAH-CRPS only.

Check:

- per-domain convergence;
- no domain dominates updates;
- leave-one-domain candidate behavior;
- no ID shortcut.

---

# 9. Third smoke — optional context

Only after stable universal core:

- freeze core hash;
- one rich domestic/public dataset;
- train optional branch with masking;
- test full optional, partial mask, full mask.

Core hash must remain unchanged.

---

# 10. Acceptance state names

Use explicit states:

- `LEGACY_ONLY`
- `V03_MODULES_UNIT_VALIDATED`
- `V04_END_TO_END_SMOKE_VALIDATED`
- `UNIVERSAL_CORE_TRAINABLE`
- `OPTIONAL_CONTEXT_TRAINABLE`
- `FORMAL_EXPERIMENT_READY`

Current repository baseline should be considered:

\[
\boxed{\texttt{V03\_MODULES\_UNIT\_VALIDATED}}
\]

not `FORMAL_EXPERIMENT_READY`.
