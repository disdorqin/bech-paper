# HCH-v2 First-Round Universal Training Protocol v0.1

**Date:** 2026-08-13  
**Depends on:** `HCH-v2 Universal Training Master Plan v0.1`  
**Audited code baseline:** `769e421530d3fa29f964a8a5b8a24a0e5eb38697`  
**Purpose:** First scientifically meaningful repo-only universal training and falsification run.  
**Explicit exclusions:** no Shandong in source training; no TCN; no Hugging Face distillation; no U2 optional covariate branch.

---

# 0. Should we run an effect experiment now?

## Verdict

\[
\boxed{\textbf{Yes, but only after the P0 training gates are fixed and T0 smoke passes.}}
\]

Why it is worth testing:

1. the mathematical decision chain is now implemented in the new pipeline;
2. legacy HCH is no longer the formal path;
3. IAH-CRPS, W1 retrieval, query-dose replay, double-event and DVG can now be connected end-to-end;
4. the remaining uncertainty is no longer “can the formula be coded?” but:
   - can a shared candidate learn across markets?
   - does Data Signature help rather than harm?
   - does the frozen candidate transfer?
   - does the DVG gate preserve normal periods?

These are empirical questions and should now be falsified by a controlled experiment.

Why **not** run a large benchmark immediately:

- the current v0.4 smoke result artifact is empty;
- current S1 reference is host-in-sample;
- S2 checkpoint selection is training-loss based;
- the current signature API is not safe for multi-domain context.

Therefore the first experiment is a **go/no-go scientific pilot**, not a final paper benchmark.

---

# 1. Hard prerequisites

Do not start R1A until all are true.

## Gate G0-1 — per-domain signature

`domain_det` is passed as sample/batch context.

No single mutable domain buffer determines universal outputs.

## Gate G0-2 — host-fit/reference separation

Replace current S1 usage with:

\[
H0/S1R/S2T/S2V/S3M/S3C/S4.
\]

## Gate G0-3 — validation-based checkpointing

Best candidate checkpoint selected by:

\[
\operatorname{MacroCRPS}_{S2V}.
\]

Not training loss.

## Gate G0-4 — identity-initialized modulation

Use:

\[
h'=(1+\Delta\gamma)h+\beta
\]

with zero-init modulation output.

## Gate G0-5 — deterministic descriptor v1 cleaned

Remove/rewrite order-sensitive `flips` and `lag1`.

## Gate G0-6 — non-empty T0

Regenerate:

`LAGO_DE x Linear`

and save non-empty evidence with roundtrip verification.

If any gate fails, generated universal weights are labeled:

`INVALID_DEV_ONLY`

and are not reused.

---

# 2. First-round scientific questions

This round answers four questions only.

### Q1 — Can one IAH candidate be trained across heterogeneous price markets?

\[
\theta_{\rm universal}
\]

must improve macro candidate quality without one market dominating.

### Q2 — Does the Data Signature actually help?

Compare:

\[
\text{Universal-NoSig}
\quad\text{vs}\quad
\text{Universal-Sig}.
\]

### Q3 — Is universal training better than merely training a local corrector?

Compare:

\[
\text{Local-Core}
\quad\text{vs}\quad
\text{Universal-Core}.
\]

### Q4 — Does a frozen universal candidate transfer to an unseen market?

Use a market family absent from universal candidate training.

Nothing about U0/MOMENT/U2 is tested in this round.

---

# 3. Dataset selection

## 3.1 Source markets

Use exactly three source series for the first pilot:

### A. `LAGO_DE`

Role:

- European electricity price market;
- negative prices exist but are not dominant;
- good anchor for conventional EPF behavior.

### B. `LAGO_PJM`

Role:

- different region/currency/market structure;
- useful to prevent a Europe-only universal prior.

### C. `NEM_SA1`

Role:

- extreme high-variance market;
- repository metadata reports high negative-price frequency and very large price range;
- deliberately stresses signed-tail correction geometry.

These three are chosen for **heterogeneity**, not because they are the “best” datasets.

## 3.2 Unseen market

Use:

### `NORD_DK1`

as the main frozen cross-market holdout.

Its candidate core receives **no gradient updates** during universal source training.

The holdout host itself may be trained locally in H0, because the scientific claim is about a model-agnostic post-hoc corrector over a frozen host, not a universally zero-shot base forecaster.

Call this:

> **zero-gradient / frozen-corrector transfer**

rather than loosely saying the whole forecasting system is zero-shot.

## 3.3 Explicitly excluded

First round does not train universal candidate on:

- Shandong DA;
- Shandong RT;
- other Chinese provinces;
- full EPEX/Nordic/UniElecPrice pool;
- generic time-series benchmarks;
- HF data.

This protects the first round from becoming a data-volume experiment before the mechanism is understood.

---

# 4. Host selection

## R1A — cheap infrastructure/scientific pilot

Use:

- Linear
- MLP

Domains:

\[
3\text{ source markets}\times2\text{ hosts}=6
\]

This is enough to test the multi-domain trainer and two different host-error regimes cheaply.

Run one fixed host seed first.

## R1B — first meaningful model-independence experiment

If R1A passes, add:

- LSTM
- PatchTST

Total source domains:

\[
3\times4=12.
\]

Do **not** add TCN in first version.

### Why not TCN

The first round needs architectural diversity, not every possible network. Linear/MLP/LSTM/PatchTST already cover:

- linear tabular;
- nonlinear tabular;
- recurrent sequence;
- Transformer/patch sequence.

TCN adds cost without creating a uniquely necessary first-round scientific category.

---

# 5. Chronological split

For each `(dataset, host)` domain, use the same date-first proportions:

| Segment | Fraction | Reads target? | Updates |
|---|---:|---|---|
| H0 | 40% | yes | host only |
| S1R | 10% | no for rank/signature construction | none |
| S2T | 16% | yes | universal/local candidate |
| S2V | 4% | yes | no gradient; checkpoint selection |
| S3M | 5% | yes | local memory |
| S3C | 5% | yes | local DVG calibration |
| S4 | 20% | evaluation only | none |

All boundaries are chronological and date-first.

Non-24h dates may follow the current `COMPLETE_24H_ONLY` policy in the first round, but:

- excluded date count;
- original hour count;
- final effective sample count

must be stored in the manifest.

No silent deletion.

---

# 6. Host training and cache generation

For every selected domain:

1. fit host on H0 only;
2. freeze host;
3. produce host predictions on:
   - S1R
   - S2T
   - S2V
   - S3M
   - S3C
   - S4
4. cache full prediction arrays;
5. hash host predictions and split manifest.

Do not refit host after seeing S1R.

Required cache metadata:

```text
dataset
host
host_seed
H0_start/end
S1R_start/end
...
feature_schema_hash
split_hash
prediction_hash
git_commit
```

---

# 7. S1R construction

For each `(market, host)`:

1. compute daily scale:
   \[
   s_d=\frac1{H_d}\sum_h|x^0_{d,h}|
   \]
2. map host prediction to:
   \[
   z^0=\operatorname{asinh}(x^0/s_d)
   \]
3. build hour-aware S1 rank reference;
4. build deterministic domain signature from **S1R only**;
5. freeze these local reference objects for the rest of the run.

No S2/S3/S4 target or prediction distribution is allowed to update S1R.

---

# 8. Deterministic signature v1 for first round

Keep this intentionally small.

Recommended descriptor vector from S1R `z0`:

\[
e^{det}=
[
q_{05},q_{25},q_{50},q_{75},q_{95},
IQR,
E|z^0|,
P(z^0<0)
].
\]

All are scale-free/order-free.

Do not use:

- dataset name;
- currency;
- market one-hot;
- target values from S2+;
- per-day realized outcome;
- unsorted lag1;
- artificial sign-flips across concatenated day boundaries.

Market/target/host IDs remain metadata for audit and stratified reporting.

---

# 9. Universal candidate model

First-round default:

- `d_model = 64`
- `d_sig = 32`
- optional branch disabled;
- core input:
  - current host \(z^0\);
  - S1 continuous rank \(u\);
  - legal cyclic time features;
  - scale-free lag/history features.

Data Signature modulation:

\[
h'=(1+\Delta\gamma(e))\odot h+\beta(e),
\]

identity-initialized.

No new temporal block is added before the first experiment.

Reason:

> if this lightweight core underfits, the experiment will tell us. Adding attention/convolution now would confound the basic question of whether the IAH + domain-conditioning formulation itself works.

---

# 10. Universal training sampler

Do not concatenate all domain-days and shuffle globally.

Use:

```text
sample domain g ~ Uniform(G)
sample a minibatch of S2T days from g
attach g's frozen S1R rank/signature
forward shared candidate
compute IAH-CRPS
update shared weights
```

This approximates:

\[
L_{\rm universal}
=
\frac1{|G|}
\sum_g E_{d\sim g}[L_g(d)].
\]

No market receives larger weight merely because it has more rows.

### Batch strategy

Prefer domain-homogeneous minibatches in R1A for debugging.

After per-sample signature tests pass, mixed-domain batches may be enabled.

Initial minibatch:

- 16–32 days per optimizer step, depending on memory;
- each day is its valid 24-hour trajectory.

---

# 11. Optimizer defaults

These are starting values, not paper-tuned hyperparameters.

### R1A

- optimizer: AdamW
- learning rate: `3e-4`
- weight decay: `1e-4`
- gradient clipping: `1.0`
- one HCH seed initially
- validation every fixed number of optimizer steps
- early stopping on macro S2V CRPS
- restore best validation checkpoint.

### R1B

Keep the same base hyperparameters unless R1A shows numerical instability.

Run HCH seeds:

\[
\{0,1,2\}
\]

while host cache may initially remain fixed at host seed 0 to isolate corrector randomness.

Later formal paper runs can expand host seeds.

Do not hyperparameter-search against S4.

---

# 12. S2 validation protocol

For every validation event compute per domain:

\[
L_g^{val}
=
\operatorname{mean}_{d\in S2V_g}
L_{\rm IAH-CRPS}(d).
\]

Checkpoint score:

\[
L_{\rm macro}
=
\frac1{|G|}\sum_g L_g^{val}.
\]

Also report:

\[
L_{\rm worst}=\max_gL_g^{val}.
\]

Compare against deterministic frozen host in the same hyperbolic geometry:

\[
L^{host}_g
=
E|z^Y-z^0|.
\]

Define:

\[
\Delta_g
=
L_g^{IAH}-L_g^{host}.
\]

Negative is better.

This is the first signal of whether the candidate distribution learns anything useful before routing/calibration.

---

# 13. Training-health diagnostics

Record these at every validation checkpoint.

## Mass health

For each domain/hour regime:

- mean \(w^-\);
- mean \(w^0\);
- mean \(w^+\);
- entropy of atom masses.

Look for:

- permanent `w0≈1` collapse;
- permanent one-tail collapse;
- extremely unstable mass switching.

These are diagnostic flags, not automatic failure by themselves.

## Shift health

Record:

- fraction `m_minus > tiny`;
- fraction `m_plus > tiny`;
- median/p95 `m_minus`, `m_plus`.

Look for:

- all-zero shifts;
- explosive shifts;
- market-specific scale leakage despite hyperbolic normalization.

## Signature health

Record:

- \(\|\Delta\gamma\|\);
- \(\|\beta\|\);
- learned signature norm;
- per-domain deterministic signature.

If modulation becomes enormous while CRPS does not improve, stop and debug.

## Gradient health

Record:

- total grad norm before clipping;
- fraction of NaN/Inf batches;
- number of scale-unidentified days.

---

# 14. R1A go/no-go rule

R1A is not judged by one leaderboard number.

## GREEN

Proceed to R1B if:

1. training and validation losses are finite;
2. best checkpoint is selected by S2V, not S2T;
3. macro S2V IAH-CRPS is better than the frozen-host deterministic CRPS baseline;
4. improvement is not generated entirely by one source domain;
5. atom masses/shifts are not trivially collapsed across every domain;
6. interleaving domain order produces identical evaluation for the same checkpoint/reference objects.

## YELLOW

Pause for targeted diagnosis if:

- macro improves but NEM or another domain is severely sacrificed;
- Signature variant is much less stable than NoSig;
- candidate improves source markets but not the unseen DK1 diagnostic.

## RED

Do not start U0 or larger training if:

- macro S2V fails to beat deterministic-host CRPS after reasonable optimization;
- validation worsens while train CRPS improves;
- domain descriptor leakage/order effects exist;
- outputs depend on which market's descriptor was set last;
- S4 chain cannot be reproduced after freeze/reload.

---

# 15. R1B experiment matrix

After R1A GREEN:

## Models

### B0 — Host Identity

No HCH correction.

### B1 — Local-Core

Train an independent candidate on each source market/host S2T only.

Same architecture and IAH-CRPS.

Purpose:

> measure whether universal sharing helps or merely dilutes local fit.

### B2 — Universal-NoSig

Shared candidate over all source domains with deterministic signature zero/disabled.

Purpose:

> isolate the value of domain-aware conditioning.

### B3 — Universal-Sig

Full first-round universal candidate with Data Signature.

This is the proposed main model.

---

# 16. Local evidence/calibration after candidate training

For B1/B2/B3, candidate weights are frozen before S3.

Per evaluation domain:

## S3M

Build target-local CAGM memory from frozen candidate outputs and outcomes.

Select `k` only using the designated forward-validation portion within S3M if the implementation retains a nested k-validation subset.

Candidate k list can begin with:

\[
\{5,10,20\}
\]

provided memory size supports them.

Any candidate k larger than available memory is invalid and must be removed, not silently clipped.

## S3C

Calibrate whole-day:

\[
E_t=\hat A_t-A_t.
\]

For:

\[
\alpha=0.10,
\]

compute the split-conformal one-sided quantile.

No S3C gradient update.

## S4

Execute iff:

\[
LCB=\hat A_q-q>0.
\]

Otherwise Identity.

---

# 17. Unseen-market evaluation on NORD_DK1

The universal candidate B2/B3 is trained **without NORD_DK1 S2 gradients**.

For the holdout market:

1. train each frozen host on DK1 H0;
2. create DK1 S1R local rank/signature;
3. do **not** update universal candidate weights;
4. candidate-only evaluation can be performed on S2V/S4;
5. build local S3M/S3C for the safe routed version;
6. evaluate final S4.

This decomposes:

### Candidate transfer

Does \(\theta_{\rm universal}\) generate a useful distribution on unseen DK1?

### Local evidence adaptation

Can target-local memory/calibration safely decide when to use that candidate?

This is exactly the distinction the HCH paper should preserve.

---

# 18. First-round effect metrics

## 18.1 Candidate metrics

Primary:

- IAH-CRPS;
- deterministic-host transformed loss;
- \(\Delta CRPS\).

Stratify by:

- market;
- host;
- negative-price hours;
- high-tail hours;
- normal hours.

## 18.2 Final point metrics

For Host vs HCH final:

- MAE;
- rMAE;
- RMSE;
- standard no-floor sMAPE.

## 18.3 Tail metrics

- `mae_on_neg`;
- negative sign miss;
- negative bias;
- p95/p99 target MAE or high-tail MAE;
- high-tail underestimation bias.

Thresholds used for reporting must be derived from a legal pre-test/reference split, not S4 target quantiles unless explicitly labeled as retrospective diagnostics.

## 18.4 Gate metrics

- execute/release rate;
- Identity rate;
- harmful execution rate;
- mean realized action gain conditional on execute;
- empirical LCB coverage;
- distribution of \(A_{\rm hat}\), \(q\), LCB.

## 18.5 Statistical support

Use paired errors on the exact same S4 timestamps.

Existing repository DM-test utility may be used as a secondary diagnostic.

Do not turn the first pilot into a “significance hunting” exercise; effect size and failure pattern are more important at R1.

---

# 19. Essential first-round comparisons

The smallest scientifically useful table is:

| Model | Source/Local | Signature | Gate | S4 MAE/rMAE | Candidate CRPS | Harm rate |
|---|---|---|---|---|---|---|
| Host | — | — | — | ✓ | deterministic baseline | — |
| Local-Core | local | ✓/same architecture | local | ✓ | ✓ | ✓ |
| Universal-NoSig | shared | no | local | ✓ | ✓ | ✓ |
| Universal-Sig | shared | yes | local | ✓ | ✓ | ✓ |

And the same table on unseen `NORD_DK1` for the universal variants.

This is enough to decide whether the current training direction deserves scale-up.

---

# 20. What counts as success?

## Scientific GREEN

The direction is worth expanding if the evidence pattern is approximately:

1. `Universal-Sig` improves macro candidate CRPS versus Host;
2. it is competitive with or better than Local-Core on average;
3. unseen DK1 candidate does not collapse;
4. DVG reduces harmful execution and preserves a meaningful nonzero release rate;
5. final MAE/rMAE improvement is not bought by major normal-period degradation.

It is **not required** that every market-host pair improves.

## Transfer YELLOW

If:

- source domains improve;
- unseen DK1 degrades;
- Local-Core clearly beats Universal-Core,

then do not add more data blindly.

Investigate:

- signature sufficiency;
- domain imbalance;
- host-error representation;
- history representation.

This is the strongest case for later U0/MOMENT distillation.

## Formulation RED

If even source macro validation cannot improve deterministic-host CRPS, then:

- do not start server U0;
- do not add all datasets;
- debug candidate representation/optimization/math implementation first.

A foundation model should not be used to hide a broken correction objective.

---

# 21. Recommended run order

```text
R0
Fix P0 gates
  ↓
T0
LAGO_DE × Linear smoke, non-empty evidence
  ↓
R1A
[LAGO_DE, LAGO_PJM, NEM_SA1] × [Linear, MLP]
Universal-Sig / NoSig
1 HCH seed
  ↓
R1A verdict
  ├── RED → debug, stop
  ├── YELLOW → targeted diagnosis
  └── GREEN
        ↓
R1B
same markets × [Linear, MLP, LSTM, PatchTST]
Local-Core / Universal-NoSig / Universal-Sig
HCH seeds 0,1,2
        ↓
Frozen holdout
NORD_DK1 × same available hosts
        ↓
Decision
whether to launch server-scale U0 distillation
```

---

# 22. Artifacts required

Every R1 run directory:

```text
R1_<timestamp>/
├── run_config.yaml
├── git_commit.txt
├── domain_manifest.csv
├── split_manifest.json
├── host_cache_manifest.csv
├── signature_by_domain.csv
├── train_curve.csv
├── val_crps_by_domain.csv
├── mass_shift_stats.csv
├── checkpoint_best.pt
├── checkpoint_hash.txt
├── candidate_s4.parquet
├── final_s4.parquet
├── gate_evidence.jsonl
├── metrics_by_domain_host.csv
├── dm_tests.csv
└── VERDICT.md
```

`VERDICT.md` must explicitly state:

- GREEN / YELLOW / RED;
- why;
- which hypotheses failed;
- whether server U0 is authorized.

---

# 23. First-round non-goals

Do not use this round to claim:

- full SOTA;
- full official PIR superiority;
- guaranteed nonstationary conformal safety;
- final business profitability;
- universal performance over all electricity markets;
- Shandong deployment success;
- foundation-model contribution.

This round exists to answer:

\[
\boxed{
\text{Does the clean universal HCH correction idea work well enough to deserve scale?}
}
\]

---

# 24. Immediate implementation checklist

Before training:

- [ ] add H0/S1R split to manifest;
- [ ] add S2T/S2V split and split hash;
- [ ] make domain descriptor a forward context;
- [ ] zero-init identity FiLM;
- [ ] remove/repair order-sensitive descriptors;
- [ ] implement UniversalCoreTrainer;
- [ ] implement macro-domain validation;
- [ ] ensure optional U2 params are disabled in first round;
- [ ] record all source datasets/hosts in universal checkpoint manifest;
- [ ] regenerate non-empty v0.4 smoke;
- [ ] freeze first-round config before looking at S4.

After these boxes are closed, R1A is worth running immediately.
