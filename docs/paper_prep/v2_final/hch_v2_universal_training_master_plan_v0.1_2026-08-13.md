# HCH-v2 Universal Training Master Plan v0.1

**Date:** 2026-08-13  
**Code baseline audited:** `disdorqin/bech-paper @ 769e421530d3fa29f964a8a5b8a24a0e5eb38697`  
**Status:** Research/training design; this document does not authorize final paper-scale experiments until the P0 gates below are closed.  
**Core principles:** 简洁有效、创新独特、底层推导；single IAH-CRPS objective; frozen host; transferable candidate knowledge + target-local evidence/calibration.

---

## 0. Executive verdict

The synchronized code is now close enough to the v0.3 mathematical design that **we should continue toward real training rather than redesign the mathematics**. The authoritative pipeline already contains the intended chain:

`S1 reference -> IAH candidate -> S3-M memory/k -> query-dose replay -> double-event -> final pi replay -> S3-C whole-day DVG -> S4 target-free inference`.

However, the repository is **not yet safe for producing the first reusable universal checkpoint**. Several remaining issues are not cosmetic; if we train before fixing them, the resulting weights may encode wrong domain context, use a biased reference distribution, or be selected by training loss rather than generalization.

Therefore:

> **Do not produce or preserve “official” universal weights until P0-1 ~ P0-5 are closed and a non-empty real-data smoke is regenerated.**

Current checked smoke result file  
`experiments/08-hch-v2/results/v0.3/smoke_v4_lago_de_linear.json`  
is empty at the audited commit, so code structure is present but an auditable successful v0.4 real-data run is not yet established by that artifact.

---

# 1. Training-blocking code audit

## 1.1 P0-1 — DataSignature must not be a single mutable domain buffer

### Current behavior

`src/hch_v2_context.py` stores deterministic domain descriptors in:

```python
self.register_buffer("domain_det", torch.zeros(d_det))
```

and later overwrites them with:

```python
set_domain_descriptors(det)
```

This is correct for a one-domain smoke, but not for universal training.

If training domains are:

\[
g \in \{(\text{market},\text{target},\text{host})\},
\]

then each sample/batch must use its own:

\[
e_g^{det}.
\]

A single mutable model buffer means the last descriptor written can silently become the context used by another domain.

### Required correction

The deterministic signature must be **data/context**, not persistent universal model state.

Preferred API:

```text
candidate_head.forward(
    host_raw,
    core_context,
    domain_det=[B, d_det],
    ...
)
```

The universal core checkpoint stores only shared parameters.  
The local/domain descriptor is stored in a separate profile/manifest.

### Acceptance test

Create two synthetic domains A/B with very different descriptors and identical network weights. Interleave them in the same epoch and mixed batch. Verify:

1. A always receives A descriptor;
2. B always receives B descriptor;
3. batch order does not change output;
4. saving/reloading the universal checkpoint does not bake one source market's descriptor into the model.

---

## 1.2 P0-2 — S1 reference is currently built from host in-sample predictions

### Current behavior

In `experiments/08-hch-v2/smoke_v4.py`:

1. the host backbone is fitted on S1;
2. the same fitted host predicts the whole dataset;
3. its S1 predictions are used to build:
   - S1 rank reference;
   - deterministic Data Signature.

That means S1 reference predictions are in-sample for the host, while S2/S3/S4 predictions are genuinely out-of-sample.

This changes the distribution of host errors/predictions seen by the reference system and may bias:

\[
u_{d,h}=\mathcal R_{\mathcal P_h}(z^0_{d,h})
\]

and the domain signature.

### Required correction

Use a chronological **host-fit/reference split**:

\[
\boxed{
H0 \rightarrow S1R \rightarrow S2T \rightarrow S2V \rightarrow S3M \rightarrow S3C \rightarrow S4
}
\]

Recommended first protocol:

| Segment | Fraction | Role |
|---|---:|---|
| H0 | 40% | fit frozen host |
| S1R | 10% | out-of-sample host predictions for rank/signature |
| S2T | 16% | candidate training |
| S2V | 4% | candidate model selection |
| S3M | 5% | memory |
| S3C | 5% | DVG calibration |
| S4 | 20% | untouched test |

This preserves the old overall 50/20/10/20 idea while making host reference out-of-sample.

Do **not** retrain the host on S1R after the reference is created in this protocol; the frozen host applied to S1R/S2/S3/S4 must be the same model.

---

## 1.3 P0-3 — Current S2 early stopping uses training loss

### Current behavior

`HCHV2UniversalPipeline.train_candidate_s2()` selects its best state from the average loss on the batches used for optimization.

This can select an overfit checkpoint and is especially unsafe under multi-domain training.

### Required correction

Create explicit:

\[
S2 = S2T \cup S2V,\quad S2T \cap S2V = \varnothing.
\]

Use:

\[
L_{\text{select}}
=
\frac{1}{|\mathcal G|}
\sum_{g \in \mathcal G}
L^{val}_g.
\]

Checkpoint selection must be based on **macro-domain validation IAH-CRPS**, not pooled/micro loss.

Also record:

\[
L_{\text{worst}}=\max_g L_g^{val}
\]

to detect a universal model that improves the mean by sacrificing one market.

Training loss remains diagnostic only.

---

## 1.4 P0-4 — A real multi-domain UniversalCoreTrainer is still missing

The current pipeline is still structurally centered on one pipeline/domain instance.

Universal training needs a trainer whose unit is:

```text
Domain = (market_series, target_id, host_backbone)
```

and which owns only the shared candidate parameters.

### Required behavior

Per optimizer step:

1. sample a domain uniformly or by an explicitly frozen macro sampler;
2. sample chronological training days from that domain's S2T;
3. attach that domain's `domain_det`;
4. use the corresponding frozen host cache and local S1 rank reference;
5. compute **IAH-CRPS only**;
6. update shared \(\theta_{\rm core}\).

The sampler must not simply concatenate all rows, because large markets/datasets would dominate.

Default first-round domain risk:

\[
\boxed{
L_{\rm universal}
=
\frac1{|\mathcal G|}
\sum_{g\in\mathcal G}
\mathbb E_{d\sim g}
[
L_{\rm IAH-CRPS}(d)
]
}
\]

The code may implement this by uniform domain sampling; no extra loss term is needed.

---

## 1.5 P0-5 — Official universal smoke must exist before keeping weights

At the audited commit, the v0.4 smoke code exists, but the checked result JSON is empty.

Before any first-round checkpoint is treated as meaningful:

1. regenerate `LAGO_DE x Linear` smoke;
2. confirm non-empty evidence;
3. confirm bundle round-trip;
4. confirm S3-M/S3-C disjointness;
5. confirm final `pi_q` is replayed for `A_hat`;
6. confirm S4 accepts no target;
7. log exclusion of non-24h days.

Any checkpoint created before this gate is **development-only** and should not enter paper tables.

---

# 2. Strong pre-training corrections (P1)

These are not all mathematical blockers, but they should be fixed before the first serious multi-market run.

## P1-1 — FiLM must start as identity

Current core uses:

\[
h'=\gamma(e)\odot h+\beta(e),
\]

while the modulation head is randomly initialized. A random domain interface can distort the core before it has learned anything.

Change to:

\[
\boxed{
h'=(1+\Delta\gamma(e))\odot h+\beta(e)
}
\]

and zero-initialize the final modulation layer so initially:

\[
h'=h.
\]

This makes DataSignature an incremental conditioning mechanism rather than a random gate over the core.

---

## P1-2 — Remove or repair order-sensitive deterministic descriptors

`compute_domain_descriptors()` currently accepts `s1_hours` but does not use it.

It flattens `s1_z0` and computes `flips` and `lag1`. If the concatenation crosses day boundaries or dates are not explicitly sorted, these quantities no longer have a clean time-series meaning.

First version recommendation:

**Keep only robust order-free descriptors**, e.g.

- q05/q25/q50/q75/q95;
- IQR;
- mean absolute hyperbolic coordinate / robust dispersion;
- positive/negative mass.

Remove `flips` and `lag1` from deterministic signature v1.

If later reintroduced, compute them on timestamp-sorted contiguous sequences and never across artificial day boundaries.

---

## P1-3 — Universal bundle and local profile must be separated

Current bundle structure can serialize one `data_signature_spec["det"]` and one source dataset ID.

For universal training:

### Universal checkpoint

Contains:

- shared candidate/core weights;
- architecture version;
- source market/host manifest;
- optimizer/config hash;
- training code commit;
- S2T/S2V definition;
- seed.

### Local/domain profile

Contains:

- S1 rank reference;
- deterministic signature;
- CAGM memory;
- selected k;
- DVG alpha/q/errors;
- target/market/host metadata;
- split hash.

This realizes the paper concept:

\[
\boxed{
\text{global transferable correction knowledge}
+
\text{local evidence/calibration}
}
\]

---

## P1-4 — U2 needs an explicit trainer and freeze contract

The optional encoder exists, but current `train_candidate_s2()` only forwards:

`host, ctx, target, valid_mask`

and therefore does not train the optional input path as a formal stage.

U2 must be a separate trainer:

1. load a frozen U1 checkpoint;
2. hash the core;
3. freeze universal core parameters;
4. enable optional encoder / explicitly allowed adapter;
5. use the **same IAH-CRPS objective**;
6. apply modality/covariate dropout;
7. after training, assert core hash unchanged.

No BCE, auxiliary reconstruction, domain-classification or tail loss is added to S2/U2.

---

## P1-5 — Optional covariate semantics should eventually be richer than five generic roles

Current roles are roughly:

- KNOWN_FUTURE
- OBSERVED_PAST
- STATIC
- CALENDAR
- OTHER

This is acceptable for first-round **Core-only** training.

Before U2, add stable semantic types such as:

- LOAD_FC
- WIND_FC
- SOLAR_FC
- GENERATION/RENEWABLE_FC
- WEATHER_FC
- ACTUAL_LAG
- DA_PRICE / PRICE_LAG
- OTHER

Availability and forecast-vs-actual legality must remain explicit.

---

# 3. What should remain unchanged

Do **not** “fix” the remaining engineering issues by changing the mathematical core.

Keep:

1. host frozen after host-fit;
2. signed hyperbolic coordinate;
3. learnable three-atom masses;
4. IAH-CRPS as the only S2/U1/U2 candidate objective;
5. CAGM exact 1D W1;
6. query supplies correction dose;
7. at most one Down + one Up contiguous event;
8. whole-day action-value split conformal DVG;
9. S4 fail-closed Identity.

Do not reintroduce:

- BCE occurrence loss;
- SmoothL1 magnitude loss;
- state loss;
- CARA/KL;
- temperature calibration;
- arbitrary spike thresholds;
- hour-level confidence gates;
- extra tail losses.

---

# 4. Overall training architecture

The master training program is:

\[
\boxed{
\text{Host cache}
\rightarrow
\text{U1-native universal core}
\rightarrow
\text{transfer falsification}
\rightarrow
\text{U0 foundation distillation}
\rightarrow
\text{U1-FM-init}
\rightarrow
\text{U2-rich}
\rightarrow
\text{frozen deployment/local calibration}
}
\]

The ordering is deliberate.

We first prove HCH can learn a useful cross-market correction prior **without** a foundation-model teacher. Otherwise a later gain from MOMENT/TTM cannot be attributed cleanly.

---

# 5. Stage T0 — code/semantic smoke

### Data

- `LAGO_DE`
- Host: `Linear`
- one seed.

### Objective

Not performance.

Verify:

- finite IAH-CRPS;
- valid mass simplex;
- nonnegative shifts;
- no target in S4;
- non-empty memory and calibration;
- selected k valid;
- non-empty evidence JSON;
- freeze/reload reproduces candidate and decision chain.

No paper claim is allowed from T0.

---

# 6. Stage T1 — U1-Native: first real universal candidate

## 6.1 Unit of domain

\[
\boxed{
g=(\text{market series},\text{target},\text{host})
}
\]

A market with two hosts supplies two different host-residual regimes, which is useful training diversity.

Market identity itself is metadata, not a predictive embedding shortcut.

## 6.2 Training loss

Only:

\[
L_{\rm IAH-CRPS}.
\]

No dataset-specific loss.

Different datasets are handled through:

- scale-free coordinates;
- local S1 rank reference;
- per-domain Data Signature;
- balanced sampling;
- later local evidence/calibration.

## 6.3 Source data strategy

First version should **not** use all repository datasets.

Use a deliberately heterogeneous subset first. Expand only after the learning dynamics are understood.

Shandong remains excluded from U1-native source training during the scientific transfer phase.

## 6.4 Host strategy

First serious set:

- Linear
- MLP
- LSTM
- PatchTST

Do not include TCN in v1 training.

Reason: the initial goal is not to maximize host count; it is to create enough distinct error families to test model-independence without unnecessary training cost.

---

# 7. Stage T2 — transfer falsification

Two distinct questions must be tested.

## 7.1 Leave-one-market-family-out

Remove an entire market family from candidate training and evaluate the frozen candidate on it.

Example:

- train on DE/PJM/NEM source families;
- evaluate frozen candidate transfer on Nordic DK1.

Avoid weak holdouts such as training on one German file and testing on another German file while calling it “unseen market”.

## 7.2 Leave-one-host-family-out

Train the universal core without one host architecture, then place the frozen corrector over that unseen host.

This supports the “model-agnostic” claim more directly than merely training on many hosts and evaluating on those same families.

---

# 8. Stage U0 — server-scale foundation representation distillation

This stage is **separate pretraining**, not an additional HCH loss.

The user plans to run this on a server, so the design can exploit larger memory/GPU resources.

## 8.1 Teacher role

Recommended first teacher:

`AutonLab/MOMENT-1-small`

Why:

- official model card exposes representation/embedding mode;
- its config uses a 512-step sequence length;
- MIT license;
- suitable for teacher-feature extraction.

The foundation model teaches a representation of temporal dynamics, **not HCH correction actions**.

## 8.2 Student role

Add a small causal `HistorySignatureEncoder`.

Input at deployment must be legally available pre-outcome history only, e.g.

\[
Y_{t-L:t-1}
\]

transformed into a signed scale-free coordinate.

Recommended historical normalization:

\[
s_{\rm hist}
=
\frac1L\sum_{\tau=t-L}^{t-1}|Y_\tau|,
\qquad
z_\tau=\operatorname{asinh}(Y_\tau/s_{\rm hist})
\]

with fail-closed handling when scale is unidentified.

Output:

\[
e^{history}\in\mathbb R^{d_h}.
\]

This becomes the learned part of the Data Signature; it does not directly emit Down/Up actions.

## 8.3 Distillation loss

Because teacher and student dimensions need not match, prefer relational distillation:

\[
K^T_{ij}=\cos(e_i^T,e_j^T),
\qquad
K^S_{ij}=\cos(e_i^S,e_j^S)
\]

\[
\boxed{
L_{U0}
=
\frac1{B^2}\sum_{i,j}
\rho(K^T_{ij}-K^S_{ij})
}
\]

where \(\rho\) can be squared error or Huber.

After U0, this loss disappears.

HCH candidate training still uses:

\[
\boxed{L=L_{\rm IAH-CRPS}}.
\]

## 8.4 Offline feature-bank strategy

Do not run MOMENT inside every HCH epoch.

Server workflow:

```text
selected HF/repo windows
        ↓
frozen MOMENT teacher
        ↓
teacher embedding bank
        ↓
unload teacher
        ↓
train small HistorySignatureEncoder
        ↓
save student weights only
```

This makes later HCH training cheap and reproducible.

## 8.5 Hugging Face data ladder

Do not download the entire Chronos collection by default. The official collection currently contains many configs and the repository is extremely large; some subsets are hundreds of GB.

Start with energy-heavy data, then scale by streaming.

### U0-A sanity corpus

- `autogluon/chronos_datasets : monash_electricity_hourly`
  - 321 series;
  - about 31 MB download in the current HF dataset card;
  - CC BY 4.0 for that config.
- `Salesforce/lotsa_data : australian_electricity_demand`
  - about 4.62 MB current artifact;
  - LOTSA repo license Apache-2.0.

### U0-B server corpus

Expand with selected hourly/energy configs from Chronos/LOTSA, prioritizing:

- electricity demand;
- solar;
- energy/load series;
- other regular hourly physical series.

Keep a **corpus manifest** with exact config, license, series count, window count and sampling weight.

### U0-C broad representation stress test

Only if U0-B helps downstream HCH:

- stream larger Chronos training corpus or selected non-energy time-series;
- maintain energy-heavy sampling;
- do not claim these data directly trained the HCH correction objective.

## 8.6 Avoid teacher-data leakage into final benchmarks

Before U0 scale-up, create a contamination registry.

If a final benchmark series is known to be in the teacher's public pretraining corpus or in our distillation corpus, distinguish:

- **representation-pretrained transfer**
from
- **strict unseen-data transfer**.

The paper must not silently present both as the same zero-shot condition.

---

# 9. TTM role

Recommended role for:

`ibm-granite/granite-timeseries-ttm-r2`

is primarily a **cheap frozen host / host-diversity generator**, not the main representation teacher.

Current official HF model card describes TTM-R2 as a very small pretrained time-series forecaster; the main model page reports roughly 805k parameters and a small safetensors artifact, Apache-2.0 licensed.

Later we can create additional correction domains:

\[
(\text{HF energy series}, \text{TTM host})
\]

without training a new PatchTST for every external series.

This is optional U1-Energy+ expansion, not first-round training.

---

# 10. Stage T4 — U1-FM-init

After U0 student is trained:

1. initialize `HistorySignatureEncoder` from U0;
2. attach its output to Data Signature;
3. train HCH candidate on the **same U1-native price datasets**;
4. keep candidate objective IAH-CRPS only.

Required controlled comparison:

\[
\text{HCH-Native}
\quad \text{vs} \quad
\text{HCH-FM-init}.
\]

Same data split, hosts, seeds and training budget.

This isolates whether foundation representation knowledge helps correction transfer.

---

# 11. Stage T5 — U2-Rich optional covariate adaptation

Only after U1 core is stable.

Use markets with legally available rich covariates, such as suitable Lago/GEFCom/domestic data.

Protocol:

1. freeze U1 core;
2. train only optional branch / explicitly approved tiny adapter;
3. use semantic feature roles;
4. randomly drop optional feature groups;
5. include full optional-dropout cases so the branch cannot assume rich data always exists;
6. retain IAH-CRPS only.

Goal:

\[
\text{rich data improves when present}
\]

while preserving:

\[
\text{price-only HCH-Core remains usable}.
\]

---

# 12. Target-market deployment / local adaptation

The reusable object should be split into:

## Global

\[
\theta_{\rm universal}
\]

possibly plus frozen HistorySignatureEncoder.

## Local target components

- host model/cache;
- S1 rank reference;
- target Data Signature profile;
- CAGM atom memory;
- selected k;
- DVG conformal q.

This produces three deployment protocols.

### F1 Universal-Frozen

Universal candidate frozen, target-local evidence/calibration built from target history.

Main practical protocol.

### F2 Zero-shot corrector candidate

Candidate core receives no target-market gradient updates.

S1 rank/signature may be built from pre-outcome host/history, but no target outcome is used to update candidate weights.

Do not call the entire forecasting system “zero-shot” if the host itself was trained on target labels.

### F3 Few-shot Safe

Universal candidate remains frozen.

Use 7/30/90-day target outcomes only to build or refresh memory/calibration.

### F4 Fine-tune baseline

Allow candidate fine-tuning only as a comparison.

It is not the preferred HCH protocol.

---

# 13. Metrics by training stage

## Candidate training / validation

Primary:

\[
\text{IAH-CRPS}
\]

and deterministic host transformed-score baseline:

\[
L_{\rm host}=|z^Y-z^0|.
\]

Report:

\[
\Delta CRPS
=
L_{\rm IAH}-L_{\rm host}.
\]

Also log:

- per-domain validation CRPS;
- macro mean;
- worst domain;
- atom mass entropy;
- mean \(w^-,w^0,w^+\);
- nonzero shift rate;
- shift magnitude;
- invalid-scale rate;
- modulation norm.

## Final S4 point forecast

Primary paper-comparable metrics:

- MAE
- rMAE
- RMSE
- standard no-floor sMAPE

Tail diagnostics:

- negative-price MAE;
- negative sign miss;
- negative bias;
- high-tail MAE / underestimation bias.

Gate/evidence:

- release rate;
- Identity rate;
- erroneous-release/harm rate;
- \(E[A_{\rm true}\mid execute]\);
- empirical LCB coverage.

Do not infer sMAPE/MAE/profit guarantees from DVG's action-value conformal statement.

---

# 14. Reproducibility artifacts required for every serious run

Every training run must save:

```text
run/
├── training_manifest.json
├── git_commit.txt
├── data_manifest.csv
├── host_cache_manifest.csv
├── domain_manifest.csv
├── split_manifest.json
├── checkpoint_best.pt
├── optimizer_config.json
├── validation_by_domain.csv
├── training_curve.csv
├── signature_stats.csv
├── mass_shift_stats.csv
└── README.md
```

Every S4 evaluation must additionally save:

```text
s4/
├── predictions.parquet
├── metrics_by_domain.csv
├── gate_evidence.jsonl
├── candidate_metrics.csv
├── dm_tests.csv
└── summary.md
```

A paper result without these artifacts is not considered auditable.

---

# 15. Decision rule for proceeding to large-scale distillation

Do **not** start server-scale U0 merely because the code can run.

Proceed from U1-native to U0 only if at least one of the following is true:

1. universal candidate improves macro validation CRPS but transfer to unseen market is weak;
2. source-market results are good but Data Signature lacks enough temporal-domain information;
3. learned core clearly benefits from multiple domains and does not collapse to Identity.

If U1-native cannot improve even its own macro S2V or source S4 conditions, U0 is unlikely to solve the underlying correction formulation; debug U1 first.

---

# 16. Current overall sequence

```text
A. Fix P0/P1 training semantics
        ↓
B. Regenerate real-data T0 smoke
        ↓
C. First-round repo-only U1-native
        ↓
D. Cross-market + local-vs-universal falsification
        ↓
E. Server U0 MOMENT representation distillation
        ↓
F. U1-FM-init controlled comparison
        ↓
G. U2-rich covariate adaptation
        ↓
H. Expanded markets / hosts
        ↓
I. Shandong held-out business validation
        ↓
J. Final formal ablations + paper tables
```

Shandong is intentionally late: it should first function as a scientifically meaningful unseen/business target rather than a source market that the universal core has already absorbed.

---

# 17. Sources / audit anchors

Repository paths audited at commit `769e4215...`:

- `src/hch_v2_context.py`
- `src/iah_candidate.py`
- `src/hch_v2_pipeline.py`
- `src/eval_manifest.py`
- `src/common.py`
- `experiments/08-hch-v2/smoke_v4.py`
- `experiments/08-hch-v2/host_cache.py`
- `experiments/08-hch-v2/results/v0.3/smoke_v4_lago_de_linear.json`

External official resources checked 2026-08-13:

- Hugging Face `AutonLab/MOMENT-1-small`
- Hugging Face `ibm-granite/granite-timeseries-ttm-r2`
- Hugging Face dataset `autogluon/chronos_datasets`
- Hugging Face dataset `Salesforce/lotsa_data`

This plan is a living training protocol. Changes are allowed, but any change affecting splits, objective semantics, domain sampling, freeze scope or final evaluation must increment the document version and be recorded in the run manifest.
