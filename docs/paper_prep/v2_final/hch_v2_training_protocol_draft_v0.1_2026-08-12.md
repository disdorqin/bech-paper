# HCH-v2 Training Protocol Draft v0.1

> Date: 2026-08-12  
> Status: **LIVING DRAFT — intended to change during early training**  
> Mathematical training score: IAH-CRPS remains the only supervised candidate score.  
> Architecture reference: `hch_v2_v0.4_universal_adaptive_architecture_design_2026-08-12.md`

---

# 0. Training objective

The training problem is not simply:

> concatenate all datasets and fit one network.

The intended objective is:

\[
\boxed{
\text{learn a compact transferable correction prior}
\rightarrow
\text{freeze it}
\rightarrow
\text{adapt observability cheaply}
\rightarrow
\text{build target-local evidence}
}
\]

Two orthogonal stage systems must be kept separate:

### Global parameter acquisition

\[
U0\rightarrow U1\rightarrow U2\rightarrow \text{Freeze}
\]

### Per-domain mathematical/evaluation split

\[
S1\rightarrow S2\rightarrow S3\text{-M}\rightarrow S3\text{-C}\rightarrow S4
\]

`U*` answers “how do we obtain reusable parameters?”  
`S*` answers “which outcomes may each mathematical component see?”

---

# 1. U0 — Optional Foundation Representation Prior Injection

## 1.1 Purpose

Our data volume is not comparable with industrial TSFM pretraining corpora.

Use public pretrained time-series models only to provide a better initialization for the HCH temporal representation.

Do not distill their forecasting head into HCH.

Target knowledge:

- periodic structure;
- local dynamics;
- trend;
- volatility/regime geometry;
- generic temporal similarity.

## 1.2 Main protocol: HCH-native first

The scientific mainline must remain reproducible without U0.

Release/evaluate:

- `HCH-native`: no foundation teacher;
- `HCH-FM-init`: optional representation-prior enhancement.

The paper’s core claim should survive without teacher contamination.

## 1.3 Teacher choice

First teacher candidate:

- MOMENT embedding model, because representation extraction is a native task.

Later optional teachers:

- Chronos representation/hidden states;
- TimesFM;
- TTM or other compact public models.

Do not start with a multi-teacher system before single-teacher value is demonstrated.

## 1.4 Offline Teacher Feature Bank

Teacher forward passes are performed once.

For selected windows:

\[
X_i\xrightarrow{T}t_i
\]

cache:

- sample/window ID;
- teacher/version;
- selected representation layer;
- embedding;
- input preprocessing hash.

Then unload teacher.

Student U0 training reads the bank; teacher does not participate online in every optimization step.

## 1.5 Representation rather than forecast distillation

Preferred first loss for U0:

### relational geometry matching

For a batch:

\[
K^T_{ij}=\cos(t_i,t_j),
\qquad
K^S_{ij}=\cos(e_i,e_j).
\]

Train initialization such that:

\[
K^S\approx K^T.
\]

This avoids requiring teacher and student latent coordinates to match dimension-by-dimension.

U0 is **pretraining initialization**, not IAH candidate training.

Therefore a distillation objective in U0 does not alter the statement:

> IAH candidate supervised optimization uses one CRPS score.

## 1.6 Compute-control tricks

- cache teacher embeddings;
- distill on a representative subset, not every hour;
- stratify coverage by dataset, season, volatility and host-error regime;
- use one teacher first;
- student remains small;
- no teacher gradients;
- no teacher during HCH U1/U2/S3/S4.

## 1.7 Contamination audit

For every public teacher record:

- known pretraining corpus;
- whether an HCH test dataset may overlap;
- whether teacher saw the target dataset family.

If overlap cannot be excluded, HCH-FM-init is reported separately from the native mainline.

---

# 2. U1 — Common-Core Training

## 2.1 Central rule

All source datasets are deliberately reduced to the **same minimum information set**.

Even rich domestic datasets do not expose optional exogenous features in U1.

Thus price-only public datasets are first-class training domains.

## 2.2 Domain definition

Training domain:

\[
g=(\text{dataset/series},\text{host backbone}).
\]

Host diversity is essential because HCH learns host-relative correction behavior.

Recommended initial host families:

- Linear/LEAR-like;
- MLP;
- LSTM/TCN;
- PatchTST/Transformer.

Foundation hosts can be added later through offline host cache.

## 2.3 Host cache

Every host is trained/frozen according to the experiment manifest.

Cache:

- timestamp;
- dataset;
- host name/version;
- host prediction;
- split hash;
- host training-data hash;
- seed.

HCH training reads cached host predictions.

Never run a large host inside every HCH optimization step.

## 2.4 Core-only inputs

U1 input must exclude optional covariates.

Use:

- \(z^0\);
- local S1 rank \(u\);
- time features;
- scale-free legal lag/history context;
- Data Signature built only from legal pre-outcome core information.

No predictive market/target identity token by default.

## 2.5 Single supervised score

For every U1 sample:

\[
\mathcal L
=
\mathcal L_{\rm IAH-CRPS}.
\]

No:

- BCE occurrence;
- magnitude L1;
- state loss;
- domain classification;
- tail loss;
- market loss;
- trading loss;
- distillation loss.

## 2.6 Domain-balanced sampling

Do not concatenate all hours and sample uniformly.

Large datasets would dominate.

Preferred sampler:

1. sample domain \(g\) approximately uniformly or with a predeclared capped weight;
2. sample a day/window from that domain;
3. compute IAH-CRPS.

Equivalent target:

\[
\mathcal L_{\rm U1}
=
\frac1{|\mathcal G|}
\sum_{g\in\mathcal G}
\mathbb E_{d\sim g}
[
\mathcal L_{\rm IAH}(d)
].
\]

This changes sampling/aggregation, not the score.

## 2.7 Do not tail-oversample without correction

Blind spike/negative-price oversampling changes the effective target distribution and therefore changes the CRPS projection.

Default:

- domain balancing is allowed;
- outcome-tail oversampling is not part of U1.

If later required, inverse-probability weighting must restore the intended expectation and be treated as a separate experiment.

## 2.8 U0-to-U1 progressive specialization

If U0 is used:

### U1-A
freeze representation encoder, train IAH projection/head.

### U1-B
unfreeze a small upper part of the core encoder with lower learning rate.

### U1-C
freeze the final \(\theta_{\rm core}^*\).

If U0 is not used:

train the compact core normally with IAH-CRPS.

This is an implementation strategy, not a paper contribution.

---

# 3. U2 — Optional Context Expansion

## 3.1 Goal

Learn how extra information helps **without rewriting the common correction prior**.

Freeze:

\[
\theta_{\rm core}^*.
\]

Train:

\[
\theta_{\rm optional},
\quad
\omega_{\rm conditioning}.
\]

## 3.2 Input

Only datasets with legal optional covariates participate in their rich form.

Optional roles:

- known-future covariates;
- observed-past covariates;
- static covariates;
- calendar extras;
- other legal pre-outcome features.

## 3.3 Residual injection

Use a zero/near-zero initialized optional branch:

\[
h_{\rm final}
=
h_{\rm core}
+
g(e,M)\odot h_{\rm optional}.
\]

At initialization, output should approximate HCH-Core.

## 3.4 Masking as information-distribution training

For a rich sample draw an availability mask:

\[
M\sim Q(M).
\]

Train the candidate under:

- all optional features;
- partial feature groups;
- selected feature dropout;
- full optional-feature dropout;
- variable history masks if legal.

The supervised score remains:

\[
\mathbb E_M[
\mathcal L_{\rm IAH-CRPS}
].
\]

Masking changes the information condition, not the objective.

## 3.5 Required degradation test

For every rich dataset:

\[
\text{all optional covariates masked}
\]

must reproduce the core-only path within numerical tolerance.

If it does not, the optional interface is not structurally safe enough.

---

# 4. U3 — Optional Few-Shot Adaptation Training (Deferred Candidate)

This stage is **not required before first experiments**.

Goal: make the released checkpoint easy for downstream users to adapt.

For a source domain, simulate a new user:

\[
D_g=D_g^{support}\cup D_g^{query}.
\]

Freeze universal core.

Update only a tiny adapter:

\[
\phi_g
\]

on support using IAH-CRPS.

Evaluate on query.

Possible future variants:

- simple adapter training;
- episodic support/query training;
- meta-initialization.

Do not introduce MAML-level complexity unless ordinary few-shot adapters are insufficient.

---

# 5. Per-Domain S1/S2/S3/S4 inside universal training

Universal parameter training must still respect information boundaries.

For every source dataset/host pair:

## S1
- host training / OOF generation;
- local rank reference;
- all normalizers/statistics permitted by the contract.

## S2
- provides candidate labels for U1/U2 optimization;
- no S3/S4 outcomes may be used to update \(\theta\).

## S3-M
- memory prefix;
- validation suffix for frozen \(k\);
- proposal policy fixed.

## S3-C
- candidate, memory, k and proposal frozen;
- calibrate only whole-action error quantile.

## S4
- untouched evaluation.

For universal core training, many source-domain S2 batches contribute to one shared \(\theta_{\rm core}\).

S1/S3 objects remain domain-local.

---

# 6. Preliminary dataset plan

This is a draft and should be revised after data-contract audit.

## 6.1 Main domestic / business realism

- Shandong DA;
- Shandong RT;
- other Chinese provinces after schema/cutoff audit.

## 6.2 Reproducible public electricity-price datasets

Priority pool already in repository:

- Lago markets;
- GEFCom price track;
- NEM SA1;
- EPEX / NordPool / PJM price-only datasets;
- UniElecPrice selectively.

## 6.3 Paired foreign DA/RT expansion

Preferred future additions:

- PJM DA/RT;
- NYISO DA/RT.

Do not delay first core smoke until every foreign market is collected.

## 6.4 Rich vs price-only protocol

Every dataset can contribute to U1 core training.

Only legally rich datasets contribute extra covariates to U2.

This avoids forcing feature-poor datasets into an artificial rich schema.

---

# 7. Preliminary experiment matrix

## 7.1 Frozen transfer

- Local HCH core;
- Universal HCH-Core;
- leave-one-market-out frozen core;
- few-shot local adapter;
- optional local fine-tune baseline.

## 7.2 Input robustness

- core only;
- core + rich features;
- core + masked rich features;
- all optional features removed.

## 7.3 Foundation initialization

- random/native initialization;
- one-teacher representation initialization;
- optional multi-teacher only after evidence.

## 7.4 Host transfer

- leave-one-host-family-out;
- mixed-host universal training;
- unseen host correction.

---

# 8. Early-training diagnostics

Do not judge training only by final MAE.

During U1/U2 track:

### Candidate
- CRPS;
- \(w^-,w^0,w^+\);
- \(m^-,m^+\);
- atom collapse frequency;
- p95/p99 and negative/near-zero subsets.

### Transfer
- per-domain CRPS;
- per-host CRPS;
- worst-domain gap;
- domain imbalance of gradients / update counts;
- performance after optional-feature removal.

### Geometry
- scale-equivariance checks;
- Data Signature distribution across domains;
- whether signatures collapse to dataset identity;
- whether new domains fall far outside source signature support.

### Local evidence
- retrieval neighbor stratification;
- \(\widehat A\) vs \(A\);
- DVG empirical coverage;
- release/harm/Identity rates.

---

# 9. Evaluation metrics draft

Main comparable statistical metrics:

- MAE;
- rMAE;
- RMSE;
- standard no-floor sMAPE.

Method-consistent diagnostics:

\[
d_s(y,\hat y)
=
|
\operatorname{asinh}(y/s)
-
\operatorname{asinh}(\hat y/s)
|.
\]

Use this as a geometry diagnostic unless the final math/experiment review promotes it to a named metric.

Tail/event/gate metrics remain necessary.

SCR/WSCR and DA/RT market-value analysis belong to evaluation, not U1/U2 optimization.

---

# 10. Training stop/go rules

Stop or redesign before large-scale training if:

1. core-only path cannot learn useful non-Identity candidates;
2. learned masses collapse globally;
3. one/two datasets dominate universal gradients despite balanced sampling;
4. feature-poor domains are systematically worse after U2;
5. optional branch fails to degrade to core when features are masked;
6. retrieval geometry does not correlate with action value;
7. DVG releases harmful actions at unacceptable empirical rate;
8. universal core is materially worse than local core on most source domains.

Do not “fix” these automatically by adding auxiliary losses.

First diagnose whether the failure belongs to:

- representation;
- candidate family;
- sampling;
- optional interface;
- retrieval;
- proposal;
- gate.

---

# 11. Compute budget strategy

First training cycle should be cheap.

1. use cached classical/neural host predictions;
2. start with HCH-native;
3. train compact U1 core on a reduced multi-market subset;
4. verify transfer and core-only behavior;
5. only then run U0 teacher bank;
6. U2 only after U1 checkpoint is stable;
7. large TSFMs are offline teachers/hosts, never part of HCH backprop graph.

The intent is to exploit public pretrained knowledge **without reproducing foundation-model-scale training**.

---

# 12. Literature grounding for this draft

The following works motivate, but do not define, HCH:

- MOMENT (ICML 2024): large heterogeneous multi-dataset time-series representation pretraining and embedding mode.
- Tiny Time Mixers (NeurIPS 2024): small models can obtain strong zero/few-shot transfer through careful multi-dataset pretraining.
- UniTime: cross-domain time-series learning faces variable-count heterogeneity, domain confusion and unequal convergence; masking is used to improve unified training.
- Moirai-MoE (ICML 2025): human-defined domain/frequency specialization is coarse; data-driven specialization can be useful.
- UniCA (ICLR 2026): heterogeneous covariates can be homogenized and injected while preserving pretrained generalization.
- CoRA (2025 preprint): frozen backbone plus non-destructive covariate injection is a useful adaptation pattern.
- DistilTS (2026 preprint): TSFM knowledge distillation is practically viable, but HCH uses distillation only as representation initialization rather than forecasting imitation.

These references support design choices; they are not evidence that the complete HCH training protocol is already validated.

---

# 13. Versioning policy

This document is intentionally mutable.

Update after each early experiment with:

- datasets actually used;
- host families actually cached;
- U0 teacher choice;
- U1 sampler weights;
- U1/U2 learning rates and freezing schedule;
- mask distribution \(Q(M)\);
- few-shot support sizes;
- stop/go findings.

The architecture document should change less frequently than this training document.
