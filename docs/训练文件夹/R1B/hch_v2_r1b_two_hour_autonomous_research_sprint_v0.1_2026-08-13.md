# HCH-v2 R1B — Two-Hour Autonomous Research Sprint v0.1

**Date:** 2026-08-13  
**Authoritative starting commit:** `main@89b5e27569555ed294ebb181fb95a191003eb5f1`  
**Parent protocol:** `hch_v2_r1b_universal_generalization_battery_v0.1_2026-08-13.md`

## Mission

The user will be unavailable for approximately two hours.

Use this window to make **substantial, auditable progress toward R1B**, while preserving the core scientific rule:

> Do not optimize the known source domains at the cost of true generalization.

This is an autonomous execution window, not authorization to redesign HCH.

The priorities are:

1. establish a reproducible server/workstation R1B environment;
2. audit actual repository support for new hosts and DK1;
3. generate trustworthy LSTM/PatchTST host caches;
4. implement the R1B generalization runner and transfer ledger;
5. run as much **single-seed candidate-level screening** as time allows;
6. perform one genuine unseen-host test and one unseen-market test;
7. in parallel, prepare read-only literature/data audits for later DA/RT and U0 stages;
8. package everything for human scientific review;
9. STOP before architecture changes, multi-seed confirmation, R1C, or U0.

---

# 0. Non-negotiable scientific state

R1A is CLOSED.

The accepted mechanism entering R1B is:

```text
Frozen Host
    ↓
Universal IAH Candidate
    ↓
analytic raw action utility
    ↓
prequential local C0/C3 calibration eligibility
    ↓
double-event proposal
    ↓
S3C DVG
```

The current scientific claims are limited to:

```text
UNIVERSAL_CANDIDATE_SUPPORTED
PREQUENTIAL_CALIBRATION_ROUTING_SUPPORTED
```

R1B must test whether these mechanisms survive:

- new host family;
- new market;
- different host feature schema;
- lower/higher host error regimes.

Do NOT attempt to prove DA/RT universality or rich-feature universality in this sprint.

---

# 1. Hard prohibitions during the unattended window

Do NOT:

- change IAH mathematics;
- change IAH-CRPS;
- add new losses;
- modify C0/C3 calibration mathematics;
- modify DVG mathematics;
- add a neural router;
- add market ID or host ID as predictive inputs;
- use S4 for model or hyperparameter selection;
- add Shandong;
- add TCN to the R1B experiment;
- launch MOMENT/U0 training;
- download huge HF corpora;
- add new source markets to “fix” weak transfer;
- tune learning rate/model width after seeing DK1 or PatchTST results;
- run 3-seed confirmatory experiments automatically;
- merge architecture changes into `main` because a result is bad.

If a shared/production-code bug is discovered:

1. reproduce it;
2. write a failing test or minimal reproduction;
3. record it in `R1B_BLOCKERS.md`;
4. prepare a proposed patch if useful;
5. **do not silently alter the scientific protocol to make the run pass.**

Experiment-only runners, audit scripts, manifests, and tests are allowed.

---

# 2. Git and branch discipline

Before any work:

```bash
git status
git rev-parse HEAD
```

Record the exact SHA.

Prefer creating:

```text
exp/r1b-screening-20260813
```

for unattended work.

Commit:

- experiment runners;
- tests;
- small manifests;
- audit docs;
- summary CSVs.

Do not commit:

- raw datasets;
- large caches;
- checkpoints;
- large parquet/npy artifacts.

Do not merge the experiment branch to `main` while the user is away.

If the user's existing workflow requires staying on `main`, then restrict commits to:

- new R1B experiment code;
- tests;
- docs;

and do not modify formal production modules without an explicit blocker report.

---

# 3. Two-hour priority schedule

This is a priority queue, not a promise that every GPU job will finish in exactly two hours.

## T+00–15 min — Environment & reproducibility

### 3.1 If the 4090 server exists

Record:

```text
GPU model
VRAM
driver
CUDA
PyTorch
Python
CPU count
RAM
disk mounts
free disk
git SHA
```

Set storage caches to the data disk, not the 30GB system disk.

Recommended:

```bash
export HF_HOME=/root/rivermind-data/hf_cache
export TORCH_HOME=/root/rivermind-data/torch_cache
export TRANSFORMERS_CACHE=/root/rivermind-data/hf_cache
```

Create:

```text
/root/rivermind-data/
├── datasets/
├── host_cache/
├── hf_cache/
├── torch_cache/
├── experiments/
└── checkpoints/
```

### 3.2 Run repository tests

Run at minimum:

- P0/P1 HCH tests;
- R1A.11 relevant tests if present;
- one minimal candidate forward/load smoke.

Save output to:

```text
R1B_PREP_<timestamp>/environment.txt
R1B_PREP_<timestamp>/tests.txt
```

### 3.3 Fail-closed rule

If the environment cannot reproduce the current code path:

STOP GPU experiment launch.

Continue only the read-only audits in §12–§15.

---

# 4. Repository capability audit

Before writing the R1B runner, confirm from code and actual execution:

## 4.1 Hosts

Current expected host set:

```text
Linear
MLP
LSTM
PatchTST
```

Verify:

- constructor works;
- sequence length is legal/cutoff-safe;
- H0-only fitting;
- device placement;
- early stopping;
- deterministic seed;
- prediction shape and alignment;
- cache provenance.

### Important naming audit

The in-repo `PatchTST` implementation is a lightweight local implementation using:

```text
patch extraction
TransformerEncoder
static-feature head
```

Do NOT automatically claim it is a faithful official PatchTST reproduction.

Create:

```text
docs/paper_prep/v2_final_prep/r1b_host_backbone_fidelity_audit_v0.1.md
```

Classify each host:

```text
canonical/basic implementation
paper-faithful reproduction
architecture-inspired local implementation
```

For the current Transformer host, recommend a scientifically honest naming convention.

This audit is mandatory before final paper comparisons.

---

# 5. Dataset and feature-schema audit

Create:

```text
docs/paper_prep/v2_final_prep/r1b_domain_feature_schema_audit_v0.1.md
```

For:

```text
LAGO_DE
LAGO_PJM
NEM_SA1
NORD_DK1
```

record:

- date coverage;
- currency;
- target;
- negative-price rate;
- number of valid days;
- host feature count;
- known-future exogenous variables;
- lagged observed variables;
- price lags;
- calendar variables;
- sequence availability;
- split counts:
  - H0
  - S1R
  - S2T
  - S2V
  - S3M
  - S3C
  - S4
- DST/excluded-day count;
- feature-schema hash.

Explicitly distinguish:

### host-input heterogeneity

The host forecaster may use different feature schemas.

### HCH-core heterogeneity

The current HCH universal core does **not** yet consume arbitrary rich optional covariates directly.

Therefore R1B can support:

> “the corrector works across hosts trained from heterogeneous market schemas”

but not yet:

> “HCH directly consumes arbitrary rich feature schemas.”

That stronger claim belongs to R1C/U2.

---

# 6. Host-cache generation

## 6.1 Mandatory first smoke

Before batch generation, run:

```text
LAGO_DE × LSTM
LAGO_DE × PatchTST
```

Verify:

- H0-only fit;
- valid-row indexing;
- no leakage assertion;
- finite predictions;
- pred hash;
- exact seven-segment metadata;
- GPU memory;
- wall-clock.

If either smoke fails:

STOP that host branch and document.

Do not tune architecture to force it through.

## 6.2 Then generate new-host caches

Priority set:

```text
LAGO_DE   × LSTM
LAGO_DE   × PatchTST
LAGO_PJM  × LSTM
LAGO_PJM  × PatchTST
NEM_SA1   × LSTM
NEM_SA1   × PatchTST
NORD_DK1  × LSTM
NORD_DK1  × PatchTST
```

Use:

```text
host_seed = 0
```

Do not run TCN.

Record:

```text
duration
peak VRAM if available
split hash
pred hash
feature schema hash
n_params
MAE by segment
S2V host transformed baseline
S4 host MAE
```

Cache generation is more valuable than launching half-finished multi-seed HCH runs.

---

# 7. Host-quality sanity screen

Before universal candidate training, create:

```text
host_quality_by_domain.csv
```

Rows:

```text
market × host
```

Columns:

- H0 validation metric if available;
- S1R MAE;
- S2V MAE;
- S4 MAE;
- transformed host error;
- residual mean;
- residual std;
- residual IQR;
- lag-1 residual ACF;
- negative-residual rate;
- large-positive residual rate;
- large-negative residual rate.

Purpose:

> determine whether R1B actually spans heterogeneous host-error regimes.

Also make one scatter:

\[
x = \text{host transformed error}
\]

\[
y = \text{future HCH candidate improvement}
\]

when HCH screening results become available.

This checks whether HCH only helps weak hosts.

---

# 8. Implement the R1B screening runner

Create:

```text
experiments/08-hch-v2/r1b_generalization_screen.py
```

The runner must separate:

```text
SOURCE TRAINING DOMAINS
TRANSFER EVALUATION DOMAINS
```

and make transfer status explicit.

Do not overload R1A scripts with hidden flags.

## 8.1 Source candidate training

Primary source markets:

```text
LAGO_DE
LAGO_PJM
NEM_SA1
```

Full seen-host training set for main screening:

```text
Linear
MLP
LSTM
PatchTST
```

Candidate parameters remain:

```text
d_model = 64
d_sig = 32
HCH_seed = 0
AdamW = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
equal-domain sampling
macro S2V CRPS selection
```

No new tuning.

---

# 9. Candidate variants — screening order

Do not automatically run everything.

## Priority 1 — LearnedSig

This is the current provisional main candidate.

Run it first.

## Priority 2 — PlainCore

True no-DataSignature/FiLM path:

\[
h'=h.
\]

This must be a **real bypass**, not deterministic descriptor = 0 while learned signature remains active.

## Priority 3 — Learned+DetSig

Only if time remains after Priority 1/2 candidate-level transfer is available.

R1A suggests deterministic 8-d signature may be redundant, so this is an ablation, not the main run.

## Do not run Local-Core first

Local-Core is useful, but it is not more valuable than obtaining true unseen-market/unseen-host evidence in the two-hour window.

---

# 10. Genuine unseen-host test — mandatory design

Create a LOHO screening configuration.

## LOHO-PatchTST

Train universal candidate only on:

```text
Linear
MLP
LSTM
```

across:

```text
LAGO_DE
LAGO_PJM
NEM_SA1
```

PatchTST must contribute:

```text
ZERO S2T GRADIENT
ZERO S2V CHECKPOINT SELECTION SIGNAL
```

Then evaluate the frozen candidate on:

```text
LAGO_DE:PatchTST
LAGO_PJM:PatchTST
NEM_SA1:PatchTST
NORD_DK1:PatchTST
```

This is one of the most scientifically valuable runs in the sprint.

Label:

```text
UNSEEN_HOST_FAMILY = PatchTST-style
```

until host-fidelity audit is complete.

---

# 11. Genuine unseen-market test

Universal candidate training receives no DK1 gradient.

DK1 may build:

- H0 local host;
- S1R rank/reference;
- local legal history;
- later local calibration evidence.

But universal candidate weights:

\[
\theta_{IAH}
\]

must remain frozen.

First evaluate **candidate-level transfer only**:

- host transformed baseline;
- IAH CRPS;
- delta CRPS;
- mass entropy;
- w0;
- m-/m+ alive rate;
- shift p50/p95;
- NaN/scale failures.

Do not immediately run the full C3/DVG chain if candidate transfer is already catastrophically broken.

---

# 12. Transfer matrix artifact

Create:

```text
R1B_SCREEN_<timestamp>/candidate_transfer_matrix.csv
```

Every evaluated domain gets:

```text
market_seen
host_seen
candidate_variant
host
market
host_baseline
iah_crps
delta_crps
mass_entropy
w0
mminus_alive
mplus_alive
shift_p95
status
```

Produce four aggregate cells:

```text
Seen market + Seen host
Seen market + Unseen host
Unseen market + Seen host
Unseen market + Unseen host
```

Do not collapse these into one macro number.

---

# 13. Generalization Ledger — mandatory

Create:

```text
experiments/08-hch-v2/results/GENERALIZATION_LEDGER.csv
```

Schema:

```text
experiment_id
change
source_macro_effect
source_worst_effect
unseen_market_effect
unseen_host_effect
unseen_market_unseen_host_effect
complexity_added
evidence_level
accepted_for_core
notes
```

Initial rows should include:

```text
LearnedSig vs PlainCore
Learned+DetSig vs LearnedSig
LOHO-PatchTST
DK1 zero-gradient transfer
```

Rule:

> A change that improves source domains while materially degrading unseen market/host cannot be recommended for the universal core.

No exceptions during the unattended sprint.

---

# 14. Candidate-stage STOP / CONTINUE rules

## CONTINUE to action-chain development evaluation only if:

- candidate inference is finite;
- no collapse;
- source macro is healthy;
- DK1 delta CRPS is not catastrophically positive;
- LOHO PatchTST is not systematically broken.

## STOP full-chain execution and report if:

### MARKET_TRANSFER_COLLAPSE

DK1 candidate is clearly worse than host across host families.

### HOST_TRANSFER_COLLAPSE

LOHO PatchTST fails across source markets.

### BOTH_TRANSFER_COLLAPSE

both fail.

### SIGNATURE_NEGATIVE_TRANSFER

LearnedSig improves source domains but clearly hurts unseen cells relative to PlainCore.

If any of these occur:

Do NOT “repair” during this sprint.

Prepare a diagnosis pack only.

---

# 15. If candidate screening is healthy and time remains

Then, and only then, run the R1A.11 local mechanism on a **small selected subset**:

Priority:

```text
NORD_DK1:Linear
NORD_DK1:PatchTST
one source LSTM
one source PatchTST
```

Use the frozen:

```text
prequential C0/C3 eligibility protocol
```

Record:

- S3M days;
- OOS days;
- selector reason;
- C0/C3;
- q;
- release;
- harmful;
- net action value;
- final MAE/rMAE.

Do not modify the gate based on new domains.

The result may be poor. That is valuable evidence.

---

# 16. Scientific side experiment A — Signature generalization probe

If LearnedSig candidate is trained and time remains, perform a **diagnostic frozen probe**.

Extract learned daily signature representation for all domains.

Run two simple probes:

```text
market classifier
host-family classifier
```

using only source-domain training representations.

Purpose:

### If nearly perfect market classification

The learned signature may largely encode domain identity / shortcut information.

### If host classification is strong

The signature may adapt to host residual regime, which could explain cross-host value.

### If both are weak but correction transfer is strong

The representation may be more invariant.

This is diagnostic only.

Do not alter training because of the probe during this sprint.

Save:

```text
signature_probe.csv
signature_embedding_summary.csv
```

No t-SNE conclusions as primary evidence.

---

# 17. Scientific side experiment B — Negative-transfer map

For every source/transfer domain, compute:

\[
\Delta_g^{cand}
=
L_g^{IAH}
-
L_g^{host}
\]

and final metric delta if available.

Create:

```text
negative_transfer_map.csv
```

Classify:

```text
IMPROVED
NEUTRAL
DEGRADED
CATASTROPHIC
```

Use predeclared descriptive thresholds from the R1B parent protocol if present.

If not present, do NOT invent hard publication thresholds during the sprint; report continuous deltas and mark threshold as pending human review.

---

# 18. Parallel read-only research task — host fidelity

While GPU jobs run, inspect the literature/official code for:

```text
PatchTST
LSTM baseline conventions
```

Use primary sources:

- original paper;
- authors' official repository if available.

Answer:

1. Is our in-repo PatchTST close enough to call PatchTST?
2. Which major components are missing/different?
3. For the paper, should it be renamed `PatchTST-style`?
4. How much effort would an official/fair host reproduction take?
5. Does this matter for the model-agnostic claim or only baseline naming?

Do NOT implement official PatchTST during this sprint unless it is trivial and isolated.

Save the audit doc only.

---

# 19. Parallel read-only research task — DA/RT public data

Create:

```text
docs/paper_prep/v2_final_prep/public_da_rt_dataset_audit_v0.1.md
```

Use **official/primary sources only**.

Audit:

```text
NYISO
ERCOT
PJM
```

For each:

- official DA price source;
- official RT price source;
- temporal granularity;
- archive history;
- zone/hub identifiers;
- publication/availability timing;
- legal forecasting cutoff implications;
- negative-price occurrence;
- download mechanism;
- redistribution/license restrictions;
- DST handling;
- approximate local storage after hourly conversion;
- recommended role:
  - R1C primary
  - R1C secondary
  - reference only.

Do not download multi-year raw data during this sprint.

Goal:

> when the user returns, we know exactly which DA/RT pair should be integrated next.

---

# 20. Parallel read-only research task — U0 / HF corpus inventory

Create:

```text
docs/paper_prep/v2_final_prep/u0_external_corpus_inventory_v0.1.md
```

No large downloads.

Inventory a limited set of high-value public time-series data:

- LOTSA energy/electricity subsets;
- Monash electricity;
- Australian electricity demand;
- selected load/solar/wind datasets;
- any already-referenced public datasets in the repo.

For each:

- exact dataset/config name;
- source;
- license;
- frequency;
- number of series;
- approximate size;
- context length compatibility;
- likely role:
  - representation distillation;
  - host diversity;
  - final benchmark;
  - exclude;
- possible overlap with foundation-model pretraining.

Do not claim overlap unless documented by the teacher's published corpus information.

---

# 21. Parallel literature scouting — inspiration, not implementation

Spend at most 20–30 minutes.

Use recent **primary papers / official repos** only.

Focus on:

```text
cross-domain time-series training
domain-balanced multi-dataset training
leave-one-dataset-out evaluation
unseen-domain adaptation
foundation representation transfer
electricity-price cross-market transfer
```

Produce:

```text
docs/paper_prep/v2_final_prep/r1b_literature_scout_v0.1.md
```

For each useful paper:

```text
paper
year
exact experimental idea
what problem it addresses
what HCH could test
what HCH should NOT copy
priority: now / R1C / U0 / later
```

Do not add a technique just because a paper reports gains.

The key question is:

> Which evaluation design can make our universality claim harder to fake?

---

# 22. Allowed exploratory ideas — report only

The AI may analyze these ideas, but must not silently integrate them into the model:

## E1 — Leave-one-market-out joint training
Would source-market withholding reveal negative transfer?

## E2 — Leave-one-host-family-out
Should R1B-B expand beyond PatchTST LOHO?

## E3 — Domain-balanced vs GroupDRO
Only if R1B shows worst-domain collapse.
Do not implement yet.

## E4 — Signature dropout
Could prevent domain-signature shortcut memorization.
Only a future hypothesis.

## E5 — Host identity dropout
If hidden host identity leaks through representation, should training randomize/erase it?
Future only.

## E6 — U0 representation initialization
Could repair unseen-host/unseen-market transfer if native HCH fails.
Future only.

## E7 — Role-based optional feature encoder
Required later for direct rich-feature universality.
Future R1C/U2, not R1B-A.

For each, write:

```text
triggering evidence required before implementation
```

This prevents idea accumulation from turning into architecture bloat.

---

# 23. Resource usage during the two hours

One GPU only.

Do not launch competing large GPU jobs simultaneously.

Recommended:

```text
GPU queue:
1. LAGO_DE LSTM smoke
2. LAGO_DE PatchTST smoke
3. remaining new-host caches
4. LearnedSig candidate screening
5. PlainCore candidate screening
6. LOHO-PatchTST
7. optional Learned+DetSig
```

CPU/network tasks can run in parallel:

```text
literature audit
DA/RT audit
U0 corpus inventory
feature-schema inventory
CSV aggregation
```

Use tmux if available.

Suggested panes:

```text
0: gpu-jobs
1: tests/log-tail
2: audits/docs
3: metrics/aggregation
```

---

# 24. Time-limit behavior

At approximately T+110 min:

STOP launching new long-running jobs.

Allow current safe job to finish if close.

Spend remaining time packaging results.

Do not leave an uncontrolled experiment cascade running because the user is away.

If one expensive cache/training run is still legitimately progressing:

record:

```text
PID
command
start time
current epoch/progress
estimated remaining time
log path
```

and leave it running only if it is a **pre-approved R1B job from this document**, not a newly invented experiment.

---

# 25. Required return package

When the user returns, the AI must provide a concise top-level report:

## A. Environment

```text
server used?
git SHA
GPU
RAM
disk
tests PASS/FAIL
```

## B. Host caches

Table:

```text
market × host
status
duration
pred hash
split hash
S2V host loss
```

## C. Candidate screening

Table:

```text
variant
training domains
source macro
source worst
DK1
LOHO PatchTST
unseen-market+unseen-host
```

## D. Generalization verdict

Exactly one provisional label:

```text
R1B_SCREEN_HEALTHY
MARKET_TRANSFER_WARNING
HOST_TRANSFER_WARNING
SIGNATURE_TRANSFER_WARNING
INFRASTRUCTURE_BLOCKED
SCREENING_INCOMPLETE
```

This is NOT the final R1B verdict.

## E. What was NOT done

Explicitly state:

- no multi-seed;
- no U0;
- no R1C;
- no architecture change;
- no S4 tuning;
- no new loss.

## F. Artifacts

List paths to:

```text
R1B_PREP_*/
R1B_SCREEN_*/
GENERALIZATION_LEDGER.csv
host fidelity audit
feature schema audit
DA/RT audit
U0 corpus inventory
literature scout
R1B_BLOCKERS.md (if any)
```

## G. Proposed next decisions

Maximum 3.

Do not automatically execute them.

---

# 26. Definition of success for this two-hour sprint

The sprint is successful even if no R1B candidate is “good”.

A successful sprint means:

1. the compute environment is trusted;
2. new hosts are either valid or honestly blocked;
3. the R1B runner exists and enforces source/transfer separation;
4. we obtain at least one genuine unseen-host or unseen-market candidate result;
5. no architecture is changed based on transfer outcomes;
6. we have a clear negative-transfer/generalization ledger;
7. future DA/RT and U0 work is better prepared;
8. the user returns to evidence, not speculation.

The sprint should optimize:

\[
\boxed{\text{information gained per two hours}}
\]

not:

\[
\boxed{\text{number of experiments launched}}.
\]
