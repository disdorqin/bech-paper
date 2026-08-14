# HCH-v2 Paper Benchmark Gate — Comparative Experiment & Architecture Decision Protocol v0.1

**Date:** 2026-08-14  
**Repository:** `disdorqin/bech-paper`  
**Purpose:** Run the first decisive paper-oriented comparison of Native HCH-v2 against frozen hosts and post-hoc competitors.  
**Status:** Authoritative DEVELOPMENT experiment order. Not final sealed-test authorization.

---

## 0. Mission

This is not a scaling or distillation experiment.

It answers:

> **Does the current HCH architecture actually beat strong host baselines and relevant post-hoc competitors on the datasets, hosts, and metrics intended for the paper?**

Internal target:

\[
oxed{	ext{HCH is Top-1 or statistically tied Top-1 on a clear majority of primary paper cells.}}
\]

Do not start large-scale supervised expansion or U0 distillation before this gate is scientifically reviewed.

---

## 1. Current HCH paper candidate

The experimentally supported main chain is now:

```text
Frozen Host
    ↓
Host-relative scale-free representation
    ↓
Learned Data Signature + FiLM
    ↓
IAH three-atom predictive correction distribution
    ↓
Analytic action utility
    ↓
Prequential evidence-gated local C0/C3 calibration
    ↓
Signed contiguous Double Event
    ↓
S3C DVG / abstention
    ↓
Final corrected prediction
```

The earlier CAGM/W1 retrieval mechanism is no longer a required main-method component after R1A showed weak action-value ranking. Keep it only as an ablation / historical alternative if useful.

---

## 2. Hard-frozen vs soft-frozen

### 2.1 Hard-frozen mathematical core

Do NOT change during the first benchmark pass:

1. host-relative asinh geometry;
2. three-atom IAH semantics;
3. masses \(w^-,w^0,w^+\);
4. positive shifts \(m^-,m^+\);
5. single discrete IAH-CRPS candidate objective;
6. frozen-host interface;
7. no market-ID predictive embedding.

A change here requires a new math/design review.

### 2.2 Strongly soft-frozen

May be modified later only when benchmark evidence identifies the corresponding failure:

- LearnedSig capacity/context;
- FiLM capacity/placement;
- candidate width;
- training sampler;
- point/action readout;
- local C3 regularization;
- prequential eligibility;
- DoubleEvent details;
- DVG calibration.

### 2.3 Deferred

Do NOT activate in this benchmark:

- U0 / MOMENT distillation;
- large-scale external-corpus training;
- MoE;
- market-ID embeddings;
- rich optional-feature branch;
- new auxiliary loss terms.

---

## 3. Method contributions this benchmark must validate

### A. Universal signed probabilistic correction core
A compact host-relative three-atom correction distribution trained by one proper CRPS objective, jointly representing signed correction probability mass and magnitude.

### B. Dataset/host-adaptive shared prior without identity tokens
Learned Data Signature + lightweight FiLM modulates one shared corrector across heterogeneous markets/hosts.

### C. Selective local decision layer
The distribution is translated into action utility, local recalibration is enabled only when prequential evidence authorizes it, and structured signed corrections are protected by DVG abstention.

---

## 4. P0 before any headline comparison

### P0-A — Correct final point replay

R1B Stage-2D currently evaluates `host_day` in its headline point metrics. Fix this first.

Use one authoritative final-output function:

\[
\pi_{\mathrm{eff}}=
egin{cases}
\pi,&	ext{DVG releases},\
0,&	ext{otherwise},
\end{cases}
\]

\[
oxed{x^{final}=s_d\sinh(z^0+\pi_{\mathrm{eff}})}.
\]

Reuse/centralize the already-correct R1A.9 replay path.

Regression tests:

```text
all Identity -> x_final == host
released nonzero pi -> x_final follows inverse-asinh exactly
bundle reload -> bit/effectively identical x_final
```

No paper benchmark is valid before P0-A passes.

### P0-B — Provenance

Record:

```text
git SHA
architecture version
math-core version
split version
host-code hashes
peer-adapter hashes
dataset hashes
```

No formal artifact may use `git_sha=unknown`.

### P0-C — PatchTST naming

Current repository Transformer host may be called `PatchTST-style` during development.

Before manuscript:
- use a faithful PatchTST reproduction, OR
- keep current implementation but never label it as official/fidelity-equivalent PatchTST.

---

## 5. Dataset matrix

### 5.1 Foreign electricity-price headline panel — PRIMARY PAPER GATE

Use:

```text
LAGO_DE
LAGO_BE
LAGO_FR
LAGO_PJM
LAGO_NP
NEM_SA1
GEFCOM14P
NORD_DK1
```

Do not remove a dataset because HCH performs poorly.

### 5.2 Foreign extended robustness panel

Run after the headline panel without architecture changes:

```text
DE_EPEX
PJM_2020
EPEX_FR
EPEX_BE
EPEX_NL
NORD_FI
NORD_NO
NORD_SE3
```

Report separately because some share market or processing families with headline data.

### 5.3 Domestic private panel

Audit actual files under:

```text
data/raw/provinces/
```

Current repository includes at least:

```text
Shandong PMOS 96/full
Shandong hourly
Ningxia 24h price
Gansu 24h price
Shaanxi 24h price
Qinghai 24h price
```

Create `domestic_dataset_audit.csv` with:

```text
dataset
target columns
DA/RT availability
frequency
information cutoff
history length
missingness
negative-price rate
valid complete days
headline eligibility
```

Mandatory headline targets if semantically valid:

```text
Shandong DA
Shandong RT
```

Other provinces enter the secondary domestic table if target/cutoff semantics pass audit.

### 5.4 Generic TS / peer-fidelity panel — NOT the main HCH paper gate

Current repo includes ETTh1/2, ETTm1/2, Electricity/ECL, Solar, Weather, Exchange and Traffic backup.

Use a SMALL subset only to verify PIR / δ-Adapter fidelity.

Do NOT force the full 24-hour signed-event HCH chain onto generic long-horizon tasks merely to enlarge the paper table.

---

## 6. Host matrix

First benchmark:

```text
H1 Linear / Ridge-style
H2 MLP / DNN
H3 LSTM
H4 PatchTST-style
```

All post-hoc methods receive the exact same frozen host predictions per cell.

Defer iTransformer/TimeMixer/TSFM until this gate is understood.

---

## 7. Post-hoc comparison matrix

For every eligible `dataset × host`:

```text
B0 Host Identity
B1 ResidualL1
B2 QuantileResidual
B3 δ-Adapter
B4 PIR
B5 HCH-Universal
B6 HCH-Local
```

### B5 HCH-Universal
Main method. One shared PUBLIC universal candidate is jointly trained across approved public paper development datasets.

### B6 HCH-Local
Same architecture/capacity trained only on target-domain candidate training data.

Purpose: separate architecture quality from sharing benefit.

---

## 8. Public/private training separation

### Public benchmark universal weights

`HCH-Universal-Public` may use only public paper datasets' training/development partitions.

Do NOT mix Shandong/private provinces into public benchmark weights.

### Domestic evaluation

Report at least:

#### CN-A — Public-universal frozen candidate transfer
Public universal candidate + legal domestic local evidence.

#### CN-B — Private target-adapted setting
Optional explicitly defined few-shot/local adaptation using domestic development labels.

Label it as private/business evaluation.

---

## 9. PIR and δ-Adapter fidelity policy

Do not spend unnecessary engineering time on competitors.

> **If the current implementation reproduces the original paper's baseline→adapter improvement direction and roughly similar magnitude on a small official reference setting, accept it and freeze it.**

### 9.1 PIR

Official PIR contains:

```text
failure identification
local revision
global retrieval revision
```

Current repo implementation is explicitly limited and lacks official retrieval.

Reference smoke:
- `PatchTST × Electricity`
- `PatchTST × ETTh1`
or another exact official script that is easier to reproduce.

Use official split/horizon/metrics for this fidelity check.

Check:

```text
Host MSE / MAE
PIR MSE / MAE
relative improvement
```

If reasonably consistent -> `PIR_ACCEPT_AS_IS`.

If clearly inconsistent:
1. add official-style global retrieval index/revision;
2. rerun fidelity;
3. only add further missing local/context details if necessary.

Do not redesign PIR.

### 9.2 δ-Adapter

Current implementation is based on official PostY/Ada-Y family but modifies sample organization, normalization and training details.

Use 1–2 official reference settings.

Check mainly:

```text
MSE
MAE
relative improvement
```

If reasonably consistent -> `DELTA_ACCEPT_AS_IS`.

If clearly inconsistent, audit only:
- official output correction placement;
- normalization;
- bounded/residual form used in selected paper configuration;
- training schedule.

Implement the minimum needed to recover paper-like behavior.

Do not tune δ-Adapter specifically to HCH datasets.

---

## 10. Metrics

### 10.1 Domestic price

PRIMARY:

```text
MAE
sMAPE (standard no-floor)
```

SECONDARY:

```text
RMSE
rMAE
CRPS candidate diagnostic
negative-price MAE
negative sign-miss rate
high-tail MAE
tail bias
event metrics where valid
```

BUSINESS SUPPLEMENT:
use the previously defined DA/RT market-value proxy only where semantically valid.

Do not use floor50 sMAPE as the paper headline.

### 10.2 Foreign electricity price

PRIMARY:

```text
MAE
sMAPE
```

SECONDARY:

```text
rMAE / MASE-style relative error when valid
RMSE
CRPS diagnostic
negative-price / high-tail metrics
```

STATISTICAL TEST:
paired forecast-loss test on the same test days, at minimum day-level Diebold-Mariano-style testing on absolute-error loss.

If faithful epftoolbox multivariate DM/GW is integrated, add as supplementary evidence.

### 10.3 Generic peer-fidelity metrics

For PIR / δ-Adapter reference reproduction:

```text
MSE
MAE
```

Do not let the generic fidelity panel dictate electricity-price metrics.

---

## 11. Chronology and leakage

For HCH price experiments preserve:

```text
H0
S1R
S2T
S2V
S3M
S3C
S4
```

Roles:

```text
H0   host fitting
S1R  rank/signature reference
S2T  candidate training
S2V  checkpoint selection
S3M  local prequential evidence/calibration
S3C  DVG calibration
S4   experiment test/development confirmation
```

Important:
R1A/R1B S4 has been inspected repeatedly. Do NOT call it pristine final manuscript test.

Create/freeze a fresh Paper Development Test period where possible, and later a sealed final period after the architecture/config is frozen.

---

## 12. HCH paper-training regime

### 12.1 Universal public training

Train one shared candidate on all approved PUBLIC paper training domains.

Use hierarchical balancing:

```text
market family
  -> dataset
    -> host
      -> day batch
```

Later with DA/RT:

```text
market family
  -> target category
    -> dataset/zone
      -> host
```

Objective remains:

\[
oxed{L=L_{\mathrm{IAH-CRPS}}.}
\]

### 12.2 Checkpoint regression guards

Promote a checkpoint only if:

1. balanced macro S2V CRPS improves or remains within tiny tolerance;
2. worst-domain CRPS does not materially regress;
3. no headline dataset violates the regression budget;
4. generalization anchors do not materially regress.

Initial development guards:

```text
per-headline-domain S2V CRPS regression <= 2%
worst-domain regression <= 3%
generalization-anchor macro regression <= 2%
```

If a candidate fails a guard -> rollback.

Training itself need not be monotonic. Accepted model versions must be protected.

---

## 13. Execution order

### B0 — P0 closure
- final prediction replay;
- provenance;
- domestic audit;
- host naming;
- peer fidelity harness.

STOP if unresolved.

### B1 — Peer fidelity smoke
Output `peer_fidelity_report.md`.

Statuses:

```text
PIR_ACCEPT_AS_IS
PIR_PATCH_RETRIEVAL_REQUIRED
DELTA_ACCEPT_AS_IS
DELTA_OFFICIAL_ALIGNMENT_REQUIRED
```

### B2 — Paper smoke

Datasets:

```text
LAGO_DE
LAGO_PJM
NEM_SA1
Shandong DA (if valid)
Shandong RT (if valid)
```

Hosts:

```text
Linear
MLP
```

Methods: B0–B6.

Purpose:
- verify real final point metrics;
- expose obvious peer gaps quickly;
- identify candidate vs readout vs local-layer failure.

### B3 — Foreign headline matrix, seed 0

8 foreign headline datasets × 4 hosts × all methods.

Generate ranks for MAE and sMAPE.

### B4 — Domestic matrix, seed 0

All audit-valid domestic datasets × 4 hosts × all methods.

### B5 — Paper Performance Gate v1

STOP. Do not auto-run multi-seed.

Human/scientific review chooses:
- freeze;
- one diagnosis-driven modification;
- or core review.

### B6 — Multi-seed confirmation

Only after GREEN or a repaired strong YELLOW:

```text
HCH seeds = 0,1,2
```

Use identical host caches when isolating HCH randomness.

---

## 14. Primary cell definition

A primary cell is:

```text
dataset × host × primary metric
```

Foreign headline:

```text
8 datasets × 4 hosts × 2 metrics = 64 primary cells
```

For each cell rank Host + all post-hoc competitors.

### Strict Top-1
Lowest point estimate.

### Tied Top-1
HCH within 0.5% relative of the best point estimate AND not significantly worse under paired testing.

Freeze the 0.5% rule before results.

### Top-2
Among best two by point estimate.

---

## 15. Internal Paper Performance Gate

These are INTERNAL development thresholds. Final paper shows the full matrix.

### 15.1 Foreign

#### STRONG GREEN

```text
Top-1/tied >= 70% of 64 primary cells
Top-2 >= 90%
```

and no headline dataset systematically fails.

#### GREEN

```text
Top-1/tied >= 60%
Top-2 >= 85%
```

plus:
- improvement over raw Host in >=90% primary cells;
- each headline dataset: Top-1/tied in >=50% of its host×metric cells OR mean gap to best peer <=2%;
- no dataset >5% worse than best peer on BOTH MAE and sMAPE;
- no >15% final-MAE safety failure.

#### YELLOW

```text
30% <= Top-1/tied < 60%
```

or strong overall score with one/few dataset families failing.

#### RED

```text
Top-1/tied < 30%
```

or HCH broadly loses materially to simple/peer post-hoc methods.

**30% is never called a majority.**

### 15.2 Domestic

For Shandong DA/RT aim higher:

- across 4 hosts × {MAE,sMAPE}, HCH Top-1/tied >=75%;
- on each DA and RT target, at least one primary metric is strict-best on a majority of hosts;
- other primary metric not materially worse;
- signed-tail metrics show intended advantage;
- business supplement non-negative / improved where valid.

Other provinces development target:

```text
Top-1/tied >= 60% of valid host×primary-metric cells
```

---

## 16. Failure diagnosis tree

### Case A — CRPS strong, final MAE/sMAPE weak

Interpretation: distribution useful, point/action readout misaligned or too conservative.

Do NOT change atoms or CRPS first.

First repair:

\[
\mu_R=w^+m^+-w^-m^-,
\]

\[
oxed{z^{point}=z^0+\mu_R.}
\]

This adds no predictive parameters.

If overcorrection occurs, development-only shrinkage grid:

\[
\lambda\in\{0.25,0.5,0.75,1.0\},
\]

\[
z^{point}=z^0+\lambda\mu_R.
\]

Select only on validation/development.

Only then consider local calibration of point readout.

### Case B — Candidate CRPS weak

Repair order:
1. optimization audit;
2. d_model 64→128 controlled test;
3. history/context length;
4. LearnedSig capacity;
5. FiLM capacity/placement;
6. hierarchical sampling;
7. optional legal covariates if failure correlates with missing information.

Do not add a new loss first.

### Case C — Mixed training causes dataset regression

Interpretation: negative transfer / mixture problem.

Repair:
- hierarchical sampling;
- family-balanced updates;
- regression guards;
- temperature sampling;
- signature dropout only if shortcut diagnostics support it.

No market-ID embedding.

### Case D — Tail metrics weak, normal MAE strong

First repair exposure/sampling:
- event-stratified day sampling;
- compensating weights if needed to preserve global risk;
- keep normal-period degradation budget.

Do not automatically add `CRPS + tail loss`.

### Case E — C3/local evidence hurts

Repair local layer only:
- stricter authorization;
- identity-anchored/shrunk isotonic;
- longer local evidence horizon;
- separate point-readout calibration from safe-action calibration.

Do not retrain universal candidate to solve a local failure.

### Case F — Broad peer loss

If after reasonable A–E repairs HCH remains `<30% Top-1/tied` and candidate CRPS is broadly weak:

```text
CORE_REVIEW_REQUIRED
```

Only then consider major redesign:
- richer support than three atoms;
- continuous residual distribution;
- new candidate/action coupling;
- alternative probabilistic head.

Major redesign requires a new math document.

---

## 17. Development tuning rule

Allowed on train/validation/dev:
- LR;
- width;
- dropout;
- context;
- batch size;
- sampler;
- point-readout shrinkage;
- calibration thresholds;
- patience/epochs.

Forbidden:

```text
look at sealed test
-> tune
-> rerun sealed test
-> tune again
```

---

## 18. Required artifacts

Root:

```text
experiments/08-hch-v2/results/PAPER_GATE_<timestamp>/
```

Required:

```text
00_RUN_CONFIG.json
01_CODE_PROVENANCE.json
02_DATASET_MANIFEST.csv
03_DOMESTIC_DATA_AUDIT.csv
04_HOST_MANIFEST.csv
05_PEER_FIDELITY_REPORT.md
06_PREDICTIONS.parquet
07_METRICS_BY_CELL.csv
08_RANKS_BY_CELL.csv
09_PRIMARY_WIN_RATE.csv
10_DATASET_LEVEL_SUMMARY.csv
11_HOST_LEVEL_SUMMARY.csv
12_DM_TESTS.csv
13_HCH_CANDIDATE_DIAGNOSTICS.csv
14_HCH_ACTION_DIAGNOSTICS.csv
15_FAILURE_MAP.csv
16_GENERALIZATION_LEDGER.csv
17_PAPER_GATE_VERDICT.md
figures/
```

Preserve predictions so all metrics can be recomputed without retraining.

---

## 19. PAPER_GATE_VERDICT.md

Must include:

### A P0 status
final replay, code SHA, host naming, data audit.

### B Peer fidelity
PIR accepted/patched; δ accepted/patched; official reference used; reproduced relative improvement.

### C Foreign scorecard
Top1/tied, Top2, win by metric/host/dataset, worst gap, host-improvement rate.

### D Domestic scorecard
Especially Shandong DA/RT MAE/sMAPE, ranks, tail metrics, business supplement.

### E Failure decomposition
Each weak cell tagged:

```text
CANDIDATE
POINT_READOUT
LOCAL_CALIBRATION
DVG
HOST
```

### F Verdict — exactly one

```text
PAPER_GATE_STRONG_GREEN
PAPER_GATE_GREEN
PAPER_GATE_YELLOW_READOUT
PAPER_GATE_YELLOW_CANDIDATE
PAPER_GATE_YELLOW_MIXTURE
PAPER_GATE_YELLOW_LOCAL
PAPER_GATE_RED_CORE
```

### G Next action
Maximum 3 evidence-driven modifications. Do not auto-execute.

---

## 20. Hard stop

After B3/B4 and Paper Gate v1:

\[
oxed{	extbf{STOP}}
\]

Do not start:
- U0;
- large-scale data expansion;
- new foundation-model experiments;
- new architecture branch;
- final sealed testing.

Return results for scientific review.

---

## 21. What this experiment decides

If GREEN:
> HCH is strong enough to become the paper method. Freeze the mathematical core, allow only minor readout/training refinements, then resume broader generalization work.

If YELLOW:
> The idea is viable, but one identified layer must be improved.

If RED:
> Do not hide behind R1B generalization. Reopen the core before scaling.

This benchmark exists to replace speculation with evidence.
