# EXECUTION PROMPT — China-5 Common Benchmark 70/10/20 PRETEST Freeze

Repository:

`D:\作业\science\solar_leak_price_model`

This prompt authorizes **PRETEST_FREEZE only** for the new common-benchmark protocol.

It does **not** authorize TEST evaluation.

It does **not** authorize any architecture rescue or scientific result claim.

Terminal objective:

`CHINA5_COMMON_BENCHMARK_PRETEST_VERIFIED`

---

## 1. Read first

Read in order:

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `experiments/AGENTS.md`
5. `experiments/STAGE_INDEX.md`
6. `docs/current/BASELINE_SPLIT_PROTOCOL_LITERATURE_AUDIT_20260912.md`
7. `docs/current/HCH_COMMON_BENCHMARK_701020_PROTOCOL_20260912.md`
8. `docs/current/HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912.md`
9. `docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md`
10. `src/core/DESIGN_CONTRACT.md`
11. `experiments/current/hch_china5_common_benchmark_701020/PROTOCOL.md`

Also read the frozen baseline-fidelity authority and the currently admitted Host config authorities before training anything.

Do not reinterpret an old blocker from memory.

---

## 2. Scientific decision already made

The old `~45/5/15/5/30` protocol remains valid only as:

`HARD_FROZEN_LOW_SHOT_TRANSFER`.

The new generic-market main benchmark uses:

`COMMON_BENCHMARK_701020_V1`

with chronological:

```text
TRAIN 70%
VAL   10% (remainder after integer train/test)
TEST  20%
```

All new main-comparison numeric rows must come from this new protocol.

Do not overwrite old result/evidence/lab rows.

---

## 3. Scope

Markets:

- `GANSU_DA`
- `SHANDONG_DA`
- `SHAANXI_DA`
- `NINGXIA_DA`
- `QINGHAI_DA`

Hosts:

- PatchTST
- TimeMixer
- iTransformer
- LSTM

Expected target-day counts:

```text
GANSU_DA     203 / 30 / 58
SHANDONG_DA  809 /116 /231
SHAANXI_DA   198 / 30 / 56
NINGXIA_DA    70 / 11 / 20
QINGHAI_DA    65 / 11 / 18
```

These are TRAIN / VAL / TEST.

If a count differs, stop and diagnose source/eligibility identity. Do not silently alter the split.

---

## 4. Phase P0 — Freeze split manifests before training

Build a fresh split manifest from the admitted eligible target-day sequence.

For every market persist:

- ordered eligible target-day IDs;
- TRAIN IDs;
- VAL IDs;
- TEST IDs;
- source hash;
- eligibility/exclusion contract identity;
- split manifest SHA256;
- target-blind TEST-ID hash.

TEST target values must remain structurally inaccessible during this PRETEST phase.

Implement an access audit that proves successful TEST-label read count = 0.

Do not reuse the old five-role labels as if they were the new split.

---

## 5. Phase P1 — New result-lab isolation

Create, if absent:

`experiments/lab/common_benchmark_results/`

with its own README/protocol/schema/empty registries/audit/snapshot infrastructure according to:

`docs/current/HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912.md`.

Do not modify numeric data or snapshots under:

`experiments/lab/strict_results/`.

Before and after this stage, hash the old strict-results lab tree and prove it is unchanged unless a purely top-level non-registry navigation document was explicitly authorized. Prefer zero modification.

The new lab contains no TEST numeric rows during PRETEST.

---

## 6. Phase P2 — Retrain 20 Hosts under the new split

Old low-shot Host checkpoints are protocol-mismatched and must not be reused.

For each of 20 market×Host coordinates:

- use the already-qualified architecture/config recipe;
- scaler fit on TRAIN only;
- train on TRAIN only;
- checkpoint/early-stop selection on VAL only;
- do not read TEST labels;
- persist checkpoint/config/source/split hashes;
- include `COMMON_BENCHMARK_701020_V1` in artifact identity;
- persist TRAIN/VAL predictions needed for baseline/method fitting.

Do not search architecture, hidden size, LR, epoch budget or seed per market from VAL outcome beyond the already-frozen selection rule.

Run Host sanity gates on TRAIN/VAL only.

No TEST Host MAE during PRETEST.

---

## 7. Phase P3 — Fit/freeze baseline objects on TRAIN/VAL only

Prepare every baseline that will later be evaluated on TEST.

### MDR

Fit under the new TRAIN support and freeze any VAL selection required by its existing contract.

### δ-Adapter

Use the official/common benchmark semantics:

- TRAIN loader/support for adapter fitting;
- VAL for early stopping/selection;
- no TEST labels.

Do not impose the old POST_TRAIN=15% restriction.

### PIR

Only for coordinates already admitted by current fidelity/compatibility authority.

Use:

- TRAIN support/retrieval/revision;
- VAL selection;
- no TEST labels.

Do not reopen a previously blocked Host merely because the new split is larger.

### COSA

Prepare/freeze initialization/configuration required before TEST. Do not execute TEST adaptation in PRETEST.

### UEC-STD / OMPB

Carry their current blockers. Do not retrain/proxy/reopen.

Persist a 20-cell×method readiness matrix with numeric-ready vs inherited-blocker status.

---

## 8. Phase P4 — Freeze proposed method before TEST exists

The alignment-safe active core has already passed source review.

First run the relevant active-core regression tests.

Persist:

- active `src/core` per-file SHA256;
- core tree digest;
- `HISTORY_DAYS=7` assertion;
- candidate architecture identity;
- safety formula identity;
- proof deleted components remain absent.

Then create a **TRAIN/VAL-only method-training manifest** for future M0/M1/M2.

It must freeze, before any TEST result:

- exact way candidate/residual training pairs are produced inside TRAIN;
- seeds;
- optimizer;
- epochs/early-stop rule if any;
- loss weights;
- train-only normalization;
- amplitude scale fitting;
- `alpha_0` fitting on VAL;
- exact seven-record VAL safety warm-start construction;
- M0/M1/M2 definitions;
- TEST prequential bank-update rule for M1/M2.

Important: if the existing experiment harness cannot support the new split or honest TRAIN support without changing scientific semantics, stop with a concrete blocker and document it. Do not improvise using old POST_TRAIN roles.

No M0/M1/M2 TEST result is permitted.

TRAIN/VAL fitting of the proposed method may be prepared/run in PRETEST only if its complete training semantics are already frozen by the protocol and it cannot access TEST.

---

## 9. Phase P5 — Thresholds/metric contract

Freeze from TRAIN only:

- q05
- q95
- S90 / extreme-day threshold under the admitted common definition

Persist exact source row/day IDs and hashes.

No TEST subset count or metric may be reported yet if obtaining it requires reading TEST targets.

Freeze the metric code and units before TEST opens.

Normal-region harm fields must clearly state whether they are absolute units or relative percentages; do not repeat the historical interpretation bug.

---

## 10. Phase P6 — Independent PRETEST verifier

Build an independent verifier that does not import the production runner.

It must check at least:

1. protocol ID exact;
2. five split manifests and expected counts;
3. chronological order;
4. source/eligibility hashes;
5. TEST-label read count = 0;
6. 20 protocol-scoped Host checkpoints/configs ready;
7. no old low-shot checkpoint substitution;
8. Host scaler uses TRAIN only;
9. Host selection uses VAL only;
10. baseline fit semantics match current admitted contracts;
11. blocker coordinates have no proxy numeric object;
12. active core tree hash frozen;
13. M0/M1/M2 definitions frozen;
14. W=7 fixed;
15. method-training manifest frozen;
16. thresholds TRAIN-only;
17. old `experiments/lab/strict_results/` tree unchanged;
18. new common-benchmark lab has no TEST numeric rows;
19. no scientific TEST metric exists anywhere in this stage;
20. no protected/old final role was repurposed as TEST.

Required token:

`CHINA5_COMMON_BENCHMARK_PRETEST_VERIFIED`

If anything fails:

`CHINA5_COMMON_BENCHMARK_PRETEST_NOT_READY`

---

## 11. Forbidden in this phase

Do NOT:

- compute TEST MAE for Host or any baseline;
- run COSA across TEST;
- run M0/M1/M2 on TEST;
- expose TEST targets to any fitting/selection code;
- reuse old low-shot predictions as new benchmark outputs;
- write new numeric rows into `experiments/lab/strict_results/`;
- delete old evidence;
- change UEC/OMPB/PIR compatibility blockers without separate authority;
- rescue/tune the alignment-safe core;
- change W=7;
- add a gate/TCN/KNN/untied amplitude/daily-bias branch;
- read any previously sealed protected-final partition for convenience.

---

## 12. Final PRETEST report

Report:

- exact split counts/hashes;
- 20 Host readiness status;
- baseline readiness matrix;
- inherited blockers;
- core tree digest;
- method-training manifest hash;
- TEST-label read count;
- old strict-results lab before/after hash;
- new common-benchmark lab scaffold status;
- exact test PASS counts;
- all implementation repairs, if any, with before/after hashes;
- confirmation that no TEST metric exists.

Stop after the PRETEST token.

Do not automatically proceed to JOINT_TEST.
