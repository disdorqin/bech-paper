# China-5 Common Benchmark 70/10/20 — Joint Baseline + Alignment-Safe Method Protocol

Date: 2026-09-12  
Protocol ID: `COMMON_BENCHMARK_701020_V1`  
Status: **DESIGN FROZEN; EXECUTION REQUIRES PHASE-SPECIFIC AUTHORIZATION**

## 0. Purpose

Replace the old low-shot market comparison as the future main generic-market benchmark with one literature-standard chronological split shared by all compared methods.

This stage covers:

```text
GANSU_DA
SHANDONG_DA
SHAANXI_DA
NINGXIA_DA
QINGHAI_DA
```

×

```text
PatchTST
TimeMixer
iTransformer
LSTM
```

The stage has two hard-separated phases:

1. **PRETEST_FREEZE** — build splits, train/freeze Hosts and methods using TRAIN/VAL only, prepare result lab; TEST labels remain structurally sealed.
2. **JOINT_TEST** — only after independent review, open the exact common TEST once and evaluate all frozen coordinates.

No human-visible baseline TEST result may be used to redesign M0/M1/M2 before JOINT_TEST.

---

## 1. Split authority

Controlling split design:

`docs/current/HCH_COMMON_BENCHMARK_701020_PROTOCOL_20260912.md`

For each generic market:

```text
n_train = int(0.70*N)
n_test  = int(0.20*N)
n_val   = N - n_train - n_test
```

Chronological target-day assignment only.

Expected counts:

| Market | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| GANSU_DA | 203 | 30 | 58 |
| SHANDONG_DA | 809 | 116 | 231 |
| SHAANXI_DA | 198 | 30 | 56 |
| NINGXIA_DA | 70 | 11 | 20 |
| QINGHAI_DA | 65 | 11 | 18 |

Any mismatch is a blocker until explained by a versioned canonical-data change.

---

## 2. Test sealing

Before JOINT_TEST authorization:

- TEST target arrays may not be loaded by training/selection code;
- no TEST MAE or subset metric may be produced;
- no result summary may contain TEST performance;
- TEST target-day IDs and a target-blind identity hash may be frozen, but target values remain closed;
- source/config/checkpoint/core hashes must be frozen before TEST opens.

Required pretest token:

`CHINA5_COMMON_BENCHMARK_PRETEST_FROZEN`

JOINT_TEST must refuse to run without a later explicit authorization token.

---

## 3. Data/feature rules

- same canonical source datasets already admitted for China-5;
- no value-conditioned row deletion;
- all feature legality inherited from audited source contracts;
- Host input contract remains the accepted Host contract;
- proposed method may use only audited forecast-known features allowed at origin;
- scalers fit TRAIN only;
- q05/q95/S90/extreme thresholds fit TRAIN targets only;
- no TEST label for any scaler, threshold or selection.

---

## 4. Host rerun

All 20 Hosts must be retrained because the data budget changed.

For each Host:

- same accepted architecture/config family as the prior strict stage;
- TRAIN fit;
- VAL checkpoint selection/early stopping;
- same seed contract unless the Host authority specifies otherwise;
- no market-specific hyperparameter search;
- persist protocol-scoped checkpoint/predictions/manifests;
- old low-shot checkpoint cannot be substituted.

Host artifact identity must include `COMMON_BENCHMARK_701020_V1` and split manifest hash.

---

## 5. Baseline rerun

### Numeric offline rows

- Host
- MatchedDirectResidual
- δ-Adapter
- PIR on already-qualified Host coordinates only

All must use the same TRAIN/VAL/TEST boundary.

δ and PIR receive the ordinary TRAIN support according to their admitted official semantics; do not restrict them to an artificial 15% POST_TRAIN block.

### Online row

COSA remains `ONLINE_TTA` and later runs strictly sequentially across TEST.

### Inherited blockers

UEC-STD / OMPB / incompatible PIR Host coordinates retain their fidelity/data/compatibility blocker unless a separate authority changes it.

The new split does not reopen a blocker.

---

## 6. Alignment-safe method freeze

Current source authority:

- `docs/current/HCH_RESIDUAL_ALIGNMENT_SAFETY_DESIGN_20260912.md`
- `src/core/DESIGN_CONTRACT.md`

Before TEST opens, persist a SHA256 tree digest of active `src/core`.

The only test-facing method variants are:

### M0 — static minimal repair

\[
\hat y=\hat y^H+\alpha_0 c.
\]

Setting: `OFFLINE_STATIC_POSTHOC`.

### M1 — uniform prequential safety

\[
\lambda_d=\min(\alpha_0,\lambda_U),\qquad
\hat y_d=\hat y_d^H+\lambda_dc_d.
\]

Setting: `PREQUENTIAL_SAFETY`.

### M2 — uniform + Shape-relevant prequential safety

\[
\lambda_d=\min(\alpha_0,\lambda_U,\lambda_S),\qquad
\hat y_d=\hat y_d^H+\lambda_dc_d.
\]

Setting: `PREQUENTIAL_SAFETY`.

No additional variant, rescue, threshold or window.

`W=7` remains fixed.

---

## 7. Proposed-method TRAIN/VAL contract must be frozen in PRETEST

PRETEST must create a method-training manifest before TEST is readable.

The manifest must state exactly:

- how honest candidate/residual training pairs are produced inside TRAIN;
- candidate-generator seed(s), epoch/optimizer/loss recipe;
- train-only residual/amplitude normalization;
- how `alpha_0` is fit from VAL;
- how the final seven legal VAL records warm-start the safety bank;
- how M1/M2 update the bank after TEST target reveal.

The method-training manifest is hash-frozen before JOINT_TEST.

A method-training choice cannot be changed after seeing any common-benchmark TEST result.

---

## 8. VAL use

VAL is visible during PRETEST and is the only common selection/calibration region.

Host:

- checkpoint/early stopping.

Static baselines:

- official validation/model-selection semantics.

Our Method:

- architecture is already frozen;
- `alpha_0` may be fitted on VAL after candidate generator is frozen;
- safety warm start uses exactly the final seven honest VAL candidate/residual records;
- no M0/M1/M2 selection is performed from VAL outcome. All three proceed to JOINT_TEST as registered variants.

---

## 9. TEST execution semantics

### Offline-static rows

Predict every TEST target without using any TEST label.

### COSA

For day d:

```text
predict d
persist prediction/hash
reveal target d
update COSA
predict d+1
```

### M1/M2

For day d:

```text
candidate c_d and Shape produced
compute lambda from the seven completed bank records
persist forecast/candidate/decision
reveal target d
append realised residual to bank
advance to d+1
```

No same-day target access.

M0 does not update from TEST labels.

---

## 10. Metrics

At minimum:

- MAE
- MSE/RMSE
- Overall gain vs exact Host
- Tail-MAE using TRAIN-only q05/q95
- Upper-tail and lower-tail MAE where nonempty
- Normal-MAE / relative normal gain-harm with units clearly distinguished
- negative-price subset where nonempty
- day-level MAE distribution

For M1/M2 additionally persist:

- `alpha_0`
- `lambda_U`
- `lambda_S` (M2)
- final `lambda`
- uniform/Shape alignment support
- n_evidence (=7)
- day-level safety decision record

No empty subset becomes numeric zero.

---

## 11. Result-setting separation

Offline ranking table:

```text
Host / MDR / δ / PIR / M0
```

Online/prequential table:

```text
Host reference / COSA / M1 / M2
```

M1/M2 may not be silently declared offline winners because they consume revealed TEST history after each day.

---

## 12. New result lab

Controlling migration plan:

`docs/current/HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912.md`

Create/use:

`experiments/lab/common_benchmark_results/`

Do not mutate the old `experiments/lab/strict_results/` registry or snapshots.

No old low-shot numeric row enters the new lab.

---

## 13. Evidence roots

Use new roots only, e.g.:

```text
experiments/evidence/hch_china5_common_benchmark_701020_20260912/
```

Subdirectories should distinguish:

```text
00_protocol
01_splits
02_hosts
03_baselines
04_method_pretest
05_joint_test
06_registry_export
07_audits
```

Never overwrite old evidence.

---

## 14. Independent verification

### PRETEST verifier

Must independently confirm:

- 5 split manifests and expected counts;
- 20 Host coordinates trained only on TRAIN, selected only on VAL;
- baseline method support matches official/common semantics;
- TEST read count = 0;
- active core tree hash frozen;
- M0/M1/M2 definitions frozen;
- all pretest checkpoint/config/manifests present;
- old strict-result lab unchanged.

Required token:

`CHINA5_COMMON_BENCHMARK_PRETEST_VERIFIED`.

### JOINT_TEST verifier

Later independently recompute metrics from raw arrays, setting separation, chronology, split identity and evidence hashes.

No JOINT_TEST token is defined here until that phase is separately authorized.

---

## 15. Stop conditions

During PRETEST:

- if TEST labels are read, stop invalid;
- if split counts differ without an admitted data-version reason, stop;
- if a baseline requires a proxy not admitted by its fidelity authority, record blocker; do not improvise;
- if an old low-shot Host artifact is substituted, stop;
- if proposed-method source/config changes after the pretest freeze, JOINT_TEST must not proceed until human adjudication.

PRETEST produces no scientific TEST result and therefore no result claim.
