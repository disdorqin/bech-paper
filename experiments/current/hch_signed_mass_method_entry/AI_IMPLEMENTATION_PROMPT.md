# AI EXECUTION PROMPT — Land the signed-mass core into a runnable China-5 method harness

Repository: `D:\作业\science\solar_leak_price_model`

This task is **implementation/integration only**. Do not run the scientific signed-mass experiment yet. Do not produce method-vs-baseline results. Do not read `PROTECTED_FINAL`.

The scientific core already exists under `src/core/` and has passed its source-readiness audit. **Do not rewrite or redesign the method.** Your task is to build the experiment-side adapters, chronological OOF training harness, frozen-Host prediction loader, provenance checks and integration tests required so that the next authorized stage can run one uniform signed-mass method across **5 domestic DA markets × 4 Hosts**.

## 1. Mandatory cold start

Read, in this order:

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `experiments/AGENTS.md`
5. `experiments/STAGE_INDEX.md`
6. `src/AGENTS.md`
7. `src/core/CORE_READINESS_AUDIT_20260912.md`
8. `src/core/README.md`
9. `src/core/DESIGN_CONTRACT.md`
10. `docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`
11. `docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`
12. `experiments/current/hch_china_host_breadth_expansion/PROTOCOL.md`
13. `experiments/current/hch_china_host_breadth_expansion/QUALIFICATION_AND_REPAIR_GATES.md`
14. the controlling handoff/summary files under the latest China Host×baseline evidence/registry roots.

Then inspect the actual `DATASET_CONTRACT.json` for:

- `GANSU_DA`
- `SHANDONG_DA`
- `SHAANXI_DA`
- `NINGXIA_DA`
- `QINGHAI_DA`

and inspect the frozen/imported Host prediction/checkpoint registries for:

- PatchTST
- TimeMixer
- iTransformer
- LSTM

If the parallel breadth job has not yet frozen one of the Hosts/cells, record it as `PENDING_EXTERNAL_HOST_ARTIFACT`; do not retrain it in this task.

---

## 2. Scientific authority

Do not invent a new architecture.

The current method is exactly the one implemented in `src/core/`:

\[
r_h = A^+S_h^+ - A^-S_h^-
\]

with predictions

\[
\hat c_h = \hat A^+\hat S_h^+ - \hat A^-\hat S_h^-.
\]

Pipeline:

```text
complete legal forecast-known tensor
+ frozen Host prediction
+ calendar / masks
+ revealed residual history
    -> deterministic coordinates
    -> ONE shared MLP stem
    -> Shape semantic view / Amplitude absolute view
    -> small TCN -> GRU32 per branch
    -> S+, S-, A+, A-
    -> exact signed-mass fusion
    -> chronological OOF nonnegative MAE alpha
    -> repaired Host forecast
```

First scientific experiment switches already exist and must remain exactly:

- `shape_semantic_context`
- `use_tcn`
- `untied_amplitude_heads`
- `rare_mass_sampling`

KNN is OFF.
No learned proposal selector.
No Bridge.
No attention/Transformer.
No MoE/router.
No province expert.
No market-specific model branch.

Do not add new scientific components while building the harness.

---

## 3. Purpose of this implementation stage

Create a new maintained experiment package:

```text
experiments/current/hch_signed_mass_method_entry/
```

It should make the already-existing `src/core` trainable/evaluable under the registered protocol without source edits.

The package must contain, at minimum:

```text
README.md
PROTOCOL.md
RUN_MANIFEST_SCHEMA.json
config.py
contracts.py
china5_adapter.py
host_prediction_loader.py
training.py
oof.py
calibration.py
metrics.py
runner.py
verify_preexecution.py
tests/
```

You may adjust filenames if repository conventions strongly prefer another layout, but keep the package compact.

Do NOT duplicate `src/core` scientific logic into `experiments/`.

---

## 4. China-5 generic dataset adapter

Implement one generic adapter driven by each market's audited dataset contract.

The adapter must output the `src.core.RawInputs` contract using:

- frozen Host forecast for the target day;
- the **complete admitted forecast-known feature tensor** from that market's `DATASET_CONTRACT.json`;
- matching feature-availability mask;
- complete legal calendar tensor;
- only already-revealed historical Host residual windows;
- target values only for training/evaluation loss, never as model input.

Important:

- do not hard-code province-specific scientific feature subsets;
- do not rename/interpret missing features by inventing a substitute physical role;
- preserve feature names/provenance in metadata;
- support different feature counts `F` across markets;
- use one common method recipe;
- dataset-specific code is allowed only for source schema/provenance adaptation, not model behavior.

The adapter must assert:

- target-day DA price input reads = 0;
- realized future exogenous reads = 0;
- `PROTECTED_FINAL` reads = 0;
- all legal features came from the controlling contract;
- chronological origin/history legality.

---

## 5. Frozen Host loader

Implement a single loader/registry interface for frozen Host predictions for the five markets and four Hosts:

```text
GANSU_DA
SHANDONG_DA
SHAANXI_DA
NINGXIA_DA
QINGHAI_DA

×

PatchTST
TimeMixer
iTransformer
LSTM
```

The next scientific experiment must cover all validly frozen coordinates in this 5×4 grid.

This implementation task must NOT retrain Hosts.

For each cell, resolve:

- exact Host artifact path;
- source/provenance hash if available;
- split/role coverage;
- target semantics (`DA -> DA next-24h`);
- prediction origin alignment;
- horizon length;
- availability of `POST_TRAIN` and `DEV_EVAL` predictions.

Produce a machine-readable readiness table:

`HOST_INPUT_READINESS.csv`

with one row for all 20 coordinates and statuses such as:

- `READY_FROZEN`
- `PENDING_EXTERNAL_HOST_ARTIFACT`
- `INVALID_SEMANTICS`
- `MISSING_ROLE`
- `BLOCKED_PROVENANCE`

Never silently substitute another Host or historical cache.

---

## 6. Training chronology implementation

Implement the registered chronology from:

`docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md`.

For each market×Host×seed:

1. use `POST_TRAIN` only for all fitting;
2. create chronological expanding OOF folds inside `POST_TRAIN`;
3. fit feature scalers and residual scale on each fold training prefix only;
4. train the signed-mass model on that fold;
5. produce OOF raw corrections for the fold holdout;
6. concatenate legal chronological OOF corrections;
7. fit the nonnegative pooled MAE scalar

\[
\alpha^*=\max\left(0,\operatorname{WeightedMedian}(r_i/c_i;|c_i|)\right);
\]

8. derive the final training epoch rule only from OOF/training history;
9. fit final model on all `POST_TRAIN`;
10. evaluate once on `DEV_EVAL` only when a later scientific-run command explicitly authorizes execution.

Implementation defaults must match the registered protocol unless a newer controlling file explicitly supersedes them:

- AdamW
- LR `1e-3`
- weight decay `1e-4`
- batch size `32`
- max epochs `50`
- patience `8`
- seeds `{7,17,37}`

Do not tune any of these per market/Host.

---

## 7. Rare-mass batching integration

Wire `src/core/sampling.py` into the trainer.

The full candidate uses:

`rare_mass_sampling=True`

with `spread` mode only:

- every training sample exactly once per epoch;
- no duplicate oversampling;
- strata fitted on legal training prefix only.

The registered ablation sets it to `False`.

Do not implement or activate duplicate oversampling in the first experiment.

---

## 8. Four removable-component configurations

Implement named configurations with no source edits required:

- `FULL`
- `NO_SHAPE_CONTEXT`
- `NO_TCN`
- `TIED_AMPLITUDE`
- `NO_RARE_MASS`

Their only differences must be:

```text
FULL:
  shape_semantic_context = True
  use_tcn = True
  untied_amplitude_heads = True
  rare_mass_sampling = True

NO_SHAPE_CONTEXT:
  shape_semantic_context = False

NO_TCN:
  use_tcn = False

TIED_AMPLITUDE:
  untied_amplitude_heads = False

NO_RARE_MASS:
  rare_mass_sampling = False
```

All other architecture/training/data settings must be identical.

Also support a later `FROZEN_SMALLEST` config generated only after scientific adjudication; do not choose it in this implementation task.

---

## 9. Metrics implementation

Implement metrics needed by the registered experiment, without computing final scientific results in this task:

Primary:
- Overall MAE
- MSE
- RMSE

Mechanism/region:
- Shape W1
- Shape cosine (secondary)
- positive-mass L1
- negative-mass L1
- Tail-MAE
- upper-tail MAE
- lower-tail MAE
- negative-price MAE and count where applicable
- Normal-region MAE / relative harm

All tail/normal definitions must be one globally declared rule or imported from an existing controlling China-5 contract. Do not create market-specific thresholds.

---

## 10. Run manifests and evidence layout

Prepare a future evidence root, but do not populate it with scientific results now:

```text
experiments/evidence/hch_signed_mass_method_20260912/
```

Every later run must record:

- market
- Host
- config variant
- seed
- code hash / git status snapshot
- Host artifact provenance
- dataset contract hash
- legal feature names
- scaler state hashes
- OOF fold boundaries
- best/final epoch rule
- fitted alpha
- protected read count
- output prediction path
- metrics path

Do not overwrite evidence from another run.

---

## 11. Tests required before scientific execution

This task may run synthetic/unit/integration tests and tiny non-scientific smoke paths only.

Required tests:

1. every China-5 dataset adapter reads only contract-admitted forecast-known features;
2. variable legal feature counts across markets are accepted by one common adapter/model path;
3. missing features remain explicit missingness and are never replaced with Host/another role;
4. frozen Host predictions align exactly with target origins/horizons;
5. historical residual windows are strictly revealed before the target origin;
6. all fitted statistics use fold-training prefix only;
7. OOF predictions are truly out-of-fold and chronological;
8. alpha is fitted only from OOF predictions and is nonnegative;
9. `DEV_EVAL` cannot influence training/early stopping/scaling/calibration;
10. `PROTECTED_FINAL` access count remains zero;
11. all five registered method configs execute without source edits;
12. `FULL` vs each ablation changes only the intended switch;
13. same pipeline supports all 20 5×4 readiness coordinates;
14. no Host/baseline retraining path exists in this package;
15. importing/using the experiment package does not mutate `src/core` or historical evidence.

Run the relevant existing `src/core` tests again as a regression gate.

---

## 12. What you must NOT do

Do not:

- redesign `src/core`;
- copy core model logic into the experiment folder;
- run D0 scientific diagnostics yet;
- train the method on full real POST_TRAIN for scientific evaluation;
- read `DEV_EVAL` outcomes for model choice;
- read `PROTECTED_FINAL` at all;
- retrain PatchTST/TimeMixer/iTransformer/LSTM;
- rerun admitted baselines;
- add KNN;
- add a learned proposal selector;
- add Bridge/attention/Transformer/MoE/router;
- create market-specific architecture or loss settings;
- tune hyperparameters from China outcomes;
- silently drop a market or Host because it is inconvenient.

If a Host artifact is still being produced by the parallel breadth job, leave the cell `PENDING_EXTERNAL_HOST_ARTIFACT` and make the harness able to consume it once frozen.

---

## 13. Deliverables

Create/update:

1. `experiments/current/hch_signed_mass_method_entry/README.md`
2. `experiments/current/hch_signed_mass_method_entry/PROTOCOL.md`
3. implementation files for adapter/Host loader/training/OOF/metrics/runner
4. `experiments/current/hch_signed_mass_method_entry/HOST_INPUT_READINESS.csv`
5. `experiments/current/hch_signed_mass_method_entry/PREEXECUTION_AUDIT.md`
6. tests
7. a final machine-readable readiness verdict, e.g. `PREEXECUTION_READINESS.json`

The final report must explicitly state:

- how many of 20 market×Host cells are currently `READY_FROZEN`;
- which cells are pending/blocked and why;
- test command(s) and exact PASS count;
- proof `PROTECTED_FINAL` read count is zero;
- proof no scientific signed-mass run was executed;
- proof no Host/baseline was retrained;
- exact files created/modified;
- whether the repository is ready for the next stage.

Terminal token only if all implementation/integration gates pass:

`SIGNED_MASS_CHINA5_METHOD_HARNESS_READY`

If some external Host artifacts are still pending but the harness itself is valid, use:

`SIGNED_MASS_CHINA5_METHOD_HARNESS_READY_WITH_EXTERNAL_HOSTS_PENDING`

If the harness has an implementation/legality defect, stop with:

`SIGNED_MASS_CHINA5_METHOD_HARNESS_NOT_READY`

After a READY token, do **not** automatically start the scientific experiment. Return control to the user.
