# HCH Final Geometry-Coupled Repair Canary Protocol

STATUS:        ACTIVE
STAGE:         hch_final_geometry_coupled_canary_20260918
KIND:          protocol
SUPERSEDED_BY: -

Scientific authority:
docs/current/HCH_FINAL_GEOMETRY_COUPLED_METHOD_FREEZE_20260918.md

Implementation authority:
docs/current/HCH_FINAL_GEOMETRY_COUPLED_IMPLEMENTATION_EXPERIMENT_PLAN_20260918.md

## 1. Scope

Implement and evaluate the frozen final HCH core on domestic TRAIN+VAL only.

Critical scientific question:

Does geometry-derived global signed correction mass improve hourly signed Shape placement beyond a flat exact-geometry decoder, while preserving or improving the already-recovered O1 domestic gains?

Foreign markets, TEST roles, exogenous-feature fitting, alternative couplings and hyperparameter search are out of scope.

## 2. Code boundary

Implement stage-local scientific code under:
experiments/current/hch_final_geometry_coupled_canary_20260918/implementation/

Independent verification under:
experiments/current/hch_final_geometry_coupled_canary_20260918/verification/

Reuse verified mathematical/source utilities from src/core as allowed by the implementation plan.

Do not mutate src/core, prior evidence, frozen O1 checkpoints or paper files in this execution stage.

## 3. Required variants

Reuse O1 read-only.

Train:
- A1 GEOM_FLAT;
- A2 GEOM_COUPLED.

Conditional after A2 promotion only:
- A0 DIRECT;
- A3 GEOM_COUPLED_NOHIST.

No fifth scientific variant.

## 4. Fixed canary panel

Markets/Hosts:
- GANSU_DA/TimeMixer
- GANSU_DA/iTransformer
- SHANDONG_DA/TimeMixer
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/iTransformer
- SHAANXI_DA/LSTM
- QINGHAI_DA/PatchTST

Seeds:
7, 17, 37.

Primary O1 values must be reused from the completed domestic O1 evidence under the exact same TRAIN+VAL role.

## 5. Training contract

A1/A2/A3:
- AdamW;
- lr 1e-3;
- weight decay 1e-4;
- batch 32;
- grad clip 1.0;
- EMA 0.995;
- max optimizer steps 2000;
- VAL every 50 steps;
- no early stop before step 800;
- then patience 8 VAL checks;
- step 0 legal;
- updates 1..400 use L_rec + (L_b+L_B+L_S)/3;
- updates 401..2000 use L_rec only.

A0 DIRECT:
- same optimizer, schedule, EMA, batch, max steps and checkpoint policy;
- L_rec only;
- near-zero direct readout.

No retry with changed settings.

## 6. Final method contract

A2 must contain:
- one shared BiGRU32;
- one day core;
- Level head;
- B scalar head;
- predicted m_plus/m_minus;
- two geometry-derived signed-mass queries;
- one shared hourly key projection;
- two masked horizon softmaxes;
- exact decoder.

A2 must not contain:
- evidence-source router;
- coordinate embeddings;
- history GRU;
- Shape-history network;
- retrieval;
- gate;
- market expert;
- market/Host ID feature.

History enters only through the deterministic W=7 geometry prior defined in the final method authority.

## 7. Access boundary

Allowed:
- TRAIN and VAL rows;
- frozen Host predictions for those roles;
- strictly revealed historical residuals needed to build the deterministic prior;
- calendar and existing deterministic Host features;
- frozen TRAIN-only scales;
- prior O1 TRAIN+VAL metrics/checkpoints for reuse identity.

Forbidden:
- V2 TEST targets/predictions/metrics;
- protected/final roles;
- target-day actual load/generation or any realized covariate;
- optional exogenous market-state trajectory fitting;
- paper-facing TEST tables for method selection.

Every data accessor must be audited.

## 8. Unit-test gate

Before scientific fits:
- geometry-prior tests pass;
- exact decoder identity tests pass;
- A1/A2 architecture matching tests pass;
- A2/A3 parameter identity tests pass;
- A3 history-mask invariance test passes;
- no forbidden active-path imports;
- near-Host step-0 audit passes;
- warm-up switch is exact after update 400;
- inference rejects target/current-residual fields;
- TEST read count is zero.

Failure blocks all scientific fits.

## 9. Primary statistic

For each method/cell:
1. median VAL MAE across seeds first;
2. then compute relative comparison.

Gain(A2 vs X) = 100 * (MAE_X - MAE_A2) / MAE_X.

Use X in {Host, O1, A1}.

Do not substitute median paired seed-level gains.

## 10. Promotion gate

A2 is supported iff all pass:

G0 validity:
- source/unit/access audit clean;
- 24/24 A1 and 24/24 A2 fits complete;
- O1 reused, never retrained;
- one scientific code state;
- all finite;
- TEST reads 0.

G1 Host safety:
- A2 Host-positive in >=7/8 cells;
- worst Host-relative cell >= -0.5%.

G2 improvement over recovered O1:
- A2 beats O1 in >=5/8 cells;
- panel median A2-vs-O1 gain >= +0.50%;
- no cell worse than O1 by >1.50%.

G3 coupling value:
- A2 beats A1 in >=6/8;
- panel median A2-vs-A1 gain >= +0.50%;
- among the six fixed Shape-limited weak cells, >=4/6 improve and median A2-vs-A1 gain >= +0.75%.

The six Shape-limited weak cells are:
- GANSU_DA/TimeMixer
- SHANDONG_DA/TimeMixer
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/iTransformer
- SHAANXI_DA/LSTM

G4 strong-control safety:
- QINGHAI_DA/PatchTST not worse than O1 by >1%.

G5 efficiency:
- exactly one learned temporal encoder;
- exactly two signed-mass queries;
- no source-router calls;
- report parameter count and latency.

G5 is a validity/reporting gate, not a performance threshold.

All G0-G5 must pass for HCH_FINAL_GC_CANARY_SUPPORTED.

## 11. Conditional ablations

Only after A2 is supported.

Run A0 DIRECT and A3 GEOM_COUPLED_NOHIST on the same eight cells x three seeds.

These are paper ablations, not rescue candidates.

Their outcome cannot replace A2.

## 12. Conditional full-20 completion

If A2 is supported, this executor may continue directly to full-20 TRAIN+VAL completion only if the launcher explicitly includes that authorization.

Reuse the eight A2 canary cells.

Train exactly 12 missing cells x three seeds = 36 new A2 fits.

No TEST opening.

## 13. Evidence

Root:
experiments/evidence/hch_final_geometry_coupled_canary_20260918/

Write:
- SOURCE_AUDIT.json
- UNIT_TEST_REPORT.json
- VARIANT_CONFIGS.json
- O1_REUSE_PROVENANCE.json
- PARAMETER_COUNTS.csv
- PER_SEED_METRICS.csv
- CELL_MEDIANS.csv
- TRAINING_CURVES.csv
- MECHANISM_DIAGNOSTICS.csv
- GATE.json
- RESULTS.md
- INDEPENDENT_VERIFICATION_REPORT.json
- STAGE_TOKEN.json

Conditional:
- ABLATION_CELL_MEDIANS.csv
- FULL20_PER_SEED.csv
- FULL20_CELL_MEDIANS.csv
- FULL20_MARKET_SUMMARY.csv
- FULL20_GATE.json

## 14. Independent verifier

The verifier must not import implementation runner modules.

It must independently verify:
- canary coordinates;
- O1 reuse hashes;
- source hashes;
- parameter counts;
- active-path forbidden-import scan;
- saved prediction decoder identities;
- seed-median-first aggregation;
- G0-G5 arithmetic;
- TEST target reads zero;
- no prior evidence/checkpoint mutation.

## 15. Terminal tokens

Exactly one critical-path token:
- HCH_FINAL_GC_CANARY_SUPPORTED
- HCH_FINAL_GC_CANARY_NOT_SUPPORTED
- HCH_FINAL_GC_CANARY_BLOCKED_<REASON>

If conditional ablations also finish, additionally record HCH_FINAL_GC_ABLATION_COMPLETE inside evidence, but the returned top-level token remains the canary token.

If authorized full-20 also finishes, evidence may contain HCH_FINAL_GC_FULL20_COMPLETE_FOR_ADJUDICATION; it does not authorize TEST.

## 16. Return format

At most 100 lines:
1. terminal token;
2. source/unit-test status;
3. A1/A2 fit counts, O1 reuse count and TEST reads;
4. one 8-cell Host/O1/A1/A2 seed-median VAL MAE table with relative gains;
5. G0-G5;
6. parameter count/latency and exact active-path deletions;
7. conditional A0/A3 or full-20 status if actually executed;
8. verifier token and evidence root;
9. one sentence limited to whether geometry-coupled global-to-local repair is supported.

At most two Markdown tables.
No architecture redesign or rescue recommendation.
