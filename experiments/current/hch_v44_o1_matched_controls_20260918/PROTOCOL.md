# HCH O1 Matched-Control Closure Protocol

STATUS:        ACTIVE
STAGE:         hch_v44_o1_matched_controls_20260918
KIND:          protocol
SUPERSEDED_BY: -

Authority:
docs/current/HCH_O1_MATCHED_CONTROL_CLOSURE_20260918.md

## 1. Scope

Run the final matched-control closure on TRAIN+VAL only.

Questions:
1. Does O1/G2 coordinate-specific identity beat parameter-identical G1 shared context?
2. Does the Phase-A-selected structured model beat a matched direct residual control?

No architecture search.

## 2. Source boundary

Read and reuse frozen src/core only.

Required source parity:
- src/core digest must match the O1/G2 reused runs;
- no src/core file may be modified;
- stage-local implementation may only contain runner/support and, if needed, a no-new-parameter G3_SHARED_DIRECT_CONTEXT_CTRL wrapper that sets tied_identity=True on the existing G3 control.

If source parity fails, block before fits.

## 3. Panel and seeds

Exactly:
- GANSU_DA/PatchTST
- GANSU_DA/LSTM
- SHANDONG_DA/PatchTST
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/PatchTST
- NINGXIA_DA/iTransformer
- QINGHAI_DA/TimeMixer

Seeds:
7,17,37.

No additional cell or seed.

## 4. Reference

Reuse existing O1/G2 runs from the completed objective-alignment/full-8 stages.

Do not retrain G2.

All 24 references must be hash-identified before any new fit.

## 5. Phase A

Train exactly 24 G1_GEOM_SHARED_CONTEXT runs.

Training:
- AdamW lr 1e-3;
- wd 1e-4;
- batch 32;
- clip 1.0;
- EMA .995;
- max 2000 updates;
- VAL every 50;
- step 0 legal;
- no early stop before 800;
- after 800 patience 8 checks;
- updates 1..400 full structured objective;
- 401+ L_rec only.

Primary statistic:
3-seed median VAL MAE per method/cell, then relative gain.

Phase-A adjudication exactly follows the design authority:
- A1 COORDINATE_IDENTITY_SUPPORTED;
- else A2 SHARED_CONTEXT_SUFFICIENT;
- else A3 ROUTING_IDENTITY_INCONCLUSIVE and stop.

Do not reinterpret thresholds.

## 6. Phase B

Only if A1 or A2 selects a structured variant.

If G2 selected:
train 24 G3_DIRECT_CONTEXT_CTRL runs.

If G1 selected:
train 24 G3_SHARED_DIRECT_CONTEXT_CTRL runs, defined only by setting tied_identity=True on the existing G3 direct-control model. No parameter or forward change beyond that identity tie.

Direct training:
- same optimizer/EMA/batch/max-step/VAL/early-stop contract;
- L_rec only throughout.

Apply B1/B2 exactly from the design authority.

## 7. Access

Allowed:
- TRAIN/VAL;
- frozen Host and legal evidence caches used by O1;
- existing O1/G2 TRAIN+VAL runs.

Forbidden:
- V2 TEST targets, predictions or metrics;
- protected/final roles;
- foreign outcomes for selection;
- target-day exogenous additions.

test_target_read_count must be 0.

## 8. Evidence and verifier

Evidence root:
experiments/evidence/hch_v44_o1_matched_controls_20260918/

Verifier must not import the scientific runner.

Check all items required by the design authority.

## 9. No rescue

If routing is inconclusive, stop.
If geometry is not supported, stop.

No new query, router, loss, input, hidden width, feature family, seed or market-specific rule.

## 10. Return format

At most 100 lines:
1. terminal token;
2. source/parity/access status;
3. fit/reuse counts;
4. Phase-A 8-cell table and exact gate arithmetic;
5. Phase-B table/gate if executed;
6. parameter parity and direct-control budget;
7. TEST reads;
8. verifier token/evidence root;
9. one scientific verdict sentence.

No redesign or next-step suggestion.
