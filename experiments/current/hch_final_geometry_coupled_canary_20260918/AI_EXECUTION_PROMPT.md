# AI Execution Prompt — Final HCH Geometry-Coupled Domestic Canary

STATUS:        ACTIVE
STAGE:         hch_final_geometry_coupled_canary_20260918
KIND:          prompt
SUPERSEDED_BY: -

Repository:
D:\作业\science\solar_leak_price_model

This task implements code and runs the registered domestic experiment. It does not redesign the method.

Before any change, read in order:
1. root AGENTS.md;
2. RESEARCH_STATE.md;
3. EXPERIMENT_LEDGER.md;
4. HANDOFF.md;
5. docs/current/HCH_FINAL_GEOMETRY_COUPLED_METHOD_FREEZE_20260918.md;
6. docs/current/HCH_FINAL_GEOMETRY_COUPLED_IMPLEMENTATION_EXPERIMENT_PLAN_20260918.md;
7. experiments/STAGE_INDEX.md;
8. experiments/AGENTS.md;
9. .agents/skills/solar-research-executor/SKILL.md;
10. this stage PROTOCOL.md.

Treat the two final HCH design documents as immutable scientific authority.

## Phase E0 — source implementation

Implement the final method only under:
experiments/current/hch_final_geometry_coupled_canary_20260918/implementation/

Reuse verified src/core geometry/decoder/loss/scales/access utilities rather than re-deriving them.

Implement:
- deterministic W=7 geometry prior;
- one shared BiGRU32;
- day repair core;
- b/B scalar heads;
- predicted m_plus/m_minus;
- two signed-mass queries;
- one shared hourly key projection;
- masked horizon Shape softmax;
- exact decoder;
- variants DIRECT, GEOM_FLAT, GEOM_COUPLED, GEOM_COUPLED_NOHIST;
- frozen O1 staged training schedule.

Do not import or copy the v4.4 evidence-source router, coordinate embeddings, history GRU, Shape-history network, retrieval, safety gate or market expert into the new active path.

## Phase E1 — correctness gate

Implement unit/contract tests required by PROTOCOL.md.

No scientific fit until the full source/access gate passes.

Independently prove:
- no target-day leakage;
- W=7 reveal boundary;
- geometry prior correctness;
- exact decoder identities;
- exact auxiliary switch after update 400;
- A1/A2 fairness;
- no active-path router/history-neural dependencies;
- TEST target reads 0.

## Phase E2 — critical canary

Fixed cells:
GANSU_DA/TimeMixer
GANSU_DA/iTransformer
SHANDONG_DA/TimeMixer
SHANDONG_DA/iTransformer
SHAANXI_DA/TimeMixer
SHAANXI_DA/iTransformer
SHAANXI_DA/LSTM
QINGHAI_DA/PatchTST

Seeds:
7,17,37.

Reuse existing O1 read-only.

Train exactly:
- 24 GEOM_FLAT fits;
- 24 GEOM_COUPLED fits.

No scientific retry with modified settings.

Evaluate G0-G5 exactly.

## Phase E3 — paper ablations

Run only if the critical canary returns supported.

Train:
- DIRECT, 24 fits;
- GEOM_COUPLED_NOHIST, 24 fits.

These runs test paper claims. They are not rescue arms and may not replace GEOM_COUPLED.

## Phase E4 — conditional full-20 development completion

This prompt authorizes E4 only if E2 is supported.

Reuse the eight GEOM_COUPLED canary cells.

Train the remaining 12 China-5 x four-Host cells x seeds 7/17/37 = 36 new GEOM_COUPLED fits.

Produce the complete TRAIN+VAL development table and market summaries.

Do not open V2 TEST.

## Optional zero-fit work

You may in parallel prepare, but not fit, a provenance manifest for target-day renewable/load/hydro/non-market-generation/bidding-space forecasts. This optional manifest must not block or alter the core experiment and does not authorize HCH-Exog training.

## Scientific prohibitions

Do not:
- change the signed-mass query formula;
- try FiLM, attention, extra layers or another coupling after outcomes;
- tune query width;
- tune loss weights or the 400-step switch;
- add a market-specific branch;
- use market/Host ID as a feature;
- add retrieval or a learned gate;
- use foreign-market outcomes;
- read TEST targets/predictions/metrics for selection;
- edit paper/current.

## Evidence and verification

Write new evidence only under:
experiments/evidence/hch_final_geometry_coupled_canary_20260918/

Use an independent verifier that does not import the scientific runner.

If you create implementation/ or verification/ children, update the stage README in the same change.

Do not update canonical state files. Return evidence for main-window adjudication.

## Return format

At most 100 lines:
1. terminal token;
2. source/unit-test status;
3. fit/reuse/access counts;
4. one 8-cell Host/O1/GEOM_FLAT/GEOM_COUPLED table;
5. G0-G5 arithmetic;
6. parameter/latency summary plus removed active-path modules;
7. DIRECT/NOHIST and full-20 status if executed;
8. independent verifier token and evidence root;
9. exactly one scientific verdict sentence.

At most two Markdown tables.
No next-method proposal.

Terminal token must be exactly one of:
- HCH_FINAL_GC_CANARY_SUPPORTED
- HCH_FINAL_GC_CANARY_NOT_SUPPORTED
- HCH_FINAL_GC_CANARY_BLOCKED_<REASON>
