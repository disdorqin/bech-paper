# AI Execution Prompt — HCH O1 Matched-Control Closure

STATUS:        ACTIVE
STAGE:         hch_v44_o1_matched_controls_20260918
KIND:          prompt
SUPERSEDED_BY: -

Repository:
D:\作业\science\solar_leak_price_model

This task does not redesign HCH. It executes already-existing v4.4 matched controls under the corrected O1 training recipe.

Read in order:
1. AGENTS.md
2. RESEARCH_STATE.md
3. EXPERIMENT_LEDGER.md
4. HANDOFF.md
5. experiments/STAGE_INDEX.md
6. experiments/AGENTS.md
7. docs/current/HCH_O1_MATCHED_CONTROL_CLOSURE_20260918.md
8. src/core/DESIGN_CONTRACT.md
9. experiments/current/hch_v44_o1_matched_controls_20260918/PROTOCOL.md
10. .agents/skills/solar-research-executor/SKILL.md

## E0 — source and reuse audit

Before fits:
- verify src/core digest equals the frozen O1/G2 reference digest;
- locate exactly 24 O1/G2 reference runs on the registered 8-cell x 3-seed panel;
- hash-persist their freeze/checkpoint identity;
- verify G1/G2 exact parameter equality using src/core.assert_variant_parity;
- verify existing G3/G2 ratio <=1.25;
- verify TEST access guards.

Do not modify src/core.

## E1 — Phase A

Train exactly 24 G1_GEOM_SHARED_CONTEXT fits using the exact O1 schedule:
- full structured objective through optimizer update 400;
- L_rec only after 400;
- AdamW 1e-3, wd 1e-4, batch 32, grad clip 1.0, EMA .995;
- max 2000 updates;
- VAL every 50;
- step 0 legal;
- no early stop before 800; then patience 8.

Same data support, scales, evidence construction and seeds as reused G2.

Compute seed-median-first cell comparisons.

Apply A1/A2/A3 exactly.

If A3 ROUTING_IDENTITY_INCONCLUSIVE, stop.

## E2 — Phase B

If Phase A selects G2:
train exactly 24 existing G3_DIRECT_CONTEXT_CTRL fits.

If Phase A selects G1:
create only a stage-local wrapper/control that instantiates the existing G3 direct model and sets tied_identity=True. It must introduce no new parameter and otherwise be byte/forward identical. Train exactly 24 of this G3_SHARED_DIRECT_CONTEXT_CTRL.

Direct control uses reconstruction-only loss throughout with the same optimizer/checkpoint budget.

Apply B1/B2 exactly.

## E3 — independent verification

Verifier must not import the runner.

Independently recompute:
- panel universe;
- reference identity;
- G1/G2 parity;
- selected direct-control identity;
- seed-median-first arithmetic;
- all gate clauses;
- TEST read count;
- no prior artifact mutation.

## Prohibitions

Do not:
- retrain G2;
- reopen signed-mass query coupling;
- add exogenous features;
- alter history window/encoders;
- change router form;
- change loss switch/lr/init;
- run G0;
- use market/Host ID;
- read V2 TEST;
- edit paper/current;
- update canonical state.

Return evidence for main-window adjudication.

## Return format

At most 100 lines:
1. terminal token;
2. source/parity/access status;
3. 24 reused G2 + new G1/direct fit counts;
4. Phase-A table and gate arithmetic;
5. Phase-B table and gate arithmetic if executed;
6. parameter parity;
7. TEST reads;
8. independent verifier token and evidence root;
9. exactly one scientific verdict sentence.

No architecture redesign and no next-step proposal.
