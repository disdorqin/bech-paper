# AI Execution Prompt — HCH v4.4 O1 Domestic-20 + International Guardrail

STATUS:        ACTIVE
STAGE:         hch_v44_o1_domestic20_guardrail_20260918
KIND:          prompt
SUPERSEDED_BY: -

Execute the stage in:
`D:\作业\science\solar_leak_price_model`

Before doing anything:
1. read root `AGENTS.md`;
2. read `RESEARCH_STATE.md`, `EXPERIMENT_LEDGER.md`, `HANDOFF.md`;
3. read `experiments/STAGE_INDEX.md`;
4. read `experiments/AGENTS.md`;
5. read `.agents/skills/solar-research-executor/SKILL.md`;
6. read the controlling design:
   `docs/current/HCH_V44_O1_DOMESTIC20_INTERNATIONAL_GUARDRAIL_DESIGN_20260918.md`;
7. read this stage's `PROTOCOL.md`.

The protocol is authoritative. Fail closed on any ambiguity.

## Execution order

### Phase D
Complete the frozen O1 recipe on the full China-5 × 4-Host TRAIN+VAL surface.

- Reuse the 24 existing O1 runs from the eight already-complete cells by hash.
- Train exactly 36 new fits: the 12 missing domestic cells × seeds 7/17/37.
- Do not retrain any existing O1 coordinate.
- Use the identical O1 trainer/recipe:
  - full coordinate loss through optimizer step 400;
  - reconstruction MAE only from step 401 onward;
  - all other settings unchanged.
- Never read V2 TEST material.
- Build the 20-cell paper-style seed-median VAL summary and evaluate D0-D4 exactly.

If D fails, do not run international fits. Continue only to Phase M and verification, then return the domestic-not-ready token.

### Phase M
Run the zero-fit gradient-conflict audit on the selected checkpoints of the 20 domestic cells.

- No optimizer.step.
- Compute the four losses on the frozen TRAIN diagnostic subset.
- Compute gradient cosine/norm statistics only on shared parameters.
- This phase is descriptive only and cannot change any recipe, gate, checkpoint, or decision.

### Phase I
Run only if D0-D4 all pass.

First perform a zero-fit international preflight for:
- LAGO_DE/PatchTST
- LAGO_DE/TimeMixer
- LAGO_PJM/PatchTST
- LAGO_PJM/TimeMixer

Confirm exact Host/split identity, legal O1 history support and frozen admitted baseline provenance without protected/final target access. The international comparison authority is `experiments/evidence/hch_frozen_method_baseline_transfer_20260911/`, specifically `baseline_admission_matrix.csv`, `best_baseline_gap_by_cell.csv`, `metrics_by_cell.csv`, `provenance.json` and `transfer_fidelity_audit.md`; do not substitute a newer proxy or retrain a baseline.

If preflight is clean:
- train exactly 12 O1 fits, seeds 7/17/37;
- keep the China recipe bit-for-bit scientifically identical;
- compare 3-seed median O1 MAE against frozen Host and frozen best admitted offline baseline;
- evaluate I0-I4.

Do not rescue a failing foreign cell. In particular, do not alter the recipe for PJM/PatchTST.

## Critical scientific rules

- Primary domestic statistic is **median MAE across seeds first, then relative gain**. Do not replace it with median paired relative gain.
- Existing V2 TEST numbers are historical contaminated evidence and cannot be used for any current selection.
- Domestic SOTA is not claimed here. A domestic PASS only requests an untouched confirmation stage.
- International performance is a guardrail. It may be somewhat below the strongest baseline only within the already-registered 1%/2% band.
- No component-loss ablation, router deletion, architecture simplification, optimizer change, loss-weight search or switch-step search.
- Do not edit `src/core/**` or `paper/**`.

## Verification

Use an independent verifier that does not import the runner. It must independently verify:
- exact reused/new coordinate counts;
- source/checkpoint hashes;
- access counters;
- domestic cell/market aggregation;
- D0-D4 arithmetic;
- Phase-M zero-fit status and gradient-table completeness;
- if Phase I ran, exact four-cell identities, frozen baseline provenance and I0-I4 arithmetic;
- no scientific-core mutation.

Write all new evidence under:
`experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918/`.

Update the stage-local README if implementation/verification children are created.

Do not update canonical state files; return the evidence for main-window adjudication.

## Return format

Return at most 110 lines and only:
1. terminal token;
2. Phase-D counts (36 new fits expected, 24 reused runs);
3. TEST/protected/final read counts;
4. one domestic 20-cell table: cell, O1 seed-median VAL MAE, Host MAE, Host-relative gain, seeds beating Host;
5. D0-D4 plus panel median, number Host-positive, number >=2%, five market medians;
6. Phase-M gradient conflict: negative-cosine fractions for b/B/S/aux-sum, plus the strongest Host-family pattern;
7. if Phase I ran, one four-cell table: O1 seed-median MAE, Host MAE, best admitted baseline MAE, gap, Host gain, Normal harm; then I0-I4;
8. independent-verifier token and evidence root;
9. one sentence: ready/not ready for separately authorized untouched confirmation.

Line budget: items 1-3 one line each; item 4 may use up to 24 lines including header; item 5 up to 8 lines; item 6 up to 8 lines; item 7 up to 18 lines including its optional table; items 8-9 up to 3 lines each. Total hard cap: 110 lines.

At most two Markdown tables.
Do not propose a next architecture, loss, optimizer or TEST action.

Terminal token must be exactly one of:
- `HCH_V44_O1_DOMESTIC20_NOT_READY`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_BLOCKED_<REASON>`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_NOT_MET`
- `HCH_V44_O1_DOMESTIC_READY_INTERNATIONAL_GUARDRAIL_PASS`
- `HCH_V44_O1_DOM20_GUARDRAIL_BLOCKED_<REASON>`
