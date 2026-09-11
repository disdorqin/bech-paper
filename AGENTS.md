# solar_leak_price_model — AI Working Instructions

This is a long-running research repository for cross-market electricity-price forecasting/post-processing, with special emphasis on extreme-price errors and model-agnostic repair.

## 1. New-window boot sequence

Do **not** recursively scan the repository.

Read in this order:

1. `RESEARCH_STATE.md` — canonical scientific state.
2. `EXPERIMENT_LEDGER.md` — experiment history and verdicts.
3. `HANDOFF.md` — latest handoff / immediate stage status.
4. the `README.md` / local `AGENTS.md` of the folder you are actually working in.
5. only the specific history/evidence files explicitly referenced by the above.

After that boot sequence, use `docs/current/README.md`, `experiments/STAGE_INDEX.md`,
and the local README/AGENTS files as the short navigation layer. Historical detail
belongs under `docs/history/`, `docs/archive/`, and `experiments/evidence/`; do not
load those trees wholesale.

Canonical state files outrank old docs, old chat memory, directory names, and historical README files.

## 2. Current mode

Current mode: **UNIFIED DA S1 SHAPE PASSED / STATE-EXCURSION SHAPE REJECTED / HORIZON-ALIGNED SEMANTIC STATE RESIDUAL CLOSURE AUTHORIZED / PAPER PREWRITING AUTHORIZED / NO MULTI-MODULE EXPANSION**.

The D0–D4 repair diagnostics are complete and adjudicated. Their scientific interpretation remains controlled by `experiments/evidence/hch_repair_diagnostics_20260908/adjudication/INDEPENDENT_DIAGNOSTIC_ADJUDICATION.md`; do not rerun them or reuse the contaminated diagnostic S3/S4 slices as untouched final truth.

The calibrated-ray proposal remains `Delta=s*alpha*u_hat`. Conditional Amplitude is removed. The Compact Verification closure is also complete and independently adjudicated at `experiments/evidence/hch_compact_verification_closure_20260909/adjudication/INDEPENDENT_COMPACT_VERIFICATION_ADJUDICATION.md`; learned Verification is removed from the leading method and must not be rescued. The current proposal is only proof-of-concept strength: six-cell relative Overall-MAE gains are about 0.11%--5.34%, so the active bottleneck is material gain rather than safety.

The Unified-DA Shape Engineering canary is complete and adjudicated. S1 Daily-Patch GRU32 Shape is the leading Shape candidate: median relative Overall-MAE gain 5.85%, 3/6 cells >=5%, maximum 10.74%, but PJM/PatchTST is a real negative cell. Shape-only conditional Amplitude and the state-intensity -> distance route are rejected. The subsequent State-Excursion Shape closure is also validly rejected: it improves all four international cells but worsens both GANSU_DA cells, so do not tune its 7-day window, MAD factor, clipping, or market-specific role subsets. The Horizon-Aligned Semantic State Residual closure (`docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`) has now been executed and is validly rejected: `HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`, with H0 and H4 passing but H1, H2 and H3 failing. The five shared horizon-local coefficients `beta` in `u_HSA=normalize(u_S1 + R beta)` genuinely help both GANSU_DA Hosts (Shape cosine `+0.0666`/`+0.1170`, Overall-MAE `+1.30`/`+4.31` pp vs S1) but do not transfer: gain vs S1 is nonnegative in only 4/6 cell medians, only 3/6 cells reach `>=5%` gain vs Host, and normal-MAE harm peaks at `1.02%`. Read `experiments/evidence/hch_horizon_aligned_state_residual_20260911/HSA_SUMMARY.md`. Do not rescue it with China-only roles, per-market role selection, per-hour/conditional `beta`, a larger step budget, or post-hoc threshold changes. There is currently no authorized successor execution; keep `S1 Daily-Patch GRU32 Shape + pooled fixed alpha` as the executable method and obtain explicit authorization before any new stage. No excursion transform, conditional Amplitude, Gate, deeper Shape network, Transformer, router, retrieval, full China panel, Shandong, threshold/hyperparameter search or protected/final evaluation.

Paper prewriting is now explicitly authorized. `paper/_drafts/HCH_FORECAST_REPAIR_PAPER_DRAFT_20260909.md` is the current first manuscript skeleton. Result statements marked `PENDING-EVIDENCE` must remain conditional until the registered experiments support them.

## 3. Research discipline

- Do not reopen a route already recorded as rejected unless there is genuinely new evidence.
- Do not silently reinterpret historical results because a file was moved or renamed.
- When a new research conclusion, supported/rejected hypothesis, major experiment, bottleneck change, route freeze, or stage transition occurs, update the canonical state files.
- The final selected method must ultimately beat the preregistered selected strong baselines on the selected datasets/headline metrics; beating Identity alone is not sufficient.
- Keep human-in-the-loop decisions explicit when they affect paper framing, information setting, dataset scope, or final method selection.

### Paper-first simplicity / anti-overdesign

- **大道至简:** prefer the smallest scientifically meaningful mechanism that can express the hypothesis. Added depth, routing, adapters, feature inventories, or interaction blocks require evidence; architectural sophistication is not itself a contribution.
- Do not pre-assign information to branches through elaborate hand-crafted feature engineering unless the scientific hypothesis requires an information boundary. Prefer a complete, legally available base tensor and let branch specialization be learned internally.
- Default branch pattern for the active HCH redesign is conceptually `z0 = Encoder(X)`, then task-specific `z_s = f_s(z0)` and `z_a = f_a(z0)`. Shape/Amplitude specialization should primarily come from their statistical targets/losses and only secondarily from minimal architectural inductive bias.
- Features are implementation/data preparation, not paper modules, unless a transformation is itself scientifically essential (for example causal normalization or scale restoration). Keep feature construction simple, transparent, and reproducible.
- A shared encoder already constitutes information sharing. Do not add a Bridge/cross-branch router merely because two branches exist; add interaction only if a controlled experiment demonstrates a mechanism-level need.
- Preserve as much aligned temporal information as practical. Do not replace rich Host/residual histories with a small set of handcrafted summaries unless an ablation shows the summaries are sufficient.
- When evaluating a proposed component, ask first: (1) what scientific object does it model, (2) can the paper explain it in one or two equations/paragraphs, (3) can another researcher implement it directly, and (4) does evidence show it is needed? If not, simplify or omit it.
- Keep reviewer-facing author material and reviewer instructions strictly separated. A method/design document intended for external review should contain only the current method itself: problem definition, legal inputs, mathematical formulation, architecture, losses, training/inference procedure, and exact reproducibility details. Do not include historical rejected alternatives, internal risk lists, suggested criticisms, reviewer questions, or arguments defending the method. Reviewer prompts should specify evaluation principles only and must not pre-seed expected flaws or conclusions.

## 4. Context discipline

Use progressive disclosure.

- Never read all of `docs/`, `experiments/`, `archive/`, or `evidence/` by default.
- Prefer a short README/index/summary first.
- Read detailed historical documents or raw machine evidence only when a current question requires them.
- Avoid creating long version chains such as `*_v0.1`, `*_v0.2`, `*_final2` when one maintained document can be edited instead.

## 5. Repository target roles

The target top-level structure is:

```text
solar_leak_price_model/
├── data/
├── docs/
├── experiments/
├── src/
├── paper/
├── RESEARCH_STATE.md
├── EXPERIMENT_LEDGER.md
├── HANDOFF.md
├── AGENTS.md
├── requirements.txt
└── .gitignore
```

### `data/`

Market-centered data storage. Target organization is market-family -> market/dataset -> metadata/source/processed files. During cleanup, prioritize readability and correct paths; do not introduce engineering-heavy migration ceremonies.

### `docs/`

AI-facing scientific memory.

Target lifecycle:

- `docs/current/` = tiny rolling synthesis only.
- `docs/history/<stage>/` = stage/version documents and compact stage README.
- `docs/archive/<stage>/` = superseded, verbose, operational, or legacy documents.

`docs/current/` must stay small. Stage documents belong in `history`, not accumulated in `current`.

### `experiments/`

Experiment code/history and machine evidence.

Target lifecycle:

- `foundation/` = long-lived benchmark/reproduction infrastructure.
- `current/` = at most one active experimental stage.
- `history/` = closed stage code and concise stage README files.
- `evidence/` = raw metrics/predictions/audits/provenance; cold storage, read mainly for audits/error investigation.
- `support/` = generic support utilities.
- `archive/` = pre-mainline/retired experiment families.

Do not treat directory presence as scientific validity; use the stage index/README and canonical state.

### `src/`

Human-facing reusable source.

Target lifecycle:

- `core/` = current scientifically accepted module code.
- `mvp/` = candidate components under validation; never silently become default core.
- `baselines/` = current paper-faithful comparison methods.
- `backbones/` = current host/backbone adapters.
- `utils/` = genuinely shared utilities.
- `archive/` = retired/replaced reusable source, organized by component.

Promotion rule: new component -> `mvp/<component>` -> experiment -> either reject to `archive/<component>` or promote/integrate into `core`. If a core component is replaced, archive the old implementation under the same component concept and update README navigation.

### `paper/`

Paper prewriting is **AUTHORIZED** as of 2026-09-09 for the current minimal Forecast Repair paper. The active manuscript skeleton is `paper/_drafts/HCH_FORECAST_REPAIR_PAPER_DRAFT_20260909.md`.

Do not promote pending claims into final results before their registered evidence exists. Historical paper assets remain non-canonical; scientific truth still comes from the root state files and controlled evidence.

## 6. Repository-reorganization execution rules

- Work one folder at a time from an explicit cleanup design/prompt.
- Do not perform broad unrelated cleanup while handling another folder.
- Runtime/cache garbage can be deleted when the folder prompt says so.
- Preserve research meaning and readable file names; this is a research workspace, not a software-engineering compliance project.
- Do not require SHA-heavy migration machinery unless a specific scientific artifact truly needs it. Practical checks such as file readability, row counts/time ranges, imports, and clear old->new notes are normally enough.
- Do not rewrite old experiment artifacts merely to make old paths look modern.
- Before moving reusable source code, inspect imports/dependencies so current execution is not broken.
- After each folder migration, return a concise before/after inventory, unresolved items, and checks performed. Wait for review before moving to the next folder.

## 7. Project instruction compatibility

The web/project-level instruction should remain consistent with this file: canonical state first, no recursive context loading, rejected routes stay closed without new evidence, folder-by-folder cleanup, and paper writing only within the scope explicitly authorized by the user.

## 8. Solar commander / executor separation

For protocolized research implementation, repair, preflight, closure, execution, or evidence tasks, use the repo-local Agent Skill:

`.agents/skills/solar-research-executor/SKILL.md`

Solar-authored controlling designs/audits define the science and phase boundary. Coding agents such as Codex/Luna are low-autonomy executors: they may implement and verify only the named phase, must fail closed on unresolved scientific/data-access ambiguity, and must stop at the registered completion/blocker token. They must not invent rescue variants, tune after results, open the next data partition, or continue into the next phase without fresh authorization.
