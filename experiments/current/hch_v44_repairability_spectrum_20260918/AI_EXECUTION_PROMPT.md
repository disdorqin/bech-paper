# AI Execution Prompt — HCH v4.4 Repairability Spectrum

STATUS:        ACTIVE
STAGE:         hch_v44_repairability_spectrum_20260918
KIND:          prompt
SUPERSEDED_BY: -

Work in:
D:\作业\science\solar_leak_price_model

Before execution read, in order:
1. root AGENTS.md;
2. RESEARCH_STATE.md;
3. EXPERIMENT_LEDGER.md;
4. HANDOFF.md;
5. experiments/STAGE_INDEX.md;
6. experiments/AGENTS.md;
7. .agents/skills/solar-research-executor/SKILL.md;
8. docs/current/HCH_V44_REPAIRABILITY_SPECTRUM_DIAGNOSTIC_20260918.md;
9. this stage's PROTOCOL.md.

The controlling question is:

When O1 remains close to the Host in the weak domestic cells, is correction headroom actually small, or is material headroom present but not recoverable from the current legal Host/history evidence?

Execute only the registered diagnostic.

### R0
Reuse all 60 completed domestic O1 runs. Do not retrain them.

Reconstruct exact residual geometry on legal TRAIN/VAL rows, then compute:
- b/B/Shape oracle swaps and registered pairwise swaps;
- exact nonnegative MAE-optimal ray scale alpha* on the current O1 correction;
- per-day repairability/geometry descriptors.

### R1
Build only the fixed legal Host/history descriptors.

Run the no-fit current-Host-to-historical-residual analogue test:
- absolute Host similarity;
- relative Host-shape similarity;
- nearest legal W=7 historical day;
- compare residual geometry against current history summary and target-aware oracle-history analogue.

No learned retrieval and no distance/temperature/K search.

### R2
Run only Ridge(alpha=1.0) diagnostic probes under five leave-one-market-out folds.

No market or Host identity features.
No hyperparameter search.
Training-fold scaling only.
Report the four registered targets and the frozen O1 correction-ratio diagnostic.

### Gate
Classify all 20 cells using the registered taxonomy.
Use the fixed seven low-gain cells from PROTOCOL.md.
Evaluate future-design A-E exactly.

Regardless of outcome, STOP. Do not implement a router, similar-day module, amplitude learner, feature branch, architecture change or foreign experiment.

### Critical boundaries
- V2 TEST reads = 0.
- protected/final reads = 0.
- optimizer steps = 0.
- existing O1 checkpoints/evidence immutable.
- src/core/** and paper/** immutable.
- no hidden retry or result-conditioned descriptor addition.

### Evidence
Write only to:
experiments/evidence/hch_v44_repairability_spectrum_20260918/

Use an independent verifier that does not import the runner.

Do not update canonical state files; return results for main-window adjudication.

## Return format

At most 110 lines:
1. terminal token;
2. reuse/access/zero-fit counts;
3. one 20-cell table: O1 gain, ray/b/B/Shape headroom, diagnosis;
4. seven-low-gain-cell summary;
5. legal Host-residual analogue results;
6. LOMO probe results;
7. future-design gate A-E;
8. one concise diagnosis paragraph;
9. independent verifier token and evidence root.

At most two Markdown tables.
No next architecture proposal.
No automatic successor stage.

Terminal token:
HCH_V44_REPAIRABILITY_SPECTRUM_COMPLETE_FOR_ADJUDICATION
or
HCH_V44_REPAIRABILITY_SPECTRUM_BLOCKED_<REASON>
