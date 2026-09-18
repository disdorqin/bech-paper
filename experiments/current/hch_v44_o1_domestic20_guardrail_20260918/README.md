# hch_v44_o1_domestic20_guardrail_20260918

Current stage: **ACTIVE / TRAIN+VAL DOMESTIC BREADTH, CONDITIONAL INTERNATIONAL GUARDRAIL / V2 TEST QUARANTINED**.

Direct children:
- `README.md` — stage index and status.
- `PROTOCOL.md` — controlling executable scientific protocol.
- `AI_EXECUTION_PROMPT.md` — local-AI launcher; inherits the protocol exactly.
- `implementation/` — stage-local runner. Inherits the closed O1 trainer
  (`probe_train.train_o1`) by import; adds no data path and no trainer.
  - `guardrail_common.py` — coordinate universe, reuse provenance pins, median-MAE-first aggregation.
  - `guardrail_runner.py` — Phase D reference check, 36-fit pool, D0-D4, access audit.
  - `run_phase_d.py` — Phase D entry point.
  - `guardrail_phase_m.py` — Phase M zero-fit shared-gradient audit.
  - `guardrail_phase_i.py` — Phase I preflight and the fits it gates.
  - `guardrail_finalize.py` — Phase M, conditional Phase I, verification, token, `RESULTS.md`.
- `verification/` — `verify.py`, the independent verifier. Imports nothing from
  `implementation/`; re-derives the universe, the gates and every aggregate from
  the protocol text and the raw artifacts.
- `build/` — scratch: run logs under `build/logs/` and the throwaway smoke fit
  under `build/smoke/`. Never evidence.

Scientific design authority:
`docs/current/HCH_V44_O1_DOMESTIC20_INTERNATIONAL_GUARDRAIL_DESIGN_20260918.md`.

Execution order is Phase D (China-5×4 Host O1 breadth) -> Phase M (zero-fit gradient audit) -> Phase I only if D passes (LAGO_DE/PJM international guardrail).

Any later-created direct child directory such as `implementation/`, `verification/` or `build/` must be added to this README in the same change.
