# Superseded artifacts — do not use as the v4.4 baseline registry

These files were produced by an earlier session on 2026-09-16 23:20 from
`experiments/current/hch_v44_baseline_comparison_freeze_20260916/AI_EXECUTION_PROMPT.md`.

They are **quarantined, not authoritative**. The controlling design document
`docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md`
supersedes that prompt:

- Design doc line 35: the correct task is "**not to retrain baselines**, but to
  independently reconstruct and freeze a V2 paper comparison registry from the
  verified raw artifacts. Any new baseline training merely to fill UEC/OMPB/PIR
  blocker cells would change method fidelity and is forbidden."
- Design doc line 144: "Expected new baseline neural fit count is `0`."
- `PROTOCOL.md` section 9: "Do not fit any model."

`AI_EXECUTION_PROMPT.md` instead ordered 50 **new** training runs and pointed at
`experiments/current/hch_v44_five_baseline_supplement_20260916/PROTOCOL.md`,
**which does not exist** anywhere under `experiments/current/`. Its target
`PARALLEL_EXECUTION_REPORT.json`, `NEW_RUN_QUEUE.csv`, `FROZEN_REUSE_INDEX.csv`,
`CELL_STATUS.csv`, `HISTORICAL_COMPLETENESS_MATRIX.csv`,
`SOURCE_AND_CONFIG_HASHES.json` and `COMPATIBILITY_ADAPTATIONS.md` therefore
describe a 100-coordinate, 50-new-fit design that was never authorized and was
not executed.

The authorized registry is the one in the parent directory
(`MATRIX_140.csv`, 140 coordinates = 90 numeric + 50 blocked, 0 new fits),
verified by `implementation/verify_registry.py` into `VERIFICATION_REPORT.json`.

These files are kept only so the superseded work is not silently destroyed.

Note also that `SOURCE_AND_CONFIG_HASHES.json` here is **not** the file the
design document requires; the design document (section 9) requires
`SOURCE_HASHES.json`, which lives in the parent directory and is produced by the
verified registry build.
