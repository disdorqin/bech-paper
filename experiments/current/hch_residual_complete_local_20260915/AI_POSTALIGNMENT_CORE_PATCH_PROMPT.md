# Local-AI prompt — RCL v4.1 post-alignment core patch

Read in order:

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
5. `docs/current/HCH_RCL_CORE_ALIGNMENT_REFACTOR_PLAN_20260915.md`
6. `docs/current/HCH_RCL_CORE_POSTALIGNMENT_AUDIT_20260915.md`
7. `experiments/current/hch_residual_complete_local_20260915/PROTOCOL.md`

This task is **source-only**. RCL scientific execution is NOT authorized.

Repair every defect A–I in `HCH_RCL_CORE_POSTALIGNMENT_AUDIT_20260915.md` without changing the scientific object.

Mandatory fixes:

- make Shape output genuinely depend on legal post-Level `rho` history by feeding the Shape GRU historical latent into the Shape head;
- remove the duplicate unused `amplitude_head`; exactly one nonnegative centered-mass head may exist in active R1;
- repair forecast-feature normalization so each feature preserves temporal variation (especially F=1); never normalize across feature identity;
- make forecast-feature availability masks explicit model inputs;
- implement a true masked median and valid-mask-correct robust normalization/statistics;
- remove the accidental second division in the Shape-history ramp channel;
- make Stage-1 freeze enforcement reject any `requires_grad=True` Stage-1 parameter before Stage-2 optimizer creation;
- make prequential q-history chronology strictly increasing and fail closed on replay/nonmonotonic append;
- make masked-horizon target/fusion semantics internally consistent; prefer exact zero correction on invalid coordinates and simplex mass only over valid positions;
- update stale RCL docs/docstrings in `shape.py`, `amplitude.py`, README/contract if needed.

Add adversarial tests named/equivalent to the 11 tests listed in audit §14. In particular prove:

1. changing only legal `rho` history changes Shape output;
2. R1 has one and only one amplitude head;
3. no unintended trainable parameter is disconnected from the primary forward/loss graph;
4. a single forecast-known feature with time variation does not collapse to zero;
5. missing feature and physical zero remain distinguishable;
6. masked median ignores invalid values;
7. Local normalization stats are invariant to arbitrary invalid-coordinate values;
8. Shape ramp is scaled once;
9. Stage-1 freeze helper fails if Level parameters still require grad;
10. q-history refuses decreasing chronology;
11. partial valid masks preserve exact valid-position RCL identities.

Do not weaken or delete existing RCL tests to obtain PASS. Do not alter the old alignment-safe archive; recompute its 15/15 SHA256 map after the patch.

Forbidden:

- any scientific fit;
- any TEST/foreign/final read;
- Host/baseline retraining;
- gate/safety/retrieval/attention/TCN/Transformer/MoE/market-specific logic;
- paper edits;
- result-conditioned architecture changes.

At completion report:

- exact source files changed;
- adversarial test results;
- full active RCL/core/parity test count;
- archive hash recheck;
- scientific fit count = 0;
- TEST label read count = 0;
- terminal token `RCL_V4_1_POSTALIGNMENT_PATCH_READY_FOR_REVIEW`.

Stop there. Do not launch the RCL experiment. The experiment still requires a separate human authorization after main-window review.
