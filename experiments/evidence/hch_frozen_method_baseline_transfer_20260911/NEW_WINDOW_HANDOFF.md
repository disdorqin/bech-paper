# New-window handoff — frozen HCH-S1 is not competitive with the admitted baselines

**Token:** `HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED`

Written because the protocol's B4 decision on the non-competitive branch requires
the branch to be frozen and a new-window handoff to be created. This document
records what was established and what must NOT be reopened here.

## What this stage established

The frozen method `S1 Daily-Patch GRU32 Shape + pooled exact OOF MAE alpha` was
replayed bit-exactly (B0 abs diff `0.0` against registered evidence for S1, Host,
and all 18 per-seed MatchedDirectResidual rows) and compared on the fixed six-cell
panel against Host, an internal same-information control, an audited delta-Adapter,
and the official PIR paper protocol. Result:

- B1 GANSU competitiveness: `True` — HCH is strict best on 2/2 GANSU Hosts.
- B2 international parity: `False` — 3/4 cells within 1.0% of the best admitted baseline; the band is broken by
  `LAGO_PJM/PatchTST` at `2.3564%`.
- B3 overall: `False`.

The paired block bootstrap agrees and is not the reason for the decision:
on `LAGO_PJM/PatchTST` the interval spans zero
(`-0.0112` to
`0.1691`),
while the three international cells HCH wins have intervals strictly below zero.

## The finding to carry forward

The frozen method's advantage is not uniform across Hosts: it is decisive where the
Host is weak or the residual geometry is strongly structured (GANSU both Hosts,
LAGO_DE both Hosts, LAGO_PJM/TimeMixer) and it *loses* to a plain per-cell
delta-adapter on `LAGO_PJM/PatchTST`, where the Host is already the strongest in the
panel and the delta-adapter's small learned correction helps rather than hurts.
On that cell the frozen S1 proposal is also 0.87% worse than the Host it is meant to
improve, so the loss is not merely 'the baseline is better' — the proposal's sign
selection is wrong there too. Any new framing should treat Host-strength-conditional
applicability (the method's own harm region), not raw accuracy, as the object of study.

## Prohibited here (protocol §9, no-rescue rule)

Do not, inside this branch: change S1; change alpha; add HSA/HRSI; reopen
conditional Amplitude or Verification; tune HCH or baseline thresholds or
hyperparameters; select markets/Hosts post hoc; or add a baseline because it looks
easier to beat. The negative result is evidence for a new framing, not a defect to
repair here.

## State at freeze

- panel: `GANSU_DA / LAGO_DE / LAGO_PJM` x `PatchTST / TimeMixer` (6 cells)
- seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] seeds present; PIR on its
  official single-seed paper contract (`2021`); Host deterministic
- bootstrap: 7-day blocks, `1000` replicates, seed `20260911` (robustness only)
- partitions touched: `S1`, `DIAG_FIT`, `DIAG_EVAL` only; `S3`/`S4`/`protected`/
  `final` and every other Chinese market: `0` reads

## Reproduction

```
python experiments/current/hch_frozen_method_baseline_transfer/run_transfer.py
python experiments/current/hch_frozen_method_baseline_transfer/audit_access.py
python experiments/current/hch_frozen_method_baseline_transfer/finalize.py
python experiments/current/hch_frozen_method_baseline_transfer/verify_transfer.py
```

PIR predictions are cached in `experiments/current/hch_frozen_method_baseline_transfer/_pir_cache/`
so re-runs are cheap; the verifier recomputes PIR metrics from those raw predictions.