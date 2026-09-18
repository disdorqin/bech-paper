# verification/ — HCH final geometry-coupled canary checks

STATUS:        ACTIVE
STAGE:         hch_final_geometry_coupled_canary_20260918
KIND:          verification
SUPERSEDED_BY: -

Authority: `PROTOCOL.md` §8 (unit-test gate) and §14 (independent verifier).

## Direct children

| path | role | may import the implementation? |
| --- | --- | --- |
| `README.md` | this file (L1 index) | — |
| `unit_tests.py` | the correctness gate of PROTOCOL §8: geometry-prior causality and identity, A1/A2 parameter-tensor identity, A2/A3 history invariance, exact-decoder identities, near-Host start-up, gradient reachability, staged-objective boundaries, access refusal, active-path source boundary, plus a non-registered schedule smoke | yes, by design |
| `verify_canary.py` | the independent verifier of PROTOCOL §14 | **no** — it re-derives every claim from the written evidence and the frozen source and never imports `gc_runner.py`, `trainer.py` or any other runner module |

## Order of use

1. `unit_tests.py` runs first and is **gating**: `gc_runner.py tests` refuses to start a
   scientific fit when its report says `all_passed: false`. A crashing check counts as a
   failed check, and the schedule smoke is part of the verdict.
2. `verify_canary.py` runs last, over the written evidence, and its verdict is reported
   alongside — but never instead of — the runner's own gate arithmetic.

## Boundary

Neither file opens a V2 TEST table. Both inherit `recovery_common.install_access_guard()`,
which refuses at the I/O boundary the nine enumerated TEST result files and any
`result.json` / `test_predictions.npz` under a seed directory.
