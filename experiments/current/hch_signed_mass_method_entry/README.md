# `hch_signed_mass_method_entry` — China-5 signed-mass method harness

**Status:** integration / harness landing complete. **No scientific result has
been produced here.** The next scientific round is separately authorized and must
cover all legal 5 markets × 4 Hosts.

This package exists so that the already-frozen signed-mass object in `src/core`
can be run uniformly over 20 `market × Host` cells without anyone re-implementing
it. It is deliberately thin: it owns the *legal data path*, the *fitting
schedule* and the *provenance*, and owns no mathematics at all.

---

## Read this in order

| file | what it is |
|---|---|
| `PROTOCOL.md` | the operational contract: roles, fitting order, configs, metrics, no-go list |
| `config.py` | every constant the harness may branch on — read this first, it is the whole registry |
| `contracts.py` | the market contracts, the role vocabulary, the readiness vocabulary |
| `host_prediction_loader.py` | reads the frozen Host artifacts; produces the 20-row readiness table |
| `china5_adapter.py` | the generic dataset adapter: contracts + frozen predictions → one `CellDataset` |
| `core_bridge.py` | the single, audited import of `src/core` |
| `training.py` | folds, scalers, losses, epochs — all delegated to the core |
| `oof.py` | the chronological expanding out-of-fold schedule and the final re-fit |
| `calibration.py` | the pooled nonnegative MAE scalar (a thin wrapper, no fitting maths) |
| `metrics.py` | metric definitions plus the frozen-threshold lookup |
| `synthetic.py` | a fully synthetic dataset, for plumbing tests only |
| `runner.py` | CLI: `readiness` / `verify` / `smoke` / `run` |
| `verify_preexecution.py` | the pre-execution verifier and its audit report |
| `tests/` | the 15 required tests, the scaler-mirror witness, and the purity assertions |

## The one idea

`src/core` is the scientific object. This package's only job is to answer, for
every row it hands the core, *"is this row legal to fit on?"* — and to make that
answer checkable rather than a matter of discipline.

Concretely, the legal data path is built so that:

* `DEV_EVAL` rows are not reachable from the fitting code at all, so no
  evaluation outcome can leak into a scaler, an early-stopping decision, or α;
* `PROTECTED_FINAL` rows do not exist in any frozen artifact's segment
  vocabulary, so "do not read them" is enforced by their absence, not by a check;
* every fitted statistic is a function of the rows it was handed and nothing
  else — `tests/test_harness.py::test_06` proves this by poisoning everything
  outside the prefix and asserting the statistics are unchanged;
* the scaler's input block is mirrored, not trusted: `tests/test_scaler_prefix.py`
  asserts bit-identity with the core's own canonicalisation for all five markets.

## Why there is no Gate, router, expert or per-market model

The registered object is a *deterministic* decomposition plus one global scalar.
Every structural addition beyond it — a gate, a benefit predictor, a bridge, an
attention block, an MoE, a province expert, per-market feature selection — would
be a new scientific proposal smuggled in as infrastructure. The ban is recorded
in `config.FORBIDDEN_SOURCE_TOKENS`, enforced by AST scans in
`verify_preexecution.audit_source` and `tests/test_harness.py::test_14`, and
restated in `PROTOCOL.md` §8.

## What this package may not do

It may not train, fine-tune or re-fit a Host or a baseline; may not read
`PROTECTED_FINAL`; may not evaluate on `DEV_EVAL` in this round; may not write
into a frozen artifact root; and may not define a second implementation of any
core routine. `tests/test_purity.py` and the verifier both assert this, and
`tests/test_harness.py::test_15` asserts it *behaviourally*: a full fit on a real
cell leaves `src/core` and all twenty frozen Host artifact directories
bit-identical.

## Quick start

```bash
python experiments/current/hch_signed_mass_method_entry/runner.py readiness
python experiments/current/hch_signed_mass_method_entry/runner.py verify
python experiments/current/hch_signed_mass_method_entry/runner.py smoke
python -m pytest experiments/current/hch_signed_mass_method_entry/tests/ -q
```

`runner.py run` prints the execution plan and fits nothing without `--execute`.
See `PROTOCOL.md` §9 for the full command list and the gating rules.

## Evidence layout

Everything this package writes lands under
`experiments/evidence/hch_signed_mass_method_20260912/`:

```
00_protocol/          thresholds provenance, RUN_MANIFEST.json
01_d0_geometry/       (empty until the scientific round)
02_crossfit/          (empty until the scientific round)
03_component_screen/  ablation records
04_frozen_method/     FULL-config records
05_baseline_comparison/  (empty until the scientific round)
06_audits/            HOST_INPUT_READINESS.csv, PREEXECUTION_AUDIT.md, SMOKE_AUDIT.json
```

The upstream breadth evidence (`hch_china_host_breadth_expansion_20260912/`) is
**read-only** from here: this package consumes its `THRESHOLD_FREEZE.json` and
its frozen Host artifacts and never writes to it.
