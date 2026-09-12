# `src/core` reorganization inventory — 2026-09-12

**This file is a provenance record, not method code.** It documents what left
`src/core/` when the signed-mass method core was established. The method itself
is described in [`README.md`](README.md) and [`DESIGN_CONTRACT.md`](DESIGN_CONTRACT.md).

Machine-readable twin: [`core_reorganization_inventory.json`](core_reorganization_inventory.json).

- Canonical repo: `D:/作业/science/solar_leak_price_model`
- Controlling document: [`CORE_REORGANIZATION_PLAN_20260912.md`](CORE_REORGANIZATION_PLAN_20260912.md)
- Archive root: `src/archive/core_pre_extreme_repair_20260912/`
- Compatibility module: `src/legacy_core_compat.py`

## Status vocabulary

| Status | Meaning |
|---|---|
| `CURRENT_REUSE` | Belongs to the current science and stays in `src/core/`. |
| `HISTORICAL_FIDELITY` | Historical V2/V2.5 core. Not current science; preserved byte-for-byte because it is evidence-bearing. |
| `OBSOLETE_UNREFERENCED` | No caller anywhere in the repo. Retained rather than deleted — its role is not certain. |
| `UNCERTAIN` | Role cannot be established. Never deleted. |

Nothing in this inventory is `UNCERTAIN`, and no file was deleted.

## Disposition summary

All 14 files were `MOVE_TO_ARCHIVE_KEEP`. The prior `src/core/` tree was the
accepted frozen **HCH V2.5 point core**
(`learned_signature → IAH candidate → IAH-CRPS → weighted_mean readout`). It is
superseded by the signed-mass decomposition, so it was archived in full rather
than partially reused.

Archive placement is **lifecycle metadata, not a new scientific verdict**
(`src/archive/README.md`). The V2.5 science is neither withdrawn nor downgraded
by the move.

## Per-file record

Hashes are SHA-256 of the archived bytes. Originals were hashed → copied →
re-hashed → verified → only then deleted; 14/14 matched with zero mismatches.

| Old path | Role | Status | Symbols | sha256 (first 12) | Bytes |
|---|---|---|---|---|---|
| `src/core/__init__.py` | package re-exports | `HISTORICAL_FIDELITY` | 22 re-exported names | `c8e2acc1f5a7` | 1316 |
| `src/core/README.md` | package documentation | `HISTORICAL_FIDELITY` | — | `30d94eae5ecd` | 933 |
| `src/core/learned_signature/__init__.py` | learned day signature encoder | `HISTORICAL_FIDELITY` | `__all__` | `14c8e0ec4733` | 218 |
| `src/core/learned_signature/context.py` | learned day signature encoder | `HISTORICAL_FIDELITY` | `compute_domain_descriptors`, `DataSignature`, `CoreContextEncoder`, `OptionalCovariateEncoder` | `c9a9d85ed2af` | 4652 |
| `src/core/iah_candidate/__init__.py` | IAH candidate head | `HISTORICAL_FIDELITY` | `__all__` | `76fde240652a` | 72 |
| `src/core/iah_candidate/candidate.py` | IAH candidate head | `HISTORICAL_FIDELITY` | `IAHCandidateHead` | `6eea4e695482` | 6356 |
| `src/core/iah_crps/__init__.py` | IAH-CRPS training loss | `HISTORICAL_FIDELITY` | `__all__` | `2bd778e4afec` | 89 |
| `src/core/iah_crps/loss.py` | IAH-CRPS training loss | `HISTORICAL_FIDELITY` | `iah_crps_loss`, `crps_manual` | `1ee6282998cd` | 2461 |
| `src/core/weighted_mean_readout/__init__.py` | raw-space weighted-mean point readout | `HISTORICAL_FIDELITY` | — | `ce08af9ce634` | 24 |
| `src/core/weighted_mean_readout/readout.py` | raw-space weighted-mean point readout | `HISTORICAL_FIDELITY` | `CANONICAL_POINT_READOUT`, `POINT_METRIC_PRIMARY`, `weighted_median`, `atom_supports`, `weighted_median_from_atoms`, `weighted_mean_from_atoms`, `weighted_mean_from_candidate`, `weighted_median_torch`, `weighted_median_from_candidate`, `weighted_median_from_day`, `weighted_mean_from_day`, `point_metrics_from_days` | `9b6490be822b` | 8089 |
| `src/core/v25_point_runtime/__init__.py` | V2.5 point runtime orchestrator | `HISTORICAL_FIDELITY` | `__all__` | `4ee144eefda0` | 74 |
| `src/core/v25_point_runtime/runtime.py` | V2.5 point runtime orchestrator | `HISTORICAL_FIDELITY` | `HCHV25PointRuntime` | `8a46b348e970` | 5430 |
| `src/core/universal_trainer/__init__.py` | universal trainer | `OBSOLETE_UNREFERENCED` | `__all__` | `eb18873a5a1f` | 108 |
| `src/core/universal_trainer/universal.py` | universal trainer | `OBSOLETE_UNREFERENCED` | `DomainBatch`, `_eval_batch_losses`, `_eval_host_baseline`, `_collect_health`, `UniversalCoreTrainer` | `6c0c06653d8c` | 12424 |

Full hashes, import/use sites and the `belongs_to_current_science` flag are in the
JSON twin.

## Import/use sites found (repo-wide scan, 2026-09-12)

Scan excluded `.git`, `__pycache__`, `node_modules`, `.mypy_cache`,
`.pytest_cache`, `src/core/` itself, and `data/`.

### Live callers (these now need the compat bootstrap)

| Old symbol | Site |
|---|---|
| `core.v25_point_runtime` | `experiments/foundation/pilot_benchmark/runners/phase_b_runner.py:372` |
| `core.v25_point_runtime` | `experiments/foundation/tests/test_v25_point_runtime.py:19` |
| `core.iah_candidate`, `core.learned_signature`, `core.iah_crps` | `experiments/foundation/tests/test_architecture_variants.py:13` |
| `core.weighted_mean_readout` | `experiments/foundation/tests/test_v25_readout_contract.py:29,40` |

### Archived callers (left byte-identical, per `src/AGENTS.md:9`)

| Old symbol | Site |
|---|---|
| `core.iah_candidate`, `core.iah_crps`, `core.learned_signature` | `src/archive/legacy_hch_runtime/pipeline.py:20,34,36` |
| `core.weighted_mean_readout` | `src/archive/legacy_hch_runtime/pipeline.py:44` |
| `core.iah_candidate` | `src/archive/v2_5_ablation_support/architecture_variants.py:14` |
| all four | `src/archive/core_pre_extreme_repair_20260912/legacy_core/v25_point_runtime/runtime.py:16-19` |

### Known dangling importer (recorded, not fixed)

`experiments/archive/support_legacy/internal_market_adapter_20260820/20260820/internal_v2_smoke.py:26`
imports `core.iah_crps.loss` absolutely and has **no** compat bootstrap. It is
archived support code, so it was not edited. It will not resolve on a bare
`python` run without `import legacy_core_compat` first. See `ARCHIVE_MANIFEST.md`.

### Gap: the old top-level re-exports have no callers

`src/core/__init__.py` re-exported 22 names. A repo-wide scan found **zero**
callers using a bare `from core import <name>` or `import core`. Every live
caller imports a submodule path. This is why the compat shim only needs to map
submodule paths, and why no extra alias surface is required.

## What the archive does *not* change

- No historical behaviour was altered. The archive is a byte-identical copy and
  the compat shim resolves the original import paths to it.
- No evidence artifact under `experiments/evidence/**` was rewritten.
- `src/mvp/hch_minimal_repair/` was **not** archived, not modified and not
  superseded. The M0 ray-decomposition object and this signed-mass core
  coexist; see `DESIGN_CONTRACT.md`.
