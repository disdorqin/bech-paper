# HCH Final Geometry-Coupled Repair Canary — Results

- Protocol: `HCH_FINAL_GEOMETRY_COUPLED_CANARY_20260918`
- Evidence root: `experiments/evidence/hch_final_geometry_coupled_canary_20260918/`
- Generated: 2026-09-18T18:20:59
- **Terminal token: `HCH_FINAL_GC_CANARY_NOT_SUPPORTED`**

## 1. Canary panel (seed-median VAL MAE, PROTOCOL §9)

| cell | Host | O1 | A1 GEOM_FLAT | A2 GEOM_COUPLED | gain vs Host | gain vs O1 | gain vs A1 |
|---|---|---|---|---|---|---|---|
| GANSU_DA__TimeMixer | 76.607 | 76.233 | 76.488 | 76.486 | 0.157% | -0.333% | 0.003% |
| GANSU_DA__iTransformer | 79.174 | 79.045 | 79.174 | 79.174 | 0.000% | -0.163% | 0.000% |
| QINGHAI_DA__PatchTST | 143.256 | 128.431 | 135.498 | 135.617 | 5.333% | -5.595% | -0.088% |
| SHAANXI_DA__LSTM | 109.456 | 109.303 | 109.342 | 109.341 | 0.105% | -0.035% | 0.000% |
| SHAANXI_DA__TimeMixer | 103.592 | 103.471 | 103.524 | 103.524 | 0.066% | -0.051% | 0.001% |
| SHAANXI_DA__iTransformer | 112.507 | 112.411 | 111.332 | 111.254 | 1.113% | 1.029% | 0.070% |
| SHANDONG_DA__TimeMixer | 78.992 | 78.492 | 77.622 | 77.655 | 1.693% | 1.067% | -0.041% |
| SHANDONG_DA__iTransformer | 75.070 | 74.850 | 72.825 | 72.841 | 2.969% | 2.683% | -0.022% |

Panel medians are computed as the median over the eight per-cell gains, not as a
median of paired seed-level gains.

## 2. Promotion gate G0–G5

### G0: PASS

```json
{
  "passed": true,
  "checks": {
    "source_audit_clean": true,
    "unit_test_gate_clean": true,
    "fits_complete_a1_24_of_24": true,
    "fits_complete_a2_24_of_24": true,
    "o1_reused_never_retrained": true,
    "o1_artifacts_unmutated": true,
    "one_scientific_code_state": true,
    "all_finite": true,
    "test_reads_zero": true
  }
}
```

### G1: PASS

```json
{
  "n_host_positive": 7,
  "n_cells": 8,
  "worst_gain_vs_host_pct": 0.0,
  "thresholds": {
    "n_host_positive": 7,
    "worst_gain_vs_host_pct": -0.5
  },
  "passed": true
}
```

### G2: FAIL

```json
{
  "n_cells_better_than_o1": 3,
  "panel_median_gain_vs_o1_pct": -0.0431532502,
  "worst_gain_vs_o1_pct": -5.5948799994,
  "thresholds": {
    "n_cells_better_than_o1": 5,
    "panel_median_gain_vs_o1_pct": 0.5,
    "worst_gain_vs_o1_pct": -1.5
  },
  "passed": false
}
```

### G3: FAIL

```json
{
  "n_cells_better_than_a1": 4,
  "panel_median_gain_vs_a1_pct": 0.000146529,
  "n_weak_cells": 6,
  "n_weak_cells_improved": 4,
  "weak_median_gain_vs_a1_pct": 0.00042289165,
  "weak_cells": [
    "GANSU_DA__TimeMixer",
    "SHAANXI_DA__LSTM",
    "SHAANXI_DA__TimeMixer",
    "SHAANXI_DA__iTransformer",
    "SHANDONG_DA__TimeMixer",
    "SHANDONG_DA__iTransformer"
  ],
  "thresholds": {
    "n_cells_better_than_a1": 6,
    "panel_median_gain_vs_a1_pct": 0.5,
    "n_weak_cells_improved": 4,
    "weak_median_gain_vs_a1_pct": 0.75
  },
  "passed": false
}
```

### G4: FAIL

```json
{
  "cell": "QINGHAI_DA__PatchTST",
  "gain_vs_o1_pct": -5.5948799994,
  "gain_vs_host_pct": 5.3327742068,
  "thresholds": {
    "min_gain_vs_o1_pct": -1.0
  },
  "passed": false
}
```

### G5: PASS

```json
{
  "learned_temporal_encoders": 1,
  "signed_mass_queries": 2,
  "source_router_calls": 0,
  "attention_modules": 0,
  "parameter_count": 10498,
  "parameter_map": "{\"day_core\": 1858, \"encoder\": 5408, \"shape_head\": 3232}",
  "latency_ms_median": 3.9077500114,
  "latency_ms_p95": 4.5443300027,
  "latency_batch": 256,
  "latency_device": "cuda",
  "ratio_to_v44_g2": 0.7023013112,
  "thresholds": {
    "learned_temporal_encoders": 1,
    "signed_mass_queries": 2,
    "source_router_calls": 0
  },
  "is_validity_reporting_gate": true,
  "passed": true
}
```

**All gates passed: False**  

## 3. Parameter count and latency

| variant | parameters | ratio to v4.4 G2 (14948) | learned temporal encoders | attention modules | signed-mass queries | source-router calls | latency median (ms) | latency p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| DIRECT | 9443 | 0.6317 | 1 | 0 | 0 | 0 | 1.036 | 2.077 |
| GEOM_FLAT | 10498 | 0.7023 | 1 | 0 | 2 | 0 | 3.079 | 6.069 |
| GEOM_COUPLED | 10498 | 0.7023 | 1 | 0 | 2 | 0 | 3.908 | 4.544 |
| GEOM_COUPLED_NOHIST | 10498 | 0.7023 | 1 | 0 | 2 | 0 | 2.738 | 5.761 |

Latency is a single-forward measurement at batch 256 on `cuda`; it is reported, not gated.

## 4. Active-path removals

- src/core/allocation.py (coordinate allocation + MaskedSoftmaxAllocator-as-router)
- src/core/encoders.py (HistoryDayEncoder, ShapeHistoryEncoder, calendar/host encoders)
- src/core/model.py (LegalEvidenceBatch, HCHV44Core, build_variants, evidence router)
- src/core/bundles.py (EvidenceSet, make_bundle)
- evidence-source routing, coordinate embeddings e_L/e_B/e_S, learned retrieval,
- safety/verification gates, market core/selector, market-specific experts,
- conditional amplitude rescue, OptionalTrajectoryEncoder

Static scan of the 8 active-path modules found 0 forbidden imports; the model class census found 0 removed classes reachable.

## 5. Access boundary

- Registered runs audited: {'n_registered': 48, 'n_fit': 47, 'n_skipped': 1, 'n_error': 0}
- Sum of per-run TEST target reads: 0
- Distinct forbidden paths blocked: []
- TEST role frames refused: 0
- V2 TEST was never opened; no TEST target, prediction or metric was read for selection.

## 6. Provenance

- O1 reused runs: 24 across 8 cells, read-only; checkpoints match their own recorded hashes: True
- One O1 code state across reused runs: True
- One source-tree digest across reused runs: True
- Prior-evidence pins unchanged at gate time: True

## 7. Disclosed caveats

- importing core.<module> executes src/core/__init__.py, which eagerly re-exports the legacy modules, so those module objects appear in sys.modules as a package-init side effect.  No active-path module imports, names or calls them: the AST scan above is the proof, and the model class census shows no removed class is reachable.
- test_rows_dropped_before_return is a per-process counter: the writer of this file reports its own process's count, not the sum over fit workers.  The authoritative evidence that no TEST row was ever materialised is that every registered run recorded test_target_read_count == 0 and test_rows_materialised == 0, and that distinct_paths_blocked is empty.
- VAL-prediction dump defect (found while verifying this stage, repaired in the same change).  The trainer wrote each run's `val_predictions.npz` from the *live* EMA rather than the selected-EMA snapshot, so for 40 of 48 runs it describes the run's last step instead of the model the freeze record describes.  No gated quantity depends on that file: G0-G5 read the freeze records, and the selection, `selected_ema.pt` and every recorded metric are computed on the selected EMA and are unaffected.  The registered MECHANISM_DIAGNOSTICS.csv is rebuilt from `val_predictions_selected_ema.npz`, a re-forward of the frozen `selected_ema.pt` that writes a *new* file and overwrites no run artifact.  The re-forward reproduces the recorded `selected_val_mae_ema` for all 48 runs (max |deviation| 1.38e-05) and reproduces the run's own npz exactly wherever that npz already was the selected pass (8/8), which is what proves it rebuilds the runs' own inputs.  No fit was retrained and no selection re-run.  Registry consequence: all 48 registered runs carry a `code_hash` for the pre-repair `trainer.py`, so their recorded hash no longer matches the file on disk.  That drift is reported in GATE.json (`runs_whose_recorded_hash_differs_from_disk`) and deliberately not gated: G0's `one_scientific_code_state` asserts homogeneity *between* runs, which still holds because all 48 share the one pre-repair hash, and a repair to a post-training code path cannot retroactively invalidate frozen fits.
- Independent verifier: INDEPENDENT_VERIFICATION_PASS (11/11 checks)

