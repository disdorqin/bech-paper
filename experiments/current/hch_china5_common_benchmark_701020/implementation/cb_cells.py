"""Common-benchmark cell assembly — the frozen baselines' own cell contract, re-roled.

`transfers.run_direct`, `transfers.run_delta` and `pir_transfer.run_pir` consume a *cell
dict* whose role labels come from the old five-role contract (`POST_TRAIN` fit block,
`DEV_EVAL` scoring block).  Under `COMMON_BENCHMARK_701020_V1` the fit block is `TRAIN`
and the first scoring block available during PRETEST is `VAL`.

Nothing about the three frozen entry points is modified here.  What this module does is
supply them the same dict shape from the protocol-scoped Host caches, with exactly two
role substitutions:

```text
TRAIN -> DIAG_FIT      (the fit block)
VAL   -> DIAG_EVAL     (the scoring block)
```

Those two names are the ones `panel.rebuild_series` and `role_of_day` understand, and
they are the *only* names that change.  The frozen baselines' own internal selection
carve (`panel.VAL_FRACTION` of the fit block) is preserved verbatim: it is part of the
adjudicated transfer semantics, so replacing it with the benchmark VAL would change the
baseline rather than re-run it.

Two things are deliberately *not* done:

* TEST is never a scoring block here.  `build_cell(market, host, ev_role=...)` refuses
  `ev_role="TEST"` while the PRETEST seal holds.
* `PROTECTED_FINAL` / `S3+S4` is never materialised — the protocol-scoped Host cache
  contains TRAIN and VAL rows only.

Re-runnability: `run_direct` / `run_delta` fit using the fit block only, so a PRETEST
run with `ev=VAL` yields the same fitted state a later JOINT_TEST run with `ev=TEST`
would.  PIR's `scale_runs` uses only `role == "train"` hours and its train windows are
enumerated from train-role day ranges, so its fitted state is likewise independent of
the evaluation block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

# Bound before any vendor snapshot can displace the top-level name `utils`.
from utils.benchmark_cache import HostPredictionCache                        # noqa: E402

for _p in (str(CB.BREADTH_IMPL), str(CB.PARENT_STAGE), str(CB.GANSU_STAGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EV = CB.EV
HOST_ROOT = EV / "02_hosts"

# The two role names the frozen helpers read.  Nothing else is substituted.
FROZEN_FIT_ROLE = "DIAG_FIT"
FROZEN_EVAL_ROLE = "DIAG_EVAL"
ROLE_SUBSTITUTION = {"TRAIN": FROZEN_FIT_ROLE, "VAL": FROZEN_EVAL_ROLE}


def host_cache_path(market: str, host_name: str) -> Path:
    return HOST_ROOT / market / host_name / "host_predictions.npz"


def load_host_cache(market: str, host_name: str) -> HostPredictionCache:
    path = host_cache_path(market, host_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{market}/{host_name}: no protocol-scoped Host cache at {path}; the old "
            f"low-shot checkpoint is not a substitute and must not be used")
    cache = HostPredictionCache.load(path)
    if cache.metadata.get("protocol_id") != CB.PROTOCOL_ID:
        raise RuntimeError(f"{market}/{host_name}: Host cache is not protocol-scoped "
                           f"({cache.metadata.get('protocol_id')!r}) — refusing to use it")
    if set(map(str, cache.segment)) - {"TRAIN", "VAL"}:
        raise RuntimeError(f"{market}/{host_name}: Host cache carries roles outside "
                           f"TRAIN/VAL; PRETEST must not hold a TEST row")
    return cache


def relabelled(cache: HostPredictionCache) -> HostPredictionCache:
    """TRAIN->DIAG_FIT / VAL->DIAG_EVAL, so the frozen helpers can read the cache."""
    seg = np.array([ROLE_SUBSTITUTION[str(s)] for s in cache.segment], dtype=object)
    return HostPredictionCache(cache.market_id, cache.physical_market_group, cache.backbone,
                               cache.timestamp, cache.context, cache.y_true, cache.host_pred,
                               seg, dict(cache.metadata))


def build_cell(market: str, host_name: str, ev_role: str = "VAL") -> dict:
    """The frozen baselines' cell contract, re-roled onto the common benchmark."""
    if ev_role == "TEST" and not CB.PROCESS_SEAL.lifted:
        raise PermissionError("build_cell: refusing a TEST scoring block while the "
                              "PRETEST seal holds")
    if ev_role not in ("VAL", "TEST"):
        raise KeyError(f"unknown evaluation role {ev_role!r}")

    from experiments.current.hch_unified_da_shape_upgrade.run_canary import make_inputs
    from experiments.current.hch_frozen_method_baseline_transfer import panel

    cache = load_host_cache(market, host_name)
    fit = cache.subset("TRAIN")
    ev = cache.subset(ev_role)
    series_cache = relabelled(cache)

    fi, floor = make_inputs(fit, cache)
    ei, _ = make_inputs(ev, cache, floor)
    scale = np.asarray(fi.scale, dtype=np.float64)
    ef = ((fit.y_true[:, :, 0] - fit.host_pred[:, :, 0])
          / np.where(np.isfinite(scale), scale, 1.0)[:, None])
    runs, span = panel.rebuild_series(series_cache)

    man = json.loads((EV / "01_splits" / market / "split_manifest.json")
                     .read_text(encoding="utf-8"))
    return {
        "protocol_id": CB.PROTOCOL_ID,
        "market": man["dataset_id"], "market_key": market, "host": host_name,
        "family": man["family"],
        "cache": cache, "cache_origin": "COMMON_BENCHMARK_NEW_EXECUTION",
        "fit": fit, "ev": ev, "fi": fi, "ei": ei, "ef": ef,
        "hp": ev.host_pred[:, :, 0].astype(np.float64),
        "scale": np.nan_to_num(np.asarray(ei.scale, dtype=np.float64), nan=0.0)[:, None],
        "avail": np.asarray(ei.repair_available),
        "n_fit": int(len(fi.x)), "n_eval": int(len(ei.x)),
        "runs": runs, "span": span, "gap_breaks": int(span["n_gap_breaks"]),
        "host_metadata": dict(cache.metadata),
        "role_substitution": dict(ROLE_SUBSTITUTION),
        "fit_role": "TRAIN", "eval_role": ev_role,
        "split_manifest_hash": man.get("split_manifest_sha256", ""),
    }


def internal_selection_carve(n_fit: int) -> tuple[np.ndarray, np.ndarray]:
    """The frozen baselines' own train/val carve inside the fit block.

    Reproduced (not re-decided) from `transfers.run_delta` and
    `pir_transfer.build_bundles`, both of which use `max(1, int(VAL_FRACTION * n_fit))`.
    Reported so the readiness matrix can state where each baseline actually selects.
    """
    from experiments.current.hch_frozen_method_baseline_transfer import panel
    n_val = max(1, int(panel.VAL_FRACTION * n_fit))
    return np.arange(0, n_fit - n_val), np.arange(n_fit - n_val, n_fit)


def cell_summary(market: str, host_name: str) -> dict:
    cell = build_cell(market, host_name, ev_role="VAL")
    tr, va = internal_selection_carve(cell["n_fit"])
    return {
        "protocol_id": CB.PROTOCOL_ID, "market": market, "host": host_name,
        "n_fit_TRAIN": cell["n_fit"], "n_eval_VAL": cell["n_eval"],
        "gap_breaks": cell["gap_breaks"], "n_runs": len(cell["runs"]),
        "baseline_internal_selection_rows": [int(len(tr)), int(len(va))],
        "scaler_fit_partition": "TRAIN",
        "baseline_selection_partition": "INTERNAL_FIT_BLOCK_CARVE(frozen VAL_FRACTION)",
        "test_used_for_fitting_or_selection": False,
    }


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", default=None)
    ns = ap.parse_args()
    markets = [ns.market] if ns.market else list(CB.MARKETS)
    for m in markets:
        for h in CB.HOSTS:
            try:
                print(json.dumps(cell_summary(m, h), ensure_ascii=False))
            except FileNotFoundError as exc:
                print(json.dumps({"market": m, "host": h, "status": "NO_HOST_CACHE",
                                  "detail": str(exc)[:120]}, ensure_ascii=False))
