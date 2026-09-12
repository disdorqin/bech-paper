"""Stage 3 — run the audited offline baselines on the China-5 frozen Hosts.

The three runnable offline baselines are executed by **importing the already
audited transfer entry points unchanged**:

  * `MatchedDirectResidual` -> `transfers.run_direct`      (frozen internal control)
  * `delta-Adapter Ada-Y`   -> `transfers.run_delta`        (official PostProcessingNet)
  * `PIR`                   -> `pir_transfer.run_pir`       (official QualityEstimator
                                                             + official refiner + official
                                                             revision/combination path)

Nothing in those modules is edited.  The only adaptation is the *cell* dict, whose
role labels are mapped onto the vocabulary those modules already understand:

    HOST_TRAIN / HOST_VAL  ->  S1        (host-fit block, residual history)
    POST_TRAIN             ->  DIAG_FIT  (baseline fitting block)
    DEV_EVAL               ->  DIAG_EVAL (evaluation block)
    PROTECTED_FINAL        ->  S4        (still sealed; never materialised)

The mapping is a pure relabelling of the same chronological partitions; it is
recorded in EVIDENCE_PROVENANCE_MAP.csv.

Run:  python experiments/current/china5_posthoc_baseline_panel/s3_baselines.py [MARKET ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (HERE, ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import panel_contract as pc                                   # noqa: E402
from src.utils.benchmark_cache import HostPredictionCache      # noqa: E402

OUT = ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
HOSTS = ("PatchTST", "TimeMixer")
SEED = 7

LEGACY_ROLE = {"HOST_TRAIN": "S1", "HOST_VAL": "S1", "POST_TRAIN": "DIAG_FIT",
               "DEV_EVAL": "DIAG_EVAL", "PROTECTED_FINAL": "S4"}


def relabelled(cache: HostPredictionCache) -> HostPredictionCache:
    """Same arrays, legacy role vocabulary. Used only to reuse frozen helper code."""
    seg = np.array([LEGACY_ROLE[str(s)] for s in cache.segment], dtype=object)
    return HostPredictionCache(cache.market_id, cache.physical_market_group, cache.backbone,
                               cache.timestamp, cache.context, cache.y_true, cache.host_pred,
                               seg, dict(cache.metadata))


def build_cell(market: str, name: str) -> dict:
    """Assemble the cell dict the audited transfer modules consume, unchanged."""
    sys.path.insert(0, str(ROOT))
    from experiments.current.hch_unified_da_shape_upgrade.run_canary import make_inputs
    from experiments.current.hch_frozen_method_baseline_transfer import panel

    cache = HostPredictionCache.load(OUT / "02_hosts" / market / name / "host_predictions.npz")
    fit, ev = cache.subset("POST_TRAIN"), cache.subset("DEV_EVAL")
    fi, floor = make_inputs(fit, cache)
    ei, _ = make_inputs(ev, cache, floor)
    scale = np.asarray(fi.scale, dtype=np.float64)
    ef = ((fit.y_true[:, :, 0] - fit.host_pred[:, :, 0])
          / np.where(np.isfinite(scale), scale, 1.0)[:, None])
    runs, span = panel.rebuild_series(relabelled(cache))
    return {
        "market": pc.MARKETS[market]["dataset_id"], "host": name,
        "cache": cache, "fit": fit, "ev": ev, "fi": fi, "ei": ei, "ef": ef,
        "hp": ev.host_pred[:, :, 0].astype(np.float64),
        "scale": np.nan_to_num(np.asarray(ei.scale, dtype=np.float64), nan=0.0)[:, None],
        "avail": np.asarray(ei.repair_available),
        "n_fit": int(len(fi.x)), "n_eval": int(len(ei.x)),
        "runs": runs, "span": span, "gap_breaks": int(span["n_gap_breaks"]),
        "host_metadata": dict(cache.metadata),
    }


def run_one(market: str, name: str, method: str, cell: dict, seed: int = SEED) -> dict:
    from experiments.current.hch_frozen_method_baseline_transfer import transfers
    if method == "MatchedDirectResidual":
        return transfers.run_direct(cell, seed)
    if method == "delta_Adapter_AdaY":
        return transfers.run_delta(cell, seed)
    if method == "PIR_paper_protocol":
        from experiments.current.hch_frozen_method_baseline_transfer import pir_transfer
        dest = OUT / "03_baselines" / "_pir_run" / f"{market}__{name}"
        return pir_transfer.run_pir(cell, dest, seed=pir_transfer.ANCHOR_SEED)
    raise KeyError(method)


METHODS = ("MatchedDirectResidual", "delta_Adapter_AdaY", "PIR_paper_protocol")


def main(argv: list[str]) -> int:
    markets = [m for m in (argv or list(pc.MARKETS)) if m in pc.MARKETS]
    for market in markets:
        for name in HOSTS:
            cell = build_cell(market, name)
            y = cell["ev"].y_true[:, :, 0].astype(np.float64)
            hp = cell["hp"]
            # Host identity row: no fitting, no correction.
            store = OUT / "03_baselines" / market / name
            store.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(store / "Host.npz", pred=hp.astype(np.float32), y_true=y.astype(np.float32))
            print(f"[S3] {market:9s} {name:10s} Host                 "
                  f"DEV_EVAL_MAE={np.mean(np.abs(y-hp)):9.4f}  n_eval={cell['n_eval']}")
            for method in METHODS:
                t0 = time.perf_counter()
                res = run_one(market, name, method, cell)
                pred = np.asarray(res["pred"], dtype=np.float64)
                if pred.shape != y.shape:
                    raise RuntimeError(f"{market}/{name}/{method}: pred {pred.shape} != {y.shape}")
                payload = {k: v for k, v in res.items() if k != "pred"}
                np.savez_compressed(store / f"{method}.npz",
                                    pred=pred.astype(np.float32), y_true=y.astype(np.float32))
                pc.dump_json(store / f"{method}.json",
                             {"market": pc.MARKETS[market]["dataset_id"], "host": name,
                              "method": method, "seed": res.get("seed", seed_of(method)),
                              "fidelity": res.get("fidelity"),
                              "target_column": pc.MARKETS[market]["target"],
                              "eval_partition": "DEV_EVAL", "fit_partition": "POST_TRAIN",
                              "n_eval": int(cell["n_eval"]), "n_fit": int(cell["n_fit"]),
                              "gap_breaks": cell["gap_breaks"],
                              "role_relabelling": LEGACY_ROLE,
                              "wall_seconds": time.perf_counter() - t0,
                              **payload})
                print(f"[S3] {market:9s} {name:10s} {method:19s} "
                      f"DEV_EVAL_MAE={np.mean(np.abs(y-pred)):9.4f}  "
                      f"{res.get('fidelity','')}  {time.perf_counter()-t0:6.1f}s")
    return 0


def seed_of(method: str) -> int:
    return SEED


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
