"""Metric contract for the China-5 post-hoc baseline panel.

Implements exactly the definitions pre-registered in
`00_protocol/PROTOCOL_FREEZE.md` sections 7.1 and 7.2, and nothing else.

Threshold policy (frozen before any result existed):  q05 / q95 / S90 are
computed once per market from HOST_TRAIN target values only, then applied
unchanged to every method, every cell and every partition.  HOST_TRAIN is
disjoint from POST_TRAIN (where baselines are fitted) and from DEV_EVAL (where
they are scored), so no threshold is fit on a day that is fitted or scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel_contract as pc


# ------------------------------------------------------------------ thresholds
def compute_thresholds(market: str, contract: pc.DayContract) -> dict:
    """Fit q05/q95/S90 on HOST_TRAIN target values only.  Never reads a closed role."""
    _, y, _, _, _, _, audit = pc.load_target_windows(market, contract, roles=("HOST_TRAIN",))
    flat = y.reshape(-1).astype(np.float64)
    spread = (y[:, :, 0].max(axis=1) - y[:, :, 0].min(axis=1)).astype(np.float64)
    return {
        "market": market,
        "dataset_id": pc.MARKETS[market]["dataset_id"],
        "fit_partition": "HOST_TRAIN",
        "n_days_fit": int(y.shape[0]),
        "n_values_fit": int(flat.size),
        "q05": float(np.percentile(flat, 5)),
        "q95": float(np.percentile(flat, 95)),
        "day_spread_p90": float(np.percentile(spread, 90)),
        "read_audit": audit,
        "production_rule": ("fixed once from HOST_TRAIN targets; applied unchanged to every "
                            "method, cell and partition; never recomputed per method"),
    }


# --------------------------------------------------------------- cell metrics
def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def cell_metrics(y_true: np.ndarray, y_pred: np.ndarray, host_pred: np.ndarray,
                 thr: dict, day_keys: np.ndarray | None = None) -> dict:
    """Metrics for one (market, host, method) cell.

    All arrays are (n_days, 24).  `host_pred` is the frozen Host's own prediction
    for the same days, used only for the harm / gain comparisons.
    """
    assert y_true.shape == y_pred.shape == host_pred.shape, "shape mismatch"
    q05, q95, s90 = thr["q05"], thr["q95"], thr["day_spread_p90"]
    e = np.abs(y_true - y_pred)
    eh = np.abs(y_true - host_pred)

    up, lo = y_true >= q95, y_true <= q05
    tail, norm = up | lo, ~(up | lo)
    neg = y_true < 0

    spread = y_true.max(axis=1) - y_true.min(axis=1)
    extreme_day = spread >= s90
    neg_day = neg.any(axis=1)
    uptail_day = up.any(axis=1)

    mae_host = _mae(y_true, host_pred)
    mae = _mae(y_true, y_pred)
    rel = (mae_host - mae) / mae_host * 100.0 if mae_host else float("nan")

    def subset(mask, arr):
        return float(arr[mask].mean()) if mask.any() else None

    m = {
        "n_days": int(y_true.shape[0]),
        "n_hours": int(y_true.size),
        "Overall_MAE": mae,
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "Host_Overall_MAE": mae_host,
        "relative_gain_vs_Host_pct": rel,
        "Tail_MAE": subset(tail, e),
        "Upper_tail_MAE": subset(up, e),
        "Lower_tail_MAE": subset(lo, e),
        "Normal_region_MAE": subset(norm, e),
        "Host_Normal_region_MAE": subset(norm, eh),
        "Normal_harm_vs_Host": (subset(norm, e) - subset(norm, eh)) if norm.any() else None,
        "n_upper_tail_hours": int(up.sum()),
        "n_lower_tail_hours": int(lo.sum()),
        "n_normal_hours": int(norm.sum()),
        "Tail_MAE_Host": subset(tail, eh),
        "Extreme_day_MAE": subset(extreme_day[:, None].repeat(24, 1), e),
        "n_extreme_days": int(extreme_day.sum()),
        "Extreme_day_MAE_Host": subset(extreme_day[:, None].repeat(24, 1), eh),
        "extreme_day_subset_status": "OK" if extreme_day.any() else "NOT_APPLICABLE_N0",
        "Upper_tail_day_MAE": subset(uptail_day[:, None].repeat(24, 1), e),
        "n_upper_tail_days": int(uptail_day.sum()),
        "upper_tail_day_subset_status": "OK" if uptail_day.any() else "NOT_APPLICABLE_N0",
    }
    if neg.any():
        m.update({
            "negative_price_subset_status": "OK",
            "n_negative_hours": int(neg.sum()),
            "Negative_price_MAE": subset(neg, e),
            "Negative_price_MAE_Host": subset(neg, eh),
            "n_negative_days": int(neg_day.sum()),
        })
    else:
        m.update({
            "negative_price_subset_status": "NOT_APPLICABLE_N0",
            "n_negative_hours": 0,
            "Negative_price_MAE": None,
            "Negative_price_MAE_Host": None,
            "n_negative_days": 0,
        })
    m["_per_day_abs_err"] = e
    m["_per_day_residual"] = y_pred - y_true
    m["_day_keys"] = day_keys
    return m


def strip_arrays(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def dump_thresholds(out_root: Path, thresholds: dict[str, dict]) -> None:
    payload = {"schema": "china5_threshold_freeze.v1", "thresholds": thresholds}
    p = out_root / "00_protocol" / "THRESHOLD_FREEZE.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
