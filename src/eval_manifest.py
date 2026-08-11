"""Evaluation manifest — unified timestamp-keyed S4 evaluation (§A1-A2).

Provides a single source of truth for S4 sample alignment across methods.
All methods evaluate on the exact same (timestamp, target_id) keys.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class EvaluationManifest:
    """Unified S4 evaluation index."""

    timestamps: np.ndarray       # [N] of pd.Timestamp
    valid_indices: np.ndarray    # [N] indices into y_full (full-price array)
    seg_indices: np.ndarray      # [N] indices into y[seg["S4"]] array
    dates: list[str]             # unique date strings
    n_hours: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "valid_idx": self.valid_indices,
            "seg_idx": self.seg_indices,
        })

    @property
    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.timestamps.astype(str).tobytes())
        h.update(self.valid_indices.tobytes())
        return h.hexdigest()[:16]


def build_s4_manifest(ds: dict, seg: dict, yhat_full: np.ndarray) -> EvaluationManifest:
    """Build unified S4 manifest from daily dataloader dates.

    Only includes hours that are:
    1. Within the S4 date range
    2. Have exactly 24 hours per day (non-DST)
    3. Have valid host_pred (not NaN, past warmup)

    Returns manifest with timestamps, valid indices, and seg indices.
    """
    ts = ds["ts"]
    y_full = ds["price"]

    s4_start_idx = int(seg["S4"][0])
    s4_end_idx = int(seg["S4"][-1])

    all_dates = sorted(set(ts.dt.date))
    s4_dates = set()
    s4_ratio = 0.20
    n_dates = len(all_dates)
    s4_start_date = all_dates[int(n_dates * 0.80)]
    for d in all_dates:
        if pd.Timestamp(d) >= pd.Timestamp(s4_start_date):
            s4_dates.add(str(d))

    timestamps = []
    valid_idx_list = []
    seg_idx_list = []

    for i, t in enumerate(ts):
        d_str = str(t.date())
        if d_str not in s4_dates:
            continue
        n_hours_today = ts[ts.dt.date == t.date()].count()
        if n_hours_today != 24:
            continue
        if np.isnan(yhat_full[i]):
            continue
        timestamps.append(t)
        valid_idx_list.append(i)
        seg_idx_list.append(len(seg_idx_list))

    return EvaluationManifest(
        timestamps=np.array(timestamps),
        valid_indices=np.array(valid_idx_list, dtype=np.int64),
        seg_indices=np.array(seg_idx_list, dtype=np.int64),
        dates=list(s4_dates),
        n_hours=len(timestamps),
    )


def evaluate_on_manifest(y_true: np.ndarray, y_pred: np.ndarray,
                         manifest: EvaluationManifest,
                         neg_thr=0.0, spike_thr=None) -> dict:
    """Evaluate predictions against truth on the exact manifest hours."""
    ae = np.abs(y_pred - y_true)
    out = {
        "n": len(y_true),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((ae ** 2).mean())),
        "neg_n": int((y_true < neg_thr).sum()),
    }
    neg = y_true < neg_thr
    out["mae_on_neg"] = float(ae[neg].mean()) if neg.sum() else None
    out["neg_miss_rate"] = float((y_pred[neg] >= neg_thr).mean()) if neg.sum() else None

    if spike_thr is not None:
        sp = y_true > spike_thr
        out["spike_n"] = int(sp.sum())
        out["mae_on_spike"] = float(ae[sp].mean()) if sp.sum() else None
        out["spike_miss_rate"] = float((y_pred[sp] <= spike_thr).mean()) if sp.sum() else None
        normal = (~neg) & (~sp)
    else:
        out["spike_n"] = out["mae_on_spike"] = out["spike_miss_rate"] = None
        normal = ~neg

    out["mae_on_normal"] = float(ae[normal].mean()) if normal.sum() else None
    return out
