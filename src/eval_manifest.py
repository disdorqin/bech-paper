"""Experiment manifest — unified date-first split + timestamp-keyed evaluation.

Per addendum §3.2: dates split first, then valid/warmup/availability mapped in.
Single source of truth for host_cache, hch_v2_data, smoke_v2, and all evaluations.

ExperimentManifest (§3.2):
    dataset_id, split_of_date, raw_to_date, valid_indices, valid_to_raw, split_hash.
    Replaces both four_segment_split() and date_based_split().

EvaluationManifest (§A1-A2):
    S4 hours only: timestamps, valid_indices (full-price array), n_hours, hash.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ExperimentManifest:
    """Unified date-first split authority. All consumers read splits from here.

    Attributes:
        dataset_id: stable dataset key
        timestamps: all raw timestamps [N_total]
        dates: unique calendar dates (sorted str)
        raw_to_date: [N_total] index into self.dates
        split_of_date: dict date_str -> "S1"|"S2"|"S3"|"S4"
        valid_indices: valid-hour raw indices [M] (past warmup, not NaN)
        valid_to_raw: [M] raw index for each valid hour
        split_hash: deterministic hash of the split boundaries
    """

    dataset_id: str
    timestamps: np.ndarray
    dates: list[str]
    raw_to_date: np.ndarray
    split_of_date: dict
    valid_indices: np.ndarray
    valid_to_raw: np.ndarray
    split_hash: str = ""

    @staticmethod
    def from_dataset(ds: dict, valid: np.ndarray, dataset_id: str = "",
                     frac=(0.50, 0.20, 0.10, 0.20)) -> "ExperimentManifest":
        """Build from raw dataset dict + valid index array.

        1. Extract all unique calendar dates from ts.
        2. Exclude 23/25h dates.
        3. Split complete dates S1/S2/S3/S4 chronologically.
        4. Map each raw index to its date and split.
        5. Compute split hash.
        """
        ts = ds["ts"]
        n_total = len(ts)

        # Step 1-2: unique dates, exclude non-24h
        all_dates = []
        for d in pd.unique(ts.dt.date):
            n_h = (ts.dt.date == d).sum()
            if n_h == 24:
                all_dates.append(d)
        all_dates = sorted(all_dates)
        date_strs = [str(d) for d in all_dates]
        n_dates = len(date_strs)

        # Step 3: split dates chronologically
        a = int(n_dates * frac[0])
        b = a + int(n_dates * frac[1])
        c = b + int(n_dates * frac[2])
        split_of_date = {}
        for i, d in enumerate(date_strs):
            if i < a:
                split_of_date[d] = "S1"
            elif i < b:
                split_of_date[d] = "S2"
            elif i < c:
                split_of_date[d] = "S3"
            else:
                split_of_date[d] = "S4"

        # Step 4: map raw indices to dates
        raw_to_date = np.full(n_total, -1, dtype=np.int32)
        date_to_idx = {d: i for i, d in enumerate(date_strs)}
        for i in range(n_total):
            d_str = str(ts.iloc[i].date())
            raw_to_date[i] = date_to_idx.get(d_str, -1)

        # Step 5: split hash
        h = hashlib.sha256()
        h.update(dataset_id.encode())
        h.update(json.dumps({d: split_of_date.get(d, "?") for d in date_strs}).encode())
        split_hash_val = h.hexdigest()[:16]

        # valid mapping (for host_cache compatibility)
        valid_arr = np.asarray(valid, dtype=np.int64)
        valid_indices_arr = np.where(valid_arr)[0].astype(np.int64)

        return ExperimentManifest(
            dataset_id=dataset_id,
            timestamps=ts.values if hasattr(ts, "values") else np.array(ts),
            dates=date_strs,
            raw_to_date=raw_to_date,
            split_of_date=split_of_date,
            valid_indices=valid_indices_arr,
            valid_to_raw=valid_indices_arr,
            split_hash=split_hash_val,
        )

    def date_is(self, date_str: str, split: str) -> bool:
        return self.split_of_date.get(date_str) == split

    def raw_idx_is(self, raw_idx: int, split: str) -> bool:
        if raw_idx < 0 or raw_idx >= len(self.raw_to_date):
            return False
        di = self.raw_to_date[raw_idx]
        if di < 0 or di >= len(self.dates):
            return False
        return self.split_of_date.get(self.dates[di]) == split

    def valid_indices_in_split(self, split: str) -> np.ndarray:
        """Valid-hour raw indices belonging to the given split."""
        mask = np.array([self.raw_idx_is(vi, split) for vi in self.valid_indices])
        return self.valid_indices[mask]

    def dates_in_split(self, split: str) -> set:
        return {d for d, s in self.split_of_date.items() if s == split}

    def build_s4_eval_manifest(self, yhat_full: np.ndarray) -> EvaluationManifest:
        """Build S4 evaluation manifest from this split authority.

        Only includes hours that:
        1. Are in S4 dates
        2. Have exactly 24 hours on that date (non-DST)
        3. Have valid host_pred (not NaN)
        """
        n_total = len(self.timestamps)
        s4_dates = self.dates_in_split("S4")

        # Precompute date string + hour count per row
        date_strs = [str(pd.Timestamp(self.timestamps[i]).date()) for i in range(n_total)]
        hour_counts = {}
        for d in date_strs:
            hour_counts[d] = hour_counts.get(d, 0) + 1

        timestamps_list = []
        valid_idx_list = []
        seg_idx_list = []

        for i in range(n_total):
            d_str = date_strs[i]
            if d_str not in s4_dates:
                continue
            if hour_counts.get(d_str, 0) != 24:
                continue
            if np.isnan(yhat_full[i]):
                continue
            timestamps_list.append(self.timestamps[i])
            valid_idx_list.append(i)
            seg_idx_list.append(len(seg_idx_list))

        return EvaluationManifest(
            timestamps=np.array(timestamps_list),
            valid_indices=np.array(valid_idx_list, dtype=np.int64),
            seg_indices=np.array(seg_idx_list, dtype=np.int64),
            dates=sorted(s4_dates),
            n_hours=len(timestamps_list),
        )


@dataclass
class EvaluationManifest:
    """Unified S4 evaluation index."""

    timestamps: np.ndarray
    valid_indices: np.ndarray
    seg_indices: np.ndarray
    dates: list[str]
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
    """DEPRECATED: use ExperimentManifest.build_s4_eval_manifest() instead.
    Kept for backward compatibility.
    """
    ts = ds["ts"]
    all_dates = sorted(set(ts.dt.date))
    s4_dates = set()
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
        if ts[ts.dt.date == t.date()].count() != 24:
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
