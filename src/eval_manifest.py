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

# P1-9: day-length policy is explicit. 23/25-hour days are excluded and recorded.
DAY_LENGTH_PROTOCOL = "COMPLETE_24H_ONLY"

# P0-2 (master plan §1.2 / protocol §5): chronological host/reference split.
# Fractions are date-first over COMPLETE days:  H0 40% | S1R 10% | S2T 16% |
# S2V 4% | S3M 5% | S3C 5% | S4 20%.
#   H0  fit frozen host only (reads target)
#   S1R OOS host predictions; builds S1 rank reference + Data Signature only
#   S2T candidate training (reads target)
#   S2V candidate validation / checkpoint selection (no gradient)
#   S3M target-local atom memory
#   S3C target-local DVG calibration
#   S4  untouched test
SPLIT_7 = ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4")
FRAC_7 = (0.40, 0.10, 0.16, 0.04, 0.05, 0.05, 0.20)


def cumulative_bounds(n_dates: int, frac) -> np.ndarray:
    """Monotone cumulative segment boundaries [0=b0 < ... < bk=n_dates].

    P0-A fix: boundaries are float-rounded cumulative fractions; the last
    boundary is forced to n_dates so bounds stay non-decreasing and cover every
    date exactly. Never mixes cumulative boundaries with segment sizes (the old
    `counts[-1] = n - sum(counts[:-1])` summed cumulative boundaries as if they
    were sizes, producing a negative non-monotone tail).
    """
    bounds = np.rint(np.asarray(frac, dtype=np.float64).cumsum()
                     * n_dates).astype(np.int64)
    bounds[-1] = n_dates
    return np.concatenate([[0], bounds])


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
    s3_m_dates: set = field(default_factory=set)
    s3_c_dates: set = field(default_factory=set)
    excluded_dates: dict = field(default_factory=dict)
    raw_to_valid_row: dict = field(default_factory=dict)
    # protocol §5: no-silent-deletion accounting
    n_original_hours: int = 0
    n_excluded_dates: int = 0

    @staticmethod
    def from_dataset(ds: dict, valid: np.ndarray, dataset_id: str = "",
                     frac=FRAC_7, s7: bool = True,
                     s3m_frac: float = 0.50) -> "ExperimentManifest":
        """Build from raw dataset dict + valid index array.

        1. Extract all unique calendar dates from ts.
        2. Exclude 23/25h dates (recorded in excluded_dates).
        3. Split complete dates chronologically.
           s7=True  -> 7-segment H0/S1R/S2T/S2V/S3M/S3C/S4 (P0-2, default).
           s7=False -> legacy 4-segment S1/S2/S3/S4 + nested S3-M/S3-C
                       (kept so historical experiments keep passing).
        4. Map each raw index to its date and split.
        5. Compute split hash (includes all segment boundaries + excluded counts).
        """
        ts = ds["ts"]
        n_total = len(ts)

        # Step 1-2: unique dates, exclude non-24h
        all_dates = []
        excluded = {}
        for d in pd.unique(ts.dt.date):
            n_h = (ts.dt.date == d).sum()
            if n_h == 24:
                all_dates.append(d)
            else:
                excluded[str(d)] = int(n_h)
        all_dates = sorted(all_dates)
        date_strs = [str(d) for d in all_dates]
        n_dates = len(date_strs)

        # Step 3: chronological date split.
        # Use float rounding so segment sizes sum to n_dates exactly.
        if s7:
            labels = list(SPLIT_7)
            f = list(frac) if len(frac) == 7 else list(FRAC_7)
        else:
            labels = ["S1", "S2", "S3", "S4"]
            f = list(frac) if len(frac) == 4 else [0.50, 0.20, 0.10, 0.20]
        bounds = cumulative_bounds(n_dates, f)
        split_of_date = {}
        for i, d in enumerate(date_strs):
            seg = int(np.searchsorted(bounds, i, side="right") - 1)
            split_of_date[d] = labels[min(seg, len(labels) - 1)]

        # Legacy-compatible aggregate view: S1 = H0+S1R, S2 = S2T+S2V,
        # S3 = S3M+S3C, S4 unchanged. Used only when s7=False is bypassed via
        # split_of_date_4 below; consumers of the 7-way split read directly.
        s3_m_dates = {d for d in date_strs if split_of_date[d] == "S3M"}
        s3_c_dates = {d for d in date_strs if split_of_date[d] == "S3C"}
        # If caller explicitly used the legacy 4-way (s7=False), nest S3.
        if not s7:
            s3_dates = [d for d in date_strs if split_of_date[d] == "S3"]
            n_s3 = len(s3_dates)
            n_s3m = int(n_s3 * s3m_frac)
            s3_m_dates = set(s3_dates[:n_s3m])
            s3_c_dates = set(s3_dates[n_s3m:])

        # Step 4: map raw indices to dates
        raw_to_date = np.full(n_total, -1, dtype=np.int32)
        date_to_idx = {d: i for i, d in enumerate(date_strs)}
        for i in range(n_total):
            d_str = str(ts.iloc[i].date())
            raw_to_date[i] = date_to_idx.get(d_str, -1)

        # Step 5: split hash (segment assignment + excluded counts)
        h = hashlib.sha256()
        h.update(dataset_id.encode())
        h.update(json.dumps({d: split_of_date.get(d, "?") for d in date_strs}).encode())
        h.update(json.dumps({"excluded": excluded}).encode())
        split_hash_val = h.hexdigest()[:16]

        # valid is an ARRAY of raw indices (build_tabular contract), not a bool mask
        valid_arr = np.asarray(valid, dtype=np.int64)
        valid_indices_arr = valid_arr
        # raw index -> valid-row position (X/y from build_tabular are valid-rows-only)
        raw_to_valid_row = {int(raw): pos for pos, raw in enumerate(valid_arr)}

        return ExperimentManifest(
            dataset_id=dataset_id,
            timestamps=ts.values if hasattr(ts, "values") else np.array(ts),
            dates=date_strs,
            raw_to_date=raw_to_date,
            split_of_date=split_of_date,
            valid_indices=valid_indices_arr,
            valid_to_raw=valid_indices_arr,
            split_hash=split_hash_val,
            s3_m_dates=s3_m_dates,
            s3_c_dates=s3_c_dates,
            excluded_dates=excluded,
            raw_to_valid_row=raw_to_valid_row,
            n_original_hours=int(n_total),
            n_excluded_dates=int(len(excluded)),
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
        """Valid-hour RAW indices in a split.

        split is a 7-way segment (H0/S1R/S2T/S2V/S3M/S3C/S4) or a legacy
        4-way aggregate (S1/S2/S3/S4). S1 => H0+S1R, S2 => S2T+S2V, S3 =>
        S3M+S3C, S4 unchanged.
        """
        if split in ("S1", "S2", "S3", "S4"):
            return self.valid_indices_in_seg4_split(split)
        mask = np.array([self.raw_idx_is(vi, split) for vi in self.valid_indices])
        return self.valid_indices[mask]

    def valid_row_in_split(self, split: str) -> np.ndarray:
        """Valid-row POSITIONS (for indexing X/y from build_tabular)."""
        raw = self.valid_indices_in_split(split)
        return np.array([self.raw_to_valid_row[int(r)] for r in raw], dtype=np.int64)

    def dates_in_split(self, split: str) -> set:
        """Dates in a 7-way segment, or the union of a legacy 4-way group."""
        if split in ("S1", "S2", "S3", "S4"):
            return self._seg7(split)
        return {d for d, s in self.split_of_date.items() if s == split}

    def dates_in_s3m(self) -> set:
        return self.s3_m_dates

    def dates_in_s3c(self) -> set:
        return self.s3_c_dates

    # ---- P0-2 aggregate view: 7-segment -> legacy 4-segment mapping ----
    _SEG4_MAP = {"H0": "S1", "S1R": "S1", "S2T": "S2", "S2V": "S2",
                 "S3M": "S3", "S3C": "S3", "S4": "S4"}

    def seg4(self, seg7: str) -> str:
        """Map a 7-way segment name to the legacy 4-way group."""
        return self._SEG4_MAP.get(seg7, seg7)

    def split_of_date_4(self) -> dict:
        """Aggregated legacy view: H0+S1R->S1, S2T+S2V->S2, S3M+S3C->S3."""
        return {d: self.seg4(s) for d, s in self.split_of_date.items()}

    def valid_indices_in_seg4_split(self, split4: str) -> np.ndarray:
        """Valid-hour RAW indices in a legacy 4-way split group."""
        mask = np.array([
            self.seg4(self.split_of_date.get(
                self.dates[int(self.raw_to_date[vi])], "")) == split4
            for vi in self.valid_indices], dtype=bool)
        return self.valid_indices[mask]

    def valid_row_in_seg4_split(self, split4: str) -> np.ndarray:
        """Valid-row positions in a legacy 4-way split group (indexes X/y)."""
        raw = self.valid_indices_in_seg4_split(split4)
        return np.array([self.raw_to_valid_row[int(r)] for r in raw], dtype=np.int64)

    def _seg7(self, seg4: str) -> set:
        return {d for d, s in self.split_of_date.items() if self.seg4(s) == seg4}

    def assert_s3_disjoint(self) -> bool:
        """Verify the date segments are pairwise disjoint.

        In 7-way mode the segments H0/S1R/S2T/S2V/S3M/S3C/S4 are checked
        directly. In legacy 4-way mode S1/S2/S3/S4 (+ nested S3-M/S3-C) are
        checked. True in both cases.
        """
        groups = {"H0": self.dates_in_split("H0"),
                  "S1R": self.dates_in_split("S1R"),
                  "S2T": self.dates_in_split("S2T"),
                  "S2V": self.dates_in_split("S2V"),
                  "S3-M": self.s3_m_dates,
                  "S3-C": self.s3_c_dates,
                  "S4": self.dates_in_split("S4")}
        names = list(groups.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if groups[names[i]] & groups[names[j]]:
                    return False
        return True

    def assert_7seg_disjoint(self) -> bool:
        """Verify the 7 P0-2 segments are pairwise disjoint and exhaustive.

        Checks: every date maps to a valid 7-way segment; the segments are
        pairwise disjoint; the union of all 7 segment date-sets equals the
        full date list (no date unmapped); every segment is non-empty.
        """
        by_seg = {s: set() for s in SPLIT_7}
        for d in self.dates:
            s = self.split_of_date.get(d)
            if s not in SPLIT_7:
                return False
            by_seg[s].add(d)
        # pairwise disjoint
        segs = list(by_seg.values())
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if segs[i] & segs[j]:
                    return False
        # exhaustive: union equals full date set
        union = set().union(*segs)
        if union != set(self.dates):
            return False
        # all non-empty
        return all(len(s) > 0 for s in segs)

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
