"""HCH v2 data layer — 24-hour episode construction, S1-only normalization,
raw/model dual-channel, target-free S4, exogenous type/mask/lag, learned-null.

DailyEpisodeBatch (§4.2 addendum):
    host_raw    [B, 24, 1]   original currency (preserves economic zero)
    host_model  [B, 24, 1]   S1 normalized, for encoder consumption
    target_raw  [B, 24, 1] | None   None for S4 predict-only
    target_model [B, 24, 1] | None  normalized target, deprecated future
    exog_value  [B, 24, N, 1]  S1 normalized exogenous values
    exog_type   [B, 24, N]    stable categorical feature ID
    exog_mask   [B, 24, N]    1=valid input, 0=null/padding
    lag_context [B, 24, D_lag]  past price/residual/actual, cutoff-safe
    time_feat   [B, 24, Dt]   hour_sin/cos, dow_sin/cos, mon_sin/cos, is_weekend
    market_id   [B]           market identifier token
    target_id   [B]           target identifier token (DA/RT)
    timestamps  list          per-hour timestamps for audit
    date_ids    list[str]     date strings for memory/audit
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import load_dataset, load_shandong, DATASETS

EPS = 1e-8


@dataclass
class DailyEpisodeBatch:
    host_raw: torch.Tensor
    host_model: torch.Tensor
    target_raw: torch.Tensor | None
    target_model: torch.Tensor | None
    exog_value: torch.Tensor
    exog_type: torch.Tensor
    exog_mask: torch.Tensor
    lag_context: torch.Tensor
    time_feat: torch.Tensor
    market_id: torch.Tensor
    target_id: torch.Tensor
    timestamps: list = field(default_factory=list)
    date_ids: list = field(default_factory=list)


def _build_time_features(ts_slice: pd.Series) -> np.ndarray:
    h = ts_slice.dt.hour.values.astype(np.float32)
    dow = ts_slice.dt.dayofweek.values.astype(np.float32)
    mon = ts_slice.dt.month.values.astype(np.float32)
    return np.column_stack([
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mon / 12),
        np.cos(2 * np.pi * mon / 12),
        (dow >= 5).astype(np.float32),
    ])


def _hour_sort_key(ts_slice: pd.Series) -> np.ndarray:
    return np.argsort(ts_slice.dt.hour.values)


class DailyEpisodeDataset(Dataset):
    """Yields one 24-hour day per __getitem__.

    Normalization: S1-only. Exogenous values are standardized per-feature
    using S1-fitted mean/std scalers. Host and target prices use S1
    price_mean/price_std for model-coordinate normalization.
    """

    def __init__(
        self,
        ds: dict,
        host_pred: np.ndarray,
        allowed_dates: set,
        price_mean: float = 0.0,
        price_std: float = 1.0,
        exog_scalers: dict | None = None,
        market_id: int = 0,
        target_id: int = 0,
        include_target: bool = True,
    ):
        self.ts = ds["ts"]
        self.price = ds["price"].astype(np.float32)
        self.host_pred = host_pred.astype(np.float32)
        self.price_mean = price_mean
        self.price_std = max(price_std, 1.0)
        self.market_id = market_id
        self.target_id = target_id
        self.include_target = include_target

        exog_fc = ds.get("exog_fc", ds.get("exog"))
        if hasattr(exog_fc, "values"):
            self.exog_fc = exog_fc.values.astype(np.float32)
        else:
            self.exog_fc = np.zeros((len(self.price), 0), dtype=np.float32)
        self.n_exog = self.exog_fc.shape[1]

        if exog_scalers is not None:
            self.exog_mean = np.array([exog_scalers.get(j, {}).get("mean", 0.0)
                                        for j in range(self.n_exog)], dtype=np.float32)
            self.exog_std = np.array([max(exog_scalers.get(j, {}).get("std", 1.0), EPS)
                                       for j in range(self.n_exog)], dtype=np.float32)
        else:
            self.exog_mean = np.zeros(self.n_exog, dtype=np.float32)
            self.exog_std = np.ones(self.n_exog, dtype=np.float32)

        exog_act = ds.get("exog_act")
        if hasattr(exog_act, "values"):
            self.exog_act = exog_act.values.astype(np.float32)
        else:
            self.exog_act = np.zeros((len(self.price), 0), dtype=np.float32)
        self.n_exog_act = self.exog_act.shape[1]

        self.dates = []
        excluded = []
        for d in pd.unique(self.ts.dt.date):
            d_str = str(d)
            if d_str not in allowed_dates:
                continue
            mask = self.ts.dt.date == d
            n_h = mask.sum()
            if n_h != 24:
                excluded.append((d_str, int(n_h)))
                continue
            self.dates.append(d_str)

        self.excluded_dates = excluded

        self._date_to_idx = {}
        self._date_timestamps = {}
        valid_dates = []
        for d in self.dates:
            d_ts = pd.Timestamp(d)
            mask = self.ts.dt.date == d_ts.date()
            idxs = np.where(mask.values)[0]
            if len(idxs) != 24:
                continue
            sort_idx = _hour_sort_key(self.ts.iloc[idxs])
            self._date_to_idx[d] = idxs[sort_idx]
            self._date_timestamps[d] = list(self.ts.iloc[idxs[sort_idx]])
            valid_dates.append(d)
        self.dates = valid_dates

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        d = self.dates[idx]
        ixs = self._date_to_idx[d]

        yhat_raw = self.host_pred[ixs].reshape(24, 1).copy()
        y_raw = self.price[ixs].reshape(24, 1).copy()

        exog_value = np.zeros((24, max(self.n_exog, 1), 1), dtype=np.float32)
        exog_type = np.zeros((24, max(self.n_exog, 1)), dtype=np.float32)
        exog_mask = np.zeros((24, max(self.n_exog, 1)), dtype=np.float32)

        if self.n_exog > 0:
            raw = self.exog_fc[ixs]
            for j in range(self.n_exog):
                col = raw[:, j]
                exog_value[:, j, 0] = np.nan_to_num(
                    (col - self.exog_mean[j]) / self.exog_std[j], nan=0.0)
                exog_type[:, j] = float(j + 1)
                exog_mask[:, j] = 1.0
        else:
            exog_mask[:, 0] = 0.0

        lag_ctx = self._build_lag_context(ixs)

        time_feat = _build_time_features(self.ts.iloc[ixs])

        yhat_norm = (yhat_raw - self.price_mean) / self.price_std
        y_norm = (y_raw - self.price_mean) / self.price_std if self.include_target else None

        return DailyEpisodeBatch(
            host_raw=torch.tensor(yhat_raw, dtype=torch.float32),
            host_model=torch.tensor(yhat_norm, dtype=torch.float32),
            target_raw=torch.tensor(y_raw, dtype=torch.float32) if self.include_target else None,
            target_model=torch.tensor(y_norm, dtype=torch.float32) if self.include_target and y_norm is not None else None,
            exog_value=torch.tensor(exog_value, dtype=torch.float32),
            exog_type=torch.tensor(exog_type, dtype=torch.float32),
            exog_mask=torch.tensor(exog_mask, dtype=torch.float32),
            lag_context=torch.tensor(lag_ctx, dtype=torch.float32),
            time_feat=torch.tensor(time_feat, dtype=torch.float32),
            market_id=torch.tensor([self.market_id], dtype=torch.long),
            target_id=torch.tensor([self.target_id], dtype=torch.long),
            timestamps=self._date_timestamps.get(d, []),
            date_ids=d,
        )

    def _build_lag_context(self, ixs):
        ctx = np.zeros((24, 5), dtype=np.float32)
        for h in range(24):
            raw_idx = ixs[h]
            lag24 = raw_idx - 24
            lag168 = raw_idx - 168
            if lag24 >= 0:
                ctx[h, 0] = (self.price[lag24] - self.price_mean) / self.price_std
                ctx[h, 1] = self.host_pred[lag24]
            if lag168 >= 0:
                ctx[h, 2] = (self.price[lag168] - self.price_mean) / self.price_std
            if lag24 >= 0:
                ctx[h, 3] = self.price[lag24] - self.host_pred[lag24]  # residual lag 24h
            ctx[h, 4] = float(h)
        return ctx


def date_based_split(ds: dict, frac=(0.50, 0.20, 0.10, 0.20)) -> dict[str, set]:
    """Split unique dates into S1/S2/S3/S4. Returns sets of date strings."""
    dates = sorted(pd.unique(ds["ts"].dt.date))
    n = len(dates)
    a = int(n * frac[0])
    b = a + int(n * frac[1])
    c = b + int(n * frac[2])
    return {
        "S1": set(str(d) for d in dates[:a]),
        "S2": set(str(d) for d in dates[a:b]),
        "S3": set(str(d) for d in dates[b:c]),
        "S4": set(str(d) for d in dates[c:]),
    }


def fit_exog_scalers(ds: dict, s1_dates: set) -> dict:
    """Fit per-feature mean/std scalers using only S1 data.
    Returns dict: {feature_idx: {"mean": float, "std": float}}
    """
    exog_fc = ds.get("exog_fc", ds.get("exog"))
    if exog_fc is None or (hasattr(exog_fc, "values") and exog_fc.shape[1] == 0):
        return {}
    vals = exog_fc.values if hasattr(exog_fc, "values") else exog_fc
    s1_mask = np.array([str(d) in s1_dates for d in ds["ts"].dt.date])
    s1_data = vals[s1_mask].astype(np.float64)
    scalers = {}
    for j in range(s1_data.shape[1]):
        col = s1_data[:, j]
        col = col[np.isfinite(col)]
        scalers[j] = {"mean": float(np.mean(col)) if len(col) > 0 else 0.0,
                       "std": float(np.std(col)) if len(col) > 0 else 1.0}
    return scalers


def collate_daily(batches: list[DailyEpisodeBatch]) -> DailyEpisodeBatch:
    B = len(batches)
    max_exog = max(b.exog_value.shape[1] for b in batches)

    def pad_exog(attr, fill=0.0):
        padded = torch.zeros(B, 24, max_exog, *batches[0].__getattribute__(attr).shape[3:])
        for i, b in enumerate(batches):
            n = b.__getattribute__(attr).shape[1]
            padded[i, :, :n] = b.__getattribute__(attr)
        return padded

    def stack_or_none(attr):
        vals = [b.__getattribute__(attr) for b in batches]
        if vals[0] is None:
            return None
        return torch.stack(vals)

    return DailyEpisodeBatch(
        host_raw=torch.stack([b.host_raw for b in batches]),
        host_model=torch.stack([b.host_model for b in batches]),
        target_raw=stack_or_none("target_raw"),
        target_model=stack_or_none("target_model"),
        exog_value=pad_exog("exog_value"),
        exog_type=pad_exog("exog_type"),
        exog_mask=pad_exog("exog_mask"),
        lag_context=torch.stack([b.lag_context for b in batches]),
        time_feat=torch.stack([b.time_feat for b in batches]),
        market_id=torch.stack([b.market_id[:1] for b in batches]),
        target_id=torch.stack([b.target_id[:1] for b in batches]),
        timestamps=[b.timestamps for b in batches],
        date_ids=[b.date_ids for b in batches],
    )


def build_dataloaders(
    ds_key: str,
    host_pred: np.ndarray,
    ds: dict,
    batch_size: int = 32,
    include_target_s4: bool = False,
) -> dict:
    splits = date_based_split(ds)

    s1_dates = splits["S1"]
    s1_mask = np.array([str(d) in s1_dates for d in ds["ts"].dt.date])
    price_s1 = ds["price"][s1_mask].astype(np.float32)
    p_mean = float(price_s1.mean())
    p_std = float(price_s1.std()) if price_s1.std() > 0 else 1.0

    exog_scalers = fit_exog_scalers(ds, s1_dates)

    loaders = {}
    for seg in ("S1", "S2", "S3", "S4"):
        inc_target = True
        if seg == "S4" and not include_target_s4:
            inc_target = False
        dset = DailyEpisodeDataset(
            ds, host_pred, splits[seg],
            price_mean=p_mean, price_std=p_std,
            exog_scalers=exog_scalers,
            include_target=inc_target,
        )
        loaders[seg] = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=(seg == "S2"),
            collate_fn=collate_daily, drop_last=False,
        )
        loaders[f"{seg}_n_days"] = len(dset)
        loaders[f"{seg}_excluded"] = dset.excluded_dates

    loaders["price_mean"] = p_mean
    loaders["price_std"] = p_std
    loaders["exog_scalers"] = exog_scalers
    loaders["splits"] = splits
    return loaders


def build_blocked_s2_loaders(
    ds: dict,
    host_pred: np.ndarray,
    batch_size: int = 32,
    n_blocks: int = 5,
) -> tuple:
    splits = date_based_split(ds)

    s1_dates = splits["S1"]
    s1_mask = np.array([str(d) in s1_dates for d in ds["ts"].dt.date])
    price_s1 = ds["price"][s1_mask].astype(np.float32)
    p_mean = float(price_s1.mean())
    p_std = float(price_s1.std()) if price_s1.std() > 0 else 1.0
    exog_scalers = fit_exog_scalers(ds, s1_dates)

    s2_dates = sorted(splits["S2"])
    if len(s2_dates) < n_blocks:
        n_blocks = max(2, len(s2_dates))

    block_size = len(s2_dates) // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(s2_dates)
        block_dates = set(s2_dates[start:end])
        dset = DailyEpisodeDataset(ds, host_pred, block_dates,
                                   price_mean=p_mean, price_std=p_std,
                                   exog_scalers=exog_scalers,
                                   include_target=True)
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_daily, drop_last=False,
        )
        blocks.append(loader)

    return blocks, p_mean, p_std, exog_scalers
