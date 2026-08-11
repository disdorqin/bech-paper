"""HCH v2 data layer — 24-hour episode construction, DST handling, exogenous tokens.

DailyEpisodeBatch schema (spec §4.2):
    host_pred  [B, 24, 1]   frozen backbone forecast
    target     [B, 24, 1]   ground truth (only S2/S3/S4; S4 never in training)
    exog       [B, 24, N, D_exog]  exogenous tokens, N varies per-market
    exog_mask  [B, 24, N]   1=valid, 0=pad/null
    time_feat  [B, 24, Dt]  hour_sin/cos, dow_sin/cos, mon_sin/cos, is_weekend
    date_ids   list[str]    date strings for audit
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import load_dataset, load_shandong, DATASETS


@dataclass
class DailyEpisodeBatch:
    host_pred: torch.Tensor
    target: torch.Tensor
    exog: torch.Tensor
    exog_mask: torch.Tensor
    time_feat: torch.Tensor
    date_ids: list


def _safe_values_at(df, idxs):
    vals = df.iloc[idxs].values
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals.astype(np.float32)


def _build_time_features(ts_slice: pd.Series) -> np.ndarray:
    h = ts_slice.dt.hour.values.astype(np.float32)
    dow = ts_slice.dt.dayofweek.values.astype(np.float32)
    mon = ts_slice.dt.month.values.astype(np.float32)
    feats = np.column_stack([
        np.sin(2 * np.pi * h / 24),
        np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * mon / 12),
        np.cos(2 * np.pi * mon / 12),
        (dow >= 5).astype(np.float32),
    ])
    return feats  # [24, 7]


def _hour_sort_key(ts_slice: pd.Series) -> np.ndarray:
    return np.argsort(ts_slice.dt.hour.values)


class DailyEpisodeDataset(Dataset):
    """Yields one 24-hour day per __getitem__.

    Excludes DST days (23/25 hours). Uses S1 stats for normalization.
    """

    def __init__(
        self,
        ds: dict,
        host_pred: np.ndarray,
        allowed_dates: set,
        price_mean: float = 0.0,
        price_std: float = 1.0,
    ):
        self.ts = ds["ts"]
        self.price = ds["price"].astype(np.float32)
        self.host_pred = host_pred.astype(np.float32)
        self.price_mean = price_mean
        self.price_std = price_std

        exog_fc = ds["exog_fc"]
        if hasattr(exog_fc, "values"):
            self.exog_fc = exog_fc.values.astype(np.float32)
        else:
            self.exog_fc = np.zeros((len(self.price), 0), dtype=np.float32)

        self.n_exog = self.exog_fc.shape[1]

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
        valid_dates = []
        for d in self.dates:
            d_ts = pd.Timestamp(d)
            mask = self.ts.dt.date == d_ts.date()
            idxs = np.where(mask.values)[0]
            if len(idxs) != 24:
                continue
            sort_idx = _hour_sort_key(self.ts.iloc[idxs])
            self._date_to_idx[d] = idxs[sort_idx]
            valid_dates.append(d)
        self.dates = valid_dates

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        d = self.dates[idx]
        ixs = self._date_to_idx[d]

        yhat = self.host_pred[ixs].reshape(24, 1).copy()
        y = self.price[ixs].reshape(24, 1).copy()

        exog = np.zeros((24, max(self.n_exog, 1), 3), dtype=np.float32)
        mask = np.zeros((24, max(self.n_exog, 1)), dtype=np.float32)
        if self.n_exog > 0:
            raw = self.exog_fc[ixs]
            for j in range(self.n_exog):
                col = raw[:, j]
                c_mean = np.nanmean(col) if not np.all(np.isnan(col)) else 0.0
                c_std = np.nanstd(col) if not np.all(np.isnan(col)) else 1.0
                exog[:, j, 0] = np.nan_to_num((col - c_mean) / max(c_std, 1e-8), nan=0.0)
                exog[:, j, 1] = float(j + 1)  # variable type id
                exog[:, j, 2] = 1.0  # availability flag
                mask[:, j] = 1.0
        else:
            mask[:, 0] = 1.0

        time_feat = _build_time_features(self.ts.iloc[ixs])

        y_norm = (y - self.price_mean) / max(self.price_std, 1.0)
        yhat_norm = (yhat - self.price_mean) / max(self.price_std, 1.0)

        return DailyEpisodeBatch(
            host_pred=torch.tensor(yhat_norm, dtype=torch.float32),
            target=torch.tensor(y_norm, dtype=torch.float32),
            exog=torch.tensor(exog, dtype=torch.float32),
            exog_mask=torch.tensor(mask, dtype=torch.float32),
            time_feat=torch.tensor(time_feat, dtype=torch.float32),
            date_ids=d,
        )


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


def collate_daily(batches: list[DailyEpisodeBatch]) -> DailyEpisodeBatch:
    max_exog = max(b.exog.shape[1] for b in batches)
    B = len(batches)

    hp = torch.stack([b.host_pred for b in batches])
    tgt = torch.stack([b.target for b in batches])
    tf = torch.stack([b.time_feat for b in batches])

    exog_padded = torch.zeros(B, 24, max_exog, 3)
    mask_padded = torch.zeros(B, 24, max_exog)
    for i, b in enumerate(batches):
        n = b.exog.shape[1]
        exog_padded[i, :, :n] = b.exog
        mask_padded[i, :, :n] = b.exog_mask

    return DailyEpisodeBatch(
        host_pred=hp, target=tgt, exog=exog_padded,
        exog_mask=mask_padded, time_feat=tf,
        date_ids=[b.date_ids for b in batches],
    )


def build_dataloaders(
    ds_key: str,
    host_pred: np.ndarray,
    ds: dict,
    batch_size: int = 32,
) -> dict:
    splits = date_based_split(ds)

    s1_dates = splits["S1"]
    s1_mask = np.array([str(d) in s1_dates for d in ds["ts"].dt.date])
    price_s1 = ds["price"][s1_mask].astype(np.float32)
    p_mean = float(price_s1.mean())
    p_std = float(price_s1.std()) if price_s1.std() > 0 else 1.0

    loaders = {}
    for seg in ("S1", "S2", "S3", "S4"):
        dset = DailyEpisodeDataset(
            ds, host_pred, splits[seg], price_mean=p_mean, price_std=p_std,
        )
        loaders[seg] = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=(seg == "S2"),
            collate_fn=collate_daily, drop_last=False,
        )
        loaders[f"{seg}_n_days"] = len(dset)
        loaders[f"{seg}_excluded"] = dset.excluded_dates

    loaders["price_mean"] = p_mean
    loaders["price_std"] = p_std
    return loaders


def build_blocked_s2_loaders(
    ds: dict,
    host_pred: np.ndarray,
    batch_size: int = 32,
    n_blocks: int = 5,
) -> list:
    """Split S2 dates into n_blocks for blocked forward cross-fitting (§5.1).

    Returns list of DataLoaders, one per block, in chronological order
    with shuffle=False. Each block's dataset uses its own date-indexing
    for OOF generation.
    """
    splits = date_based_split(ds)

    s1_dates = splits["S1"]
    s1_mask = np.array([str(d) in s1_dates for d in ds["ts"].dt.date])
    price_s1 = ds["price"][s1_mask].astype(np.float32)
    p_mean = float(price_s1.mean())
    p_std = float(price_s1.std()) if price_s1.std() > 0 else 1.0

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
                                   price_mean=p_mean, price_std=p_std)
        loader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_daily, drop_last=False,
        )
        blocks.append(loader)

    return blocks, p_mean, p_std
