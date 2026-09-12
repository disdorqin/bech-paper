"""Shared dataset/task contract for the China-5 post-hoc baseline panel (2026-09-12).

Day-ahead price -> day-ahead price, next-24h, hourly, univariate target.

This module is deliberately read-projected: every reader takes an explicit set of
authorized natural days and refuses to touch any coordinate belonging to a closed
role (PROTECTED_FINAL).  Nothing here may use fill/forward-fill imputation.

Governing rules (see 00_protocol/PROTOCOL_FREEZE.md):
  * chronological roles only, never a random split
  * scalers / quantiles / thresholds fit on the corresponding TRAIN partition only
  * `*实际值` (realized), `实时电价` (realized RT) and `竞价空间预测值` are never inputs
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()

SEQ = 168          # causal context hours
H = 24             # forecast horizon (next 24 h)
TZ = "Asia/Shanghai"

# ---------------------------------------------------------------- market spec
# `target` is the DA price column.  `legal` lists *only* source-labelled 预测值
# columns whose target-day values are published before the day-ahead auction of
# the target day.  The bidding-space forecast 竞价空间预测值 is deliberately
# excluded project-wide: the GANSU audit could not confirm its pre-DA
# publication time, and the same source dialect is used by every province file.
EXCLUDED_FEATURES = {
    "竞价空间预测值": "pre-DA publication time unconfirmed (GANSU audit precedent)",
}

MARKETS: dict[str, dict] = {
    "SHANDONG": dict(
        dataset_id="SHANDONG_DA", region="Shandong, China", public=False,
        path="data/CHINA/SHANDONG/source/shandong_pmos_hourly.csv",
        reader="csv", encoding="gbk", tcol="时刻", target="日前电价",
    ),
    "SHAANXI": dict(
        dataset_id="SHAANXI_DA", region="Shaanxi, China", public=True,
        path="data/CHINA/SHAANXI/陕西24h电价数据集(1).xlsx",
        reader="xlsx", tcol="时刻", target="日前电价",
    ),
    "NINGXIA": dict(
        dataset_id="NINGXIA_DA", region="Ningxia, China", public=True,
        path="data/CHINA/NINGXIA/宁夏24h电价数据集.xlsx",
        reader="xlsx", tcol="时刻", target="日前电价",
    ),
    "QINGHAI": dict(
        dataset_id="QINGHAI_DA", region="Qinghai, China", public=True,
        path="data/CHINA/QINGHAI/青海24h电价数据集.xlsx",
        reader="xlsx", tcol="时刻", target="日前电价",
    ),
}

# Registry rows that exist as *data-admission* facts but cannot host a run.
NON_RUNNABLE = {
    "SHANXI": dict(
        dataset_id="SHANXI_DA", status="ABSENT_FROM_REGISTRY",
        reason=("No Shanxi dataset file exists: `data/CHINA/` holds only "
                "GANSU/LIAONING/NINGXIA/QINGHAI/SHAANXI/SHANDONG, and "
                "`data/MARKET_INDEX.csv` has no Shanxi row. Shanxi appears only as a "
                "*candidate* row `SHANXI_DA_RT` in the archived v2.5 research table "
                "`docs/archive/v2_5_details/HCH_V2.5_DATA_TRAINING_RESEARCH_20260820/"
                "dataset_registry_candidates.csv`, classified `C_diagnostic_only` with "
                "0-1 normalised features and an undisclosed licence (open item D5). It was "
                "never admitted to the registry. RESEARCH_STATE forbids inventing or "
                "silently substituting a market, so no Shanxi cell is reported."),
    ),
    "LIAONING": dict(
        dataset_id="LIAONING_DA", status="DATA_BLOCKED",
        reason=("Registered as public=true but role=candidate with admission unresolved; "
                "data/CHINA/LIAONING contains only portal HTML captures, no price dataset."),
    ),
}

# --------------------------------------------------------------- role layout
# Boundaries are expressed as fractions of the eligible-episode count and are
# identical in every market.  They reproduce the already-frozen GANSU_DA
# boundaries exactly: HOST_TRAIN+HOST_VAL == old S1 == first 50%,
# POST_TRAIN+DEV_EVAL == old S2 == 50..70%.
ROLE_FRACTIONS = dict(host_train=0.45, host_val=0.05, post_train=0.15,
                      dev_eval=0.05, protected_final=0.30)
HOST_VAL_FRACTION_OF_S1 = 0.10     # tail 10% of the host-fit block, as in GANSU_DA

ROLES = ("HOST_TRAIN", "HOST_VAL", "POST_TRAIN", "DEV_EVAL", "PROTECTED_FINAL")
OPEN_ROLES = ("HOST_TRAIN", "HOST_VAL", "POST_TRAIN", "DEV_EVAL")
CLOSED_ROLES = ("PROTECTED_FINAL",)

FORBIDDEN_TARGET_DAY_INPUTS = ("日前电价", "实时电价", "竞价空间预测值")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_path(market: str) -> Path:
    return ROOT / MARKETS[market]["path"]


def read_header(market: str) -> list[str]:
    spec = MARKETS[market]
    path = source_path(market)
    if spec["reader"] == "xlsx":
        return [str(c) for c in pd.read_excel(path, nrows=0).columns]
    return [str(c) for c in pd.read_csv(path, nrows=0, encoding=spec["encoding"]).columns]


def read_timestamps_only(market: str) -> pd.Series:
    """Read just the timestamp column.  Never touches price/feature values."""
    spec = MARKETS[market]
    path = source_path(market)
    tcol = spec["tcol"]
    if spec["reader"] == "xlsx":
        df = pd.read_excel(path, usecols=[tcol])
    else:
        df = pd.read_csv(path, usecols=[tcol], encoding=spec["encoding"])
    return pd.to_datetime(df[tcol], errors="coerce")


def read_projected(market: str, columns: list[str], keep_positions: set[int]) -> pd.DataFrame:
    """Read only the requested rows of the requested columns.

    Raises if any requested column is missing.  The row projection is applied at
    parse time (`skiprows`) so closed-role coordinates are never materialised.
    """
    spec = MARKETS[market]
    path = source_path(market)
    tcol = spec["tcol"]
    if tcol not in columns:
        columns = [tcol] + list(columns)
    skip = [i + 1 for i in range(_row_count(market)) if i not in keep_positions]
    if spec["reader"] == "xlsx":
        df = pd.read_excel(path, usecols=columns, skiprows=skip)
    else:
        df = pd.read_csv(path, usecols=columns, skiprows=skip, encoding=spec["encoding"])
    df.columns = [str(c).strip() for c in df.columns]
    df[tcol] = pd.to_datetime(df[tcol], errors="raise")
    return df


_ROW_COUNT_CACHE: dict[str, int] = {}


def _row_count(market: str) -> int:
    if market not in _ROW_COUNT_CACHE:
        spec = MARKETS[market]
        path = source_path(market)
        if spec["reader"] == "xlsx":
            df = pd.read_excel(path, usecols=[spec["tcol"]])
        else:
            df = pd.read_csv(path, usecols=[spec["tcol"]], encoding=spec["encoding"])
        _ROW_COUNT_CACHE[market] = len(df)
    return _ROW_COUNT_CACHE[market]


# ------------------------------------------------------------- day structure
@dataclass
class DayContract:
    """Timestamp-only contract: role assignment derived before any price is read."""
    timestamps: pd.DatetimeIndex
    positions: dict[str, np.ndarray]     # natural day (YYYY-MM-DD) -> row positions
    segments: dict[str, str]             # natural day -> role
    counts: dict[str, int]

    @property
    def open_days(self) -> list[str]:
        return [d for d, s in self.segments.items() if s in OPEN_ROLES]

    @property
    def closed_days(self) -> list[str]:
        return [d for d, s in self.segments.items() if s in CLOSED_ROLES]


def _day_of(ts: pd.Series) -> pd.Series:
    """Hour-ending convention: the row labelled D 01:00 .. D+1 00:00 belongs to day D.

    `hour_ending_label` is verified empirically by `p0_audit`; a row labelled
    00:00 is the 24th (last) hour of the *previous* natural day.
    """
    return (ts - pd.Timedelta(hours=1)).dt.date.astype(str)


def build_day_contract(market: str) -> DayContract:
    ts = read_timestamps_only(market)
    if ts.isna().any():
        raise RuntimeError(f"{market}: unparseable timestamps")
    if ts.duplicated().any():
        raise RuntimeError(f"{market}: duplicate timestamps")
    if not ts.is_monotonic_increasing:
        raise RuntimeError(f"{market}: timestamps not sorted")
    ts = ts.reset_index(drop=True)
    days = _day_of(ts)
    groups = {str(day): g.index.to_numpy() for day, g in pd.DataFrame({"d": days}).groupby("d", sort=True)}

    episodes: list[tuple[str, np.ndarray]] = []
    for day, pos in groups.items():
        pos = np.asarray(pos, int)
        td = ts.iloc[pos].to_numpy(dtype="datetime64[ns]")
        if len(pos) != 24:
            continue
        if not np.all(np.diff(td).astype("timedelta64[h]") == np.timedelta64(1, "h")):
            continue
        start = int(pos[0])
        if start < SEQ:
            continue
        # The causal context must be the calendar-correct 168 hours immediately
        # before `origin`, not merely 168 contiguous *file* rows.  Diffing
        # positions start-SEQ..start-1 alone hides a hole that ends exactly at
        # start-1 (its 25h step is the excluded boundary), so the window is
        # checked over start-SEQ..start inclusive: SEQ+1 slots, all 1h apart.
        hist = ts.iloc[start - SEQ:start + 1].to_numpy(dtype="datetime64[ns]")
        if len(hist) != SEQ + 1:
            continue
        if not np.all(np.diff(hist).astype("timedelta64[h]") == np.timedelta64(1, "h")):
            continue
        episodes.append((day, pos))

    n = len(episodes)
    if n < 40:
        raise RuntimeError(f"{market}: only {n} eligible episodes")
    # Same deterministic boundaries as the frozen GANSU_DA contract.
    b1, b2, b3 = int(round(.50 * n)), int(round(.70 * n)), int(round(.80 * n))
    segments: dict[str, str] = {}
    for i, (day, _) in enumerate(episodes):
        segments[day] = ("HOST_TRAIN" if i < b1 else
                         "POST_TRAIN" if i < b2 else
                         "PROTECTED_FINAL")
    # carve HOST_VAL from the tail of the host-fit block
    s1 = [d for d, _ in episodes if segments[d] == "HOST_TRAIN"]
    nval = max(1, int(np.floor(HOST_VAL_FRACTION_OF_S1 * len(s1))))
    for d in s1[len(s1) - nval:]:
        segments[d] = "HOST_VAL"
    # carve DEV_EVAL from the tail of the post block
    s2 = [d for d, _ in episodes if segments[d] == "POST_TRAIN"]
    ncut = max(1, int(np.floor(.25 * len(s2)))) if s2 else 0
    for d in s2[len(s2) - ncut:]:
        segments[d] = "DEV_EVAL"

    positions = {d: p for d, p in episodes}
    counts = {r: sum(v == r for v in segments.values()) for r in ROLES}
    if b3 != sum(1 for i, _ in enumerate(episodes) if i < b3):
        raise RuntimeError("boundary drift")
    return DayContract(ts, positions, segments, counts)


def split_manifest(contract: DayContract, market: str) -> dict:
    return {
        "schema": "china5_da_development_split.v1",
        "dataset_id": MARKETS[market]["dataset_id"],
        "physical_market_group": market,
        "source": MARKETS[market]["path"],
        "source_sha256": sha256_file(source_path(market)),
        "target_column": MARKETS[market]["target"],
        "task": "DA_to_DA_next24h",
        "timezone": TZ,
        "hour_label_convention": "hour_ending",
        "role_boundaries": {"s1_end": 0.50, "s2_end": 0.70, "s3_end": 0.80,
                            "host_val_tail_of_s1": HOST_VAL_FRACTION_OF_S1,
                            "dev_eval_tail_of_s2": 0.25},
        "counts": contract.counts,
        "open_roles": list(OPEN_ROLES),
        "closed_roles": list(CLOSED_ROLES),
        "forbidden_target_day_inputs": list(FORBIDDEN_TARGET_DAY_INPUTS),
        "excluded_features": EXCLUDED_FEATURES,
        "boundary_policy": ("identical deterministic natural-day proportions in every market; "
                            "timestamp-only construction, built before any price value is read"),
    }


# ------------------------------------------------------------------- readers
def load_target_windows(market: str, contract: DayContract, roles=OPEN_ROLES):
    """Projected target read: only the authorized roles' target day + prior context."""
    allowed = [d for d, s in contract.segments.items() if s in roles]
    target_pos: set[int] = set()
    for d in allowed:
        target_pos.update(contract.positions[d].tolist())
    needed: set[int] = set(target_pos)
    for d in allowed:
        start = int(contract.positions[d][0])
        needed.update(range(start - SEQ, start))
    forbidden: set[int] = set()
    for d in contract.closed_days:
        forbidden.update(contract.positions[d].tolist())
    if target_pos & forbidden or needed & forbidden:
        raise RuntimeError(f"{market}: PROTECTED_FINAL coordinate requested")

    keep = sorted(needed)
    df = read_projected(market, [MARKETS[market]["target"]], set(keep))
    tcol = MARKETS[market]["tcol"]
    tgt = MARKETS[market]["target"]
    ser = pd.to_numeric(df[tgt], errors="raise")
    mp = {pd.Timestamp(t): float(v) for t, v in zip(df[tcol], ser)}
    if len(mp) != len(df):
        raise RuntimeError(f"{market}: target projection collapsed")

    ctx, y, stamps, segs = [], [], [], []
    for d in allowed:
        pos = contract.positions[d]
        origin = pd.Timestamp(contract.timestamps[pos[0]])
        # day D occupies rows origin+0h .. origin+23h (hour-ending labels:
        # 01:00 D .. 00:00 D+1); context is the 168 hours immediately before origin.
        hts = [origin - pd.Timedelta(hours=k) for k in range(SEQ, 0, -1)]
        tts = [origin + pd.Timedelta(hours=k) for k in range(H)]
        if any(t not in mp for t in hts + tts):
            raise RuntimeError(f"{market}: projected target read incomplete for {d}")
        ctx.append(np.asarray([mp[t] for t in hts], np.float32)[:, None])
        y.append(np.asarray([mp[t] for t in tts], np.float32)[:, None])
        stamps.append(str(origin))
        segs.append(contract.segments[d])
    return (np.stack(ctx), np.stack(y), np.asarray(stamps), np.asarray(segs),
            MARKETS[market]["dataset_id"], market, {
                "target_column": tgt,
                "target_day_values_read": len(target_pos),
                "causal_context_values_read": len(needed - target_pos),
                "protected_final_target_values_read": 0,
            })


def load_legal_state(market: str, origins, legal_cols: list[str]) -> tuple[np.ndarray, dict]:
    """Read the authorized forecast-time feature block for the target-day hours."""
    tcol = MARKETS[market]["tcol"]
    header = read_header(market)
    if any(c not in header for c in [tcol] + legal_cols):
        missing = [c for c in legal_cols if c not in header]
        raise RuntimeError(f"{market}: registered legal columns missing: {missing}")
    if any(c in FORBIDDEN_TARGET_DAY_INPUTS for c in legal_cols):
        raise RuntimeError(f"{market}: forbidden target-day input in legal set")

    wanted = {o + pd.Timedelta(hours=h) for o in pd.to_datetime(origins) for h in range(H)}
    ts = read_timestamps_only(market)
    idx = {pd.Timestamp(t): i for i, t in enumerate(ts)}
    if not wanted <= set(idx):
        raise RuntimeError(f"{market}: legal-state timestamps not all present")
    keep = {idx[t] for t in wanted}
    df = read_projected(market, legal_cols, keep)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol).sort_index()
    if set(df.index) != wanted:
        raise RuntimeError(f"{market}: state reader escaped authorized timestamps")
    for c in legal_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not np.isfinite(df[legal_cols].to_numpy()).all():
        raise RuntimeError(f"{market}: non-finite legal state")
    z = np.stack([df.loc[[o + pd.Timedelta(hours=h) for h in range(H)], legal_cols]
                  .to_numpy(np.float32) for o in pd.to_datetime(origins)])
    return z, {
        "columns": list(legal_cols),
        "rows_read": len(df),
        "target_day_DA_price_input_reads": 0,
        "realized_future_input_reads": 0,
        "protected_final_reads": 0,
        "publication_semantics": ("all included columns are source-labelled 预测值 (forecast); "
                                  "竞价空间预测值 excluded because pre-DA publication time is unconfirmed"),
    }


def legal_columns(market: str) -> list[str]:
    header = read_header(market)
    tcol = MARKETS[market]["tcol"]
    out = []
    for c in header:
        c = c.strip()
        if c == tcol or "电价" in c or c in EXCLUDED_FEATURES:
            continue
        if c.endswith("预测值"):
            out.append(c)
    return out


def realized_columns(market: str) -> list[str]:
    header = read_header(market)
    tcol = MARKETS[market]["tcol"]
    return [c.strip() for c in header
            if c.strip() != tcol and ("实际值" in c or "电价" in c)]


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
