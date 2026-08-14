"""Phase 3 common utilities: public-data loading, cutoff-safe features,
four-segment rolling-origin isolation, and evaluation metrics.

DESIGN CONTRACT (anti-leakage, non-negotiable)
----------------------------------------------
Day-ahead convention (identical to Lago 2021 open benchmark):
  For a target hour t on delivery day D, the information set available at the
  forecast cutoff (day D-1) is:
    * all prices up to and including day D-1        -> y_{t-24}, y_{t-48}, ...
    * exogenous FORECASTS for day D                 -> load fc, RE fc at time t
    * exogenous ACTUALS up to day D-1               -> must be lagged >= 24h
    * calendar variables for day D                  -> always known
  y_t itself and any function of it NEVER enters the feature matrix.
  (This is the L101 lesson: a single y_true-derived feature invalidates a run.)

Four-segment rolling-origin isolation:
  S1 backbone-train | S2 corrector-train | S3 conformal-calibration | S4 test
  chronological, non-overlapping. The backbone is FROZEN after S1, so its
  predictions on S2/S3/S4 are genuinely out-of-sample; the corrector is frozen
  after S2; the conformal routing is frozen after S3.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

# ---------------------------------------------------------------- datasets ---
# exog_fc  : exogenous variables that are FORECASTS for the target hour -> safe at t
# exog_act : exogenous variables that are ACTUALS -> must be lagged >= 24h
DATASETS = {
    # ---- Lago 2021 open benchmark (Zenodo 4624804) -- EPF de-facto standard
    "LAGO_DE":  dict(path="raw/lago_benchmark/DE.csv",  currency="EUR",
                     tier="L1", note="EPEX-DE, 1.03% neg price, main EU market"),
    "LAGO_BE":  dict(path="raw/lago_benchmark/BE.csv",  currency="EUR", tier="L2"),
    "LAGO_FR":  dict(path="raw/lago_benchmark/FR.csv",  currency="EUR", tier="L2"),
    "LAGO_PJM": dict(path="raw/lago_benchmark/PJM.csv", currency="USD", tier="L2"),
    "LAGO_NP":  dict(path="raw/lago_benchmark/NP.csv",  currency="EUR",
                     tier="L3", note="NEGATIVE CONTROL: 0% negative price"),
    # ---- GEFCom2014-P (classic competition benchmark, historical regime)
    "GEFCOM14P": dict(path="raw/gefcom2014/GEFCom2014P_hourly.csv", currency="USD",
                       tier="L3", note="NEGATIVE CONTROL: 0.000% negative price (2011-2013)"),
    # ---- NEM SA1 2024 (high negative-price market for validation)
    "NEM_SA1":  dict(path="raw/nem_aemo/clean/SA1_2024_hourly.csv", currency="AUD",
                     tier="L1", note="SA1 2024: 26.0% negative price hours, -718..14131"),
    # ---- EPEX-DE 2020-2025 (German day-ahead, Energi Data Service)
    "DE_EPEX":  dict(path="raw/epex_markets/DE_EPEX_2020_2025.csv", currency="EUR",
                     tier="L1", note="EPEX-DE 2020-2025: 3.6% neg (2025:8%), -500..2326"),
    # ---- PJM 2020-2025 (US day-ahead, UniElecPrice, 0% neg — normal control)
    "PJM_2020":  dict(path="raw/epex_markets/PJM_2020_2025.csv", currency="USD",
                      tier="L2", note="PJM 2020-2025: 0% negative price, normal market control"),
    # ---- European power markets 2022-2024 (UniElecPrice)
    "EPEX_FR":   dict(path="raw/epex_markets/EPEX_FR_2022_2024.csv", currency="EUR", tier="L1"),
    "EPEX_BE":   dict(path="raw/epex_markets/EPEX_BE_2022_2024.csv", currency="EUR", tier="L1"),
    "EPEX_NL":   dict(path="raw/epex_markets/EPEX_NL_2022_2024.csv", currency="EUR", tier="L1"),
    "NORD_FI":   dict(path="raw/epex_markets/Finland_2022_2024.csv", currency="EUR", tier="L1"),
    "NORD_NO":   dict(path="raw/epex_markets/NordPool_NO_2022_2024.csv", currency="EUR", tier="L1"),
    "NORD_SE3":  dict(path="raw/epex_markets/NordPool_SE3_2022_2024.csv", currency="EUR", tier="L1"),
    "NORD_DK1":  dict(path="raw/epex_markets/NordPool_DK1_2022_2024.csv", currency="EUR", tier="L1"),
}

# 国内省份注册表(不含 shandong,其走 load_shandong)。列方言与 shandong 相同:
# *预测值 → exog_fc,*实际值 → exog_act,日前/实时电价 → target。文件均为 xlsx。
PROVINCES = {
    "ningxia": dict(file="宁夏24h电价数据集.xlsx",   currency="CNY", note="NX 2025-10..2026-04, 0% negative"),
    "gansu":   dict(file="甘肃24h电价数据集.xlsx",   currency="CNY", note="GS 2024-07..2026-04, 0% negative"),
    "shaanxi": dict(file="陕西24h电价数据集(1).xlsx", currency="CNY", note="SN 2025-01..2026-03, 0% negative"),
    "qinghai": dict(file="青海24h电价数据集.xlsx",   currency="CNY", note="QH 2025-09..2026-04, 0% negative"),
}
PROVINCE_KEYS = [f"{p}_{m}" for p in PROVINCES for m in ("DA", "RT")]   # 8 keys

_LAGO_EXOG_FC = True   # Lago columns are all "* Forecast" -> safe at target hour


def load_dataset(key: str) -> dict:
    """Return dict(ts, price, exog_fc:DataFrame, exog_act:DataFrame, meta)."""
    spec = DATASETS[key]
    path = os.path.join(DATA, spec["path"])
    df = pd.read_csv(path)

    if key.startswith("LAGO_"):
        first = df.columns[0]
        df = df.rename(columns={first: "timestamp"})
        price_col = [c for c in df.columns if c.lower().startswith("price")
                     or "price" in c.lower()][0]
        exog_cols = [c for c in df.columns if c not in ("timestamp", price_col)]
        exog_fc, exog_act = exog_cols, []
    elif key.startswith("NEM_"):
        price_col = "price"
        # AEMO TOTALDEMAND/column names may vary
        demand_cols = [c for c in df.columns if "DEMAND" in c.upper() or c == "demand"]
        if demand_cols:
            df = df.rename(columns={demand_cols[0]: "demand"})
        exog_fc, exog_act = [], ["demand"] if "demand" in df.columns else []
    elif key == "GEFCOM14P":
        price_col = "price"
        # GEFCom2014-P exogenous loads are FORECASTS by competition design
        exog_fc, exog_act = ["load_system_fc", "load_zonal_fc"], []
    elif key in ("DE_EPEX", "PJM_2020", "EPEX_FR", "EPEX_BE", "EPEX_NL",
                 "NORD_FI", "NORD_NO", "NORD_SE3", "NORD_DK1"):
        price_col = "price"
        exog_fc, exog_act = [], []
    else:
        raise KeyError(key)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    for c in exog_fc + exog_act:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].ffill().bfill()

    return dict(
        key=key, ts=df["timestamp"], price=df[price_col].astype(float).to_numpy(),
        exog_fc=df[exog_fc].astype(float) if exog_fc else pd.DataFrame(index=df.index),
        exog_act=df[exog_act].astype(float) if exog_act else pd.DataFrame(index=df.index),
        meta=spec,
    )


# ---------------------------------------------------------------- features ---
PRICE_LAGS = (24, 48, 72, 168)     # Lago standard: D-1, D-2, D-3, D-7
ACT_LAGS = (24, 168)               # actual exogenous must be lagged >= 24h
SEQ_LEN = 168                      # sequence window for LSTM / Transformer


def build_tabular(ds: dict) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Cutoff-safe tabular design matrix. Returns X, y, names, valid_idx."""
    y = ds["price"]
    ts = ds["ts"]
    n = len(y)
    cols, names = [], []

    s = pd.Series(y)
    for L in PRICE_LAGS:
        cols.append(s.shift(L).to_numpy()); names.append(f"price_lag{L}")
    # aggregates of the last fully-observed day (D-1 -> lags 24..47)
    prev_day = s.shift(24).rolling(24, min_periods=24)
    cols += [prev_day.mean().to_numpy(), prev_day.min().to_numpy(),
             prev_day.max().to_numpy(), prev_day.std().to_numpy()]
    names += ["prevday_mean", "prevday_min", "prevday_max", "prevday_std"]
    # weekly stats (lag 24 .. 191)
    prev_week = s.shift(24).rolling(168, min_periods=168)
    cols += [prev_week.mean().to_numpy(), prev_week.std().to_numpy()]
    names += ["prevweek_mean", "prevweek_std"]

    for c in ds["exog_fc"].columns:                       # forecast -> usable at t
        v = ds["exog_fc"][c].to_numpy()
        cols.append(v); names.append(f"fc_{c}")
        cols.append(pd.Series(v).shift(24).to_numpy()); names.append(f"fc_{c}_lag24")
    for c in ds["exog_act"].columns:                      # actual -> must lag
        v = pd.Series(ds["exog_act"][c].to_numpy())
        for L in ACT_LAGS:
            cols.append(v.shift(L).to_numpy()); names.append(f"act_{c}_lag{L}")

    hr = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    mon = ts.dt.month.to_numpy()
    cols += [np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
             np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
             np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
             (dow >= 5).astype(float)]
    names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "mon_sin", "mon_cos", "is_weekend"]

    X = np.column_stack(cols).astype(np.float64)
    warm = max(max(PRICE_LAGS), 24 + 168, max(ACT_LAGS) if ACT_LAGS else 0)
    valid = np.arange(warm, n)
    ok = np.isfinite(X[valid]).all(axis=1)
    valid = valid[ok]
    return X[valid], y[valid], names, valid


def build_sequences(ds: dict, valid: np.ndarray) -> np.ndarray:
    """(N, SEQ_LEN) price window ending at t-24 -> strictly <= cutoff."""
    y = ds["price"]
    out = np.empty((len(valid), SEQ_LEN), dtype=np.float32)
    for i, t in enumerate(valid):
        out[i] = y[t - 24 - SEQ_LEN + 1: t - 24 + 1]
    return out


def assert_no_leakage(ds: dict, X: np.ndarray, y: np.ndarray, valid: np.ndarray,
                      names: list[str]) -> None:
    """Hard guard: no column may correlate ~perfectly with y_t, and a shuffled
    target must destroy nothing (columns are functions of the past only)."""
    yv = y
    for j, nm in enumerate(names):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        r = abs(np.corrcoef(col, yv)[0, 1])
        assert r < 0.999, f"LEAK SUSPECT: column {nm} corr={r:.5f} with y_t"
    # structural check: recompute a lag column directly and compare
    j = names.index("price_lag24")
    ref = ds["price"][valid - 24]
    assert np.allclose(X[:, j], ref, equal_nan=False), "price_lag24 misaligned"


def assert_no_future_leakage(X: np.ndarray, y: np.ndarray, names: list[str],
                             future_offsets: tuple[int, ...] = (1, 2, 24, 48, 168)
                             ) -> dict:
    """Hard guard (user requirement): no design-matrix column may be a function
    of any FUTURE price y_{t+k} (k>=1).  For each column compare against y
    shifted by future offsets (correlation + exact-value collision) over valid
    rows.  Raises AssertionError on a near-perfect future match; otherwise
    returns a per-column audit dict for the admission report.

    X / y are the build_tabular-compressed matrices (valid rows are consecutive
    hourly timestamps after warm-up, so shifting within the array aligns to
    future hours; sparse gaps from dropped non-finite rows only widen the
    safety margin against false positives, never against false negatives).
    """
    report = {}
    yv = np.asarray(y, dtype=np.float64)
    n = len(yv)
    for j, nm in enumerate(names):
        col = np.asarray(X[:, j], dtype=np.float64)
        if np.std(col) < 1e-12:
            report[nm] = {"leak": False, "kind": "const"}
            continue
        best = None  # (corr, exact_frac, offset) over future offsets
        for k in future_offsets:
            fut = np.full(n, np.nan, dtype=np.float64)
            fut[: n - k] = yv[k:]
            m = np.isfinite(col) & np.isfinite(fut)
            if m.sum() < 50:
                continue
            r = abs(np.corrcoef(col[m], fut[m])[0, 1])
            exact = float(np.mean(np.abs(col[m] - fut[m]) < 1e-6))
            if best is None or r > best[0]:
                best = (r, exact, int(k))
        if best is None:
            report[nm] = {"leak": False, "kind": "no_overlap"}
            continue
        r, exact, k = best
        if r >= 0.999 or exact >= 0.999:
            raise AssertionError(
                f"FUTURE PRICE LEAK: column {nm} ~= y[t+{k}] (corr={r:.5f}, "
                f"exact={exact:.3f})")
        report[nm] = {"leak": False, "kind": "ok", "max_future_corr": float(r),
                      "argmax_offset": int(k), "exact_frac": exact}
    return report


def verify_forecast_vs_actual(ds: dict) -> dict:
    """Per-column: is each fc_{c} a genuine forecast or a backfilled actual?

    A real day-ahead forecast deviates from the same-hour realized value; if a
    '预测值' column is (near-)identical to its '实际值' twin, it is a backfilled
    actual and must NOT be treated as forecast-visible at t. Returns
    {fc_col: {kind: genuine|backfilled_actual|no_act_pair, corr, exact}}.
    """
    out = {}
    fc, act = ds["exog_fc"], ds["exog_act"]
    act_map = {c.replace("预测值", "实际值"): c for c in fc.columns}
    for c in fc.columns:
        pair = act_map.get(c)
        v = pd.to_numeric(fc[c], errors="coerce").to_numpy(float)
        if pair is None or pair not in act.columns:
            out[c] = {"kind": "no_act_pair", "corr": None, "exact": None}
            continue
        a = pd.to_numeric(act[pair], errors="coerce").to_numpy(float)
        m = np.isfinite(v) & np.isfinite(a)
        if m.sum() < 50:
            out[c] = {"kind": "no_act_pair", "corr": None, "exact": None}
            continue
        corr = abs(float(np.corrcoef(v[m], a[m])[0, 1]))
        exact = float(np.mean(np.abs(v[m] - a[m]) < 1e-6))
        out[c] = {"kind": ("backfilled_actual" if (corr >= 0.999 or exact >= 0.999)
                           else "genuine"),
                  "corr": corr, "exact": exact}
    return out


# ------------------------------------------------------------------ splits ---
def four_segment_split(n: int, frac=(0.50, 0.20, 0.10, 0.20)) -> dict:
    a = int(n * frac[0])
    b = a + int(n * frac[1])
    c = b + int(n * frac[2])
    return dict(S1=np.arange(0, a), S2=np.arange(a, b),
                S3=np.arange(b, c), S4=np.arange(c, n))


# ----------------------------------------------------------------- metrics ---
def _hac_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    g0 = float((d * d).sum() / n)
    v = g0
    for k in range(1, lag + 1):
        gk = float((d[k:] * d[:-k]).sum() / n)
        v += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(v, 1e-12)


def dm_test(e_base: np.ndarray, e_new: np.ndarray, lag: int = 24) -> dict:
    """One-sided Diebold-Mariano on absolute errors. H1: new is better."""
    d = np.abs(e_base) - np.abs(e_new)
    n = len(d)
    if n < 30 or np.allclose(d, 0):
        return dict(dm_stat=None, p_value=None, mean_gain=float(d.mean()) if n else None)
    var = _hac_var(d, lag)
    stat = float(d.mean() / np.sqrt(var / n))
    from math import erf, sqrt
    p = 1.0 - 0.5 * (1.0 + erf(stat / sqrt(2.0)))   # one-sided upper tail
    return dict(dm_stat=stat, p_value=float(p), mean_gain=float(d.mean()))


def evaluate(y: np.ndarray, pred: np.ndarray, naive: np.ndarray | None,
             neg_thr: float = 0.0, spike_thr: float | None = None) -> dict:
    err = pred - y
    ae = np.abs(err)
    out = dict(n=int(len(y)), mae=float(ae.mean()),
               rmse=float(np.sqrt((err ** 2).mean())))
    if naive is not None:
        nm = float(np.abs(naive - y).mean())
        out["rmae"] = float(out["mae"] / nm) if nm > 0 else None
    # negative-price branch
    neg = y < neg_thr
    out["neg_n"] = int(neg.sum())
    if neg.sum() > 0:
        out["neg_miss_rate"] = float((pred[neg] >= neg_thr).mean())
        out["mae_on_neg"] = float(ae[neg].mean())
        out["bias_on_neg"] = float(err[neg].mean())
    else:
        out["neg_miss_rate"] = out["mae_on_neg"] = out["bias_on_neg"] = None
    # positive spike branch
    if spike_thr is None:
        spike_thr = float(np.quantile(y, 0.99))
    sp = y > spike_thr
    out["spike_thr"] = float(spike_thr)
    out["spike_n"] = int(sp.sum())
    out["mae_on_spike"] = float(ae[sp].mean()) if sp.sum() else None
    out["spike_miss_rate"] = float((pred[sp] <= spike_thr).mean()) if sp.sum() else None
    # tail of the error distribution (worst 5%)
    k = max(1, int(0.05 * len(ae)))
    out["tail_rmse"] = float(np.sqrt((np.sort(ae)[-k:] ** 2).mean()))
    # normal regime (neither negative nor spike) -- degradation budget check
    normal = (~neg) & (~sp)
    out["mae_on_normal"] = float(ae[normal].mean()) if normal.sum() else None
    return out


def weekly_naive(y_full: np.ndarray, valid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Lago-recommended rMAE denominator: seasonal (weekly) naive forecast."""
    t = valid[idx]
    return y_full[t - 168]


# ============================================================= 山东数据加载 ===
def load_shandong(
    price_col: str = "日前电价",
    encoding: str = "gbk",
) -> dict:
    """Load Shandong hourly data (23 columns, 2022-2026).

    Returns dict(ts, price, exog_fc:DataFrame, exog_act:DataFrame, meta)
    with all exogenous columns split into fc (预测值) and act (实际值).
    """
    path = os.path.join(DATA, "raw", "provinces", "shandong_pmos_hourly.csv")
    df = pd.read_csv(path, encoding=encoding)
    df.columns = df.columns.str.strip()
    ts_col = [c for c in df.columns if "时刻" in c or "time" in c.lower()][0]
    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found; available: {list(df.columns)}")

    # exogenous: all cols except timestamp, prices, 全省负荷预测总值
    price_cols = {c for c in df.columns if "电价" in c}
    exog_cols = [c for c in df.columns if c not in price_cols
                 and "时刻" not in c and "timestamp" not in c]

    exog_fc = [c for c in exog_cols if "预测" in c]
    exog_act = [c for c in exog_cols if "实际" in c and c not in exog_fc]

    for c in exog_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].ffill().bfill()

    price = pd.to_numeric(df[price_col], errors="coerce").ffill().bfill().to_numpy()

    return dict(
        key="shandong",
        ts=df["timestamp"],
        price=price.astype(float),
        exog_fc=df[exog_fc].astype(float) if exog_fc else pd.DataFrame(index=df.index),
        exog_act=df[exog_act].astype(float) if exog_act else pd.DataFrame(index=df.index),
        meta=dict(path=path, price_col=price_col,
                  n_cols=len(df.columns), currency="CNY",
                  tier="L1",
                  note="Shandong spot market, 11-13% negative price hours"),
    )


# =============================================== 通用省份加载(非 shandong) ====
def load_province(province: str, price_col: str = "日前电价") -> dict:
    """Generic provincial loader for the four non-Shandong province files.

    Same column dialect as load_shandong: 时刻→ts, *预测值→exog_fc,
    *实际值→exog_act, 日前/实时电价→price. Files are .xlsx. Returns the same
    dict schema as load_dataset / load_shandong, so build_tabular /
    assert_no_leakage / ExperimentManifest work unchanged. Each province trains
    its own host on its own column set — no cross-province schema alignment.
    """
    if province not in PROVINCES:
        raise KeyError(f"unknown province {province!r}; have {list(PROVINCES)}")
    spec = PROVINCES[province]
    path = os.path.join(DATA, "raw", "provinces", spec["file"])
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    ts_col = [c for c in df.columns if "时刻" in c or "time" in c.lower()][0]
    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found in {province}; "
                       f"available: {list(df.columns)}")

    price_cols = {c for c in df.columns if "电价" in c}
    exog_cols = [c for c in df.columns if c not in price_cols
                 and "时刻" not in c and "timestamp" not in c]
    exog_fc = [c for c in exog_cols if "预测" in c]
    exog_act = [c for c in exog_cols if "实际" in c and c not in exog_fc]
    for c in exog_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].ffill().bfill()

    price = pd.to_numeric(df[price_col], errors="coerce").ffill().bfill().to_numpy()
    return dict(
        key=province, ts=df["timestamp"], price=price.astype(float),
        exog_fc=df[exog_fc].astype(float) if exog_fc else pd.DataFrame(index=df.index),
        exog_act=df[exog_act].astype(float) if exog_act else pd.DataFrame(index=df.index),
        meta=dict(path=path, price_col=price_col, n_cols=len(df.columns),
                  currency=spec["currency"], tier="L1", note=spec["note"],
                  province=province),
    )


# ===================================================== episode 评估指标 ====
def extract_episodes(prices: np.ndarray, threshold: float = 0.0) -> list[tuple[int, int]]:
    """Extract maximal contiguous intervals where price < threshold.

    Returns list of (start_idx, end_idx) inclusive.
    """
    mask = prices < threshold
    episodes = []
    in_ep = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_ep:
            start = i
            in_ep = True
        if not m and in_ep:
            episodes.append((start, i - 1))
            in_ep = False
    if in_ep:
        episodes.append((start, len(prices) - 1))
    return episodes


def episode_match(
    true_eps: list[tuple[int, int]],
    pred_eps: list[tuple[int, int]],
    max_boundary_gap: int = 3,
) -> dict:
    """Match predicted episodes to true episodes via greedy overlap + dummy.

    Returns dict with:
      n_true, n_pred, n_matched, n_miss (true unmatched), n_fp (pred unmatched),
      boundary_mae, matched_pairs (list of (ti, pi)),
      recall: n_matched / n_true,  complete_miss_rate: n_miss / n_true.
    """
    from scipy.optimize import linear_sum_assignment

    nt, np_ = len(true_eps), len(pred_eps)
    if nt == 0 and np_ == 0:
        return dict(n_true=0, n_pred=0, n_matched=0, n_miss=0, n_fp=0,
                    boundary_mae=0.0, recall=1.0, complete_miss_rate=0.0)

    if nt == 0:
        return dict(n_true=0, n_pred=np_, n_matched=0, n_miss=0, n_fp=np_,
                    boundary_mae=0.0, recall=1.0, complete_miss_rate=0.0)

    if np_ == 0:
        return dict(n_true=nt, n_pred=0, n_matched=0, n_miss=nt, n_fp=0,
                    boundary_mae=0.0, recall=0.0, complete_miss_rate=1.0)

    # cost matrix: overlap-based (negative IoU)
    cost = np.ones((nt, np_)) * 1e6
    for i, (ts, te) in enumerate(true_eps):
        for j, (ps, pe) in enumerate(pred_eps):
            intersection = min(te, pe) - max(ts, ps) + 1
            union = max(te, pe) - min(ts, ps) + 1
            if intersection > 0:
                cost[i, j] = 1.0 - intersection / union

    row_ind, col_ind = linear_sum_assignment(cost)
    n_matched = 0
    boundary_sum = 0.0
    matched = []
    true_matched = set()
    pred_matched = set()
    for i, j in zip(row_ind, col_ind):
        ts, te = true_eps[i]
        ps, pe = pred_eps[j]
        gap = abs(ts - ps) + abs(te - pe)
        if gap <= max_boundary_gap * 2 and cost[i, j] < 1e5:
            n_matched += 1
            boundary_sum += gap / 2.0
            matched.append((i, j))
            true_matched.add(i)
            pred_matched.add(j)

    n_miss = nt - len(true_matched)
    n_fp = np_ - len(pred_matched)
    boundary_mae = boundary_sum / n_matched if n_matched else 0.0
    recall = n_matched / nt if nt else 1.0
    complete_miss_rate = n_miss / nt if nt else 0.0

    return dict(
        n_true=nt, n_pred=np_, n_matched=n_matched,
        n_miss=n_miss, n_fp=n_fp, boundary_mae=boundary_mae,
        recall=recall, complete_miss_rate=complete_miss_rate,
    )


def episode_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    base: np.ndarray,
    neg_thr: float = 0.0,
) -> dict:
    """Compute episode-level evaluation for predictions vs base.

    Returns dict with base/pred episode counts, recall, complete-miss,
    false-episode rate, boundary MAE.
    """
    true_eps = extract_episodes(y, neg_thr)
    pred_eps = extract_episodes(pred, neg_thr)
    base_eps = extract_episodes(base, neg_thr)

    pred_match = episode_match(true_eps, pred_eps)
    base_match = episode_match(true_eps, base_eps)

    return dict(
        n_true_episodes=len(true_eps),
        n_base_episodes=len(base_eps),
        n_pred_episodes=len(pred_eps),
        base_episode_recall=base_match["recall"],
        our_episode_recall=pred_match["recall"],
        base_complete_miss=base_match["complete_miss_rate"],
        our_complete_miss=pred_match["complete_miss_rate"],
        base_false_events=base_match["n_fp"],
        our_false_events=pred_match["n_fp"],
        base_boundary_mae=base_match["boundary_mae"],
        our_boundary_mae=pred_match["boundary_mae"],
    )
