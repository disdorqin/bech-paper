"""Hurdle Correction Head (HCH) — a model-agnostic selective extreme-price corrector.

Signature (compatible with any frozen-backbone predictor):
    hch = HurdleCorrectionHead(neg_thr=0.0, seed=0)
    hch.fit(Z, yhat, y)           # S2: learn occurrence×magnitude decomposition
    hch.calibrate(Z, yhat, y)     # S3: paired risk certificate + shrinkage λ
    y_corrected, diag = hch.apply(Z, yhat)   # S4: selective correction

Design
------
Bi-Hurdle: for each signed tail (neg / pos), the head learns:
  1. OCCURRENCE:  P(event | Z)  — "should this hour be corrected?"
  2. MAGNITUDE:   E[y - yhat | event, Z]  — "by how much?"

The correction is δ = λ · P(event|Z) · E[y−yhat | event, Z], triggered only
when P(event|Z) > τ = 0.5 and the S3 certificate grants λ > 0.  Otherwise
output is exactly the frozen base (bit-exact identity fallback).

SCARR (Signed-tail Conformal Action-Risk Routing)
-------------------------------------------------
S3 chooses λ ∈ [0, 1] per branch to maximise a block-bootstrap LCB on mean
absolute-error reduction, subject to a conformal safety quantile constraint.
If no λ > 0 clears both tiers the branch ABSTAINS (λ = 0 → identity).

Related work (Hurdle / Tweedie / Zero-inflated):
  Hunt 2025 (arXiv:2509.08369), Wang et al. 2019 (arXiv:1912.07753),
  Kong et al. 2020 (arXiv:2010.16040), Wang & Gao 2023 (arXiv:2310.07435).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
BOOT_B = 500
BOOT_BLOCK = 24
MIN_EVENTS_CLF = 10
MIN_EVENTS_REG = 10

_BOOT_CACHE: dict = {}


def _boot_index(n: int, B: int, block: int, seed: int) -> np.ndarray:
    key = (n, B, block, seed)
    if key not in _BOOT_CACHE:
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(B, nb))
        offs = np.arange(block)
        _BOOT_CACHE[key] = (starts[:, :, None] + offs[None, None, :]).reshape(B, -1)[:, :n]
    return _BOOT_CACHE[key]


def block_bootstrap_lcb(g: np.ndarray, alpha: float = 0.10, B: int = BOOT_B,
                        block: int = BOOT_BLOCK, seed: int = 0) -> float:
    n = len(g)
    if n == 0:
        return 0.0
    if n <= block:
        return float(g.mean())
    idx = _boot_index(n, B, block, seed)
    return float(np.quantile(g[idx].mean(axis=1), alpha))


def _fit_clf(Z, lab, seed=0):
    import lightgbm as lgb
    pos = int(lab.sum())
    if pos < MIN_EVENTS_CLF or pos == len(lab):
        return None
    w = np.where(lab == 1, len(lab) / (2.0 * pos),
                 len(lab) / (2.0 * max(len(lab) - pos, 1)))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=8,
                           verbose=-1)
    m.fit(Z, lab, sample_weight=w)
    return m


def _fit_reg(Z, tgt, seed=0, objective="regression_l1"):
    import lightgbm as lgb
    if len(tgt) < MIN_EVENTS_REG:
        if not len(tgt):
            return 0.0
        return float(np.median(tgt))
    m = lgb.LGBMRegressor(objective=objective, n_estimators=300,
                          learning_rate=0.05, num_leaves=15,
                          min_child_samples=10, random_state=seed, n_jobs=8,
                          verbose=-1)
    m.fit(Z, tgt)
    return m


def _pred_reg(m, Z):
    if m is None:
        return np.zeros(len(Z))
    if isinstance(m, float):
        return np.full(len(Z), m)
    return m.predict(Z)


# ------------------------------------------------------------ corrector features
def build_corrector_features(
    X: np.ndarray, names: list[str], yhat: np.ndarray,
    y: np.ndarray, hour: np.ndarray, dayid: np.ndarray,
    oos_start: int,
) -> tuple[np.ndarray, list[str]]:
    n = len(yhat)
    cols, zn = [], []

    cols.append(yhat)
    zn.append("yhat")

    df = pd.DataFrame(dict(day=dayid, yhat=yhat))
    g = df.groupby("day")["yhat"]
    day_mean = g.transform("mean").to_numpy()
    day_min = g.transform("min").to_numpy()
    day_max = g.transform("max").to_numpy()
    day_std = g.transform("std").fillna(0.0).to_numpy()
    day_rank = g.rank(pct=True).to_numpy()
    cols += [day_mean, day_min, day_max, day_std, day_rank,
             yhat - day_mean, (yhat - day_min) / (day_max - day_min + EPS)]
    zn += ["yhat_daymean", "yhat_daymin", "yhat_daymax", "yhat_daystd",
           "yhat_dayrank", "yhat_dev_daymean", "yhat_daypos"]

    for nm in ("prevday_mean", "prevweek_mean", "prevweek_std", "prevday_min",
               "prevday_max"):
        if nm in names:
            v = X[:, names.index(nm)]
            cols.append(v); zn.append(nm)
            if nm.endswith("mean") or nm.endswith("min") or nm.endswith("max"):
                cols.append(yhat - v); zn.append(f"yhat_minus_{nm}")

    resid = np.full(n, np.nan)
    resid[oos_start:] = y[oos_start:] - yhat[oos_start:]
    rs = pd.Series(resid)
    for L in (24, 48, 168):
        cols.append(rs.shift(L).to_numpy()); zn.append(f"resid_lag{L}")
    roll = rs.shift(24).rolling(168, min_periods=24)
    cols += [roll.mean().to_numpy(), roll.std().to_numpy(),
             roll.min().to_numpy(), roll.max().to_numpy()]
    zn += ["resid_w_mean", "resid_w_std", "resid_w_min", "resid_w_max"]

    keep = [nm for nm in names if nm.startswith(
        ("fc_", "act_", "hour_", "dow_", "mon_", "is_weekend", "price_lag"))]
    for nm in keep:
        cols.append(X[:, names.index(nm)]); zn.append(f"x_{nm}")

    Z = np.column_stack(cols).astype(np.float64)
    avail = np.isfinite(Z).all(axis=1).astype(float)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.column_stack([Z, avail]); zn.append("feat_available")
    return Z, zn


# ----------------------------------------------------- HurdleCorrectionHead
class HurdleCorrectionHead:
    """Bi-Hurdle selective corrector.

    Parameters
    ----------
    neg_thr : absolute negative-price threshold (0 by market convention).
    alpha  : confidence level for action-risk certificate.
    rho    : Tier-2 safety budget: conformal harm quantile ≤ rho × baseline MAE.
    pos_q  : segment-relative quantile for positive-spike label (avoids
             absolute-threshold failures under regime drift).
    """

    def __init__(
        self,
        neg_thr: float = 0.0,
        alpha: float = 0.10,
        rho: float = 0.50,
        pos_q: float = 0.99,
        seed: int = 0,
    ):
        self.neg_thr = neg_thr
        self.alpha = alpha
        self.rho = rho
        self.pos_q = pos_q
        self.seed = seed
        self.tau = {"neg": 0.5, "pos": 0.5}
        self.lam = {"neg": 0.0, "pos": 0.0}
        self.clf: dict = {}
        self.reg: dict = {}
        self.info: dict = {}
        self._calib_extra = None

    # -------------------------------------------------------- fit (S2) ------
    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        n = len(y)
        cut = int(n * 0.75)
        a, b = np.arange(cut), np.arange(cut, n)
        spike_thr = float(np.quantile(y, self.pos_q))
        self.info["spike_thr"] = spike_thr

        lab = {
            "neg": (y < self.neg_thr).astype(int),
            "pos": (y > spike_thr).astype(int),
        }
        resid = y - yhat

        for br in ("neg", "pos"):
            L = lab[br]
            self.clf[br] = _fit_clf(Z[a], L[a], self.seed)
            if self.clf[br] is None:
                self.reg[br] = None
                continue
            ev = a[L[a] == 1]
            self.reg[br] = _fit_reg(Z[ev], resid[ev], self.seed,
                                    objective="regression_l1")
            self.info[f"n_{br}_events_S2"] = int(len(ev))

        self._calib_extra = (Z[b], yhat[b], y[b])
        return self

    def _prob(self, Z, br):
        m = self.clf.get(br)
        if m is None:
            return np.zeros(len(Z))
        return m.predict_proba(Z)[:, 1]

    def _raw_delta(self, Z, br, p):
        m = _pred_reg(self.reg.get(br), Z)
        return m * p

    # ---------------------------------------------------- calibrate (S3) ----
    def calibrate(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        if self._calib_extra is not None:
            Ze, he, ye = self._calib_extra
            Z = np.vstack([Ze, Z])
            yhat = np.concatenate([he, yhat])
            y = np.concatenate([ye, y])
        base = np.abs(y - yhat)
        n = len(y)
        grid = np.round(np.arange(0.0, 1.01, 0.05), 3)
        cal = {"_n_calib": int(n),
               "_s2b_reused": bool(self._calib_extra is not None)}
        for br in ("neg", "pos"):
            if self.clf.get(br) is None:
                self.lam[br] = 0.0
                cal[br] = dict(status="disabled(no training events)", n_fire=0)
                continue
            p = self._prob(Z, br)
            other = self._prob(Z, "pos" if br == "neg" else "neg")
            route = (p > self.tau[br]) & (p >= other)
            nfire = int(route.sum())
            if nfire < 10:
                self.lam[br] = 0.0
                cal[br] = dict(status="abstain(too few calib events)", n_fire=nfire)
                continue
            draw = self._raw_delta(Z, br, p)[route]
            yy, hh, bb = y[route], yhat[route], base[route]
            budget = float(self.rho * bb.mean())
            k = int(np.ceil((nfire + 1) * (1.0 - self.alpha)))
            best_lam, best_lcb, rec = 0.0, 0.0, []
            for lam in grid:
                H = np.abs(yy - (hh + lam * draw)) - bb
                q = float(np.sort(H)[k - 1]) if k <= nfire else np.inf
                g_full = np.zeros(n)
                g_full[route] = -H
                lcb = block_bootstrap_lcb(g_full, self.alpha, seed=self.seed)
                rec.append((float(lam), q, float(-H.mean()), lcb))
                feasible = (q <= budget) and (lcb > 0)
                if not feasible:
                    continue
                if lcb > best_lcb:
                    best_lam, best_lcb = float(lam), lcb
            self.lam[br] = best_lam
            cal[br] = dict(
                status="certified" if best_lam > 0 else "abstain(no lambda certified)",
                n_fire=nfire, lam=best_lam, lcb_mean_gain=round(best_lcb, 5),
                conformal_rank=k, branch_base_mae=float(bb.mean()),
                harm_budget=budget, rho=self.rho, alpha=self.alpha,
                grid=[dict(lam=l, harm_q=None if not np.isfinite(q) else round(q, 3),
                           mean_gain=round(g, 4), lcb=round(lc, 5))
                      for l, q, g, lc in rec[::4]])
        self.info["scarr"] = cal
        return self

    # -------------------------------------------------------- apply (S4) ----
    def apply(self, Z: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, dict]:
        p_neg = self._prob(Z, "neg")
        p_pos = self._prob(Z, "pos")
        delta = np.zeros(len(yhat))
        route = np.zeros(len(yhat), dtype=int)

        fire_neg = (p_neg > self.tau["neg"]) & (p_neg >= p_pos) & (self.lam["neg"] > 0)
        fire_pos = (p_pos > self.tau["pos"]) & (p_pos > p_neg) & (self.lam["pos"] > 0)
        if fire_neg.any():
            delta[fire_neg] = self.lam["neg"] * self._raw_delta(Z, "neg", p_neg)[fire_neg]
            route[fire_neg] = -1
        if fire_pos.any():
            delta[fire_pos] = self.lam["pos"] * self._raw_delta(Z, "pos", p_pos)[fire_pos]
            route[fire_pos] = 1

        diag = dict(
            n_fire_neg=int(fire_neg.sum()),
            n_fire_pos=int(fire_pos.sum()),
            n_total=int(len(yhat)),
            fire_rate=float((route != 0).mean()),
            lam_neg=self.lam["neg"],
            lam_pos=self.lam["pos"],
            tau_neg=self.tau["neg"],
            tau_pos=self.tau["pos"],
        )
        return yhat + delta, diag
