"""R1A.9 Action-Threshold / Utility Calibration Audit (plan v0.1).

Question (plan §0): a CRPS-good 3-atom predictive distribution — is its
translation to reliable action-utility only a tiny, monotone, action-relevant
post-hoc calibration away?

Structure:
  B0  Raw threshold calibration: P(B^±=1 | w^±=p) ?= p, esp. near p=0.5.
  C0  Raw IAH  (s̃ = s)                         — == current production action.
  C1  Shared monotone affine (4 scalars, source equal-weighted).
  C2  Local monotone affine  (per market:host, diagnostic upper bound).
  C3  Local monotone isotonic (per market:host, low-capacity upper bound).
Then each calibrator is pushed through the EXISTING double-event optimizer
(§7) and refits its OWN DVG q on S3C (§8). S4 = development confirmation
only. Verdict ∈ {UNIVERSAL_ACTION_CALIBRATION_SUPPORTED,
LOCAL_ACTION_CALIBRATION_SUPPORTED, ACTION_CALIBRATION_PARTIAL,
MONOTONE_CALIBRATION_INSUFFICIENT}.

No candidate retrain; no CRPS/DVG/CAGM math change; no new production module.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import r1a5_diag as D
import r1a_run as R
import r1a6_value_recovery as V6
from hch_v2_bundle import HCHV2Bundle
from double_event import double_event_proposal
from query_replay import estimate_realized_A, form_final_pi

EPS = 1e-12
FOCUS = ("LAGO_PJM:MLP", "LAGO_PJM:Linear", "NEM_SA1:MLP", "LAGO_DE:MLP")


# ---------------------------------------------------------------- helpers ----
def _fl(t) -> np.ndarray:
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a.reshape(-1).astype(np.float64)


def _iso(d) -> str:
    return str(pd.Timestamp(d).date())


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _auc(score, label) -> float | None:
    try:
        return float(roc_auc_score(np.asarray(label).ravel(),
                                   np.asarray(score, dtype=float).ravel()))
    except Exception:
        return None


def _pr_auc(score, label) -> float | None:
    try:
        return float(average_precision_score(np.asarray(label).ravel(),
                      np.asarray(score, dtype=float).ravel()))
    except Exception:
        return None


def _brier(prob, label) -> float:
    p = np.asarray(prob, dtype=float).ravel()
    l = np.asarray(label, dtype=float).ravel()
    return float(np.mean((p - l) ** 2))


def _ece(prob, label, nbins=10) -> float:
    p = np.asarray(prob, dtype=float).ravel()
    l = np.asarray(label, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    ece = 0.0
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - l[m].mean())
    return float(ece)


def _rel_bins(score, label, nbins=10):
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=float).ravel()
    qs = np.quantile(score, np.linspace(0, 1, nbins + 1))
    out = []
    for i in range(nbins):
        lo = -np.inf if i == 0 else qs[i]
        hi = np.inf if i == nbins - 1 else qs[i + 1]
        m = (score >= lo) & (score <= hi) if i == nbins - 1 \
            else (score >= lo) & (score < hi)
        out.append({"bin": i, "lo": float(lo), "hi": float(hi),
                    "n": int(m.sum()),
                    "mean_score": float(score[m].mean()) if m.sum() else None,
                    "mean_label": float(label[m].mean()) if m.sum() else None})
    return out


def _sp(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(stats.spearmanr(a, b)[0])


def _order_stat_q(errors, alpha) -> float:
    E = np.sort(np.asarray(errors, dtype=np.float64))
    n = len(E)
    rnk = int(np.ceil((1.0 - alpha) * (n + 1)))
    rnk = min(rnk, n)
    return float(E[rnk - 1])


# ------------------------------------------------------------ calibrators ----
class Calibrator:
    name = "base"

    def fit(self, train_slices) -> None:  # list of dicts
        pass

    def apply(self, s, direction, domain=None) -> np.ndarray:
        raise NotImplementedError


class RawIAH(Calibrator):
    name = "C0_raw"

    def apply(self, s, direction, domain=None):
        return np.asarray(s, dtype=float)


def _clip_affine(s, a, b):
    return np.clip(a * s + b, -1.0, 1.0)


def _fit_affine(s, Y, wts):
    """Minimize weighted MSE of clip(a*s+b,-1,1) vs Y, a>=0."""
    def obj(p):
        a, b = p
        return float(np.mean(wts * (_clip_affine(s, a, b) - Y) ** 2))
    res = minimize(obj, np.array([1.0, 0.0]), method="L-BFGS-B",
                   bounds=[(0.0, 10.0), (-5.0, 5.0)])
    return float(res.x[0]), float(res.x[1])


class SharedAffine(Calibrator):
    """C1 — four shared scalars, source domains equal-weighted."""
    name = "C1_shared_affine"

    def fit(self, train_slices):
        for direction, sk, yk in (("d", "sd", "Yd"), ("u", "su", "Yu")):
            ss, yy, ww = [], [], []
            nd = len([1 for sl in train_slices if len(sl[sk])])
            for sl in train_slices:
                s = np.asarray(sl[sk], dtype=float)
                y = np.asarray(sl[yk], dtype=float)
                mk = np.isfinite(s) & np.isfinite(y)
                s, y = s[mk], y[mk]
                if len(s) == 0:
                    continue
                w = np.full(len(s), 1.0 / (nd * len(s)))
                ss.append(s); yy.append(y); ww.append(w)
            if not ss:
                self._set(direction, 1.0, 0.0)
                continue
            s_all = np.concatenate(ss); y_all = np.concatenate(yy)
            w_all = np.concatenate(ww)
            a, b = _fit_affine(s_all, y_all, w_all)
            self._set(direction, a, b)

    def _set(self, direction, a, b):
        self.params = getattr(self, "params", {})
        self.params[direction] = (a, b)

    def apply(self, s, direction, domain=None):
        a, b = getattr(self, "params", {}).get(direction, (1.0, 0.0))
        return _clip_affine(np.asarray(s, dtype=float), a, b)


class LocalAffine(Calibrator):
    """C2 — per (market,host) direction affine map (diagnostic upper bound)."""
    name = "C2_local_affine"

    def fit(self, train_slices):
        self.params = {}
        for sl in train_slices:
            dom = sl["domain"]
            self.params[dom] = {}
            for direction, sk, yk in (("d", "sd", "Yd"), ("u", "su", "Yu")):
                s = np.asarray(sl[sk], dtype=float)
                y = np.asarray(sl[yk], dtype=float)
                mk = np.isfinite(s) & np.isfinite(y)
                s, y = s[mk], y[mk]
                if len(s) < 4:
                    self.params[dom][direction] = (1.0, 0.0)
                    continue
                self.params[dom][direction] = _fit_affine(s, y, np.ones(len(s)))

    def apply(self, s, direction, domain=None):
        a, b = self.params.get(domain, {}).get(direction, (1.0, 0.0))
        return _clip_affine(np.asarray(s, dtype=float), a, b)


class LocalIsotonic(Calibrator):
    """C3 — per (market,host) monotone isotonic (low-capacity upper bound)."""
    name = "C3_local_isotonic"

    def fit(self, train_slices):
        self.params = {}
        for sl in train_slices:
            dom = sl["domain"]
            self.params[dom] = {}
            for direction, sk, yk in (("d", "sd", "Yd"), ("u", "su", "Yu")):
                s = np.asarray(sl[sk], dtype=float)
                y = np.asarray(sl[yk], dtype=float)
                mk = np.isfinite(s) & np.isfinite(y)
                s, y = s[mk], y[mk]
                if len(s) < 8:
                    self.params[dom][direction] = None
                    continue
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso.fit(s, y)
                self.params[dom][direction] = iso

    def apply(self, s, direction, domain=None):
        iso = self.params.get(domain, {}).get(direction)
        s = np.asarray(s, dtype=float)
        if iso is None:
            return np.clip(s, -1.0, 1.0)
        return np.clip(iso.predict(s), -1.0, 1.0)


# --------------------------------------------------------- data collection ----
def collect_domain(artifact_dir: Path, ds_key: str, bb: str,
                   variant: str = "learned_sig",
                   head: Optional[torch.nn.Module] = None,
                   info: Optional["DomainInfo"] = None) -> dict:
    """Build day-level action-chain domain data.

    head is None (default): load the frozen R1A checkpoint from artifact_dir.
    head given (Stage-2D): rebuild the pipe from the in-memory trained head.
    info given: reuse a pre-built DomainInfo (head-independent; avoids re-running
      prepare_domain per head eval). Same results, faster.
    """
    old = D.NEW_TO_OLD[variant]
    domain = f"{ds_key}:{bb}"
    if info is None:
        info = R.prepare_domain(ds_key, bb)
    det_np = R.det_for_variant(variant, info)
    if head is not None:
        pipe, _val_days, _s3c_days, problems = D.pipe_from_head(head, info,
                                                                variant)
    else:
        bundle = HCHV2Bundle.load(str(artifact_dir / f"checkpoint_{old}_{ds_key}_{bb}.pt"))
        pipe, _val_days, _s3c_days, problems = D.rebuild_pipe_from_bundle(
            bundle, info, variant)
    if len(pipe.memory) == 0:
        raise ValueError(f"{domain}: empty static memory")

    s3m_all = sorted(info.exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * R.S3M_MEM_FRAC)
    mem_dates = list(pipe.memory.dates)
    val_dates = s3m_all[n_mem:]
    s3c_dates = sorted(info.exp.dates_in_split("S3C"))
    s4_dates = sorted(info.exp.dates_in_split("S4"))
    qmap = dict(zip(s4_dates, D.s4_quarters(s4_dates)))

    def block_of(d):
        if d in set(mem_dates):
            return "train"
        if d in set(val_dates):
            return "val"
        if d in set(s3c_dates):
            return "s3c"
        if d in qmap:
            return "dev"
        return None

    ts = info.ds["ts"]
    y_full = info.ds["price"].astype(np.float64)
    s_full = np.asarray(info.s_full, dtype=np.float64)
    z0_full = np.asarray(info.z0_full, dtype=np.float64)

    key_sink: list = []
    hook = pipe.candidate_head.core_encoder.signature.learned_proj.register_forward_hook(
        lambda m, i, o: key_sink.append(o.detach().cpu().numpy()))
    daymap: dict = {}
    try:
        all_dates = sorted(set(s3m_all) | set(s3c_dates) | set(s4_dates))
        for d in all_dates:
            day, _k = V6.make_day_key(pipe, info, d, det_np, key_sink)
            if day is None:
                continue
            daymap[d] = day
    finally:
        hook.remove()

    days: list = []
    hour_split = {sp: {k: [] for k in ("wm", "wp", "mm", "mp", "r",
                                       "Bd", "Bu", "Yd", "Yu", "sd", "su")}
                  for sp in ("train", "val", "s3c", "dev")}
    for d in all_dates:
        if d not in daymap:
            continue
        blk = block_of(d)
        if blk is None:
            continue
        cand = daymap[d]["candidate"]
        z0 = _fl(cand["z0"])
        zY = np.asarray(daymap[d]["target_zY"]).reshape(-1)
        mm = np.nan_to_num(_fl(cand["m_minus"]), nan=0.0)
        mp = np.nan_to_num(_fl(cand["m_plus"]), nan=0.0)
        wm = np.nan_to_num(_fl(cand["w_minus"]), nan=0.0)
        w0 = np.nan_to_num(_fl(cand["w_zero"]), nan=0.0)
        wp = np.nan_to_num(_fl(cand["w_plus"]), nan=0.0)
        vm = _fl(cand["valid_mask"]).astype(bool)
        if not vm.any():
            continue
        idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
        r = zY - z0
        gd = np.abs(r) - np.abs(r + mm)
        gu = np.abs(r) - np.abs(r - mp)
        Bd = (gd > 0).astype(np.float64)
        Bu = (gu > 0).astype(np.float64)
        Yd = np.where(mm > 0, gd / np.maximum(mm, EPS), np.nan)
        Yu = np.where(mp > 0, gu / np.maximum(mp, EPS), np.nan)
        sd = 2.0 * wm - 1.0
        su = 2.0 * wp - 1.0

        for h in np.where(vm)[0]:
            hb = hour_split[blk]
            hb["wm"].append(wm[h]); hb["wp"].append(wp[h])
            hb["mm"].append(mm[h]); hb["mp"].append(mp[h])
            hb["r"].append(r[h]); hb["Bd"].append(Bd[h]); hb["Bu"].append(Bu[h])
            hb["Yd"].append(Yd[h]); hb["Yu"].append(Yu[h])
            hb["sd"].append(sd[h]); hb["su"].append(su[h])

        s_day = float(s_full[idxs[0]]) if np.isfinite(s_full[idxs[0]]) else \
            float(np.mean(np.abs(info.yhat_full[idxs].astype(np.float64))))
        days.append({
            "date": d, "block": blk, "quarter": qmap.get(d),
            "z0": z0, "zY": zY, "mm": mm, "mp": mp, "wm": wm, "wp": wp,
            "vm": vm, "r": r, "gd": gd, "gu": gu, "sd": sd, "su": su,
            "host_day": info.yhat_full[idxs].astype(np.float64),
            "price": y_full[idxs], "s_day": s_day,
        })
    return {"domain": domain, "days": days, "hour_split": hour_split,
            "problems": problems, "k": int(pipe.k), "n_mem0": len(pipe.memory)}


def hour_table(hour_split, sp, direction="d"):
    """Concatenated hour-level arrays for a split and direction, m>0 hours."""
    hb = hour_split[sp]
    mag = hb["mp"] if direction == "u" else hb["mm"]
    keep = np.asarray(mag) > 0
    return {k: np.asarray(hb[k])[keep] for k in ("wm", "wp", "mm", "mp", "r",
                                                 "Bd", "Bu", "Yd", "Yu",
                                                 "sd", "su")}


# ---------------------------------------------------------- per-calibrator ----
def evaluate_days(cal, domain_data):
    """Apply a calibrator through the double-event optimizer for every day."""
    rows = []
    for day in domain_data["days"]:
        s̃d = cal.apply(day["sd"], "d", domain_data["domain"])
        s̃u = cal.apply(day["su"], "u", domain_data["domain"])
        g̃d = day["mm"] * s̃d
        g̃u = day["mp"] * s̃u
        prop = double_event_proposal(g̃d, g̃u)
        pi = form_final_pi(day["mm"], day["mp"], prop["I_down"], prop["I_up"])
        vm = day["vm"]
        A_hat = float((g̃d * (pi < 0) + g̃u * (pi > 0))[vm].sum() / vm.sum())
        A_true = estimate_realized_A(day["z0"], day["zY"], pi, vm)
        rows.append({"date": day["date"], "block": day["block"],
                     "quarter": day["quarter"], "A_hat": A_hat,
                     "A_true": A_true, "fire": bool(np.any(pi[vm] != 0.0)),
                     "pi": pi, "s_day": day["s_day"], "z0": day["z0"],
                     "price": day["price"], "vm": vm})
    return rows


def dvg_and_s4(cal_rows, alpha=0.10):
    """Per-calibrator DVG q from S3C, then S4 gated selective metrics."""
    s3c_E = np.array([r["A_hat"] - r["A_true"] for r in cal_rows
                      if r["block"] == "s3c"], dtype=np.float64)
    if len(s3c_E) >= 2:
        q = _order_stat_q(s3c_E, alpha)
    else:
        q = None
    s4 = [r for r in cal_rows if r["block"] == "dev"]
    out = {"n_calib": int(len(s3c_E)), "q": q, "n_eval": len(s4)}
    if q is not None and s4:
        lcb = np.array([r["A_hat"] for r in s4]) - q
        released = lcb > 0
        A = np.array([r["A_true"] for r in s4])
        out["release_rate"] = float(released.mean())
        out["identity_rate"] = float((~released).mean())
        rel = A[released]
        out["harmful_rate"] = float((rel < 0).mean()) if len(rel) else None
        out["mean_gain_release"] = float(rel.mean()) if len(rel) else None
        out["net_value"] = float(A[released].sum() / len(A))
        out["coverage"] = float((A >= lcb).mean())
        out["A_hat_med"] = float(np.median([r["A_hat"] for r in s4]))
        out["A_true_med"] = float(np.median(A))
        out["_released"] = released
        out["_rows"] = s4
    else:
        out["release_rate"] = None
        out["identity_rate"] = None
        out["harmful_rate"] = None
        out["mean_gain_release"] = None
        out["net_value"] = None
        out["coverage"] = None
        out["_released"] = None
        out["_rows"] = s4
    return out


def val_metrics(cal_rows):
    """Selection metrics on S3M-suffix, execute iff A_hat>0 (no gate yet)."""
    val = [r for r in cal_rows if r["block"] == "val"]
    out = {"n_eval": len(val)}
    if not val:
        return {**out, "release_rate": None, "harmful_rate": None,
                "mean_gain_release": None, "net_value": None}
    A = np.array([r["A_true"] for r in val])
    ex = np.array([r["A_hat"] for r in val]) > 0
    rel = A[ex]
    out["release_rate"] = float(ex.mean())
    out["harmful_rate"] = float((rel < 0).mean()) if len(rel) else None
    out["mean_gain_release"] = float(rel.mean()) if len(rel) else None
    out["net_value"] = float(A[ex].sum() / len(A))
    return out


def point_metrics(dvg_out, host_mae_usd):
    """S4 final forecast: replay corrected prediction via s*sinh(z0+pi)."""
    rows = dvg_out.get("_rows", [])
    released = dvg_out.get("_released")
    if released is None or not rows:
        return {}
    price_all, pred_all, host_all = [], [], []
    ae_final, ae_host, smape = [], [], []
    for r, rel in zip(rows, released):
        pi_eff = r["pi"] if rel else np.zeros_like(r["pi"])
        pred = r["s_day"] * np.sinh(r["z0"] + pi_eff)
        host = r["s_day"] * np.sinh(r["z0"])
        vm = r["vm"]
        p = r["price"][vm]
        ae_final.append(np.abs(p - pred[vm]))
        ae_host.append(np.abs(p - host[vm]))
        denom = np.abs(p) + np.abs(pred[vm]) + EPS
        smape.append(2.0 * np.abs(p - pred[vm]) / denom)
        price_all.append(p); pred_all.append(pred[vm]); host_all.append(host[vm])
    price_all = np.concatenate(price_all)
    final_mae = float(np.mean(np.concatenate(ae_final)))
    host_mae = float(np.mean(np.concatenate(ae_host)))
    smape_v = float(np.mean(np.concatenate(smape)))
    scale = float(np.mean(np.abs(price_all)))
    return {"final_mae_usd": final_mae, "host_mae_usd": host_mae,
            "final_rmae": final_mae / scale if scale > 0 else None,
            "host_rmae": host_mae / scale if scale > 0 else None,
            "smape_nofloor": smape_v,
            "degradation": final_mae - host_mae,
            "degradation_frac": (final_mae - host_mae) / host_mae
            if host_mae > 0 else None}


def calibration_metrics(cal, domain_data, sp, direction):
    """Utility calibration of s̃ vs B/Y on a split (for calibration_validation)."""
    h = hour_table(domain_data["hour_split"], sp, "d" if direction == "d" else "u")
    if direction == "d":
        s = h["sd"]; Y = h["Yd"]; B = h["Bd"]
    else:
        s = h["su"]; Y = h["Yu"]; B = h["Bu"]
    m = np.isfinite(Y)
    s, Y, B = s[m], Y[m], B[m]
    if len(s) < 8:
        return {"n": 0, "ece_B": None, "brier_B": None, "util_slope": None,
                "util_intercept": None, "spearman": None}
    s̃ = cal.apply(s, direction, domain_data["domain"])
    p̃ = (s̃ + 1.0) / 2.0
    ece_B = _ece(p̃, B)
    brier_B = _brier(p̃, B)
    slope, intercept = np.polyfit(s̃, Y, 1) if np.std(s̃) > 0 else (np.nan, np.nan)
    return {"n": int(len(s)), "ece_B": ece_B, "brier_B": brier_B,
            "util_slope": float(slope), "util_intercept": float(intercept),
            "spearman": _sp(s̃, Y)}


# ------------------------------------------------------------------- plots ----
def _reliability_points(score, label, nbins=12):
    rows = _rel_bins(score, label, nbins)
    xs = [r["mean_score"] for r in rows if r["n"] and r["mean_score"] is not None]
    ys = [r["mean_label"] for r in rows if r["n"] and r["mean_label"] is not None]
    ns = [r["n"] for r in rows if r["n"]]
    return xs, ys, ns


def _focus_axes(fig):
    axs = fig.subplots(2, 2)
    for ax, dom in zip(axs.ravel(), FOCUS):
        ax.set_title(dom, fontsize=8)
        ax.tick_params(labelsize=7)
    return axs.ravel()


def _fig_reliability(all_hr, direction, out_fig):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    for ax, dom in zip(axs.ravel(), FOCUS):
        h = all_hr[dom]
        s = h["wd"] if direction == "d" else h["wu"]
        B = h["Bd"] if direction == "d" else h["Bu"]
        xs, ys, _ = _reliability_points(s, B)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(xs, ys, "o-", ms=4, color="steelblue")
        ax.axvspan(0.5, 1.0, color="salmon", alpha=0.15)
        ax.set_xlabel("w  (IAH P(B=1))", fontsize=7)
        ax.set_ylabel("empirical P(B=1)", fontsize=7)
        ax.set_title(dom, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"B0 raw threshold calibration — {'Down' if direction=='d' else 'Up'}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / f"reliability_{direction}.png", dpi=130)
    plt.close(fig)


def _fig_boundary(all_hr, direction, out_fig):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    for ax, dom in zip(axs.ravel(), FOCUS):
        h = all_hr[dom]
        s = h["wd"] if direction == "d" else h["wu"]
        B = h["Bd"] if direction == "d" else h["Bu"]
        lo, hi = 0.40, 0.60
        xs, ys, ns = [], [], []
        for edge in np.arange(lo, hi, 0.01):
            m = (s >= edge) & (s < edge + 0.01)
            if m.sum() >= 20:
                xs.append(float(s[m].mean()))
                ys.append(float(B[m].mean()))
                ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        if xs:
            sc = ax.scatter(xs, ys, c=ns, s=14, cmap="viridis")
        ax.axvline(0.5, color="salmon", ls=":", lw=1)
        ax.set_xlim(0.38, 0.62); ax.set_ylim(0.0, 1.0)
        ax.set_title(dom, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"B0 boundary zoom (w in [0.40,0.60]) — "
                 f"{'Down' if direction=='d' else 'Up'}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / f"boundary_zoom_{direction}.png", dpi=130)
    plt.close(fig)


def _fig_utility(all_hr, direction, out_fig):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    for ax, dom in zip(axs.ravel(), FOCUS):
        h = all_hr[dom]
        s = h["sd"] if direction == "d" else h["su"]
        Y = h["Yd"] if direction == "d" else h["Yu"]
        m = np.isfinite(Y)
        s, Y = s[m], Y[m]
        rows = _rel_bins(s, Y, 12)
        xs = [r["mean_score"] for r in rows if r["n"] and r["mean_score"] is not None]
        ys = [r["mean_label"] for r in rows if r["n"] and r["mean_label"] is not None]
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.plot(xs, ys, "o-", ms=4, color="seagreen")
        ax.set_xlabel("s = 2w-1", fontsize=7)
        ax.set_ylabel("E[Y] = E[g/m]", fontsize=7)
        ax.set_title(dom, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"A4 continuous utility calibration — "
                 f"{'Down' if direction=='d' else 'Up'}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_fig / f"utility_calibration_{direction}.png", dpi=130)
    plt.close(fig)


def _fig_net_value(s4_sum, out_fig):
    df = pd.DataFrame(s4_sum)
    piv = df.pivot_table(index="domain", columns="calibrator",
                         values="s4_net", aggfunc="first")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(piv))
    w = 0.2
    for j, col in enumerate(piv.columns):
        ax.bar(x + (j - 1.5) * w, piv[col], w, label=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("net daily action value (S4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "net_value_s4.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variant", type=str, default="learned_sig")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--force-iso", action="store_true")
    args = ap.parse_args()

    results_dir = HERE / "results"
    if args.artifacts:
        artifact_dir = Path(args.artifacts)
    else:
        dirs = sorted(results_dir.glob("R1A_[0-9]*"), key=lambda p: p.name)
        if not dirs:
            raise SystemExit("no R1A_* artifact dir under results/")
        artifact_dir = dirs[-1]
    out_dir = Path(args.out) if args.out else \
        results_dir / f"R1A9_CAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.9] frozen artifacts: {artifact_dir}")
    print(f"[R1A.9] out: {out_dir}")

    domain_data = {}
    for ds_key, bb in R.DOMAINS:
        print(f"[R1A.9] collect {ds_key}:{bb} ...", flush=True)
        dd = collect_domain(artifact_dir, ds_key, bb, args.variant)
        domain_data[dd["domain"]] = dd
        print(f"    k={dd['k']} n_mem0={dd['n_mem0']} "
              f"n_days={len(dd['days'])} probs={len(dd['problems'])}")
        for pr in dd["problems"]:
            print(f"      WARN {pr}")

    # ---- calibrator fit (S3M-prefix hours only) ----
    train_slices = []
    for dom, dd in domain_data.items():
        t_d = hour_table(dd["hour_split"], "train", "d")
        t_u = hour_table(dd["hour_split"], "train", "u")
        train_slices.append({"domain": dom, "sd": t_d["sd"], "Yd": t_d["Yd"],
                             "su": t_u["su"], "Yu": t_u["Yu"]})
    calibrators = [RawIAH(), SharedAffine(), LocalAffine(), LocalIsotonic()]
    for cal in calibrators:
        cal.fit(train_slices)

    # ---- evaluate every calibrator through the optimizer ----
    cal_rows = {cal.name: {dom: evaluate_days(cal, dd)
                           for dom, dd in domain_data.items()}
                for cal in calibrators}
    s4_sum = []
    for cal in calibrators:
        for dom, dd in domain_data.items():
            rows = cal_rows[cal.name][dom]
            v = val_metrics(rows)
            dv = dvg_and_s4(rows, args.alpha)
            pm = point_metrics(dv, None)
            row = {"domain": dom, "calibrator": cal.name,
                   "val_n": v["n_eval"], "val_release": v["release_rate"],
                   "val_harmful": v["harmful_rate"],
                   "val_gain_release": v["mean_gain_release"],
                   "val_net": v["net_value"],
                   "s4_n": dv["n_eval"], "s4_release": dv["release_rate"],
                   "s4_identity": dv["identity_rate"],
                   "s4_harmful": dv["harmful_rate"],
                   "s4_gain_release": dv["mean_gain_release"],
                   "s4_net": dv["net_value"], "q": dv["q"],
                   "coverage": dv["coverage"],
                   "A_hat_med": dv["A_hat_med"], "A_true_med": dv["A_true_med"]}
            row.update(pm)
            s4_sum.append(row)

    # ---- calibration_validation.csv (per domain, calibrator, split, dir) ----
    cal_val_rows = []
    for cal in calibrators:
        for dom, dd in domain_data.items():
            for split in ("val", "s3c", "dev"):
                for direction in ("d", "u"):
                    cm = calibration_metrics(cal, dd, split, direction)
                    cal_val_rows.append({
                        "domain": dom, "calibrator": cal.name, "split": split,
                        "direction": direction, **cm})

    # ---- B0 raw threshold calibration ----
    raw_rows, boundary_rows = [], []
    all_hr = {}
    for dom, dd in domain_data.items():
        merged = {k: [] for k in ("wd", "wu", "Bd", "Bu", "sd", "su", "Yd", "Yu")}
        for sp in ("train", "val", "s3c", "dev"):
            t = hour_table(dd["hour_split"], sp, "d")
            merged["wd"].append(t["wm"]); merged["Bd"].append(t["Bd"])
            merged["sd"].append(t["sd"]); merged["Yd"].append(t["Yd"])
            t2 = hour_table(dd["hour_split"], sp, "u")
            merged["wu"].append(t2["wp"]); merged["Bu"].append(t2["Bu"])
            merged["su"].append(t2["su"]); merged["Yu"].append(t2["Yu"])
        all_hr[dom] = {k: np.concatenate(v) for k, v in merged.items()}
        for direction, sk, bk, yk in (("d", "wd", "Bd", "Yd"),
                                      ("u", "wu", "Bu", "Yu")):
            w = all_hr[dom][sk]; B = all_hr[dom][bk]
            Y = all_hr[dom][yk]
            m = np.isfinite(Y) & np.isfinite(w) & np.isfinite(B)
            w, B = w[m], B[m]
            n = len(w)
            if n < 16:
                continue
            brier = _brier(w, B)
            ece = _ece(w, B)
            wc = np.clip(w, 1e-3, 1 - 1e-3)
            lr = LogisticRegression()
            lr.fit(wc.reshape(-1, 1), B)
            intercept = float(lr.intercept_[0])
            slope = float(lr.coef_[0, 0])
            fire = w > 0.5
            raw_rows.append({
                "domain": dom, "direction": direction, "n": n,
                "brier": brier, "ece": ece, "calib_intercept": intercept,
                "calib_slope": slope, "auc": _auc(w, B), "pr_auc": _pr_auc(w, B),
                "overall_benefit_rate": float(B.mean()),
                "fire_benefit_rate": float(B[fire].mean()) if fire.sum() else None,
                "fire_n": int(fire.sum()),
            })
            for rb in _rel_bins(w, B):
                boundary_rows.append({"domain": dom, "direction": direction,
                                      "range": "full", **rb})
            for edge in np.arange(0.40, 0.60, 0.01):
                mm = (w >= edge) & (w < edge + 0.01)
                if mm.sum() < 20:
                    continue
                boundary_rows.append({"domain": dom, "direction": direction,
                                      "range": "boundary",
                                      "bin": float(edge), "lo": edge,
                                      "hi": edge + 0.01, "n": int(mm.sum()),
                                      "mean_score": float(w[mm].mean()),
                                      "mean_label": float(B[mm].mean())})

    # ---- figures ----
    _fig_reliability(all_hr, "d", out_fig)
    _fig_reliability(all_hr, "u", out_fig)
    _fig_boundary(all_hr, "d", out_fig)
    _fig_boundary(all_hr, "u", out_fig)
    _fig_utility(all_hr, "d", out_fig)
    _fig_utility(all_hr, "u", out_fig)
    _fig_net_value(s4_sum, out_fig)

    # ---- CSVs ----
    pd.DataFrame(raw_rows).to_csv(out_dir / "raw_threshold_calibration.csv",
                                  index=False)
    pd.DataFrame(boundary_rows).to_csv(out_dir / "boundary_calibration.csv",
                                       index=False)
    util_rows = []
    for dom, dd in domain_data.items():
        for direction in ("d", "u"):
            sk = "sd" if direction == "d" else "su"
            yk = "Yd" if direction == "d" else "Yu"
            s = all_hr[dom][sk]; Y = all_hr[dom][yk]
            m = np.isfinite(Y)
            s, Y = s[m], Y[m]
            rows = _rel_bins(s, Y, 12)
            for rb in rows:
                util_rows.append({"domain": dom, "direction": direction, **rb})
    pd.DataFrame(util_rows).to_csv(out_dir / "normalized_utility_calibration.csv",
                                   index=False)
    cal_params = []
    for direction, label in (("d", "down"), ("u", "up")):
        c1 = calibrators[1]
        a, b = getattr(c1, "params", {}).get(direction, (1.0, 0.0))
        cal_params.append({"calibrator": "C1_shared_affine", "domain": "shared",
                           "direction": label, "a": a, "b": b})
    c2 = calibrators[2]; c3 = calibrators[3]
    for dom in domain_data:
        for direction, label in (("d", "down"), ("u", "up")):
            a, b = c2.params.get(dom, {}).get(direction, (1.0, 0.0))
            cal_params.append({"calibrator": "C2_local_affine", "domain": dom,
                               "direction": label, "a": a, "b": b})
            iso = c3.params.get(dom, {}).get(direction)
            xs = getattr(iso, "X_thresholds_", None) if iso is not None else None
            ys = getattr(iso, "y_thresholds_", None) if iso is not None else None
            n_th = int(len(xs)) if xs is not None else None
            lo = float(np.min(ys)) if ys is not None and len(ys) else None
            hi = float(np.max(ys)) if ys is not None and len(ys) else None
            cal_params.append({"calibrator": "C3_local_isotonic", "domain": dom,
                               "direction": label, "a": lo, "b": hi,
                               "n_thresholds": n_th})
    pd.DataFrame(cal_params).to_csv(out_dir / "calibrator_params.csv",
                                    index=False)
    pd.DataFrame(cal_val_rows).to_csv(out_dir / "calibration_validation.csv",
                                      index=False)
    s4_df = pd.DataFrame(s4_sum)
    s4_df.to_csv(out_dir / "s4_action_metrics.csv", index=False)
    dvg_rows = [{"domain": r["domain"], "calibrator": r["calibrator"],
                 "q": r["q"], "n_calib": None, "s4_release": r["s4_release"],
                 "s4_harmful": r["s4_harmful"],
                 "s4_gain_release": r["s4_gain_release"],
                 "coverage": r["coverage"]} for r in s4_sum]
    pd.DataFrame(dvg_rows).to_csv(out_dir / "dvg_metrics.csv", index=False)
    s4_df[["domain", "calibrator", "host_mae_usd", "host_rmae",
           "final_mae_usd", "final_rmae", "smape_nofloor", "degradation",
           "degradation_frac"]].to_csv(out_dir / "final_point_metrics.csv",
                                       index=False)
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{_git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "alpha": args.alpha,
                   "force_iso": args.force_iso,
                   "domains": [f"{d}:{b}" for d, b in R.DOMAINS]}, f, indent=2)

    # ------------------------------------------------------------- verdict ----
    def _f(x):
        return f"{x:.3f}" if x is not None and pd.notna(x) else "NaN"

    def _g(x):
        return f"{x:+.4f}" if x is not None and pd.notna(x) else "NaN"

    pjm = s4_df[s4_df["domain"] == "LAGO_PJM:MLP"]
    c1_pjm = pjm[pjm["calibrator"] == "C1_shared_affine"]
    c2_pjm = pjm[pjm["calibrator"] == "C2_local_affine"]
    c3_pjm = pjm[pjm["calibrator"] == "C3_local_isotonic"]
    c0_pjm = pjm[pjm["calibrator"] == "C0_raw"]

    def route(row):
        A = row["s4_gain_release"] is not None and \
            row["s4_gain_release"] > 0 and row["s4_net"] > 0
        rel = row["s4_release"]
        harm = row["s4_harmful"]
        deg = row["degradation_frac"]
        # release==0 => trivially no harmful releases (harm is NaN there)
        B = (rel is not None and rel <= 0.15
             and (rel == 0.0 or (harm is not None and harm <= 0.10))
             and deg is not None and abs(deg) <= 0.03)
        return A, B

    def retention_ok(cal_name):
        for dom in ("NEM_SA1:MLP", "NEM_SA1:Linear", "LAGO_DE:MLP"):
            r0 = s4_df[(s4_df["domain"] == dom)
                       & (s4_df["calibrator"] == "C0_raw")].iloc[0]
            rc = s4_df[(s4_df["domain"] == dom)
                       & (s4_df["calibrator"] == cal_name)].iloc[0]
            n0 = r0["s4_net"]; nc = rc["s4_net"]
            rel0 = r0["s4_release"]; relc = rc["s4_release"]
            if nc is None:
                return False, dom
            if n0 is not None and n0 > 0 and nc < 0.5 * n0:
                return False, dom
            if rel0 is not None and relc is not None and rel0 > 0.05 \
                    and relc < 0.5 * rel0:
                return False, dom
        return True, None

    c1a, c1b = route(c1_pjm.iloc[0]) if len(c1_pjm) else (False, False)
    c2a, c2b = route(c2_pjm.iloc[0]) if len(c2_pjm) else (False, False)
    c3a, c3b = route(c3_pjm.iloc[0]) if len(c3_pjm) else (False, False)
    ret_ok, ret_dom = retention_ok("C1_shared_affine")

    verdict = "MONOTONE_CALIBRATION_INSUFFICIENT"
    reason = ""
    if (c1a or c1b) and ret_ok:
        verdict = "UNIVERSAL_ACTION_CALIBRATION_SUPPORTED"
        reason = ("C1 shared affine achieves PJM:MLP Route "
                  + ("A" if c1a else "B")
                  + f" ({'recalibrated action' if c1a else 'calibrated abstention'}) "
                    "without crushing NEM/DE.")
    elif c2a or c2b or c3a or c3b:
        verdict = "LOCAL_ACTION_CALIBRATION_SUPPORTED"
        reason = ("C1 shared affine fails on PJM:MLP; a local (per market:host) "
                  "monotone calibration achieves Route "
                  + ("A" if (c2a or c3a) else "B") + ".")
    else:
        # partial: any meaningful improvement vs C0 on PJM:MLP or elsewhere?
        part = False
        if len(c0_pjm):
            c0r = c0_pjm.iloc[0]
            for r in (c1_pjm.iloc[0] if len(c1_pjm) else None,
                      c2_pjm.iloc[0] if len(c2_pjm) else None,
                      c3_pjm.iloc[0] if len(c3_pjm) else None):
                if r is None:
                    continue
                if r["s4_harmful"] is not None and c0r["s4_harmful"] is not None \
                        and r["s4_harmful"] < c0r["s4_harmful"] - 0.05:
                    part = True
                if r["s4_net"] is not None and c0r["s4_net"] is not None \
                        and r["s4_net"] > c0r["s4_net"] + 0.005:
                    part = True
        if part:
            verdict = "ACTION_CALIBRATION_PARTIAL"
            reason = ("No calibrator reaches a full PJM:MLP success route, but "
                      "some monotone map materially improves harmful/ net value "
                      "vs C0.")
        else:
            reason = ("C1/C2/C3 monotone recalibration cannot turn the frozen "
                      "3-atom distribution into reliable action utility on "
                      "PJM:MLP (nor recover NEM/DE without regression). "
                      "Richer action mapping / richer information required.")

    L = []
    L.append("# R1A.9 CALIBRATION VERDICT — Action-Threshold / Utility Calibration")
    L.append("")
    L.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- code commit: `{_git_head()}`")
    L.append(f"- frozen artifacts: `{artifact_dir.name}` (variant "
             f"`{args.variant}`, no retrain)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"### **{verdict}**")
    L.append("")
    L.append(f"- reason: {reason}")
    L.append("")
    L.append("## B0 — Raw threshold calibration (pooled, action-relevant m>0 hours)")
    for _, r in pd.DataFrame(raw_rows).iterrows():
        L.append(f"- {r['domain']} {r['direction']}: n={r['n']} "
                 f"brier={_f(r['brier'])} ece={_f(r['ece'])} "
                 f"slope={_f(r['calib_slope'])} intercept={_f(r['calib_intercept'])} "
                 f"auc={_f(r['auc'])} "
                 f"fire_benefit_rate(w>0.5)={_f(r['fire_benefit_rate'])} "
                 f"overall={_f(r['overall_benefit_rate'])}")
    L.append("")
    L.append("## C1 shared affine params (equal-weighted)")
    for row in cal_params:
        if row["calibrator"] == "C1_shared_affine":
            L.append(f"- {row['direction']}: a={_f(row['a'])} b={_f(row['b'])}")
    L.append("")
    L.append("## S4 selective action metrics (development confirmation)")
    for _, r in s4_df.iterrows():
        L.append(f"- {r['domain']} {r['calibrator']}: "
                 f"release={_f(r['s4_release'])} harm={_f(r['s4_harmful'])} "
                 f"gain|rel={_g(r['s4_gain_release'])} "
                 f"net={_g(r['s4_net'])} q={_f(r['q'])} "
                 f"final_rmae={_f(r['final_rmae'])} "
                 f"degrad_frac={_f(r['degradation_frac'])}")
    L.append("")
    L.append("## Decision tree (§12)")
    L.append(f"- PJM:MLP C0: release={_f(c0_pjm.iloc[0]['s4_release'])} "
             f"harm={_f(c0_pjm.iloc[0]['s4_harmful'])}")
    if len(c1_pjm):
        L.append(f"- PJM:MLP C1 (shared affine): release="
                 f"{_f(c1_pjm.iloc[0]['s4_release'])} "
                 f"harm={_f(c1_pjm.iloc[0]['s4_harmful'])} "
                 f"net={_g(c1_pjm.iloc[0]['s4_net'])} "
                 f"-> Route {'A' if c1a else 'B' if c1b else 'FAIL'}")
    if len(c2_pjm):
        L.append(f"- PJM:MLP C2 (local affine): release="
                 f"{_f(c2_pjm.iloc[0]['s4_release'])} "
                 f"harm={_f(c2_pjm.iloc[0]['s4_harmful'])} "
                 f"net={_g(c2_pjm.iloc[0]['s4_net'])} "
                 f"-> Route {'A' if c2a else 'B' if c2b else 'FAIL'}")
    if len(c3_pjm):
        L.append(f"- PJM:MLP C3 (local isotonic): release="
                 f"{_f(c3_pjm.iloc[0]['s4_release'])} "
                 f"harm={_f(c3_pjm.iloc[0]['s4_harmful'])} "
                 f"net={_g(c3_pjm.iloc[0]['s4_net'])} "
                 f"-> Route {'A' if c3a else 'B' if c3b else 'FAIL'}")
    ret_txt = "PASS" if ret_ok else f"FAIL at {ret_dom}"
    L.append(f"- NEM/DE retention under C1: {ret_txt}")
    L.append("")
    L.append("## NEM/DE retention vs C0 (S4 net value, §11 hard constraint)")
    for cal_name in ("C1_shared_affine", "C2_local_affine", "C3_local_isotonic"):
        for dom in ("NEM_SA1:MLP", "NEM_SA1:Linear", "LAGO_DE:MLP"):
            r0 = s4_df[(s4_df["domain"] == dom)
                       & (s4_df["calibrator"] == "C0_raw")].iloc[0]
            rc = s4_df[(s4_df["domain"] == dom)
                       & (s4_df["calibrator"] == cal_name)].iloc[0]
            n0 = r0["s4_net"]; nc = rc["s4_net"]
            ratio = (nc / n0) if (n0 is not None and n0 != 0) else None
            L.append(f"- {dom} {cal_name}: net {_g(n0)} -> {_g(nc)} "
                     f"(ratio={_f(ratio) if ratio is not None else 'n/a'}, "
                     f"release {_f(r0['s4_release'])} -> "
                     f"{_f(rc['s4_release'])})")
    L.append("")
    L.append("## Notes")
    L.append("- Chronology: S3M-prefix fit, S3M-suffix selection, S3C DVG-q "
             "only, S4 dev confirmation. No S4 tuning.")
    L.append("- Each calibrator refits its own DVG q on S3C (§8); old W1 q never "
             "reused.")
    L.append("- C0 raw == current production action; its S4 row is the R1A.8 "
             "cross-check.")
    L.append("- Final point metrics replay s*sinh(z0+pi); no-floor sMAPE uses "
             "only a 1e-12 numerical floor.")
    L.append("- R1B stays paused; a richer action mapping is authorized ONLY if "
             "verdict is not MONOTONE_CALIBRATION_INSUFFICIENT.")
    L.append("")
    verdict_text = "\n".join(L)
    with open(out_dir / "CALIBRATION_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)
    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.9] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
