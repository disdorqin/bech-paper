"""R1A.8 Decision Calibration & Safe Abstention Audit (plan v0.1).

Four questions (plan §2):
  Q1 A0 — is the current IAH full-atom action the absolute-loss Bayes decision?
  Q2 A1 — materiality: strong-host/low-margin vs large-but-unpredictable margin.
  Q3 A2 — can frozen legal pre-outcome features predict Down/Up benefit?
  Q4 A3 — does the 3-atom distribution carry continuous direction signal?
  Q5 A4 — can IAH-native value + fresh DVG safely abstain on non-actionable
          domains?

Pipeline mirrors R1A.7 (same frozen candidate, same splits, same daymap).
No retrain; no new trainable production component. Probe 2 (tiny MLP) is a
diagnostic upper-bound only.

Verdict ∈ {SAFE_ABSTENTION_SUPPORTED, ACTION_MAPPING_FAILURE,
            ACTION_SIGNAL_UNRESOLVED} (plus IMPLEMENTATION_BUG if A0 fails).
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

# Windows GBK console cannot encode e.g. U+207B "⁻"; force UTF-8 on std streams
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-12


# ----------------------------------------------------------------- helpers ----
def _fl(t) -> np.ndarray:
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a.reshape(-1).astype(np.float64)


def _iso(d) -> str:
    return str(pd.Timestamp(d).date())


def _sp(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    try:
        return float(stats.spearmanr(a, b)[0])
    except Exception:
        return None


def _pear(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _auc(score, label) -> float | None:
    """Rank-based AUC (Mann–Whitney U) with tied-rank averaging."""
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=bool).ravel()
    if len(score) < 2 or label.sum() == 0 or label.sum() == len(label):
        return None
    try:
        return float(roc_auc_score(label, score))
    except Exception:
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(len(score))
        ranks[order] = np.arange(1, len(score) + 1)
        srt = score[order]
        i = 0
        while i < len(srt):
            j = i
            while j + 1 < len(srt) and srt[j + 1] == srt[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        n_pos = int(label.sum())
        n_neg = len(label) - n_pos
        U = float(ranks[label].sum() - n_pos * (n_pos + 1) / 2.0)
        return U / (n_pos * n_neg)


def _pr_auc(score, label) -> float | None:
    try:
        return float(average_precision_score(label, score))
    except Exception:
        return None


def _brier(prob, label) -> float:
    p = np.asarray(prob, dtype=float).ravel()
    l = np.asarray(label, dtype=float).ravel()
    return float(np.mean((p - l) ** 2))


def _ece(prob, label, nbins=10) -> float:
    prob = np.asarray(prob, dtype=float).ravel()
    label = np.asarray(label, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, nbins - 1)
    ece = 0.0
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(prob)) * abs(prob[m].mean() - label[m].mean())
    return float(ece)


def _rel_bins(score, label, nbins=10):
    """10-bin reliability table: rows of (bin, lo, hi, n, mean_score, mean_label)."""
    score = np.asarray(score, dtype=float).ravel()
    label = np.asarray(label, dtype=float).ravel()
    qs = np.quantile(score, np.linspace(0, 1, nbins + 1))
    out = []
    for i in range(nbins):
        lo = -np.inf if i == 0 else qs[i]
        hi = np.inf if i == nbins - 1 else qs[i + 1]
        if i < nbins - 1:
            m = (score >= lo) & (score < hi)
        else:
            m = (score >= lo) & (score <= hi)
        out.append({"bin": i, "lo": float(lo), "hi": float(hi),
                    "n": int(m.sum()),
                    "mean_score": float(score[m].mean()) if m.sum() else None,
                    "mean_label": float(label[m].mean()) if m.sum() else None})
    return out


def _rel_spearman(score, label, nbins=10) -> float | None:
    rows = _rel_bins(score, label, nbins)
    xs = [r["mean_score"] for r in rows if r["n"] > 0 and r["mean_score"] is not None]
    ys = [r["mean_label"] for r in rows if r["n"] > 0 and r["mean_label"] is not None]
    if len(xs) >= 3 and np.std(ys) > 0:
        return float(stats.spearmanr(xs, ys)[0])
    return None


def _top_decile_gain(score, gain):
    """mean continuous true gain in top decile of score vs overall mean."""
    score = np.asarray(score, dtype=float).ravel()
    gain = np.asarray(gain, dtype=float).ravel()
    n_top = max(1, len(score) // 10)
    top = np.argsort(score)[-n_top:]
    return float(gain[top].mean()), float(gain.mean())


def _bayes_action(wm, wp, mm, mp, count_ties=True):
    """Per-hour absolute-loss Bayes median act (plan §3).

    act = -m^- where w^- > 0.5; +m^+ where w^+ > 0.5; 0 otherwise.
    Exactly one of w^-, w^0, w^+ can exceed 0.5. Tie at exactly 0.5 ->
    deterministic identity (counted separately).
    """
    b = np.zeros_like(mm)
    tie = np.zeros_like(mm, dtype=bool)
    down = wm > 0.5
    up = wp > 0.5
    b[down] = -mm[down]
    b[up] = +mp[up]
    if count_ties:
        tie = (np.abs(wm - 0.5) < 1e-9) | (np.abs(wp - 0.5) < 1e-9)
    return b, down, up, tie


# ---------------------------------------------------------------- probe fit ---
def _fit_logistic_probe(Xtr, ytr, Xva, yva, Xde, yde):
    """Fixed L2 logistic regression (Probe 1). Returns val/dev probs + model."""
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    clf.fit(Xtr, ytr)
    pva = clf.predict_proba(Xva)[:, 1]
    pde = clf.predict_proba(Xde)[:, 1]
    return pva, pde, clf


def _fit_mlp_probe(Xtr, ytr, Xva, yva, Xde, yde, seed=0):
    """Tiny 2-layer MLP (width 32) diagnostic upper-bound (Probe 2).

    Early stopping on the chronological last 20% of the train split, so the
    true validation split stays a clean test set. Returns val/dev probs.
    """
    torch.manual_seed(seed)
    n_feat = Xtr.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(n_feat, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()

    # chronological early-stop split inside train
    n_tr = len(Xtr)
    n_fit = int(n_tr * 0.8)
    Xf = torch.tensor(Xtr[:n_fit], dtype=torch.float32)
    yf = torch.tensor(ytr[:n_fit], dtype=torch.float32)
    Xe = torch.tensor(Xtr[n_fit:], dtype=torch.float32)
    ye = torch.tensor(ytr[n_fit:], dtype=torch.float32)
    if len(Xe) < 16 or len(np.unique(ytr[n_fit:])) < 2:
        Xe, ye = Xf, yf  # degenerate early-split -> fit/stop on same data

    best_auc, best_state, patience = 0.0, None, 0
    for _ep in range(150):
        model.train()
        perm = torch.randperm(len(Xf))
        for i in range(0, len(Xf), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(Xf[idx]).squeeze(-1), yf[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pe = torch.sigmoid(model(Xe)).numpy()
        auc = _auc(pe, ye) or 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy()
        pde = torch.sigmoid(model(torch.tensor(Xde, dtype=torch.float32))).numpy()
    return pva, pde


# ------------------------------------------------------------- per domain ----
def analyze_domain(artifact_dir: Path, ds_key: str, bb: str,
                   variant: str = "learned_sig", alpha: float = 0.10,
                   force_mlp: bool = False) -> dict:
    old = D.NEW_TO_OLD[variant]
    domain = f"{ds_key}:{bb}"
    info = R.prepare_domain(ds_key, bb)
    bundle = HCHV2Bundle.load(str(artifact_dir / f"checkpoint_{old}_{ds_key}_{bb}.pt"))
    det_np = R.det_for_variant(variant, info)
    pipe, _val_days, _s3c_days, problems = D.rebuild_pipe_from_bundle(
        bundle, info, variant)
    k = int(pipe.k)
    n_mem0 = len(pipe.memory)
    if n_mem0 == 0:
        raise ValueError(f"{domain}: empty static memory")

    # ---- split bookkeeping (mirror r1a7) ----
    s3m_all = sorted(info.exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * R.S3M_MEM_FRAC)
    mem_dates = list(pipe.memory.dates)
    val_dates = s3m_all[n_mem:]
    s3c_dates = sorted(info.exp.dates_in_split("S3C"))
    s4_dates = sorted(info.exp.dates_in_split("S4"))
    qmap = dict(zip(s4_dates, D.s4_quarters(s4_dates)))

    def block_of(d):
        if d in set(mem_dates):
            return "S3M-mem"
        if d in set(val_dates):
            return "S3M-val"
        if d in set(s3c_dates):
            return "S3C"
        if d in qmap:
            return f"S4Q{qmap[d] + 1}"
        return None

    # ---- daymap + learned keys (same as r1a7) ----
    key_sink: list = []
    hook = pipe.candidate_head.core_encoder.signature.learned_proj.register_forward_hook(
        lambda m, i, o: key_sink.append(o.detach().cpu().numpy()))
    daymap: dict = {}
    try:
        all_dates = sorted(set(s3m_all) | set(s3c_dates) | set(s4_dates))
        for d in all_dates:
            day, key = V6.make_day_key(pipe, info, d, det_np, key_sink)
            if day is None or key is None:
                continue
            daymap[d] = {"day": day, "key": key}
    finally:
        hook.remove()

    ts = info.ds["ts"]
    y_full = info.ds["price"].astype(np.float64)

    # ---- A0 counters / A1 per-day rows / A2 feature buckets / A3 / A4 ----
    a0 = {"domain": domain,
          "n_days": 0, "n_valid_hours": 0,
          "exact_match": 0, "bayes_wants_abstain": 0,
          "current_acts_bayes_no": 0, "direction_mismatch": 0, "ties": 0,
          "current_day_fire": 0, "bayes_day_fire": 0,
          "current_hour_fire": 0.0, "bayes_hour_fire": 0.0}
    a1_rows = []          # per-day materiality
    feat_buckets = {"train": {"X": [], "Bd": [], "Bu": [], "Gd": [], "Gu": []},
                    "val": {"X": [], "Bd": [], "Bu": [], "Gd": [], "Gu": []},
                    "dev": {"X": [], "Bd": [], "Bu": [], "Gd": [], "Gu": []}}
    a3_rows = {"domain": domain, "n_hours": 0,
               "spearman(muR,r)": None, "sign_acc": None,
               "decile_monotone_spearman": None,
               "P(sign muR>0|r>0)": None, "P(sign muR<0|r<0)": None}
    a4_calib_E = []       # S3C errors for q
    a4_day_rows = []      # S4 release rows

    def split_for(d):
        if d in set(mem_dates):
            return "train"
        if d in set(val_dates) or d in set(s3c_dates):
            return "val"
        return "dev"

    for d in all_dates:
        if d not in daymap:
            continue
        day = daymap[d]["day"]
        cand = day["candidate"]
        z0 = _fl(cand["z0"])
        zY = np.asarray(day["target_zY"]).reshape(-1)
        mm = _fl(cand["m_minus"]); mp = _fl(cand["m_plus"])
        wm = _fl(cand["w_minus"]); w0 = _fl(cand["w_zero"]); wp = _fl(cand["w_plus"])
        vm = _fl(cand["valid_mask"]).astype(bool)
        r = zY - z0
        if not vm.any():
            continue
        blk = block_of(d)
        split = split_for(d)
        idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
        host_day = info.yhat_full[idxs].astype(np.float64)

        # true directional gains
        gtd = np.abs(r) - np.abs(r + mm)
        gtu = np.abs(r) - np.abs(r - mp)
        # IAH analytic gains
        gI_down = mm * (2.0 * wm - 1.0)
        gI_up = mp * (2.0 * wp - 1.0)
        # current double-event action + values
        prop = double_event_proposal(gI_down, gI_up)
        pi = form_final_pi(mm, mp, prop["I_down"], prop["I_up"])
        A_I = float((gI_down * (pi < 0) + gI_up * (pi > 0))[vm].sum()
                    / vm.sum())
        A_true = estimate_realized_A(z0, zY, pi, vm)
        o = D.oracle_for_day({"z0": cand["z0"], "m_minus": cand["m_minus"],
                              "m_plus": cand["m_plus"],
                              "valid_mask": cand["valid_mask"]}, zY)
        oA = float(o["A"])
        fire = bool(np.any(pi[vm] != 0.0))

        # ---- A0: Bayes median rule vs current action ----
        b, b_down, b_up, tie = _bayes_action(wm, wp, mm, mp)
        bayes_fire = bool(np.any(b[vm] != 0.0))
        vv = vm
        a0["n_days"] += 1
        a0["n_valid_hours"] += int(vv.sum())
        a0["ties"] += int(tie[vv].sum())
        p = pi[vv]; bv = b[vv]
        both_nonzero = (p != 0) & (bv != 0)
        a0["exact_match"] += int(((p == 0) & (bv == 0)).sum()
                                 + (both_nonzero
                                    & (np.sign(p) == np.sign(bv))).sum())
        a0["bayes_wants_abstain"] += int(((p == 0) & (bv != 0)).sum())
        a0["current_acts_bayes_no"] += int(((p != 0) & (bv == 0)).sum())
        a0["direction_mismatch"] += int(
            (both_nonzero & (np.sign(p) != np.sign(bv))).sum())
        a0["current_day_fire"] += int(fire)
        a0["bayes_day_fire"] += int(bayes_fire)
        a0["current_hour_fire"] += float((p != 0).mean())
        a0["bayes_hour_fire"] += float((bv != 0).mean())

        # ---- A1: materiality per day ----
        price_vv = y_full[idxs][vv]
        host_mae_usd = float(np.mean(np.abs(price_vv - host_day[vv])))
        host_rmae = (host_mae_usd / float(np.mean(np.abs(price_vv)))
                     if float(np.mean(np.abs(price_vv))) > 0 else None)
        host_err_z = float(np.mean(np.abs(r)[vv]))   # z-space baseline for gains
        price_scale = float(np.mean(np.abs(price_vv)))
        a1_rows.append({
            "domain": domain, "date": _iso(d), "block": blk, "split": split,
            "host_mae_usd": host_mae_usd, "host_rmae": host_rmae,
            "host_err_z": host_err_z, "price_scale": price_scale,
            "oracle_gain": oA, "iah_gain": float(A_true),
            "iah_A_I": float(A_I), "fire": int(fire),
            "bayes_fire": int(bayes_fire),
            "mean_abs_gtd": float(np.mean(np.abs(gtd)[vv])),
            "mean_abs_gtu": float(np.mean(np.abs(gtu)[vv])),
            "oracle_best_dir": float(max(np.mean(np.abs(gtd)[vv]),
                                         np.mean(np.abs(gtu)[vv]))),
            "gain_host_ratio": oA / host_err_z if host_err_z > 0 else None,
        })

        # ---- A2: hourly features (frozen pre-outcome only) ----
        hours = ts.iloc[idxs].dt.hour.values
        ctx = R.build_core_context(host_day, hours, pipe.s1_rank_ref,
                                   info.z0_full, info.s_full, y_full, idxs)
        key = np.asarray(daymap[d]["key"], dtype=np.float64)
        Hv = int(vm.sum())
        feats = np.column_stack([
            wm[vv], w0[vv], wp[vv], mm[vv], mp[vv],
            gI_down[vv], gI_up[vv], z0[vv],
            ctx[vv].astype(np.float64),
            np.tile(key, (Hv, 1)),
        ])
        bucket = feat_buckets[split]
        bucket["X"].append(feats)
        bucket["Bd"].append((gtd > 0)[vv].astype(np.float64))
        bucket["Bu"].append((gtu > 0)[vv].astype(np.float64))
        bucket["Gd"].append(gtd[vv].astype(np.float64))
        bucket["Gu"].append(gtu[vv].astype(np.float64))

        # ---- A3: pooled continuous signal ----
        muR = wp * mp - wm * mm
        rv = r[vv]; mur = muR[vv]
        n_new = int(vv.sum())
        old_n = a3_rows["n_hours"]
        a3_rows["n_hours"] += n_new
        if old_n == 0:
            a3_rows["_r"] = rv.copy(); a3_rows["_mu"] = mur.copy()
        else:
            a3_rows["_r"] = np.concatenate([a3_rows["_r"], rv])
            a3_rows["_mu"] = np.concatenate([a3_rows["_mu"], mur])

        # ---- A4: IAH-native DVG calibration/release ----
        if blk == "S3C":
            a4_calib_E.append(float(A_I) - float(A_true))
        elif blk is not None and blk.startswith("S4Q"):
            a4_day_rows.append({"domain": domain, "date": _iso(d),
                                "block": blk, "A_I": float(A_I),
                                "A_true": float(A_true),
                                "host_mae": host_err_z,   # z-space baseline
                                "host_rmae": host_rmae,
                                "price_scale": price_scale,
                                "fire": int(fire)})

    # ---- A0 aggregates ----
    a0["current_day_fire_rate"] = a0["current_day_fire"] / a0["n_days"] \
        if a0["n_days"] else None
    a0["bayes_day_fire_rate"] = a0["bayes_day_fire"] / a0["n_days"] \
        if a0["n_days"] else None
    a0["current_hour_fire_rate"] = a0["current_hour_fire"] / a0["n_days"] \
        if a0["n_days"] else None
    a0["bayes_hour_fire_rate"] = a0["bayes_hour_fire"] / a0["n_days"] \
        if a0["n_days"] else None
    n_vh = max(1, a0["n_valid_hours"])
    a0["exact_match_frac"] = a0["exact_match"] / n_vh
    a0["bayes_wants_abstain_frac"] = a0["bayes_wants_abstain"] / n_vh
    a0["current_acts_bayes_no_frac"] = a0["current_acts_bayes_no"] / n_vh
    a0["direction_mismatch_frac"] = a0["direction_mismatch"] / n_vh
    a0["tie_frac"] = a0["ties"] / n_vh
    a0.pop("current_hour_fire", None); a0.pop("bayes_hour_fire", None)

    # ---- A3 aggregates ----
    r_all = a3_rows.pop("_r"); mu_all = a3_rows.pop("_mu")
    nz = np.abs(r_all) > 1e-9
    if len(r_all) >= 2 and np.std(mu_all) > 0 and np.std(r_all) > 0:
        a3_rows["spearman(muR,r)"] = _sp(mu_all, r_all)
        if nz.sum() >= 2:
            a3_rows["sign_acc"] = float(
                np.mean(np.sign(mu_all[nz]) == np.sign(r_all[nz])))
        pos = nz & (r_all > 0)
        neg = nz & (r_all < 0)
        if pos.sum() >= 1:
            a3_rows["P(sign muR>0|r>0)"] = float((mu_all[pos] > 0).mean())
        if neg.sum() >= 1:
            a3_rows["P(sign muR<0|r<0)"] = float((mu_all[neg] < 0).mean())
        # decile monotonicity: mean-mu^R per decile vs mean realized residual
        qs = np.quantile(mu_all, np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            lo = -np.inf if i == 0 else qs[i]
            hi = np.inf if i == 9 else qs[i + 1]
            m = (mu_all >= lo) & (mu_all <= hi)
            if m.sum() >= 1:
                xs.append(mu_all[m].mean()); ys.append(r_all[m].mean())
        if len(xs) >= 3 and np.std(ys) > 0:
            a3_rows["decile_monotone_spearman"] = float(
                stats.spearmanr(xs, ys)[0])
    else:
        a3_rows["spearman(muR,r)"] = None
        a3_rows["sign_acc"] = None

    # ---- A2: probe ----
    probe_rows, rel_rows = [], []
    Xtr = np.vstack(feat_buckets["train"]["X"]) if feat_buckets["train"]["X"] \
        else np.zeros((0, 53))
    Xva = np.vstack(feat_buckets["val"]["X"]) if feat_buckets["val"]["X"] \
        else np.zeros((0, 53))
    Xde = np.vstack(feat_buckets["dev"]["X"]) if feat_buckets["dev"]["X"] \
        else np.zeros((0, 53))
    scaler = StandardScaler()
    if len(Xtr) >= 10:
        scaler.fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva) if len(Xva) else Xva
        Xde_s = scaler.transform(Xde) if len(Xde) else Xde
    else:
        Xtr_s, Xva_s, Xde_s = Xtr, Xva, Xde

    for direction, Bk, Gk in (("down", "Bd", "Gd"), ("up", "Bu", "Gu")):
        ytr = np.concatenate(feat_buckets["train"][Bk]) \
            if feat_buckets["train"][Bk] else np.zeros(0)
        yva = np.concatenate(feat_buckets["val"][Bk]) \
            if feat_buckets["val"][Bk] else np.zeros(0)
        yde = np.concatenate(feat_buckets["dev"][Bk]) \
            if feat_buckets["dev"][Bk] else np.zeros(0)
        gva = np.concatenate(feat_buckets["val"][Gk]) \
            if feat_buckets["val"][Gk] else np.zeros(0)
        gde = np.concatenate(feat_buckets["dev"][Gk]) \
            if feat_buckets["dev"][Gk] else np.zeros(0)
        if len(ytr) < 16 or len(np.unique(ytr)) < 2 or len(yva) < 2:
            continue

        # Probe 1: fixed L2 logistic
        pva1, pde1, _ = _fit_logistic_probe(Xtr_s, ytr, Xva_s, yva, Xde_s, yde)
        auc_v1 = _auc(pva1, yva) if len(yva) else None
        auc_d1 = _auc(pde1, yde) if len(yde) else None
        for split, prob, y, g in (("val", pva1, yva, gva),
                                  ("dev", pde1, yde, gde)):
            if len(y) < 2:
                continue
            prob = np.asarray(prob, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()
            g = np.asarray(g, dtype=float).ravel()
            topG, allG = _top_decile_gain(prob, g)
            pos_rate = float(y.mean())
            rel_rows += [{"domain": domain, "direction": direction,
                          "probe": "logistic", "split": split, **rb}
                         for rb in _rel_bins(prob, y)]
            probe_rows.append({
                "domain": domain, "direction": direction, "probe": "logistic",
                "split": split, "n": int(len(y)), "n_pos": int(y.sum()),
                "pos_rate": pos_rate, "auc": _auc(prob, y),
                "pr_auc": _pr_auc(prob, y), "brier": _brier(prob, y),
                "ece": _ece(prob, y),
                "rel_spearman": _rel_spearman(prob, y),
                "top10_gain_mean": topG, "all_gain_mean": allG,
                "top10_enrich_gain": topG - allG,
                "top10_hit_lift": float((y[prob >= np.quantile(prob, 0.9)]
                                         if len(y) > 9 else y).mean()) - pos_rate,
            })
        # Probe 2: tiny MLP, only when Probe 1 insufficient (val AUC < 0.60)
        if force_mlp or (auc_v1 is not None and auc_v1 < 0.60):
            pva2, pde2 = _fit_mlp_probe(Xtr_s, ytr, Xva_s, yva, Xde_s, yde)
            for split, prob, y, g in (("val", pva2, yva, gva),
                                      ("dev", pde2, yde, gde)):
                if len(y) < 2:
                    continue
                prob = np.asarray(prob, dtype=float).ravel()
                y = np.asarray(y, dtype=float).ravel()
                g = np.asarray(g, dtype=float).ravel()
                topG, allG = _top_decile_gain(prob, g)
                rel_rows += [{"domain": domain, "direction": direction,
                              "probe": "mlp", "split": split, **rb}
                             for rb in _rel_bins(prob, y)]
                probe_rows.append({
                    "domain": domain, "direction": direction, "probe": "mlp",
                    "split": split, "n": int(len(y)), "n_pos": int(y.sum()),
                    "pos_rate": float(y.mean()), "auc": _auc(prob, y),
                    "pr_auc": _pr_auc(prob, y), "brier": _brier(prob, y),
                    "ece": _ece(prob, y), "rel_spearman": _rel_spearman(prob, y),
                    "top10_gain_mean": topG, "all_gain_mean": allG,
                    "top10_enrich_gain": topG - allG,
                    "top10_hit_lift": float(
                        (y[prob >= np.quantile(prob, 0.9)]
                         if len(y) > 9 else y).mean()) - float(y.mean()),
                })

    # ---- A4: IAH-native DVG recalibration on S3C, release on S4 ----
    a4_summary = {"domain": domain, "alpha": alpha,
                  "n_calib": len(a4_calib_E), "q": None,
                  "n_eval": len(a4_day_rows), "release_rate": None,
                  "identity_rate": None, "harmful_release_rate": None,
                  "realized_gain_release": None, "host_mae": None,
                  "final_mae": None, "final_rmae": None, "degradation": None,
                  "degradation_frac": None, "coverage": None,
                  "A_I_mean": None, "A_I_median": None}
    if len(a4_calib_E) >= 2:
        E = np.sort(np.asarray(a4_calib_E, dtype=np.float64))
        n = len(E)
        rnk = int(np.ceil((1.0 - alpha) * (n + 1)))
        rnk = min(rnk, n)
        q = float(E[rnk - 1])
        a4_summary["q"] = q
        if a4_day_rows:
            df = pd.DataFrame(a4_day_rows)
            lcb = df["A_I"].to_numpy() - q
            rel = lcb > 0
            a4_summary["n_eval"] = len(df)
            a4_summary["release_rate"] = float(rel.mean())
            a4_summary["identity_rate"] = float((~rel).mean())
            rel_idx = np.where(rel)[0]
            At_rel = df["A_true"].to_numpy()[rel_idx]
            a4_summary["harmful_release_rate"] = float(
                (At_rel < 0).mean()) if len(At_rel) else None
            a4_summary["realized_gain_release"] = float(
                At_rel.mean()) if len(At_rel) else None
            host_mae = float(df["host_mae"].mean())          # z-space baseline
            host_rmae = float(df["host_rmae"].mean())        # $ / mean|price|
            final_mae = float(
                (df["host_mae"].to_numpy() - rel * df["A_true"].to_numpy()).mean())
            a4_summary["host_mae"] = host_mae
            a4_summary["host_rmae"] = host_rmae
            a4_summary["final_mae"] = final_mae
            # rMAE_final = rMAE_host scaled by the same relative error change
            a4_summary["final_rmae"] = \
                host_rmae * (final_mae / host_mae) \
                if host_mae > 0 and host_rmae is not None and host_rmae == host_rmae \
                else None
            a4_summary["degradation"] = final_mae - host_mae
            a4_summary["degradation_frac"] = (final_mae - host_mae) / host_mae \
                if host_mae > 0 else None
            a4_summary["coverage"] = float((df["A_true"].to_numpy() >= lcb).mean())
            a4_summary["A_I_mean"] = float(df["A_I"].mean())
            a4_summary["A_I_median"] = float(df["A_I"].median())
            df["lcb"] = lcb
            df["release"] = rel.astype(int)
            df["q"] = q
            a4_day_rows = df.to_dict("records")

    return {"a0": a0, "a1_rows": a1_rows, "probe_rows": probe_rows,
            "rel_rows": rel_rows, "a3": a3_rows,
            "a4_summary": a4_summary, "a4_day_rows": a4_day_rows,
            "problems": problems, "k": k, "n_mem0": n_mem0,
            "n_days": a0["n_days"]}


# ------------------------------------------------------------- figure utils ----
def _fig_bayes_fire(a0df, out_fig):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(a0df))
    w = 0.36
    ax.bar(x - w / 2, a0df["bayes_day_fire_rate"], w, label="Bayes median rule")
    ax.bar(x + w / 2, a0df["current_day_fire_rate"], w,
           label="current double-event")
    ax.set_xticks(x)
    ax.set_xticklabels(a0df["domain"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("day fire rate")
    ax.set_title("A0: per-hour Bayes median vs current full-atom day fire")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig / "a0_bayes_vs_current_fire.png", dpi=130)
    plt.close(fig)


def _fig_probe_auc(probe_df, out_fig):
    rows = probe_df[(probe_df["split"] == "val")
                    & (probe_df["probe"] == "logistic")].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = rows.pivot_table(index="domain", columns="direction",
                             values="auc", aggfunc="first")
    n_dom = len(pivot)
    x = np.arange(n_dom)
    w = 0.36
    down = pivot["down"] if "down" in pivot.columns \
        else pd.Series(np.nan, index=pivot.index)
    up = pivot["up"] if "up" in pivot.columns \
        else pd.Series(np.nan, index=pivot.index)
    ax.bar(x - w / 2, down, w, label="Down AUC")
    ax.bar(x + w / 2, up, w, label="Up AUC")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axhline(0.55, color="r", ls=":", lw=0.8, label="unpredictable gate (0.55)")
    ax.axhline(0.60, color="g", ls=":", lw=0.8, label="mapping gate (0.60)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("validation ROC-AUC")
    ax.set_title("A2: logistic actionability probe (validation split)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_fig / "a2_probe_val_auc.png", dpi=130)
    plt.close(fig)


def _fig_muR(a3df, out_fig):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(a3df["decile_monotone_spearman"], np.zeros(len(a3df)), s=10,
               alpha=0.0)
    ax.bar(np.arange(len(a3df)), a3df["spearman(muR,r)"], color="steelblue")
    ax.set_xticks(np.arange(len(a3df)))
    ax.set_xticklabels(a3df["domain"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Spearman(muR, r_true)")
    ax.set_title("A3: continuous distribution signal mu^R vs realized residual")
    fig.tight_layout()
    fig.savefig(out_fig / "a3_muR_signal.png", dpi=130)
    plt.close(fig)


def _fig_abstention(a4df, out_fig):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    doms = a4df["domain"]
    x = np.arange(len(a4df))
    axes[0].bar(x, a4df["release_rate"].fillna(0), color="steelblue")
    axes[0].set_xticks(x); axes[0].set_xticklabels(doms, rotation=30,
                                                   ha="right", fontsize=7)
    axes[0].set_title("release rate")
    axes[1].bar(x, a4df["realized_gain_release"].fillna(0), color="seagreen")
    axes[1].set_xticks(x); axes[1].set_xticklabels(doms, rotation=30,
                                                   ha="right", fontsize=7)
    axes[1].set_title("realized gain | release")
    axes[2].bar(x, a4df["degradation_frac"].fillna(0) * 100, color="indianred")
    axes[2].set_xticks(x); axes[2].set_xticklabels(doms, rotation=30,
                                                   ha="right", fontsize=7)
    axes[2].set_title("degradation vs host (%)")
    fig.tight_layout()
    fig.savefig(out_fig / "a4_safe_abstention.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variant", type=str, default="learned_sig")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--force-mlp", action="store_true")
    args = ap.parse_args()

    results_dir = HERE / "results"
    if args.artifacts:
        artifact_dir = Path(args.artifacts)
    else:
        dirs = sorted(results_dir.glob("R1A_[0-9]*"), key=lambda p: p.name)
        if not dirs:
            raise SystemExit("no R1A_* artifact dir found under results/")
        artifact_dir = dirs[-1]
    if not artifact_dir.is_dir():
        raise SystemExit(f"artifact dir not found: {artifact_dir}")

    out_dir = Path(args.out) if args.out else \
        results_dir / f"R1A8_DECISION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.8] frozen artifacts: {artifact_dir}")
    print(f"[R1A.8] out: {out_dir}")
    print(f"[R1A.8] variant={args.variant} alpha={args.alpha} "
          f"force_mlp={args.force_mlp}")

    all_a0, all_a1, all_probe, all_rel = [], [], [], []
    all_a3, all_a4s, all_a4d = [], [], []
    for ds_key, bb in R.DOMAINS:
        print(f"[R1A.8] {ds_key}:{bb} ...", flush=True)
        res = analyze_domain(artifact_dir, ds_key, bb, args.variant,
                             alpha=args.alpha, force_mlp=args.force_mlp)
        all_a0.append(res["a0"])
        all_a1.extend(res["a1_rows"])
        all_probe.extend(res["probe_rows"])
        all_rel.extend(res["rel_rows"])
        all_a3.append(res["a3"])
        all_a4s.append(res["a4_summary"])
        all_a4d.extend(res["a4_day_rows"])
        print(f"    k={res['k']} n_mem0={res['n_mem0']} "
              f"n_days={res['n_days']} probs={len(res['problems'])}")
        for pr in res["problems"]:
            print(f"      WARN {pr}")

    a0df = pd.DataFrame(all_a0)
    a1df = pd.DataFrame(all_a1)
    probe_df = pd.DataFrame(all_probe)
    rel_df = pd.DataFrame(all_rel)
    a3df = pd.DataFrame(all_a3)
    a4s_df = pd.DataFrame(all_a4s)
    a4d_df = pd.DataFrame(all_a4d)

    # ---- CSVs ----
    a0df.to_csv(out_dir / "bayes_equivalence.csv", index=False)
    a1df.to_csv(out_dir / "materiality_by_day.csv", index=False)
    agg = a1df.groupby("domain").agg(
        n_days=("host_err_z", "size"),
        host_mae_usd=("host_mae_usd", "mean"),
        host_rmae=("host_rmae", "mean"),
        host_err_z=("host_err_z", "mean"),
        oracle_gain_mean=("oracle_gain", "mean"),
        oracle_gain_median=("oracle_gain", "median"),
        oracle_gain_p10=("oracle_gain", lambda s: np.percentile(s, 10)),
        oracle_gain_p90=("oracle_gain", lambda s: np.percentile(s, 90)),
        oracle_gain_p95=("oracle_gain", lambda s: np.percentile(s, 95)),
        iah_gain_mean=("iah_gain", "mean"),
        fire_rate=("fire", "mean"),
        bayes_fire_rate=("bayes_fire", "mean"),
        mean_abs_gtd=("mean_abs_gtd", "mean"),
        mean_abs_gtu=("mean_abs_gtu", "mean"),
        gain_host_ratio=("gain_host_ratio", "mean"),
    )
    # fire-day realized gain / non-fire-day oracle gain (conditional on current action)
    fire_rows = a1df[a1df["fire"] == 1].groupby("domain")["iah_gain"].mean()
    nf_rows = a1df[a1df["fire"] == 0].groupby("domain")["oracle_gain"].mean()
    agg["fire_day_realized_gain"] = fire_rows
    agg["nonfire_day_oracle_gain"] = nf_rows
    agg["nonfire_frac"] = 1.0 - agg["fire_rate"]
    agg.to_csv(out_dir / "materiality_by_domain.csv")
    probe_df.to_csv(out_dir / "actionability_probe.csv", index=False)
    rel_df.to_csv(out_dir / "probe_reliability.csv", index=False)
    a3df.to_csv(out_dir / "continuous_signal.csv", index=False)
    a4d_df.to_csv(out_dir / "iah_dvg_metrics.csv", index=False)
    a4s_df.to_csv(out_dir / "safe_abstention_summary.csv", index=False)

    # ---- figures ----
    _fig_bayes_fire(a0df, out_fig)
    if len(probe_df):
        _fig_probe_auc(probe_df, out_fig)
    _fig_muR(a3df, out_fig)
    _fig_abstention(a4s_df, out_fig)

    # ---- code_commit + config ----
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{_git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "alpha": args.alpha,
                   "force_mlp": args.force_mlp,
                   "domains": [f"{d}:{b}" for d, b in R.DOMAINS]}, f, indent=2)

    # ---- verdict (plan §14 decision tree) ----
    pjm_a0 = a0df[a0df["domain"] == "LAGO_PJM:MLP"].iloc[0]
    bayes_pjm = pjm_a0["bayes_day_fire_rate"]
    cur_pjm = pjm_a0["current_day_fire_rate"]
    a0_ok = bool((bayes_pjm is not None and bayes_pjm <= 0.25)
                 and (cur_pjm is not None and cur_pjm <= 0.25))

    pjm_probe = probe_df[(probe_df["domain"] == "LAGO_PJM:MLP")
                         & (probe_df["split"] == "val")]
    best_val_auc = None
    if len(pjm_probe):
        best_val_auc = float(pjm_probe["auc"].max())
    probe1_val = pjm_probe[pjm_probe["probe"] == "logistic"] if len(pjm_probe) else \
        pd.DataFrame()
    probe1_val_mean = float(probe1_val["auc"].mean()) if len(probe1_val) else None
    pjm_a4 = a4s_df[a4s_df["domain"] == "LAGO_PJM:MLP"]
    pjm_a4 = pjm_a4.iloc[0] if len(pjm_a4) else None

    reason = ""
    verdict = "ACTION_SIGNAL_UNRESOLVED"
    if not a0_ok:
        verdict = "IMPLEMENTATION_BUG"
        reason = (f"A0: current day fire {cur_pjm:.3f} vs Bayes median day fire "
                  f"{bayes_pjm:.3f} on PJM:MLP; low fire is NOT an exact Bayes "
                  f"consequence (see bayes_wants_abstain / "
                  f"current_acts_bayes_no fractions).")
    elif best_val_auc is not None and best_val_auc >= 0.60:
        verdict = "ACTION_MAPPING_FAILURE"
        reason = (f"A2: best validation probe AUC on PJM:MLP = {best_val_auc:.3f} "
                  f">= 0.60 -> frozen representation carries predictable benefit; "
                  f"(w,m)->action mapping is lossy.")
    elif best_val_auc is not None and best_val_auc <= 0.55:
        a4_ok = bool(pjm_a4 is not None
                     and (pjm_a4["release_rate"] is None
                          or pjm_a4["release_rate"] <= 0.15)
                     and (pjm_a4["harmful_release_rate"] is None
                          or pjm_a4["harmful_release_rate"] <= 0.10)
                     and (pjm_a4["degradation_frac"] is None
                          or abs(pjm_a4["degradation_frac"]) <= 0.03))
        if a4_ok:
            verdict = "SAFE_ABSTENTION_SUPPORTED"
            reason = (f"A2 probe unpredictable on PJM:MLP (best val AUC "
                      f"{best_val_auc:.3f} <= 0.55); A4 IAH-native DVG releases "
                      f"{pjm_a4['release_rate']:.3f} of S4 days, degradation "
                      f"{pjm_a4['degradation_frac']:.4f} -> safe abstention.")
        else:
            verdict = "ACTION_SIGNAL_UNRESOLVED"
            reason = (f"A2 probe unpredictable (best val AUC {best_val_auc:.3f}) "
                      f"but A4 abstention not clean (release "
                      f"{pjm_a4['release_rate'] if pjm_a4 is not None else None}, "
                      f"harm {pjm_a4['harmful_release_rate'] if pjm_a4 is not None else None}).")
    else:
        verdict = "ACTION_SIGNAL_UNRESOLVED"
        reason = (f"A2 probe on PJM:MLP in gray zone (best val AUC "
                  f"{best_val_auc} between 0.55 and 0.60).")

    # ---- DECISION_VERDICT.md ----
    def _f(x):
        return f"{x:.3f}" if x is not None and pd.notna(x) else "NaN"

    L = []
    L.append("# R1A.8 DECISION VERDICT — Bayes action, materiality, actionability probe, safe abstention")
    L.append("")
    L.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- code commit: `{_git_head()}`")
    L.append(f"- frozen R1A artifacts: `{artifact_dir.name}` "
             f"(variant `{args.variant}`, no retrain)")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"### **{verdict}**")
    L.append("")
    L.append(f"- reason: {reason}")
    L.append("")
    L.append("## A0 — Bayes-action equivalence")
    for _, r in a0df.iterrows():
        L.append(f"- {r['domain']}: current day-fire={_f(r['current_day_fire_rate'])} "
                 f"Bayes day-fire={_f(r['bayes_day_fire_rate'])} | "
                 f"hourly exact-match={r['exact_match_frac']:.4f} "
                 f"Bayes-wants-but-abstain={r['bayes_wants_abstain_frac']:.4f} "
                 f"acts-but-Bayes-no={r['current_acts_bayes_no_frac']:.4f} "
                 f"dir-mismatch={r['direction_mismatch_frac']:.4f} "
                 f"ties={r['tie_frac']:.4f}")
    L.append("")
    L.append("## A1 — Materiality")
    for _, r in agg.iterrows():
        L.append(f"- {r.name}: host_err_z={_f(r['host_err_z'])} "
                 f"host_mae_usd={_f(r['host_mae_usd'])} "
                 f"rMAE={_f(r['host_rmae'])} "
                 f"oracle_gain med={_f(r['oracle_gain_median'])} "
                 f"p95={_f(r['oracle_gain_p95'])} "
                 f"fire_rate={_f(r['fire_rate'])} "
                 f"fire_day_gain={_f(r['fire_day_realized_gain'])} "
                 f"nonfire_oracle_gain={_f(r['nonfire_day_oracle_gain'])} "
                 f"|gT↓|={_f(r['mean_abs_gtd'])} |gT↑|={_f(r['mean_abs_gtu'])} "
                 f"gain/host={_f(r['gain_host_ratio'])}")
    L.append("")
    L.append("## A2 — Frozen actionability probe (validation split, PJM:MLP focus)")
    pjm_v = probe_df[probe_df["domain"] == "LAGO_PJM:MLP"]
    for _, r in pjm_v.iterrows():
        L.append(f"- {r['domain']} {r['direction']} {r['probe']} {r['split']}: "
                 f"n={r['n']} pos_rate={_f(r['pos_rate'])} "
                 f"AUC={_f(r['auc'])} PR-AUC={_f(r['pr_auc'])} "
                 f"Brier={_f(r['brier'])} ECE={_f(r['ece'])} "
                 f"rel10={_f(r['rel_spearman'])} "
                 f"top10_enrich_gain={r['top10_enrich_gain']:+.4f}")
    L.append("")
    L.append("## A3 — Continuous distribution signal mu^R")
    for _, r in a3df.iterrows():
        L.append(f"- {r['domain']}: n={r['n_hours']} "
                 f"Spearman(muR,r)={_f(r['spearman(muR,r)'])} "
                 f"sign_acc={_f(r['sign_acc'])} "
                 f"decile_mono={_f(r['decile_monotone_spearman'])} "
                 f"P(muR>0|r>0)={_f(r['P(sign muR>0|r>0)'])} "
                 f"P(muR<0|r<0)={_f(r['P(sign muR<0|r<0)'])}")
    L.append("")
    L.append("## A4 — IAH-native DVG + safe abstention (S4 release)")
    for _, r in a4s_df.iterrows():
        L.append(f"- {r['domain']}: n_calib={int(r['n_calib'])} q={_f(r['q'])} "
                 f"n_eval={int(r['n_eval'])} release={_f(r['release_rate'])} "
                 f"identity={_f(r['identity_rate'])} "
                 f"harm_release={_f(r['harmful_release_rate'])} "
                 f"gain|release={_f(r['realized_gain_release'])} "
                 f"host_mae={_f(r['host_mae'])} final_mae={_f(r['final_mae'])} "
                 f"degrad={_f(r['degradation'])} "
                 f"degrad_frac={_f(r['degradation_frac'])} "
                 f"cov={_f(r['coverage'])} A_I med={_f(r['A_I_median'])}")
    L.append("")
    L.append("## Decision tree (§14)")
    L.append(f"- A0 low-fire is exact Bayes consequence: "
             f"{'YES' if a0_ok else 'NO'} "
             f"(PJM:MLP Bayes day-fire={_f(bayes_pjm)} current={_f(cur_pjm)})")
    L.append(f"- Frozen features predict benefit (PJM:MLP val best AUC): "
             f"{_f(best_val_auc)} -> "
             f"{'predictable' if (best_val_auc or 0) >= 0.60 else ('unpredictable' if (best_val_auc or 1) <= 0.55 else 'gray-zone')}")
    if pjm_a4 is not None:
        L.append(f"- A4 abstention health (PJM:MLP): release="
                 f"{_f(pjm_a4['release_rate'])} harm={_f(pjm_a4['harmful_release_rate'])} "
                 f"degrad_frac={_f(pjm_a4['degradation_frac'])}")
    L.append("")
    L.append("## Notes")
    L.append("- R1A source S4 = DEVELOPMENT DATA; probe thresholds are dev gates, "
             "not theorems.")
    L.append("- Probe 2 (tiny MLP) runs only when Probe 1 val AUC < 0.60 "
             "(plan §5) unless --force-mlp; it is a diagnostic upper bound.")
    L.append("- No candidate retrain; no CRPS/DVG/CAGM math change; no new "
             "trainable production component (plan §7 forbids blind gamma sweep).")
    L.append("- R1B stays paused until SAFE_ABSTENTION_SUPPORTED or an "
             "authorized ACTION_MAPPING_FAILURE calibration layer (plan §11).")
    L.append("")

    verdict_text = "\n".join(L)
    with open(out_dir / "DECISION_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.8] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
