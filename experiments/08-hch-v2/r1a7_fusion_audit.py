"""R1A.7 — Prior–Local value fusion (F0–F4) + PJM:MLP failure audit.

Spec: docs/paper_prep/v2_final_prep/hch_v2_r1a7_prior_local_fusion_failure_audit_compute_plan_v0.1_2026-08-13.md

Scope (per user): ONLY R1A.7. NO R1B, NO candidate retrain, NO CRPS change,
NO DVG change, no new trainable network. Frozen R1A candidate (variant
learned_sig). R1A source S4 = DEVELOPMENT DATA (plan §10).

Core (§3): IAH-native analytic directional gains g^I and V4-prequential
weighted local directional gains g^L are fused BEFORE the double-event
optimizer:

    g^{(\lambda)} = (1-\lambda) g^I + \lambda g^L,  \lambda in {0,.25,.5,.75,1}

double_event(g^{(\lambda)}) -> pi_q^{(\lambda)}.  BOTH the IAH prior and the
local replay then evaluate the SAME final action pi_q^{(\lambda)}:

    A^I_q(pi),  A^L_q(pi),  A^{(\lambda)}_q = (1-\lambda) A^I_q + \lambda A^L_q

Boundaries: lambda=0 == V1 (IAHNativeValue, plan §3,§6); lambda=1 == V4 with
the §3-proper weighted proposal (R1A.6's V4 kept the uniform proposal — plan
§2.2 caveat, so F4 is V4 generalized, not numerically identical).

λ protocol (§4): only the 5 fixed λ; selection ONLY on S3M-validation, in
order worst-domain Spearman -> macro Spearman -> top10 realized-gain
enrichment. No per-market/per-day/learned λ. "不要为了 NEM 的大收益牺牲 PJM".

Failure audit (§5): 5.1 same-market host contrast (PJM:Linear vs PJM:MLP);
5.2 IAH directional calibration (AUC/reliability of w± vs true benefit B±);
5.3 learned-key degeneracy (norm, pairwise cosine, NN distance, concentration,
distinct-neighbor count); 5.4 local weight reliability (ESS_q, top weight,
median distance, error vs ESS, Spearman by ESS tercile). Diagnostics only.

Gate (§7) evaluated on the S3M-val-selected λ over full eval data; bootstrap
= 7-day moving block, 500+ samples, deltas vs V0 (continuity) and vs F0 (the
fusion's own marginal effect). Verdict (§8) names Case A (atom/action
calibration), B (key degeneracy), or C (local evidence high variance) when
fusion stays YELLOW.

Outputs -> R1A7_FUSION_<ts>/ :
    code_commit.txt, fusion_config.json, fusion_by_day.csv,
    fusion_metrics_by_domain.csv, fusion_metrics_by_block.csv,
    top_decile_enrichment.csv, lambda_selection.csv, bootstrap_intervals.csv,
    audit_host_contrast.csv, audit_residual_drift.csv,
    audit_directional_calib.csv, audit_key_degeneracy.csv,
    audit_weight_reliability.csv,
    figures/{spearman_by_lambda,decile_gain,ess_error,dirmismatch_lambda}_*.png,
    FUSION_VERDICT.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Windows GBK console cannot encode e.g. U+207B "⁻"; force UTF-8 on std streams
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1a5_diag as D
import r1a_run as R
import r1a6_value_recovery as V6
from hch_v2_bundle import HCHV2Bundle
from double_event import double_event_proposal
from query_replay import (estimate_realized_A, form_final_pi,
                          replay_query_dose)

EPS = 1e-12
LAMBDAS = [0.0, 0.25, 0.50, 0.75, 1.0]
EST_TAGS = [f"F{i}" for i in range(len(LAMBDAS))]


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


def _bin_rel(score, label, nbins=10):
    """Bin score into deciles; Spearman of bin-mean score vs bin-mean label."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    qs = np.quantile(score, np.linspace(0, 1, nbins + 1))
    xs, ys = [], []
    for i in range(nbins):
        lo = -np.inf if i == 0 else qs[i]
        hi = np.inf if i == nbins - 1 else qs[i + 1]
        if i < nbins - 1:
            m = (score >= lo) & (score < hi)
        else:
            m = (score >= lo) & (score <= hi)
        if m.sum() >= 1:
            xs.append(score[m].mean())
            ys.append(label[m].mean())
    if len(xs) >= 3 and np.std(ys) > 0:
        return float(stats.spearmanr(xs, ys)[0])
    return None


def _crps3(mminus, mzero, mplus, wminus, wzero, wplus, err):
    """CRPS of the 3-atom IAH measure at observed err (per hour)."""
    atoms = np.stack([mminus, mzero, mplus], axis=-1)
    ws = np.stack([wminus, wzero, wplus], axis=-1)
    e = err[..., None]
    term1 = np.sum(ws * np.abs(atoms - e), axis=-1)
    term2 = np.zeros_like(err)
    for i in range(3):
        for j in range(3):
            term2 += ws[..., i] * ws[..., j] * np.abs(atoms[..., i] - atoms[..., j])
    return term1 - 0.5 * term2


def _hit_lift(Ah, At):
    n = len(Ah)
    if n < 2:
        return None
    n_top = max(1, n // 10)
    top_idx = np.argsort(Ah)[-n_top:]
    pos_all = float((At > 0).mean())
    pos_top = float((At[top_idx] > 0).mean())
    return pos_top - pos_all


# --------------------------------------------------- weighted local gains -----
def weighted_directional_gains(memory, neighbor_indices, weights,
                               m_minus_q, m_plus_q, query_valid):
    """g^L: per-hour local directional gain, weighted mean over neighbors.

    Same contract as query_replay.build_directional_gains (invalid hours
    excluded from the mean), but each neighbor contributes with weight
    omega_j = exp[-D_j/(tau+eps)]/sum instead of uniformly.
    """
    H = len(m_minus_q)
    g_down_accum = np.zeros(H, dtype=np.float64)
    g_up_accum = np.zeros(H, dtype=np.float64)
    wsum = np.zeros(H, dtype=np.float64)
    for idx, w in zip(neighbor_indices, weights):
        r_down = replay_query_dose(memory.z0[idx], memory.target_zY[idx],
                                   -np.asarray(m_minus_q, dtype=np.float64),
                                   memory.valid_mask[idx], query_valid)
        r_up = replay_query_dose(memory.z0[idx], memory.target_zY[idx],
                                 np.asarray(m_plus_q, dtype=np.float64),
                                 memory.valid_mask[idx], query_valid)
        vd = r_down["valid"]
        g_down_accum[vd] += w * r_down["g"][vd]
        g_up_accum[vd] += w * r_up["g"][vd]
        wsum[vd] += w
    g_hat_down = np.divide(g_down_accum, wsum, out=np.zeros_like(g_down_accum),
                           where=wsum > 0)
    g_hat_up = np.divide(g_up_accum, wsum, out=np.zeros_like(g_up_accum),
                         where=wsum > 0)
    return {"g_hat_down": g_hat_down, "g_hat_up": g_hat_up}


def weighted_action_value(memory, neighbor_indices, weights, pi_q, query_valid):
    """A^L(pi) = sum_j omega_j A_{q->j}(pi) — local value of a given action."""
    total = 0.0
    for idx, w in zip(neighbor_indices, weights):
        r = replay_query_dose(memory.z0[idx], memory.target_zY[idx], pi_q,
                              memory.valid_mask[idx], query_valid)
        total += w * r["A"]
    return float(total)


def fusion_day(preq_mem, nbr, D_j, mm, mp, wm, wp, z0, zY, vm,
               gI_down, gI_up):
    """All λ for one day (plan §3). Returns (rows, weights_dict|None).

    rows: list over λ of {lam, pi, A_I, A_L, A_hat, A_true}.
    weights_dict (for §5.4) = {tau, wts, D_j} when nbr non-empty else None.
    Empty nbr => every λ reduces to F0 (λ=0); rows carry identical A_hat/A_true
    for all λ so the per-λ aggregations stay complete.
    """
    if not nbr:
        prop = double_event_proposal(gI_down, gI_up)
        pi = form_final_pi(mm, mp, prop["I_down"], prop["I_up"])
        A_I = float((gI_down * (pi < 0) + gI_up * (pi > 0))[vm].sum()
                    / vm.sum()) if vm.any() else 0.0
        A_true = estimate_realized_A(z0, zY, pi, vm)
        rows = [{"lam": lam, "pi": pi.copy(), "A_I": A_I, "A_L": 0.0,
                 "A_hat": A_I, "A_true": A_true} for lam in LAMBDAS]
        return rows, None

    tau = float(np.median(D_j))
    wts = np.exp(-D_j / (tau + EPS))
    wts = wts / (wts.sum() + EPS)
    gL = weighted_directional_gains(preq_mem, nbr, wts, mm, mp, vm)
    rows = []
    for lam in LAMBDAS:
        g_down = (1.0 - lam) * gI_down + lam * gL["g_hat_down"]
        g_up = (1.0 - lam) * gI_up + lam * gL["g_hat_up"]
        prop = double_event_proposal(g_down, g_up)
        pi = form_final_pi(mm, mp, prop["I_down"], prop["I_up"])
        A_I = float((gI_down * (pi < 0) + gI_up * (pi > 0))[vm].sum()
                    / vm.sum()) if vm.any() else 0.0
        A_L = weighted_action_value(preq_mem, nbr, wts, pi, vm)
        A_hat = (1.0 - lam) * A_I + lam * A_L
        A_true = estimate_realized_A(z0, zY, pi, vm)
        rows.append({"lam": lam, "pi": pi, "A_I": A_I, "A_L": A_L,
                     "A_hat": A_hat, "A_true": A_true})
    return rows, {"tau": tau, "wts": wts, "D_j": np.asarray(D_j, dtype=np.float64)}


def _direction_mismatch(mm, mp, z0, zY, vm, pi):
    """Fraction of executed valid hours whose chosen direction has non-positive
    true benefit (g_true_down<=0 for down actions, g_true_up<=0 for up)."""
    r = zY - z0
    g_true_down = np.abs(r) - np.abs(r + mm)
    g_true_up = np.abs(r) - np.abs(r - mp)
    exec_down = (pi < 0) & vm
    exec_up = (pi > 0) & vm
    n_exec = int(exec_down.sum() + exec_up.sum())
    if n_exec == 0:
        return None
    bad = int((exec_down & (g_true_down <= 0)).sum()
              + (exec_up & (g_true_up <= 0)).sum())
    return bad / n_exec


# ------------------------------------------------------------- per domain ----
def analyze_domain(artifact_dir: Path, ds_key: str, bb: str,
                   variant: str = "learned_sig") -> dict:
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

    # ---- split bookkeeping (mirror r1a6) ----
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

    # ---- capture learned keys + candidate outputs for every day ----
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

    # ---- cross-check my forward atoms == frozen pipe memory atoms ----
    for j, d in enumerate(mem_dates):
        if d not in daymap:
            problems.append(f"{domain}: memory date {d} missing from daymap")
            continue
        day = daymap[d]["day"]
        for attr in ("z0", "w_minus", "w_zero", "w_plus", "m_minus", "m_plus",
                     "valid_mask"):
            mine = _fl(day["candidate"][attr])
            ref = np.asarray(getattr(pipe.memory, attr)[j], dtype=np.float64)
            if not np.allclose(mine, ref, atol=1e-9):
                problems.append(f"{domain}: my {attr}[{d}] != frozen memory")

    eval_dates = sorted(d for d in daymap
                        if block_of(d) in ("S3M-val", "S3C") or
                        (block_of(d) or "").startswith("S4Q"))
    if not eval_dates:
        raise ValueError(f"{domain}: no evaluation days")

    # ---- prequential expanding memory with learned keys (F1–F4) ----
    preq_mem = V6.KeyMemory()
    for d in mem_dates:
        if d in daymap:
            preq_mem.add_day(d, daymap[d]["day"]["candidate"],
                             daymap[d]["day"]["target_zY"], daymap[d]["key"])

    # ---- oracle per day (estimator-independent) ----
    oracleA: dict = {}
    for d in eval_dates:
        cand = daymap[d]["day"]["candidate"]
        zY = np.asarray(daymap[d]["day"]["target_zY"]).reshape(-1)
        o = D.oracle_for_day({"z0": cand["z0"], "m_minus": cand["m_minus"],
                              "m_plus": cand["m_plus"],
                              "valid_mask": cand["valid_mask"]}, zY)
        oracleA[d] = float(o["A"])

    # ---- main per-day loop ----
    rows = []            # fusion by-day rows
    v0_rows = []         # CurrentW1Static (for bootstrap baseline)
    weight_days = []     # §5.4 per-day ESS/top-weight/median-distance/error
    calib_hours = {      # §5.2 pooled per-hour vectors
        "r": [], "g_true_down": [], "g_true_up": [],
        "wm": [], "wp": [], "gI_down": [], "gI_up": [],
    }
    n_fire_lam0 = 0      # days where the pure-IAH double-event fires (§5.2)
    atom_scale = []      # per-day median |m^-| over valid hours (§5.2)
    keys_all = np.vstack([daymap[d]["key"] for d in sorted(daymap)])

    def _basic(d):
        day = daymap[d]["day"]
        cand = day["candidate"]
        z0 = _fl(cand["z0"])
        zY = np.asarray(day["target_zY"]).reshape(-1)
        mm = _fl(cand["m_minus"]); mp = _fl(cand["m_plus"])
        wm = _fl(cand["w_minus"]); wp = _fl(cand["w_plus"])
        vm = _fl(cand["valid_mask"]).astype(bool)
        return cand, z0, zY, mm, mp, wm, wp, vm

    for d in eval_dates:
        blk = block_of(d); q = qmap.get(d)
        cand, z0, zY, mm, mp, wm, wp, vm = _basic(d)
        gI_down = mm * (2.0 * wm - 1.0)
        gI_up = mp * (2.0 * wp - 1.0)
        oA = oracleA[d]

        # V0 baseline (W1 + static memory + uniform mean) for bootstrap continuity
        A0_hat, A0_true = pipe._replay_value(daymap[d]["day"], k)
        v0_rows.append({"domain": domain, "block": blk, "quarter": q,
                        "date": _iso(d), "A_hat": float(A0_hat),
                        "A_true": float(A0_true)})

        nbr, dists = preq_mem.neighbors_cos(daymap[d]["key"], k)
        D_j = np.asarray([dists[j] for j in nbr], dtype=np.float64) if nbr else \
            np.zeros(0)
        fres, wdict = fusion_day(preq_mem, nbr, D_j, mm, mp, wm, wp, z0, zY, vm,
                                 gI_down, gI_up)
        lam1 = fres[-1]
        for fr in fres:
            lam = fr["lam"]
            A_hat, A_true, pi = fr["A_hat"], fr["A_true"], fr["pi"]
            eta = max(A_true, 0.0) / max(oA, EPS)
            missed = 1 if (oA > 1e-9 and A_true <= 0.0) else 0
            dm = _direction_mismatch(mm, mp, z0, zY, vm, pi)
            rows.append({"domain": domain, "block": blk, "quarter": q,
                         "date": _iso(d), "lambda": float(lam),
                         "estimator": EST_TAGS[LAMBDAS.index(lam)],
                         "A_hat": float(A_hat), "A_I": float(fr["A_I"]),
                         "A_L": float(fr["A_L"]), "A_true": float(A_true),
                         "E": float(A_hat - A_true), "A_oracle": float(oA),
                         "eta": float(eta), "missed": int(missed),
                         "direction_mismatch": float(dm) if dm is not None else None,
                         "n_neighbors": int(len(nbr)), "k": int(k)})
        # §5.2 pure-IAH action-fire diagnostics (λ=0 double-event fires?)
        fired0 = bool(np.any(fres[0]["pi"] != 0.0))
        if fired0:
            n_fire_lam0 += 1
        atom_scale.append(float(np.median(np.abs(mm)[vm])))
        # §5.4 (weights are λ-independent; use the λ=1 branch)
        if wdict is not None:
            wts = wdict["wts"]
            ess = float(1.0 / np.sum(wts ** 2))
            weight_days.append({"domain": domain, "date": _iso(d), "block": blk,
                                "ESS": ess, "top_weight": float(wts.max()),
                                "median_D": float(np.median(wdict["D_j"])),
                                "tau": wdict["tau"],
                                "A_hat": float(lam1["A_hat"]),
                                "A_true": float(lam1["A_true"]),
                                "abs_err": float(abs(lam1["A_hat"] - lam1["A_true"])),
                                "n_neighbors": int(len(nbr))})
        # §5.2 pooled valid hours
        r = zY - z0
        vv = vm
        calib_hours["r"].append(r[vv])
        calib_hours["g_true_down"].append((np.abs(r) - np.abs(r + mm))[vv])
        calib_hours["g_true_up"].append((np.abs(r) - np.abs(r - mp))[vv])
        calib_hours["wm"].append(wm[vv])
        calib_hours["wp"].append(wp[vv])
        calib_hours["gI_down"].append(gI_down[vv])
        calib_hours["gI_up"].append(gI_up[vv])

        # after outcome: append day's pre-outcome key + atoms + zY + mask
        preq_mem.add_day(d, day["candidate"], day["target_zY"], daymap[d]["key"])

    pdf = pd.DataFrame(rows)
    v0df = pd.DataFrame(v0_rows)

    # ---- per domain x λ aggregation (§7) ----
    metrics, block_metrics, topdec = [], [], []
    for (lam, est), g in pdf.groupby(["lambda", "estimator"]):
        Ah = g["A_hat"].to_numpy(); At = g["A_true"].to_numpy()
        n = len(g)
        n_top = max(1, n // 10)
        top_idx = np.argsort(Ah)[-n_top:]
        pos_all = float((At > 0).mean()) if n else None
        pos_top = float((At[top_idx] > 0).mean())
        lift = (pos_top - pos_all) if pos_all is not None else None
        dms = g["direction_mismatch"].dropna()
        m = {
            "domain": domain, "lambda": float(lam), "estimator": est, "n": n,
            "spearman": _sp(Ah, At), "pearson": _pear(Ah, At),
            "mae": float(np.mean(np.abs(Ah - At))) if n else None,
            "bias": float(np.mean(Ah - At)) if n else None,
            "P(A>0|Ah>0)": float((At > 0)[Ah > 0].mean())
                           if (Ah > 0).any() else None,
            "P(A>0|top10)": pos_top, "P(A>0|all)": pos_all,
            "hit_lift_top10": lift,
            "mean_A_true_top10": float(At[top_idx].mean()),
            "median_A_true_top10": float(np.median(At[top_idx])),
            "mean_A_true_all": float(At.mean()) if n else None,
            "realized_A_mean": float(At.mean()) if n else None,
            "eta_mean": float(g["eta"].mean()) if n else None,
            "missed_positive_rate": float(g["missed"].mean()) if n else None,
            "direction_mismatch": float(dms.mean()) if len(dms) else None,
        }
        metrics.append(m)
        for qq in (1, 2, 3, 4):
            gq = g[g["quarter"] == qq]
            m[f"rho_S4Q{qq}"] = _sp(gq["A_hat"].to_numpy(),
                                    gq["A_true"].to_numpy()) if len(gq) > 1 else None
        m["s4_quarter_drift"] = None
        if m.get("rho_S4Q1") is not None and m.get("rho_S4Q4") is not None:
            m["s4_quarter_drift"] = m["rho_S4Q4"] - m["rho_S4Q1"]
        for blk, gb in g.groupby("block"):
            Ahb = gb["A_hat"].to_numpy(); Atb = gb["A_true"].to_numpy()
            block_metrics.append({
                "domain": domain, "lambda": float(lam), "estimator": est,
                "block": blk, "n": len(gb), "spearman": _sp(Ahb, Atb),
                "mae": float(np.mean(np.abs(Ahb - Atb))) if len(gb) else None,
                "bias": float(np.mean(Ahb - Atb)) if len(gb) else None,
                "mean_A_hat": float(Ahb.mean()) if len(gb) else None,
                "mean_A_true": float(Atb.mean()) if len(gb) else None,
            })
        topdec.append({
            "domain": domain, "lambda": float(lam), "estimator": est,
            "n_top": n_top, "P(A>0|top10)": pos_top, "P(A>0|all)": pos_all,
            "hit_lift_top10": lift,
            "mean_A_true_top10": float(At[top_idx].mean()),
            "median_A_true_top10": float(np.median(At[top_idx])),
            "mean_A_true_all": float(At.mean()) if n else None,
        })

    # ---- §5.1 host-contrast + residual drift ----
    host_rows, daily_med = [], []
    ts = info.ds["ts"]
    y_full = info.ds["price"].astype(np.float64)
    for d in eval_dates:
        cand = daymap[d]["day"]["candidate"]
        idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
        yhat = info.yhat_full[idxs].astype(np.float64)
        price = y_full[idxs]
        vm = _fl(cand["valid_mask"]).astype(bool)
        z0 = _fl(cand["z0"])
        zY = np.asarray(daymap[d]["day"]["target_zY"]).reshape(-1)
        wm = _fl(cand["w_minus"]); wp = _fl(cand["w_plus"])
        w0 = _fl(cand["w_zero"])
        mm = _fl(cand["m_minus"]); mp = _fl(cand["m_plus"])
        err = zY - z0
        if not vm.any():
            continue
        host_mae = float(np.mean(np.abs(price[vm] - yhat[vm])))
        iah_crps = _crps3(-mm[vm], np.zeros(vm.sum()), mp[vm],
                          wm[vm], w0[vm], wp[vm], err[vm])
        host_ae = float(np.mean(np.abs(err[vm])))
        daily_med.append(float(np.median(err[vm])))
        host_rows.append({
            "domain": domain, "date": _iso(d), "block": block_of(d),
            "host_mae": host_mae,
            "host_rmse": float(np.sqrt(np.mean((price[vm] - yhat[vm]) ** 2))),
            "host_rmae": (host_mae / float(np.mean(np.abs(price[vm])))
                          if np.mean(np.abs(price[vm])) > 0 else None),
            "hyp_host_err": host_ae,
            "iah_crps": float(np.mean(iah_crps)),
            "crps_delta": host_ae - float(np.mean(iah_crps)),
            "resid_mean": float(np.mean(err[vm])),
            "resid_std": float(np.std(err[vm])),
            "resid_iqr": float(np.percentile(err[vm], 75)
                               - np.percentile(err[vm], 25)),
            "sign_balance": float((err[vm] > 0).mean()),
            "lag1_autocorr": _sp(err[vm][:-1], err[vm][1:])
                             if vm.sum() > 2 else None,
        })
    host_df = pd.DataFrame(host_rows)
    resid_all = np.concatenate(calib_hours["r"]) if calib_hours["r"] else \
        np.zeros(0)
    sig2 = float(np.std(resid_all)) if resid_all.size else 0.0
    daily_med = np.asarray(daily_med, dtype=np.float64)
    daily_ac = _sp(daily_med[:-1], daily_med[1:]) if len(daily_med) > 2 else None
    host_agg = []
    for dom, g in host_df.groupby("domain"):
        lag1 = np.asarray([x for x in g["lag1_autocorr"] if pd.notna(x)])
        host_agg.append({
            "domain": dom, "n": len(g),
            "host_mae": float(np.mean(g["host_mae"].dropna())),
            "host_rmse": float(np.sqrt(np.mean(g["host_rmse"].dropna() ** 2))),
            "host_rmae": float(np.mean(g["host_rmae"].dropna())),
            "hyp_host_err": float(np.mean(g["hyp_host_err"].dropna())),
            "iah_crps": float(np.mean(g["iah_crps"].dropna())),
            "crps_delta": float(np.mean(g["crps_delta"].dropna())),
            "resid_mean": float(np.mean(g["resid_mean"].dropna())),
            "resid_std": float(np.mean(g["resid_std"].dropna())),
            "resid_iqr": float(np.mean(g["resid_iqr"].dropna())),
            "sign_balance": float(np.mean(g["sign_balance"].dropna())),
            "lag1_autocorr": float(np.mean(lag1)) if lag1.size else None,
            "daily_autocorr": daily_ac,
            "P(resid>+2sd)": float((resid_all > 2 * sig2).mean()),
            "P(resid<-2sd)": float((resid_all < -2 * sig2).mean()),
        })
    host_agg_df = pd.DataFrame(host_agg)
    # residual drift by temporal block (S1R/S2V pre-correction host residual is
    # captured by hyp_host_err; here we show the S3 -> S4 evolution)
    drift_rows = []
    for dom, g in host_df.groupby("domain"):
        for blk, gb in g.groupby("block"):
            drift_rows.append({"domain": dom, "block": blk, "n": len(gb),
                               "resid_mean": float(np.mean(gb["resid_mean"])),
                               "abs_resid_mean": float(np.mean(
                                   np.abs(np.asarray(gb["resid_mean"]))))})
    drift_df = pd.DataFrame(drift_rows)

    # ---- §5.2 directional calibration (pooled valid hours) ----
    r_all = np.concatenate(calib_hours["r"]) if calib_hours["r"] else \
        np.zeros(0)
    gtd = np.concatenate(calib_hours["g_true_down"])
    gtu = np.concatenate(calib_hours["g_true_up"])
    wm_all = np.concatenate(calib_hours["wm"])
    wp_all = np.concatenate(calib_hours["wp"])
    gId = np.concatenate(calib_hours["gI_down"])
    gIu = np.concatenate(calib_hours["gI_up"])
    Bm = gtd > 0.0
    Bp = gtu > 0.0
    fire_rate = (n_fire_lam0 / len(eval_dates)
                 if eval_dates else float("nan"))
    calib_rows = [{
        "domain": domain, "n_hours": len(r_all),
        "P(B->0)": float(np.mean(Bm)), "P(B+>0)": float(np.mean(Bp)),
        "AUC(w-,B-)": _auc(wm_all, Bm), "AUC(w+,B+)": _auc(wp_all, Bp),
        "Spearman(gI_down,g_true_down)": _sp(gId, gtd),
        "Spearman(gI_up,g_true_up)": _sp(gIu, gtu),
        "Spearman(w-,g_true_down)": _sp(wm_all, gtd),
        "Spearman(w+,g_true_up)": _sp(wp_all, gtu),
        "rel10_spearman_down": _bin_rel(wm_all, Bm),
        "rel10_spearman_up": _bin_rel(wp_all, Bp),
        "median_abs_mminus": float(np.median(atom_scale)) if atom_scale else None,
        "fire_rate_lambda0": float(fire_rate),
    }]

    # ---- §5.3 learned-key degeneracy ----
    K = keys_all
    nk = np.linalg.norm(K, axis=1) + EPS
    cos = np.clip((K @ K.T) / (nk[:, None] * nk[None, :]), -1.0, 1.0)
    Dmat = 1.0 - cos
    iu = np.triu_indices(len(K), k=1)
    off = Dmat[iu]
    uniq = set()
    jac = []
    prev = None
    for d in eval_dates:
        nbr_q, _ = preq_mem.neighbors_cos(daymap[d]["key"], k)
        if not nbr_q:
            continue
        cur = set(nbr_q)
        uniq |= cur
        if prev is not None:
            inter = len(cur & prev)
            union = len(cur | prev)
            jac.append(inter / union if union else 0.0)
        prev = cur
    key_agg = {
        "domain": domain, "n_days": len(K),
        "key_norm_mean": float(np.mean(nk)), "key_norm_std": float(np.std(nk)),
        "pairwise_cos_mean": float(np.mean(cos[iu])),
        "pairwise_cos_median": float(np.median(cos[iu])),
        "pairwise_dist_mean": float(np.mean(off)),
        "pairwise_dist_median": float(np.median(off)),
        "pairwise_dist_std": float(np.std(off)),
        "dist_concentration": float(np.std(off) / (np.median(off) + EPS)),
        "nn_dist_mean": float(np.mean(np.min(Dmat + np.eye(len(K)) * 2.0,
                                            axis=1))),
        "unique_neighbors": len(uniq),
        "memory_days": n_mem0,
        "neighbor_coverage": len(uniq) / n_mem0 if n_mem0 else None,
        "stickiness_jaccard": float(np.mean(jac)) if jac else None,
    }

    # ---- §5.4 local weight reliability (per-day aggregated) ----
    wdf = pd.DataFrame(weight_days)
    wr_rows = []
    if len(wdf):
        qs = np.quantile(wdf["ESS"], [0.1, 0.5, 0.9])
        terciles = np.quantile(wdf["ESS"], [1 / 3, 2 / 3])
        wr = {
            "domain": domain, "n_days": len(wdf),
            "ESS_p10": float(qs[0]), "ESS_median": float(qs[1]),
            "ESS_p90": float(qs[2]),
            "top_weight_median": float(np.median(wdf["top_weight"])),
            "median_D_median": float(np.median(wdf["median_D"])),
            "Spearman(abs_err,ESS)": _sp(wdf["abs_err"].to_numpy(),
                                         wdf["ESS"].to_numpy()),
            "Spearman(Ahat,Atrue)": _sp(wdf["A_hat"].to_numpy(),
                                        wdf["A_true"].to_numpy()),
        }
        for name, lo, hi in (("low", -np.inf, terciles[0]),
                             ("mid", terciles[0], terciles[1]),
                             ("high", terciles[1], np.inf)):
            g = wdf[(wdf["ESS"] > lo) & (wdf["ESS"] <= hi)]
            if len(g) > 1:
                wr[f"rho_{name}_ESS"] = _sp(g["A_hat"].to_numpy(),
                                            g["A_true"].to_numpy())
            else:
                wr[f"rho_{name}_ESS"] = None
        wr_rows.append(wr)

    return {"rows": rows, "v0_rows": v0_rows, "metrics": metrics,
            "block_metrics": block_metrics, "topdec": topdec,
            "host_agg": host_agg_df, "drift": drift_df,
            "calib": calib_rows, "key_deg": key_agg,
            "weight_rel": wr_rows, "weight_days": weight_days,
            "problems": problems, "k": k, "n_mem0": n_mem0,
            "n_eval_days": len(eval_dates)}


# ------------------------------------------------------- macro aggregation ----
def macro_metrics(metrics: list[dict]) -> dict:
    """Per-λ macro summary over 6 domains (worst, excl-best, PJM:MLP check)."""
    out = {}
    for lam in LAMBDAS:
        ms = [m for m in metrics if np.isclose(m["lambda"], lam)]
        sps = [m["spearman"] for m in ms if m["spearman"] is not None]
        macro = float(np.mean(sps)) if sps else float("nan")
        n_pos = int(sum(1 for s in sps if s > 0))
        worst = float(min(sps)) if sps else float("nan")
        excl = float(np.mean(sorted(sps)[:-1])) if len(sps) >= 2 else macro
        lifts = [m["hit_lift_top10"] for m in ms
                 if m["hit_lift_top10"] is not None]
        dms = [m["direction_mismatch"] for m in ms
               if m["direction_mismatch"] is not None]
        pjm_mlp = next((m["spearman"] for m in ms
                        if m["domain"] == "LAGO_PJM:MLP"), None)
        out[float(lam)] = {
            "macro_spearman": macro, "n_pos_domains": n_pos,
            "worst_domain_rho": worst, "macro_excl_best": excl,
            "macro_hit_lift": float(np.mean(lifts)) if lifts else float("nan"),
            "macro_direction_mismatch": float(np.mean(dms)) if dms else float("nan"),
            "pjm_mlp_rho": pjm_mlp,
        }
    return out


def lambda_select(val_metrics: list[dict]) -> tuple[float, dict]:
    """§4 provisional λ selection (S3M-val rows only): worst -> macro -> top10."""
    macro = {}
    for lam in LAMBDAS:
        ms = [m for m in val_metrics if np.isclose(m["lambda"], lam)]
        sps = [m["spearman"] for m in ms if m["spearman"] is not None]
        lifts = [m["hit_lift_top10"] for m in ms
                 if m["hit_lift_top10"] is not None]
        macro[float(lam)] = {
            "worst_domain_rho": float(min(sps)) if sps else float("nan"),
            "macro_spearman": float(np.mean(sps)) if sps else float("nan"),
            "macro_hit_lift": float(np.mean(lifts)) if lifts else float("nan"),
        }
    best_lam, best_key = None, None
    for lam in LAMBDAS:
        m = macro[float(lam)]
        key = (m["worst_domain_rho"], m["macro_spearman"], m["macro_hit_lift"])
        if best_key is None or key > best_key:
            best_key, best_lam = key, lam
    return best_lam, macro


# ---------------------------------------------------------------- bootstrap ---
def _block_bootstrap_delta(est_map, base_map, rng, n_boot=500, block=7):
    """Moving-block bootstrap of macro Spearman delta + top10-gain delta
    (estimator vs base). Same block indices per domain for pairing."""
    domains = [d for d in base_map if d in est_map]
    sp_delta, t10_delta = [], []
    for _b in range(n_boot):
        est_sp, base_sp, est_t10, base_t10 = [], [], [], []
        for dom in domains:
            Ah_e, At_e = est_map[dom]
            Ah_b, At_b = base_map[dom]
            n = len(At_e)
            nblocks = int(np.ceil(n / block))
            high = max(1, n - block + 1)
            starts = rng.integers(0, high, size=nblocks)
            idx = np.concatenate([np.arange(s, min(s + block, n))
                                  for s in starts])
            if len(idx) < 2:
                continue
            sp_e = _sp(Ah_e[idx], At_e[idx])
            sp_b = _sp(Ah_b[idx], At_b[idx])
            if sp_e is not None and sp_b is not None:
                est_sp.append(sp_e); base_sp.append(sp_b)
            n_top = max(1, len(idx) // 10)
            t_e = float(At_e[idx][np.argsort(Ah_e[idx])[-n_top:]].mean())
            t_b = float(At_b[idx][np.argsort(Ah_b[idx])[-n_top:]].mean())
            est_t10.append(t_e); base_t10.append(t_b)
        if not est_sp:
            continue
        sp_delta.append(float(np.mean(est_sp) - np.mean(base_sp)))
        t10_delta.append(float(np.mean(est_t10) - np.mean(base_t10)))

    def ci(a):
        return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) \
            if a else (float("nan"), float("nan"))

    return ci(sp_delta), ci(t10_delta)


def bootstrap_all(all_rows, v0_rows, n_boot=500, block=7, seed=0):
    pdf = pd.DataFrame(all_rows)
    v0df = pd.DataFrame(v0_rows)
    domains = sorted(pdf["domain"].unique())
    rng = np.random.default_rng(seed)
    v0_map = {}
    for dom in domains:
        g = v0df[v0df["domain"] == dom].sort_values("date")
        v0_map[dom] = (g["A_hat"].to_numpy(), g["A_true"].to_numpy())
    f0_map = {}
    for dom in domains:
        g = pdf[(pdf["domain"] == dom)
                & np.isclose(pdf["lambda"], 0.0)].sort_values("date")
        f0_map[dom] = (g["A_hat"].to_numpy(), g["A_true"].to_numpy())
    out = []
    for lam in LAMBDAS:
        est_map = {}
        for dom in domains:
            g = pdf[(pdf["domain"] == dom)
                    & np.isclose(pdf["lambda"], lam)].sort_values("date")
            est_map[dom] = (g["A_hat"].to_numpy(), g["A_true"].to_numpy())
        sp_vs0, t10_vs0 = _block_bootstrap_delta(est_map, v0_map, rng,
                                                 n_boot=n_boot, block=block)
        if lam > 0.0:
            sp_vsF0, t10_vsF0 = _block_bootstrap_delta(est_map, f0_map, rng,
                                                       n_boot=n_boot,
                                                       block=block)
        else:
            sp_vsF0, t10_vsF0 = ((0.0, 0.0), (0.0, 0.0))
        out.append({
            "lambda": float(lam), "estimator": EST_TAGS[LAMBDAS.index(lam)],
            "macro_spearman_delta_vs_V0_ci_lo": sp_vs0[0],
            "macro_spearman_delta_vs_V0_ci_hi": sp_vs0[1],
            "top10_gain_delta_vs_V0_ci_lo": t10_vs0[0],
            "top10_gain_delta_vs_V0_ci_hi": t10_vs0[1],
            "macro_spearman_delta_vs_F0_ci_lo": sp_vsF0[0],
            "macro_spearman_delta_vs_F0_ci_hi": sp_vsF0[1],
            "top10_gain_delta_vs_F0_ci_lo": t10_vsF0[0],
            "top10_gain_delta_vs_F0_ci_hi": t10_vsF0[1],
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures -----
def plot_figures(out_fig: Path, pdf: pd.DataFrame, domain: str):
    dom = domain.replace(":", "_")
    g = pdf[pdf["domain"] == domain]
    colors = plt.cm.viridis(np.linspace(0, 1, len(LAMBDAS)))

    # spearman by lambda
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lam, gl in g.groupby("lambda"):
        Ah = gl["A_hat"].to_numpy(); At = gl["A_true"].to_numpy()
        ax.plot([float(lam)], [_sp(Ah, At)], "o", ms=6,
                color=colors[LAMBDAS.index(lam)])
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(LAMBDAS)
    ax.set_xlabel("lambda"); ax.set_ylabel("Spearman(A_hat, A_true)")
    ax.set_title(f"{domain} fusion Spearman by lambda")
    fig.tight_layout()
    fig.savefig(out_fig / f"spearman_by_lambda_{dom}.png", dpi=110)
    plt.close(fig)

    # decile gain per lambda
    fig, ax = plt.subplots(figsize=(6, 4))
    for lam, gl in g.groupby("lambda"):
        Ah = gl["A_hat"].to_numpy(); At = gl["A_true"].to_numpy()
        if len(Ah) < 10:
            continue
        qs = np.quantile(Ah, np.arange(0.1, 1.001, 0.1))
        means = []
        for i in range(10):
            lo = -np.inf if i == 0 else qs[i - 1]
            m = (Ah >= lo) & (Ah <= qs[i])
            means.append(At[m].mean() if m.any() else np.nan)
        ax.plot(np.arange(1, 11), means, "-o", ms=3,
                color=colors[LAMBDAS.index(lam)], label=f"F{LAMBDAS.index(lam)}")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("A_hat decile (1=lowest)"); ax.set_ylabel("mean A_true")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title(f"{domain} realized gain by predicted-value decile")
    fig.tight_layout(); fig.savefig(out_fig / f"decile_gain_{dom}.png", dpi=110)
    plt.close(fig)

    # direction mismatch by lambda
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lam, gl in g.groupby("lambda"):
        dm = gl["direction_mismatch"].dropna()
        ax.plot([float(lam)], [float(dm.mean()) if len(dm) else 0.0], "o", ms=6,
                color=colors[LAMBDAS.index(lam)])
    ax.set_xticks(LAMBDAS)
    ax.set_xlabel("lambda"); ax.set_ylabel("mean direction mismatch")
    ax.set_title(f"{domain} direction mismatch by lambda")
    fig.tight_layout()
    fig.savefig(out_fig / f"dirmismatch_lambda_{dom}.png", dpi=110)
    plt.close(fig)


def plot_ess_error(out_fig: Path, wdf: pd.DataFrame, domain: str):
    dom = domain.replace(":", "_")
    if wdf is None or not len(wdf):
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(wdf["ESS"], wdf["abs_err"], s=10, alpha=0.5)
    ax.set_xlabel("ESS_q = 1/sum(omega^2)"); ax.set_ylabel("|A_hat(F4) - A_true|")
    ax.set_title(f"{domain} local weight reliability (ESS vs error)")
    fig.tight_layout(); fig.savefig(out_fig / f"ess_error_{dom}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variant", type=str, default="learned_sig")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--boot-seed", type=int, default=0)
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
        results_dir / f"R1A7_FUSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.7] frozen artifacts: {artifact_dir}")
    print(f"[R1A.7] out: {out_dir}")
    print(f"[R1A.7] variant={args.variant} n_boot={args.n_boot} "
          f"block={args.block}")

    variant = args.variant
    all_rows, all_v0, all_metrics, all_block, all_topdec = [], [], [], [], []
    all_host, all_drift, all_calib, all_key, all_wrel, all_wdays = \
        [], [], [], [], [], []
    domain_info = {}
    for ds_key, bb in R.DOMAINS:
        print(f"[R1A.7] {ds_key}:{bb} ...", flush=True)
        res = analyze_domain(artifact_dir, ds_key, bb, variant)
        all_rows.extend(res["rows"])
        all_v0.extend(res["v0_rows"])
        all_metrics.extend(res["metrics"])
        all_block.extend(res["block_metrics"])
        all_topdec.extend(res["topdec"])
        all_host.extend(res["host_agg"].to_dict("records"))
        all_drift.extend(res["drift"].to_dict("records"))
        all_calib.extend(res["calib"])
        all_key.append(res["key_deg"])
        all_wrel.extend(res["weight_rel"])
        all_wdays.extend(res["weight_days"])
        domain_info[f"{ds_key}:{bb}"] = {"k": res["k"], "n_mem0": res["n_mem0"],
                                         "n_eval_days": res["n_eval_days"]}
        print(f"    k={res['k']} n_mem0={res['n_mem0']} "
              f"n_eval={res['n_eval_days']} probs={len(res['problems'])}")
        for pr in res["problems"]:
            print(f"      WARN {pr}")

    pdf = pd.DataFrame(all_rows)

    # ---- F0 == V1 cross-check against R1A.6 ----
    vcheck = None
    r6_dirs = sorted(results_dir.glob("R1A_VALUE_*"), key=lambda p: p.name)
    if r6_dirs:
        ref = pd.read_csv(r6_dirs[-1] / "value_by_day.csv")
        ref = ref[ref["estimator"] == "IAHNativeValue"].copy()
        mine = pdf[np.isclose(pdf["lambda"], 0.0)].copy()
        ref["date"] = pd.to_datetime(ref["date"])
        mine["date"] = pd.to_datetime(mine["date"])
        merged = ref.merge(mine, on=["domain", "block", "date"],
                           suffixes=("_r6", "_f7"))
        if len(merged):
            dA = float(np.max(np.abs(merged["A_hat_r6"] - merged["A_hat_f7"])))
            dT = float(np.max(np.abs(merged["A_true_r6"] - merged["A_true_f7"])))
            vcheck = {"n_matched": int(len(merged)),
                      "max_dA_hat": dA, "max_dA_true": dT,
                      "ok": bool(dA <= 1e-9 and dT <= 1e-9)}
    print(f"[R1A.7] F0 vs R1A.6 V1 cross-check: "
          f"{'PASS' if vcheck and vcheck['ok'] else 'N/A'} {vcheck}")

    # ---- S3M-val λ selection (§4) ----
    valpdf = pdf[pdf["block"] == "S3M-val"]
    val_metrics = []
    for (dom, lam), g in valpdf.groupby(["domain", "lambda"]):
        Ah = g["A_hat"].to_numpy(); At = g["A_true"].to_numpy()
        val_metrics.append({"domain": dom, "lambda": float(lam),
                            "spearman": _sp(Ah, At),
                            "hit_lift_top10": _hit_lift(Ah, At)})
    sel_lam, sel_table = lambda_select(val_metrics)
    lam_sel_df = pd.DataFrame([{"lambda": lam,
                                **{f"S3M-val_{k2}": v2 for k2, v2 in
                                   sel_table[float(lam)].items()},
                                "selected": bool(np.isclose(lam, sel_lam))}
                               for lam in LAMBDAS])
    print(f"[R1A.7] S3M-val λ selection: selected λ={sel_lam} "
          f"{sel_table[float(sel_lam)]}")

    # ---- full-eval macro + gate on the selected λ ----
    macro = macro_metrics(all_metrics)
    sel_m = macro[float(sel_lam)]
    gate = {
        "macro_spearman>=0.20": bool(sel_m["macro_spearman"] >= 0.20),
        "6/6>0 or worst>-0.05": bool(
            sel_m["n_pos_domains"] == 6 or sel_m["worst_domain_rho"] > -0.05),
        "excl_best>=0.15": bool(sel_m["macro_excl_best"] >= 0.15),
        "top10_hit_lift>=0.20": bool(sel_m["macro_hit_lift"] >= 0.20),
        "PJM:MLP not outlier": bool(
            sel_m["pjm_mlp_rho"] is not None and sel_m["pjm_mlp_rho"] >= -0.05),
    }
    green = all(gate.values())

    # ---- CSVs ----
    pdf.to_csv(out_dir / "fusion_by_day.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(out_dir / "fusion_metrics_by_domain.csv",
                                     index=False)
    pd.DataFrame(all_block).to_csv(out_dir / "fusion_metrics_by_block.csv",
                                   index=False)
    pd.DataFrame(all_topdec).to_csv(out_dir / "top_decile_enrichment.csv",
                                    index=False)
    lam_sel_df.to_csv(out_dir / "lambda_selection.csv", index=False)
    boot = bootstrap_all(all_rows, all_v0, n_boot=args.n_boot,
                         block=args.block, seed=args.boot_seed)
    boot.to_csv(out_dir / "bootstrap_intervals.csv", index=False)

    # ---- verdict + case diagnosis (§8, evidence-driven) ----
    # Priority: CASE_A (pure-IAH prior fails at λ=0, no local evidence)
    #         -> CASE_B (fusion significantly harmful AND key degeneracy)
    #         -> CASE_C (ESS-gated recovery: high-ESS tercile genuinely positive)
    #         -> MIXED.
    case = None
    cofactors = []
    if not green:
        f0_map = {m["domain"]: m["spearman"] for m in all_metrics
                  if np.isclose(m["lambda"], 0.0)}
        f0_worst_dom, f0_worst = min(f0_map.items(), key=lambda kv: kv[1])
        pjm = next((c for c in all_calib if c["domain"] == "LAGO_PJM:MLP"), None)
        pjm_w = next((w for w in all_wrel if w["domain"] == "LAGO_PJM:MLP"), None)
        pjm_k = next((k for k in all_key if k["domain"] == "LAGO_PJM:MLP"), None)
        b4 = boot[boot["lambda"] == 1.0]
        f4_lo = float(b4["macro_spearman_delta_vs_F0_ci_lo"].iloc[0]) \
            if len(b4) else float("nan")
        f4_hi = float(b4["macro_spearman_delta_vs_F0_ci_hi"].iloc[0]) \
            if len(b4) else float("nan")
        fuse_harmful = bool(len(b4)) and bool(f4_hi < 0.0)
        key_degen = bool(pjm_k and pjm_k["pairwise_cos_mean"] > 0.95)
        hi_ess = (pjm_w.get("rho_high_ESS") if pjm_w else None) or 0.0
        lo_ess = (pjm_w.get("rho_low_ESS") if pjm_w else None) or 0.0
        # 1) CASE_A: λ=0 (pure IAH prior, ZERO local evidence) already fails on
        #    the worst domain -> atom/action calibration separation
        if f0_worst < -0.05:
            case = "CASE_A"
        # 2) CASE_B: local evidence is actively harmful (bootstrap Δρ vs F0 at
        #    λ=1 excludes 0 negatively) AND the failed domain's keys collapse
        elif fuse_harmful and key_degen:
            case = "CASE_B"
        # 3) CASE_C: ESS-gated recovery — high-ESS tercile genuinely positive
        #    while low-ESS is not (reliability-adaptive λ could fix)
        elif hi_ess > 0.10 and hi_ess - lo_ess > 0.15:
            case = "CASE_C"
        else:
            case = "MIXED"
        if fuse_harmful and key_degen:
            cofactors.append("CASE_B")
        if pjm_w and hi_ess <= 0.0 and lo_ess <= 0.0:
            cofactors.append("CASE_C-excluded")
    pd.DataFrame(all_host).to_csv(out_dir / "audit_host_contrast.csv",
                                  index=False)
    pd.DataFrame(all_drift).to_csv(out_dir / "audit_residual_drift.csv",
                                   index=False)
    pd.DataFrame(all_calib).to_csv(out_dir / "audit_directional_calib.csv",
                                   index=False)
    pd.DataFrame(all_key).to_csv(out_dir / "audit_key_degeneracy.csv",
                                 index=False)
    pd.DataFrame(all_wrel).to_csv(out_dir / "audit_weight_reliability.csv",
                                  index=False)

    # ---- figures ----
    for dom in [f"{d}:{b}" for d, b in R.DOMAINS]:
        plot_figures(out_fig, pdf, dom)
    wdays_df = pd.DataFrame(all_wdays)
    for dom in [f"{d}:{b}" for d, b in R.DOMAINS]:
        plot_ess_error(out_fig, wdays_df[wdays_df["domain"] == dom], dom)

    # ---- code_commit + config ----
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{_git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"frozen artifacts: {artifact_dir.name}\n")
        f.write(f"variant: {variant}\n")
        f.write("R1A.7 F0-F4 only; NO R1B; no candidate retrain; "
                "no CRPS/DVG change; no new trainable network.\n")
        f.write("R1A source S4 = DEVELOPMENT DATA (plan §10).\n")
    with open(out_dir / "fusion_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "plan": "hch_v2_r1a7_prior_local_fusion_failure_audit_compute_plan_v0.1_2026-08-13.md",
            "frozen_artifacts": artifact_dir.name,
            "variant": variant,
            "lambdas": LAMBDAS,
            "lambda_selection": {"rule": "S3M-val only; worst -> macro -> top10",
                                 "selected": sel_lam,
                                 "table": sel_table},
            "fusion": "g^(λ)=(1-λ)g^I+λg^L -> double_event -> pi^(λ); "
                      "A^(λ)=(1-λ)A^I(pi)+λA^L(pi); same final action",
            "gI": "g^I_down=m-(2w--1), g^I_up=m+(2w+-1) (IAH closed form)",
            "gL": "weighted per-hour local directional gain, "
                  "omega_j=exp(-D/(tau+eps))/sum, tau=median(D_j)",
            "domain_k": {d: v["k"] for d, v in domain_info.items()},
            "domain_n_mem0": {d: v["n_mem0"] for d, v in domain_info.items()},
            "bootstrap": {"block": args.block, "n": args.n_boot,
                          "seed": args.boot_seed},
            "s4_status": "R1A source S4 = DEVELOPMENT DATA (plan §10); "
                         "not final confirmatory evidence",
        }, f, indent=2)

    # ---- FUSION_VERDICT ----
    lines = []
    lines.append("# R1A.7 FUSION VERDICT — Prior–Local shrinkage (F0–F4) + PJM:MLP audit")
    lines.append("")
    lines.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- code commit: `{_git_head()}`")
    lines.append(f"- frozen R1A artifacts: `{artifact_dir.name}` "
                 f"(variant `{variant}`, no retrain)")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    label = "PRIOR_LOCAL_FUSION" if green else "FUSION_UNRESOLVED"
    gate_status = "GREEN" if green else "YELLOW"
    lines.append(f"### **{label}** (gate status: {gate_status}"
                 + (f", {case}" if not green else "") + ")")
    lines.append("")
    lines.append("Decision evidence (§7 gate, evaluated on the §4-selected "
                 f"λ={sel_lam}):")
    for k, v in gate.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append(f"Selected λ (S3M-validation only, §4 order worst→macro→top10): "
                 f"**λ={sel_lam}**")
    lines.append("")
    lines.append("## §4 λ selection table (S3M-val only)")
    for _, r in lam_sel_df.iterrows():
        lines.append(f"- λ={r['lambda']}: worst={r['S3M-val_worst_domain_rho']:.3f} "
                     f"macro={r['S3M-val_macro_spearman']:.3f} "
                     f"hit-lift={r['S3M-val_macro_hit_lift']:.3f} "
                     f"{'<- SELECTED' if r['selected'] else ''}")
    lines.append("")

    lines.append("## Full-eval macro Spearman by λ (mean over 6 domains)")
    for lam in LAMBDAS:
        m = macro[float(lam)]
        lines.append(f"- **F{LAMBDAS.index(lam)}** (λ={lam}): "
                     f"macro rho=`{m['macro_spearman']:.3f}` "
                     f"({m['n_pos_domains']}/6 domains > 0, "
                     f"worst=`{m['worst_domain_rho']:.3f}`, "
                     f"excl-best=`{m['macro_excl_best']:.3f}`, "
                     f"hit-lift(top10)=`{m['macro_hit_lift']:.3f}`, "
                     f"dir-mismatch=`{m['macro_direction_mismatch']:.3f}`, "
                     f"PJM:MLP rho=`{m['pjm_mlp_rho']:.3f}`)")
    lines.append("")

    lines.append("## Bootstrap (moving-block, block=7d, "
                 f"{args.n_boot} samples, 95% CI)")
    for _, r in boot.iterrows():
        lines.append(f"- F{LAMBDAS.index(r['lambda'])} (λ={r['lambda']}): "
                     f"Δρ vs V0 CI[{r['macro_spearman_delta_vs_V0_ci_lo']:+.3f}, "
                     f"{r['macro_spearman_delta_vs_V0_ci_hi']:+.3f}] | "
                     f"Δρ vs F0 CI[{r['macro_spearman_delta_vs_F0_ci_lo']:+.3f}, "
                     f"{r['macro_spearman_delta_vs_F0_ci_hi']:+.3f}] | "
                     f"Δtop10-gain vs F0 "
                     f"CI[{r['top10_gain_delta_vs_F0_ci_lo']:+.4f}, "
                     f"{r['top10_gain_delta_vs_F0_ci_hi']:+.4f}]")
    lines.append("")

    lines.append("## Per-domain Spearman (all λ, all blocks pooled)")
    md = pd.DataFrame(all_metrics)
    piv = md.pivot_table(index="domain", columns="lambda", values="spearman",
                         aggfunc="first")
    for dom, row in piv.iterrows():
        parts = [f"λ={c:.2f}={v:.3f}" if pd.notna(v) else f"λ={c:.2f}=NaN"
                 for c, v in row.items()]
        lines.append("- " + dom + ": " + ", ".join(parts))
    lines.append("")

    lines.append("## F0 == R1A.6 V1 cross-check")
    if vcheck:
        lines.append(f"- matched {vcheck['n_matched']} days; "
                     f"max|dA_hat|={vcheck['max_dA_hat']:.3e} "
                     f"max|dA_true|={vcheck['max_dA_true']:.3e} -> "
                     f"{'PASS (F0 == V1 exactly)' if vcheck['ok'] else 'FAIL'}")
    else:
        lines.append("- R1A_VALUE value_by_day.csv not found; skipped")
    lines.append("")

    lines.append("## §5 PJM:MLP Failure Audit summary")
    lines.append("")
    lines.append("### 5.1 Same-market host contrast (PJM:Linear vs PJM:MLP)")
    ha = pd.DataFrame(all_host)
    for dom, g in ha.groupby("domain"):
        r = g.iloc[0]
        lines.append(f"- {dom}: host MAE={r['host_mae']:.3f} "
                     f"rMAE={r['host_rmae']:.3f} "
                     f"hyp-err={r['hyp_host_err']:.4f} "
                     f"IAH-CRPS={r['iah_crps']:.4f} "
                     f"CRPSΔ={r['crps_delta']:+.4f} "
                     f"resid_μ={r['resid_mean']:.4f} σ={r['resid_std']:.4f} "
                     f"IQR={r['resid_iqr']:.4f} signB={r['sign_balance']:.3f} "
                     f"lag1={r['lag1_autocorr']:.3f} "
                     f"daily={r['daily_autocorr']:.3f}")
    lines.append("")

    lines.append("### 5.2 IAH directional calibration")
    for c in all_calib:
        def _f(x):
            return f"{x:.3f}" if x is not None else "NaN"
        lines.append(f"- {c['domain']}: AUC(w-,B-)={_f(c['AUC(w-,B-)'])} "
                     f"AUC(w+,B+)={_f(c['AUC(w+,B+)'])} "
                     f"ρ(gI↓,gT↓)={_f(c['Spearman(gI_down,g_true_down)'])} "
                     f"ρ(gI↑,gT↑)={_f(c['Spearman(gI_up,g_true_up)'])} "
                     f"rel10↓={_f(c['rel10_spearman_down'])} "
                     f"rel10↑={_f(c['rel10_spearman_up'])} "
                     f"P(B->0)={c['P(B->0)']:.3f} P(B+>0)={c['P(B+>0)']:.3f} "
                     f"med|m⁻|={_f(c.get('median_abs_mminus'))} "
                     f"fire@λ=0={_f(c.get('fire_rate_lambda0'))}")
    lines.append("")

    lines.append("### 5.3 Learned-key degeneracy")
    for kd in all_key:
        lines.append(f"- {kd['domain']}: ||k||={kd['key_norm_mean']:.3f}±"
                     f"{kd['key_norm_std']:.3f} cos_pair_mean="
                     f"{kd['pairwise_cos_mean']:.3f} "
                     f"dist_med={kd['pairwise_dist_median']:.4f} "
                     f"conc={kd['dist_concentration']:.3f} "
                     f"NN={kd['nn_dist_mean']:.4f} "
                     f"uniq_nbr={kd['unique_neighbors']}/{kd['memory_days']} "
                     f"cov={kd['neighbor_coverage']:.3f} "
                     f"stickiness={kd['stickiness_jaccard']:.3f}")
    lines.append("")

    lines.append("### 5.4 Local weight reliability (F4 branch)")
    for w in all_wrel:
        lines.append(f"- {w['domain']}: ESS_med={w['ESS_median']:.2f} "
                     f"[{w['ESS_p10']:.2f},{w['ESS_p90']:.2f}] "
                     f"top_w={w['top_weight_median']:.3f} "
                     f"medD={w['median_D_median']:.4f} "
                     f"ρ(|err|,ESS)={_f(w['Spearman(abs_err,ESS)'])} "
                     f"ρ(Ah,At)={_f(w['Spearman(Ahat,Atrue)'])} "
                     f"tercile-low={_f(w.get('rho_low_ESS'))} "
                     f"mid={_f(w.get('rho_mid_ESS'))} high={_f(w.get('rho_high_ESS'))}")
    lines.append("")

    lines.append("## Case diagnosis (§8)")
    if green:
        lines.append("- GREEN: prior–local fusion restored action-value ranking "
                     "within the §7 gate; R1B may resume per §9.")
    else:
        lines.append(f"- YELLOW, case = **{case}**"
                     + (f", co-factors: {', '.join(cofactors)}"
                        if cofactors else "") + ".")
        lines.append(f"  - Evidence: worst-domain ρ at λ=0 (pure IAH prior, ZERO "
                     f"local evidence) = **{f0_worst_dom} → {f0_worst:.3f}**; "
                     f"bootstrap Δρ(λ=1 vs F0) CI=[{f4_lo:+.3f},{f4_hi:+.3f}]"
                     f"{' (harmful)' if fuse_harmful else ''}; "
                     f"PJM:MLP key cos_pair_mean="
                     f"{pjm_k['pairwise_cos_mean']:.3f}"
                     f"{' (degenerate)' if key_degen else ''}; "
                     f"PJM:MLP ESS terciles low={lo_ess:+.3f} "
                     f"high={hi_ess:+.3f}.")
        lines.append("  - CASE_A: atom/action calibration failure → next question "
                     "is IAH probability mass → action-value calibration.")
        lines.append("  - CASE_B: representation/key degeneracy → richer "
                     "representation / foundation rep / MOMENT U0.")
        lines.append("  - CASE_C: local evidence high variance → "
                     "reliability-adaptive shrinkage / ESS-based λ / robust "
                     "local estimator.")
        lines.append("- R1B stays paused (§9) until GREEN or a single quantified "
                     "failure mechanism is located.")
    lines.append("")

    lines.append("## Notes")
    lines.append("- R1A source S4 = DEVELOPMENT DATA (plan §10): selected λ is "
                 "NOT final confirmatory evidence; confirm on R1B unseen hosts / "
                 "NORD_DK1 / Shandong holdout.")
    lines.append("- F4 (λ=1) = V4 with the §3-proper weighted PROPOSAL (R1A.6's "
                 "V4 kept the uniform proposal; plan §2.2 caveat). So F4 generalizes "
                 "V4, not a byte-identical rerun.")
    lines.append("- λ selection used S3M-validation ONLY (§4); full-eval numbers "
                 "are reported for every λ but the gate applies to the selected λ.")
    lines.append("- No candidate retrain; no CRPS/DVG/CAGM math change; no new "
                 "trainable network (plan §6).")
    lines.append("")

    verdict_text = "\n".join(lines)
    with open(out_dir / "FUSION_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.7] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
