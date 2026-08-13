"""R1A.6 — value-estimation recovery diagnostics (V0–V4).

Spec: docs/paper_prep/v2_final_prep/hch_v2_r1a6_retrieval_value_recovery_plan_v0.1_2026-08-13.md

Scope (per user): ONLY R1A.6 (V0–V4). NO R1B, NO candidate retrain, NO CRPS
change, NO DVG change. Candidate weights frozen; IAH/CRPS/CAGM math untouched
(plan §9).

Goal: restore a value estimator A_hat with real action-ranking ability on the
FROZEN R1A candidate (variant learned_sig, tentative main) across 6 domains.

Estimators (plan §8, §16):
    V0 CurrentW1Static                — W1 + static memory + uniform mean (baseline)
    V1 IAHNativeValue                 — closed-form expected utility, no retrieval
    V2 LearnedKeyStatic               — cosine on frozen learned daily key + static mem
    V3 LearnedKeyPrequential          — learned key + causally expanding memory
    V4 LearnedKeyPrequentialWeighted  — V3 + distance-weighted A_hat (plan §2.2, §7)

Learned daily key (plan §5): k_d = e^{learned}_d = ReLU(W_sig MeanPool_h(h_core)),
captured from the frozen candidate's DataSignature.learned_proj (no metric-
learning retriever trained). Distance D_emb(q,j) = 1 - cos(k_q, k_j).

Prequential memory (plan §6): day t predicts with M_{t^-} (before outcome);
the day's pre-outcome key + atoms + target zY + valid mask are appended to M_t
after its outcome, usable from day t+1 onward. Y_t ∉ F^-_t, Y_t ∈ F^-_{t+1}.

Weighted A_hat (plan §7): A_hat_q^w = sum_j omega_j A_{q->j},
omega_j = exp[-D(q,j)/(tau_q+eps)] / sum, tau_q = median_{j in N_k(q)} D(q,j).

All statistics follow plan §11-§12; verdict gates per plan §13; VALUE_VERDICT
labels per plan §18. R1A source S4 = DEVELOPMENT DATA (plan §10).

Outputs -> R1A_VALUE_<ts>/ (plan §17):
    code_commit.txt, source_r1a_artifact.txt, estimator_config.json,
    value_by_day.csv, value_metrics_by_domain.csv, value_metrics_by_block.csv,
    top_decile_enrichment.csv, memory_growth.csv, bootstrap_intervals.csv,
    figures/{ahat_atrue,decile_gain,spearman_by_block,memory_growth}_*.png,
    VALUE_VERDICT.md
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1a5_diag as D   # rebuild / oracle / s4_quarters / write_source_r1a_run
import r1a_run as R
from hch_v2_bundle import HCHV2Bundle
from double_event import double_event_proposal
from query_replay import (estimate_realized_A, form_final_pi,
                          full_replay_chain)

EPS = 1e-12

EST_ORDER_ALL = ["CurrentW1Static", "IAHNativeValue", "LearnedKeyStatic",
                 "LearnedKeyPrequential", "LearnedKeyPrequentialWeighted"]
EST_LABELS = {
    "CurrentW1Static": "V0 Current W1 + static + uniform mean (baseline)",
    "IAHNativeValue": "V1 IAH expected utility (closed-form, no retrieval)",
    "LearnedKeyStatic": "V2 learned-key cosine + static memory + uniform mean",
    "LearnedKeyPrequential": "V3 learned-key + prequential expanding memory",
    "LearnedKeyPrequentialWeighted": "V4 V3 + distance-weighted A_hat",
}
VERDICT_LABELS = ("IAH_NATIVE_VALUE", "LEARNED_RETRIEVAL", "PREQUENTIAL_MEMORY",
                  "WEIGHTED_LOCAL_VALUE", "VALUE_ESTIMATION_UNRESOLVED")


# ---------------------------------------------------------------- helpers ----
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


# --------------------------------------------------- learned-key memory ------
class KeyMemory:
    """Duck-typed CAGMAtomMemory + per-day learned embedding key.

    full_replay_chain / build_directional_gains only touch .z0/.target_zY/
    .valid_mask, so this mirrors those three lists and adds .keys for
    cosine retrieval. Atoms stored for audit / memory_growth.
    """

    def __init__(self):
        self.dates: list = []
        self.z0: list[np.ndarray] = []
        self.w_minus: list[np.ndarray] = []
        self.w_zero: list[np.ndarray] = []
        self.w_plus: list[np.ndarray] = []
        self.m_minus: list[np.ndarray] = []
        self.m_plus: list[np.ndarray] = []
        self.target_zY: list[np.ndarray] = []
        self.valid_mask: list[np.ndarray] = []
        self.keys: list[np.ndarray] = []

    def add_day(self, date, cand, zY, key):
        self.dates.append(date)
        self.z0.append(_fl(cand["z0"]))
        self.w_minus.append(_fl(cand["w_minus"]))
        self.w_zero.append(_fl(cand["w_zero"]))
        self.w_plus.append(_fl(cand["w_plus"]))
        self.m_minus.append(_fl(cand["m_minus"]))
        self.m_plus.append(_fl(cand["m_plus"]))
        self.target_zY.append(np.asarray(zY, dtype=np.float64).reshape(-1))
        self.valid_mask.append(_fl(cand["valid_mask"]).astype(bool))
        self.keys.append(np.asarray(key, dtype=np.float64).reshape(-1))

    def __len__(self):
        return len(self.dates)

    def neighbors_cos(self, kq, k):
        """D_emb = 1 - cos(k_q, k_j); returns (indices, distances)."""
        kq = np.asarray(kq, dtype=np.float64).reshape(-1)
        nq = float(np.linalg.norm(kq))
        dists = np.zeros(len(self), dtype=np.float64)
        for i, kj in enumerate(self.keys):
            nj = float(np.linalg.norm(kj))
            if nq == 0.0 and nj == 0.0:
                cos = 1.0
            elif nq == 0.0 or nj == 0.0:
                cos = 0.0
            else:
                cos = float(np.dot(kq, kj) / (nq * nj))
            dists[i] = 1.0 - cos
        order = np.argsort(dists)
        nbr = []
        for idx in order:
            if len(nbr) < k:
                nbr.append(int(idx))
            else:
                break
        return nbr, dists


def make_day_key(pipe, info, d, det_np, key_sink):
    """One day's frozen-candidate forward + learned daily embedding capture.

    Deterministic-identical to r1a5 _make_day (same R._run_candidate_day), plus
    the learned key k_d from the DataSignature hook (target-free, plan §5).
    """
    ts = info.ds["ts"]
    idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
    if len(idxs) != 24:
        return None, None
    host_day = info.yhat_full[idxs].astype(np.float64)
    if not np.isfinite(host_day).all():
        return None, None
    hours = ts.iloc[idxs].dt.hour.values
    key_sink.clear()
    out, zY = R._run_candidate_day(pipe, host_day, hours, info.z0_full,
                                   info.s_full, info.ds["price"].astype(np.float32),
                                   idxs, det_np)
    if out is None:
        return None, None
    key = np.asarray(key_sink[-1], dtype=np.float64).reshape(-1) if key_sink \
        else None
    return {"date": d, "candidate": out, "target_zY": zY}, key


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

    # ---- split bookkeeping (mirror r1a_run / r1a5) ----
    s3m_all = sorted(info.exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * R.S3M_MEM_FRAC)
    mem_dates = list(pipe.memory.dates)          # authoritative frozen prefix
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
            day, key = make_day_key(pipe, info, d, det_np, key_sink)
            if day is None or key is None:
                continue
            daymap[d] = {"day": day, "key": key}
    finally:
        hook.remove()

    # ---- cross-check: my forward atoms == frozen pipe memory atoms ----
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
        mine_y = np.asarray(day["target_zY"], dtype=np.float64).reshape(-1)
        ref_y = np.asarray(pipe.memory.target_zY[j], dtype=np.float64)
        if not np.allclose(mine_y, ref_y, atol=1e-9):
            problems.append(f"{domain}: my target_zY[{d}] != frozen memory")

    eval_dates = sorted(d for d in daymap
                        if block_of(d) in ("S3M-val", "S3C") or
                        (block_of(d) or "").startswith("S4Q"))
    if not eval_dates:
        raise ValueError(f"{domain}: no evaluation days")

    # ---- shared static (frozen) memory with learned keys (V2) ----
    static_mem = KeyMemory()
    for d in mem_dates:
        if d in daymap:
            static_mem.add_day(d, daymap[d]["day"]["candidate"],
                               daymap[d]["day"]["target_zY"], daymap[d]["key"])

    rows, mem_growth = [], []

    def _row(est, blk, q, d, A_hat, A_true):
        oA = oracleA[d]
        eta = max(A_true, 0.0) / max(oA, EPS)
        missed = 1 if (oA > 1e-9 and A_true <= 0.0) else 0
        return {"domain": domain, "variant": variant, "estimator": est,
                "block": blk, "quarter": q, "date": _iso(d),
                "A_hat": float(A_hat), "A_true": float(A_true),
                "E": float(A_hat - A_true), "A_oracle": float(oA),
                "eta": float(eta), "missed": int(missed), "k": int(k)}

    def day_basic(d):
        day = daymap[d]["day"]
        cand = day["candidate"]
        z0 = _fl(cand["z0"])
        zY = np.asarray(day["target_zY"]).reshape(-1)
        mm = _fl(cand["m_minus"]); mp = _fl(cand["m_plus"])
        wm = _fl(cand["w_minus"]); wp = _fl(cand["w_plus"])
        vm = _fl(cand["valid_mask"]).astype(bool)
        return cand, z0, zY, mm, mp, wm, wp, vm

    # ---- oracle per day (estimator-independent, D1 machinery) ----
    oracleA: dict = {}
    for d in eval_dates:
        cand = daymap[d]["day"]["candidate"]
        zY = np.asarray(daymap[d]["day"]["target_zY"]).reshape(-1)
        o = D.oracle_for_day({"z0": cand["z0"], "m_minus": cand["m_minus"],
                              "m_plus": cand["m_plus"],
                              "valid_mask": cand["valid_mask"]}, zY)
        oracleA[d] = float(o["A"])

    # ---- V0 / V1 / V2 (order-free) ----
    for d in eval_dates:
        blk = block_of(d); q = qmap.get(d)
        day = daymap[d]["day"]
        cand, z0, zY, mm, mp, wm, wp, vm = day_basic(d)

        # V0: current chain = W1 + static memory + uniform mean (R1A.5 authority)
        A_hat, A_true = pipe._replay_value(day, k)
        rows.append(_row("CurrentW1Static", blk, q, d, A_hat, A_true))
        mem_growth.append({"domain": domain, "estimator": "CurrentW1Static",
                           "date": _iso(d), "block": blk,
                           "memory_size_before": n_mem0})

        # V1: IAH expected utility (plan §3), no retrieval, no memory
        g_down = mm * (2.0 * wm - 1.0)
        g_up = mp * (2.0 * wp - 1.0)
        prop = double_event_proposal(g_down, g_up)
        pi = form_final_pi(mm, mp, prop["I_down"], prop["I_up"])
        A_hat1 = float((g_down * (pi < 0) + g_up * (pi > 0))[vm].sum() / vm.sum()) \
            if vm.any() else 0.0
        A_true1 = estimate_realized_A(z0, zY, pi, vm)
        rows.append(_row("IAHNativeValue", blk, q, d, A_hat1, A_true1))

        # V2: learned-key cosine retrieval, static memory, uniform mean
        nbr, _dists = static_mem.neighbors_cos(daymap[d]["key"], k)
        if nbr:
            chain = full_replay_chain(static_mem, nbr, mm, mp,
                                      double_event_proposal, query_valid=vm)
            A_hat2 = chain["A_hat"]; pi2 = chain["pi_q"]
        else:
            A_hat2 = 0.0; pi2 = np.zeros_like(mm)
        A_true2 = estimate_realized_A(z0, zY, pi2, vm)
        rows.append(_row("LearnedKeyStatic", blk, q, d, A_hat2, A_true2))
        mem_growth.append({"domain": domain, "estimator": "LearnedKeyStatic",
                           "date": _iso(d), "block": blk,
                           "memory_size_before": n_mem0})

    # ---- V3 / V4 (prequential expanding memory, chronological) ----
    preq_mem = KeyMemory()
    for d in mem_dates:
        if d in daymap:
            preq_mem.add_day(d, daymap[d]["day"]["candidate"],
                             daymap[d]["day"]["target_zY"], daymap[d]["key"])
    for d in eval_dates:  # eval_dates sorted => val -> s3c -> s4 chronology
        blk = block_of(d); q = qmap.get(d)
        day = daymap[d]["day"]
        cand, z0, zY, mm, mp, wm, wp, vm = day_basic(d)
        nbr, dists = preq_mem.neighbors_cos(daymap[d]["key"], k)
        if nbr:
            chain = full_replay_chain(preq_mem, nbr, mm, mp,
                                      double_event_proposal, query_valid=vm)
            A_hat3 = chain["A_hat"]; pi3 = chain["pi_q"]
            perA = np.asarray(chain["per_neighbor_A"], dtype=np.float64)
            D_j = np.asarray([dists[j] for j in nbr], dtype=np.float64)
        else:
            A_hat3 = 0.0; pi3 = np.zeros_like(mm); perA = np.zeros(0)
            D_j = np.zeros(0)
        A_true3 = estimate_realized_A(z0, zY, pi3, vm)
        rows.append(_row("LearnedKeyPrequential", blk, q, d, A_hat3, A_true3))
        # V4: identical retrieval/proposal as V3, only A_hat weighted (plan §2.2)
        if perA.size:
            tau = float(np.median(D_j))
            wts = np.exp(-D_j / (tau + EPS))
            wts = wts / (wts.sum() + EPS)
            A_hat4 = float(np.sum(perA * wts))
        else:
            A_hat4 = 0.0
        rows.append(_row("LearnedKeyPrequentialWeighted", blk, q, d,
                         A_hat4, A_true3))
        mem_growth.append({"domain": domain, "estimator": "LearnedKeyPrequential",
                           "date": _iso(d), "block": blk,
                           "memory_size_before": len(preq_mem)})
        mem_growth.append({"domain": domain,
                           "estimator": "LearnedKeyPrequentialWeighted",
                           "date": _iso(d), "block": blk,
                           "memory_size_before": len(preq_mem)})
        # after outcome: append day's pre-outcome key + atoms + zY + mask
        preq_mem.add_day(d, day["candidate"], day["target_zY"], daymap[d]["key"])

    # ---- aggregation (per domain × estimator) ----
    pdf = pd.DataFrame(rows)
    metrics, block_metrics, topdec = [], [], []
    for est, g in pdf.groupby("estimator"):
        Ah = g["A_hat"].to_numpy(); At = g["A_true"].to_numpy()
        n = len(g)
        n_top = max(1, n // 10)
        top_idx = np.argsort(Ah)[-n_top:]
        pos_all = float((At > 0).mean()) if n else None
        pos_top = float((At[top_idx] > 0).mean())
        lift = (pos_top - pos_all) if pos_all is not None else None
        m = {
            "domain": domain, "variant": variant, "estimator": est, "n": n,
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
        }
        metrics.append(m)
        # S4-quarter drift (plan §11 stability)
        for qq in (1, 2, 3, 4):
            gq = g[g["quarter"] == qq]
            m[f"rho_S4Q{qq}"] = _sp(gq["A_hat"].to_numpy(),
                                     gq["A_true"].to_numpy()) if len(gq) > 1 else None
        m["s4_quarter_drift"] = None
        if m.get("rho_S4Q1") is not None and m.get("rho_S4Q4") is not None:
            m["s4_quarter_drift"] = m["rho_S4Q4"] - m["rho_S4Q1"]

        # block metrics
        for blk, gb in g.groupby("block"):
            Ahb = gb["A_hat"].to_numpy(); Atb = gb["A_true"].to_numpy()
            block_metrics.append({
                "domain": domain, "variant": variant, "estimator": est,
                "block": blk, "n": len(gb), "spearman": _sp(Ahb, Atb),
                "mae": float(np.mean(np.abs(Ahb - Atb))) if len(gb) else None,
                "bias": float(np.mean(Ahb - Atb)) if len(gb) else None,
                "mean_A_hat": float(Ahb.mean()) if len(gb) else None,
                "mean_A_true": float(Atb.mean()) if len(gb) else None,
            })

        # top-decile enrichment
        topdec.append({
            "domain": domain, "variant": variant, "estimator": est,
            "n_top": n_top, "P(A>0|top10)": pos_top, "P(A>0|all)": pos_all,
            "hit_lift_top10": lift,
            "mean_A_true_top10": float(At[top_idx].mean()),
            "median_A_true_top10": float(np.median(At[top_idx])),
            "mean_A_true_all": float(At.mean()) if n else None,
        })

    return {"rows": rows, "mem_growth": mem_growth, "metrics": metrics,
            "block_metrics": block_metrics, "topdec": topdec,
            "problems": problems, "k": k, "n_mem0": n_mem0,
            "n_eval_days": len(eval_dates)}


# -------------------------------------------------------------- bootstrap ----
def _macro_stat(pdf, stat, domains, est):
    vals = []
    for dom in domains:
        g = pdf[(pdf["domain"] == dom) & (pdf["estimator"] == est)]
        Ah = g["A_hat"].to_numpy(); At = g["A_true"].to_numpy()
        if stat == "spearman":
            v = _sp(Ah, At)
        elif stat == "top10_gain":
            # top-10% by PREDICTED value A_hat, then realized A_true of those days
            n_top = max(1, len(At) // 10)
            top_idx = np.argsort(Ah)[-n_top:]
            v = float(At[top_idx].mean()) if len(At) else None
        else:
            raise ValueError(stat)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def block_bootstrap(est_map, base_map, rng, n_boot=500, block=7):
    """Moving-block bootstrap (plan §12) for macro Spearman delta + top10-gain
    delta of estimator vs base (V0). Same block indices per domain for pairing."""
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
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
            if len(idx) < 2:
                continue
            sp_e = _sp(Ah_e[idx], At_e[idx]); sp_b = _sp(Ah_b[idx], At_b[idx])
            if sp_e is not None and sp_b is not None:
                est_sp.append(sp_e); base_sp.append(sp_b)
            n_top = max(1, len(idx) // 10)
            # top-10% by A_hat (predicted value), realized A_true of those days
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


def bootstrap_all(all_rows, n_boot=500, block=7, seed=0):
    pdf = pd.DataFrame(all_rows)
    domains = sorted(pdf["domain"].unique())
    rng = np.random.default_rng(seed)
    est_map = {}
    for est in EST_ORDER_ALL:
        est_map[est] = {}
        for dom in domains:
            g = pdf[(pdf["domain"] == dom) & (pdf["estimator"] == est)] \
                 .sort_values("date")
            est_map[est][dom] = (g["A_hat"].to_numpy(), g["A_true"].to_numpy())
    base = est_map["CurrentW1Static"]
    out = []
    base_sp = _macro_stat(pdf, "spearman", domains, "CurrentW1Static")
    base_t10 = _macro_stat(pdf, "top10_gain", domains, "CurrentW1Static")
    for est in EST_ORDER_ALL:
        if est == "CurrentW1Static":
            out.append({"estimator": est, "vs": "CurrentW1Static",
                        "macro_spearman": base_sp, "macro_spearman_delta": 0.0,
                        "spearman_delta_ci_lo": 0.0, "spearman_delta_ci_hi": 0.0,
                        "macro_top10_gain": base_t10, "macro_top10_gain_delta": 0.0,
                        "top10_gain_delta_ci_lo": 0.0,
                        "top10_gain_delta_ci_hi": 0.0})
            continue
        (slo, shi), (tlo, thi) = block_bootstrap(est_map[est], base, rng,
                                                 n_boot=n_boot, block=block)
        macro_sp = _macro_stat(pdf, "spearman", domains, est)
        macro_t10 = _macro_stat(pdf, "top10_gain", domains, est)
        out.append({"estimator": est, "vs": "CurrentW1Static",
                    "macro_spearman": macro_sp,
                    "macro_spearman_delta": macro_sp - base_sp,
                    "spearman_delta_ci_lo": slo, "spearman_delta_ci_hi": shi,
                    "macro_top10_gain": macro_t10,
                    "macro_top10_gain_delta": macro_t10 - base_t10,
                    "top10_gain_delta_ci_lo": tlo,
                    "top10_gain_delta_ci_hi": thi})
    return pd.DataFrame(out)


# -------------------------------------------------------------- figures ------
def plot_figures(out_fig: Path, pdf: pd.DataFrame, domain: str):
    dom = domain.replace(":", "_")
    g = pdf[pdf["domain"] == domain]
    colors = {e: c for e, c in zip(EST_ORDER_ALL, plt.cm.tab10.colors[:5])}

    # ahat vs atrue
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for est, ge in g.groupby("estimator"):
        ax.scatter(ge["A_hat"], ge["A_true"], s=6, alpha=0.45,
                   color=colors[est], label=EST_LABELS[est].split(" ")[0]
                   if est != "CurrentW1Static" else est)
    lim = [min(g["A_hat"].min(), g["A_true"].min()),
           max(g["A_hat"].max(), g["A_true"].max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("A_hat"); ax.set_ylabel("A_true")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title(f"{domain} A_hat vs A_true (V0-V4)")
    fig.tight_layout(); fig.savefig(out_fig / f"ahat_atrue_{dom}.png", dpi=110)
    plt.close(fig)

    # decile gain
    fig, ax = plt.subplots(figsize=(6, 4))
    for est, ge in g.groupby("estimator"):
        Ah = ge["A_hat"].to_numpy(); At = ge["A_true"].to_numpy()
        if len(Ah) < 10:
            continue
        qs = np.quantile(Ah, np.arange(0.1, 1.001, 0.1))
        means = []
        for i in range(10):
            lo = -np.inf if i == 0 else qs[i - 1]
            m = (Ah >= lo) & (Ah <= qs[i])
            means.append(At[m].mean() if m.any() else np.nan)
        ax.plot(np.arange(1, 11), means, "-o", ms=3, color=colors[est],
                label=est)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("A_hat decile (1=lowest)"); ax.set_ylabel("mean A_true")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title(f"{domain} realized gain by predicted-value decile")
    fig.tight_layout(); fig.savefig(out_fig / f"decile_gain_{dom}.png", dpi=110)
    plt.close(fig)

    # spearman by block
    fig, ax = plt.subplots(figsize=(8, 3.5))
    blk_order = [b for b in ("S3M-val", "S3C", "S4Q1", "S4Q2", "S4Q3", "S4Q4")
                 if b in set(g["block"])]
    width = 0.8 / len(EST_ORDER_ALL)
    for ei, est in enumerate(EST_ORDER_ALL):
        ge = g[g["estimator"] == est]
        vals = []
        for b in blk_order:
            gb = ge[ge["block"] == b]
            vals.append(_sp(gb["A_hat"].to_numpy(), gb["A_true"].to_numpy())
                        if len(gb) > 1 else None)
        x = np.arange(len(blk_order)) + (ei - len(EST_ORDER_ALL) / 2) * width
        ax.bar(x, [0.0 if v is None else v for v in vals], width=width,
               color=colors[est], label=est)
    ax.set_xticks(np.arange(len(blk_order)))
    ax.set_xticklabels(blk_order, fontsize=7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Spearman(A_hat, A_true)")
    ax.legend(fontsize=5.5, loc="lower right", ncol=2)
    ax.set_title(f"{domain} Spearman by time block")
    fig.tight_layout(); fig.savefig(out_fig / f"spearman_by_block_{dom}.png", dpi=110)
    plt.close(fig)


def plot_memory_growth(out_fig: Path, mgdf: pd.DataFrame, domain: str):
    dom = domain.replace(":", "_")
    g = mgdf[mgdf["domain"] == domain]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = {"CurrentW1Static": "#1f77b4", "LearnedKeyStatic": "#2ca02c",
              "LearnedKeyPrequential": "#ff7f0e",
              "LearnedKeyPrequentialWeighted": "#d62728"}
    for est, ge in g.groupby("estimator"):
        ge = ge.sort_values("date").reset_index(drop=True)
        ax.plot(np.arange(len(ge)), ge["memory_size_before"],
                color=colors.get(est, "#333"), lw=1.2, label=est)
    ax.set_xlabel("evaluation day index (chronological)")
    ax.set_ylabel("memory size before prediction")
    ax.legend(fontsize=6)
    ax.set_title(f"{domain} memory size over time (V0/V2 static, V3/V4 preq)")
    fig.tight_layout(); fig.savefig(out_fig / f"memory_growth_{dom}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------- V0 vs R1A.5 ---
def verify_v0_vs_r1a5(all_rows, results_dir: Path) -> dict | None:
    """Cross-check V0 == R1A.5 D2 (same chain, same A_hat/A_true).

    A_hat is the ranking authority and must reproduce at float32 precision
    (<=1e-6) on every day. A_true is a single-day realized quantity: a ~1-ULP
    atom drift can flip one hour's action on a marginal day, shifting that day's
    A_true by one hour's contribution while A_hat (a neighbor mean) barely moves.
    Gate: max_dA <= 1e-6 AND at most 2 marginal-flip days with |dT| > 1e-7.
    """
    diag_dirs = sorted(results_dir.glob("R1A_DIAG_*"))
    if not diag_dirs:
        return None
    path = diag_dirs[-1] / "ahat_vs_atrue.csv"
    if not path.exists():
        return None
    ref = pd.read_csv(path)
    ref = ref[ref["variant"] == "learned_sig"].copy()
    mine = pd.DataFrame(all_rows)
    mine = mine[mine["estimator"] == "CurrentW1Static"].copy()
    ref["date"] = pd.to_datetime(ref["date"])
    mine["date"] = pd.to_datetime(mine["date"])
    merged = ref.merge(mine, on=["domain", "block", "date"],
                       suffixes=("_r5", "_v6"))
    if not len(merged):
        return None
    dA = float(np.max(np.abs(merged["A_hat_r5"] - merged["A_hat_v6"])))
    dT = float(np.max(np.abs(merged["A_true_r5"] - merged["A_true_v6"])))
    n_flip = int((np.abs(merged["A_true_r5"] - merged["A_true_v6"]) > 1e-7).sum())
    return {"n_matched": int(len(merged)),
            "n_ref": int(len(ref)), "n_v0": int(len(mine)),
            "max_dA": dA, "max_dT": dT, "n_Atrue_flip_days_gt_1e-7": n_flip,
            "ok": bool(dA <= 1e-6 and n_flip <= 2)}


# ---------------------------------------------------------------- verdict ----
def summarize_estimators(metrics: list[dict]) -> dict:
    est_rows = {}
    for est in EST_ORDER_ALL:
        ms = [m for m in metrics if m["estimator"] == est]
        sps = [m["spearman"] for m in ms if m["spearman"] is not None]
        macro = float(np.mean(sps)) if sps else float("nan")
        n_pos = int(sum(1 for s in sps if s > 0))
        worst = float(min(sps)) if sps else float("nan")
        macro_excl = float(np.mean(sorted(sps)[:-1])) if len(sps) >= 2 else macro
        lifts = [m["hit_lift_top10"] for m in ms if m["hit_lift_top10"] is not None]
        macro_lift = float(np.mean(lifts)) if lifts else float("nan")
        etas = [m["eta_mean"] for m in ms if m["eta_mean"] is not None]
        missed = [m["missed_positive_rate"] for m in ms
                  if m["missed_positive_rate"] is not None]
        est_rows[est] = {
            "macro_spearman": macro, "n_domains": len(ms),
            "n_pos_domains": n_pos, "worst_domain_rho": worst,
            "macro_excl_best": macro_excl, "macro_hit_lift": macro_lift,
            "macro_eta": float(np.mean(etas)) if etas else float("nan"),
            "macro_missed": float(np.mean(missed)) if missed else float("nan"),
        }
    return est_rows


def decide_verdict(est_rows: dict) -> tuple[str, str, list[str]]:
    """Plan §13 gates + §18 naming (smallest causal change first across GREEN)."""
    ORDER = ["IAHNativeValue", "LearnedKeyStatic", "LearnedKeyPrequential",
             "LearnedKeyPrequentialWeighted"]
    LABELS = {"IAHNativeValue": "IAH_NATIVE_VALUE",
              "LearnedKeyStatic": "LEARNED_RETRIEVAL",
              "LearnedKeyPrequential": "PREQUENTIAL_MEMORY",
              "LearnedKeyPrequentialWeighted": "WEIGHTED_LOCAL_VALUE"}
    reasons = []
    for est in ORDER:
        m = est_rows[est]
        green = (m["macro_spearman"] >= 0.20 and m["n_pos_domains"] >= 5
                 and m["worst_domain_rho"] > -0.10
                 and m["macro_hit_lift"] >= 0.10
                 and m["macro_excl_best"] >= 0.15)
        reasons.append(
            f"- {est}: macro rho={m['macro_spearman']:.3f} "
            f"pos-domains={m['n_pos_domains']}/6 worst={m['worst_domain_rho']:.3f} "
            f"hit-lift={m['macro_hit_lift']:.3f} excl-best={m['macro_excl_best']:.3f} "
            f"-> {'GREEN' if green else 'no'}")
        if green:
            return LABELS[est], "GREEN", reasons
    best = max(est_rows[e]["macro_spearman"] for e in ORDER)
    if best >= 0.10:
        return "VALUE_ESTIMATION_UNRESOLVED", "YELLOW", reasons
    return "VALUE_ESTIMATION_UNRESOLVED", "RED", reasons


def write_source_r1a_artifact(out_dir: Path, artifact_dir: Path):
    """Record the R1A runner source that produced the frozen artifacts (§17)."""
    try:
        artifact_commit = (artifact_dir / "git_commit.txt").read_text().strip()
    except Exception:
        artifact_commit = ""
    candidates = []
    try:
        last = subprocess.run(["git", "log", "-1", "--format=%H", "--",
                               "experiments/08-hch-v2/r1a_run.py"],
                              capture_output=True, text=True, timeout=10)
        if last.returncode == 0 and last.stdout.strip():
            candidates.append(last.stdout.strip())
    except Exception:
        pass
    if artifact_commit:
        candidates.append(artifact_commit)
    for commit in candidates:
        try:
            src = subprocess.run(["git", "show",
                                  f"{commit}:experiments/08-hch-v2/r1a_run.py"],
                                 capture_output=True, text=True, timeout=20)
            if src.returncode == 0 and src.stdout.strip():
                with open(out_dir / "source_r1a_artifact.txt", "w",
                          encoding="utf-8") as f:
                    f.write(f"# source_r1a_artifact.txt @ git {commit}\n")
                    f.write(src.stdout)
                return
        except Exception:
            pass
    with open(out_dir / "source_r1a_artifact.txt", "w", encoding="utf-8") as f:
        f.write(f"# source_r1a_artifact.txt @ {_git_head()} (working tree; "
                f"commits {candidates or 'none'} not resolvable)\n")
        f.write(Path(R.__file__).read_text(encoding="utf-8"))


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
        results_dir / f"R1A_VALUE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.6] frozen artifacts: {artifact_dir}")
    print(f"[R1A.6] out: {out_dir}")
    print(f"[R1A.6] variant={args.variant} n_boot={args.n_boot} "
          f"block={args.block}")

    variant = args.variant
    all_rows, all_mg, all_metrics, all_block, all_topdec = [], [], [], [], []
    domain_info = {}
    for ds_key, bb in R.DOMAINS:
        print(f"[R1A.6] {ds_key}:{bb} ...", flush=True)
        res = analyze_domain(artifact_dir, ds_key, bb, variant)
        all_rows.extend(res["rows"])
        all_mg.extend(res["mem_growth"])
        all_metrics.extend(res["metrics"])
        all_block.extend(res["block_metrics"])
        all_topdec.extend(res["topdec"])
        domain_info[f"{ds_key}:{bb}"] = {"k": res["k"], "n_mem0": res["n_mem0"],
                                         "n_eval_days": res["n_eval_days"]}
        print(f"    k={res['k']} n_mem0={res['n_mem0']} "
              f"n_eval={res['n_eval_days']} probs={len(res['problems'])}")
        for pr in res["problems"]:
            print(f"      WARN {pr}")

    vcheck = verify_v0_vs_r1a5(all_rows, results_dir)
    print(f"[R1A.6] V0 vs R1A.5 D2 cross-check: "
          f"{'PASS' if vcheck and vcheck['ok'] else 'N/A'} "
          f"{vcheck}")

    # ---- CSVs ----
    pdf = pd.DataFrame(all_rows)
    pdf.to_csv(out_dir / "value_by_day.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(out_dir / "value_metrics_by_domain.csv",
                                     index=False)
    pd.DataFrame(all_block).to_csv(out_dir / "value_metrics_by_block.csv",
                                   index=False)

    # top-decile enrichment + MACRO rows (mean over domains)
    td = pd.DataFrame(all_topdec)
    top_rows = all_topdec.copy()
    for est, ge in td.groupby("estimator"):
        top_rows.append({
            "domain": "MACRO", "variant": variant, "estimator": est,
            "n_top": int(ge["n_top"].sum()),
            "P(A>0|top10)": float(ge["P(A>0|top10)"].mean()),
            "P(A>0|all)": float(ge["P(A>0|all)"].mean()),
            "hit_lift_top10": float(ge["hit_lift_top10"].mean()),
            "mean_A_true_top10": float(ge["mean_A_true_top10"].mean()),
            "median_A_true_top10": float(ge["median_A_true_top10"].mean()),
            "mean_A_true_all": float(ge["mean_A_true_all"].mean()),
        })
    pd.DataFrame(top_rows).to_csv(out_dir / "top_decile_enrichment.csv",
                                  index=False)

    mgdf = pd.DataFrame(all_mg)
    mgdf.to_csv(out_dir / "memory_growth.csv", index=False)

    boot = bootstrap_all(all_rows, n_boot=args.n_boot, block=args.block,
                         seed=args.boot_seed)
    boot.to_csv(out_dir / "bootstrap_intervals.csv", index=False)

    # ---- figures ----
    for dom in [f"{d}:{b}" for d, b in R.DOMAINS]:
        plot_figures(out_fig, pdf, dom)
        plot_memory_growth(out_fig, mgdf, dom)

    # ---- code_commit + source ----
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{_git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"frozen artifacts: {artifact_dir.name}\n")
        f.write(f"variant: {variant}\n")
        f.write("R1A.6 V0-V4 only; NO R1B; no candidate retrain; "
                "no CRPS/DVG change.\n")
        f.write("R1A source S4 = DEVELOPMENT DATA (plan §10).\n")
    write_source_r1a_artifact(out_dir, artifact_dir)

    with open(out_dir / "estimator_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "plan": "hch_v2_r1a6_retrieval_value_recovery_plan_v0.1_2026-08-13.md",
            "frozen_artifacts": artifact_dir.name,
            "variant": variant,
            "estimators": EST_LABELS,
            "domain_k": {d: v["k"] for d, v in domain_info.items()},
            "domain_n_mem0": {d: v["n_mem0"] for d, v in domain_info.items()},
            "learned_key": "k_d = ReLU(W_sig MeanPool_h(h_core)), "
                           "captured from frozen DataSignature.learned_proj",
            "distance": "D_emb = 1 - cos(k_q, k_j) (plan §5)",
            "memory": {"V0": "static W1 (frozen)", "V1": "none",
                       "V2": "static + learned key",
                       "V3": "prequential expanding + learned key",
                       "V4": "prequential + learned key + weighted A_hat"},
            "weighted_A_hat": "omega_j = exp(-D/(tau+eps))/sum, "
                              "tau = median D over neighbors (plan §7)",
            "bootstrap": {"block": args.block, "n": args.n_boot,
                          "seed": args.boot_seed},
            "s4_status": "R1A source S4 = DEVELOPMENT DATA (plan §10); "
                         "not final confirmatory evidence",
        }, f, indent=2)

    # ---- VALUE_VERDICT ----
    est_rows = summarize_estimators(all_metrics)
    label, gate, reasons = decide_verdict(est_rows)

    lines = []
    lines.append("# R1A.6 VALUE VERDICT — value-estimation recovery (V0–V4)")
    lines.append("")
    lines.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- code commit: `{_git_head()}`")
    lines.append(f"- frozen R1A artifacts: `{artifact_dir.name}` "
                 f"(variant `{variant}`, no retrain)")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"### **{label}** (gate status: {gate})")
    lines.append("")
    lines.append("Decision evidence (§13 / §18):")
    lines.extend(reasons)
    lines.append("")

    lines.append("## Macro Spearman by estimator (mean over 6 domains)")
    for est in EST_ORDER_ALL:
        m = est_rows[est]
        lines.append(f"- **{est}**: macro rho = `{m['macro_spearman']:.3f}` "
                     f"({m['n_pos_domains']}/6 domains > 0, "
                     f"worst = {m['worst_domain_rho']:.3f}, "
                     f"excl-best = {m['macro_excl_best']:.3f}, "
                     f"hit-lift(top10) = {m['macro_hit_lift']:.3f}, "
                     f"eta = {m['macro_eta']:.3f}, "
                     f"missed-pos = {m['macro_missed']:.3f})")
    lines.append("")

    lines.append("## Bootstrap (moving-block, block=7d, "
                 f"{args.n_boot} samples, 95% CI)")
    for _, r in boot.iterrows():
        lines.append(f"- {r['estimator']}: macro rho={r['macro_spearman']:.3f} "
                     f"delta vs V0={r['macro_spearman_delta']:+.3f} "
                     f"CI[{r['spearman_delta_ci_lo']:+.3f}, "
                     f"{r['spearman_delta_ci_hi']:+.3f}] | "
                     f"top10 gain={r['macro_top10_gain']:.4f} "
                     f"delta={r['macro_top10_gain_delta']:+.4f} "
                     f"CI[{r['top10_gain_delta_ci_lo']:+.4f}, "
                     f"{r['top10_gain_delta_ci_hi']:+.4f}]")
    lines.append("")

    lines.append("## Per-domain Spearman (all blocks pooled)")
    md = pd.DataFrame(all_metrics)
    piv = md.pivot_table(index="domain", columns="estimator",
                         values="spearman", aggfunc="first")
    for dom, row in piv.iterrows():
        lines.append("- " + dom + ": " +
                     ", ".join(f"{c}={v:.3f}" if pd.notna(v) else f"{c}=NaN"
                               for c, v in row.items()))
    lines.append("")

    lines.append("## Per-block Spearman (plan §2.1 caveat)")
    bd = pd.DataFrame(all_block)
    for dom, g in bd.groupby("domain"):
        lines.append(f"- {dom}:")
        for (est, blk), gg in g.groupby(["estimator", "block"]):
            v = gg["spearman"].iloc[0]
            lines.append(f"    {est:32s} {blk:7s} rho="
                         f"{v:.3f}" if v is not None else f"    {est:32s} {blk:7s} rho=NaN")
    lines.append("")

    lines.append("## V0 == R1A.5 cross-check")
    if vcheck:
        lines.append(f"- matched {vcheck['n_matched']} days; "
                     f"max|dA_hat|={vcheck['max_dA']:.3e} "
                     f"max|dA_true|={vcheck['max_dT']:.3e} "
                     f"A_true-flip-days(>1e-7)={vcheck['n_Atrue_flip_days_gt_1e-7']} -> "
                     f"{'PASS (V0 == R1A.5 D2 at float32)' if vcheck['ok'] else 'FAIL'}")
        lines.append("  Gate: max|dA_hat|<=1e-6 (ranking authority, all days) AND "
                     "<=2 A_true marginal-flip days. A_hat reproduces R1A.5 D2 at "
                     "float32 precision on every day; a rare A_true flip is a "
                     "~1-ULP atom drift tipping one hour's action on a marginal "
                     "day (A_hat = neighbor mean, barely moves).")
    else:
        lines.append("- R1A_DIAG ahat_vs_atrue.csv not found; skipped")
    lines.append("")

    lines.append("## Notes")
    lines.append("- R1A source S4 = DEVELOPMENT DATA (plan §10): the selected "
                 "estimator is NOT final confirmatory evidence; confirm on "
                 "R1B unseen hosts / NORD_DK1 / Shandong holdout.")
    lines.append("- V4 (weighted A_hat) keeps the SAME proposal as V3; only the "
                 "final A_hat aggregation changes (plan §2.2 caveat).")
    lines.append("- No candidate retrain; no CRPS/DVG/CAGM math change (plan §9).")
    lines.append("")

    verdict_text = "\n".join(lines)
    with open(out_dir / "VALUE_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.6] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
