"""R1A.5 — retrospective causal diagnostics (D1–D5) + audit-fix verification.

Spec: docs/paper_prep/v2_final_prep/hch_v2_r1a_review_r1a5_diagnostics_r1b_revision_v0.1_2026-08-13.md

Scope (per user): ONLY R1A.5 D1–D5 + the two §2 code-audit fixes. NO model
change, NO candidate retrain, NO D6 (prequential baselines), no R1B edits.

Data: frozen R1A per-domain bundles under
    experiments/08-hch-v2/results/R1A_<ts>/checkpoint_{old_variant}_{ds}_{bb}.pt
(old_variant = nosig|sig). The deterministic S3 chain (S1 ref -> memory -> k ->
DVG) is REBUILT from the bundle's universal state + domain data (frozen weights,
inference only), then cross-checked against the bundle's stored local state.
This yields pipe_before (pre-freeze equivalent). pipe_after = from_bundle.

Outputs -> R1A_DIAG_<ts>/  (artifacts per §19; D6 CSV intentionally absent):
    code_commit.txt, source_r1a_run.txt,
    oracle_actionability_by_domain.csv, ahat_vs_atrue.csv, ahat_deciles.csv,
    proposal_efficiency.csv, neighbor_ablation.csv,
    calibration_error_timeseries.csv, calibration_drift_by_block.csv,
    figures/{ahat_vs_atrue,error_vs_time,calibration_q_drift,oracle_vs_actual_gain}_*.png,
    DIAG_VERDICT.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
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

import r1a_run as R
from common import load_dataset
from eval_manifest import ExperimentManifest
from hch_v2_bundle import HCHV2Bundle
from hch_v2_pipeline import HCHV2UniversalPipeline
from s1_rank import S1RankReference
from query_replay import (estimate_realized_A, estimate_action_value,
                          form_final_pi, replay_query_dose)
from double_event import double_event_proposal

# Old R1A artifact variant key -> renamed variant key (review §2.1)
OLD_TO_NEW = {"nosig": "learned_sig", "sig": "learned_det_sig"}
NEW_TO_OLD = {v: k for k, v in OLD_TO_NEW.items()}
EPS = 1e-12


# ---------------------------------------------------------------- helpers ----
def _n(t) -> np.ndarray:
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return a.reshape(-1).astype(np.float64)


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _make_day(pipe, info, d, det_np) -> dict | None:
    """Replicate r1a_run.run_domain_chain.make_day on a frozen candidate."""
    ts = info.ds["ts"]
    idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
    if len(idxs) != 24:
        return None
    host_day = info.yhat_full[idxs].astype(np.float64)
    if not np.isfinite(host_day).all():
        return None
    hours = ts.iloc[idxs].dt.hour.values
    out, zY = R._run_candidate_day(pipe, host_day, hours, info.z0_full,
                                   info.s_full,
                                   info.ds["price"].astype(np.float32),
                                   idxs, det_np)
    if out is None:
        return None
    return {"date": d, "candidate": out, "target_zY": zY}


# ------------------------------------------------ rebuild + audit (D0) ------
def rebuild_pipe_from_bundle(bundle: HCHV2Bundle, info, variant: str):
    """Deterministic pre-freeze equivalent rebuilt from the bundle + domain data.

    candidate weights + S1 ref from the bundle; memory/k/DVG rebuilt exactly as
    r1a_run did (frozen inference). Returns (pipe, problems: list[str]).
    """
    problems: list[str] = []
    det_np = R.det_for_variant(variant, info)
    alpha = bundle.dvg_alpha if bundle.dvg_alpha is not None else R.ALPHA
    pipe = HCHV2UniversalPipeline(d_core_context=R.D_CORE_CONTEXT,
                                  d_model=R.D_MODEL, d_value=R.D_VALUE,
                                  alpha=alpha, k=None, seed=R.SEED)
    pipe.candidate_head.load_state_dict(bundle.core_model_state)
    pipe.candidate_head.eval()
    pipe.fit_s1_reference(info.s1_z0, info.s1_hours)   # fresh build (as R1A)

    if variant == "learned_sig":
        pipe._domain_det = np.zeros(8)
        pipe.candidate_head.core_encoder.signature.set_domain_descriptors(
            np.zeros(8))
    else:
        # byte-identical to r1a_run.run_domain_chain (fit_s1_signature)
        pipe.fit_s1_signature(info.s1_z0, info.s1_hours)

    # ---- S3-M memory + forward-validation (replicated exactly) ----
    s3m_all = sorted(info.exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * R.S3M_MEM_FRAC)
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]
    mem_days = [md for md in (_make_day(pipe, info, d, det_np)
                              for d in mem_dates) if md is not None]
    if not mem_days:
        raise ValueError(f"{info.ds_key} x {info.bb}: no memory days")
    pipe.fit_s3_memory(mem_days)
    val_days = [vd for vd in (_make_day(pipe, info, d, det_np)
                              for d in val_dates) if vd is not None]

    k_selected = pipe.select_s3m_k(list(R.K_CANDIDATES), val_days)
    if k_selected != bundle.frozen_k:
        problems.append(f"select_s3m_k={k_selected} != frozen_k={bundle.frozen_k}")
    pipe.k = bundle.frozen_k

    # ---- S3-C DVG ----
    s3c_days = [sd for sd in (_make_day(pipe, info, d, det_np)
                              for d in sorted(info.exp.dates_in_split("S3C")))
                if sd is not None]
    pipe.calibrate_s3c(s3c_days)

    # ---- cross-check rebuild vs bundle local state ----
    if list(pipe.memory.dates) != list(bundle.memory_dates):
        problems.append("memory dates differ from bundle")
    am = bundle.atom_memory
    if am is not None:
        for attr in ("z0", "w_minus", "w_zero", "w_plus", "m_minus", "m_plus",
                     "target_zY", "valid_mask"):
            arr = [np.asarray(x) for x in am[attr]]
            for j, (x, y) in enumerate(zip(arr, getattr(pipe.memory, attr))):
                if not np.allclose(np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64), atol=1e-9):
                    problems.append(f"memory.{attr}[{j}] mismatch")
                    break
    if bundle.dvg_q is not None:
        if abs(float(pipe.dvg.q) - float(bundle.dvg_q)) > 1e-12:
            problems.append(f"dvg q {pipe.dvg.q} != bundle {bundle.dvg_q}")
        if len(pipe.dvg.errors) != len(bundle.dvg_errors):
            problems.append("dvg errors length mismatch")
        elif not np.allclose(pipe.dvg.errors, bundle.dvg_errors, atol=1e-9):
            problems.append("dvg errors differ from bundle")
    if bundle.data_signature_spec and variant == "learned_det_sig":
        spec_det = np.asarray(bundle.data_signature_spec["det"], dtype=np.float64)
        if not np.allclose(spec_det, info.det_real, atol=1e-9):
            problems.append("bundle det != info.det_real")
    return pipe, val_days, s3c_days, problems


def run_full_chain_roundtrip(pipe_before, pipe_after, info, s4_days, det_np,
                             n_check=3) -> dict:
    """§2.2 audit on the REBUILT pre-freeze pipe vs reloaded pipe."""
    return R._roundtrip_check(pipe_before, pipe_after, info, s4_days, det_np,
                              n_check=n_check)


# ------------------------------------------------------------- D1 oracle ----
def oracle_for_day(cand, zY: np.ndarray) -> dict:
    """True-directional-gain oracle on one day (review §7).

    g_true_down = |r|-|r+m-|, g_true_up = |r|-|r-m+|, r = zY - z0.
    Feed true gains to the SAME double-event optimizer.
    """
    z0 = _n(cand["z0"])
    m_minus = _n(cand["m_minus"])
    m_plus = _n(cand["m_plus"])
    vm = _n(cand["valid_mask"]).astype(bool)
    r = np.asarray(zY, dtype=np.float64) - z0
    g_down = np.abs(r) - np.abs(r + m_minus)
    g_up = np.abs(r) - np.abs(r - m_plus)
    prop = double_event_proposal(g_down, g_up)
    pi = form_final_pi(m_minus, m_plus, prop["I_down"], prop["I_up"])
    A = estimate_realized_A(z0, zY, pi, vm)
    return {"g_down": g_down, "g_up": g_up, "proposal": prop, "pi": pi,
            "A": A, "vm": vm, "z0": z0}


# -------------------------------------------------------------- D4 engine ----
def _per_neighbor_directional_gains(memory, nbr, m_minus, m_plus, query_valid):
    """Per-neighbor per-hour directional gains for pluggable aggregation."""
    H = len(m_minus)
    gd = np.zeros((len(nbr), H), dtype=np.float64)
    gu = np.zeros((len(nbr), H), dtype=np.float64)
    v = np.zeros((len(nbr), H), dtype=bool)
    for j, idx in enumerate(nbr):
        z0_j = np.asarray(memory.z0[idx], dtype=np.float64)
        zY_j = np.asarray(memory.target_zY[idx], dtype=np.float64)
        vj = np.asarray(memory.valid_mask[idx], dtype=bool)
        rd = replay_query_dose(z0_j, zY_j, -m_minus.astype(np.float64),
                               vj, query_valid)
        ru = replay_query_dose(z0_j, zY_j, m_plus.astype(np.float64),
                               vj, query_valid)
        gd[j] = rd["g"]; gu[j] = ru["g"]; v[j] = rd["valid"]
    return gd, gu, v


def _aggregate_gains(gd, gu, v, method, weights=None, trim=0.1):
    """Aggregate per-neighbor gains hour-wise by method.

    mean / median / trimmed-mean / distance-weighted mean (review §10, Eq).
    Invalid hours excluded per hour (MEDIUM-10 semantics).
    """
    H = gd.shape[1]
    g_d = np.zeros(H); g_u = np.zeros(H)
    for h in range(H):
        sel = v[:, h]
        if not sel.any():
            continue
        if method == "mean":
            g_d[h] = gd[sel, h].mean(); g_u[h] = gu[sel, h].mean()
        elif method == "median":
            g_d[h] = np.median(gd[sel, h]); g_u[h] = np.median(gu[sel, h])
        elif method == "trimmed":
            n = int(sel.sum()); k = int(np.floor(trim * n))
            if k * 2 >= n:
                g_d[h] = np.median(gd[sel, h]); g_u[h] = np.median(gu[sel, h])
            else:
                sd = np.sort(gd[sel, h])[k:n - k]
                su = np.sort(gu[sel, h])[k:n - k]
                g_d[h] = sd.mean(); g_u[h] = su.mean()
        elif method == "dist_weighted":
            w = np.asarray(weights, dtype=np.float64)[sel]
            w = w / (w.sum() + EPS)
            g_d[h] = (gd[sel, h] * w).sum(); g_u[h] = (gu[sel, h] * w).sum()
    return g_d, g_u


def _select_neighbors(memory, out, k, method, dists, seed=0):
    """Neighbor selection: w1 / recent / random (review §10)."""
    if method == "w1":
        return memory.get_neighbors(dists, k)
    if method == "recent":
        n = len(memory)
        return list(range(max(0, n - k), n))
    if method == "random":
        rng = np.random.default_rng(seed)
        n = len(memory)
        idx = rng.choice(n, size=min(k, n), replace=False)
        return sorted(int(i) for i in idx)
    raise ValueError(method)


# ----------------------------------------------------------------- D5 -------
def s4_quarters(s4_days) -> list[int]:
    n = len(s4_days)
    q = int(np.ceil(n / 4))
    blocks = []
    for i in range(n):
        blocks.append(i // q if q else 0)
    return blocks


# ============================================================== per domain ----
@dataclass
class DomainDiag:
    variant: str
    domain: str
    ds_key: str
    bb: str
    problems: list = field(default_factory=list)
    roundtrip: dict = field(default_factory=dict)
    oracle_rows: list = field(default_factory=list)       # per S4 day (D1/D3)
    d2_rows: list = field(default_factory=list)           # per day all blocks
    d3_rows: list = field(default_factory=list)           # per S4 day
    d4_rows: list = field(default_factory=list)           # per sel x agg
    d5_ts: list = field(default_factory=list)             # per day all blocks
    val_candidates: list = field(default_factory=list)    # for D4


def analyze_domain(artifact_dir: Path, ds_key: str, bb: str, variant_new: str,
                   n_check: int = 3) -> DomainDiag:
    old = NEW_TO_OLD[variant_new]
    diag = DomainDiag(variant=variant_new, domain=f"{ds_key}:{bb}",
                      ds_key=ds_key, bb=bb)
    info = R.prepare_domain(ds_key, bb)
    bundle = HCHV2Bundle.load(str(artifact_dir / f"checkpoint_{old}_{ds_key}_{bb}.pt"))
    det_np = R.det_for_variant(variant_new, info)

    # ---- D0: rebuild pre-freeze pipe + verify bundle consistency ----
    pipe, val_days, s3c_days, problems = rebuild_pipe_from_bundle(bundle, info,
                                                                  variant_new)
    diag.problems = problems
    pipe_after = HCHV2UniversalPipeline.from_bundle(bundle)

    # ---- S4 days (ctx via reloaded ref, matching R1A) ----
    ts = info.ds["ts"]
    s4_days = []
    for d in sorted(info.exp.dates_in_split("S4")):
        idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = info.yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = R.build_core_context(host_day, hours, pipe_after.s1_rank_ref,
                                   info.z0_full, info.s_full,
                                   info.ds["price"].astype(np.float32), idxs)
        s4_days.append({"date": d, "idxs": idxs, "host_day": host_day,
                        "hours": hours, "ctx": ctx})
    if not s4_days:
        raise ValueError(f"{diag.domain}: no S4 days")

    # ---- §2.2 full-chain roundtrip (rebuilt-before vs reloaded-after) ----
    diag.roundtrip = run_full_chain_roundtrip(pipe, pipe_after, info, s4_days,
                                              det_np, n_check=n_check)

    # ---- S4 evidence ----
    batch_host = torch.tensor(np.stack([d["host_day"].reshape(24, 1)
                                        for d in s4_days]), dtype=torch.float32)
    batch_ctx = torch.tensor(np.stack([d["ctx"] for d in s4_days]),
                             dtype=torch.float32)
    ev = pipe_after.predict_s4(batch_host, batch_ctx,
                               valid_mask=torch.ones(len(s4_days), 24),
                               domain_det=R.det_for(det_np, len(s4_days)))
    cand = ev["candidate"]
    y_full = info.ds["price"].astype(np.float32)
    q_frozen = float(ev["q"]) if ev["q"] is not None else float("inf")

    quarters = s4_quarters(s4_days)
    for i, day in enumerate(s4_days):
        if float(cand["scale_valid"][i]) < 0.5:
            continue
        target_day = y_full[day["idxs"]].astype(np.float64)
        s_d = float(cand["s"][i])
        z0 = _n(cand["z0"][i])
        zY = np.arcsinh(target_day / max(s_d, 1e-12))
        vm = _n(cand["valid_mask"][i]).astype(bool)
        m_minus = _n(cand["m_minus"][i]); m_plus = _n(cand["m_plus"][i])

        A_hat = float(ev["A_hat"][i])
        pi_q = np.asarray(ev["pi"][i])
        A_true = estimate_realized_A(z0, zY, pi_q, vm)
        E = A_hat - A_true
        lcb = float(ev["lcb"][i])
        action = ev["final_action"][i]
        proposal = ev["proposals"][i]

        # D1 oracle (S4)
        o = oracle_for_day({"z0": cand["z0"][i], "m_minus": cand["m_minus"][i],
                            "m_plus": cand["m_plus"][i],
                            "valid_mask": cand["valid_mask"][i]}, zY)
        A_oracle = o["A"]

        # oracle raw MAE
        x_oracle = s_d * np.sinh(z0 + o["pi"])
        mae_host = np.abs(day["host_day"] - target_day)[vm].mean()
        mae_oracle = np.abs(x_oracle - target_day)[vm].mean()

        # D1 row
        down_evt = o["proposal"]["I_down"] is not None
        up_evt = o["proposal"]["I_up"] is not None
        diag.oracle_rows.append({
            "date": day["date"], "A_oracle": A_oracle, "pi_oracle": o["pi"],
            "down_event": down_evt, "up_event": up_evt,
            "mae_host": float(mae_host), "mae_oracle": float(mae_oracle),
            "A_proposal": A_true, "proposal": proposal,
        })

        # D2 + D5 rows
        block = f"S4Q{quarters[i] + 1}"
        diag.d2_rows.append({"variant": variant_new, "domain": diag.domain,
                             "block": block, "date": day["date"],
                             "A_hat": A_hat, "A_true": A_true, "E": E})
        diag.d5_ts.append({"variant": variant_new, "domain": diag.domain,
                           "block": block, "date": day["date"],
                           "A_hat": A_hat, "A_true": A_true, "E": E,
                           "q_frozen": q_frozen, "lcb": lcb, "action": action})

        # D3 row (S4)
        I_d_p = proposal["I_down"]; I_d_o = o["proposal"]["I_down"]
        I_u_p = proposal["I_up"]; I_u_o = o["proposal"]["I_up"]
        set_p = (set(range(I_d_p[0], I_d_p[1] + 1)) if I_d_p else set()) | \
                (set(range(I_u_p[0], I_u_p[1] + 1)) if I_u_p else set())
        set_o = (set(range(I_d_o[0], I_d_o[1] + 1)) if I_d_o else set()) | \
                (set(range(I_u_o[0], I_u_o[1] + 1)) if I_u_o else set())
        union = set_p | set_o
        iou = len(set_p & set_o) / len(union) if union else 1.0
        down_union = set(range(I_d_p[0], I_d_p[1] + 1)) if I_d_p else set()
        down_o = set(range(I_d_o[0], I_d_o[1] + 1)) if I_d_o else set()
        du = down_union | down_o
        down_iou = len(down_union & down_o) / len(du) if du else 1.0
        up_union = set(range(I_u_p[0], I_u_p[1] + 1)) if I_u_p else set()
        up_o = set(range(I_u_o[0], I_u_o[1] + 1)) if I_u_o else set()
        uu = up_union | up_o
        up_iou = len(up_union & up_o) / len(uu) if uu else 1.0

        direction_mismatch = float(
            np.mean((np.sign(pi_q) != np.sign(o["pi"]))[vm])) if vm.any() else 0.0
        eta = max(A_true, 0.0) / max(A_oracle, EPS)
        missed = 1 if (A_oracle > 1e-9 and A_true <= 0.0) else 0

        diag.d3_rows.append({
            "variant": variant_new, "domain": diag.domain, "date": day["date"],
            "A_proposal": A_true, "A_oracle": A_oracle, "eta": eta,
            "down_iou": down_iou, "up_iou": up_iou,
            "direction_mismatch": direction_mismatch, "missed": missed,
        })

    # ---- S3M-val + S3C (A_hat, A_true) ----
    for block, days in (("S3M-val", val_days), ("S3C", s3c_days)):
        for day in days:
            A_hat, A_true = pipe._replay_value(day, pipe.k)
            E = A_hat - A_true
            diag.d2_rows.append({"variant": variant_new, "domain": diag.domain,
                                 "block": block, "date": day["date"],
                                 "A_hat": A_hat, "A_true": A_true, "E": E})
            diag.d5_ts.append({"variant": variant_new, "domain": diag.domain,
                               "block": block, "date": day["date"],
                               "A_hat": A_hat, "A_true": A_true, "E": E,
                               "q_frozen": q_frozen, "lcb": None,
                               "action": None})

    # ---- D4: neighbor ablation on S3M-val ----
    k = int(pipe.k)
    n_mem = len(pipe.memory)
    for vd in val_days:
        out = vd["candidate"]
        z0 = _n(out["z0"]); zY = np.asarray(vd["target_zY"]).reshape(-1)
        m_minus = _n(out["m_minus"]); m_plus = _n(out["m_plus"])
        vm = _n(out["valid_mask"]).astype(bool)
        dists = pipe.memory.build_retrieval_index(out)
        o = oracle_for_day({"z0": out["z0"], "m_minus": out["m_minus"],
                            "m_plus": out["m_plus"],
                            "valid_mask": out["valid_mask"]}, zY)
        for sel in ("w1", "recent", "random"):
            nbr = _select_neighbors(pipe.memory, out, min(k, n_mem), sel, dists)
            if not nbr:
                continue
            gd, gu, v = _per_neighbor_directional_gains(pipe.memory, nbr,
                                                        m_minus, m_plus, vm)
            D_j = np.asarray([dists[j] for j in nbr], dtype=np.float64)
            med = float(np.median(D_j)) if D_j.size else 0.0
            weights = np.exp(-D_j / (med + EPS))
            for agg in ("mean", "median", "trimmed", "dist_weighted"):
                g_d, g_u = _aggregate_gains(gd, gu, v, agg,
                                            weights=weights)
                prop = double_event_proposal(g_d, g_u)
                pi = form_final_pi(m_minus, m_plus, prop["I_down"],
                                   prop["I_up"])
                A_hat = estimate_action_value(pipe.memory, nbr, pi, vm)["A_hat"]
                A_true = estimate_realized_A(z0, zY, pi, vm)
                eta = max(A_true, 0.0) / max(o["A"], EPS)
                diag.d4_rows.append({
                    "variant": variant_new, "domain": diag.domain,
                    "date": vd["date"], "selection": sel, "aggregation": agg,
                    "A_hat": A_hat, "A_true": A_true, "eta": eta,
                })

    return diag


# ------------------------------------------------------------ aggregation ----
def aggregate_domain(diag: DomainDiag) -> dict:
    dom = diag.domain
    # D1
    o = pd.DataFrame(diag.oracle_rows)
    n = len(o)
    d1 = {
        "domain": dom, "variant": diag.variant, "n_days": n,
        "oracle_positive_day_rate": float((o["A_oracle"] > 1e-9).mean()) if n else None,
        "A_oracle_mean": float(o["A_oracle"].mean()) if n else None,
        "A_oracle_median": float(o["A_oracle"].median()) if n else None,
        "A_oracle_p10": float(o["A_oracle"].quantile(0.10)) if n else None,
        "A_oracle_p90": float(o["A_oracle"].quantile(0.90)) if n else None,
        "identity_oracle_optimal_ratio": float((o["A_oracle"] <= 1e-9).mean()) if n else None,
        "down_event_freq": float(o["down_event"].mean()) if n else None,
        "up_event_freq": float(o["up_event"].mean()) if n else None,
        "host_mae": float(o["mae_host"].mean()) if n else None,
        "oracle_mae": float(o["mae_oracle"].mean()) if n else None,
        "oracle_raw_mae_improvement": float(
            ((o["mae_host"] - o["mae_oracle"]) / o["mae_host"]).mean()) if n else None,
    }
    # D2
    d2 = pd.DataFrame(diag.d2_rows)
    A_hat = d2["A_hat"].to_numpy(); A_true = d2["A_true"].to_numpy()
    d2s = {
        "domain": dom, "variant": diag.variant, "n": len(d2),
        "pearson": float(np.corrcoef(A_hat, A_true)[0, 1]) if len(d2) > 1 else None,
        "spearman": float(stats.spearmanr(A_hat, A_true)[0]) if len(d2) > 1 else None,
        "mae": float(np.mean(np.abs(A_hat - A_true))) if len(d2) else None,
        "bias": float(np.mean(A_hat - A_true)) if len(d2) else None,
        "P(A>0|Ahat>0)": float((A_true > 0)[A_hat > 0].mean())
                         if (A_hat > 0).any() else None,
        "P(A>0|Ahat_top10)": float((A_true > 0)[np.argsort(A_hat)[-max(1, len(d2)//10):]].mean())
                             if len(d2) else None,
    }
    # D2 deciles
    dec = []
    if len(d2) >= 10:
        qs = np.quantile(A_hat, np.arange(0.1, 1.001, 0.1))
        for d_i in range(10):
            lo = -np.inf if d_i == 0 else qs[d_i - 1]
            hi = qs[d_i]
            m = (A_hat >= lo) & (A_hat <= hi)
            dec.append({"domain": dom, "variant": diag.variant,
                        "decile": d_i, "n": int(m.sum()),
                        "mean_A_hat": float(A_hat[m].mean()) if m.any() else None,
                        "mean_A_true": float(A_true[m].mean()) if m.any() else None})
    # D3
    p = pd.DataFrame(diag.d3_rows)
    d3 = {
        "domain": dom, "variant": diag.variant, "n": len(p),
        "eta_mean": float(p["eta"].mean()) if len(p) else None,
        "eta_median": float(p["eta"].median()) if len(p) else None,
        "eta_oracle_pos_days": float(p.loc[p["A_oracle"] > 1e-9, "eta"].mean())
                               if (p["A_oracle"] > 1e-9).any() else None,
        "down_iou": float(p["down_iou"].mean()) if len(p) else None,
        "up_iou": float(p["up_iou"].mean()) if len(p) else None,
        "direction_mismatch": float(p["direction_mismatch"].mean()) if len(p) else None,
        "missed_positive_rate": float(p["missed"].mean()) if len(p) else None,
    }
    # D4
    d4 = []
    if diag.d4_rows:
        df = pd.DataFrame(diag.d4_rows)
        for (sel, agg), g in df.groupby(["selection", "aggregation"]):
            Ah = g["A_hat"].to_numpy(); At = g["A_true"].to_numpy()
            d4.append({
                "domain": dom, "variant": diag.variant,
                "selection": sel, "aggregation": agg, "n": len(g),
                "mae": float(np.mean(np.abs(Ah - At))),
                "spearman": float(stats.spearmanr(Ah, At)[0]) if len(g) > 1 else None,
                "precision": float((At > 0)[Ah > 0].mean()) if (Ah > 0).any() else None,
                "efficiency": float(g["eta"].mean()),
            })
    # D5
    ts = pd.DataFrame(diag.d5_ts)
    q_frozen = float(ts["q_frozen"].iloc[0]) if len(ts) else float("inf")
    s3c_E = ts.loc[ts["block"] == "S3C", "E"]
    d5_blocks = []
    for block, g in ts.groupby("block"):
        E = g["E"].to_numpy()
        cov = float((E <= q_frozen).mean()) if len(E) and np.isfinite(q_frozen) else None
        row = {
            "domain": dom, "variant": diag.variant, "block": block, "n": len(E),
            "median_E": float(np.median(E)) if len(E) else None,
            "q90_E": float(np.quantile(E, 0.90)) if len(E) else None,
            "q95_E": float(np.quantile(E, 0.95)) if len(E) else None,
            "max_E": float(np.max(E)) if len(E) else None,
            "fixed_q_coverage": cov,
        }
        if block.startswith("S4") and len(s3c_E):
            w = stats.wasserstein_distance(s3c_E.to_numpy(), E)
            ks = stats.kstest(s3c_E.to_numpy(), E)
            row["wasserstein_vs_S3C"] = float(w)
            row["ks_vs_S3C_stat"] = float(ks.statistic)
            row["ks_vs_S3C_p"] = float(ks.pvalue)
        d5_blocks.append(row)
    return {"d1": d1, "d2": d2s, "deciles": dec, "d3": d3, "d4": d4,
            "d5_blocks": d5_blocks}


# --------------------------------------------------------------- figures ----
def plot_figures(out_fig, diag: DomainDiag, agg: dict):
    d2 = pd.DataFrame(diag.d2_rows)
    ts = pd.DataFrame(diag.d5_ts)
    o = pd.DataFrame(diag.oracle_rows)
    p = pd.DataFrame(diag.d3_rows)
    dom = diag.domain.replace(":", "_")
    colors = {"S3M-val": "#7f7f7f", "S3C": "#1f77b4", "S4Q1": "#ff7f0e",
              "S4Q2": "#ffa500", "S4Q3": "#d62728", "S4Q4": "#9467bd"}

    # ahat_vs_atrue
    fig, ax = plt.subplots(figsize=(5, 5))
    for block, g in d2.groupby("block"):
        ax.scatter(g["A_hat"], g["A_true"], s=8, alpha=0.5,
                   color=colors.get(block, "#333"), label=block)
    lim = [min(d2["A_hat"].min(), d2["A_true"].min()),
           max(d2["A_hat"].max(), d2["A_true"].max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("A_hat"); ax.set_ylabel("A_true"); ax.legend(fontsize=6)
    ax.set_title(f"{dom} [{diag.variant}] A_hat vs A_true")
    fig.tight_layout(); fig.savefig(out_fig / f"ahat_vs_atrue_{dom}.png", dpi=110)
    plt.close(fig)

    # error_vs_time
    fig, ax = plt.subplots(figsize=(7, 4))
    t0 = ts["date"].iloc[0] if len(ts) else None
    x = pd.to_datetime(ts["date"])
    x = (x - x.min()).dt.total_seconds() / 86400.0
    for block, g in ts.groupby("block"):
        ax.scatter(x[g.index], g["E"], s=8, alpha=0.5,
                   color=colors.get(block, "#333"), label=block)
    qf = float(ts["q_frozen"].iloc[0])
    if np.isfinite(qf):
        ax.axhline(qf, color="k", ls="--", lw=0.9, label=f"q={qf:.4f}")
    ax.set_xlabel("days from first diag day"); ax.set_ylabel("E = A_hat - A_true")
    ax.legend(fontsize=6); ax.set_title(f"{dom} [{diag.variant}] E over time")
    fig.tight_layout(); fig.savefig(out_fig / f"error_vs_time_{dom}.png", dpi=110)
    plt.close(fig)

    # calibration_q_drift
    blocks = agg["d5_blocks"]
    bnames = [b["block"] for b in blocks]
    cov = [b["fixed_q_coverage"] for b in blocks]
    q90 = [b["q90_E"] for b in blocks]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    xpos = np.arange(len(bnames))
    ax.bar(xpos - 0.18, cov, width=0.36, label="fixed-q coverage")
    ax.bar(xpos + 0.18, q90, width=0.36, label="q90(E)")
    ax.set_xticks(xpos); ax.set_xticklabels(bnames, rotation=30, fontsize=7)
    ax.axhline(0.90, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7); ax.set_title(f"{dom} [{diag.variant}] DVG drift by block")
    fig.tight_layout(); fig.savefig(out_fig / f"calibration_q_drift_{dom}.png", dpi=110)
    plt.close(fig)

    # oracle_vs_actual_gain
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(o["A_oracle"], o["A_proposal"], s=8, alpha=0.5, color="#2ca02c")
    lim = [min(o["A_oracle"].min(), o["A_proposal"].min()),
           max(o["A_oracle"].max(), o["A_proposal"].max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("A_oracle"); ax.set_ylabel("A_proposal (realized)")
    ax.set_title(f"{dom} [{diag.variant}] oracle vs actual gain")
    fig.tight_layout(); fig.savefig(out_fig / f"oracle_vs_actual_gain_{dom}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------ verdict -------
VERDICT_LABELS = ("CANDIDATE_ACTIONABILITY", "RETRIEVAL_VALUE_ESTIMATION",
                  "EVENT_PROPOSAL", "DVG_CALIBRATION_DRIFT", "MIXED")


def decide_verdict(aggs: list[dict]) -> tuple[str, list[str]]:
    """Review §18 decision tree. Runs on the MAIN variant (LearnedSig)."""
    main = [a for a in aggs if a["d1"]["variant"] == "learned_sig"]
    if not main:
        main = aggs
    dom_d1 = [a["d1"] for a in main]
    dom_d2 = [a["d2"] for a in main]
    dom_d3 = [a["d3"] for a in main]
    dom_d5 = [a["d5_blocks"] for a in main]

    pos_rate = float(np.mean([d["oracle_positive_day_rate"] for d in dom_d1
                              if d["oracle_positive_day_rate"] is not None]))
    sp = float(np.mean([d["spearman"] for d in dom_d2
                        if d["spearman"] is not None]))
    eta_pos = float(np.mean([d["eta_oracle_pos_days"] for d in dom_d3
                             if d["eta_oracle_pos_days"] is not None]))
    missed = float(np.mean([d["missed_positive_rate"] for d in dom_d3
                            if d["missed_positive_rate"] is not None]))
    drift_frac = float(np.mean([
        1.0 for a in main
        if _s4_drift(a["d5_blocks"])])) if main else 0.0

    evidence = [
        f"- D1 oracle positive-day rate (mean over domains) = {pos_rate:.3f}",
        f"- D2 Spearman(A_hat,A_true) (mean) = {sp:.3f}",
        f"- D3 proposal efficiency on oracle-positive days (mean) = {eta_pos:.3f}",
        f"- D3 missed-positive-event rate (mean) = {missed:.3f}",
        f"- D5 S4-Q4 q90(E) > S3C q90(E) fraction of domains = {drift_frac:.2f}",
    ]
    if pos_rate < 0.50:
        label = "CANDIDATE_ACTIONABILITY"
    elif sp < 0.15:
        label = "RETRIEVAL_VALUE_ESTIMATION"
    elif eta_pos < 0.50 or missed > 0.50:
        label = "EVENT_PROPOSAL"
    elif drift_frac >= 0.50:
        label = "DVG_CALIBRATION_DRIFT"
    else:
        label = "MIXED"
    return label, evidence


def _s4_drift(blocks: list[dict]) -> bool:
    s3c = next((b for b in blocks if b["block"] == "S3C"), None)
    q4 = next((b for b in blocks if b["block"] == "S4Q4"), None)
    if s3c is None or q4 is None or s3c["q90_E"] is None or q4["q90_E"] is None:
        return False
    return float(q4["q90_E"]) > float(s3c["q90_E"])


def write_source_r1a_run(out_dir: Path, artifact_dir: Path):
    """Record the runner source that produced the R1A artifacts (§19).

    Priority: last COMMITTED version of r1a_run.py (the R1A source, pre-audit),
    then the artifact's recorded git_commit.txt, then the working tree.
    """
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
                with open(out_dir / "source_r1a_run.txt", "w",
                          encoding="utf-8") as f:
                    f.write(f"# source_r1a_run.txt @ git {commit}\n")
                    f.write(src.stdout)
                return
        except Exception:
            pass
    with open(out_dir / "source_r1a_run.txt", "w", encoding="utf-8") as f:
        f.write(f"# source_r1a_run.txt @ {_git_head()} (current working tree; "
                f"commits {candidates or 'none'} not resolvable)\n")
        f.write(Path(R.__file__).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default=None,
                    help="R1A artifact dir (default latest R1A_* under results/)")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir; default R1A_DIAG_<timestamp>")
    ap.add_argument("--n-check", type=int, default=3)
    args = ap.parse_args()

    results_dir = HERE / "results"
    if args.artifacts:
        artifact_dir = Path(args.artifacts)
    else:
        # R1A_<timestamp> only — never R1A_DIAG_* output dirs.
        dirs = sorted(results_dir.glob("R1A_[0-9]*"), key=lambda p: p.name)
        if not dirs:
            raise SystemExit("no R1A_* artifact dir found under results/")
        artifact_dir = dirs[-1]
    if not artifact_dir.is_dir():
        raise SystemExit(f"artifact dir not found: {artifact_dir}")

    out_dir = Path(args.out) if args.out else \
        results_dir / f"R1A_DIAG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "figures"
    out_fig.mkdir(exist_ok=True)
    print(f"[R1A.5] artifacts: {artifact_dir}")
    print(f"[R1A.5] out: {out_dir}")

    n_check = args.n_check

    diags, aggs = [], []
    for old_variant in ("nosig", "sig"):
        variant_new = OLD_TO_NEW[old_variant]
        for ds_key, bb in R.DOMAINS:
            print(f"[R1A.5] {variant_new} {ds_key}:{bb} ...", flush=True)
            diag = analyze_domain(artifact_dir, ds_key, bb, variant_new,
                                  n_check=n_check)
            diags.append(diag)
            agg = aggregate_domain(diag)
            aggs.append(agg)
            print(f"    roundtrip ok={diag.roundtrip['ok']} "
                  f"n_checked={diag.roundtrip['n_checked']} "
                  f"probs={len(diag.problems)}")
            if diag.problems:
                for pr in diag.problems:
                    print(f"      WARN {pr}")

    # ---- CSVs ----
    oracle_rows = []
    for diag in diags:
        for r in diag.oracle_rows:
            oracle_rows.append({"variant": diag.variant, "domain": diag.domain,
                                **{k: (v if not isinstance(v, np.ndarray) else None)
                                   for k, v in r.items()}})
    pd.DataFrame(oracle_rows).to_csv(out_dir / "oracle_actionability_by_domain.csv",
                                     index=False)

    d2_all = pd.concat([pd.DataFrame(diag.d2_rows) for diag in diags])
    d2_all.to_csv(out_dir / "ahat_vs_atrue.csv", index=False)

    dec_all = pd.concat([pd.DataFrame(a["deciles"]) for a in aggs if a["deciles"]])
    if len(dec_all):
        dec_all.to_csv(out_dir / "ahat_deciles.csv", index=False)

    d3_all = pd.concat([pd.DataFrame(diag.d3_rows) for diag in diags])
    d3_all.to_csv(out_dir / "proposal_efficiency.csv", index=False)

    d4_all = pd.concat([pd.DataFrame(diag.d4_rows) for diag in diags if diag.d4_rows])
    d4_all.to_csv(out_dir / "neighbor_ablation.csv", index=False)

    ts_all = pd.concat([pd.DataFrame(diag.d5_ts) for diag in diags])
    ts_all.to_csv(out_dir / "calibration_error_timeseries.csv", index=False)

    d5_all = pd.concat([pd.DataFrame(a["d5_blocks"]) for a in aggs])
    d5_all.to_csv(out_dir / "calibration_drift_by_block.csv", index=False)

    # aggregated summaries + roundtrip
    d1_sum = pd.DataFrame([a["d1"] for a in aggs])
    d1_sum.to_csv(out_dir / "_d1_summary.csv", index=False)
    d2_sum = pd.DataFrame([a["d2"] for a in aggs])
    d2_sum.to_csv(out_dir / "_d2_summary.csv", index=False)
    d3_sum = pd.DataFrame([a["d3"] for a in aggs])
    d3_sum.to_csv(out_dir / "_d3_summary.csv", index=False)
    pd.DataFrame([{"variant": d.variant, "domain": d.domain, **d.roundtrip}
                  for d in diags]).to_csv(out_dir / "roundtrip_audit.csv",
                                          index=False)

    # ---- figures (main variant per domain) ----
    for diag in diags:
        if diag.variant != "learned_sig":
            continue
        agg = next(a for a in aggs if a["d1"]["domain"] == diag.domain
                   and a["d1"]["variant"] == "learned_sig")
        plot_figures(out_fig, diag, agg)

    # ---- code_commit + source ----
    with open(out_dir / "code_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{_git_head()}\n")
        f.write(f"run: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"artifacts analyzed: {artifact_dir.name}\n")
        f.write("D6 (prequential_calibration_baselines) NOT run "
                "(user scope: D1-D5 only)\n")
    write_source_r1a_run(out_dir, artifact_dir)

    # ---- DIAG_VERDICT ----
    label, evidence = decide_verdict(aggs)
    lines = []
    lines.append("# R1A.5 DIAG VERDICT — causal action-chain diagnostics")
    lines.append("")
    lines.append(f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- code commit: `{_git_head()}`")
    lines.append(f"- R1A artifacts: `{artifact_dir.name}` (frozen, no retrain)")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"### **{label}**")
    lines.append("")
    lines.append("Decision-tree evidence (§18):")
    lines.extend(evidence)
    lines.append("")

    lines.append("## D0 — audit fixes verification")
    rt = pd.DataFrame([{"variant": d.variant, "domain": d.domain,
                        **d.roundtrip} for d in diags])
    ok_all = bool(rt["ok"].all())
    lines.append(f"- §2.2 full-chain round-trip (scale/rank/atoms/shifts/W1/"
                 f"neighbors/intervals/pi/A_hat/q/LCB/action/x_final, "
                 f"{int(rt['n_checked'].max())} queries): "
                 f"{'PASS' if ok_all else 'FAIL'} on {len(rt)}/12 "
                 f"(variant x domain)")
    n_rebuild_probs = sum(len(d.problems) for d in diags)
    lines.append(f"- §2.1 rename + interpretation applied (LearnedSig / "
                 f"Learned+DetSig); rebuilt-chain consistency problems: "
                 f"{n_rebuild_probs}")
    if n_rebuild_probs:
        for d in diags:
            for pr in d.problems:
                lines.append(f"    - {d.domain}: {pr}")
    lines.append("")

    lines.append("## D1 — Candidate Actionability Oracle")
    for _, r in d1_sum.iterrows():
        lines.append(f"- {r['domain']} [{r['variant']}]: n={int(r['n_days'])} "
                     f"pos-day-rate={r['oracle_positive_day_rate']:.3f} "
                     f"A_oracle med={r['A_oracle_median']:.4f} "
                     f"p10={r['A_oracle_p10']:.4f} p90={r['A_oracle_p90']:.4f} "
                     f"identity-opt={r['identity_oracle_optimal_ratio']:.3f} "
                     f"raw-MAE impr={r['oracle_raw_mae_improvement']:.3f} "
                     f"down={r['down_event_freq']:.3f} up={r['up_event_freq']:.3f}")
    lines.append("")

    lines.append("## D2 — Retrieval / A_hat quality")
    for _, r in d2_sum.iterrows():
        lines.append(f"- {r['domain']} [{r['variant']}]: n={int(r['n'])} "
                     f"pearson={r['pearson']:.3f} spearman={r['spearman']:.3f} "
                     f"MAE={r['mae']:.4f} bias={r['bias']:+.4f} "
                     f"P(A>0|Ah>0)={r['P(A>0|Ahat>0)']:.3f} "
                     f"P(A>0|Ah top10)={r['P(A>0|Ahat_top10)']:.3f}")
    lines.append("")

    lines.append("## D3 — Proposal Efficiency")
    for _, r in d3_sum.iterrows():
        lines.append(f"- {r['domain']} [{r['variant']}]: "
                     f"eta(mean)={r['eta_mean']:.3f} "
                     f"eta(oracle-pos)={r['eta_oracle_pos_days']:.3f} "
                     f"down_IoU={r['down_iou']:.3f} up_IoU={r['up_iou']:.3f} "
                     f"dir-mismatch={r['direction_mismatch']:.3f} "
                     f"missed-pos={r['missed_positive_rate']:.3f}")
    lines.append("")

    lines.append("## D4 — Neighbor Evidence (S3M forward-validation)")
    d4_dom = d4_all[d4_all["variant"] == "learned_sig"]
    for dom, g in d4_dom.groupby("domain"):
        lines.append(f"- {dom}:")
        for (sel, agg), gg in g.groupby(["selection", "aggregation"]):
            Ah = gg["A_hat"].to_numpy(); At = gg["A_true"].to_numpy()
            mae = float(np.mean(np.abs(Ah - At)))
            sp = float(stats.spearmanr(Ah, At)[0]) if len(gg) > 1 else float("nan")
            prec = float((At > 0)[Ah > 0].mean()) if (Ah > 0).any() else float("nan")
            eff = float(gg["eta"].mean())
            lines.append(f"    {sel:8s} x {agg:13s}  MAE={mae:.4f} "
                         f"Spearman={sp:.3f} prec={prec:.3f} eff={eff:.3f}")
    lines.append("")

    lines.append("## D5 — DVG Calibration Drift")
    for _, r in d5_all.iterrows():
        w = r.get("wasserstein_vs_S3C")
        ks_p = r.get("ks_vs_S3C_p")
        lines.append(f"- {r['domain']} [{r['variant']}] {r['block']}: "
                     f"n={int(r['n'])} med_E={r['median_E']:.4f} "
                     f"q90={r['q90_E']:.4f} q95={r['q95_E']:.4f} "
                     f"max={r['max_E']:.4f} "
                     f"fixed-q coverage={r['fixed_q_coverage']:.3f}"
                     + (f" | W1={w:.4f} KS_p={ks_p:.3f}" if w is not None else ""))
    lines.append("")

    lines.append("## Notes")
    lines.append("- D1–D5 are retrospective on frozen R1A artifacts; no model "
                 "change, no candidate retrain (§6).")
    lines.append("- D6 (prequential calibration baselines) NOT run — excluded "
                 "by user scope (only D1–D5).")
    lines.append("- Once R1A S4 informs architecture, S4 is no longer a final "
                 "confirmation set (§6).")
    lines.append("")
    verdict_text = "\n".join(lines)
    with open(out_dir / "DIAG_VERDICT.md", "w", encoding="utf-8") as f:
        f.write(verdict_text)

    print("\n==========================================================")
    print(verdict_text)
    print(f"\n[R1A.5] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
