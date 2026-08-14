"""Phase4 P2 — universal-core CAVM confirmation (paper-config, design §4/§5/§9).

The P2 extension trained a candidate head PER target market (d_model=32) and
found only LAGO_DE direction-stable (GATE_NOT_YET_PASS, 99f75c5). This script
runs the SAME E0-E3 controlled chain but with the FROZEN paper-config universal
core (head_vA, d_model=64, learned_sig) instead of per-market candidate heads:

    load_head(seed)  ->  HCHV2UniversalPipeline(d_model=64, memory_mode="cavm")
    fit_s1_reference (target-local S1, legal)
    S3-M / S3-C      (target-local, legal)
    CAVM ledger from the same S3-M days (core_context + zeros det)
    predict_s4 under E0(w1) / E1(cavm 1,0) / E2(cavm 0,1) / E3(cavm 1,1)

Paper-config invariants (§4.4, enforced here):
  - descriptor for learned_sig is ZEROS (det_for_variant), so c_sig in the CAVM
    key is identically 0 — no market identity leaks into the retrieval key.
  - the candidate head is ONE frozen shared instance per seed (never retrained
    on the target), so every E0-E3 difference is attributable to retrieval.
  - universal core parameters are never updated on the target (S1/S3-M/S3-C
    only build local context/memory/DVG).

Per cell the script emits C0-C3 offline diagnostics (design §5):
  C0 candidate support  : frac_m_alive, dose mean/p95, candidate-vs-host oracle
  C1 proposal support   : proposal_empty_rate, interval lengths, A_hat>0 frac
  C2 DVG gating         : no-LCB execute vs with-LCB execute, blocked days
  C3 realized gain      : exec_count, exec_mean_A_true, harm, point deltas

Outputs (results/phase4/ucore_p2/): per-cell JSON, matrix.csv, summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import r1a_run as R                                        # noqa: E402
from hch_v2_pipeline import HCHV2UniversalPipeline         # noqa: E402
from query_replay import estimate_realized_A               # noqa: E402
from p2_cavm_experiment import (                           # noqa: E402
    pd_date, build_core_context, _run_candidate, _scale_z0,
)
from p1_round1 import load_head, readouts_from_atoms       # noqa: E402

OUT = HERE / "results" / "phase4" / "ucore_p2"
OUT.mkdir(parents=True, exist_ok=True)

# design §4.3 quick screen: 3 target markets x 3 hosts x 3 seeds
TARGETS = ["LAGO_DE", "LAGO_NP", "shandong_DA"]
HOSTS = ["Linear", "MLP", "PatchTST"]
SEEDS = [0, 1, 2]
D_CORE = 13
D_MODEL = 64        # frozen universal core width (NOT P2's 32)
D_VALUE = 0         # schema-agnostic core: no exogenous covariates consumed
ALPHA = R.ALPHA     # 0.10
K_CANDIDATES = (5, 10, 20)
K_VALIDATION_FRAC = 0.25
EPS = 1e-12


# ------------------------------------------------------------- chain build ----
def run(dataset_key: str, backbone: str, seed: int,
        k_validation_frac: float = K_VALIDATION_FRAC,
        k_candidates: tuple = K_CANDIDATES, alpha: float = ALPHA) -> dict:
    """One E0-E3 chain for (target, host, seed) on the FROZEN universal core.

    No S2 training happens here — head_vA is the shared paper-config core.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    info = R.prepare_domain(dataset_key, backbone, seed=seed)
    ds = info.ds
    ts = ds["ts"]
    y_full = ds["price"].astype(np.float32)
    yhat_full = info.yhat_full
    z0_full, s_full = info.z0_full, info.s_full
    exp = info.exp

    # ---- frozen universal core (paper config) ----
    head = load_head(seed, torch)
    pipe = HCHV2UniversalPipeline(d_core_context=D_CORE, d_model=D_MODEL,
                                  d_value=D_VALUE, alpha=alpha, k=None,
                                  seed=seed, memory_mode="cavm")
    pipe.candidate_head.load_state_dict(head.state_dict())
    pipe.candidate_head.eval()
    pipe.fit_s1_reference(info.s1_z0, info.s1_hours)

    # learned_sig -> deterministic descriptor ZEROS (§4.4: no market identity).
    det_day = np.zeros(8, dtype=np.float32)
    pipe._domain_det = det_day
    pipe.candidate_head.core_encoder.signature.set_domain_descriptors(det_day)

    det_broadcast = torch.tensor(det_day, dtype=torch.float32).unsqueeze(0)

    # ---- S3-M: memory prefix + k-validation suffix ----
    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - k_validation_frac))
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]

    def make_day(d):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            return None
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            return None
        hours = ts.iloc[idxs].dt.hour.values
        out, zY = _run_candidate(pipe, host_day, hours, z0_full, s_full,
                                 y_full, idxs)
        if out is None:
            return None
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                 y_full, idxs)
        return {"date": d, "candidate": out, "target_zY": zY,
                "core_context": ctx, "domain_det": det_day}

    mem_days = [md for md in (make_day(d) for d in mem_dates) if md is not None]
    pipe.fit_s3_memory(mem_days)
    val_days = [vd for vd in (make_day(d) for d in val_dates) if vd is not None]
    k = pipe.select_s3m_k(list(k_candidates), val_days)
    s3c_days = [sd for sd in (make_day(d) for d in sorted(exp.dates_in_s3c()))
                if sd is not None]
    q_info = pipe.calibrate_s3c(s3c_days)
    pipe.fit_cavm_memory(mem_days)

    # ---- S4 batch (shared across regimes; predict_s4 is target-free) ----
    s4_hosts, s4_ctxs, s4_dates, s4_y = [], [], [], []
    for d in sorted(exp.dates_in_split("S4")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        host_day = yhat_full[idxs].astype(np.float64)
        if not np.isfinite(host_day).all():
            continue
        hours = ts.iloc[idxs].dt.hour.values
        ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                 y_full, idxs)
        s4_hosts.append(host_day.reshape(24, 1))
        s4_ctxs.append(ctx)
        s4_dates.append(d)
        s4_y.append(y_full[idxs].astype(np.float64))

    n_s4 = len(s4_hosts)
    batch_host = torch.tensor(np.stack(s4_hosts), dtype=torch.float32)
    batch_ctx = torch.tensor(np.stack(s4_ctxs), dtype=torch.float32)
    domain_det = torch.tensor(det_day, dtype=torch.float32).unsqueeze(0).expand(n_s4, -1)

    # ---- negative-price availability (honest market-type label) ----
    y_all = np.concatenate(s4_y) if s4_y else np.zeros(0)
    neg_frac = float(np.mean(y_all < 0)) if len(y_all) else None

    def predict(memory_mode, lam):
        pipe.memory_mode = memory_mode
        pipe.set_cavm_retrieval(lam[0], lam[1])
        return pipe.predict_s4(batch_host, batch_ctx, domain_det=domain_det)

    def summarize(ev):
        n = n_s4
        acts = [ev["final_action"][i] for i in range(n)]
        n_exec = sum(a == "execute" for a in acts)
        y_hat = np.stack([ev["x_final"][i].detach().cpu().numpy().ravel()
                          for i in range(n)]) if n else np.zeros((0, 24))
        y_true = np.stack(s4_y) if n else np.zeros((0, 24))
        mae = float(np.mean(np.abs(y_hat - y_true)))
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))

        a_hats, a_trues = [], []
        for i in range(n):
            cand = ev["candidate"]
            if float(cand["scale_valid"][i]) < 0.5:
                a_hats.append(np.nan); a_trues.append(np.nan)
                continue
            a_hats.append(float(ev["A_hat"][i]))
            s = float(cand["s"][i])
            z0 = cand["z0"][i].detach().cpu().numpy().ravel()
            zY = np.arcsinh(s4_y[i] / s)
            vm = cand["valid_mask"][i].detach().cpu().numpy().ravel().astype(bool)
            pi = np.asarray(ev["pi"][i], dtype=np.float64).ravel()
            a_trues.append(float(estimate_realized_A(z0, zY, pi, vm)))

        a_t = np.asarray(a_trues, dtype=np.float64)
        exec_mask = np.array([a == "execute" for a in acts], dtype=bool)
        a_t_exec = a_t[exec_mask]
        harm = float(np.mean(a_t_exec < 0)) if a_t_exec.size else None
        return {
            "n_days": n, "n_execute": int(n_exec),
            "execute_rate": n_exec / max(n, 1),
            "MAE": mae, "RMSE": rmse,
            "mean_A_hat": float(np.nanmean(a_hats)) if a_hats else None,
            "mean_A_true": float(np.nanmean(a_t)) if a_t.size else None,
            "exec_mean_A_true": float(np.nanmean(a_t_exec)) if a_t_exec.size else None,
            "exec_harm_rate": harm,
            "mean_lcb": float(np.mean([float(ev["lcb"][i]) for i in range(n)]))
                if n else None,
            "actions": acts, "A_hat": a_hats, "A_true": a_trues,
            "neighbors": [list(ev["neighbors"][i]) for i in range(n)],
            "proposals": [ev["proposals"][i] for i in range(n)],
            "pi": [np.asarray(ev["pi"][i], dtype=np.float64).ravel().tolist()
                   for i in range(n)],
        }

    modes = [
        ("E0_w1", "w1", (1.0, 0.0)),
        ("E1_cavm_10", "cavm", (1.0, 0.0)),
        ("E2_cavm_01", "cavm", (0.0, 1.0)),
        ("E3_cavm_11", "cavm", (1.0, 1.0)),
    ]
    results = {}
    ev_e0 = None
    for name, mm, lam in modes:
        ev = predict(mm, lam)
        results[name] = summarize(ev)
        if name == "E0_w1":
            ev_e0 = ev

    # ---- day-by-day deltas vs E0 ----
    delta = {}
    e0 = results["E0_w1"]
    for name in ("E1_cavm_10", "E2_cavm_01", "E3_cavm_11"):
        r = results[name]
        nbr_changed = sum(r["neighbors"][i] != e0["neighbors"][i]
                          for i in range(n_s4))
        act_changed = sum(r["actions"][i] != e0["actions"][i]
                          for i in range(n_s4))
        delta[name] = {
            "neighbor_changed_days": int(nbr_changed),
            "action_changed_days": int(act_changed),
            "MAE_delta": r["MAE"] - e0["MAE"],
            "RMSE_delta": r["RMSE"] - e0["RMSE"],
            "exec_rate_delta": r["execute_rate"] - e0["execute_rate"],
            "A_true_delta": (r["mean_A_true"] - e0["mean_A_true"])
                if r["mean_A_true"] is not None and e0["mean_A_true"] is not None
                else None,
            "exec_mean_A_true_delta": (r["exec_mean_A_true"] - e0["exec_mean_A_true"])
                if r["exec_mean_A_true"] is not None and e0["exec_mean_A_true"] is not None
                else None,
            "exec_harm_rate_delta": (r["exec_harm_rate"] - e0["exec_harm_rate"])
                if r["exec_harm_rate"] is not None and e0["exec_harm_rate"] is not None
                else None,
        }

    # ---- C0-C3 offline diagnostics (design §5, on E0/W1 reference evidence) ----
    diag = diagnostics(ev_e0, s4_hosts, s4_y)

    return {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "frozen_head": f"TRAINER_CMP_20260814_seeds012/head_vA_seed{seed}.pt",
        "d_model": D_MODEL, "variant": "learned_sig",
        "descriptor": "zeros(8)  # learned_sig deterministic det",
        "n_S3M": len(mem_days), "selected_k": k, "q": q_info["q"],
        "n_S4_days": n_s4, "s1r_days": _s1r_count(exp, ts),
        "neg_price_frac_s4": neg_frac,
        "cavm_key_version": pipe.cavm_key_builder.version,
        "cavm_key_dim": pipe.cavm_key_builder.dim,
        "cavm_global_days": len(pipe.cavm_global),
        "modes": results,
        "delta_vs_E0": delta,
        "diagnostics": diag,
    }


# ------------------------------------------------------------- C0-C3 diag ----
def diagnostics(ev, s4_hosts, s4_y) -> dict:
    """design §5 C0-C3 offline decomposition on the W1/E0 reference evidence.

    C0 candidate support, C1 proposal support, C2 DVG gating, C3 realized gain.
    None of these alter the formal S4 modes above.
    s4_hosts: list of [24] raw host-day arrays (host oracle).
    """
    n = len(s4_y)
    cand = ev["candidate"]

    def _row(name: str, i: int) -> np.ndarray:
        a = cand[name][i].detach().cpu().numpy()
        return np.asarray(a).reshape(-1)

    valid = np.stack([_row("valid_mask", i) for i in range(n)])  # [n, H]
    m_minus = np.stack([_row("m_minus", i) for i in range(n)])
    m_plus = np.stack([_row("m_plus", i) for i in range(n)])
    wm = np.stack([_row("w_minus", i) for i in range(n)])
    wz = np.stack([_row("w_zero", i) for i in range(n)])
    wp = np.stack([_row("w_plus", i) for i in range(n)])
    z0 = np.stack([_row("z0", i) for i in range(n)])
    scale_v = np.stack([_row("scale_valid", i) for i in range(n)]).ravel()
    s = np.stack([_row("s", i) for i in range(n)]).ravel()
    acts = ev["final_action"]
    proposals = ev["proposals"]
    a_hat = np.asarray(ev["A_hat"], dtype=np.float64)
    lcb = np.asarray(ev["lcb"], dtype=np.float64)
    # A_true offline, same estimator as summarize (labels revealed after predict)
    a_true = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if scale_v[i] <= 0.5 or s[i] <= 0:
            continue
        zY = np.arcsinh(np.asarray(s4_y[i], dtype=np.float64) / s[i])
        pi = np.asarray(ev["pi"][i], dtype=np.float64).ravel()
        a_true[i] = float(estimate_realized_A(z0[i], zY, pi, valid[i].astype(bool)))

    # ---- C0: candidate support (universal core capacity on the target) ----
    frac_mm, frac_mp, dose_mm, dose_mp, p95_mm, p95_mp = [], [], [], [], [], []
    oracle_delta_mae, wm_mae, host_mae = [], [], []
    for i in range(n):
        v = valid[i].astype(bool)
        sv = scale_v[i] > 0.5
        if not sv or not v.any():
            frac_mm.append(np.nan); frac_mp.append(np.nan)
            dose_mm.append(np.nan); dose_mp.append(np.nan)
            p95_mm.append(np.nan); p95_mp.append(np.nan)
            continue
        mm, mp = m_minus[i][v], m_plus[i][v]
        frac_mm.append(float(np.mean(mm > EPS)))
        frac_mp.append(float(np.mean(mp > EPS)))
        dose_mm.append(float(np.mean(mm)))
        dose_mp.append(float(np.mean(mp)))
        p95_mm.append(float(np.quantile(mm, 0.95)) if mm.size else np.nan)
        p95_mp.append(float(np.quantile(mp, 0.95)) if mp.size else np.nan)
        # candidate oracle point readout (weighted_mean) vs host, raw space
        s_d = float(s[i])
        if s_d > 0:
            rd = readouts_from_atoms(z0[i][v], wm[i][v], wz[i][v], wp[i][v],
                                     m_minus[i][v], m_plus[i][v], s_d, EPS)
            yt = np.asarray(s4_y[i], dtype=np.float64)[v]
            hh = np.asarray(s4_hosts[i], dtype=np.float64).reshape(-1)[v]
            wm_mae.append(float(np.mean(np.abs(rd["weighted_mean"] - yt))))
            host_mae.append(float(np.mean(np.abs(hh - yt))))
    c0 = {
        "frac_m_minus_alive": round(float(np.nanmean(frac_mm)), 4)
            if frac_mm else None,
        "frac_m_plus_alive": round(float(np.nanmean(frac_mp)), 4)
            if frac_mp else None,
        "dose_m_minus_mean": round(float(np.nanmean(dose_mm)), 5)
            if dose_mm else None,
        "dose_m_plus_mean": round(float(np.nanmean(dose_mp)), 5)
            if dose_mp else None,
        "dose_m_minus_p95": round(float(np.nanmean(p95_mm)), 5)
            if p95_mm else None,
        "dose_m_plus_p95": round(float(np.nanmean(p95_mp)), 5)
            if p95_mp else None,
        "oracle_wm_mae": round(float(np.mean(wm_mae)), 5) if wm_mae else None,
        "host_mae": round(float(np.mean(host_mae)), 5) if host_mae else None,
        "oracle_wm_vs_host": round(float(np.mean(wm_mae) - np.mean(host_mae)), 5)
            if wm_mae and host_mae else None,
    }

    # ---- C1: proposal support (no LCB applied) ----
    prop_nonempty = 0
    n_down, n_up, len_down, len_up = [], [], [], []
    for i in range(n):
        p = proposals[i]
        i_d = p.get("I_down") if isinstance(p, dict) else None
        i_u = p.get("I_up") if isinstance(p, dict) else None
        if isinstance(p, dict) and (i_d is not None or i_u is not None):
            prop_nonempty += 1
        if isinstance(p, dict) and i_d is not None:
            n_down.append(i); len_down.append(i_d[1] - i_d[0] + 1)
        if isinstance(p, dict) and i_u is not None:
            n_up.append(i); len_up.append(i_u[1] - i_u[0] + 1)
    a_hat_pos = int(np.sum(np.isfinite(a_hat) & (a_hat > 0)))
    c1 = {
        "proposal_empty_rate": round(1.0 - prop_nonempty / max(n, 1), 4),
        "proposal_nonempty_days": prop_nonempty,
        "down_days": len(n_down), "up_days": len(n_up),
        "down_interval_len_mean": round(float(np.mean(len_down)), 2)
            if len_down else None,
        "up_interval_len_mean": round(float(np.mean(len_up)), 2)
            if len_up else None,
        "A_hat_pos_frac": round(a_hat_pos / max(n, 1), 4),
    }

    # ---- C2: DVG gating (no-LCB vs with-LCB, offline) ----
    # no-LCB execute rule: proposal non-empty AND A_hat > 0.
    no_lcb_exec = np.zeros(n, dtype=bool)
    for i in range(n):
        p = proposals[i]
        nonempty = (isinstance(p, dict)
                    and (p.get("I_down") is not None or p.get("I_up") is not None))
        no_lcb_exec[i] = bool(nonempty and np.isfinite(a_hat[i]) and a_hat[i] > 0)
    with_lcb_exec = np.array([a == "execute" for a in acts], dtype=bool)
    blocked = no_lcb_exec & ~with_lcb_exec
    c2 = {
        "no_lcb_exec_rate": round(float(no_lcb_exec.mean()), 4) if n else None,
        "with_lcb_exec_rate": round(float(with_lcb_exec.mean()), 4) if n else None,
        "lcb_blocked_days": int(blocked.sum()),
        "blocked_among_pos_ahat": round(float(blocked.sum() / max(no_lcb_exec.sum(), 1)), 4)
            if no_lcb_exec.any() else None,
        # realized A_true on no-LCB execute days vs with-LCB execute days
        "no_lcb_exec_mean_A_true": round(float(np.nanmean(a_true[no_lcb_exec])), 5)
            if no_lcb_exec.any() and np.isfinite(a_true[no_lcb_exec]).any() else None,
        "with_lcb_exec_mean_A_true": round(float(np.nanmean(a_true[with_lcb_exec])), 5)
            if with_lcb_exec.any() and np.isfinite(a_true[with_lcb_exec]).any() else None,
        "lcb_mean": round(float(np.mean(lcb)), 5) if n else None,
    }

    # ---- C3: realized gain on executed days ----
    y_hat = ev["x_final"]
    y_hat_np = np.stack([y_hat[i].detach().cpu().numpy().reshape(-1)
                         for i in range(n)]) if n else np.zeros((0, 24))
    y_true = np.stack([np.asarray(y, dtype=np.float64).reshape(-1)
                       for y in s4_y]) if n else np.zeros((0, 24))
    host_np = np.stack([np.asarray(h, dtype=np.float64).reshape(-1)
                        for h in s4_hosts]) if n else np.zeros((0, 24))
    mae_all = float(np.mean(np.abs(y_hat_np - y_true)))
    host_all = float(np.mean(np.abs(host_np - y_true)))
    # normal-regime degradation: identity days point error vs host
    id_mask = ~with_lcb_exec
    mae_id = float(np.mean(np.abs(y_hat_np[id_mask] - y_true[id_mask]))) if id_mask.any() else None
    a_t_exec = a_true[with_lcb_exec]
    c3 = {
        "exec_count": int(with_lcb_exec.sum()),
        "exec_mean_A_true": round(float(np.nanmean(a_t_exec)), 5)
            if with_lcb_exec.any() and np.isfinite(a_t_exec).any() else None,
        "exec_harm_rate": round(float(np.mean(a_t_exec < 0)), 4)
            if with_lcb_exec.any() and np.isfinite(a_t_exec).any() else None,
        "final_mae": round(mae_all, 5),
        "host_mae": round(host_all, 5),
        "point_delta_vs_host": round(mae_all - host_all, 5),
        "identity_days_mae": round(mae_id, 5) if mae_id is not None else None,
        "normal_regime_degradation": round(mae_id - host_all, 5)
            if mae_id is not None else None,
    }
    return {"C0": c0, "C1": c1, "C2": c2, "C3": c3}


def _s1r_count(exp, ts) -> int:
    n = 0
    for d in sorted(exp.dates_in_split("S1R")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) == 24:
            n += 1
    return n


# --------------------------------------------------------------- aggregate ----
def _flat_row(cell: dict) -> list[dict]:
    base = {"cell": f"{cell['dataset']}:{cell['backbone']}:s{cell['seed']}",
            "market": cell["dataset"], "host": cell["backbone"],
            "seed": cell["seed"], "k": cell["selected_k"],
            "q": cell["q"], "s1r_days": cell["s1r_days"]}
    rows = []
    for name in ("E0_w1", "E1_cavm_10", "E2_cavm_01", "E3_cavm_11"):
        m = cell["modes"][name]
        d = cell["delta_vs_E0"].get(name, {})
        rows.append({**base, "mode": name,
                     "n_days": m["n_days"], "n_execute": m["n_execute"],
                     "execute_rate": m["execute_rate"],
                     "MAE": m["MAE"], "RMSE": m["RMSE"],
                     "mean_A_hat": m["mean_A_hat"],
                     "mean_A_true": m["mean_A_true"],
                     "exec_mean_A_true": m["exec_mean_A_true"],
                     "exec_harm_rate": m["exec_harm_rate"],
                     "delta_MAE": d.get("MAE_delta"),
                     "delta_exec_rate": d.get("exec_rate_delta"),
                     "delta_A_true": d.get("A_true_delta"),
                     "delta_exec_A_true": d.get("exec_mean_A_true_delta"),
                     "delta_harm": d.get("exec_harm_rate_delta"),
                     "action_changed_days": d.get("action_changed_days"),
                     "neighbor_changed_days": d.get("neighbor_changed_days"),
                     "neg_price_frac_s4": cell["neg_price_frac_s4"]})
    return rows


def _per_market(rows: list[dict], mode: str) -> dict:
    out = {}
    for mk in TARGETS:
        sub = [r for r in rows if r["market"] == mk and r["mode"] == mode]
        if not sub:
            out[mk] = {"n_cells": 0}
            continue
        dmae = [r["delta_MAE"] for r in sub if r["delta_MAE"] is not None]
        dat = [r["delta_A_true"] for r in sub if r["delta_A_true"] is not None]
        dharm = [r["delta_harm"] for r in sub if r["delta_harm"] is not None]
        point_helps = sum(1 for x in dmae if x < 0) if dmae else 0
        action_helps = sum(1 for x in dat if x > 0) if dat else 0
        both_helps = sum(1 for x, a in zip(dmae, dat) if x < 0 and a > 0)
        both_opp = sum(1 for x, a in zip(dmae, dat) if x < 0 and a < 0)
        by_seed = {}
        for r in sub:
            by_seed.setdefault(r["seed"], []).append(r["delta_MAE"])
        seeds_point_help = sum(
            1 for s, ds in by_seed.items()
            if ds and sum(1 for x in ds if x < 0) >= 2)
        n_cells = len(sub)
        point_help_frac = point_helps / max(n_cells, 1)
        out[mk] = {
            "n_cells": n_cells, "n_seeds": len(by_seed),
            "point_help_frac": round(point_help_frac, 3),
            "mean_delta_MAE": round(float(np.mean(dmae)), 4) if dmae else None,
            "point_help_cells": point_helps,
            "mean_delta_A_true": round(float(np.mean(dat)), 4) if dat else None,
            "action_help_cells": action_helps,
            "both_help_cells": both_helps,
            "both_opposite_cells": both_opp,
            "mean_delta_harm": round(float(np.mean(dharm)), 4) if dharm else None,
            "seeds_with_point_help": seeds_point_help,
            "market_improves": bool(point_help_frac >= 0.5 and point_helps > 0
                                    and dmae and float(np.mean(dmae)) < 0
                                    and action_helps >= 1),
        }
    return out


def build_summary(rows: list[dict]) -> dict:
    e2 = [r for r in rows if r["mode"] == "E2_cavm_01"]
    e3 = [r for r in rows if r["mode"] == "E3_cavm_11"]
    per_mkt = {"E2": _per_market(rows, "E2_cavm_01"),
               "E3": _per_market(rows, "E3_cavm_11")}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    summary = {
        "protocol": "hch_v2_phase4_cavm_p2ext_gate_review_and_next_design_v0.1 §4.3/§5/§9",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": R._git_head(),
        "matrix": {"targets": TARGETS, "hosts": HOSTS, "seeds": SEEDS,
                   "n_cells": len({r["cell"] for r in rows})},
        "frozen_core": "TRAINER_CMP_20260814_seeds012 head_vA (d_model=64, learned_sig, zeros det)",
        "per_market": per_mkt,
        "overall_E2_vs_E0": {
            "n_cells": len(e2),
            "mean_delta_MAE": mean([r["delta_MAE"] for r in e2]),
            "mean_delta_A_true": mean([r["delta_A_true"] for r in e2]),
            "point_help_cells": sum(1 for r in e2 if r["delta_MAE"] is not None and r["delta_MAE"] < 0),
            "action_help_cells": sum(1 for r in e2 if r["delta_A_true"] is not None and r["delta_A_true"] > 0),
            "opposite_cells": sum(1 for r in e2 if r["delta_MAE"] is not None
                                  and r["delta_A_true"] is not None
                                  and (r["delta_MAE"] < 0) != (r["delta_A_true"] > 0)),
        },
        "overall_E3_vs_E0": {
            "n_cells": len(e3),
            "mean_delta_MAE": mean([r["delta_MAE"] for r in e3]),
            "mean_delta_A_true": mean([r["delta_A_true"] for r in e3]),
            "point_help_cells": sum(1 for r in e3 if r["delta_MAE"] is not None and r["delta_MAE"] < 0),
            "action_help_cells": sum(1 for r in e3 if r["delta_A_true"] is not None and r["delta_A_true"] > 0),
            "opposite_cells": sum(1 for r in e3 if r["delta_MAE"] is not None
                                  and r["delta_A_true"] is not None
                                  and (r["delta_MAE"] < 0) != (r["delta_A_true"] > 0)),
        },
    }
    summary["case_verdict"] = _case_verdict_impl(summary, rows)
    return summary


def _case_verdict_impl(s: dict, rows: list[dict]) -> dict:
    """design §6 branch decision from summary + flat rows (cells w/o C0-C3).

    main() recomputes the verdict with per-cell diagnostics via
    _case_verdict_from_cells_impl; this is the flat-rows fallback so the
    function is pure and callable without the diagnostics store.
    """
    per = s["per_market"]["E2"]
    mk_improve = {mk: bool(v.get("market_improves"))
                  for mk, v in per.items()}
    improving = [mk for mk, v in mk_improve.items() if v]
    n_mk = sum(1 for v in mk_improve.values() if v)
    lag0 = per.get("LAGO_NP", {})
    lag0_regress = bool(lag0.get("n_cells", 0) > 0
                        and lag0.get("mean_delta_MAE", 0) > 0
                        and lag0.get("point_help_frac", 0)
                        <= 1 / max(lag0.get("n_cells", 1), 1))
    seeds_ok = all(per[mk].get("seeds_with_point_help", 0) >= 2
                   for mk in improving) if improving else False
    if n_mk >= 2 and not lag0_regress and seeds_ok:
        case = "A"
    elif n_mk == 1 and improving == ["LAGO_DE"]:
        case = "D"
    else:
        case = "D"
    return {"case": case, "n_market_improving": n_mk,
            "market_improving": sorted(improving),
            "LAGO_NP_regress": lag0_regress, "seeds_ok": seeds_ok}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=str, default=",".join(TARGETS))
    ap.add_argument("--hosts", type=str, default=",".join(HOSTS))
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--cell", type=str, default=None,
                    help="single cell 'MARKET:HOST:SEED' for sanity (skips matrix)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cell:
        mk, bb, sd = args.cell.split(":")
        cell = run(mk, bb, int(sd))
        with open(out_dir / f"sanity_{mk}_{bb}_s{sd}.json", "w",
                  encoding="utf-8") as f:
            json.dump(cell, f, indent=2, default=str)
        print(f"sanity cell {mk}:{bb}:s{sd} done -> {out_dir / f'sanity_{mk}_{bb}_s{sd}.json'}")
        return

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    cells_store = []
    rows = []
    failures = []
    for mk in targets:
        for bb in hosts:
            for sd in seeds:
                tag = f"{mk}:{bb}:s{sd}"
                print(f"[ucore] {tag} ...", flush=True)
                try:
                    cell = run(mk, bb, sd)
                except Exception as e:
                    failures.append({"cell": tag, "error": str(e)})
                    print(f"    FAIL {tag}: {e!r}", flush=True)
                    continue
                with open(out_dir / f"{mk}_{bb}_s{sd}.json", "w",
                          encoding="utf-8") as f:
                    json.dump(cell, f, indent=2, default=str)
                cells_store.append(cell)
                rows.extend(_flat_row(cell))
                r = cell["modes"]["E2_cavm_01"]
                d = cell["diagnostics"]
                print(f"    E2 MAE={r['MAE']:.3f} (Δ={cell['delta_vs_E0']['E2_cavm_01']['MAE_delta']:+.3f}) "
                      f"C0.alive[m-,m+]={d['C0']['frac_m_minus_alive']},{d['C0']['frac_m_plus_alive']} "
                      f"C1.empty={d['C1']['proposal_empty_rate']} "
                      f"k={cell['selected_k']}", flush=True)
    if failures:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"[ucore] {len(failures)} cells failed: {failures}", flush=True)
    if not rows:
        print("[ucore] no successful cells — aborting summary", flush=True)
        return

    with open(out_dir / "matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    summary["case_verdict"] = _case_verdict_from_cells_impl(summary, cells_store)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== design §6 case verdict =====")
    print(json.dumps(summary["case_verdict"], ensure_ascii=False, indent=2))
    print(f"\n[ucore] artifacts: {out_dir}")


def _case_verdict_from_cells_impl(s: dict, cells: list[dict]) -> dict:
    """design §6 A/B/C/D decision using aggregated C0-C2 from real cells."""
    per = s["per_market"]["E2"]
    mk_improve = {mk: bool(v.get("market_improves"))
                  for mk, v in per.items()}
    improving = [mk for mk, v in mk_improve.items() if v]
    n_mk = sum(1 for v in mk_improve.values() if v)

    # LAGO_NP regression check (design §6 case A gate)
    lag0 = per.get("LAGO_NP", {})
    lag0_regress = bool(lag0.get("n_cells", 0) > 0
                        and lag0.get("mean_delta_MAE", 0) > 0
                        and lag0.get("point_help_frac", 0)
                        <= 1 / max(lag0.get("n_cells", 1), 1))

    # seed consistency: every improving market must hold in >=2 of 3 seeds
    seeds_ok = all(per[mk].get("seeds_with_point_help", 0) >= 2
                   for mk in improving) if improving else False

    # aggregated C0/C1/C2 over cells
    def agg(key, fn):
        vals = [fn(c["diagnostics"][key]) for c in cells]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    c0_mminus = agg("C0", lambda d: d["frac_m_minus_alive"])
    c0_mplus = agg("C0", lambda d: d["frac_m_plus_alive"])
    c0_oracle = agg("C0", lambda d: d["oracle_wm_vs_host"])
    c1_empty = agg("C1", lambda d: d["proposal_empty_rate"])
    c1_ahat_pos = agg("C1", lambda d: d["A_hat_pos_frac"])
    c2_nolcb_rate = agg("C2", lambda d: d["no_lcb_exec_rate"])
    c2_withlcb_rate = agg("C2", lambda d: d["with_lcb_exec_rate"])
    c2_nolcb_atrue = agg("C2", lambda d: d["no_lcb_exec_mean_A_true"])
    c2_blocked = agg("C2", lambda d: d["lcb_blocked_days"])
    c3_exec = agg("C3", lambda d: d["exec_count"])

    candidate_capacity = (c0_mminus is not None and c0_mplus is not None
                          and c0_mminus >= 0.05 and c0_mplus >= 0.05)
    proposal_support = (c1_empty is not None and c1_ahat_pos is not None
                        and c1_empty < 0.9 and c1_ahat_pos > 0.1)
    no_lcb_profitable = (c2_nolcb_atrue is not None and c2_nolcb_atrue > 0)
    lcb_overconservative = (c2_withlcb_rate is not None and c2_nolcb_rate is not None
                            and c2_withlcb_rate < 0.1 and c2_nolcb_rate > 0.3)

    # Branch logic (design §6). Order: A -> C (capacity) -> B -> D.
    if n_mk >= 2 and not lag0_regress and seeds_ok:
        case = "A"
    elif not candidate_capacity:
        case = "C"   # universal core candidate support insufficient on target
    elif proposal_support and no_lcb_profitable and lcb_overconservative:
        case = "B"   # candidate + proposal capacity, but LCB too conservative
    elif n_mk == 1 and improving == ["LAGO_DE"]:
        case = "D"
    else:
        case = "D"
    return {
        "case": case,
        "n_market_improving": n_mk,
        "market_improving": sorted(improving),
        "LAGO_NP_regress": lag0_regress,
        "seeds_ok": seeds_ok,
        "agg_C0_m_minus_alive": round(c0_mminus, 4) if c0_mminus is not None else None,
        "agg_C0_m_plus_alive": round(c0_mplus, 4) if c0_mplus is not None else None,
        "agg_C0_oracle_wm_vs_host": round(c0_oracle, 4) if c0_oracle is not None else None,
        "agg_C1_proposal_empty_rate": round(c1_empty, 4) if c1_empty is not None else None,
        "agg_C1_A_hat_pos_frac": round(c1_ahat_pos, 4) if c1_ahat_pos is not None else None,
        "agg_C2_no_lcb_exec_rate": round(c2_nolcb_rate, 4) if c2_nolcb_rate is not None else None,
        "agg_C2_with_lcb_exec_rate": round(c2_withlcb_rate, 4) if c2_withlcb_rate is not None else None,
        "agg_C2_no_lcb_mean_A_true": round(c2_nolcb_atrue, 5) if c2_nolcb_atrue is not None else None,
        "agg_C2_LCB_blocked_days": round(c2_blocked, 3) if c2_blocked is not None else None,
        "agg_C3_exec_count": round(c3_exec, 3) if c3_exec is not None else None,
        "candidate_capacity": bool(candidate_capacity),
        "proposal_support": bool(proposal_support),
        "no_lcb_profitable": bool(no_lcb_profitable),
        "lcb_overconservative": bool(lcb_overconservative),
    }


if __name__ == "__main__":
    main()
