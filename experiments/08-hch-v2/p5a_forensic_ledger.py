"""Phase 5A/5B — forensic hour-level ledger + four-utility offline recompute.

Design: docs/训练文件夹/对比实验/
    hch_v2_phase5_dual_geometry_action_value_research_direction_v0.1_2026-08-15.md
    §6.1 (ledger) / §6.2 (U0-U3) / §6.3 (5C gate) / §10 (boundaries).

Question: Shandong_DA×PatchTST executed days show POSITIVE scale-free A_true
but DEGRADED raw-yuan MAE. Hypothesis (§0): the action value is learned in
day-scale asinh geometry while the paper metric is raw MAE -> target mismatch
(Jacobian overshoot §2.2: x^π−x^0 = s·cosh(ξ)·π; day-varying weights do not
preserve raw ordering §2.3).

Boundaries (§10, enforced here):
  - IAH-CRPS / three-atom candidate / query-dose replay / double-event
    structure / alpha are UNCHANGED;
  - no new loss, event head, market/host ID, hard thresholds, P4;
  - S4 targets are used ONLY as revealed-labels diagnostics, never to select
    formulas or parameters;
  - existing results and scripts are untouched; new artifact dir
    results/phase5/forensic/.

The forensic chain mirrors p2_cavm_universal_core.run() (same seed, same chain
order) but CAPTURES the full E2 evidence + raw S4 arrays, because the saved
cell JSON only holds summarized quantities. Reproducibility is asserted against
results/phase4/ucore_p2/{mk}_{bb}_s{sd}.json (E2 MAE, n_execute, per-day A_true).
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

import r1a_run as R                                          # noqa: E402
import p2_cavm_universal_core as P2                          # noqa: E402
from hch_v2_pipeline import HCHV2UniversalPipeline            # noqa: E402
from query_replay import estimate_realized_A                  # noqa: E402
from p2_cavm_experiment import (                              # noqa: E402
    pd_date, build_core_context, _run_candidate,
)
from p1_round1 import load_head                               # noqa: E402

OUT = HERE / "results" / "phase5" / "forensic"
OUT.mkdir(parents=True, exist_ok=True)
P2_OUT = HERE / "results" / "phase4" / "ucore_p2"

D_CORE = P2.D_CORE
D_MODEL = P2.D_MODEL
D_VALUE = P2.D_VALUE
ALPHA = P2.ALPHA
K_CANDIDATES = P2.K_CANDIDATES
K_VALIDATION_FRAC = P2.K_VALIDATION_FRAC
EPS = P2.EPS

# §6.1 audit cells: Shandong PatchTST (puzzle) / Shandong Linear (control) /
# LAGO_DE (positive example, all hosts).
AUDIT_CELLS = (
    [("shandong_DA", "PatchTST", s) for s in (0, 1, 2)]
    + [("shandong_DA", "Linear", s) for s in (0, 1, 2)]
    + [(mk, bb, s) for mk in ("LAGO_DE",)
       for bb in ("Linear", "MLP", "PatchTST") for s in (0, 1, 2)]
)

# tail_state diagnostic bucket (pure diagnostic, not a gate): day mean of the
# S1 continuous mid-rank u∈[0,1] (core_context col 0), bucketed at 0.75/0.25.
TAIL_HIGH = 0.75
TAIL_LOW = 0.25


# ------------------------------------------------------------ S_d (§3.2) ----
def frozen_domain_scale(info) -> float:
    """S_d = median over S1R days of daily mean|host forecast| (§3.2).

    Uses only S1R host forecasts (info.yhat_full / info.s_full), zero S4
    dependence. Host cache is per (dataset, backbone) with no seed, so S_d is
    shared across seeds — the domain d=(market, target/mode, host).
    """
    ts = info.ds["ts"]
    yhat = info.yhat_full.astype(np.float64)
    scales = []
    for d in sorted(info.exp.dates_in_split("S1R")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) != 24:
            continue
        hd = yhat[idxs]
        if not np.isfinite(hd).all() or hd.size == 0:
            continue
        s = float(np.mean(np.abs(hd)))
        if s > 0:
            scales.append(s)
    return float(np.median(scales)) if scales else float("nan")


def _iv(iv):
    """Normalize a proposal interval (tuple (l,r), None, or [] fallback)."""
    if iv is None:
        return None
    if isinstance(iv, (list, tuple)) and len(iv) == 2:
        return (int(iv[0]), int(iv[1]))
    return None


def _tail_bucket(mean_u: float) -> str:
    if not np.isfinite(mean_u):
        return "normal"
    if mean_u >= TAIL_HIGH:
        return "high"
    if mean_u <= TAIL_LOW:
        return "low"
    return "normal"


# ------------------------------------------------------- forensic chain -----
def run_forensic(dataset_key: str, backbone: str, seed: int) -> dict:
    """Mirror P2.run() on the frozen universal core, capturing full evidence.

    Returns per-day arrays + executed-day ledger + verification results. The
    chain is deterministic for a given seed (same as P2.run); reproducibility
    is asserted against the saved cell JSON in _verify_cell.
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
    S_d = frozen_domain_scale(info)

    head = load_head(seed, torch)
    pipe = HCHV2UniversalPipeline(d_core_context=D_CORE, d_model=D_MODEL,
                                  d_value=D_VALUE, alpha=ALPHA, k=None,
                                  seed=seed, memory_mode="cavm")
    pipe.candidate_head.load_state_dict(head.state_dict())
    pipe.candidate_head.eval()
    pipe.fit_s1_reference(info.s1_z0, info.s1_hours)

    # learned_sig -> deterministic descriptor ZEROS (§4.4: no market identity).
    det_day = np.zeros(8, dtype=np.float32)
    pipe._domain_det = det_day
    pipe.candidate_head.core_encoder.signature.set_domain_descriptors(det_day)

    # ---- S3-M: memory prefix + k-validation suffix (same as P2.run) ----
    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - K_VALIDATION_FRAC))
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
    k = pipe.select_s3m_k(list(K_CANDIDATES), val_days)
    s3c_days = [sd for sd in (make_day(d) for d in sorted(exp.dates_in_s3c()))
                if sd is not None]
    q_info = pipe.calibrate_s3c(s3c_days)
    pipe.fit_cavm_memory(mem_days)

    # ---- S4 batch ----
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

    # ---- E2 (cavm λ=(0,1)): the deployed intervention mode ----
    pipe.memory_mode = "cavm"
    pipe.set_cavm_retrieval(0.0, 1.0)
    ev = pipe.predict_s4(batch_host, batch_ctx, domain_det=domain_det)

    cand = ev["candidate"]
    n = n_s4
    z0_arr = np.stack([cand["z0"][i].detach().cpu().numpy().reshape(-1)
                       for i in range(n)]) if n else np.zeros((0, 24))
    s_arr = np.array([float(cand["s"][i]) for i in range(n)],
                     dtype=np.float64)
    sv_arr = np.array([float(cand["scale_valid"][i]) for i in range(n)],
                      dtype=np.float64)
    valid_arr = np.stack([cand["valid_mask"][i].detach().cpu().numpy()
                          .reshape(-1).astype(bool) for i in range(n)]
                         ) if n else np.zeros((0, 24), dtype=bool)
    m_minus_arr = np.stack([cand["m_minus"][i].detach().cpu().numpy()
                            .reshape(-1) for i in range(n)]) if n else np.zeros((0, 24))
    m_plus_arr = np.stack([cand["m_plus"][i].detach().cpu().numpy()
                           .reshape(-1) for i in range(n)]) if n else np.zeros((0, 24))
    hosts_arr = np.stack([h.reshape(-1) for h in s4_hosts]) if n else np.zeros((0, 24))
    target_arr = np.stack(s4_y) if n else np.zeros((0, 24))
    pi_arr = np.stack([np.asarray(ev["pi"][i], dtype=np.float64).ravel()
                       for i in range(n)]) if n else np.zeros((0, 24))
    corrected_arr = np.stack([ev["x_final"][i].detach().cpu().numpy()
                              .reshape(-1) for i in range(n)]) if n else np.zeros((0, 24))
    rank_u_arr = np.stack([s4_ctxs[i][:, 0] for i in range(n)]) if n else np.zeros((0, 24))
    actions = ev["final_action"]
    proposals = ev["proposals"]
    a_hat = np.array([float(x) if np.isfinite(float(x)) else np.nan
                      for x in ev["A_hat"]], dtype=np.float64)
    lcb = np.array([float(x) for x in ev["lcb"]], dtype=np.float64)
    q_ev = float(ev["q"]) if ev["q"] is not None else float("nan")

    # ---- derived per-hour fields (diagnostic; §6.1 / §6.2) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        zY_arr = np.where(s_arr[:, None] > 0,
                          np.arcsinh(target_arr / np.maximum(s_arr[:, None], EPS)),
                          0.0)
    g_z = np.abs(zY_arr - z0_arr) - np.abs(zY_arr - z0_arr - pi_arr)
    g_r = np.abs(target_arr - hosts_arr) - np.abs(target_arr - corrected_arr)
    jac = s_arr[:, None] * np.cosh(z0_arr)
    g_mfav = g_r / S_d if np.isfinite(S_d) and S_d > 0 else np.full_like(g_r, np.nan)
    g_daynorm = g_r / np.maximum(s_arr[:, None], EPS)

    exec_mask = np.array([a == "execute" for a in actions], dtype=bool)
    no_lcb_exec = np.zeros(n, dtype=bool)
    for i in range(n):
        p = proposals[i]
        nonempty = (isinstance(p, dict)
                    and (_iv(p.get("I_down")) is not None
                         or _iv(p.get("I_up")) is not None))
        no_lcb_exec[i] = bool(nonempty and np.isfinite(a_hat[i]) and a_hat[i] > 0)

    # ---- per-executed-day aggregates ----
    day_rows = []
    hour_rows = []
    u0_day = np.full(n, np.nan)
    u3_day = np.full(n, np.nan)
    for i in range(n):
        vm = valid_arr[i]
        sv = sv_arr[i] > 0.5 and s_arr[i] > 0
        if not exec_mask[i] or not sv:
            continue
        u0 = estimate_realized_A(z0_arr[i], zY_arr[i], pi_arr[i], vm)
        u0_day[i] = u0
        # U3 day = day_MAE_host - day_MAE_corrected (valid hours only)
        mae_host = float(np.mean(np.abs(hosts_arr[i][vm] - target_arr[i][vm]))) if vm.any() else np.nan
        mae_corr = float(np.mean(np.abs(corrected_arr[i][vm] - target_arr[i][vm]))) if vm.any() else np.nan
        u3 = mae_host - mae_corr
        u3_day[i] = u3
        u1 = u3 / s_arr[i] if s_arr[i] > 0 else np.nan
        u2 = u3 / S_d if np.isfinite(S_d) and S_d > 0 else np.nan

        mean_u = float(np.mean(rank_u_arr[i][vm])) if vm.any() else float("nan")
        tail = _tail_bucket(mean_u)
        idown = _iv(proposals[i].get("I_down")) if isinstance(proposals[i], dict) else None
        iup = _iv(proposals[i].get("I_up")) if isinstance(proposals[i], dict) else None
        p_idown = proposals[i] if isinstance(proposals[i], dict) else {}

        day_rows.append({
            "cell": f"{dataset_key}:{backbone}:s{seed}", "dataset": dataset_key,
            "backbone": backbone, "seed": seed, "date": s4_dates[i],
            "tail_state": tail, "mean_rank_u": round(mean_u, 6),
            "n_valid": int(vm.sum()), "scale_valid": float(sv),
            "scale_day": s_arr[i], "scale_domain": S_d,
            "U0_asinh": u0, "U1_raw_sday": u1, "U2_raw_Sd": u2, "U3_raw": u3,
            "day_mae_host": mae_host, "day_mae_corrected": mae_corr,
            "day_mae_delta": u3,
            "A_hat": a_hat[i], "q": q_ev, "lcb": lcb[i],
            "down_len": (idown[1] - idown[0] + 1) if idown else 0,
            "up_len": (iup[1] - iup[0] + 1) if iup else 0,
        })

        for h in range(24):
            in_down = bool(idown is not None and idown[0] <= h <= idown[1])
            in_up = bool(iup is not None and iup[0] <= h <= iup[1])
            ev_st = "in_I_down" if in_down else ("in_I_up" if in_up else "outside")
            hour_rows.append({
                "cell": f"{dataset_key}:{backbone}:s{seed}", "dataset": dataset_key,
                "backbone": backbone, "seed": seed, "date": s4_dates[i], "hour": h,
                "tail_state": tail, "valid": bool(vm[h]),
                "host_raw": float(hosts_arr[i, h]), "target_raw": float(target_arr[i, h]),
                "corrected_raw": float(corrected_arr[i, h]),
                "scale_day": s_arr[i], "scale_domain": S_d,
                "z0": float(z0_arr[i, h]), "zY": float(zY_arr[i, h]),
                "pi": float(pi_arr[i, h]),
                "gain_z": float(g_z[i, h]), "gain_raw": float(g_r[i, h]),
                "gain_mfav": float(g_mfav[i, h]),
                "jacobian_proxy": float(jac[i, h]), "rank_u": float(rank_u_arr[i, h]),
                "event_state": ev_st,
                "proposal_I_down_l": (idown[0] if idown else None),
                "proposal_I_down_r": (idown[1] if idown else None),
                "proposal_I_up_l": (iup[0] if iup else None),
                "proposal_I_up_r": (iup[1] if iup else None),
                "A_hat": a_hat[i], "q": q_ev, "lcb": lcb[i], "action": "execute",
            })

    return {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "selected_k": k, "q_cal": q_info["q"], "S_d": S_d,
        "n_s4": n, "n_exec": int(exec_mask.sum()),
        "n_no_lcb_exec": int(no_lcb_exec.sum()),
        "s4_dates": s4_dates, "actions": actions,
        "z0": z0_arr, "s": s_arr, "scale_valid": sv_arr, "valid": valid_arr,
        "m_minus": m_minus_arr, "m_plus": m_plus_arr,
        "hosts": hosts_arr, "target": target_arr, "pi": pi_arr,
        "corrected": corrected_arr, "zY": zY_arr, "rank_u": rank_u_arr,
        "g_z": g_z, "g_r": g_r, "g_mfav": g_mfav, "g_daynorm": g_daynorm,
        "jac": jac, "exec_mask": exec_mask, "no_lcb_exec": no_lcb_exec,
        "a_hat": a_hat, "lcb": lcb, "proposals": proposals,
        "u0_day": u0_day, "u3_day": u3_day,
        "day_rows": day_rows, "hour_rows": hour_rows,
        "ev_q": q_ev,
    }


# ----------------------------------------------------- verification (§6.1 Q3) ----
def _verify_cell(cd: dict, saved: dict | None) -> dict:
    checks = {}
    z0, s, hosts = cd["z0"], cd["s"], cd["hosts"]
    n = cd["n_s4"]
    rt = np.abs(s[:, None] * np.sinh(z0) - hosts)
    rt_tol = 1e-3 * np.maximum(np.abs(hosts), 1.0).max() + 1e-6 if n else 1.0
    checks["roundtrip_max"] = float(rt.max()) if n else None
    checks["roundtrip_ok"] = bool(n and rt.max() <= rt_tol)

    exe = cd["exec_mask"]
    if exe.any():
        xp = s[exe, None] * np.sinh(z0[exe] + cd["pi"][exe])
        corr = cd["corrected"][exe]
        diff = np.abs(xp - corr)
        inv_tol = 1e-3 * np.maximum(np.abs(corr), 1.0).max() + 1e-6
        checks["exec_inverse_max"] = float(diff.max())
        checks["exec_inverse_ok"] = bool(diff.max() <= inv_tol)
        # interval indexing: pi nonzero hours inside I_down ∪ I_up, value == ∓m
        pi_ok = True
        for idx in np.where(exe)[0]:
            p = cd["proposals"][idx]
            idown = _iv(p.get("I_down")) if isinstance(p, dict) else None
            iup = _iv(p.get("I_up")) if isinstance(p, dict) else None
            pi = cd["pi"][idx]
            nz = np.where(pi != 0)[0]
            for h in nz:
                in_d = bool(idown is not None and idown[0] <= h <= idown[1])
                in_u = bool(iup is not None and iup[0] <= h <= iup[1])
                if in_d:
                    ok = np.isclose(pi[h], -cd["m_minus"][idx][h], atol=1e-9)
                elif in_u:
                    ok = np.isclose(pi[h], cd["m_plus"][idx][h], atol=1e-9)
                else:
                    ok = False
                pi_ok = pi_ok and bool(ok)
        checks["interval_pi_ok"] = pi_ok
    else:
        checks["exec_inverse_max"] = None
        checks["exec_inverse_ok"] = True
        checks["interval_pi_ok"] = True

    # U0 anchors to the saved cell JSON E2 A_true on executed days
    if saved is not None and "modes" in saved:
        j_a = saved["modes"].get("E2_cavm_01", {}).get("A_true", [])
        u0_diff = []
        for i in np.where(exe)[0]:
            if i < len(j_a) and np.isfinite(j_a[i]):
                u0_diff.append(float(abs(cd["u0_day"][i] - float(j_a[i]))))
        checks["u0_vs_json_max"] = max(u0_diff) if u0_diff else None
        # vacuous pass when no executed days to compare
        checks["u0_vs_json_ok"] = bool(not u0_diff or max(u0_diff) <= 1e-6)
        j_mae = saved["modes"].get("E2_cavm_01", {}).get("MAE")
        j_nx = saved["modes"].get("E2_cavm_01", {}).get("n_execute")
        checks["mae_vs_json_max"] = (float(abs(cd["e2_mae"] - float(j_mae)))
                                     if j_mae is not None else None)
        checks["mae_vs_json_ok"] = (j_mae is None
                                    or abs(cd["e2_mae"] - float(j_mae)) <= 1e-4)
        checks["n_exec_vs_json"] = (None if j_nx is None
                                    else bool(cd["n_exec"] == int(j_nx)))
    else:
        checks["u0_vs_json_ok"] = None
        checks["mae_vs_json_ok"] = None

    # M1: U2 == U3/S_d exactly (construction, mask/units alignment check)
    if np.isfinite(cd["S_d"]) and cd["S_d"] > 0:
        m1 = []
        for r in cd["day_rows"]:
            expect = r["U3_raw"] / cd["S_d"]
            m1.append(float(abs(r["U2_raw_Sd"] - expect))
                      / max(abs(expect), 1e-9))
        checks["M1_max_rel"] = max(m1) if m1 else None
        checks["M1_ok"] = bool(not m1 or max(m1) <= 1e-9)
        # U3 == day_MAE_host - day_MAE_corrected (24h aggregation identity)
        agg = [float(abs(r["U3_raw"] - r["day_mae_delta"])) for r in cd["day_rows"]]
        checks["U3_agg_max"] = max(agg) if agg else None
        checks["U3_agg_ok"] = bool(not agg or max(agg) <= 1e-9)
    else:
        checks["M1_ok"] = None
        checks["U3_agg_ok"] = None

    # candidate scale vs mean|host| over valid hours (scale consistency)
    s_expect = []
    for i in np.where(cd["exec_mask"])[0]:
        vm = cd["valid"][i]
        if vm.any():
            s_expect.append(float(np.mean(np.abs(cd["hosts"][i][vm]))))
    if s_expect:
        se = np.array(s_expect)
        checks["scale_vs_meanhost_max"] = float(np.abs(se - cd["s"][cd["exec_mask"]]).max())
        checks["scale_ok"] = bool(float(np.abs(se - cd["s"][cd["exec_mask"]]).max()) <= 1e-3)
    else:
        checks["scale_ok"] = True

    checks["all_ok"] = bool(
        checks.get("roundtrip_ok", True)
        and checks.get("exec_inverse_ok", True)
        and checks.get("interval_pi_ok", True)
        and checks.get("u0_vs_json_ok", True) is not False
        and checks.get("mae_vs_json_ok", True) is not False
        and checks.get("n_exec_vs_json", True) is not False
        and checks.get("M1_ok", True) is not False
        and checks.get("U3_agg_ok", True) is not False
        and checks.get("scale_ok", True)
    )
    return checks


def _e2_mae(cd: dict) -> float:
    exe = cd["exec_mask"]
    if cd["n_s4"] == 0:
        return float("nan")
    return float(np.mean(np.abs(cd["corrected"] - cd["target"])))


# ------------------------------------------------------------ analysis ------
def _confusion_matrix(sign_a, sign_b):
    """2x2 (excluding 0) counts of (sign_a, sign_b) pairs."""
    out = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "n": 0}
    for a, b in zip(sign_a, sign_b):
        sa = np.sign(a)
        sb = np.sign(b)
        if sa == 0 or sb == 0:
            continue
        out["n"] += 1
        if sa > 0 and sb > 0:
            out["pp"] += 1
        elif sa > 0 and sb < 0:
            out["pn"] += 1
        elif sa < 0 and sb > 0:
            out["np"] += 1
        else:
            out["nn"] += 1
    return out


def _hour_analysis(cd: dict) -> dict:
    exe = cd["exec_mask"]
    if not exe.any():
        return {}
    vm = cd["valid"][exe]
    gz = cd["g_z"][exe][vm]
    gr = cd["g_r"][exe][vm]
    jac = cd["jac"][exe][vm]
    pi = cd["pi"][exe][vm]
    z0 = cd["z0"][exe][vm]
    flip = (gz > 0) & (gr < 0)
    harm = gr < 0
    total_harm = float(np.sum(-gr[harm])) if harm.any() else 0.0
    flip_harm = float(np.sum(-gr[flip])) if flip.any() else 0.0
    jac_med_flip = float(np.median(jac[flip])) if flip.any() else float("nan")
    jac_med_non = float(np.median(jac[~flip])) if (~flip).any() else float("nan")
    conf_hour = _confusion_matrix(gz, gr)
    return {
        "n_exec_hours": int(vm.sum()),
        "hour_confusion": conf_hour,
        "hour_flip_rate": float(flip.mean()) if vm.any() else None,
        "hour_harm_rate": float(harm.mean()) if vm.any() else None,
        "flip_share_of_raw_harm": float(flip_harm / total_harm) if total_harm > 0 else 0.0,
        "jac_median_flip": jac_med_flip,
        "jac_median_nonflip": jac_med_non,
        "jac_flip_ratio": float(jac_med_flip / jac_med_non) if jac_med_non > 0 else float("nan"),
        "mean_abs_z0_exec": float(np.mean(np.abs(z0))),
        "mean_jac_exec": float(np.mean(jac)),
        # decomposition by jacobian_proxy quintiles
        "jac_quint": _quint_stats(jac, gz, gr),
        # decomposition by |z0| quintiles
        "z0_quint": _quint_stats(np.abs(z0), gz, gr),
        # by pi direction
        "dir_down": _dir_stats(pi, gz, gr, "<0"),
        "dir_up": _dir_stats(pi, gz, gr, ">0"),
        "dir_zero": _dir_stats(pi, gz, gr, "==0"),
    }


def _quint_stats(x, gz, gr):
    if x.size < 5:
        return None
    out = []
    for q in (0.0, 0.2, 0.4, 0.6, 0.8):
        lo, hi = float(np.quantile(x, q)), float(np.quantile(x, q + 0.2))
        m = (x >= lo) & (x <= hi)
        out.append({"q_lo": lo, "q_hi": hi, "n": int(m.sum()),
                    "mean_gain_z": float(np.mean(gz[m])),
                    "mean_gain_raw": float(np.mean(gr[m])),
                    "flip_rate": float(np.mean((gz[m] > 0) & (gr[m] < 0)))})
    return out


def _dir_stats(pi, gz, gr, op):
    m = {"<0": pi < 0, ">0": pi > 0, "==0": pi == 0}[op]
    if not m.any():
        return {"op": op, "n": 0}
    return {"op": op, "n": int(m.sum()),
            "mean_gain_z": float(np.mean(gz[m])),
            "mean_gain_raw": float(np.mean(gr[m])),
            "flip_rate": float(np.mean((gz[m] > 0) & (gr[m] < 0))),
            "harm_rate": float(np.mean(gr[m] < 0))}


def _day_analysis(cd: dict) -> dict:
    if not cd["day_rows"]:
        return {}
    u0 = np.array([r["U0_asinh"] for r in cd["day_rows"]])
    u3 = np.array([r["U3_raw"] for r in cd["day_rows"]])
    u2 = np.array([r["U2_raw_Sd"] for r in cd["day_rows"]])
    conf = _confusion_matrix(u0, u3)
    return {
        "n_exec_days": len(cd["day_rows"]),
        "day_confusion": conf,
        "day_flip_rate": float(np.mean((u0 > 0) & (u3 < 0))) if len(u0) else None,
        "mean_U0": float(np.mean(u0)),
        "mean_U1": float(np.mean([r["U1_raw_sday"] for r in cd["day_rows"]])),
        "mean_U2": float(np.mean(u2)),
        "mean_U3": float(np.mean(u3)),
        "exec_raw_harm_rate": float(np.mean(u3 < 0)),
        "exec_raw_help_rate": float(np.mean(u3 > 0)),
        "tail_crosstab": _tail_crosstab(cd),
    }


def _tail_crosstab(cd: dict) -> dict:
    out = {}
    for r in cd["day_rows"]:
        t = r["tail_state"]
        e = out.setdefault(t, {"n": 0, "sum_U0": 0.0, "sum_U3": 0.0})
        e["n"] += 1
        e["sum_U0"] += r["U0_asinh"]
        e["sum_U3"] += r["U3_raw"]
    for t, e in out.items():
        e["mean_U0"] = e["sum_U0"] / e["n"]
        e["mean_U3"] = e["sum_U3"] / e["n"]
        e.pop("sum_U0"); e.pop("sum_U3")
    return out


# ------------------------------------------------------------------ main ----
def _flat_ledger(cd: dict) -> list[dict]:
    rows = cd["hour_rows"]
    return rows


def _flat_day(cd: dict) -> list[dict]:
    return cd["day_rows"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default=None,
                    help="single cell 'MARKET:HOST:SEED' for sanity")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cell:
        mk, bb, sd = args.cell.split(":")
        cell = run_forensic(mk, bb, int(sd))
        cell["e2_mae"] = _e2_mae(cell)
        saved_path = P2_OUT / f"{mk}_{bb}_s{sd}.json"
        saved = json.load(open(saved_path, encoding="utf-8")) if saved_path.exists() else None
        v = _verify_cell(cell, saved)
        print(json.dumps(v, indent=2))
        print(f"sanity {mk}:{bb}:s{sd} n_exec={cell['n_exec']} "
              f"e2_mae={cell['e2_mae']:.4f} S_d={cell['S_d']:.4f}")
        return

    cells_store = []
    failures = []
    for mk, bb, sd in AUDIT_CELLS:
        tag = f"{mk}:{bb}:s{sd}"
        print(f"[forensic] {tag} ...", flush=True)
        try:
            cd = run_forensic(mk, bb, sd)
            cd["e2_mae"] = _e2_mae(cd)
        except Exception as e:
            failures.append({"cell": tag, "error": str(e)})
            print(f"    FAIL {tag}: {e!r}", flush=True)
            continue
        saved_path = P2_OUT / f"{mk}_{bb}_s{sd}.json"
        saved = json.load(open(saved_path, encoding="utf-8")) if saved_path.exists() else None
        cd["verify"] = _verify_cell(cd, saved)
        cd["hour_analysis"] = _hour_analysis(cd)
        cd["day_analysis"] = _day_analysis(cd)
        cells_store.append(cd)
        _write_csv(out_dir / f"ledger_{mk}_{bb}_s{sd}.csv", _flat_ledger(cd))
        _write_csv(out_dir / f"day_summary_{mk}_{bb}_s{sd}.csv", _flat_day(cd))
        v = cd["verify"]
        print(f"    n_exec={cd['n_exec']} S_d={cd['S_d']:.4f} "
              f"MAE={cd['e2_mae']:.4f} verify_ok={v['all_ok']} "
              f"day_flip={cd['day_analysis'].get('day_flip_rate')} "
              f"mean_U0={cd['day_analysis'].get('mean_U0')} "
              f"mean_U3={cd['day_analysis'].get('mean_U3')}", flush=True)

    if failures:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"[forensic] {len(failures)} cells failed: {failures}", flush=True)
    if not cells_store:
        print("[forensic] no successful cells — aborting", flush=True)
        return

    rows = []
    for cd in cells_store:
        d = cd["day_analysis"]
        h = cd["hour_analysis"]
        v = cd["verify"]
        rows.append({
            "cell": f"{cd['dataset']}:{cd['backbone']}:s{cd['seed']}",
            "market": cd["dataset"], "host": cd["backbone"], "seed": cd["seed"],
            "k": cd["selected_k"], "q": cd["q_cal"], "S_d": round(cd["S_d"], 6),
            "n_s4": cd["n_s4"], "n_exec": cd["n_exec"],
            "n_no_lcb_exec": cd["n_no_lcb_exec"],
            "e2_mae": round(cd["e2_mae"], 5),
            "n_exec_days": d.get("n_exec_days"),
            "mean_U0": round(d["mean_U0"], 6) if "mean_U0" in d else None,
            "mean_U1": round(d["mean_U1"], 6) if "mean_U1" in d else None,
            "mean_U2": round(d["mean_U2"], 6) if "mean_U2" in d else None,
            "mean_U3": round(d["mean_U3"], 6) if "mean_U3" in d else None,
            "exec_raw_harm_rate": round(d["exec_raw_harm_rate"], 4) if "exec_raw_harm_rate" in d else None,
            "day_flip_rate": round(d["day_flip_rate"], 4) if "day_flip_rate" in d else None,
            "hour_flip_rate": round(h["hour_flip_rate"], 4) if h and "hour_flip_rate" in h else None,
            "hour_harm_rate": round(h["hour_harm_rate"], 4) if h and "hour_harm_rate" in h else None,
            "flip_share_of_raw_harm": round(h["flip_share_of_raw_harm"], 4) if h and "flip_share_of_raw_harm" in h else None,
            "jac_flip_ratio": round(h["jac_flip_ratio"], 3) if h and "jac_flip_ratio" in h else None,
            "mean_abs_z0_exec": round(h["mean_abs_z0_exec"], 4) if h and "mean_abs_z0_exec" in h else None,
            "mean_jac_exec": round(h["mean_jac_exec"], 4) if h and "mean_jac_exec" in h else None,
            "verify_all_ok": v["all_ok"],
            "verify_roundtrip": v.get("roundtrip_ok"),
            "verify_exec_inverse": v.get("exec_inverse_ok"),
            "verify_u0_anchor": v.get("u0_vs_json_ok"),
            "verify_M1": v.get("M1_ok"),
            "verify_U3_agg": v.get("U3_agg_ok"),
            "verify_scale": v.get("scale_ok"),
        })

    with open(out_dir / "utility_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    verdict = _build_verdict(cells_store, rows, failures)
    with open(out_dir / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print("\n===== phase5 forensic verdict =====")
    print(json.dumps(verdict["verdict"], ensure_ascii=False, indent=2))
    print(f"\n[forensic] artifacts: {out_dir}")


def _build_verdict(cells_store: list[dict], rows: list[dict],
                   failures: list[dict]) -> dict:
    verify_fail = [r for r in rows if not r["verify_all_ok"]]
    by_host = {}
    for r in rows:
        if r["market"] == "shandong_DA":
            by_host.setdefault(r["host"], []).append(r)

    # divergence: executed days where sign(U0)>0 but sign(U3)<0
    day_conf = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "n": 0}
    hour_conf = {"pp": 0, "pn": 0, "np": 0, "nn": 0, "n": 0}
    flip_jac = []
    flip_nonjac = []
    for cd in cells_store:
        da = cd["day_analysis"]
        ha = cd["hour_analysis"]
        if "day_confusion" in da:
            for k in day_conf:
                day_conf[k] += da["day_confusion"].get(k, 0)
        if ha and "hour_confusion" in ha:
            for k in hour_conf:
                hour_conf[k] += ha["hour_confusion"].get(k, 0)
        if ha and ha.get("jac_median_flip") is not None:
            flip_jac.append(ha["jac_median_flip"])
            flip_nonjac.append(ha["jac_median_nonflip"])

    shandong_patchtst = by_host.get("PatchTST", [])
    sd_pt_day_flips = [r["day_flip_rate"] for r in shandong_patchtst
                       if r.get("day_flip_rate") is not None]
    exec_days_total = sum(r.get("n_exec_days") or 0 for r in rows)
    flip_days_total = day_conf.get("pn", 0)  # U0>0 & U3<0
    hour_flip_frac = (hour_conf.get("pn", 0) / max(hour_conf.get("n", 0), 1))
    jac_ratio_med = float(np.median(flip_jac) / np.median(flip_nonjac)) if flip_jac and flip_nonjac else float("nan")

    has_bug = bool(verify_fail) or bool(failures)
    divergence = bool(
        exec_days_total > 0
        and (flip_days_total / exec_days_total) >= 0.3
        and hour_flip_frac >= 0.1
        and np.isfinite(jac_ratio_med) and jac_ratio_med >= 1.5)

    if has_bug and divergence:
        case = "BOTH"
    elif has_bug:
        case = "BUG"
    elif divergence:
        case = "METRIC_MISMATCH"
    else:
        case = "INCONCLUSIVE"

    return {
        "protocol": "hch_v2_phase5_dual_geometry_action_value_research_direction_v0.1 §6.1/§6.2",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": R._git_head(),
        "audit_cells": [r["cell"] for r in rows],
        "n_cells": len(rows),
        "failures": failures,
        "verification": {
            "cells_ok": len(rows) - len(verify_fail), "cells_total": len(rows),
            "failed_cells": [r["cell"] for r in verify_fail],
        },
        "confusion": {
            "day": day_conf, "hour": hour_conf,
            "day_flip_frac": float(day_conf["pn"] / max(exec_days_total, 1)),
            "hour_flip_frac": hour_flip_frac,
            "exec_days_total": exec_days_total,
            "jac_flip_vs_nonflip_median_ratio": round(jac_ratio_med, 3)
                if np.isfinite(jac_ratio_med) else None,
        },
        "shandong": {
            "patchtst_mean_day_flip": float(np.mean(sd_pt_day_flips)) if sd_pt_day_flips else None,
            "patchtst_cells": len(shandong_patchtst),
            "linear_mean_day_flip": float(np.mean(
                [r["day_flip_rate"] for r in by_host.get("Linear", [])
                 if r.get("day_flip_rate") is not None])) if by_host.get("Linear") else None,
        },
        "verdict": {
            "case": case,
            "has_bug": has_bug,
            "divergence": divergence,
            "criteria": {
                "verify_fail_cells": [r["cell"] for r in verify_fail],
                "day_flip_frac": round(day_conf["pn"] / max(exec_days_total, 1), 4),
                "hour_flip_frac": round(hour_flip_frac, 4),
                "jac_ratio_median": round(jac_ratio_med, 3)
                    if np.isfinite(jac_ratio_med) else None,
                "flip_threshold_day": 0.3, "flip_threshold_hour": 0.1,
                "jac_ratio_threshold": 1.5,
            },
            "note": ("METRIC_MISMATCH = code correct but asinh action value "
                     "systematically disagrees with raw MAE on executed days, "
                     "flip hours concentrated at high jacobian_proxy. "
                     "BUG = any verification check failed (fix first, §6.1). "
                     "BOTH = verification failure AND divergence. "
                     "INCONCLUSIVE = code correct, no clean divergence."),
        },
    }


if __name__ == "__main__":
    main()
