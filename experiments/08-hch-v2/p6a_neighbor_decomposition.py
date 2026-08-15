"""Phase 6.0 — E2-vs-E0 neighbor-set decomposition (retrieval / day-selection quality).

Design: docs/训练文件夹/对比实验/
    hch_v2_phase6_retrieval_reliability_generalization_research_plan_v0.1_2026-08-15.md
    §5 (Phase 6.0 audit matrix) / §4 (APR, NOT implemented here) / §2 (boundaries).

Question: under the same frozen universal core, same candidate atoms and same
DVG, why does CAVM λ=(0,1) (E2) fail to select the high-value neighbors that W1
(E0) selects on Shandong×PatchTST (fewer executed days 31/44/50 vs 35/60/60,
slightly worse point MAE)? Phase 5A/B (§0.1) already ruled out metric mismatch;
this is a retrieval / day-selection quality question.

The chain mirrors P2.run / p5a.run_forensic EXACTLY (same seed, same stage
order, same make_day / build_core_context / _run_candidate). Candidate atoms are
computed once; the three arms E0(w1)/E2(cavm 0,1)/E3(cavm 1,1) only differ in
neighbor selection + distance metric. Everything downstream (uniform-mean
directional gains -> double-event proposal -> uniform A_hat -> DVG LCB) is
replayed OFFLINE on the aligned ledger so the four counterfactuals
(Δ_retrieval_set / Δ_weight_k / Δ_proposal / Δ_LCB) are reproducible.

Boundaries (§2, enforced):
  - IAH-CRPS / three-atom candidate / query-dose replay / double-event
    structure / alpha / LCB / DVG are UNCHANGED.
  - No APR, no new loss/event head/market-host ID/hard thresholds/P4.
  - S4 targets are used ONLY as revealed-label diagnostics (A_true / realized
    raw gain), never to select formulas or parameters.
  - Existing results and scripts untouched; new artifact dir results/phase6/decomp/.

Reproducibility anchors: all three arm MAE / n_execute / per-day A_true /
neighbors reproduce results/phase4/ucore_p2/{mk}_{bb}_s{sd}.json to tolerance
(any mismatch -> hard stop).
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
from context_action_memory import composite_distance          # noqa: E402
from query_replay import (                                    # noqa: E402
    estimate_realized_A, estimate_action_value, form_final_pi,
)
from double_event import double_event_proposal                # noqa: E402
from p2_cavm_experiment import (                              # noqa: E402
    pd_date, build_core_context, _run_candidate,
)
from p1_round1 import load_head                               # noqa: E402
from r1a5_diag import _per_neighbor_directional_gains, _aggregate_gains  # noqa: E402
from r1a7_fusion_audit import (                               # noqa: E402
    weighted_directional_gains, weighted_action_value,
)
from p5a_forensic_ledger import frozen_domain_scale           # noqa: E402

OUT = HERE / "results" / "phase6" / "decomp"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)
P2_OUT = HERE / "results" / "phase4" / "ucore_p2"

D_CORE = P2.D_CORE
D_MODEL = P2.D_MODEL
D_VALUE = P2.D_VALUE
ALPHA = P2.ALPHA
K_CANDIDATES = P2.K_CANDIDATES
K_VALIDATION_FRAC = P2.K_VALIDATION_FRAC
EPS = 1e-12

# Audit matrix (design §5 Phase 6.0): 3 markets x 3 hosts x 3 seeds.
TARGETS = P2.TARGETS
HOSTS = P2.HOSTS
SEEDS = P2.SEEDS

# Three arms: (json_key, memory_mode, (lambda_atom, lambda_ctx))
ARMS = [
    ("E0_w1", "w1", (1.0, 0.0)),
    ("E2_cavm_01", "cavm", (0.0, 1.0)),
    ("E3_cavm_11", "cavm", (1.0, 1.0)),
]

# Verdict thresholds (design §5 Phase 6.0 rules, quantized).
DATA_SUPPORT_EMPTY = 0.9    # both arms proposal-empty rate >= this -> DATA_SUPPORT
K_WEIGHT_JACC = 0.7         # median Jaccard >= this -> K_WEIGHT
LOW_JACC = 0.5              # median Jaccard < this enables PROPOSAL/KEY branch
SIGN_AGREE_HIGH = 0.6       # action-sign agreement >= this -> PROPOSAL
SIGN_AGREE_LOW = 0.4        # action-sign agreement <= this -> KEY


# ------------------------------------------------------------ json utils ----
def _jnum(v):
    """Normalize a JSON value to float (str-typed numbers -> float)."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _jdefault(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------ shared utils ----
def _norm01_row(a):
    """Monotone /max normalisation (same semantics as CAVM._norm01)."""
    a = np.asarray(a, dtype=np.float64)
    m = float(a.max()) if a.size else 0.0
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(a)
    return a / m


def _prop_empty(p) -> bool:
    """True when a proposal dict carries no usable Down/Up interval."""
    if not isinstance(p, dict):
        return True
    return not (p.get("I_down") is not None or p.get("I_up") is not None)


def _sign_code_proposal(p) -> str:
    """Direction-presence code of a proposal: '', 'D', 'U', 'DU'."""
    if not isinstance(p, dict):
        return ""
    return (("D" if p.get("I_down") is not None else "")
            + ("U" if p.get("I_up") is not None else ""))


def _ess(d):
    """n_eff = 1/sum(w^2) for softmax weights w=exp(-D/tau), tau=median(D)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 1:
        return float("nan")
    tau = float(np.median(d))
    w = np.exp(-d / (tau + EPS))
    w = w / (w.sum() + EPS)
    ss = float(np.sum(w ** 2))
    return float(1.0 / ss) if ss > 0 else float("nan")


def _margin(d):
    """(second-nearest - nearest)/max(nearest,eps) on a finite distance set."""
    d = np.asarray(d, dtype=np.float64)
    d = np.sort(d[np.isfinite(d)])
    if d.size < 2 or d[0] <= 0:
        return float("nan")
    return float((d[1] - d[0]) / d[0])


def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2 or np.all(x[m] == x[m][0]) or np.all(y[m] == y[m][0]):
            return float("nan")
        rho, _ = spearmanr(x[m], y[m])
        return float(rho)
    except Exception:
        return float("nan")


def _fmed(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else None


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))


# ------------------------------------------------------------ replay ----
def replay_chain(mem, nbr, mm, mp, z0_q, zY_q, vm, agg="mean", weights=None,
                 proposal_fn=None):
    """Full offline replay chain on a given neighbor set (design §5).

    agg in {"mean","median","trimmed","weighted"}. proposal_fn defaults to
    double_event_proposal; pass lambda gd,gu: double_event_proposal(gd, zeros)
    for a down-only variant. Returns gains, proposal, pi, A_hat, A_true.
    """
    if not nbr:
        pi = np.zeros(len(mm), dtype=np.float64)
        return {"proposal": {"I_down": None, "I_up": None},
                "pi": pi, "A_hat": 0.0,
                "A_true": estimate_realized_A(z0_q, zY_q, pi, vm),
                "proposal_nonempty": False}
    if agg == "weighted":
        gains = weighted_directional_gains(mem, nbr, weights, mm, mp, vm)
        gd, gu = gains["g_hat_down"], gains["g_hat_up"]
    else:
        gd, gu = _aggregate_gains(*_per_neighbor_directional_gains(
            mem, nbr, mm, mp, vm), agg)
    pf = proposal_fn or double_event_proposal
    prop = pf(gd, gu)
    pi = form_final_pi(mm, mp, prop["I_down"], prop["I_up"])
    if agg == "weighted":
        A_hat = weighted_action_value(mem, nbr, weights, pi, vm)
    else:
        A_hat = estimate_action_value(mem, nbr, pi, vm)["A_hat"]
    A_true = estimate_realized_A(z0_q, zY_q, pi, vm)
    return {"gd": gd, "gu": gu, "proposal": prop, "pi": pi, "A_hat": A_hat,
            "A_true": A_true,
            "proposal_nonempty": not _prop_empty(prop)}


def _replay_signed(mem, nbr, mm, mp, z0_q, zY_q, vm, s_q, host_q, target_q,
                   q, agg="mean", weights=None, down_only=False,
                   lcb_mode="with"):
    """Offline replay with an explicit LCB decision + realized raw gain u3.

    u3 = day_MAE_host - day_MAE_corrected (valid hours) on the query day when
    execute; NaN otherwise (same per-day definition as p5a). lcb_mode:
      "with"  -> execute iff A_hat - q > 0   (deployed DVG gate)
      "none"  -> execute iff proposal non-empty AND A_hat > 0
    """
    mm = np.asarray(mm, dtype=np.float64)
    mp = np.asarray(mp, dtype=np.float64)
    if down_only:
        def pf(gd, gu):
            return double_event_proposal(gd, np.zeros_like(gd))
    else:
        pf = None
    r = replay_chain(mem, nbr, mm, mp, z0_q, zY_q, vm, agg=agg, weights=weights,
                     proposal_fn=pf)
    if lcb_mode == "none":
        execute = bool(r["proposal_nonempty"] and r["A_hat"] > 0)
    else:
        execute = bool(r["A_hat"] - q > 0)
    u3 = float("nan")
    if execute:
        xf = np.asarray(s_q, dtype=np.float64) * np.sinh(z0_q + r["pi"])
        if vm.any():
            mh = float(np.mean(np.abs(host_q[vm] - target_q[vm])))
            mc = float(np.mean(np.abs(xf[vm] - target_q[vm])))
            u3 = mh - mc
    return {"execute": execute, "A_hat": r["A_hat"], "A_true": r["A_true"],
            "u3": u3}


def _neighbor_sign_stability(mem, nbr, mm, mp, vm):
    """Within-arm per-neighbor sign stability vs the uniform-mean aggregate.

    For every hour where the aggregate directional gain |g| > EPS, the fraction
    of neighbors whose own per-hour gain shares the aggregate sign. Mean over
    down/up and over such hours. NaN when no neighbor / no meaningful hour.
    """
    gd, gu, v = _per_neighbor_directional_gains(mem, nbr, mm, mp, vm)
    gd_agg, gu_agg = _aggregate_gains(gd, gu, v, "mean")
    parts = []
    for g_j, g_agg in ((gd, gd_agg), (gu, gu_agg)):
        hz = np.where(np.abs(g_agg) > EPS)[0]
        if not len(hz):
            continue
        agrees = []
        for h in hz:
            sel = v[:, h]
            if not sel.any():
                continue
            agrees.append(float(np.mean(np.sign(g_j[sel, h])
                                        == np.sign(g_agg[h]))))
        if agrees:
            parts.append(float(np.mean(agrees)))
    return float(np.mean(parts)) if parts else float("nan")


# ------------------------------------------------------------------ chain ----
def run_decomp(dataset_key: str, backbone: str, seed: int) -> dict:
    """One (market, host, seed) cell: three-arm retrieval decomposition."""
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

    # Index alignment (§5): E0 indices into pipe.memory.dates, E2/E3 into
    # pipe.cavm_global.dates; both built from the SAME mem_days in order.
    assert list(pipe.memory.dates) == list(pipe.cavm_global.dates), (
        f"{dataset_key}:{backbone}:s{seed} memory/cavm date order drifted")
    M = len(pipe.cavm_global)
    mem_date_list = list(pipe.cavm_global.dates)
    assert M == len(mem_days)

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

    # ---- three arms (neighbor selection is the ONLY thing that differs) ----
    evs = {}
    for name, mm_, lam in ARMS:
        pipe.memory_mode = mm_
        pipe.set_cavm_retrieval(lam[0], lam[1])
        evs[name] = pipe.predict_s4(batch_host, batch_ctx, domain_det=domain_det)
    cand = evs["E0_w1"]["candidate"]

    # ---- shared per-query distance matrices (candidate is mode-independent) ----
    w1_full = np.full((n_s4, M), np.inf)
    for b in range(n_s4):
        w1_full[b] = pipe.memory.build_retrieval_index(
            _cand_view(cand, b))
    keys = evs["E2_cavm_01"].get("context_keys", [])
    ctx_full = np.full((n_s4, M), np.inf)
    if keys:
        for b in range(n_s4):
            ctx_full[b] = pipe.cavm_global.context_distances(
                np.asarray(keys[b], dtype=np.float64))
    nw1 = np.stack([_norm01_row(w1_full[b]) for b in range(n_s4)])
    nctx = np.stack([_norm01_row(ctx_full[b]) for b in range(n_s4)])

    # ---- per-query atom arrays (shared across arms) ----
    z0_arr = np.stack([cand["z0"][i].detach().cpu().numpy().reshape(-1)
                       for i in range(n_s4)]) if n_s4 else np.zeros((0, 24))
    s_arr = np.array([float(cand["s"][i]) for i in range(n_s4)], dtype=np.float64)
    sv_arr = np.array([float(cand["scale_valid"][i]) for i in range(n_s4)],
                      dtype=np.float64)
    valid_arr = np.stack([cand["valid_mask"][i].detach().cpu().numpy()
                          .reshape(-1).astype(bool) for i in range(n_s4)]
                         ) if n_s4 else np.zeros((0, 24), dtype=bool)
    m_minus_arr = np.stack([cand["m_minus"][i].detach().cpu().numpy()
                            .reshape(-1) for i in range(n_s4)]) if n_s4 else np.zeros((0, 24))
    m_plus_arr = np.stack([cand["m_plus"][i].detach().cpu().numpy()
                           .reshape(-1) for i in range(n_s4)]) if n_s4 else np.zeros((0, 24))
    hosts_arr = np.stack([h.reshape(-1) for h in s4_hosts]) if n_s4 else np.zeros((0, 24))
    target_arr = np.stack(s4_y) if n_s4 else np.zeros((0, 24))
    with np.errstate(divide="ignore", invalid="ignore"):
        zY_arr = np.where(s_arr[:, None] > 0,
                          np.arcsinh(target_arr / np.maximum(s_arr[:, None], EPS)),
                          0.0)

    # ---- per-arm chain evidence (deployed pipeline) ----
    arm = {}
    for name, mm_, lam in ARMS:
        ev = evs[name]
        nbr = [list(ev["neighbors"][i]) for i in range(n_s4)]
        props = ev["proposals"]
        pi_list = ev["pi"]
        a_hat = np.array([float(x) for x in ev["A_hat"]], dtype=np.float64)
        lcb = np.array([float(x) for x in ev["lcb"]], dtype=np.float64)
        acts = ev["final_action"]
        xf = np.stack([ev["x_final"][i].detach().cpu().numpy().reshape(-1)
                       for i in range(n_s4)]) if n_s4 else np.zeros((0, 24))
        exec_mask = np.array([a == "execute" for a in acts], dtype=bool)

        a_true = np.full(n_s4, np.nan)
        for i in range(n_s4):
            if sv_arr[i] > 0.5 and s_arr[i] > 0:
                a_true[i] = estimate_realized_A(
                    z0_arr[i], zY_arr[i],
                    np.asarray(pi_list[i], dtype=np.float64).ravel(), valid_arr[i])

        u3 = np.full(n_s4, np.nan)
        for i in range(n_s4):
            vm = valid_arr[i]
            if not vm.any():
                continue
            mh = float(np.mean(np.abs(hosts_arr[i][vm] - target_arr[i][vm])))
            mc = float(np.mean(np.abs(xf[i][vm] - target_arr[i][vm])))
            u3[i] = mh - mc

        mae = float(np.mean(np.abs(xf - target_arr))) if n_s4 else float("nan")
        pipe_nonempty = np.array([not _prop_empty(props[i])
                                  for i in range(n_s4)], dtype=bool)

        # offline replay of the deployed neighbor set (internal consistency)
        off_ahat = np.full(n_s4, np.nan)
        off_nonempty = np.zeros(n_s4, dtype=bool)
        for i in range(n_s4):
            if not nbr[i]:
                continue
            r = replay_chain(pipe.cavm_global, nbr[i], m_minus_arr[i],
                             m_plus_arr[i], z0_arr[i], zY_arr[i], valid_arr[i],
                             agg="mean")
            off_ahat[i] = r["A_hat"]
            off_nonempty[i] = r["proposal_nonempty"]

        arm[name] = {
            "nbr": nbr, "props": props, "pi": pi_list, "a_hat": a_hat,
            "lcb": lcb, "acts": acts, "xf": xf, "exec": exec_mask,
            "a_true": a_true, "u3": u3, "mae": mae,
            "proposal_nonempty": pipe_nonempty,
            "no_lcb_exec": pipe_nonempty & (a_hat > 0),
            "off_ahat_max_abs_diff": float(np.nanmax(np.abs(off_ahat - a_hat)))
                if n_s4 else 0.0,
            "off_nonempty_agree": float(np.mean(pipe_nonempty == off_nonempty))
                if n_s4 else 1.0,
        }

    # ---- per-query inter-arm metrics ----
    pqs = _per_query_metrics(arm, pipe.cavm_global, m_minus_arr, m_plus_arr,
                             valid_arr, w1_full, nw1, nctx, n_s4)

    # ---- four-way counterfactual decomposition (offline replay) ----
    decomp = _counterfactuals(arm, pipe.cavm_global, m_minus_arr, m_plus_arr,
                              z0_arr, zY_arr, valid_arr, hosts_arr, target_arr,
                              s_arr, w1_full, nw1, nctx, q_info["q"], n_s4)

    # ---- anchors vs saved cell JSON (hard stop on mismatch) ----
    checks = _anchor_arms(dataset_key, backbone, seed, arm)
    checks["off_ahat_e2_max_abs_diff"] = arm["E2_cavm_01"]["off_ahat_max_abs_diff"]
    checks["off_nonempty_agree_e2"] = arm["E2_cavm_01"]["off_nonempty_agree"]
    checks["n_s4"] = n_s4

    verdict = _cell_verdict(arm, pqs, decomp)

    # ---- emitted long-format rows (consumed by main) ----
    out = _build_out_rows(dataset_key, backbone, seed, s4_dates, mem_date_list,
                          arm, pqs, decomp, w1_full, nw1, nctx)

    return {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "selected_k": k, "q": q_info["q"], "S_d": S_d,
        "n_s4": n_s4, "n_mem": M,
        "n_exec_e0": int(arm["E0_w1"]["exec"].sum()),
        "n_exec_e2": int(arm["E2_cavm_01"]["exec"].sum()),
        "n_exec_e3": int(arm["E3_cavm_11"]["exec"].sum()),
        "arms": {name: _arm_summary(a) for name, a in arm.items()},
        "retrieval": pqs["cell_agg"],
        "decomposition": decomp["summary"],
        "checks": checks,
        "verdict": verdict,
        "_out_rows": out,
    }


# ------------------------------------------------------------ per-query -----
def _per_query_metrics(arm, mem, m_minus_arr, m_plus_arr, valid_arr,
                       w1_full, nw1, nctx, n_s4) -> dict:
    e0, e2, e3 = arm["E0_w1"], arm["E2_cavm_01"], arm["E3_cavm_11"]
    rows = []
    agg = {"jaccard_e0_e2": [], "jaccard_e0_e3": [],
           "n_eff": {"E0_w1": [], "E2_cavm_01": [], "E3_cavm_11": []},
           "margin_w1": {"E0_w1": [], "E2_cavm_01": [], "E3_cavm_11": []},
           "stab": {"E0_w1": [], "E2_cavm_01": [], "E3_cavm_11": []},
           "rank_corr_full": [], "rank_corr_shared": [],
           "sign_agree_fire": [], "fire_mask": []}
    for b in range(n_s4):
        n0 = set(e0["nbr"][b]); n2 = set(e2["nbr"][b]); n3 = set(e3["nbr"][b])
        jac_e0e2 = _jaccard(n0, n2)
        jac_e0e3 = _jaccard(n0, n3)
        jac_e2e3 = _jaccard(n2, n3)
        mw = {
            "E0_w1": _margin(w1_full[b][list(n0)]) if n0 else float("nan"),
            "E2_cavm_01": _margin(w1_full[b][list(n2)]) if n2 else float("nan"),
            "E3_cavm_11": _margin(w1_full[b][list(n3)]) if n3 else float("nan"),
        }
        n_eff = {}
        for name, lam in (("E0_w1", (1.0, 0.0)), ("E2_cavm_01", (0.0, 1.0)),
                          ("E3_cavm_11", (1.0, 1.0))):
            nbset = {"E0_w1": n0, "E2_cavm_01": n2, "E3_cavm_11": n3}[name]
            n_eff[name] = _ess(lam[0] * nw1[b][list(nbset)]
                               + lam[1] * nctx[b][list(nbset)]) if nbset else float("nan")
        d_e2 = nctx[b]                                   # E2 composite (λ_ctx=1)
        rho_full = _spearman(w1_full[b], d_e2)
        shared = sorted(n0 & n2)
        rho_shared = (_spearman(w1_full[b][shared], d_e2[shared])
                      if len(shared) >= 2 else float("nan"))
        sc0 = _sign_code_proposal(e0["props"][b])
        sc2 = _sign_code_proposal(e2["props"][b])
        sc3 = _sign_code_proposal(e3["props"][b])
        agree_e0e2 = 1 if sc0 == sc2 else 0
        fire = bool(e0["exec"][b] or e2["exec"][b] or e3["exec"][b])
        stab = {}
        for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11"):
            nb = {"E0_w1": e0, "E2_cavm_01": e2, "E3_cavm_11": e3}[name]["nbr"][b]
            stab[name] = (_neighbor_sign_stability(
                mem, nb, m_minus_arr[b], m_plus_arr[b], valid_arr[b])
                if nb else float("nan"))
        rows.append({
            "b": b, "jaccard_e0_e2": jac_e0e2, "jaccard_e0_e3": jac_e0e3,
            "jaccard_e2_e3": jac_e2e3,
            "n_eff_e0": n_eff["E0_w1"], "n_eff_e2": n_eff["E2_cavm_01"],
            "n_eff_e3": n_eff["E3_cavm_11"],
            "margin_w1_e0": mw["E0_w1"], "margin_w1_e2": mw["E2_cavm_01"],
            "margin_w1_e3": mw["E3_cavm_11"],
            "rank_corr_w1_vs_e2composite": rho_full,
            "rank_corr_shared": rho_shared,
            "sign_agree_e0_e2": agree_e0e2,
            "sign_agree_e0_e3": 1 if sc0 == sc3 else 0,
            "fire": int(fire),
            "stab_e0": stab["E0_w1"], "stab_e2": stab["E2_cavm_01"],
            "stab_e3": stab["E3_cavm_11"],
            "n_nbr_e0": len(n0), "n_nbr_e2": len(n2), "n_nbr_e3": len(n3),
            "exec_e0": int(e0["exec"][b]), "exec_e2": int(e2["exec"][b]),
            "exec_e3": int(e3["exec"][b]),
            "u3_e0": e0["u3"][b], "u3_e2": e2["u3"][b], "u3_e3": e3["u3"][b],
            "a_true_e0": e0["a_true"][b], "a_true_e2": e2["a_true"][b],
            "a_true_e3": e3["a_true"][b],
        })
        agg["jaccard_e0_e2"].append(jac_e0e2)
        agg["jaccard_e0_e3"].append(jac_e0e3)
        for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11"):
            agg["n_eff"][name].append(n_eff[name])
            agg["margin_w1"][name].append(mw[name])
            agg["stab"][name].append(stab[name])
        agg["rank_corr_full"].append(rho_full)
        agg["rank_corr_shared"].append(rho_shared)
        if fire:
            agg["sign_agree_fire"].append(agree_e0e2)
        agg["fire_mask"].append(int(fire))

    cell_agg = {
        "median_jaccard_e0_e2": _fmed(agg["jaccard_e0_e2"]),
        "median_jaccard_e0_e3": _fmed(agg["jaccard_e0_e3"]),
        "mean_jaccard_e0_e2": (float(np.mean(agg["jaccard_e0_e2"]))
                               if agg["jaccard_e0_e2"] else None),
        "mean_jaccard_e0_e3": (float(np.mean(agg["jaccard_e0_e3"]))
                               if agg["jaccard_e0_e3"] else None),
        "median_n_eff": {name: _fmed(agg["n_eff"][name])
                         for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11")},
        "median_margin_w1": {name: _fmed(agg["margin_w1"][name])
                             for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11")},
        "median_rank_corr_w1_vs_e2composite": _fmed(agg["rank_corr_full"]),
        "median_rank_corr_shared": _fmed(agg["rank_corr_shared"]),
        "action_sign_agreement_e0_e2_fire": (float(np.mean(agg["sign_agree_fire"]))
                                             if agg["sign_agree_fire"] else None),
        "n_fire_days": int(sum(agg["fire_mask"])),
        "median_stab": {name: _fmed(agg["stab"][name])
                        for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11")},
    }
    return {"rows": rows, "cell_agg": cell_agg}


# ------------------------------------------------------------ counterfacts ----
def _counterfactuals(arm, mem, m_minus_arr, m_plus_arr, z0_arr, zY_arr,
                     valid_arr, hosts_arr, target_arr, s_arr, w1_full, nw1,
                     nctx, q, n_s4) -> dict:
    e0, e2 = arm["E0_w1"], arm["E2_cavm_01"]
    rows = []
    agg = {key: {"n_exec": 0, "sum_A_true": 0.0, "sum_u3": 0.0,
                 "action_flip": 0} for key in (
        "baseline_e2", "delta_retrieval_set", "weighted", "down_only",
        "median", "trimmed", "no_lcb", "lcb_q_half", "lcb_q_1p5")}
    for b in range(n_s4):
        n0 = e0["nbr"][b]; n2 = e2["nbr"][b]
        mm, mp = m_minus_arr[b], m_plus_arr[b]
        vm = valid_arr[b]
        kw = dict(mm=mm, mp=mp, z0_q=z0_arr[b], zY_q=zY_arr[b], vm=vm,
                  s_q=s_arr[b], host_q=hosts_arr[b], target_q=target_arr[b])
        base = _replay_signed(mem, n2, agg="mean", q=q, **kw)
        cf_set = _replay_signed(mem, n0, agg="mean", q=q, **kw)
        if n2:
            D2 = nctx[b][n2]                            # E2 composite dists
            tau = float(np.median(D2))
            wts = np.exp(-D2 / (tau + EPS)); wts = wts / (wts.sum() + EPS)
            cf_w = _replay_signed(mem, n2, agg="weighted", weights=wts, q=q, **kw)
            cf_down = _replay_signed(mem, n2, agg="mean", down_only=True, q=q, **kw)
            cf_med = _replay_signed(mem, n2, agg="median", q=q, **kw)
            cf_trim = _replay_signed(mem, n2, agg="trimmed", q=q, **kw)
            cf_nolcb = _replay_signed(mem, n2, agg="mean", lcb_mode="none", q=q, **kw)
            cf_qh = _replay_signed(mem, n2, agg="mean", q=q * 0.5, **kw)
            cf_q15 = _replay_signed(mem, n2, agg="mean", q=q * 1.5, **kw)
        else:
            cf_w = cf_down = cf_med = cf_trim = base
            cf_nolcb = cf_qh = cf_q15 = base

        _acc(agg["baseline_e2"], base)
        _acc(agg["delta_retrieval_set"], cf_set)
        if base["execute"] != cf_set["execute"]:
            agg["delta_retrieval_set"]["action_flip"] += 1
        _acc(agg["weighted"], cf_w)
        _acc(agg["down_only"], cf_down)
        _acc(agg["median"], cf_med)
        _acc(agg["trimmed"], cf_trim)
        _acc(agg["no_lcb"], cf_nolcb)
        _acc(agg["lcb_q_half"], cf_qh)
        _acc(agg["lcb_q_1p5"], cf_q15)

        rows.append({
            "b": b, "base_exec": int(base["execute"]),
            "base_A_true": base["A_true"], "base_u3": base["u3"],
            "retrieval_set_exec": int(cf_set["execute"]),
            "retrieval_set_A_true": cf_set["A_true"],
            "retrieval_set_u3": cf_set["u3"],
            "weighted_exec": int(cf_w["execute"]),
            "weighted_A_true": cf_w["A_true"],
            "down_only_exec": int(cf_down["execute"]),
            "down_only_A_true": cf_down["A_true"],
            "median_exec": int(cf_med["execute"]),
            "median_A_true": cf_med["A_true"],
            "trimmed_exec": int(cf_trim["execute"]),
            "trimmed_A_true": cf_trim["A_true"],
            "no_lcb_exec": int(cf_nolcb["execute"]),
            "no_lcb_A_true": cf_nolcb["A_true"],
            "lcb_q_half_exec": int(cf_qh["execute"]),
            "lcb_q_1p5_exec": int(cf_q15["execute"]),
        })

    summary = {}
    for key, d in agg.items():
        s = {"n_exec": d["n_exec"]}
        if d["n_exec"]:
            s["exec_mean_A_true"] = d["sum_A_true"] / d["n_exec"]
            s["exec_mean_u3"] = d["sum_u3"] / d["n_exec"]
        if key == "delta_retrieval_set":
            s["action_flip_days"] = d["action_flip"]
        summary[key] = s
    return {"rows": rows, "summary": summary}


def _acc(d, r):
    d["n_exec"] += int(r["execute"])
    if r["execute"]:
        if np.isfinite(r["A_true"]):
            d["sum_A_true"] += r["A_true"]
        if np.isfinite(r["u3"]):
            d["sum_u3"] += r["u3"]


# ------------------------------------------------------------ anchors ----
def _anchor_arms(dataset_key: str, backbone: str, seed: int, arm: dict) -> dict:
    saved_path = P2_OUT / f"{dataset_key}_{backbone}_s{seed}.json"
    if not saved_path.exists():
        return {"anchored": False, "reason": "no_saved_json"}
    saved = json.load(open(saved_path, encoding="utf-8"))
    checks = {}
    ok = True
    for name, _, _ in ARMS:
        jm = saved["modes"].get(name, {})
        j_mae = jm.get("MAE")
        j_nx = jm.get("n_execute")
        j_at = jm.get("A_true", [])
        j_nbr = jm.get("neighbors", [])
        a = arm[name]
        dmae = abs(a["mae"] - _jnum(j_mae)) if j_mae is not None else None
        mae_ok = (j_mae is None) or (dmae <= 1e-5)
        nx_ok = (j_nx is None) or (a["exec"].sum() == int(j_nx))
        at_max, at_ok = 0.0, True
        for i in range(len(a["a_true"])):
            if i >= len(j_at):
                continue
            jv, mv = _jnum(j_at[i]), a["a_true"][i]
            if np.isnan(mv):
                same = bool(np.isnan(jv))
            else:
                same = bool(np.isfinite(jv) and abs(mv - jv) <= 1e-6)
            if not same:
                at_ok = False
            elif not np.isnan(mv):
                at_max = max(at_max, abs(mv - jv))
        nbr_ok = True
        for i in range(len(a["nbr"])):
            if i >= len(j_nbr):
                continue
            if a["nbr"][i] != [int(x) for x in j_nbr[i]]:
                nbr_ok = False
        checks[f"{name}_mae_diff"] = round(dmae, 8) if dmae is not None else None
        checks[f"{name}_mae_ok"] = bool(mae_ok)
        checks[f"{name}_n_exec_ok"] = bool(nx_ok)
        checks[f"{name}_a_true_max_diff"] = round(at_max, 8)
        checks[f"{name}_a_true_ok"] = bool(at_ok)
        checks[f"{name}_neighbors_ok"] = bool(nbr_ok)
        ok = ok and mae_ok and nx_ok and at_ok and nbr_ok
    checks["anchored"] = bool(ok)
    return checks


# ------------------------------------------------------------ summaries -----
def _arm_summary(a: dict) -> dict:
    exe = a["exec"]
    at = a["a_true"]
    at_exe = at[exe]
    u3_exe = a["u3"][exe]
    n = len(exe)
    out = {
        "n_exec": int(exe.sum()),
        "execute_rate": round(float(exe.mean()), 4) if n else None,
        "MAE": round(a["mae"], 5),
        "proposal_empty_rate": (round(float((~a["proposal_nonempty"]).mean()), 4)
                                if n else None),
        "no_lcb_exec_rate": round(float(a["no_lcb_exec"].mean()), 4) if n else None,
        "lcb_blocked_days": int((a["no_lcb_exec"] & ~exe).sum()),
        "mean_A_true": (round(float(np.nanmean(at)), 5)
                        if np.isfinite(at).any() else None),
        "exec_mean_A_true": (round(float(np.nanmean(at_exe)), 5)
                             if at_exe.size and np.isfinite(at_exe).any() else None),
        "exec_harm_rate": (round(float(np.mean(at_exe < 0)), 4)
                           if at_exe.size else None),
        "exec_mean_u3": (round(float(np.nanmean(u3_exe)), 4)
                         if u3_exe.size and np.isfinite(u3_exe).any() else None),
        "mean_lcb": (round(float(np.nanmean(a["lcb"])), 5)
                     if n and np.isfinite(a["lcb"]).any() else None),
    }
    return out


def _cell_verdict(arm: dict, pqs: dict, decomp: dict) -> dict:
    e0, e2 = arm["E0_w1"], arm["E2_cavm_01"]
    n = len(e0["proposal_nonempty"])
    prop_empty_e0 = float((~e0["proposal_nonempty"]).mean()) if n else 1.0
    prop_empty_e2 = float((~e2["proposal_nonempty"]).mean()) if n else 1.0
    ca = pqs["cell_agg"]
    med_jac = ca["median_jaccard_e0_e2"]
    act_agree = ca["action_sign_agreement_e0_e2_fire"]
    nolcb_at = e2["a_true"][e2["no_lcb_exec"]]
    no_lcb_mean_At = (float(np.nanmean(nolcb_at))
                      if e2["no_lcb_exec"].any() and np.isfinite(nolcb_at).any()
                      else None)
    lcb_exec_rate = float(e2["exec"].mean()) if n else 0.0
    lcb_blocked = int((e2["no_lcb_exec"] & ~e2["exec"]).sum())

    # decomposition leader among the four non-degenerate content axes
    base_n = decomp["summary"]["baseline_e2"]["n_exec"]
    shifts = {k: decomp["summary"][k]["n_exec"] - base_n
              for k in ("delta_retrieval_set", "weighted", "median", "trimmed")}
    leader = max(shifts, key=lambda k: abs(shifts[k])) if any(shifts.values()) else "none"

    # primary category (design §4.4 / plan §5, quantized; gray-zone overlap
    # [0.5, 0.7) is resolved by action-sign agreement, same discriminator as
    # the low-overlap branch)
    if prop_empty_e0 >= DATA_SUPPORT_EMPTY and prop_empty_e2 >= DATA_SUPPORT_EMPTY:
        cat = "DATA_SUPPORT"
    elif (no_lcb_mean_At is not None and no_lcb_mean_At > 0
          and lcb_exec_rate == 0):
        cat = "CALIBRATION"
    elif med_jac is not None and med_jac >= K_WEIGHT_JACC:
        cat = "K_WEIGHT"
    elif act_agree is None:
        cat = "MIXED"
    elif act_agree >= SIGN_AGREE_HIGH:
        cat = "PROPOSAL"
    elif act_agree <= SIGN_AGREE_LOW:
        cat = "KEY"
    else:
        cat = "MIXED"
    return {
        "category": cat,
        "decomposition_leader": leader,
        "evidence": {
            "proposal_empty_e0": round(prop_empty_e0, 4),
            "proposal_empty_e2": round(prop_empty_e2, 4),
            "median_jaccard_e0_e2": round(med_jac, 4) if med_jac is not None else None,
            "action_sign_agreement_fire": (round(act_agree, 4)
                                           if act_agree is not None else None),
            "no_lcb_exec_mean_A_true": (round(no_lcb_mean_At, 5)
                                        if no_lcb_mean_At is not None else None),
            "with_lcb_exec_rate": round(lcb_exec_rate, 4),
            "lcb_blocked_days": lcb_blocked,
        },
    }


# ------------------------------------------------------------ emitted rows ----
def _build_out_rows(dataset_key, backbone, seed, s4_dates, mem_date_list,
                    arm, pqs, decomp, w1_full, nw1, nctx) -> dict:
    cell = f"{dataset_key}:{backbone}:s{seed}"
    pq_rows = []
    for r in pqs["rows"]:
        pq_rows.append({"cell": cell, "market": dataset_key, "host": backbone,
                        "seed": seed, "date": s4_dates[r["b"]], **r})
    # long neighbor table: one row per (query, arm, neighbor)
    nbr_rows = []
    nbr_sets = {name: [set(arm[name]["nbr"][b]) for b in range(len(s4_dates))]
                for name in ("E0_w1", "E2_cavm_01", "E3_cavm_11")}
    for b in range(len(s4_dates)):
        for name, lam in (("E0_w1", (1.0, 0.0)), ("E2_cavm_01", (0.0, 1.0)),
                          ("E3_cavm_11", (1.0, 1.0))):
            nbr = arm[name]["nbr"][b]
            for rank, idx in enumerate(nbr):
                nbr_rows.append({
                    "cell": cell, "market": dataset_key, "host": backbone,
                    "seed": seed, "date": s4_dates[b], "arm": name,
                    "rank": rank, "neighbor_idx": idx,
                    "neighbor_date": mem_date_list[idx],
                    "dist_raw_w1": float(w1_full[b, idx]),
                    "norm_w1": float(nw1[b, idx]),
                    "norm_ctx": float(nctx[b, idx]),
                    "dist_composite": float(lam[0] * nw1[b, idx]
                                            + lam[1] * nctx[b, idx]),
                    "in_e0": int(idx in nbr_sets["E0_w1"][b]),
                    "in_e2": int(idx in nbr_sets["E2_cavm_01"][b]),
                    "in_e3": int(idx in nbr_sets["E3_cavm_11"][b]),
                })
    cf_rows = []
    for r in decomp["rows"]:
        cf_rows.append({"cell": cell, "market": dataset_key, "host": backbone,
                        "seed": seed, "date": s4_dates[r["b"]], **r})
    return {"pq": pq_rows, "nbr": nbr_rows, "cf": cf_rows}


# ------------------------------------------------------------------ main ----
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _matrix_row(cell: dict) -> dict:
    r = cell["retrieval"]
    v = cell["verdict"]["evidence"]
    d = cell["decomposition"]
    return {
        "cell": f"{cell['dataset']}:{cell['backbone']}:s{cell['seed']}",
        "market": cell["dataset"], "host": cell["backbone"], "seed": cell["seed"],
        "k": cell["selected_k"], "q": round(cell["q"], 6),
        "n_s4": cell["n_s4"], "n_mem": cell["n_mem"],
        "n_exec_e0": cell["n_exec_e0"], "n_exec_e2": cell["n_exec_e2"],
        "n_exec_e3": cell["n_exec_e3"],
        "mae_e0": cell["arms"]["E0_w1"]["MAE"],
        "mae_e2": cell["arms"]["E2_cavm_01"]["MAE"],
        "mae_e3": cell["arms"]["E3_cavm_11"]["MAE"],
        "exec_mean_u3_e0": cell["arms"]["E0_w1"]["exec_mean_u3"],
        "exec_mean_u3_e2": cell["arms"]["E2_cavm_01"]["exec_mean_u3"],
        "med_jaccard_e0_e2": r["median_jaccard_e0_e2"],
        "mean_jaccard_e0_e2": r["mean_jaccard_e0_e2"],
        "med_n_eff_e0": (r["median_n_eff"] or {}).get("E0_w1"),
        "med_n_eff_e2": (r["median_n_eff"] or {}).get("E2_cavm_01"),
        "med_rank_corr": r["median_rank_corr_w1_vs_e2composite"],
        "sign_agree_fire": r["action_sign_agreement_e0_e2_fire"],
        "n_fire_days": r["n_fire_days"],
        "med_stab_e2": (r["median_stab"] or {}).get("E2_cavm_01"),
        "prop_empty_e0": v["proposal_empty_e0"],
        "prop_empty_e2": v["proposal_empty_e2"],
        "cf_retrieval_set_exec": d["delta_retrieval_set"]["n_exec"],
        "cf_retrieval_set_u3": d["delta_retrieval_set"].get("exec_mean_u3"),
        "cf_weighted_exec": d["weighted"]["n_exec"],
        "cf_down_only_exec": d["down_only"]["n_exec"],
        "cf_no_lcb_exec": d["no_lcb"]["n_exec"],
        "cf_lcb_q_half_exec": d["lcb_q_half"]["n_exec"],
        "cf_lcb_q_1p5_exec": d["lcb_q_1p5"]["n_exec"],
        "verdict": cell["verdict"]["category"],
        "decomp_leader": cell["verdict"].get("decomposition_leader"),
        "lcb_blocked_days_e2": v["lcb_blocked_days"],
        "anchored": cell["checks"].get("anchored"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=str, default=None,
                    help="single cell 'MARKET:HOST:SEED' for sanity")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="re-build CSVs/verdict/figs from saved per-cell JSONs "
                         "(no recompute)")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    log = []

    def L(msg):
        log.append(msg)
        print(msg, flush=True)

    if args.aggregate:
        cells_store, failures = [], []
        for p in sorted(out_dir.glob("decomp_*.json")):
            with open(p, encoding="utf-8") as f:
                cells_store.append(json.load(f))
        fail_path = out_dir / "failures.json"
        if fail_path.exists():
            with open(fail_path, encoding="utf-8") as f:
                failures = json.load(f)
        L(f"[decomp] aggregate: {len(cells_store)} cells loaded from {out_dir}")
        _aggregate(out_dir, cells_store, failures, log, args)
        return

    if args.cell:
        mk, bb, sd = args.cell.split(":")
        cell = run_decomp(mk, bb, int(sd))
        with open(out_dir / f"decomp_{mk}_{bb}_s{sd}.json", "w",
                  encoding="utf-8") as f:
            json.dump(cell, f, ensure_ascii=False, indent=2, default=_jdefault)
        print(json.dumps(cell["checks"], ensure_ascii=False, indent=2))
        print(json.dumps(cell["verdict"], ensure_ascii=False, indent=2))
        print(f"sanity {mk}:{bb}:s{sd} anchored={cell['checks'].get('anchored')} "
              f"cat={cell['verdict']['category']} "
              f"leader={cell['verdict'].get('decomposition_leader')} "
              f"E2 n_exec={cell['n_exec_e2']} MAE={cell['arms']['E2_cavm_01']['MAE']}")
        return

    cells_store, failures = [], []
    for mk in TARGETS:
        for bb in HOSTS:
            for sd in SEEDS:
                tag = f"{mk}:{bb}:s{sd}"
                L(f"[decomp] {tag} ...")
                try:
                    cell = run_decomp(mk, bb, sd)
                except Exception as e:
                    failures.append({"cell": tag, "error": str(e)})
                    L(f"    FAIL {tag}: {e!r}")
                    continue
                if not cell["checks"].get("anchored", False):
                    failures.append({"cell": tag, "error": "ANCHOR MISMATCH",
                                     "checks": cell["checks"]})
                    L(f"    ANCHOR FAIL {tag}: {cell['checks']}")
                    continue
                with open(out_dir / f"decomp_{mk}_{bb}_s{sd}.json", "w",
                          encoding="utf-8") as f:
                    json.dump(cell, f, ensure_ascii=False, indent=2,
                              default=_jdefault)
                cells_store.append(cell)
                v = cell["verdict"]["evidence"]
                L(f"    anchored k={cell['selected_k']} "
                  f"n_exec E0/E2/E3={cell['n_exec_e0']}/{cell['n_exec_e2']}/{cell['n_exec_e3']} "
                  f"med_jac={cell['retrieval']['median_jaccard_e0_e2']} "
                  f"agree_fire={v['action_sign_agreement_fire']} "
                  f"cat={cell['verdict']['category']}")
    if failures:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2, default=_jdefault)
        L(f"[decomp] {len(failures)} cells failed / hard-stopped")
    if not cells_store:
        L("[decomp] no successful cells — aborting")
        return
    _aggregate(out_dir, cells_store, failures, log, args)


def _aggregate(out_dir: Path, cells_store: list[dict], failures: list[dict],
               log: list[str], args) -> None:
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    def L(msg):
        log.append(msg)
        print(msg, flush=True)

    pq_rows, nbr_rows, cf_rows, matrix_rows = [], [], [], []
    for cell in cells_store:
        pq_rows.extend(cell["_out_rows"]["pq"])
        nbr_rows.extend(cell["_out_rows"]["nbr"])
        cf_rows.extend(cell["_out_rows"]["cf"])
        matrix_rows.append(_matrix_row(cell))
    _write_csv(out_dir / "per_query_metrics.csv", pq_rows)
    _write_csv(out_dir / "neighbor_diff_long.csv", nbr_rows)
    _write_csv(out_dir / "counterfactual_long.csv", cf_rows)
    _write_csv(out_dir / "matrix.csv", matrix_rows)

    verdict = _build_verdict(cells_store, matrix_rows, failures)
    with open(out_dir / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    with open(out_dir / "matrix_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")

    if not args.no_figs:
        try:
            _make_figs(cells_store, matrix_rows, figs_dir)
            L("[decomp] figures written to " + str(figs_dir))
        except Exception as e:
            L(f"[decomp] figures FAILED: {e!r}")
    print("\n===== phase6.0 decomp verdict =====")
    print(json.dumps(verdict["verdict"], ensure_ascii=False, indent=2))
    print(f"\n[decomp] artifacts: {out_dir}")


def _build_verdict(cells_store: list[dict], matrix_rows: list[dict],
                   failures: list[dict]) -> dict:
    per_market = {}
    for mk in TARGETS:
        cells = [c for c in cells_store if c["dataset"] == mk]
        counts = {}
        for c in cells:
            counts[c["verdict"]["category"]] = counts.get(
                c["verdict"]["category"], 0) + 1
        rj = [c["retrieval"] for c in cells]
        per_market[mk] = {
            "n_cells": len(cells),
            "category_counts": counts,
            "median_jaccard_e0_e2": _fmed([r["median_jaccard_e0_e2"]
                                           for r in rj if r.get("median_jaccard_e0_e2") is not None]),
            "sign_agree_fire_mean": (float(np.mean([r["action_sign_agreement_e0_e2_fire"]
                                                    for r in rj if r.get("action_sign_agreement_e0_e2_fire") is not None]))
                                     if any(r.get("action_sign_agreement_e0_e2_fire") is not None
                                            for r in rj) else None),
            "median_n_exec_e2": int(np.median([c["n_exec_e2"] for c in cells])),
            "median_n_exec_e0": int(np.median([c["n_exec_e0"] for c in cells])),
        }

    total = {"n_cells": len(cells_store),
             "category_counts": {},
             "median_jaccard_e0_e2": _fmed([r["med_jaccard_e0_e2"]
                                            for r in matrix_rows
                                            if r["med_jaccard_e0_e2"] is not None]),
             "sign_agree_fire_mean": (float(np.mean([r["sign_agree_fire"]
                                                     for r in matrix_rows
                                                     if r["sign_agree_fire"] is not None]))
                                      if any(r["sign_agree_fire"] is not None
                                             for r in matrix_rows) else None)}
    for c in cells_store:
        total["category_counts"][c["verdict"]["category"]] = total[
            "category_counts"].get(c["verdict"]["category"], 0) + 1

    return {
        "protocol": ("hch_v2_phase6_retrieval_reliability_generalization_"
                     "research_plan_v0.1 §5 Phase 6.0"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": R._git_head(),
        "matrix": {"targets": TARGETS, "hosts": HOSTS, "seeds": SEEDS,
                   "n_cells": len(cells_store)},
        "failures": failures,
        "per_market": per_market,
        "total": total,
        "verdict": _verdict_note(per_market, total),
    }


def _verdict_note(per_market: dict, total: dict) -> dict:
    """Descriptive cross-cell gate note (§5: per-market, not one macro mean).

    The categories are per-cell; the cross-cell reading separates the markets
    so LAGO_NP's action-empty boundary role is never conflated with a
    retrieval failure, and the Shandong host split is reported explicitly.
    """
    cats = total["category_counts"]
    dominant = (max(cats, key=cats.get) if cats else "OTHER")
    gate = {
        "LAGO_DE": bool(per_market.get("LAGO_DE", {}).get("n_cells", 0) > 0),
        "shandong_DA": bool(per_market.get("shandong_DA", {}).get("n_cells", 0) > 0),
    }
    return {
        "category_counts": cats,
        "dominant_category": dominant,
        "per_market": {mk: {"category_counts": d["category_counts"],
                            "median_jaccard_e0_e2": d["median_jaccard_e0_e2"]}
                       for mk, d in per_market.items()},
        "note": ("Per-cell category + per-market median Jaccard and fire "
                 "agreement are in verdict.per_market. Cross-cell verdict is "
                 "described in the report; the single dominant category here "
                 "is indicative only, not a gate. Gate R (proceed to Phase 6.1 "
                 "APR) is decided per market in the report."),
    }


# ---------------------------------------------------------------- figures ----
def _make_figs(cells_store: list[dict], matrix_rows: list[dict],
               figs_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1: raw-W1 distance of selected neighbors per arm ----
    data = {"E0_w1": [], "E2_cavm_01": [], "E3_cavm_11": []}
    for cell in cells_store:
        for r in cell["_out_rows"]["nbr"]:
            d = r["dist_raw_w1"]
            if np.isfinite(d):
                data[r["arm"]].append(d)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([data[a] for a in ("E0_w1", "E2_cavm_01", "E3_cavm_11")],
               labels=("E0 w1", "E2 ctx", "E3 mix"))
    ax.set_ylabel("raw W1 distance of selected neighbors")
    ax.set_title("Fig1: selected-neighbor raw W1 distance (pooled cells)")
    fig.tight_layout(); fig.savefig(figs_dir / "fig1_dist.png", dpi=110)
    plt.close(fig)

    # ---- 2: per-query Jaccard E0∩E2 ----
    jac = []
    for cell in cells_store:
        jac.extend([r["jaccard_e0_e2"] for r in cell["_out_rows"]["pq"]
                    if np.isfinite(r["jaccard_e0_e2"])])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(jac, bins=30)
    ax.axvline(0.5, color="r", ls="--", lw=1, label="LOW_JACC")
    ax.axvline(0.7, color="g", ls="--", lw=1, label="K_WEIGHT")
    ax.set_xlabel("Jaccard(E0 neighbors ∩ E2 neighbors)")
    ax.set_ylabel("query-days")
    ax.legend()
    ax.set_title("Fig2: E0/E2 neighbor-set overlap (pooled cells)")
    fig.tight_layout(); fig.savefig(figs_dir / "fig2_overlap.png", dpi=110)
    plt.close(fig)

    # ---- 3: n_eff (ESS) per arm ----
    neff = {"E0_w1": [], "E2_cavm_01": [], "E3_cavm_11": []}
    for cell in cells_store:
        for r in cell["_out_rows"]["pq"]:
            for a, key in (("E0_w1", "n_eff_e0"), ("E2_cavm_01", "n_eff_e2"),
                           ("E3_cavm_11", "n_eff_e3")):
                if np.isfinite(r[key]):
                    neff[a].append(r[key])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([neff[a] for a in ("E0_w1", "E2_cavm_01", "E3_cavm_11")],
               labels=("E0 w1", "E2 ctx", "E3 mix"))
    ax.set_ylabel("ESS (n_eff) of selected neighbors")
    ax.set_title("Fig3: neighbor effective sample size per arm")
    fig.tight_layout(); fig.savefig(figs_dir / "fig3_neff.png", dpi=110)
    plt.close(fig)

    # ---- 4: action-sign agreement (fire-restricted) per market ----
    fig, ax = plt.subplots(figsize=(7, 4))
    markets = list(TARGETS)
    agree = []
    for mk in markets:
        vals = [r["sign_agree_fire"] for r in matrix_rows
                if r["market"] == mk and r["sign_agree_fire"] is not None]
        agree.append(float(np.mean(vals)) if vals else 0.0)
    ax.bar(markets, agree)
    ax.axhline(0.6, color="g", ls="--", lw=1, label="SIGN_AGREE_HIGH")
    ax.axhline(0.4, color="r", ls="--", lw=1, label="SIGN_AGREE_LOW")
    ax.set_ylabel("mean fire-restricted action-sign agreement E0 vs E2")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Fig4: action-sign agreement by market")
    fig.tight_layout(); fig.savefig(figs_dir / "fig4_sign_agreement.png", dpi=110)
    plt.close(fig)

    # ---- 5: proposal -> A_hat>0 -> LCB -> execute funnel per arm ----
    funnel = {a: {"n_days": 0, "prop": 0, "no_lcb": 0, "exec": 0}
              for a in ("E0_w1", "E2_cavm_01", "E3_cavm_11")}
    for cell in cells_store:
        for a, key in (("E0_w1", "E0_w1"), ("E2_cavm_01", "E2_cavm_01"),
                       ("E3_cavm_11", "E3_cavm_11")):
            s = cell["arms"][key]
            funnel[a]["n_days"] += cell["n_s4"]
            funnel[a]["prop"] += int(round(cell["n_s4"]
                                           * (1 - (s["proposal_empty_rate"] or 0))))
            funnel[a]["no_lcb"] += int(round(cell["n_s4"]
                                             * (s["no_lcb_exec_rate"] or 0)))
            funnel[a]["exec"] += s["n_exec"]
    fig, ax = plt.subplots(figsize=(7, 4))
    stages = ["n_days", "prop", "no_lcb", "exec"]
    x = np.arange(len(stages))
    width = 0.25
    for j, a in enumerate(("E0_w1", "E2_cavm_01", "E3_cavm_11")):
        ax.bar(x + (j - 1) * width,
               [funnel[a][s] for s in stages], width, label=a)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.legend()
    ax.set_title("Fig5: proposal->A_hat>0->LCB->execute funnel (pooled counts)")
    fig.tight_layout(); fig.savefig(figs_dir / "fig5_funnel.png", dpi=110)
    plt.close(fig)

    # ---- 6: E2 - E0 per-day raw gain on executed days (per market) ----
    fig, axes = plt.subplots(1, len(markets), figsize=(11, 3.5), sharey=True)
    for ax, mk in zip(axes, markets):
        ds = []
        for cell in cells_store:
            if cell["dataset"] != mk:
                continue
            for r in cell["_out_rows"]["pq"]:
                if (r["exec_e0"] or r["exec_e2"]) and np.isfinite(r["u3_e2"]) \
                        and np.isfinite(r["u3_e0"]):
                    ds.append(r["u3_e2"] - r["u3_e0"])
        ax.hist(ds, bins=30)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(mk)
        ax.set_xlabel("u3(E2) - u3(E0)")
    fig.suptitle("Fig6: per-day realized raw-gain delta E2 vs E0 (executed days)")
    fig.tight_layout(); fig.savefig(figs_dir / "fig6_gain_delta.png", dpi=110)
    plt.close(fig)


def _cand_view(cand, b):
    """Single-day candidate view (mirrors hch_v2_pipeline._day_view)."""
    return {
        "w_minus": cand["w_minus"][b:b + 1],
        "w_zero": cand["w_zero"][b:b + 1],
        "w_plus": cand["w_plus"][b:b + 1],
        "m_minus": cand["m_minus"][b:b + 1],
        "m_plus": cand["m_plus"][b:b + 1],
        "valid_mask": cand["valid_mask"][b:b + 1],
    }


if __name__ == "__main__":
    main()
