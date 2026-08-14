"""Phase4 P2 extension — multi-market × multi-seed E0-E3 gate (design §9.2).

The P2 probe (p2_cavm_experiment.py) was single-market (LAGO_DE) single-seed.
This script runs the SAME controlled chain across the §4.2 representative target
markets × 3 hosts × 3 seeds to satisfy the §9.2 first-phase pass criteria:

  1. >= 2 distinct market types direction-stable improvement
  2. 3 seeds, direction not carried by a single seed
  3. LAGO_NP no reproducible significant regression
  4. action value and point prediction not opposite
  5. (streaming-only downgrade — P3 already showed local is append-only)
  6. not point-only: action track must be reported independently

Protocol (unchanged from P2):
  - host comes from the shared cache (R.prepare_domain; R1B/Stage-2C convention:
    host shared across seeds, seed varies the candidate head training).
  - candidate head trained per market on its OWN S2T/S2V (controlled setting;
    the ONLY variable between E0 and E2/E3 is the retrieval key, §3.1).
  - k selected on S3-M by W1 forward validation; lambda FIXED grid {0,1}/{1,1},
    NOT tuned on S4 (red line).
  - S4 target-free (predict_s4); A_true computed offline with the SAME
    estimate_realized_A after labels reveal.

Markets (§4.2): LAGO_DE (EU negative price, probe continuity), LAGO_NP (mild
market — §9.2-3 critical), shandong_DA (CN real negative price + spikes),
shaanxi_RT (strong host), gansu_DA (thin sample). All host caches exist.

Outputs (results/phase4/p2_ext/): per-cell JSON, matrix.csv, summary.json
with the §9.2 verdict (descriptive, honest).
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

import r1a_run as R                                    # noqa: E402
from hch_v2_pipeline import HCHV2UniversalPipeline     # noqa: E402
from query_replay import estimate_realized_A           # noqa: E402
from p2_cavm_experiment import (                       # noqa: E402
    pd_date, build_core_context, _run_candidate, _scale_z0,
)

OUT = HERE / "results" / "phase4" / "p2_ext"
OUT.mkdir(parents=True, exist_ok=True)

MARKETS = ["LAGO_DE", "LAGO_NP", "shandong_DA", "shaanxi_RT", "gansu_DA"]
HOSTS = ["Linear", "MLP", "PatchTST"]
SEEDS = [0, 1, 2]
D_CORE = 13
D_MODEL = 32


# ------------------------------------------------------------- chain build ----
def run(dataset_key: str, backbone: str, seed: int,
        k_validation_frac: float = 0.25,
        k_candidates: tuple = (5, 10, 20), alpha: float = 0.10,
        epochs: int = 8, patience: int = 4) -> dict:
    """One controlled E0-E3 chain for (market, host, seed).

    Data front-end = R.prepare_domain (shared host cache, province-aware);
    downstream = P2 probe chain (build_core_context/_run_candidate/predict_s4).
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

    pipe = HCHV2UniversalPipeline(d_core_context=D_CORE, d_model=D_MODEL,
                                  alpha=alpha, k=None, seed=seed,
                                  memory_mode="cavm")
    pipe.fit_s1_reference(info.s1_z0, info.s1_hours)
    pipe.fit_s1_signature(info.s1_z0, info.s1_hours)

    det_broadcast = None
    if pipe._domain_det is not None:
        det_b = torch.tensor(np.asarray(pipe._domain_det, dtype=np.float32),
                             dtype=torch.float32).unsqueeze(0)
        det_broadcast = det_b

    def _s2_batches_for(split):
        batches = []
        for d in sorted(exp.dates_in_split(split)):
            idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
            if len(idxs) != 24:
                continue
            host_day = yhat_full[idxs].astype(np.float64)
            if not np.isfinite(host_day).all():
                continue
            hours = ts.iloc[idxs].dt.hour.values
            ctx = build_core_context(host_day, hours, pipe, z0_full, s_full,
                                     y_full, idxs)
            batches.append((
                torch.tensor(host_day.reshape(1, 24, 1), dtype=torch.float32),
                torch.tensor(ctx.reshape(1, 24, -1), dtype=torch.float32),
                torch.tensor(y_full[idxs].astype(np.float64).reshape(1, 24, 1),
                             dtype=torch.float32),
                torch.ones(1, 24),
                det_broadcast.clone(),
            ))
        return batches

    s2_batches = _s2_batches_for("S2T")
    s2v_batches = _s2_batches_for("S2V")
    s2_loss = pipe.train_candidate_s2(s2_batches, s2v_batches=s2v_batches,
                                      epochs=epochs, lr=1e-3, patience=patience)

    # ---- S3-M: memory prefix + k-validation suffix ----
    s3m_all = sorted(exp.dates_in_s3m())
    n_mem = int(len(s3m_all) * (1.0 - k_validation_frac))
    mem_dates, val_dates = s3m_all[:n_mem], s3m_all[n_mem:]
    det_day = pipe._domain_det

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
    domain_det = None
    if pipe._domain_det is not None:
        det_t = torch.tensor(np.asarray(pipe._domain_det, dtype=np.float32),
                             dtype=torch.float32)
        domain_det = det_t.unsqueeze(0).expand(n_s4, -1)

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
        }

    modes = [
        ("E0_w1", "w1", (1.0, 0.0)),
        ("E1_cavm_10", "cavm", (1.0, 0.0)),
        ("E2_cavm_01", "cavm", (0.0, 1.0)),
        ("E3_cavm_11", "cavm", (1.0, 1.0)),
    ]
    results = {}
    for name, mm, lam in modes:
        ev = predict(mm, lam)
        results[name] = summarize(ev)

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

    return {
        "dataset": dataset_key, "backbone": backbone, "seed": seed,
        "S2_loss": float(s2_loss), "n_S3M": len(mem_days),
        "n_S2T": len(s2_batches), "n_S2V": len(s2v_batches),
        "selected_k": k, "q": q_info["q"],
        "n_S4_days": n_s4,
        "s1r_days": _s1r_count(exp, ts),
        "neg_price_frac_s4": neg_frac,
        "cavm_key_version": pipe.cavm_key_builder.version,
        "cavm_key_dim": pipe.cavm_key_builder.dim,
        "cavm_global_days": len(pipe.cavm_global),
        "modes": results,
        "delta_vs_E0": delta,
    }


def _s1r_count(exp, ts) -> int:
    import pandas as pd
    n = 0
    for d in sorted(exp.dates_in_split("S1R")):
        idxs = np.where((ts.dt.date == pd_date(d)).values)[0]
        if len(idxs) == 24:
            n += 1
    return n


# --------------------------------------------------------------- aggregate ----
def _flat_row(cell: dict) -> dict:
    """One row per (cell, mode) for matrix.csv."""
    base = {"cell": f"{cell['dataset']}:{cell['backbone']}:s{cell['seed']}",
            "market": cell["dataset"], "host": cell["backbone"],
            "seed": cell["seed"]}
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
                     "mean_lcb": m["mean_lcb"],
                     "delta_MAE": d.get("MAE_delta"),
                     "delta_RMSE": d.get("RMSE_delta"),
                     "delta_exec_rate": d.get("exec_rate_delta"),
                     "delta_A_true": d.get("A_true_delta"),
                     "delta_exec_A_true": d.get("exec_mean_A_true_delta"),
                     "delta_harm": d.get("exec_harm_rate_delta"),
                     "action_changed_days": d.get("action_changed_days"),
                     "neighbor_changed_days": d.get("neighbor_changed_days"),
                     "S2_loss": cell["S2_loss"], "selected_k": cell["selected_k"],
                     "q": cell["q"], "s1r_days": cell["s1r_days"],
                     "neg_price_frac_s4": cell["neg_price_frac_s4"]})
    return rows


def _per_market(rows: list[dict], mode: str) -> dict:
    """Direction-stability per market for a retrieval mode (E2/E3 vs E0)."""
    out = {}
    for mk in MARKETS:
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
        # seed consistency: does the point improvement hold in >=2 of 3 seeds?
        by_seed = {}
        for r in sub:
            by_seed.setdefault(r["seed"], []).append(r["delta_MAE"])
        seeds_point_help = sum(
            1 for s, ds in by_seed.items()
            if ds and sum(1 for x in ds if x < 0) >= 2)
        n_cells = len(sub)
        point_help_frac = point_helps / max(n_cells, 1)
        out[mk] = {
            "n_cells": n_cells,
            "n_seeds": len(by_seed),
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
        "protocol": "hch_v2_phase4_cavm_experiment_design_v0.1_2026-08-14 §9.2",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": R._git_head(),
        "matrix": {"markets": MARKETS, "hosts": HOSTS, "seeds": SEEDS,
                   "n_cells": len({r["cell"] for r in rows})},
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
            "harm_up_cells": sum(1 for r in e2 if r["delta_harm"] is not None and r["delta_harm"] > 0),
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
            "harm_up_cells": sum(1 for r in e3 if r["delta_harm"] is not None and r["delta_harm"] > 0),
        },
    }
    summary["verdict"] = _verdict(summary)
    return summary


def _verdict(s: dict) -> dict:
    per = s["per_market"]["E2"]
    # §9.2-1 "方向稳定改善": 该市场多数 cell(≥1/2)点预测改善 + 均值改善
    #   + 至少 1 cell 动作价值同向(非 point-only)。
    mk_improve = {mk: bool(v.get("market_improves"))
                  for mk, v in per.items()}
    n_mk_improve = sum(1 for v in mk_improve.values() if v)
    improving_mk = [mk for mk, v in mk_improve.items() if v]
    # §9.2-2 "3 seeds 方向非单一 seed 支撑": 只对声称改善的市场判定
    seeds_ok = all(per[mk].get("seeds_with_point_help", 0) >= 2
                   for mk in improving_mk) if improving_mk else False
    lag0 = per.get("LAGO_NP", {})
    lag0_regress = (lag0.get("n_cells", 0) > 0
                    and lag0.get("mean_delta_MAE", 0) > 0
                    and lag0.get("point_help_frac", 0) <= 1 / max(lag0.get("n_cells", 1), 1))
    opp = s["overall_E2_vs_E0"]["opposite_cells"]
    harm = s["overall_E2_vs_E0"]["harm_up_cells"]

    checks = {
        "1_market_types_ge2": n_mk_improve,
        "2_seeds_ge2_of_3": bool(seeds_ok),
        "3_lag0_no_repro_regression": not lag0_regress,
        "4_action_not_opposite_point": opp == 0,
        "5_streaming_downgrade": "n/a (P3: local append-only, never consumed)",
        "6_not_point_only": s["overall_E2_vs_E0"]["action_help_cells"] > 0,
    }
    verdict = ("GATE_PASS" if (n_mk_improve >= 2 and seeds_ok and not lag0_regress
                               and opp == 0)
               else "GATE_NOT_YET_PASS")
    return {"verdict": verdict, "checks": checks,
            "market_improving_types": sorted(improving_mk)}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=str, default=",".join(MARKETS))
    ap.add_argument("--hosts", type=str, default=",".join(HOSTS))
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--cell", type=str, default=None,
                    help="single cell 'MARKET:HOST:SEED' for sanity (skips matrix)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--resummarize", type=str, default=None,
                    help="rebuild summary.json from existing cell JSONs in DIR "
                         "(recompute with current _per_market/_verdict logic)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resummarize:
        src = Path(args.resummarize)
        cells = sorted(src.glob("*_s*.json"))
        cells = [c for c in cells if not c.name.startswith("sanity_")]
        rows = []
        missing = []
        for c in cells:
            try:
                cell = json.loads(c.read_text(encoding="utf-8"))
            except Exception as e:
                missing.append({"cell": c.name, "error": f"load {e!r}"})
                continue
            flat = _flat_row(cell)
            rows.extend(flat if isinstance(flat, list) else [flat])
        if missing:
            with open(src / "failures_reload.json", "w", encoding="utf-8") as f:
                json.dump(missing, f, ensure_ascii=False, indent=2)
        if not rows:
            print(f"[p2ext] no rows recovered from {src}", flush=True)
            return
        summary = build_summary(rows)
        with open(src / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[p2ext] resummarized {len(rows)} rows from {src} -> summary.json")
        print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
        return

    if args.cell:
        mk, bb, sd = args.cell.split(":")
        cell = run(mk, bb, int(sd))
        with open(out_dir / f"sanity_{mk}_{bb}_s{sd}.json", "w",
                  encoding="utf-8") as f:
            json.dump(cell, f, indent=2, default=str)
        print(f"sanity cell {mk}:{bb}:s{sd} done")
        return

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    failures = []
    for mk in markets:
        for bb in hosts:
            for sd in seeds:
                tag = f"{mk}:{bb}:s{sd}"
                print(f"[p2ext] {tag} ...", flush=True)
                try:
                    cell = run(mk, bb, sd)
                except Exception as e:  # one cell must not kill the matrix
                    failures.append({"cell": tag, "error": str(e)})
                    print(f"    FAIL {tag}: {e!r}", flush=True)
                    continue
                with open(out_dir / f"{mk}_{bb}_s{sd}.json", "w",
                          encoding="utf-8") as f:
                    json.dump(cell, f, indent=2, default=str)
                rows.extend(_flat_row(cell))
                r = cell["modes"]["E2_cavm_01"]
                print(f"    E2 MAE={r['MAE']:.3f} (Δ={cell['delta_vs_E0']['E2_cavm_01']['MAE_delta']:+.3f}) "
                      f"A_true={r['mean_A_true'] if r['mean_A_true'] is None else round(r['mean_A_true'],4)} "
                      f"k={cell['selected_k']}", flush=True)
    if failures:
        with open(out_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"[p2ext] {len(failures)} cells failed: {failures}", flush=True)
    if not rows:
        print("[p2ext] no successful cells — aborting summary", flush=True)
        return

    with open(out_dir / "matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = build_summary(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== §9.2 verdict =====")
    print(json.dumps(summary["verdict"], ensure_ascii=False, indent=2))
    print(f"\n[p2ext] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
