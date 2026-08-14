"""P1 round-1: point readout matrix + six-object point table + safe-action track.

Docs: hch_v2_p1_readout_training_data_sota_experiment_design_prompt_v0.1_2026-08-14 (§7.1).

Runs (each in a fresh run dir, never overwrites old results):
  r0  T0 control curves      — export P0A_RERUN training history (train_loss /
                                s2v_macro / s2v_per_domain / health / sampling).
  r1  five point readouts    — identity / weighted-mean / weighted-median /
                                sMAPE-Bayes-action / global-shrink over
                                market x host x seed. S2V compares + fits the ONE
                                global shrink alpha; S4 evaluated once, frozen.
  r2  six fixed objects      — B0 Identity, B1 Residual-L1, B2 QuantileResidual-
                                LGBM, B3 delta-Adapter Ada-Y, B4 PIR, HCH
                                (pre-registered readout). Point table only.
  r3  safe-action track      — raw-IAH double-event proposal + DVG gated
                                release (decoupled from point readout).

Pre-registered main readouts (§3.2): MAE -> weighted median; RMSE -> weighted
mean; sMAPE -> sMAPE-Bayes-action; identity is the degenerate fallback.

sMAPE_eps (§3.3): eps = 1e-6 * median(|y|) frozen from S1R/S2T training target
scale, per dataset, recorded in dataset_registry.

Hard constraints (§0.2): no change to IAH-CRPS core, query-dose replay, double
event proposal, whole-day action calibration, LCB gating, or the six fixed
comparison objects. S4 labels never used for training/readout/hyper-parameter
selection. Point-readout wins are NOT reported as safe-action wins.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "09-paper-gate" / "runner"))

import r1a_run as R
import r1a9_action_calibration as M
from r1b_generalization_screen import (SEED, SOURCE_MARKETS, HOSTS, build_head,
                                       det_for_variant)
from _final_point import raw_metrics, final_metrics
from run_matrix import cell_table, run_baselines, BASELINES

# ---------------------------------------------------------------- constants --
HEADLINE_DS = ["LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
               "NEM_SA1", "GEFCOM14P", "NORD_DK1"]
HOSTS4 = ["Linear", "MLP", "LSTM", "PatchTST"]
SEEDS = [0, 1, 2]

READOUTS = ["identity", "weighted_mean", "weighted_median",
            "bayes_action_smape", "global_shrink"]
MAIN_READOUT = {"mae": "weighted_median", "rmse": "weighted_mean",
                "smape": "bayes_action_smape"}

P0A_HEAD = HERE / "results" / "P0A_RERUN_20260814" / "learned_sig_main_head.pt"
P0A_REPORT = HERE / "results" / "P0A_RERUN_20260814" / "training_reports.json"
TRAINER_CMP = HERE / "results" / "TRAINER_CMP_20260814_seeds012"

GRID_N = 101        # sMAPE Bayes action one-dim grid resolution
SHRINK_GRID = np.linspace(0.0, 2.0, 401)   # global alpha grid (diagnostic)


def cell_name(mk: str, bb: str) -> str:
    return f"{mk}:{bb}"


def cells_all():
    return [cell_name(mk, bb) for mk in HEADLINE_DS for bb in HOSTS4]


# -------------------------------------------------------------- sMAPE utils --
def cell_eps(info) -> float:
    """Frozen sMAPE eps = 1e-6 * median(|y_train|); y_train = S2T target."""
    y_full = info.ds["price"].astype(np.float64)
    ts = info.ds["ts"]
    vals = []
    for d in sorted(info.exp.dates_in_split("S2T")):
        idxs = np.where((ts.dt.date == R.pd_date(d)).values)[0]
        if len(idxs) == 24:
            vals.append(y_full[idxs])
    if not vals:
        vals = [np.abs(y_full[np.isfinite(y_full)])]
    med = float(np.median(np.abs(np.concatenate(vals))))
    return float(1e-6 * max(med, 1e-9))


def smape_eps_arr(y, x, eps: float) -> np.ndarray:
    """Per-hour no-floor sMAPE in % with frozen eps (§3.3)."""
    den = np.abs(y) + np.abs(x) + eps
    return 200.0 * np.abs(y - x) / np.maximum(den, 1e-12)


# ------------------------------------------------------------ readout layer --
def readouts_from_atoms(z0, wm, wz, wp, mm, mp, s, eps: float,
                        grid_n: int = GRID_N) -> dict:
    """Five point readouts from the three-atom hyperbolic distribution.

    z0/wm/wz/wp/mm/mp: [H] per-hour. s: scalar day scale. eps: frozen sMAPE eps.
    Returns raw-space [H] arrays keyed by readout name (identity, weighted_mean,
    weighted_median, bayes_action_smape).
    """
    z_minus = z0 - mm
    z_plus = z0 + mp
    x_down = s * np.sinh(z_minus)
    x_ident = s * np.sinh(z0)
    x_up = s * np.sinh(z_plus)
    H = len(z0)
    out = {"identity": x_ident.copy(),
           "weighted_mean": wm * x_down + wz * x_ident + wp * x_up}

    # weighted median (lower median of the 3-atom distribution; MAE Bayes action)
    wm_med = np.zeros(H)
    for h in range(H):
        vals = np.array([x_down[h], x_ident[h], x_up[h]])
        ws = np.array([wm[h], wz[h], wp[h]])
        order = np.argsort(vals)
        cum = np.cumsum(ws[order])
        wm_med[h] = vals[order][np.searchsorted(cum, 0.5)]
    out["weighted_median"] = wm_med

    # sMAPE Bayes action: 1-dim search on the finite interval spanned by the
    # three support points, minimizing expected sMAPE_eps. Fixed rule — no per-
    # test-set parameter fitting.
    ba = np.zeros(H)
    for h in range(H):
        atoms = np.array([x_down[h], x_ident[h], x_up[h]])
        ws = np.array([wm[h], wz[h], wp[h]])
        lo, hi = float(atoms.min()), float(atoms.max())
        if hi - lo < 1e-9:
            ba[h] = atoms[0]
            continue
        grid = np.unique(np.concatenate([atoms, np.linspace(lo, hi, grid_n)]))
        loss = np.zeros(len(grid))
        for a in range(3):
            den = np.abs(grid) + np.abs(atoms[a]) + eps
            loss += ws[a] * (200.0 * np.abs(grid - atoms[a]) / np.maximum(den, 1e-12))
        ba[h] = grid[int(np.argmin(loss))]
    out["bayes_action_smape"] = ba
    return out


def apply_global_shrink(x_read_raw, host, alpha: float) -> np.ndarray:
    """§3.2 global shrink: x = host + alpha * (x_read_raw - host)."""
    return host + alpha * (np.asarray(x_read_raw) - np.asarray(host))


def fit_global_shrink(host_c, raw_c, y_c, eps: float) -> dict:
    """Fit ONE global alpha on aggregated S2V minimizing sMAPE_eps (diagnostic).
    host_c/raw_c/y_c: concatenated hour arrays across cells.
    """
    alpha_mae, best_mae, alpha_sm = None, None, None
    best_sm = None
    best_sm_loss = float("inf")
    best_mae_loss = float("inf")
    for a in SHRINK_GRID:
        x = apply_global_shrink(raw_c, host_c, a)
        sm = float(np.mean(smape_eps_arr(y_c, x, eps)))
        if sm < best_sm_loss:
            best_sm_loss, alpha_sm = sm, a
        ae = np.abs(y_c - x)
        m = float(np.mean(ae))
        if m < best_mae_loss:
            best_mae_loss, alpha_mae = m, a
    host_sm = float(np.mean(smape_eps_arr(y_c, host_c, eps)))
    host_mae = float(np.mean(np.abs(y_c - host_c)))
    return {
        "alpha_smape": round(float(alpha_sm), 4),
        "smape_at_alpha": round(best_sm_loss, 6),
        "alpha_mae": round(float(alpha_mae), 4),
        "mae_at_alpha": round(best_mae_loss, 6),
        "identity_smape_s2v": round(host_sm, 6),
        "identity_mae_s2v": round(host_mae, 6),
        "n_hours": int(len(y_c)),
    }


# ------------------------------------------------------------- S2V scanning --
def s2v_scan(head, info, det_np, eps: float) -> dict:
    """Readout diagnostics on the real S2V split (info.s2v_batches)."""
    det_t = R.det_for(det_np, 1)
    pred_all = {k: [] for k in ("identity", "weighted_mean", "weighted_median",
                                "bayes_action_smape")}
    y_all, host_all = [], []
    n_hours = 0
    with torch.no_grad():
        for batch in info.s2v_batches:
            host, ctx, tgt, vm = batch[:4]
            det_b = det_t.expand(host.shape[0], -1) if det_t.shape[0] != host.shape[0] else det_t
            out = head(host, ctx, valid_mask=vm, domain_det=det_b)
            for i in range(host.shape[0]):
                vh = (vm[i] > 0.5) & torch.isfinite(tgt[i, :, 0])
                vh_np = vh.cpu().numpy()
                if not vh_np.any():
                    continue
                z0 = out["z0"][i].cpu().numpy()
                wm = out["w_minus"][i].cpu().numpy()
                wz = out["w_zero"][i].cpu().numpy()
                wp = out["w_plus"][i].cpu().numpy()
                mm = out["m_minus"][i].cpu().numpy()
                mp = out["m_plus"][i].cpu().numpy()
                hb = host[i].squeeze(-1).cpu().numpy()
                s = float(np.maximum(np.abs(hb).mean(), 1e-12))
                y = tgt[i, :, 0].cpu().numpy()
                preds = readouts_from_atoms(z0, wm, wz, wp, mm, mp, s, eps)
                for k in pred_all:
                    pred_all[k].append(preds[k][vh_np])
                y_all.append(y[vh_np])
                host_all.append((s * np.sinh(z0))[vh_np])
                n_hours += int(vh_np.sum())
    yc = np.concatenate(y_all) if y_all else np.zeros(0)
    hc = np.concatenate(host_all) if host_all else np.zeros(0)
    metrics = {}
    for k, v in pred_all.items():
        if not v:
            continue
        xc = np.concatenate(v)
        ae = np.abs(yc - xc)
        metrics[k] = {
            "mae": round(float(ae.mean()), 6),
            "rmse": round(float(np.sqrt((ae ** 2).mean())), 6),
            "smape_eps": round(float(np.mean(smape_eps_arr(yc, xc, eps))), 6),
        }
    # S2V per-readout selection signal (diagnostic only; S4 is frozen once).
    if metrics:
        best = min(metrics, key=lambda k: metrics[k]["mae"])
    else:
        best = None
    return {"n_hours": n_hours, "metrics": metrics, "best_by_mae_s2v": best,
            "arrays": {"host": hc, "y": yc,
                       "weighted_median": np.concatenate(pred_all["weighted_median"])
                       if pred_all["weighted_median"] else np.zeros(0)},
            "eps": eps}


# -------------------------------------------------------------- S4 eval ------
def s4_eval(dd: dict, readout: str, eps: float, alpha: float | None = None) -> dict:
    """Frozen S4 evaluation of a readout (no tuning on S4)."""
    preds_all, y_all, host_all, days_d, day_mae_p, day_mae_h = [], [], [], [], [], []
    n_days = 0
    for day in dd["days"]:
        if day["block"] != "dev":
            continue
        z0 = np.asarray(day["z0"]); mm = np.asarray(day["mm"])
        mp = np.asarray(day["mp"]); wm = np.asarray(day["wm"])
        wp = np.asarray(day["wp"]); wz = 1.0 - wm - wp
        s = float(day["s_day"])
        y = np.asarray(day["price"], dtype=np.float64)
        vm = np.asarray(day["vm"]).astype(bool)
        pr = readouts_from_atoms(z0, wm, wz, wp, mm, mp, s, eps)
        host = s * np.sinh(z0)
        if readout == "global_shrink":
            x = apply_global_shrink(pr["weighted_median"], host, alpha)
        elif readout == "identity":
            x = pr["identity"]
        else:
            x = np.asarray(pr[readout])
        preds_all.append(x[vm]); y_all.append(y[vm]); host_all.append(host[vm])
        n_days += 1
        days_d.append(str(day["date"]))
        day_mae_p.append(float(np.abs(y[vm] - x[vm]).mean()))
        day_mae_h.append(float(np.abs(y[vm] - host[vm]).mean()))
    if not preds_all:
        return {"n_days": 0, "n_hours": 0}
    yc = np.concatenate(y_all); xc = np.concatenate(preds_all)
    hc = np.concatenate(host_all)
    ae = np.abs(yc - xc)
    tail = np.abs(yc) >= np.quantile(np.abs(yc), 0.95)
    neg = yc < 0
    host_ae = np.abs(yc - hc)
    return {
        "n_days": n_days, "n_hours": int(len(yc)), "readout": readout,
        "mae": round(float(ae.mean()), 6),
        "rmse": round(float(np.sqrt((ae ** 2).mean())), 6),
        "smape_eps": round(float(np.mean(smape_eps_arr(yc, xc, eps))), 6),
        "host_mae": round(float(host_ae.mean()), 6),
        "improvement_vs_host": round((host_ae.mean() - ae.mean()) / host_ae.mean(), 5)
                               if host_ae.mean() > 0 else None,
        "high_tail_mae": round(float(ae[tail].mean()), 6) if tail.any() else None,
        "neg_price_mae": round(float(ae[neg].mean()), 6) if neg.any() else None,
        "days": days_d, "day_mae_pred": np.asarray(day_mae_p),
        "day_mae_host": np.asarray(day_mae_h),
    }


# ------------------------------------------------------------- head loader --
def load_head(seed: int, torch) -> torch.nn.Module:
    pt = TRAINER_CMP / f"head_vA_seed{seed}.pt"
    head = build_head("learned_sig")
    head.load_state_dict(torch.load(pt, map_location="cpu"))
    head.eval()
    return head


# ------------------------------------------------------------ curve export --
def _dump_curves(out_dir: Path, history: list[dict]):
    df = pd.DataFrame(history)
    per = pd.DataFrame([{**{"epoch": h["epoch"]}, **h.get("per_domain", {})}
                        for h in history])
    upd = pd.DataFrame([{**{"epoch": h["epoch"]},
                         **h.get("updates_per_domain", {})}
                        for h in history])
    health = pd.DataFrame([{**{"epoch": h["epoch"]},
                            **(h.get("health") or {})} for h in history])
    df.to_csv(out_dir / "train_loss_curve.csv", index=False)
    per.to_csv(out_dir / "s2v_per_domain_curve.csv", index=False)
    upd.to_csv(out_dir / "domain_sampling.csv", index=False)
    health.to_csv(out_dir / "health_curve.csv", index=False)
    # s2v macro + worst
    pd.DataFrame({"epoch": df["epoch"], "macro_s2v": df["macro_s2v"],
                  "worst_s2v": df["worst_s2v"]}).to_csv(
        out_dir / "s2v_macro_curve.csv", index=False)
    return {
        "epochs_run": int(df["epoch"].max()) + 1,
        "best_macro_s2v": float(df["macro_s2v"].min()),
        "best_epoch": int(df.loc[df["macro_s2v"].idxmin(), "epoch"]),
        "final_macro_s2v": float(df["macro_s2v"].iloc[-1]),
        "final_train_loss": float(df["train_loss"].iloc[-1]) if "train_loss" in df else None,
        "worst_s2v": float(df["worst_s2v"].min()),
    }


# ---------------------------------------------------------------- r0 ---------
def run_r0(out_dir: Path) -> dict:
    rep = json.load(open(P0A_REPORT, encoding="utf-8"))["LearnedSig_main"]
    curves = _dump_curves(out_dir, rep["history"])
    shutil.copy(P0A_HEAD, out_dir / "t0_control_head.pt")
    return {
        "head": str(P0A_HEAD),
        "source_report": str(P0A_REPORT),
        "curves": curves,
    }


# ---------------------------------------------------------------- r1 ---------
def run_r1(out_dir: Path, seeds: list[int], cells: list[str]) -> dict:
    shrink_fit = None
    s2v_rows, s4_rows, s4_full = [], [], []
    # Phase A: S2V scans (fast) — collect seed0 arrays to fit ONE global alpha.
    seed0_arrays = {"host": [], "y": [], "weighted_median": []}
    eps_by_cell = {}
    infos = {}
    for cell in cells:
        mk, bb = cell.split(":")
        info = R.prepare_domain(mk, bb, seed=SEED)
        infos[cell] = info
        eps_by_cell[cell] = cell_eps(info)
    for cell in cells:
        info = infos[cell]
        det_np = det_for_variant("learned_sig", info)
        head0 = load_head(0, torch)
        s2v = s2v_scan(head0, info, det_np, eps_by_cell[cell])
        del head0
        if s2v["arrays"]["y"].size:
            seed0_arrays["host"].append(s2v["arrays"]["host"])
            seed0_arrays["y"].append(s2v["arrays"]["y"])
            seed0_arrays["weighted_median"].append(s2v["arrays"]["weighted_median"])
    if seed0_arrays["y"]:
        hc = np.concatenate(seed0_arrays["host"])
        yc = np.concatenate(seed0_arrays["y"])
        rc = np.concatenate(seed0_arrays["weighted_median"])
        eps0 = float(np.median(list(eps_by_cell.values())))
        shrink_fit = fit_global_shrink(hc, rc, yc, eps0)
        alpha = shrink_fit["alpha_smape"]
    else:
        alpha = 1.0
        shrink_fit = {"alpha_smape": 1.0, "note": "no S2V hours"}

    # Phase B: full S2V + S4 per cell x seed.
    for cell in cells:
        info = infos[cell]
        det_np = det_for_variant("learned_sig", info)
        for seed in seeds:
            head = load_head(seed, torch)
            s2v = s2v_scan(head, info, det_np, eps_by_cell[cell])
            dd = M.collect_domain(None, cell.split(":")[0], cell.split(":")[1],
                                  "learned_sig", head=head)
            del head
            for name in READOUTS:
                s4 = s4_eval(dd, name, eps_by_cell[cell], alpha=alpha)
                row = {"cell": cell, "seed": seed, "readout": name,
                       "eps": eps_by_cell[cell], "alpha": alpha,
                       "n_hours": s4.get("n_hours", 0),
                       "mae": s4.get("mae"), "rmse": s4.get("rmse"),
                       "smape_eps": s4.get("smape_eps"),
                       "host_mae": s4.get("host_mae"),
                       "improvement_vs_host": s4.get("improvement_vs_host"),
                       "high_tail_mae": s4.get("high_tail_mae"),
                       "neg_price_mae": s4.get("neg_price_mae"),
                       "n_days": s4.get("n_days")}
                s4_rows.append(row)
                s4_full.append({**row,
                                "days": s4.get("days", []),
                                "day_mae_pred": s4.get("day_mae_pred", []).tolist(),
                                "day_mae_host": s4.get("day_mae_host", []).tolist()})
            for k, m in (s2v.get("metrics") or {}).items():
                s2v_rows.append({"cell": cell, "seed": seed, "readout": k,
                                 "s2v_mae": m.get("mae"), "s2v_rmse": m.get("rmse"),
                                 "s2v_smape": m.get("smape_eps")})
            s2v_rows.append({"cell": cell, "seed": seed, "readout": "global_shrink",
                             "s2v_mae": None, "s2v_rmse": None, "s2v_smape": None})
    pd.DataFrame(s4_rows).to_csv(out_dir / "readout_matrix.csv", index=False)
    pd.DataFrame(s2v_rows).to_csv(out_dir / "readout_s2v.csv", index=False)
    with open(out_dir / "global_shrink_fit.json", "w", encoding="utf-8") as f:
        json.dump(shrink_fit, f, ensure_ascii=False, indent=2)
    # Cache S4 readout metrics for R2 (HCH point table) without re-running.
    # s4_full keeps day-level arrays so R2 can compute paired block-bootstrap CIs.
    hch_metrics = {}
    for row in s4_full:
        key = (row["cell"], row["seed"], row["readout"])
        hch_metrics[f"{key[0]}|{key[1]}|{key[2]}"] = row
    with open(out_dir / "hch_s4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(hch_metrics, f, ensure_ascii=False)
    return {"shrink_fit": shrink_fit, "n_cells": len(cells),
            "n_readout_rows": len(s4_rows)}


# ---------------------------------------------------------------- r2 ---------
def _day_mae_arrays(hours):
    """(dates, pred_day_mae, host_day_mae) aligned per S4 day."""
    ser = {}
    for h in hours:
        d = str(h["date"])
        ser.setdefault(d, []).append(
            (float(np.mean(np.abs(np.asarray(h["pred"], float) - np.asarray(h["y"], float)))),
             float(np.mean(np.abs(np.asarray(h["host"], float) - np.asarray(h["y"], float))))))
    dates = sorted(ser)
    if not dates:
        return [], np.array([]), np.array([])
    pm = np.array([ser[d][0][0] for d in dates])
    hm = np.array([ser[d][0][1] for d in dates])
    return dates, pm, hm


def _recompute_metrics(hours, eps: float) -> dict:
    """Uniform MAE/RMSE/sMAPE_eps from hour rows (all objects share §3.3 eps)."""
    if not hours:
        return {}
    y = np.concatenate([np.asarray(h["y"], dtype=np.float64) for h in hours])
    p = np.concatenate([np.asarray(h["pred"], dtype=np.float64) for h in hours])
    ae = np.abs(y - p)
    return {"mae": round(float(ae.mean()), 6),
            "rmse": round(float(np.sqrt((ae ** 2).mean())), 6),
            "smape_eps": round(float(np.mean(smape_eps_arr(y, p, eps))), 6),
            "n_hours": int(len(y))}


def _ci_append(row: dict, diff: np.ndarray, day_mae_host: np.ndarray) -> dict:
    """Paired block-bootstrap 95% CI of mean day diff (host - method, >0 helps)."""
    if len(diff) == 0 or len(day_mae_host) == 0:
        return {**row, "mean_day_diff": None, "ci_lo": None, "ci_hi": None,
                "n_days": 0}
    ci = block_bootstrap_ci(diff)
    return {**row, "mean_day_diff": round(float(diff.mean()), 6),
            "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_days": ci["n"]}


def run_r2(out_dir: Path, seeds: list[int], cells: list[str]) -> dict:
    # HCH pre-registered readouts come from R1 cache.
    cache = json.load(open(out_dir / "hch_s4_metrics.json", encoding="utf-8"))
    rows_met, rows_tail = [], []
    eps_by_cell = {}
    infos = {}
    for cell in cells:
        mk, bb = cell.split(":")
        info = R.prepare_domain(mk, bb, seed=SEED)
        infos[cell] = info
        eps_by_cell[cell] = cell_eps(info)
    for cell in cells:
        mk, bb = cell.split(":")
        table = cell_table(mk, bb)
        if table is None or "days" not in table:
            continue
        for seed in seeds:
            # B0-B4 via run_matrix (identical S3M/S3C/S4 protocol).
            met, hours = run_baselines(table, cell)
            eps = eps_by_cell[cell]
            # Uniform metrics per baseline method (same eps definition as HCH).
            for name in BASELINES:
                hrs = [h for h in hours if h.get("method") == name]
                uni = _recompute_metrics(hrs, eps)
                if not uni:
                    continue
                _dates, pm, hm = _day_mae_arrays(hrs)
                base = {"cell": cell, "seed": seed, "method": name,
                        "metric_role": "primary", "readout": "n/a",
                        **uni, "host_mae": None, "improvement_vs_host": None}
                rows_met.append(_ci_append(base, hm - pm, hm))
                _tail_rows(hrs, cell, seed, name, eps, rows_tail)
            # HCH: pre-registered readout for the primary point table.
            for metric, readout in MAIN_READOUT.items():
                key = f"{cell}|{seed}|{readout}"
                r1 = cache.get(key)
                if r1 is None:
                    continue
                dp = np.asarray(r1.get("day_mae_pred", []) or [], dtype=np.float64)
                dh = np.asarray(r1.get("day_mae_host", []) or [], dtype=np.float64)
                base = {"cell": cell, "seed": seed, "method": "HCH",
                        "metric_role": metric, "readout": readout,
                        "mae": r1["mae"], "rmse": r1["rmse"],
                        "smape_eps": r1["smape_eps"],
                        "host_mae": r1["host_mae"],
                        "improvement_vs_host": r1["improvement_vs_host"],
                        "n_hours": r1["n_hours"], "n_days": r1["n_days"]}
                rows_met.append(_ci_append(base, dh - dp, dh))
                rows_tail.append({"cell": cell, "seed": seed,
                                  "method": f"HCH:{readout}",
                                  "high_tail_mae": r1.get("high_tail_mae"),
                                  "neg_price_mae": r1.get("neg_price_mae"),
                                  "n_hours": r1.get("n_hours")})
    pd.DataFrame(rows_met).to_csv(out_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(rows_tail).to_csv(out_dir / "tail_metrics.csv", index=False)
    return {"n_method_rows": len(rows_met), "n_tail_rows": len(rows_tail)}


def _tail_rows(hours, cell, seed, method, eps, sink):
    if not hours:
        return
    y = np.concatenate([np.asarray(h["y"], dtype=np.float64) for h in hours])
    p = np.concatenate([np.asarray(h["pred"], dtype=np.float64) for h in hours])
    ae = np.abs(y - p)
    tail = np.abs(y) >= np.quantile(np.abs(y), 0.95)
    neg = y < 0
    sink.append({
        "cell": cell, "seed": seed, "method": method,
        "high_tail_mae": round(float(ae[tail].mean()), 6) if tail.any() else None,
        "high_tail_rate": round(float(tail.mean()), 4),
        "neg_price_mae": round(float(ae[neg].mean()), 6) if neg.any() else None,
        "neg_price_rate": round(float(neg.mean()), 4),
        "n_hours": int(len(y)),
    })


# ---------------------------------------------------------------- r3 ---------
def run_r3(out_dir: Path, seeds: list[int], cells: list[str]) -> dict:
    """Safe-action track on seed0 (seed dimension is covered by R1/R2 point
    tables; §6.4 safety metrics are not seed-averaged)."""
    rows = []
    for cell in cells:
        mk, bb = cell.split(":")
        head = load_head(0, torch)
        dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
        del head
        c0 = M.RawIAH()
        cal_rows = M.evaluate_days(c0, dd)
        dv = M.dvg_and_s4(cal_rows, R.ALPHA)
        s4_rows_r = [r for r in cal_rows if r["block"] == "dev"]
        dose_hours = [float((np.asarray(r["pi"])[np.asarray(r["vm"]).astype(bool)]
                             != 0.0).mean()) for r in s4_rows_r]
        rows.append({
            "cell": cell, "seed": 0, "method": "HCH_C0_RAWIAH_DVG",
            "proposal": "double_event_unconditional",
            "gate": "LCB>0 (DVG split-conformal)",
            "n_calib": dv.get("n_calib"), "q": dv.get("q"),
            "n_eval": dv.get("n_eval"),
            "release_rate": dv.get("release_rate"),
            "identity_rate": dv.get("identity_rate"),
            "harmful_rate": dv.get("harmful_rate"),
            "mean_gain_release": dv.get("mean_gain_release"),
            "net_value": dv.get("net_value"),
            "coverage": dv.get("coverage"),
            "mean_dose_hours_released": round(float(np.mean(dose_hours)), 4)
            if dose_hours else None,
        })
    pd.DataFrame(rows).to_csv(out_dir / "action_safety.csv", index=False)
    return {"n_rows": len(rows)}


# ------------------------------------------------------ registry / split -----
def _ds_level_meta(ds_key: str) -> dict:
    """Dataset-level metadata (host-independent): load once per dataset."""
    from common import build_tabular
    from eval_manifest import ExperimentManifest
    ds = R.load_dataset(ds_key)
    ts = ds["ts"]
    y = ds["price"].astype(np.float64)
    X, yt, names, valid = build_tabular(ds)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id=ds_key)
    med = float(np.median(np.abs(y)))
    n_neg = int((y < 0).sum())
    return {"ds": ds, "y": y, "med": med, "n_neg": n_neg,
            "exp": exp, "ts": ts, "valid": valid}


def build_registry_split(out_dir: Path, cells: list[str]) -> None:
    """Dataset registry + split manifest. Only needs ds-level info (loads each
    dataset once, not per host), so it is fast and host-independent."""
    reg_rows, split_rows = [], []
    ds_cache = {}
    for cell in cells:
        mk, bb = cell.split(":")
        if mk not in ds_cache:
            ds_cache[mk] = _ds_level_meta(mk)
        m = ds_cache[mk]
        ts = m["ts"]; y = m["y"]; exp = m["exp"]
        split_map = {}
        for sp in ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4"):
            dates = sorted(exp.dates_in_split(sp))
            split_map[sp] = f"{dates[0]}..{dates[-1]}" if dates else "n/a"
        tz = str(ts.dt.tz) if hasattr(ts.dt, "tz") else "n/a"
        reg_rows.append({
            "dataset": mk, "host": bb, "cell": cell,
            "timezone": tz,
            "start": str(ts.iloc[0].date()), "end": str(ts.iloc[-1].date()),
            "n_days": int(len(ts) // 24) if len(ts) >= 24 else int(len(ts)),
            "n_hours": int(len(ts)),
            "median_abs_price": round(m["med"], 4),
            "neg_price_rate": round(m["n_neg"] / len(y), 5),
            "smape_eps": None,   # filled below via cell_eps (needs info)
            "target_role": "day-ahead price" if "RT" not in mk else "real-time price",
            "license": ("LAGO/EPEX public" if mk in ("LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM")
                        else "Nord Pool public" if mk in ("LAGO_NP", "NORD_DK1")
                        else "NEM public" if mk == "NEM_SA1" else "GEFCOM public"),
            "in_training": ("source" if mk in SOURCE_MARKETS
                            else "transfer" if mk in ("LAGO_BE", "LAGO_FR", "LAGO_NP")
                            else "holdout"),
        })
        for sp, rng in split_map.items():
            split_rows.append({"cell": cell, "split": sp, "date_range": rng,
                               "split_hash": exp.split_hash})
    # Backfill frozen sMAPE eps from R1 readout matrix if already produced.
    rm = out_dir / "readout_matrix.csv"
    if rm.exists():
        rdf = pd.read_csv(rm)
        eps_map = {c: e for c, e in zip(rdf["cell"], rdf["eps"]) if not pd.isna(e)}
        for row in reg_rows:
            row["smape_eps"] = eps_map.get(row["cell"])
    pd.DataFrame(reg_rows).to_csv(out_dir / "dataset_registry.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "split_manifest.csv", index=False)


# ------------------------------------------------------------ seed summary ---
def block_bootstrap_ci(diff: np.ndarray, block=7, n=5000, alpha=0.05) -> dict:
    """Block bootstrap 95% CI of the mean of a day-level difference series."""
    if len(diff) == 0:
        return {"ci_lo": None, "ci_hi": None, "n": 0}
    T = len(diff)
    n_blk = max(int(np.ceil(T / block)), 1)
    rng = np.random.default_rng(0)
    means = np.empty(n)
    off = np.arange(block)[None, :] + np.arange(n_blk)[:, None] * block
    for b in range(n):
        starts = rng.integers(0, max(T - block + 1, 1), n_blk)
        idx = (starts[:, None] + off[:1]) % T
        idx = idx.ravel()[:T]
        means[b] = diff[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "n": int(T)}


def run_seed_summary(out_dir: Path) -> pd.DataFrame:
    """Per-cell seed spread of the pre-registered MAE readout (weighted median).
    Also adds host MAE for context. Day-level bootstrap lives in R2."""
    df = pd.read_csv(out_dir / "readout_matrix.csv")
    df = df[df["readout"] == "weighted_median"]
    rows = []
    for cell, g in df.groupby("cell"):
        mae_vals = g["mae"].dropna().tolist()
        host_vals = g["host_mae"].dropna().tolist()
        rows.append({
            "cell": cell, "n_seeds": int(len(g)),
            "mae_mean": round(float(np.mean(mae_vals)), 6) if mae_vals else None,
            "mae_std": round(float(np.std(mae_vals)), 6) if len(mae_vals) > 1 else None,
            "mae_min": round(float(np.min(mae_vals)), 6) if mae_vals else None,
            "mae_max": round(float(np.max(mae_vals)), 6) if mae_vals else None,
            "host_mae_mean": round(float(np.mean(host_vals)), 6) if host_vals else None,
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "seed_summary.csv", index=False)
    return out


# ---------------------------------------------------------------- verdict ----
def write_verdict(out_dir: Path, r0: dict, r1: dict, r2: dict, r3: dict,
                  git_sha: str) -> None:
    if r0.get("curves"):
        conv = ("CONVERGED" if (r0["curves"].get("final_train_loss") is not None
                                and r0["curves"]["final_train_loss"]
                                < r0["curves"]["best_macro_s2v"] * 0.5)
                else "CHECK_CURVES")
    else:
        conv = "UNKNOWN (r0 not run)"
    verdict = {
        "convergence_status": conv,
        "best_epoch": r0.get("curves", {}).get("best_epoch"),
        "best_checkpoint": str(P0A_HEAD),
        "candidate_crps": r0.get("curves", {}).get("best_macro_s2v"),
        "point_readout_selected": {k: v for k, v in MAIN_READOUT.items()},
        "global_shrink_alpha": (r1.get("shrink_fit") or {}).get("alpha_smape"),
        "point_metrics": "baseline_comparison.csv",
        "tail_metrics": "tail_metrics.csv",
        "action_metrics": "action_safety.csv",
        "transfer_status": "NOT_TESTED (round-1 only; round-3 adds source/leave-out)",
        "negative_transfer_status": "NOT_TESTED",
        "SOTA_status": "NOT_TESTED",
        "next_recommendation": ("round-2 training-paradigm directions (T2 sampling first), "
                                "then round-3 incremental data admission"),
    }
    with open(out_dir / "VERDICT.md", "w", encoding="utf-8") as f:
        f.write("# P1 Round-1 Verdict\n\n")
        for k, v in verdict.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\ngit_sha: {git_sha}\n")
        f.write(f"shrink_fit:\n```json\n"
                f"{json.dumps((r1.get('shrink_fit') or {}), ensure_ascii=False, indent=2)}\n```\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="r0,r1,r2,r3")
    ap.add_argument("--cells", type=str, default="")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if not cells:
        cells = cells_all()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"P1_{datetime.now().strftime('%Y%m%d')}_{git_sha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[P1] out={out_dir} cells={len(cells)} seeds={seeds} runs={runs}",
          flush=True)
    manifest = {"git_sha": git_sha, "date": datetime.now().isoformat(),
                "command": " ".join(sys.argv), "cells": cells, "seeds": seeds,
                "runs": {}}

    if "r0" in runs:
        manifest["r0"] = run_r0(out_dir)
    if "r1" in runs:
        manifest["r1"] = run_r1(out_dir, seeds, cells)
    if "r2" in runs:
        manifest["r2"] = run_r2(out_dir, seeds, cells)
    if "r3" in runs:
        manifest["r3"] = run_r3(out_dir, seeds, cells)

    build_registry_split(out_dir, cells)
    try:
        run_seed_summary(out_dir)
    except Exception as e:
        print(f"[warn] seed_summary skipped: {e}", flush=True)
    write_verdict(out_dir, manifest.get("r0") or {}, manifest.get("r1") or {},
                  manifest.get("r2") or {}, manifest.get("r3") or {}, git_sha)

    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
