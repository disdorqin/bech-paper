"""WP-6 B3 foreign headline matrix — 8 datasets x 4 hosts x B0-B6.

Protocol: hch_v2_paper_benchmark_gate v0.1  §5.1 / §7 / §10.2 / §14 / §15 / §16.

Primary cells: 8 headline datasets x 4 hosts x {MAE, sMAPE} = 64.
Methods: B0 Identity, B1 ResidualL1, B2 QuantileResidual, B3 delta-Adapter,
         B4 PIR(limited_official), B5 HCH-Universal, B6 HCH-Local.

Evidence windows (mirror HCH A2 chain, fair comparison):
  fit = S3M(train+val)   cal = S3C   eval = S4(dev)
HCH final output = P0-A replay: x_final = s*sinh(z0 + pi_eff), pi_eff = pi iff
DVG releases the day else 0.

Outputs (results/{run}/):
  06_PREDICTIONS.parquet          hour-level preds per (cell, method)
  07_METRICS_BY_CELL.csv          scalar metrics (MAE/sMAPE primary)
  08_RANKS_BY_CELL.csv            per-cell ranks + strict/tied Top-1, Top-2 (B5)
  09_PRIMARY_WIN_RATE.csv         Top-1/tied & Top-2 rates over 64 cells + gate level
  10_DATASET_LEVEL_SUMMARY.csv
  11_HOST_LEVEL_SUMMARY.csv
  12_DM_TESTS.csv                 day-level DM (abs-error loss): B5 vs B0..B4,B6
  13_HCH_CANDIDATE_DIAGNOSTICS.csv  B5 per-cell: delta_crps, a2_net, selected, CRPS-vs-point sign
  14_HCH_ACTION_DIAGNOSTICS.csv   B5 DVG: release/identity/harmful rates, q, coverage
  15_FAILURE_MAP.csv              weak/divergent B5 cells tagged (protocol §16)
  16_GENERALIZATION_LEDGER.csv    delta_crps by market_seen/host_seen aggregation
  config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
HCH = PAPER.parent / "08-hch-v2"
PEERS = PAPER.parent / "07-route-e" / "peers"
ROOT = PAPER.parent.parent

for p in (ROOT / "src", HCH, PEERS):
    sys.path.insert(0, str(p))

import r1a_run as R                                # noqa: E402
import r1a9_action_calibration as M                # noqa: E402
import r1a11_prequential_calibration_router as P   # noqa: E402
from common import build_tabular                   # noqa: E402
from r1b_generalization_screen import (            # noqa: E402
    HOSTS, SEED, EvalDomain, build_head, train_candidate,
)
from r1b_stage2a_panel import eval_panel_domain    # noqa: E402
from baselines_v2 import Identity, ResidualL1, QuantileResidualLGBM  # noqa: E402
from official_adapters import DeltaAdapterLimited, PIRLimited        # noqa: E402
from _final_point import raw_metrics, final_metrics                  # noqa: E402

HEADLINE_DS = ["LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
               "NEM_SA1", "GEFCOM14P", "NORD_DK1"]
SOURCE_MK = {"LAGO_DE", "LAGO_PJM", "NEM_SA1"}      # R1B source; others are transfer
BASELINES = ("B0_Identity", "B1_ResidualL1", "B2_QuantileResidual",
             "B3_DeltaAdapter", "B4_PIR")
PREQ_CFG = {"burn_in": P.BURN_IN_DAYS, "block": P.BLOCK_DAYS,
            "min_oos_days": P.GATE_A_MIN_OOS_DAYS,
            "harm_threshold": P.GATE_B_HARMFUL_THRESHOLD,
            "n_resample": P.N_RESAMPLE, "alpha": P.ALPHA}
TIED_REL = 0.005      # §14 frozen 0.5% tied rule
DM_ALPHA = 0.05


# ------------------------------------------------------------------ data ----
def cell_table(ds_key: str, bb: str) -> dict | None:
    """Hour/day structures for S3M/S3C/S4 with cache-backed host predictions."""
    try:
        info = R.prepare_domain(ds_key, bb, seed=SEED)
    except Exception as e:
        return {"error": f"prepare_domain: {e}"}
    X, y, _names, valid = build_tabular(info.ds)
    ts = info.ds["ts"]
    yhat_v = info.yhat_full[valid].astype(np.float64)
    dates_arr = ts.iloc[valid].dt.date.to_numpy()

    exp = info.exp
    s3m_all = sorted(exp.dates_in_split("S3M"))
    n_mem = int(len(s3m_all) * R.S3M_MEM_FRAC)
    mem_set = set(s3m_all[:n_mem])
    val_set = set(s3m_all[n_mem:])
    s3c_set = set(exp.dates_in_split("S3C"))
    s4_dates = sorted(exp.dates_in_split("S4"))

    def _rows(d):
        return dates_arr == R.pd_date(d)

    def _block(d):
        if d in mem_set:
            return "train"
        if d in val_set:
            return "val"
        if d in s3c_set:
            return "s3c"
        return "dev"

    days = []
    all_dates = sorted(mem_set | val_set | s3c_set | set(s4_dates))
    for d in all_dates:
        m = _rows(d)
        if not m.any():
            continue
        days.append({
            "date": d, "block": _block(d),
            "X": X[m], "yhat": yhat_v[m], "y": y[m],
            "host_day": yhat_v[m], "price_day": y[m],
        })
    return {"info": info, "days": days, "s4_dates": s4_dates}


def _concat(days, blocks):
    rows = [d for d in days if d["block"] in blocks]
    if not rows:
        return None, None, None
    return (np.concatenate([d["X"] for d in rows]),
            np.concatenate([d["yhat"] for d in rows]),
            np.concatenate([d["y"] for d in rows]))


def _s4_days(table):
    return [d for d in table["days"] if d["block"] == "dev"]


def _arr_or_empty(x):
    return np.asarray(x) if x is not None else np.array([])


def _day_mae_from_hours(hours: list[dict]) -> np.ndarray:
    """Per-day MAE from hour rows (baselines)."""
    ser = {}
    for h in hours:
        ser.setdefault(h["date"], []).append(
            float(np.mean(np.abs(np.asarray(h["pred"]) - np.asarray(h["y"])))))
    return np.array([ser[k] for k in sorted(ser)]) if ser else np.array([])


# ----------------------------------------------------------- baselines -----
def run_baselines(table: dict, cell: str) -> tuple[list[dict], list[dict]]:
    """Return (metric_rows, hour_rows). hour_rows = [{date, pred, host, y}]."""
    Zf, yhf, yf = _concat(table["days"], ("train", "val"))
    Zc, yhc, yc = _concat(table["days"], ("s3c",))
    s4 = _s4_days(table)
    if Zf is None or not s4:
        return ([{"cell": cell, "method": "ERROR",
                  "note": "no fit or eval rows"}], [])
    out_met, out_hour = [], []

    def _mk(name):
        if name == "B0_Identity":
            return Identity()
        if name == "B1_ResidualL1":
            return ResidualL1(seed=0)
        if name == "B2_QuantileResidual":
            return QuantileResidualLGBM()
        if name == "B3_DeltaAdapter":
            return DeltaAdapterLimited(hidden_dim=128, epochs=30, lr=1e-3, seed=0)
        if name == "B4_PIR":
            return PIRLimited(pred_len=24, epochs=30, lr=1e-3, seed=0)
        raise KeyError(name)

    for name in BASELINES:
        m = _mk(name)
        try:
            m.fit(Zf, yhf, yf)
            if name == "B2_QuantileResidual" and Zc is not None:
                m.calibrate(Zc, yhc, yc)
            pred_days, host_days, price_days, hours = [], [], [], []
            for d in s4:
                corr = m.predict(d["X"], d["yhat"])
                pred_days.append(corr)
                host_days.append(d["host_day"])
                price_days.append(d["price_day"])
                hours.append({"date": str(d["date"]), "pred": np.asarray(corr),
                              "host": np.asarray(d["host_day"]),
                              "y": np.asarray(d["price_day"])})
            met = raw_metrics(pred_days, host_days, price_days)
            out_met.append({"cell": cell, "method": name, "status": "OK", **met})
            out_hour.extend({**h, "cell": cell, "method": name} for h in hours)
        except Exception as e:
            out_met.append({"cell": cell, "method": name, "status": "FAIL",
                            "note": str(e)[:300]})
    return out_met, out_hour


# ---------------------------------------------------------- HCH A2 chain ----
def hch_hour_rows(dv: dict) -> list[dict]:
    """Hour-level x_final from a dvg output (P0-A replay, §4)."""
    rows = dv.get("_rows", []) or []
    rel = dv.get("_released")
    out = []
    for i, r in enumerate(rows):
        rl = bool(rel[i]) if rel is not None else False
        pi_eff = r["pi"] if rl else np.zeros_like(r["pi"])
        pred = np.asarray(r["s_day"]) * np.sinh(np.asarray(r["z0"]) + pi_eff)
        host = np.asarray(r["s_day"]) * np.sinh(np.asarray(r["z0"]))
        out.append({"date": str(r["date"]), "pred": pred, "host": host,
                    "y": np.asarray(r["price"])})
    return out


def run_hch_chain(ds_key: str, bb: str, head, cell: str, method: str,
                  evaluator, dom: EvalDomain) -> dict:
    """A2 evidence-gated action chain -> final metrics + hour rows + diagnostics."""
    dd = M.collect_domain(None, ds_key, bb, "learned_sig", head=head)
    c0 = M.RawIAH()
    rows_c0 = M.evaluate_days(c0, dd)
    dv_a1 = M.dvg_and_s4(rows_c0, R.ALPHA)
    folds, _proto, _days, _n = evaluator.build_rolling_folds(dd)
    oos = evaluator.collect_paired_oos_rows(cell, folds, c0) if folds else []
    sel, _reason, _s0, _s3, _gates, _boot = evaluator.select(cell, oos, folds)
    if sel == "C3":
        s_d, y_d = P.hour_rows(
            [d for d in dd["days"] if d["block"] in ("train", "val")], "d")
        s_u, y_u = P.hour_rows(
            [d for d in dd["days"] if d["block"] in ("train", "val")], "u")
        final = M.LocalIsotonic()
        final.fit([{"domain": cell, "sd": s_d, "Yd": y_d,
                    "su": s_u, "Yu": y_u}])
        rows_final = M.evaluate_days(final, dd)
        dv = M.dvg_and_s4(rows_final, R.ALPHA)
    else:
        dv = dv_a1

    fm = final_metrics(dv)
    scalar = {k: v for k, v in fm.items()
              if k not in ("day_mae_final", "day_mae_host", "days")
              and not isinstance(v, np.ndarray)}
    pan = eval_panel_domain(head, dom, "learned_sig")
    out = {"cell": cell, "method": method, "status": "OK", "selected": sel,
           "delta_crps": pan.get("delta_crps"),
           "host_crps": pan.get("host_baseline"),
           "cand_crps": pan.get("iah_crps"),
           "a1_net": dv_a1.get("net_value"),
           "a2_net": dv.get("net_value"),
           "release_rate": dv.get("release_rate"),
           "identity_rate": dv.get("identity_rate"),
           "harmful_rate": dv.get("harmful_rate"),
           "q": dv.get("q"), "coverage": dv.get("coverage"),
           "n_calib": dv.get("n_calib"), "n_eval": dv.get("n_eval"),
           "day_mae_final": np.asarray(fm["day_mae_final"])
                           if fm.get("day_mae_final") is not None else np.array([]),
           "day_mae_host": np.asarray(fm["day_mae_host"])
                           if fm.get("day_mae_host") is not None else np.array([]),
           **scalar}
    out["hours"] = hch_hour_rows(dv)
    return out


# ------------------------------------------------------------ DM (day-level)
def _nw_var(d: np.ndarray, lag: int) -> float:
    T = len(d)
    gam = [float(np.mean(d[: T - k] * d[k:])) for k in range(lag + 1)]
    w = 1.0 - np.arange(lag + 1) / (lag + 1)
    return gam[0] + 2.0 * sum(gam[k] * w[k] for k in range(1, lag + 1))


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """Day-level Diebold-Mariano on absolute-error loss.

    d_t = loss_a - loss_b (negative d => method A better that day).
    Newey-West HAC variance, lag = floor(T^(1/3)).
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    T = min(len(a), len(b))
    if T < 4:
        return {"n_days": int(T), "dm": None, "p": None,
                "mean_diff": None, "b5_wins": None}
    d = a[:T] - b[:T]
    mu = float(d.mean())
    var = _nw_var(d, max(1, int(T ** (1.0 / 3.0))))
    se = np.sqrt(max(var, 1e-12) / T)
    dm = mu / se if se > 0 else 0.0
    p = 2.0 * (1.0 - norm.cdf(abs(dm)))
    return {"n_days": int(T), "dm": round(dm, 4), "p": round(p, 5),
            "mean_diff": round(mu, 5), "b5_wins": round(float((d < 0).mean()), 4)}


# ------------------------------------------------------------ ranks / gate ----
def tied_best(score_b5: float, best: float) -> bool:
    return score_b5 <= best * (1.0 + TIED_REL)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--universal-head", type=str, default=None,
                    help="B5 head checkpoint. Default: latest 09-paper-gate "
                         "WP5_PUBLIC head if present, else P0A_RERUN.")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--hosts", nargs="+", default=None,
                    help="restrict hosts (default all 4)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-b6", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else PAPER / "results" / \
        f"B3_MATRIX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- B5 head resolution: latest PASSED WP-5 public head > P0A_RERUN head ----
    def _guard_ok(head_pt: Path) -> bool:
        """§12.2: skip WP5 heads whose guard_report.json pass=false (ROLLBACK)."""
        guard_f = head_pt.parent / "guard_report.json"
        if not guard_f.exists():
            return True                    # no report -> assume ok (pre-guard run)
        try:
            return bool(json.loads(guard_f.read_text()).get("pass", True))
        except Exception:
            return True

    wp5_cands = sorted(HCH.glob("results/WP5_PUBLIC_*/learned_sig_main_head.pt"))
    wp5_head = None
    for cand in reversed(wp5_cands):       # newest first
        if _guard_ok(cand):
            wp5_head = cand
            break
        print(f"[head] skip ROLLBACK WP5 head {cand.parent.name}", flush=True)
    p0a_head = HCH / "results" / "P0A_RERUN_20260814" / "learned_sig_main_head.pt"
    head_path = args.universal_head
    if head_path is None:
        head_path = str(wp5_head) if wp5_head is not None else \
            str(p0a_head) if p0a_head.exists() else None
    if head_path is None:
        print("[FATAL] no universal head checkpoint. Train one first (WP-5 step 3).")
        sys.exit(2)

    ds_list = args.datasets or HEADLINE_DS
    hosts_list = args.hosts or HOSTS
    cells = [(ds, bb) for ds in ds_list for bb in hosts_list]
    evaluator = P.PrequentialCalibrationEvaluator(PREQ_CFG)

    met_rows = []                       # every method metric row (07)
    hour_rows = []                      # every hour row (06 parquet)
    b5_diag = []                        # B5 diagnostics (13/14/15/16)
    per_cell = {}                       # cell -> {method -> {"met": dict, "day_mae": ndarray}}

    head = build_head("learned_sig")
    import torch
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.eval()

    for ds, bb in cells:
        cell = f"{ds}:{bb}"
        print(f"\n===== {cell} =====", flush=True)
        table = cell_table(ds, bb)
        if table is None or table.get("error"):
            print(f"  SKIP: {table.get('error') if table else 'no cache'}", flush=True)
            met_rows.append({"cell": cell, "method": "SKIP",
                             "note": table.get("error") if table else "no cache"})
            continue
        dom = EvalDomain(info=table["info"], market=ds, host=bb, name=cell)
        per_cell[cell] = {}

        if not args.skip_baselines:
            bm, bh = run_baselines(table, cell)
            for r in bm:
                print(f"  {r['method']}: mae={r.get('mae')} "
                      f"smape={r.get('smape_nofloor')} {r.get('status')}", flush=True)
                met_rows.append(r)
                if r.get("status") == "OK":
                    hours_m = [h for h in bh if h["method"] == r["method"]]
                    per_cell[cell][r["method"]] = {
                        "met": r, "day_mae": _day_mae_from_hours(hours_m)}
            hour_rows.extend(bh)

        # ---- B5 universal ----
        r5 = run_hch_chain(ds, bb, head, cell, "B5_HCH-Universal", evaluator, dom)
        print(f"  B5: mae={r5.get('mae')} smape={r5.get('smape_nofloor')} "
              f"sel={r5.get('selected')} dCRPS={r5.get('delta_crps')}", flush=True)
        met_rows.append({k: v for k, v in r5.items() if k not in ("hours",)})
        per_cell[cell]["B5_HCH-Universal"] = {
            "met": r5, "day_mae": _arr_or_empty(r5.get("day_mae_final"))}
        b5_diag.append(r5)
        for h in r5.get("hours", []):
            hour_rows.append({**h, "cell": cell, "method": "B5_HCH-Universal"})

        # ---- B6 local ----
        if not args.skip_b6:
            head_l, _rep = train_candidate("learned_sig", [dom], seed=SEED)
            r6 = run_hch_chain(ds, bb, head_l, cell, "B6_HCH-Local", evaluator, dom)
            print(f"  B6: mae={r6.get('mae')} smape={r6.get('smape_nofloor')} "
                  f"sel={r6.get('selected')}", flush=True)
            met_rows.append({k: v for k, v in r6.items() if k not in ("hours",)})
            per_cell[cell]["B6_HCH-Local"] = {
                "met": r6, "day_mae": _arr_or_empty(r6.get("day_mae_final"))}
            for h in r6.get("hours", []):
                hour_rows.append({**h, "cell": cell, "method": "B6_HCH-Local"})

    # ============================ write outputs ============================
    def _write(name, df):
        path = out_dir / name
        df.to_csv(path, index=False)
        return path

    # 06 predictions parquet: one row per hour (cell, method, date, hour, ...)
    pred_records = []
    for h in hour_rows:
        pred = np.asarray(h["pred"], dtype=np.float64)
        host = np.asarray(h["host"], dtype=np.float64)
        y = np.asarray(h["y"], dtype=np.float64)
        for i in range(len(pred)):
            pred_records.append({"cell": h["cell"], "method": h["method"],
                                 "date": h["date"], "hour": int(i),
                                 "pred": float(pred[i]), "host": float(host[i]),
                                 "y": float(y[i])})
    pred_df = pd.DataFrame(pred_records)
    pred_df.to_parquet(out_dir / "06_PREDICTIONS.parquet", index=False)

    # 07 metrics
    met_cols = ["cell", "method", "status", "selected", "n_hours", "mae",
                "smape_nofloor", "rmse", "rmae", "neg_price_rate",
                "neg_price_mae", "high_tail_rate", "high_tail_mae",
                "host_mae", "host_rmae", "degradation", "degradation_frac",
                "delta_crps", "a1_net", "a2_net", "note"]
    _write("07_METRICS_BY_CELL.csv",
           pd.DataFrame([{k: r.get(k) for k in met_cols} for r in met_rows]))

    # ---- DM tests (B5 vs each peer + B5 vs Host) ----
    dm_rows = []
    for cell in sorted(per_cell):
        loss5 = per_cell[cell].get("B5_HCH-Universal", {}).get("day_mae")
        if loss5 is None or not len(loss5):
            continue
        for m in ("B0_Identity", "B1_ResidualL1", "B2_QuantileResidual",
                  "B3_DeltaAdapter", "B4_PIR", "B6_HCH-Local"):
            lm = per_cell[cell].get(m, {}).get("day_mae")
            if lm is None or not len(lm):
                continue
            res = dm_test(loss5, lm)
            dm_rows.append({"cell": cell, "vs": m,
                            "B5_sig_better": bool(res.get("p") is not None
                                                  and res["p"] < DM_ALPHA
                                                  and res["mean_diff"] < 0),
                            **res})
    _write("12_DM_TESTS.csv", pd.DataFrame(dm_rows))

    # ---- ranks over primary cells ----
    rank_rows = []
    for cell in sorted(per_cell):
        cell_mets = per_cell[cell]
        for metric, key in (("MAE", "mae"), ("sMAPE", "smape_nofloor")):
            scores = {m: float(cell_mets[m]["met"][key])
                      for m in cell_mets if key in cell_mets[m]["met"]}
            if len(scores) < 2 or "B5_HCH-Universal" not in scores:
                continue
            order = sorted(scores, key=lambda m: scores[m])
            best, b5 = order[0], "B5_HCH-Universal"
            rank_rows.append({
                "cell": cell, "metric": metric,
                "b5_score": round(scores[b5], 6), "best_method": best,
                "best_score": round(scores[best], 6),
                "gap_to_best_pct": round((scores[b5] - scores[best]) /
                                         scores[best] * 100, 3) if scores[best] > 0 else None,
                "b5_rank": order.index(b5) + 1, "n_methods": len(order),
                "strict_top1": bool(best == b5),
                "tied_top1": tied_best(scores[b5], scores[best]),
                "top2": order.index(b5) < 2,
            })
    _write("08_RANKS_BY_CELL.csv", pd.DataFrame(rank_rows))

    # ---- win rate + host-better fraction ----
    n = len(rank_rows)
    top1 = sum(1 for r in rank_rows if r["strict_top1"] or r["tied_top1"])
    top2 = sum(1 for r in rank_rows if r["top2"])
    # host-better: B5 MAE < B0 MAE, over MAE cells only (n/2 cells)
    hb_num = hb_den = 0
    for r in rank_rows:
        if r["metric"] != "MAE":
            continue
        cell_mets = per_cell[r["cell"]]
        b0 = cell_mets.get("B0_Identity", {}).get("met", {}).get("mae")
        if b0 is not None:
            hb_den += 1
            if r["b5_score"] < b0:
                hb_num += 1
    hb_frac = hb_num / hb_den if hb_den else None
    wr = pd.DataFrame([{
        "n_primary_cells": n, "n_strict_top1":
            sum(1 for r in rank_rows if r["strict_top1"]),
        "n_tied_top1": top1, "n_top2": top2,
        "rate_top1_tied": round(top1 / n, 4) if n else None,
        "rate_top2": round(top2 / n, 4) if n else None,
        "host_better_frac": round(hb_frac, 4) if hb_frac is not None else None,
    }])
    _write("09_PRIMARY_WIN_RATE.csv", wr)

    # ---- HCH diagnostics ----
    diag_cols = ["cell", "selected", "delta_crps", "cand_crps", "host_crps",
                 "a1_net", "a2_net", "release_rate", "identity_rate",
                 "harmful_rate", "q", "coverage", "mae", "host_mae",
                 "degradation_frac"]
    _write("13_HCH_CANDIDATE_DIAGNOSTICS.csv",
           pd.DataFrame([{k: d.get(k) for k in diag_cols} for d in b5_diag]))
    act_cols = ["cell", "selected", "release_rate", "identity_rate",
                "harmful_rate", "q", "coverage", "n_calib", "n_eval"]
    _write("14_HCH_ACTION_DIAGNOSTICS.csv",
           pd.DataFrame([{k: d.get(k) for k in act_cols} for d in b5_diag]))

    # ---- FAILURE_MAP (protocol §16) ----
    fm_rows = []
    for d in b5_diag:
        mae, hmae, dc = d.get("mae"), d.get("host_mae"), d.get("delta_crps")
        if mae is None or hmae is None or dc is None:
            tag, note = "NO_DATA", "missing metrics"
        else:
            point_ok = mae <= hmae * (1.0 + 1e-9)
            rel = (mae - hmae) / max(hmae, 1e-9)
            if d.get("selected") == "C3" and abs(rel) < 1e-6:
                tag, note = "ABSTAIN_SAFE", "DVG abstains; final == host"
            elif dc < 0 and point_ok and rel < -0.01:
                tag, note = "CANDIDATE", "CRPS<0 and point improves"
            elif dc < 0 and not point_ok:
                tag, note = "POINT_READOUT", "CRPS<0 but point MAE worse -> Case A"
            elif dc >= 0 and point_ok:
                tag, note = "POINT_ONLY", "point ok, CRPS not negative"
            elif dc >= 0:
                tag, note = "CRPS_WEAK", "CRPS>=0 on S2V panel"
            else:
                tag, note = "NEUTRAL", "near-flat"
        fm_rows.append({"cell": d["cell"], "tag": tag, "note": note,
                        "mae": mae, "host_mae": hmae, "delta_crps": dc,
                        "degradation_frac": d.get("degradation_frac"),
                        "selected": d.get("selected")})
    _write("15_FAILURE_MAP.csv", pd.DataFrame(fm_rows))

    # ---- generalization ledger (source vs transfer markets) ----
    _write("16_GENERALIZATION_LEDGER.csv",
           pd.DataFrame([{"cell": d["cell"],
                          "market_seen": int(d["cell"].split(":")[0] in SOURCE_MK),
                          "delta_crps": d.get("delta_crps"),
                          "mae_rel": d.get("degradation_frac")} for d in b5_diag]))

    # ---- config ----
    with open(out_dir / "config.json", "w") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "datasets": ds_list, "hosts": hosts_list,
                   "methods": list(BASELINES) + ["B5_HCH-Universal", "B6_HCH-Local"],
                   "universal_head": head_path,
                   "evidence": "fit=S3M(train+val) cal=S3C eval=S4",
                   "tied_rel": TIED_REL, "dm_alpha": DM_ALPHA}, f, indent=2)
    print(f"\n[matrix] done -> {out_dir}")


if __name__ == "__main__":
    main()
