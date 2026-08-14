"""WP-4 B2 paper smoke — 5 markets × Linear/MLP × B0–B6 (protocol §7).

Cells:
  LAGO_DE / LAGO_PJM / NEM_SA1 / shandong_DA / shandong_RT  ×  Linear / MLP

Methods:
  B0 Identity         host forecast unchanged
  B1 ResidualL1       LGBM L1 on frozen-base residuals (baselines_v2)
  B2 QuantileResidual q=0.1/0.5/0.9 residual quantiles, S3C width cutoff
  B3 δ-Adapter        production DeltaAdapterLimited (PostY/Ada-Y)
  B4 PIR              production PIRLimited (QE+Refiner, no retrieval)
  B5 HCH-Universal    shared public LearnedSig_main (seed0, 12 source) + A2 chain
  B6 HCH-Local        same arch trained on target S2T/S2V only + A2 chain

Final point metrics (protocol §4): HCH uses x_final = s·sinh(z0+pi_eff) via
_final_point.final_point_metrics; baselines use raw-space raw_metrics with the
identical metric definitions.

Evidence windows (mirror the HCH action chain, fair comparison):
  fit  = S3M first-n_mem days ("train" mem) + rest ("val")  → same as C3 local
  cal  = S3C days (B2 width cutoff)
  eval = S4 days (dev)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
    SOURCE_MARKETS, HOSTS, SEED, EvalDomain, build_head, train_candidate,
)
from baselines_v2 import Identity, ResidualL1, QuantileResidualLGBM  # noqa: E402
from official_adapters import DeltaAdapterLimited, PIRLimited        # noqa: E402
from _final_point import raw_metrics, final_metrics                  # noqa: E402

SMOKE_DS = ["LAGO_DE", "LAGO_PJM", "NEM_SA1", "shandong_DA", "shandong_RT"]
SMOKE_BB = ["Linear", "MLP"]
PREQ_CFG = {"burn_in": P.BURN_IN_DAYS, "block": P.BLOCK_DAYS,
            "min_oos_days": P.GATE_A_MIN_OOS_DAYS,
            "harm_threshold": P.GATE_B_HARMFUL_THRESHOLD,
            "n_resample": P.N_RESAMPLE, "alpha": P.ALPHA}

# ------------------------------------------------------------------ data ----
def cell_table(ds_key: str, bb: str) -> dict | None:
    """Hour/day structures for S3M/S3C/S4 with cache-backed host predictions.

    Returns None if cache missing/invalid. X/y/yhat are valid-row compressed
    (len = n_valid), consistent with the host cache contract (P0-B).
    """
    try:
        info = R.prepare_domain(ds_key, bb, seed=SEED)
    except Exception as e:
        return {"error": f"prepare_domain: {e}"}
    X, y, _names, valid = build_tabular(info.ds)
    ts = info.ds["ts"]
    y_full = info.ds["price"]
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


def _s4_day_slices(days):
    return [d for d in days if d["block"] == "dev"]


# ----------------------------------------------------------- baselines -----
def run_baselines(table: dict, cell: str) -> list[dict]:
    Zf, yhf, yf = _concat(table["days"], ("train", "val"))
    Zc, yhc, yc = _concat(table["days"], ("s3c",))
    s4 = _s4_day_slices(table["days"])
    if Zf is None or not s4:
        return [{"cell": cell, "method": "ERROR", "note": "no fit or eval rows"}]

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

    rows = []
    for name in ("B0_Identity", "B1_ResidualL1", "B2_QuantileResidual",
                 "B3_DeltaAdapter", "B4_PIR"):
        m = _mk(name)
        try:
            m.fit(Zf, yhf, yf)
            if name == "B2_QuantileResidual" and Zc is not None:
                m.calibrate(Zc, yhc, yc)
            pred_days, host_days, price_days = [], [], []
            for d in s4:
                corr = m.predict(d["X"], d["yhat"])
                pred_days.append(corr)
                host_days.append(d["host_day"])
                price_days.append(d["price_day"])
            met = raw_metrics(pred_days, host_days, price_days)
            rows.append({"cell": cell, "method": name, "status": "OK", **met})
        except Exception as e:
            rows.append({"cell": cell, "method": name, "status": "FAIL",
                         "note": str(e)[:300]})
    return rows


# ---------------------------------------------------------- HCH A2 chain ----
def run_hch_chain(ds_key: str, bb: str, head, cell: str, method: str,
                  evaluator) -> dict:
    """A2 evidence-gated action chain (identical to stage2d) -> final metrics."""
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
    return {"cell": cell, "method": method, "status": "OK", "selected": sel,
            **scalar}


# --------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--universal-head", type=str, default=None,
                    help="path to saved LearnedSig_main head (B5). Default: "
                         "P0A_RERUN checkpoint if present.")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict smoke datasets (default all 5)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-hch", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else PAPER / "results" / \
        f"B2_SMOKE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_head = HCH / "results" / "P0A_RERUN_20260814" / "learned_sig_main_head.pt"
    head_path = args.universal_head or (str(default_head) if default_head.exists()
                                        else None)

    ds_list = args.datasets or SMOKE_DS
    cells = [(ds, bb) for ds in ds_list for bb in SMOKE_BB]
    evaluator = P.PrequentialCalibrationEvaluator(PREQ_CFG)
    all_rows = []

    for ds, bb in cells:
        cell = f"{ds}:{bb}"
        print(f"\n===== {cell} =====", flush=True)
        table = cell_table(ds, bb)
        if table is None or table.get("error"):
            print(f"  SKIP: {table.get('error') if table else 'no cache'}", flush=True)
            all_rows.append({"cell": cell, "method": "SKIP",
                             "note": table.get("error") if table else "no cache"})
            continue
        if not args.skip_baselines:
            for r in run_baselines(table, cell):
                print(f"  {r['method']}: mae={r.get('mae')} "
                      f"smape={r.get('smape_nofloor')} {r.get('status')}", flush=True)
                all_rows.append(r)
        if not args.skip_hch:
            dom = EvalDomain(info=table["info"], market=ds, host=bb, name=cell)
            if head_path:
                head = build_head("learned_sig")
                head.load_state_dict(
                    __import__("torch").load(head_path, map_location="cpu"))
                head.eval()
                r = run_hch_chain(ds, bb, head, cell, "B5_HCH-Universal", evaluator)
                print(f"  B5: mae={r.get('mae')} smape={r.get('smape_nofloor')} "
                      f"sel={r.get('selected')}", flush=True)
                all_rows.append(r)
            else:
                print("  B5 SKIP (no universal head checkpoint)", flush=True)
            print(f"  [B6 local train] {cell} ...", flush=True)
            head_l, _rep = train_candidate("learned_sig", [dom], seed=SEED)
            r6 = run_hch_chain(ds, bb, head_l, cell, "B6_HCH-Local", evaluator)
            print(f"  B6: mae={r6.get('mae')} smape={r6.get('smape_nofloor')} "
                  f"sel={r6.get('selected')}", flush=True)
            all_rows.append(r6)

    # ---- write CSV (stable column order) ----
    import csv
    cols = ["cell", "method", "status", "selected", "n_hours", "mae",
            "smape_nofloor", "rmse", "rmae", "neg_price_rate", "neg_price_mae",
            "high_tail_rate", "high_tail_mae", "host_mae", "host_rmae",
            "degradation", "degradation_frac", "note"]
    with open(out_dir / "smoke_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in cols})
    cfg = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "datasets": ds_list, "hosts": SMOKE_BB,
           "methods": ["B0-B4 baselines", "B5 HCH-Universal", "B6 HCH-Local"],
           "universal_head": head_path,
           "evidence": "fit=S3M(train+val) cal=S3C eval=S4 (mirrors A2 chain)"}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[smoke] done -> {out_dir / 'smoke_matrix.csv'}")


if __name__ == "__main__":
    main()
