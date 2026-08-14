"""P1 Round-2, T2: frozen S4 readout evaluation with weighted_mean.

Same evaluation path as P1 round-1 run_r1 (collect_domain + s4_eval), but the
candidate heads come from full-coverage training (results/P1_T2_0478cf7) instead
of T0 equal-sampling (TRAINER_CMP vA). Paired comparison T2 vs T0 on S4.

Readout fixed: weighted_mean (round-1 zero-degradation readout, S2V-optimal).
S4 labels never used for training/selection. Nothing here tunes on S4.

Outputs in <t2_dir>:
  t2_s4_metrics.json     per (cell, seed) S4 weighted_mean for T2
  t2_vs_t0_s4_paired.csv per (cell, seed) paired T2/T0 + delta
  t2_s4_summary.json     macro means + win rates (weighted_mean)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import r1a_run as R                          # noqa: E402
import r1a9_action_calibration as M          # noqa: E402
from r1b_generalization_screen import build_head, det_for_variant, SEED  # noqa: E402
from p1_round1 import cells_all, cell_eps, s4_eval, READOUTS  # noqa: E402

T2_DIR = HERE / "results" / "P1_T2_0478cf7"
P1_DIR = HERE / "results" / "P1_20260814_0478cf7"
SEEDS = [0, 1, 2]
READOUT = "weighted_mean"


def load_t2_head(seed: int) -> torch.nn.Module:
    pt = T2_DIR / f"t2_head_seed{seed}.pt"
    head = build_head("learned_sig")
    head.load_state_dict(torch.load(pt, map_location="cpu"))
    head.eval()
    return head


def main():
    cells = cells_all()
    eps_by_cell = {}
    infos = {}
    for cell in cells:
        mk, bb = cell.split(":")
        info = R.prepare_domain(mk, bb, seed=SEED)
        infos[cell] = info
        eps_by_cell[cell] = cell_eps(info)

    t2_rows = []
    for cell in cells:
        mk, bb = cell.split(":")
        info = infos[cell]
        det_np = det_for_variant("learned_sig", info)
        for seed in SEEDS:
            head = load_t2_head(seed)
            dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
            del head
            s4 = s4_eval(dd, READOUT, eps_by_cell[cell])
            t2_rows.append({
                "cell": cell, "seed": seed, "readout": READOUT,
                "n_hours": s4.get("n_hours", 0),
                "mae": s4.get("mae"), "rmse": s4.get("rmse"),
                "smape_eps": s4.get("smape_eps"),
                "host_mae": s4.get("host_mae"),
                "improvement_vs_host": s4.get("improvement_vs_host"),
                "high_tail_mae": s4.get("high_tail_mae"),
                "neg_price_mae": s4.get("neg_price_mae"),
                "n_days": s4.get("n_days"),
            })
            print(f"  {cell} seed{seed}: mae={s4.get('mae')} "
                  f"rmse={s4.get('rmse')} smape={s4.get('smape_eps')}", flush=True)

    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(T2_DIR / "t2_s4_metrics.csv", index=False)
    with open(T2_DIR / "t2_s4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(t2_rows, f, ensure_ascii=False, indent=2)

    # --- paired vs round-1 T0 (same path, weighted_mean) ---
    ref = json.load(open(P1_DIR / "hch_s4_metrics.json", encoding="utf-8"))
    pairs = []
    for r in t2_rows:
        key = f"{r['cell']}|{r['seed']}|{READOUT}"
        ref_row = ref.get(key)
        if ref_row is None:
            continue
        pairs.append({
            "cell": r["cell"], "seed": r["seed"],
            "t2_mae": r["mae"], "t0_mae": ref_row.get("mae"),
            "d_mae": (r["mae"] - ref_row.get("mae")) if r["mae"] is not None and ref_row.get("mae") is not None else None,
            "t2_rmse": r["rmse"], "t0_rmse": ref_row.get("rmse"),
            "d_rmse": (r["rmse"] - ref_row.get("rmse")) if r["rmse"] is not None and ref_row.get("rmse") is not None else None,
            "t2_smape": r["smape_eps"], "t0_smape": ref_row.get("smape_eps"),
            "d_smape": (r["smape_eps"] - ref_row.get("smape_eps")) if r["smape_eps"] is not None and ref_row.get("smape_eps") is not None else None,
            "t2_imp_host": r["improvement_vs_host"],
            "t0_imp_host": ref_row.get("improvement_vs_host"),
            "host_mae": r["host_mae"],
        })
    pdf = pd.DataFrame(pairs)
    pdf.to_csv(T2_DIR / "t2_vs_t0_s4_paired.csv", index=False)

    # --- summary ---
    def macro(col):
        v = pdf[col].dropna()
        return float(v.mean()) if len(v) else None

    def win_rate(dcol):
        v = pdf[dcol].dropna()
        if not len(v):
            return None
        return float((v < 0).mean())   # negative delta = T2 better

    summary = {
        "readout": READOUT,
        "n_pairs": int(len(pdf)),
        "macro_mae": {"t2": macro("t2_mae"), "t0": macro("t0_mae"),
                      "delta_t2_minus_t0": macro("d_mae")},
        "macro_rmse": {"t2": macro("t2_rmse"), "t0": macro("t0_rmse"),
                       "delta_t2_minus_t0": macro("d_rmse")},
        "macro_smape": {"t2": macro("t2_smape"), "t0": macro("t0_smape"),
                        "delta_t2_minus_t0": macro("d_smape")},
        "win_rate_t2_mae": win_rate("d_mae"),
        "win_rate_t2_rmse": win_rate("d_rmse"),
        "win_rate_t2_smape": win_rate("d_smape"),
        "n_better_mae": int((pdf["d_mae"] < 0).sum()),
        "n_worse_mae": int((pdf["d_mae"] > 0).sum()),
    }
    with open(T2_DIR / "t2_s4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[done] t2_s4_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
