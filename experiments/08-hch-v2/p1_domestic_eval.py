"""D2: domestic benchmark — frozen round-1 head transferred to Chinese provinces.

cells = 10 modes (shandong_DA/RT + 4 provinces × DA/RT) × HOSTS4 (4 hosts) = 40.
Head  = round-1 vA seed0 FROZEN head (p1_round1.load_head(0), trained on the 12
        international source domains). Readout fixed = weighted_mean.
        S4 labels never used for training/selection; nothing tunes on S4.

Independent cells list — does NOT touch cells_all() (the 32 international cells).

Outputs in results/domestic/:
  domestic_s4_metrics.csv/json   per (cell) S4 weighted_mean
  domestic_summary.json          macro per mode + improvement_vs_host + honest
                                 notes (S1R days, neg-price availability)
"""
from __future__ import annotations

import csv
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
from common import PROVINCE_KEYS              # noqa: E402
from p1_round1 import HOSTS4, cell_eps, s4_eval, load_head  # noqa: E402

OUT = HERE / "results" / "domestic"
OUT.mkdir(parents=True, exist_ok=True)
READOUT = "weighted_mean"
SEED = 0

# 独立 cells 列表: 10 国内模式 × 4 host,不动 cells_all() 的 32 国外 cell。
DOMESTIC_MODES = ["shandong_DA", "shandong_RT"] + list(PROVINCE_KEYS)
CELLS = [f"{mk}:{bb}" for mk in DOMESTIC_MODES for bb in HOSTS4]


def load_s1r_days() -> dict:
    """S1R day counts from the admission audit (honest thin-reference notes)."""
    path = OUT / "admission_report.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["dataset"]: int(r["S1R_days"]) for r in csv.DictReader(f)}


def main():
    s1r = load_s1r_days()
    head = load_head(SEED, torch)

    rows = []
    for cell in CELLS:
        mk, bb = cell.split(":")
        info = R.prepare_domain(mk, bb, seed=SEED)
        eps = cell_eps(info)
        dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
        s4 = s4_eval(dd, READOUT, eps)
        row = {
            "cell": cell, "mode": mk, "host": bb, "readout": READOUT,
            "n_hours": s4.get("n_hours", 0), "n_days": s4.get("n_days", 0),
            "s1r_days": s1r.get(mk),
            "mae": s4.get("mae"), "rmse": s4.get("rmse"),
            "smape_eps": s4.get("smape_eps"),
            "host_mae": s4.get("host_mae"),
            "improvement_vs_host": s4.get("improvement_vs_host"),
            "high_tail_mae": s4.get("high_tail_mae"),
            # 诚实: 0% 负价省份无 neg_price_mae(数据根本没有负价),山东才有。
            "neg_price_mae": s4.get("neg_price_mae"),
        }
        rows.append(row)
        print(f"  {cell:26s} n={row['n_hours']:5d} mae={row['mae']} "
              f"host={row['host_mae']} imp={row['improvement_vs_host']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "domestic_s4_metrics.csv", index=False)
    with open(OUT / "domestic_s4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # ---- summary: per-mode macro + per-host macro + overall ----
    def macro(sub, col):
        v = sub[col].dropna()
        return round(float(v.mean()), 6) if len(v) else None

    per_mode, per_host = {}, {}
    for mk in DOMESTIC_MODES:
        sub = df[df["mode"] == mk]
        imp = sub["improvement_vs_host"].dropna()
        per_mode[mk] = {
            "n_cells": int(len(sub)),
            "macro_mae": macro(sub, "mae"),
            "macro_rmse": macro(sub, "rmse"),
            "macro_smape": macro(sub, "smape_eps"),
            "macro_host_mae": macro(sub, "host_mae"),
            "mean_improvement_vs_host": round(float(imp.mean()), 5) if len(imp) else None,
            "n_cells_correction_helps": int((imp > 0).sum()),
            "n_cells_correction_hurts": int((imp < 0).sum()),
            "s1r_days": s1r.get(mk),
            "neg_price_mae_available": bool(sub["neg_price_mae"].notna().any()),
        }
    for bb in HOSTS4:
        sub = df[df["host"] == bb]
        per_host[bb] = {
            "macro_mae": macro(sub, "mae"),
            "macro_host_mae": macro(sub, "host_mae"),
            "mean_improvement_vs_host": macro(sub, "improvement_vs_host"),
        }

    summary = {
        "cells": len(rows), "readout": READOUT, "head_seed": SEED,
        "head_source": "round-1 vA seed0 (trained on 12 international source domains)",
        "per_mode": per_mode, "per_host": per_host,
        "macro_overall": {
            "macro_mae": macro(df, "mae"),
            "macro_rmse": macro(df, "rmse"),
            "macro_smape": macro(df, "smape_eps"),
            "macro_host_mae": macro(df, "host_mae"),
            "mean_improvement_vs_host": macro(df, "improvement_vs_host"),
            "n_cells_correction_helps": int((df["improvement_vs_host"] > 0).sum()),
            "n_cells_correction_hurts": int((df["improvement_vs_host"] < 0).sum()),
        },
    }
    with open(OUT / "domestic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[done] domestic_s4_metrics.json / domestic_summary.json")
    print(json.dumps(summary["macro_overall"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
