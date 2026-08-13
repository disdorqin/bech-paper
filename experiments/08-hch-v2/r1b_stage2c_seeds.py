"""R1B Stage-2C — HCH seed stability (§15).

Only run after Stage-2A and Stage-2B are healthy. Varies the HCH training seed
(1, 2) for the primary candidate LearnedSig_main (host caches stay seed=0) and
evaluates the full 40-cell panel (12 source + 20 Linear/MLP holdout from
Stage-2A + 8 deep LSTM/PatchTST from Stage-2B). Seed-0 rows are loaded from the
parent Stage-2A/Stage-2B panel CSVs, so the artifact is a self-contained 3-seed
table.

Coverage (§15): source 12 domains ✓, DK1 4 domains ✓ (Linear/MLP from Stage-2A
H1 + LSTM/PatchTST from Stage-2B deep), broad fast Linear/MLP panel ✓.

STOP (documented operationalization):
  SEED_INSTABILITY — for any holdout category (UNSEEN_MARKET,
  UNSEEN_DATASET_SAME_MARKET, UNSEEN_SCHEMA_REGIME), either
    * the macro sign flips across seeds 0/1/2, or
    * any seed has macro >= 0, or
    * any seed has frac<0 < 2/3
  while seed 0 was healthy. seed0 GREEN alone is not robust generalization
  evidence if seed1/2 break.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEED, EPOCHS, PATIENCE,
    EvalDomain, with_membership, train_candidate, _summarize_reports,
)
from r1b_stage2a_panel import provenance, eval_panel_domain, agg_by_category, _write_csv
from r1b_stage2b_panel import REP_MARKETS, DEEP_HOSTS

SEEDS = [1, 2]
HOLDOUT_CATS = ["UNSEEN_MARKET", "UNSEEN_DATASET_SAME_MARKET", "UNSEEN_SCHEMA_REGIME"]

# Stage-2A holdout Linear/MLP cells (H1+H2+H3), same set as the 32-cell panel
STAGE2A_HOLD_LM = [
    ("EPEX_FR", "Linear"), ("EPEX_FR", "MLP"),
    ("EPEX_BE", "Linear"), ("EPEX_BE", "MLP"),
    ("EPEX_NL", "Linear"), ("EPEX_NL", "MLP"),
    ("NORD_FI", "Linear"), ("NORD_FI", "MLP"),
    ("NORD_NO", "Linear"), ("NORD_NO", "MLP"),
    ("NORD_SE3", "Linear"), ("NORD_SE3", "MLP"),
    ("NORD_DK1", "Linear"), ("NORD_DK1", "MLP"),
    ("DE_EPEX", "Linear"), ("DE_EPEX", "MLP"),
    ("PJM_2020", "Linear"), ("PJM_2020", "MLP"),
    ("GEFCOM14P", "Linear"), ("GEFCOM14P", "MLP"),
]
DEEP_CELLS = [(mk, bb) for mk in REP_MARKETS for bb in DEEP_HOSTS]
SOURCE_CELLS = [(mk, bb) for mk in SOURCE_MARKETS for bb in HOSTS]
EVAL_CELLS = SOURCE_CELLS + STAGE2A_HOLD_LM + DEEP_CELLS  # 12 + 20 + 8 = 40


def load_seed0(parent2a: Path, parent2b: Path) -> dict[str, dict]:
    """Extract seed-0 rows for the 40 cells from parent Stage-2A/2B CSVs."""
    rows = {}
    for p in (parent2a / "panel_matrix_LearnedSig_main.csv",
              parent2b / "panel_matrix_LearnedSig_main.csv"):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["evaluation_dataset"]] = r
    out = {}
    for mk, bb in EVAL_CELLS:
        name = f"{mk}:{bb}"
        if name not in rows:
            raise ValueError(f"seed0 cell {name} missing from parent CSVs")
        out[name] = rows[name]
    return out


def _cat_stats(rows_by_cell: dict[str, dict], cat: str) -> dict | None:
    ds = [float(r["delta_crps"]) for r in rows_by_cell.values()
          if r["transfer_category"] == cat and r["delta_crps"] not in (None, "")]
    if not ds:
        return None
    return {"n": len(ds), "macro": round(float(np.mean(ds)), 6),
            "frac_lt_0": round(float(np.mean([1.0 if d < 0 else 0.0 for d in ds])), 4),
            "worst": round(float(np.max(ds)), 6)}


def stability(per_seed: dict[int, dict[str, dict]]) -> dict:
    """§15 3/3 sign consistency + no seed-1/2 holdout degradation."""
    flags = {}
    for cat in HOLDOUT_CATS:
        rows = [per_seed[s].get(cat) for s in (SEED, *SEEDS)]
        if any(r is None for r in rows):
            flags[cat] = "MISSING"
            continue
        macros = [r["macro"] for r in rows]
        fracs = [r["frac_lt_0"] for r in rows]
        signs_same = (macros[0] < 0) == (macros[1] < 0) == (macros[2] < 0)
        all_neg = all(m < 0 for m in macros)
        frac_ok = all(f >= 2 / 3 for f in fracs)
        flags[cat] = {
            "macro_0_1_2": macros, "frac_lt_0_0_1_2": fracs,
            "sign_consistent_3of3": signs_same, "all_macro_lt_0": all_neg,
            "frac_lt_0_ok": frac_ok,
            "ok": bool(signs_same and all_neg and frac_ok),
        }
    ok = all(v.get("ok", False) for v in flags.values())
    return {"per_category": flags, "SEED_STABLE": ok,
            "stop_note": ("SEED_INSTABILITY — pause before Stage-2D" if not ok
                          else "STABLE — 3/3 sign consistency, no seed-1/2 collapse")}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None)
    ap.add_argument("--parent-stage2a", type=str, required=True)
    ap.add_argument("--parent-stage2b", type=str, required=True)
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2C_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_root = HERE / "results"
    seed0 = load_seed0(res_root / args.parent_stage2a, res_root / args.parent_stage2b)

    # ---- prepare 40 cells (caches fixed seed=0) ----
    doms = {}
    for mk, bb in EVAL_CELLS:
        name = f"{mk}:{bb}"
        print(f"[prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    eval_doms = with_membership([doms[n] for n in doms], SOURCE_MARKETS, HOSTS)
    train_doms = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]

    # ---- train LearnedSig_main at seeds 1, 2 ----
    per_seed: dict[int, dict[str, dict]] = {}
    reports = {}
    for s in SEEDS:
        print(f"\n===== STAGE-2C train LearnedSig_main seed={s} =====", flush=True)
        head, report = train_candidate("learned_sig", train_doms, seed=s)
        reports[f"LearnedSig_seed{s}"] = report
        rows = [eval_panel_domain(head, d, "learned_sig") for d in eval_doms]
        _write_csv(out_dir / f"panel_matrix_LearnedSig_seed{s}.csv", rows)
        by_cell = {r["evaluation_dataset"]: r for r in rows}
        per_seed[s] = {c: _cat_stats(by_cell, c) for c in ("SOURCE_SEEN", *HOLDOUT_CATS)}

    # ---- seed0 stats from parents ----
    per_seed[SEED] = {c: _cat_stats(seed0, c) for c in ("SOURCE_SEEN", *HOLDOUT_CATS)}

    # ---- 3-seed table + stability ----
    table_rows = []
    for s in (SEED, *SEEDS):
        for c, stats in per_seed[s].items():
            if stats is None:
                continue
            table_rows.append({"seed": s, "category": c, **stats})
    _write_csv(out_dir / "seed_table.csv", table_rows)

    stab = stability(per_seed)
    with open(out_dir / "stage2c_summary.json", "w") as f:
        json.dump({"seeds": {str(s): per_seed[s] for s in (SEED, *SEEDS)},
                   "stability": stab}, f, indent=2, default=str)

    # ---- provenance + config ----
    config = {
        "protocol": "hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(args.commit),
        "parent_stage2a_dir": args.parent_stage2a,
        "parent_stage2b_dir": args.parent_stage2b,
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "epochs": EPOCHS, "patience": PATIENCE,
                             "lr": 3e-4, "weight_decay": 1e-4, "clip": 1.0},
        "seed0_source": "parent Stage-2A/2B LearnedSig_main panels",
        "seeds_varied": [1, 2], "host_cache_seed": 0,
        "eval_cells": len(EVAL_CELLS),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump(_summarize_reports(reports), f, indent=2)

    # ---- human summary ----
    L = [f"R1B Stage-2C HCH seed stability (LearnedSig_main) — summary",
         f"date: {config['date']}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"parent2a={args.parent_stage2a} parent2b={args.parent_stage2b}",
         f"seeds: {SEED}(from parents) + {SEEDS}; eval cells: {len(EVAL_CELLS)}", ""]
    for s in (SEED, *SEEDS):
        L.append(f"== seed {s} ==")
        for c in ("SOURCE_SEEN", *HOLDOUT_CATS):
            st = per_seed[s][c]
            if st:
                L.append(f"  {c:30s} n={st['n']:2d} macro={st['macro']} "
                         f"frac<0={st['frac_lt_0']} worst={st['worst']}")
        L.append("")
    L.append("== §15 seed stability ==")
    for c, v in stab["per_category"].items():
        L.append(f"  {c}: {v}")
    L.append(f"  VERDICT: {stab['stop_note']}")
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[stage2c] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
