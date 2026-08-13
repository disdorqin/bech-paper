"""R1B Stage-2F — Local-Core comparison (§23).

Quantify universal-sharing source-fit cost / benefit.

  Universal : LearnedSig_main trained on 12 source domains (seed=0).
  Local-Core: same arch trained on a SINGLE domain only.
    - per source domain d: local_crps[d] vs universal_crps[d].
        cost[d] = universal_crps[d] - local_crps[d]   (>0 = sharing costs)
    - DK1 full-shot: local core trained on DK1 x 4 hosts.
        gap = universal_crps(DK1) - fullshot_crps(DK1) (<0 = universal beats
        a DK1-native model) — labeled TARGET_TRAINED_FULLSHOT_UPPER_BOUND.

Verdict:
  SOURCE_FIT_COST_HARMFUL  macro cost > 0.02
  SOURCE_FIT_COST_NEUTRAL  |macro cost| <= 0.02
  SOURCE_FIT_HELPS         macro cost < -0.02
"""
from __future__ import annotations

import argparse
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

import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEED, EPOCHS, PATIENCE, EvalDomain, train_candidate,
)
from r1b_stage2a_panel import provenance, eval_panel_domain, _write_csv

COST_TOL = 0.02
ALL_SOURCE = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None)
    ap.add_argument("--parent-stage2e", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2F_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. prep all source + DK1 domains ----
    all_names = ALL_SOURCE + [f"NORD_DK1:{bb}" for bb in HOSTS]
    doms = {}
    for name in all_names:
        mk, bb = name.split(":")
        print(f"[prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)

    # ---- 1. universal head (deterministic, identical to Stage-2D/2E) ----
    univ_train = [doms[n] for n in ALL_SOURCE]
    print(f"\n===== STAGE-2F train UNIVERSAL ({len(univ_train)} source) =====", flush=True)
    univ_head, univ_report = train_candidate("learned_sig", univ_train, seed=SEED)
    univ_crps = {}
    for name in all_names:
        pan = eval_panel_domain(univ_head, doms[name], "learned_sig")
        univ_crps[name] = pan["iah_crps"]
    print("universal eval done", flush=True)

    # ---- 2. source Local-Core (train on ONE domain) ----
    cost_rows = []
    for name in ALL_SOURCE:
        print(f"[local-core] {name} ...", flush=True)
        head_l, rep_l = train_candidate("learned_sig", [doms[name]], seed=SEED)
        pan_l = eval_panel_domain(head_l, doms[name], "learned_sig")
        pan_u = eval_panel_domain(univ_head, doms[name], "learned_sig")
        local_c, univ_c = pan_l["iah_crps"], pan_u["iah_crps"]
        cost = (univ_c - local_c) if (univ_c and local_c) else None
        cost_rows.append({
            "domain": name, "market": name.split(":")[0],
            "local_crps": local_c, "universal_crps": univ_c,
            "cost_univ_minus_local": round(cost, 6) if cost is not None else None,
            "local_delta_vs_host": pan_l["delta_crps"],
            "universal_delta_vs_host": pan_u["delta_crps"],
            "local_epochs_run": rep_l.get("epochs_run"),
        })
        print(f"      {name}: univ={univ_c} local={local_c} cost={cost}", flush=True)
    _write_csv(out_dir / "source_fit_cost.csv", cost_rows)

    # ---- 3. DK1 full-shot (train on DK1 x 4 hosts) ----
    dk1_doms = [doms[f"NORD_DK1:{bb}"] for bb in HOSTS]
    print(f"\n===== STAGE-2F train DK1 FULL-SHOT ({len(dk1_doms)} hosts) =====", flush=True)
    dk1_head, dk1_report = train_candidate("learned_sig", dk1_doms, seed=SEED)
    dk1_rows = []
    for bb in HOSTS:
        name = f"NORD_DK1:{bb}"
        pan_fs = eval_panel_domain(dk1_head, doms[name], "learned_sig")
        pan_u = eval_panel_domain(univ_head, doms[name], "learned_sig")
        dk1_rows.append({
            "domain": name,
            "universal_crps": pan_u["iah_crps"],
            "fullshot_crps": pan_fs["iah_crps"],
            "gap_univ_minus_fullshot": round(pan_u["iah_crps"] - pan_fs["iah_crps"], 6)
            if pan_u["iah_crps"] and pan_fs["iah_crps"] else None,
            "universal_delta": pan_u["delta_crps"],
            "fullshot_delta": pan_fs["delta_crps"],
            "label": "TARGET_TRAINED_FULLSHOT_UPPER_BOUND",
        })
        print(f"      {name}: univ={pan_u['iah_crps']} "
              f"fullshot={pan_fs['iah_crps']}", flush=True)
    _write_csv(out_dir / "dk1_fullshot.csv", dk1_rows)

    # ---- 4. verdict ----
    costs = [r["cost_univ_minus_local"] for r in cost_rows
             if r["cost_univ_minus_local"] is not None]
    macro_cost = float(np.mean(costs)) if costs else None
    dk1_gaps = [r["gap_univ_minus_fullshot"] for r in dk1_rows
                if r["gap_univ_minus_fullshot"] is not None]
    dk1_gap = float(np.mean(dk1_gaps)) if dk1_gaps else None

    if macro_cost is not None and macro_cost > COST_TOL:
        fit_verdict = "SOURCE_FIT_COST_HARMFUL"
    elif macro_cost is not None and macro_cost < -COST_TOL:
        fit_verdict = "SOURCE_FIT_HELPS"
    else:
        fit_verdict = "SOURCE_FIT_COST_NEUTRAL"
    if dk1_gap is not None and dk1_gap < 0:
        dk1_verdict = "TARGET_TRAINED_FULLSHOT_SURPASSED"
    else:
        dk1_verdict = "TARGET_TRAINED_FULLSHOT_UPPER_BOUND"

    stops = {
        "macro_source_fit_cost": macro_cost,
        "source_fit_verdict": fit_verdict,
        "dk1_univ_minus_fullshot_gap": dk1_gap,
        "dk1_verdict": dk1_verdict,
        "note": (f"universal sharing on source: {fit_verdict} (macro cost "
                 f"{macro_cost}); DK1: {dk1_verdict} (gap {dk1_gap})"),
    }

    # ---- 5. artifacts ----
    config = {
        "protocol": "hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md §23",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(args.commit),
        "parent_stage2e_dir": args.parent_stage2e,
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "seed": 0, "epochs": EPOCHS, "patience": PATIENCE},
        "local_core_training": "single source domain (source); DK1 x 4 hosts (full-shot)",
        "cost_tolerance": COST_TOL,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump({"universal": univ_report, "dk1_fullshot": dk1_report},
                  f, indent=2, default=str)
    with open(out_dir / "localcore_summary.json", "w") as f:
        json.dump({"stops": stops}, f, indent=2, default=str)

    L = [f"R1B Stage-2F Local-Core comparison — summary",
         f"date: {config['date']}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"parent_stage2e: {args.parent_stage2e}", ""]
    L.append(f"{'domain':18s} {'universal':>10s} {'local':>10s} {'cost':>10s}")
    for r in cost_rows:
        L.append(f"{r['domain']:18s} {str(r['universal_crps']):>10s} "
                 f"{str(r['local_crps']):>10s} "
                 f"{str(r['cost_univ_minus_local']):>10s}")
    L.append("")
    L.append("DK1 full-shot:")
    for r in dk1_rows:
        L.append(f"  {r['domain']:18s} univ={r['universal_crps']} "
                 f"fullshot={r['fullshot_crps']} "
                 f"gap={r['gap_univ_minus_fullshot']}")
    L.append("")
    L.append(f"macro source-fit cost = {macro_cost}  ->  {fit_verdict}")
    L.append(f"DK1 univ - fullshot gap = {dk1_gap}  ->  {dk1_verdict}")
    L.append(f"  VERDICT: {stops['note']}")
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[stage2f] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
