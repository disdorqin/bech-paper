"""R1B Stage-2E — action-chain extension to FR / PJM2020 / GEFCOM (§19).

Same frozen LearnedSig_main (seed=0, 12 source) as Stage-2D, same A0/A1/A2
chain. 6 extension domains: EPEX_FR / PJM_2020 / GEFCOM14P x Linear/PatchTST.

§19 rules:
  - selector must default C0 if local evidence too short (gates are NOT lowered).
  - no forced C3.

STOP operationalizations (extension-specific):
  SAFETY_FAILURE        any domain MAE_rel > 0.15
  EXT_ACTION_COLLAPSE   macro A2 over 6 ext domains <= 0, or any A2 < -0.01
  GATING_HURTS          any domain A2 < A1 - 0.01
  SELECTOR_RESPECTED    domains with insufficient prequential evidence -> C0
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

import r1a9_action_calibration as M
import r1a11_prequential_calibration_router as P
import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEED, EPOCHS, PATIENCE, EvalDomain, train_candidate,
)
from r1b_stage2a_panel import provenance, eval_panel_domain, _write_csv
from _final_point import final_metrics
from r1b_stage2d_action_chain import PREQ_CFG, MAE_SAFETY

EXT_MARKETS = ["EPEX_FR", "PJM_2020", "GEFCOM14P"]
EXT_HOSTS = ["Linear", "PatchTST"]
EXT_DOMAINS = [f"{mk}:{bb}" for mk in EXT_MARKETS for bb in EXT_HOSTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None)
    ap.add_argument("--parent-stage2d", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2E_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. train + freeze primary candidate (same deterministic seed-0 head) ----
    doms = {}
    all_names = ([f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]
                 + EXT_DOMAINS)
    for name in all_names:
        mk, bb = name.split(":")
        print(f"[prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    train_doms = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]
    print(f"\n===== STAGE-2E train LearnedSig_main ({len(train_doms)} source) =====",
          flush=True)
    head, report = train_candidate("learned_sig", train_doms, seed=SEED)

    # ---- 1. per-domain A0/A1/A2 chain (same as Stage-2D) ----
    evaluator = P.PrequentialCalibrationEvaluator(PREQ_CFG)
    ac_rows, sel_rows = [], []
    for name in EXT_DOMAINS:
        mk, bb = name.split(":")
        print(f"[chain] {name} ...", flush=True)
        dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
        for pr in dd["problems"]:
            print(f"      WARN {pr}", flush=True)

        c0 = M.RawIAH()
        rows_c0 = M.evaluate_days(c0, dd)
        dv_a1 = M.dvg_and_s4(rows_c0, R.ALPHA)

        folds, proto, _days, n_total = evaluator.build_rolling_folds(dd)
        oos = evaluator.collect_paired_oos_rows(name, folds, c0) if folds else []
        sel, reason, s0, s3, gates, boot = evaluator.select(name, oos, folds)
        if sel == "C3":
            s_d, y_d = P.hour_rows(
                [d for d in dd["days"] if d["block"] in ("train", "val")], "d")
            s_u, y_u = P.hour_rows(
                [d for d in dd["days"] if d["block"] in ("train", "val")], "u")
            final = M.LocalIsotonic()
            final.fit([{"domain": name, "sd": s_d, "Yd": y_d,
                        "su": s_u, "Yu": y_u}])
            rows_final = M.evaluate_days(final, dd)
            dv_a2 = M.dvg_and_s4(rows_final, R.ALPHA)
        else:
            rows_final = rows_c0
            dv_a2 = dv_a1
        calibrator = "C3_local_isotonic" if sel == "C3" else "C0_raw"

        pan = eval_panel_domain(head, doms[name], "learned_sig")
        # P0-A: headline point metrics = TRUE final output of SELECTED policy.
        fm = final_metrics(dv_a2)
        fm = {k: v for k, v in fm.items() if not isinstance(v, np.ndarray)}

        sel_rows.append({
            "domain": name, "total_s3m_days": n_total,
            "n_blocks": len(folds), "n_oos_days": len(oos),
            "selected": sel, "reason": reason, "calibrator": calibrator,
            "gate_A": gates.get("A_prequential"),
            "gate_B": gates.get("B_raw_problem"),
            "gate_C": gates.get("C_value_improve"),
            "gate_D": gates.get("D_bootstrap"),
            "lcb90": boot.get("lcb") if boot else None,
            "mean_paired_delta": boot.get("mean_delta") if boot else None,
        })
        fm_scalar = {k: v for k, v in fm.items()
                     if k not in ("days",) and not isinstance(v, np.ndarray)}
        ac_rows.append({
            "domain": name, "market": mk, "host": bb,
            "transfer": "EXTENSION",
            "host_crps": pan["host_baseline"], "cand_crps": pan["iah_crps"],
            "delta_crps": pan["delta_crps"],
            "host_mae_panel": pan["host_raw_mae"], "cand_mae_panel": pan["cand_mae"],
            "mae_rel": pan["mae_rel_deg"], "safety": pan["safety"],
            "a1_net": dv_a1.get("net_value"), "a2_net": dv_a2.get("net_value"),
            "a1_release": dv_a1.get("release_rate"),
            "a2_release": dv_a2.get("release_rate"),
            "a1_harm": dv_a1.get("harmful_rate"),
            "a2_harm": dv_a2.get("harmful_rate"),
            "selected": sel, "reason": reason,
            **{("final_" + k): v for k, v in fm_scalar.items()
               if k not in ("mae", "smape_nofloor")},
            "final_mae": fm_scalar.get("mae"),
            "final_smape": fm_scalar.get("smape_nofloor"),
        })
        print(f"      {name}: sel={sel} A1_net={dv_a1.get('net_value')} "
              f"A2_net={dv_a2.get('net_value')}", flush=True)

    _write_csv(out_dir / "ext_action_chain_matrix.csv", ac_rows)
    _write_csv(out_dir / "ext_selector.csv", sel_rows)

    # ---- 2. STOP / verdict ----
    def _macro(key):
        vals = [r[key] for r in ac_rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    macro_a1, macro_a2 = _macro("a1_net"), _macro("a2_net")
    safety_fail = [r["domain"] for r in ac_rows
                   if r["mae_rel"] is not None and r["mae_rel"] > MAE_SAFETY]
    ext_collapse = [r["domain"] for r in ac_rows
                    if r["a2_net"] is not None and r["a2_net"] < -0.01]
    gating_hurt = [r["domain"] for r in ac_rows
                   if r["a1_net"] is not None and r["a2_net"] is not None
                   and r["a2_net"] < r["a1_net"] - 0.01]
    low_evidence = [r for r in sel_rows if r["gate_A"] is False]
    selector_respected = all(r["selected"] == "C0" for r in low_evidence)

    stops = {
        "SAFETY_FAILURE": bool(safety_fail), "SAFETY_FAILURE_domains": safety_fail,
        "EXT_ACTION_COLLAPSE": bool(macro_a2 is not None and macro_a2 <= 0)
        or bool(ext_collapse),
        "EXT_COLLAPSE_domains": ext_collapse,
        "ext_macro_A1": macro_a1, "ext_macro_A2": macro_a2,
        "GATING_HURTS": bool(gating_hurt),
        "GATING_HURTS_domains": gating_hurt,
        "SELECTOR_RESPECTED": bool(selector_respected),
        "C3_authorized": [r["domain"] for r in sel_rows if r["selected"] == "C3"],
        "STOP": bool(safety_fail or (macro_a2 is not None and macro_a2 <= 0)
                     or ext_collapse or gating_hurt),
        "note": ("STOP before Local-Core" if
                 (safety_fail or (macro_a2 is not None and macro_a2 <= 0)
                  or ext_collapse or gating_hurt)
                 else "CONTINUE — extension action chain healthy -> Local-Core"),
    }

    # ---- 3. artifacts ----
    config = {
        "protocol": "hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md §19",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(args.commit),
        "parent_stage2d_dir": args.parent_stage2d,
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "seed": 0, "epochs": EPOCHS, "patience": PATIENCE},
        "extension_domains": EXT_DOMAINS,
        "prequential_cfg": PREQ_CFG,
        "mae_safety_redline": MAE_SAFETY,
        "note": "selector gates NOT lowered; insufficient evidence must default C0 (§19)",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump({"LearnedSig_main": report}, f, indent=2, default=str)
    with open(out_dir / "ext_summary.json", "w") as f:
        json.dump({"stops": stops}, f, indent=2, default=str)

    L = [f"R1B Stage-2E action-chain extension (FR/PJM2020/GEFCOM) — summary",
         f"date: {config['date']}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"parent_stage2d: {args.parent_stage2d}", ""]
    L.append(f"{'domain':18s} {'sel':5s} {'A1_net':>8s} {'A2_net':>8s} "
             f"{'delta_crps':>10s} {'mae_rel':>8s} {'safety':>6s}")
    for r in ac_rows:
        L.append(f"{r['domain']:18s} {str(r['selected']):5s} "
                 f"{str(r['a1_net']):>8s} {str(r['a2_net']):>8s} "
                 f"{str(r['delta_crps']):>10s} {str(r['mae_rel']):>8s} "
                 f"{r['safety']:>6s}")
    L.append("")
    L.append(f"extension macro: A1={macro_a1} A2={macro_a2}")
    L.append("")
    L.append("== §19 STOP rules ==")
    for k in ("SAFETY_FAILURE", "EXT_ACTION_COLLAPSE", "GATING_HURTS",
              "SELECTOR_RESPECTED"):
        L.append(f"  {k}: {stops[k]}")
    L.append(f"  C3_authorized: {stops['C3_authorized']}")
    L.append(f"  VERDICT: {stops['note']}")
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[stage2e] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
