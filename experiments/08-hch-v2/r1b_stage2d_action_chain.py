"""R1B Stage-2D — Full action-chain on canonical 16 (§17-§22).

Primary universal candidate: LearnedSig_main (frozen, seed=0, 12 source
domains). Canonical 16 = 3 source markets x 4 seen hosts + NORD_DK1 x 4 hosts.

Three policies per domain:
  A0 — Host Identity (pi=0, do nothing; the host baseline).
  A1 — Raw IAH action: analytic C0 utility -> Double Event -> own S3C DVG.
  A2 — Evidence-gated local calibration: prequential C0/C3 eligibility
       (R1A.11 gates A/B/C/D, default-to-C0) -> selected utility map ->
       Double Event -> selected estimator's own S3C DVG.

Metrics (§20): candidate CRPS / host baseline / delta; selector S3M days,
rolling OOS days, selected C0/C3, reason, map stability, LCB; DVG q, coverage,
release, Identity, harmful release, mean gain|release, net daily action value;
forecast MAE / rMAE / RMSE / no-floor sMAPE / negative-price / high-tail.

Safety (§21): any transfer-domain MAE degradation > 15% -> SAFETY_FAILURE.
Continuous degradation is reported, not just the threshold.

STOP operationalizations:
  SAFETY_FAILURE             any domain MAE_rel > 0.15
  SOURCE_ACTION_UNHEALTHY    macro A2 net over 12 source domains <= 0
  TRANSFER_ACTION_COLLAPSE   DK1 macro A2 net < -0.01, or DK1 A2 < A1 - 0.01
  GATING_HURTS               any domain A2 net < A1 net - 0.01
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

import r1a9_action_calibration as M
import r1a11_prequential_calibration_router as P
import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEED, EPOCHS, PATIENCE,
    EvalDomain, train_candidate,
)
from r1b_stage2a_panel import provenance, eval_panel_domain, _write_csv

CANONICAL = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS] + \
            [f"NORD_DK1:{bb}" for bb in HOSTS]          # 12 + 4 = 16
SOURCE_16 = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]
DK1_16 = [f"NORD_DK1:{bb}" for bb in HOSTS]

PREQ_CFG = {"burn_in": P.BURN_IN_DAYS, "block": P.BLOCK_DAYS,
            "min_oos_days": P.GATE_A_MIN_OOS_DAYS,
            "harm_threshold": P.GATE_B_HARMFUL_THRESHOLD,
            "n_resample": P.N_RESAMPLE, "alpha": P.ALPHA}
MAE_SAFETY = 0.15


def forecast_metrics(dd) -> dict:
    """Point-forecast metrics on S4 (dev) days (§20). cand point = host."""
    pred, act = [], []
    for day in dd["days"]:
        if day["block"] != "dev":
            continue
        pred.append(day["host_day"])
        act.append(day["price"])
    if not pred:
        return {"n_hours": 0}
    p = np.concatenate(pred).astype(np.float64)
    a = np.concatenate(act).astype(np.float64)
    e = p - a
    mae = float(np.mean(np.abs(e)))
    rmae = float(mae / max(np.mean(np.abs(a)), 1e-9))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    denom = np.abs(p) + np.abs(a)
    smape = 200.0 * np.abs(e) / np.maximum(denom, 1e-9)
    out = {"n_hours": int(len(p)), "mae": round(mae, 6),
           "rmae": round(rmae, 6), "rmse": round(rmse, 6),
           "smape_nofloor": round(float(np.mean(smape)), 6),
           "neg_price_rate": round(float(np.mean(a < 0)), 4),
           "neg_price_mae": round(float(np.mean(np.abs(e[a < 0]))), 6)
           if (a < 0).any() else None,
           "high_tail_rate": round(float(np.mean(np.abs(a) >=
                                                 np.quantile(np.abs(a), 0.95))), 4),
           "high_tail_mae": round(float(np.mean(np.abs(e[np.abs(a) >=
                          np.quantile(np.abs(a), 0.95)]))), 6)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None)
    ap.add_argument("--parent-stage2c", type=str, default=None)
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2D_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. train + freeze primary candidate (same as Stage-2A) ----
    doms = {}
    for name in CANONICAL:
        mk, bb = name.split(":")
        print(f"[prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    train_doms = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]
    print(f"\n===== STAGE-2D train LearnedSig_main ({len(train_doms)} source) =====",
          flush=True)
    head, report = train_candidate("learned_sig", train_doms, seed=SEED)

    # ---- 1. per-domain action chain (A0/A1/A2) ----
    evaluator = P.PrequentialCalibrationEvaluator(PREQ_CFG)
    ac_rows, sel_rows, dvg_rows = [], [], []
    for name in CANONICAL:
        mk, bb = name.split(":")
        print(f"[chain] {name} ...", flush=True)
        dd = M.collect_domain(None, mk, bb, "learned_sig", head=head)
        for pr in dd["problems"]:
            print(f"      WARN {pr}", flush=True)

        # ---- A0 host identity ----
        c0 = M.RawIAH()

        # ---- A1 raw IAH action ----
        rows_c0 = M.evaluate_days(c0, dd)
        dv_a1 = M.dvg_and_s4(rows_c0, R.ALPHA)

        # ---- A2 evidence-gated (prequential C0/C3) ----
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

        # ---- candidate CRPS + safety (reuse panel eval on trained head) ----
        pan = eval_panel_domain(head, doms[name], "learned_sig")
        fm = forecast_metrics(dd)

        dvg_rows.append({"domain": name, "policy": "A1", **{k: dv_a1.get(k)
                         for k in ("q", "release_rate", "identity_rate",
                                   "harmful_rate", "mean_gain_release",
                                   "net_value", "coverage", "n_calib", "n_eval")}})
        dvg_rows.append({"domain": name, "policy": "A2",
                         "q": dv_a2.get("q"),
                         "release_rate": dv_a2.get("release_rate"),
                         "identity_rate": dv_a2.get("identity_rate"),
                         "harmful_rate": dv_a2.get("harmful_rate"),
                         "mean_gain_release": dv_a2.get("mean_gain_release"),
                         "net_value": dv_a2.get("net_value"),
                         "coverage": dv_a2.get("coverage"),
                         "n_calib": dv_a2.get("n_calib"),
                         "n_eval": dv_a2.get("n_eval")})

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
            "c0_fire": s0["n_fire"] if s0 else 0,
            "c3_fire": s3["n_fire"] if s3 else 0,
            "c0_net": s0["net"] if s0 else None,
            "c3_net": s3["net"] if s3 else None,
        })

        ac_rows.append({
            "domain": name, "market": mk, "host": bb,
            "transfer": "SOURCE" if name in SOURCE_16 else "DK1_TRANSFER",
            # candidate / host CRPS
            "host_crps": pan["host_baseline"], "cand_crps": pan["iah_crps"],
            "delta_crps": pan["delta_crps"],
            # MAE safety (§21)
            "host_mae": pan["host_raw_mae"], "cand_mae": pan["cand_mae"],
            "mae_rel": pan["mae_rel_deg"], "safety": pan["safety"],
            # A1 / A2 DVG net
            "a1_net": dv_a1.get("net_value"), "a2_net": dv_a2.get("net_value"),
            "a1_release": dv_a1.get("release_rate"),
            "a2_release": dv_a2.get("release_rate"),
            "a1_harm": dv_a1.get("harmful_rate"),
            "a2_harm": dv_a2.get("harmful_rate"),
            "selected": sel, "reason": reason,
            **{k: v for k, v in fm.items()},
        })
        print(f"      {name}: sel={sel} A1_net={dv_a1.get('net_value')} "
              f"A2_net={dv_a2.get('net_value')}", flush=True)

    _write_csv(out_dir / "action_chain_matrix.csv", ac_rows)
    _write_csv(out_dir / "selector_by_domain.csv", sel_rows)
    _write_csv(out_dir / "dvg_by_policy.csv", dvg_rows)

    # ---- 2. STOP / verdict ----
    def _macro(rows, key, dom_filter=None):
        vals = [r[key] for r in rows if r[key] is not None
                and (dom_filter is None or dom_filter(r))]
        return float(np.mean(vals)) if vals else None

    src_a1 = _macro(ac_rows, "a1_net", lambda r: r["transfer"] == "SOURCE")
    src_a2 = _macro(ac_rows, "a2_net", lambda r: r["transfer"] == "SOURCE")
    dk1_a1 = _macro(ac_rows, "a1_net", lambda r: r["transfer"] == "DK1_TRANSFER")
    dk1_a2 = _macro(ac_rows, "a2_net", lambda r: r["transfer"] == "DK1_TRANSFER")
    safety_fail = [r["domain"] for r in ac_rows
                   if r["mae_rel"] is not None and r["mae_rel"] > MAE_SAFETY]
    gating_hurt = [r["domain"] for r in ac_rows
                   if r["a1_net"] is not None and r["a2_net"] is not None
                   and r["a2_net"] < r["a1_net"] - 0.01]
    dk1_worse = dk1_a2 is not None and dk1_a1 is not None \
        and dk1_a2 < dk1_a1 - 0.01

    stops = {
        "SAFETY_FAILURE": bool(safety_fail),
        "SAFETY_FAILURE_domains": safety_fail,
        "SOURCE_ACTION_UNHEALTHY": bool(src_a2 is not None and src_a2 <= 0),
        "source_macro_A2": src_a2, "source_macro_A1": src_a1,
        "TRANSFER_ACTION_COLLAPSE": bool(dk1_a2 is not None and dk1_a2 < -0.01)
        or bool(dk1_worse),
        "dk1_macro_A2": dk1_a2, "dk1_macro_A1": dk1_a1,
        "dk1_worse_than_A1": dk1_worse,
        "GATING_HURTS": bool(gating_hurt),
        "GATING_HURTS_domains": gating_hurt,
        "STOP": bool(safety_fail or (src_a2 is not None and src_a2 <= 0)
                     or (dk1_a2 is not None and dk1_a2 < -0.01) or dk1_worse
                     or gating_hurt),
        "note": ("STOP before extension / Local-Core" if
                 (safety_fail or (src_a2 is not None and src_a2 <= 0)
                  or (dk1_a2 is not None and dk1_a2 < -0.01) or dk1_worse
                  or gating_hurt)
                 else "CONTINUE — canonical-16 action chain healthy"),
    }

    # ---- 3. artifacts ----
    config = {
        "protocol": "hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(args.commit),
        "parent_stage2c_dir": args.parent_stage2c,
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "seed": 0, "epochs": EPOCHS, "patience": PATIENCE},
        "canonical_domains": CANONICAL,
        "policies": {"A0": "host_identity_pi0", "A1": "raw_IAH_C0_double_event",
                     "A2": "prequential_C0/C3_gated_double_event"},
        "prequential_cfg": PREQ_CFG,
        "mae_safety_redline": MAE_SAFETY,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump({"LearnedSig_main": report}, f, indent=2, default=str)
    with open(out_dir / "stage2d_summary.json", "w") as f:
        json.dump({"stops": stops}, f, indent=2, default=str)

    L = [f"R1B Stage-2D full action-chain on canonical 16 — summary",
         f"date: {config['date']}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"parent_stage2c: {args.parent_stage2c}", ""]
    L.append(f"{'domain':20s} {'sel':5s} {'A1_net':>8s} {'A2_net':>8s} "
             f"{'delta_crps':>10s} {'mae_rel':>8s} {'safety':>6s}")
    for r in ac_rows:
        L.append(f"{r['domain']:20s} {str(r['selected']):5s} "
                 f"{str(r['a1_net']):>8s} {str(r['a2_net']):>8s} "
                 f"{str(r['delta_crps']):>10s} {str(r['mae_rel']):>8s} "
                 f"{r['safety']:>6s}")
    L.append("")
    L.append(f"source macro: A1={src_a1} A2={src_a2}")
    L.append(f"DK1 macro:   A1={dk1_a1} A2={dk1_a2}")
    L.append("")
    L.append("== §21/§22 STOP rules ==")
    for k in ("SAFETY_FAILURE", "SOURCE_ACTION_UNHEALTHY",
              "TRANSFER_ACTION_COLLAPSE", "GATING_HURTS"):
        L.append(f"  {k}: {stops[k]}")
    L.append(f"  VERDICT: {stops['note']}")
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n[stage2d] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
