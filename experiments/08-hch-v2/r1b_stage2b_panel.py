"""R1B Stage-2B — predeclared deep-host stress (§13-§14).

Only run after Stage-2A is healthy (§13). Two parts:

PART-1 main deep-host panel: LearnedSig_main / PlainCore_main (frozen, seed=0,
12 source domains, all 4 hosts seen) evaluated on the pre-frozen representative
markets x LSTM/PatchTST hosts — ZERO candidate gradient / ZERO S2V signal:
    REP = [EPEX_FR, PJM_2020, GEFCOM14P, NORD_DK1] x {LSTM, PatchTST}  (8 cells)

PART-2 LOHO unseen-host stress (§14): LearnedSig_LOHO / PlainCore_LOHO trained
with Linear+MLP+LSTM hosts ONLY (PatchTST excluded from S2T gradient and S2V
selection), evaluated on PatchTST at:
    SOURCE_MARKETS + NORD_DK1 + EPEX_FR + PJM_2020 + GEFCOM14P            (7 cells)
giving the true UNSEEN_HOST x multiple market/dataset-shift matrix.

STOP (documented operationalizations):
  DEEP_HOST_COLLAPSE        : >1/3 of the 8 main deep-holdout cells delta>0.
  LOHO_HOST_COLLAPSE        : >1/3 of the 4 LOHO holdout-market PatchTST cells
                              (DK1/EPEX_FR/PJM_2020/GEFCOM14P) delta>0.
  SIGNATURE_DEEP_NEGATIVE   : LearnedSig better than PlainCore on source deep
                              cells but worse on deep holdout (macro).
Any STOP -> pause before Stage-2C / action-chain (§16).
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from host_cache import cache_one
import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEEN_HOSTS_LOHO, SEED, EPOCHS, PATIENCE,
    EvalDomain, with_membership, train_candidate, _summarize_reports,
)
from r1b_stage2a_panel import (
    provenance, eval_panel_domain, agg_by_category, gen_gap, _write_csv,
)

# --------------------------------------------------------------- config ----
# §13 pre-frozen representative markets (chosen before looking at results).
REP_MARKETS = ["EPEX_FR", "PJM_2020", "GEFCOM14P", "NORD_DK1"]
DEEP_HOSTS = ["LSTM", "PatchTST"]          # §13 deep hosts
LOHO_UNSEEN_HOST = "PatchTST"              # §14

# §14 LOHO eval set: source markets + DK1 + EPEX_FR + PJM_2020 + GEFCOM14P
LOHO_EVAL_MARKETS = SOURCE_MARKETS + ["NORD_DK1", "EPEX_FR", "PJM_2020", "GEFCOM14P"]

VARIANTS = ["learned_sig", "plain_core"]


def _deep_holdout_cells() -> list[str]:
    return [f"{mk}:{bb}" for mk in REP_MARKETS for bb in DEEP_HOSTS]


def _loho_cells() -> list[str]:
    return [f"{mk}:{LOHO_UNSEEN_HOST}" for mk in LOHO_EVAL_MARKETS]


def stop_rules(main_sig: dict, main_plain: dict,
               loho_sig: dict, loho_plain: dict) -> dict:
    """§13/§14 Stage-2B STOP rules (documented operationalizations)."""
    def _frac_pos(rows, pred):
        ds = [r["delta_crps"] for r in rows if r["delta_crps"] is not None
              and pred(r)]
        return (sum(1 for d in ds if d > 0) / len(ds)) if ds else None

    # ---- PART-1: main deep-holdout (8 cells, host_seen=True) ----
    main_ho_sig = [r for r in main_sig if r["transfer_category"] != "SOURCE_SEEN"]
    main_ho_plain = [r for r in main_plain if r["transfer_category"] != "SOURCE_SEEN"]
    fp_sig = _frac_pos(main_ho_sig, lambda r: True)
    fp_plain = _frac_pos(main_ho_plain, lambda r: True)
    deep_collapse = fp_sig is not None and fp_sig > 1 / 3

    # per-market mean over deep-holdout (main LearnedSig)
    mkt_means = {}
    for mk in REP_MARKETS:
        ds = [r["delta_crps"] for r in main_ho_sig if r["market"] == mk
              and r["delta_crps"] is not None]
        mkt_means[mk] = round(float(np.mean(ds)), 6) if ds else None

    # sig-vs-plain: better on source deep, worse on deep holdout?
    def _deep_macro(rows, host_set):
        ds = [r["delta_crps"] for r in rows
              if r["transfer_category"] == "SOURCE_SEEN"
              and r["host"] in host_set and r["delta_crps"] is not None]
        return float(np.mean(ds)) if ds else None

    sig_src_deep = _deep_macro(main_sig, DEEP_HOSTS)
    plain_src_deep = _deep_macro(main_plain, DEEP_HOSTS)
    def _macro(rows):
        ds = [r["delta_crps"] for r in rows if r["delta_crps"] is not None]
        return float(np.mean(ds)) if ds else None
    sig_ho_deep, plain_ho_deep = _macro(main_ho_sig), _macro(main_ho_plain)
    sig_neg_deep = bool(
        sig_src_deep is not None and plain_src_deep is not None
        and sig_ho_deep is not None and plain_ho_deep is not None
        and sig_src_deep < plain_src_deep and sig_ho_deep > plain_ho_deep)

    # ---- PART-2: LOHO unseen-host stress ----
    # holdout markets only (DK1/EPEX_FR/PJM_2020/GEFCOM14P), source sanity kept apart
    loho_ho_sig = [r for r in loho_sig if r["market"] not in SOURCE_MARKETS]
    loho_ho_plain = [r for r in loho_plain if r["market"] not in SOURCE_MARKETS]
    fp_loho = _frac_pos(loho_ho_sig, lambda r: True)
    fp_loho_plain = _frac_pos(loho_ho_plain, lambda r: True)
    loho_collapse = fp_loho is not None and fp_loho > 1 / 3

    # LOHO source-market sanity (candidate on unseen host, seen market)
    loho_src_sig = [r for r in loho_sig if r["market"] in SOURCE_MARKETS]
    fp_src = _frac_pos(loho_src_sig, lambda r: True)

    any_stop = bool(deep_collapse or loho_collapse or sig_neg_deep)
    return {
        "DEEP_HOST_COLLAPSE": deep_collapse,
        "DEEP_HOST_detail": {
            "n_cells": len(main_ho_sig),
            "frac_pos_sig": round(fp_sig, 4) if fp_sig is not None else None,
            "frac_pos_plain": round(fp_plain, 4) if fp_plain is not None else None,
            "per_market_mean_sig": mkt_means,
            "source_deep_macro_sig": sig_src_deep,
            "source_deep_macro_plain": plain_src_deep,
            "holdout_deep_macro_sig": sig_ho_deep,
            "holdout_deep_macro_plain": plain_ho_deep,
        },
        "LOHO_HOST_COLLAPSE": loho_collapse,
        "LOHO_detail": {
            "n_holdout_cells": len(loho_ho_sig),
            "frac_pos_sig": round(fp_loho, 4) if fp_loho is not None else None,
            "frac_pos_plain": round(fp_loho_plain, 4) if fp_loho_plain is not None else None,
            "source_cells_n": len(loho_src_sig),
            "source_frac_pos_sig": round(fp_src, 4) if fp_src is not None else None,
        },
        "SIGNATURE_DEEP_NEGATIVE": sig_neg_deep,
        "STOP": any_stop,
        "stop_note": ("STOP before Stage-2C / action-chain" if any_stop
                      else "CONTINUE — no §13/§14 deep-host collapse"),
    }


def gen_gap_deep(src_deep: float, ho_deep: float) -> dict:
    eps = 1e-9
    return {
        "source_deep_macro": round(float(src_deep), 6),
        "holdout_deep_macro": round(float(ho_deep), 6),
        "G_transfer_deep": round(float(ho_deep - src_deep), 6),
        "R_retain_deep": round(float(abs(ho_deep) / (abs(src_deep) + eps)), 4),
    }


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None,
                    help="declared_source_commit (P0-3)")
    ap.add_argument("--parent-stage2a", type=str, default=None,
                    help="R1B_STAGE2A_<ts> dir that authorized this run")
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2B_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. deep-host caches (LSTM/PatchTST, host_seed=0) ----
    if not args.skip_cache:
        for mk in REP_MARKETS:
            for bb in DEEP_HOSTS:
                cache_dir = HERE / "results" / "cache" / mk / bb
                if (cache_dir / "pred_full.npy").exists():
                    print(f"[cache] {mk} x {bb} SKIP", flush=True)
                    continue
                print(f"[cache] {mk} x {bb} ...", flush=True)
                rec = cache_one(mk, bb, seed=0)
                print(f"[cache] {mk} x {bb} OK split_hash={rec['split_hash']}", flush=True)
    else:
        print("[cache] skipped (--skip-cache)")

    # ---- 1. prepare domains ----
    cells_main = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS] + \
                 _deep_holdout_cells()
    doms = {}
    for name in cells_main + _loho_cells():
        if name in doms:
            continue
        mk, bb = name.split(":")
        print(f"[prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)

    main_eval = with_membership([doms[n] for n in cells_main], SOURCE_MARKETS, HOSTS)
    loho_eval = with_membership([doms[n] for n in _loho_cells()],
                                SOURCE_MARKETS, SEEN_HOSTS_LOHO)

    train_doms_main = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]
    train_doms_loho = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in SEEN_HOSTS_LOHO]

    # ---- 2. train + eval 4 frozen candidates ----
    heads, reports, rows, cats = {}, {}, {}, {}
    specs = [
        ("learned_sig", "LearnedSig_main", train_doms_main, main_eval),
        ("plain_core", "PlainCore_main", train_doms_main, main_eval),
        ("learned_sig", "LearnedSig_LOHO", train_doms_loho, loho_eval),
        ("plain_core", "PlainCore_LOHO", train_doms_loho, loho_eval),
    ]
    for variant, label, train_doms, eval_doms in specs:
        print(f"\n===== STAGE-2B train {label} ({len(train_doms)} domains) =====", flush=True)
        head, report = train_candidate(variant, train_doms)
        heads[label], reports[label] = head, report
        rows[label] = [eval_panel_domain(head, d, variant) for d in eval_doms]
        cats[label] = agg_by_category(rows[label])
        _write_csv(out_dir / f"panel_matrix_{label}.csv", rows[label])

    # ---- 3. gaps + STOP ----
    def _source_deep_macro(rowset, host_set):
        ds = [r["delta_crps"] for r in rowset
              if r["transfer_category"] == "SOURCE_SEEN"
              and r["host"] in host_set and r["delta_crps"] is not None]
        return float(np.mean(ds)) if ds else float("nan")

    def _holdout_macro(rowset):
        ds = [r["delta_crps"] for r in rowset
              if r["transfer_category"] != "SOURCE_SEEN"
              and r["delta_crps"] is not None]
        return float(np.mean(ds)) if ds else float("nan")

    gaps = {}
    for label in ("LearnedSig_main", "PlainCore_main"):
        gaps[label] = gen_gap_deep(_source_deep_macro(rows[label], DEEP_HOSTS),
                                   _holdout_macro(rows[label]))
    stops = stop_rules(rows["LearnedSig_main"], rows["PlainCore_main"],
                       rows["LearnedSig_LOHO"], rows["PlainCore_LOHO"])

    # ---- 4. ledger (§24) ----
    ledger = []
    for label in ("LearnedSig_main", "PlainCore_main",
                  "LearnedSig_LOHO", "PlainCore_LOHO"):
        for r in rows[label]:
            worst = max((x["delta_crps"] for x in rows[label]
                         if x["delta_crps"] is not None), default=None)
            is_main = label.endswith("_main")
            ledger.append({
                "experiment_id": "R1B_STAGE2B",
                "candidate_variant": r["candidate_variant"],
                "training_market_set": "LAGO_DE,LAGO_PJM,NEM_SA1",
                "training_host_set": ("Linear,MLP,LSTM,PatchTST" if is_main
                                      else "Linear,MLP,LSTM (LOHO)"),
                "evaluation_dataset": r["evaluation_dataset"],
                "transfer_category": r["transfer_category"],
                "source_macro_delta": gaps.get(label, {}).get("source_deep_macro"),
                "holdout_delta": r["delta_crps"],
                "worst_domain_delta": worst,
                "seed_consistency": "single_seed_0 (Stage-2C pending)",
                "final_mae_effect": r["mae_rel_deg"],
                "safety_effect": r["safety"],
                "complexity_added": "none",
                "accepted_for_universal_core": "PENDING" if stops["STOP"]
                else "PENDING_R1B_FINAL",
                "notes": f"group={r['holdout_group']} schema={r['schema_class']} "
                         f"status={r['status']}",
            })
    _write_csv(out_dir / "generalization_ledger_v2.csv", ledger)

    # ---- 5. artifacts ----
    config = {
        "protocol": "hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(args.commit),
        "parent_stage2a_dir": args.parent_stage2a,
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "seed": 0, "epochs": EPOCHS, "patience": PATIENCE,
                             "lr": 3e-4, "weight_decay": 1e-4, "clip": 1.0},
        "source_markets": SOURCE_MARKETS, "source_hosts": HOSTS,
        "deep_hosts": DEEP_HOSTS,
        "deep_holdout_markets": REP_MARKETS,   # §13 predeclared
        "loho_hosts": SEEN_HOSTS_LOHO, "loho_unseen_host": LOHO_UNSEEN_HOST,
        "loho_eval_markets": LOHO_EVAL_MARKETS,
        "zero_gradient_holdouts": True,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump(_summarize_reports(reports), f, indent=2)
    with open(out_dir / "stage2b_summary.json", "w") as f:
        json.dump({"aggregates_by_category": cats, "gaps": gaps,
                   "stop_rules": stops}, f, indent=2, default=str)

    cat_rows = []
    for label, cat in cats.items():
        for c, v in cat.items():
            cat_rows.append({"candidate": label, "category": c, **v})
    _write_csv(out_dir / "transfer_taxonomy.csv", cat_rows)

    # ---- 6. human summary ----
    lines = _build_summary(config, cats, gaps, stops)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[stage2b] artifacts: {out_dir}")


def _build_summary(config, cats, gaps, stops) -> list[str]:
    L = [f"R1B Stage-2B deep-host stress (LSTM/PatchTST) — summary",
         f"date: {config['date']}",
         f"parent_stage2a: {config.get('parent_stage2a')}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"sha256(runner)={config['provenance']['sha256_runner'][:12]}",
         ""]
    for label, cat in cats.items():
        L.append(f"== {label} ==")
        for c, v in cat.items():
            L.append(f"  {c:32s} n={v['n_domains']:2d} macro={v['macro_delta']} "
                     f"frac<0={v['frac_delta_lt_0']} worst={v['worst_delta']}")
        if label in gaps:
            g = gaps[label]
            L.append(f"  GAP deep: src={g['source_deep_macro']} "
                     f"holdout={g['holdout_deep_macro']} "
                     f"G_deep={g['G_transfer_deep']} R_retain={g['R_retain_deep']}")
        L.append("")
    L.append("== §13/§14 STOP rules ==")
    for k in ("DEEP_HOST_COLLAPSE", "LOHO_HOST_COLLAPSE",
              "SIGNATURE_DEEP_NEGATIVE"):
        L.append(f"  {k}: {stops[k]}")
    L.append(f"  VERDICT: {stops['stop_note']}")
    return L


if __name__ == "__main__":
    main()
