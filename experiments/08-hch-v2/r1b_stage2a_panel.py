"""R1B Stage-2A — broad frozen Linear/MLP zero-gradient holdout panel.

Stage-2 protocol §5-§12. Candidates are the Stage-1 MAIN variants
(LearnedSig_main, PlainCore_main), retrained deterministically (seed=0, same
12 source domains) so the frozen weights reproduce Stage-1 exactly. All new
holdout domains contribute ZERO candidate gradient and ZERO S2V selection
signal (§5).

Holdout groups (§6):
  H1 unseen market  : EPEX_FR, EPEX_BE, EPEX_NL, NORD_FI, NORD_NO, NORD_SE3, NORD_DK1
  H2 same-market    : DE_EPEX, PJM_2020
  H3 historical     : GEFCOM14P
Fast panel (§8): host caches Linear + MLP only, host_seed = 0.

Outputs per domain (§10): host transformed baseline, IAH CRPS, delta CRPS,
mass entropy, w0, m-/m+ alive, shift p50/p95, scale valid frac, host raw MAE,
candidate MAE, MAE rel degradation.

Aggregates: transfer taxonomy (§9), macro by category, fraction delta<0,
p10/p50/p90, worst, G_transfer / R_retain (§11), STOP rules (§12),
GENERALIZATION_LEDGER v2 (§24), P0-3 provenance (declared_source_commit +
sha256 of runner + core src files).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
from universal_trainer import _eval_batch_losses, _eval_host_baseline

import r1a_run as R
from r1b_generalization_screen import (
    SOURCE_MARKETS, HOSTS, SEED, EPOCHS, PATIENCE, DET_ZERO,
    EvalDomain, with_membership, build_head, det_for_variant, domain_batches,
    domain_health, train_candidate, _summarize_reports,
)

# --------------------------------------------------------------- config ----
H1 = ["EPEX_FR", "EPEX_BE", "EPEX_NL", "NORD_FI", "NORD_NO", "NORD_SE3", "NORD_DK1"]
H2 = ["DE_EPEX", "PJM_2020"]
H3 = ["GEFCOM14P"]
FAST_HOSTS = ["Linear", "MLP"]          # §8 fast panel
ALL_HOLDOUT_MARKETS = H1 + H2 + H3

HOLDOUT_GROUP = {mk: g for g, mks in (("H1", H1), ("H2", H2), ("H3", H3))
                 for mk in mks}

FAMILY = {
    "LAGO_DE": "DE", "DE_EPEX": "DE",
    "LAGO_PJM": "PJM", "PJM_2020": "PJM",
    "NEM_SA1": "NEM",
    "NORD_DK1": "NORDIC", "NORD_FI": "NORDIC", "NORD_NO": "NORDIC",
    "NORD_SE3": "NORDIC",
    "EPEX_FR": "WEST_EU", "EPEX_BE": "WEST_EU", "EPEX_NL": "WEST_EU",
    "GEFCOM14P": "GEFCOM",
}
SOURCE_FAMILIES = {"DE", "PJM", "NEM"}

VARIANTS = ["learned_sig", "plain_core"]
VARIANT_LABELS = {"learned_sig": "LearnedSig", "plain_core": "PlainCore"}


def transfer_category(market: str, host_seen: bool) -> str:
    """§9 taxonomy — category determined by market role + host membership."""
    if market in SOURCE_MARKETS:
        return "SOURCE_SEEN" if host_seen else "UNSEEN_HOST"
    if market in H1:
        return "UNSEEN_MARKET" if host_seen else "UNSEEN_MARKET_AND_HOST"
    if market in H2:
        return "UNSEEN_DATASET_SAME_MARKET" if host_seen else "UNSEEN_MARKET_AND_HOST"
    if market in H3:
        return "UNSEEN_SCHEMA_REGIME" if host_seen else "UNSEEN_MARKET_AND_HOST"
    raise ValueError(f"unknown market {market}")


def schema_class(market: str) -> str:
    if market.startswith("LAGO_"):
        return "lago_exog_fc"
    if market == "NEM_SA1":
        return "nem_aemo"
    if market == "GEFCOM14P":
        return "gefcom14_exog_fc"
    return "epex_price_only"


# ------------------------------------------------------------- provenance ----
def _sha256(relpath: str) -> str:
    try:
        p = ROOT / relpath
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return "missing"


def provenance(declared_commit: str | None) -> dict:
    """P0-R1B-S3: declared source commit + sha256 of the files that determine
    the run's behavior. If the server has a real .git, git_sha is authoritative
    and declared_source_commit cross-checks it."""
    git_sha = R._git_head()
    return {
        "git_sha": git_sha,                      # server .git HEAD if present
        "declared_source_commit": declared_commit or git_sha,
        "sha256_runner": _sha256(Path(__file__).resolve().relative_to(ROOT).as_posix()),
        "sha256_iah_candidate": _sha256("src/iah_candidate.py"),
        "sha256_universal_trainer": _sha256("src/universal_trainer.py"),
        "sha256_iah_crps_loss": _sha256("src/iah_crps_loss.py"),
        # DataSignature class lives in src/hch_v2_context.py (no data_signature.py)
        "sha256_data_signature": _sha256("src/hch_v2_context.py"),
        "provenance_note": ("git_sha from server .git if available; "
                            "declared_source_commit is the local commit whose "
                            "files were synced to the server (P0-R1B-S3)."),
    }


# ---------------------------------------------------------------- panel ----
def eval_panel_domain(head: nn.Module, dom: EvalDomain, variant: str) -> dict:
    """One panel row: CRPS + mass health + raw MAE + taxonomy (§9/§10)."""
    det_np = det_for_variant(variant, dom.info)
    det_t = R.det_for(det_np, 1)
    iah_crps, _ = _eval_batch_losses(head, dom.info.s2v_batches, det_t)
    host_base = _eval_host_baseline(dom.info.s2v_batches)
    h = domain_health(head, dom, det_np)

    host_ae, cand_ae, scale_days = [], [], 0
    with torch.no_grad():
        for batch in dom.info.s2v_batches:
            host, ctx, tgt, vm = batch[:4]
            det_b = det_t.expand(host.shape[0], -1) if det_t.shape[0] != host.shape[0] else det_t
            out = head(host, ctx, valid_mask=vm, domain_det=det_b)
            vh = (vm > 0.5) & torch.isfinite(tgt.squeeze(-1))
            if not vh.any():
                continue
            host_ae.append((tgt.squeeze(-1) - host.squeeze(-1))[vh].abs()
                           .detach().cpu().numpy())
            sv = out["scale_valid"] > 0.5                      # [B]
            scale_days += int(sv.sum())
            # torch.where broadcast quirk: cond must be [B,1,1] to broadcast over
            # [B,H,1] x_identity/host (a [B,1] cond FAILS — PyTorch does not
            # left-pad the condition rank; reproduces on torch 2.x 4090 server).
            cand_pred = torch.where(sv.unsqueeze(-1).unsqueeze(-1),
                                    out["x_identity"], host)
            cand_ae.append((tgt.squeeze(-1) - cand_pred.squeeze(-1))[vh].abs()
                           .detach().cpu().numpy())
    host_raw_mae = float(np.concatenate(host_ae).mean()) if host_ae else float("nan")
    cand_mae = float(np.concatenate(cand_ae).mean()) if cand_ae else float("nan")
    mae_rel = ((cand_mae - host_raw_mae) / host_raw_mae
               if np.isfinite(cand_mae) and np.isfinite(host_raw_mae)
               and host_raw_mae > 0 else float("nan"))

    delta = (iah_crps - host_base if np.isfinite(iah_crps)
             and np.isfinite(host_base) else float("nan"))
    market, host_bb = dom.market, dom.host
    status = "OK"
    if not (np.isfinite(iah_crps) and np.isfinite(delta)):
        status = "NAN_OR_INF"
    elif delta > 0.05:
        status = "HOST_BETTER"

    return {
        "candidate_variant": VARIANT_LABELS[variant],
        "market": market, "host": host_bb,
        "evaluation_dataset": f"{market}:{host_bb}",
        "holdout_group": HOLDOUT_GROUP.get(market, "SOURCE"),
        "market_family_seen": int(FAMILY.get(market, "?") in SOURCE_FAMILIES),
        "dataset_seen": int(market in SOURCE_MARKETS),
        "host_seen": int(dom.host_seen),
        "schema_class": schema_class(market),
        "transfer_category": transfer_category(market, dom.host_seen),
        "host_baseline": round(float(host_base), 6) if np.isfinite(host_base) else None,
        "iah_crps": round(float(iah_crps), 6) if np.isfinite(iah_crps) else None,
        "delta_crps": round(float(delta), 6) if np.isfinite(delta) else None,
        "mass_entropy": round(h["mass_entropy"], 6),
        "w0": round(h["w0"], 6),
        "mminus_alive": round(h["mminus_alive"], 6),
        "mplus_alive": round(h["mplus_alive"], 6),
        "shift_p50": round(h["shift_p50"], 6),
        "shift_p95": round(h["shift_p95"], 6),
        "scale_valid_frac": round(scale_days / len(dom.info.s2v_batches), 4)
                            if dom.info.s2v_batches else None,
        "host_raw_mae": round(host_raw_mae, 4) if np.isfinite(host_raw_mae) else None,
        "cand_mae": round(cand_mae, 4) if np.isfinite(cand_mae) else None,
        "mae_rel_deg": round(mae_rel, 5) if np.isfinite(mae_rel) else None,
        "safety": ("SAFETY_FAILURE" if np.isfinite(mae_rel) and mae_rel > 0.15
                   else ("ok" if np.isfinite(mae_rel) else "n/a")),
        "status": status,
    }


def agg_by_category(rows: list[dict]) -> dict:
    """Per transfer-category aggregate (§10)."""
    out = {}
    for cat in ("SOURCE_SEEN", "UNSEEN_MARKET", "UNSEEN_DATASET_SAME_MARKET",
                "UNSEEN_SCHEMA_REGIME", "UNSEEN_HOST", "UNSEEN_MARKET_AND_HOST"):
        sub = [r for r in rows if r["transfer_category"] == cat]
        d = [r["delta_crps"] for r in sub if r["delta_crps"] is not None]
        if not sub:
            continue
        out[cat] = {
            "n_domains": len(sub),
            "macro_delta": round(float(np.mean(d)), 6) if d else None,
            "frac_delta_lt_0": round(float(np.mean([x < 0 for x in d])), 4) if d else None,
            "p10": round(float(np.percentile(d, 10)), 6) if d else None,
            "p50": round(float(np.percentile(d, 50)), 6) if d else None,
            "p90": round(float(np.percentile(d, 90)), 6) if d else None,
            "worst_delta": round(float(np.max(d)), 6) if d else None,
            "best_delta": round(float(np.min(d)), 6) if d else None,
        }
    return out


def gen_gap(src12: float, src_ll: float, holdout: float) -> dict:
    """§11 generalization gap + retain ratio (more negative = better)."""
    eps = 1e-9
    return {
        "source_macro_12dom": round(float(src12), 6),
        "source_macro_ll": round(float(src_ll), 6),      # source Linear/MLP only
        "holdout_macro": round(float(holdout), 6),
        "G_transfer_vs_12": round(float(holdout - src12), 6),
        "G_transfer_vs_ll": round(float(holdout - src_ll), 6),
        "R_retain_vs_12": round(float(abs(holdout) / (abs(src12) + eps)), 4),
        "R_retain_vs_ll": round(float(abs(holdout) / (abs(src_ll) + eps)), 4),
    }


def stop_rules(sig: dict, plain: dict, sig_rows: list[dict], plain_rows: list[dict]
               ) -> dict:
    """§12 Stage-2A STOP rules (evaluated on LearnedSig primary).

    Operationalization (documented, falsifiable):
      MARKET_PANEL_COLLAPSE   : >1/3 of H1 domains (Linear/MLP) delta_crps > 0.
      DATASET_SHIFT_COLLAPSE  : >=3 of 4 H2 domains delta>0 AND any market's
                                mean delta > 0.03 (clear degradation, not noise).
      HISTORICAL_SCHEMA_COLLAPSE: all H3 domains (GEFCOM Linear+MLP) delta>0.
      SIGNATURE_NEGATIVE_TRANSFER: LearnedSig better than PlainCore on SOURCE
                                but worse on the holdout union (macro).
    """
    h1_sig = [r for r in sig_rows if r["holdout_group"] == "H1"
              and r["delta_crps"] is not None]
    n_pos_h1 = sum(1 for r in h1_sig if r["delta_crps"] > 0)
    market_collapse = n_pos_h1 > len(h1_sig) / 3

    h2_sig = [r for r in sig_rows if r["holdout_group"] == "H2"
              and r["delta_crps"] is not None]
    h2_pos = sum(1 for r in h2_sig if r["delta_crps"] > 0)
    h2_mkt_means = {}
    for mk in H2:
        ds = [r["delta_crps"] for r in h2_sig if r["market"] == mk]
        h2_mkt_means[mk] = float(np.mean(ds)) if ds else None
    ds_collapse = (h2_pos >= 3
                   and any(m is not None and m > 0.03 for m in h2_mkt_means.values()))

    h3_sig = [r for r in sig_rows if r["holdout_group"] == "H3"
              and r["delta_crps"] is not None]
    h3_pos = sum(1 for r in h3_sig if r["delta_crps"] > 0)
    hist_collapse = h3_pos == len(h3_sig) and len(h3_sig) > 0 and h3_pos >= 1

    def _m(cat_agg, cat): return (cat_agg.get(cat) or {}).get("macro_delta")
    sig_src = _m(sig, "SOURCE_SEEN"); plain_src = _m(plain, "SOURCE_SEEN")
    holdout_union = ("UNSEEN_MARKET", "UNSEEN_DATASET_SAME_MARKET", "UNSEEN_SCHEMA_REGIME")
    sig_ho = np.mean([_m(sig, c) for c in holdout_union if _m(sig, c) is not None])
    plain_ho = np.mean([_m(plain, c) for c in holdout_union if _m(plain, c) is not None])
    sig_better_source = sig_src is not None and plain_src is not None \
        and sig_src < plain_src
    sig_worse_holdout = sig_ho > plain_ho
    sig_neg_transfer = bool(sig_better_source and sig_worse_holdout)

    any_stop = market_collapse or ds_collapse or hist_collapse or sig_neg_transfer
    return {
        "MARKET_PANEL_COLLAPSE": market_collapse,
        "MARKET_PANEL_detail": {"h1_n": len(h1_sig), "h1_pos_delta": n_pos_h1,
                                "threshold_gt_1of3": len(h1_sig) / 3},
        "DATASET_SHIFT_COLLAPSE": ds_collapse,
        "DATASET_SHIFT_detail": {"h2_n": len(h2_sig), "h2_pos_delta": h2_pos,
                                 "per_market_mean": h2_mkt_means},
        "HISTORICAL_SCHEMA_COLLAPSE": hist_collapse,
        "HISTORICAL_detail": {"h3_n": len(h3_sig), "h3_pos_delta": h3_pos},
        "SIGNATURE_NEGATIVE_TRANSFER": sig_neg_transfer,
        "SIG_detail": {"sig_source_macro": sig_src, "plain_source_macro": plain_src,
                       "sig_holdout_macro": float(sig_ho), "plain_holdout_macro": float(plain_ho)},
        "STOP": any_stop,
        "stop_note": ("STOP before full action-chain" if any_stop
                      else "CONTINUE — no §12 collapse"),
    }


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--commit", type=str, default=None,
                    help="declared_source_commit (P0-3)")
    ap.add_argument("--skip-cache", action="store_true",
                    help="assume holdout caches already generated")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_STAGE2A_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 0. holdout host caches (Linear/MLP, host_seed=0) ----
    if not args.skip_cache:
        for mk in ALL_HOLDOUT_MARKETS:
            for bb in FAST_HOSTS:
                cache_dir = HERE / "results" / "cache" / mk / bb
                if (cache_dir / "pred_full.npy").exists():
                    print(f"[cache] {mk} x {bb} SKIP", flush=True)
                    continue
                print(f"[cache] {mk} x {bb} ...", flush=True)
                rec = cache_one(mk, bb, seed=0)
                print(f"[cache] {mk} x {bb} OK split_hash={rec['split_hash']}", flush=True)
    else:
        print("[cache] skipped (--skip-cache)")

    # ---- 1. prepare 32 domains (12 source x 4 host + 20 holdout x 2 host) ----
    doms = {}
    for mk in SOURCE_MARKETS + ALL_HOLDOUT_MARKETS:
        hosts = HOSTS if mk in SOURCE_MARKETS else FAST_HOSTS
        for bb in hosts:
            name = f"{mk}:{bb}"
            print(f"[prep] {name} ...", flush=True)
            info = R.prepare_domain(mk, bb, seed=SEED)
            doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    main_membership = with_membership(
        [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS + ALL_HOLDOUT_MARKETS
         for bb in (HOSTS if mk in SOURCE_MARKETS else FAST_HOSTS)],
        SOURCE_MARKETS, HOSTS)                      # main: all 4 hosts seen

    train_doms = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]

    # ---- 2. train frozen main candidates (deterministic seed=0) ----
    heads, reports, rows, cats = {}, {}, {}, {}
    for variant in VARIANTS:
        label = VARIANT_LABELS[variant]
        print(f"\n===== STAGE-2A train {label}_main (12 source domains) =====", flush=True)
        head, report = train_candidate(variant, train_doms)
        heads[variant], reports[f"{label}_main"] = head, report
        rows[f"{label}_main"] = [eval_panel_domain(head, d, variant)
                                 for d in main_membership]
        cats[f"{label}_main"] = agg_by_category(rows[f"{label}_main"])
        _write_csv(out_dir / f"panel_matrix_{label}_main.csv", rows[f"{label}_main"])

    # ---- 3. generalization gap + STOP rules ----
    def _macro_all(cat_agg):
        vs = [v["macro_delta"] for v in cat_agg.values()
              if v["macro_delta"] is not None]
        return float(np.mean(vs)) if vs else float("nan")

    gap, stops = {}, {}
    for variant, label in (("learned_sig", "LearnedSig"), ("plain_core", "PlainCore")):
        cat = cats[f"{label}_main"]
        src12 = _macro_all({k: v for k, v in cat.items() if k == "SOURCE_SEEN"})
        src_ll = float(np.mean([r["delta_crps"] for r in rows[f"{label}_main"]
                                if r["transfer_category"] == "SOURCE_SEEN"
                                and r["host"] in FAST_HOSTS]))
        ho = float(np.mean([r["delta_crps"] for r in rows[f"{label}_main"]
                            if r["transfer_category"] != "SOURCE_SEEN"]))
        gap[f"{label}_main"] = gen_gap(src12, src_ll, ho)
    stops = stop_rules(cats["LearnedSig_main"], cats["PlainCore_main"],
                       rows["LearnedSig_main"], rows["PlainCore_main"])

    # ---- 4. ledger v2 (§24) ----
    ledger = []
    for label in ("LearnedSig_main", "PlainCore_main"):
        worst = max((r["delta_crps"] for r in rows[label]
                     if r["delta_crps"] is not None), default=None)
        for r in rows[label]:
            ledger.append({
                "experiment_id": "R1B_STAGE2A",
                "candidate_variant": r["candidate_variant"],
                "training_market_set": "LAGO_DE,LAGO_PJM,NEM_SA1",
                "training_host_set": "Linear,MLP,LSTM,PatchTST",
                "evaluation_dataset": r["evaluation_dataset"],
                "transfer_category": r["transfer_category"],
                "source_macro_delta": gap[label]["source_macro_12dom"],
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
        "candidate_params": {"d_model": 64, "d_sig": 32, "d_value": 0,
                             "seed": 0, "epochs": EPOCHS, "patience": PATIENCE,
                             "lr": 3e-4, "weight_decay": 1e-4, "clip": 1.0},
        "source_markets": SOURCE_MARKETS,
        "source_hosts": HOSTS,
        "h1_unseen_market": H1, "h2_same_market": H2, "h3_historical": H3,
        "fast_hosts": FAST_HOSTS, "host_seed": 0,
        "zero_gradient_holdouts": True,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump(_summarize_reports(reports), f, indent=2)
    with open(out_dir / "stage2a_summary.json", "w") as f:
        json.dump({"aggregates_by_category": cats, "generalization_gap": gap,
                   "stop_rules": stops}, f, indent=2, default=str)

    # taxonomy csv
    cat_rows = []
    for label, cat in cats.items():
        for c, v in cat.items():
            cat_rows.append({"candidate": label, "category": c, **v})
    _write_csv(out_dir / "transfer_taxonomy.csv", cat_rows)

    # ---- 6. human summary ----
    lines = _build_summary(config, cats, gap, stops, rows)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[stage2a] artifacts: {out_dir}")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_summary(config, cats, gap, stops, rows) -> list[str]:
    L = [f"R1B Stage-2A broad frozen holdout panel — summary",
         f"date: {config['date']}",
         f"git_sha: {config['provenance']['git_sha']} "
         f"(declared {config['provenance']['declared_source_commit']})",
         f"sha256(runner)={config['provenance']['sha256_runner'][:12]}",
         ""]
    for label, cat in cats.items():
        L.append(f"== {label} ==")
        for c, v in cat.items():
            L.append(f"  {c:32s} n={v['n_domains']:2d} macro={v['macro_delta']} "
                     f"frac<0={v['frac_delta_lt_0']} "
                     f"p10/50/90={v['p10']}/{v['p50']}/{v['p90']} "
                     f"worst={v['worst_delta']}")
        L.append(f"  GAP: holdout={gap[label]['holdout_macro']} "
                 f"src12={gap[label]['source_macro_12dom']} "
                 f"G_vs12={gap[label]['G_transfer_vs_12']} "
                 f"G_vsLL={gap[label]['G_transfer_vs_ll']} "
                 f"R_retain={gap[label]['R_retain_vs_12']}")
        L.append("")
    L.append("== §12 STOP rules ==")
    for k in ("MARKET_PANEL_COLLAPSE", "DATASET_SHIFT_COLLAPSE",
              "HISTORICAL_SCHEMA_COLLAPSE", "SIGNATURE_NEGATIVE_TRANSFER"):
        L.append(f"  {k}: {stops[k]}")
    L.append(f"  VERDICT: {stops['stop_note']}")
    return L


if __name__ == "__main__":
    main()
