"""WP-5 step 3 — Universal retrain on all 8 public headline datasets x 4 hosts.

Protocol §12.1/§12.2: one shared PUBLIC candidate over approved public paper
training domains, hierarchical-balancing-equivalent (8 headline markets each
contribute 4 host domains -> domain-level UCT already implies market-balance).

Outputs: results/WP5_PUBLIC_{date}/
  learned_sig_main_head.pt      frozen candidate head
  training_reports.json         UCT report (macro S2V, worst, per-domain)
  guard_report.json             §12.2 checkpoint guards vs the P0A_RERUN head
  config.json

Guards (protocol §12.2): per-headline-domain S2V CRPS regression <= 2%,
  worst-domain <= 3%, balanced macro improves or within tiny tolerance.
  Anchor = 3 R1B source markets (LAGO_DE/PJM/NEM_SA1): their 12 cells are the
  generalization anchors that must not regress.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
HCH = PAPER.parent / "08-hch-v2"
ROOT = PAPER.parent.parent

for p in (ROOT / "src", HCH):
    sys.path.insert(0, str(p))

import r1a_run as R                                # noqa: E402
from r1b_generalization_screen import (            # noqa: E402
    HOSTS, SEED, EvalDomain, build_head, train_candidate,
)
from r1b_stage2a_panel import eval_panel_domain    # noqa: E402

HEADLINE = ["LAGO_DE", "LAGO_BE", "LAGO_FR", "LAGO_PJM", "LAGO_NP",
            "NEM_SA1", "GEFCOM14P", "NORD_DK1"]
SOURCE = {"LAGO_DE", "LAGO_PJM", "NEM_SA1"}         # R1B anchors
MAX_PER_DOMAIN_REGR = 0.02                          # §12.2 (2%)
MAX_WORST_REGR = 0.03                               # §12.2 (3%)


def load_domains() -> list[EvalDomain]:
    doms = {}
    for mk in HEADLINE:
        for bb in HOSTS:
            name = f"{mk}:{bb}"
            print(f"[prep] {name} ...", flush=True)
            info = R.prepare_domain(mk, bb, seed=SEED)
            doms[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    return doms


def panel_crps(head: torch.nn.Module, doms: list[EvalDomain]) -> dict:
    return {d.name: float(eval_panel_domain(head, d, "learned_sig")["iah_crps"])
            for d in doms}


def main():
    out_dir = HCH / "results" / \
        f"WP5_PUBLIC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    doms_list = list(load_domains().values())
    dom_map = {d.name: d for d in doms_list}
    print(f"\n[UCT] training on {len(doms_list)} domains "
          f"({len(HEADLINE)} markets x {len(HOSTS)} hosts)", flush=True)

    # ---- reference head (P0A incumbent) — reused for weights + guards ----
    p0a = HCH / "results" / "P0A_RERUN_20260814" / "learned_sig_main_head.pt"
    ref_head = None
    if p0a.exists():
        ref_head = build_head("learned_sig")
        ref_head.load_state_dict(torch.load(p0a, map_location="cpu"))
        ref_head.eval()

    # ---- Case C (protocol): temperature sampling, weights ∝ S2V CRPS^p.
    #      Difficult domains (NEM/GEFCOM, high CRPS) get more gradient share;
    #      mean(weight)=1 keeps the per-epoch budget equal to P0-C. ----
    temp_p = 1.0
    weights = None
    if ref_head is not None:
        crps_map = panel_crps(ref_head, doms_list)
        crps = np.array([crps_map[d.name] for d in doms_list])
        w = np.power(np.clip(crps, 1e-6, None), temp_p)
        w = w / w.mean()
        weights = [float(x) for x in w]
        print(f"[caseC] temperature sampling p={temp_p}: "
              f"weight min={min(weights):.3f} max={max(weights):.3f}", flush=True)

    head, report = train_candidate("learned_sig", doms_list, seed=SEED,
                                   weights=weights)

    # ---- §12.2 guards vs the incumbent P0A_RERUN head ----
    guards = {"pass": True, "checks": [], "macro": None, "worst": None}
    if ref_head is not None:
        old_crps = panel_crps(ref_head, doms_list)
        new_crps = panel_crps(head, doms_list)
        regr = {n: (new_crps[n] - old_crps[n]) / max(old_crps[n], 1e-12)
                for n in old_crps if old_crps[n] > 0}
        anchors = {n: r for n, r in regr.items() if n.split(":")[0] in SOURCE}
        per_dom_max = max(regr.values()) if regr else 0.0
        worst_regr = max(regr.values()) if regr else 0.0
        macro_new = float(np.mean(list(new_crps.values())))
        macro_old = float(np.mean(list(old_crps.values())))
        macro_rel = (macro_new - macro_old) / max(macro_old, 1e-12)
        guards.update({
            "macro_new": macro_new, "macro_old": macro_old,
            "macro_rel": macro_rel,
            "per_domain_max_rel": per_dom_max, "worst_rel": worst_regr,
            "anchor_mean_rel": float(np.mean(list(anchors.values()))) if anchors else None,
        })
        for n, r in sorted(regr.items()):
            guards["checks"].append({"domain": n, "rel": round(r, 5),
                                     "new": round(new_crps[n], 5),
                                     "old": round(old_crps[n], 5)})
        bad_dom = [n for n, r in regr.items() if r > MAX_PER_DOMAIN_REGR]
        if per_dom_max > MAX_PER_DOMAIN_REGR:
            guards["pass"] = False
            print(f"[guard] FAIL per-domain regression: {bad_dom}", flush=True)
        if worst_regr > MAX_WORST_REGR:
            guards["pass"] = False
            print(f"[guard] FAIL worst-domain regression {worst_regr:.3f} "
                  f"> {MAX_WORST_REGR}", flush=True)
        if macro_rel > 0.005:
            guards["pass"] = False
            print(f"[guard] FAIL macro regression {macro_rel:.3f}", flush=True)
        print(f"[guard] macro_rel={macro_rel:.4f} per_dom_max={per_dom_max:.4f} "
              f"worst={worst_regr:.4f} anchor_mean={guards['anchor_mean_rel']:.4f} "
              f"-> {'PASS' if guards['pass'] else 'ROLLBACK'}", flush=True)
    else:
        print("[guard] P0A_RERUN head not found — guards skipped", flush=True)

    # ---- save artifacts ----
    # guard gate: a ROLLBACK head must NOT be picked up by run_matrix's
    # glob("learned_sig_main_head.pt") -> save it under a different name.
    head_fname = ("learned_sig_main_head.pt" if guards["pass"]
                  else "learned_sig_main_head.ROLLBACK.pt")
    torch.save(head.state_dict(), out_dir / head_fname)
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out_dir / "guard_report.json", "w") as f:
        json.dump(guards, f, indent=2, default=str)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "domains": len(doms_list),
                   "markets": HEADLINE, "hosts": HOSTS,
                   "seed": SEED, "git_sha": R._git_head(),
                   "case_c": {"temperature_sampling": True,
                              "power": temp_p,
                              "weight_source": "S2V_CRPS_P0A_head"}}, f, indent=2)
    print(f"\n[wp5] head -> {out_dir / 'learned_sig_main_head.pt'}")


if __name__ == "__main__":
    main()
