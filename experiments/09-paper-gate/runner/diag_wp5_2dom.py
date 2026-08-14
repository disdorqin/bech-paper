"""Diagnostic: 2-domain UniversalCoreTrainer smoke (WP-5 crash investigation).

Verifies the multi-domain UCT path is healthy after the 32-domain WP5 retrain
died silently (no traceback, ~20min CPU). Runs 2 domains x 12 epochs (~1-2min).
"""
from __future__ import annotations

import sys
import faulthandler
from datetime import datetime
from pathlib import Path

faulthandler.enable()

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
HCH = PAPER.parent / "08-hch-v2"
ROOT = PAPER.parent.parent
for p in (ROOT / "src", HCH):
    sys.path.insert(0, str(p))

import r1a_run as R                                # noqa: E402
from r1b_generalization_screen import HOSTS, SEED, EvalDomain, build_head, train_candidate  # noqa: E402

DOMS = [("LAGO_DE", "Linear"), ("NORD_DK1", "MLP")]


def main():
    print(f"[diag] {datetime.now().strftime('%H:%M:%S')} start 2-domain UCT", flush=True)
    doms = []
    for mk, bb in DOMS:
        name = f"{mk}:{bb}"
        print(f"[diag] prep {name}", flush=True)
        info = R.prepare_domain(mk, bb, seed=SEED)
        doms.append(EvalDomain(info=info, market=mk, host=bb, name=name))
    print("[diag] training...", flush=True)
    head, report = train_candidate("learned_sig", doms, seed=SEED)
    print(f"[diag] DONE macro_s2v={report.get('macro_s2v')} "
          f"worst={report.get('worst_s2v')}", flush=True)


if __name__ == "__main__":
    main()
