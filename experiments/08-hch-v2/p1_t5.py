"""P1 Round-3, T5: domestic mixing-ratio grid + transfer matrix.

Hypothesis doc: docs/训练文件夹/对比实验/hch_v2_p1_t5_mixing_grid_hypothesis_prompt_v0.1_2026-08-14.md
  H5: adding admitted domestic domains (shandong_DA/gansu_DA/shaanxi_DA x 4 hosts)
      to universal-head training at domestic gradient share r should not hurt the
      32 foreign headline cells (transfer matrix) and should improve domestic
      holdout S4.

Mechanism (verified): UniversalCoreTrainer weights -> rng.choice(p=w/sum(w)),
total updates = n_domains*K unchanged; K recomputed on the enlarged pool.
r-controlled: foreign domains weight 1 each, domestic weight w_d = N_for*r/(N_dom*(1-r))
so domestic total gradient share == r. r=0 baseline = T0 equal heads (TRAINER_CMP vA).

Eval: 32 foreign headline cells (cells_all) + 40 domestic cells (10 modes x 4 hosts),
weighted_mean readout, S4 frozen confirm. r chosen on S2V (checkpoint selection only).

Run:
  python p1_t5.py --seeds 0 --epochs 3 --r-grid 0.15 --domains LAGO_DE:Linear,NEM_SA1:Linear,shandong_DA:Linear --out results/P1_T5_SMOKE
  python p1_t5.py --seeds 0,1,2 --r-grid 0.15,0.30 --out results/P1_T5_<gitsha>
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from r1b_generalization_screen import (  # noqa: E402
    SOURCE_MARKETS, HOSTS, EvalDomain, train_candidate,
)
import r1a_run as R                       # noqa: E402
import r1a9_action_calibration as M       # noqa: E402
from p1_round1 import HOSTS4, HEADLINE_DS, cell_eps, s4_eval, load_head  # noqa: E402
from common import PROVINCE_KEYS           # noqa: E402

FOREIGN_TRAIN = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]          # 12
DOMESTIC_TRAIN = [f"{mk}:{bb}" for mk in ("shandong_DA", "gansu_DA", "shaanxi_DA")
                  for bb in HOSTS4]                                               # 12
FOREIGN_CELLS = [f"{mk}:{bb}" for mk in HEADLINE_DS for bb in HOSTS4]            # 32
DOMESTIC_MODES = ["shandong_DA", "shandong_RT"] + list(PROVINCE_KEYS)            # 10
DOMESTIC_CELLS = [f"{mk}:{bb}" for mk in DOMESTIC_MODES for bb in HOSTS4]        # 40
T0_DIR = HERE / "results" / "TRAINER_CMP_20260814_seeds012"
READOUT = "weighted_mean"


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def prepare_subset(domains: list[str]) -> dict[str, EvalDomain]:
    out = {}
    for name in domains:
        mk, bb = name.split(":")
        print(f"[t5/prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=0)
        out[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    return out


def r_weights(n_for: int, n_dom: int, r: float) -> list[float]:
    """Foreign weight 1 each; domestic weight s.t. domestic total share == r."""
    assert 0.0 < r < 1.0
    w_d = n_for * r / (n_dom * (1.0 - r))
    return [1.0] * n_for + [w_d] * n_dom


def load_head_state(r: float, seed: int, out_root: Path) -> torch.nn.Module:
    if r == 0.0:
        return load_head(seed, torch)     # T0 equal vA head (TRAINER_CMP)
    path = out_root / f"r{r:g}" / f"t5_head_seed{seed}.pt"
    head = load_head(0, torch)            # same architecture
    head.load_state_dict(torch.load(path, map_location="cpu"))
    return head


def eval_cell(head: torch.nn.Module, mk: str, bb: str, info) -> dict:
    """S4 eval of one head on one cell, reusing a pre-built DomainInfo."""
    eps = cell_eps(info)
    dd = M.collect_domain(None, mk, bb, "learned_sig", head=head, info=info)
    s4 = s4_eval(dd, READOUT, eps)
    return {
        "mae": s4.get("mae"), "rmse": s4.get("rmse"),
        "smape_eps": s4.get("smape_eps"),
        "host_mae": s4.get("host_mae"),
        "improvement_vs_host": s4.get("improvement_vs_host"),
        "neg_price_mae": s4.get("neg_price_mae"),
    }


def _eval_cell_worker(args) -> list[dict]:
    """One cell per worker: prepare once (head-independent), then eval every
    (seed, r) head. Spawn-safe (module-level, picklable args)."""
    cell, out_root, combos = args
    mk, bb = cell.split(":")
    info = R.prepare_domain(mk, bb, seed=0)
    rows = []
    for seed, r in combos:
        head = load_head_state(r, seed, Path(out_root))
        row = eval_cell(head, mk, bb, info)
        rows.append({"seed": seed, "r": r, "cell": cell, "mode": mk, "host": bb,
                     "domain_tier": "foreign" if mk in HEADLINE_DS else "domestic",
                     **row})
    return rows


def t5_verdict(foreign_macro: dict, domestic_macro: dict, r_grid: list[float],
               seeds: list[int]) -> tuple[str, list[str]]:
    """Doc §3 rule, operationalized exactly as written:

      KEEP  foreign: 32-cell MACRO mean (across seeds) ΔMAE <= +0.5% noise band
      KEEP  domestic: some r with >=2/3 seeds domestic <= -2%
      REJECT foreign: some r with >=2/3 seeds foreign > +0.5%  (明确负迁移)
      REJECT domestic: no r reaches the domestic-improvement bar
      both keep_r and reject_r non-empty -> INCONCLUSIVE (r 之间不一致).

    (KEEP 国外条件按 doc 用"宏平均"= 跨 seed 均值;REJECT 国外条件按 doc 用
    "2/3 seed 一致"。二者并存时若同一 r 都触发 -> INCONCLUSIVE。)
    """
    reasons = []
    if not foreign_macro or not any(domestic_macro.get(r) for r in r_grid):
        # sanity with --no-domestic-holdout (or an empty tier): no verdict
        reasons.append("tier missing (foreign or domestic holdout) -> verdict n/a")
        return "N/A", reasons
    for r in r_grid:
        fw = foreign_macro.get(r, [])
        dm = domestic_macro.get(r, [])
        reasons.append(f"r={r:g}: foreign_macro_mean={float(np.mean(fw)):+.4f} "
                       f"per-seed={[round(x,4) for x in fw]}  "
                       f"domestic_per-seed={[round(x,4) for x in dm]}")
    dom_ok = [r for r in r_grid
              if sum(1 for m in domestic_macro.get(r, []) if m <= -0.02) >= 2]
    keep_r = [r for r in r_grid
              if float(np.mean(foreign_macro.get(r, []))) <= 0.005 and r in dom_ok]
    reject_r = [r for r in r_grid
                if sum(1 for m in foreign_macro.get(r, []) if m > 0.005) >= 2]
    reasons.append(f"domestic_improve_r(≥2/3 seeds ≤-2%)={dom_ok}  "
                   f"keep_r(foreign_macro_mean≤+0.5% ∩ dom_ok)={keep_r}  "
                   f"foreign_reject_r(≥2/3 seeds >+0.5%)={reject_r}")
    if not dom_ok:
        return "REJECT", reasons
    if keep_r and not reject_r:
        return "KEEP", reasons
    if reject_r and not keep_r:
        return "REJECT", reasons
    return "INCONCLUSIVE", reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--r-grid", type=str, default="0.15,0.30")
    ap.add_argument("--domains", type=str, default="",
                    help="comma-separated subset for sanity; default=24 pool")
    ap.add_argument("--eval-skip", action="store_true",
                    help="skip the S4 eval (train + weights sanity only)")
    ap.add_argument("--no-domestic-holdout", action="store_true",
                    help="eval only the 32 foreign cells (sanity)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    r_grid = [float(x) for x in args.r_grid.split(",") if x.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    sanity = bool(domains)
    if not domains:
        domains = FOREIGN_TRAIN + DOMESTIC_TRAIN

    out_root = Path(args.out) if args.out else \
        HERE / "results" / f"P1_T5_{git_sha()}"
    out_root.mkdir(parents=True, exist_ok=True)

    n_for = sum(1 for d in domains if d in FOREIGN_TRAIN)
    n_dom = len(domains) - n_for
    print(f"[t5] r-grid={r_grid} seeds={seeds} epochs={args.epochs} "
          f"pool={len(domains)} (foreign={n_for}, domestic={n_dom}) "
          f"sanity={sanity} out={out_root}", flush=True)

    doms = prepare_subset(domains)
    train_doms = [doms[n] for n in domains]
    n_g = {d.name: len(d.info.s2t_batches) for d in train_doms}
    print(f"  N_g per domain: {n_g}", flush=True)

    weights = {r: r_weights(n_for, n_dom, r) for r in r_grid}
    for r, w in weights.items():
        domestic_share = sum(w[n_for:]) / sum(w)
        print(f"  r={r:g}: w_dom={w[n_for]:.4f} (domestic gradient share = "
              f"{domestic_share:.3f})", flush=True)
        assert abs(domestic_share - r) < 1e-9

    manifest = {"seeds": seeds, "epochs": args.epochs, "r_grid": r_grid,
                "pool": domains, "n_foreign": n_for, "n_domestic": n_dom,
                "weights": {f"r{r:g}": [round(w, 5) for w in ws]
                            for r, ws in weights.items()},
                "K": int(np.median([len(d.info.s2t_batches) for d in train_doms])),
                "heads": {}}

    for r in r_grid:
        arm_dir = out_root / f"r{r:g}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            print(f"\n[t5] r={r:g} seed={seed} train ...", flush=True)
            head, rep = train_candidate("learned_sig", train_doms, seed=seed,
                                        weights=weights[r], sampling="equal",
                                        epochs=args.epochs)
            torch.save(head.state_dict(), arm_dir / f"t5_head_seed{seed}.pt")
            with open(arm_dir / f"training_report_seed{seed}.json", "w",
                      encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            manifest["heads"][f"r{r:g}_seed{seed}"] = {
                "best_macro_s2v": rep["best_macro_s2v"],
                "worst_s2v_at_best": rep["worst_s2v_at_best"],
                "epochs_run": rep["epochs_run"],
                "per_epoch_updates_total": [
                    sum(h["updates_per_domain"].values()) for h in rep["history"]],
            }
            print(f"   best_macro_s2v={rep['best_macro_s2v']:.5f} "
                  f"updates/epoch={manifest['heads'][f'r{r:g}_seed{seed}']['per_epoch_updates_total']}",
                  flush=True)
        with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    if args.eval_skip:
        print(f"\n[done (eval skipped)] {out_root}", flush=True)
        return

    # ---- S4 eval: per (seed, r) on foreign 32 + domestic 40 ----
    # Each cell is prepared ONCE (head-independent) and shared across the
    # (seed x r) heads. Cells are distributed across worker processes.
    eval_cells = [f"{mk}:{bb}" for mk in HEADLINE_DS for bb in HOSTS4]  # 32
    if not args.no_domestic_holdout:
        eval_cells += [f"{mk}:{bb}" for mk in DOMESTIC_MODES for bb in HOSTS4]  # 40
    combos = [(seed, r) for seed in seeds for r in [0.0] + r_grid]

    eval_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_eval_cell_worker, (c, str(out_root), combos)): c
                   for c in eval_cells}
        done = 0
        for fut in as_completed(futures):
            cell = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"  [EVAL-ERROR] {cell}: {e}", flush=True)
                continue
            eval_rows.extend(rows)
            done += 1
            r0 = rows[0]
            print(f"  [{done}/{len(eval_cells)}] {cell:24s} "
                  f"mae(r0,s0)={r0['mae']} imp={r0['improvement_vs_host']}", flush=True)

    with open(out_root / "eval_rows.json", "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, ensure_ascii=False, indent=2)
    for seed in seeds:
        with open(out_root / f"eval_rows_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump([r_ for r_ in eval_rows if r_["seed"] == seed], f,
                      ensure_ascii=False, indent=2)

    # ---- aggregates ----
    def rel_macro(r: float, tier: str) -> list[float]:
        """Per-seed mean relative ΔMAE (r vs 0) over a tier."""
        out_m = []
        for seed in seeds:
            base = {x["cell"]: x["mae"] for x in eval_rows
                    if x["seed"] == seed and x["r"] == 0.0 and x["domain_tier"] == tier}
            diffs = []
            for x in eval_rows:
                if x["seed"] == seed and x["r"] == r and x["domain_tier"] == tier:
                    m0 = base[x["cell"]]
                    diffs.append((x["mae"] - m0) / m0 if m0 else 0.0)
            out_m.append(float(np.mean(diffs)))
        return out_m

    foreign_macro = {r: rel_macro(r, "foreign") for r in r_grid}
    domestic_macro = {r: rel_macro(r, "domestic") for r in r_grid}

    # per-cell transfer matrix
    with open(out_root / "transfer_matrix.csv", "w", encoding="utf-8") as f:
        header = ["cell", "tier", "seed", "r", "mae_r", "mae_0", "dmae_rel"]
        f.write(",".join(header) + "\n")
        for seed in seeds:
            base = {x["cell"]: x for x in eval_rows
                    if x["seed"] == seed and x["r"] == 0.0}
            for x in eval_rows:
                if x["seed"] != seed or x["r"] == 0.0:
                    continue
                m0 = base[x["cell"]]["mae"]
                f.write(f"{x['cell']},{x['domain_tier']},{seed},{x['r']:g},"
                        f"{x['mae']},{m0},{(x['mae']-m0)/m0 if m0 else ''}\n")

    summary = {"foreign_macro_dmae_rel": {f"r{r:g}": foreign_macro[r] for r in r_grid},
               "domestic_macro_dmae_rel": {f"r{r:g}": domestic_macro[r] for r in r_grid}}
    verdict, reasons = t5_verdict(foreign_macro, domestic_macro, r_grid, seeds)
    summary["verdict"] = verdict
    summary["reasons"] = reasons
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nVERDICT: {verdict}")
    for r_ in reasons:
        print(f"  {r_}")
    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
