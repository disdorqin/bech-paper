"""P1 Round-2, T2: market/host balanced sampling (full-coverage).

Direction T2 (P1 doc §5.2/§7.2): fix day-exposure imbalance.
  Current T0 = P0-C equal-domain sampling: K = median(N_g) updates/domain/epoch.
    - long domains (LAGO_DE/PJM, S2T 348d -> N_g=22) fully covered (N_g==K)
    - short domains (NEM_SA1, S2T 58d -> N_g=4) repeated 5.5x per epoch
  T2 = full-coverage sampling (UniversalCoreTrainer sampling="full_coverage"):
    every domain's FULL S2T batch set visited exactly once per epoch
    (no truncation, no repetition). Total updates/epoch = sum N_g (~192 vs 264);
    per-domain gradient weight becomes proportional to data volume.
    Honest side-effect: NEM gradient share 16/192 = 8.3% (vs 1/3 equal-weight).

Protocol (doc §5.3): every direction paired with T0; fixed seed[0,1,2],
day exposure, split, candidate loss; only batch sampler / domain counts change.
Eval: S2V macro CRPS paired vs T0 (TRAINER_CMP vA), then S4 readout later.

Run:
  python p1_t2.py --seeds 0 --epochs 3 --domains LAGO_DE:Linear,NEM_SA1:Linear --out results/P1_T2_SMOKE_20260814
  python p1_t2.py --seeds 0,1,2 --out results/P1_T2_<gitsha>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from r1b_generalization_screen import (  # noqa: E402
    SOURCE_MARKETS, HOSTS, EvalDomain, train_candidate,
)
import r1a_run as R  # noqa: E402


def prepare_subset(domains: list[str], seed: int = 0) -> dict[str, EvalDomain]:
    """Load only the requested source domains (subset for sanity)."""
    out = {}
    for name in domains:
        mk, bb = name.split(":")
        print(f"[t2/prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=seed)
        out[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
        print(f"   S2T={len(info.s2t_batches)} S2V={len(info.s2v_batches)}", flush=True)
    return out


def dump_curves(out_dir: Path, report: dict, seed: int):
    """Export the 5 P1-style curve CSVs from a training report history."""
    hist = report["history"]
    n = len(hist)
    # train_loss / s2v_macro
    with open(out_dir / f"train_loss_curve.csv", "w") as f:
        f.write("epoch,train_loss\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['train_loss']:.6f}\n")
    with open(out_dir / f"s2v_macro_curve.csv", "w") as f:
        f.write("epoch,macro_s2v,worst_s2v\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['macro_s2v']:.6f},{h['worst_s2v']:.6f}\n")
    # s2v_per_domain
    doms = list(hist[0]["per_domain"].keys())
    with open(out_dir / f"s2v_per_domain_curve.csv", "w") as f:
        f.write("epoch," + ",".join(doms) + "\n")
        for h in hist:
            f.write(h["epoch"].__str__() + "," + ",".join(
                f"{h['per_domain'][d]:.6f}" for d in doms) + "\n")
    # health
    h0 = hist[0]["health"]
    hcols = list(h0.keys())
    with open(out_dir / f"health_curve.csv", "w") as f:
        f.write("epoch," + ",".join(hcols) + "\n")
        for h in hist:
            f.write(h["epoch"].__str__() + "," + ",".join(
                f"{h['health'][c]:.6f}" for c in hcols) + "\n")
    # domain_sampling (updates per domain per epoch)
    with open(out_dir / f"domain_sampling.csv", "w") as f:
        f.write("epoch," + ",".join(doms) + "\n")
        for h in hist:
            upd = h["updates_per_domain"]
            f.write(h["epoch"].__str__() + "," + ",".join(
                str(upd[d]) for d in doms) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--domains", type=str, default="",
                    help="comma-separated subset for sanity; default=12 source")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        domains = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"P1_T2_{R.SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[t2] sampling=full_coverage seeds={seeds} epochs={args.epochs} "
          f"domains={len(domains)} out={out_dir}", flush=True)

    doms = prepare_subset(domains, seed=0)
    train_doms = [doms[n] for n in domains]

    manifest = {"seeds": seeds, "epochs": args.epochs,
                "sampling": "full_coverage",
                "domains": domains, "n_domains": len(domains),
                "heads": {}}
    for seed in seeds:
        print(f"\n[t2] train seed={seed} ...", flush=True)
        head, rep = train_candidate("learned_sig", train_doms, seed=seed,
                                    sampling="full_coverage",
                                    epochs=args.epochs)
        head_path = out_dir / f"t2_head_seed{seed}.pt"
        torch.save(head.state_dict(), head_path)
        with open(out_dir / f"training_report_seed{seed}.json", "w",
                  encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        dump_curves(out_dir, rep, seed)
        manifest["heads"][str(seed)] = {
            "path": str(head_path),
            "best_macro_s2v": rep["best_macro_s2v"],
            "worst_s2v_at_best": rep["worst_s2v_at_best"],
            "epochs_run": rep["epochs_run"],
            "n_domains": rep["n_domains"],
            "per_epoch_updates_total": [
                sum(h["updates_per_domain"].values()) for h in rep["history"]],
        }
        print(f"   seed{seed}: best_macro_s2v={rep['best_macro_s2v']:.4f} "
              f"worst={rep['worst_s2v_at_best']:.4f} "
              f"epochs_run={rep['epochs_run']}", flush=True)

    with open(out_dir / "t2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # S2V paired comparison vs T0 (TRAINER_CMP vA, same pipeline equal-sampling)
    t0 = {}
    t0rep = HERE / "results" / "TRAINER_CMP_20260814_seeds012" / "report.json"
    if t0rep.exists():
        data = json.load(open(t0rep, encoding="utf-8"))
        for s in ("0", "1", "2"):
            v = data.get("runs", {}).get(f"vA_seed{s}")
            if isinstance(v, dict):
                t0[s] = v.get("best_macro_s2v")
    rows = []
    for seed in seeds:
        t2m = manifest["heads"][str(seed)]["best_macro_s2v"]
        t0m = t0.get(str(seed))
        rows.append({"seed": seed, "T2_fullcoverage_best_s2v": t2m,
                     "T0_equal_best_s2v": t0m,
                     "delta_T2_minus_T0": (t2m - t0m) if t0m is not None else None})
    with open(out_dir / "s2v_paired_comparison.csv", "w", encoding="utf-8") as f:
        if rows:
            f.write("seed,T2_fullcoverage_best_s2v,T0_equal_best_s2v,delta_T2_minus_T0\n")
            for r in rows:
                f.write(f"{r['seed']},{r['T2_fullcoverage_best_s2v']:.6f},"
                        f"{r['T0_equal_best_s2v'] if r['T0_equal_best_s2v'] is not None else ''},"
                        f"{r['delta_T2_minus_T0'] if r['delta_T2_minus_T0'] is not None else ''}\n")
    print(f"\n[done] {out_dir}", flush=True)
    print("S2V paired comparison:", rows, flush=True)


if __name__ == "__main__":
    main()
