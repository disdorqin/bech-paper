"""P1 Round-3, T4: difficulty-based soft sampling weights (re-targeted).

Hypothesis doc: docs/训练文件夹/对比实验/hch_v2_p1_t4_soft_weight_hypothesis_prompt_v0.1_2026-08-14.md
  H4: low-data/high-difficulty domains (NEM_SA1, S2T 58d) get too few gradient
      steps under equal-domain sampling. Soft weights w_g from host-side,
      forecast-visible difficulty stats, budget preserved (mean(w)~=1, total
      n_domains*K updates/epoch unchanged), should help NEM without hurting
      DE/PJM -> beat T0 equal on S2V macro CRPS.

Arms (weights_from_host_stats modes):
  T4-A  host_s1r_mae   primary     training-only (reads S1R price; §5.2 labeled)
  T4-B  inv_nbatch     control     target-free  (data-volume inverse = T2 reverse)
  (T4-C volatility optional probe, noisy in pre-check; run only if A/B show signal)

Control = T0 equal (TRAINER_CMP vA seeds012), same pipeline, same seeds, same
best-epoch rule. Eval = S2V macro CRPS paired vs T0 + per-domain delta-vs-host
(method validated to reproduce the T2 table exactly).

Run:
  python p1_t4.py --seeds 0 --epochs 3 --modes host_s1r_mae --domains LAGO_DE:Linear,NEM_SA1:Linear --out results/P1_T4_SMOKE
  python p1_t4.py --seeds 0,1,2 --out results/P1_T4_<gitsha>
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from r1b_generalization_screen import (  # noqa: E402
    SOURCE_MARKETS, HOSTS, EvalDomain, train_candidate, weights_from_host_stats,
)
import r1a_run as R  # noqa: E402

T0_REPORT = HERE / "results" / "TRAINER_CMP_20260814_seeds012" / "report.json"
MODES = ("host_s1r_mae", "inv_nbatch", "volatility")
DE_PJM = ("LAGO_DE", "LAGO_PJM")
NEM = "NEM_SA1"


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def prepare_subset(domains: list[str], seed: int = 0) -> dict[str, EvalDomain]:
    out = {}
    for name in domains:
        mk, bb = name.split(":")
        print(f"[t4/prep] {name} ...", flush=True)
        info = R.prepare_domain(mk, bb, seed=seed)
        out[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
    return out


def best_epoch_index(rep: dict) -> int:
    best = rep.get("best_macro_s2v")
    for i, h in enumerate(rep["history"]):
        if best is not None and abs(h.get("macro_s2v", float("nan")) - best) < 1e-6:
            return i
    return int(min(range(len(rep["history"])), key=lambda i: rep["history"][i].get(
        "macro_s2v", float("inf"))))


def dump_curves(out_dir: Path, rep: dict):
    hist = rep["history"]
    doms = list(hist[0]["per_domain"].keys())
    with open(out_dir / "s2v_per_domain_curve.csv", "w") as f:
        f.write("epoch," + ",".join(doms) + "\n")
        for h in hist:
            f.write(f"{h['epoch']}," + ",".join(f"{h['per_domain'][d]:.6f}" for d in doms) + "\n")
    with open(out_dir / "s2v_macro_curve.csv", "w") as f:
        f.write("epoch,macro_s2v,worst_s2v\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['macro_s2v']:.6f},{h['worst_s2v']:.6f}\n")
    with open(out_dir / "train_loss_curve.csv", "w") as f:
        f.write("epoch,train_loss\n")
        for h in hist:
            f.write(f"{h['epoch']},{h['train_loss']:.6f}\n")


def load_t0() -> dict:
    """T0 equal-sampling per-domain delta-vs-host @best + macro, per seed."""
    data = json.load(open(T0_REPORT, encoding="utf-8"))
    t0 = {}
    for s in ("0", "1", "2"):
        v = data["runs"].get(f"vA_seed{s}")
        if isinstance(v, dict):
            t0[s] = {
                "best_macro_s2v": v.get("best_macro_s2v"),
                "domain_delta_crps": v.get("domain_delta_crps", {}),
            }
    return t0


def per_domain_delta_vs_t0(rep: dict, t0: dict) -> tuple[dict, float]:
    """Per-domain Δ(T4−T0) in delta-vs-host space @best epoch + macro Δ.

    Δ(d) = T4_delta_best(d) − T0_delta_best(d). Negative = T4 better.
    mean over the 12 domains == macro Δ (validated: reproduces T2 table exactly).
    """
    bi = best_epoch_index(rep)
    d4 = rep["history"][bi]["delta"]
    d0 = t0["domain_delta_crps"]
    delta = {k: d4[k] - d0[k] for k in d0}
    return delta, sum(delta.values()) / len(delta)


def keep_verdict(macro_deltas: list[float], de_pjm_degrade: list[int],
                 nem_improve: list[int], arm: str) -> tuple[str, list[str]]:
    """Doc §5 KEEP rule, operationalized:
      - 3/3 seeds macro Δ < 0
      - NEM: per seed ≥2/4 domains improve (Δ<0)
      - DE/PJM: per seed 0/8 domains degrade (Δ>0)
    Returns (verdict, reasons).
    """
    reasons = []
    ok_macro = all(d < 0 for d in macro_deltas)
    reasons.append(f"macro Δ<0 3/3: {['%.5f' % d for d in macro_deltas]} -> {'OK' if ok_macro else 'FAIL'}")
    ok_nem = all(c >= 2 for c in nem_improve)
    reasons.append(f"NEM improve ≥2/4 per seed: {nem_improve} -> {'OK' if ok_nem else 'FAIL'}")
    ok_depjm = all(c == 0 for c in de_pjm_degrade)
    reasons.append(f"DE/PJM degrade ==0 per seed: {de_pjm_degrade} -> {'OK' if ok_depjm else 'FAIL'}")
    if ok_macro and ok_nem and ok_depjm:
        return "KEEP", reasons
    if any(d < 0 for d in macro_deltas) or any(c >= 2 for c in nem_improve):
        return "INCONCLUSIVE", reasons
    return "REJECT", reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--modes", type=str, default="host_s1r_mae,inv_nbatch")
    ap.add_argument("--domains", type=str, default="",
                    help="comma-separated subset for sanity; default=12 source")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    modes = [m for m in args.modes.split(",") if m.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        domains = [f"{mk}:{bb}" for mk in SOURCE_MARKETS for bb in HOSTS]

    out_root = Path(args.out) if args.out else \
        HERE / "results" / f"P1_T4_{git_sha()}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[t4] modes={modes} seeds={seeds} epochs={args.epochs} "
          f"domains={len(domains)} out={out_root}", flush=True)

    doms = prepare_subset(domains, seed=0)
    train_doms = [doms[n] for n in domains]
    weights = {m: weights_from_host_stats(train_doms, mode=m) for m in modes}
    for m, w in weights.items():
        per_mkt = {}
        for name, wg in zip(domains, w):
            per_mkt.setdefault(name.split(":")[0], []).append(round(wg, 3))
        print(f"  weights[{m}]: " + " ".join(f"{k}{v}" for k, v in per_mkt.items()), flush=True)

    t0 = load_t0()
    manifest = {"seeds": seeds, "epochs": args.epochs, "sampling": "equal",
                "modes": modes, "domains": domains, "weights": {
                    m: [round(w, 6) for w in ws] for m, ws in weights.items()},
                "arms": {}, "t0_source": str(T0_REPORT)}

    arms_summary = {}
    for mode in modes:
        arm_dir = out_root / mode
        arm_dir.mkdir(parents=True, exist_ok=True)
        # sanity: weights must be finite, positive, mean ~1
        assert all(w > 0 for w in weights[mode]), f"{mode} non-positive weight"
        assert abs(sum(weights[mode]) / len(weights[mode]) - 1.0) < 1e-6, \
            f"{mode} budget not preserved (mean {sum(weights[mode])/len(weights[mode]):.4f})"

        per_seed = {}
        for seed in seeds:
            print(f"\n[t4] arm={mode} seed={seed} ...", flush=True)
            head, rep = train_candidate("learned_sig", train_doms, seed=seed,
                                        weights=weights[mode], sampling="equal",
                                        epochs=args.epochs)
            torch.save(head.state_dict(), arm_dir / f"t4_head_seed{seed}.pt")
            with open(arm_dir / f"training_report_seed{seed}.json", "w",
                      encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
            dump_curves(arm_dir, rep)
            per_seed[str(seed)] = {
                "best_macro_s2v": rep["best_macro_s2v"],
                "worst_s2v_at_best": rep["worst_s2v_at_best"],
                "epochs_run": rep["epochs_run"],
                "best_epoch": best_epoch_index(rep),
            }
            print(f"   best_macro_s2v={rep['best_macro_s2v']:.5f} "
                  f"worst={rep['worst_s2v_at_best']:.5f}", flush=True)
        arms_summary[mode] = per_seed
        manifest["arms"][mode] = per_seed
        with open(out_root / f"{mode}_manifest.json", "w", encoding="utf-8") as f:
            json.dump({**manifest, "arms": {mode: per_seed}}, f, ensure_ascii=False, indent=2)

    # ---- paired comparison + per-domain table ----
    rows = []
    per_domain_rows = []
    verdicts = {}
    for mode in modes:
        macro_deltas, depjm_degs, nem_impr = [], [], []
        for seed in seeds:
            rep = json.load(open(out_root / mode / f"training_report_seed{seed}.json",
                                 encoding="utf-8"))
            delta, macro = per_domain_delta_vs_t0(rep, t0[str(seed)])
            per_domain_rows.append({"arm": mode, "seed": seed, "macro_delta": macro,
                                    **delta})
            t4m = arms_summary[mode][str(seed)]["best_macro_s2v"]
            t0m = t0[str(seed)]["best_macro_s2v"]
            rows.append({"arm": mode, "seed": seed, "T4_best_s2v": t4m,
                         "T0_equal_best_s2v": t0m, "delta_T4_minus_T0": t4m - t0m})
            macro_deltas.append(macro)
            de = [v for k, v in delta.items() if k.startswith(DE_PJM[0]) or k.startswith(DE_PJM[1])]
            nem = [v for k, v in delta.items() if k.startswith(NEM)]
            depjm_degs.append(sum(1 for v in de if v > 0))
            nem_impr.append(sum(1 for v in nem if v < 0))
        verdict, reasons = keep_verdict(macro_deltas, depjm_degs, nem_impr, mode)
        verdicts[mode] = {"verdict": verdict, "reasons": reasons,
                          "macro_deltas": macro_deltas,
                          "de_pjm_n_degrade_per_seed": depjm_degs,
                          "nem_n_improve_per_seed": nem_impr}
        print(f"\n[{mode}] VERDICT: {verdict}")
        for r in reasons:
            print(f"    {r}", flush=True)

    with open(out_root / "s2v_paired_comparison.csv", "w", encoding="utf-8") as f:
        f.write("arm,seed,T4_best_s2v,T0_equal_best_s2v,delta_T4_minus_T0\n")
        for r in rows:
            f.write(f"{r['arm']},{r['seed']},{r['T4_best_s2v']:.6f},"
                    f"{r['T0_equal_best_s2v']:.6f},{r['delta_T4_minus_T0']:.6f}\n")
    with open(out_root / "per_domain_comparison.csv", "w", encoding="utf-8") as f:
        keys = list(per_domain_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in per_domain_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(out_root / "verdicts.json", "w", encoding="utf-8") as f:
        json.dump(verdicts, f, ensure_ascii=False, indent=2)

    print(f"\n[done] {out_root}", flush=True)
    for m, v in verdicts.items():
        print(f"  {m}: {v['verdict']}  (macro_deltas={v['macro_deltas']})", flush=True)


if __name__ == "__main__":
    main()
