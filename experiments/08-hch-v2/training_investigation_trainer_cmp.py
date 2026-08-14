"""HCH-v2 §5 跨域训练方式对照(Version A vs B)——只改一个因素。

Version A (当前): UniversalCoreTrainer 逐 batch sequential step,
   每域 K=median(n_batches) 次 update/epoch, 等域采样(P0-C)。
Version B (macro-domain gradient accumulation): 每个 macro-step 对每个域
   各取一个 batch, loss_g/|G| 累积, 统一 optimizer.step(); 每 epoch 跑 K 轮。

保持完全一致: EPOCHS=12, lr=3e-4, wd=1e-4, clip=1.0, seed, head 结构,
   checkpoint 选择 = macro S2V IAH-CRPS, 12 source domains。
输出 results/TRAINER_CMP_<ts>/ 新目录。不改 src/。

诚实纪律(v0.2 §9): 每次只改一个因素; 不把"训练轮数增加后偶然变好"写成
模块有效性; 结果全量记录, 不只报改进方。
"""
from __future__ import annotations

import argparse
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

from iah_crps_loss import iah_crps_loss
from universal_trainer import (DomainBatch, _eval_batch_losses,
                               _eval_host_baseline, _collect_health)
import r1a_run as R
from r1b_generalization_screen import (SOURCE_MARKETS, HOSTS, SEED, LR, WD,
                                       CLIP, EPOCHS, PATIENCE, build_head,
                                       det_for_variant, train_candidate)

VERSION_A_CFG = {"epochs": EPOCHS, "lr": LR, "wd": WD, "clip": CLIP,
                 "patience": PATIENCE, "seed": SEED}


def train_version_a(train_doms, seed=SEED) -> tuple[nn.Module, dict]:
    """Current sequential per-domain update — reuse the R1B formal runner's
    train_candidate so Version A is byte-identical to R1B (strongest control)."""
    return train_candidate("learned_sig", train_doms, seed=seed)


def train_version_b(train_doms, epochs=EPOCHS, lr=LR, weight_decay=WD,
                    clip=CLIP, patience=PATIENCE, seed=SEED
                    ) -> tuple[nn.Module, dict]:
    """Macro-domain gradient accumulation: per macro-step, each domain's next
    batch contributes loss/|G|; one optimizer.step() per macro-step."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    head = build_head("learned_sig")
    domains = []
    for d in train_doms:
        det = det_for_variant("learned_sig", d.info)
        domains.append(DomainBatch(name=d.name, s2t_batches=d.info.s2t_batches,
                                   s2v_batches=d.info.s2v_batches,
                                   domain_det=det))
    domains = [d for d in domains if d.s2t_batches]
    n_domains = len(domains)
    K = int(np.median([len(d.s2t_batches) for d in domains]))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_state, best_macro, worst_at_best = None, float("inf"), float("inf")
    pat = 0
    history = []

    def _eval_all():
        per_g, worst, host_g, delta_g = {}, float("-inf"), {}, {}
        for d in domains:
            loss, n = _eval_batch_losses(head, d.s2v_batches, d.det_tensor(1))
            per_g[d.name] = float(loss)
            host_g[d.name] = _eval_host_baseline(d.s2v_batches)
            if np.isfinite(loss):
                delta_g[d.name] = float(loss) - host_g[d.name]
                worst = max(worst, float(loss))
        macro = (float(np.mean([v for v in per_g.values() if np.isfinite(v)]))
                 if per_g else float("nan"))
        health = _collect_health(head, domains)
        return macro, worst, per_g, host_g, delta_g, health

    for ep in range(epochs):
        cursors = [0] * n_domains
        opt.zero_grad()
        epoch_losses, grad_norms, nan_batches = [], [], 0
        scale_invalid = 0
        updates = [0] * n_domains
        for _t in range(K):
            opt.zero_grad()   # 每个 macro-step 统一累积 → step
            for g in range(n_domains):
                d = domains[g]
                nb = len(d.s2t_batches)
                bi = cursors[g] % nb
                cursors[g] += 1
                batch = d.s2t_batches[bi]
                host, ctx, target, vm = batch[:4]
                det = batch[4] if len(batch) >= 5 else d.det_tensor(host.shape[0])
                if det is not None and det.shape[0] != host.shape[0]:
                    det = det.expand(host.shape[0], -1)
                out = head(host, ctx, valid_mask=vm, domain_det=det)
                loss = iah_crps_loss(out, target)
                (loss / n_domains).backward()
                if not np.isfinite(float(loss.detach())):
                    nan_batches += 1
                scale_invalid += int((out["scale_valid"] < 0.5).sum())
                updates[g] += 1
                epoch_losses.append(float(loss.detach()))
            gnorm = nn.utils.clip_grad_norm_(head.parameters(), clip).item()
            grad_norms.append(float(gnorm))
            if not np.isfinite(gnorm):
                nan_batches += 1
            opt.step()

        macro, worst, per_g, host_g, delta_g, health = _eval_all()
        updates_per_domain = {domains[g].name: updates[g] for g in range(n_domains)}
        history.append({"epoch": ep, "macro_s2v": macro, "worst_s2v": worst,
                        "per_domain": per_g, "host_baseline": host_g,
                        "delta": delta_g, "health": health,
                        "updates_per_domain": updates_per_domain,
                        "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                        "grad_health": {"mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                                        "nan_inf_batches": nan_batches,
                                        "scale_unidentified_days": scale_invalid}})
        print(f"  [B] ep{ep}: macro_s2v={macro:.4f} worst={worst:.4f} "
              f"train={history[-1]['train_loss']:.4f}")
        if np.isfinite(macro) and macro < best_macro - 1e-5:
            best_macro = macro
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            worst_at_best = worst
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"  [B] early stop at ep{ep}; best_macro={best_macro:.4f}")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"best_macro_s2v": float(best_macro),
                  "worst_s2v_at_best": float(worst_at_best),
                  "n_domains": n_domains, "epochs_run": len(history),
                  "history": history, "method": "B_macro_accum"}


def eval_transfer(head, dom) -> dict:
    """Frozen transfer eval on S2V (macro CRPS + worst + per-domain delta)."""
    det = det_for_variant("learned_sig", dom.info)
    db = DomainBatch(name=dom.name, s2v_batches=dom.info.s2v_batches,
                     domain_det=det)
    loss, _ = _eval_batch_losses(head, db.s2v_batches, db.det_tensor(1))
    host = _eval_host_baseline(db.s2v_batches)
    return {"domain": dom.name, "iah_crps": float(loss), "host_base": float(host),
            "delta_crps": float(loss) - float(host)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--domains-src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"TRAINER_CMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 12 source domains (same as R1B main screen)
    doms = {}
    markets = [m.strip() for m in (args.domains_src or ",".join(SOURCE_MARKETS)).split(",") if m.strip()]
    for mk in markets:
        for bb in HOSTS:
            name = f"{mk}:{bb}"
            print(f"[prep] {name}", flush=True)
            info = R.prepare_domain(mk, bb, seed=SEED)
            doms[name] = type("D", (), {"info": info, "name": name, "market": mk, "host": bb})()
    train_doms = list(doms.values())

    report = {"config": {"epochs": args.epochs, "seeds": seeds,
                         "source_domains": [d.name for d in train_doms]},
              "runs": {}}
    for seed in seeds:
        for ver in ("A", "B"):
            t0 = time.time()
            print(f"\n===== Version {ver} seed {seed} =====", flush=True)
            if ver == "A":
                head, rep = train_version_a(train_doms, seed=seed)
            else:
                head, rep = train_version_b(train_doms, epochs=args.epochs,
                                            seed=seed)
            rows = [eval_transfer(head, d) for d in train_doms]
            deltas = [r["delta_crps"] for r in rows if np.isfinite(r["delta_crps"])]
            key = f"v{ver}_seed{seed}"
            report["runs"][key] = {
                "best_macro_s2v": rep["best_macro_s2v"],
                "worst_s2v_at_best": rep["worst_s2v_at_best"],
                "epochs_run": rep["epochs_run"],
                "domain_delta_crps": {r["domain"]: round(r["delta_crps"], 6)
                                      for r in rows},
                "macro_delta_crps": round(float(np.mean(deltas)), 6) if deltas else None,
                "domain_var": round(float(np.var(deltas)), 6) if len(deltas) >= 2 else None,
                "worst_delta_crps": round(float(np.max(deltas)), 6) if deltas else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  {key}: macro_delta={report['runs'][key]['macro_delta_crps']} "
                  f"best_macro_s2v={rep['best_macro_s2v']:.5f} "
                  f"worst={rep['worst_s2v_at_best']:.5f}")
            torch.save(head.state_dict(),
                       out_dir / f"head_v{ver}_seed{seed}.pt")

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[done] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
