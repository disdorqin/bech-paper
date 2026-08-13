"""R1B Generalization Screen — source/transfer separation enforced (sprint §8-13).

One-shot single-seed candidate screening. NEVER uses S4 for selection, never
tunes after results, never runs multi-seed confirmation (sprint §1).

Scientific contract:
  - SOURCE TRAINING DOMAINS and TRANSFER EVALUATION DOMAINS are explicit.
  - Candidate weights theta_IAH are frozen during every transfer evaluation.
  - DK1 contributes ZERO gradient (unseen market, §11).
  - LOHO-PatchTST contributes ZERO S2T gradient and ZERO S2V selection signal
    for the candidate that generalizes to it (§10).

Variants (sprint §9):
  learned_sig  (P1, provisional main): learned per-day Data Signature pool +
               identity-init FiLM; deterministic descriptor zeroed (R1A naming).
  plain_core   (P2, real bypass): h' = h. No DataSignature, no FiLM — a true
               no-signature control, not deterministic-descriptor=0 while the
               learned pool remains active.

Artifacts (R1B_SCREEN_<ts>/):
  candidate_transfer_matrix.csv   (§12 schema, one row per evaluated domain)
  loho_patchtst_matrix.csv        (LOHO candidate over all 16 domains)
  dk1_zero_gradient.csv           (§11 DK1 rows, candidate-level only)
  training_reports.json           (macro S2V CRPS, L_worst, per-domain, epochs)
  four_cell_aggregate.csv         (§12: Seen/Unseen market x host cells)
  config.json                     (full config + git SHA)
  summary.txt                     (human verdict + STOP/CONTINUE decision)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from iah_candidate import IAHCandidateHead
from universal_trainer import DomainBatch, UniversalCoreTrainer, _eval_batch_losses, _eval_host_baseline

import r1a_run as R  # reuse prepare_domain, det_for_variant, D_* config


# ---------------------------------------------------------------- config ----
SOURCE_MARKETS = ["LAGO_DE", "LAGO_PJM", "NEM_SA1"]
UNSEEN_MARKET = "NORD_DK1"
HOSTS = ["Linear", "MLP", "LSTM", "PatchTST"]
SEEN_HOSTS_LOHO = ["Linear", "MLP", "LSTM"]   # §10: PatchTST excluded
UNSEEN_HOST = "PatchTST"

D_CORE_CONTEXT = R.D_CORE_CONTEXT          # 13
D_MODEL = R.D_MODEL                        # 64
D_SIG = R.D_SIG                            # 32
D_VALUE = R.D_VALUE                        # 0 (optional branch off)
SEED = R.SEED                              # 0
LR = R.LR                                  # 3e-4
WD = R.WD                                  # 1e-4
CLIP = R.CLIP                              # 1.0
EPOCHS = 12
PATIENCE = 4

VARIANTS = ["learned_sig", "plain_core"]
VARIANT_LABELS = {"learned_sig": "LearnedSig", "plain_core": "PlainCore"}

DET_ZERO = np.zeros(8, dtype=np.float64)

# ------------------------------------------------------------- PlainCore ----
class _PlainCoreEncoder(nn.Module):
    """True no-DataSignature/FiLM path (sprint §9 P2): h' = h.

    Identical proj+LayerNorm as the core, but the DataSignature module and
    the FiLM modulation (1+Δγ)·h + β are REMOVED, not zeroed. This is the
    ablation control for the learned signature.
    """

    def __init__(self, d_core_in: int, d_model: int = 64, d_sig: int = 32):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_core_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, core_input: torch.Tensor,
                domain_det: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(self.proj(core_input))


class PlainCoreHead(IAHCandidateHead):
    """IAH head with the DataSignature/FiLM stage fully bypassed."""

    def __init__(self, d_core_context: int, d_model: int = 64,
                 d_value: int = 0, d_sig: int = 32):
        super().__init__(d_core_context, d_model, d_value=d_value, d_sig=d_sig)
        self.core_encoder = _PlainCoreEncoder(d_core_context + 1, d_model, d_sig)


def build_head(variant: str) -> nn.Module:
    if variant == "plain_core":
        return PlainCoreHead(D_CORE_CONTEXT, D_MODEL, d_value=D_VALUE, d_sig=D_SIG)
    return IAHCandidateHead(D_CORE_CONTEXT, D_MODEL, d_value=D_VALUE, d_sig=D_SIG)


def det_for_variant(variant: str, info: R.DomainInfo) -> np.ndarray:
    """PlainCore ignores det; learned_sig uses zeros; learned_det_sig uses S1R."""
    if variant == "plain_core":
        return DET_ZERO
    return R.det_for_variant(variant, info)


# ------------------------------------------------------- domain preparation ----
@dataclass
class EvalDomain:
    info: R.DomainInfo
    market: str
    host: str
    name: str

    @property
    def market_seen(self) -> bool:
        return self.market in SOURCE_MARKETS

    @property
    def host_seen(self) -> bool:
        return self.host != UNSEEN_HOST


def prepare_all() -> dict[str, EvalDomain]:
    """Load all 16 domains: 4 markets x 4 hosts (caches must exist)."""
    out = {}
    for mk in SOURCE_MARKETS + [UNSEEN_MARKET]:
        for bb in HOSTS:
            name = f"{mk}:{bb}"
            print(f"[r1b/prep] {name} ...", flush=True)
            info = R.prepare_domain(mk, bb, seed=SEED)
            out[name] = EvalDomain(info=info, market=mk, host=bb, name=name)
            print(f"   S2T={len(info.s2t_batches)} S2V={len(info.s2v_batches)}")
    return out


def domain_batches(dom: EvalDomain) -> DomainBatch:
    det = det_for_variant("learned_sig", dom.info)  # zeros for sig; ignored by core
    return DomainBatch(name=dom.name,
                       s2t_batches=dom.info.s2t_batches,
                       s2v_batches=dom.info.s2v_batches,
                       domain_det=det)


# ----------------------------------------------------- candidate-level eval ----
def domain_health(head: nn.Module, dom: EvalDomain, det_np: np.ndarray) -> dict:
    """Per-domain mass/shift/scale health over S2V (transfer eval, no chain)."""
    w_m, w_z, w_p, m_m, m_p = [], [], [], [], []
    det_t = R.det_for(det_np, 1)  # [1, d_det]
    with torch.no_grad():
        for batch in dom.info.s2v_batches:
            host, ctx, _, vm = batch[:4]
            det_b = det_t.expand(host.shape[0], -1) if det_t.shape[0] != host.shape[0] else det_t
            out = head(host, ctx, valid_mask=vm, domain_det=det_b)
            vh = (vm > 0.5) & torch.isfinite(out["z0"])
            cnt = float(vh.float().sum().clamp(min=1))
            w_m.append(float((out["w_minus"] * vh.float()).sum() / cnt))
            w_z.append(float((out["w_zero"] * vh.float()).sum() / cnt))
            w_p.append(float((out["w_plus"] * vh.float()).sum() / cnt))
            m_m.append(out["m_minus"][vh].detach().cpu().numpy())
            m_p.append(out["m_plus"][vh].detach().cpu().numpy())
    wm, wz, wp = float(np.mean(w_m)), float(np.mean(w_z)), float(np.mean(w_p))
    p = np.clip([wm, wz, wp], 1e-12, None)
    entropy = float(-(p * np.log(p)).sum())
    mm = np.concatenate(m_m) if m_m else np.zeros(1)
    mp = np.concatenate(m_p) if m_p else np.zeros(1)
    tiny = 1e-4
    shifts = np.concatenate([mm, mp])
    return {
        "mass_entropy": entropy,
        "w0": wz,
        "mminus_alive": float(np.mean(mm > tiny)),
        "mplus_alive": float(np.mean(mp > tiny)),
        "shift_p50": float(np.median(shifts)),
        "shift_p95": float(np.percentile(shifts, 95)),
        "scale_invalid_days": 0,  # filled below if available
    }


def eval_domain(head: nn.Module, dom: EvalDomain, variant: str) -> dict:
    """Candidate-level transfer row for one domain (frozen theta_IAH)."""
    det_np = det_for_variant(variant, dom.info)
    det_t = R.det_for(det_np, 1)
    iah_crps, _ = _eval_batch_losses(head, dom.info.s2v_batches, det_t)
    host_base = _eval_host_baseline(dom.info.s2v_batches)
    h = domain_health(head, dom, det_np)
    delta = iah_crps - host_base if np.isfinite(iah_crps) and np.isfinite(host_base) else float("nan")
    status = "OK"
    if not (np.isfinite(iah_crps) and np.isfinite(delta)):
        status = "NAN_OR_INF"
    elif delta > 0.05:
        status = "HOST_BETTER"  # candidate clearly worse than host on S2V
    return {
        "market_seen": int(dom.market_seen),
        "host_seen": int(dom.host_seen),
        "candidate_variant": VARIANT_LABELS.get(variant, variant),
        "host": dom.host,
        "market": dom.market,
        "host_baseline": round(float(host_base), 6),
        "iah_crps": round(float(iah_crps), 6) if np.isfinite(iah_crps) else None,
        "delta_crps": round(float(delta), 6) if np.isfinite(delta) else None,
        "mass_entropy": round(h["mass_entropy"], 6),
        "w0": round(h["w0"], 6),
        "mminus_alive": round(h["mminus_alive"], 6),
        "mplus_alive": round(h["mplus_alive"], 6),
        "shift_p50": round(h["shift_p50"], 6),
        "shift_p95": round(h["shift_p95"], 6),
        "status": status,
    }


def train_candidate(variant: str, train_domains: list[EvalDomain]) -> tuple[nn.Module, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    head = build_head(variant)
    domains = [domain_batches(d) for d in train_domains]
    trainer = UniversalCoreTrainer(head, seed=SEED)
    report = trainer.train(domains, epochs=EPOCHS, lr=LR, weight_decay=WD,
                           clip=CLIP, patience=PATIENCE)
    head.eval()
    return head, report


def aggregate_cells(rows: list[dict]) -> list[dict]:
    """§12 four aggregate cells (mean delta_crps + worst)."""
    cells = []
    for seen_mkt in (1, 0):
        for seen_host in (1, 0):
            sub = [r for r in rows if r["market_seen"] == seen_mkt and r["host_seen"] == seen_host]
            d = [r["delta_crps"] for r in sub if r["delta_crps"] is not None]
            cells.append({
                "market_seen": int(seen_mkt), "host_seen": int(seen_host),
                "label": f"{'Seen' if seen_mkt else 'Unseen'}_market/"
                         f"{'Seen' if seen_host else 'Unseen'}_host",
                "n_domains": len(sub),
                "mean_delta_crps": round(float(np.mean(d)), 6) if d else None,
                "worst_delta_crps": round(float(np.max(d)), 6) if d else None,
                "best_delta_crps": round(float(np.min(d)), 6) if d else None,
            })
    return cells


# ------------------------------------------------------------------ main ----
def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--variants", type=str, default=",".join(VARIANTS))
    ap.add_argument("--only-loho", action="store_true",
                    help="run LOHO-PatchTST candidate only (skip main 12-domain screen)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    EPOCHS = args.epochs

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        assert v in VARIANTS, f"unknown variant {v}"

    out_dir = Path(args.out) if args.out else \
        HERE / "results" / f"R1B_SCREEN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    doms = prepare_all()

    # ---- candidate configurations ----
    # main: 3 source markets x 4 hosts = 12 source domains
    main_train = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in HOSTS]
    # LOHO: 3 source markets x 3 seen hosts = 9 source domains (PatchTST excluded)
    loho_train = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS for bb in SEEN_HOSTS_LOHO]
    # eval set: all 16 domains (12 source + 4 DK1)
    eval_order = [doms[f"{mk}:{bb}"] for mk in SOURCE_MARKETS + [UNSEEN_MARKET] for bb in HOSTS]

    config = {
        "protocol": "hch_v2_r1b_two_hour_autonomous_research_sprint_v0.1_2026-08-13.md",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": R._git_head(),
        "source_markets": SOURCE_MARKETS,
        "unseen_market": UNSEEN_MARKET,
        "hosts": HOSTS,
        "seen_hosts_loho": SEEN_HOSTS_LOHO,
        "unseen_host": UNSEEN_HOST,
        "candidate_params": {"d_model": D_MODEL, "d_sig": D_SIG, "d_value": D_VALUE,
                             "seed": SEED, "lr": LR, "weight_decay": WD,
                             "grad_clip": CLIP, "epochs": EPOCHS, "patience": PATIENCE},
        "variants": variants,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_rows, reports, cells_out = {}, {}, {}

    for variant in variants:
        label = VARIANT_LABELS[variant]
        if not args.only_loho:
            print(f"\n===== MAIN SCREENING {label} (12 source domains) =====", flush=True)
            head, report = train_candidate(variant, main_train)
            reports[f"{label}_main"] = report
            rows = [eval_domain(head, d, variant) for d in eval_order]
            all_rows[f"{label}_main"] = rows
            cells_out[f"{label}_main"] = aggregate_cells(rows)
            _write_matrix(out_dir / f"matrix_{label}_main.csv", rows, label)

        print(f"\n===== LOHO-PatchTST {label} (9 source domains, no PatchTST grad) =====", flush=True)
        head_l, report_l = train_candidate(variant, loho_train)
        reports[f"{label}_LOHO"] = report_l
        rows_l = [eval_domain(head_l, d, variant) for d in eval_order]
        all_rows[f"{label}_LOHO"] = rows_l
        cells_out[f"{label}_LOHO"] = aggregate_cells(rows_l)
        _write_matrix(out_dir / f"matrix_{label}_LOHO.csv", rows_l, label)

    # ---- DK1 zero-gradient extract (§11) ----
    dk1_rows = []
    for key, rows in all_rows.items():
        for r in rows:
            if r["market"] == UNSEEN_MARKET:
                dk1_rows.append({"screen": key, **r})
    _write_matrix(out_dir / "dk1_zero_gradient.csv", dk1_rows, "DK1")

    # ---- training reports ----
    with open(out_dir / "training_reports.json", "w") as f:
        json.dump(_summarize_reports(reports), f, indent=2)

    # ---- four-cell aggregate ----
    with open(out_dir / "four_cell_aggregate.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        keys = list(cells_out.keys())
        cols = ["screen"] + list(cells_out[keys[0]][0].keys()) if keys else ["screen"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, cells in cells_out.items():
            for c in cells:
                w.writerow({"screen": key, **c})

    # ---- summary + STOP/CONTINUE ----
    lines = _build_summary(config, cells_out, reports)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[r1b] artifacts: {out_dir}")


def _write_matrix(path: Path, rows: list[dict], tag: str):
    import csv
    base = ["market_seen", "host_seen", "candidate_variant", "host", "market",
            "host_baseline", "iah_crps", "delta_crps", "mass_entropy", "w0",
            "mminus_alive", "mplus_alive", "shift_p50", "shift_p95", "status"]
    # extra keys (e.g. "screen" on DK1 rows) go first so the writer never
    # rejects a row it must serialize (crash fix: dk1_zero_gradient write).
    extra = []
    if rows:
        extra = [k for k in rows[0].keys() if k not in base]
    cols = extra + base
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _summarize_reports(reports: dict) -> dict:
    out = {}
    for key, rep in reports.items():
        out[key] = {
            "best_macro_s2v": rep.get("best_macro_s2v"),
            "worst_s2v_at_best": rep.get("worst_s2v_at_best"),
            "epochs_run": rep.get("epochs_run"),
            "per_domain_val": rep["history"][-1].get("per_domain") if rep.get("history") else {},
            "health": rep["history"][-1].get("health") if rep.get("history") else {},
        }
    return out


def _build_summary(config: dict, cells: dict, reports: dict) -> list[str]:
    lines = [
        f"R1B Generalization Screen — summary",
        f"date: {config['date']}",
        f"git: {config['git_sha']}",
        "",
        "== training ==",
    ]
    for key, rep in reports.items():
        lines.append(f"  {key}: macro_s2v={rep.get('best_macro_s2v'):.5f} "
                     f"worst={rep.get('worst_s2v_at_best'):.5f} "
                     f"epochs={rep.get('epochs_run')}")
    lines.append("")
    lines.append("== four-cell aggregate (mean delta_crps, + = worse than host) ==")
    for key, cs in cells.items():
        lines.append(f"  [{key}]")
        for c in cs:
            lines.append(f"    {c['label']:26s} n={c['n_domains']:2d} "
                         f"mean={c['mean_delta_crps']} worst={c['worst_delta_crps']}")
    lines.append("")
    lines.append("== STOP/CONTINUE (sprint §14, descriptive only) ==")
    for key, cs in cells.items():
        main_ok = any(c["label"] == "Seen_market/Seen_host" and c["n_domains"] > 0
                      for c in cs)
        dk1 = [c for c in cs if c["label"].startswith("Unseen_market") and c["n_domains"] > 0]
        lo = [c for c in cs if "Unseen_host" in c["label"] and c["n_domains"] > 0]
        dk1_worst = max((c["worst_delta_crps"] for c in dk1 if c["worst_delta_crps"] is not None), default=None)
        lo_worst = max((c["worst_delta_crps"] for c in lo if c["worst_delta_crps"] is not None), default=None)
        verdict = "HEALTHY"
        if dk1_worst is not None and dk1_worst > 0.10:
            verdict = "DK1_WORSE_THAN_HOST"
        if lo_worst is not None and lo_worst > 0.10:
            verdict = verdict + "+LOHO_WORSE" if verdict != "HEALTHY" else "LOHO_WORSE_THAN_HOST"
        lines.append(f"  {key}: {verdict} (dk1_worst={dk1_worst}, loho_worst={lo_worst})")
    return lines


if __name__ == "__main__":
    main()
