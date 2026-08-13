"""UniversalCoreTrainer — multi-domain IAH-CRPS training (P0-4) + macro
domain validation (P0-3).

Math/protocol: first-round protocol §9-§12; master plan §1.3-§1.4.

Domain unit:
    g = (market_series, target, host_backbone)
Each domain owns:
    - frozen S1R rank reference (implicit, applied upstream by caller)
    - frozen S1R Data Signature descriptor (domain_det)
    - S2T training batches (for gradient)
    - S2V validation batches (for checkpoint selection)

Training loss (only objective):
    L_universal = (1/|G|) Σ_g E_{d~g}[ L_IAH-CRPS(d) ]
implemented by uniform domain sampling + domain-homogeneous minibatches.

Checkpoint selection (P0-3): a snapshot is kept when macro-domain S2V CRPS
improves; the pooled/micro loss is never used for selection. L_worst = max_g
L_g^val is recorded to detect a universal model that improves the mean by
sacrificing one market.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from iah_crps_loss import iah_crps_loss


@dataclass
class DomainBatch:
    """One (market, host) domain's S2 data + frozen signature context.

    s2t_batches: list of (host [B,H,1], ctx [B,H,D], target [B,H,1], vm [B,H])
    s2v_batches: list of (host [B,H,1], ctx [B,H,D], target [B,H,1], vm [B,H])
    domain_det: [d_det] numpy frozen S1R descriptor, or None for no-signature.
    name: audit label like "LAGO_DE:Linear".
    """
    name: str
    s2t_batches: list = field(default_factory=list)
    s2v_batches: list = field(default_factory=list)
    domain_det: Optional[np.ndarray] = None

    def det_tensor(self, n: int) -> Optional[torch.Tensor]:
        """Broadcast this domain's descriptor to n rows (or None if no-sig)."""
        if self.domain_det is None:
            return None
        det = torch.tensor(np.asarray(self.domain_det, dtype=np.float32),
                           dtype=torch.float32)
        return det.unsqueeze(0).expand(n, -1)


def _eval_batch_losses(candidate_head: nn.Module,
                       batches: list,
                       domain_det: Optional[torch.Tensor]) -> tuple[float, int]:
    """Mean IAH-CRPS over a list of batches (eval mode, no grad)."""
    if not batches:
        return float("nan"), 0
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            if len(batch) >= 5:
                host, ctx, target, vm, det = batch[:5]
            else:
                host, ctx, target, vm = batch
                det = domain_det
            # P0-D: det must be the real batch size. [1, d_det] would crash the
            # DataSignature cat for B > 1 (shape [1,d] vs learned [B,d_sig]).
            if det is not None and det.shape[0] != host.shape[0]:
                det = det.expand(host.shape[0], -1)
            out = candidate_head(host, ctx, valid_mask=vm, domain_det=det)
            loss = iah_crps_loss(out, target)
            total += float(loss)
            n += 1
    return total / max(n, 1), n


def _eval_host_baseline(batches: list) -> float:
    """Protocol §12 host baseline L^host_g = E|zY - z0| over S2V batches.

    The deterministic frozen host is the identity corrector in hyperbolic
    geometry: predictive measure = delta_{z0}. Its IAH-CRPS reduces to
    E|zY - z0| (w⁰=1, w⁻=w⁺=0, m⁻=m⁺=0).
    """
    if not batches:
        return float("nan")
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            host, _, target, vm = batch[:4]
            s = host.squeeze(-1).abs().mean(dim=1, keepdim=True).clamp(min=1e-12)
            z0 = torch.asinh(host.squeeze(-1).to(torch.float64)
                             / s.to(torch.float64)).to(torch.float32)
            zY = torch.asinh(target.squeeze(-1).to(torch.float64)
                             / s.to(torch.float64)).to(torch.float32)
            diff = (zY - z0).abs()
            vh = vm.squeeze(-1) if vm.dim() == 3 else vm
            vh = (vh > 0.5) & torch.isfinite(target.squeeze(-1))
            cnt = vh.sum(dim=1).clamp(min=1)
            total += float((diff * vh.float()).sum(dim=1).div(cnt).mean())
            n += 1
    return total / max(n, 1)


def _collect_health(candidate_head: nn.Module, domains: list) -> dict:
    """Protocol §13 training-health diagnostics over S2V batches.

    Mass: mean w⁻/w⁰/w⁺ + entropy. Shift: fraction m>tiny, median/p95.
    Signature: mean |Δγ|, |β|. Recorded at every validation checkpoint.
    """
    w_m, w_z, w_p, m_m, m_p, dg_n, b_n = [], [], [], [], [], [], []
    with torch.no_grad():
        for d in domains:
            for batch in d.s2v_batches:
                host, ctx, target, vm = batch[:4]
                # P0-D: 5-tuple carries its own det; 4-tuple uses the domain's
                # descriptor expanded to the real batch size (never [1, d_det]).
                det = batch[4] if len(batch) >= 5 else d.det_tensor(host.shape[0])
                if det is not None and det.shape[0] != host.shape[0]:
                    det = det.expand(host.shape[0], -1)
                out = candidate_head(host, ctx, valid_mask=vm, domain_det=det)
                vh = (vm > 0.5) if vm.dim() <= 2 else (vm.squeeze(-1) > 0.5)
                cnt = float(vh.float().sum().clamp(min=1))
                w_m.append(float((out["w_minus"] * vh.float()).sum() / cnt))
                w_z.append(float((out["w_zero"] * vh.float()).sum() / cnt))
                w_p.append(float((out["w_plus"] * vh.float()).sum() / cnt))
                m_m.append(out["m_minus"][vh].detach().cpu().numpy())
                m_p.append(out["m_plus"][vh].detach().cpu().numpy())
                # signature FiLM norms (P1-1: identity-init => start near 0)
                core_input = torch.cat(
                    [out["z0"].unsqueeze(-1), ctx], dim=-1)
                h = candidate_head.core_encoder.proj(core_input)
                dg, beta = candidate_head.core_encoder.signature(h, det)
                dg_n.append(float(dg.abs().mean()))
                b_n.append(float(beta.abs().mean()))
    if not w_m:
        return {}
    wm, wz, wp = float(np.mean(w_m)), float(np.mean(w_z)), float(np.mean(w_p))
    p = np.clip([wm, wz, wp], 1e-12, None)
    entropy = float(-(p * np.log(p)).sum())
    m_minus = np.concatenate(m_m) if m_m else np.zeros(1)
    m_plus = np.concatenate(m_p) if m_p else np.zeros(1)
    tiny = 1e-4
    return {
        "mean_w_minus": wm, "mean_w_zero": wz, "mean_w_plus": wp,
        "mass_entropy": entropy,
        "frac_m_minus_alive": float(np.mean(m_minus > tiny)),
        "frac_m_plus_alive": float(np.mean(m_plus > tiny)),
        "med_m_minus": float(np.median(m_minus)),
        "p95_m_minus": float(np.percentile(m_minus, 95)),
        "med_m_plus": float(np.median(m_plus)),
        "p95_m_plus": float(np.percentile(m_plus, 95)),
        "mean_abs_delta_gamma": float(np.mean(dg_n)),
        "mean_abs_beta": float(np.mean(b_n)),
    }


class UniversalCoreTrainer:
    """Trains a single shared IAHCandidateHead across many domains."""

    def __init__(self, candidate_head: nn.Module, seed: int = 0):
        self.head = candidate_head
        self.seed = seed

    def train(self, domains: list[DomainBatch], epochs: int = 8,
              lr: float = 3e-4, weight_decay: float = 1e-4,
              clip: float = 1.0, patience: int = 3) -> dict:
        """True equal-domain sampling + macro S2V checkpoint selection.

        domains: list of DomainBatch (one per (market, host)).
        Returns training report: best macro S2V CRPS, L_worst, per-domain val,
        and per-epoch updates_per_domain (P0-C audit).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        domains = [d for d in domains if d.s2t_batches]
        assert domains, "UniversalCoreTrainer: no domain has S2T batches"
        n_domains = len(domains)
        n_batches = [len(d.s2t_batches) for d in domains]
        # P0-C: L_universal = (1/|G|) Σ_g E_{d~g}[L_g] requires every domain to
        # receive the SAME number of optimizer updates per epoch. K = median_g
        # N_g; longer domains sample without replacement, shorter with, so a
        # long market never gets more gradient weight for being longer.
        K = int(np.median(n_batches))
        print(f"[UCT] {n_domains} domains, epochs={epochs}, lr={lr}, "
              f"K={K}/domain/epoch")

        opt = torch.optim.AdamW(self.head.parameters(), lr=lr,
                                weight_decay=weight_decay)
        best_state, best_macro, worst_at_best = None, float("inf"), float("inf")
        pat = 0
        history = []

        def _eval_all() -> tuple[float, float, dict, dict, dict, dict]:
            """Macro S2V CRPS + L_worst + per-domain CRPS + baselines + health."""
            per_g, worst, host_g, delta_g = {}, float("-inf"), {}, {}
            for d in domains:
                loss, n = _eval_batch_losses(self.head, d.s2v_batches,
                                             d.det_tensor(1))
                per_g[d.name] = float(loss)
                host_g[d.name] = _eval_host_baseline(d.s2v_batches)
                if np.isfinite(loss):
                    delta_g[d.name] = float(loss) - host_g[d.name]
                    if loss > worst:
                        worst = float(loss)
            macro = float(np.mean([v for v in per_g.values()
                                   if np.isfinite(v)])) if per_g else float("nan")
            health = _collect_health(self.head, domains)
            return macro, worst, per_g, host_g, delta_g, health

        for ep in range(epochs):
            epoch_losses = []
            grad_norms, nan_batches, scale_invalid_days = [], 0, 0
            updates = [0] * n_domains
            # shuffled schedule: every domain appears exactly K times
            schedule = np.repeat(np.arange(n_domains), K)
            rng.shuffle(schedule)
            pools = [list(range(len(domains[g].s2t_batches)))
                     for g in range(n_domains)]
            for g in schedule:
                d = domains[g]
                pool = pools[g]
                if not pool:  # exhausted -> sample with replacement
                    pools[g] = list(range(len(d.s2t_batches)))
                    pool = pools[g]
                bi = pool.pop()
                batch = d.s2t_batches[bi]
                host, ctx, target, vm = batch[:4]
                # P0-D: det is always the real batch size, never [1, d_det].
                det = batch[4] if len(batch) >= 5 else d.det_tensor(host.shape[0])
                if det is not None and det.shape[0] != host.shape[0]:
                    det = det.expand(host.shape[0], -1)
                opt.zero_grad()
                out = self.head(host, ctx, valid_mask=vm, domain_det=det)
                loss = iah_crps_loss(out, target)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.head.parameters(),
                                                     clip).item()
                grad_norms.append(float(grad_norm))
                if not np.isfinite(float(loss.detach())) or not np.isfinite(grad_norm):
                    nan_batches += 1
                scale_invalid_days += int((out["scale_valid"] < 0.5).sum())
                opt.step()
                updates[g] += 1
                epoch_losses.append(float(loss.detach()))
            updates_per_domain = {domains[g].name: updates[g]
                                  for g in range(n_domains)}
            assert all(v == K for v in updates), \
                f"P0-C imbalance: {updates_per_domain} (expect {K}/domain)"

            macro, worst, per_g, host_g, delta_g, health = _eval_all()
            history.append({"epoch": ep, "macro_s2v": macro,
                            "worst_s2v": worst, "per_domain": per_g,
                            "host_baseline": host_g, "delta": delta_g,
                            "health": health,
                            "updates_per_domain": updates_per_domain,
                            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                            "grad_health": {
                                "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                                "nan_inf_batches": nan_batches,
                                "scale_unidentified_days": scale_invalid_days,
                            }})
            print(f"  ep{ep}: macro_s2v={macro:.4f} worst={worst:.4f} "
                  f"train={history[-1]['train_loss']:.4f} | "
                  f"{' '.join(f'{k}={v:.3f}' for k, v in per_g.items())}")

            if np.isfinite(macro) and macro < best_macro - 1e-5:
                best_macro, best_state = macro, {k: v.clone() for k, v in
                                                 self.head.state_dict().items()}
                worst_at_best = worst
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    print(f"  early stop at ep{ep}; best_macro={best_macro:.4f}")
                    break

        if best_state is not None:
            self.head.load_state_dict(best_state)

        return {
            "best_macro_s2v": float(best_macro),
            "worst_s2v_at_best": float(worst_at_best),
            "n_domains": n_domains,
            "epochs_run": len(history),
            "history": history,
        }
