"""IAH-CRPS Loss — single training objective (Eq 10-11).

Math: hch_v2_iah_crps_final_math_core_v0.3 §2.3

For a 3-atom conditional predictive measure:
    F = w⁻ δ_{z⁻} + w⁰ δ_{z⁰} + w⁺ δ_{z⁺}

The CRPS is (Eq 10):
    L = Σ w^a |zY - z^a| − w⁻(1−w⁻)m⁻ − w⁺(1−w⁺)m⁺

The second line is the exact simplification of the spread term
    −0.5 Σ_{a,b} w^a w^b |z^a − z^b|
for the 3-atom structure. It is NOT an extra regularization term.
"""
from __future__ import annotations

import torch


def iah_crps_loss(candidate: dict, target_raw: torch.Tensor) -> torch.Tensor:
    """Compute IAH-CRPS loss over valid hours, averaged per day.

    Args:
        candidate: dict from IAHCandidateHead.forward(), containing:
            s, z0, z_minus, z_plus, w_minus, w_zero, w_plus,
            m_minus, m_plus, valid_mask
        target_raw: [B, H, 1] raw-price ground truth

    Returns:
        scalar loss = mean over days of mean over valid hours of CRPS
    """
    s = candidate["s"]                       # [B]
    s_safe = s.clamp(min=1e-12)              # avoid div by zero

    # Convert target to hyperbolic coordinates (Eq 3)
    target_64 = target_raw.to(torch.float64).squeeze(-1)
    s_safe_64 = s_safe.to(torch.float64)
    zY = torch.asinh(target_64 / s_safe_64.unsqueeze(-1)).to(torch.float32)  # [B, H]

    z0 = candidate["z0"]         # [B, H]
    z_minus = candidate["z_minus"]  # [B, H]
    z_plus = candidate["z_plus"]    # [B, H]
    w_minus = candidate["w_minus"]  # [B, H]
    w_zero = candidate["w_zero"]    # [B, H]
    w_plus = candidate["w_plus"]    # [B, H]
    m_minus = candidate["m_minus"]  # [B, H]
    m_plus = candidate["m_plus"]    # [B, H]

    # Term 1: weighted absolute errors (Eq 10, first line)
    term1 = (w_minus * torch.abs(zY - z_minus)
             + w_zero * torch.abs(zY - z0)
             + w_plus * torch.abs(zY - z_plus))

    # Term 2: spread simplification (Eq 10, second line)
    # = −w⁻(1−w⁻)m⁻ − w⁺(1−w⁺)m⁺
    term2 = (-w_minus * (1.0 - w_minus) * m_minus
             - w_plus * (1.0 - w_plus) * m_plus)

    loss_per_hour = term1 + term2  # [B, H]

    # Average over valid hours per day, then over days
    valid = candidate["valid_mask"].to(loss_per_hour.device)  # [B, H]
    loss_per_day = (loss_per_hour * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B]
    return loss_per_day.mean()


def crps_manual(zY_val: float, z0_val: float,
                w_minus_val: float, w_plus_val: float,
                m_minus_val: float, m_plus_val: float) -> float:
    """Manual CRPS computation for one hour (testing oracle).

    Eq (10): L = Σ w^a |zY − z^a| − w⁻(1−w⁻)m⁻ − w⁺(1−w⁺)m⁺
    where z⁻ = z0 − m⁻, z⁺ = z0 + m⁺, w₀ = 1 − w⁻ − w⁺.
    """
    z_minus = z0_val - m_minus_val
    z_plus = z0_val + m_plus_val
    w_zero_val = 1.0 - w_minus_val - w_plus_val

    term1 = (w_minus_val * abs(zY_val - z_minus)
             + w_zero_val * abs(zY_val - z0_val)
             + w_plus_val * abs(zY_val - z_plus))

    term2 = (-w_minus_val * (1.0 - w_minus_val) * m_minus_val
             - w_plus_val * (1.0 - w_plus_val) * m_plus_val)

    return float(term1 + term2)
