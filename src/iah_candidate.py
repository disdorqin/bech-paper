"""IAH Candidate Head — host-anchored bi-tail conditional atom candidates.

Math: hch_v2_iah_crps_final_math_core_v0.3 §2.1-§2.2

Coordinate transform (Eq 2-4):
    s_d = mean(|host_raw_d|) per day
    z0 = asinh(host_raw / s_d)       # scale-equivariant hyperbolic coordinate

Three-atom conditional measure (Eq 6-8):
    w = softmax([l_minus, 0, l_plus])  # conditional mass (center logit=0)
    m = ReLU(r)                         # hyperbolic displacement dose
    z_minus = z0 - m_minus, z_plus = z0 + m_plus
    x_a = s * sinh(z_a)                # inverse transform to raw price
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IAHCandidateHead(nn.Module):
    """Conditional 3-atom candidate head in hyperbolic coordinates.

    Input: host_raw [B, H, 1], context [B, H, D] (includes u, time_feat, exog, lag)
    Output: dict with z0, weights, shifts, inverse-transformed actions, valid mask.
    """

    def __init__(self, d_context: int, d_hidden: int = 64):
        super().__init__()
        self.context_net = nn.Sequential(
            nn.Linear(d_context, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.mass_head = nn.Linear(d_hidden, 2)   # l_minus, l_plus
        self.shift_head = nn.Linear(d_hidden, 2)  # r_minus, r_plus

    def _compute_scale(self, host_raw: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """s_d = mean(|host_raw|) over VALID AND FINITE hours per day (Eq 2).

        Returns s=0 for days with no valid+finite host value (SCALE_UNIDENTIFIED).
        """
        host_abs = host_raw.squeeze(-1).abs()  # [B, H]
        finite_mask = torch.isfinite(host_abs)

        if valid_mask is None:
            eff_mask = finite_mask
        else:
            vm = valid_mask.to(host_raw.device)
            if vm.dim() == 3:
                vm = vm.squeeze(-1)  # [B, H]
            eff_mask = finite_mask & (vm > 0.5)

        valid_count = eff_mask.sum(dim=1)  # [B]
        finite_sum = torch.where(eff_mask, host_abs,
                                 torch.zeros_like(host_abs)).sum(dim=1)  # [B]
        # If no valid+finite hour, s=0 (not 0/0). Avoid div-by-zero via masked divide.
        s = torch.where(valid_count > 0, finite_sum / valid_count.clamp(min=1),
                        torch.zeros_like(finite_sum))
        return s

    def forward(self, host_raw: torch.Tensor, context: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None):
        B, H, _ = host_raw.shape
        s = self._compute_scale(host_raw, valid_mask)  # [B]

        # SCALE_UNIDENTIFIED iff no valid+finite hour or s == 0
        scale_valid = (s > 0).float()  # [B]

        # Implementation masking (NOT a modified scale definition): a dummy safe
        # denominator only for the asinh/sinh computations. Its value can never
        # change the mathematical output because scale-invalid days have their
        # candidate output overwritten to Identity below.
        s_safe = s.clamp(min=1e-12)

        # Hyperbolic coordinates (float64 for asinh/sinh stability)
        host_raw_64 = host_raw.to(torch.float64)
        s_safe_64 = s_safe.to(torch.float64)
        z0_64 = torch.asinh(host_raw_64.squeeze(-1) / s_safe_64.unsqueeze(-1))  # [B, H]
        z0 = z0_64.to(torch.float32)

        # Context encoding → mass logits + raw shifts
        feat = self.context_net(context)  # [B, H, d_hidden]

        l_raw = self.mass_head(feat)  # [B, H, 2]
        l_minus = l_raw[..., 0]       # [B, H]
        l_plus = l_raw[..., 1]        # [B, H]

        # Softmax with center logit fixed at 0 (Eq 6)
        logits = torch.stack([l_minus, torch.zeros_like(l_minus), l_plus], dim=-1)  # [B, H, 3]
        w = F.softmax(logits, dim=-1)
        w_minus, w_zero, w_plus = w[..., 0], w[..., 1], w[..., 2]

        # Displacement doses (Eq 7)
        r_raw = self.shift_head(feat)  # [B, H, 2]
        m_minus = F.relu(r_raw[..., 0])  # [B, H]
        m_plus = F.relu(r_raw[..., 1])   # [B, H]

        # Candidate positions in z-coordinates (Eq 8)
        z_minus = z0 - m_minus
        z_plus = z0 + m_plus

        # Inverse transform to raw prices [B, H, 1]
        s_safe_32 = s_safe.to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        x_identity = s_safe_32 * torch.sinh(z0_64).unsqueeze(-1).to(torch.float32)
        x_down = s_safe_32 * torch.sinh(z_minus.to(torch.float64)).unsqueeze(-1).to(torch.float32)
        x_up = s_safe_32 * torch.sinh(z_plus.to(torch.float64)).unsqueeze(-1).to(torch.float32)

        # Scale-invalid days: broadcast mask and use torch.where (preserves autograd)
        day_mask = scale_valid.unsqueeze(-1)  # [B, 1] broadcasts with [B, H]
        w_minus = torch.where(day_mask > 0.5, w_minus, torch.zeros_like(w_minus))
        w_zero = torch.where(day_mask > 0.5, w_zero, torch.ones_like(w_zero))
        w_plus = torch.where(day_mask > 0.5, w_plus, torch.zeros_like(w_plus))
        m_minus = torch.where(day_mask > 0.5, m_minus, torch.zeros_like(m_minus))
        m_plus = torch.where(day_mask > 0.5, m_plus, torch.zeros_like(m_plus))
        z_minus = torch.where(day_mask > 0.5, z_minus, z0)
        z_plus = torch.where(day_mask > 0.5, z_plus, z0)

        day_mask_3d = day_mask.unsqueeze(-1)  # [B, 1, 1]
        x_identity = torch.where(day_mask_3d > 0.5, x_identity, host_raw)
        x_down = torch.where(day_mask_3d > 0.5, x_down, host_raw)
        x_up = torch.where(day_mask_3d > 0.5, x_up, host_raw)

        return {
            "s": s,
            "scale_valid": scale_valid,
            "z0": z0,
            "z_minus": z_minus,
            "z_plus": z_plus,
            "w_minus": w_minus,
            "w_zero": w_zero,
            "w_plus": w_plus,
            "m_minus": m_minus,
            "m_plus": m_plus,
            "x_identity": x_identity,
            "x_down": x_down,
            "x_up": x_up,
            "valid_mask": valid_mask if valid_mask is not None
                          else torch.ones(B, H, device=host_raw.device),
        }
