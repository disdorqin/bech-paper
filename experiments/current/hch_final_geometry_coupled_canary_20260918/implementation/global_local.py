"""Day repair core and geometry-derived signed-mass Shape head (freeze §5.2-§5.5).

``DayRepairCore``
    ``d = LayerNorm(W_d [MaskedMean_h(u_h) || z_day])``
    ``b_hat = w_b^T d + c_b``                      (near-Host: exactly zero at init)
    ``B_hat = s_B softplus(w_B^T d + c_B + beta0)`` (near-Host: ``0.01 s_B`` at init)

``SignedMassQueryShapeHead``
    ``m_hat^+/- = B_hat + [ +/- H b_hat ]_+``
    ``s_m       = s_B + H s_b``  (deterministic, TRAIN-frozen; no fourth scale)
    ``q^sigma   = W_q^sigma [ d || log(1 + m_hat^sigma / s_m) ]``
    ``k_h       = W_k u_h``
    ``S_hat^sigma = MaskedSoftmax_h( <q^sigma, k_h> / sqrt(d_q) )``

This is the *only* global-to-local coupling in the method.  It is not an evidence
router: the query selects no source, it conditions where the predicted day-level
signed mass is placed across hours.  There is no value projection, no multi-head
attention and no attention stack.

The GEOM_FLAT control uses **the identical parameter tensors** and hard-zeroes the
signed-mass scalar input (``[d || 0]``), so the A1/A2 contrast is exactly the
presence of the geometry-derived mass scalar.

All arithmetic that defines an identity is forced to fp32: the query, the key, the
logits and both horizon simplexes are computed in fp32 irrespective of autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.contracts import HORIZON, TOKEN_DIM
from core.geometry import masked_mean, masked_softmax
from core.heads import softplus_inverse

from fused_encoder import WIDTH

#: Fraction of ``s_B`` the balanced-mass head starts at (frozen near-Host bias).
B_INIT_FRACTION = 0.01
#: Query dimension ``d_q``.  Not tunable after outcomes.
QUERY_DIM = 32
#: Standard deviation of the small query/key initialization.
QUERY_INIT_STD = 1e-3


class DayRepairCore(nn.Module):
    """Masked day pooling, one affine to width 32, and the Level / B heads."""

    def __init__(
        self,
        day_dim: int,
        s_B: float,
        width: int = WIDTH,
        dropout: float = 0.1,
        token_dim: int = TOKEN_DIM,
    ):
        super().__init__()
        if not s_B > 0.0:
            raise ValueError("s_B must be positive and TRAIN-frozen")
        self.width = int(width)
        self.day_dim = int(day_dim)
        self.affine = nn.Linear(width + day_dim, width)
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        # Level: the final affine starts at exactly zero -> b_hat == 0 at init.
        self.b_affine = nn.Linear(width, 1)
        nn.init.zeros_(self.b_affine.weight)
        nn.init.zeros_(self.b_affine.bias)
        # Balanced mass: zero affine plus one softplus offset -> B_hat ~ 0.01 s_B.
        self.B_affine = nn.Linear(width, 1)
        nn.init.zeros_(self.B_affine.weight)
        nn.init.zeros_(self.B_affine.bias)
        self.register_buffer("s_B", torch.tensor(float(s_B)), persistent=True)
        self.register_buffer("offset", torch.tensor(softplus_inverse(B_INIT_FRACTION)))
        self._token_dim = int(token_dim)

    def forward(
        self,
        u: torch.Tensor,
        hour_valid: torch.Tensor,
        z_day: torch.Tensor,
    ) -> tuple:
        mask = torch.as_tensor(hour_valid).to(torch.bool)
        pooled = masked_mean(u, mask.to(u.dtype), dim=1)
        state = torch.cat([pooled, torch.as_tensor(z_day).float()], dim=-1)
        d = self.drop(self.norm(self.affine(state)))
        b_hat = self.b_affine(d).squeeze(-1)
        B_hat = self.s_B * torch.nn.functional.softplus(self.B_affine(d).squeeze(-1) + self.offset)
        return d, b_hat, B_hat


class SignedMassQueryShapeHead(nn.Module):
    """Two geometry-derived signed queries against one shared hourly key."""

    def __init__(self, width: int = WIDTH, d_q: int = QUERY_DIM, init_std: float = QUERY_INIT_STD):
        super().__init__()
        self.d_q = int(d_q)
        self.query_plus = nn.Linear(width + 1, d_q)
        self.query_minus = nn.Linear(width + 1, d_q)
        self.key = nn.Linear(width, d_q)
        for module in (self.query_plus, self.query_minus, self.key):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def signed_masses(b_hat: torch.Tensor, B_hat: torch.Tensor, horizon: int = HORIZON) -> tuple:
        """``m_hat^+ = B_hat + [H b_hat]_+`` and the mirror image."""
        b = torch.as_tensor(b_hat).float().reshape(-1)
        amp = torch.as_tensor(B_hat).float().reshape(-1)
        return amp + torch.relu(horizon * b), amp + torch.relu(-horizon * b)

    def forward(
        self,
        d: torch.Tensor,
        u: torch.Tensor,
        b_hat: torch.Tensor,
        B_hat: torch.Tensor,
        hour_valid: torch.Tensor,
        s_b: float,
        s_B: float,
        coupled: bool,
        horizon: int = HORIZON,
    ) -> dict:
        m_plus, m_minus = self.signed_masses(b_hat, B_hat, horizon)
        # Deterministic TRAIN-frozen mass scale: same signed-mass unit as m^+/-.
        s_m = float(s_B) + float(horizon) * float(s_b)
        if not s_m > 0.0:
            raise ValueError("s_m must be positive; it is a TRAIN-frozen scale")

        zero = torch.zeros_like(m_plus)
        scalar_plus = torch.log1p(m_plus / s_m) if coupled else zero
        scalar_minus = torch.log1p(m_minus / s_m) if coupled else zero

        d = torch.as_tensor(d).float()
        q_plus = self.query_plus(torch.cat([d, scalar_plus.unsqueeze(-1)], dim=-1)).float()
        q_minus = self.query_minus(torch.cat([d, scalar_minus.unsqueeze(-1)], dim=-1)).float()
        keys = self.key(torch.as_tensor(u).float()).float()

        scale = float(self.d_q) ** 0.5
        logits_plus = (q_plus.unsqueeze(1) * keys).sum(dim=-1) / scale
        logits_minus = (q_minus.unsqueeze(1) * keys).sum(dim=-1) / scale
        mask = torch.as_tensor(hour_valid).to(torch.bool)
        s_plus_hat = masked_softmax(logits_plus, mask, dim=-1)
        s_minus_hat = masked_softmax(logits_minus, mask, dim=-1)
        return {
            "s_plus_hat": s_plus_hat,
            "s_minus_hat": s_minus_hat,
            "m_plus": m_plus,
            "m_minus": m_minus,
            "s_m": s_m,
            "logits_plus": logits_plus,
            "logits_minus": logits_minus,
            "coupled": bool(coupled),
        }
