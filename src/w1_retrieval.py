"""W1 distance + residual atom measure (Eq 15-16).

Math: hch_v2_iah_crps_final_math_core_v0.3 §3.1

Residual atom measure (Eq 15):
    R̂_{d,h} = w⁻ δ_{-m⁻} + w⁰ δ₀ + w⁺ δ_{m⁺}

W1 distance between two such measures (Eq 16):
    D(q,j) = mean_h W1(R̂_{q,h}, R̂_{j,h})

For unequal-mass 3-atom measures, W1 is computed exactly by merging
the cumulative mass breakpoints of both CDFs.
"""
from __future__ import annotations

import numpy as np
import torch


def w1_3atom(w_minus_a: float, w_zero_a: float, w_plus_a: float,
             m_minus_a: float, m_plus_a: float,
             w_minus_b: float, w_zero_b: float, w_plus_b: float,
             m_minus_b: float, m_plus_b: float) -> float:
    """Exact W1 distance between two 3-atom residual measures.

    W1 = integral |F_a(x) - F_b(x)| dx, computed by breaking at all
    atom positions and integrating the piecewise-constant |CDF diff|.
    """
    # Collect (position, delta_mass_a, delta_mass_b) at each atom
    pos_a = [(-float(m_minus_a), 0), (0.0, 1), (float(m_plus_a), 2)]
    w_a = [float(w_minus_a), float(w_zero_a), float(w_plus_a)]
    pos_b = [(-float(m_minus_b), 0), (0.0, 1), (float(m_plus_b), 2)]
    w_b = [float(w_minus_b), float(w_zero_b), float(w_plus_b)]

    events = []  # (position, dmass_a, dmass_b)
    for (p, _), w in zip(pos_a, w_a):
        if w > 1e-15:
            events.append((p, w, 0.0))
    for (p, _), w in zip(pos_b, w_b):
        if w > 1e-15:
            events.append((p, 0.0, w))

    # Merge events at the same position
    events.sort(key=lambda x: x[0])
    merged = []
    for pos, da, db in events:
        if merged and abs(merged[-1][0] - pos) < 1e-14:
            prev = merged.pop()
            merged.append((pos, prev[1] + da, prev[2] + db))
        else:
            merged.append((pos, da, db))

    if not merged:
        return 0.0

    # Integrate |CDF_a - CDF_b| between consecutive breakpoints
    cum_a, cum_b = 0.0, 0.0
    prev_pos = merged[0][0]
    w1 = 0.0

    for pos, da, db in merged:
        dx = pos - prev_pos
        if dx > 0 and (cum_a > 0 or cum_b > 0 or abs(cum_a - cum_b) > 0):
            w1 += abs(cum_a - cum_b) * dx
        cum_a += da
        cum_b += db
        prev_pos = pos

    return float(w1)


def residual_atom_measure(w_minus: torch.Tensor, w_zero: torch.Tensor,
                          w_plus: torch.Tensor, m_minus: torch.Tensor,
                          m_plus: torch.Tensor) -> dict:
    """Extract residual atom measures from candidate output.
    Returns numpy-friendly dict for storage/retrieval.
    """
    return {
        "w_minus": w_minus.detach().cpu().numpy(),
        "w_zero": w_zero.detach().cpu().numpy(),
        "w_plus": w_plus.detach().cpu().numpy(),
        "m_minus": m_minus.detach().cpu().numpy(),
        "m_plus": m_plus.detach().cpu().numpy(),
    }


def day_w1_distance(w1_a: np.ndarray, m1_a: np.ndarray,
                    w1_b: np.ndarray, m1_b: np.ndarray,
                    valid_a: np.ndarray, valid_b: np.ndarray) -> float:
    """Day-level W1 distance (Eq 16): mean over valid shared hours.

    Args:
        w1_a/w1_b: [H, 3] arrays of weights (w⁻, w⁰, w⁺)
        m1_a/m1_b: [H, 2] arrays of doses (m⁻, m⁺)
        valid_a/valid_b: [H] boolean valid-hour masks
    """
    shared = valid_a.astype(bool) & valid_b.astype(bool)
    if shared.sum() < 1:
        return float('inf')

    total = 0.0
    count = 0
    for h in range(len(shared)):
        if not shared[h]:
            continue
        d = w1_3atom(
            float(w1_a[h, 0]), float(w1_a[h, 1]), float(w1_a[h, 2]),
            float(m1_a[h, 0]), float(m1_a[h, 1]),
            float(w1_b[h, 0]), float(w1_b[h, 1]), float(w1_b[h, 2]),
            float(m1_b[h, 0]), float(m1_b[h, 1]),
        )
        total += d
        count += 1
    return total / count


# ======================== CAGM Atom Memory ===================================
class CAGMAtomMemory:
    """Stores per-day residual atom measures for W1-based retrieval.

    Each entry: date, timestamp, z0 [H], w [H,3], m [H,2], target_zY [H] (for replay only).
    Retrieval uses W1 distance (Eq 16); target_zY is stored separately from the key.
    """

    def __init__(self):
        self.dates: list[str] = []
        self.z0: list[np.ndarray] = []          # [H] per day
        self.w_minus: list[np.ndarray] = []     # [H]
        self.w_zero: list[np.ndarray] = []      # [H]
        self.w_plus: list[np.ndarray] = []      # [H]
        self.m_minus: list[np.ndarray] = []     # [H]
        self.m_plus: list[np.ndarray] = []      # [H]
        self.target_zY: list[np.ndarray] = []   # [H] zY for replay (NOT in key)
        self.valid_mask: list[np.ndarray] = []  # [H] boolean

    def add_day(self, date: str, candidate: dict, target_zY: np.ndarray):
        """Store one day's residual atom measures."""
        self.dates.append(date)
        self.z0.append(candidate["z0"].detach().cpu().numpy().squeeze())
        self.w_minus.append(candidate["w_minus"].detach().cpu().numpy().squeeze())
        self.w_zero.append(candidate["w_zero"].detach().cpu().numpy().squeeze())
        self.w_plus.append(candidate["w_plus"].detach().cpu().numpy().squeeze())
        self.m_minus.append(candidate["m_minus"].detach().cpu().numpy().squeeze())
        self.m_plus.append(candidate["m_plus"].detach().cpu().numpy().squeeze())
        self.target_zY.append(target_zY)
        self.valid_mask.append(
            candidate["valid_mask"].detach().cpu().numpy().squeeze().astype(bool)
        )

    def __len__(self):
        return len(self.dates)

    def build_retrieval_index(self, query_candidate: dict) -> np.ndarray:
        """Compute W1 distances from query to all stored days.

        The retrieval key is the residual atom measure R̂ (Eq 15).
        Target zY is NOT part of the key — per addendum §5.1 constraint.

        Returns: distances [M]
        """
        q_w = np.stack([
            query_candidate["w_minus"].detach().cpu().numpy().squeeze(),
            query_candidate["w_zero"].detach().cpu().numpy().squeeze(),
            query_candidate["w_plus"].detach().cpu().numpy().squeeze(),
        ], axis=-1)  # [H, 3]
        q_m = np.stack([
            query_candidate["m_minus"].detach().cpu().numpy().squeeze(),
            query_candidate["m_plus"].detach().cpu().numpy().squeeze(),
        ], axis=-1)  # [H, 2]
        q_valid = query_candidate["valid_mask"].detach().cpu().numpy().squeeze().astype(bool)

        distances = np.full(len(self), float("inf"))
        for i in range(len(self)):
            d_w = np.stack([self.w_minus[i], self.w_zero[i], self.w_plus[i]], axis=-1)
            d_m = np.stack([self.m_minus[i], self.m_plus[i]], axis=-1)
            distances[i] = day_w1_distance(q_w, q_m, d_w, d_m, q_valid, self.valid_mask[i])
        return distances

    def get_neighbors(self, distances: np.ndarray, k: int) -> list[int]:
        """Return indices of k nearest neighbors (excluding self if distance=0)."""
        order = np.argsort(distances)
        neighbors = []
        for idx in order:
            if distances[idx] < float("inf") and len(neighbors) < k:
                neighbors.append(int(idx))
        return neighbors

