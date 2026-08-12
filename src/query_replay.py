"""Query-dose replay (Eq 17-22).

Math: hch_v2_iah_crps_final_math_core_v0.3 §3.2

Query dose: π_q ∈ {−m⁻_q, 0, +m⁺_q} per hour.
Replay on history day j: z_replay_j = z0_j + π_q  (Eq 18)
Hourly gain: g = |r_z| − |r_z − π_q|  (Eq 19), where r_z = zY_j − z0_j
Daily action value: A_q→j = mean_valid_hours(g)  (Eq 21)
Estimated value: Â_q = mean_{j∈N_k} A_q→j  (Eq 22)

Key constraint: replay uses QUERY dose π_q on HISTORY host z0_j.
Not history's own dose π_j.
"""
from __future__ import annotations

import numpy as np


def replay_query_dose(z0_hist: np.ndarray, target_zY_hist: np.ndarray,
                      pi_query: np.ndarray, valid_hist: np.ndarray) -> dict:
    """Replay a query dose vector on one history day.

    Args:
        z0_hist: [H] history day's z0 (Identity) values
        target_zY_hist: [H] history day's zY (target) values
        pi_query: [H] query dose per hour (−m⁻, 0, or +m⁺)
        valid_hist: [H] boolean valid-hour mask

    Returns:
        {"g": [H] hourly gains, "A": float daily action value, "n_valid": int}
    """
    r_z = target_zY_hist.astype(np.float64) - z0_hist.astype(np.float64)
    pi_q = np.asarray(pi_query, dtype=np.float64)
    g = np.abs(r_z) - np.abs(r_z - pi_q)
    g = g * valid_hist.astype(np.float64)  # zero out invalid hours
    n_valid = valid_hist.sum()
    A = float(g.sum() / max(n_valid, 1))  # Eq 21
    return {"g": g, "A": A, "n_valid": int(n_valid)}


def estimate_action_value(memory, neighbor_indices: list[int],
                          pi_query: np.ndarray) -> dict:
    """Estimate action value of query dose on k nearest neighbors (Eq 22).

    Args:
        memory: CAGMAtomMemory instance
        neighbor_indices: indices of retrieved neighbors
        pi_query: [H] query dose per hour

    Returns:
        {"A_hat": float, "per_neighbor_A": [float], "per_neighbor_g": [ndarray]}
    """
    per_neighbor_A = []
    per_neighbor_g = []
    for idx in neighbor_indices:
        result = replay_query_dose(
            memory.z0[idx], memory.target_zY[idx],
            pi_query, memory.valid_mask[idx],
        )
        per_neighbor_A.append(result["A"])
        per_neighbor_g.append(result["g"])

    if per_neighbor_A:
        A_hat = float(np.mean(per_neighbor_A))
    else:
        A_hat = 0.0

    return {"A_hat": A_hat, "per_neighbor_A": per_neighbor_A,
            "per_neighbor_g": per_neighbor_g}


def verify_gain_bound(pi_query: np.ndarray, g: np.ndarray,
                      valid: np.ndarray) -> bool:
    """Verify |g_h| ≤ |π_h| for all valid hours (Eq 20)."""
    g_valid = g[valid.astype(bool)]
    pi_valid = pi_query[valid.astype(bool)]
    return bool((np.abs(g_valid) <= np.abs(pi_valid) + 1e-10).all())


def build_directional_gains(memory, neighbor_indices: list[int],
                            m_minus_q: np.ndarray, m_plus_q: np.ndarray) -> dict:
    """Estimate per-hour directional gains g_hat (Eq 23).

    Replays the FULL Down dose (−m⁻_q) and FULL Up dose (+m⁺_q) on each
    neighbor, then averages per hour.

    Args:
        memory: CAGMAtomMemory
        neighbor_indices: retrieved neighbor indices
        m_minus_q: [H] query Down dose
        m_plus_q:  [H] query Up dose

    Returns:
        {"g_hat_down": [H], "g_hat_up": [H]}
    """
    H = len(m_minus_q)
    g_down_accum = np.zeros(H, dtype=np.float64)
    g_up_accum = np.zeros(H, dtype=np.float64)
    n = 0

    for idx in neighbor_indices:
        z0_j = memory.z0[idx].astype(np.float64)
        zY_j = memory.target_zY[idx].astype(np.float64)
        valid_j = memory.valid_mask[idx].astype(bool)

        # Full Down dose
        r_down = replay_query_dose(z0_j, zY_j, -m_minus_q.astype(np.float64), valid_j)
        # Full Up dose
        r_up = replay_query_dose(z0_j, zY_j, m_plus_q.astype(np.float64), valid_j)

        g_down_accum += r_down["g"]
        g_up_accum += r_up["g"]
        n += 1

    if n == 0:
        return {"g_hat_down": np.zeros(H), "g_hat_up": np.zeros(H)}

    return {"g_hat_down": g_down_accum / n, "g_hat_up": g_up_accum / n}


def form_final_pi(m_minus_q: np.ndarray, m_plus_q: np.ndarray,
                  I_down, I_up) -> np.ndarray:
    """Form final sparse action vector pi_q (Eq 26).

    pi[h] = -m_minus[h] if h in I_down, +m_plus[h] if h in I_up, 0 otherwise.
    """
    H = len(m_minus_q)
    pi = np.zeros(H, dtype=np.float64)
    if I_down is not None:
        l, r = I_down
        for h in range(l, r + 1):
            pi[h] = -m_minus_q[h]
    if I_up is not None:
        l, r = I_up
        for h in range(l, r + 1):
            pi[h] = m_plus_q[h]
    return pi


def full_replay_chain(memory, neighbor_indices, m_minus_q, m_plus_q,
                      proposal_fn) -> dict:
    """Complete replay chain (P0-5):

    directional replay -> proposal -> final pi_q -> final replay -> A_hat_q

    Args:
        memory, neighbor_indices: as above
        m_minus_q, m_plus_q: [H] query doses
        proposal_fn: callable(g_hat_down, g_hat_up) -> double_event_proposal dict

    Returns:
        {"g_hat_down", "g_hat_up", "proposal", "pi_q",
         "A_hat" (from final replay), "per_neighbor_A"}
    """
    gains = build_directional_gains(memory, neighbor_indices,
                                    m_minus_q, m_plus_q)
    proposal = proposal_fn(gains["g_hat_down"], gains["g_hat_up"])
    pi_q = form_final_pi(m_minus_q, m_plus_q,
                         proposal["I_down"], proposal["I_up"])

    # Final replay with the final sparse pi_q (NOT pre-proposal directional gains)
    value = estimate_action_value(memory, neighbor_indices, pi_q)

    return {
        "g_hat_down": gains["g_hat_down"],
        "g_hat_up": gains["g_hat_up"],
        "proposal": proposal,
        "pi_q": pi_q,
        "A_hat": value["A_hat"],
        "per_neighbor_A": value["per_neighbor_A"],
    }
