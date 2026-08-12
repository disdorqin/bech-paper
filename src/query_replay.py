"""Query-dose replay (Eq 17-22).

Math: hch_v2_iah_crps_final_math_core_v0.3 §3.2

Query dose: π_q ∈ {−m⁻_q, 0, +m⁺_q} per hour.
Replay on history day j: z_replay_j = z0_j + π_q  (Eq 18)
Hourly gain: g = |r_z| − |r_z − π_q|  (Eq 19), where r_z = zY_j − z0_j
Daily action value: A_q→j = mean over |H_q ∩ H_j| of g  (Eq 21)
Estimated value: Â_q = mean_{j∈N_k} A_q→j  (Eq 22)

Key constraint: replay uses QUERY dose π_q on HISTORY host z0_j.
Not history's own dose π_j.
"""
from __future__ import annotations

import numpy as np


def replay_query_dose(z0_hist: np.ndarray, target_zY_hist: np.ndarray,
                      pi_query: np.ndarray, valid_hist: np.ndarray,
                      query_valid: np.ndarray | None = None) -> dict:
    """Replay a query dose vector on one history day.

    Args:
        z0_hist: [H] history day's z0 (Identity) values
        target_zY_hist: [H] history day's zY (target) values
        pi_query: [H] query dose per hour (−m⁻, 0, or +m⁺)
        valid_hist: [H] boolean valid-hour mask of the HISTORY day
        query_valid: [H] boolean valid-hour mask of the QUERY day (None => all)

    Returns:
        {"g": [H] hourly gains (unmasked), "A": float daily action value over
         the intersection |H_q ∩ H_j|, "n_valid": int, "valid": [H] bool}
    """
    r_z = target_zY_hist.astype(np.float64) - z0_hist.astype(np.float64)
    pi_q = np.asarray(pi_query, dtype=np.float64)
    g = np.abs(r_z) - np.abs(r_z - pi_q)  # Eq 19, unmasked

    vh = np.asarray(valid_hist, dtype=bool)
    if query_valid is None:
        eff_valid = vh
    else:
        eff_valid = vh & np.asarray(query_valid, dtype=bool)

    n_valid = int(eff_valid.sum())
    A = float(g[eff_valid].sum() / n_valid) if n_valid > 0 else 0.0  # Eq 21
    return {"g": g, "A": A, "n_valid": n_valid, "valid": eff_valid}


def estimate_action_value(memory, neighbor_indices: list[int],
                          pi_query: np.ndarray,
                          query_valid: np.ndarray | None = None) -> dict:
    """Estimate action value of query dose on k nearest neighbors (Eq 22)."""
    per_neighbor_A = []
    per_neighbor_g = []
    for idx in neighbor_indices:
        result = replay_query_dose(
            memory.z0[idx], memory.target_zY[idx],
            pi_query, memory.valid_mask[idx], query_valid,
        )
        per_neighbor_A.append(result["A"])
        per_neighbor_g.append(result["g"])

    A_hat = float(np.mean(per_neighbor_A)) if per_neighbor_A else 0.0
    return {"A_hat": A_hat, "per_neighbor_A": per_neighbor_A,
            "per_neighbor_g": per_neighbor_g}


def verify_gain_bound(pi_query: np.ndarray, g: np.ndarray,
                      valid: np.ndarray) -> bool:
    """Verify |g_h| ≤ |π_h| for all valid hours (Eq 20)."""
    g_valid = g[valid.astype(bool)]
    pi_valid = np.asarray(pi_query)[valid.astype(bool)]
    return bool((np.abs(g_valid) <= np.abs(pi_valid) + 1e-10).all())


def build_directional_gains(memory, neighbor_indices: list[int],
                            m_minus_q: np.ndarray, m_plus_q: np.ndarray,
                            query_valid: np.ndarray | None = None) -> dict:
    """Estimate per-hour directional gains g_hat (Eq 23).

    Invalid hours are EXCLUDED from the mean (MEDIUM-10), not zeroed.
    Each hour's gain is averaged over neighbors where that hour is valid in
    BOTH the history day and the query day.
    """
    H = len(m_minus_q)
    g_down_accum = np.zeros(H, dtype=np.float64)
    g_up_accum = np.zeros(H, dtype=np.float64)
    count = np.zeros(H, dtype=np.float64)

    for idx in neighbor_indices:
        z0_j = memory.z0[idx].astype(np.float64)
        zY_j = memory.target_zY[idx].astype(np.float64)
        valid_j = memory.valid_mask[idx].astype(bool)

        r_down = replay_query_dose(z0_j, zY_j, -m_minus_q.astype(np.float64),
                                   valid_j, query_valid)
        r_up = replay_query_dose(z0_j, zY_j, m_plus_q.astype(np.float64),
                                 valid_j, query_valid)
        vd = r_down["valid"]
        g_down_accum[vd] += r_down["g"][vd]
        g_up_accum[vd] += r_up["g"][vd]
        count[vd] += 1.0

    g_hat_down = np.divide(g_down_accum, count,
                           out=np.zeros_like(g_down_accum), where=count > 0)
    g_hat_up = np.divide(g_up_accum, count,
                         out=np.zeros_like(g_up_accum), where=count > 0)
    return {"g_hat_down": g_hat_down, "g_hat_up": g_hat_up}


def form_final_pi(m_minus_q: np.ndarray, m_plus_q: np.ndarray,
                  I_down, I_up) -> np.ndarray:
    """Form final sparse action vector pi_q (Eq 26)."""
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


def estimate_realized_A(z0: np.ndarray, zY: np.ndarray, pi_q: np.ndarray,
                        valid_mask: np.ndarray | None = None) -> float:
    """Realized whole-day action value on the query day itself (Eq 21).

    r_z = zY - z0, A = mean over valid hours of (|r_z| - |r_z - pi_q|).
    Invalid hours are excluded when valid_mask is provided.
    """
    r_z = np.asarray(zY, dtype=np.float64) - np.asarray(z0, dtype=np.float64)
    g = np.abs(r_z) - np.abs(r_z - np.asarray(pi_q, dtype=np.float64))
    if valid_mask is None:
        return float(np.mean(g))
    vm = np.asarray(valid_mask, dtype=bool)
    n = int(vm.sum())
    return float(g[vm].sum() / n) if n > 0 else 0.0


def full_replay_chain(memory, neighbor_indices, m_minus_q, m_plus_q,
                      proposal_fn, query_valid: np.ndarray | None = None) -> dict:
    """Complete replay chain (P0-5):

    directional replay -> proposal -> final pi_q -> final replay -> A_hat_q
    """
    gains = build_directional_gains(memory, neighbor_indices,
                                    m_minus_q, m_plus_q, query_valid)
    proposal = proposal_fn(gains["g_hat_down"], gains["g_hat_up"])
    pi_q = form_final_pi(m_minus_q, m_plus_q,
                         proposal["I_down"], proposal["I_up"])

    value = estimate_action_value(memory, neighbor_indices, pi_q, query_valid)

    return {
        "g_hat_down": gains["g_hat_down"],
        "g_hat_up": gains["g_hat_up"],
        "proposal": proposal,
        "pi_q": pi_q,
        "A_hat": value["A_hat"],
        "per_neighbor_A": value["per_neighbor_A"],
    }
