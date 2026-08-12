"""Double-Event Proposal — O(H²) exact enumeration (Eq 23-30).

Math: hch_v2_iah_crps_final_math_core_v0.3 §4
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def double_event_proposal(g_hat_down: np.ndarray, g_hat_up: np.ndarray) -> dict:
    """Find optimal non-overlapping Down+Up interval pair (Eq 25-30).

    O(H²) via: prefix Down intervals + precomputed prefix/suffix max Up.
    For H=24 this is exact enumeration, not an approximation.
    """
    H = len(g_hat_down)
    assert len(g_hat_up) == H

    down_pf = np.zeros(H + 1)
    up_pf = np.zeros(H + 1)
    for h in range(H):
        down_pf[h + 1] = down_pf[h] + g_hat_down[h]
        up_pf[h + 1] = up_pf[h] + g_hat_up[h]

    def S(l, r, pf):
        return pf[r + 1] - pf[l]

    # U_L[t]: max Up interval S_+(l,r) with r ≤ t (Eq 27)
    U_L_val = np.zeros(H)
    U_L_r = np.full(H, -1, dtype=int)  # best Up end position at each t
    for t in range(H):
        best, br = 0.0, -1
        for r in range(t + 1):
            for l in range(r + 1):
                v = up_pf[r + 1] - up_pf[l]
                if v > best:
                    best, br = v, r
        U_L_val[t] = best
        U_L_r[t] = br

    # U_R[t]: max Up interval S_+(l,r) with l ≥ t (Eq 28)
    U_R_val = np.zeros(H)
    U_R_l = np.full(H, -1, dtype=int)
    for t in range(H):
        best, bl = 0.0, -1
        for l in range(t, H):
            for r in range(l, H):
                v = up_pf[r + 1] - up_pf[l]
                if v > best:
                    best, bl = v, l
        U_R_val[t] = best
        U_R_l[t] = bl

    # Reconstruct actual Up interval for a given (t, side)
    def find_up_ending_at(r_end):
        if r_end < 0:
            return None
        best, bl, br = 0.0, -1, -1
        for l in range(r_end + 1):
            v = up_pf[r_end + 1] - up_pf[l]
            if v > best:
                best, bl, br = v, l, r_end
        return (bl, br) if best > 0 else None

    def find_up_starting_at(l_start):
        if l_start < 0 or l_start >= H:
            return None
        best, bl, br = 0.0, -1, -1
        for r in range(l_start, H):
            v = up_pf[r + 1] - up_pf[l_start]
            if v > best:
                best, bl, br = v, l_start, r
        return (bl, br) if best > 0 else None

    # Best single-direction
    best_single_down = 0.0
    best_dl, best_dr = -1, -1
    for l in range(H):
        for r in range(l, H):
            v = down_pf[r + 1] - down_pf[l]
            if v > best_single_down:
                best_single_down = v
                best_dl, best_dr = l, r

    best_single_up = 0.0
    best_ul, best_ur = -1, -1
    for l in range(H):
        for r in range(l, H):
            v = up_pf[r + 1] - up_pf[l]
            if v > best_single_up:
                best_single_up = v
                best_ul, best_ur = l, r

    best_total = max(best_single_down, best_single_up, 0.0)
    best_pair = (best_dl if best_single_down == best_total else -1,
                 best_dr if best_single_down == best_total else -1,
                 best_ul if best_single_up == best_total else -1,
                 best_ur if best_single_up == best_total else -1)

    # Enumerate Down intervals + best non-overlapping Up (Eq 29-30)
    for l in range(H):
        for r in range(l, H):
            s_down = down_pf[r + 1] - down_pf[l]

            # Best Up before l (U_L[l-1], Eq 27)
            u_before = U_L_val[l - 1] if l > 0 else 0.0
            # Best Up after r (U_R[r+1], Eq 28)
            u_after = U_R_val[r + 1] if r + 1 < H else 0.0

            u_disj = max(u_before, u_after)  # Eq 29
            total = s_down + u_disj

            if total > best_total:
                best_total = total
                if u_disj > 0 and u_before >= u_after:
                    up_interval = find_up_ending_at(U_L_r[l - 1]) if l > 0 else None
                elif u_disj > 0:
                    up_interval = find_up_starting_at(U_R_l[r + 1]) if r + 1 < H else None
                else:
                    up_interval = None

                best_pair = (l, r,
                             up_interval[0] if up_interval else -1,
                             up_interval[1] if up_interval else -1)

    dl, dr, ul, ur = best_pair

    return {
        "I_down": (dl, dr) if dl >= 0 else None,
        "I_up": (ul, ur) if ul >= 0 else None,
        "total_value": float(best_total),
        "down_only_value": float(best_single_down),
        "up_only_value": float(best_single_up),
    }


def brute_force_proposal(g_hat_down: np.ndarray, g_hat_up: np.ndarray) -> dict:
    """O(H^4) exhaustive search for validation."""
    H = len(g_hat_down)
    down_pf = np.zeros(H + 1)
    up_pf = np.zeros(H + 1)
    for h in range(H):
        down_pf[h + 1] = down_pf[h] + g_hat_down[h]
        up_pf[h + 1] = up_pf[h] + g_hat_up[h]

    def S(l, r, pf):
        return pf[r + 1] - pf[l]

    best_total = 0.0
    best_dl, best_dr = -1, -1
    best_ul, best_ur = -1, -1

    # Single direction
    for l in range(H):
        for r in range(l, H):
            vd = S(l, r, down_pf)
            if vd > best_total:
                best_total = vd
                best_dl, best_dr = l, r
                best_ul, best_ur = -1, -1
            vu = S(l, r, up_pf)
            if vu > best_total:
                best_total = vu
                best_dl, best_dr = -1, -1
                best_ul, best_ur = l, r

    # Dual direction
    for dl in range(H):
        for dr in range(dl, H):
            for ul in range(H):
                for ur in range(ul, H):
                    if dr < ul or ur < dl:
                        vd = S(dl, dr, down_pf)
                        vu = S(ul, ur, up_pf)
                        if vd + vu > best_total:
                            best_total = vd + vu
                            best_dl, best_dr = dl, dr
                            best_ul, best_ur = ul, ur

    return {
        "I_down": (best_dl, best_dr) if best_dl >= 0 else None,
        "I_up": (best_ul, best_ur) if best_ul >= 0 else None,
        "total_value": float(best_total),
    }
