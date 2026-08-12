"""Double-Event Proposal — O(H²) exact enumeration (Eq 23-30).

Math: hch_v2_iah_crps_final_math_core_v0.3 §4

U_L/U_R are computed by Kadane max-subarray (O(H) each), matching the math
core's Eq 27-28. The Down-interval enumeration is O(H²). Total O(H²).
"""
from __future__ import annotations

import numpy as np


def _kadane_prefix(g: np.ndarray):
    """U_L(t) = max(0, max_{l<=r<=t} sum g[l..r]), with argmax interval.

    Returns (val [H], l [H], r [H]) where l/r are the argmax interval for
    the best segment ending at or before each t (or -1 if empty).
    """
    H = len(g)
    val = np.zeros(H)
    l_arr = np.full(H, -1, dtype=int)
    r_arr = np.full(H, -1, dtype=int)
    cur, cur_l = 0.0, 0
    best, bl, br = 0.0, -1, -1
    for t in range(H):
        if cur <= 0:
            cur, cur_l = float(g[t]), t
        else:
            cur += float(g[t])
        if cur > best:
            best, bl, br = cur, cur_l, t
        val[t], l_arr[t], r_arr[t] = best, bl, br
    return val, l_arr, r_arr


def _kadane_suffix(g: np.ndarray):
    """U_R(t) = max(0, max_{t<=l<=r} sum g[l..r]), with argmax interval."""
    H = len(g)
    val = np.zeros(H)
    l_arr = np.full(H, -1, dtype=int)
    r_arr = np.full(H, -1, dtype=int)
    cur, cur_r = 0.0, H - 1
    best, bl, br = 0.0, -1, -1
    for t in range(H - 1, -1, -1):
        if cur <= 0:
            cur, cur_r = float(g[t]), t
        else:
            cur += float(g[t])
        if cur > best:
            best, bl, br = cur, t, cur_r
        val[t], l_arr[t], r_arr[t] = best, bl, br
    return val, l_arr, r_arr


def double_event_proposal(g_hat_down: np.ndarray, g_hat_up: np.ndarray) -> dict:
    """Find optimal non-overlapping Down+Up interval pair (Eq 25-30).

    O(H²): Kadane O(H) prefix/suffix for Up, then O(H²) Down enumeration.
    For H=24 this is exact enumeration, not an approximation.
    """
    H = len(g_hat_down)
    assert len(g_hat_up) == H

    down_pf = np.zeros(H + 1)
    up_pf = np.zeros(H + 1)
    for h in range(H):
        down_pf[h + 1] = down_pf[h] + g_hat_down[h]
        up_pf[h + 1] = up_pf[h] + g_hat_up[h]

    # U_L / U_R via Kadane (O(H)), Eq 27-28
    U_L_val, U_L_l, U_L_r = _kadane_prefix(g_hat_up)
    U_R_val, U_R_l, U_R_r = _kadane_suffix(g_hat_up)

    # Best single-direction via Kadane
    best_down = 0.0
    bdl, bdr = -1, -1
    cur, cur_l = 0.0, 0
    for t in range(H):
        if cur <= 0:
            cur, cur_l = float(g_hat_down[t]), t
        else:
            cur += float(g_hat_down[t])
        if cur > best_down:
            best_down, bdl, bdr = cur, cur_l, t

    best_up = U_R_val[0]  # best Up anywhere
    bul, bur = (U_R_l[0], U_R_r[0]) if best_up > 0 else (-1, -1)

    best_total = 0.0
    best_pair = (-1, -1, -1, -1)
    if best_down > best_total:
        best_total = best_down
        best_pair = (bdl, bdr, -1, -1)
    if best_up > best_total:
        best_total = best_up
        best_pair = (-1, -1, bul, bur)

    # Enumerate Down intervals + best non-overlapping Up (Eq 29-30)
    for l in range(H):
        for r in range(l, H):
            s_down = down_pf[r + 1] - down_pf[l]
            u_before = U_L_val[l - 1] if l > 0 else 0.0
            u_after = U_R_val[r + 1] if r + 1 < H else 0.0
            u_disj = max(u_before, u_after)
            total = s_down + u_disj

            if total > best_total:
                best_total = total
                if u_disj > 0 and u_before >= u_after:
                    ul = U_L_l[l - 1] if l > 0 else -1
                    ur = U_L_r[l - 1] if l > 0 else -1
                elif u_disj > 0:
                    ul = U_R_l[r + 1] if r + 1 < H else -1
                    ur = U_R_r[r + 1] if r + 1 < H else -1
                else:
                    ul, ur = -1, -1
                best_pair = (l, r, ul, ur)

    dl, dr, ul, ur = best_pair
    return {
        "I_down": (dl, dr) if dl >= 0 else None,
        "I_up": (ul, ur) if ul >= 0 else None,
        "total_value": float(best_total),
        "down_only_value": float(best_down),
        "up_only_value": float(best_up),
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
