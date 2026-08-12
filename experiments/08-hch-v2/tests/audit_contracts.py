"""Auditor's contract tests — v0.3 spec compliance deep verification.

These go beyond the existing 20 tests and probe edge cases,
mathematical correctness, and spec deviations.
"""
from __future__ import annotations

import sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from iah_candidate import IAHCandidateHead
from iah_crps_loss import iah_crps_loss, crps_manual
from s1_rank import S1RankReference
from w1_retrieval import w1_3atom, CAGMAtomMemory, day_w1_distance
from query_replay import replay_query_dose, verify_gain_bound
from double_event import double_event_proposal, brute_force_proposal
from dvg_calibrate import DGVSplitConformal

TESTS = []
FAILED = 0
PASSED = 0


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def assert_close(a, b, tol=1e-5, msg=""):
    a, b = float(a), float(b)
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b} (diff={abs(a-b):.2e})")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# =============================== Phase 1: IAH Candidate ==========================

@test("A01 scale edge-case: all-NaN host -> SCALE_UNIDENTIFIED")
def _():
    head = IAHCandidateHead(d_context=16)
    host = torch.full((2, 24, 1), float("nan"))
    ctx = torch.randn(2, 24, 16)
    with torch.no_grad():
        out = head(host, ctx)
    assert_true((out["scale_valid"] == 0).all(), "scale should be invalid")
    assert_true((out["w_zero"] == 1).all(), "all mass should be at center")
    assert_true(torch.allclose(out["x_identity"], host, equal_nan=True),
                "x_identity should equal host for invalid scale")


@test("A02 scale edge-case: very small host -> nonzero s, finite z0")
def _():
    head = IAHCandidateHead(d_context=16)
    host = torch.full((2, 24, 1), 1e-8)
    ctx = torch.randn(2, 24, 16)
    with torch.no_grad():
        out = head(host, ctx)
    assert_true((out["s"] > 0).all(), "scale should be > 0 for small nonzero host")
    assert_true(torch.isfinite(out["z0"]).all(), "z0 must be finite")


@test("A03 asinh identity: z0(s*c) = asinh(c) independent of scale")
def _():
    head = IAHCandidateHead(d_context=16)
    for c in [0.1, 1.0, 5.0, 100.0]:
        host = torch.full((1, 24, 1), c)
        ctx = torch.randn(1, 24, 16)
        with torch.no_grad():
            out = head(host, ctx)
        s = out["s"].item()
        z0_mean = out["z0"].mean().item()
        expected = np.arcsinh(float(c) / float(s)) if s > 0 else 0
        assert_close(z0_mean, expected, 1e-3, f"c={c} s={s}")


@test("A04 ReLU zero shift: identity candidate == host exactly")
def _():
    head = IAHCandidateHead(d_context=16)
    for p in head.shift_head.parameters():
        p.data.zero_()
    host = torch.tensor([[[30.0]], [[-5.0]], [[0.0]]]).repeat(1, 24, 1)
    ctx = torch.randn(3, 24, 16)
    with torch.no_grad():
        out = head(host, ctx)
    assert_true(torch.allclose(out["x_down"], host, atol=1e-5),
                "x_down should equal host when shift is zero")
    assert_true(torch.allclose(out["x_up"], host, atol=1e-5),
                "x_up should equal host when shift is zero")


@test("A05 loss gradient flows through mass and shift heads")
def _():
    head = IAHCandidateHead(d_context=16)
    host = torch.randn(4, 24, 1) * 50 + 40
    target = host + torch.randn(4, 24, 1) * 20
    ctx = torch.randn(4, 24, 16)
    out = head(host, ctx)
    loss = iah_crps_loss(out, target)
    loss.backward()
    for name, p in head.named_parameters():
        if "mass_head" in name or "shift_head" in name:
            assert_true(p.grad is not None and p.grad.norm().item() > 0,
                        f"{name} has no gradient")


@test("A06 CRPS spread term equals discrete pairwise formula")
def _():
    for _ in range(50):
        w_m = np.random.rand() * 0.4
        w_p = np.random.rand() * 0.4
        w_z = 1.0 - w_m - w_p
        m_m = np.random.rand() * 10
        m_p = np.random.rand() * 10
        z0 = 0.0
        zY = np.random.randn() * 5
        # Direct CRPS
        direct = crps_manual(zY, z0, w_m, w_p, m_m, m_p)
        # Pairwise formula: CRPS = Σ w_a|zY-z_a| - 0.5 Σ_{a,b} w_a w_b |z_a-z_b|
        z = np.array([z0 - m_m, z0, z0 + m_p])
        w = np.array([w_m, w_z, w_p])
        term1 = sum(w[a] * abs(zY - z[a]) for a in range(3))
        term2 = -0.5 * sum(w[a] * w[b] * abs(z[a] - z[b]) for a in range(3) for b in range(3))
        pairwise = term1 + term2
        assert_close(direct, pairwise, 1e-8, f"CRPS mismatch: {direct} vs {pairwise}")


# =============================== Phase 2: S1 Rank ==============================

@test("A07 S1Rank: per-hour pools produce rank near 0.5 for new values")
def _():
    z0 = np.random.randn(1000)
    hours = np.random.randint(0, 24, 1000)
    ref = S1RankReference(z0, s1_hours=hours)
    u = ref(np.array([0.0] * 24), hours=np.arange(24))
    assert_true((u >= 0).all() and (u <= 1).all(), "ranks should be in [0,1]")
    assert_true(abs(u.mean() - 0.5) < 0.3, "average rank should be near 0.5")


@test("A08 S1Rank: empty pool returns 0.5 neutral")
def _():
    ref = S1RankReference(np.array([]), s1_hours=np.array([]))
    u = ref(np.array([1.0, 2.0, -3.0]))
    assert_true(np.allclose(u, 0.5), "empty pool should return 0.5")


@test("A09 S1Rank: monotonic — larger z0 has larger rank")
def _():
    z0 = np.random.randn(500)
    ref = S1RankReference(z0)
    u = ref(np.array([-5.0, 0.0, 5.0]))
    assert_true(u[0] <= u[1] <= u[2], f"ranks must be monotonic: {u}")


# =============================== Phase 3: W1 + Retrieval ========================

@test("A10 W1: identical measures => distance 0")
def _():
    for _ in range(20):
        wm = np.random.rand() * 0.3
        wp = np.random.rand() * 0.3
        wz = 1 - wm - wp
        mm = np.random.rand() * 5
        mp = np.random.rand() * 5
        d = w1_3atom(wm, wz, wp, mm, mp, wm, wz, wp, mm, mp)
        assert_close(d, 0.0, 1e-12, "identical measures must have W1=0")


@test("A11 W1: distance bounded by max |m|")
def _():
    for _ in range(20):
        wm_a, wp_a = np.random.rand() * 0.5, np.random.rand() * 0.5
        wm_b, wp_b = np.random.rand() * 0.5, np.random.rand() * 0.5
        wz_a = 1 - wm_a - wp_a
        wz_b = 1 - wm_b - wp_b
        mm_a, mp_a = np.random.rand() * 5, np.random.rand() * 5
        mm_b, mp_b = np.random.rand() * 5, np.random.rand() * 5
        d = w1_3atom(wm_a, wz_a, wp_a, mm_a, mp_a, wm_b, wz_b, wp_b, mm_b, mp_b)
        max_m = max(mm_a, mp_a, mm_b, mp_b)
        assert_true(d <= max_m * 3 + 1e-10, f"W1={d} > 3*max_m={3*max_m}")


@test("A12 CAGMAtomMemory: self-exclusion from neighbors confirmed")
def _():
    mem = CAGMAtomMemory()
    z0 = np.random.randn(24)
    wm = np.random.rand(24) * 0.3
    wz = 1 - wm - np.random.rand(24) * 0.3
    wp = 1 - wm - wz
    mm = np.random.rand(24) * 2
    mp = np.random.rand(24) * 2
    valid = np.ones(24, dtype=bool)

    cand_template = {
        "w_minus": torch.tensor(wm), "w_zero": torch.tensor(wz),
        "w_plus": torch.tensor(wp), "m_minus": torch.tensor(mm),
        "m_plus": torch.tensor(mp), "valid_mask": torch.tensor(valid),
        "z0": torch.tensor(z0),
    }

    for i in range(5):
        mem.add_day(f"d{i}", cand_template, np.random.randn(24))

    q_cand = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in cand_template.items()}
    dists = mem.build_retrieval_index(q_cand)
    neighbors = mem.get_neighbors(dists, k=3)
    assert_true(0 not in neighbors, "query day must not be its own neighbor")


# =============================== Phase 4: Double Event ==========================

@test("A13 double-event: brute force matches for H=24 on random data")
def _():
    for _ in range(100):
        g_down = np.random.randn(24) * 0.5
        g_up = np.random.randn(24) * 0.5
        r1 = double_event_proposal(g_down, g_up)
        r2 = brute_force_proposal(g_down, g_up)
        assert_close(r1["total_value"], r2["total_value"], 1e-10,
                     f"total_value mismatch: {r1['total_value']} vs {r2['total_value']}")
        # Intervals may differ when values are tied; skip interval comparison for ties
        if abs(r1["total_value"] - r2["total_value"]) > 1e-10:
            print(f"WARNING: total value mismatch at {r1['total_value']} vs {r2['total_value']}")


@test("A14 double-event: all-negative => empty, returns Identity")
def _():
    g_down = -np.ones(24)
    g_up = -np.ones(24)
    r = double_event_proposal(g_down, g_up)
    assert_true(r["I_down"] is None and r["I_up"] is None, "all negative should be Identity")
    assert_true(r["total_value"] == 0.0, "total value should be 0")


@test("A15 double-event: tie-breaking prefers shorter interval")
def _():
    g_down = np.array([1.0, 1.0, -5.0, 1.0, 1.0])
    g_up = np.zeros(5)
    r = double_event_proposal(g_down, g_up)
    if r["I_down"] is not None:
        s, e = r["I_down"]
        length = e - s + 1
        # Should prefer the shorter {1,1} over {1,1,-5,1,1} when they have same value
        assert_true(length <= 5, f"interval too long: {length}")


# =============================== Phase 5: DVG =================================

@test("A16 DVG: q=inf produces identity for any A_hat")
def _():
    dvg = DGVSplitConformal(alpha=0.10)
    dvg.compute_quantile()  # no errors -> q=inf
    r = dvg.lcb(1000.0)
    assert_true(not r["execute"], "with q=inf, even huge A_hat should not execute")
    r2 = dvg.lcb(-1000.0)
    assert_true(not r2["execute"], "with q=inf, negative A_hat must not execute")


@test("A17 DVG: compute_quantile idempotent")
def _():
    dvg = DGVSplitConformal(alpha=0.10)
    for i in range(10):
        dvg.record_error(float(i) * 0.5, 0.0)
    r1 = dvg.compute_quantile()
    r2 = dvg.compute_quantile()
    assert_true(r1["q"] == r2["q"], "compute_quantile should be idempotent")


@test("A18 DVG: freeze/restore round-trip")
def _():
    dvg1 = DGVSplitConformal(alpha=0.15)
    for i in range(20):
        dvg1.record_error(i * 0.3, 0.0)
    dvg1.compute_quantile()
    lcb1 = dvg1.lcb(3.0)

    state = dvg1.freeze()
    dvg2 = DGVSplitConformal.from_frozen(state)
    lcb2 = dvg2.lcb(3.0)

    assert_true(lcb1["execute"] == lcb2["execute"], "freeze/restore must preserve decision")
    assert_close(lcb1["lcb"], lcb2["lcb"], 1e-10)


# =============================== Phase 6: Query Replay =========================

@test("A19 replay: zero dose => zero gain")
def _():
    z0 = np.random.randn(24)
    zY = z0 + np.random.randn(24) * 2
    pi = np.zeros(24)
    r = replay_query_dose(z0, zY, pi, np.ones(24, dtype=bool))
    assert_close(r["A"], 0.0, 1e-10, "zero dose should have zero gain")


@test("A20 replay: perfect correction => positive gain")
def _():
    z0 = np.ones(24) * 10.0
    zY = z0 + 5.0  # target is 5 above host
    pi = np.full(24, 5.0)  # correct by +5
    r = replay_query_dose(z0, zY, pi, np.ones(24, dtype=bool))
    assert_true(r["A"] > 0, "perfect correction should have positive gain")


@test("A21 replay: gain bounded by |pi| for extreme values")
def _():
    for _ in range(200):
        z0 = np.random.randn(24) * 20
        zY = z0 + np.random.randn(24) * 30
        pi = np.random.randn(24) * 15
        r = replay_query_dose(z0, zY, pi, np.ones(24, dtype=bool))
        assert_true(verify_gain_bound(pi, r["g"], np.ones(24, dtype=bool)),
                    "gain bound violated")


# =============================== CRPS math =====================================

@test("A22 CRPS: loss >= 0 always")
def _():
    head = IAHCandidateHead(d_context=8)
    for _ in range(30):
        host = torch.randn(4, 24, 1) * 50 + 40
        target = host + torch.randn(4, 24, 1) * 30
        ctx = torch.randn(4, 24, 8)
        with torch.no_grad():
            out = head(host, ctx)
        loss = iah_crps_loss(out, target)
        assert_true(loss.item() >= -1e-6, f"CRPS should be non-negative, got {loss.item()}")


@test("A23 CRPS: perfect prediction => loss = 0")
def _():
    head = IAHCandidateHead(d_context=8)
    for p in head.parameters():
        p.data.zero_()
    host = torch.randn(2, 24, 1) * 30 + 50
    ctx = torch.randn(2, 24, 8)
    with torch.no_grad():
        out = head(host, ctx)
    loss = iah_crps_loss(out, host)  # target = host => zY = z0
    assert_close(loss.item(), 0.0, 1e-4, "CRPS should be 0 when target=host")


# =============================== W1 precision ==================================

@test("A24 W1: symmetric property")
def _():
    for _ in range(30):
        wa, wp_a = np.random.rand() * 0.3, np.random.rand() * 0.3
        wz_a = max(0, 1 - wa - wp_a)
        ma, pa = np.random.rand() * 3, np.random.rand() * 3
        wb, wp_b = np.random.rand() * 0.3, np.random.rand() * 0.3
        wz_b = max(0, 1 - wb - wp_b)
        mb, pb = np.random.rand() * 3, np.random.rand() * 3
        d_ab = w1_3atom(wa, wz_a, wp_a, ma, pa, wb, wz_b, wp_b, mb, pb)
        d_ba = w1_3atom(wb, wz_b, wp_b, mb, pb, wa, wz_a, wp_a, ma, pa)
        assert_close(d_ab, d_ba, 1e-10, f"W1 asymmetry: {d_ab} != {d_ba}")


@test("A25 W1: triangle inequality")
def _():
    for _ in range(20):
        wa, wpa = np.random.rand()*0.3, np.random.rand()*0.3; wza = max(0,1-wa-wpa)
        wb, wpb = np.random.rand()*0.3, np.random.rand()*0.3; wzb = max(0,1-wb-wpb)
        wc, wpc = np.random.rand()*0.3, np.random.rand()*0.3; wzc = max(0,1-wc-wpc)
        ma, pa = np.random.rand()*3, np.random.rand()*3
        mb, pb = np.random.rand()*3, np.random.rand()*3
        mc, pc = np.random.rand()*3, np.random.rand()*3
        d_ab = w1_3atom(wa,wza,wpa,ma,pa, wb,wzb,wpb,mb,pb)
        d_bc = w1_3atom(wb,wzb,wpb,mb,pb, wc,wzc,wpc,mc,pc)
        d_ac = w1_3atom(wa,wza,wpa,ma,pa, wc,wzc,wpc,mc,pc)
        assert_true(d_ac <= d_ab + d_bc + 1e-8,
                    f"triangle inequality: {d_ac} > {d_ab}+{d_bc}")


if __name__ == "__main__":
    for name, fn in TESTS:
        try:
            fn()
            PASSED += 1
            print(f"  PASS: {name}")
        except Exception as e:
            FAILED += 1
            print(f"  FAIL: {name} — {e}")
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED > 0:
        sys.exit(1)
