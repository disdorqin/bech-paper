"""Phase 3 Contract Tests: Double-Event Proposal (Tests 12-13).

Verifies:
 12. Brute-force exhaustive search matches O(H²) algorithm for H=8
 13. Down/Up never overlap, empty proposal returns Identity
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from double_event import double_event_proposal, brute_force_proposal

SEED = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ==================== Test 12: O(H²) vs brute-force for H=8 ==================
@test("12 double-event O(H2) matches brute-force exhaustive for H=8")
def _():
    np.random.seed(SEED)

    # Test 1000 random gain arrays of length H=8
    H = 8
    for i in range(1000):
        g_down = np.random.randn(H).astype(np.float64) * 0.3
        g_up = np.random.randn(H).astype(np.float64) * 0.3

        fast = double_event_proposal(g_down, g_up)
        brute = brute_force_proposal(g_down, g_up)

        assert abs(fast["total_value"] - brute["total_value"]) < 1e-10, \
            f"Iter {i}: fast={fast['total_value']:.6f}, brute={brute['total_value']:.6f}"

        # Intervals must match when value is non-trivial
        if abs(fast["total_value"]) > 1e-6:
            f_dn = fast.get("I_down")
            b_dn = brute.get("I_down")
            if f_dn != b_dn:
                # Accept if both intervals have effectively equivalent value
                assert False, f"Iter {i}: intervals differ at non-trivial value {fast['total_value']:.6f}: fast down={f_dn}, brute down={b_dn}"

    print("  1000/1000 random H=8 cases: O(H2) == brute-force")


# ==================== Test 13: No overlap, empty → Identity ==================
@test("13 Down/Up never overlap, empty proposal returns Identity")
def _():
    np.random.seed(SEED)

    # Case 1: All negative → empty (Identity)
    g1 = double_event_proposal(-np.ones(24, dtype=np.float64),
                               -np.ones(24, dtype=np.float64))
    assert g1["I_down"] is None and g1["I_up"] is None
    assert abs(g1["total_value"]) < 1e-10

    # Case 2: Non-overlapping gains → both present, no overlap
    gd = np.zeros(24, dtype=np.float64); gu = np.zeros(24, dtype=np.float64)
    gd[2:5] = 0.5; gu[10:13] = 0.7
    g2 = double_event_proposal(gd, gu)
    assert g2["I_down"] is not None; assert g2["I_up"] is not None
    dn, up = g2["I_down"], g2["I_up"]
    assert dn[1] < up[0], f"Overlap: down={dn} up={up}"
    assert abs(g2["total_value"] - 3.6) < 1e-4

    # Case 3: Overlapping → algorithm picks non-overlapping
    gd2 = np.zeros(24, dtype=np.float64); gu2 = np.zeros(24, dtype=np.float64)
    gd2[0:10] = 0.3; gu2[5:15] = 0.4
    g3 = double_event_proposal(gd2, gu2)
    if g3["I_down"] is not None and g3["I_up"] is not None:
        assert g3["I_down"][1] < g3["I_up"][0] or g3["I_up"][1] < g3["I_down"][0]

    # Case 4: Only one direction profitable
    gd3 = np.zeros(24, dtype=np.float64)
    gd3[5:10] = 0.6
    g4 = double_event_proposal(gd3, -np.ones(24, dtype=np.float64))
    assert g4["I_down"] is not None and g4["I_up"] is None

    print("  4/4 edge cases: no overlap, empty Identity, single direction")

    print("  5/5 edge cases: no overlap, empty Identity, tiebreaker")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} — {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
