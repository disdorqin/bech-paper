"""Phase 2 Contract Tests: W1 retrieval + query-dose replay (Tests 9-11).

Verifies:
  9. Changing target doesn't change retrieval key (W1 distance stays same)
 10. Replay uses query dose pi_q, not history dose pi_j
 11. |g| ≤ |π| (hourly gain bounded by dose magnitude)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from iah_candidate import IAHCandidateHead
from w1_retrieval import CAGMAtomMemory, day_w1_distance
from query_replay import replay_query_dose, estimate_action_value, verify_gain_bound

SEED = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def make_synthetic_day(head, host, ctx):
    """Helper: run candidate head and return zY for a synthetic day."""
    with torch.no_grad():
        out = head(host.unsqueeze(0), ctx.unsqueeze(0))
    target = host + torch.randn_like(host) * 5
    s = out["s"].item()
    s_safe = max(s, 1e-12)
    zY = np.arcsinh((target.squeeze(-1) / s_safe).numpy())
    return out, zY


# ==================== Test 9: Key independent of target =======================
@test("09 retrieval key does not depend on target")
def _():
    torch.manual_seed(SEED)
    head = IAHCandidateHead(d_core_context=4, d_model=32).eval()

    # Build memory with 5 synthetic days (only host+context matters, not target)
    memory = CAGMAtomMemory()
    hosts = []
    for i in range(5):
        host = torch.randn(24, 1) * 50 + 100 + i * 10
        ctx = torch.randn(24, 4)
        out, zY_fake = make_synthetic_day(head, host, ctx)
        memory.add_day(f"day_{i}", out, zY_fake)
        hosts.append(host)

    # Query day: get distances with one target
    q_host = torch.randn(24, 1) * 50 + 100
    q_ctx = torch.randn(24, 4)
    q_out, q_zY_a = make_synthetic_day(head, q_host, q_ctx)

    dist_a = memory.build_retrieval_index(q_out)

    # Same query, different target — distance should be identical
    q_out_b, q_zY_b = make_synthetic_day(head, q_host, q_ctx)
    dist_b = memory.build_retrieval_index(q_out_b)

    np.testing.assert_array_equal(dist_a, dist_b)
    print(f"  distances: {dist_a}")


# ==================== Test 10: Replay uses query dose =========================
@test("10 replay uses query dose pi_q, not history dose pi_j")
def _():
    np.random.seed(SEED)

    # History day j
    z0_j = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
    zY_j = np.array([0.3, -0.5, 0.8, 0.1], dtype=np.float64)
    valid_j = np.ones(4, dtype=bool)

    # History dose (π_j) — should NOT be used
    pi_j = np.array([-0.50, 0.40, -0.30, 0.0], dtype=np.float64)

    # Query dose (π_q) — should be used
    pi_q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Scenario A: replay π_q=0 on history j → all gains=0
    result_A = replay_query_dose(z0_j, zY_j, pi_q, valid_j)

    # Scenario B: replay π_j on history j → non-zero gains
    result_B = replay_query_dose(z0_j, zY_j, pi_j, valid_j)

    # Zero query dose → A=0; non-zero history dose → A≠0
    assert abs(result_A["A"]) < 1e-10, f"Zero dose should give A=0, got {result_A['A']}"
    assert abs(result_B["A"]) > 1e-6, f"Non-zero dose should give A≠0"

    # Verify each hour: Eq 18 z_replay = z0_j + pi_q, Eq 19 g = |r_z| - |r_z - pi_q|
    for h in range(4):
        z_replay = z0_j[h] + pi_q[h]
        r_z = zY_j[h] - z0_j[h]
        g_manual = abs(r_z) - abs(zY_j[h] - z_replay)
        assert abs(result_A["g"][h] - g_manual) < 1e-10, \
            f"h={h}: g={result_A['g'][h]:.6f}, manual={g_manual:.6f}"

    print(f"  zero-dose A={result_A['A']:.4f}, non-zero-dose B={result_B['A']:.4f}")


# ==================== Test 11: |g| ≤ |π| =====================================
@test("11 hourly gain bounded by dose: |g| ≤ |π|")
def _():
    np.random.seed(SEED)

    # Generate 100 random scenarios
    for _ in range(100):
        z0 = np.random.randn(24).astype(np.float64) * 0.5
        zY = z0 + np.random.randn(24).astype(np.float64) * 0.3
        pi = (np.random.rand(24).astype(np.float64) - 0.5) * 0.4
        valid = np.ones(24, dtype=bool)

        result = replay_query_dose(z0, zY, pi, valid)

        assert verify_gain_bound(pi, result["g"], valid), \
            f"Bound violated: max_g={np.abs(result['g']).max():.4f}, max_pi={np.abs(pi).max():.4f}"

    # Edge case: pi = 0 everywhere → g = 0 everywhere
    result_zero = replay_query_dose(z0, zY, np.zeros(24), valid)
    assert np.allclose(result_zero["g"], 0), "Zero dose should give zero gain"

    print(f"  100 random cases + edge case: all bounded")


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
