"""P1 correction tests — rank ties, self-exclusion, one-day, context, Data Signature."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from s1_rank import S1RankReference
from w1_retrieval import CAGMAtomMemory, w1_3atom
from hch_v2_context import CoreContextEncoder, OptionalCovariateEncoder, DataSignature
from eval_manifest import DAY_LENGTH_PROTOCOL

TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


@test("rank: tied pool maps identical query to 0.5")
def _():
    pool = np.array([1.0] * 100)  # all identical
    ref = S1RankReference(pool)
    u = ref(np.array([1.0]))  # query identical to pool
    assert abs(u[0] - 0.5) < 1e-6, f"identical query should be 0.5, got {u[0]}"
    # smaller query -> lower rank
    u_lo = ref(np.array([0.5]))
    assert u_lo[0] < 0.5
    # larger query -> higher rank
    u_hi = ref(np.array([2.0]))
    assert u_hi[0] > 0.5


@test("rank: mid-rank convention matches manual counts")
def _():
    pool = np.array([0.0, 1.0, 1.0, 1.0, 2.0])  # 1.0 appears 3 times
    ref = S1RankReference(pool)
    # query = 1.0: count(<1)=1, count(==1)=3 -> (1 + 1.5)/5 = 0.5
    u = ref(np.array([1.0]))
    assert abs(u[0] - 0.5) < 1e-6, f"mid-rank of 1.0 should be 0.5, got {u[0]}"
    # query = 0.0: count(<0)=0, count(==0)=1 -> 0.5/5 = 0.1
    u0 = ref(np.array([0.0]))
    assert abs(u0[0] - 0.1) < 1e-6


@test("self-exclusion by ID, not distance (two identical days remain neighbors)")
def _():
    mem = CAGMAtomMemory()
    # Two days with identical atom measures -> W1 = 0
    for i in range(2):
        cand = {"w_minus": torch.tensor([[0.3] * 24]), "w_zero": torch.tensor([[0.4] * 24]),
                "w_plus": torch.tensor([[0.3] * 24]), "m_minus": torch.tensor([[0.2] * 24]),
                "m_plus": torch.tensor([[0.2] * 24]),
                "z0": torch.tensor([[0.0] * 24]),
                "valid_mask": torch.ones(1, 24)}
        mem.add_day(f"d{i}", cand, np.zeros(24))

    # Query identical to day 0
    q = {"w_minus": torch.tensor([[0.3] * 24]), "w_zero": torch.tensor([[0.4] * 24]),
         "w_plus": torch.tensor([[0.3] * 24]), "m_minus": torch.tensor([[0.2] * 24]),
         "m_plus": torch.tensor([[0.2] * 24]),
         "valid_mask": torch.ones(1, 24)}
    dists = mem.build_retrieval_index(q)
    # Both days have W1=0
    assert abs(dists[0]) < 1e-10 and abs(dists[1]) < 1e-10

    # Exclude day 0 by ID -> only day 1 remains (even though identical)
    nbr = mem.get_neighbors(dists, k=2, exclude_idx=0)
    assert 0 not in nbr, f"day 0 should be excluded by ID, got {nbr}"
    assert 1 in nbr, f"day 1 (identical) should remain neighbor, got {nbr}"


@test("add_day enforces single-day batch")
def _():
    mem = CAGMAtomMemory()
    # batch size 2 should raise
    cand = {"w_minus": torch.randn(2, 24), "w_zero": torch.randn(2, 24),
            "w_plus": torch.randn(2, 24), "m_minus": torch.randn(2, 24),
            "m_plus": torch.randn(2, 24), "z0": torch.randn(2, 24),
            "valid_mask": torch.ones(2, 24)}
    try:
        mem.add_day("d", cand, np.zeros(24))
        assert False, "should raise for batch>1"
    except ValueError:
        pass


@test("core encoder + Data Signature produce finite modulation")
def _():
    torch.manual_seed(0)
    B, H = 2, 24
    d_in, d_model = 8, 32
    enc = CoreContextEncoder(d_in, d_model)
    core = torch.randn(B, H, d_in)
    z0 = torch.randn(B, H)
    core[..., 0] = z0  # dimension 0 is z0 (CoreContextEncoder convention)
    out = enc(core)
    assert out.shape == (B, H, d_model)
    assert torch.isfinite(out).all()

    # Data Signature deterministic descriptors are scale-free (finite, correct dim)
    sig = DataSignature(d_model, 8)
    det = sig.compute_deterministic(z0)
    assert det.shape == (B, 8)
    assert torch.isfinite(det).all()


@test("optional branch zero-init preserves near-core output")
def _():
    torch.manual_seed(0)
    opt = OptionalCovariateEncoder(d_value=4, d_model=32)
    B, H, N = 2, 24, 3
    values = torch.randn(B, H, N, 4)
    roles = torch.zeros(B, H, N, dtype=torch.long)
    masks = torch.ones(B, H, N)
    out = opt(values, roles, masks)
    # zero-init residual: output should be near zero
    assert torch.abs(out).max() < 1e-6, f"zero-init should give ~0, got {out.abs().max()}"


@test("DAY_LENGTH_PROTOCOL is COMPLETE_24H_ONLY")
def _():
    assert DAY_LENGTH_PROTOCOL == "COMPLETE_24H_ONLY"


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
            import traceback; traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
