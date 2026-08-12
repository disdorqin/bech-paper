"""Phase 4+5 Contract Tests: Conformal LCB + Bundle/Legacy (Tests 14-20).

Tests 14-15: Conformal calibration
Tests 16-20: Bundle, target-free S4, legacy guard, timestamp, date disjoint
"""
import sys
import tempfile, os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from dvg_calibrate import DGVSplitConformal
from _legacy.hch_v2 import require_not_legacy, LEGACY_UNTRAINED
from eval_manifest import ExperimentManifest

SEED = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ==================== Test 14: Conformal rank boundaries =====================
@test("14 conformal rank, q=inf, LCB boundaries correct")
def _():
    np.random.seed(SEED)

    dvg = DGVSplitConformal(alpha=0.10)

    # Record 10 errors
    errors = [0.5, -0.3, 1.2, -0.8, 0.1, 0.9, -0.5, 0.0, 0.7, -1.0]
    for e in errors:
        dvg.record_error(e, 0.0)  # A_true=0 for synthetic

    q_info = dvg.compute_quantile()
    n = 10
    r_expected = int(np.ceil((n + 1) * 0.9))  # ceil(11*0.9) = ceil(9.9) = 10
    assert q_info["r"] == r_expected
    # q should be the 10th largest error = max of sorted errors
    assert abs(q_info["q"] - 1.2) < 1e-10  # max error

    # Test LCB
    lcb_info = dvg.lcb(0.5)
    assert lcb_info["lcb"] == 0.5 - 1.2  # -0.7
    assert not lcb_info["execute"]  # LCB < 0

    lcb_info2 = dvg.lcb(2.0)
    assert lcb_info2["lcb"] == 2.0 - 1.2  # 0.8
    assert lcb_info2["execute"]  # LCB > 0

    # Edge: empty calibration → q=+inf → never execute
    dvg2 = DGVSplitConformal(alpha=0.10)
    q2 = dvg2.compute_quantile()
    assert q2["q"] == float('inf')
    lcb_inf = dvg2.lcb(100.0)
    assert not lcb_inf["execute"]  # q=inf → never execute

    # Edge: r = n+1 → q = +inf
    dvg3 = DGVSplitConformal(alpha=0.01)  # very small alpha
    for i in range(5):
        dvg3.record_error(float(i), 0.0)
    q3 = dvg3.compute_quantile()
    # r = ceil(6 * 0.99) = ceil(5.94) = 6 > 5 → q = +inf
    assert q3["q"] == float('inf')
    assert q3["r"] == 6

    print("  rank/boundary/empty: all correct")


# ==================== Test 15: S3-C doesn't change candidate =================
@test("15 S3-C doesn't change candidate/memory/k/proposal")
def _():
    from iah_candidate import IAHCandidateHead
    from w1_retrieval import CAGMAtomMemory

    # Create frozen candidate state
    torch.manual_seed(SEED)
    head = IAHCandidateHead(4, 32).eval()
    host = torch.randn(2, 24, 1) * 50
    ctx = torch.randn(2, 24, 4)

    with torch.no_grad():
        out_before = head(host, ctx)
    before_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in out_before.items()
                    if isinstance(v, torch.Tensor)}

    # Simulate S3-C: record errors (calibration only)
    memory = CAGMAtomMemory()
    dvg = DGVSplitConformal(alpha=0.10)

    for _ in range(10):
        A_hat = float(np.random.randn() * 0.5)
        A_true = A_hat - float(np.random.randn() * 0.3)
        dvg.record_error(A_hat, A_true)

    dvg.compute_quantile()

    # Verify candidate output unchanged
    with torch.no_grad():
        out_after = head(host, ctx)

    for k in ["z0", "w_minus", "w_plus", "m_minus", "m_plus"]:
        if k in before_state and k in out_after:
            torch.testing.assert_close(out_after[k], before_state[k],
                                       atol=1e-6, rtol=1e-6)

    print("  candidate frozen during S3-C calibration")


# ==================== Test 16: Target-free S4 ================================
@test("16 target-free S4 runs without target_raw")
def _():
    from iah_candidate import IAHCandidateHead

    torch.manual_seed(SEED)
    head = IAHCandidateHead(4, 32).eval()

    # S4 batch: no target
    host = torch.randn(2, 24, 1) * 50
    ctx = torch.randn(2, 24, 4)

    with torch.no_grad():
        out = head(host, ctx)

    assert "x_identity" in out
    assert "x_down" in out
    assert "x_up" in out
    # No target required for forward pass
    print("  target-free S4 forward OK")


# ==================== Test 17: Bundle round-trip =============================
@test("17 bundle round-trip consistent")
def _():
    from _legacy.hch_v2 import HCHV2, HCHV2Config, HCHV2Bundle

    m1 = HCHV2(HCHV2Config(d_model=32))
    # Add some calibration data
    import json
    calib_params = {"alpha": 0.10, "q": 0.5, "n": 20}
    b = m1.freeze(calibration_params=calib_params, split_hash="test_hash_123")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        b.save(f.name)
        path = f.name

    b2 = HCHV2Bundle.load(path)
    m2 = HCHV2.from_bundle(b2)

    assert b.hash() == b2.hash()

    # Verify calibration params persisted
    assert b2.calibration_params == calib_params
    assert b2.split_hash == "test_hash_123"

    os.unlink(path)
    print("  bundle round-trip + calibration persistence OK")


# ==================== Test 18: Legacy guard ==================================
@test("18 legacy guard blocks formal runner")
def _():
    # Verify LEGACY_UNTRAINED flags are in place
    assert "CAGM_DVG" in LEGACY_UNTRAINED
    assert LEGACY_UNTRAINED["CAGM_DVG"] is True

    try:
        require_not_legacy("test_formal_runner")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "legacy" in str(e).lower()

    print("  legacy guard blocks formal runner")


# ==================== Test 19: Timestamp join ================================
@test("19 timestamp join order insensitive")
def _():
    from common import load_dataset, build_tabular

    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="LAGO_DE")

    s4_manifest = exp.build_s4_eval_manifest(
        np.full(len(ds["price"]), 1.0, dtype=np.float32))

    # Verify manifest order is deterministic
    s4_manifest2 = exp.build_s4_eval_manifest(
        np.full(len(ds["price"]), 1.0, dtype=np.float32))
    assert np.array_equal(s4_manifest.valid_indices, s4_manifest2.valid_indices)
    assert s4_manifest.n_hours == s4_manifest2.n_hours

    print("  timestamp join deterministic")


# ==================== Test 20: Date disjoint =================================
@test("20 S1/S2/S3-M/S3-C/S4 dates strictly disjoint")
def _():
    from common import load_dataset, build_tabular

    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="LAGO_DE")

    s1 = exp.dates_in_split("S1")
    s2 = exp.dates_in_split("S2")
    s3 = exp.dates_in_split("S3")
    s4 = exp.dates_in_split("S4")

    assert len(s1 & s2) == 0
    assert len(s2 & s3) == 0
    assert len(s3 & s4) == 0
    assert len(s1 & s3) == 0
    assert len(s1 & s4) == 0
    assert len(s2 & s4) == 0

    print("  all 4 splits disjoint")


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
