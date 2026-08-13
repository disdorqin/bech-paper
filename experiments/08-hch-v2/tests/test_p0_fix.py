"""P0-A..P0-D final pre-training fix tests (R1A gate).

Per docs/paper_prep/v2_final_prep/hch_v2_r1a_final_pretraining_fix_instruction_v0.1:
  P0-A  cumulative_bounds(): monotone, last == n_dates, exact n=500 counts
  P0-B  host fit indexes X/y by valid_row_in_split (raw indices would leak)
  P0-C  UniversalCoreTrainer: updates_A == updates_B under 10x batch imbalance
  P0-D  domain_det expanded to the real batch size on train/val/health paths
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from eval_manifest import ExperimentManifest, FRAC_7, SPLIT_7, cumulative_bounds
from common import load_dataset, build_tabular
from universal_trainer import UniversalCoreTrainer, DomainBatch
from iah_candidate import IAHCandidateHead

TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


@test("P0-A: cumulative_bounds(n=500) exact 7-segment counts")
def _():
    b = cumulative_bounds(500, FRAC_7)
    assert list(b) == [0, 200, 250, 330, 350, 375, 400, 500], f"got {b}"
    # monotone + exhaustive + exact segment sizes
    assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))
    assert b[-1] == 500
    assert list(np.diff(b)) == [200, 50, 80, 20, 25, 25, 100]
    # guard: the old broken idiom (counts[-1] = n - sum(cumulative counts))
    # produced a negative non-monotone tail — this test is not vacuous.
    cum = np.cumsum(FRAC_7)
    old = [int(round(500 * c)) for c in cum]
    old[-1] = 500 - sum(old[:-1])
    assert old[-1] < 0, "guard: old bug produced a negative boundary"


@test("P0-A: real LAGO_DE manifest boundaries monotone + exact counts")
def _():
    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="LAGO_DE")
    n_dates = len(exp.dates)
    counts = {s: len(exp.dates_in_split(s)) for s in SPLIT_7}
    assert sum(counts.values()) == n_dates
    cum = np.rint(np.cumsum(FRAC_7) * n_dates).astype(int)
    sizes = np.diff([0, *cum])
    for i, s in enumerate(SPLIT_7):
        assert counts[s] == sizes[i], f"{s}: {counts[s]} != {sizes[i]}"
    b = cumulative_bounds(n_dates, FRAC_7)
    assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))
    assert b[-1] == n_dates


@test("P0-B: host fit must use valid rows, not raw indices")
def _():
    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="LAGO_DE")
    raw = exp.valid_indices_in_split("H0")
    row = exp.valid_row_in_split("H0")
    # the trap is real: raw indices != compressed-row positions
    assert not np.array_equal(raw, row), "P0-B test would be vacuous"
    assert row.max() < len(X), "valid rows must be in-bounds for X/y"
    # every correct fit row's raw hour belongs to H0 dates (no leak, no shift)
    fit_dates = {exp.dates[int(exp.raw_to_date[int(valid[r])])] for r in row}
    assert fit_dates <= exp.dates_in_split("H0"), "host fit must be H0-only"
    # the raw-index version would include non-H0 (S1R) hours — proves leak
    shifted = {exp.dates[int(exp.raw_to_date[int(valid[r])])] for r in raw
               if r < len(X)}
    assert not (shifted <= exp.dates_in_split("H0")), \
        "raw indexing would leak S1R into host fit"


@test("P0-C: 10x batch imbalance -> updates_A == updates_B")
def _():
    def mk_batch():
        host = torch.rand(1, 24, 1) + 1.0
        ctx = torch.randn(1, 24, 5)
        target = host * (0.5 + torch.rand(1, 24, 1))
        vm = torch.ones(1, 24)
        return (host, ctx, target, vm)

    n_a, n_b = 100, 10
    A = DomainBatch(name="A", s2t_batches=[mk_batch() for _ in range(n_a)],
                    s2v_batches=[mk_batch()], domain_det=None)
    B = DomainBatch(name="B", s2t_batches=[mk_batch() for _ in range(n_b)],
                    s2v_batches=[mk_batch()], domain_det=None)

    def run_once():
        torch.manual_seed(0)
        head = IAHCandidateHead(d_core_context=5, d_model=16)
        t = UniversalCoreTrainer(head, seed=0)
        return t.train([A, B], epochs=1, lr=1e-3)

    r1, r2 = run_once(), run_once()
    up1 = r1["history"][0]["updates_per_domain"]
    up2 = r2["history"][0]["updates_per_domain"]
    assert up1 == {"A": 55, "B": 55}, f"got {up1}"  # K = median(100, 10)
    assert up2 == up1, f"seed reproducibility broken: {up1} vs {up2}"
    assert r1["best_macro_s2v"] == r2["best_macro_s2v"], \
        "same seed must reproduce the checkpoint result"


@test("P0-D: det expanded to real batch size on train/val/health (B=4)")
def _():
    torch.manual_seed(0)
    head = IAHCandidateHead(d_core_context=5, d_model=16)
    host = torch.rand(4, 24, 1) + 1.0
    ctx = torch.randn(4, 24, 5)
    target = host * 1.1
    vm = torch.ones(4, 24)
    det = np.ones(8, dtype=np.float64)  # 8-dim S1 descriptor
    d = DomainBatch(name="D", s2t_batches=[(host, ctx, target, vm)],
                    s2v_batches=[(host, ctx, target, vm)], domain_det=det)
    rep = UniversalCoreTrainer(head, seed=0).train([d], epochs=1, lr=1e-3)
    # would crash in DataSignature cat for B>1 with [1, d_det]; must run
    assert np.isfinite(rep["best_macro_s2v"]), "macro S2V must be finite"
    assert rep["history"][0]["health"], "health diagnostics must be populated"


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
            import traceback
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
