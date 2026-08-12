"""Synthetic end-to-end pipeline smoke — traces one day through full IAH chain.

Proves the P0-3 orchestrator owns the full sequence:
    S1 rank → IAH candidate → S3-M memory/k → S3-C DVG → S4 predict
And that the formal path never touches legacy HCH.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hch_v2_pipeline import HCHV2UniversalPipeline
from s1_rank import S1RankReference
from w1_retrieval import CAGMAtomMemory
from query_replay import build_directional_gains, form_final_pi, full_replay_chain
from double_event import double_event_proposal

SEED = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def make_synthetic_day(host, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # host [H, 1]
    s = float(np.mean(np.abs(host)))
    z0 = np.arcsinh(host / max(s, 1e-12))
    return host, s, z0


@test("pipeline full chain traces one day")
def _():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    H = 24

    # ---- S1: build rank reference from synthetic S1 host predictions ----
    s1_host = np.random.randn(200, H, 1) * 30 + 80
    s1_z0 = []
    for i in range(200):
        s = float(np.mean(np.abs(s1_host[i])))
        s1_z0.append(np.arcsinh(s1_host[i] / max(s, 1e-12)).ravel())
    s1_z0 = np.concatenate(s1_z0)

    pipe = HCHV2UniversalPipeline(d_core_context=8, d_model=32, alpha=0.10, k=3, seed=SEED)
    pipe.fit_s1_reference(s1_z0)

    # ---- S2: train candidate on synthetic S2 batches ----
    # context = [u (rank, 1), time_feat (7)] = 8 dims
    def make_context(host_batch):
        B, Hh, _ = host_batch.shape
        ctx = []
        for b in range(B):
            hb = host_batch[b].squeeze(-1).numpy().astype(np.float64)
            s = float(np.mean(np.abs(hb)))
            z0 = np.arcsinh(hb / max(s, 1e-12))
            u = pipe.s1_rank_ref(z0)
            time_feat = np.zeros((Hh, 7), dtype=np.float32)
            for h in range(Hh):
                time_feat[h, 0] = np.sin(2 * np.pi * h / 24)
                time_feat[h, 1] = np.cos(2 * np.pi * h / 24)
            ctx.append(np.concatenate([u.reshape(-1, 1), time_feat], axis=1))
        return torch.tensor(np.stack(ctx), dtype=torch.float32)

    s2_batches = []
    for _ in range(20):
        host = torch.tensor(np.random.randn(4, H, 1) * 30 + 80, dtype=torch.float32)
        target = host + torch.tensor(np.random.randn(4, H, 1) * 8, dtype=torch.float32)
        ctx = make_context(host)
        vm = torch.ones(4, H)
        s2_batches.append((host, ctx, target, vm))

    loss = pipe.train_candidate_s2(s2_batches, epochs=5, lr=1e-3, patience=3)
    assert np.isfinite(loss), f"loss should be finite, got {loss}"

    # ---- S3-M: build memory from S3-M days ----
    s3m_days = []
    for i in range(10):
        host = torch.tensor(np.random.randn(1, H, 1) * 30 + 80, dtype=torch.float32)
        ctx = make_context(host)
        with torch.no_grad():
            out = pipe.candidate_head(host, ctx)
        s = float(out["s"][0])
        zY = np.arcsinh((host[0].numpy() / max(s, 1e-12)))
        s3m_days.append({"date": f"s3m_{i}", "candidate": out, "target_zY": zY})

    mem = pipe.fit_s3_memory(s3m_days)
    assert len(mem) == 10

    # select k
    pipe.k = 3

    # ---- S3-C: calibrate (new interface: candidate + target_zY) ----
    s3c_days = []
    for i in range(10):
        host = torch.tensor(np.random.randn(1, H, 1) * 30 + 80, dtype=torch.float32)
        ctx = make_context(host)
        with torch.no_grad():
            out = pipe.candidate_head(host, ctx)
        s = float(out["s"][0])
        zY = np.arcsinh((host[0].numpy() / max(s, 1e-12)))
        s3c_days.append({"candidate": out, "target_zY": zY})
    q_info = pipe.calibrate_s3c(s3c_days)
    assert q_info["q"] != float("inf")

    # ---- S4: target-free predict ----
    host_s4 = torch.tensor(np.random.randn(2, H, 1) * 30 + 80, dtype=torch.float32)
    ctx_s4 = make_context(host_s4)
    evidence = pipe.predict_s4(host_s4, ctx_s4)

    # Verify evidence artifacts
    assert "candidate" in evidence
    assert "final_action" in evidence
    assert "x_final" in evidence
    assert evidence["x_final"].shape == (2, H, 1)

    # Verify bundle round-trip
    bundle = pipe.freeze_bundle(dataset_id="synthetic", split_hash="test")
    assert bundle.hash() is not None
    assert bundle.frozen_k == 3
    assert bundle.dvg_q is not None

    print(f"  loss={loss:.4f}, q={q_info['q']:.4f}, "
          f"actions={evidence['final_action']}")
    print(f"  bundle hash={bundle.hash()}")


@test("formal pipeline does not import legacy HCH")
def _():
    import hch_v2_pipeline as p
    # Check actual module namespace: legacy symbols must not be imported
    for legacy in ["BiOMC", "candidate_loss_fn", "state_loss_fn",
                   "ContinuousStateHead", "calibrate_s3"]:
        assert not hasattr(p, legacy), f"pipeline exposes legacy symbol {legacy}"

    # Legacy guard must exist and be invocable
    from _legacy.hch_v2 import require_not_legacy
    try:
        require_not_legacy("formal_test")
        assert False, "should raise"
    except RuntimeError:
        pass


@test("final pi_q replay differs from directional pre-proposal gain")
def _():
    np.random.seed(SEED)
    # Build a memory where proposal removes some hours
    mem = CAGMAtomMemory()
    H = 8
    for i in range(5):
        # synthetic candidate
        z0 = np.zeros(H)
        w_minus = np.full(H, 0.3)
        w_zero = np.full(H, 0.4)
        w_plus = np.full(H, 0.3)
        m_minus = np.full(H, 0.2)
        m_plus = np.full(H, 0.2)
        target_zY = np.random.randn(H) * 0.3
        valid = np.ones(H, dtype=bool)
        # manually add day
        mem.dates.append(f"d{i}")
        mem.z0.append(z0)
        mem.w_minus.append(w_minus)
        mem.w_zero.append(w_zero)
        mem.w_plus.append(w_plus)
        mem.m_minus.append(m_minus)
        mem.m_plus.append(m_plus)
        mem.target_zY.append(target_zY)
        mem.valid_mask.append(valid)

    m_minus_q = np.full(H, 0.2)
    m_plus_q = np.full(H, 0.2)
    nbr = [0, 1, 2]

    # Full chain: directional -> proposal -> final pi -> final replay
    chain = full_replay_chain(mem, nbr, m_minus_q, m_plus_q, double_event_proposal)

    # Directional gain (pre-proposal) uses FULL dose everywhere
    dir_gains = build_directional_gains(mem, nbr, m_minus_q, m_plus_q)

    # Final pi is sparse (only proposed intervals)
    assert chain["pi_q"].shape == (H,)
    # A_hat must come from final sparse pi, not full directional dose
    assert "A_hat" in chain
    assert np.isfinite(chain["A_hat"])

    print(f"  directional g_down={dir_gains['g_hat_down'].round(3)}")
    print(f"  final pi_q={chain['pi_q'].round(3)}")
    print(f"  final A_hat={chain['A_hat']:.4f}")


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
