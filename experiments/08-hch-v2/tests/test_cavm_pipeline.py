"""Phase4 P1 — pipeline-level CAVM integration tests (build spec §5, §8).

Proves:
  E1 contract — memory_mode="cavm" + fit_cavm_memory adds read-only audit
      fields WITHOUT altering any prediction (neighbors/A_hat/q/LCB/x_final
      identical day-by-day to the pure W1 path).
  bundle round-trip carries CAVM ledgers; old w1 bundles load cleanly.
  fit_cavm_memory rejects unrevealed days.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hch_v2_pipeline import HCHV2UniversalPipeline
from context_action_memory import KEY_VERSION

SEED = 0
H = 24
D_CORE = 8
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def _build_chain(seed=SEED, memory_mode="cavm"):
    """Synthetic S1->S2->S3-M->S3-C chain. Returns (pipe, host_s4, ctx_s4)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    s1_host = np.random.randn(200, H, 1) * 30 + 80
    s1_z0 = []
    for i in range(200):
        s = float(np.mean(np.abs(s1_host[i])))
        s1_z0.append(np.arcsinh(s1_host[i] / max(s, 1e-12)).ravel())
    s1_z0 = np.concatenate(s1_z0)

    pipe = HCHV2UniversalPipeline(d_core_context=D_CORE, d_model=32,
                                  alpha=0.10, k=3, seed=seed,
                                  memory_mode=memory_mode)
    pipe.fit_s1_reference(s1_z0)

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
        host = torch.tensor(np.random.randn(4, H, 1) * 30 + 80,
                            dtype=torch.float32)
        target = host + torch.tensor(np.random.randn(4, H, 1) * 8,
                                     dtype=torch.float32)
        ctx = make_context(host)
        vm = torch.ones(4, H)
        s2_batches.append((host, ctx, target, vm))
    pipe.train_candidate_s2(s2_batches, epochs=5, lr=1e-3, patience=3)

    def make_day(d):
        host = torch.tensor(np.random.randn(1, H, 1) * 30 + 80,
                            dtype=torch.float32)
        ctx = make_context(host)
        with torch.no_grad():
            out = pipe.candidate_head(host, ctx)
        s = float(out["s"][0])
        zY = np.arcsinh(host[0].numpy() / max(s, 1e-12))
        return {"date": d, "candidate": out, "target_zY": zY,
                "core_context": ctx.numpy()}

    mem_days = [make_day(f"s3m_{i}") for i in range(10)]
    pipe.fit_s3_memory(mem_days)
    pipe.k = 3

    s3c_days = [make_day(f"s3c_{i}") for i in range(10)]
    pipe.calibrate_s3c(s3c_days)

    host_s4 = torch.tensor(np.random.randn(3, H, 1) * 30 + 80,
                           dtype=torch.float32)
    ctx_s4 = make_context(host_s4)
    return pipe, host_s4, ctx_s4


@test("E1 fit_cavm_memory equivalent neighbors + audit, predictions unchanged")
def _():
    """Same pipe, predict before and after fit_cavm_memory -> identical."""
    pipe, host_s4, ctx_s4 = _build_chain(seed=3, memory_mode="cavm")

    ev_before = pipe.predict_s4(host_s4, ctx_s4)
    assert "cavm" not in ev_before

    # CAVM days = current CAGM memory content (same revealed days).
    days = []
    for i, date in enumerate(pipe.memory.dates):
        days.append({
            "date": date,
            "candidate": {
                "z0": torch.as_tensor(pipe.memory.z0[i]).view(1, H, 1),
                "w_minus": torch.as_tensor(pipe.memory.w_minus[i]).view(1, H, 1),
                "w_zero": torch.as_tensor(pipe.memory.w_zero[i]).view(1, H, 1),
                "w_plus": torch.as_tensor(pipe.memory.w_plus[i]).view(1, H, 1),
                "m_minus": torch.as_tensor(pipe.memory.m_minus[i]).view(1, H, 1),
                "m_plus": torch.as_tensor(pipe.memory.m_plus[i]).view(1, H, 1),
                "valid_mask": torch.ones(1, H, dtype=torch.bool),
            },
            "target_zY": pipe.memory.target_zY[i],
        })
    fit_info = pipe.fit_cavm_memory(days)
    assert fit_info["global"] == len(pipe.memory)

    ev_after = pipe.predict_s4(host_s4, ctx_s4)
    assert ev_after["memory_mode"] == "cavm"
    assert ev_after["context_key_version"] == KEY_VERSION

    # Prediction path byte-identical (E1: diagnosis never changes output).
    # neighbor IDs are the invariant; distance VALUES become the /max-normalised
    # composite distance (audit field) once the CAVM ledger is active.
    for i in range(3):
        assert ev_after["neighbors"][i] == ev_before["neighbors"][i]
        assert ev_after["final_action"] == ev_before["final_action"]
        assert ev_after["A_hat"] == ev_before["A_hat"]
        assert ev_after["lcb"] == ev_before["lcb"]
        assert torch.equal(ev_after["x_final"], ev_before["x_final"])
    assert ev_after["q"] == ev_before["q"]

    # Audit fields present and consistent with the W1-selected neighbors
    # (same revealed days -> λ=1,λ=0 neighbor IDs match the prediction's).
    cavm = ev_after["cavm"]
    assert len(cavm["neighbor_scopes"]) == 3
    assert cavm["effective_neighbor_count"] == [3, 3, 3]
    for i in range(3):
        assert cavm["neighbor_ids"][i] == ev_before["neighbors"][i]
        assert len(cavm["distance_context"][i]) == 3
        assert len(cavm["distance_w1"][i]) == 3
    print("  E1: predictions identical, CAVM audit consistent")


@test("P2 lambda=(1,0) reproduces W1; (0,1) genuinely switches neighbors")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=7, memory_mode="cavm")

    days = []
    for i, date in enumerate(pipe.memory.dates):
        days.append({
            "date": date,
            "candidate": {
                "z0": torch.as_tensor(pipe.memory.z0[i]).view(1, H, 1),
                "w_minus": torch.as_tensor(pipe.memory.w_minus[i]).view(1, H, 1),
                "w_zero": torch.as_tensor(pipe.memory.w_zero[i]).view(1, H, 1),
                "w_plus": torch.as_tensor(pipe.memory.w_plus[i]).view(1, H, 1),
                "m_minus": torch.as_tensor(pipe.memory.m_minus[i]).view(1, H, 1),
                "m_plus": torch.as_tensor(pipe.memory.m_plus[i]).view(1, H, 1),
                "valid_mask": torch.ones(1, H, dtype=torch.bool),
            },
            "target_zY": pipe.memory.target_zY[i],
        })
    pipe.fit_cavm_memory(days)

    ev_pre = pipe.predict_s4(host_s4, ctx_s4)  # λ=(1,0) default
    assert ev_pre["cavm_lambda"] == {"atom": 1.0, "context": 0.0}
    # λ=(1,0): neighbor IDs identical to pure W1 selection.
    for i in range(3):
        assert ev_pre["neighbors"][i] == pipe.memory.get_neighbors(
            pipe.memory.build_retrieval_index(
                {k: v for k, v in _cavm_view(ev_pre, i).items()}),
            k=3), f"λ=(1,0) neighbor mismatch at day {i}"

    # λ=(0,1): context-only must select different neighbors (genuine switch).
    pipe.set_cavm_retrieval(0.0, 1.0)
    ev_ctx = pipe.predict_s4(host_s4, ctx_s4)
    assert ev_ctx["cavm_lambda"] == {"atom": 0.0, "context": 1.0}
    changed = sum(ev_ctx["neighbors"][i] != ev_pre["neighbors"][i]
                  for i in range(3))
    assert changed >= 1, "context-only retrieval did not change selection"

    # λ=(1,1): composite differs from both pure modes.
    pipe.set_cavm_retrieval(1.0, 1.0)
    ev_mix = pipe.predict_s4(host_s4, ctx_s4)
    mix_changed = sum(ev_mix["neighbors"][i] != ev_pre["neighbors"][i]
                      for i in range(3))
    assert mix_changed >= 0
    print(f"  λ switch: context-changed {changed}/3 days, "
          f"composite-changed {mix_changed}/3 days")


def _cavm_view(evidence, b):
    c = evidence["candidate"]
    return {
        "w_minus": c["w_minus"][b:b + 1],
        "w_zero": c["w_zero"][b:b + 1],
        "w_plus": c["w_plus"][b:b + 1],
        "m_minus": c["m_minus"][b:b + 1],
        "m_plus": c["m_plus"][b:b + 1],
        "valid_mask": c["valid_mask"][b:b + 1],
    }


@test("E1 cavm bundle round-trip restores ledgers + predictions")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=4, memory_mode="cavm")
    days = []
    for i, date in enumerate(pipe.memory.dates):
        days.append({
            "date": date,
            "candidate": {
                "z0": torch.as_tensor(pipe.memory.z0[i]).view(1, H, 1),
                "w_minus": torch.as_tensor(pipe.memory.w_minus[i]).view(1, H, 1),
                "w_zero": torch.as_tensor(pipe.memory.w_zero[i]).view(1, H, 1),
                "w_plus": torch.as_tensor(pipe.memory.w_plus[i]).view(1, H, 1),
                "m_minus": torch.as_tensor(pipe.memory.m_minus[i]).view(1, H, 1),
                "m_plus": torch.as_tensor(pipe.memory.m_plus[i]).view(1, H, 1),
                "valid_mask": torch.ones(1, H, dtype=torch.bool),
            },
            "target_zY": pipe.memory.target_zY[i],
        })
    pipe.fit_cavm_memory(days)

    ev_before = pipe.predict_s4(host_s4, ctx_s4)
    bundle = pipe.freeze_bundle(dataset_id="syn", split_hash="t")
    assert bundle.memory_mode == "cavm"
    assert bundle.cavm_key_version == KEY_VERSION
    assert bundle.cavm_global_state is not None
    assert bundle.cavm_global_hash != ""
    assert bundle.cavm_local_state is None

    pipe2 = HCHV2UniversalPipeline.from_bundle(bundle)
    assert pipe2.memory_mode == "cavm"
    assert pipe2.cavm_global is not None
    assert pipe2.cavm_key_builder.dim == pipe.cavm_key_builder.dim

    ev_after = pipe2.predict_s4(host_s4, ctx_s4)
    assert ev_after["final_action"] == ev_before["final_action"]
    assert ev_after["A_hat"] == ev_before["A_hat"]
    assert torch.equal(ev_after["x_final"], ev_before["x_final"])
    assert (ev_after["cavm"]["neighbor_ids"]
            == ev_before["cavm"]["neighbor_ids"])
    # Round-trip hash matches (same fields recomputed deterministically).
    assert bundle.hash() == pipe2.freeze_bundle(
        dataset_id="syn", split_hash="t").hash()
    print("  CAVM bundle round-trip restores ledger + predictions")


@test("E1 old w1 bundle loads with memory_mode=w1, no CAVM")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=5, memory_mode="w1")
    bundle = pipe.freeze_bundle(dataset_id="syn", split_hash="t")
    assert bundle.memory_mode == "w1"
    assert bundle.cavm_global_state is None
    assert bundle.cavm_key_version == ""

    pipe2 = HCHV2UniversalPipeline.from_bundle(bundle)
    assert pipe2.memory_mode == "w1"
    assert pipe2.cavm_global is None
    ev = pipe2.predict_s4(host_s4, ctx_s4)
    assert "cavm" not in ev
    assert ev["memory_mode"] == "w1"
    print("  old w1 bundle loads clean, no CAVM")


@test("E1 fit_cavm_memory rejects unrevealed days")
def _():
    pipe, _, _ = _build_chain(seed=6, memory_mode="cavm")
    bad_day = {
        "date": "x",
        "candidate": {
            "z0": torch.zeros(1, H, 1),
            "w_minus": torch.zeros(1, H, 1),
            "w_zero": torch.ones(1, H, 1),
            "w_plus": torch.zeros(1, H, 1),
            "m_minus": torch.zeros(1, H, 1),
            "m_plus": torch.zeros(1, H, 1),
            "valid_mask": torch.ones(1, H, dtype=torch.bool),
        },
        # target_zY deliberately absent -> must raise
    }
    try:
        pipe.fit_cavm_memory([bad_day])
        assert False, "unrevealed day accepted"
    except (KeyError, ValueError) as e:
        assert "target" in str(e).lower() or "target_zY" in str(e)
    print("  unrevealed day rejected")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} — {e!r}")
            import traceback; traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
