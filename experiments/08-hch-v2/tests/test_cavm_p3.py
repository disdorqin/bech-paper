"""Phase4 P3 — observe_outcome() local-memory tests (build spec §5.2, §8).

Proves:
  §8.3 local update cannot change universal state — predictions before/after
      observe_outcome are byte-identical (local never consumed by predict_s4).
      Turning local memory off == global-only.
  §8.4 time order — a day's context_key is recorded at prediction time; the
      revealed label only enters through target_zY (A_true), never the key.
  §8.1 observe_outcome only callable post-reveal (missing target -> error;
      prediction-time key required -> error).
  observe is OFF by default (strictly frozen S4); toggled on via policy.
  local ledger round-trips through freeze/from_bundle with hash coverage.
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


def _build_chain(seed=SEED, memory_mode="cavm", n_mem=6, n_s4=3):
    """Synthetic S1->S2->S3-M->S3-C chain + fit_cavm_memory + S4 batch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    s1_host = np.random.randn(120, H, 1) * 30 + 80
    s1_z0 = []
    for i in range(120):
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
    for _ in range(12):
        host = torch.tensor(np.random.randn(4, H, 1) * 30 + 80,
                            dtype=torch.float32)
        target = host + torch.tensor(np.random.randn(4, H, 1) * 8,
                                     dtype=torch.float32)
        ctx = make_context(host)
        vm = torch.ones(4, H)
        s2_batches.append((host, ctx, target, vm))
    pipe.train_candidate_s2(s2_batches, epochs=4, lr=1e-3, patience=3)

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

    mem_days = [make_day(f"mem_{i}") for i in range(n_mem)]
    pipe.fit_s3_memory(mem_days)
    pipe.k = 3
    s3c_days = [make_day(f"cal_{i}") for i in range(4)]
    pipe.calibrate_s3c(s3c_days)
    pipe.fit_cavm_memory(mem_days)

    host_s4 = torch.tensor(np.random.randn(n_s4, H, 1) * 30 + 80,
                           dtype=torch.float32)
    ctx_s4 = make_context(host_s4)
    return pipe, host_s4, ctx_s4


@test("P3 observe off by default -> applied=False, local stays None")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=3)
    ev = pipe.predict_s4(host_s4, ctx_s4)
    assert "context_keys" in ev
    assert len(ev["context_keys"]) == 3
    r = pipe.observe_outcome(0, np.zeros(H), ev)
    assert r["applied"] is False and r["reason"] == "observe_disabled"
    assert pipe.cavm_local is None
    print("  observe off -> no local ledger, strictly frozen S4")


@test("P3 observe on: local appends; predictions byte-identical before/after")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=4)
    pipe.set_cavm_update_policy(observe=True)
    ev = pipe.predict_s4(host_s4, ctx_s4)
    assert "query_ids" not in ev
    ev["query_ids"] = ["s4_a", "s4_b", "s4_c"]

    # Predicted values (A_hat, pi, neighbors, x_final) frozen BEFORE observe.
    pre = {k: (torch.clone(v) if torch.is_tensor(v) else list(v))
           for k, v in ev.items() if k in ("A_hat", "pi", "neighbors",
                                           "x_final", "final_action", "lcb")}

    r0 = pipe.observe_outcome("s4_a", np.random.randn(H) * 10, ev)
    assert r0["applied"] and r0["n_local"] == 1 and r0["scope"] == "local"
    r1 = pipe.observe_outcome("s4_b", np.random.randn(H) * 10, ev)
    assert r1["n_local"] == 2
    assert pipe.cavm_global is not None
    assert len(pipe.cavm_global) == 6  # global ledger untouched by observe
    assert pipe.cavm_local.scope == "local"

    # Universal state: same pipe, re-predict -> identical evidence.
    def same(a, b):
        a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        for x, y in zip(a.ravel(), b.ravel()):
            if isinstance(x, str) or isinstance(y, str):
                if x != y:
                    return False
            elif not np.array_equal(np.asarray(x, dtype=float),
                                    np.asarray(y, dtype=float)):
                return False
        return True

    ev2 = pipe.predict_s4(host_s4, ctx_s4)
    for k in ("A_hat", "pi", "neighbors", "final_action", "lcb"):
        assert same(ev2[k], pre[k]), f"local observe changed {k}"
    assert torch.equal(ev2["x_final"], pre["x_final"])
    # k / lambda / q untouched.
    assert pipe.k == 3
    assert pipe.cavm_lambda == (1.0, 0.0)
    assert pipe.dvg.q is not None and pipe.dvg.q > 0
    # local ledger carries post-reveal audit values.
    e0 = pipe.cavm_local.experiences[0]
    assert e0.A_hat is not None and e0.A_true is not None
    assert abs(e0.action_error - (e0.A_hat - e0.A_true)) < 1e-12
    assert e0.timestamp != ""
    print("  local observe: universal untouched, audit fields recorded")


@test("P3 local never consumed -> turning observe off == global-only")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=5)
    pipe.set_cavm_update_policy(observe=True)
    ev = pipe.predict_s4(host_s4, ctx_s4)
    for i in range(3):
        pipe.observe_outcome(i, np.random.randn(H) * 10, ev)
    assert len(pipe.cavm_local) == 3

    pipe.set_cavm_update_policy(observe=False)
    ev_off = pipe.predict_s4(host_s4, ctx_s4)
    for i in range(3):
        assert ev_off["neighbors"][i] == ev["neighbors"][i]
    assert torch.equal(ev_off["x_final"], ev["x_final"])
    print("  local ledger present but predictions read global only")


@test("P3 key recorded at prediction time; target change never alters key")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=6)
    ev = pipe.predict_s4(host_s4, ctx_s4)
    key0 = np.array(ev["context_keys"][0], dtype=np.float64).copy()
    assert np.any(np.isfinite(key0))

    # Revealed label differs wildly; the stored key must be byte-identical
    # (info isolation §8.1: modifying query target cannot change the key).
    pipe.set_cavm_update_policy(observe=True)
    zY_alt = np.random.randn(H) * 100
    r = pipe.observe_outcome(0, zY_alt, ev)
    assert r["applied"]
    assert np.array_equal(ev["context_keys"][0], key0)
    assert np.array_equal(pipe.cavm_local.experiences[0].context_key, key0)
    print("  context_key immune to revealed label")


@test("P3 guards: post-reveal only (missing target / no key -> error)")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=7)
    pipe.set_cavm_update_policy(observe=True)
    ev = pipe.predict_s4(host_s4, ctx_s4)

    # w1-mode prediction has no context_keys -> observe must refuse.
    pipe_w1, h, c = _build_chain(seed=8, memory_mode="w1")
    pipe_w1.set_cavm_update_policy(observe=True)
    ev_w1 = pipe_w1.predict_s4(h, c)
    assert "context_keys" not in ev_w1
    r = pipe_w1.observe_outcome(0, np.zeros(H), ev_w1)
    assert r["applied"] is False and "cavm_not_active" in r["reason"]

    # Missing target -> error.
    try:
        pipe.observe_outcome(0, None, ev)
        assert False, "missing target accepted"
    except ValueError as e:
        assert "target" in str(e).lower()

    # Unknown query_id -> error.
    try:
        pipe.observe_outcome("does_not_exist", np.zeros(H), ev)
        assert False, "unresolvable query_id accepted"
    except ValueError:
        pass
    print("  guards: target required, key required, query_id resolvable")


@test("P3 A_true matches offline estimate_realized_A (scale-free target)")
def _():
    """observe_outcome's internal A_true must equal a direct recomputation."""
    from query_replay import estimate_realized_A as era
    pipe, host_s4, ctx_s4 = _build_chain(seed=11)
    pipe.set_cavm_update_policy(observe=True)
    ev = pipe.predict_s4(host_s4, ctx_s4)

    for i in range(3):
        s = float(ev["candidate"]["s"][i])
        # Scale-free revealed target, built from the day's own scale.
        zY = np.arcsinh(np.random.randn(H) * 8 / s)
        r = pipe.observe_outcome(i, zY, ev)

        z0 = ev["candidate"]["z0"][i].detach().cpu().numpy().ravel()
        vm = ev["candidate"]["valid_mask"][i].detach().cpu().numpy().ravel()
        pi = np.asarray(ev["pi"][i], dtype=np.float64).ravel()
        offline = float(era(z0, zY, pi, vm.astype(bool)))
        assert abs(r["A_true"] - offline) < 1e-9, \
            f"day {i}: internal {r['A_true']} != offline {offline}"
        assert abs(r["action_error"] - (r["A_hat"] - offline)) < 1e-9
    print("  observe A_true == offline estimate_realized_A (scale-free)")


@test("P3 local ledger round-trips through bundle + hash coverage")
def _():
    pipe, host_s4, ctx_s4 = _build_chain(seed=9)
    pipe.set_cavm_update_policy(observe=True)
    ev = pipe.predict_s4(host_s4, ctx_s4)
    for i in range(3):
        pipe.observe_outcome(i, np.random.randn(H) * 10, ev)
    assert len(pipe.cavm_local) == 3

    b1 = pipe.freeze_bundle(dataset_id="syn", split_hash="t")
    assert b1.cavm_local_state is not None
    assert b1.cavm_local_hash != ""
    assert b1.cavm_update_policy.get("observe") is True
    h1 = b1.hash()

    pipe2 = HCHV2UniversalPipeline.from_bundle(b1)
    assert pipe2.cavm_local is not None and len(pipe2.cavm_local) == 3
    assert pipe2.cavm_update_policy.get("observe") is True
    b2 = pipe2.freeze_bundle(dataset_id="syn", split_hash="t")
    assert b2.hash() == h1, "local-ledger round-trip hash mismatch"

    # Local lives in the local package only: extract_universal has no local.
    u = b1.extract_universal()
    assert "cavm_local_state" not in u
    assert u["cavm_global_state"] is not None
    print("  local ledger round-trips; extract_universal excludes local")


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
