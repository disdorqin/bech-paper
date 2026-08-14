"""Phase4 CAVM — context_action_memory.py unit tests (build spec §8.1-8.6).

Covers the standalone module only. Pipeline wiring / predict_s4 target-free /
observe_outcome / §5.1.1 interleaved-domain isolation are added once P1
integrates the module into HCHV2UniversalPipeline.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from context_action_memory import (
    CAVMExperience,
    ContextKeyBuilder,
    ContextActionMemory,
    context_distance,
    atom_w1_distance,
    composite_distance,
    KEY_VERSION,
)
from w1_retrieval import CAGMAtomMemory

SEED = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


H = 24
D_CORE = 13
D_SIG = 8


# ------------------------------------------------------------------ helpers
def _core_ctx(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.5, size=(H, D_CORE))


def _domain_det(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3, 3, size=D_SIG)


def _candidate(seed=0, z0=None, valid=None):
    """Fake candidate dict, torch tensors so CAGMAtomMemory can consume it too."""
    rng = np.random.default_rng(seed)
    H_ = H
    z0v = np.zeros(H_) if z0 is None else np.asarray(z0, dtype=np.float64)
    cand = {
        "z0": torch.as_tensor(z0v, dtype=torch.float64).view(1, H_, 1),
        "w_minus": torch.as_tensor(np.clip(rng.normal(0.05, 0.04, H_), 0, None)
                                   ).view(1, H_, 1),
        "w_zero": torch.as_tensor(np.clip(rng.normal(0.90, 0.04, H_), 0, None)
                                  ).view(1, H_, 1),
        "w_plus": torch.as_tensor(np.clip(rng.normal(0.05, 0.04, H_), 0, None)
                                  ).view(1, H_, 1),
        "m_minus": torch.as_tensor(np.clip(rng.normal(0.02, 0.02, H_), 0, None)
                                   ).view(1, H_, 1),
        "m_plus": torch.as_tensor(np.clip(rng.normal(0.02, 0.02, H_), 0, None)
                                  ).view(1, H_, 1),
        "valid_mask": torch.as_tensor(
            np.ones(H_, dtype=bool) if valid is None else np.asarray(valid, bool)
        ).view(1, H_),
    }
    return cand


def _experience(date, cand, kb, target=None, seed=0, domain="LAGO_DE"):
    """Build a revealed CAVMExperience; target required."""
    tgt = np.zeros(H) if target is None else np.asarray(target, np.float64)
    key = kb.build(cand, _core_ctx(seed), cand["valid_mask"], _domain_det(seed))
    return CAVMExperience(
        date=date, context_key=key,
        z0=cand["z0"], w_minus=cand["w_minus"], w_zero=cand["w_zero"],
        w_plus=cand["w_plus"], m_minus=cand["m_minus"], m_plus=cand["m_plus"],
        target_zY=tgt, valid_mask=cand["valid_mask"],
        audit_domain=domain)


def _fill_cagm(cagm, cand, date):
    cagm.add_day(date, cand, np.zeros(H))


# ==================== T1: key builder dims / version / health ================
@test("1 key builder fixed dim, version, no NaN/Inf")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE, d_sig=D_SIG,
                           n_optional_roles=0)
    assert kb.version == KEY_VERSION == "cavm-key-v1"
    expected = 8 + 5 + 2 * D_CORE + D_SIG + 8 + 3 * 0
    assert kb.dim == expected

    cand = _candidate(seed=0)
    key = kb.build(cand, _core_ctx(0), cand["valid_mask"], _domain_det(0))
    assert key.shape == (expected,)
    assert np.isfinite(key).all()
    assert not np.isnan(key).any()
    print("  dim", expected, "clean key OK")


# ==================== T2: info isolation — banned fields rejected ============
@test("2 target/residual/action gain into key builder raise")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    base = _candidate(seed=0)

    for banned in ("target_raw", "target_zY", "residual",
                   "action_gain", "A_hat", "A_true", "action_error"):
        bad = dict(base)
        bad[banned] = np.zeros(H)
        try:
            kb.build(bad, _core_ctx(0), base["valid_mask"], _domain_det(0))
            assert False, f"should have rejected {banned!r}"
        except ValueError as e:
            assert banned in str(e)
    print("  banned label/action fields all rejected")


# ==================== T3: deterministic key + all-invalid fallback ===========
@test("3 deterministic; all-invalid hours safe zero fallback")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    cand_a = _candidate(seed=1)
    key_a = kb.build(cand_a, _core_ctx(1), cand_a["valid_mask"], _domain_det(1))
    key_a2 = kb.build(cand_a, _core_ctx(1), cand_a["valid_mask"], _domain_det(1))
    assert np.array_equal(key_a, key_a2)

    # All hours invalid -> every hourly stat falls back to 0, no NaN/Inf.
    # domain_det is a day-level descriptor (not gated by valid hours) and stays.
    vm = np.zeros(H, dtype=bool)
    cand_bad = _candidate(seed=2, valid=vm)
    det = _domain_det(2)
    key_bad = kb.build(cand_bad, _core_ctx(2), cand_bad["valid_mask"], det)
    assert np.isfinite(key_bad).all()
    # Layout: shape(8) | dyn(5) | time(2*D) | sig(D_SIG) | atom(8)
    n_time = 2 * D_CORE
    hourly_blocks = np.concatenate([
        key_bad[:8], key_bad[8:13],
        key_bad[13:13 + n_time], key_bad[13 + n_time + D_SIG:],
    ])
    assert np.count_nonzero(hourly_blocks) == 0
    np.testing.assert_allclose(key_bad[13 + n_time:13 + n_time + D_SIG], det)
    print("  deterministic + all-invalid zero fallback OK")


# ==================== T4: context_distance = euclidean =======================
@test("4 context_distance equals manual euclidean")
def _():
    rng = np.random.default_rng(0)
    qk = rng.normal(size=20)
    keys = rng.normal(size=(5, 20))
    got = context_distance(qk, keys)
    want = np.array([np.linalg.norm(k - qk) for k in keys])
    np.testing.assert_allclose(got, want, atol=1e-12)
    assert context_distance(qk, np.zeros((0, 20))).size == 0
    print("  context_distance == euclidean")


# ==================== T5: atom_w1_distance == CAGM retrieval =================
@test("5 atom_w1_distance matches CAGMAtomMemory.build_retrieval_index")
def _():
    cagm = CAGMAtomMemory()
    cam = ContextActionMemory("global", ContextKeyBuilder(d_core_context=D_CORE))
    kb = cam.key_builder

    for i in range(3):
        c = _candidate(seed=10 + i)
        _fill_cagm(cagm, c, f"d{i}")
        cam.add_revealed_day(_experience(f"d{i}", c, kb, domain="LAGO_DE"))

    q = _candidate(seed=99)
    w1_cagm = cagm.build_retrieval_index(q)
    w1_cam = cam.atom_w1_distances(q)
    assert w1_cam.shape == (3,)
    np.testing.assert_allclose(w1_cam, w1_cagm, atol=1e-12)

    # Order of neighbors identical to W1-only (ID-based, no self in memory).
    assert np.array_equal(np.argsort(w1_cam), np.argsort(w1_cagm))
    print("  atom_w1_distance == CAGM retrieval, order preserved")


# ==================== T6: composite λ=1,0 reproduces W1-only =================
@test("6 lambda_atom=1,lambda_ctx=0 reproduces W1-only neighbor order")
def _():
    cam = ContextActionMemory("global", ContextKeyBuilder(d_core_context=D_CORE))
    cagm = CAGMAtomMemory()
    kb = cam.key_builder
    for i in range(5):
        c = _candidate(seed=20 + i)
        _fill_cagm(cagm, c, f"m{i}")
        cam.add_revealed_day(_experience(f"m{i}", c, kb, domain="LAGO_DE"))

    q = _candidate(seed=88)
    qk = kb.build(q, _core_ctx(88), q["valid_mask"], _domain_det(88))

    res = cam.query(qk, q, k=3, lambda_atom=1.0, lambda_ctx=0.0)
    w1_only = cagm.get_neighbors(cagm.build_retrieval_index(q), k=3)
    assert res["neighbor_ids"] == w1_only, \
        f"CAVM {res['neighbor_ids']} vs W1 {w1_only}"
    assert res["effective_neighbor_count"] == 3
    # distance_w1 reported is the /max-normalized value, monotone-preserved.
    assert all(np.diff(res["distance_w1"]) >= -1e-12)

    # And lambda_atom=0,lambda_ctx=1 gives pure context order.
    res_ctx = cam.query(qk, q, k=3, lambda_atom=0.0, lambda_ctx=1.0)
    ctx_order = np.argsort(cam.context_distances(qk))[:3]
    assert list(res_ctx["neighbor_ids"]) == [int(i) for i in ctx_order]
    print("  composite retrieval contracts hold")


# ==================== T7: add_revealed_day guards + CAGM lists ===============
@test("7 add_revealed_day requires revealed target; lists compatible")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    cam = ContextActionMemory("global", kb)
    c = _candidate(seed=30)

    # Unrevealed target must be rejected.
    bad = CAVMExperience(date="x", context_key=np.zeros(kb.dim), z0=c["z0"],
                         w_minus=c["w_minus"], w_zero=c["w_zero"],
                         w_plus=c["w_plus"], m_minus=c["m_minus"],
                         m_plus=c["m_plus"], target_zY=None,
                         valid_mask=c["valid_mask"])
    try:
        cam.add_revealed_day(bad)
        assert False, "unrevealed target accepted"
    except ValueError as e:
        assert "revealed" in str(e) or "target_zY" in str(e)

    cam.add_revealed_day(_experience("d0", c, kb, domain="LAGO_DE"))
    # CAGMAtomMemory-compatible list attributes present.
    for attr in ("dates", "z0", "w_minus", "w_zero", "w_plus", "m_minus",
                 "m_plus", "target_zY", "valid_mask"):
        assert len(getattr(cam, attr)) == 1, attr
    assert cam.dates == ["d0"]
    assert cam.z0[0].shape == (H,)
    print("  reveal guard + compatible lists OK")


# ==================== T8: query self-exclusion + empty fallback ==============
@test("8 query excludes self by ID; empty memory falls back to count 0")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    cam = ContextActionMemory("local", kb)

    # Empty memory -> effective_neighbor_count 0, caller must fall back.
    q = _candidate(seed=40)
    qk = kb.build(q, _core_ctx(40), q["valid_mask"], _domain_det(40))
    empty = cam.query(qk, q, k=3)
    assert empty["effective_neighbor_count"] == 0
    assert empty["neighbor_ids"] == []
    assert empty["source_scope"] == "local"

    # Self in memory with identical candidate -> excluded even if nearest.
    c = _candidate(seed=50)
    cam.add_revealed_day(_experience("self", c, kb, domain="LAGO_DE"))
    # Query day has the SAME atom measure as stored 'self'.
    same = _candidate(seed=50)
    qk2 = kb.build(same, _core_ctx(50), same["valid_mask"], _domain_det(50))
    res = cam.query(qk2, same, k=3, exclude_date="self")
    assert res["self_excluded"] is True
    assert res["effective_neighbor_count"] == 0  # only 'self' was there
    # Without exclude_date, the identical day is a legal perfect neighbor.
    res2 = cam.query(qk2, same, k=3)
    assert res2["neighbor_ids"] == [0]
    print("  ID-based self-exclusion + empty fallback OK")


# ==================== T9: freeze / from_frozen round-trip ====================
@test("9 freeze/from_frozen round-trip preserves key, neighbors, distances")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    cam = ContextActionMemory("global", kb)
    for i in range(4):
        c = _candidate(seed=60 + i)
        cam.add_revealed_day(_experience(f"m{i}", c, kb, domain="LAGO_DE"))

    state = cam.freeze()
    assert state["key_version"] == KEY_VERSION
    assert state["key_dim"] == kb.dim
    cam2 = ContextActionMemory.from_frozen(state)

    assert cam2.scope == "global"
    assert cam2.dates == cam.dates
    q = _candidate(seed=77)
    qk = kb.build(q, _core_ctx(77), q["valid_mask"], _domain_det(77))
    r1 = cam.query(qk, q, k=2)
    r2 = cam2.query(qk, q, k=2)
    assert r1["neighbor_ids"] == r2["neighbor_ids"]
    np.testing.assert_allclose(r1["distance_total"], r2["distance_total"],
                               atol=1e-12)
    np.testing.assert_allclose(r1["distance_context"], r2["distance_context"],
                               atol=1e-12)
    print("  freeze/from_frozen round-trip identical")


# ==================== T10: market_id audit field is inert ====================
@test("10 market_id audit field does not affect key or query")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    c1 = _candidate(seed=70)
    c2 = _candidate(seed=70)
    assert "market_id" not in c1  # guard would not reject it, but it must be inert

    k1 = kb.build(c1, _core_ctx(70), c1["valid_mask"], _domain_det(70))
    # Same physical inputs, only a foreign audit key added -> identical key.
    c2b = dict(c2)
    c2b["market_id"] = "SHANDONG_DA"
    k2 = kb.build(c2b, _core_ctx(70), c2b["valid_mask"], _domain_det(70))
    assert np.array_equal(k1, k2)

    cam = ContextActionMemory("global", kb)
    cam.add_revealed_day(_experience("m0", c1, kb, domain="LAGO_DE"))
    r_a = cam.query(k1, c1, k=1)
    c3 = _candidate(seed=70)
    c3["market_id"] = "SHANDONG_DA"
    r_b = cam.query(kb.build(c3, _core_ctx(70), c3["valid_mask"],
                             _domain_det(70)), c3, k=1)
    assert r_a["neighbor_ids"] == r_b["neighbor_ids"]
    np.testing.assert_allclose(r_a["distance_total"], r_b["distance_total"])
    print("  market_id inert in key and retrieval")


# ==================== T11: optional roles — empty → dim 0, mismatch raises ==
@test("11 optional empty -> no dims; role count mismatch raises")
def _():
    kb0 = ContextKeyBuilder(d_core_context=D_CORE, n_optional_roles=0)
    c = _candidate(seed=80)
    k0 = kb0.build(c, _core_ctx(80), c["valid_mask"], _domain_det(80))
    assert k0.shape[0] == kb0.dim

    kb1 = ContextKeyBuilder(d_core_context=D_CORE, n_optional_roles=2)
    ov = np.ones((H, 2))
    k1 = kb1.build(c, _core_ctx(80), c["valid_mask"], _domain_det(80),
                   optional_values=ov, optional_roles=["fc", "act"],
                   optional_masks=np.ones((H, 2)))
    assert k1.shape[0] == kb1.dim
    assert np.isfinite(k1).all()

    # Role count mismatch raises (never silently truncates).
    try:
        kb1.build(c, _core_ctx(80), c["valid_mask"], _domain_det(80),
                  optional_values=np.ones((H, 3)), optional_roles=["a", "b", "c"],
                  optional_masks=np.ones((H, 3)))
        assert False, "role mismatch accepted"
    except ValueError as e:
        assert "n_optional_roles" in str(e)
    print("  optional dims / mismatch guard OK")


# ==================== T12: identical atom days are legal neighbors ===========
@test("12 two identical atom days are legal perfect neighbors (P1-2)")
def _():
    kb = ContextKeyBuilder(d_core_context=D_CORE)
    cam = ContextActionMemory("global", kb)
    c0 = _candidate(seed=90)
    c1 = _candidate(seed=90)  # same seed -> identical atom measures
    cam.add_revealed_day(_experience("a", c0, kb, domain="LAGO_DE"))
    cam.add_revealed_day(_experience("b", c1, kb, domain="LAGO_DE"))
    assert cam.atom_w1_distances(c0)[1] == 0.0  # W1=0 between identical days

    q = _candidate(seed=90)
    qk = kb.build(q, _core_ctx(90), q["valid_mask"], _domain_det(90))
    res = cam.query(qk, q, k=2, lambda_atom=1.0, lambda_ctx=0.0)
    assert res["effective_neighbor_count"] == 2
    print("  identical-day W1=0 neighbors legal")


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
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
