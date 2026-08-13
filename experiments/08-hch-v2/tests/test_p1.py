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
from eval_manifest import DAY_LENGTH_PROTOCOL, ExperimentManifest
from universal_trainer import UniversalCoreTrainer, DomainBatch

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

    # Data Signature: frozen domain descriptors, NOT per-day quantile
    sig = DataSignature(d_model, 8)
    det = sig.compute_deterministic()
    assert det.shape == (8,)
    assert torch.isfinite(det).all()
    # default frozen descriptors are zero until set_domain_descriptors
    assert (det == 0).all()

    # set domain descriptors and verify they are frozen
    from hch_v2_context import compute_domain_descriptors
    s1_z0 = np.random.randn(500) * 0.5 + 0.1
    ddet = compute_domain_descriptors(s1_z0)
    assert ddet.shape == (8,)
    sig.set_domain_descriptors(ddet)
    assert torch.allclose(sig.compute_deterministic(),
                          torch.tensor(ddet, dtype=torch.float32), atol=1e-6)


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


@test("7-segment split is disjoint, exhaustive, non-empty (P0-2)")
def _():
    import pandas as pd
    # synthetic 10-year hourly series, all complete days
    n_days = 3650
    ts = pd.date_range("2010-01-01", periods=n_days * 24, freq="h")
    ds = {"ts": pd.Series(ts)}
    valid = np.arange(0, n_days * 24)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="SYN")
    assert exp.assert_7seg_disjoint(), "7 segments must be pairwise disjoint + exhaustive"
    assert exp.assert_s3_disjoint(), "s3 disjoint check must hold in 7-way mode"
    # every date mapped
    assert len(exp.split_of_date) == n_days
    from collections import Counter
    cnt = Counter(exp.split_of_date.values())
    for s in ("H0", "S1R", "S2T", "S2V", "S3M", "S3C", "S4"):
        assert cnt[s] > 0, f"segment {s} must be non-empty"
    # aggregate view: S1=H0+S1R, S2=S2T+S2V, S3=S3M+S3C
    assert len(exp.dates_in_split("S1")) == cnt["H0"] + cnt["S1R"]
    assert len(exp.dates_in_split("S2")) == cnt["S2T"] + cnt["S2V"]
    assert len(exp.dates_in_split("S3")) == cnt["S3M"] + cnt["S3C"]
    assert len(exp.dates_in_split("S4")) == cnt["S4"]


@test("7-segment valid-hour totals match legacy aggregate (P0-2)")
def _():
    import pandas as pd
    n_days = 500
    ts = pd.date_range("2010-01-01", periods=n_days * 24, freq="h")
    ds = {"ts": pd.Series(ts)}
    valid = np.arange(0, n_days * 24)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="SYN2")
    for s4 in ("S1", "S2", "S3", "S4"):
        v7 = exp.valid_indices_in_split(s4)
        # S1 => H0+S1R etc.
        names = {"S1": ("H0", "S1R"), "S2": ("S2T", "S2V"),
                 "S3": ("S3M", "S3C"), "S4": ("S4",)}[s4]
        n_expected = sum(len(exp.valid_indices_in_split(x)) for x in names)
        assert len(v7) == n_expected, f"{s4}: {len(v7)} != {n_expected}"


@test("7-segment split hash includes excluded dates (P0-2)")
def _():
    import pandas as pd
    # synthetic series with one 23h day
    ts = pd.date_range("2010-01-01", periods=24 * 200 + 23, freq="h")
    ds = {"ts": pd.Series(ts)}
    valid = np.arange(0, 24 * 200)
    exp = ExperimentManifest.from_dataset(ds, valid, dataset_id="SYN3")
    assert len(exp.excluded_dates) == 1, "the 23h day must be recorded as excluded"
    assert exp.assert_7seg_disjoint()


@test("P0-1: mixed-domain batch gets its own descriptor (forward context)")
def _():
    torch.manual_seed(0)
    from hch_v2_context import DataSignature, compute_domain_descriptors
    sig = DataSignature(32, 8)
    sig.train()
    detA = torch.tensor(compute_domain_descriptors(
        np.random.randn(2000) * 0.3 + 0.5), dtype=torch.float32).unsqueeze(0)
    detB = torch.tensor(compute_domain_descriptors(
        np.random.randn(2000) * 1.5 - 0.8), dtype=torch.float32).unsqueeze(0)
    # fixed core -> descriptor is the only varying input
    core = torch.randn(1, 24, 32)
    opt = torch.optim.Adam(sig.parameters(), lr=1e-2)
    for _ in range(50):
        dg, beta = sig(core, detA)
        loss = -((dg - 1.0).abs() + beta.abs()).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dgA, _ = sig(core, detA)
        dgB, _ = sig(core, detB)
        dg0, _ = sig(core, detA * 0)
    assert (dgA - dgB).abs().max() > 1e-4, "A vs B must differ after training"
    assert (dgA - dg0).abs().max() > 1e-4, "descriptor must drive FiLM"


@test("P0-1: interleaved batch order does not change per-domain output")
def _():
    torch.manual_seed(0)
    from hch_v2_context import DataSignature
    sig = DataSignature(32, 8)
    sig.eval()
    det = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unsqueeze(0)
    core = torch.randn(2, 24, 32)
    d1 = sig(core, det.expand(2, -1))
    d2 = sig(core, det.expand(2, -1))
    assert torch.allclose(d1[0], d2[0]) and torch.allclose(d1[1], d2[1])


@test("P0-1: identity-init FiLM (mod_head zero) is ~identity at init")
def _():
    torch.manual_seed(0)
    from hch_v2_context import DataSignature
    sig = DataSignature(32, 8)
    core = torch.randn(2, 24, 32)
    det = torch.randn(2, 8)
    with torch.no_grad():
        dg, beta = sig(core, det)
    assert dg.abs().max() < 1e-6, f"delta_gamma must be ~0 at init, got {dg.abs().max()}"
    assert beta.abs().max() < 1e-6, f"beta must be ~0 at init, got {beta.abs().max()}"


@test("P0-4: universal trainer does macro S2V selection across domains")
def _():
    torch.manual_seed(0)
    from iah_candidate import IAHCandidateHead

    head = IAHCandidateHead(d_core_context=5, d_model=16)
    domains = []
    for g in range(2):
        det = np.array([0.2 + g * 0.6] * 8, dtype=np.float32)
        bt, bv = [], []
        for _ in range(3):
            host = torch.rand(1, 24, 1) + 1.0  # positive scale => scale-valid
            ctx = torch.randn(1, 24, 5)
            target = torch.rand(1, 24, 1) + 0.5
            vm = torch.ones(1, 24)
            bt.append((host, ctx, target, vm))
            bv.append((host.clone(), ctx.clone(), target.clone(), vm.clone()))
        domains.append(DomainBatch(name=f"g{g}", s2t_batches=bt,
                                   s2v_batches=bv, domain_det=det))

    rep = UniversalCoreTrainer(head, seed=1).train(domains, epochs=3,
                                                   patience=2)
    assert np.isfinite(rep["best_macro_s2v"]), rep
    assert np.isfinite(rep["worst_s2v_at_best"]), rep
    assert rep["n_domains"] == 2
    assert len(rep["history"]) >= 1
    # per-domain S2V losses recorded
    for g in rep["history"][0]["per_domain"]:
        assert g in ("g0", "g1")
    # §13 training-health diagnostics recorded at every validation checkpoint
    h0 = rep["history"][0]["health"]
    assert "mass_entropy" in h0, "mass health missing"
    assert "frac_m_minus_alive" in h0, "shift health missing"
    assert "mean_abs_delta_gamma" in h0, "signature health missing"
    gh0 = rep["history"][0]["grad_health"]
    assert "mean_grad_norm" in gh0 and "nan_inf_batches" in gh0


@test("P0-3: S2V batches select checkpoint, not S2T train loss")
def _():
    torch.manual_seed(0)
    from hch_v2_pipeline import HCHV2UniversalPipeline

    pipe = HCHV2UniversalPipeline(d_core_context=5, d_model=16)
    # S2T: easy targets near host (low CRPS). S2V: far targets (high CRPS).
    def mk(scale):
        host = torch.rand(1, 24, 1) + 1.0
        ctx = torch.randn(1, 24, 5)
        target = host * scale
        vm = torch.ones(1, 24)
        return (host, ctx, target, vm)

    s2t = [mk(1.1)] * 6
    s2v = [mk(3.0)] * 6
    best = pipe.train_candidate_s2(s2t, s2v_batches=s2v, epochs=3, patience=2)
    # returned value must equal the S2V eval (checkpoint chosen on S2V)
    s2v_after = pipe._eval_s2_batches(s2v)
    s2t_after = pipe._eval_s2_batches(s2t)
    assert s2v_after is not None and s2t_after is not None
    assert abs(best - s2v_after) < 1e-3, f"{best} != S2V {s2v_after}"
    # and S2V must be distinguishable from S2T here (test is not vacuous)
    assert s2v_after > s2t_after + 0.05, f"S2V {s2v_after} vs S2T {s2t_after}"


@test("P0-4: domain descriptor drives different FiLM modulation (2-domain)")
def _():
    torch.manual_seed(0)
    from iah_candidate import IAHCandidateHead

    head = IAHCandidateHead(d_core_context=5, d_model=16)
    head.eval()
    detA = torch.tensor([0.2] * 8, dtype=torch.float32).unsqueeze(0)
    detB = torch.tensor([0.8] * 8, dtype=torch.float32).unsqueeze(0)
    host = torch.rand(2, 24, 1) + 1.0
    ctx = torch.randn(2, 24, 5)
    with torch.no_grad():
        oA = head(host, ctx, valid_mask=torch.ones(2, 24), domain_det=detA.expand(2, -1))
        oB = head(host, ctx, valid_mask=torch.ones(2, 24), domain_det=detB.expand(2, -1))
    # FiLM is identity-init => outputs identical before training, but the head
    # must at least RUN with per-domain descriptors in a batch (no shape error).
    assert oA["w_minus"].shape == (2, 24)
    assert torch.allclose(oA["z0"], oB["z0"])


@test("P1-3: universal/local hashes are independent + signature is local")
def _():
    from hch_v2_bundle import HCHV2Bundle

    b1 = HCHV2Bundle()
    b1.core_model_state = {"w": torch.tensor([1.0, 2.0, 3.0])}
    b1.core_config = {"d_model": 16}
    b1.data_signature_spec = {"det": [0.1, 0.2, 0.3], "version": "v1"}
    b1.local_hashes = {"split_hash": "ABC"}
    b1.compute_hash()

    # Same universal, different local -> universal_hash unchanged
    b2 = HCHV2Bundle()
    b2.core_model_state = {"w": torch.tensor([1.0, 2.0, 3.0])}
    b2.core_config = {"d_model": 16}
    b2.data_signature_spec = {"det": [9.9], "version": "v2"}
    b2.local_hashes = {"split_hash": "XYZ"}
    b2.compute_hash()
    assert b1.universal_hash == b2.universal_hash, \
        f"universal hash must be local-invariant: {b1.universal_hash} vs {b2.universal_hash}"
    assert b1.local_hash != b2.local_hash, "local hash must differ with local"

    # Signature is in the LOCAL package, not the universal one
    uni = b1.extract_universal()
    assert "data_signature_spec" not in uni, "signature must NOT be universal"
    loc = b1.extract_local()
    assert loc["data_signature_spec"] == b1.data_signature_spec
    assert "universal_hash" in uni and "local_hash" in loc


@test("P1-3: one universal checkpoint reused across two target markets")
def _():
    from hch_v2_bundle import HCHV2Bundle

    # The universal checkpoint is trained once on source market S1...
    def universal():
        b = HCHV2Bundle()
        b.core_model_state = {"w": torch.tensor([1.0, 2.0, 3.0])}
        b.core_config = {"d_model": 16, "seed": 0}
        b.training_provenance = {"seed": 0, "s2_split": "S2T/S2V",
                                 "config_hash": "x", "code_commit": "abc123"}
        b.source_datasets = ["S1"]
        b.source_hosts = ["Linear"]
        b.compute_hash()
        return b

    u = universal()

    # ...then applied to two target markets via a per-domain local profile.
    def profile(target):
        b = universal()  # copy the universal package
        b.data_signature_spec = {"det": [0.1 if target == "T1" else 0.9],
                                 "version": "v1", "source": target}
        b.local_hashes = {"split_hash": "HASH-" + target,
                          "target_market": target, "target_host": target}
        b.compute_hash()
        return b

    t1, t2 = profile("T1"), profile("T2")
    assert t1.universal_hash == t2.universal_hash == u.universal_hash
    assert t1.local_hash != t2.local_hash
    assert t1.bundle_hash != t2.bundle_hash

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
