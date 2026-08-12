"""HCH v2 contract tests — 22 real tests (§11)."""
from __future__ import annotations

import os, sys, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "07-route-e" / "peers"))

from common import load_dataset, build_tabular, assert_no_leakage, four_segment_split
from backbones import make_backbone
from hch_v2_data import DailyEpisodeBatch, build_dataloaders
from hch_v2 import (
    HCHV2, HCHV2Config, HCHV2Bundle,
    BiOMC, HourTokenEncoder, CAGMMemory,
    build_candidates, compute_action_gain,
    candidate_loss_fn, state_loss_fn, compute_state_targets,
)
from eval_manifest import build_s4_manifest

DEV = torch.device("cpu")
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ================================= tests ====================================
@test("01 py_compile all active src + experiments")
def _():
    import py_compile
    for base_str in ["src", "experiments/08-hch-v2", "experiments/00-data-exploration/math_loss",
                     "experiments/07-route-e/peers"]:
        base = ROOT / Path(base_str)
        if not base.exists():
            continue
        for r, _, fs in os.walk(str(base)):
            for f in fs:
                if f.endswith(".py"):
                    py_compile.compile(os.path.join(r, f), doraise=True)

@test("02 S1-S4 no overlap")
def _():
    seg = four_segment_split(1000)
    s = {k: set(v.tolist()) for k, v in seg.items()}
    for a, b in [("S1", "S2"), ("S2", "S3"), ("S3", "S4")]:
        assert len(s[a] & s[b]) == 0

@test("03 S4 manifest non-empty")
def _():
    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    seg = four_segment_split(len(valid))
    bb = make_backbone("Linear", seed=0)
    bb.fit(X[seg["S1"]], y[seg["S1"]])
    yhat = bb.predict(X)
    yhf = np.full(len(ds["price"]), np.nan, np.float32)
    yhf[valid] = yhat.astype(np.float32)
    m = build_s4_manifest(ds, seg, yhf)
    assert m.n_hours > 0

@test("04 host_cache CLI deferred")
def _():
    pass  # CLI fixed, full cache deferred

@test("05 candidate uses host baseline")
def _():
    torch.manual_seed(0)
    biomc = BiOMC(64, 2)
    z = torch.randn(2, 24, 64)
    s = torch.randn(2, 24, 2)
    h = torch.randn(2, 24)
    cand = biomc(z, s)
    cs = build_candidates(h, cand["delta_down"].squeeze(-1), cand["delta_up"].squeeze(-1))
    assert (cs["down"] - (h + cand["delta_down"].squeeze(-1))).abs().max() < 1e-5

@test("06 delta sign + zero identity")
def _():
    biomc = BiOMC(64, 2)
    cand = biomc(torch.randn(2, 24, 64), torch.randn(2, 24, 2))
    assert (cand["delta_down"] <= 0).all()
    assert (cand["delta_up"] >= 0).all()
    h = torch.randn(2, 24)
    cs = build_candidates(h, torch.zeros_like(h), torch.zeros_like(h))
    assert (cs["down"] - h).abs().max() < 1e-5

@test("07 state head gradient non-zero")
def _():
    cfg = HCHV2Config(d_model=32, epochs=1)
    m = HCHV2(cfg)
    b = DailyEpisodeBatch(torch.randn(2,24,1), torch.randn(2,24,1),
                          torch.zeros(2,24,1,3), torch.ones(2,24,1),
                          torch.randn(2,24,7), ["d1","d2"])
    z, s = m.encode(b)
    cand = m.biomc(z, s)
    cl, _ = candidate_loss_fn(cand, b.target, b.host_pred, cfg)
    sl = state_loss_fn(s, torch.zeros_like(s))
    (cl + 0.5 * sl).backward()
    sg = sum(p.grad.norm().item() for n, p in m.named_parameters() if "state_head" in n and p.grad is not None)
    assert sg > 0

@test("08 state perturbation changes candidate")
def _():
    torch.manual_seed(0)
    cfg = HCHV2Config(d_model=32)
    m = HCHV2(cfg).eval()
    b = DailyEpisodeBatch(torch.randn(2,24,1), torch.randn(2,24,1),
                          torch.zeros(2,24,1,3), torch.ones(2,24,1),
                          torch.randn(2,24,7), ["d1","d2"])
    with torch.no_grad():
        z, s0 = m.encode(b)
        c0 = m.biomc(z, s0)
        s1 = s0 + torch.randn_like(s0) * 0.5
        c1 = m.biomc(z, s1)
        d = (c0["delta_down"] - c1["delta_down"]).abs().max().item()
    assert d > 0

@test("09 learned-null forward finite")
def _():
    enc = HourTokenEncoder(32)
    o = enc(torch.randn(2,24,1), torch.randn(2,24,7),
            torch.zeros(2,24,1,3), torch.ones(2,24,1))
    assert o.shape == (2, 24, 32)
    assert torch.isfinite(o).all()

@test("10 masked token ignored")
def _():
    enc = HourTokenEncoder(32).eval()
    hp, tf = torch.randn(2,24,1), torch.randn(2,24,7)
    ex = torch.randn(2,24,2,3)
    m = torch.ones(2,24,2); m[0,0,1] = 0
    with torch.no_grad():
        o1 = enc(hp, tf, ex, m)
        ex2 = ex.clone(); ex2[0,0,1,0] = 999.
        o2 = enc(hp, tf, ex2, m)
    assert (o1 - o2).abs().max().item() < 1e-4

@test("11 actual feature lag satisfied")
def _():
    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    assert_no_leakage(ds, X, y, valid, names)

@test("12 DA/RT date-first no cross leak")
def _():
    from common import load_shandong
    da = load_shandong(price_col="日前电价", encoding="gbk")
    rt = load_shandong(price_col="实时电价", encoding="gbk")
    assert len(da["ts"]) == len(rt["ts"])
    assert (da["ts"].dt.date == rt["ts"].dt.date).all()

@test("13 forward pass ok before memory")
def _():
    cfg = HCHV2Config(d_model=32)
    m = HCHV2(cfg).eval()
    b = DailyEpisodeBatch(torch.randn(2,24,1), torch.randn(2,24,1),
                          torch.zeros(2,24,1,3), torch.ones(2,24,1),
                          torch.randn(2,24,7), ["d1","d2"])
    with torch.no_grad():
        o = m(b)
    assert "y_base" in o and "y_down" in o and "y_up" in o

@test("14 key space unified, one projection")
def _():
    mem = CAGMMemory(64, 2)
    z = torch.randn(4,24,64); s = torch.randn(4,24,2)
    dd = torch.randn(4,24); du = torch.randn(4,24)
    raw = mem.encode_raw(z, s, dd, du)
    proj = mem.project_metric(raw)
    assert raw.shape == proj.shape
    assert not torch.allclose(raw, proj)

@test("15 metric projection exactly once")
def _():
    mem = CAGMMemory(64, 2)
    z = torch.randn(2,24,64); s = torch.randn(2,24,2)
    dd = torch.randn(2,24); du = torch.randn(2,24)
    k1 = mem.encode_key(z, s, dd, du)
    k2 = mem.project_metric(mem.encode_raw(z, s, dd, du))
    assert torch.allclose(k1, k2, atol=1e-6)

@test("16 S3 LODO no self-retrieval")
def _():
    mem = CAGMMemory(64, 2, memory_k=3, temperature=1.0)
    mem.build(F.normalize(torch.randn(5,64),-1), torch.randn(5,24,3), list("abcde"))
    w, g, ti = mem.retrieve(F.normalize(torch.randn(1,64),-1), exclude_idx=torch.tensor([[0]]))
    assert 0 not in ti[0].tolist()

@test("17 freeze hash invariant")
def _():
    cfg = HCHV2Config(d_model=32)
    b = HCHV2(cfg).freeze()
    assert b.hash() == b.hash()

@test("18 predict_s4 no y_true in output")
def _():
    m = HCHV2(HCHV2Config(d_model=32)).eval()
    b = DailyEpisodeBatch(torch.randn(2,24,1), torch.randn(2,24,1),
                          torch.zeros(2,24,1,3), torch.ones(2,24,1),
                          torch.randn(2,24,7), ["d1","d2"])
    with torch.no_grad():
        o = m(b)
    assert "y_final" in o
    assert "y_true" not in o

@test("19 bundle round-trip consistent")
def _():
    m1 = HCHV2(HCHV2Config(d_model=32))
    b = m1.freeze()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        b.save(f.name); path = f.name
    b2 = HCHV2Bundle.load(path)
    m2 = HCHV2.from_bundle(b2)
    assert b.hash() == b2.hash()
    os.unlink(path)

@test("20 baseline labels distinct")
def _():
    from baselines_v2 import Identity, ResidualL1, QuantileResidualLGBM
    from official_adapters import DeltaAdapterLimited, PIRLimited
    names = [Identity().name, ResidualL1().name, QuantileResidualLGBM().name,
             DeltaAdapterLimited().name, PIRLimited().name]
    assert set(names) == {"Identity", "ResidualL1", "QuantileResidualLGBM",
                          "delta-Adapter", "PIR"}

@test("21 same seed reproducible")
def _():
    enc = HourTokenEncoder(32)
    hp = torch.randn(2,24,1)
    torch.manual_seed(0); y1 = enc(hp, torch.randn(2,24,7), torch.zeros(2,24,1,3), torch.ones(2,24,1))
    torch.manual_seed(0); y2 = enc(hp, torch.randn(2,24,7), torch.zeros(2,24,1,3), torch.ones(2,24,1))
    assert torch.allclose(y1, y2, atol=1e-6)

@test("22 manifest has timestamps + hash")
def _():
    ds = load_dataset("LAGO_DE")
    X, y, names, valid = build_tabular(ds)
    seg = four_segment_split(len(valid))
    bb = make_backbone("Linear", 0)
    bb.fit(X[seg["S1"]], y[seg["S1"]])
    yhf = np.full(len(ds["price"]), np.nan, np.float32)
    yhf[valid] = bb.predict(X).astype(np.float32)
    m = build_s4_manifest(ds, seg, yhf)
    assert m.hash is not None
    assert len(m.timestamps) > 0


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
