"""Adversarial tests for the HSA Shape closure.

Run:  python experiments/current/hch_horizon_aligned_state_residual/test_closure.py
      python experiments/current/hch_horizon_aligned_state_residual/test_closure.py --smoke

Fast tests exercise the mechanism-level claims the protocol gates depend on.
`--smoke` additionally replays one real (market, host, seed) cell end-to-end
through the closure pipeline and checks the frozen S1 replay tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.current.hch_horizon_aligned_state_residual.run_closure import (
    LEGAL, MARKETS, N_ROLES, REPLAY_TOL, ROLES, HSAResidual, calibrate_hsa_oof,
    calibrate_oof, fit_beta, fit_beta_final, hsa_oof, load_role_schema, new_beta_model,
    predict_beta, role_table, standardized_state,
)
from experiments.current.hch_unified_da_shape_upgrade.run_canary import read_state
from src.mvp.hch_minimal_repair.crossfit import chronological_blocks
from utils.timing import parameter_count

PASSED: list[str] = []


def check(name, cond):
    if not bool(cond):
        raise AssertionError(f"FAILED: {name}")
    PASSED.append(name)


def t01_beta_model_shape():
    m = HSAResidual()
    check("T01 exactly five trainable parameters", parameter_count(m) == N_ROLES)
    check("T01 beta initialised to exactly zero", float(m.beta.detach().abs().max()) == 0.0)
    check("T01 no hidden layer or submodule", set(type(x) for x in m.modules()) == {HSAResidual})
    check("T01 no reset_parameters hook that could wipe beta",
          not any(hasattr(x, "reset_parameters") for x in m.modules()))


def t02_zero_init_replays_s1_exactly():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(64, 24)).astype(np.float32)
    u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    R = rng.normal(size=(64, 24, N_ROLES)).astype(np.float32)
    out = predict_beta(np.zeros(N_ROLES), u, R)
    check("T02 beta=0 replays S1 directions within tolerance",
          float(np.max(np.abs(out - u))) <= REPLAY_TOL)
    check("T02 beta=0 output is still unit norm", np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5))
    # A nonzero beta must actually move the direction, else the gate is vacuous.
    moved = predict_beta(np.array([0.5, 0, 0, 0, 0], float), u, R)
    check("T02 nonzero beta changes the direction", float(np.max(np.abs(moved - u))) > 1e-3)


def t03_role_table_pooling_and_structural_zero():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5, 24, 3)).astype(np.float32)
    mapping = ((0, 1), (2,), (), (), ())
    R = role_table(z, mapping)
    check("T03 role table has 24 x 5 shape", R.shape == (5, 24, 5))
    check("T03 mean pooling within role", np.allclose(R[:, :, 0], z[:, :, [0, 1]].mean(2), atol=1e-6))
    check("T03 single-channel role equals that channel", np.allclose(R[:, :, 1], z[:, :, 2], atol=1e-6))
    check("T03 structurally absent roles are exactly zero", np.all(R[:, :, 2:] == 0))
    check("T03 pooling is order invariant within a role",
          np.allclose(role_table(z, ((1, 0), (2,), (), (), ()))[:, :, 0], R[:, :, 0], atol=1e-6))


def t04_schema_is_primary_and_legal_only():
    mapping, manifest = load_role_schema()
    check("T04 five roles per market mapping", all(len(mapping[m]) == N_ROLES for m in MARKETS))
    names, idx_ok = set(), True
    for m in MARKETS:
        for role in mapping[m]:
            for i in role:
                idx_ok &= 0 <= i < len(LEGAL[m])
                names.add(LEGAL[m][i])
    check("T04 every mapped index is in range for its market", idx_ok)
    check("T04 every mapped channel is a registered legal column",
          names <= {c for m in MARKETS for c in LEGAL[m]})
    check("T04 GANSU report-only aggregate excluded", "新能源总加预测值" not in names)
    sm = manifest[manifest.semantic_role.eq("SUPPLY_MARGIN_FC")]
    check("T04 SUPPLY_MARGIN_FC structurally absent in every market",
          len(sm) == len(MARKETS) and bool(sm.structurally_absent.all()))
    check("T04 only SUPPLY_MARGIN_FC is absent for all three markets",
          manifest.groupby("semantic_role").structurally_absent.all().pipe(lambda s: s[s].index.tolist()) == ["SUPPLY_MARGIN_FC"])
    # GANSU must contribute exactly six primary channels across four roles.
    g = manifest[(manifest.market == "GANSU_DA") & manifest.n_channels_in_role.gt(0)]
    check("T04 GANSU uses six primary channels", int(g.n_channels_in_role.sum()) == 6)
    check("T04 GANSU covers four roles", sorted(g.semantic_role) == ["DEMAND_FC", "INTERCHANGE_FC", "MUST_RUN_FC", "RENEWABLE_FC"])


def t05_real_state_reads_zero_forbidden():
    from experiments.current.hch_anchored_compact_amplitude.run_closure import load_frozen_cache
    c, _ = load_frozen_cache("GANSU_DA", "PatchTST")
    origins = np.r_[c.subset("DIAG_FIT").timestamp, c.subset("DIAG_EVAL").timestamp]
    z, a = read_state("GANSU_DA", origins)
    check("T05 state reader reads no target-day DA price", a["target_day_DA_input_reads"] == 0)
    check("T05 state reader reads no realized RT price", a["realized_RT_input_reads"] == 0)
    check("T05 state reader reads no S3/S4 partition", a["S3_S4_reads"] == 0)
    check("T05 state tensor is finite", bool(np.isfinite(z).all()))
    check("T05 state tensor channel count matches LEGAL", z.shape[2] == len(LEGAL["GANSU_DA"]))


def t06_beta_oof_chronology():
    n = 400
    rng = np.random.default_rng(2)
    u = rng.normal(size=(n, 24)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    state = rng.normal(size=(n, 24, len(LEGAL["LAGO_DE"]))).astype(np.float32)
    e = rng.normal(size=(n, 24)).astype(np.float32)
    av = np.ones(n, bool)
    mapping = ((0,), (1,), (), (), ())
    out = hsa_oof(u, state, e, av, mapping, 7)
    b = out["blocks"]
    check("T06 three beta folds for B2..B4", len(out["folds"]) == 3)
    check("T06 every beta fold self-excluded", all(r["self_excluded"] for r in out["records"]))
    for r in out["records"]:
        k = int(r["block"][1:])
        tr_max = max(int(b[f"B{j}"].max()) for j in range(1, k))
        check(f"T06 block {r['block']} beta train rows strictly precede proposals",
              r["beta_train_max_index"] < r["proposal_min_index"] and r["beta_train_max_index"] == tr_max)
    filled = np.isfinite(out["u"]).all(axis=1)
    check("T06 HSA OOF defined exactly on B2..B4", np.array_equal(np.flatnonzero(filled),
                                                                  np.sort(np.concatenate([b[f"B{k}"] for k in range(2, 5)]))))
    check("T06 B0/B1 have no HSA OOF direction", not np.isfinite(out["u"][b["B0"]]).any() and not np.isfinite(out["u"][b["B1"]]).any())


def t07_alpha_chronology_and_rowset():
    n = 400
    rng = np.random.default_rng(3)
    u = rng.normal(size=(n, 24)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    e = (2.0 * u + 0.1 * rng.normal(size=(n, 24))).astype(np.float32)
    av = np.ones(n, bool)
    b = chronological_blocks(n)
    alpha, d, complete, allrows, rec = calibrate_oof(e, u, av, b)
    check("T07 alpha folds calibrated strictly before proposals",
          all(r["calibration_max_index"] < r["proposal_min_index"] for r in rec))
    check("T07 three alpha folds", len(rec) == 3)
    check("T07 complete row set is B2..B4 available",
          np.array_equal(complete, np.concatenate([b[f"B{k}"] for k in range(2, 5)])))
    check("T07 allrows row set is B1..B4 available",
          np.array_equal(allrows, np.concatenate([b[f"B{k}"] for k in range(1, 5)])))
    check("T07 alpha recovers the planted amplitude", abs(alpha - 2.0) < 0.05)
    check("T07 HSA alpha row set is strictly inside the S1 alpha row set",
          set(complete.tolist()) < set(allrows.tolist()))


def t08_beta_actually_trains():
    rng = np.random.default_rng(4)
    n = 600
    R = rng.normal(size=(n, 24, N_ROLES)).astype(np.float32)
    mapping = ((0,), (1,), (), (), ())
    beta_true = np.array([1.2, -0.8, 0.6, 0.0, 0.0])
    u_s1 = rng.normal(size=(n, 24))
    u_s1 /= np.linalg.norm(u_s1, axis=1, keepdims=True)
    # Plant an exact attainable optimum of the HSA objective at beta_true.
    z = u_s1 + R @ beta_true
    target = z / np.linalg.norm(z, axis=1, keepdims=True)
    check("T08 planted optimum is not reachable at beta=0",
          float(np.mean(np.sum((u_s1 - target) ** 2, 1))) > 1e-3)
    loss0 = float(np.mean(np.sum((u_s1 - target) ** 2, 1)))
    beta, info = fit_beta(u_s1, R, target, np.ones(n, bool), 7)
    check("T08 beta training cuts the Shape objective by >95%", info["objective"] < 0.05 * loss0)
    check("T08 fitted beta is nontrivial", float(np.abs(beta).max()) > 1e-2)
    # The fixed 1000-step budget under-converges in magnitude, so assert direction only.
    cos = float(beta @ beta_true / (np.linalg.norm(beta) * np.linalg.norm(beta_true)))
    check("T08 fitted beta recovers the planted direction", cos > 0.99)
    beta2, info2 = fit_beta(u_s1, R, target, np.ones(n, bool), 7)
    check("T08 beta fit is deterministic under a fixed seed", np.array_equal(beta, beta2))
    # beta starts at exactly zero with no stochastic layer, so the RNG seed is a
    # no-op for beta: identical inputs must give identical coefficients. Seed
    # variation in the closure therefore comes only from the frozen S1 OOF
    # directions differing across seeds, not from beta's own optimisation.
    beta3, _ = fit_beta(u_s1, R, target, np.ones(n, bool), 8)
    check("T08 beta fit is seed-invariant for identical inputs", np.array_equal(beta, beta3))


def t09_final_beta_uses_only_oof_rows():
    n = 300
    rng = np.random.default_rng(5)
    u = rng.normal(size=(n, 24)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    state = rng.normal(size=(n, 24, 2)).astype(np.float32)
    e = rng.normal(size=(n, 24)).astype(np.float32)
    av = np.ones(n, bool)
    mapping = ((0,), (1,), (), (), ())
    b = chronological_blocks(n)
    allrows = np.concatenate([b[f"B{k}"] for k in range(1, 5)])
    beta, info = fit_beta_final(u, state, e, av, mapping, allrows, 7)
    check("T09 final beta trains only on legal OOF rows", 0 < info["n_train_rows"] <= len(allrows))
    check("T09 final beta has five finite coefficients", len(beta) == N_ROLES and np.isfinite(beta).all())
    check("T09 final beta verified zero-init replay", info["zero_init_replay_max_abs_diff"] <= REPLAY_TOL)


def t10_standardization_is_train_only():
    rng = np.random.default_rng(6)
    train = (rng.normal(size=(40, 24, 2)) * 5 + 100).astype(np.float32)
    test = rng.normal(size=(10, 24, 2)).astype(np.float32)
    a, b = standardized_state(train, test)
    check("T10 train standardisation uses train statistics", np.allclose(a.reshape(-1, 48).mean(0), 0, atol=1e-5))
    check("T10 test standardisation does not use test statistics",
          not np.allclose(b.reshape(-1, 48).mean(0), 0, atol=1e-3))
    # Swapping in extra non-primary columns must not change primary-role values.
    full = np.concatenate([train, rng.normal(size=(40, 24, 1)).astype(np.float32)], axis=2)
    af, bf = standardized_state(full, np.concatenate([test, rng.normal(size=(10, 24, 1)).astype(np.float32)], axis=2))
    check("T10 per-coordinate standardisation is independent across channels",
          np.allclose(af[:, :, :2], a, atol=1e-6) and np.allclose(bf[:, :, :2], b, atol=1e-6))


def smoke_one_cell():
    """End-to-end replay of one real cell through the closure pipeline."""
    from experiments.current.hch_anchored_compact_amplitude.run_closure import (
        SHAPE_EVIDENCE, fit_shape_outputs, load_frozen_cache, pooled_alpha, shape_oof,
    )
    from experiments.current.hch_unified_da_shape_upgrade.run_canary import make_inputs

    m, h, seed = "GANSU_DA", "PatchTST", 7
    cache, _ = load_frozen_cache(m, h)
    fit = cache.subset("DIAG_FIT"); ev = cache.subset("DIAG_EVAL")
    fi, floor = make_inputs(fit, cache); ei, _ = make_inputs(ev, cache, floor)
    origins = np.r_[fit.timestamp, ev.timestamp]
    z, _ = read_state(m, origins); nf = len(fit.timestamp)
    sf, se = z[:nf], z[nf:]
    ef = (fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]) / np.where(np.isfinite(fi.scale), fi.scale, 1)[:, None]
    oof = shape_oof(fi.x, sf, ef, fi.repair_available, seed)
    _, u_eval, _, info = fit_shape_outputs(fi.x, sf, ef, fi.repair_available, ei.x, se, seed)
    check("smoke S1 OOF directions are unit norm on B1..B4",
          np.allclose(np.linalg.norm(oof["u"][np.isfinite(oof["u"]).all(1)], axis=1), 1.0, atol=1e-5))
    mapping, _ = load_role_schema()
    hsa = hsa_oof(oof["u"], sf, ef, fi.repair_available, mapping[m], seed)
    b = hsa["blocks"]
    left = np.isfinite(hsa["u"]).all(1)
    complete_h = np.concatenate([b[f"B{k}"][fi.repair_available[b[f"B{k}"]]] for k in range(2, 5)])
    check("smoke HSA OOF fills exactly the B2..B4 available rows",
          np.array_equal(np.flatnonzero(left), np.sort(complete_h)))
    check("smoke HSA OOF directions are unit norm",
          np.allclose(np.linalg.norm(hsa["u"][left], axis=1), 1.0, atol=1e-5))
    has_dir = np.isfinite(hsa["u"]).all(1)
    alpha_hsa, _, alpha_rows, hrec = calibrate_hsa_oof(ef, hsa["u"], fi.repair_available, b, has_dir)
    check("smoke alpha_HSA is positive and finite", np.isfinite(alpha_hsa) and alpha_hsa > 0)
    check("smoke alpha_HSA row set carries only finite HSA directions",
          np.isfinite(hsa["u"][alpha_rows]).all())
    check("smoke alpha_HSA chronology starts at B3 and self-excludes",
          [r["block"] for r in hrec] == ["B3", "B4"] and all(r["self_excluded"] for r in hrec))
    check("smoke alpha_HSA row set excludes B1, which has no HSA direction",
          np.intersect1d(alpha_rows, b["B1"]).size == 0)
    # The S1 alpha row set spans B1..B4, but B1 carries no HSA direction. Reusing
    # it would only appear to work because `weighted_median_scalar` filters rows by
    # `np.abs(u) > 1e-12` and `NaN > 1e-12` is False, so direction-less rows are
    # silently dropped. That makes legality incidental rather than explicit, which is
    # why calibrate_hsa_oof narrows the row set instead of relying on the NaN filter.
    naive_rows = np.concatenate([b[f"B{k}"][fi.repair_available[b[f"B{k}"]]] for k in range(1, 5)])
    dropped = np.setdiff1d(naive_rows, alpha_rows)
    check("smoke the S1 alpha row set really does carry HSA-direction-less rows",
          dropped.size > 0 and not np.isfinite(hsa["u"][dropped]).any())
    check("smoke the NaN weight filter is what hides those rows from pooled_alpha",
          abs(pooled_alpha(ef, hsa["u"], naive_rows) - alpha_hsa) < 1e-12)
    allrows = np.concatenate([b[f"B{k}"][fi.repair_available[b[f"B{k}"]]] for k in range(1, 5)])
    beta, finfo = fit_beta_final(oof["u"], sf, ef, fi.repair_available, mapping[m], allrows, seed)
    check("smoke final beta is five finite coefficients", len(beta) == 5 and np.isfinite(beta).all())
    zf, ze = standardized_state(sf, se)
    u_hsa = predict_beta(beta, u_eval, role_table(ze, mapping[m]))
    check("smoke HSA eval directions are unit norm", np.allclose(np.linalg.norm(u_hsa, axis=1), 1.0, atol=1e-5))
    old = pd.read_csv(SHAPE_EVIDENCE / "metrics_by_seed.csv")
    ref = old[(old.method.eq("S1_DailyPatch_GRU32")) & (old.market == m) & (old.host == h) & (old.seed == seed)].iloc[0]
    check("smoke S1 final parameter count matches frozen evidence",
          int(info["shape_parameter_count"]) == int(ref.parameter_count))
    check("smoke S1 alpha matches frozen evidence", abs(pooled_alpha(ef, oof["u"], allrows) - float(ref.alpha)) < 1e-9)
    print(f"  smoke beta={np.round(beta, 5).tolist()} alpha_HSA={alpha_hsa:.6f} "
          f"S1_params={info['shape_parameter_count']} HSA_params={info['shape_parameter_count'] + 5}")


def t11_single_thread_invariant():
    # src/backbones/backbones.py calls torch.set_num_threads(4) at import time.
    # run_closure must re-pin to 1 after its imports, or S1 trains with 4 BLAS
    # threads and the frozen replay breaks by ~1e-3 in alpha.
    check("T11 importing the runner leaves torch single-threaded", torch.get_num_threads() == 1)
    torch.set_num_threads(4)
    import importlib
    import experiments.current.hch_horizon_aligned_state_residual.run_closure as rc
    importlib.reload(rc)
    check("T11 re-importing the runner re-pins torch to one thread", torch.get_num_threads() == 1)


def t12_role_mapping_scope():
    # `load_role_schema` returns a dict keyed by market; every consumer wants ONE
    # market's five-role tuple. Passing the whole dict must fail closed rather than
    # silently building a role table -- it did not, once, in main()'s per-cell loop.
    mapping, _ = load_role_schema()
    rng = np.random.default_rng(3)
    z = rng.normal(size=(4, 24, 3)).astype(np.float32)
    try:
        role_table(z, mapping)
        raised = False
    except RuntimeError:
        raised = True
    check("T12 role_table fails closed on the whole multi-market schema dict", raised)
    check("T12 schema dict is keyed by exactly the three audited markets", set(mapping) == set(MARKETS))
    for m in MARKETS:
        check(f"T12 {m} single-market mapping carries exactly five roles", len(mapping[m]) == N_ROLES)
        check(f"T12 {m} role slots are index tuples, not column names",
              all(isinstance(t, tuple) and all(isinstance(i, int) for i in t) for t in mapping[m]))


def main():
    t01_beta_model_shape()
    t02_zero_init_replays_s1_exactly()
    t03_role_table_pooling_and_structural_zero()
    t04_schema_is_primary_and_legal_only()
    t05_real_state_reads_zero_forbidden()
    t06_beta_oof_chronology()
    t07_alpha_chronology_and_rowset()
    t08_beta_actually_trains()
    t09_final_beta_uses_only_oof_rows()
    t10_standardization_is_train_only()
    t11_single_thread_invariant()
    t12_role_mapping_scope()
    if "--smoke" in sys.argv:
        smoke_one_cell()
    for name in PASSED:
        print(f"PASS {name}")
    print(f"\n{len(PASSED)} checks passed")


if __name__ == "__main__":
    main()
