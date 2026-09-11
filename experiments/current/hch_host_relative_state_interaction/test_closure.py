"""Adversarial tests for the HRSI final structural closure.

Run:  python experiments/current/hch_host_relative_state_interaction/test_closure.py
      python experiments/current/hch_host_relative_state_interaction/test_closure.py --smoke

Fast tests exercise exactly the mechanism-level claims the protocol gates depend
on: the causal seven-day Host-bias direction, the fixed five-role interaction, the
single global five-parameter beta, the macro-balanced cell loss, and the
chronology of both the beta supervision and the pooled-alpha calibration.

`--smoke` additionally runs one real (market, host, seed) cell end-to-end and
checks the frozen S1 replay tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\作业\science\solar_leak_price_model").resolve()
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.current.hch_host_relative_state_interaction.run_closure import (
    BIAS_CHANNEL, EPS, HIST_DAYS, HOSTS, LEGAL, MARKETS, MASK_CHANNEL, N_ROLES, REPLAY_TOL,
    ROLES, HRSIResidual, beta_oof, bias_from_residual_block, calibrate_hrsi_oof,
    fit_global_beta, host_bias_direction, interaction, load_role_schema, macro_balanced_loss,
    new_beta_model, pooled_alpha, predict_beta, role_table, standardized_state,
)
from experiments.current.hch_anchored_compact_amplitude.run_closure import load_frozen_cache
from experiments.current.hch_unified_da_shape_upgrade.run_canary import make_inputs
from src.mvp.hch_minimal_repair.crossfit import chronological_blocks
from utils.timing import parameter_count

PASSED: list[str] = []


def check(name, cond):
    if not bool(cond):
        raise AssertionError(f"FAILED: {name}")
    PASSED.append(name)


def unit(v):
    v = np.asarray(v, np.float64).reshape(-1, 24) if np.asarray(v).ndim != 2 else np.asarray(v, np.float64)
    return (v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)).astype(np.float32)


def raises(fn):
    try:
        fn(); return False
    except RuntimeError:
        return True


def tensors(pools):
    return [(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32),
             torch.as_tensor(c, dtype=torch.float32)) for a, b, c in pools]


def mbl(model, pools):
    with torch.no_grad():
        return float(macro_balanced_loss(model, tensors(pools)))


# --------------------------------------------------------------------------
# T01-T02  the single global parameterisation
# --------------------------------------------------------------------------

def t01_beta_model_shape():
    m = HRSIResidual()
    check("T01 exactly five trainable parameters", parameter_count(m) == N_ROLES)
    check("T01 beta initialised to exactly zero", float(m.beta.detach().abs().max()) == 0.0)
    check("T01 no hidden layer or submodule", set(type(x) for x in m.modules()) == {HRSIResidual})
    check("T01 no reset_parameters hook that could wipe beta",
          not any(hasattr(x, "reset_parameters") for x in m.modules()))


def t02_zero_init_replays_s1_exactly():
    rng = np.random.default_rng(0)
    u = unit(rng.normal(size=(64, 24)))
    Z = rng.normal(size=(64, 24, N_ROLES)).astype(np.float32)
    out = predict_beta(np.zeros(N_ROLES), u, Z)
    check("T02 beta=0 replays S1 directions within tolerance", float(np.max(np.abs(out - u))) <= REPLAY_TOL)
    check("T02 beta=0 output is still unit norm", np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5))
    moved = predict_beta(np.full(N_ROLES, 0.3), u, Z)
    check("T02 a non-zero beta changes the directions", float(np.max(np.abs(moved - u))) > 1e-3)
    check("T02 output remains unit norm for non-zero beta",
          np.allclose(np.linalg.norm(moved, axis=1), 1.0, atol=1e-5))
    check("T02 the projection is linear in beta before normalization",
          np.allclose(predict_beta(np.full(N_ROLES, 0.3), u, Z),
                      (lambda z: z / (np.linalg.norm(z, axis=1, keepdims=True) + EPS))(u + 0.3 * Z.sum(axis=2)),
                      atol=1e-5))


# --------------------------------------------------------------------------
# T03-T06  the causal Host-bias direction
# --------------------------------------------------------------------------

def t03_bias_is_same_hour_mean_over_seven_days():
    """`b[h]` must be the same-hour mean over the seven frozen historical days.

    The 168-vector is `[7 days x 24 hours]`, so horizon hour `h` lives at positions
    `h, 24+h, ..., 144+h` -- the same reshape `DailyPatchShape` uses. A defect here
    (for example an off-by-one day stride) would smear the bias across horizon
    hours, so it is pinned against a hand-built block.
    """
    n = 3
    hour_values = np.arange(1, 25, dtype=np.float64)
    resn = np.tile(hour_values, (n, HIST_DAYS)); msk = np.ones_like(resn)
    msk[1, :] = 0.0; resn[1, :] = 0.0
    for d in (0, 3):
        msk[1, d * 24:(d + 1) * 24] = 1.0; resn[1, d * 24:(d + 1) * 24] = hour_values
    msk[2, :] = 0.0; resn[2, :] = 0.0
    bhat, bias, counts = bias_from_residual_block(resn, msk)

    check("T03 day counts per hour are exactly the available days",
          np.array_equal(counts[0], np.full(24, 7.0)) and np.array_equal(counts[1], np.full(24, 2.0))
          and np.array_equal(counts[2], np.zeros(24)))
    check("T03 hour h aggregates the same hour of every available day",
          np.allclose(bias[0], hour_values, atol=1e-12))
    check("T03 a partially available hour averages only over its available days",
          np.allclose(bias[1], hour_values, atol=1e-12))
    check("T03 full-coverage rows carry a unit bias direction",
          np.isclose(np.linalg.norm(bhat[0]), 1.0, atol=1e-6))
    check("T03 an hour with no historical residual contributes a zero numerator",
          np.allclose(bias[2], 0.0, atol=1e-12))
    check("T03 a zero-norm bias yields exactly b_hat = 0 rather than a NaN",
          np.array_equal(bhat[2], np.zeros(24)) and np.isfinite(bhat).all())
    # A bias that is constant across days must survive as a pure hourly profile.
    resn2 = np.zeros((1, HIST_DAYS * 24)); msk2 = np.ones_like(resn2)
    resn2[0, ::24] = 5.0  # only hour 0 of each day
    b2, bias2, _ = bias_from_residual_block(resn2, msk2)
    check("T03 a single non-zero hour yields a one-hot bias direction",
          np.isclose(b2[0, 0], 1.0, atol=1e-6) and np.allclose(b2[0, 1:], 0.0, atol=1e-12))
    check("T03 the raw bias is not L2-normalized in place", np.isclose(bias2[0, 0], 5.0, atol=1e-12))


def t04_bias_fails_closed_on_contract_violations():
    zeros = np.zeros((2, HIST_DAYS * 24)); mask_off = np.zeros_like(zeros)
    check("T04 wrong history width is rejected",
          raises(lambda: bias_from_residual_block(np.zeros((2, 120)), np.zeros((2, 120)))))
    check("T04 a non-zero residual in a masked-out coordinate is rejected",
          raises(lambda: bias_from_residual_block(np.full((2, HIST_DAYS * 24), 0.5), mask_off)))
    check("T04 a shape mismatch between residual and mask is rejected",
          raises(lambda: bias_from_residual_block(zeros, mask_off[:, :100])))
    check("T04 legal all-masked input is accepted",
          isinstance(bias_from_residual_block(zeros, mask_off)[0], np.ndarray))


def t05_real_bias_matches_the_frozen_repair_input():
    """The bias must come from the frozen 624-D repair input, not a re-derivation."""
    cache, _ = load_frozen_cache("LAGO_DE", "PatchTST")
    fi, _ = make_inputs(cache.subset("DIAG_FIT"), cache)
    check("T05 the frozen repair input is 624-D", fi.x.shape[1] == 624)
    check("T05 the mask channel equals RepairInput.residual_mask",
          np.array_equal(fi.x[:, MASK_CHANNEL].astype(np.float32), fi.residual_mask.astype(np.float32)))
    bhat, _, _ = host_bias_direction(fi.x, fi.residual_mask)
    direct, _, _ = bias_from_residual_block(fi.x[:, BIAS_CHANNEL], fi.x[:, MASK_CHANNEL])
    check("T05 host_bias_direction is exactly the frozen-input read", np.array_equal(bhat, direct))
    check("T05 masked historical residuals are exactly zero in the frozen input",
          np.all(fi.x[:, BIAS_CHANNEL][fi.residual_mask == 0] == 0.0))
    # (a) mask bit cleared in BOTH the frozen array and the mask channel, so the
    #     layout comparison passes and only the "masked coordinate must be exactly
    #     zero" rule can catch the non-zero residual.
    bad = fi.x.copy(); bad[0, MASK_CHANNEL.start] = 0.0; bad[0, BIAS_CHANNEL.start] = 1.0
    m2 = fi.residual_mask.copy(); m2[0, 0] = False
    check("T05 host_bias_direction fails closed when a masked coordinate is non-zero",
          raises(lambda: host_bias_direction(bad, m2)))
    # (b) the mask channel no longer matches the registered RepairInput mask.
    bad2 = fi.x.copy(); bad2[0, MASK_CHANNEL.start] = 1.0 - bad2[0, MASK_CHANNEL.start]
    check("T05 host_bias_direction fails closed when the mask channel is altered",
          raises(lambda: host_bias_direction(bad2, fi.residual_mask)))
    short = fi.x[:, :600].copy()
    check("T05 a non-registered state width is rejected", raises(lambda: host_bias_direction(short, fi.residual_mask)))


def t06_real_bias_profile_is_causal_and_available():
    """Availability structure and `b_hat` norms must match the protocol's causal rule."""
    for h in HOSTS:
        cache, _ = load_frozen_cache("GANSU_DA", h)
        fi, _ = make_inputs(cache.subset("DIAG_FIT"), cache)
        bhat, _, cnt = host_bias_direction(fi.x, fi.residual_mask)
        covered = cnt.max(axis=1) > 0
        check(f"T06 GANSU_DA/{h}: rows with any history have unit bias direction",
              np.allclose(np.linalg.norm(bhat[covered], axis=1), 1.0, atol=1e-5))
        check(f"T06 GANSU_DA/{h}: rows with no history carry exactly b_hat = 0",
              np.array_equal(bhat[~covered], np.zeros((int((~covered).sum()), 24), np.float32)))
        check(f"T06 GANSU_DA/{h}: bias availability equals repair availability",
              np.array_equal(covered, fi.repair_available))
        check(f"T06 GANSU_DA/{h}: no hour uses more than the frozen seven days", cnt.max() <= HIST_DAYS)

        cache, _ = load_frozen_cache("LAGO_DE", h)
        fi, _ = make_inputs(cache.subset("DIAG_FIT"), cache)
        bhat_l, _, cnt_l = host_bias_direction(fi.x, fi.residual_mask)
        check(f"T06 LAGO_DE/{h}: full seven-day coverage gives a unit bias direction",
              np.allclose(np.linalg.norm(bhat_l, axis=1), 1.0, atol=1e-5))
        check(f"T06 LAGO_DE/{h}: every hour reaches the full seven days", cnt_l.min() == HIST_DAYS)
        # Bias availability must be exactly the row set S1 calls repair-available;
        # otherwise the two mechanisms disagree about which days they may look at.
        check(f"T06 LAGO_DE/{h}: bias availability equals repair availability",
              np.array_equal(cnt_l.max(axis=1) > 0, fi.repair_available))
        check(f"T06 LAGO_DE/{h}: the frozen seven-day window is never exceeded",
              cnt_l.max() == HIST_DAYS)


# --------------------------------------------------------------------------
# T07-T09  the shared parameter and the macro-balanced objective
# --------------------------------------------------------------------------

def t07_interaction_is_the_registered_product():
    rng = np.random.default_rng(1)
    bhat = unit(rng.normal(size=(5, 24)))
    R = rng.normal(size=(5, 24, N_ROLES)).astype(np.float32)
    Z = interaction(bhat, R)
    check("T07 interaction shape is [rows, 24, 5]", Z.shape == (5, 24, N_ROLES))
    check("T07 Z[h,k] = b_hat[h] * R[h,k]", np.allclose(Z, bhat[:, :, None] * R, atol=1e-6))
    check("T07 a zero bias direction disables every role channel",
          np.allclose(interaction(np.zeros_like(bhat), R), 0.0))
    check("T07 a row-alignment mismatch is rejected",
          raises(lambda: interaction(bhat[:3], R)))
    check("T07 a role table with the wrong arity is rejected",
          raises(lambda: interaction(bhat, R[:, :, :3])))


def t08_macro_balanced_loss_ignores_cell_size():
    """Equal total Shape weight per cell: cell size must not change the objective."""
    rng = np.random.default_rng(2)
    # Cell 0 is deliberately the hard cell (target = -u, squared error 4 per row);
    # cells 1..5 are already solved (target = u, squared error 0). Cell 0 is then
    # repeated 100x, so the panel has 125 rows instead of 30. A macro-balanced
    # objective cannot notice; a row-weighted one is dominated by the big cell.
    u0 = unit(rng.normal(size=(5, 24)))
    Z0 = rng.normal(size=(5, 24, N_ROLES)).astype(np.float32)
    six_small = [(u0, Z0, -u0)] + [(unit(rng.normal(size=(5, 24))),
                                    rng.normal(size=(5, 24, N_ROLES)).astype(np.float32), None) for _ in range(5)]
    six_small = [(a, b, c if c is not None else a) for a, b, c in six_small]
    one_huge = [(np.tile(u0, (100, 1)), np.tile(Z0, (100, 1, 1)), np.tile(-u0, (100, 1)))] + six_small[1:]
    model = new_beta_model()
    flat = lambda pools: sum(float(np.sum((predict_beta(np.zeros(N_ROLES), a, b) - c) ** 2)) for a, b, c in pools) / sum(len(a) for a, _, _ in pools)
    check("T08 the macro-balanced objective is invariant to cell size",
          np.isclose(mbl(model, one_huge), mbl(model, six_small), atol=1e-5))
    check("T08 a row-weighted objective would not be invariant (so this is a real distinction)",
          abs(flat(one_huge) - flat(six_small)) > 0.1)
    manual = np.mean([float(np.mean(np.sum((predict_beta(np.zeros(N_ROLES), a, b) - c) ** 2, axis=1)))
                      for a, b, c in six_small])
    check("T08 objective equals the unweighted mean of per-cell mean Shape errors",
          np.isclose(mbl(model, six_small), manual, atol=1e-5))
    check("T08 the macro-balanced objective refuses a non-six-cell panel",
          raises(lambda: macro_balanced_loss(model, tensors(six_small[:5]))))


def t09_global_beta_trains_and_is_shared():
    rng = np.random.default_rng(3)
    n = 40
    Z = rng.normal(size=(6, n, 24, N_ROLES)).astype(np.float32)
    raw = rng.normal(size=(6, n, 24))
    u = (raw / (np.linalg.norm(raw, axis=2, keepdims=True) + EPS)).astype(np.float32)
    # One planted panel-wide direction change: a single shared beta can recover it.
    planted = np.array([1.0, 0.0, 0.5, 0.0, 0.0], np.float32)
    direction = np.einsum("cnhr,r->cnh", Z, planted)
    shifted = u + 0.35 * direction
    target = (shifted / (np.linalg.norm(shifted, axis=2, keepdims=True) + EPS)).astype(np.float32)
    pools = [(u[c], Z[c], target[c]) for c in range(6)]
    beta, info = fit_global_beta(pools, 7)
    check("T09 the fitted beta has exactly five components", beta.shape == (N_ROLES,))
    check("T09 the fitted beta is not the zero initialisation", float(np.abs(beta).max()) > 1e-3)
    check("T09 beta=0 replay was verified before training", info["zero_init_replay_max_abs_diff"] <= REPLAY_TOL)
    zero = mbl(new_beta_model(), pools)
    check("T09 training reduces the macro-balanced objective", info["objective"] < zero)
    check("T09 the fold training row count is reported", info["n_train_rows"] == 6 * n)
    check("T09 per-cell row counts are reported", info["rows_per_cell"] == [n] * 6)
    check("T09 the training time is measured", info["training_seconds"] > 0.0)
    preds = [predict_beta(beta, u[c], Z[c]) for c in range(6)]
    check("T09 every cell is predicted by the same beta vector",
          all(np.array_equal(p, predict_beta(beta, u[c], Z[c])) for c, p in enumerate(preds)))
    check("T09 all predictions are unit-norm directions",
          all(np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5) for p in preds))
    check("T09 the fitted beta recovers the planted direction up to scale",
          np.corrcoef(beta, planted)[0, 1] > 0.9)
    check("T09 a global beta refuses a panel that is not exactly six cells", raises(lambda: fit_global_beta(pools[:4], 7)))
    empty = (pools[0][0][:0], pools[0][1][:0], pools[0][2][:0])
    check("T09 an empty cell is rejected rather than silently dropped from the mean",
          raises(lambda: fit_global_beta([empty] + pools[1:], 7)))


# --------------------------------------------------------------------------
# T10-T12  schema, roles, state normalization
# --------------------------------------------------------------------------

def t10_role_table_pooling_and_structural_zero():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(7, 24, 5)).astype(np.float32)
    mapping = ((0, 1), (2,), (), (3,), (4,))
    R = role_table(z, mapping)
    check("T10 role table is [rows, 24, 5]", R.shape == (7, 24, N_ROLES))
    check("T10 a multi-channel role is the plain channel mean",
          np.allclose(R[:, :, 0], z[:, :, [0, 1]].mean(axis=2), atol=1e-6))
    check("T10 a single-channel role is that channel", np.allclose(R[:, :, 1], z[:, :, 2], atol=1e-6))
    check("T10 a structurally absent role is an exact zero column",
          np.array_equal(R[:, :, 2], np.zeros((7, 24), np.float32)))
    check("T10 pooling is deterministic and channel-order-free",
          np.array_equal(R, role_table(z, ((1, 0), (2,), (), (3,), (4,)))))
    for bad in ((0, 1), (0, 1, 2, 3), ((0,),) * 6):
        check("T10 role_table fails closed unless exactly five roles are mapped",
              raises(lambda b=bad: role_table(z, b)))


def t11_role_mapping_scope_is_single_market():
    """The per-cell fit must receive one market's five-tuple, never the whole schema.

    This mirrors the HSA defect where the multi-market schema dict was passed into
    the per-cell role table; there it crashed loudly, and this test pins the same
    boundary so a refactor cannot reintroduce the broader object silently.
    """
    mapping, manifest = load_role_schema()
    check("T11 role_table fails closed on the whole multi-market schema dict",
          raises(lambda: role_table(np.zeros((3, 24, 5), np.float32), mapping)))
    check("T11 schema dict is keyed by exactly the three audited markets", set(mapping) == set(MARKETS))
    for m in MARKETS:
        check(f"T11 {m} single-market mapping carries exactly five roles", len(mapping[m]) == N_ROLES)
        check(f"T11 {m} role slots are index tuples, not column names",
              all(isinstance(t, tuple) and all(isinstance(i, int) for i in t) for t in mapping[m]))
    check("T11 every mapped index is inside the registered LEGAL columns",
          all(0 <= i < len(LEGAL[m]) for m in MARKETS for t in mapping[m] for i in t))
    check("T11 only the five audited semantic roles appear", set(manifest.semantic_role) == set(ROLES))
    check("T11 no market maps more than one channel into the same role slot twice",
          all(len(set(t)) == len(t) for m in MARKETS for t in mapping[m]))


def t12_standardization_is_train_only():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(30, 24, 5)); b = rng.normal(size=(10, 24, 5)) + 50.0
    za, zb = standardized_state(a, b)
    af = a.reshape(len(a), -1); bf = b.reshape(len(b), -1)
    check("T12 output shapes survive the round trip", za.shape == a.shape and zb.shape == b.shape)
    check("T12 training rows are standardized with their own statistics",
          np.allclose(za.reshape(len(a), -1).mean(0), 0.0, atol=1e-6))
    check("T12 evaluation rows use the training statistics, not their own",
          np.allclose(zb.reshape(len(b), -1).mean(0), (bf.mean(0) - af.mean(0)) / (af.std(0) + 1e-6), atol=1e-6))
    check("T12 evaluation rows are not centred on zero", not np.allclose(zb.reshape(len(b), -1).mean(0), 0.0, atol=1e-3))
    _, other = standardized_state(a, rng.normal(size=(7, 24, 5)))
    za2, _ = standardized_state(a, rng.normal(size=(7, 24, 5)))
    check("T12 the training transform does not depend on the evaluation rows", np.array_equal(za, za2))
    check("T12 a different evaluation set produces a different transform of itself",
          not np.array_equal(zb, other))


# --------------------------------------------------------------------------
# T13-T15  chronology of beta supervision and alpha calibration
# --------------------------------------------------------------------------

def t13_pooled_alpha_rowset_must_be_explicit():
    """The pooled alpha is a discrete statistic: its row set must be stated, not inferred.

    `weighted_median_scalar` keeps only rows with `|u| > 1e-12`, and NaN compares
    False, so rows without a direction are dropped silently. That is the property
    which made the HSA alpha row-set defect numerically invisible; HRSI therefore
    restricts the row set explicitly to rows carrying a finite HRSI direction.
    """
    e = np.tile(np.array([[1.0, 2.0]]), (6, 1))
    u = np.tile(np.array([[1.0, 0.0]]), (6, 1)); u[3:] = -1.0; u[3:, 1] = 0.0
    narrow = pooled_alpha(e, u, np.arange(3)); wide = pooled_alpha(e, u, np.arange(6))
    check("T13 pooled alpha is sensitive to which rows are supplied", narrow != wide)
    nan_dir = np.tile(np.array([[np.nan, np.nan]]), (3, 1))
    check("T13 NaN-direction rows are silently ignored by the frozen statistic",
          pooled_alpha(e, np.vstack([u[:3], nan_dir]), np.arange(6)) == narrow)
    check("T13 pooled alpha is clamped non-negative", pooled_alpha(e, -np.abs(u), np.arange(6)) >= 0.0)


def t14_alpha_calibration_is_self_excluded_and_shifted():
    """HRSI can only start calibrating once a beta exists, so it starts at B2."""
    rng = np.random.default_rng(6)
    n = 101
    blocks = chronological_blocks(n)
    e = rng.normal(size=(n, 24)).astype(np.float64)
    u = unit(rng.normal(size=(n, 24)))
    av = np.ones(n, bool); has_dir = np.ones(n, bool)
    alpha, d, complete, records = calibrate_hrsi_oof(e, u, av, blocks, has_dir)
    check("T14 two calibrated folds are produced (B3, B4)", len(records) == 2)
    check("T14 every calibration fold is self-excluded", all(r["self_excluded"] for r in records))
    check("T14 every calibration index precedes every proposal index",
          all(r["calibration_max_index"] < r["proposal_min_index"] for r in records))
    check("T14 the first HRSI fold is B3, not B1", [r["block"] for r in records] == ["B3", "B4"])
    check("T14 the alpha row set is exactly B2..B4",
          np.array_equal(complete, np.concatenate([blocks["B2"], blocks["B3"], blocks["B4"]])))
    check("T14 a finite non-negative alpha is returned", np.isfinite(alpha) and alpha >= 0.0)
    proposed = np.concatenate([blocks["B3"], blocks["B4"]])
    check("T14 proposals are produced exactly on the calibrated folds B3..B4",
          np.isfinite(d[proposed]).all()
          and np.isnan(d[np.concatenate([blocks["B1"], blocks["B2"]])]).all())
    check("T14 the alpha row set is exactly B2..B4 with finite directions",
          np.isfinite(u[complete]).all()
          and len(complete) == len(blocks["B2"]) + len(blocks["B3"]) + len(blocks["B4"]))
    has_dir2 = has_dir.copy(); has_dir2[blocks["B3"][:2]] = False
    _, d2, complete2, _ = calibrate_hrsi_oof(e, u, av, blocks, has_dir2)
    check("T14 rows without an HRSI direction leave the alpha row set",
          np.isnan(d2[blocks["B3"][:2]]).all() and len(complete2) == len(complete) - 2)
    check("T14 the excluded rows would otherwise have been calibrated on",
          np.array_equal(complete2, np.setdiff1d(complete, blocks["B3"][:2])))


def t15_beta_oof_is_panel_global_and_chronological():
    """One beta per (seed, fold) must cover all six cells and never see the fold."""
    rng = np.random.default_rng(7)
    n = 61
    blocks = chronological_blocks(n)
    cells = {}
    for m in MARKETS:
        for h in HOSTS:
            state = rng.normal(size=(n, 24, 5)).astype(np.float32)
            cells[m, h] = {"market": m, "host": h, "blocks": blocks,
                           "oof_u": unit(rng.normal(size=(n, 24))),
                           "ef": rng.normal(size=(n, 24)) * 0.5 + 0.2,
                           "av": np.ones(n, bool), "has_dir": np.ones(n, bool),
                           "bhat_fit": unit(rng.normal(size=(n, 24))),
                           "state_fit_raw": state, "state_eval_raw": state[:5],
                           "role_map": ((0,), (1,), (2,), (3,), (4,))}
    per_cell_u, folds, chron, seconds = beta_oof(cells, 7)
    check("T15 one beta record is written per fold", len(folds) == 3)
    check("T15 every fold beta covers all six cells", all(f["cells_covered"] == 6 for f in folds))
    check("T15 folds are B2, B3, B4 in order", [f["fold"] for f in folds] == ["B2", "B3", "B4"])
    check("T15 every fold beta has exactly five components",
          all([f[f"beta_{r}"] for r in ROLES] and
              len([k for k in f if k.startswith("beta_") and k != "beta_sha256"]) == N_ROLES for f in folds))
    check("T15 each fold beta is reported with a content hash", all(len(f["beta_sha256"]) == 64 for f in folds))
    check("T15 beta training time is accumulated", seconds > 0.0)
    beta_records = [r for r in chron if r["record_type"] == "beta"]
    check("T15 every cell is predicted for every fold", len(beta_records) == 3 * 6)
    check("T15 a fold's beta is used by all six cells",
          all(len({r["beta_sha256"] for r in beta_records if r["block"] == b}) == 1 for b in ("B2", "B3", "B4")))
    check("T15 the fold beta training scope is the six-cell pool",
          all(r["beta_train_scope"] == "six-cell pooled B1..B(k-1)" for r in beta_records))
    check("T15 every beta record is marked self-excluded", all(r["self_excluded"] for r in beta_records))
    b234 = np.concatenate([blocks["B2"], blocks["B3"], blocks["B4"]])
    check("T15 HRSI directions exist exactly on the rows that received a beta",
          all(np.array_equal(np.isfinite(per_cell_u[k]).all(axis=1), np.isin(np.arange(n), b234)) for k in cells))
    check("T15 B1 never receives an HRSI direction",
          all(np.isnan(per_cell_u[k][blocks["B1"]]).all() for k in cells))
    check("T15 all produced HRSI directions are unit norm",
          all(np.allclose(np.linalg.norm(per_cell_u[k][b234], axis=1), 1.0, atol=1e-5) for k in cells))


def t16_single_thread_invariant():
    """Importing the runner must have pinned torch to one thread for S1 replay."""
    check("T16 torch is pinned to a single thread by the runner import", torch.get_num_threads() == 1)


def t17_frozen_clock_and_replay_tolerances():
    check("T17 the frozen S1 metric replay tolerance is unchanged", REPLAY_TOL == 1e-6)
    check("T17 the historical window is exactly the seven frozen days", HIST_DAYS == 7)
    check("T17 the bias channel is the frozen normalized residual block",
          (BIAS_CHANNEL.start, BIAS_CHANNEL.stop) == (168, 336))
    check("T17 the mask channel is the frozen residual mask block",
          (MASK_CHANNEL.start, MASK_CHANNEL.stop) == (336, 504))


def t18_rowset_diagnostic_is_a_real_rescoring():
    """The disclosed row-set diagnostic must measure something, and must not gate.

    HRSI's alpha lives on B2..B4 and frozen S1's on B1..B4. Because the pooled
    alpha is a discrete weighted median, that shift can move the scalar and
    therefore the reported Overall-MAE by itself. The runner must (a) actually
    rescore S1 under HRSI's row set -- so the column is not a copy of the frozen
    number -- and (b) keep every diagnostic column out of the gated metric names,
    so the disclosure can never silently become a gate input.
    """
    from experiments.current.hch_host_relative_state_interaction.run_closure import (
        ROWSET_SENS_COLS, score,
    )
    rng = np.random.default_rng(11)
    n, seed = 60, 7
    y = 50.0 + 10.0 * rng.normal(size=(n, 24))
    hp = y + 3.0 * rng.normal(size=(n, 24))
    ee = y - hp
    scale = np.full((n, 1), 10.0)
    c = {"y": y, "hp": hp, "ee": ee, "scale": scale,
         "tail": np.zeros((n, 24), bool), "normal": np.ones((n, 24), bool),
         "avail": np.ones(n, bool), "n_eval": n, "s1_inference_seconds": 0.0}
    u_dir = unit(rng.normal(size=(n, 24)))
    ef = ee.copy()
    blocks = chronological_blocks(n)
    narrow = np.arange(n // 2)
    wide = np.arange(n)
    a_narrow, a_wide = pooled_alpha(ef, u_dir, narrow), pooled_alpha(ef, u_dir, wide)
    check("T18 the two alpha row sets really do disagree on this fixture", a_narrow != a_wide)
    met_n, _ = score(c, u_dir, a_narrow, ef, u_dir, narrow, 1, 0.0)
    met_w, _ = score(c, u_dir, a_wide, ef, u_dir, wide, 1, 0.0)
    check("T18 rescoring on the shifted row set changes Overall_MAE",
          abs(met_n["Overall_MAE"] - met_w["Overall_MAE"]) > 0.0)
    check("T18 rescoring on the shifted row set changes the reported gain",
          abs(met_n["relative_gain_pct"] - met_w["relative_gain_pct"]) > 0.0)
    # The diagnostic may only move the amplitude row set. If it also moved the
    # direction-dependent numbers it would be a second experiment, not a
    # disclosure about the first one.
    check("T18 the row-set shift leaves the Shape mechanism untouched",
          met_n["shape_cosine"] == met_w["shape_cosine"]
          and met_n["wrong_hemisphere_rate"] == met_w["wrong_hemisphere_rate"])
    gated = {"Overall_MAE", "Tail_MAE", "Normal_MAE", "relative_gain_pct",
             "shape_cosine", "wrong_hemisphere_rate", "alpha"}
    check("T18 no row-set diagnostic column is a gated metric name",
          not (set(ROWSET_SENS_COLS) & gated))
    check("T18 the diagnostic discloses both the refit alpha and the rescored gain",
          "S1_alpha_refit_on_HRSI_rowset" in ROWSET_SENS_COLS
          and "S1_aligned_rowset_relative_gain_pct" in ROWSET_SENS_COLS
          and "HRSI_minus_S1_aligned_overall_gain_pp" in ROWSET_SENS_COLS)
    check("T18 the frozen S1 number stays reported alongside the aligned one",
          "S1_frozen_Overall_MAE" in ROWSET_SENS_COLS)


def smoke_one_cell():
    """One real cell end-to-end: frozen S1 replay plus the alpha that HRSI reuses."""
    from experiments.current.hch_host_relative_state_interaction.run_closure import (
        SHAPE_EVIDENCE, TOL, _s1_alpha, _s1_complete, s1_artifacts, score,
    )
    old = pd.read_csv(SHAPE_EVIDENCE / "metrics_by_seed.csv")
    old = old[old.method.eq("S1_DailyPatch_GRU32")]
    m, h, seed = "LAGO_DE", "PatchTST", 7
    cache, _ = load_frozen_cache(m, h); fit = cache.subset("DIAG_FIT"); ev = cache.subset("DIAG_EVAL")
    fi, floor = make_inputs(fit, cache); ei, _ = make_inputs(ev, cache, floor)
    origins = np.r_[fit.timestamp, ev.timestamp]
    from experiments.current.hch_unified_da_shape_upgrade.run_canary import read_state
    z, access = read_state(m, origins); nf = len(fit.timestamp)
    check("SMOKE forbidden state reads are zero",
          access["target_day_DA_input_reads"] == 0 and access["realized_RT_input_reads"] == 0
          and access["S3_S4_reads"] == 0)
    ef = (fit.y_true[:, :, 0] - fit.host_pred[:, :, 0]) / np.where(np.isfinite(fi.scale), fi.scale, 1)[:, None]
    oof, u_eval, info = s1_artifacts(fi, ei, z[:nf], z[nf:], ef, seed)
    cell = {"blocks": oof["blocks"], "oof_u": oof["u"]}
    alpha, _, _, _ = _s1_alpha(ef, cell, fi.repair_available)
    y = ev.y_true[:, :, 0]; hp = ev.host_pred[:, :, 0]
    q05, q95 = np.quantile(fit.y_true[:, :, 0], [.05, .95])
    met, _ = score(
        {"y": y, "hp": hp, "tail": (y <= q05) | (y >= q95), "normal": ~((y <= q05) | (y >= q95)),
         "ee": (y - hp) / np.where(np.isfinite(ei.scale), ei.scale, 1)[:, None],
         "scale": np.nan_to_num(ei.scale, nan=0)[:, None], "avail": ei.repair_available,
         "n_eval": len(ei.x), "s1_inference_seconds": info["inference_seconds"]},
        u_eval, alpha, ef, oof["u"], _s1_complete(cell, fi.repair_available), info["parameter_count"], 0.0)
    ref = old[(old.market == m) & (old.host == h) & (old.seed == seed)].iloc[0]
    for col, key in (("Overall_MAE", "Overall_MAE"), ("Tail_MAE", "Tail_MAE"),
                     ("Normal_MAE", "Normal_MAE"), ("alpha", "alpha"),
                     ("shape_cosine", "shape_cosine"), ("wrong_hemisphere_rate", "wrong_hemisphere_rate")):
        check(f"SMOKE {m}/{h} seed{seed} {key} replays the frozen S1 evidence",
              abs(met[col] - float(ref[key])) <= TOL)


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("canonical repo required")
    t01_beta_model_shape()
    t02_zero_init_replays_s1_exactly()
    t03_bias_is_same_hour_mean_over_seven_days()
    t04_bias_fails_closed_on_contract_violations()
    t05_real_bias_matches_the_frozen_repair_input()
    t06_real_bias_profile_is_causal_and_available()
    t07_interaction_is_the_registered_product()
    t08_macro_balanced_loss_ignores_cell_size()
    t09_global_beta_trains_and_is_shared()
    t10_role_table_pooling_and_structural_zero()
    t11_role_mapping_scope_is_single_market()
    t12_standardization_is_train_only()
    t13_pooled_alpha_rowset_must_be_explicit()
    t14_alpha_calibration_is_self_excluded_and_shifted()
    t15_beta_oof_is_panel_global_and_chronological()
    t16_single_thread_invariant()
    t17_frozen_clock_and_replay_tolerances()
    t18_rowset_diagnostic_is_a_real_rescoring()
    if "--smoke" in sys.argv:
        smoke_one_cell()
    for name in PASSED:
        print(f"PASS {name}")
    print(f"\n{len(PASSED)} checks passed")


if __name__ == "__main__":
    main()
