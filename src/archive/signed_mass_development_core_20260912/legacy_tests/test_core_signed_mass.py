"""Unit tests for the signed-mass core mathematics and information contracts.

These are synthetic unit tests only. They do not train a scientific model, read a
benchmark split, or access protected data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import core  # noqa: E402
from core import (  # noqa: E402
    DeterministicFeatureBuilder, KNNShapeContextProvider, LocalTCN,
    RareMassSampler, RareMassSamplerConfig, RawInputs, ShapeMemoryBank,
    SignedMassRepairModel, TrainFrozenScaler, apply_mae_scalar, default_config,
    decompose_residual, fit_mae_scalar, fit_residual_scale, shape_wasserstein1,
    within_day_robust_normalize,
)

RESIDUAL = torch.tensor([
    [3.0, -1.0, 0.0, 2.0, -4.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, -2.0, 0.0, 0.0, 0.0],
])


def _raw(batch=4, horizon=24, windows=7, n_features=3, n_calendar=2, seed=0):
    torch.manual_seed(seed)
    host = torch.randn(batch, horizon) * 20 + 50
    features = torch.randn(batch, horizon, n_features) * 10 + 30
    feature_mask = torch.ones_like(features)
    calendar = torch.randn(batch, horizon, n_calendar) if n_calendar else None
    return RawInputs(
        host=host,
        valid_mask=torch.ones(batch, horizon),
        forecast_features=features,
        forecast_feature_mask=feature_mask,
        calendar=calendar,
        past_residual=torch.randn(batch, windows, horizon) if windows else None,
    )


def _batch(batch=4, horizon=24, windows=7, n_features=3, n_calendar=2, seed=0):
    raw = _raw(batch, horizon, windows, n_features, n_calendar, seed)
    return DeterministicFeatureBuilder().build(raw)


def _model(n_features=3, n_calendar=2):
    return SignedMassRepairModel(default_config(n_features, n_calendar)).eval()


# --- exact signed-mass object -------------------------------------------------

def test_residual_reconstructs_exactly_from_masses_and_shapes():
    targets = decompose_residual(RESIDUAL)
    rebuilt = (targets.mass_positive.unsqueeze(-1) * targets.shape_positive
               - targets.mass_negative.unsqueeze(-1) * targets.shape_negative)
    assert torch.allclose(rebuilt, RESIDUAL, atol=1e-6)


def test_every_shape_is_a_nonnegative_simplex():
    targets = decompose_residual(RESIDUAL)
    for shape in (targets.shape_positive, targets.shape_negative):
        assert (shape >= 0).all()
        assert torch.allclose(shape.sum(dim=-1), torch.ones(shape.shape[0]), atol=1e-6)

    model = _model()
    with torch.no_grad():
        out = model(_batch())
    for shape in (out.shape.positive, out.shape.negative):
        assert (shape >= 0).all()
        assert torch.allclose(shape.sum(dim=-1), torch.ones(shape.shape[0]), atol=1e-5)


def test_shape_is_scale_invariant_and_amplitude_is_scale_equivariant():
    base = decompose_residual(RESIDUAL)
    for scale in (0.25, 1.0, 7.5):
        scaled = decompose_residual(RESIDUAL * scale)
        assert torch.allclose(scaled.shape_positive, base.shape_positive, atol=1e-6)
        assert torch.allclose(scaled.shape_negative, base.shape_negative, atol=1e-6)
        assert torch.allclose(scaled.mass_positive, base.mass_positive * scale, atol=1e-5)
        assert torch.allclose(scaled.mass_negative, base.mass_negative * scale, atol=1e-5)


def test_zero_mass_days_are_finite_and_contribute_nothing():
    targets = decompose_residual(torch.zeros(2, 6))
    assert torch.isfinite(targets.shape_positive).all()
    assert torch.isfinite(targets.shape_negative).all()
    assert (targets.mass_positive == 0).all() and (targets.mass_negative == 0).all()
    assert torch.allclose(targets.shape_positive.sum(-1), torch.ones(2), atol=1e-6)

    one_sided = decompose_residual(torch.tensor([[2.0, 0.0, 1.0]]))
    assert one_sided.mass_negative.item() == 0.0
    assert torch.allclose(one_sided.shape_positive,
                          torch.tensor([[2 / 3, 0.0, 1 / 3]]), atol=1e-6)


# --- deterministic preprocessing / legality ----------------------------------

def test_masked_out_steps_do_not_leak_into_robust_coordinates():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, 5)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]).reshape(1, 5)
    before = within_day_robust_normalize(x, mask)
    perturbed = x.clone()
    perturbed[0, 2] = 1e6
    after = within_day_robust_normalize(perturbed, mask)
    assert torch.allclose(before[mask.bool()], after[mask.bool()], atol=1e-6)


def test_complete_forecast_feature_tensor_enters_one_shared_base_token():
    for n_features in (0, 1, 3, 9):
        n_calendar = 3
        batch = _batch(n_features=n_features, n_calendar=n_calendar)
        expected = 2 + 2 * n_features + n_calendar
        assert batch.base_tokens.shape[-1] == expected
        config = default_config(n_features, n_calendar)
        assert config.d_base_tokens == expected
        with torch.no_grad():
            out = SignedMassRepairModel(config).eval()(batch)
        assert out.correction.shape == batch.host.shape


def test_missing_feature_is_zero_filled_only_with_explicit_availability_mask():
    host = torch.tensor([[10.0, 11.0, 12.0]])
    features = torch.tensor([[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]]])
    fmask = torch.ones_like(features)
    fmask[0, 1, 0] = 0.0

    raw = RawInputs(host=host, valid_mask=torch.ones_like(host),
                    forecast_features=features,
                    forecast_feature_mask=fmask)
    batch = DeterministicFeatureBuilder().build(raw)
    # layout: [host, f0, f1, valid, mask_f0, mask_f1]
    assert batch.base_tokens[0, 1, 1].item() == 0.0
    assert batch.base_tokens[0, 1, 4].item() == 0.0

    changed = features.clone()
    changed[0, 1, 0] = 1e9
    raw2 = RawInputs(host=host, valid_mask=torch.ones_like(host),
                     forecast_features=changed,
                     forecast_feature_mask=fmask)
    batch2 = DeterministicFeatureBuilder().build(raw2)
    assert torch.allclose(batch.base_tokens, batch2.base_tokens)


def test_train_frozen_scaler_ignores_unavailable_feature_entries():
    x = torch.tensor([[1.0, 1000.0], [2.0, 4.0], [3.0, 6.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    scaler = TrainFrozenScaler().fit(x, mask)
    transformed = scaler.transform(x, mask)
    assert transformed[0, 1].item() == 0.0
    assert torch.isfinite(transformed).all()


def test_all_calendar_coordinates_are_preserved_not_just_the_first():
    raw = _raw(n_features=2, n_calendar=3)
    batch = DeterministicFeatureBuilder().build(raw)
    numeric_width = 1 + 2
    assert torch.allclose(batch.base_tokens[..., numeric_width:numeric_width + 3],
                          raw.calendar)


def test_shape_history_removes_scale_but_amplitude_history_keeps_it():
    torch.manual_seed(2)
    r = torch.randn(2, 5, 24)
    builder = DeterministicFeatureBuilder(residual_scale=3.0)
    s1 = builder.build_shape_history(r)
    s2 = builder.build_shape_history(7.0 * r)
    assert torch.allclose(s1, s2, atol=2e-5)

    a1 = builder.build_amplitude_history(r)
    a2 = builder.build_amplitude_history(7.0 * r)
    assert torch.allclose(a2, 7.0 * a1, atol=2e-5)


def test_training_only_residual_scale_is_positive_and_scale_equivariant():
    r = torch.tensor([-2.0, -1.0, 0.0, 3.0, 5.0])
    s = fit_residual_scale(r)
    assert s > 0
    assert abs(fit_residual_scale(4 * r) - 4 * s) < 1e-5


# --- local temporal encoder ---------------------------------------------------

def test_local_tcn_never_looks_forward_in_time():
    tcn = LocalTCN(channels=3, hidden=8).eval()
    x = torch.randn(2, 3, 12)
    with torch.no_grad():
        base = tcn(x)
        perturbed = x.clone()
        perturbed[:, :, 7:] += 100.0
        after = tcn(perturbed)
    assert torch.allclose(base[:, :, :7], after[:, :, :7], atol=1e-5)


# --- Shape metric -------------------------------------------------------------

def test_wasserstein1_matches_a_hand_computed_toy():
    p = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    assert shape_wasserstein1(p, q, reduction="none").item() == 2.0
    assert shape_wasserstein1(p, p, reduction="none").item() == 0.0


# --- pooled MAE calibration ---------------------------------------------------

def test_pooled_alpha_matches_a_brute_force_nonnegative_grid():
    candidate = torch.tensor([2.0, -3.0, 0.0, 4.0, 1.0])
    residual = torch.tensor([1.0, -1.0, 5.0, 3.0, 0.5])
    alpha = fit_mae_scalar(candidate, residual)

    grid = torch.linspace(0.0, 3.0, 300_001)
    losses = torch.stack([(residual - g * candidate).abs().sum() for g in grid])
    best = grid[losses.argmin()].item()
    assert abs(alpha - best) < 1e-3
    assert fit_mae_scalar(torch.zeros(4), torch.tensor([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_registered_alpha_clips_a_negative_unconstrained_solution_to_zero():
    candidate = torch.tensor([1.0, 1.0, 1.0])
    residual = torch.tensor([-3.0, -2.0, -1.0])
    assert fit_mae_scalar(candidate, residual) == 0.0
    assert fit_mae_scalar(candidate, residual, nonnegative=False) < 0.0


def test_apply_mae_scalar_is_identity_at_one_and_rejects_negative_alpha():
    candidate = torch.tensor([1.0, -2.0, 3.0])
    assert torch.allclose(apply_mae_scalar(candidate, 1.0), candidate)
    assert torch.allclose(apply_mae_scalar(candidate, 0.5), candidate * 0.5)
    try:
        apply_mae_scalar(candidate, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative registered alpha must be refused")


# --- optional KNN context remains closed -------------------------------------

def test_shape_memory_bank_rejects_future_or_unpermitted_windows():
    bank = ShapeMemoryBank(is_permitted=lambda i: i < 50, enabled=True, k=2)
    bank.add(np.zeros(4), np.ones(4) / 4, np.ones(4) / 4,
             revealed_index=10, origin_index=20)
    for kwargs in (dict(revealed_index=20, origin_index=20),
                   dict(revealed_index=25, origin_index=20),
                   dict(revealed_index=60, origin_index=70)):
        try:
            bank.add(np.zeros(4), np.ones(4) / 4, np.ones(4) / 4, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"bank accepted an illegal window: {kwargs}")
    assert len(bank) == 1


def test_knn_context_is_disabled_by_default_and_returns_none():
    bank = ShapeMemoryBank(is_permitted=lambda i: True, enabled=True)
    bank.add(np.zeros(4), np.ones(4) / 4, np.ones(4) / 4,
             revealed_index=1, origin_index=2)
    provider = KNNShapeContextProvider(bank)
    assert provider(np.zeros((1, 4))) is None
    config = default_config(3, 2)
    assert config.knn_enabled is False and config.shape_context_dim == 0


# --- model generality ---------------------------------------------------------

def test_one_recipe_serves_several_value_regimes_without_market_logic():
    n_features, n_calendar = 2, 1
    model = _model(n_features, n_calendar)

    def regime(scale, offset, seed):
        torch.manual_seed(seed)
        host = torch.randn(3, 24) * scale + offset
        features = torch.randn(3, 24, n_features) * scale + offset
        return DeterministicFeatureBuilder().build(RawInputs(
            host=host,
            valid_mask=torch.ones(3, 24),
            forecast_features=features,
            calendar=torch.zeros(3, 24, n_calendar),
            past_residual=torch.randn(3, 7, 24) * scale,
        ))

    outputs = [model(regime(2.0, 10.0, 1)), model(regime(40.0, 300.0, 2))]
    for out in outputs:
        assert out.correction.shape == (3, 24)
        assert torch.isfinite(out.correction).all()
    assert not torch.allclose(outputs[0].correction, outputs[1].correction)


def test_forward_accepts_any_horizon_and_is_not_hardcoded_to_24():
    n_features, n_calendar = 2, 1
    model = _model(n_features, n_calendar)
    for horizon in (1, 12, 24, 48, 96):
        batch = _batch(batch=2, horizon=horizon, n_features=n_features,
                       n_calendar=n_calendar, seed=horizon)
        with torch.no_grad():
            out = model(batch)
        assert out.correction.shape == (2, horizon)
        assert torch.allclose(out.shape.positive.sum(-1), torch.ones(2), atol=1e-5)


def test_model_runs_without_any_history_windows():
    batch = _batch(batch=3, windows=0, n_features=2, n_calendar=1)
    assert batch.shape_history.shape[1] == 0
    assert batch.amplitude_history.shape[1] == 0
    with torch.no_grad():
        out = _model(2, 1)(batch)
    assert torch.isfinite(out.correction).all()


def test_predict_applies_calibration_scalar_to_host():
    batch = _batch()
    model = _model()
    with torch.no_grad():
        out = model(batch)
        assert torch.allclose(model.predict(batch), batch.host + out.correction)
        assert torch.allclose(model.predict(batch, 0.25), batch.host + 0.25 * out.correction)
        assert torch.allclose(model.predict(batch, 0.0), batch.host)


# --- training-only rare-mass ordering ----------------------------------------

def test_rare_mass_spread_uses_every_training_sample_exactly_once():
    sampler = RareMassSampler(RareMassSamplerConfig(enabled=True, seed=3)).fit(
        torch.arange(100.0) % 7, torch.arange(100.0) % 3)
    batches = sampler.batches(5, seed=3)
    concatenated = torch.cat([torch.as_tensor(b) for b in batches])
    assert sorted(concatenated.tolist()) == list(range(100))
    assert sum(sampler.stratum_counts().values()) == 100


def test_sampling_is_gated_and_has_no_inference_surface():
    try:
        RareMassSampler(RareMassSamplerConfig()).batches(2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("disabled sampler must refuse to run")

    sampler = RareMassSampler(RareMassSamplerConfig(enabled=True)).fit(
        torch.rand(10), torch.rand(10))
    try:
        sampler.oversampled_indices(5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("oversampling must be explicitly enabled")

    forbidden = ("transform", "predict", "apply", "correct")
    assert not any(name in dir(RareMassSampler) for name in forbidden)


def test_oversampling_positive_and_negative_pools_do_not_mix_ordinary_strata():
    # High positive = indices 8,9; high negative = 0,1 under q=0.8.
    pos = torch.arange(10.0)
    neg = torch.arange(10.0).flip(0)

    pos_sampler = RareMassSampler(RareMassSamplerConfig(
        enabled=True, mode="oversample", upper_quantile=0.8,
        positive_fraction=1.0, negative_fraction=0.0, seed=1)).fit(pos, neg)
    pos_idx = pos_sampler.oversampled_indices(50, seed=1)
    assert set(pos_idx.tolist()).issubset({8, 9})

    neg_sampler = RareMassSampler(RareMassSamplerConfig(
        enabled=True, mode="oversample", upper_quantile=0.8,
        positive_fraction=0.0, negative_fraction=1.0, seed=1)).fit(pos, neg)
    neg_idx = neg_sampler.oversampled_indices(50, seed=1)
    assert set(neg_idx.tolist()).issubset({0, 1})


def test_registered_removal_flags_are_executable_without_source_edits():
    batch = _batch(n_features=2, n_calendar=1)
    config = default_config(
        2, 1,
        shape_semantic_context=False,
        use_tcn=False,
        untied_amplitude_heads=False,
    )
    model = SignedMassRepairModel(config).eval()
    with torch.no_grad():
        out = model(batch)
    assert out.correction.shape == batch.host.shape
    assert torch.allclose(out.amplitude.positive, out.amplitude.negative), \
        "tied-head ablation must use the same mass readout for both signs"


def test_core_public_surface_imports_and_exports_are_resolvable():
    for name in core.__all__:
        assert hasattr(core, name), f"{name} is exported but missing"
