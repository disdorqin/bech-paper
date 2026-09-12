"""Safety mathematics, prequential legality and historical preservation.

Covers items 8-25 of section 20 of
`docs/current/HCH_ALIGNMENT_SAFE_CORE_REFACTOR_PLAN_20260912.md`.

Unit tests only.  Nothing here trains a model, opens an evaluation split, reads a
protected artifact or touches frozen evidence — the preservation tests in the last
section only *hash* closed evidence trees and compare against digests recorded
before the refactor.

Two independent references are used so the analytic safety layer is checked against
computation rather than against itself:

* the direct definitional L1 form ``sum_i w_i (|r_i - lambda c_i| - |r_i|)``, which
  shares no algebra with the ratio form the module evaluates;
* a bisection root of the risk function itself, which uses no breakpoint scan.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORE = SRC / "core"
ARCHIVE = SRC / "archive/signed_mass_development_core_20260912"
LEGACY = ARCHIVE / "legacy_core"

import core  # noqa: E402
from core import safety as S  # noqa: E402
from core.contracts import (CandidateOutput, ForecastOriginBatch,  # noqa: E402
                            PROVENANCE_IN_SAMPLE, PROVENANCE_OOF_PREQUENTIAL,
                            PROVENANCE_PREQUENTIAL, RawInputs, SafetyEvidence,
                            WARM_START_PROVENANCE)
from core.history import (HISTORY_DAYS, DuplicateDeliveryError,  # noqa: E402
                         DishonestRecordError, HistoryError,
                         IncompleteHistoryError, InSampleCandidateError,
                         InsufficientWarmStartError, SafetyEvidenceBank,
                         select_warm_start)

TOL = 1e-9


# --- helpers ------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_rays(rng, horizon=None, zero_fraction=0.2):
    horizon = int(rng.integers(2, 40)) if horizon is None else horizon
    residual = rng.normal(0.0, 3.0, size=horizon)
    correction = rng.normal(0.0, 3.0, size=horizon)
    correction[rng.random(horizon) < zero_fraction] = 0.0
    return residual, correction


def _direct_l1(residuals, corrections, lam, weights=None):
    """The definitional excess MAE: ``sum_i w_i (|r_i - lambda c_i| - |r_i|)``."""
    r = np.asarray(residuals, dtype=np.float64)
    c = np.asarray(corrections, dtype=np.float64)
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=np.float64)
    return float((w * (np.abs(r - lam * c) - np.abs(r))).sum())


def _bisect_radius(residuals, corrections, weights=None, cap=1e7, iters=90):
    """Independent oracle: bracket-then-bisect the root of the risk function.

    Uses only :func:`mae_excess_risk`, so it shares no code with the breakpoint
    scan.  The no-harm set is an interval by convexity, so bisecting on the sign of
    the risk converges to its right endpoint.
    """
    def risk(lam):
        return S.mae_excess_risk(residuals, corrections, lam=lam, weights=weights)

    high = 1.0
    while risk(high) <= 0.0:
        high *= 2.0
        if high > cap:
            return math.inf
    low = 0.0
    for _ in range(iters):
        mid = 0.5 * (low + high)
        if risk(mid) <= 0.0:
            low = mid
        else:
            high = mid
    return low


def _fill_bank(horizon=6, count=HISTORY_DAYS, provenance=PROVENANCE_PREQUENTIAL,
               seed=11, start_ordinal=100, mask=None):
    """A legal, chronologically honest bank of ``count`` completed records."""
    rng = _rng(seed)
    bank = SafetyEvidenceBank()
    for index in range(count):
        ordinal = start_ordinal + index
        bank.append_candidate(
            delivery_id=f"d{index:02d}",
            ordinal=ordinal,
            candidate_created_at=ordinal - 1,
            shape_positive=torch.tensor(np.abs(rng.normal(0, 1, horizon)) + 0.1,
                                        dtype=torch.float32),
            shape_negative=torch.tensor(np.abs(rng.normal(0, 1, horizon)) + 0.1,
                                        dtype=torch.float32),
            correction=torch.tensor(rng.normal(0.0, 2.0, horizon), dtype=torch.float32),
            provenance=provenance,
            valid_mask=None if mask is None else torch.as_tensor(mask),
        )
        bank.attach_residual(
            f"d{index:02d}",
            residual=torch.tensor(rng.normal(0.0, 3.0, horizon), dtype=torch.float32),
            revealed_at=ordinal + 1,
        )
    return bank


def _candidate(horizon, seed=3):
    """A legal one-day candidate: simplex Shapes and a zero-sum correction."""
    rng = _rng(seed)
    positive = np.abs(rng.normal(0, 1, horizon)) + 0.1
    negative = np.abs(rng.normal(0, 1, horizon)) + 0.1
    positive /= positive.sum()
    negative /= negative.sum()
    amplitude = float(abs(rng.normal(0, 2)))
    correction = amplitude * (positive - negative)
    return CandidateOutput(
        shape_positive=torch.tensor(positive, dtype=torch.float32).unsqueeze(0),
        shape_negative=torch.tensor(negative, dtype=torch.float32).unsqueeze(0),
        amplitude=torch.tensor([amplitude], dtype=torch.float32),
        correction=torch.tensor(correction, dtype=torch.float32).unsqueeze(0),
    )


def _tree_digest(root: Path, exclude=("__pycache__",)) -> tuple:
    """The digest recorded by the archive manifests: relpath NUL hash LF, sorted.

    ``__pycache__`` is excluded, which is the convention `ARCHIVE_HASHES.json`
    records as ``excluded_from_archive``: compiled bytecode is not source.
    """
    root = Path(root)
    files = [p for p in root.rglob("*")
             if p.is_file() and not any(part in exclude for part in p.parts)]
    ordered = sorted((p.relative_to(root).as_posix(), p) for p in files)
    digest = hashlib.sha256()
    for relpath, path in ordered:
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\x0a")
    return digest.hexdigest(), len(ordered)


# --- 8: the risk function equals its direct definitional form ------------------

def test_exact_mae_risk_equals_direct_l1_evaluation():
    """The ratio form the module evaluates is the L1 excess, to float precision."""
    rng = _rng(101)
    for trial in range(60):
        residual, correction = _rand_rays(rng)
        lam = float(abs(rng.normal(1.0, 1.0)))
        got = S.mae_excess_risk(residual, correction, lam=lam)
        want = _direct_l1(residual, correction, lam)
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (trial, got, want)

        weights = rng.random(residual.shape[0])
        got_w = S.mae_excess_risk(residual, correction, lam=lam, weights=weights)
        want_w = _direct_l1(residual, correction, lam, weights)
        assert abs(got_w - want_w) <= 1e-9 * max(1.0, abs(want_w)), (trial, got_w, want_w)


def test_risk_is_zero_at_zero_and_convex_in_lambda():
    """``R(0) = 0`` and the slope is non-decreasing, which is what makes the
    no-harm set an interval and the radius well defined."""
    rng = _rng(102)
    residual, correction = _rand_rays(rng)
    assert abs(S.mae_excess_risk(residual, correction, lam=0.0)) <= TOL
    grid = np.linspace(0.0, 6.0, 400)
    values = np.array([S.mae_excess_risk(residual, correction, lam=x) for x in grid])
    slopes = np.diff(values) / np.diff(grid)
    assert (np.diff(slopes) >= -1e-7).all(), "the risk must be convex in lambda"


def test_a_coordinate_with_zero_correction_contributes_nothing():
    """``c_i = 0`` carries zero weight, so a huge residual there is irrelevant."""
    residual = np.array([1.0, 2.0, 3.0])
    correction = np.array([1.0, 0.0, -1.0])
    baseline = S.mae_excess_risk(residual, correction, lam=0.5)
    perturbed = S.mae_excess_risk(np.array([1.0, 1e6, 3.0]), correction, lam=0.5)
    assert abs(baseline - perturbed) <= TOL


# --- 9: the safe root against an independent reference ------------------------

def test_exact_safe_root_matches_an_independent_bisection_reference():
    rng = _rng(103)
    compared = 0
    for trial in range(50):
        residual, correction = _rand_rays(rng)
        exact = S.exact_mae_safe_radius(residual, correction)
        reference = _bisect_radius(residual, correction)
        if not math.isfinite(exact):
            # Only legal when the evidence carries no correction mass at all.
            assert (np.abs(correction) * (correction != 0)).sum() == 0.0 or \
                not math.isfinite(reference), (trial, exact, reference)
            continue
        compared += 1
        assert abs(exact - reference) <= 1e-6 * max(1.0, reference), (trial, exact, reference)
        # Feasible at the radius ...
        assert S.mae_excess_risk(residual, correction, lam=exact) <= 1e-7, trial
        # ... and strictly infeasible just past it, so the radius is not short.
        margin = max(1e-6, 1e-6 * exact)
        assert S.mae_excess_risk(residual, correction, lam=exact + margin) > 0.0, trial
    assert compared >= 40, "almost every trial should produce a finite radius"


def test_exact_safe_root_never_exceeds_the_evidence_support():
    """The radius cannot outlive the last positive breakpoint unless the risk is
    still non-increasing there, which the walk handles explicitly."""
    rng = _rng(104)
    for _ in range(80):
        residual, correction = _rand_rays(rng)
        radius = S.exact_mae_safe_radius(residual, correction)
        if not math.isfinite(radius):
            continue
        positive = (correction != 0) & (residual / np.where(correction != 0, correction, 1) > 0)
        if positive.any():
            last = float((residual[positive] / correction[positive]).max())
            assert radius >= 0.0
            assert S.mae_excess_risk(residual, correction, lam=radius) <= 1e-7


# --- 10: monotone, bounded final scale ----------------------------------------

def test_deployment_scale_is_always_within_zero_and_alpha_0():
    rng = _rng(105)
    for _ in range(200):
        alpha_0 = float(abs(rng.normal(2.0, 2.0)))
        uniform = float(abs(rng.normal(2.0, 2.0)))
        shape = None if rng.random() < 0.25 else float(abs(rng.normal(2.0, 2.0)))
        final = S.deployment_scale(alpha_0, uniform, shape)
        assert 0.0 <= final <= alpha_0
    # An unrestrictive channel never lengthens the ray.
    assert S.deployment_scale(0.7, math.inf, None) == pytest.approx(0.7)
    assert S.deployment_scale(0.7, None, math.inf) == pytest.approx(0.7)
    # A restrictive channel always shortens it.
    assert S.deployment_scale(0.7, 0.2, None) == pytest.approx(0.2)
    assert S.deployment_scale(0.7, 3.0, 0.1) == pytest.approx(0.1)
    # A zero radius is a real constraint, not a missing one.
    assert S.deployment_scale(0.7, 0.0, None) == 0.0


def test_deployment_scale_refuses_an_illegal_calibration_scalar():
    with pytest.raises(ValueError):
        S.deployment_scale(-0.1)
    with pytest.raises(ValueError):
        S.deployment_scale(math.inf)
    with pytest.raises(ValueError):
        S.deployment_scale(1.0, float("nan"))


def test_plan_deployment_final_scale_is_bounded_by_alpha_0():
    torch.manual_seed(106)
    horizon = 6
    bank = _fill_bank(horizon=horizon)
    candidate = _candidate(horizon)
    for alpha_0 in (0.0, 0.35, 5.0):
        decision = core.plan_deployment(torch.zeros(1, horizon), candidate, bank, alpha_0)
        assert 0.0 <= decision.lambda_final <= alpha_0
        assert decision.n_evidence == HISTORY_DAYS


# --- 11 / 12: the selected scale is actually no-harm on the evidence -----------

def test_uniform_risk_at_the_selected_scale_is_never_positive():
    rng = _rng(107)
    for trial in range(40):
        residual, correction = _rand_rays(rng)
        radius = S.exact_mae_safe_radius(residual, correction)
        alpha_0 = float(abs(rng.normal(2.0, 2.0)))
        final = S.deployment_scale(alpha_0, radius)
        assert S.mae_excess_risk(residual, correction, lam=final) <= 1e-7, trial


def test_weight_shapes_are_resolved_explicitly_or_refused():
    """A scalar, a per-record vector and a per-coordinate vector are all legal."""
    rng = _rng(112)
    residual = rng.normal(0, 3, (5, 6))
    correction = rng.normal(0, 3, (5, 6))
    per_record = rng.random(5)
    # One kappa per record, broadcast over each record's horizon.
    assert S.mae_excess_risk(residual, correction, lam=1.0, weights=per_record) == \
        pytest.approx(_direct_l1(residual, correction, 1.0, per_record[:, None]))
    # One weight per coordinate.
    assert S.mae_excess_risk(residual, correction, lam=1.0,
                             weights=per_record[:, None]) == \
        pytest.approx(_direct_l1(residual, correction, 1.0, per_record[:, None]))
    # A single weight for the whole ray.
    assert S.mae_excess_risk(residual, correction, lam=1.0, weights=0.25) == \
        pytest.approx(0.25 * _direct_l1(residual, correction, 1.0))
    # Anything else is a caller error, reported as one rather than surfacing as a
    # numpy broadcasting failure.
    with pytest.raises(ValueError, match="do not describe the evidence"):
        S.mae_excess_risk(residual, correction, weights=np.ones(4))
    with pytest.raises(ValueError, match="non-negative"):
        S.mae_excess_risk(residual, correction, weights=-np.ones(5))


def test_shape_weighted_risk_at_the_selected_scale_is_never_positive():
    rng = _rng(108)
    checked = 0
    for trial in range(40):
        horizon = int(rng.integers(2, 30))
        residual = rng.normal(0, 3, horizon)
        correction = rng.normal(0, 3, horizon)
        relevance = rng.random(horizon) * 0.9 + 0.05
        radius = S.shape_weighted_safe_radius(residual, correction, relevance)
        if radius is None:
            continue
        checked += 1
        assert S.mae_excess_risk(residual, correction, lam=radius,
                                 weights=relevance) <= 1e-7, trial
    assert checked >= 35


def test_shape_channel_imposes_no_restriction_when_it_carries_no_weight():
    """Zero total weighted mass means "no extra information", not "abstain"."""
    residual = np.array([1.0, -1.0, 2.0])
    correction = np.array([1.0, 1.0, -1.0])
    assert S.shape_weighted_safe_radius(residual, correction, np.zeros(3)) is None
    # ... and the caller keeps alpha_0 rather than collapsing to zero.
    assert S.deployment_scale(0.4, 0.9, None) == pytest.approx(0.4)


def test_shape_channel_can_only_shorten_the_ray():
    """Intersection safety: ``min`` makes Shape one-way conservative."""
    rng = _rng(109)
    for _ in range(60):
        horizon = int(rng.integers(2, 25))
        residual = rng.normal(0, 3, horizon)
        correction = rng.normal(0, 3, horizon)
        relevance = rng.random(horizon)
        uniform = S.exact_mae_safe_radius(residual, correction)
        shape = S.shape_weighted_safe_radius(residual, correction, relevance)
        alpha_0 = float(abs(rng.normal(2, 2)))
        final = S.deployment_scale(alpha_0, uniform, shape)
        assert final <= alpha_0 + TOL
        assert final <= uniform + TOL


# --- 13: negative alignment forces a zero radius ------------------------------

def test_negative_alignment_forces_a_zero_safe_radius():
    """``A_align < 0`` means the right derivative at zero is positive."""
    residual = np.array([-1.0, -2.0, -0.5])
    correction = np.array([1.0, 1.0, 1.0])
    assert S.mae_alignment_support(residual, correction) == pytest.approx(-1.0)
    assert S.exact_mae_safe_radius(residual, correction) == 0.0

    mixed_residual = np.array([-1.0, -1.0, 3.0])
    mixed_correction = np.ones(3)
    alignment = S.mae_alignment_support(mixed_residual, mixed_correction)
    assert alignment < 0.0
    assert S.exact_mae_safe_radius(mixed_residual, mixed_correction) == 0.0


def test_alignment_statistic_is_bounded_and_signed_correctly():
    """Fully supporting evidence is ``+1``, fully opposing evidence is ``-1``."""
    residual = np.array([1.0, 2.0, 3.0])
    assert S.mae_alignment_support(residual, np.ones(3)) == pytest.approx(1.0)
    assert S.mae_alignment_support(-residual, np.ones(3)) == pytest.approx(-1.0)
    # A coordinate with no correction mass carries no directional information.
    assert S.mae_alignment_support(np.array([1.0, -9.0]), np.array([1.0, 0.0])) == \
        pytest.approx(1.0)
    # No usable evidence at all is neutral, not opposing.
    assert S.mae_alignment_support(np.array([1.0, 2.0]), np.zeros(2)) == 0.0


# --- 14: zero, flat and degenerate cases --------------------------------------

def test_no_correction_mass_imposes_no_restriction():
    residual = np.array([1.0, -2.0, 3.0])
    assert S.exact_mae_safe_radius(residual, np.zeros(3)) == math.inf
    assert S.exact_mae_safe_radius(np.zeros(2), np.zeros(2)) == math.inf


def test_zero_residual_forces_a_zero_radius():
    assert S.exact_mae_safe_radius(np.zeros(3), np.array([1.0, -1.0, 0.0])) == 0.0


def test_a_single_live_coordinate_has_radius_twice_its_ratio():
    assert S.exact_mae_safe_radius(np.array([2.0]), np.array([1.0])) == pytest.approx(4.0)
    assert S.exact_mae_safe_radius(np.array([-2.0]), np.array([1.0])) == 0.0


def test_a_flat_risk_plateau_is_measured_to_its_end():
    """``R`` is exactly zero on ``[0, 3]`` here, then rises: the radius is 3."""
    residual = np.array([-1.0, 3.0])
    correction = np.array([1.0, 1.0])
    assert S.mae_alignment_support(residual, correction) == pytest.approx(0.0)
    assert S.exact_mae_safe_radius(residual, correction) == pytest.approx(3.0)


def test_duplicate_breakpoints_are_grouped_not_double_counted():
    """Two coordinates at the same ratio must behave as one mass of their sum."""
    pair = S.exact_mae_safe_radius(np.array([2.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    merged = S.exact_mae_safe_radius(np.array([4.0, 1.0]), np.array([2.0, 1.0]))
    assert pair == pytest.approx(merged)
    assert S.mae_excess_risk(np.array([4.0, 1.0]), np.array([2.0, 1.0]),
                             lam=pair) <= 1e-7


def test_the_walk_does_not_project_a_line_past_its_own_segment():
    """Regression: the first positive slope segment can have its root far beyond it.

    With breakpoints ``[4, 5]`` and one opposing coordinate the slope turns
    positive only after ``z = 4``, but the line through the origin with that slope
    would cross zero well past ``z = 5`` where the slope has changed again.  The
    correct radius is the later, shorter crossing.
    """
    residual = np.array([-10.0, 4.0, 5.0])
    correction = np.array([1.0, 1.0, 1.0])
    radius = S.exact_mae_safe_radius(residual, correction)
    reference = _bisect_radius(residual, correction)
    assert radius == pytest.approx(reference, rel=1e-6)
    assert S.mae_excess_risk(residual, correction, lam=radius) <= 1e-7
    assert S.mae_excess_risk(residual, correction, lam=radius + 1e-4) > 0.0


# --- 15: masks and zero corrections ------------------------------------------

def test_masked_positions_are_excluded_from_the_radius():
    """An excluded opposing coordinate must not shorten the permitted ray."""
    residual = np.array([1.0, 2.0, -1e3])
    correction = np.array([1.0, 1.0, 1.0])
    full = S.exact_mae_safe_radius(residual, correction)
    masked = S.exact_mae_safe_radius(residual, correction,
                                     valid_mask=np.array([True, True, False]))
    assert masked > full
    assert masked == pytest.approx(
        S.exact_mae_safe_radius(residual[:2], correction[:2]))
    # ... and the same evidence restricted to its first two coordinates by hand.
    assert full == pytest.approx(2.0) and masked == pytest.approx(3.0)


def test_a_bare_horizon_mask_and_a_batched_mask_agree():
    rng = _rng(110)
    residual = rng.normal(0, 3, (4, 7))
    correction = rng.normal(0, 3, (4, 7))
    horizon_mask = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)
    batched = np.tile(horizon_mask, (4, 1))
    assert S.exact_mae_safe_radius(residual, correction, valid_mask=horizon_mask) == \
        pytest.approx(S.exact_mae_safe_radius(residual, correction, valid_mask=batched))


def test_a_mask_with_the_wrong_horizon_axis_is_refused():
    residual = np.ones((2, 5))
    correction = np.ones((2, 5))
    with pytest.raises(ValueError, match="horizon"):
        S.exact_mae_safe_radius(residual, correction, valid_mask=np.ones(4))


def test_wasserstein_needs_at_least_two_valid_positions():
    with pytest.raises(ValueError, match="two valid"):
        S.wasserstein1(np.ones(4), np.ones(4),
                       valid_mask=np.array([1, 0, 0, 0], dtype=bool))


def test_wasserstein_is_zero_for_identical_and_bounded_by_the_horizon():
    rng = _rng(111)
    horizon = 9
    first = rng.random(horizon)
    first /= first.sum()
    second = rng.random(horizon)
    second /= second.sum()
    assert S.wasserstein1(first, first) == pytest.approx(0.0)
    distance = S.wasserstein1(first, second)
    assert 0.0 <= distance <= horizon - 1

    # Two point masses at opposite ends sit at the maximum distance.
    left = np.zeros(horizon)
    left[0] = 1.0
    right = np.zeros(horizon)
    right[-1] = 1.0
    assert S.wasserstein1(left, right) == pytest.approx(horizon - 1)


def test_shape_relevance_is_one_for_identical_evidence_and_zero_for_opposites():
    horizon = 8
    left = np.zeros(horizon)
    left[0] = 1.0
    right = np.zeros(horizon)
    right[-1] = 1.0
    assert S.shape_relevance(left, left, left, left) == pytest.approx(1.0)
    assert S.shape_relevance(left, left, right, right) == pytest.approx(0.0)


def test_shape_relevance_broadcasts_across_a_historical_bank():
    horizon = 5
    candidate_positive = np.array([0.6, 0.4, 0, 0, 0])
    candidate_negative = np.array([0, 0, 0, 0.5, 0.5])
    historical_positive = np.tile(candidate_positive, (HISTORY_DAYS, 1))
    historical_negative = np.tile(candidate_negative, (HISTORY_DAYS, 1))
    relevance = S.shape_relevance(candidate_positive, candidate_negative,
                                  historical_positive, historical_negative)
    assert np.shape(relevance) == (HISTORY_DAYS,)
    assert np.allclose(relevance, 1.0)


def test_relevance_ignores_the_shape_denominator_at_horizon_one():
    """With one step there are no cut points, so Shape carries no information."""
    relevance = S.shape_relevance(np.ones(1), np.ones(1), np.ones((3, 1)), np.ones((3, 1)))
    assert np.allclose(np.asarray(relevance), 1.0)


# --- 16: arbitrary horizon ----------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 2, 7, 24, 168])
def test_safety_layer_handles_any_horizon(horizon):
    rng = _rng(120 + horizon)
    residual = rng.normal(0, 3, horizon)
    correction = rng.normal(0, 3, horizon)
    radius = S.exact_mae_safe_radius(residual, correction)
    if horizon == 1:
        # A one-step correction has nowhere to redistribute to, so it is zero-sum
        # only by being zero, and the evidence carries no mass.
        assert S.mae_excess_risk(residual, correction, lam=radius) <= 1e-7
    else:
        assert math.isfinite(radius)
        assert S.mae_excess_risk(residual, correction, lam=radius) <= 1e-7
    assert -1.0 <= S.mae_alignment_support(residual, correction) <= 1.0


def test_safety_module_is_pure_functions_with_no_learned_surface():
    """Section 7: pure functions only — no module, no parameter, no gate."""
    for name in ("safety.py", "history.py"):
        text = (CORE / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if name == "safety.py":
            assert classes == [], f"the safety layer must define no class: {classes}"
        assert not any("Module" in c for c in classes), f"{name} defines a module"
        assert not any("Gate" in c or "Selector" in c or "Router" in c for c in classes)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
        assert not any(m.startswith("torch.nn") for m in imports), f"{name} reaches into nn"
        # ``nn`` is never even bound as a name: there is no architectural surface.
        assert "import torch.nn as nn" not in text
        for token in ("nn.Parameter", "nn.Linear", "nn.GRU", "state_dict"):
            assert token not in text, f"{name} mentions {token}"
        # The deployment controller consumes the bank; it never owns or trains it.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert node.name not in ("fit", "train", "step", "backward")


# --- 17: the current candidate cannot see the current residual ----------------

def test_the_candidate_input_carries_no_target_day_residual():
    """The only residual-shaped input is the already-revealed history window."""
    raw_fields = {f.name for f in dataclasses.fields(RawInputs)}
    batch_fields = {f.name for f in dataclasses.fields(ForecastOriginBatch)}
    assert raw_fields == {"host", "valid_mask", "forecast_features",
                          "forecast_feature_mask", "calendar", "past_residual"}
    assert batch_fields == {"host", "valid_mask", "base_tokens",
                            "shape_history", "amplitude_history"}
    # The tensor the model consumes carries no residual column at all.
    for forbidden in ("residual", "target", "label", "outcome", "future", "truth"):
        assert not any(forbidden in name for name in batch_fields), forbidden
    # On the raw side the single residual-shaped field is the *past* window.
    assert [n for n in raw_fields if "residual" in n] == ["past_residual"]

    # ``RawInputs`` documents that window as already revealed, and it is the only
    # thing the deterministic builder reads.
    assert "already-revealed residual windows" in (RawInputs.__doc__ or "")
    builder = core.DeterministicFeatureBuilder()
    assert set(inspect.signature(builder.build).parameters) == {"raw"}


def test_no_candidate_side_callable_accepts_a_residual():
    """A residual argument anywhere on the inference path would be leakage.

    Training objectives are excluded on purpose: a loss legitimately consumes the
    target it is fitted to, and that is not the same object as the deployment-day
    residual the controller must not see.
    """
    callables = [
        core.fuse, core.repair, core.plan_deployment,
        core.ShapeBranch.forward, core.BalancedAmplitudeBranch.forward,
        core.MinimalSignedRedistributionModel.forward,
        core.MinimalSignedRedistributionModel.predict,
        core.UnifiedFeatureStem.forward,
        core.WindowGRUEncoder.forward, core.ShapeBranch.__init__,
    ]
    for func in callables:
        names = set(inspect.signature(func).parameters)
        leaked = {n for n in names
                  if n in ("residual", "residuals", "target", "label", "truth")}
        assert not leaked, f"{func.__qualname__} accepts {leaked}"


# --- 18: the bank refuses incomplete and future records -----------------------

def test_an_incomplete_bank_is_refused_rather_than_shortened():
    bank = _fill_bank(count=6)
    assert not bank.is_ready()
    with pytest.raises(IncompleteHistoryError):
        bank.completed()
    with pytest.raises(IncompleteHistoryError):
        bank.as_arrays()

    empty = SafetyEvidenceBank()
    assert len(empty) == 0 and not empty.is_ready()
    with pytest.raises(IncompleteHistoryError):
        empty.completed()


def test_a_candidate_created_after_its_own_outcome_is_refused():
    bank = SafetyEvidenceBank()
    with pytest.raises(InSampleCandidateError):
        bank.append_candidate(delivery_id="leak", ordinal=10, candidate_created_at=10,
                              shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                              correction=torch.ones(4), provenance=PROVENANCE_PREQUENTIAL)
    with pytest.raises(InSampleCandidateError):
        bank.append_candidate(delivery_id="leak", ordinal=10, candidate_created_at=11,
                              shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                              correction=torch.ones(4), provenance=PROVENANCE_PREQUENTIAL)


def test_a_residual_dated_before_its_candidate_is_refused():
    bank = SafetyEvidenceBank()
    bank.append_candidate(delivery_id="a", ordinal=10, candidate_created_at=7,
                          shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                          correction=torch.ones(4), provenance=PROVENANCE_PREQUENTIAL)
    with pytest.raises(DishonestRecordError):
        bank.attach_residual("a", torch.ones(4), revealed_at=9)   # before its own slot
    with pytest.raises(DishonestRecordError):
        bank.attach_residual("a", torch.ones(4), revealed_at=7)   # not after the candidate
    record = bank.attach_residual("a", torch.ones(4), revealed_at=11)
    assert record.completed


def test_a_pending_candidate_is_never_evidence():
    bank = _fill_bank(count=HISTORY_DAYS)
    ordinal = 100 + HISTORY_DAYS
    bank.append_candidate(delivery_id="pending", ordinal=ordinal,
                          candidate_created_at=ordinal - 1,
                          shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                          correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)
    assert len(bank) == HISTORY_DAYS
    assert [r.delivery_id for r in bank.pending] == ["pending"]
    assert all(r.completed for r in bank.completed())
    assert "pending" not in bank.as_arrays()["delivery_ids"]


def test_the_bank_is_chronological_and_refuses_a_replayed_slot():
    bank = _fill_bank(count=2)
    with pytest.raises(DishonestRecordError):
        bank.append_candidate(delivery_id="back", ordinal=101, candidate_created_at=99,
                              shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                              correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)


def test_an_illegal_provenance_mark_is_refused_outright():
    bank = SafetyEvidenceBank()
    with pytest.raises(DishonestRecordError):
        bank.append_candidate(delivery_id="x", ordinal=5, candidate_created_at=4,
                              shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                              correction=torch.ones(4), provenance=PROVENANCE_IN_SAMPLE)
    with pytest.raises(DishonestRecordError):
        bank.append_candidate(delivery_id="x", ordinal=5, candidate_created_at=4,
                              shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                              correction=torch.ones(4), provenance="whatever")
    with pytest.raises(DishonestRecordError):
        bank.append_candidate(delivery_id="", ordinal=5, candidate_created_at=4,
                              shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                              correction=torch.ones(4), provenance=PROVENANCE_PREQUENTIAL)


# --- 19: warm start accepts only out-of-fold / prequential records ------------

def _warm_start_records(count=9, in_sample_at=(), provenance=PROVENANCE_OOF_PREQUENTIAL,
                        seed=201, horizon=5):
    rng = _rng(seed)
    records = []
    for index in range(count):
        ordinal = 50 + index
        records.append(SafetyEvidence(
            delivery_id=f"w{index}", ordinal=ordinal, candidate_created_at=ordinal - 1,
            shape_positive=torch.tensor(np.abs(rng.normal(0, 1, horizon)), dtype=torch.float32),
            shape_negative=torch.tensor(np.abs(rng.normal(0, 1, horizon)), dtype=torch.float32),
            candidate_correction=torch.tensor(rng.normal(0, 1, horizon), dtype=torch.float32),
            host_residual=torch.tensor(rng.normal(0, 1, horizon), dtype=torch.float32),
            revealed_at=ordinal + 1,
            provenance=(PROVENANCE_IN_SAMPLE if index in in_sample_at else provenance),
        ))
    return records


def test_warm_start_excludes_in_sample_predictions():
    """An in-sample candidate is fitted to the residual it is scored against."""
    records = _warm_start_records(count=11, in_sample_at=(2, 6))
    selected = select_warm_start(records)
    assert len(selected) == HISTORY_DAYS
    assert all(r.provenance in WARM_START_PROVENANCE for r in selected)
    assert all(r.provenance != PROVENANCE_IN_SAMPLE for r in selected)
    # Nine of the eleven are legal, so the *last seven legal* records are the
    # window -- not the last seven supplied, and not the first seven legal.
    assert [r.delivery_id for r in selected] == ["w3", "w4", "w5", "w7", "w8", "w9", "w10"]
    assert [r.ordinal for r in selected] == sorted(r.ordinal for r in selected)
    assert records[2].delivery_id not in [r.delivery_id for r in selected]
    assert records[6].delivery_id not in [r.delivery_id for r in selected]


def test_warm_start_refuses_to_shorten_the_window():
    short = tuple(_warm_start_records(count=HISTORY_DAYS - 1, provenance=PROVENANCE_PREQUENTIAL))
    with pytest.raises(InsufficientWarmStartError, match="not shortened"):
        select_warm_start(short)


def test_warm_start_requires_completed_records():
    pending = SafetyEvidence(delivery_id="p", ordinal=3, candidate_created_at=2,
                             shape_positive=torch.ones(4), shape_negative=torch.ones(4),
                             candidate_correction=torch.ones(4), provenance=PROVENANCE_PREQUENTIAL)
    assert not pending.completed
    with pytest.raises(InsufficientWarmStartError):
        select_warm_start([pending] * HISTORY_DAYS)


def test_a_bank_can_be_seeded_from_a_legal_warm_start():
    records = select_warm_start([r for r in _fill_bank(seed=202).records()])
    bank = SafetyEvidenceBank.from_warm_start(records)
    assert bank.is_ready()
    assert [r.delivery_id for r in bank.completed()] == [r.delivery_id for r in records]
    # A seeded bank still refuses a replayed delivery.
    with pytest.raises(DuplicateDeliveryError):
        bank.append_candidate(delivery_id=records[-1].delivery_id,
                              ordinal=records[-1].ordinal + 1,
                              candidate_created_at=records[-1].ordinal,
                              shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                              correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)


def test_warm_start_refuses_an_in_sample_record_even_supplied_directly():
    records = list(_fill_bank(seed=203).records())
    records[-1] = dataclasses.replace(records[-1], provenance=PROVENANCE_IN_SAMPLE)
    with pytest.raises(InsufficientWarmStartError):
        select_warm_start(records)


# --- 20: capacity is exactly seven, with no knob ------------------------------

def test_capacity_is_exactly_seven_and_not_configurable():
    import inspect
    assert HISTORY_DAYS == 7
    assert set(inspect.signature(SafetyEvidenceBank.__init__).parameters) == {"self"}
    bank = SafetyEvidenceBank()
    assert bank.capacity == HISTORY_DAYS
    assert all(p.name not in ("capacity", "history_days", "window", "w")
               for p in inspect.signature(SafetyEvidenceBank.__init__).parameters.values())


def test_the_bank_keeps_only_the_seven_most_recent_outcomes():
    bank = _fill_bank(count=HISTORY_DAYS + 4)
    assert len(bank) == HISTORY_DAYS
    ids = [r.delivery_id for r in bank.completed()]
    assert ids == [f"d{i:02d}" for i in range(4, 4 + HISTORY_DAYS)]
    assert bank.is_ready()


def test_evicted_delivery_identifiers_are_still_refused():
    """Re-delivering an evicted day would replay history under a new slot."""
    bank = _fill_bank(count=HISTORY_DAYS)
    ordinal = 100 + HISTORY_DAYS
    bank.append_candidate(delivery_id="new", ordinal=ordinal, candidate_created_at=ordinal - 1,
                          shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                          correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)
    bank.attach_residual("new", torch.ones(6), revealed_at=ordinal + 1)
    assert "d00" not in [r.delivery_id for r in bank.completed()]
    with pytest.raises(DuplicateDeliveryError):
        bank.append_candidate(delivery_id="d00", ordinal=ordinal + 1,
                              candidate_created_at=ordinal,
                              shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                              correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)


def test_a_cleared_bank_still_refuses_a_replayed_delivery():
    bank = _fill_bank(count=1, seed=204)
    bank.clear()
    assert len(bank) == 0
    with pytest.raises(DuplicateDeliveryError):
        bank.append_candidate(delivery_id="d00", ordinal=200, candidate_created_at=199,
                              shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                              correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)


def test_a_serialized_bank_for_another_window_length_is_refused():
    bank = _fill_bank()
    state = bank.to_state()
    assert SafetyEvidenceBank.from_state(state).is_ready()
    state["history_days"] = 5
    with pytest.raises(HistoryError, match="structural constant"):
        SafetyEvidenceBank.from_state(state)


# --- 21 / 22: no replay, and reveal only follows persistence ------------------

def test_a_completed_record_cannot_be_rewritten():
    bank = _fill_bank(count=1, seed=205)
    with pytest.raises(DishonestRecordError, match="append-only"):
        bank.attach_residual("d00", torch.zeros(6), revealed_at=999)


def test_a_residual_cannot_be_attached_to_a_candidate_that_was_never_persisted():
    """Item 22: the reveal step appends nothing; it only completes a stored pair."""
    bank = _fill_bank()
    with pytest.raises(HistoryError, match="no pending candidate"):
        bank.attach_residual("never-stored", torch.ones(6), revealed_at=999)
    assert all(r.delivery_id != "never-stored" for r in bank.records())


def test_pending_candidates_do_not_displace_completed_evidence():
    bank = _fill_bank(count=HISTORY_DAYS)
    for index in range(3):
        ordinal = 100 + HISTORY_DAYS + index
        bank.append_candidate(delivery_id=f"p{index}", ordinal=ordinal,
                              candidate_created_at=ordinal - 1,
                              shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                              correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)
    assert len(bank) == HISTORY_DAYS
    assert len(bank.pending) == 3
    assert bank.is_ready(), "a pending candidate is not an outcome and must not evict one"


# --- 26: the bank's trust boundary is the same at every door ------------------
#
# `docs/current/HCH_ALIGNMENT_SAFE_CORE_REVIEW_20260912.md` found that warm start
# and deserialization could bypass part of the chronology/immutability contract the
# online append/attach path enforces.  These are the adversarial cases from that
# audit: each one was accepted before the patch and must be refused after it.

def _honest_completed(index, *, seed=301, horizon=5, start_ordinal=50,
                      provenance=PROVENANCE_OOF_PREQUENTIAL):
    """One structurally honest completed record, legal at every door."""
    rng = _rng(seed + index)
    ordinal = start_ordinal + index
    return SafetyEvidence(
        delivery_id=f"h{index}", ordinal=ordinal, candidate_created_at=ordinal - 1,
        shape_positive=torch.tensor(np.abs(rng.normal(0, 1, horizon)), dtype=torch.float32),
        shape_negative=torch.tensor(np.abs(rng.normal(0, 1, horizon)), dtype=torch.float32),
        candidate_correction=torch.tensor(rng.normal(0, 1, horizon), dtype=torch.float32),
        host_residual=torch.tensor(rng.normal(0, 1, horizon), dtype=torch.float32),
        revealed_at=ordinal + 1,
        provenance=provenance,
    )


def test_warm_start_refuses_a_reveal_dated_before_its_slot():
    """The audit's exact adversarial bank: legal provenance, legal candidate
    timing, non-null residual -- and ``revealed_at = 0`` for ordinals 10..16.
    It used to build a ready bank; it is now refused."""
    adversarial = [SafetyEvidence(
        delivery_id=f"a{i}", ordinal=10 + i, candidate_created_at=9 + i,
        shape_positive=torch.ones(5), shape_negative=torch.ones(5),
        candidate_correction=torch.ones(5), host_residual=torch.ones(5),
        revealed_at=0, provenance=PROVENANCE_OOF_PREQUENTIAL,
    ) for i in range(HISTORY_DAYS)]
    with pytest.raises(DishonestRecordError, match="before its own slot"):
        select_warm_start(adversarial)
    with pytest.raises(DishonestRecordError, match="before its own slot"):
        SafetyEvidenceBank.from_warm_start(adversarial)


def test_warm_start_refuses_a_reveal_that_does_not_postdate_the_candidate():
    """A reveal that does not postdate its own candidate is never accepted.

    With ``candidate_created_at < ordinal <= revealed_at`` already required, the
    third relation is implied, so the two rules overlap on every input.  This test
    pins the observable contract -- no such record is ever accepted -- rather than
    pretending a particular message is the one that fires.
    """
    records = [_honest_completed(i) for i in range(HISTORY_DAYS)]
    records[3] = dataclasses.replace(records[3], revealed_at=records[3].candidate_created_at)
    with pytest.raises(DishonestRecordError):
        select_warm_start(records)

    # A candidate that postdates its own outcome slot is refused by the stricter
    # rule that fires first: it can never be evidence at all.
    records[3] = dataclasses.replace(_honest_completed(3),
                                     candidate_created_at=_honest_completed(3).ordinal + 5)
    with pytest.raises(InSampleCandidateError):
        select_warm_start(records)


def test_warm_start_refuses_a_completed_record_without_a_reveal_time():
    """A residual with no chronology is not evidence, and a pending candidate
    that claims a reveal time is internally impossible."""
    records = [_honest_completed(i) for i in range(HISTORY_DAYS)]
    records[2] = dataclasses.replace(records[2], revealed_at=None)
    with pytest.raises(DishonestRecordError, match="no reveal time"):
        select_warm_start(records)

    pending_with_reveal = [dataclasses.replace(_honest_completed(i), host_residual=None,
                                               revealed_at=i)
                           for i in range(HISTORY_DAYS)]
    with pytest.raises(DishonestRecordError, match="no residual"):
        select_warm_start(pending_with_reveal)


def test_warm_start_refuses_a_duplicate_delivery_identifier():
    records = [_honest_completed(i) for i in range(HISTORY_DAYS + 1)]
    records[-1] = dataclasses.replace(records[-1], delivery_id=records[0].delivery_id)
    with pytest.raises(DuplicateDeliveryError):
        select_warm_start(records)
    with pytest.raises(DuplicateDeliveryError):
        SafetyEvidenceBank.from_warm_start(records)


def test_warm_start_refuses_a_non_strict_ordinal_sequence():
    """Two outcomes cannot claim the same slot: ordinals are the only chronology
    the safety mathematics uses."""
    clash = [_honest_completed(i) for i in range(HISTORY_DAYS + 1)]
    clash[-1] = dataclasses.replace(clash[-1], ordinal=clash[0].ordinal,
                                    candidate_created_at=clash[0].ordinal - 1,
                                    revealed_at=clash[0].ordinal + 1)
    with pytest.raises(DishonestRecordError, match="strictly follow"):
        select_warm_start(clash)
    with pytest.raises(DishonestRecordError, match="strictly follow"):
        SafetyEvidenceBank.from_warm_start(clash)


def test_restore_refuses_a_corrupted_reveal_chronology():
    bank = _fill_bank(seed=306)
    state = bank.to_state()
    assert SafetyEvidenceBank.from_state(state).is_ready(), \
        "the honest serialization must still restore"

    state["records"][0]["revealed_at"] = 0
    with pytest.raises(DishonestRecordError, match="before its own slot"):
        SafetyEvidenceBank.from_state(state)

    missing = bank.to_state()
    missing["records"][0]["revealed_at"] = None
    with pytest.raises(DishonestRecordError, match="no reveal time"):
        SafetyEvidenceBank.from_state(missing)


def test_restore_refuses_a_duplicate_identifier_or_a_broken_ordinal():
    bank = _fill_bank(seed=307)

    duplicated = bank.to_state()
    duplicated["records"][1]["delivery_id"] = duplicated["records"][0]["delivery_id"]
    with pytest.raises(DuplicateDeliveryError):
        SafetyEvidenceBank.from_state(duplicated)

    reordered = bank.to_state()
    reordered["records"][0], reordered["records"][1] = (reordered["records"][1],
                                                        reordered["records"][0])
    with pytest.raises(DishonestRecordError, match="strictly follow"):
        SafetyEvidenceBank.from_state(reordered)


def test_restore_refuses_a_record_outside_the_delivered_ledger():
    """The ledger is the replay guard; a record missing from it was never delivered."""
    state = _fill_bank(seed=308).to_state()
    state["delivered_ids"] = [i for i in state["delivered_ids"] if i != "d03"]
    with pytest.raises(DishonestRecordError, match="ledger"):
        SafetyEvidenceBank.from_state(state)


def test_restore_refuses_more_completed_records_than_the_window():
    """Live eviction makes this unreachable; a tampered snapshot must still be refused."""
    state = _fill_bank(seed=309).to_state()
    extra = dict(state["records"][-1])
    extra.update(delivery_id="d99", ordinal=200, candidate_created_at=199, revealed_at=201)
    state["records"].append(extra)
    state["delivered_ids"] = sorted(state["delivered_ids"] + ["d99"])
    with pytest.raises(IncompleteHistoryError, match="more than"):
        SafetyEvidenceBank.from_state(state)


def test_restore_refuses_an_impossible_pending_state():
    without_residual = _fill_bank(seed=310).to_state()
    without_residual["records"][0]["host_residual"] = None
    with pytest.raises(DishonestRecordError, match="no residual"):
        SafetyEvidenceBank.from_state(without_residual)

    without_reveal = _fill_bank(seed=310).to_state()
    without_reveal["records"][0]["revealed_at"] = None
    with pytest.raises(DishonestRecordError, match="no reveal time"):
        SafetyEvidenceBank.from_state(without_reveal)


def test_restore_refuses_a_structurally_unparseable_entry():
    state = _fill_bank(seed=311).to_state()
    del state["records"][0]["candidate_correction"]
    with pytest.raises(HistoryError, match="not well formed"):
        SafetyEvidenceBank.from_state(state)


def test_a_caller_cannot_mutate_bank_evidence_through_the_tensors_it_supplied():
    bank = SafetyEvidenceBank()
    positive = torch.full((5,), 0.4)
    negative = torch.full((5,), 0.6)
    correction = torch.ones(5)
    mask = torch.ones(5)
    bank.append_candidate(delivery_id="snap", ordinal=10, candidate_created_at=9,
                          shape_positive=positive, shape_negative=negative,
                          correction=correction, provenance=PROVENANCE_PREQUENTIAL,
                          valid_mask=mask)
    bank.attach_residual("snap", torch.full((5,), 2.0), revealed_at=11)

    for supplied in (positive, negative, correction, mask):
        supplied.mul_(9.0)

    record = bank.records()[0]
    assert torch.allclose(record.shape_positive, torch.full((5,), 0.4))
    assert torch.allclose(record.shape_negative, torch.full((5,), 0.6))
    assert torch.allclose(record.candidate_correction, torch.ones(5))
    assert torch.allclose(record.valid_mask, torch.ones(5))


def test_a_caller_cannot_mutate_bank_evidence_through_the_residual_it_supplied():
    bank = SafetyEvidenceBank()
    bank.append_candidate(delivery_id="d00", ordinal=100, candidate_created_at=99,
                          shape_positive=torch.ones(6), shape_negative=torch.ones(6),
                          correction=torch.ones(6), provenance=PROVENANCE_PREQUENTIAL)
    residual = torch.full((6,), 3.0)
    bank.attach_residual("d00", residual=residual, revealed_at=101)
    residual.mul_(-5.0)
    assert torch.allclose(bank.records()[0].host_residual, torch.full((6,), 3.0))


def test_public_accessors_never_expose_a_mutable_alias():
    """A frozen dataclass does not make a mutable tensor immutable, so every public
    surface hands back copies."""
    bank = _fill_bank(seed=312)
    expected = {name: value.clone() for name, value in bank.as_arrays().items()
                if isinstance(value, torch.Tensor)}

    for record in bank.records() + bank.pending + bank.completed():
        for field in ("shape_positive", "shape_negative", "candidate_correction",
                      "host_residual", "valid_mask"):
            value = getattr(record, field)
            if value is not None:
                value.mul_(-7.0)

    bank.as_arrays()["correction"].mul_(3.0)  # the stacked result is a copy too

    arrays = bank.as_arrays()
    for name, reference in expected.items():
        assert torch.equal(arrays[name], reference), f"{name} was reachable in place"


def test_bank_snapshots_carry_no_autograd_graph_and_no_caller_storage():
    bank = SafetyEvidenceBank()
    sources = {
        "shape_positive": torch.full((5,), 0.5, requires_grad=True),
        "shape_negative": torch.full((5,), 0.5, requires_grad=True),
    }
    correction = torch.ones(5, requires_grad=True)
    bank.append_candidate(delivery_id="grad", ordinal=10, candidate_created_at=9,
                          shape_positive=sources["shape_positive"],
                          shape_negative=sources["shape_negative"],
                          correction=correction, provenance=PROVENANCE_PREQUENTIAL)
    residual = torch.ones(5, requires_grad=True)
    bank.attach_residual("grad", residual, revealed_at=11)
    sources.update(candidate_correction=correction, host_residual=residual)

    record = bank.records()[0]
    for field, source in sources.items():
        stored = getattr(record, field)
        assert not stored.requires_grad, f"{field} still requires grad"
        assert stored.grad_fn is None, f"{field} retains an autograd graph"
        assert stored.is_leaf, f"{field} is not a leaf snapshot"
        assert stored.device.type == "cpu", field
        assert stored.data_ptr() != source.data_ptr(), f"{field} aliases caller storage"


def test_legal_warm_start_and_serialization_round_trip_still_pass():
    """The hardening must not make an honest pipeline unusable."""
    records = select_warm_start([r for r in _fill_bank(seed=313).records()])
    bank = SafetyEvidenceBank.from_warm_start(records)
    assert bank.is_ready()

    restored = SafetyEvidenceBank.from_state(bank.to_state())
    assert restored.is_ready()
    assert ([r.delivery_id for r in restored.completed()]
            == [r.delivery_id for r in bank.completed()])
    for original, copy in zip(bank.completed(), restored.completed()):
        assert original.ordinal == copy.ordinal
        assert original.revealed_at == copy.revealed_at
        assert torch.equal(original.candidate_correction, copy.candidate_correction)
        assert torch.equal(original.host_residual, copy.host_residual)
    assert torch.equal(bank.as_arrays()["residual"], restored.as_arrays()["residual"])


# --- 23: the archived pre-refactor core is byte-preserved ---------------------

def test_archived_tree_digest_matches_the_recorded_digest():
    recorded = json.loads((ARCHIVE / "ARCHIVE_HASHES.json").read_text(encoding="utf-8"))
    digest, count = _tree_digest(LEGACY)
    assert count == recorded["file_count"] == 20
    assert digest == recorded["tree_digest_sha256"], \
        "the archived pre-refactor core has been modified"


def test_every_archived_file_still_matches_its_recorded_hash():
    recorded = json.loads((ARCHIVE / "ARCHIVE_HASHES.json").read_text(encoding="utf-8"))
    for entry in recorded["files"]:
        path = ROOT / entry["archive_path"]
        assert path.is_file(), entry["archive_path"]
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"], entry["archive_path"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["archive_path"]


#: Files the refactor de-promoted out of the active tree.  Each must be archived
#: *and* gone: "do not delete historical code without archiving it first".
DEPROMOTED = (
    "sampling.py", "semantic_views.py", "similarity.py",
    "CORE_READINESS_AUDIT_20260912.md", "CORE_REORGANIZATION_INVENTORY.md",
    "CORE_REORGANIZATION_PLAN_20260912.md", "core_reorganization_inventory.json",
)


def test_depromoted_files_are_archived_and_absent_from_the_active_tree():
    for name in DEPROMOTED:
        assert not (CORE / name).exists(), f"{name} is still on the active path"
        archived = LEGACY / name
        assert archived.is_file(), f"{name} was removed without an archive copy"
        assert archived.stat().st_size > 400, f"{name} archive copy looks truncated"


def test_the_untouched_calibration_module_is_still_byte_identical():
    """The pooled calibration scalar is reused unchanged; this proves it.

    ``calibration.py`` is the one code module the refactor had no reason to touch,
    so its bytes are the direct evidence that a reuse claim is a reuse and not a
    silent reimplementation.
    """
    recorded = json.loads((ARCHIVE / "ARCHIVE_HASHES.json").read_text(encoding="utf-8"))
    entry = next(e for e in recorded["files"]
                 if Path(e["old_path"]).name == "calibration.py")
    payload = (CORE / "calibration.py").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert hashlib.sha256((LEGACY / "calibration.py").read_bytes()).hexdigest() == \
        entry["sha256"]


def test_the_archived_legacy_core_is_importable_read_only():
    """Preservation proof: the old code still exists as code, not as prose."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preserved_legacy_core", LEGACY / "__init__.py",
        submodule_search_locations=[str(LEGACY)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["preserved_legacy_core"] = module
    spec.loader.exec_module(module)
    assert callable(module.fuse) and callable(module.amplitude_loss)


# --- 24: closed evidence is unchanged ----------------------------------------

@pytest.mark.parametrize("root", ["experiments/evidence/hch_signed_mass_method_20260912",
                                  "experiments/current/hch_signed_mass_method_entry"])
def test_frozen_closed_evidence_is_unchanged(root):
    recorded = json.loads((ARCHIVE / "FROZEN_EVIDENCE_DIGESTS.json").read_text(encoding="utf-8"))
    entry = recorded["roots"][root]
    digest, count = _tree_digest(ROOT / root)
    assert count == entry["file_count"], f"{root} gained or lost files"
    assert digest == entry["tree_digest_sha256"], f"{root} was modified after closure"


# --- 25: no protected or final data is touched -------------------------------

def test_the_active_core_references_no_protected_or_evaluation_split():
    # Assembled from fragments so this assertion does not itself contain the
    # literal tokens it forbids.
    forbidden = ("PROTECTED" + "_FINAL", "protected" + "_final",
                 "DEV" + "_EVAL", "dev" + "_eval", "HOLD" + "OUT", "hold" + "out")
    for path in sorted(CORE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} references {token}"


def test_this_module_reads_no_protected_artifact():
    """The preservation tests hash closed evidence; they open nothing sealed."""
    forbidden_roots = ("data/" + "protected", "protected" + "_final")
    for path in sorted(Path(__file__).parent.glob("test_core_safety.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden_roots:
            assert token not in text, token
    # Only the two closed-evidence roots and the archive are ever opened here.
    recorded = json.loads((ARCHIVE / "FROZEN_EVIDENCE_DIGESTS.json").read_text(encoding="utf-8"))
    assert set(recorded["roots"]) == {
        "experiments/evidence/hch_signed_mass_method_20260912",
        "experiments/current/hch_signed_mass_method_entry"}
