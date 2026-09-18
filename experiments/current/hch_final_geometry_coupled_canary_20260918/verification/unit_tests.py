"""Correctness gate for the HCH final geometry-coupled canary (PROTOCOL §8).

Failure of any check blocks **all** scientific fits.  This module is the source /
access / contract gate, not the independent verifier: it legitimately imports the
stage's own modules to test them.  The independent verifier
(``verify_canary.py``) is a separate program that never imports the runner.

Run:  python unit_tests.py [--no-smoke] [--device cpu|cuda]
Writes ``experiments/evidence/hch_final_geometry_coupled_canary_20260918/UNIT_TEST_REPORT.json``.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]
IMPL = STAGE / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import gc_common as GC  # noqa: E402
import gc_data as GD  # noqa: E402
import trainer as TR  # noqa: E402
from final_model import (  # noqa: E402
    DIRECT, GEOM_COUPLED, GEOM_COUPLED_NOHIST, GEOM_FLAT,
    FORBIDDEN_INPUT_TOKENS, HCHFinalCore, IllegalInputError,
    assert_legal_input_fields, assert_variant_parity,
)
from fused_encoder import CALENDAR_HOUR_CHANNELS, CORE_HOUR_CHANNELS  # noqa: E402
from geometry_prior import (  # noqa: E402
    GeometryPriorError, build_geometry_prior, prior_identities,
)

U = GC.U
ACTIVE_PATH = (
    "gc_common.py", "geometry_prior.py", "fused_encoder.py", "global_local.py",
    "final_model.py", "gc_data.py", "trainer.py", "gc_runner.py",
)
FORBIDDEN_MODULES = ("core.allocation", "core.model", "core.encoders", "core.bundles",
                     "core.rescues", "core")
FORBIDDEN_SYMBOLS = ("MaskedSoftmaxAllocator", "allocate_coordinates", "HistoryDayEncoder",
                     "ShapeHistoryEncoder", "OptionalTrajectoryEncoder", "EvidenceSet",
                     "make_bundle", "Rescue", "router", "e_L", "e_B", "e_S")
CALIBRATION_CELL = ("GANSU_DA", "TimeMixer")
#: Registered sanity threshold on the step-0 correction/residual ratio.
NEAR_HOST_RATIO_MAX = 0.05

CHECKS: list = []


def check(group: str, name: str):
    def deco(fn):
        CHECKS.append({"group": group, "name": name, "fn": fn})
        return fn
    return deco


# --------------------------------------------------------------------- helpers
def _day(residual, day_index: int = 1, provenance: str = "oof"):
    from core.history import RevealedResidualDay

    res = torch.as_tensor(np.asarray(residual, dtype=np.float32))
    day = dt.date(2024, 1, day_index)
    return RevealedResidualDay(
        day=day, residual=res, reveal_time=U.reveal_of(day), provenance=provenance,
        valid=torch.ones(24, dtype=torch.bool),
    )


def _window(residuals, start: int = 1):
    return [_day(r, start + i) for i, r in enumerate(residuals)]


def _scales():
    from core.geometry import residual_geometry
    from core.scales import CoordinateScales

    rng = np.random.default_rng(0)
    res = rng.normal(0.0, 1.0, size=(64, 24)).astype(np.float32)
    geom = residual_geometry(torch.as_tensor(res))
    ids = [
        (dt.date(2023, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(64)
    ]
    return CoordinateScales.from_train(
        geom.m1.detach().double(), geom.b.detach().double(), geom.B.detach().double(),
        ids, split="TRAIN", source="unit_test_synthetic",
    )


# ------------------------------------------------------------- geometry prior
@check("geometry_prior", "causal_window_boundary_enforced")
def _prior_boundary():
    res = np.zeros(24, dtype=np.float32)
    res[:4] = 1.0
    window = _window([res] * 7)
    target = dt.date(2024, 1, 20)
    origin = U.origin_of(target)
    prior = build_geometry_prior([window], target_days=[target], origins=[origin])
    ok = bool(prior.avail[0] > 0)

    from core.history import RevealedResidualDay

    bad = list(window)
    bad[-1] = RevealedResidualDay(
        day=target, residual=window[-1].residual, reveal_time=U.reveal_of(target),
        provenance="oof", valid=torch.ones(24, dtype=torch.bool),
    )
    refused_target = False
    try:
        build_geometry_prior([bad], target_days=[target], origins=[origin])
    except GeometryPriorError:
        refused_target = True

    late = list(window)
    late[-1] = RevealedResidualDay(
        day=dt.date(2024, 1, 19), residual=window[-1].residual,
        reveal_time=U.reveal_of(dt.date(2024, 1, 19)),
        provenance="oof", valid=torch.ones(24, dtype=torch.bool),
    )
    refused_late = False
    try:
        build_geometry_prior([late], target_days=[target], origins=[origin])
    except GeometryPriorError:
        refused_late = True
    return {
        "passed": ok and refused_target and refused_late,
        "target_day_refused": refused_target,
        "late_reveal_refused": refused_late,
        "availability_without_target_day": float(prior.avail[0].item()),
    }


@check("geometry_prior", "median_b_and_B_exact")
def _prior_median():
    rng = np.random.default_rng(7)
    residuals = [rng.normal(0.0, 1.0, 24).astype(np.float32) for _ in range(7)]
    prior = build_geometry_prior([_window(residuals)])
    from core.geometry import residual_geometry

    geom = residual_geometry(torch.as_tensor(np.stack(residuals)))
    want_b = float(np.median(geom.b.numpy()))
    want_B = float(np.median(geom.B.numpy()))
    got_b = float(prior.b_bar[0].item())
    got_B = float(prior.B_bar[0].item())
    return {
        "passed": abs(want_b - got_b) < 1e-6 and abs(want_B - got_B) < 1e-6,
        "median_b_expected": want_b, "median_b_got": got_b,
        "median_B_expected": want_B, "median_B_got": got_B,
    }


@check("geometry_prior", "signed_shape_barycenter_simplex")
def _prior_barycenter():
    rng = np.random.default_rng(11)
    residuals = [rng.normal(0.5, 1.0, 24).astype(np.float32) for _ in range(7)]
    prior = build_geometry_prior([_window(residuals)])
    ident = prior_identities(prior)
    return {
        "passed": bool(ident["positive_shape_simplex_or_zero"]
                       and ident["negative_shape_simplex_or_zero"]
                       and ident["shapes_nonnegative"]
                       and ident["all_finite"]),
        "identities": ident,
        "sum_plus": float(prior.s_plus_bar[0].sum().item()),
        "sum_minus": float(prior.s_minus_bar[0].sum().item()),
    }


@check("geometry_prior", "zero_sign_mass_fallback")
def _prior_zero_mass():
    pos = np.full(24, 3.0, dtype=np.float32)
    neg = np.full(24, -3.0, dtype=np.float32)
    zero = np.zeros(24, dtype=np.float32)
    p_pos = build_geometry_prior([_window([pos] * 7)])
    p_neg = build_geometry_prior([_window([neg] * 7)])
    p_zero = build_geometry_prior([_window([zero] * 7)])
    ok = (
        float(p_pos.avail_minus[0].item()) == 0.0
        and float(p_pos.s_minus_bar.abs().sum().item()) == 0.0
        and float(p_pos.avail_plus[0].item()) == 1.0
        and float(p_neg.avail_plus[0].item()) == 0.0
        and float(p_neg.s_plus_bar.abs().sum().item()) == 0.0
        and float(p_zero.avail_plus[0].item()) == 0.0
        and float(p_zero.avail_minus[0].item()) == 0.0
        and float(p_zero.B_bar[0].item()) == 0.0
    )
    return {"passed": bool(ok),
            "pos_avail": [float(p_pos.avail_plus[0]), float(p_pos.avail_minus[0])],
            "neg_avail": [float(p_neg.avail_plus[0]), float(p_neg.avail_minus[0])],
            "zero_avail": [float(p_zero.avail_plus[0]), float(p_zero.avail_minus[0])]}


@check("geometry_prior", "valid_mask_respected_and_day_dropped")
def _prior_valid_mask():
    rng = np.random.default_rng(13)
    residuals = [rng.normal(0.0, 1.0, 24).astype(np.float32) for _ in range(7)]
    window = _window(residuals)
    from core.history import RevealedResidualDay

    masked = residuals[3].copy()
    masked[12:] = np.nan
    v = torch.zeros(24, dtype=torch.bool)
    v[:12] = True
    window[3] = RevealedResidualDay(
        day=window[3].day, residual=torch.as_tensor(masked), reveal_time=window[3].reveal_time,
        provenance="oof", valid=v,
    )
    prior = build_geometry_prior([window])

    from core.geometry import residual_geometry

    geom = residual_geometry(torch.as_tensor(residuals[3][None, :]), v[None, :])
    want_b = float(geom.b[0].item())

    # A day with no eligible hour at all contributes nothing.
    empty = list(window)
    empty[5] = RevealedResidualDay(
        day=empty[5].day, residual=torch.zeros(24), reveal_time=empty[5].reveal_time,
        provenance="oof", valid=torch.zeros(24, dtype=torch.bool),
    )
    prior_empty = build_geometry_prior([empty])
    return {
        "passed": int(prior.n_days_used[0].item()) == 7 and int(prior_empty.n_days_used[0].item()) == 6,
        "n_days_used": int(prior.n_days_used[0].item()),
        "n_days_used_with_one_empty_day": int(prior_empty.n_days_used[0].item()),
        "masked_day_b_expected": want_b,
        "available_days": 7,
    }


# --------------------------------------------------------- architecture identity
def _model(variant, scales=None):
    """A freshly seeded model in deterministic (``eval``) mode.

    Every architecture-identity claim in this suite is a *structural* claim: that
    A1 equals A2 once the signed-mass scalar is removed, that A3 is invariant to
    its prior, that step 0 reproduces the Host.  Dropout is a stochastic
    regulariser, so those claims can only be stated as bit-equality with dropout
    disabled; leaving it active would make every ``torch.equal`` comparison a coin
    flip on the dropout mask rather than a statement about the architecture.  The
    training path sets ``train()``/``eval()`` itself in ``trainer.evaluate``.
    """
    scales = scales or _scales()
    torch.manual_seed(1234)
    return HCHFinalCore(variant, scales, dropout=0.1).eval()


def _fake_input(batch: int = 6, scales=None):
    from final_model import FinalInput
    from geometry_prior import GeometryPriorBatch

    scales = scales or _scales()
    g = torch.Generator().manual_seed(99)
    prior = GeometryPriorBatch(
        b_bar=torch.randn(batch, generator=g) * 0.1,
        B_bar=torch.rand(batch, generator=g) * 0.5,
        s_plus_bar=torch.softmax(torch.randn(batch, 24, generator=g), dim=-1),
        s_minus_bar=torch.softmax(torch.randn(batch, 24, generator=g), dim=-1),
        avail_plus=torch.ones(batch), avail_minus=torch.ones(batch), avail=torch.ones(batch),
        n_days_used=torch.full((batch,), 7, dtype=torch.int64),
    )
    return FinalInput(
        host_hour_channels=torch.randn(batch, 24, 5, generator=g),
        calendar_hour=torch.randn(batch, 24, CALENDAR_HOUR_CHANNELS, generator=g),
        calendar_day=torch.randn(batch, 5, generator=g),
        host_day_descriptors=torch.randn(batch, 13, generator=g),
        hour_valid=torch.ones(batch, 24),
        prior=prior,
        date=[f"2024-01-{i + 1:02d}" for i in range(batch)],
    )


@check("architecture_identity", "a1_a2_identical_parameter_tensors")
def _a1_a2_parity():
    a1, a2 = _model(GEOM_FLAT), _model(GEOM_COUPLED)
    s1, s2 = a1.state_dict(), a2.state_dict()
    same_keys = sorted(s1) == sorted(s2)
    same_shapes = same_keys and all(s1[k].shape == s2[k].shape for k in s1)
    report = assert_variant_parity({GEOM_FLAT: a1, GEOM_COUPLED: a2})
    return {
        "passed": bool(same_keys and same_shapes
                       and report["checks"]["a1_a2_equal_parameter_count"]),
        "same_state_dict_keys": bool(same_keys),
        "same_shapes": bool(same_shapes),
        "parameter_count": a2.parameter_count(),
        "checks": report["checks"],
    }


@check("architecture_identity", "a1_a2_delta_is_only_the_signed_mass_scalar")
def _a1_a2_delta():
    scales = _scales()
    a1, a2 = _model(GEOM_FLAT, scales), _model(GEOM_COUPLED, scales)
    a1.load_state_dict(a2.state_dict())
    inp = _fake_input(8, scales)
    with torch.no_grad():
        out_a1 = a1(inp)
        out_a2 = a2(inp)
        a2.coupled = False
        out_a2_flat = a2(inp)
        a2.coupled = True
    live = float((out_a2.m_plus - out_a2.m_minus).abs().mean().item())
    scalar = torch.log1p(out_a2.m_plus / out_a2.diagnostics["s_m"])
    return {
        "passed": bool(torch.equal(out_a1.correction, out_a2_flat.correction)
                       and not torch.equal(out_a1.correction, out_a2.correction)
                       and float(scalar.abs().max().item()) > 0.0),
        "a1_equals_a2_with_scalar_forced_zero": bool(
            torch.equal(out_a1.correction, out_a2_flat.correction)
        ),
        "coupled_differs_from_flat": bool(
            not torch.equal(out_a1.correction, out_a2.correction)
        ),
        "max_signed_mass_scalar": float(scalar.abs().max().item()),
        "max_abs_correction_difference": float(
            (out_a1.correction - out_a2.correction).abs().max().item()
        ),
        "mean_abs_m_plus_minus_gap": live,
    }


@check("architecture_identity", "a2_a3_identical_and_a3_history_invariant")
def _a2_a3():
    scales = _scales()
    a2, a3 = _model(GEOM_COUPLED, scales), _model(GEOM_COUPLED_NOHIST, scales)
    s2, s3 = a2.state_dict(), a3.state_dict()
    a3.load_state_dict(s2)
    inp = _fake_input(8, scales)
    with torch.no_grad():
        out_ref = a3(inp)
        perturbed = _fake_input(8, scales)
        p = perturbed.prior
        from geometry_prior import GeometryPriorBatch

        perturbed.prior = GeometryPriorBatch(
            b_bar=p.b_bar * 0 + 37.0, B_bar=p.B_bar * 0 + 11.0,
            s_plus_bar=torch.full_like(p.s_plus_bar, 1.0 / 24),
            s_minus_bar=torch.full_like(p.s_minus_bar, 1.0 / 24),
            avail_plus=p.avail_plus, avail_minus=p.avail_minus, avail=p.avail,
            n_days_used=p.n_days_used,
        )
        out_pert = a3(perturbed)
    report = assert_variant_parity({GEOM_COUPLED: a2, GEOM_COUPLED_NOHIST: a3})
    return {
        "passed": bool(sorted(s2) == sorted(s3)
                       and all(s2[k].shape == s3[k].shape for k in s2)
                       and report["checks"]["a2_a3_equal_parameter_count"]
                       and torch.equal(out_ref.correction, out_pert.correction)),
        "a3_invariant_to_prior_mutation": bool(
            torch.equal(out_ref.correction, out_pert.correction)
        ),
        "parameter_count": a3.parameter_count(),
    }


@check("architecture_identity", "direct_within_parameter_budget")
def _direct_budget():
    models = {v: _model(v) for v in (DIRECT, GEOM_FLAT, GEOM_COUPLED, GEOM_COUPLED_NOHIST)}
    report = assert_variant_parity(models)
    return {
        "passed": bool(report["checks"]["direct_within_budget"]),
        "parameter_counts": report["parameter_counts"],
        "direct_over_a2_ratio": report["direct_over_a2_ratio"],
        "budget": 1.25,
    }


@check("architecture_identity", "direct_uses_same_day_core")
def _direct_shares_core():
    d, a2 = _model(DIRECT), _model(GEOM_COUPLED)
    ok = (
        "day_core.affine.weight" in d.state_dict()
        and d.state_dict()["day_core.affine.weight"].shape
        == a2.state_dict()["day_core.affine.weight"].shape
        and d.state_dict()["encoder.gru.weight_ih_l0"].shape
        == a2.state_dict()["encoder.gru.weight_ih_l0"].shape
        and d.encoder.in_channels == CORE_HOUR_CHANNELS
    )
    return {"passed": bool(ok), "shared_module_shapes_match": bool(ok)}


# ------------------------------------------------------------------ exact decoder
@check("exact_decoder", "identities_hold_on_model_output")
def _decoder_identities():
    from core.decoder import prediction_identities

    scales = _scales()
    model = _model(GEOM_COUPLED, scales)
    with torch.no_grad():
        out = model(_fake_input(16, scales))
    from core.decoder import DecodedPrediction

    pred = DecodedPrediction(
        b_hat=out.b_hat, B_hat=out.B_hat, s_plus_hat=out.s_plus_hat,
        s_minus_hat=out.s_minus_hat, a_plus=out.a_plus, a_minus=out.a_minus,
        correction=out.correction, overlap_gap=out.overlap_gap,
    )
    ident = prediction_identities(pred)
    return {"passed": bool(all(ident.values())), "identities": ident}


@check("exact_decoder", "one_signed_and_zero_targets_finite")
def _decoder_one_signed():
    from core.geometry import residual_geometry

    scales = _scales()
    model = _model(GEOM_COUPLED, scales)
    rng = np.random.default_rng(5)
    residuals = np.stack([
        np.full(24, 2.0),                # all-positive
        np.full(24, -2.0),               # all-negative
        np.zeros(24),                    # exactly zero
        rng.normal(0.0, 1.0, 24),        # mixed
    ]).astype(np.float32)
    geom = residual_geometry(torch.as_tensor(residuals))
    inp = _fake_input(4, scales)
    with torch.no_grad():
        out = model(inp)
        loss = model.compute_loss(out, geom, torch.as_tensor(residuals), scales)
    return {
        "passed": bool(torch.isfinite(out.correction).all() and torch.isfinite(loss["total"])),
        "finite": bool(torch.isfinite(out.correction).all()),
        "loss_finite": bool(torch.isfinite(loss["total"])),
    }


@check("exact_decoder", "shape_simplex_under_mask")
def _decoder_simplex():
    from core.geometry import masked_softmax

    logits = torch.randn(4, 24)
    full = masked_softmax(logits, torch.ones(4, 24, dtype=torch.bool), dim=-1)
    partial_mask = torch.zeros(4, 24, dtype=torch.bool)
    partial_mask[:, :6] = True
    partial = masked_softmax(logits, partial_mask, dim=-1)
    dead = masked_softmax(logits, torch.zeros(4, 24, dtype=torch.bool), dim=-1)
    return {
        "passed": bool(
            torch.allclose(full.sum(-1), torch.ones(4), atol=1e-5)
            and torch.allclose(partial.sum(-1), torch.ones(4), atol=1e-5)
            and float(partial[:, 6:].abs().max().item()) == 0.0
            and float(dead.abs().max().item()) == 0.0
        ),
        "full_sums_to_one": bool(torch.allclose(full.sum(-1), torch.ones(4), atol=1e-5)),
        "dead_row_is_zero": float(dead.abs().max().item()) == 0.0,
    }


# --------------------------------------------------------------- near-Host start
_STARTUP_CACHE: dict = {}


def _calibration_data(device: str):
    if "data" not in _STARTUP_CACHE:
        market, host = CALIBRATION_CELL
        GC.RC.install_access_guard()
        _STARTUP_CACHE["data"] = GD.to_device(GD.build_cell_data(market, host), device)
    return _STARTUP_CACHE["data"]


@check("near_host_startup", "step0_output_finite_and_near_host")
def _step0():
    device = _STARTUP_CACHE.get("device", "cpu")
    data = _calibration_data(device)
    model = HCHFinalCore(GEOM_COUPLED, data["scales"], dropout=0.1).to(device)
    idx = GD.diag_index(len(data["train_rows"]), device)
    with torch.no_grad():
        out = model(GD.select_input(data["train_input"], idx))
        c = out.correction
        r = data["train_res"][idx]
        ratio = float(c.abs().mean().item() / max(float(r.abs().mean().item()), 1e-12))
        b0 = float(out.b_hat.abs().max().item())
        B0 = float((out.B_hat / model.s_B).mean().item())
    return {
        "passed": bool(torch.isfinite(c).all() and ratio <= NEAR_HOST_RATIO_MAX and b0 == 0.0),
        "correction_ratio_step0": ratio,
        "threshold": NEAR_HOST_RATIO_MAX,
        "max_abs_b_hat_at_init": b0,
        "B_hat_over_s_B_at_init": B0,
        "finite": bool(torch.isfinite(c).all()),
    }


@check("near_host_startup", "gradient_reaches_query_key_and_day_core")
def _gradient_reach():
    import torch as _t

    device = _STARTUP_CACHE.get("device", "cpu")
    data = _calibration_data(device)
    _t.manual_seed(3)
    model = HCHFinalCore(GEOM_COUPLED, data["scales"], dropout=0.0).to(device)
    idx = _t.arange(0, min(32, len(data["train_rows"])), device=device)
    model.train()
    out = model(GD.select_input(data["train_input"], idx))
    terms = model.compute_loss(
        out, GD.slice_geometry(data["train_target"], idx), data["train_res"][idx], data["scales"]
    )
    terms["total"].backward()
    want = ("shape_head.query_plus.weight", "shape_head.query_minus.weight",
            "shape_head.key.weight", "day_core.affine.weight", "day_core.b_affine.weight",
            "day_core.B_affine.weight", "encoder.gru.weight_ih_l0", "encoder.input_affine.weight")
    norms = {}
    for name in want:
        param = dict(model.named_parameters())[name]
        norms[name] = None if param.grad is None else float(param.grad.norm().item())
    return {
        "passed": all(v is not None and v > 0.0 for v in norms.values()),
        "grad_norms": norms,
    }


# ------------------------------------------------------------- training schedule
@check("training_schedule", "staged_objective_boundaries")
def _schedule_boundaries():
    terms = {"total": 5.0, "L_rec": 1.0}
    probes = (1, 2, 399, 400, 401, 402, 2000)
    seq = {i: TR.staged_objective(terms, i, True) for i in probes}
    aux = [i for i in probes if seq[i][1]]
    direct = TR.staged_objective(terms, 1, False)
    # Update indices are 1-based: there is no update 0, and accepting one would
    # make "which updates used the full objective" ambiguous.
    zero_refused = False
    try:
        TR.staged_objective(terms, 0, True)
    except ValueError:
        zero_refused = True
    return {
        "passed": bool(
            seq[1] == (5.0, True) and seq[400] == (5.0, True)
            and seq[401] == (1.0, False) and seq[2000] == (1.0, False)
            and direct == (1.0, False)
            and aux == [1, 2, 399, 400]
            and zero_refused
        ),
        "auxiliary_updates": aux,
        "update_zero_refused": zero_refused,
        "switch_step": TR.SWITCH_STEP,
        "first_reconstruction_only_update": 401,
        "direct_uses_reconstruction_only": direct == (1.0, False),
    }


@check("training_schedule", "defaults_match_completed_o1_recipe")
def _schedule_matches_o1():
    import inspect

    sys.path.insert(0, str(GC.PROBE_IMPL))
    import probe_common as PC  # prior stage, read-only

    sig = inspect.signature(TR.train_final)
    got = {
        "max_steps": sig.parameters["max_steps"].default,
        "val_every": sig.parameters["val_every"].default,
        "no_stop_before": sig.parameters["no_stop_before"].default,
        "patience_checks": sig.parameters["patience_checks"].default,
        "switch_step": TR.SWITCH_STEP,
    }
    # ``switch_step`` is a module constant of the O1 stage, not a recipe field,
    # so it is compared against the constant rather than looked up in the recipe.
    want = {k: PC.RECIPE_O1[k] for k in got if k != "switch_step"}
    want["switch_step"] = int(PC.SWITCH_STEP)
    from core.training_support import build_primary_train_config

    cfg = build_primary_train_config()
    optimizer_ok = (
        cfg.optimizer == PC.RECIPE_O1["optimizer"]
        and float(cfg.lr) == float(PC.RECIPE_O1["lr"])
        and float(cfg.weight_decay) == float(PC.RECIPE_O1["weight_decay"])
        and int(cfg.batch_size) == int(PC.RECIPE_O1["batch_size"])
        and float(cfg.grad_clip) == float(PC.RECIPE_O1["grad_clip"])
        and float(cfg.ema_decay) == float(PC.RECIPE_O1["ema_decay"])
        and cfg.eval_weights == PC.RECIPE_O1["eval_weights"]
    )
    return {
        "passed": bool(got == want and optimizer_ok),
        "schedule": got, "o1_schedule": want, "optimizer_matches_o1": bool(optimizer_ok),
    }


@check("training_schedule", "ema_selection_rule_is_min_over_all_checks")
def _selection_rule():
    src = (IMPL / "trainer.py").read_text(encoding="utf-8")
    ok = (
        "stats[\"val_mae_ema\"] < best[\"val_mae\"] - 1e-7" in src
        and "record_check(0, 0, True, None, None, False)" in src
        and "ema.copy_to(model)" in src
    )
    return {"passed": bool(ok), "source_rule_present": bool(ok)}


# ------------------------------------------------------------------------- access
@check("access", "final_input_rejects_target_and_residual_fields")
def _input_rejects():
    raised = {}
    for token in FORBIDDEN_INPUT_TOKENS:
        try:
            assert_legal_input_fields({f"day_{token}_field": 1})
            raised[token] = False
        except IllegalInputError:
            raised[token] = True
    return {"passed": all(raised.values()), "per_token": raised}


@check("access", "forbidden_token_list_matches_frozen_source")
def _token_list_matches():
    tree = ast.parse((REPO / "src/core/model.py").read_text(encoding="utf-8"))
    frozen = None
    for node in tree.body:
        # The frozen declaration is annotated (``FORBIDDEN_BATCH_TOKENS: Tuple[...] =``),
        # so it parses as an AnnAssign rather than a plain Assign.
        targets = []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        if any(getattr(t, "id", None) == "FORBIDDEN_BATCH_TOKENS" for t in targets):
            frozen = ast.literal_eval(node.value)
    return {
        "passed": frozen is not None and tuple(frozen) == tuple(FORBIDDEN_INPUT_TOKENS),
        "frozen": list(frozen) if frozen else None,
        "local": list(FORBIDDEN_INPUT_TOKENS),
    }


@check("access", "test_role_refused_and_no_test_rows_returned")
def _test_role_refused():
    before = int(GC.RC.ACCESS_STATE["test_role_frame_refusals"])
    raised = False
    try:
        GC.RC.pretest_role_frame(CALIBRATION_CELL[0], CALIBRATION_CELL[1], "TEST")
    except PermissionError:
        raised = True
    return {
        "passed": bool(raised and int(GC.RC.ACCESS_STATE["test_rows_returned"]) == 0
                       and int(GC.RC.ACCESS_STATE["test_role_frame_refusals"]) > before),
        "test_role_refusals": int(GC.RC.ACCESS_STATE["test_role_frame_refusals"]),
        "test_rows_returned": int(GC.RC.ACCESS_STATE["test_rows_returned"]),
        "distinct_paths_blocked": list(GC.RC.ACCESS_STATE["distinct_paths_blocked"]),
    }


@check("access", "legit_frames_are_train_and_val_only")
def _legit_frames():
    built = GC.RC.ACCESS_STATE["legit_frames_built"]
    return {
        "passed": set(built) == {"TRAIN", "VAL"} and built["TRAIN"] > 0 and built["VAL"] > 0,
        "legit_frames_built": dict(built),
    }


# ----------------------------------------------------------------- source boundary
@check("source_boundary", "no_forbidden_import_on_the_active_path")
def _no_forbidden_imports():
    found = {}
    for name in ACTIVE_PATH:
        path = IMPL / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [a.name for a in node.names if a.name in FORBIDDEN_MODULES]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in FORBIDDEN_MODULES or mod.startswith("core.model"):
                    hits.append(mod)
        if hits:
            found[name] = sorted(set(hits))
    return {"passed": not found, "violations": found, "files_scanned": list(ACTIVE_PATH)}


@check("source_boundary", "no_removed_module_on_the_active_path")
def _no_removed_symbols():
    found = {}
    for name in ACTIVE_PATH:
        path = IMPL / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[-1])
        hits = sorted(names & set(FORBIDDEN_SYMBOLS))
        if hits:
            found[name] = hits
    return {"passed": not found, "violations": found}


@check("source_boundary", "model_has_no_forbidden_submodule")
def _no_forbidden_submodule():
    model = _model(GEOM_COUPLED)
    classes = {type(m).__name__ for m in model.modules()}
    hits = sorted(classes & set(FORBIDDEN_SYMBOLS))
    return {"passed": not hits, "model_classes": sorted(classes), "violations": hits}


@check("source_boundary", "one_temporal_encoder_two_queries_no_router")
def _efficiency_contract():
    model = _model(GEOM_COUPLED)
    cfg = model.variant_config()
    from core.training_support import assert_rescue_absent, build_primary_train_config

    assert_rescue_absent(build_primary_train_config())
    return {
        "passed": bool(cfg["learned_temporal_encoders"] == 1 and cfg["signed_mass_queries"] == 2
                       and cfg["source_router_calls"] == 0),
        "variant_config": cfg,
    }


# ------------------------------------------------------------------------ runner
def _run_smoke(device: str) -> dict:
    """A short non-registered schedule smoke: the switch must be observed at 400.

    This fit is **not** one of the 48 registered critical-path fits and writes no
    evidence run.  It exists only to prove that the registered staged objective is
    the one the trainer actually applies.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gc_smoke_") as tmp:
        record = TR.train_final(
            GEOM_COUPLED, CALIBRATION_CELL[0], CALIBRATION_CELL[1], 7, Path(tmp),
            max_steps=500, val_every=50, device=device, write_outputs=False,
        )
    curve = record["curve"]
    post = [e for e in curve if int(e["step"]) > TR.SWITCH_STEP and not e["is_initial"]]
    pre = [e for e in curve if 0 < int(e["step"]) <= TR.SWITCH_STEP]
    pre_ok = all(e["used_full_loss_at_this_step"] and e["loss_schedule"] == "L_rec_plus_aux"
                 for e in pre)
    post_ok = all((not e["used_full_loss_at_this_step"]) and e["loss_schedule"] == "L_rec_only"
                  for e in post)
    switch_marked = [e["step"] for e in curve if e["objective_switch_here"]]
    return {
        "passed": bool(pre_ok and post_ok and post and switch_marked == [TR.SWITCH_STEP]),
        "max_steps": 500,
        "optimizer_steps_run": int(record["optimization"]["slots_available"]),
        "stopped_at_step": record["optimization"]["stopped_at_step"],
        "switch_step": record["optimization"]["switch_step"],
        "checks_before_or_at_switch": len(pre),
        "checks_after_switch": len(post),
        "all_pre_switch_used_auxiliaries": bool(pre_ok),
        "all_post_switch_used_reconstruction_only": bool(post_ok),
        "objective_switch_here_steps": switch_marked,
        "selected_step": record["selected_step"],
        "selected_val_mae_ema": record["selected_val_mae_ema"],
        "is_registered_fit": record["is_registered_fit"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--json", default=str(GC.EVID / "UNIT_TEST_REPORT.json"))
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    _STARTUP_CACHE["device"] = device
    GC.RC.install_access_guard()
    GC.RC.worker_env()

    results = []
    for spec in CHECKS:
        started = dt.datetime.now().isoformat(timespec="seconds")
        try:
            detail = spec["fn"]()
        except Exception as exc:  # a crashing check is a failing check
            detail = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-6:],
            }
        results.append({
            "group": spec["group"], "name": spec["name"],
            "passed": bool(detail.get("passed")), "detail": detail, "started": started,
        })

    smoke = None
    if not args.no_smoke:
        try:
            smoke = _run_smoke(device)
        except Exception as exc:
            smoke = {"error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc().splitlines()[-6:]}

    failed = [r for r in results if not r["passed"]]
    groups = {}
    for r in results:
        g = groups.setdefault(r["group"], {"n": 0, "failed": 0})
        g["n"] += 1
        g["failed"] += 0 if r["passed"] else 1
    smoke_ok = None if smoke is None else bool(smoke.get("passed"))
    if smoke is not None:
        groups.setdefault("training_schedule", {"n": 0, "failed": 0})
    all_passed = (not failed) and (smoke_ok is not False)
    report = {
        "schema": "hch_final_gc_unit_test_report.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "device": device,
        "n_checks": len(results),
        "n_failed": len(failed),
        "schedule_smoke_passed": smoke_ok,
        "all_passed": bool(all_passed),
        "groups": groups,
        "checks": results,
        "schedule_smoke": smoke,
        "note": (
            "Failure of any check blocks all scientific fits (PROTOCOL §8). The smoke is a "
            "non-registered 500-step fit on GANSU_DA/TimeMixer written outside the evidence "
            "runs tree; it is not one of the 48 critical-path fits and it is gating only "
            "because it is the one direct observation of the staged schedule."
        ),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    GC.write_json(Path(args.json), report)
    print(f"[unit-tests] {len(results) - len(failed)}/{len(results)} passed, "
          f"smoke={smoke_ok} -> {args.json}")
    for r in failed:
        print(f"  FAIL {r['group']}::{r['name']}: {r['detail'].get('error', r['detail'])}")
    if smoke_ok is False:
        print(f"  FAIL schedule_smoke: {json.dumps(smoke, default=str)[:400]}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
