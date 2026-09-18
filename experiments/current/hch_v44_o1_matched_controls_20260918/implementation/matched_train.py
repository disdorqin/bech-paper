"""Variant-parameterised O1 trainer for the matched-control closure stage.

The training loop below is a **deliberately minimal copy** of
``probe_train.train_o1``.  Everything that is variant-agnostic is still
*imported* rather than copied -- ``recovery_train.build_cell_data``,
``to_device``, ``build_optimizer``, ``evaluate``, ``grad_norms_by_group``,
``_bind_prior``, and ``probe_train.objective`` / ``diag_components`` /
``_assert_component_consistency`` -- so the optimizer, the data support, the
scales, the per-check evaluation and the loss schedule are produced by the
identical code bytes the 24 reused O1/G2 reference runs used.

Exactly three things differ, and ``verification/verify_matched.py`` re-checks
them independently:

1. the model is built for the *requested* variant instead of the hardcoded
   ``G2_GEOM_COORD_CONTEXT`` (``recovery_train.build_model`` hardcodes it, which
   is why this module exists at all);
2. the four structured loss components are only diagnosed for a structured
   model -- the direct control has no Level / balanced-mass / Shape readouts, so
   there is nothing to diagnose and no ``raw_*`` field to reconcile;
3. the freeze record names the variant it actually trained.

Deltas 1 and 3 are inert for a structured model: with ``variant ==
G2_GEOM_COORD_CONTEXT`` this module would reproduce ``train_o1`` byte for byte
in behaviour.  Delta 2 never fires for a structured model.

The registered O1 recipe is fixed here and not a parameter: AdamW 1e-3,
wd 1e-4, batch 32, clip 1.0, EMA 0.995, <=2000 updates, VAL every 50, step 0
legal, no early stop before 800, then patience 8 checks, full structured
objective through update 400 and ``L_rec`` alone afterwards.

There is no TEST path in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matched_common as MC
import probe_common as PC
import probe_train as PT
import recovery_common as RC
import recovery_train as RT
# Imported under their bare names so the copied training loop below stays
# textually identical to the reference loop apart from the declared deltas.
from recovery_train import _bind_prior, grad_norms_by_group  # noqa: PLC2701

SWITCH_STEP = PC.SWITCH_STEP
U = RC.U

#: Variants this stage is forbidden to fit.  G2 is the reused reference and must
#: never be retrained; G0 is out of scope for the whole stage.
FORBIDDEN_VARIANTS = (MC.G0, MC.G2)

#: The variants this stage may fit, and the phase each belongs to.
ALLOWED_VARIANTS = (MC.G1, MC.G3, MC.G3_SHARED)


# ------------------------------------------------------------------ direct-arm shims
def evaluate_matched(model, ema, data: dict, diag_idx, profile, device: str,
                     grad_norms: dict) -> dict:
    """``recovery_train.evaluate`` for a structured model; a reduced form for G3.

    ``recovery_train.evaluate`` reads ``comp["raw_mae"]``, ``raw_level_abs``,
    ``raw_mass_abs`` and ``shape_w1``, which only ``structured_loss`` produces.
    A non-structured model is dispatched to ``_evaluate_direct``, whose only
    differences are the four structured-diagnostic lines; the primary metric
    (``val_mae_ema``), the Host MAE, the correction ratio, the raw-weight MAE and
    the entire ``_chunked_forward`` / ``_stats`` path are identical.
    """
    if bool(getattr(model, "structured", True)):
        return RT.evaluate(model, ema, data, diag_idx, profile, device, grad_norms)
    return _evaluate_direct(model, ema, data, diag_idx, device, grad_norms)


def _evaluate_direct(model, ema, data: dict, diag_idx, device: str,
                     grad_norms: dict) -> dict:
    """``recovery_train.evaluate`` with the structured diagnostic block reduced.

    ``diag_reconstruction_mae`` keeps the structured arm's own definition -- the
    raw per-day MAE over eligible hours, RMSE-free, computed here straight off
    the correction tensor through the same ``daily_mae`` / ``masked_row_mean``
    pair ``structured_loss`` uses -- so the reconciliation
    ``L_rec * s_r == diag_reconstruction_mae`` remains a real cross-check rather
    than a tautology.  Level / mass / Shape have no direct-arm counterpart and
    are recorded as ``None``.
    """
    import torch
    from core.losses import daily_mae, masked_row_mean
    from core.training_support import fp32_region

    P_ = _bind_prior()
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training

    # --- EMA weights: the primary evaluation
    ema.copy_to(model)
    model.eval()
    val_outs = RT._chunked_forward(model, data["val_batch"])
    train_outs = RT._chunked_forward(model, data["train_batch"])
    val_stats = RT._stats(val_outs, data["val_res"])
    train_stats = RT._stats(train_outs, data["train_res"])

    diag_batch = P_.select_batch(data["train_batch"], diag_idx)
    diag_res = data["train_res"][diag_idx]
    diag_target = P_.slice_geometry(data["train_target"], diag_idx)
    with torch.no_grad(), fp32_region(device):
        out = model(diag_batch)
        comp = model.compute_loss(out, diag_target, diag_res, data["scales"])
        per_day, row_mask = daily_mae(diag_res, out.correction, diag_target.valid)
        diag_recon = float(masked_row_mean(per_day, row_mask).detach().item())

    # --- raw weights: diagnostic only, never selected on
    model.load_state_dict(live)
    model.eval()
    raw_out = RT._chunked_forward(model, data["val_batch"])
    raw_val_mae = RT._stats(raw_out, data["val_res"])["mae"]

    model.load_state_dict(live)
    model.train(was_training)

    return {
        "val_mae_ema": val_stats["mae"],
        "val_mae_raw": raw_val_mae,
        "val_host_mae": val_stats["mean_abs_residual"],
        "val_gain_vs_host_pct": float(
            (val_stats["mean_abs_residual"] - val_stats["mae"])
            / max(val_stats["mean_abs_residual"], 1e-12) * 100.0
        ),
        "val_correction_ratio": val_stats["correction_ratio"],
        "val_mean_abs_correction": val_stats["mean_abs_correction"],
        "train_mae_ema": train_stats["mae"],
        "train_host_mae": train_stats["mean_abs_residual"],
        "train_correction_ratio": train_stats["correction_ratio"],
        "diag_reconstruction_mae": diag_recon,
        "level_mae": None,
        "mass_mae": None,
        "shape_w1": None,
        "grad_norms_preclip": grad_norms,
        "direct_arm_note": (
            "structured diagnostics (level/mass/shape) are undefined for the "
            "direct control; diag_reconstruction_mae is the same raw per-day MAE "
            "definition the structured arm reports"
        ),
    }


def diag_components_matched(model, ema, data: dict, diag_idx, device: str) -> dict:
    """The O1 per-check component block, for structured and direct models alike."""
    if bool(getattr(model, "structured", True)):
        return PT.diag_components(model, ema, data, diag_idx, device)

    import torch
    from core.training_support import fp32_region

    P_ = _bind_prior()
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    ema.copy_to(model)
    model.eval()
    with torch.no_grad(), fp32_region(device):
        out = model(P_.select_batch(data["train_batch"], diag_idx))
        comp = model.compute_loss(
            out, P_.slice_geometry(data["train_target"], diag_idx),
            data["train_res"][diag_idx], data["scales"],
        )
    model.load_state_dict(live)
    model.train(was_training)
    return {
        "L_rec": float(comp["L_rec"].detach().item()),
        "L_b": None, "L_B": None, "L_S": None, "aux_mean": None,
        "total": float(comp["total"].detach().item()),
    }


#: The components and the reference's diagnostic fields are fp32 tensors, so the
#: scale round-trip carries fp32 rounding; 1e-5 is far below any real model change
#: while staying above the representation noise.
COMPONENT_TOL = PT.COMPONENT_TOL


def _assert_direct_consistency(comp: dict, stats: dict, scales) -> None:
    """The direct arm's one surviving reconciliation identity."""
    lhs = comp["L_rec"] * float(scales.s_r)
    rhs = float(stats["diag_reconstruction_mae"])
    if abs(lhs - rhs) > COMPONENT_TOL * max(1.0, abs(rhs)):
        raise RuntimeError(
            f"direct-arm component inconsistency: L_rec*s_r == {lhs!r} "
            f"but diag_reconstruction_mae == {rhs!r}")


# ------------------------------------------------------------------ trainer
def train_matched(
    market: str,
    host: str,
    seed: int,
    out_dir: Path,
    *,
    variant: str,
    max_steps: int = 2000,
    val_every: int = 50,
    no_stop_before: int = 800,
    patience_checks: int = 8,
    device: str = "cuda",
    recipe_id: str = MC.RECIPE_ID,
    switch_step: int = SWITCH_STEP,
    write_outputs: bool = True,
) -> dict:
    """One registered run of ``variant`` under the O1 schedule.  TRAIN+VAL only."""
    import torch
    from core.training_support import (
        EMA, DifficultyInterleaver, assert_primary_profile, build_primary_train_config,
        clip_gradients, daily_difficulty, fp32_region, neural_autocast,
    )

    if variant in FORBIDDEN_VARIANTS:
        raise ValueError(f"variant {variant!r} is forbidden in this stage")
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if int(switch_step) != SWITCH_STEP:
        raise ValueError("SWITCH_STEP is fixed and non-tunable at 400")

    RC.install_access_guard()
    RC.worker_env()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = build_primary_train_config()
    assert_primary_profile(cfg.profile)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        device = "cpu"

    data = RT.to_device(RT.build_cell_data(market, host), device)
    scales = data["scales"]
    n = len(data["train_rows"])
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    epochs_needed = int(np.ceil(max_steps * cfg.batch_size / n)) + 2

    # --- delta 1: the variant is explicit instead of hardcoded to G2 ----------
    model = MC.build_model_for_variant(data, device, cfg.profile, variant)
    # -------------------------------------------------------------------------
    structured = bool(getattr(model, "structured", True))
    optimizer = RT.build_optimizer(model, cfg.profile, None)
    ema = EMA(model, decay=cfg.ema_decay)
    interleaver = DifficultyInterleaver(
        ids=[r.day.isoformat() for r in data["train_rows"]],
        difficulty=[float(d) for d in daily_difficulty(data["train_res"].detach().cpu())],
        split="TRAIN", bins=4, seed=seed,
    )
    audit = interleaver.audit(epochs_needed)
    if not audit["all_epochs_exact_once"]:
        raise RuntimeError("difficulty interleaver failed the once-per-epoch contract")

    diag_idx = torch.as_tensor(
        np.unique(np.linspace(0, n - 1, min(n, RT.DIAG_MAX_ROWS)).astype(np.int64)),
        dtype=torch.long,
    )
    diag_idx = diag_idx.to(device)

    def _lr_now() -> float:
        return float(optimizer.param_groups[0]["lr"])

    curve: list[dict] = []
    best = {"val_mae": float("inf"), "step": -1, "shadow": None, "check": -1}
    check_index = 0

    def record_check(step: int, samples_seen: int, is_initial: bool, grad_norms: dict | None,
                     step_loss: float | None, used_full: bool) -> None:
        nonlocal check_index, best
        stats = evaluate_matched(model, ema, data, diag_idx, cfg.profile, device,
                                 grad_norms or {})
        # --- delta 2: structured components only exist for a structured model
        comp = diag_components_matched(model, ema, data, diag_idx, device)
        if structured:
            PT._assert_component_consistency(comp, stats, scales)
        else:
            _assert_direct_consistency(comp, stats, scales)
        # ---------------------------------------------------------------------
        entry = {
            "check": check_index, "step": int(step), "is_initial": bool(is_initial),
            "samples_seen": int(samples_seen),
            "epoch_equivalent": float(samples_seen / n),
            "lr": _lr_now(),
            **stats,
            # --- O1 additions -------------------------------------------------
            "train_reconstruction_mae": float(stats["diag_reconstruction_mae"]),
            "step_loss_optimized": (None if step_loss is None else float(step_loss)),
            # The step-0 calibration check precedes any update, so it has no loss of
            # its own; every other check reports the objective its own update used.
            "is_optimization_step": not bool(is_initial),
            "used_full_loss_at_this_step": (None if is_initial else bool(used_full)),
            "loss_schedule": ("L_rec_plus_aux" if used_full else "L_rec_only"),
            "objective_switch_here": bool(int(step) == SWITCH_STEP),
            "L_rec": comp["L_rec"], "L_b": comp["L_b"], "L_B": comp["L_B"], "L_S": comp["L_S"],
            "L_total_frozen": comp["total"],
            # The direct control has no auxiliary term at all; ``None`` here means
            # "does not exist", not "was not measured".
            "aux_mean": (None if not structured
                         else (comp["L_b"] + comp["L_B"] + comp["L_S"]) / 3.0),
            # -------------------------------------------------------------------
            "selected": False,
        }
        if stats["val_mae_ema"] < best["val_mae"] - 1e-7:
            best = {
                "val_mae": stats["val_mae_ema"], "step": int(step), "check": check_index,
                "shadow": {k: v.clone() for k, v in ema.shadow.items()},
            }
            entry["selected"] = True
        entry["best_val_mae_ema_so_far"] = float(best["val_mae"])
        entry["best_step_so_far"] = int(best["step"])
        curve.append(entry)
        check_index += 1

    record_check(0, 0, True, None, None, False)

    samples_seen = 0
    stream: list[int] = []
    epoch = 0
    step = 0
    stopped_at = None
    while step < max_steps:
        if len(stream) < cfg.batch_size:
            stream.extend(interleaver.epoch_order(epoch))
            epoch += 1
            continue
        idx = torch.as_tensor(stream[:cfg.batch_size], dtype=torch.long, device=device)
        del stream[:cfg.batch_size]
        model.train()
        with neural_autocast(cfg.profile, device):
            out = model(_bind_prior().select_batch(data["train_batch"], idx))
        with fp32_region(device):
            sub_target = _bind_prior().slice_geometry(data["train_target"], idx)
            sub_res = data["train_res"][idx]
            loss, used_full = PT.objective(
                model.compute_loss(out, sub_target, sub_res, data["scales"]), step + 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        step += 1
        samples_seen += cfg.batch_size
        is_check = (step % val_every == 0)
        # Gradients are read from the real training step, before clipping.
        norms = grad_norms_by_group(model) if is_check else None
        clip_gradients(model, cfg.grad_clip)
        optimizer.step()
        ema.update(model)

        if is_check:
            record_check(step, samples_seen, False, norms, float(loss.detach()), used_full)
            if step >= no_stop_before and (check_index - 1 - best["check"]) >= patience_checks:
                stopped_at = step
                break

    if best["shadow"] is None:
        raise RuntimeError("no EMA checkpoint selected")

    n_full = min(int(stopped_at or max_steps), SWITCH_STEP)
    selected_entry = next(e for e in curve if e["step"] == best["step"] and e["selected"])
    post = [e for e in curve if int(e["step"]) > SWITCH_STEP]
    opt_post = min(post, key=lambda e: float(e["val_mae_ema"])) if post else None

    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "selected_ema.pt"
        torch.save(best["shadow"], ckpt)
        keys = sorted(best["shadow"])
        ckpt_hash = U.array_hash(*[best["shadow"][k].detach().cpu().numpy() for k in keys])
    else:
        ckpt, ckpt_hash = None, None

    record = {
        "schema": "v44_o1_matched_control_run.v1",
        "protocol_id": MC.PROTOCOL_ID,
        "recipe_id": recipe_id,
        "market": market, "host": host, "seed": int(seed),
        # --- delta 3: the record names the variant actually trained -----------
        "variant": variant,
        "base_variant": MC.G3 if variant == MC.G3_SHARED else variant,
        "tied_identity": bool(getattr(model, "tied_identity", None)),
        "structured_decoder": structured,
        # ---------------------------------------------------------------------
        "device": device,
        "optimization": {
            "optimizer": cfg.optimizer, "lr": _lr_now(),
            "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
            "grad_clip": cfg.grad_clip, "ema_decay": cfg.ema_decay,
            "eval_weights": cfg.eval_weights,
            "max_steps": int(max_steps), "val_every": int(val_every),
            "no_stop_before": int(no_stop_before), "patience_checks": int(patience_checks),
            "readout_init": None,
            "stopped_at_step": stopped_at,
            "slots_available": int(max_steps),
        },
        "objective_probe": {
            "switch_step": SWITCH_STEP,
            "updates_full_loss": n_full,
            "updates_reconstruction_only": int(max(0, (stopped_at or max_steps) - SWITCH_STEP)),
            "loss_full": "L_rec + (L_b + L_B + L_S)/3",
            "loss_after_switch": "L_rec",
            "loss_effective_for_direct_control": "L_rec throughout (total == L_rec)",
            "changed_vs_reference": (
                ["loss schedule after update 400"] if structured else
                ["loss is L_rec only throughout; no auxiliary term exists"]),
            "unchanged_vs_reference": [
                "src/core/**", "architecture", "b/B/S+/S- geometry and decoder",
                "feature set", "Host/history/calendar evidence", "initialization",
                "AdamW", "lr", "batch_size", "weight_decay", "grad_clip",
                "ema_decay and EMA-primary selection", "seeds", "data split",
                "history support", "TRAIN-only scales", "max_steps", "val_every",
                "no_stop_before", "patience_checks", "per-cell tuning (none)",
            ],
            "components_evaluated_on": "registered diagnostic subset, EMA weights, no_grad",
        },
        "n_train_rows": n,
        "train_rows_excluded": data["train_excluded"],
        "n_val_rows": len(data["val_rows"]),
        "steps_per_epoch_old_semantics": steps_per_epoch,
        "epochs_generated": int(epoch),
        "diag_subset": {"n_rows": int(diag_idx.numel()),
                        "rule": "np.linspace over registered TRAIN order, capped at 256"},
        "interleaver_audit": {
            "n_train": audit["n_train"], "bins": audit["bins"],
            "bin_sizes": audit["bin_sizes"], "all_epochs_exact_once": audit["all_epochs_exact_once"],
            "epochs_audited": epochs_needed, "oversampling": False, "reweighting": False,
        },
        "selected_step": int(best["step"]),
        "selected_check": int(best["check"]),
        "selected_val_mae_ema": float(best["val_mae"]),
        "selected_metrics": {k: v for k, v in selected_entry.items() if k != "selected"},
        "selected_after_switch": bool(int(best["step"]) > SWITCH_STEP),
        "val_optimum_post_switch": {
            "step": int(opt_post["step"]) if opt_post else None,
            "val_mae_ema": float(opt_post["val_mae_ema"]) if opt_post else None,
            "better_than_at_switch": bool(
                opt_post is not None and float(opt_post["val_mae_ema"])
                < float(next(e for e in curve if int(e["step"]) == SWITCH_STEP)["val_mae_ema"])
            ),
        },
        "selected_ema_checkpoint": (
            str(ckpt.relative_to(RC.REPO)).replace("\\", "/") if ckpt else None),
        "selected_ema_checkpoint_sha256": U.sha256(ckpt) if ckpt else None,
        "selected_ema_parameter_hash": ckpt_hash,
        "epoch_history_check_count": len(curve),
        "scales": {
            "fingerprint": data["arts"]["fingerprint"], "s_r": data["arts"]["s_r"],
            "s_b": data["arts"]["s_b"], "s_B": data["arts"]["s_B"],
            "n_train_days_fit": data["arts"]["n_train_days_fit"],
        },
        "history_support_hash": data["support"]["support_hash"],
        "history_oof_days": data["support"]["oof_days"],
        "train_day_ids_hash": U.sha256_bytes(
            json.dumps([r.day.isoformat() for r in data["train_rows"]]).encode()),
        "val_day_ids_hash": U.sha256_bytes(
            json.dumps([r.day.isoformat() for r in data["val_rows"]]).encode()),
        "recovery_common_sha256": U.sha256(RC.HERE),
        "recovery_train_sha256": U.sha256(Path(RT.__file__).resolve()),
        "o1_probe_train_sha256": U.sha256(Path(PT.__file__).resolve()),
        "o1_probe_code_hash": PC.probe_tree_digest(),
        "probe_code_hash": PC.probe_tree_digest(),
        "matched_code_hash": MC.matched_tree_digest(),
        "matched_train_sha256": U.sha256(Path(__file__).resolve()),
        "source_tree_digest": RC.source_tree_digest(),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "test_rows_dropped_from_joint_cache": int(
            RC.ACCESS_STATE["test_rows_dropped_before_return"]),
        "optional_features_active": False,
        "rescue_used": False,
        "is_registered_fit": True,
        "access_state": json.loads(json.dumps(RC.ACCESS_STATE)),
    }
    if write_outputs:
        U.json_dump(out_dir / "freeze.json", record)
        U.json_dump(out_dir / "training_curve.json", curve)
    return {"record": record, "curve": curve}
