"""O1 — auxiliary-warm-up (updates 1..400) then reconstruction-only (401..2000).

Everything except the loss schedule is the closed recovery stage's own registered
R1 run: this module *imports* ``recovery_train`` read-only and reuses
``build_cell_data``, ``to_device``, ``build_model``, ``build_optimizer``,
``grad_norms_by_group`` and ``evaluate`` unchanged, so the primary metric is
produced by the identical code path the reference used.

The training loop below is a deliberately minimal copy of
``recovery_train.train_step_based``.  Exactly three lines differ, and the diff is
independently re-checked by ``verification/verify_probe.py``:

1. the loss line selects the registered objective for the current update;
2. ``record_check`` additionally receives the step's own optimization loss and
   which objective produced it;
3. the step-0 ``record_check`` call passes the two new arguments as ``None``/``False``.

Together with ``recovery_train.evaluate`` being called verbatim, the first 400
updates consume the same random stream and the same loss as the reference, so the
checkpoints at steps 0..400 are comparable to R1's by construction.

There is no TEST path in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import probe_common as PC
import recovery_common as RC
import recovery_train as RT
# Imported under their bare names so the copied training loop below is
# textually identical to the reference loop apart from the two registered
# lines; ``verify_probe.py`` asserts that equality by diff.
from recovery_train import _bind_prior, grad_norms_by_group  # noqa: PLC2701

SWITCH_STEP = PC.SWITCH_STEP
U = RC.U

#: The frozen component keys, in the order the paper writes the loss.
COMPONENTS = ("L_rec", "L_b", "L_B", "L_S")


def objective(terms: dict, update_index: int) -> tuple:
    """The registered O1 loss schedule.

    ``update_index`` is 1-based: the update that takes the optimizer from 0 to 1
    completed updates is update 1.  Updates 1..400 use the frozen full loss;
    updates 401.. use the reconstruction term alone.
    """
    if int(update_index) <= SWITCH_STEP:
        return terms["total"], True
    return terms["L_rec"], False


def diag_components(model, ema, data: dict, diag_idx, device: str) -> dict:
    """The four frozen loss components on the registered diagnostic subset.

    Computed with the EMA weights, on the same subset and through the same
    ``compute_loss`` the reference's evaluation already uses, under ``no_grad`` in
    eval mode, so no random stream is consumed and the optimization is untouched.
    """
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
    return {k: float(comp[k].detach().item()) for k in COMPONENTS + ("total",)}


#: The components and the reference's diagnostic fields are fp32 tensors, so the
#: scale round-trip carries fp32 rounding; 1e-5 is far below any real model change
#: while staying above the representation noise.
COMPONENT_TOL = 1e-5


def _assert_component_consistency(comp: dict, stats: dict, scales) -> None:
    """The new fields must reconcile with the reference's own diagnostic fields."""
    pairs = (
        ("L_rec", "diag_reconstruction_mae", scales.s_r),
        ("L_b", "level_mae", scales.s_b),
        ("L_B", "mass_mae", scales.s_B),
    )
    for loss_key, stat_key, scale in pairs:
        lhs = comp[loss_key] * float(scale)
        rhs = float(stats[stat_key])
        if abs(lhs - rhs) > COMPONENT_TOL * max(1.0, abs(rhs)):
            raise RuntimeError(
                f"loss-component inconsistency: {loss_key}*s == {lhs!r} but {stat_key} == {rhs!r}"
            )
    rebuilt = comp["L_rec"] + (comp["L_b"] + comp["L_B"] + comp["L_S"]) / 3.0
    if abs(rebuilt - comp["total"]) > COMPONENT_TOL * max(1.0, abs(comp["total"])):
        raise RuntimeError(
            f"frozen loss formula violated: components rebuild {rebuilt!r} != total {comp['total']!r}"
        )


def train_o1(
    market: str,
    host: str,
    seed: int,
    out_dir: Path,
    *,
    max_steps: int = 2000,
    val_every: int = 50,
    no_stop_before: int = 800,
    patience_checks: int = 8,
    device: str = "cuda",
    recipe_id: str = "O1_AUX_WARMUP_400_THEN_MAE",
    switch_step: int = SWITCH_STEP,
    write_outputs: bool = True,
) -> dict:
    """One registered O1 run.  TRAIN+VAL only; TEST is unreachable."""
    import torch
    from core.training_support import (
        EMA, DifficultyInterleaver, assert_primary_profile, build_primary_train_config,
        clip_gradients, daily_difficulty, fp32_region, neural_autocast,
    )

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

    model = RT.build_model(data, device, cfg.profile, None)
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
        stats = RT.evaluate(model, ema, data, diag_idx, cfg.profile, device, grad_norms or {})
        comp = diag_components(model, ema, data, diag_idx, device)
        _assert_component_consistency(comp, stats, scales)
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
            "aux_mean": (comp["L_b"] + comp["L_B"] + comp["L_S"]) / 3.0,
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
            loss, used_full = objective(model.compute_loss(out, sub_target, sub_res, data["scales"]), step + 1)
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
        "schema": "v44_objective_probe_run.v1",
        "protocol_id": PC.PROTOCOL_ID,
        "recipe_id": recipe_id,
        "market": market, "host": host, "seed": int(seed),
        "variant": RC.VARIANT,
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
            "changed_vs_reference": ["loss schedule after update 400"],
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
        "probe_code_hash": PC.probe_tree_digest(),
        "source_tree_digest": RC.source_tree_digest(),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "test_rows_dropped_from_joint_cache": int(RC.ACCESS_STATE["test_rows_dropped_before_return"]),
        "optional_features_active": False,
        "rescue_used": False,
        "is_registered_fit": True,
        "access_state": json.loads(json.dumps(RC.ACCESS_STATE)),
    }
    if write_outputs:
        U.json_dump(out_dir / "freeze.json", record)
        U.json_dump(out_dir / "training_curve.json", curve)
    return {"record": record, "curve": curve}
