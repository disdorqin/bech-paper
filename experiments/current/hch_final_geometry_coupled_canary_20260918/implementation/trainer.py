"""The frozen O1 staged-objective training schedule for the final method.

The optimization contract is copied from the completed O1 evidence, not tuned:

* ``max_steps = 2000`` optimizer steps, validation every ``50`` steps;
* step 0 is evaluated before the first update and is a legal checkpoint;
* updates ``1..400`` minimize ``L_rec + (L_b + L_B + L_S)/3``;
* updates ``401..2000`` minimize ``L_rec`` only;
* no early stop before step ``800``; afterwards patience is 8 validation checks;
* selection = minimum EMA VAL MAE over **all** checks including step 0;
* AdamW, lr 1e-3, weight decay 1e-4, batch 32, grad clip 1.0, EMA decay 0.995.

The DIRECT arm has no coordinate auxiliaries, so it uses ``L_rec`` throughout with
the identical step budget, optimizer, EMA, batch and checkpoint policy.

There is no TEST path in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import gc_common as GC
import gc_data as GD

ROLE_TRAIN, ROLE_VAL = GC.ROLE_TRAIN, GC.ROLE_VAL
SWITCH_STEP = GC.SWITCH_STEP


def staged_objective(terms: dict, update_index: int, structured: bool = True,
                     switch_step: int = SWITCH_STEP) -> tuple:
    """The frozen staged objective.  No weight or switch search is permitted.

    ``update_index`` is the 1-based index of the update about to be applied, so
    updates ``1..switch_step`` use the full geometric objective and everything
    after it uses reconstruction only.  The unit-test suite pins both boundaries.
    """
    if int(update_index) < 1:
        raise ValueError("update_index is 1-based; there is no update 0")
    if not structured:
        return terms["L_rec"], False
    if int(update_index) <= int(switch_step):
        return terms["total"], True
    return terms["L_rec"], False


def _stats(outs, residual) -> dict:
    import torch

    correction = outs["correction"]
    c_abs = correction.abs().mean()
    r_abs = residual.abs().mean()
    return {
        "mean_abs_correction": float(c_abs.item()),
        "mean_abs_residual": float(r_abs.item()),
        "correction_ratio": float((c_abs / torch.clamp(r_abs, min=1e-12)).item()),
        "mae": float((residual - correction).abs().mean().item()),
    }


def _chunked_forward(model, inp, chunk: int = 512) -> dict:
    """Chunked ``no_grad`` fp32 forward; returns concatenated output fields."""
    import torch
    from core.training_support import fp32_region

    n = int(inp.host_hour_channels.shape[0])
    device = str(inp.host_hour_channels.device)
    fields = ("correction", "b_hat", "B_hat", "s_plus_hat", "s_minus_hat",
              "a_plus", "a_minus", "overlap_gap", "m_plus", "m_minus")
    parts: dict = {f: [] for f in fields}
    with torch.no_grad(), fp32_region(device):
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n), device=device)
            out = model(GD.select_input(inp, idx))
            for f in fields:
                value = getattr(out, f, None)
                if value is not None:
                    parts[f].append(value)
    return {f: (torch.cat(v, dim=0) if v else None) for f, v in parts.items()}


def grad_norms_by_group(model) -> dict:
    """Pre-clip gradient norms of the real training step, grouped by block."""
    groups = {
        "encoder": ("encoder",),
        "day_core": ("day_core",),
        "shape_head": ("shape_head",),
        "direct_head": ("direct_head",),
    }
    out = {}
    for name, prefixes in groups.items():
        total = 0.0
        seen = 0
        for pname, param in model.named_parameters():
            if param.grad is None or not pname.startswith(prefixes):
                continue
            total += float(param.grad.detach().pow(2).sum().item())
            seen += 1
        out[name] = float(np.sqrt(total)) if seen else None
    return out


def _component_diagnostics(model, inp, target, residual, scales, valid) -> dict:
    """Component errors on a fixed diagnostic subset (never a selection metric)."""
    import torch
    from core.losses import component_losses
    from core.training_support import fp32_region

    with torch.no_grad(), fp32_region(str(inp.host_hour_channels.device)):
        out = model(inp)
        comp = component_losses(
            correction=out.correction, b_hat=out.b_hat, B_hat=out.B_hat,
            s_plus_hat=out.s_plus_hat, s_minus_hat=out.s_minus_hat,
            residual=residual, target=target, scales=scales, valid=valid,
        )
    return {k: float(v.detach().item()) if torch.is_tensor(v) else v for k, v in comp.items()}


def evaluate(model, ema, data, diag_idx, module) -> dict:
    """One validation check.  EMA is the primary weight set; raw is diagnostic."""
    import torch
    from core.training_support import fp32_region

    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training

    ema.copy_to(model)
    model.eval()
    val_outs = _chunked_forward(model, data["val_input"])
    train_outs = _chunked_forward(model, data["train_input"])
    val_stats = _stats(val_outs, data["val_res"])
    train_stats = _stats(train_outs, data["train_res"])

    diag = {}
    if model.structured:
        sub_target = GD.slice_geometry(data["train_target"], diag_idx)
        diag = _component_diagnostics(
            model, GD.select_input(data["train_input"], diag_idx), sub_target,
            data["train_res"][diag_idx], data["scales"], sub_target.valid,
        )

    model.load_state_dict(live)
    model.eval()
    raw_val = _stats(_chunked_forward(model, data["val_input"]), data["val_res"])["mae"]

    model.load_state_dict(live)
    model.train(was_training)

    host_mae = val_stats["mean_abs_residual"]
    return {
        "val_mae_ema": val_stats["mae"],
        "val_mae_raw": raw_val,
        "val_host_mae": host_mae,
        "val_gain_vs_host_pct": float(
            (host_mae - val_stats["mae"]) / max(host_mae, 1e-12) * 100.0
        ),
        "val_correction_ratio": val_stats["correction_ratio"],
        "val_mean_abs_correction": val_stats["mean_abs_correction"],
        "train_mae_ema": train_stats["mae"],
        "train_host_mae": train_stats["mean_abs_residual"],
        "train_correction_ratio": train_stats["correction_ratio"],
        "diag_reconstruction_mae": diag.get("reconstruction"),
        "diag_level": diag.get("level"),
        "diag_B": diag.get("B"),
        "diag_shape": diag.get("shape"),
    }


def _save_val_predictions(path: Path, outs: dict, data: dict) -> dict:
    """Persist VAL predictions so the decoder identities can be re-checked."""
    import torch

    payload = {"residual": data["val_res"].detach().cpu().numpy()}
    host = np.stack([np.asarray(r.host_pred, dtype=np.float32) for r in data["val_rows"]])
    payload["host_pred"] = host
    payload["hour_valid"] = data["val_input"].hour_valid.detach().cpu().numpy()
    payload["day_id"] = np.asarray(data["val_days"], dtype=object).astype("U10")
    for name, value in outs.items():
        if value is None:
            continue
        payload[name] = value.detach().cpu().numpy()
    np.savez_compressed(path, **payload)
    return {
        "path": str(path.name),
        "sha256": GC.sha256_file(path),
        "fields": sorted(payload),
        "n_rows": int(payload["residual"].shape[0]),
    }


def train_final(
    variant: str,
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
    recipe_id: str = GC.RECIPE_GC,
    write_outputs: bool = True,
) -> dict:
    """One registered canary fit.  TRAIN+VAL only; TEST is unreachable."""
    import torch
    from core.training_support import (
        EMA, DifficultyInterleaver, assert_primary_profile, build_primary_optimizer,
        build_primary_train_config, clip_gradients, daily_difficulty, fp32_region,
        neural_autocast,
    )

    from final_model import HCHFinalCore

    GC.RC.install_access_guard()
    GC.init_worker()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = build_primary_train_config()
    assert_primary_profile(cfg.profile)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        device = "cpu"

    data = GD.to_device(GD.build_cell_data(market, host), device)
    n = len(data["train_rows"])
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    epochs_needed = int(np.ceil(max_steps * cfg.batch_size / n)) + 2

    torch.manual_seed(seed)
    model = HCHFinalCore(variant, data["scales"], dropout=cfg.dropout).to(device)
    optimizer = build_primary_optimizer(model, cfg.profile)
    ema = EMA(model, decay=cfg.ema_decay)
    interleaver = DifficultyInterleaver(
        ids=[r.day.isoformat() for r in data["train_rows"]],
        difficulty=[float(d) for d in daily_difficulty(data["train_res"].detach().cpu())],
        split="TRAIN", bins=4, seed=seed,
    )
    audit = interleaver.audit(epochs_needed)
    if not audit["all_epochs_exact_once"]:
        raise RuntimeError("difficulty interleaver failed the once-per-epoch contract")

    diag_idx = GD.diag_index(n, device)
    structured = bool(model.structured)

    def objective(terms: dict, update_index: int) -> tuple:
        return staged_objective(terms, update_index, structured)

    def _lr_now() -> float:
        return float(optimizer.param_groups[0]["lr"])

    curve: list[dict] = []
    best = {"val_mae": float("inf"), "step": -1, "shadow": None, "check": -1}
    check_index = 0

    def record_check(step: int, samples_seen: int, is_initial: bool,
                     grad_norms: dict | None, step_loss, used_full) -> None:
        nonlocal check_index, best
        stats = evaluate(model, ema, data, diag_idx, model)
        entry = {
            "check": check_index, "step": int(step), "is_initial": bool(is_initial),
            "samples_seen": int(samples_seen),
            "epoch_equivalent": float(samples_seen / n),
            "lr": _lr_now(),
            **stats,
            "step_loss_optimized": (None if step_loss is None else float(step_loss)),
            "is_optimization_step": not bool(is_initial),
            "used_full_loss_at_this_step": (None if is_initial else bool(used_full)),
            "loss_schedule": ("L_rec_plus_aux" if used_full else "L_rec_only"),
            "objective_switch_here": bool(int(step) == SWITCH_STEP),
            "grad_norms_preclip": grad_norms or {},
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
            out = model(GD.select_input(data["train_input"], idx))
        with fp32_region(device):
            sub_target = GD.slice_geometry(data["train_target"], idx)
            sub_res = data["train_res"][idx]
            terms = model.compute_loss(out, sub_target, sub_res, data["scales"])
            loss, used_full = objective(terms, step + 1)
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
    selected_entry = next(e for e in curve if e["step"] == best["step"] and e["selected"])

    ckpt_hash = None
    predictions = None
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "selected_ema.pt"
        torch.save(best["shadow"], ckpt)
        keys = sorted(best["shadow"])
        ckpt_hash = GC.U.array_hash(*[best["shadow"][k].detach().cpu().numpy() for k in keys])

        # Re-evaluate the *selected* EMA weights and persist the VAL predictions.
        # ``best["shadow"]`` is the EMA snapshot taken at the selected check, i.e.
        # exactly the weights ``selected_metrics`` describes and ``selected_ema.pt``
        # holds.  The live ``ema`` keeps moving after that check, so copying it here
        # would describe the last step of the run instead of the selected one and
        # would break ``mean|r - c| == selected_val_mae_ema`` for every early stop.
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best["shadow"])
        model.eval()
        val_outs = _chunked_forward(model, data["val_input"])
        predictions = _save_val_predictions(out_dir / "val_predictions.npz", val_outs, data)
        model.load_state_dict(live)

    freeze = {
        "schema": "hch_final_gc_run.v1",
        "protocol_id": GC.PROTOCOL_ID,
        "recipe_id": recipe_id,
        "variant": variant,
        "market": market, "host": host, "seed": int(seed),
        "device": device,
        "optimization": {
            "optimizer": cfg.optimizer, "lr": _lr_now(),
            "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
            "grad_clip": cfg.grad_clip, "ema_decay": cfg.ema_decay,
            "eval_weights": cfg.eval_weights,
            "max_steps": int(max_steps), "val_every": int(val_every),
            "no_stop_before": int(no_stop_before), "patience_checks": int(patience_checks),
            "switch_step": int(SWITCH_STEP),
            "objective": ("L_rec + (L_b + L_B + L_S)/3 through step 400, L_rec afterwards"
                          if structured else "L_rec throughout (DIRECT has no coordinate auxiliaries)"),
            "stopped_at_step": stopped_at,
            "slots_available": int(max_steps),
        },
        "variant_config": model.variant_config(),
        "n_train_rows": n,
        "n_val_rows": len(data["val_rows"]),
        "train_rows_excluded": data["train_excluded"],
        "steps_per_epoch": steps_per_epoch,
        "epochs_generated": int(epoch),
        "diag_subset": {"n_rows": int(diag_idx.numel()),
                        "rule": "np.linspace over registered TRAIN order, capped at 256"},
        "interleaver_audit": {
            "n_train": audit["n_train"], "bins": audit["bins"],
            "bin_sizes": audit["bin_sizes"],
            "all_epochs_exact_once": audit["all_epochs_exact_once"],
            "epochs_audited": epochs_needed, "oversampling": False, "reweighting": False,
        },
        "selected_step": int(best["step"]),
        "selected_check": int(best["check"]),
        "selected_val_mae_ema": float(best["val_mae"]),
        "selected_metrics": {k: v for k, v in selected_entry.items() if k != "selected"},
        "selected_after_switch": bool(int(best["step"]) > SWITCH_STEP),
        "checkpoint": (str(Path(out_dir, "selected_ema.pt").relative_to(GC.REPO)).replace("\\", "/")
                       if write_outputs else None),
        "selected_ema_checkpoint_sha256": ckpt_hash,
        "selected_ema_parameter_hash": (
            GC.U.array_hash(*[best["shadow"][k].detach().cpu().numpy() for k in sorted(best["shadow"])])
            if write_outputs else None
        ),
        "val_predictions": predictions,
        "check_count": len(curve),
        "frame": GD.frame_summary(data),
        "code_hash": GC.sha256_file(Path(__file__)),
        "common_code_hash": GC.sha256_file(GC.HERE),
        "source_tree_digest": GC.RC.source_tree_digest(),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "test_rows_dropped_from_joint_cache": int(
            GC.RC.ACCESS_STATE["test_rows_dropped_before_return"]
        ),
        "optional_features_active": False,
        "rescue_used": False,
        "is_registered_fit": True,
        "access_state": json.loads(json.dumps(GC.RC.ACCESS_STATE)),
    }
    if write_outputs:
        GC.write_json(out_dir / "freeze.json", freeze)
        GC.write_json(out_dir / "training_curve.json", curve)
    # The curve is returned in memory as well as (optionally) written, so the unit
    # gate can assert the staged schedule on a non-registered smoke run.
    return {**freeze, "curve": curve}
