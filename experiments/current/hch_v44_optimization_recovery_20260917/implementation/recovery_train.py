"""Step-based recovery training for the frozen v4.4 G2 object (TRAIN+VAL only).

The scientific object is untouched: same architecture, geometry, features, loss,
initialization, seeds, batch size, weight decay, dropout, grad clip and EMA.  The
only changed thing is the **budget semantics**: optimization is registered in
optimizer steps with a burn-in, instead of in epochs.

Registered schedule (PROTOCOL.md §7):

* ``max_steps = 2000``, validation every ``50`` steps;
* step 0 is evaluated before the first update and remains a legal checkpoint;
* no early stop before step ``800``; afterwards patience is 8 validation checks
  (400 steps), so the earliest possible stop is step 1200;
* selection = minimum EMA VAL MAE over **all** checks including step 0.

There is no TEST path in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import recovery_common as RC

P = None  # bound lazily by _bind_prior() so importing this module has no side effect


def _bind_prior():
    """Import the completed stage's read-only runtime once per process."""
    global P
    if P is None:
        import sys

        if str(RC.PRIOR_IMPL) not in sys.path:
            sys.path.insert(0, str(RC.PRIOR_IMPL))
        import pipeline as _p  # noqa: PLC0415

        P = _p
    return P


ROLE_TRAIN, ROLE_VAL = RC.ROLE_TRAIN, RC.ROLE_VAL
DIAG_MAX_ROWS = 256


# --------------------------------------------------------------------------- data
def build_cell_data(market: str, host: str) -> dict:
    """Legal TRAIN/VAL frames, history support, TRAIN-only scales and batches."""
    import torch

    P_ = _bind_prior()
    support = P_.load_shadow_support(market, host)
    index = {d: P_.revealed(d, r, "oof") for d, r in support["residual"].items()}

    train_frame = RC.pretest_role_frame(market, host, ROLE_TRAIN)
    val_frame = RC.pretest_role_frame(market, host, ROLE_VAL)
    RC.assert_pretest_only(train_frame["days"], market)
    RC.assert_pretest_only(val_frame["days"], market)

    arts = P_.scale_artifacts(market, host, train_frame)
    scales, scaler = arts["scales"], arts["scaler"]

    train_rows, train_excluded = P_.build_rows(train_frame, index, ROLE_TRAIN)
    if len(train_rows) < 8:
        raise RuntimeError(f"{RC.U.cell_key(market, host)}: only {len(train_rows)} eligible TRAIN rows")

    val_index = dict(index)
    val_rows = []
    for i, day in enumerate(val_frame["days"]):
        if not val_frame["finite"][i]:
            continue
        window = RC.U.history_window(day, set(val_index))
        if window is None:
            continue
        val_rows.append(P_.Row(
            day=day, role=ROLE_VAL,
            host_pred=np.asarray(val_frame["host_pred"][i], dtype=np.float64),
            residual=np.asarray(val_frame["y_true"][i] - val_frame["host_pred"][i], dtype=np.float64),
            history=[val_index[d] for d in window], ordinal=i,
        ))
        val_index[day] = P_.revealed(
            day, val_frame["y_true"][i] - val_frame["host_pred"][i], "prequential"
        )
    if not val_rows:
        raise RuntimeError(f"{RC.U.cell_key(market, host)}: no eligible VAL rows")
    RC.assert_pretest_only([r.day for r in val_rows], market)

    train_batch, train_target, train_res = P_.make_batch(train_rows, scales, scaler)
    val_batch, val_target, val_res = P_.make_batch(val_rows, scales, scaler)
    return {
        "market": market, "host": host, "support": support, "scales": scales, "scaler": scaler,
        "arts": arts, "train_rows": train_rows, "val_rows": val_rows,
        "train_excluded": train_excluded,
        "train_batch": train_batch, "train_target": train_target, "train_res": train_res,
        "val_batch": val_batch, "val_target": val_target, "val_res": val_res,
    }


_BATCH_FIELDS = (
    "host", "host_hour_channels", "host_day_descriptors", "calendar_hour", "calendar_day",
    "history_day", "history_day_mask", "history_shape", "history_available", "hour_valid",
)


def to_device(data: dict, device: str) -> dict:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        data["device"] = "cpu"
        return data
    P_ = _bind_prior()
    for key in ("train_batch", "val_batch"):
        b = data[key]
        for f in _BATCH_FIELDS:
            setattr(b, f, getattr(b, f).to(device))
    for key in ("train_target", "val_target"):
        data[key] = P_.move_geometry(P_.slice_geometry(data[key], slice(None)), device)
    for key in ("train_res", "val_res"):
        data[key] = data[key].to(device)
    data["device"] = device
    return data


# --------------------------------------------------------------------------- model
def build_model(data: dict, device: str, profile, readout_init: str | None = None):
    """Fresh ``HCHV44Core`` at the frozen initialization, plus the R2b variant."""
    import torch
    from core.contracts import G2_GEOM_COORD_CONTEXT
    from core.model import HCHV44Core

    model = HCHV44Core(G2_GEOM_COORD_CONTEXT, data["scales"], profile=profile).to(device)
    if readout_init is not None:
        if readout_init != "level_affine_n0_1e-3":
            raise ValueError(f"unknown readout_init {readout_init!r}")
        # R2b: the only authorized initialization change (Level final affine only).
        with torch.no_grad():
            model.level_head.affine.weight.normal_(0.0, 1e-3)
            model.level_head.affine.bias.zero_()
    return model


def build_optimizer(model, profile, lr: float | None = None):
    """`build_primary_optimizer` verbatim, with R2a's single authorized override."""
    import torch
    from core.training_support import build_primary_optimizer

    if lr is None or float(lr) == float(profile.lr):
        return build_primary_optimizer(model, profile)
    if profile.optimizer != "AdamW":
        raise ValueError("the primary optimizer is frozen as AdamW")
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr), weight_decay=profile.weight_decay,
    )


# --------------------------------------------------------------------------- eval
def _chunked_forward(model, batch, chunk: int = 512):
    """EMA/raw forward over a full row set, chunked, under ``no_grad``."""
    import torch
    from core.training_support import fp32_region

    P_ = _bind_prior()
    n = int(batch.host.shape[0])
    outs = []
    with torch.no_grad(), fp32_region(str(batch.host.device)):
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n), device=batch.host.device)
            outs.append(model(P_.select_batch(batch, idx)))
    return outs


def _concat(outs, field):
    import torch

    parts = [getattr(o, field) for o in outs if getattr(o, field, None) is not None]
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def _stats(outs, residual) -> dict:
    """Correction / residual magnitudes and the frozen component metrics."""
    import torch
    from core.losses import structured_loss

    correction = _concat(outs, "correction")
    res = residual
    c_abs = correction.abs().mean()
    r_abs = res.abs().mean()
    return {
        "mean_abs_correction": float(c_abs.item()),
        "mean_abs_residual": float(r_abs.item()),
        "correction_ratio": float((c_abs / torch.clamp(r_abs, min=1e-12)).item()),
        "mae": float((res - correction).abs().mean().item()),
    }


def grad_norms_by_group(model) -> dict:
    """Pre-clip gradient norms of the real training step, grouped by block.

    Read straight off the training step's own gradients (no extra backward), so
    the diagnostic never perturbs the optimization stream and never needs a
    second forward in a mode cuDNN would reject.
    """
    groups = {
        "level": ("level_head",), "mass": ("mass_head",), "shape": ("shape_head",),
        "shared": ("host_encoder", "history_day_encoder", "shape_history_encoder",
                   "calendar_encoder"),
        "direct": ("direct_head",),
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


def evaluate(model, ema, data, diag_idx, profile, device: str, grad_norms: dict) -> dict:
    """One validation check.  EMA is the primary weight set; raw is diagnostic."""
    import torch
    from core.training_support import fp32_region

    P_ = _bind_prior()
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training

    # --- EMA weights: the primary evaluation
    ema.copy_to(model)
    model.eval()
    val_outs = _chunked_forward(model, data["val_batch"])
    train_outs = _chunked_forward(model, data["train_batch"])
    val_stats = _stats(val_outs, data["val_res"])
    train_stats = _stats(train_outs, data["train_res"])

    diag_batch = P_.select_batch(data["train_batch"], diag_idx)
    diag_res = data["train_res"][diag_idx]
    diag_target = P_.slice_geometry(data["train_target"], diag_idx)
    with torch.no_grad(), fp32_region(device):
        out = model(diag_batch)
        comp = model.compute_loss(out, diag_target, diag_res, data["scales"])

    # --- raw weights: diagnostic only, never selected on
    model.load_state_dict(live)
    model.eval()
    raw_out = _chunked_forward(model, data["val_batch"])
    raw_val_mae = _stats(raw_out, data["val_res"])["mae"]

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
        "diag_reconstruction_mae": float(comp["raw_mae"].detach().item()),
        "level_mae": float(comp["raw_level_abs"].detach().item()),
        "mass_mae": float(comp["raw_mass_abs"].detach().item()),
        "shape_w1": float(comp["shape_w1"].mean().detach().item()),
        "grad_norms_preclip": grad_norms,
    }


# --------------------------------------------------------------------------- train
def train_step_based(
    market: str,
    host: str,
    seed: int,
    out_dir: Path,
    *,
    max_steps: int = 2000,
    val_every: int = 50,
    no_stop_before: int = 800,
    patience_checks: int = 8,
    lr: float | None = None,
    readout_init: str | None = None,
    device: str = "cuda",
    recipe_id: str = "R1_STEP_BUDGET_2000",
) -> dict:
    """One registered recovery run.  TRAIN+VAL only; TEST is unreachable."""
    import torch
    from core.training_support import (
        EMA, DifficultyInterleaver, assert_primary_profile, build_primary_train_config,
        clip_gradients, daily_difficulty, fp32_region, neural_autocast,
    )

    RC.install_access_guard()
    RC.worker_env()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = build_primary_train_config()
    assert_primary_profile(cfg.profile)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        device = "cpu"

    data = to_device(build_cell_data(market, host), device)
    n = len(data["train_rows"])
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    epochs_needed = int(np.ceil(max_steps * cfg.batch_size / n)) + 2

    model = build_model(data, device, cfg.profile, readout_init)
    optimizer = build_optimizer(model, cfg.profile, lr)
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
        np.unique(np.linspace(0, n - 1, min(n, DIAG_MAX_ROWS)).astype(np.int64)),
        dtype=torch.long,
    )
    diag_idx = diag_idx.to(device)

    def _lr_now() -> float:
        return float(optimizer.param_groups[0]["lr"])

    curve: list[dict] = []
    best = {"val_mae": float("inf"), "step": -1, "shadow": None, "check": -1}
    check_index = 0

    def record_check(step: int, samples_seen: int, is_initial: bool, grad_norms: dict | None) -> None:
        nonlocal check_index, best
        stats = evaluate(model, ema, data, diag_idx, cfg.profile, device, grad_norms or {})
        entry = {
            "check": check_index, "step": int(step), "is_initial": bool(is_initial),
            "samples_seen": int(samples_seen),
            "epoch_equivalent": float(samples_seen / n),
            "lr": _lr_now(),
            **stats,
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

    record_check(0, 0, True, None)

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
            loss = model.compute_loss(out, sub_target, sub_res, data["scales"])["total"]
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
            record_check(step, samples_seen, False, norms)
            if step >= no_stop_before and (check_index - 1 - best["check"]) >= patience_checks:
                stopped_at = step
                break

    if best["shadow"] is None:
        raise RuntimeError("no EMA checkpoint selected")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "selected_ema.pt"
    torch.save(best["shadow"], ckpt)
    keys = sorted(best["shadow"])
    ckpt_hash = RC.U.array_hash(*[best["shadow"][k].detach().cpu().numpy() for k in keys])
    selected_entry = next(e for e in curve if e["step"] == best["step"] and e["selected"])

    record = {
        "schema": "v44_optimization_recovery_run.v1",
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
            "readout_init": readout_init,
            "stopped_at_step": stopped_at,
            "slots_available": int(max_steps),
        },
        "n_train_rows": n,
        "train_rows_excluded": data["train_excluded"],
        "n_val_rows": len(data["val_rows"]),
        "steps_per_epoch_old_semantics": steps_per_epoch,
        "epochs_generated": int(epoch),
        "diag_subset": {"n_rows": int(diag_idx.numel()), "rule": "np.linspace over registered TRAIN order, capped at 256"},
        "interleaver_audit": {
            "n_train": audit["n_train"], "bins": audit["bins"],
            "bin_sizes": audit["bin_sizes"], "all_epochs_exact_once": audit["all_epochs_exact_once"],
            "epochs_audited": epochs_needed, "oversampling": False, "reweighting": False,
        },
        "selected_step": int(best["step"]),
        "selected_check": int(best["check"]),
        "selected_val_mae_ema": float(best["val_mae"]),
        "selected_metrics": {k: v for k, v in selected_entry.items() if k not in ("selected",)},
        "selected_ema_checkpoint": str(ckpt.relative_to(RC.REPO)).replace("\\", "/"),
        "selected_ema_checkpoint_sha256": RC.U.sha256(ckpt),
        "selected_ema_parameter_hash": ckpt_hash,
        "epoch_history_check_count": len(curve),
        "scales": {
            "fingerprint": data["arts"]["fingerprint"], "s_r": data["arts"]["s_r"],
            "s_b": data["arts"]["s_b"], "s_B": data["arts"]["s_B"],
            "n_train_days_fit": data["arts"]["n_train_days_fit"],
        },
        "history_support_hash": data["support"]["support_hash"],
        "history_oof_days": data["support"]["oof_days"],
        "train_day_ids_hash": RC.U.sha256_bytes(json.dumps([r.day.isoformat() for r in data["train_rows"]]).encode()),
        "val_day_ids_hash": RC.U.sha256_bytes(json.dumps([r.day.isoformat() for r in data["val_rows"]]).encode()),
        "code_hash": RC.U.sha256(RC.HERE),
        "common_code_hash": RC.U.sha256(RC.HERE.parent / "recovery_common.py"),
        "source_tree_digest": RC.source_tree_digest(),
        "test_target_read_count": 0,
        "test_rows_materialised": 0,
        "test_rows_dropped_from_joint_cache": int(RC.ACCESS_STATE["test_rows_dropped_before_return"]),
        "optional_features_active": False,
        "rescue_used": False,
        "is_registered_fit": True,
        "access_state": json.loads(json.dumps(RC.ACCESS_STATE)),
    }
    RC.U.json_dump(out_dir / "freeze.json", record)
    RC.U.json_dump(out_dir / "training_curve.json", curve)
    return record
