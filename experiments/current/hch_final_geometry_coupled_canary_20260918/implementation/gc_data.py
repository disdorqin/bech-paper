"""Legal TRAIN/VAL data path for the final geometry-coupled canary.

Every frame comes from ``recovery_common.pretest_role_frame``, which refuses any
role other than ``TRAIN``/``VAL`` and counts the refusal, and every row is proved
to lie outside the sealed segment by ``assert_pretest_only`` before a batch is
built.  The causal history window, the TRAIN-only coordinate scales and the frozen
Host features come from the completed full-panel stage's ``pipeline``, read-only,
so the canary's inputs are the identical objects rather than a re-implementation.

The one thing this stage adds is the deterministic W=7 geometry prior, built from
the same revealed residual days the history bundle would have used.

There is no TEST path in this module and none is importable from it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import gc_common as GC

ROLE_TRAIN, ROLE_VAL = GC.ROLE_TRAIN, GC.ROLE_VAL
DIAG_MAX_ROWS = 256


def build_cell_data(market: str, host: str) -> dict:
    """Legal TRAIN/VAL rows, TRAIN-only scales and the final-method inputs."""
    import torch
    from core.geometry import residual_geometry

    U = GC.U
    P_ = GC.lazy_pipeline()

    support = P_.load_shadow_support(market, host)
    index = {d: P_.revealed(d, r, "oof") for d, r in support["residual"].items()}

    train_frame = GC.RC.pretest_role_frame(market, host, ROLE_TRAIN)
    val_frame = GC.RC.pretest_role_frame(market, host, ROLE_VAL)
    GC.RC.assert_pretest_only(train_frame["days"], market)
    GC.RC.assert_pretest_only(val_frame["days"], market)

    arts = P_.scale_artifacts(market, host, train_frame)
    scales, scaler = arts["scales"], arts["scaler"]

    train_rows, train_excluded = P_.build_rows(train_frame, index, ROLE_TRAIN)
    if len(train_rows) < 8:
        raise RuntimeError(f"{GC.cell_key(market, host)}: only {len(train_rows)} eligible TRAIN rows")

    val_index = dict(index)
    val_rows = []
    for i, day in enumerate(val_frame["days"]):
        if not val_frame["finite"][i]:
            continue
        window = U.history_window(day, set(val_index))
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
        raise RuntimeError(f"{GC.cell_key(market, host)}: no eligible VAL rows")
    GC.RC.assert_pretest_only([r.day for r in val_rows], market)

    train_input, train_res = build_inputs(train_rows, scaler)
    val_input, val_res = build_inputs(val_rows, scaler)
    return {
        "market": market,
        "host": host,
        "support": support,
        "scales": scales,
        "scaler": scaler,
        "arts": arts,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "val_days": [r.day.isoformat() for r in val_rows],
        "train_excluded": train_excluded,
        "train_input": train_input,
        "val_input": val_input,
        "train_res": train_res,
        "val_res": val_res,
        "train_target": residual_geometry(train_res),
        "val_target": residual_geometry(val_res),
    }


def reference_scales(market: str, host: str):
    """TRAIN-only coordinate scales for one cell, without building any row.

    Used by the source audit and the parameter/latency measurement, both of which
    need a real ``CoordinateScales`` but no data.
    """
    P_ = GC.lazy_pipeline()
    train_frame = GC.RC.pretest_role_frame(market, host, ROLE_TRAIN)
    GC.RC.assert_pretest_only(train_frame["days"], market)
    return P_.scale_artifacts(market, host, train_frame)["scales"]


def build_inputs(rows: list, scaler):
    """``FinalInput`` plus the raw residual tensor for a list of legal rows."""
    import torch
    from core.calendar import day_channels, hour_channels
    from core.feature_engineering import host_day_descriptors, host_hour_channels

    from final_model import FinalInput
    from fused_encoder import CALENDAR_HOUR_CHANNELS
    from geometry_prior import build_geometry_prior

    if not rows:
        raise ValueError("empty row list")
    U = GC.U
    host = torch.as_tensor(np.stack([r.host_pred for r in rows]).astype(np.float32))
    residual = torch.as_tensor(np.stack([r.residual for r in rows]).astype(np.float32))
    days = [r.day for r in rows]
    dates = [d.isoformat() for d in days]
    prior = build_geometry_prior(
        [r.history for r in rows], target_days=days, origins=[U.origin_of(d) for d in days]
    )
    inp = FinalInput(
        host_hour_channels=host_hour_channels(host, scaler),
        calendar_hour=hour_channels(dates),
        calendar_day=day_channels(dates),
        host_day_descriptors=host_day_descriptors(host, scaler),
        hour_valid=torch.ones(len(rows), U.HORIZON),
        prior=prior,
        date=dates,
    )
    return inp, residual


def select_input(inp, idx):
    """Row-subset of a ``FinalInput`` (same object type, sliced fields)."""
    from final_model import FinalInput
    from geometry_prior import GeometryPriorBatch

    prior = inp.prior
    sub = GeometryPriorBatch(**{f: getattr(prior, f)[idx] for f in prior.__dataclass_fields__})
    return FinalInput(
        host_hour_channels=inp.host_hour_channels[idx],
        calendar_hour=inp.calendar_hour[idx],
        calendar_day=inp.calendar_day[idx],
        host_day_descriptors=inp.host_day_descriptors[idx],
        hour_valid=inp.hour_valid[idx],
        prior=sub,
        date=[inp.date[i] for i in idx.tolist()] if inp.date else (),
    )


def slice_geometry(geom, idx):
    from core.geometry import ResidualGeometry

    take = {
        "b": geom.b[idx], "B": geom.B[idx], "P": geom.P[idx], "N": geom.N[idx],
        "s_plus": geom.s_plus[idx], "s_minus": geom.s_minus[idx],
        "valid": geom.valid[idx], "m1": geom.m1[idx], "horizon": geom.horizon[idx],
        "eta": geom.eta[idx],
        "pos_active": geom.pos_active[idx], "neg_active": geom.neg_active[idx],
    }
    return ResidualGeometry(**take)


def to_device(data: dict, device: str) -> dict:
    import torch

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        data["device"] = "cpu"
        return data
    for key in ("train_input", "val_input"):
        inp = data[key]
        for f in ("host_hour_channels", "calendar_hour", "calendar_day",
                  "host_day_descriptors", "hour_valid"):
            setattr(inp, f, getattr(inp, f).to(device))
        inp.prior = inp.prior.to(device)
    for key in ("train_target", "val_target"):
        data[key] = GC.lazy_pipeline().move_geometry(
            slice_geometry(data[key], slice(None)), device
        )
    for key in ("train_res", "val_res"):
        data[key] = data[key].to(device)
    data["device"] = device
    return data


def synthetic_input(batch: int, scales, seed: int = 0, device: str = "cpu"):
    """A synthetic legal ``FinalInput`` used only to measure parameter count and latency.

    It carries no data from any market: the point is a fixed, reproducible
    inference-cost measurement, not a scientific input.
    """
    import torch

    from final_model import FinalInput
    from fused_encoder import CALENDAR_HOUR_CHANNELS
    from geometry_prior import GeometryPriorBatch

    U = GC.U
    g = torch.Generator().manual_seed(int(seed))
    prior = GeometryPriorBatch(
        b_bar=torch.randn(batch, generator=g) * 0.1,
        B_bar=torch.rand(batch, generator=g) * 0.5,
        s_plus_bar=torch.softmax(torch.randn(batch, U.HORIZON, generator=g), dim=-1),
        s_minus_bar=torch.softmax(torch.randn(batch, U.HORIZON, generator=g), dim=-1),
        avail_plus=torch.ones(batch), avail_minus=torch.ones(batch), avail=torch.ones(batch),
        n_days_used=torch.full((batch,), U.HISTORY_DAYS, dtype=torch.int64),
    ).to(device)
    return FinalInput(
        host_hour_channels=torch.randn(batch, U.HORIZON, 5, generator=g).to(device),
        calendar_hour=torch.randn(batch, U.HORIZON, CALENDAR_HOUR_CHANNELS, generator=g).to(device),
        calendar_day=torch.randn(batch, 5, generator=g).to(device),
        host_day_descriptors=torch.randn(batch, 13, generator=g).to(device),
        hour_valid=torch.ones(batch, U.HORIZON, device=device),
        prior=prior,
        date=[f"synthetic-{i:05d}" for i in range(batch)],
    )


def diag_index(n_rows: int, device: str = "cpu"):
    """The registered diagnostic subset: ``np.linspace`` over TRAIN order, cap 256."""
    import torch

    idx = torch.as_tensor(
        np.unique(np.linspace(0, n_rows - 1, min(n_rows, DIAG_MAX_ROWS)).astype(np.int64)),
        dtype=torch.long,
    )
    return idx.to(device)


def frame_summary(data: dict) -> dict:
    return {
        "market": data["market"],
        "host": data["host"],
        "n_train_rows": len(data["train_rows"]),
        "n_val_rows": len(data["val_rows"]),
        "train_excluded": data["train_excluded"],
        "train_day_ids_hash": GC.U.sha256_bytes(
            __import__("json").dumps([r.day.isoformat() for r in data["train_rows"]]).encode()
        ),
        "val_day_ids_hash": GC.U.sha256_bytes(
            __import__("json").dumps([r.day.isoformat() for r in data["val_rows"]]).encode()
        ),
        "history_support_hash": data["support"]["support_hash"],
        "history_oof_days": data["support"]["oof_days"],
        "scales": {
            "fingerprint": data["arts"]["fingerprint"],
            "s_r": data["arts"]["s_r"],
            "s_b": data["arts"]["s_b"],
            "s_B": data["arts"]["s_B"],
            "n_train_days_fit": data["arts"]["n_train_days_fit"],
        },
        "val_days": list(data.get("val_days", [])),
        "prior_availability_fraction": float(data["val_input"].prior.avail.mean().item()),
        "prior_days_used_min": int(data["val_input"].prior.n_days_used.min().item()),
    }
