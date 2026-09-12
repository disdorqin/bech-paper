"""Chronological ``POST_TRAIN`` trainer and the frozen inference path.

Every fitted quantity in this module is fitted on a caller-supplied **training
prefix**.  The module has no notion of ``DEV_EVAL``: it cannot see a role, only a
list of row positions, and the caller (``oof.py``) is the only place roles are
resolved.  That is deliberate -- it makes "``DEV_EVAL`` never influenced fitting"
a property of the call graph rather than a promise.

Nothing scientific is defined here.  The model, the feature canonicalization, the
fusion, the losses, the sampler and the calibration all come from ``src/core``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .china5_adapter import CellDataset
from .contracts import HarnessError, LegalityError
from .core_bridge import attr


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_model(n_features: int, switches: Mapping[str, bool], seed: int,
                amplitude_scale: float = C.AMPLITUDE_SCALE_FLOOR):
    """Build one ``SignedMassRepairModel`` for a market's legal feature count.

    ``F`` is the only thing that varies across markets; the recipe does not.  The
    four switches are the registered removable components and are the *only*
    thing an ablation may change.

    ``amplitude_scale`` is the design's ``s_A``: the training-prefix robust mass
    scale that gives the two softplus heads their units.  It is a *fitted
    statistic*, not a hyperparameter -- the caller passes the value
    :func:`fit_amplitude_scale` produced for the very prefix this model is
    trained on, and nothing about it is tuned or ablatic.
    """
    unknown = set(switches) - set(C.SWITCH_NAMES)
    if unknown:
        raise HarnessError(f"unknown model switch(es): {sorted(unknown)}")
    scale = float(amplitude_scale)
    if not (scale > 0.0) or not np.isfinite(scale):
        raise HarnessError(f"amplitude_scale must be finite and positive, got {scale}")

    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    default_config = attr("default_config")
    SignedMassRepairModel = attr("SignedMassRepairModel")

    cfg = default_config(
        n_forecast_features=int(n_features),
        n_calendar_features=C.N_CALENDAR_FEATURES,
        history_windows=C.HISTORY_WINDOWS,
        knn_enabled=False,                       # KNN is OFF and stays OFF
        shape_semantic_context=bool(switches["shape_semantic_context"]),
        use_tcn=bool(switches["use_tcn"]),
        untied_amplitude_heads=bool(switches["untied_amplitude_heads"]),
        # Recorded for provenance only: the flag is inert inside the model and
        # is consumed by the batch orderer.  It changes no architecture.
        rare_mass_sampling=bool(switches["rare_mass_sampling"]),
        amplitude_scale=scale,
    )
    if cfg.knn_enabled or cfg.shape_context_dim != 0:
        raise HarnessError("KNN must remain disabled in the signed-mass harness")
    if cfg.temporal_rnn != "gru":
        raise HarnessError("the registered default encoder is GRU32")
    return SignedMassRepairModel(cfg)


def _torch():
    import torch

    return torch


# --------------------------------------------------------------------------
# Fitted statistics -- training prefix only
# --------------------------------------------------------------------------
@dataclass
class FittedStats:
    """Every statistic the method fits, with the prefix it was fitted on."""

    scaler_state: Dict[str, list]
    residual_scale: float
    mass_center: List[float]
    mass_scale: List[float]
    n_fit_rows: int
    fit_rows_sha256: str
    amplitude_scale: float = C.AMPLITUDE_SCALE_FLOOR
    amplitude_scale_rule: Optional[Dict[str, Any]] = None
    strata: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scaler_state": self.scaler_state,
            "residual_scale": self.residual_scale,
            "mass_center": self.mass_center,
            "mass_scale": self.mass_scale,
            "n_fit_rows": self.n_fit_rows,
            "fit_rows_sha256": self.fit_rows_sha256,
            "amplitude_scale": self.amplitude_scale,
            "amplitude_scale_rule": self.amplitude_scale_rule,
            "rare_mass_strata": self.strata,
        }


def fit_amplitude_scale(mass_positive, mass_negative) -> Tuple[float, Dict[str, Any]]:
    """The design's robust Amplitude output scale ``s_A`` for one legal prefix.

    .. math::

        s_A = \\operatorname{median}\\{A_d^+, A_d^- : A_d^{\\pm} > 0\\}

    The pool is *both* sign channels of the same prefix, so the two heads share
    one set of units and upper/lower asymmetry has to come from the heads rather
    than from giving them different hand-set scales.  The floor applies only
    when the pool is empty or degenerate, which the design names as the one
    admissible use of a fixed number.

    This is the harness's own arithmetic because ``src/core`` exposes the scale
    as a *configuration input* (``ModelConfig.amplitude_scale``) rather than as
    a fitting routine; the design authority specifies the rule and the core
    applies it.  No model, loss or fusion behaviour is defined here.
    """
    pos = np.asarray(mass_positive, dtype=np.float64).reshape(-1)
    neg = np.asarray(mass_negative, dtype=np.float64).reshape(-1)
    pool = np.concatenate([pos, neg])
    pool = pool[np.isfinite(pool) & (pool > 0.0)]

    rule: Dict[str, Any] = {
        "rule": "median{A_d^+, A_d^- : A_d^pm > 0} over the fitting prefix",
        "shared_by_both_sign_heads": True,
        "floor": float(C.AMPLITUDE_SCALE_FLOOR),
        "n_days": int(pos.size),
        "n_positive_mass_days": int((pos > 0.0).sum()),
        "n_negative_mass_days": int((neg > 0.0).sum()),
        "n_pooled_terms": int(pool.size),
    }
    if pool.size == 0:
        rule["degenerate"] = "NO_POSITIVE_MASS_IN_PREFIX"
        rule["value"] = float(C.AMPLITUDE_SCALE_FLOOR)
        return float(C.AMPLITUDE_SCALE_FLOOR), rule

    value = float(np.median(pool))
    if not np.isfinite(value) or value <= 0.0:
        rule["degenerate"] = "NON_POSITIVE_MEDIAN"
        rule["raw_median"] = value
        rule["value"] = float(C.AMPLITUDE_SCALE_FLOOR)
        return float(C.AMPLITUDE_SCALE_FLOOR), rule

    rule["value"] = value
    return value, rule


def _rows_digest(rows: np.ndarray) -> str:
    import hashlib

    payload = np.ascontiguousarray(rows, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest().upper()


def fit_stats(dataset: CellDataset, fit_rows: Sequence[int],
              switches: Mapping[str, bool],
              seed: int) -> FittedStats:
    """Fit the scaler, the residual scale and the rare-mass strata.

    ``fit_rows`` is the caller's training prefix.  The prompt's chronology is
    enforced by the caller passing a prefix; this function additionally refuses
    roles it must never see, as a backstop.
    """
    torch = _torch()
    TrainFrozenScaler = attr("TrainFrozenScaler")
    DeterministicFeatureBuilder = attr("DeterministicFeatureBuilder")
    fit_residual_scale = attr("fit_residual_scale")
    fit_mass_normalization = attr("fit_mass_normalization")
    decompose_residual = attr("decompose_residual")

    rows = np.asarray(fit_rows, dtype=np.int64)
    if rows.size == 0:
        raise LegalityError("fit_stats called with an empty training prefix")
    _reject_forbidden_rows(dataset, rows, "scaler/residual-scale fitting")

    numeric, numeric_mask = dataset.numeric_block(rows)
    scaler = TrainFrozenScaler().fit(torch.as_tensor(numeric),
                                     torch.as_tensor(numeric_mask))

    residual = torch.as_tensor(dataset.residual[rows], dtype=torch.float32)
    valid = torch.as_tensor(dataset.valid_mask[rows], dtype=torch.bool)
    residual_scale = float(fit_residual_scale(residual, valid))

    targets = decompose_residual(residual, valid)
    center, scale = fit_mass_normalization(targets.mass_positive,
                                           targets.mass_negative)
    amplitude_scale, amplitude_rule = fit_amplitude_scale(targets.mass_positive,
                                                          targets.mass_negative)

    strata = None
    if switches["rare_mass_sampling"]:
        RareMassSampler = attr("RareMassSampler")
        RareMassSamplerConfig = attr("RareMassSamplerConfig")
        sampler = RareMassSampler(RareMassSamplerConfig(
            enabled=True, mode="spread", seed=int(seed)))
        sampler.fit(targets.mass_positive, targets.mass_negative)
        strata = {str(k): int(v) for k, v in sampler.stratum_counts().items()}

    return FittedStats(
        scaler_state=scaler.state(),
        residual_scale=residual_scale,
        mass_center=[float(x) for x in center],
        mass_scale=[float(x) for x in scale],
        n_fit_rows=int(rows.size),
        fit_rows_sha256=_rows_digest(rows),
        amplitude_scale=float(amplitude_scale),
        amplitude_scale_rule=amplitude_rule,
        strata=strata,
    )


def build_builder(stats: FittedStats):
    """Reconstruct the frozen canonicalizer from a fitted state."""
    TrainFrozenScaler = attr("TrainFrozenScaler")
    DeterministicFeatureBuilder = attr("DeterministicFeatureBuilder")
    return DeterministicFeatureBuilder(
        base_scaler=TrainFrozenScaler.from_state(stats.scaler_state),
        residual_scale=float(stats.residual_scale))


def _reject_forbidden_rows(dataset: CellDataset, rows: np.ndarray, what: str) -> None:
    """Backstop: a fitting call must never be handed a non-fitting role."""
    allowed = set()
    for role in C.FITTING_ROLES:
        allowed.update(int(x) for x in dataset.positions(role))
    allowed.update(int(x) for x in _inner_train_pool(dataset))
    stray = sorted(int(x) for x in rows if int(x) not in allowed)
    if stray:
        roles = sorted({dataset.panel.episodes[i].role for i in stray[:32]})
        raise LegalityError(
            f"{dataset.key}: {what} was handed {len(stray)} row(s) outside the "
            f"POST_TRAIN fitting prefix (roles present: {roles}); refusing")


def _inner_train_pool(dataset: CellDataset) -> np.ndarray:
    """Rows of ``POST_TRAIN`` -- the only pool a fitting prefix may be drawn from."""
    return dataset.positions(C.ROLE_POST_TRAIN)


# --------------------------------------------------------------------------
# One fold's fit
# --------------------------------------------------------------------------
@dataclass
class FoldFit:
    state_dict: Dict[str, Any]
    stats: FittedStats
    best_epoch: int
    epochs_run: int
    best_inner_val_mae: float
    history: List[Dict[str, float]] = field(default_factory=list)


def make_optimizer(model, lr: float, weight_decay: float):
    torch = _torch()
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def batch_order(dataset: CellDataset, train_rows: np.ndarray, stats: FittedStats,
                switches: Mapping[str, bool], seed: int, epoch: int,
                batch_size: int) -> List[np.ndarray]:
    """One epoch's batches.

    ``rare_mass_sampling=True`` uses the core's ``spread`` batching, which visits
    every training sample **exactly once** per epoch and never duplicates one.
    ``rare_mass_sampling=False`` is plain seeded shuffling.  Both are deterministic
    functions of ``(seed, epoch)``.
    """
    n_batches = max(1, math.ceil(train_rows.size / batch_size))
    if switches["rare_mass_sampling"]:
        RareMassSampler = attr("RareMassSampler")
        RareMassSamplerConfig = attr("RareMassSamplerConfig")
        decompose_residual = attr("decompose_residual")
        torch = _torch()

        residual = torch.as_tensor(dataset.residual[train_rows], dtype=torch.float32)
        valid = torch.as_tensor(dataset.valid_mask[train_rows], dtype=torch.bool)
        targets = decompose_residual(residual, valid)
        sampler = RareMassSampler(RareMassSamplerConfig(
            enabled=True, mode="spread", seed=int(seed)))
        sampler.fit(targets.mass_positive, targets.mass_negative)
        local = sampler.batches(n_batches, seed=int(seed) + int(epoch))
        return [train_rows[np.asarray(b, dtype=np.int64)] for b in local]

    rng = np.random.default_rng((int(seed) * 1_000_003) + int(epoch))
    shuffled = rng.permutation(train_rows)
    return [shuffled[i:i + batch_size] for i in range(0, shuffled.size, batch_size)]


def _step_loss(model, builder, dataset: CellDataset, rows: np.ndarray,
               stats: FittedStats, switches: Mapping[str, bool]):
    """One forward pass and the registered combined loss."""
    torch = _torch()
    decompose_residual = attr("decompose_residual")
    combined_loss = attr("combined_loss")
    repair = attr("repair")

    raw = dataset.to_raw_inputs(rows)
    batch = builder.build(raw)
    out = model(batch)

    target = torch.as_tensor(dataset.target[rows], dtype=torch.float32)
    residual = torch.as_tensor(dataset.residual[rows], dtype=torch.float32)
    valid = torch.as_tensor(dataset.valid_mask[rows], dtype=torch.bool)
    masses = decompose_residual(residual, valid)

    prediction = repair(batch.host, out.correction, 1.0)

    parts = combined_loss(
        prediction=prediction,
        target=target,
        shape_positive_predicted=out.shape.positive,
        shape_positive_target=masses.shape_positive,
        shape_negative_predicted=out.shape.negative,
        shape_negative_target=masses.shape_negative,
        amplitude_positive_predicted=out.amplitude.positive,
        amplitude_positive_target=masses.mass_positive,
        amplitude_negative_predicted=out.amplitude.negative,
        amplitude_negative_target=masses.mass_negative,
        mass_positive=masses.mass_positive,
        mass_negative=masses.mass_negative,
        valid_mask=valid,
        weight_repair=C.LOSS_WEIGHTS["repair"],
        weight_shape=C.LOSS_WEIGHTS["shape"],
        weight_amplitude=C.LOSS_WEIGHTS["amplitude"],
        shape_metric=C.LOSS_WEIGHTS["shape_metric"],
        mass_center=torch.tensor(stats.mass_center, dtype=torch.float32),
        mass_scale=torch.tensor(stats.mass_scale, dtype=torch.float32),
    )
    return parts, out


def evaluate_mae(model, builder, dataset: CellDataset, rows: np.ndarray) -> float:
    """Inner-validation criterion: MAE of the raw correction (``alpha = 1``).

    ``alpha`` is a single global scalar fitted *after* the folds, so no fold may
    early-stop on a calibrated objective; the raw-correction MAE is the honest
    fold-local criterion.  ``DEV_EVAL`` is unreachable from here because the
    caller only ever passes a prefix of the fold's own training block.
    """
    torch = _torch()
    repair = attr("repair")
    repair_mae = attr("repair_mae")

    if rows.size == 0:
        return float("nan")
    model.eval()
    total = 0.0
    weight = 0
    with torch.no_grad():
        for start in range(0, rows.size, C.TRAINING["batch_size"]):
            chunk = rows[start:start + C.TRAINING["batch_size"]]
            batch = builder.build(dataset.to_raw_inputs(chunk))
            out = model(batch)
            prediction = repair(batch.host, out.correction, 1.0)
            target = torch.as_tensor(dataset.target[chunk], dtype=torch.float32)
            valid = torch.as_tensor(dataset.valid_mask[chunk], dtype=torch.bool)
            value = repair_mae(prediction, target, valid, reduction="sum")
            total += float(value)
            weight += int(valid.sum())
    model.train()
    return total / max(1, weight)


#: The branch arrays every inference call returns.  The two Shape arrays and
#: the two mass arrays are the model's **direct** outputs, read off the branch
#: heads.  They are deliberately not recovered by re-decomposing ``correction``:
#: ``S_hat+`` and ``S_hat-`` may overlap at the same horizon and cancel inside
#: ``c``, so a decomposition of the net correction would describe a different
#: object than the one the branches actually produced.
BRANCH_ARRAYS = ("shape_positive", "shape_negative",
                 "mass_positive", "mass_negative", "correction")


def predict_corrections(model, builder, dataset: CellDataset,
                        rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Frozen inference: raw correction and the direct branch outputs."""
    torch = _torch()

    idx = np.asarray(rows, dtype=np.int64)
    if idx.size == 0:
        return {"correction": np.zeros((0, C.HORIZON), dtype=np.float64),
                "shape_positive": np.zeros((0, C.HORIZON), dtype=np.float64),
                "shape_negative": np.zeros((0, C.HORIZON), dtype=np.float64),
                "mass_positive": np.zeros(0, dtype=np.float64),
                "mass_negative": np.zeros(0, dtype=np.float64)}

    model.eval()
    collected: Dict[str, List[np.ndarray]] = {k: [] for k in BRANCH_ARRAYS}
    with torch.no_grad():
        for start in range(0, idx.size, C.TRAINING["batch_size"]):
            chunk = idx[start:start + C.TRAINING["batch_size"]]
            batch = builder.build(dataset.to_raw_inputs(chunk))
            out = model(batch)
            for name, tensor in (("correction", out.correction),
                                 ("shape_positive", out.shape.positive),
                                 ("shape_negative", out.shape.negative),
                                 ("mass_positive", out.amplitude.positive),
                                 ("mass_negative", out.amplitude.negative)):
                collected[name].append(
                    tensor.detach().cpu().numpy().astype(np.float64))
    model.train()
    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def train_fold(dataset: CellDataset, train_rows: np.ndarray, val_rows: np.ndarray,
               switches: Mapping[str, bool], seed: int,
               max_epochs: Optional[int] = None) -> FoldFit:
    """Train one fold on ``train_rows`` and early-stop on ``val_rows``.

    Both arguments are positions inside the fold's own training block.  The
    holdout block is not passed in and is not reachable from here.
    """
    torch = _torch()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    val_rows = np.asarray(val_rows, dtype=np.int64)
    if train_rows.size == 0:
        raise LegalityError(f"{dataset.key}: empty fold-training prefix")

    # The inner-validation suffix must not be used to fit anything.
    stats = fit_stats(dataset, train_rows, switches, seed)
    builder = build_builder(stats)

    model = build_model(dataset.n_features, switches, seed,
                        amplitude_scale=stats.amplitude_scale)
    optimizer = make_optimizer(model, C.TRAINING["lr"], C.TRAINING["weight_decay"])

    epochs = int(max_epochs if max_epochs is not None else C.TRAINING["max_epochs"])
    patience = int(C.TRAINING["patience"])
    batch_size = int(C.TRAINING["batch_size"])

    best_mae = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None
    since_improved = 0
    history: List[Dict[str, float]] = []
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        epoch_total = 0.0
        n_batches = 0
        for rows in batch_order(dataset, train_rows, stats, switches, seed, epoch,
                                batch_size):
            optimizer.zero_grad(set_to_none=True)
            parts, _ = _step_loss(model, builder, dataset, rows, stats, switches)
            loss = parts["total"]
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach())
            n_batches += 1

        val_mae = evaluate_mae(model, builder, dataset, val_rows)
        history.append({
            "epoch": float(epoch),
            "train_loss": epoch_total / max(1, n_batches),
            "inner_val_mae": float(val_mae),
        })

        if np.isfinite(val_mae) and val_mae < best_mae - 1e-9:
            best_mae = float(val_mae)
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs_run
        best_mae = float("nan")

    return FoldFit(state_dict=best_state, stats=stats, best_epoch=best_epoch,
                   epochs_run=epochs_run, best_inner_val_mae=best_mae,
                   history=history)


def load_state(model, state_dict: Dict[str, Any]):
    model.load_state_dict(state_dict)
    return model


def final_epoch_rule(best_epochs: Sequence[int]) -> Dict[str, Any]:
    """Derive the final-fit epoch count from fold history only.

    The registered rule is the median of the folds' early-stopped best epochs,
    clipped to ``[1, max_epochs]``.  It uses nothing but training history, so it
    cannot be influenced by any held-out outcome.
    """
    values = [int(e) for e in best_epochs if int(e) > 0]
    if not values:
        return {"rule": "median_of_oof_best_epochs", "value": 1,
                "fold_best_epochs": [], "note": "no fold produced a usable epoch"}
    median = int(np.median(values))
    value = int(min(C.TRAINING["max_epochs"], max(1, median)))
    return {
        "rule": "clip(median(oof fold best epochs), 1, max_epochs)",
        "value": value,
        "fold_best_epochs": values,
        "fold_best_epochs_median": median,
        "max_epochs": int(C.TRAINING["max_epochs"]),
    }
