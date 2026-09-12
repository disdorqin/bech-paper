"""Chronological expanding out-of-fold fitting inside ``POST_TRAIN``.

The whole file resolves *one* role: ``POST_TRAIN``.  ``DEV_EVAL`` is not
importable from here, and ``PROTECTED_FINAL`` cannot exist in a dataset at all
(the adapter refuses to materialise it).  That is what makes the prompt's
chronology a structural property rather than a discipline.

The order of operations is the registered one:

1. split ``POST_TRAIN`` into a mandatory initial prefix plus contiguous holdout
   blocks (expanding, never sliding);
2. for each fold, fit every statistic on that fold's own training prefix only,
   train, and predict the holdout block;
3. concatenate the holdout predictions in chronological order;
4. fit the single nonnegative pooled MAE scalar on those out-of-fold
   predictions alone;
5. derive the final epoch count from fold training history alone;
6. re-fit on the whole of ``POST_TRAIN`` for that many epochs.

Step 6's re-fit is the only model that an evaluation may ever use, and it is
produced after ``alpha`` is already frozen, so no evaluation outcome can reach
back into it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import calibration
from . import config as C
from .china5_adapter import CellDataset
from .contracts import LegalityError
from .core_bridge import attr
from . import training as T


# --------------------------------------------------------------------------
# Fold plan
# --------------------------------------------------------------------------
@dataclass
class Fold:
    """One expanding fold.  Every row list is POSITIONAL into the dataset."""

    index: int
    train_rows: np.ndarray          # fold training prefix
    inner_train_rows: np.ndarray    # prefix minus its inner-validation suffix
    inner_val_rows: np.ndarray      # chronological suffix of the prefix
    holdout_rows: np.ndarray        # the block this fold predicts

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fold": int(self.index),
            "n_train": int(self.train_rows.size),
            "n_inner_train": int(self.inner_train_rows.size),
            "n_inner_val": int(self.inner_val_rows.size),
            "n_holdout": int(self.holdout_rows.size),
            "train_row_span": _span(self.train_rows),
            "holdout_row_span": _span(self.holdout_rows),
        }


@dataclass
class OofPlan:
    market: str
    host: str
    post_train_rows: np.ndarray
    folds: List[Fold]
    n_post_train: int

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def oof_rows(self) -> np.ndarray:
        """Every row that receives an out-of-fold prediction, chronologically."""
        if not self.folds:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([f.holdout_rows for f in self.folds])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "host": self.host,
            "n_post_train": int(self.n_post_train),
            "n_folds": int(self.n_folds),
            "min_train_days": int(C.OOF_MIN_TRAIN_DAYS),
            "min_holdout_days": int(C.OOF_MIN_HOLDOUT_DAYS),
            "max_folds": int(C.OOF_MAX_FOLDS),
            "n_oof_rows": int(self.oof_rows.size),
            "n_unpredicted_prefix_rows": int(self.n_post_train - self.oof_rows.size),
            "folds": [f.as_dict() for f in self.folds],
        }


def _span(rows: np.ndarray) -> Optional[List[int]]:
    if rows.size == 0:
        return None
    return [int(rows.min()), int(rows.max())]


def plan_folds(dataset: CellDataset) -> OofPlan:
    """Build the expanding fold plan from ``POST_TRAIN`` alone.

    The fold count comes from the one global rule in :mod:`config`; it is a
    function of the amount of legal training data, never of a market name.
    """
    post = dataset.positions(C.ROLE_POST_TRAIN)
    if post.size == 0:
        raise LegalityError(f"{dataset.key}: no POST_TRAIN rows; OOF is impossible")

    # The positions are already chronological (the adapter asserts it), but the
    # plan re-sorts by chronological position so the contract is local and
    # explicit rather than inherited.
    post = np.sort(post.astype(np.int64))
    n_post = int(post.size)
    n_folds = C.oof_fold_count(n_post)
    if n_folds == 0:
        raise LegalityError(
            f"{dataset.key}: POST_TRAIN has {n_post} days, which is too few for "
            f"the registered expanding OOF rule "
            f"(need >= {C.OOF_MIN_TRAIN_DAYS + C.OOF_MIN_HOLDOUT_DAYS})")

    remainder = n_post - C.OOF_MIN_TRAIN_DAYS
    holdout = remainder // n_folds

    folds: List[Fold] = []
    for k in range(n_folds):
        start = C.OOF_MIN_TRAIN_DAYS + k * holdout
        end = n_post if k == n_folds - 1 else C.OOF_MIN_TRAIN_DAYS + (k + 1) * holdout
        train_rows = post[:start]
        holdout_rows = post[start:end]
        if train_rows.size == 0 or holdout_rows.size == 0:
            raise LegalityError(f"{dataset.key}: degenerate fold {k}")
        n_val = max(C.OOF_MIN_INNER_VAL_DAYS,
                    int(round(C.OOF_INNER_VAL_FRACTION * train_rows.size)))
        n_val = min(n_val, train_rows.size - 1)
        inner_val = train_rows[train_rows.size - n_val:]
        inner_train = train_rows[:train_rows.size - n_val]
        if inner_train.size == 0:
            raise LegalityError(
                f"{dataset.key}: fold {k} left no inner-training rows after "
                f"holding out {n_val} for early stopping")
        folds.append(Fold(index=k, train_rows=train_rows,
                          inner_train_rows=inner_train, inner_val_rows=inner_val,
                          holdout_rows=holdout_rows))

    plan = OofPlan(market=dataset.market, host=dataset.host,
                   post_train_rows=post, folds=folds, n_post_train=n_post)
    _assert_plan_is_expanding_and_disjoint(plan)
    return plan


def _assert_plan_is_expanding_and_disjoint(plan: OofPlan) -> None:
    """Fold k trains only on rows strictly earlier than its own holdout block."""
    seen: List[int] = []
    for fold in plan.folds:
        if fold.holdout_rows.size and fold.train_rows.size:
            if int(fold.train_rows.max()) >= int(fold.holdout_rows.min()):
                raise LegalityError(
                    f"{plan.market}/{plan.host}: fold {fold.index} trains on a row "
                    "that is not strictly earlier than its holdout block")
        overlap = set(int(x) for x in fold.holdout_rows) & set(seen)
        if overlap:
            raise LegalityError(
                f"{plan.market}/{plan.host}: fold {fold.index} holdout overlaps an "
                f"earlier fold ({len(overlap)} rows)")
        seen.extend(int(x) for x in fold.holdout_rows)
        # The inner-validation suffix must be a suffix of the fold's own prefix.
        if fold.inner_val_rows.size and fold.inner_train_rows.size:
            if int(fold.inner_train_rows.max()) >= int(fold.inner_val_rows.min()):
                raise LegalityError(
                    f"{plan.market}/{plan.host}: fold {fold.index} inner train/val "
                    "split is not chronological")


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
@dataclass
class FittedMethod:
    """A fully fitted method: frozen ``alpha`` plus the whole-prefix model."""

    market: str
    host: str
    switches: Mapping[str, bool]
    seed: int
    alpha: float
    alpha_objective: Dict[str, Any]
    epoch_rule: Dict[str, Any]
    stats: T.FittedStats
    state_dict: Dict[str, Any]
    oof_plan: Dict[str, Any]
    oof_correction_sha256: str
    folds: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.market}/{self.host}"

    def model_and_builder(self):
        builder = T.build_builder(self.stats)
        model = T.build_model(_n_features_from(self.stats), self.switches, self.seed,
                              amplitude_scale=self.stats.amplitude_scale)
        T.load_state(model, self.state_dict)
        model.eval()
        return model, builder

    def predict(self, dataset: CellDataset,
                rows: Sequence[int]) -> Dict[str, Any]:
        """Calibrated prediction plus the **direct** branch outputs.

        ``rows`` must belong to a revealed, non-fitting role and may never
        include a sealed one; the guard is repeated here so no caller can route
        around it.

        The returned ``shape_positive`` / ``shape_negative`` /
        ``mass_positive`` / ``mass_negative`` are the branch heads' own outputs,
        not a re-decomposition of ``correction``.  Mechanism metrics are defined
        against these arrays, because ``S_hat+`` and ``S_hat-`` can overlap at a
        horizon and cancel inside the net correction.
        """
        idx = np.asarray(rows, dtype=np.int64)
        _assert_revealed(dataset, idx)
        builder = T.build_builder(self.stats)
        model = T.build_model(_n_features_from(self.stats), self.switches, self.seed,
                              amplitude_scale=self.stats.amplitude_scale)
        T.load_state(model, self.state_dict)
        raw = T.predict_corrections(model, builder, dataset, idx)
        correction = raw["correction"]
        host = dataset.host_forecast[idx].astype(np.float64)
        prediction = calibration.apply_pooled_alpha(host, correction, self.alpha)
        return {
            "shape_positive": raw["shape_positive"],
            "shape_negative": raw["shape_negative"],
            "mass_positive": raw["mass_positive"],
            "mass_negative": raw["mass_negative"],
            "correction": correction,
            "prediction": prediction,
            "host": host,
            "target": dataset.target[idx].astype(np.float64),
            "valid_mask": dataset.valid_mask[idx].astype(bool),
            "identity": episode_identity(dataset, idx),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "market": self.market,
            "host": self.host,
            "seed": int(self.seed),
            "switches": dict(self.switches),
            "alpha": float(self.alpha),
            "alpha_objective": self.alpha_objective,
            "epoch_rule": self.epoch_rule,
            "fitted_stats": self.stats.as_dict(),
            "oof_plan": self.oof_plan,
            "oof_correction_sha256": self.oof_correction_sha256,
            "folds": self.folds,
            "provenance": self.provenance,
        }


def episode_identity(dataset: CellDataset, rows: np.ndarray) -> Dict[str, Any]:
    """Stable day/episode identity for a block of positions.

    The artifact index is the address; the label and target day are carried
    alongside it so a reader can join without re-deriving either from a
    timestamp spelling the frozen artifact never promised to keep.
    """
    idx = np.asarray(rows, dtype=np.int64)
    episodes = dataset.panel.episodes
    picked = [episodes[int(i)] for i in idx]
    return {
        "episode_index": [int(e.index) for e in picked],
        "role": [str(e.role) for e in picked],
        "label": [str(e.label) for e in picked],
        "target_day": [str(e.target_day) for e in picked],
        "run_length": [int(e.run_length) for e in picked],
        "label_epoch_seconds": [int(np.datetime64(e.label, "s").astype(np.int64))
                                for e in picked],
    }


def _n_features_from(stats: T.FittedStats) -> int:
    """Recover ``F`` from the fitted scaler state (1 Host column + F features)."""
    return len(stats.scaler_state["center"]) - 1


def fit_high_mass_thresholds(dataset: CellDataset) -> Dict[str, Any]:
    """The cell's frozen ``q90(A+)`` / ``q90(A-)``, fitted on ``POST_TRAIN`` alone.

    This is the A3 artifact.  It is produced once per cell **before** any
    ``DEV_EVAL`` metric exists, from the same partition OOF fitting uses, so the
    subset a high-mass metric reports on is fixed independently of the outcome
    being measured.  ``DEV_EVAL`` is not reachable from this function: the rows
    come from ``dataset.positions(POST_TRAIN)``, and the residuals are the Host's
    own frozen residuals, so no Host is retrained either.
    """
    from . import metrics as M

    rows = dataset.positions(C.ROLE_POST_TRAIN)
    if rows.size == 0:
        raise LegalityError(f"{dataset.key}: no POST_TRAIN rows; q90 is undefined")
    rows = np.sort(rows.astype(np.int64))
    record = M.fit_high_mass_thresholds(
        dataset.residual[rows].astype(np.float64),
        dataset.valid_mask[rows].astype(bool),
        fit_rows=rows)
    record["cell"] = dataset.key
    record["market"] = dataset.market
    record["host"] = dataset.host
    record["n_post_train_rows"] = int(rows.size)
    return record


def _assert_revealed(dataset: CellDataset, rows: np.ndarray) -> None:
    """Every requested row must belong to a revealed, non-sealed role.

    The check is phrased as membership in the revealed union rather than as an
    intersection with a sealed role, because a sealed role is not merely empty
    here -- ``CellDataset.positions`` refuses to even name it.  Asking the
    dataset for the sealed set would turn the guard into the very access it is
    meant to prevent.
    """
    allowed = set()
    for role in C.REVEALED_HISTORY_ROLES:
        try:
            allowed.update(int(x) for x in dataset.positions(role))
        except LegalityError:
            continue
    stray = sorted(int(x) for x in rows if int(x) not in allowed)
    if stray:
        raise LegalityError(
            f"{dataset.key}: prediction requested for {len(stray)} row(s) outside "
            f"the revealed roles {sorted(C.REVEALED_HISTORY_ROLES)}; refusing")


def fit_method(dataset: CellDataset, switches: Mapping[str, bool], seed: int,
               max_epochs: Optional[int] = None,
               on_fold=None) -> FittedMethod:
    """Run the full registered fitting chain for one (cell, config, seed)."""
    plan = plan_folds(dataset)

    fold_summaries: List[Dict[str, Any]] = []
    corrections: List[np.ndarray] = []
    row_blocks: List[np.ndarray] = []
    best_epochs: List[int] = []

    for fold in plan.folds:
        fit = T.train_fold(dataset, fold.inner_train_rows, fold.inner_val_rows,
                           switches, seed, max_epochs=max_epochs)
        builder = T.build_builder(fit.stats)
        model = T.build_model(dataset.n_features, switches, seed,
                              amplitude_scale=fit.stats.amplitude_scale)
        T.load_state(model, fit.state_dict)
        raw = T.predict_corrections(model, builder, dataset, fold.holdout_rows)

        corrections.append(raw["correction"])
        row_blocks.append(fold.holdout_rows)
        best_epochs.append(int(fit.best_epoch))
        summary = dict(fold.as_dict())
        summary.update({
            "best_epoch": int(fit.best_epoch),
            "epochs_run": int(fit.epochs_run),
            "best_inner_val_mae": float(fit.best_inner_val_mae),
            "fit_rows_sha256": fit.stats.fit_rows_sha256,
            "scaler_fitted_on_rows": int(fit.stats.n_fit_rows),
            "residual_scale": float(fit.stats.residual_scale),
            "amplitude_scale": float(fit.stats.amplitude_scale),
            "amplitude_scale_rule": fit.stats.amplitude_scale_rule,
            "history": fit.history,
        })
        fold_summaries.append(summary)
        if on_fold is not None:
            on_fold(summary)

    oof_rows = np.concatenate(row_blocks)
    oof_correction = np.concatenate(corrections, axis=0)
    order = np.argsort(oof_rows, kind="stable")
    oof_rows = oof_rows[order]
    oof_correction = oof_correction[order]

    alpha, alpha_objective = fit_alpha(dataset, oof_rows, oof_correction)

    epoch_rule = T.final_epoch_rule(best_epochs)

    # Final re-fit: the whole of POST_TRAIN, for the fold-derived epoch count,
    # after alpha is already frozen.
    final_stats = T.fit_stats(dataset, plan.post_train_rows, switches, seed)
    builder = T.build_builder(final_stats)
    model = T.build_model(dataset.n_features, switches, seed,
                          amplitude_scale=final_stats.amplitude_scale)
    optimizer = T.make_optimizer(model, C.TRAINING["lr"], C.TRAINING["weight_decay"])
    for epoch in range(1, int(epoch_rule["value"]) + 1):
        for rows in T.batch_order(dataset, plan.post_train_rows, final_stats, switches,
                                  seed, epoch, int(C.TRAINING["batch_size"])):
            optimizer.zero_grad(set_to_none=True)
            parts, _ = T._step_loss(model, builder, dataset, rows, final_stats, switches)
            parts["total"].backward()
            optimizer.step()
    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    return FittedMethod(
        market=dataset.market,
        host=dataset.host,
        switches=dict(switches),
        seed=int(seed),
        alpha=float(alpha),
        alpha_objective=alpha_objective,
        epoch_rule=epoch_rule,
        stats=final_stats,
        state_dict=final_state,
        oof_plan=plan.as_dict(),
        oof_correction_sha256=_array_digest(oof_correction),
        folds=fold_summaries,
        provenance={
            "final_fit_rows": int(plan.post_train_rows.size),
            "final_fit_rows_sha256": final_stats.fit_rows_sha256,
            "final_fit_epochs": int(epoch_rule["value"]),
            "oof_rows_sha256": T._rows_digest(oof_rows),
        },
    )


def fit_alpha(dataset: CellDataset, oof_rows: np.ndarray,
              oof_correction: np.ndarray):
    """Fit the single nonnegative pooled MAE scalar on out-of-fold predictions.

    Delegates entirely to :mod:`calibration`, which in turn delegates to
    ``src/core``: the objective, the exact weighted-median solution and the
    nonnegativity constraint are the core's, not this harness's.
    """
    if oof_rows.size == 0:
        raise LegalityError(f"{dataset.key}: no out-of-fold rows; alpha is undefined")

    residual = dataset.residual[oof_rows].astype(np.float64)
    valid = dataset.valid_mask[oof_rows].astype(bool)
    candidate = oof_correction.astype(np.float64)

    alpha, record = calibration.fit_pooled_alpha(
        candidate, residual, valid, key=f"{dataset.key} OOF alpha")
    record["n_oof_rows"] = int(oof_rows.size)
    return alpha, record


def _array_digest(x: np.ndarray) -> str:
    payload = np.ascontiguousarray(x, dtype=np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest().upper()
