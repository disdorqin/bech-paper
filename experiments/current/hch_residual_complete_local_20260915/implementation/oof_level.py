"""Stage-1 causal OOF Level cross-fitting, and the final frozen refit.

Two objects come out of this module, and the protocol turns on their difference:

* **causal OOF Levels** — one prediction per LOCAL_TRAIN day, each produced by a
  model fitted only on days strictly earlier than that day's block.  These build
  the Local training targets, so an optimistic value here would train Local
  against a residual it could never see.
* **the final frozen Level predictor** — refitted on the legal TRAIN support once
  the OOF objects are frozen, and never updated again.  VAL inference reads only
  this one.

The fold plan is not a free choice.  It is :func:`rcl_contract.oof_partition`'s
40% warm-up + B1..B4, and B1 is warm-start history support only: it is never a
Local training target, because the earliest block that has a *strictly earlier*
OOF block to be trained against is B2.

Both models must be the same architecture.  :func:`level_architecture_fingerprint`
is the structural identity the run records for each, and the runner asserts the
two fingerprints are equal before the final predictor is frozen — a refit that
quietly changed the Level model would make the VAL conditioner describe a
different function than the TRAIN targets were built against.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from core import CoarseLevelModel, ModelConfig, build_level_target

from level_condition_stats import OofLevelRecord
from rcl_contract import (LOCAL_TRAIN_BLOCKS, OOF_BLOCKS, WARMUP_FRACTION,
                          oof_partition)

__all__ = [
    "LEVEL_FIT_TAIL_FRACTION",
    "FINAL_REFIT_TAIL_FRACTION",
    "OofFold",
    "plan_oof_folds",
    "assert_prefix_precedes_target",
    "level_architecture_fingerprint",
    "robust_scale",
    "history_scale",
    "level_target_scale",
    "fit_level_model",
    "fit_oof_fold",
    "seeded_median",
    "lower_median_across_seeds",
    "OofLevelPanel",
    "FinalLevelPredictor",
    "FrozenLevelEnsemble",
    "final_level_refit",
    "assert_same_level_architecture",
]

#: Inner chronological early-stop tail, as a fraction of a fold's training prefix.
LEVEL_FIT_TAIL_FRACTION = 0.15

#: The same tail for the final refit (protocol section 5.4).
FINAL_REFIT_TAIL_FRACTION = 0.15

#: Registered Stage-1 optimizer settings, inherited from protocol section 11.
LEVEL_RECIPE = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "max_epochs": 50,
    "patience": 8,
    "grad_clip": 1.0,
}


@dataclass(frozen=True)
class OofFold:
    """One causal OOF fold: a target block and the prefix that may train it.

    ``prefix_ids`` are the day indices the fold's model is fitted on, split into
    ``fit_ids`` and the inner ``early_ids`` tail.  ``prefix_ids`` is *strictly*
    earlier than ``target_ids`` — that is the whole content of "causal", and
    :func:`assert_prefix_precedes_target` is what makes it checkable rather than
    asserted in prose.
    """

    fold: int
    block: str
    target_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    fit_ids: tuple[int, ...]
    early_ids: tuple[int, ...]
    seeds: tuple[int, ...] = (7, 17, 37)

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "block": self.block,
            "target_ids_sha256": _ids_sha256(self.target_ids),
            "prefix_ids_sha256": _ids_sha256(self.prefix_ids),
            "n_target_days": len(self.target_ids),
            "n_prefix_days": len(self.prefix_ids),
            "n_fit_days": len(self.fit_ids),
            "n_early_days": len(self.early_ids),
            "target_first_day": self.target_ids[0] if self.target_ids else None,
            "prefix_last_day": self.prefix_ids[-1] if self.prefix_ids else None,
            "seeds": list(self.seeds),
        }


def _ids_sha256(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(d)) for d in ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_oof_folds(n_days: int, seeds: Sequence[int] = (7, 17, 37),
                   tail_fraction: float = LEVEL_FIT_TAIL_FRACTION
                   ) -> tuple[OofFold, ...]:
    """The registered causal fold plan for a TRAIN length of ``n_days``.

    Each fold's prefix is *everything before its own block*, not a fixed-size
    sliding window: an expanding prefix is what makes the later blocks' OOF
    predictions comparable to the final refit rather than to a smaller model.

    **B1 gets a fold like the others, and its prefix is the warm-up region.**
    Protocol section 5.3 makes B1 the Local history's warm start, and Local history
    is post-Level ``q = r - b_tilde*1``: without a *causal* ``b_tilde`` of its own,
    B1 has no q and cannot be history support at all.  A plan that skipped B1 would
    leave the one block the protocol assigns a role to as the only block with no
    prediction.  Warm-up days are legal Stage-1 support — they are TRAIN data and
    they are not sealed — and they are exactly "earlier target days" for B1, so the
    prefix rule is the same one every other block uses.
    """
    partition = oof_partition(n_days)
    folds: list[OofFold] = []
    for index, name in enumerate(OOF_BLOCKS, start=1):
        target = tuple(int(d) for d in partition[name])
        if not target:
            raise ValueError(f"block {name} is empty for {n_days} TRAIN days")
        prefix = tuple(d for d in range(target[0]))
        if not prefix:
            raise ValueError(
                f"fold {name} has no legal training prefix: block starts at day "
                f"{target[0]} and nothing earlier exists")
        tail = max(2, int(-(-len(prefix) * float(tail_fraction) // 1)))
        if tail >= len(prefix):
            raise ValueError(
                f"fold {name}: a {tail}-day early-stop tail leaves no fitting days")
        fold = OofFold(fold=index, block=name, target_ids=target,
                       prefix_ids=prefix, fit_ids=prefix[:-tail],
                       early_ids=prefix[-tail:], seeds=tuple(int(s) for s in seeds))
        assert_prefix_precedes_target(fold)
        folds.append(fold)
    return tuple(folds)


def assert_prefix_precedes_target(fold: OofFold) -> None:
    """Fail unless the fold's training prefix is strictly earlier than its target.

    Checked on the actual day indices rather than on the block names, so a plan
    built by some other route cannot pass by being labelled correctly.
    """
    if not fold.prefix_ids or not fold.target_ids:
        raise ValueError(f"fold {fold.fold} ({fold.block}) has an empty side")
    if max(fold.prefix_ids) >= min(fold.target_ids):
        raise AssertionError(
            f"fold {fold.fold} ({fold.block}): training prefix reaches day "
            f"{max(fold.prefix_ids)} but the target block starts at day "
            f"{min(fold.target_ids)}; an OOF fold must train strictly earlier")
    overlap = set(fold.prefix_ids) & set(fold.target_ids)
    if overlap:
        raise AssertionError(
            f"fold {fold.fold} ({fold.block}): prefix and target overlap on "
            f"{sorted(overlap)[:8]}")
    if list(fold.prefix_ids) != sorted(fold.prefix_ids):
        raise AssertionError("prefix day indices must be chronologically ordered")
    if set(fold.fit_ids) | set(fold.early_ids) != set(fold.prefix_ids):
        raise AssertionError("fit and early-stop days must partition the prefix")
    if max(fold.fit_ids) >= min(fold.early_ids):
        raise AssertionError("the early-stop tail must be the prefix's last days")


def level_architecture_fingerprint(config: ModelConfig,
                                   model: Optional[torch.nn.Module] = None) -> str:
    """Structural identity of the Stage-1 architecture.

    Hashes the module tree's class names and parameter shapes, plus the config.
    Two Level models with the same fingerprint are the same function class with
    the same capacity; a refit that changed either one produces a different
    fingerprint, which is exactly what the run must refuse.
    """
    if model is None:
        model = CoarseLevelModel(config)
    tree = [(name, type(module).__name__)
            for name, module in model.named_modules()]
    shapes = [(name, tuple(param.shape))
              for name, param in model.named_parameters()]
    payload = json.dumps(
        {"config": sorted(vars(config).items()), "tree": tree, "params": shapes},
        sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_same_level_architecture(config: ModelConfig) -> str:
    """The OOF and final Level models share one architecture by construction.

    Both are built from this single call site, so the fingerprint comparison the
    runner performs is a check on the *build path* rather than on two independent
    literals that could drift apart.
    """
    return level_architecture_fingerprint(config)


def seed_everything(seed: int) -> None:
    """Seed before model construction, as protocol section 11 requires."""
    import numpy as np

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seeded_median(values: Sequence[float]) -> float:
    """Median across seeds, lower-middle on an even count (repository convention)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("no seed values to reduce")
    return ordered[(len(ordered) - 1) // 2]


def lower_median_across_seeds(per_seed):
    """Elementwise lower-middle median of a ``(n_seeds, ...)`` stack.

    The same convention as :func:`seeded_median`, applied per coordinate rather
    than to a scalar.  ``np.median`` would average the two middle seeds on an even
    count, which is a *different* estimator than the one the rest of the repository
    uses; two median conventions in one module is how a reduced value stops being
    reproducible from the code.
    """
    import numpy as np

    stacked = np.asarray(per_seed, dtype=np.float64)
    if stacked.ndim < 1 or stacked.shape[0] == 0:
        raise ValueError("no seed values to reduce")
    ordered = np.sort(stacked, axis=0)
    return ordered[(ordered.shape[0] - 1) // 2]


def robust_scale(values) -> float:
    """``1.4826 * median(|x|)`` — the repository's robust scale convention."""
    import numpy as np

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("cannot fit a scale from an empty selection")
    return float(max(1e-6, 1.4826 * np.median(np.abs(flat))))


def history_scale(residual, fit_ids: Sequence[int]) -> float:
    """The Level model's history normaliser, fitted on ``fit_ids`` **only**.

    Fitting this on the whole authorized surface is the leak the audit names: a
    fold's history scale would carry the magnitude of blocks it is not allowed to
    have seen, and the final refit's scale would carry VAL.  Both are TRAIN-only
    quantities, but only ``fit_ids`` is the subset a *causal* fold may use.
    """
    import numpy as np

    rows = np.asarray(list(fit_ids), dtype=int)
    return robust_scale(np.asarray(residual)[rows])


def level_target_scale(residual, fit_ids: Sequence[int]) -> float:
    """``s_b`` — the scale of the Stage-1 **target** ``b* = mean(r)`` on ``fit_ids``.

    Deliberately not the raw hourly residual scale.  The two differ by roughly the
    horizon length, so normalising ``|b_hat - b*|`` by the hourly scale would make
    the registered Stage-1 loss an order of magnitude smaller than every other term
    and quietly de-weight it.  ``b*`` is the quantity being predicted, so ``b*`` is
    the quantity whose spread sets the unit.
    """
    import numpy as np

    rows = np.asarray(list(fit_ids), dtype=int)
    return robust_scale(np.asarray(residual)[rows].mean(axis=1))


def fit_level_model(host, residual, residual_history, config: ModelConfig,
                    fit_ids: Sequence[int], early_ids: Sequence[int],
                    seed: int, device="cpu",
                    recipe: Optional[dict] = None) -> dict:
    """Fit one Stage-1 Level model with an inner chronological early-stop tail.

    Returns the best state dict, the chosen epoch and the inner-tail MAE.  The
    architecture is always :class:`core.CoarseLevelModel` built from ``config`` —
    the single build site the architecture fingerprint is taken from.

    Both scales are fitted from ``fit_ids`` alone.  Neither the target block, nor
    the inner early-stop tail, nor any later day contributes to either one.
    """
    import numpy as np
    from core import LevelFeatureBuilder, RawForecastInputs, coarse_level_loss

    settings = dict(LEVEL_RECIPE)
    if recipe:
        settings.update(recipe)
    fit_ids = list(fit_ids)
    early_ids = list(early_ids)
    if set(fit_ids) & set(early_ids):
        raise ValueError("the early-stop tail must be disjoint from the fit days")
    seed_everything(seed)

    residual_scale = history_scale(residual, fit_ids)
    s_b = level_target_scale(residual, fit_ids)
    builder = LevelFeatureBuilder(config, residual_scale=residual_scale)
    model = CoarseLevelModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])

    fit_ids = list(fit_ids)
    early_ids = list(early_ids)
    best = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = np.random.default_rng(seed * 1000003 + epoch).permutation(fit_ids)
        for start in range(0, len(order), settings["batch_size"]):
            chunk = list(order[start:start + settings["batch_size"]])
            host_b = torch.as_tensor(host[chunk], dtype=torch.float32, device=device)
            batch = builder(RawForecastInputs(
                host=host_b, valid_mask=torch.ones_like(host_b),
                past_residual=torch.as_tensor(residual_history[chunk],
                                              dtype=torch.float32, device=device)))
            target = build_level_target(
                torch.as_tensor(residual[chunk], dtype=torch.float32, device=device))
            # ``s_b``, never ``residual_scale``: the Level target's own unit.
            loss = coarse_level_loss(model(batch).level, target, s_b).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            optimizer.step()
        score = _inner_tail_mae(model, builder, host, residual, residual_history,
                                early_ids, device)
        if score < best - 1e-7:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return {"state_dict": best_state, "best_epoch": best_epoch,
            "inner_tail_mae": best, "residual_scale": residual_scale, "s_b": s_b,
            "fit_ids": tuple(int(d) for d in fit_ids),
            "early_ids": tuple(int(d) for d in early_ids),
            "epochs_run": epoch, "recipe": settings}


def _inner_tail_mae(model, builder, host, residual, residual_history, early_ids,
                    device) -> float:
    import numpy as np
    from core import RawForecastInputs

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(early_ids), 32):
            chunk = list(early_ids[start:start + 32])
            host_b = torch.as_tensor(host[chunk], dtype=torch.float32, device=device)
            batch = builder(RawForecastInputs(
                host=host_b, valid_mask=torch.ones_like(host_b),
                past_residual=torch.as_tensor(residual_history[chunk],
                                              dtype=torch.float32, device=device)))
            preds.append(model(batch).level.detach().cpu().numpy())
    predicted = np.concatenate(preds) if preds else np.zeros(0)
    truth = np.asarray(residual)[early_ids].mean(axis=1)
    return float(np.mean(np.abs(truth - predicted)))


def predict_level(model: torch.nn.Module, host, residual_history, indices,
                  builder, device="cpu"):
    """Deterministic Level predictions over ``indices``."""
    import numpy as np
    from core import RawForecastInputs

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            chunk = list(indices[start:start + 32])
            host_b = torch.as_tensor(host[chunk], dtype=torch.float32, device=device)
            batch = builder(RawForecastInputs(
                host=host_b, valid_mask=torch.ones_like(host_b),
                past_residual=torch.as_tensor(residual_history[chunk],
                                              dtype=torch.float32, device=device)))
            out.append(model(batch).level.detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def fit_oof_fold(fold: OofFold, host, residual, residual_history,
                 config: ModelConfig, device="cpu") -> dict:
    """Fit one fold at each seed and reduce to the fold's causal OOF Level.

    The reduction across seeds is the lower-middle median, matching the
    repository's robust convention: a single unlucky seed should not become a Local
    training target.
    """
    import numpy as np
    from core import LevelFeatureBuilder

    assert_prefix_precedes_target(fold)
    per_seed = []
    states = {}
    for seed in fold.seeds:
        fitted = fit_level_model(host, residual, residual_history, config,
                                 fold.fit_ids, fold.early_ids, seed, device)
        # Predict with the *same* residual scale the fold was fitted under.  A
        # freshly recomputed scale here would feed the fitted weights a
        # differently scaled history than they saw in training.
        builder = LevelFeatureBuilder(config,
                                      residual_scale=fitted["residual_scale"])
        model = CoarseLevelModel(config).to(device)
        model.load_state_dict(fitted["state_dict"])
        per_seed.append(predict_level(model, host, residual_history,
                                      fold.target_ids, builder, device))
        states[seed] = fitted
    stacked = np.stack(per_seed)
    return {
        "fold": fold.fold, "block": fold.block,
        "target_ids": list(fold.target_ids),
        "level": lower_median_across_seeds(stacked),
        "per_seed_level": stacked,
        "per_seed_best_epoch": {s: states[s]["best_epoch"] for s in states},
        "per_seed_inner_tail_mae": {s: states[s]["inner_tail_mae"] for s in states},
        # The scale supports travel with the fold so the verifier can check that
        # each scale was fitted from exactly these days and no others.
        "fit_ids": list(fold.fit_ids),
        "early_ids": list(fold.early_ids),
        "per_seed_residual_scale": {s: states[s]["residual_scale"] for s in states},
        "per_seed_s_b": {s: states[s]["s_b"] for s in states},
        "provenance": "oof",
        "fold_provenance": fold.as_dict(),
    }


@dataclass
class OofLevelPanel:
    """Causal OOF coarse Levels over the OOF blocks, with their provenance.

    Only LOCAL_TRAIN days may seed the Level conditioner or a Local target.  The
    panel refuses ``in_sample`` and ``final`` values outright rather than letting a
    caller pass whichever array it happens to have: an in-sample Level is
    optimistic on exactly the days it was fitted on, and using one would train
    Local against a residual that does not exist at inference.
    """

    values: dict[int, float] = field(default_factory=dict)
    provenance: dict[int, str] = field(default_factory=dict)
    fold_records: list[dict] = field(default_factory=list)

    def add_fold(self, fold_result: dict, provenance: str = "oof") -> None:
        if provenance != "oof":
            raise ValueError(
                f"an OOF panel only accepts causal OOF Levels, got {provenance!r}")
        for day, value in zip(fold_result["target_ids"], fold_result["level"]):
            day = int(day)
            if day in self.values:
                raise ValueError(f"day {day} already has an OOF Level")
            self.values[day] = float(value)
            self.provenance[day] = provenance
        self.fold_records.append({
            "fold": fold_result["fold"], "block": fold_result["block"],
            "provenance": provenance, **fold_result["fold_provenance"]})

    def level_for(self, ids: Sequence[int]):
        """Levels for ``ids`` in order.  Every day must be present and OOF."""
        import numpy as np

        out = []
        for day in ids:
            day = int(day)
            if day not in self.values:
                raise KeyError(f"day {day} has no causal OOF Level")
            if self.provenance[day] != "oof":
                raise ValueError(
                    f"day {day} carries provenance {self.provenance[day]!r}; only "
                    "causal OOF Levels are legal Local inputs")
            out.append(self.values[day])
        return np.asarray(out, dtype=np.float32)

    def records_for(self, days: Sequence[int]) -> list[OofLevelRecord]:
        """The conditioner's fitting observations for ``days``."""
        return [OofLevelRecord(day_index=int(d), value=self.values[int(d)],
                               provenance="oof") for d in days]

    def local_train_records(self, n_days: int) -> list[OofLevelRecord]:
        """Records for exactly ``LOCAL_TRAIN = B2 u B3 u B4`` and nothing else."""
        from rcl_contract import local_train_days

        days = [d for d in local_train_days(n_days) if d in self.values]
        return self.records_for(days)

    def sha256(self) -> str:
        payload = json.dumps(
            {str(k): round(float(v), 10) for k, v in sorted(self.values.items())},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FinalLevelPredictor:
    """The frozen Stage-1 refit used for every VAL day — **all three seeds**.

    The TRAIN Local targets are built on a coordinate-wise median over seeds
    7/17/37 of the causal OOF Level.  A final refit that trained three seeds and
    then kept only seed 7 would define the inference Level by a *different*
    aggregation rule than the one the targets were built against, so the Local
    model would be trained against one predictor and scored against another.  All
    three states are therefore retained and the inference Level is the same
    coordinate-wise median — :func:`lower_median_across_seeds`, the repository's
    lower-middle convention, not ``np.median``'s averaging one.

    ``architecture_fingerprint`` must equal the OOF fingerprint.  ``frozen`` is a
    recorded fact rather than a convention, and the object holds no training state
    that VAL could update: the ensemble reads the states, never writes them.
    """

    state_dicts: tuple[dict, ...]
    seeds: tuple[int, ...]
    config: ModelConfig
    architecture_fingerprint: str
    oof_architecture_fingerprint: str
    fit_ids: tuple[int, ...]
    early_ids: tuple[int, ...]
    best_epochs: tuple[int, ...]
    residual_scale: float
    s_b: float
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.frozen:
            raise ValueError("the final Level predictor is frozen by construction")
        if self.architecture_fingerprint != self.oof_architecture_fingerprint:
            raise AssertionError(
                "the final Level refit changed architecture: OOF fingerprint "
                f"{self.oof_architecture_fingerprint} vs final "
                f"{self.architecture_fingerprint}")
        if len(self.state_dicts) != len(self.seeds):
            raise ValueError(
                f"{len(self.state_dicts)} final Level states for {len(self.seeds)} "
                "registered seeds; the ensemble must be one state per seed")
        if len(self.seeds) < 1:
            raise ValueError("the final Level ensemble needs at least one seed")

    def as_dict(self) -> dict:
        return {
            "architecture_fingerprint": self.architecture_fingerprint,
            "oof_architecture_fingerprint": self.oof_architecture_fingerprint,
            "seeds": [int(s) for s in self.seeds],
            "n_ensemble_members": len(self.state_dicts),
            "aggregation": "coordinate_wise_lower_median_across_seeds",
            "state_dict_sha256": [_state_dict_sha256(s) for s in self.state_dicts],
            "fit_ids_sha256": _ids_sha256(self.fit_ids),
            "early_ids_sha256": _ids_sha256(self.early_ids),
            "best_epochs": [int(e) for e in self.best_epochs],
            "residual_scale": self.residual_scale,
            "s_b": self.s_b,
            "frozen": self.frozen,
        }


def _state_dict_sha256(state_dict: dict) -> str:
    """Order-independent digest of a state dict's tensors."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(np.ascontiguousarray(
            tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


class FrozenLevelEnsemble:
    """The frozen final refit as a prediction object: median over its seeds.

    Materialized once per cell and then read-only.  :meth:`predict` is the only
    way to obtain an inference Level, so no call site can accidentally use one
    member's output where the registered ensemble was meant.
    """

    def __init__(self, predictor: FinalLevelPredictor, device="cpu") -> None:
        self.predictor = predictor
        self.device = device
        self.residual_scale = float(predictor.residual_scale)
        self.models = []
        for state in predictor.state_dicts:
            model = CoarseLevelModel(predictor.config).to(device)
            model.load_state_dict(state)
            for param in model.parameters():
                param.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):
                raise AssertionError("the final Level ensemble must be frozen")
            model.eval()
            self.models.append(model)

    def __len__(self) -> int:
        return len(self.models)

    def predict(self, host, residual_history, indices, builder, device=None):
        """Coordinate-wise median Level over the ensemble members."""
        import numpy as np

        device = device or self.device
        per_seed = [predict_level(model, host, residual_history, indices, builder,
                                  device)
                    for model in self.models]
        return lower_median_across_seeds(np.stack(per_seed))


def final_level_refit(host, residual, residual_history, config: ModelConfig,
                      seeds: Sequence[int] = (7, 17, 37), device="cpu",
                      oof_fingerprint: Optional[str] = None
                      ) -> FinalLevelPredictor:
    """Refit the identical Level architecture on the legal TRAIN support.

    The fitting support is **every** TRAIN day.  Once the OOF objects are frozen the
    final predictor is no longer estimating an out-of-fold quantity and may use all
    of it; warm-up days are TRAIN data and are not sealed, and every OOF fold's
    prefix already contains them.  Excluding them here would make the final refit
    train on strictly less than the B4 fold it is meant to supersede — a refit that
    is weaker than the cross-fit it replaces is not the "legal TRAIN support"
    section 5.4 asks for.  What the refit may *not* do is change shape — hence the
    fingerprint comparison before freezing.

    All ``len(seeds)`` states are retained.  Discarding all but the first is the
    aggregation mismatch the audit names, and it is not a saving: the fits happen
    either way.
    """
    train_ids = tuple(range(len(host)))
    if not train_ids:
        raise ValueError("no legal TRAIN support for the final Level refit")
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("the final Level refit needs at least one seed")
    tail = max(2, int(-(-len(train_ids) * FINAL_REFIT_TAIL_FRACTION // 1)))
    fit_ids, early_ids = train_ids[:-tail], train_ids[-tail:]

    fingerprint = oof_fingerprint or assert_same_level_architecture(config)
    fitted_states = []
    for seed in seeds:
        fitted = fit_level_model(host, residual, residual_history, config,
                                 fit_ids, early_ids, seed, device)
        model = CoarseLevelModel(config).to(device)
        model.load_state_dict(fitted["state_dict"])
        for param in model.parameters():
            param.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise AssertionError("the final Level predictor must be frozen")
        # All seeds share one architecture by construction; compare each anyway so
        # a divergent build path cannot pass unnoticed.
        assert level_architecture_fingerprint(config, model) == fingerprint
        fitted_states.append(fitted)

    # A seed-invariant quantity: every member sees the same fit_ids, so this is one
    # number, not an average of three.
    scales = {round(float(f["residual_scale"]), 12) for f in fitted_states}
    if len(scales) != 1:
        raise AssertionError(
            f"the ensemble members disagree on the history scale: {sorted(scales)}")
    s_b_values = {round(float(f["s_b"]), 12) for f in fitted_states}
    if len(s_b_values) != 1:
        raise AssertionError(
            f"the ensemble members disagree on s_b: {sorted(s_b_values)}")

    return FinalLevelPredictor(
        state_dicts=tuple(f["state_dict"] for f in fitted_states),
        seeds=seeds, config=config,
        architecture_fingerprint=fingerprint,
        oof_architecture_fingerprint=fingerprint,
        fit_ids=fit_ids, early_ids=early_ids,
        best_epochs=tuple(int(f["best_epoch"]) for f in fitted_states),
        residual_scale=float(fitted_states[0]["residual_scale"]),
        s_b=float(fitted_states[0]["s_b"]))


def level_fit_entry(runner_callable: Callable, *args, **kwargs):
    """Guard used by the runner so no fold fit can start without an authorization.

    The runner passes its own gated entry here; the indirection exists so a test
    can install a spy and prove the runner never reaches a fit at import time.
    """
    if not kwargs.pop("authorized", False):
        raise PermissionError(
            "Stage-1 fits require explicit scientific-execution authorization; "
            "harness preparation does not run them")
    return runner_callable(*args, **kwargs)
