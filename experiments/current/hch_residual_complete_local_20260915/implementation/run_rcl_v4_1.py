"""RCL v4.1 scientific runner — source only, **not authorized to execute**.

Importing this module fits nothing and reads nothing: every import is pure and
the only path that reaches a fit is :func:`main`, behind an explicit token.  The
harness-preparation task implements the whole run so the protocol has exactly one
code path, exercises it structurally, and still leaves the scientific-fit count at
zero.

Order, and why each step is where it is:

1. **Boundary.**  :func:`data_boundary.load_authorized_cell`, then
   :meth:`AuthorizedCell.chronological_layout`.  Loading first is deliberate: if
   the boundary refuses, nothing downstream has started.  The layout check is what
   licenses the single chronological index space the rest of the run addresses days
   in — TRAIN day ``d`` is authorized position ``d`` and VAL follows it.
2. **Stage-1 causal OOF cross-fit.**  Folds come from
   :func:`rcl_contract.oof_partition`; each fold trains strictly earlier than its
   own target block.  These OOF Levels are what the Local targets are built on.
3. **Scales.**  Both are fitted from LOCAL_TRAIN only: the Level conditioner from
   the causal OOF coarse Levels, the q-geometry scales from the LOCAL_TRAIN q
   targets.  Both must come after the OOF panel is closed — fitting either one
   earlier would let a later fold's Level reach an earlier day's scaling.
4. **Panel.**  ``LOCAL_TRAIN = B2 u B3 u B4``, each day carrying the OOF Level it
   was assigned.
5. **Final Level refit**, frozen, with its architecture fingerprint compared to the
   OOF build.  After the panels exist so the refit cannot be tuned against them,
   and before any VAL day is seen.
6. **Stage-2 fits**, one per cell x variant x seed.  The four variants share the
   frozen Level, the panel, the seeds and the recipe; only the parameterization
   differs.
7. **VAL prequential evaluation**, CAL then EVAL, on the raw outputs.
8. **Write.**  Closed-schema per-fit records, the panel closure aggregate, the
   per-cell closure inputs the verifier recomputes from, and the run index.

Nothing here is a scientific result.  This module has never been run.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core import (LocalNormalizationStats, LevelFeatureBuilder,
                  ResidualCompleteLocalFeatureBuilder, build_base_tokens,
                  build_q, build_residual_complete_targets, default_config,
                  wasserstein1)

from mvp.hch_residual_complete_local_v4_1 import (EXPERIMENT_ONLY_VARIANTS,
                                                  PRIMARY_VARIANT,
                                                  build_candidate)

from data_boundary import (AuditCounters, D_BASE_TOKENS,
                           load_authorized_cell)
from level_condition_stats import LevelConditionStats
from local_training import (LOCAL_HISTORY_DAYS, UntiedMassStats,
                            build_local_train_panel, train_stage2,
                            untied_mass_targets)
from oof_level import (FrozenLevelEnsemble, OofLevelPanel,
                       assert_same_level_architecture, final_level_refit,
                       fit_oof_fold, lower_median_across_seeds,
                       plan_oof_folds, predict_level)
from prequential import (run_val_prequential, secondary_calibration,
                         seed_windows_from_oof)
from rcl_contract import (CELLS, SEEDS, TEST_LABEL_READ_COUNT, local_train_days,
                          oof_partition)
from result_schema import (aggregate_closure_ratio, ids_sha256,
                           make_cell_record, make_run_index, write_json)

#: The token a caller must present before ``main`` will run a scientific fit.
AUTHORIZATION_TOKEN = "HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_SCIENTIFIC_EXECUTION"

PROTOCOL_ID = "HCH_RESIDUAL_COMPLETE_LOCAL_V4_1_20260915"
EVIDENCE_ROOT = Path(
    "experiments/evidence/hch_residual_complete_local_v4_1_20260915")

VARIANT_ORDER = (PRIMARY_VARIANT,) + EXPERIMENT_ONLY_VARIANTS


# --------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSurfaces:
    """One cell's authorized arrays plus the index space the run addresses them in.

    ``residual_history`` is the raw-Host-residual window array for every authorized
    day.  It is legal for any day because window ``d`` ends at day ``d-1``: a day's
    own outcome is never in it.  Stage 1 reads it; Stage 2 never does, which is why
    it lives here rather than on the Local panel.
    """

    market: str
    host: str
    host_forecast: "np.ndarray"
    target: "np.ndarray"
    residual: "np.ndarray"
    valid_mask: "np.ndarray"
    day_ids: "np.ndarray"
    n_train: int
    n_val: int

    @property
    def n_days(self) -> int:
        return int(self.residual.shape[0])

    @property
    def val_ids(self) -> list[int]:
        """VAL day indices in the run's single chronological index space."""
        return list(range(self.n_train, self.n_train + self.n_val))

    def residual_history(self, window: int = LOCAL_HISTORY_DAYS) -> "np.ndarray":
        out = np.zeros((self.n_days, window, self.residual.shape[-1]),
                       dtype=np.float32)
        for day in range(self.n_days):
            start = max(0, day - window)
            span = self.residual[start:day]
            if len(span):
                out[day, -len(span):] = span
        return out

    def train_thresholds(self) -> dict:
        """TRAIN-only reference quantiles for the tail diagnostics.

        Thresholds come from TRAIN and are then applied unchanged to EVAL.  Reading
        a quantile off EVAL would let the split that is being scored choose where
        its own tails begin, which is the classic way a tail metric becomes
        uninformative without any number looking wrong.
        """
        train_target = self.target[:self.n_train]
        train_daily = np.abs(self.residual[:self.n_train]).mean(axis=1)
        q_low, q_high = np.quantile(train_target, [0.05, 0.95])
        return {"target_q05": float(q_low), "target_q95": float(q_high),
                "daily_error_q90": float(np.quantile(train_daily, 0.90)),
                "source": "TRAIN_ONLY"}


def cell_surfaces(cell) -> CellSurfaces:
    """Wrap an authorized cell in the runner's chronological index space."""
    n_train, n_val = cell.chronological_layout()
    return CellSurfaces(
        market=cell.market, host=cell.host,
        host_forecast=np.asarray(cell.host_forecast, dtype=np.float32),
        target=np.asarray(cell.target, dtype=np.float32),
        residual=np.asarray(cell.residual(), dtype=np.float32),
        valid_mask=np.asarray(cell.valid_mask, dtype=np.float32),
        day_ids=np.asarray(cell.day_ids),
        n_train=n_train, n_val=n_val)


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------
def stage1_oof(surfaces: CellSurfaces, config) -> OofLevelPanel:
    """Fit every causal fold on the TRAIN part and collect the OOF Levels."""
    panel = OofLevelPanel()
    history = surfaces.residual_history()
    for fold in plan_oof_folds(surfaces.n_train, SEEDS):
        result = fit_oof_fold(fold, surfaces.host_forecast, surfaces.residual,
                              history, config, "cpu")
        panel.add_fold(result, provenance="oof")
    return panel


def fit_b4_tail_days(n_train: int) -> list[int]:
    """The last ``LOCAL_HISTORY_DAYS`` days of B4 — the registered VAL seed."""
    b4 = list(oof_partition(n_train)["B4"])
    if len(b4) < LOCAL_HISTORY_DAYS:
        raise ValueError(
            f"B4 has {len(b4)} days; the VAL seed needs {LOCAL_HISTORY_DAYS} and a "
            "short B4 is a partition change, not a reason to seed from B1 or warm-up")
    return b4[-LOCAL_HISTORY_DAYS:]


def fit_conditioner(oof_panel: OofLevelPanel, n_train: int) -> LevelConditionStats:
    """Step 3a: TRAIN-only Level conditioner from LOCAL_TRAIN causal OOF Levels.

    ``local_train_days`` is passed in so a warm-up or B1 record is refused rather
    than quietly included, and the fit never consults a q/delta statistic: this
    object scales a coarse-Level quantity and nothing else.  It is the sole
    authority on that scaling — nothing downstream may call
    ``LocalNormalizationStats.normalize_level_condition`` instead.
    """
    records = oof_panel.local_train_records(n_train)
    if not records:
        raise ValueError("no LOCAL_TRAIN causal OOF Levels to fit the conditioner")
    return LevelConditionStats.fit(records, local_train_days(n_train))


def _local_train_q(surfaces: CellSurfaces, oof_panel: OofLevelPanel, n_train: int):
    """``(q, valid_mask)`` over LOCAL_TRAIN, through the core's authority.

    One site, so the q-geometry scales and R2's signed-mass scales cannot be
    fitted on two subtly different arrays: they are two statistics *of the same
    targets*, and a second construction is how they would stop being that.
    """
    days = local_train_days(n_train)
    rows = np.asarray(days, dtype=int)
    level = np.asarray([oof_panel.values[int(d)] for d in days], dtype=np.float32)
    mask = torch.as_tensor(surfaces.valid_mask[rows], dtype=torch.float32)
    q = build_q(torch.as_tensor(surfaces.residual[rows], dtype=torch.float32),
                torch.as_tensor(level, dtype=torch.float32), mask)
    return q, mask


def fit_local_stats(surfaces: CellSurfaces, oof_panel: OofLevelPanel,
                    n_train: int) -> LocalNormalizationStats:
    """Step 3b: q-geometry scales from the LOCAL_TRAIN targets only.

    The q values are built by the core from the LOCAL_TRAIN residual and its causal
    OOF Level, before any panel exists, so the scales cannot depend on how the
    panel later happens to be sliced.
    """
    q, mask = _local_train_q(surfaces, oof_panel, n_train)
    return LocalNormalizationStats.fit(q, mask)


def fit_untied_stats(surfaces: CellSurfaces, oof_panel: OofLevelPanel,
                     n_train: int) -> UntiedMassStats:
    """R2's native signed-mass scales, fitted on the same LOCAL_TRAIN q."""
    q, mask = _local_train_q(surfaces, oof_panel, n_train)
    return UntiedMassStats.fit(q, mask)


def build_panel(surfaces: CellSurfaces, oof_panel: OofLevelPanel,
                stats: LocalNormalizationStats, conditioner: LevelConditionStats,
                untied_stats: UntiedMassStats, n_train: int):
    """Step 4: the LOCAL_TRAIN panel over B2 u B3 u B4, with B1 as history.

    ``list(range(surfaces.n_days))`` is the whole authorized index space, because
    the panel is where the target support (B2 u B3 u B4) and the history support
    (B1 u B2 u B3 u B4) are separated; narrowing the argument here would make the
    history support unreachable and reintroduce the all-zero B2 warm start.
    """
    return build_local_train_panel(
        surfaces.market, surfaces.host, list(range(surfaces.n_days)),
        surfaces.host_forecast, surfaces.residual, surfaces.valid_mask,
        oof_panel, n_train, stats, conditioner, untied_stats,
        coarse_level_source=f"oof:{surfaces.market}/{surfaces.host}")


def local_target_sha256(panel) -> str:
    """Digest of the LOCAL_TRAIN ``q`` the panel was actually built on.

    Deliberately distinct from ``coarse_level_sha256``, which identifies the Level
    artifact.  This one identifies the *targets*: a panel whose Level matched but
    whose residual rows were sliced differently, or which was built against a
    different OOF vintage, changes this digest while leaving the Level digest
    untouched.  Collapsing the two fields into one value would retire the check
    they exist to make, so the verifier recomputes this from the cell and the
    recorded OOF Levels rather than from the panel.

    Over the **target** support, not the history support: q over B1 is history the
    model is handed, and hashing it here would make the digest change when a
    history window changed without any target moving.  ``panel.target_q()`` is the
    B2 u B3 u B4 rows, which is what the verifier recomputes.
    """
    q = np.ascontiguousarray(panel.target_q(), dtype=np.float32)
    return hashlib.sha256(q.tobytes()).hexdigest()


def load_frozen_level(predictor, device="cpu") -> FrozenLevelEnsemble:
    """Materialize the frozen Stage-1 refit as a read-only ensemble.

    The returned object is the *only* way VAL obtains a coarse Level, and it is a
    median over the same seeds the OOF panel was reduced with.  Handing VAL a
    single member instead would score the Local model against a differently
    aggregated Level than the one its targets were built on.
    """
    return FrozenLevelEnsemble(predictor, device)


def final_level_members(ensemble: FrozenLevelEnsemble, surfaces: CellSurfaces,
                        config) -> dict:
    """Each ensemble member's VAL Level, plus the median VAL actually reads.

    Published so the median is checkable rather than merely declared.  The audit's
    defect C was a refit that trained three seeds and then kept one; a record
    saying ``n_ensemble_members = 3`` cannot distinguish that from a genuine
    median, whereas three per-member vectors can: the verifier takes their
    coordinate-wise lower median itself and compares it with the Level every VAL
    emission reports.  The members are also the evidence that the three states are
    *different* fits, so an ensemble of one state repeated three times fails.
    """
    builder = LevelFeatureBuilder(config, residual_scale=ensemble.residual_scale)
    history = surfaces.residual_history()
    ids = surfaces.val_ids
    per_seed = np.stack([
        predict_level(model, surfaces.host_forecast, history, ids, builder, "cpu")
        for model in ensemble.models])
    return {
        "seeds": [int(s) for s in ensemble.predictor.seeds],
        "val_day_ids": [int(d) for d in ids],
        "aggregation": "coordinate_wise_lower_median_across_seeds",
        "per_seed_level": {str(int(seed)): _floats(row)
                           for seed, row in zip(ensemble.predictor.seeds, per_seed)},
        "median_level": _floats(lower_median_across_seeds(per_seed)),
    }


# --------------------------------------------------------------------------
# VAL prequential evaluation
# --------------------------------------------------------------------------
def evaluate_val(model, ensemble: FrozenLevelEnsemble, config, stats,
                 surfaces: CellSurfaces, oof_panel: OofLevelPanel,
                 val_ids) -> dict:
    """Step 7: chronological CAL/EVAL prequential evaluation for one fit.

    ``build_batch`` is the only thing that may read a day's inputs, and it reads
    them from the day's own row plus the already-revealed history it is handed.
    The Level feature builder is built with the **refit's own** residual scale: the
    frozen weights were fitted under that scale, and recomputing a fresh one here
    would feed them a differently scaled history than they saw in training.
    """
    level_builder = LevelFeatureBuilder(config,
                                        residual_scale=ensemble.residual_scale)
    history = surfaces.residual_history()
    local_builder = ResidualCompleteLocalFeatureBuilder(config, stats)

    def one_hot():
        return torch.ones(1, surfaces.host_forecast.shape[-1], dtype=torch.float32)

    def build_batch(day: int, history_windows):
        host = torch.as_tensor(surfaces.host_forecast[[day]], dtype=torch.float32)
        mask = one_hot()
        # One registered call site: the ensemble's coordinate-wise median over its
        # frozen members, evaluated through the same predictor the OOF panel used.
        level_value = float(ensemble.predict(
            surfaces.host_forecast, history, [day], level_builder)[0])
        level = torch.as_tensor([level_value], dtype=torch.float32)
        windows = list(history_windows)[-LOCAL_HISTORY_DAYS:]
        past_q = torch.as_tensor(np.stack(windows), dtype=torch.float32).unsqueeze(0)
        if past_q.shape[1] < LOCAL_HISTORY_DAYS:
            past_q = torch.cat([
                torch.zeros(1, LOCAL_HISTORY_DAYS - past_q.shape[1],
                            past_q.shape[-1], dtype=torch.float32), past_q], dim=1)
        batch = local_builder(
            host=host, valid_mask=mask,
            base_tokens=build_base_tokens(host, mask), coarse_level=level,
            past_q=past_q)
        return level_value, batch

    def reveal(day: int):
        return surfaces.residual[day], surfaces.valid_mask[day]

    seed = seed_windows_from_oof(oof_panel, lambda day: surfaces.residual[day],
                                 fit_b4_tail_days(surfaces.n_train))
    run = run_val_prequential(model, build_batch, reveal, val_ids, seed, "cpu")
    return {"run": run,
            "calibration": secondary_calibration(
                run, lambda day: surfaces.residual[day])}


def _subset_mean(values, selector) -> float:
    """Mean of ``values`` over ``selector``, or NaN when the region is empty.

    An empty region is reported as NaN rather than 0.0: a tail that no EVAL day
    falls into is *undefined*, and a zero would read as a perfect score.
    """
    if not selector.any():
        return float("nan")
    return float(np.mean(np.asarray(values)[selector]))


def val_metrics(surfaces: CellSurfaces, run, calibration) -> tuple[dict, dict]:
    """Protocol section 8 metrics for one fit, from raw arrays.

    Two errors, and only these two, are ever formed::

        e_H = r            (the Host's own residual)
        e_R = r - c        (the repaired residual)

    ``r`` is a residual, so every comparison is residual-against-residual.  The
    audit's finding was that several subset metrics compared a *price* against a
    *residual*; those lines are gone rather than re-signed, because a repair that
    moved the price toward the residual would have scored as a large error there
    and the number would still have looked plausible.

    ``q`` is built against the day's **predicted** coarse Level, which is what
    section 8 defines: ``q = r - b_hat*1``, with ``z_hat`` the emitted Local
    correction — not the full correction, which already contains ``b_hat``.

    Every threshold is the TRAIN-only quantile from
    :meth:`CellSurfaces.train_thresholds`; nothing here chooses a cut from EVAL.
    """
    eval_ids = list(run.split.eval_ids)
    cal_ids = list(run.split.cal_ids)
    emissions = run.emissions_for(eval_ids)
    thresholds = surfaces.train_thresholds()

    y_eval = surfaces.target[eval_ids]
    r_eval = surfaces.residual[eval_ids]
    correction = run.predictions(eval_ids)          # ``c``, the full correction
    level_prediction = np.asarray([e.coarse_level for e in emissions],
                                  dtype=np.float64)
    delta_prediction = np.asarray([e.delta for e in emissions], dtype=np.float64)
    amplitude_prediction = np.asarray([e.amplitude for e in emissions],
                                      dtype=np.float64)
    local = np.stack([np.asarray(e.local) for e in emissions])   # ``z_hat``

    e_host = r_eval
    e_repaired = r_eval - correction

    # The exact residual-complete decomposition of the realised day, taken against
    # the *predicted* Level so its ``q`` is the q section 8 names.  A zero Level
    # here would decompose the raw residual and silently report a different
    # quantity under the same name.
    host_targets = build_residual_complete_targets(
        torch.as_tensor(r_eval, dtype=torch.float32),
        torch.as_tensor(level_prediction, dtype=torch.float32),
        torch.as_tensor(surfaces.valid_mask[eval_ids], dtype=torch.float32))
    candidate_targets = build_residual_complete_targets(
        torch.as_tensor(r_eval - e_repaired, dtype=torch.float32),
        torch.as_tensor(level_prediction, dtype=torch.float32),
        torch.as_tensor(surfaces.valid_mask[eval_ids], dtype=torch.float32))

    shape_predicted = (torch.as_tensor(np.stack([e.shape_positive for e in emissions]),
                                       dtype=torch.float32),
                       torch.as_tensor(np.stack([e.shape_negative for e in emissions]),
                                       dtype=torch.float32))

    def emitted_shape_w1(targets) -> float:
        """Mean W1 of both emitted Shapes against a reference decomposition.

        The horizon is fully valid in this protocol (the HCH surfaces carry an
        all-ones mask), so the cumulative form of W1 needs no mask handling; a
        partially masked cell would need the masked variant instead.
        """
        positive, negative = shape_predicted
        w_pos = wasserstein1(positive, targets.shape_positive)
        w_neg = wasserstein1(negative, targets.shape_negative)
        return float((0.5 * (w_pos + w_neg)).mean())

    upper = y_eval >= thresholds["target_q95"]
    lower = y_eval <= thresholds["target_q05"]
    normal = ~(upper | lower)
    extreme = upper | lower
    failure_tail = np.broadcast_to(
        (np.abs(r_eval).mean(axis=1) >= thresholds["daily_error_q90"])[:, None],
        r_eval.shape)

    level_truth = r_eval.mean(axis=1)
    delta_truth = level_truth - level_prediction
    stage1_floor = float(np.mean(np.abs(delta_truth)))
    delta_closure_error = float(np.mean(np.abs(delta_truth - delta_prediction)))
    host_mae = float(np.mean(np.abs(e_host)))
    repaired_mae = float(np.mean(np.abs(e_repaired)))
    # Harm, not improvement: a day the repair made worse contributes, a day it
    # improved contributes nothing.  The opposite sign reads as a benefit and is
    # the audit's finding.
    harm = np.maximum(0.0, np.abs(e_repaired) - np.abs(e_host))

    cal_correction = run.predictions(cal_ids)
    cal_residual = surfaces.residual[cal_ids]
    alpha = float(calibration["alpha"])

    native_plus = [e.mass_plus for e in emissions]
    native_minus = [e.mass_minus for e in emissions]
    have_native = all(v is not None for v in native_plus + native_minus)
    native_targets = untied_mass_targets(
        torch.as_tensor(r_eval - level_prediction[:, None], dtype=torch.float32),
        torch.as_tensor(surfaces.valid_mask[eval_ids], dtype=torch.float32))

    metrics = {
        # Stage 1: the coarse Level, and the floor ``delta`` has to clear.
        "level_mae": stage1_floor,
        "level_zero_reference_mae": float(np.mean(np.abs(level_truth))),
        "stage1_floor": stage1_floor,
        "delta_mae": delta_closure_error,
        "delta_zero_reference_mae": stage1_floor,
        "delta_closure_error": delta_closure_error,
        "closure_ratio_cell": (delta_closure_error / stage1_floor
                               if stage1_floor > 0 else float("nan")),
        # Local geometry, against the q-centred target decomposition.
        "amplitude_mae": float(np.mean(np.abs(
            np.asarray(host_targets.amplitude, dtype=np.float64)
            - amplitude_prediction))),
        "shape_w1": emitted_shape_w1(host_targets),
        "q_reconstruction_mae": float(np.mean(np.abs(
            (r_eval - level_prediction[:, None]) - local))),
        # Overall, in residual-error space on both sides.
        "host_overall_mae": host_mae,
        "repaired_overall_mae": repaired_mae,
        "gain_vs_host_pct": 100.0 * (host_mae - repaired_mae) / host_mae,
        # Subsets: always ``|e_R|``.
        "upper_mae": _subset_mean(np.abs(e_repaired), upper),
        "lower_mae": _subset_mean(np.abs(e_repaired), lower),
        "extreme_mae": _subset_mean(np.abs(e_repaired), extreme),
        "tail_mae": _subset_mean(np.abs(e_repaired), failure_tail),
        "normal_mae": _subset_mean(np.abs(e_repaired), normal),
        "normal_host_mae": _subset_mean(np.abs(e_host), normal),
        "normal_harm": _subset_mean(harm, normal),
        # CAL is scored on CAL, with the CAL-fitted scalar — never on EVAL.
        "cal_mae": float(np.mean(np.abs(
            cal_residual - alpha * cal_correction))),
        "cal_raw_mae": float(np.mean(np.abs(
            cal_residual - cal_correction))),
        "cal_host_mae": float(np.mean(np.abs(cal_residual))),
    }
    diagnostics = {
        "alpha_secondary": alpha,
        "eval_mae_calibrated_secondary": float(np.mean(np.abs(
            r_eval - calibration["eval_calibrated"]))),
        "centered_amplitude_gap_host": float(np.mean(np.abs(
            np.asarray(host_targets.amplitude, dtype=np.float64)
            - amplitude_prediction))),
        "centered_amplitude_gap_candidate": float(np.mean(np.abs(
            np.asarray(candidate_targets.amplitude, dtype=np.float64)
            - amplitude_prediction))),
        "centered_shape_mae_host": emitted_shape_w1(host_targets),
        "centered_shape_mae_candidate": emitted_shape_w1(candidate_targets),
        # R2's own coordinates against their own targets; ``None`` elsewhere,
        # because a derived value here would defeat the check it exists for.
        "native_mass_plus_mae": (
            float(np.mean(np.abs(np.asarray(native_plus, dtype=np.float64)
                                 - np.asarray(native_targets.a_plus,
                                              dtype=np.float64))))
            if have_native else None),
        "native_mass_minus_mae": (
            float(np.mean(np.abs(np.asarray(native_minus, dtype=np.float64)
                                 - np.asarray(native_targets.a_minus,
                                              dtype=np.float64))))
            if have_native else None),
        "emission_precedes_reveal": bool(run.emission_precedes_reveal()),
        "n_eval_days": len(eval_ids),
    }
    return metrics, diagnostics


def closure_inputs(run, surfaces: CellSurfaces) -> dict:
    """The three vectors the panel-level closure ratio is defined on."""
    ids = list(run.split.cal_ids) + list(run.split.eval_ids)
    emissions = run.emissions_for(ids)
    level_prediction = np.asarray([e.coarse_level for e in emissions],
                                  dtype=np.float64)
    r = surfaces.residual[ids]
    return {"level_truth": r.mean(axis=1).astype(np.float64),
            "level_prediction": level_prediction,
            "delta_prediction": np.asarray([e.delta for e in emissions],
                                           dtype=np.float64)}


def _floats(values) -> list:
    return [float(x) for x in np.asarray(values).reshape(-1)]


def emission_records(run) -> list[dict]:
    """The full emission log, in delivery order, as plain JSON.

    The whole reported coordinate set travels with each day — including both
    Shapes — because the verifier recomputes the protocol's diagnostics from this
    file and must not have to take any of them on the runner's word.  Delivery
    order is preserved rather than sorted: the order *is* the prequential claim, so
    a sorted dump would destroy the evidence for it.
    """
    return [{"day": e.day_index, "coarse_level": e.coarse_level,
             "delta": e.delta, "amplitude": e.amplitude,
             "history_length": e.history_length,
             "mass_plus": e.mass_plus, "mass_minus": e.mass_minus,
             "shape_positive": _floats(e.shape_positive),
             "shape_negative": _floats(e.shape_negative),
             "local": _floats(e.local),
             "full_correction": _floats(e.full_correction)}
            for e in run.emissions]


def val_surface_records(surfaces: CellSurfaces, run) -> dict:
    """The VAL truth surface the verifier scores against, plus the split.

    Only VAL is written.  It is an authorized segment, and without it "independent
    verification" could only mean re-reading the runner's own numbers.
    """
    ids = surfaces.val_ids
    return {
        "market": surfaces.market, "host": surfaces.host,
        "n_train": surfaces.n_train, "n_val": surfaces.n_val,
        "authorized_rows": surfaces.n_days,
        "val_day_ids": [int(d) for d in ids],
        "day_ids": [str(d) for d in surfaces.day_ids[ids]],
        "host_forecast": _floats(surfaces.host_forecast[ids]),
        "target": _floats(surfaces.target[ids]),
        "residual": _floats(surfaces.residual[ids]),
        "valid_mask": _floats(surfaces.valid_mask[ids]),
        "horizon": int(surfaces.residual.shape[-1]),
        "cal_ids": [int(d) for d in run.split.cal_ids],
        "eval_ids": [int(d) for d in run.split.eval_ids],
    }


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def run_cell(market: str, host: str, counters: AuditCounters, device="cpu"):
    """Every variant x seed for one cell, from boundary load to written record.

    Returns ``(records, artifacts)``.  The artifacts are the surfaces the verifier
    recomputes from, not summaries of the run: the OOF Levels it re-fits the
    conditioner on, the VAL truth it re-scores against, and the emission log it
    re-derives every diagnostic from.
    """
    cell = load_authorized_cell(market, host, counters)
    surfaces = cell_surfaces(cell)
    config = default_config(D_BASE_TOKENS)

    oof_panel = stage1_oof(surfaces, config)
    conditioner = fit_conditioner(oof_panel, surfaces.n_train)
    stats = fit_local_stats(surfaces, oof_panel, surfaces.n_train)
    untied_stats = fit_untied_stats(surfaces, oof_panel, surfaces.n_train)
    panel = build_panel(surfaces, oof_panel, stats, conditioner, untied_stats,
                        surfaces.n_train)

    fingerprint = assert_same_level_architecture(config)
    predictor = final_level_refit(surfaces.host_forecast, surfaces.residual,
                                 surfaces.residual_history(), config, SEEDS,
                                 device, oof_fingerprint=fingerprint)
    ensemble = load_frozen_level(predictor, device)

    records: list[dict] = []
    emissions: dict[str, list[dict]] = {}
    closures: dict[str, dict] = {}
    val_records = None
    for variant in VARIANT_ORDER:
        collected = {"level_truth": [], "level_prediction": [], "delta_prediction": []}
        for seed in SEEDS:
            started = time.time()
            # Any ensemble member serves as the freeze witness: all of them are
            # frozen, hold disjoint parameters, and are checked by identity.  The
            # ensemble as a whole is never handed to Stage 2 — it is a prediction
            # object, not a module the optimizer may walk into.
            fitted = train_stage2(variant, ensemble.models[0], config, panel, seed,
                                  device, authorized=True)
            counters.seeds_run += 1
            model = build_candidate(variant, config,
                                    level_center=conditioner.center,
                                    level_scale=conditioner.scale,
                                    horizon=surfaces.residual.shape[-1]).to(device)
            model.load_state_dict(fitted["state_dict"])
            model.eval()

            evaluation = evaluate_val(model, ensemble, config, stats, surfaces,
                                      oof_panel, surfaces.val_ids)
            run = evaluation["run"]
            if val_records is None:
                val_records = val_surface_records(surfaces, run)
            emissions[f"{variant}__seed_{seed}"] = emission_records(run)
            metrics, diagnostics = val_metrics(surfaces, run,
                                               evaluation["calibration"])
            for key, value in closure_inputs(run, surfaces).items():
                collected[key].append(value)
            records.append(make_cell_record(
                market=market, host=host, variant=variant, seed=seed,
                metrics=metrics, diagnostics=diagnostics,
                provenance={
                    "protocol_id": PROTOCOL_ID,
                    "n_parameters": fitted["n_parameters"],
                    "best_epoch": fitted["best_epoch"],
                    "epochs_run": fitted["epochs_run"],
                    "fit_seconds": time.time() - started,
                    # The hash of the state at construction, before a single
                    # gradient step: this is the artifact that proves the seed was
                    # set before the model existed.
                    "initial_state_sha256": fitted["initial_state_sha256"],
                    "final_state_sha256": fitted["final_state_sha256"],
                    "coarse_level_source": panel.coarse_level_source,
                    "coarse_level_sha256": panel.coarse_level_sha256,
                    "level_condition_fingerprint": conditioner.fingerprint(),
                    "oof_level_architecture_fingerprint": fingerprint,
                    "final_level_architecture_fingerprint":
                        predictor.architecture_fingerprint,
                    "local_target_sha256": local_target_sha256(panel),
                    "local_fit_ids_sha256": ids_sha256(panel.fit_ids),
                    # The two supports, hashed separately.  ``target`` is B2 u B3 u B4
                    # and ``history`` is B1 u B2 u B3 u B4: the audit's defect D was a
                    # B1 history read against an all-zero warm start, and the only way
                    # to check the fix from outside is to publish both sets and let
                    # the verifier recompute them from the partition.
                    "local_target_ids_sha256": ids_sha256(panel.day_ids),
                    "local_history_ids_sha256": ids_sha256(panel.support_ids),
                    "source_sha256": cell.source_sha256,
                    "split_hash": cell.split_hash,
                    "threshold_source": "TRAIN_ONLY",
                }))
        closures[variant] = {
            key: np.concatenate(values) for key, values in collected.items()}

    artifacts = {
        "oof_levels": {
            "market": market, "host": host, "n_train": surfaces.n_train,
            "values": {str(int(d)): float(v)
                       for d, v in sorted(oof_panel.values.items())},
            "provenance": {str(int(d)): oof_panel.provenance[d]
                           for d in sorted(oof_panel.provenance)},
            "folds": oof_panel.fold_records,
            "conditioner": {"center": conditioner.center,
                            "scale": conditioner.scale,
                            "n_observations": conditioner.n_observations,
                            "fitted_days": [int(d) for d in conditioner.fitted_days],
                            "fingerprint": conditioner.fingerprint()},
            "final_level": predictor.as_dict(),
            "final_level_val": final_level_members(ensemble, surfaces, config),
        },
        "val": val_records,
        "emissions": emissions,
        "closures": {v: {k: _floats(x) for k, x in arrays.items()}
                     for v, arrays in closures.items()},
    }
    return records, artifacts


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized", default="",
                        help="the registered scientific-execution token")
    parser.add_argument("--evidence-root", default=str(EVIDENCE_ROOT))
    args = parser.parse_args(argv)
    if args.authorized != AUTHORIZATION_TOKEN:
        raise PermissionError(
            "the RCL v4.1 scientific run requires the registered execution token; "
            "harness preparation implements this path and does not execute it")

    torch.set_num_threads(1)
    counters = AuditCounters()
    root = Path(args.evidence_root)
    records: list[dict] = []
    by_variant: dict[str, dict[str, list]] = {v: {} for v in VARIANT_ORDER}

    for market, host in CELLS:
        cell_records, artifacts = run_cell(market, host, counters)
        records.extend(cell_records)
        stem = f"{market}__{host}"
        write_json(root / f"00_protocol/oof_levels/{stem}.json", artifacts["oof_levels"])
        write_json(root / f"01_val/{stem}.json", artifacts["val"])
        for key, payload in artifacts["emissions"].items():
            write_json(root / f"06_candidates/emissions/{stem}__{key}.json", payload)
        for variant, arrays in artifacts["closures"].items():
            # The closure ratio is a panel aggregate, but its inputs are per cell:
            # writing them here is what lets the verifier recompute the panel
            # number instead of re-reading the number it is checking.
            write_json(root / f"06_candidates/closure/{stem}__{variant}.json", arrays)
            for key, value in arrays.items():
                by_variant[variant].setdefault(key, []).append(np.asarray(value))

    closure = {variant: aggregate_closure_ratio(
        np.concatenate(arrays["level_truth"]),
        np.concatenate(arrays["level_prediction"]),
        np.concatenate(arrays["delta_prediction"]))
        for variant, arrays in by_variant.items()}

    write_json(root / "00_protocol/RUN_MANIFEST.json",
               {"protocol_id": PROTOCOL_ID, "cells": [list(c) for c in CELLS],
                "seeds": list(SEEDS), "variants": list(VARIANT_ORDER),
                "test_label_read_count": TEST_LABEL_READ_COUNT,
                "authorization_token": AUTHORIZATION_TOKEN})
    index_path = write_json(
        root / "06_candidates/RESULT_INDEX.json",
        make_run_index(records, protocol_id=PROTOCOL_ID,
                       counters=counters.as_dict(), cells=CELLS, seeds=SEEDS,
                       terminal_state="PENDING_INDEPENDENT_VERIFICATION",
                       closure=closure))
    for record in records:
        write_json(root / "06_candidates/records" /
                   f"{record['market']}__{record['host']}__{record['variant']}"
                   f"__seed_{record['seed']}.json", record)
    print(f"wrote {len(records)} records; index sha256 {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
