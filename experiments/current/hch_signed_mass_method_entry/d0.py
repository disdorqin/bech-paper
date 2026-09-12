"""D0 -- the no-training geometry report over ``POST_TRAIN`` (PART A5).

D0 exists to answer a question that must be settled **before** any network is
trained: is the object the method factorises actually there?  If the residual
mass is concentrated on a handful of days, if the positive and negative shapes
are indistinguishable from uniform, or if the factorization does not reconstruct
the residual it claims to explain, then a Shape/Amplitude method has nothing to
learn and the component screen later in this experiment would be measuring noise.

It is deliberately **descriptive**.  Nothing here gates the run, nothing here
tunes anything, and nothing here is a novelty test: the only outcomes that stop
execution are structural -- a factorization that fails to reconstruct, a role
that should not be reachable, a read of a sealed partition.  Every threshold it
uses for the "extreme / tail / negative price" relations is read from the frozen
breadth-stage threshold file, never re-derived from the partition being
described.

D0 does **not** enable KNN, does not train anything, and touches ``POST_TRAIN``
only.  ``DEV_EVAL`` rows are never materialised by this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from . import metrics as M
from . import raw_evidence as RE
from .china5_adapter import CellDataset
from .contracts import HarnessError, LegalityError

__all__ = ["D0_BY_CELL_FIELDS", "analyse_cell", "run"]


#: Per-cell scalar columns of ``D0_BY_CELL.csv``.  Nested records (peak-position
#: histograms, the threshold-relation block) are emitted into the machine
#: readable JSON; the CSV keeps the flat numbers a reader compares by eye.
D0_BY_CELL_FIELDS = (
    "market", "host", "cell", "n_post_train_days", "n_entries",
    "mass_positive_mean", "mass_positive_median", "mass_positive_q25",
    "mass_positive_q75", "mass_positive_iqr", "mass_positive_std",
    "mass_negative_mean", "mass_negative_median", "mass_negative_q25",
    "mass_negative_q75", "mass_negative_iqr", "mass_negative_std",
    "mass_positive_near_zero_fraction", "mass_negative_near_zero_fraction",
    "mass_positive_exact_zero_fraction", "mass_negative_exact_zero_fraction",
    "mass_asymmetry_median_ratio", "mass_asymmetry_mean_ratio",
    "n_days_defined_positive", "n_days_defined_negative",
    "n_days_defined_both", "n_days_defined_neither",
    "shape_positive_entropy_mean", "shape_negative_entropy_mean",
    "shape_positive_concentration_mean", "shape_negative_concentration_mean",
    "shape_positive_entropy_uniform_fraction",
    "shape_negative_entropy_uniform_fraction",
    "peak_positive_modal_hour", "peak_positive_modal_fraction",
    "peak_negative_modal_hour", "peak_negative_modal_fraction",
    "w1_adjacent_pair_mean", "w1_random_pair_mean", "w1_pair_ratio",
    "reconstruction_max_abs_err", "reconstruction_max_rel_err",
    "knn_enabled", "roles_used", "dev_eval_rows_materialised",
    "protected_final_read_count",
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman rank correlation, computed without scipy.

    Ties get average ranks; a constant input has no rank variance and the
    correlation is undefined, which is reported as ``None`` rather than as a
    fabricated zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return None
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom <= 0.0:
        return None
    return float((rx * ry).sum() / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _entropy(shape: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats of each simplex row (``0 log 0 = 0``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shape > 0.0, shape * np.log(shape), 0.0)
    return -terms.sum(axis=1)


def _modal_peak(shape: np.ndarray) -> Tuple[Optional[int], float]:
    """The most frequent ``argmax`` horizon and its share of the days."""
    if shape.shape[0] == 0:
        return None, 0.0
    peaks = np.argmax(shape, axis=1)
    counts = np.bincount(peaks, minlength=shape.shape[1])
    hour = int(np.argmax(counts))
    return hour, float(counts[hour] / peaks.size)


def _w1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return M.wasserstein1(a, b)


def _near_zero_fraction(mass: np.ndarray) -> float:
    """Share of days whose mass is negligible *relative to this cell's own scale*.

    A fixed absolute cut would call every Gansu day "near zero" and no Ningxia
    day, which says nothing about either.  The reference is the cell's own median
    positive mass; the floor of 1.0 only keeps a tiny-mass cell from turning its
    rounding noise into a threshold.
    """
    positive = mass[mass > 0.0]
    reference = float(np.median(positive)) if positive.size else 1.0
    threshold = C.D0_NEAR_ZERO_FRACTION * max(1.0, reference)
    return float((mass <= threshold).mean())


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------
def analyse_cell(dataset: CellDataset) -> Dict[str, Any]:
    """The full D0 geometry record for one ``market x Host`` cell.

    Raises only on a structural failure: an empty ``POST_TRAIN``, a
    factorization whose reconstruction error is not round-off, or a role that
    the module was never allowed to touch.
    """
    rows = dataset.positions(C.ROLE_POST_TRAIN)
    if rows.size == 0:
        raise LegalityError(f"{dataset.key}: POST_TRAIN is empty; D0 is undefined")
    rows = np.sort(rows.astype(np.int64))

    residual = dataset.residual[rows].astype(np.float64)
    valid = dataset.valid_mask[rows].astype(bool)
    target = dataset.target[rows].astype(np.float64)

    # Exactly the core's factorization, on the masked residual: a masked step
    # contributes to neither sign, and the identity below is what proves the
    # two masses and two shapes still reconstruct the residual they came from.
    masked = residual * valid
    parts = M.shape_decomposition(masked)
    a_pos, a_neg = parts["mass_positive"], parts["mass_negative"]
    s_pos, s_neg = parts["shape_positive"], parts["shape_negative"]
    defined_pos, defined_neg = parts["defined_positive"], parts["defined_negative"]

    rebuilt = a_pos[:, None] * s_pos - a_neg[:, None] * s_neg
    recon_abs = float(np.max(np.abs(rebuilt - masked))) if masked.size else 0.0
    scale = float(np.max(np.abs(masked))) if masked.size else 0.0
    recon_rel = recon_abs / scale if scale > 0.0 else 0.0
    if recon_abs > 1e-6 + 1e-6 * scale:
        raise HarnessError(
            f"{dataset.key}: the signed-mass factorization does not reconstruct "
            f"the residual it was formed from (max |r - (A+S+ - A-S-)| = "
            f"{recon_abs:.3e}); D0 cannot describe an object that is not there")

    thresholds = M.load_thresholds(dataset.market)
    q05, q95 = thresholds["q05"], thresholds["q95"]

    near_zero_pos = _near_zero_fraction(a_pos)
    near_zero_neg = _near_zero_fraction(a_neg)

    ok_pos, ok_neg = s_pos[defined_pos], s_neg[defined_neg]
    horizon = s_pos.shape[1]
    uniform_entropy = float(np.log(horizon))
    ent_pos = _entropy(ok_pos) if ok_pos.size else np.zeros(0)
    ent_neg = _entropy(ok_neg) if ok_neg.size else np.zeros(0)
    conc_pos = ok_pos.max(axis=1) if ok_pos.size else np.zeros(0)
    conc_neg = ok_neg.max(axis=1) if ok_neg.size else np.zeros(0)
    peak_pos_hour, peak_pos_frac = _modal_peak(ok_pos)
    peak_neg_hour, peak_neg_frac = _modal_peak(ok_neg)

    pair = _pair_diagnostics(s_pos, defined_pos)

    entries_upper = (target >= q95) & valid
    entries_lower = (target <= q05) & valid
    entries_negative = (target < 0.0) & valid
    relation = {
        "frozen_thresholds": {"q05": q05, "q95": q95,
                              "source": thresholds["threshold_source"],
                              "source_sha256": thresholds["threshold_source_sha256"],
                              "recomputed_for_this_evaluation": False},
        "upper_tail": _subset_relation(a_pos, a_neg, entries_upper),
        "lower_tail": _subset_relation(a_pos, a_neg, entries_lower),
        "negative_price": _subset_relation(a_pos, a_neg, entries_negative),
        "n_days_with_upper_tail_entry": int(entries_upper.any(axis=1).sum()),
        "n_days_with_lower_tail_entry": int(entries_lower.any(axis=1).sum()),
        "n_days_with_negative_price": int(entries_negative.any(axis=1).sum()),
    }

    positive = a_pos[a_pos > 0.0]
    negative = a_neg[a_neg > 0.0]

    record: Dict[str, Any] = {
        "market": dataset.market,
        "host": dataset.host,
        "cell": dataset.key,
        "n_post_train_days": int(rows.size),
        "n_entries": int(valid.sum()),
        "horizon": int(horizon),
        "mass_positive_mean": float(a_pos.mean()),
        "mass_positive_median": float(np.median(a_pos)),
        "mass_positive_q25": float(np.quantile(a_pos, 0.25)),
        "mass_positive_q75": float(np.quantile(a_pos, 0.75)),
        "mass_positive_iqr": float(np.quantile(a_pos, 0.75) - np.quantile(a_pos, 0.25)),
        "mass_positive_std": float(a_pos.std(ddof=1)) if a_pos.size > 1 else 0.0,
        "mass_negative_mean": float(a_neg.mean()),
        "mass_negative_median": float(np.median(a_neg)),
        "mass_negative_q25": float(np.quantile(a_neg, 0.25)),
        "mass_negative_q75": float(np.quantile(a_neg, 0.75)),
        "mass_negative_iqr": float(np.quantile(a_neg, 0.75) - np.quantile(a_neg, 0.25)),
        "mass_negative_std": float(a_neg.std(ddof=1)) if a_neg.size > 1 else 0.0,
        "mass_positive_near_zero_fraction": near_zero_pos,
        "mass_negative_near_zero_fraction": near_zero_neg,
        "mass_positive_exact_zero_fraction": float((a_pos == 0.0).mean()),
        "mass_negative_exact_zero_fraction": float((a_neg == 0.0).mean()),
        "mass_asymmetry_median_ratio": _ratio(float(np.median(a_pos)),
                                              float(np.median(a_neg))),
        "mass_asymmetry_mean_ratio": _ratio(float(a_pos.mean()), float(a_neg.mean())),
        "n_days_defined_positive": int(defined_pos.sum()),
        "n_days_defined_negative": int(defined_neg.sum()),
        "n_days_defined_both": int((defined_pos & defined_neg).sum()),
        "n_days_defined_neither": int((~defined_pos & ~defined_neg).sum()),
        "shape_positive_entropy_mean": _mean_or_none(ent_pos),
        "shape_negative_entropy_mean": _mean_or_none(ent_neg),
        "shape_positive_entropy_uniform_fraction": (
            _fraction(ent_pos, uniform_entropy)),
        "shape_negative_entropy_uniform_fraction": (
            _fraction(ent_neg, uniform_entropy)),
        "shape_positive_concentration_mean": _mean_or_none(conc_pos),
        "shape_negative_concentration_mean": _mean_or_none(conc_neg),
        "peak_positive_modal_hour": peak_pos_hour,
        "peak_positive_modal_fraction": peak_pos_frac,
        "peak_negative_modal_hour": peak_neg_hour,
        "peak_negative_modal_fraction": peak_neg_frac,
        "shape_positive_entropy_uniform_nats": uniform_entropy,
        "w1_adjacent_pair_mean": pair["adjacent_mean"],
        "w1_random_pair_mean": pair["random_mean"],
        "w1_pair_ratio": _ratio(pair["adjacent_mean"], pair["random_mean"]),
        "w1_pair_seed": int(C.D0_RANDOM_PAIR_SEED),
        "w1_pair_n_adjacent": pair["n_adjacent"],
        "w1_pair_n_random": pair["n_random"],
        "reconstruction_max_abs_err": recon_abs,
        "reconstruction_max_rel_err": recon_rel,
        "reconstruction_identity": "r = A+S+ - A-S- on the masked POST_TRAIN residual",
        "role_threshold_relations": relation,
        "shape_positive_peak_histogram": _peak_histogram(ok_pos),
        "shape_negative_peak_histogram": _peak_histogram(ok_neg),
        "knn_enabled": False,
        "roles_used": [C.ROLE_POST_TRAIN],
        "dev_eval_rows_materialised": 0,
        "protected_final_read_count": 0,
        "notes": (
            "Descriptive only.  Nothing in D0 gates the run, tunes a parameter or "
            "enables KNN; the adjacent-vs-random W1 comparison is a similarity "
            "diagnostic, not a novelty test."),
    }
    return record


def _ratio(numerator: Optional[float], denominator: Optional[float]
           ) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return float(values.mean()) if values.size else None


def _fraction(values: np.ndarray, threshold: float) -> Optional[float]:
    """Share of days whose entropy is within 1% of the uniform maximum."""
    if values.size == 0:
        return None
    return float((values >= 0.99 * threshold).mean())


def _peak_histogram(shape: np.ndarray) -> Dict[str, int]:
    if shape.shape[0] == 0:
        return {}
    counts = np.bincount(np.argmax(shape, axis=1), minlength=shape.shape[1])
    return {str(h): int(c) for h, c in enumerate(counts) if c > 0}


def _pair_diagnostics(shape: np.ndarray, defined: np.ndarray
                      ) -> Dict[str, Any]:
    """Adjacent-day vs deterministic random-day W1, as a similarity diagnostic.

    The random partner is drawn from a fixed seed over the *defined* days only,
    so the comparison is reproducible to the byte and never depends on a library
    RNG default.  A ratio near 1 means "yesterday looks like a random other day",
    which is a statement about the data, not a gate: no threshold on this ratio
    is registered anywhere.
    """
    idx = np.flatnonzero(defined)
    if idx.size < 3:
        return {"adjacent_mean": None, "random_mean": None,
                "n_adjacent": 0, "n_random": 0}

    adjacent = [(idx[k], idx[k + 1]) for k in range(idx.size - 1)]
    rng = np.random.default_rng(C.D0_RANDOM_PAIR_SEED)
    n_random = len(adjacent)
    left = rng.integers(0, idx.size, size=n_random)
    right = rng.integers(0, idx.size, size=n_random)
    random_pairs = [(idx[a], idx[b]) for a, b in zip(left, right)
                    if a != b]

    adj_w1 = [_w1(shape[[a]], shape[[b]])[0] for a, b in adjacent]
    rnd_w1 = [_w1(shape[[a]], shape[[b]])[0] for a, b in random_pairs]
    return {
        "adjacent_mean": float(np.mean(adj_w1)) if adj_w1 else None,
        "random_mean": float(np.mean(rnd_w1)) if rnd_w1 else None,
        "n_adjacent": int(len(adj_w1)),
        "n_random": int(len(rnd_w1)),
    }


def _subset_relation(a_pos: np.ndarray, a_neg: np.ndarray,
                     entries: np.ndarray) -> Dict[str, Any]:
    """How the residual masses relate to one frozen target subset.

    ``entries`` is the ``(N, H)`` indicator of the subset.  The block reports the
    mass levels inside and outside the days that contain it, plus the rank
    correlation between the mass and the per-day share of subset entries.  All
    of it is descriptive: no threshold on any of these numbers is registered.
    """
    in_days = entries.any(axis=1)
    share = entries.mean(axis=1)
    return {
        "n_days_in_subset": int(in_days.sum()),
        "n_days_outside_subset": int((~in_days).sum()),
        "mass_positive_mean_in_subset": (float(a_pos[in_days].mean())
                                         if in_days.any() else None),
        "mass_positive_mean_outside_subset": (float(a_pos[~in_days].mean())
                                              if (~in_days).any() else None),
        "mass_negative_mean_in_subset": (float(a_neg[in_days].mean())
                                         if in_days.any() else None),
        "mass_negative_mean_outside_subset": (float(a_neg[~in_days].mean())
                                              if (~in_days).any() else None),
        "spearman_mass_positive_vs_entry_share": _spearman(a_pos, share),
        "spearman_mass_negative_vs_entry_share": _spearman(a_neg, share),
    }


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------
def run(cells: Sequence[Tuple[str, str]],
        build=None,
        out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run D0 over every cell and write the three required artifacts.

    ``build`` is injected so the caller owns dataset construction (and with it
    the readiness gate); this function never opens an artifact itself and never
    resolves a role other than ``POST_TRAIN``.
    """
    if build is None:
        from .china5_adapter import build_cell_dataset

        build = build_cell_dataset

    records: List[Dict[str, Any]] = []
    for market, host in cells:
        dataset = build(market, host)
        records.append(analyse_cell(dataset))

    target_dir = RE.confine(Path(out_dir) if out_dir
                            else RE.evidence_root() / "01_d0_geometry",
                            "D0 output directory")
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = RE.confine(target_dir / "D0_BY_CELL.csv", "D0 CSV")
    json_path = RE.confine(target_dir / "D0_BY_CELL.json", "D0 JSON")
    summary_path = RE.confine(target_dir / "D0_SUMMARY.md", "D0 summary")

    from .contracts import write_csv

    write_csv([{f: _csv_value(r.get(f)) for f in D0_BY_CELL_FIELDS}
               for r in records], csv_path, D0_BY_CELL_FIELDS)
    json_path.write_text(
        json.dumps({"schema": "signed_mass_d0_geometry.v1",
                    "n_cells": len(records),
                    "roles_used": [C.ROLE_POST_TRAIN],
                    "cells": records},
                   ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    summary_path.write_text(_render_summary(records), encoding="utf-8")

    return {"cells": records, "csv": str(csv_path), "json": str(json_path),
            "summary": str(summary_path),
            "max_reconstruction_abs_err": max(
                (r["reconstruction_max_abs_err"] for r in records), default=0.0)}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _render_summary(records: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    add = lines.append
    add("# D0 — signed-mass geometry over `POST_TRAIN`")
    add("")
    add(f"- cells: {len(records)}")
    add(f"- roles used: `{C.ROLE_POST_TRAIN}` only; `DEV_EVAL` never materialised")
    add("- `PROTECTED_FINAL` read count: 0")
    add("- KNN: disabled (D0 does not enable it)")
    add("- training: none — this report is pure arithmetic on frozen residuals")
    add("")
    add("## What D0 is, and is not")
    add("")
    add("D0 describes the object the method factorises: how the Host's residual mass")
    add("is distributed across days, whether its signed Shapes carry geometry beyond")
    add("uniform, and whether the factorization reconstructs the residual exactly.")
    add("")
    add("It is **descriptive**.  Nothing here gates the run, tunes a parameter or")
    add("tests novelty; the only failures that stop execution are structural")
    add("(reconstruction, roles, sealed reads).  The adjacent-vs-random W1 column is a")
    add("similarity diagnostic with no registered threshold.")
    add("")
    add("## Per cell")
    add("")
    add("| market | host | days | A+ median | A- median | defined + / − | "
        "entropy + / − | peak + / − | W1 adj/rand | recon err |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        add("| {market} | {host} | {n} | {ap:.4g} | {an:.4g} | {dp} / {dn} | "
            "{ep} / {en} | {pp} / {pn} | {wa:.4g} / {wr:.4g} | {rc:.2e} |".format(
                market=r["market"], host=r["host"], n=r["n_post_train_days"],
                ap=r["mass_positive_median"], an=r["mass_negative_median"],
                dp=r["n_days_defined_positive"], dn=r["n_days_defined_negative"],
                ep=_fmt(r["shape_positive_entropy_mean"]),
                en=_fmt(r["shape_negative_entropy_mean"]),
                pp=r["peak_positive_modal_hour"], pn=r["peak_negative_modal_hour"],
                wa=r["w1_adjacent_pair_mean"] or float("nan"),
                wr=r["w1_random_pair_mean"] or float("nan"),
                rc=r["reconstruction_max_abs_err"]))
    add("")
    add("## Structural checks")
    add("")
    worst = max((r["reconstruction_max_abs_err"] for r in records), default=0.0)
    add(f"- max reconstruction error over all cells: `{worst:.3e}`")
    add(f"- cells with a non-empty positive-Shape day set: "
        f"{sum(1 for r in records if r['n_days_defined_positive'] > 0)}/{len(records)}")
    add(f"- cells with a non-empty negative-Shape day set: "
        f"{sum(1 for r in records if r['n_days_defined_negative'] > 0)}/{len(records)}")
    add(f"- cells with at least one exactly-zero-mass day in either sign: "
        f"{sum(1 for r in records if (r['mass_positive_exact_zero_fraction'] > 0) or (r['mass_negative_exact_zero_fraction'] > 0))}"
        f"/{len(records)}")
    add("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
