"""Independent, read-only verification of the executed development run (A6).

This module is the audit that has to be able to say *no*.  It is written against
the raw artifacts rather than the result tables, because a result table is a
claim about numbers and the numbers themselves are the only witness that the
claim is true.

**What "independent" means here, precisely.**  Every reported number is
recomputed from the persisted arrays using this module's own arithmetic --
its own MAE, its own Wasserstein distance, its own simplex construction, its own
tolerance rule -- and then compared with the number the run recorded.  Nothing
is imported from :mod:`metrics` to do the recomputation, so a sign error, a
transposed axis or a mis-masked mean inside the harness shows up as a
disagreement instead of being faithfully reproduced.  The frozen inputs that are
*definitions* rather than computations -- the per-market quantiles, the per-cell
``q90`` pair -- are read from their frozen files, because re-deriving a
definition would only test that two copies of the same rule agree.

**What is not re-typed.**  The *adjudication rules* -- which metric each
component is judged on, the five retention conditions, the six development gates
-- come from :mod:`aggregate`, because those are the registered rules and a
second transcription of them would be a chance to disagree with the
registration rather than a chance to catch an error.  The independence that
matters is over the numbers: the verifier feeds its *recomputed* cell table into
the same rules and requires the outcome to match the one on disk.  A verdict that
survives that is derived, not echoed; the file is compared against the
derivation and is never read as an input to it.

**Read-only.**  Nothing here writes outside ``06_audits/``, and nothing here
fits, evaluates or re-runs a model.  The one expensive thing it does is reload
each cell's frozen Host panel to prove the Host predictions in the raw artifacts
are the frozen ones -- a retrained Host is otherwise indistinguishable from a
correct one in every downstream number.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import aggregate as A
from . import config as C
from . import protocol
from . import raw_evidence as RE
from .contracts import HarnessError, sha256_file, write_csv
from .verify_preexecution import Audit

__all__ = ["run", "recompute_metrics", "RESULT_AUDIT_SCHEMA"]

RESULT_AUDIT_SCHEMA = "signed_mass_result_audit.v1"

#: Relative tolerance for a recomputed mean against the recorded one.  Both are
#: float64 means over the same values; the only legitimate difference is the
#: summation order, which this is far wider than and far tighter than any real
#: discrepancy.
_TOL = 1e-9


def _close(a: Any, b: Any, tol: float = _TOL) -> bool:
    fa, fb = A._number(a), A._number(b)
    if fa is None or fb is None:
        return fa is None and fb is None
    return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)


def _bool(value: Any) -> Optional[bool]:
    """A boolean read back from a CSV cell.

    ``write_csv`` stringifies every value, so a recorded condition arrives as the
    text ``"True"``.  Comparing it with :func:`_close` would report every field
    as a mismatch -- ``_number("True")`` is ``None`` -- which would make the audit
    fail loudly and for the wrong reason.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    return None


# --------------------------------------------------------------------------
# Independent recomputation
# --------------------------------------------------------------------------
def _mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Mean absolute error over the masked entries, or ``None`` if none."""
    values = np.abs(np.asarray(pred, dtype=np.float64)
                    - np.asarray(target, dtype=np.float64))[mask]
    return float(values.mean()) if values.size else None


def _simplex(residual: np.ndarray) -> Dict[str, np.ndarray]:
    """``A^pm``, ``S^pm`` and the defined-day flags, computed from scratch.

    Written out rather than called so the verifier is not testing the harness
    against itself.  A zero mass yields a uniform simplex and ``defined=False``;
    the uniform value is never read, since every consumer filters on ``defined``
    first.
    """
    r = np.asarray(residual, dtype=np.float64)
    pos = np.where(r > 0.0, r, 0.0)
    neg = np.where(r < 0.0, -r, 0.0)
    a_pos, a_neg = pos.sum(axis=1), neg.sum(axis=1)
    h = r.shape[1]
    s_pos = np.full(r.shape, 1.0 / h)
    s_neg = np.full(r.shape, 1.0 / h)
    ok_pos, ok_neg = a_pos > 0.0, a_neg > 0.0
    s_pos[ok_pos] = pos[ok_pos] / a_pos[ok_pos, None]
    s_neg[ok_neg] = neg[ok_neg] / a_neg[ok_neg, None]
    return {"mass_positive": a_pos, "mass_negative": a_neg,
            "shape_positive": s_pos, "shape_negative": s_neg,
            "defined_positive": ok_pos, "defined_negative": ok_neg}


def _w1(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``sum_{h<H-1} |F_u(h) - F_v(h)|`` per row: the exact W1 on ``{0..H-1}``."""
    return np.abs(np.cumsum(u, axis=1)[:, :-1] - np.cumsum(v, axis=1)[:, :-1]).sum(axis=1)


def recompute_metrics(y: np.ndarray, hp: np.ndarray, mp: np.ndarray,
                      mask: np.ndarray, thresholds: Mapping[str, Any],
                      high_mass: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """Every reported number, recomputed from the four arrays alone."""
    y = np.asarray(y, dtype=np.float64)
    hp = np.asarray(hp, dtype=np.float64)
    mp = np.asarray(mp, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)

    q05, q95 = float(thresholds["q05"]), float(thresholds["q95"])
    spread = float(thresholds["day_spread_p90"])

    lower = (y <= q05) & mask
    upper = (y >= q95) & mask
    tail = lower | upper
    normal = mask & ~tail
    negative = (y < 0.0) & mask
    spread_days = (y.max(axis=1) - y.min(axis=1)) >= spread
    high_spread = np.broadcast_to(spread_days[:, None], y.shape) & mask

    mae_m, mae_h = _mae(mp, y, mask), _mae(hp, y, mask)
    normal_m, normal_h = _mae(mp, y, normal), _mae(hp, y, normal)

    realised = _simplex((y - hp) * mask)
    valid_days = mask.any(axis=1)

    out: Dict[str, Any] = {
        "n_days": int(y.shape[0]),
        "n_entries_evaluated": int(mask.sum()),
        "n_days_any_valid": int(valid_days.sum()),
        "overall_mae": mae_m,
        "host_mae": mae_h,
        "mae_gain_abs": (None if (mae_m is None or mae_h is None) else mae_h - mae_m),
        "mae_gain_pct": (None if not mae_h else 100.0 * (mae_h - mae_m) / mae_h),
        "tail_mae": _mae(mp, y, tail),
        "tail_mae_host": _mae(hp, y, tail),
        "tail_n_entries": int(tail.sum()),
        "upper_tail_mae": _mae(mp, y, upper),
        "upper_tail_mae_host": _mae(hp, y, upper),
        "lower_tail_mae": _mae(mp, y, lower),
        "lower_tail_mae_host": _mae(hp, y, lower),
        "normal_mae": normal_m,
        "normal_mae_host": normal_h,
        "normal_relative_harm_pct": (None if not normal_h
                                     else 100.0 * (normal_m - normal_h) / normal_h),
        "negative_price_mae": _mae(mp, y, negative),
        "negative_price_mae_host": _mae(hp, y, negative),
        "negative_price_n_days": int(negative.any(axis=1).sum()),
        "high_spread_day_mae": _mae(mp, y, high_spread),
        "high_spread_n_days": int(spread_days.sum()),
    }
    out["_realised"] = realised
    out["_valid_days"] = valid_days

    if high_mass is not None:
        label = f"q{int(round(float(high_mass['quantile']) * 100))}"
        for sign in ("positive", "negative"):
            realised_mass = realised[f"mass_{sign}"]
            threshold = float(high_mass.get(f"{label}_{sign}", 0.0))
            picked = valid_days & (realised_mass > 0.0) & (realised_mass >= threshold)
            out[f"high_mass_{sign}_n_days"] = int(picked.sum())
            out[f"_picked_{sign}"] = picked
    return out


def _branch_numbers(raw: Mapping[str, Any], recomputed: Mapping[str, Any]
                    ) -> Dict[str, Any]:
    """Direct-branch numbers: the identity, the simplex, the mass and Shape errors.

    The two identities are the method's own algebra, so they are checked before
    any comparison against a recorded value: an artifact whose branch arrays do
    not rebuild its correction cannot be evidence about the branches, whatever
    its metrics say.
    """
    s_pos = np.asarray(raw["shape_positive"], dtype=np.float64)
    s_neg = np.asarray(raw["shape_negative"], dtype=np.float64)
    a_pos = np.asarray(raw["mass_positive"], dtype=np.float64).reshape(-1)
    a_neg = np.asarray(raw["mass_negative"], dtype=np.float64).reshape(-1)
    correction = np.asarray(raw["correction"], dtype=np.float64)
    hp = np.asarray(raw["host_prediction"], dtype=np.float64)
    mp = np.asarray(raw["prediction"], dtype=np.float64)
    y = np.asarray(raw["target"], dtype=np.float64)
    mask = np.asarray(raw["valid_mask"]).astype(bool)

    rebuilt = a_pos[:, None] * s_pos - a_neg[:, None] * s_neg
    recon = float(np.max(np.abs(rebuilt - correction))) if correction.size else 0.0

    scale = float(np.max(np.abs(correction))) if correction.size else 0.0
    tolerance = 1e-6 + 1e-5 * max(1.0, scale)

    alpha = raw["meta"]["alpha"]
    alpha_err = float(np.max(np.abs(hp + float(alpha) * correction - mp))) \
        if mp.size else 0.0
    pred_tol = 1e-6 + 1e-5 * max(1.0, float(np.max(np.abs(mp))) if mp.size else 0.0)

    row_sums_pos = s_pos.sum(axis=1)
    row_sums_neg = s_neg.sum(axis=1)
    simplex_err = max(
        float(np.max(np.abs(row_sums_pos - 1.0))) if row_sums_pos.size else 0.0,
        float(np.max(np.abs(row_sums_neg - 1.0))) if row_sums_neg.size else 0.0)
    # The Shape rows are a ``softmax`` evaluated in float32 and cast to float64
    # for persistence, so a row sums to 1 only to *float32* precision: summing H
    # nonnegative terms each carrying relative error ~eps32 leaves an absolute
    # residual of order H*eps32.  The tolerance is derived from that arithmetic
    # and from the stored horizon rather than fixed, because the core is required
    # to work at any H.  The first version of this check used 1e-9 -- a float64
    # figure -- and therefore failed all 360 artifacts, including every fit whose
    # shapes are exactly the simplex the softmax produced; a bound nothing can
    # satisfy is a miscalibrated instrument, not a strict one.  What it still
    # catches is the failure that matters: an unnormalised or mis-scaled Shape is
    # wrong by O(1), not by 1e-7.
    simplex_tol = float(max(1e-9, max(s_pos.shape[1] if s_pos.ndim == 2 else 0,
                                      s_neg.shape[1] if s_neg.ndim == 2 else 0))
                        * np.finfo(np.float32).eps)
    nonneg = bool(s_pos.min(initial=0.0) >= -1e-12
                  and s_neg.min(initial=0.0) >= -1e-12
                  and a_pos.min(initial=0.0) >= -1e-12
                  and a_neg.min(initial=0.0) >= -1e-12)

    realised = recomputed["_realised"]
    valid_days = recomputed["_valid_days"]
    mass_pos_err = np.abs(a_pos - realised["mass_positive"])[valid_days]
    mass_neg_err = np.abs(a_neg - realised["mass_negative"])[valid_days]

    def _shape(sign: str) -> Dict[str, Any]:
        real, pred = realised[f"shape_{sign}"], (s_pos if sign == "positive" else s_neg)
        ok = realised[f"defined_{sign}"] & np.isfinite(
            a_pos if sign == "positive" else a_neg)
        if not ok.any():
            return {"w1": None, "n": 0}
        return {"w1": float(_w1(real[ok], pred[ok]).mean()), "n": int(ok.sum())}

    shape_pos, shape_neg = _shape("positive"), _shape("negative")
    out: Dict[str, Any] = {
        "branch_reconstruction_max_abs_err": recon,
        "branch_reconstruction_tolerance": tolerance,
        "branch_reconstruction_ok": bool(recon <= tolerance),
        "alpha_consistency_max_abs_err": alpha_err,
        "alpha_consistency_tolerance": pred_tol,
        "alpha_consistency_ok": bool(alpha_err <= pred_tol),
        "shape_simplex_max_abs_err": simplex_err,
        "shape_simplex_tolerance": simplex_tol,
        "shape_simplex_ok": bool(simplex_err <= simplex_tol),
        "nonnegative_masses_and_shapes": nonneg,
        "mass_positive_l1": (float(mass_pos_err.mean())
                             if mass_pos_err.size else None),
        "mass_negative_l1": (float(mass_neg_err.mean())
                             if mass_neg_err.size else None),
        "shape_w1_positive": shape_pos["w1"],
        "shape_w1_negative": shape_neg["w1"],
        "shape_days_defined_positive": shape_pos["n"],
        "shape_days_defined_negative": shape_neg["n"],
    }
    for sign in ("positive", "negative"):
        picked = recomputed.get(f"_picked_{sign}")
        if picked is None:
            continue
        realised_mass = realised[f"mass_{sign}"]
        pred_mass = a_pos if sign == "positive" else a_neg
        out[f"high_mass_{sign}_l1"] = (float(
            np.abs(pred_mass[picked] - realised_mass[picked]).mean())
            if picked.any() else None)
    return out


# --------------------------------------------------------------------------
# Loading what the run produced
# --------------------------------------------------------------------------
def _grid_rows(status_path: Path) -> List[Dict[str, str]]:
    rows = A.read_csv(status_path)
    if not rows:
        raise HarnessError(
            f"the component grid has not been run: {status_path} is missing or "
            "empty.  There is nothing to verify, and an audit that reports "
            "'no failures' over an empty grid would be the most misleading "
            "artifact in the package.")
    return rows


def _recorded_switches(row: Mapping[str, Any]) -> Optional[Dict[str, bool]]:
    """The switch vector a status row records, or ``None`` if it records none.

    ``None`` is a finding rather than an absence to be papered over: a row that
    does not say which vector produced it cannot be checked against the artifact,
    and falling back to the registry by name would make the check pass on the
    strength of a label.
    """
    value = row.get("switches")
    if isinstance(value, Mapping) and value:
        return {str(k): bool(v) for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        if isinstance(parsed, Mapping) and parsed:
            return {str(k): bool(v) for k, v in parsed.items()}
    return None


def _load_raw(row: Mapping[str, Any], audit: Audit) -> Optional[Dict[str, Any]]:
    """Read one raw artifact, refusing a path outside the evidence root."""
    path = Path(str(row.get("raw_path")))
    if not path.is_absolute():
        path = C.REPO_ROOT / path
    try:
        confined = RE.confine(path, "raw artifact read")
    except Exception as exc:  # noqa: BLE001 - a stray path is a finding
        audit.check("artifacts", f"confined::{row.get('cell')}::{row.get('config')}"
                    f"::s{row.get('seed')}", False, repr(exc))
        return None
    if not confined.exists():
        audit.check("artifacts", f"present::{row.get('cell')}::{row.get('config')}"
                    f"::s{row.get('seed')}", False, {"path": str(confined)})
        return None
    raw = RE.read_raw_artifact(confined)
    return raw


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------
def _frozen_record(root: Path) -> Dict[str, Any]:
    """The frozen-method record, or ``{}`` when nothing was frozen.

    Its label, route and switch vector are the run's own statement of *which*
    recipe produced the numbers the verdict rests on.  Reading them is not the
    same as trusting them: :func:`audit_gates` checks the label against the
    vector derived from the screen, and the artifact identity checks in
    :func:`audit_fits` check the vector against every persisted fit.
    """
    path = root / "04_frozen_method" / "FROZEN_METHOD_CONFIG.json"
    if not path.exists():
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return body if isinstance(body, Mapping) else {}


def frozen_rows(root: Path, audit: Audit) -> List[Dict[str, str]]:
    """The frozen method's own fits, when it was fitted rather than reused.

    The screen and the frozen run are different status tables, and the reported
    comparison is built from the *frozen* one.  An audit that read only the
    screen would recompute 300 fits and then adjudicate a configuration whose
    predictions it never opened -- and, because the frozen label is not a
    registered name, it would fail to find them and report an empty panel.  So
    both tables are audited, and the frozen rows are added under their own label.
    """
    record = _frozen_record(root)
    route = str(record.get("route") or "")
    if route != "FITTED_DERIVED_VECTOR":
        audit.check("frozen", "frozen_rows_not_applicable", True,
                    {"route": route or "none"})
        return []
    path = root / "04_frozen_method" / "FROZEN_STATUS.csv"
    rows = A.read_csv(path)
    audit.check("frozen", "frozen_status_table_present", bool(rows),
                {"path": str(path), "n": len(rows)})
    return rows


def audit_coverage(rows: Sequence[Mapping[str, Any]], audit: Audit) -> Dict[str, Any]:
    """The panel is exactly the registered 20 x 5 x 3, with no survivors-only gap."""
    ok = [r for r in rows if str(r.get("status", "")).startswith("OK")]
    failed = [r for r in rows if not str(r.get("status", "")).startswith("OK")]
    cells = sorted({str(r.get("cell")) for r in ok})
    expected = sorted(f"{m}::{h}" for m, h in C.CELLS)
    configs = sorted({str(r.get("config")) for r in ok})
    seeds = sorted({int(float(r["seed"])) for r in ok})

    audit.check("coverage", "no_failed_fits", not failed,
                {"n_failed": len(failed),
                 "failures": [{"cell": r.get("cell"), "config": r.get("config"),
                               "seed": r.get("seed"), "why": r.get("failure")}
                              for r in failed[:20]]})
    audit.check("coverage", "twenty_cells", cells == expected,
                {"n": len(cells), "missing": sorted(set(expected) - set(cells))})
    audit.check("coverage", "five_configs", configs == sorted(C.CONFIG_VARIANTS),
                {"configs": configs})
    audit.check("coverage", "three_registered_seeds",
                seeds == sorted(int(s) for s in C.TRAINING["seeds"]), {"seeds": seeds})
    complete = len(ok) == len(C.CELLS) * len(C.CONFIG_VARIANTS) * len(C.TRAINING["seeds"])
    audit.check("coverage", "three_hundred_fits", complete, {"n": len(ok)})
    per_cell = {c: sum(1 for r in ok if str(r.get("cell")) == c) for c in cells}
    ragged = {c: n for c, n in per_cell.items()
              if n != len(C.CONFIG_VARIANTS) * len(C.TRAINING["seeds"])}
    audit.check("coverage", "every_cell_fully_crossed", not ragged, {"ragged": ragged})
    return {"n_ok": len(ok), "n_failed": len(failed), "cells": cells,
            "configs": configs, "seeds": seeds}


def audit_fits(rows: Sequence[Mapping[str, Any]], audit: Audit,
               thresholds_by_market: Mapping[str, Mapping[str, Any]],
               high_mass: Mapping[str, Mapping[str, Any]],
               host_recheck: bool = True) -> Dict[str, Any]:
    """Recompute every fit and compare it with what was recorded.

    Returns the recomputed records -- the verifier's own cell table, built from
    the raw arrays -- which the decision and gate checks are then fed.  The
    recorded numbers never reach those checks except as the objects of a
    comparison, which is the whole design.
    """
    recomputed: List[Dict[str, Any]] = []
    worst: Dict[str, Tuple[float, str]] = {}
    n_algebra_bad = 0
    n_metric_bad = 0
    sealed_roles: List[str] = []
    knn_on: List[str] = []

    for row in rows:
        if not str(row.get("status", "")).startswith("OK"):
            continue
        tag = f"{row.get('cell')}::{row.get('config')}::s{row.get('seed')}"
        raw = _load_raw(row, audit)
        if raw is None:
            continue
        meta = raw["meta"]

        # Identity: the artifact must be the fit the table says it is.  The
        # switch vector is compared against the *table's* own record of it, not
        # against a registry lookup by name: the frozen smallest method is
        # generally not a registered configuration, so a lookup would either
        # raise or, worse, substitute a different vector and still pass.
        expected_switches = _recorded_switches(row)
        identity_ok = (str(meta.get("config")) == str(row.get("config"))
                       and int(meta.get("seed")) == int(float(row["seed"]))
                       and str(meta.get("partition")) == C.ROLE_DEV_EVAL
                       and expected_switches is not None
                       and dict(meta.get("switches") or {}) == expected_switches)
        audit.check("artifacts", f"identity::{tag}", identity_ok,
                    {"config": meta.get("config"), "seed": meta.get("seed"),
                     "partition": meta.get("partition")})
        audit.check("artifacts", f"hash::{tag}",
                    str(raw["raw_sha256"]) == str(row.get("raw_sha256")),
                    {"on_disk": raw["raw_sha256"], "indexed": row.get("raw_sha256")})

        roles = set(str(r) for r in (meta.get("identity") or {}).get("role", []))
        if roles - {C.ROLE_DEV_EVAL}:
            sealed_roles.append(f"{tag}:{sorted(roles)}")
        if meta.get("knn_enabled") is not False:
            knn_on.append(tag)

        y = np.asarray(raw["target"], dtype=np.float64)
        hp = np.asarray(raw["host_prediction"], dtype=np.float64)
        mp = np.asarray(raw["prediction"], dtype=np.float64)
        mask = np.asarray(raw["valid_mask"]).astype(bool)
        market = str(row.get("market"))

        numbers = recompute_metrics(y, hp, mp, mask, thresholds_by_market[market],
                                    high_mass.get(str(row.get("cell"))))
        branch = _branch_numbers(raw, numbers)

        rec: Dict[str, Any] = {
            "market": market, "host": row.get("host"), "cell": row.get("cell"),
            "config": row.get("config"), "seed": int(float(row["seed"])),
            "status": "OK", "raw_sha256": raw["raw_sha256"],
            "switches": row.get("switches"),
        }
        for key in ("alpha", "amplitude_scale", "n_parameters", "final_epochs",
                    "n_folds", "fit_seconds", "inference_seconds"):
            rec[key] = A._number(meta.get(key))
        for key, value in numbers.items():
            if not key.startswith("_"):
                rec[key] = value
        for key, value in branch.items():
            if key.endswith("_ok"):
                continue
            rec[key] = value
        recomputed.append(rec)

        # -- algebra first: a broken identity makes every metric moot
        for name in ("branch_reconstruction_ok", "alpha_consistency_ok",
                     "shape_simplex_ok", "nonnegative_masses_and_shapes"):
            if not branch[name]:
                n_algebra_bad += 1
                audit.check("algebra", f"{name}::{tag}", False,
                            {"reconstruction": branch["branch_reconstruction_max_abs_err"],
                             "alpha": branch["alpha_consistency_max_abs_err"],
                             "simplex": branch["shape_simplex_max_abs_err"]})

        # -- then the recorded numbers against the recomputed ones
        compared = 0
        mismatched: List[str] = []
        for field, mine in rec.items():
            if field not in rec or field in ("market", "host", "cell", "config",
                                             "seed", "status", "raw_sha256",
                                             "switches"):
                continue
            if field not in row:
                continue
            theirs = row.get(field)
            if theirs in (None, "") and mine is None:
                continue
            compared += 1
            if not _close(mine, theirs):
                mismatched.append(f"{field}: recomputed {mine} vs recorded {theirs}")
            else:
                delta = abs(float(mine) - float(theirs)) if isinstance(mine, (int, float)) \
                    else 0.0
                prev = worst.get(field)
                if prev is None or delta > prev[0]:
                    worst[field] = (delta, tag)
        if mismatched:
            n_metric_bad += 1
            audit.check("metrics", f"recomputed_matches_recorded::{tag}", False,
                        {"n_compared": compared, "mismatches": mismatched[:12],
                         "n_mismatched": len(mismatched)})

    audit.check("algebra", "every_identity_holds", n_algebra_bad == 0,
                {"n_violations": n_algebra_bad})
    audit.check("metrics", "every_reported_number_recomputes", n_metric_bad == 0,
                {"n_fits_compared": len(recomputed), "n_fits_with_a_mismatch": n_metric_bad,
                 "worst_absolute_difference_per_field":
                     {k: {"abs": v[0], "at": v[1]} for k, v in sorted(worst.items())}})
    audit.check("sealed", "no_sealed_role_in_any_artifact", not sealed_roles,
                {"offenders": sealed_roles[:10]})
    audit.check("sealed", "knn_disabled_everywhere", not knn_on,
                {"offenders": knn_on[:10]})
    # Every row that claims a status of OK must have been recomputed.  Stated as
    # an identity against the rows handed in rather than against the screen's
    # arithmetic, because the frozen method adds a sixth configuration and a
    # check hard-coded to 300 would quietly stop meaning anything the moment the
    # frozen run was included -- or, worse, would be "fixed" by not including it.
    claimed = sum(1 for r in rows if str(r.get("status", "")).startswith("OK"))
    audit.check("artifacts", "n_fits_recomputed", len(recomputed) == claimed,
                {"n_recomputed": len(recomputed), "n_claimed_ok": claimed,
                 "n_screen": len(C.CELLS) * len(C.CONFIG_VARIANTS)
                 * len(C.TRAINING["seeds"])})
    return {"records": recomputed, "n_algebra_bad": n_algebra_bad,
            "n_metric_bad": n_metric_bad,
            "worst": {k: {"abs": v[0], "at": v[1]} for k, v in sorted(worst.items())},
            "host_recheck_requested": bool(host_recheck)}


def audit_host_untouched(rows: Sequence[Mapping[str, Any]], audit: Audit) -> Dict[str, Any]:
    """The Host prediction in every artifact is the frozen Host's own output.

    Without this check a retrained Host would be invisible: its predictions would
    look like predictions, the repair would fit around them, and every metric
    would be internally consistent.  Reloading the frozen panel is the only way
    to tell the two apart, so it is done even though it is the expensive part of
    this audit.
    """
    from . import runner

    checked, offenders = 0, []
    per_cell: Dict[str, Dict[str, Any]] = {}
    for market, host in C.CELLS:
        cell_rows = [r for r in rows if str(r.get("cell")) == f"{market}::{host}"
                     and str(r.get("status", "")).startswith("OK")]
        if not cell_rows:
            offenders.append(f"{market}::{host}: no evaluated fit to check")
            continue
        _contract, panel, dataset, read_audit = runner.load_cell(market, host)
        if int(read_audit.protected_final_reads) != 0:
            offenders.append(f"{market}::{host}: protected_final_reads="
                             f"{read_audit.protected_final_reads}")
        for row in cell_rows:
            raw = _load_raw(row, audit)
            if raw is None:
                continue
            meta = raw["meta"]
            positions = np.asarray(meta["row_positions"], dtype=np.int64)
            # The frozen panel stores (N, H, 1); the artifact stores (N, H).
            # Flattening both sides to (N, -1) compares the values without
            # requiring the two writers to agree on a trailing axis.
            frozen = np.asarray(panel.host_pred, dtype=np.float64)[positions]
            frozen_target = np.asarray(panel.y_true, dtype=np.float64)[positions]
            got_host = np.asarray(raw["host_prediction"], dtype=np.float64)
            got_target = np.asarray(raw["target"], dtype=np.float64)
            frozen = frozen.reshape(frozen.shape[0], -1)
            frozen_target = frozen_target.reshape(frozen_target.shape[0], -1)
            got_host = got_host.reshape(got_host.shape[0], -1)
            got_target = got_target.reshape(got_target.shape[0], -1)
            same_host = bool(np.array_equal(got_host, frozen))
            same_target = bool(np.array_equal(got_target, frozen_target))
            checked += 1
            if not (same_host and same_target):
                offenders.append(
                    f"{market}::{host}::{row.get('config')}::s{row.get('seed')}: "
                    f"host_max|diff|={float(np.max(np.abs(got_host - frozen))):.3e} "
                    f"target_max|diff|={float(np.max(np.abs(got_target - frozen_target))):.3e}")
        per_cell[f"{market}::{host}"] = {
            "n_checked": len(cell_rows),
            "protected_final_reads": int(read_audit.protected_final_reads),
            "target_day_price_input_reads":
                int(read_audit.target_day_price_input_reads),
            "realized_future_input_reads":
                int(read_audit.realized_future_input_reads),
        }

    audit.check("host", "frozen_host_predictions_unmodified", not offenders,
                {"n_artifacts_checked": checked, "offenders": offenders[:10]})
    audit.check("sealed", "no_protected_final_read_anywhere",
                all(v["protected_final_reads"] == 0 for v in per_cell.values()),
                {"per_cell": per_cell})
    return {"n_checked": checked, "offenders": offenders, "per_cell": per_cell}


def audit_decisions(records: Sequence[Mapping[str, Any]], audit: Audit,
                    root: Path) -> Dict[str, Any]:
    """Re-derive the component decisions from the recomputed numbers."""
    cells = A.cell_table(records)
    decisions = A.component_decisions(records, cells)
    vector = A.frozen_vector(decisions)

    on_disk_path = root / "COMPONENT_DECISIONS.csv"
    on_disk = A.read_csv(on_disk_path)
    audit.check("decisions", "decision_table_present", bool(on_disk),
                {"path": str(on_disk_path)})
    by_name = {str(r.get("component")): r for r in on_disk}
    mismatches: List[str] = []
    for decision in decisions:
        mine = by_name.get(str(decision["component"]))
        if mine is None:
            mismatches.append(f"{decision['component']}: absent from the table")
            continue
        if str(mine.get("decision")) != str(decision["decision"]):
            mismatches.append(
                f"{decision['component']}: derived {decision['decision']} vs "
                f"recorded {mine.get('decision')} (reason on disk: "
                f"{mine.get('reason')}; derived: {decision['reason']})")
        for field in ("n_cells_primary_improved", "condition_1_majority_cells",
                      "condition_2_tail_or_relevant_improves",
                      "condition_3_overall_non_worse",
                      "condition_4_normal_harm_bounded",
                      "condition_5_not_one_cell_or_seed"):
            if field not in mine:
                continue
            derived = decision.get(field)
            recorded = mine.get(field)
            if isinstance(derived, bool) or isinstance(recorded, bool):
                same = _bool(derived) is not None and _bool(derived) == _bool(recorded)
            else:
                same = _close(recorded, derived)
            if not same:
                mismatches.append(
                    f"{decision['component']}.{field}: derived "
                    f"{derived} vs recorded {recorded}")
    audit.check("decisions", "derived_decisions_match_the_table", not mismatches,
                {"n_components": len(decisions), "mismatches": mismatches})
    audit.check("decisions", "decision_covers_every_registered_switch",
                sorted(by_name) == sorted(C.SWITCH_NAMES),
                {"on_disk": sorted(by_name), "registered": sorted(C.SWITCH_NAMES)})
    return {"cells": cells, "decisions": decisions, "vector": vector}


def _independent_join(strict: Sequence[Mapping[str, str]]
                      ) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int]:
    """The best setting-matched strict offline comparator per cell, re-selected.

    The join inside :mod:`aggregate` is one of the things under audit, so it
    cannot also be the thing that checks itself.  This repeats the selection
    from the frozen table with its own grouping and its own arithmetic, and
    counts the rows the rule excluded.  That count is what makes the exclusion
    observable: a rank that reports "no blocked row entered" over a table whose
    blocked rows were never seen would say the same thing.

    Every exclusion the rule makes is counted, and there are two kinds.  A row
    whose result is not numeric is a blocker.  A row whose setting is not ours --
    the ``OFFLINE_STATIC_CONTROL`` row, and the Host it belongs to -- is excluded
    for a different reason: it is a valid measurement of something else.  Both
    must be visible in the count, or the rank cannot be distinguished from one
    that silently ranked them.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_excluded = 0
    for row in strict:
        if row.get("method") == "Host":
            continue
        key = (str(row.get("market")), str(row.get("host")))
        value = A._number(row.get("Overall_MAE"))
        if not str(row.get("result_status", "")).startswith("NUMERIC") \
                or value is None or str(row.get("setting")) != C.SETTING_MATCHED_STRICT:
            n_excluded += 1
            continue
        entry = best.setdefault(key, {"method": row.get("method"),
                                      "Overall_MAE": value,
                                      "status": row.get("result_status"),
                                      "n_candidates": 0})
        entry["n_candidates"] += 1
        if value < entry["Overall_MAE"]:
            entry.update({"method": row.get("method"), "Overall_MAE": value,
                          "status": row.get("result_status")})
    return best, n_excluded


def audit_gates(cells: Sequence[Mapping[str, Any]],
                decisions: Sequence[Mapping[str, Any]],
                vector: Mapping[str, Any], audit: Audit, root: Path) -> Dict[str, Any]:
    """Re-derive the six development gates and the one legal verdict token.

    The verdict file is read *after* the derivation, only to compare.  Nothing
    from it is an input, which is the property that makes this an audit rather
    than a recital.
    """
    strict = A.read_csv(C.STRICT_OFFLINE_TABLE)
    online = A.read_csv(C.ONLINE_SUPPLEMENTARY_TABLE)
    audit.check("comparison", "frozen_strict_table_present", bool(strict),
                {"path": str(C.STRICT_OFFLINE_TABLE)})

    # Which configuration was actually run is a *claim* the frozen record makes,
    # so it is read and then checked rather than reconstructed.  Deriving it here
    # from the decision table alone silently produced the placeholder ``FROZEN``
    # for a derived vector, and ``baseline_comparison`` then found no rows for it
    # and returned an empty table -- an audit that reports zero cells compared
    # against zero cells, which reads as "nothing to report" rather than as the
    # label mismatch it is.
    frozen_record = _frozen_record(root)
    label = str(frozen_record.get("frozen_label") or "")
    registered = vector["matches_registered_config"]
    if registered is not None:
        label_ok = label == registered
    else:
        label_ok = bool(label) and label not in C.CONFIG_VARIANTS
    audit.check("frozen", "frozen_label_matches_the_derived_vector", label_ok,
                {"frozen_label": label, "registered_match": registered,
                 "route": frozen_record.get("route")})
    audit.check("frozen", "the_frozen_record_carries_the_derived_switches",
                dict(frozen_record.get("switches") or {}) == dict(vector["switches"]),
                {"recorded": frozen_record.get("switches"),
                 "derived": dict(vector["switches"])})

    config = label or (registered or "FROZEN")
    comparison = A.baseline_comparison(cells, config, strict)
    audit.check("comparison", "twenty_cell_comparison", len(comparison) == 20,
                {"n": len(comparison)})
    hosts = A._host_rows(strict)
    joined = [r for r in comparison if (r["market"], r["host"]) in hosts]
    audit.check("comparison", "every_cell_joined_to_its_host_row",
                len(joined) == 20, {"n_joined": len(joined)})
    no_comparator = [r["coordinate_id"] for r in comparison
                     if not r.get("strict_comparator_count")]
    audit.check("comparison", "every_cell_has_a_setting_matched_comparator",
                not no_comparator, {"cells_without": no_comparator})
    # The control setting is reported beside the rank and must never be inside
    # it.  Both columns come from the same selection pass, so a rank that
    # admitted the control would fill them with the same method and read as
    # orderly -- the two agreeing is exactly what a bug here looks like.
    ranked_control = [r["coordinate_id"] for r in comparison
                      if r.get("control_method")
                      and r.get("best_strict_comparator") == r.get("control_method")]
    audit.check("comparison", "the_control_is_reported_but_never_ranked",
                not ranked_control, {"cells": ranked_control})

    # -- the join, recomputed independently ---------------------------------
    mine, n_excluded = _independent_join(strict)
    audit.check("comparison", "blocked_rows_were_actually_excluded",
                n_excluded > 0, {"n_rows_excluded_from_the_rank": n_excluded})
    join_mismatches = []
    for row in comparison:
        key = (row["market"], row["host"])
        theirs = mine.get(key)
        if theirs is None:
            join_mismatches.append(f"{row['coordinate_id']}: no comparator found "
                                   "by independent selection")
            continue
        if str(row.get("best_strict_comparator")) != str(theirs["method"]):
            join_mismatches.append(
                f"{row['coordinate_id']}: recorded best {row.get('best_strict_comparator')!r} "
                f"vs recomputed {theirs['method']!r}")
        elif not _close(row.get("best_strict_comparator_Overall_MAE"),
                        theirs["Overall_MAE"]):
            join_mismatches.append(
                f"{row['coordinate_id']}: recorded best MAE "
                f"{row.get('best_strict_comparator_Overall_MAE')} vs recomputed "
                f"{theirs['Overall_MAE']}")
        elif int(row.get("strict_comparator_count") or 0) != theirs["n_candidates"]:
            join_mismatches.append(
                f"{row['coordinate_id']}: recorded candidate count "
                f"{row.get('strict_comparator_count')} vs recomputed "
                f"{theirs['n_candidates']}")
    audit.check("comparison", "baseline_join_recomputes", not join_mismatches,
                {"n_cells": len(comparison), "mismatches": join_mismatches})

    gates = A.development_gates(comparison, cells, config, decisions, vector)
    derived = A.verdict_token(gates)

    verdict_path = root / "VERDICT.json"
    payload = json.loads(verdict_path.read_text(encoding="utf-8")) \
        if verdict_path.exists() else {}
    audit.check("gates", "verdict_file_present", bool(payload), {"path": str(verdict_path)})
    audit.check("gates", "derived_verdict_equals_the_recorded_one",
                str(payload.get("verdict")) == derived,
                {"derived": derived, "recorded": payload.get("verdict"),
                 "gates_derived": {k: v["pass"] for k, v in gates["gates"].items()},
                 "gates_recorded": {k: v.get("pass") for k, v in
                                    (payload.get("gates") or {}).get("gates", {}).items()}})
    audit.check("gates", "one_legal_token",
                derived in (C.VERDICT_CANDIDATE, C.VERDICT_NOT_SUPPORTED),
                {"derived": derived})
    audit.check("gates", "protected_final_not_authorized",
                payload.get("protected_final_authorized") in (False, None),
                {"recorded": payload.get("protected_final_authorized")})

    supplementary = A.supplementary_view(cells, config, online)
    audit.check("comparison", "cosa_stays_out_of_the_strict_rank",
                all(r["pooled_with_strict_rank"] is False for r in supplementary),
                {"n": len(supplementary)})
    return {"comparison": comparison, "gates": gates, "derived": derived,
            "recorded": payload.get("verdict"), "supplementary": supplementary,
            "config": config, "n_excluded_from_rank": n_excluded}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(root: Optional[Path] = None, out_dir: Optional[Path] = None,
        host_recheck: bool = True) -> int:
    """Verify the executed run end to end; returns a process exit code."""
    base = Path(root) if root is not None else RE.evidence_root()
    audit = Audit()

    status_path = base / "02_crossfit" / "GRID_STATUS.csv"
    screen_rows = _grid_rows(status_path)
    coverage = audit_coverage(screen_rows, audit)
    # The screen's own coverage is the 20 x 5 x 3 panel; the frozen method's fits
    # are a sixth configuration and would make ``five_configs`` false, so they are
    # audited as records rather than folded into the screen's coverage.
    rows = screen_rows + frozen_rows(base, audit)

    thresholds_by_market = {
        market: json.loads(C.THRESHOLD_FREEZE.read_text(encoding="utf-8"))
        ["thresholds"][market] for market in C.MARKETS}
    frozen_hm = protocol.load_thresholds()
    audit.check("protocol", "high_mass_thresholds_frozen",
                bool(frozen_hm and len(frozen_hm.get("cells", [])) == 20),
                {"path": str(protocol.high_mass_path()),
                 "n_cells": len((frozen_hm or {}).get("cells", []))})
    high_mass = {str(r["cell"]): dict(r)
                 for r in (frozen_hm or {}).get("cells", [])}

    fits = audit_fits(rows, audit, thresholds_by_market, high_mass,
                      host_recheck=host_recheck)
    host = audit_host_untouched(rows, audit) if host_recheck else {"skipped": True}
    decisions = audit_decisions(fits["records"], audit, base)
    gates = audit_gates(decisions["cells"], decisions["decisions"],
                        decisions["vector"], audit, base)

    payload = {
        "schema": RESULT_AUDIT_SCHEMA,
        "evidence_root": str(base),
        "root_report": (str(base / "FINAL_AUDIT.md") if out_dir is None else None),
        "coverage": coverage,
        "fits": {"n_records": len(fits["records"]),
                 "n_algebra_violations": fits["n_algebra_bad"],
                 "n_metric_mismatches": fits["n_metric_bad"],
                 "worst_absolute_difference_per_field": fits["worst"]},
        "host": host,
        "frozen_config": gates["config"],
        "frozen_vector": decisions["vector"],
        "decisions": decisions["decisions"],
        "gates": gates["gates"],
        "derived_verdict": gates["derived"],
        "recorded_verdict": gates["recorded"],
        "checks": audit.checks,
        "n_checks": len(audit.checks),
        "n_failed": len(audit.failures),
        "result": "PASS" if not audit.failures else "FAIL",
        "method": ("Every number above was recomputed from the persisted raw "
                   "arrays by this module's own arithmetic; the adjudication "
                   "rules are the registered ones and the verdict is derived "
                   "before the recorded file is read."),
    }

    target_dir = RE.confine(Path(out_dir) if out_dir
                            else base / "06_audits", "result audit directory")
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = RE.confine(target_dir / "RESULT_AUDIT.json", "result audit JSON")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    sort_keys=True, default=str), encoding="utf-8")
    write_csv([{"group": c["group"], "check": c["check"], "ok": c["ok"],
                "detail": json.dumps(c["detail"], ensure_ascii=False, default=str)}
               for c in audit.checks],
              RE.confine(target_dir / "RESULT_AUDIT_CHECKS.csv", "result audit checks"),
              ("group", "check", "ok", "detail"))
    report = _render(payload, audit)
    (RE.confine(target_dir / "RESULT_AUDIT.md", "result audit report")).write_text(
        report, encoding="utf-8")

    # The entry protocol names ``FINAL_AUDIT.md`` as a root artifact.  It is the
    # same report as the one under ``06_audits``; the root copy exists so a
    # reader who opens the evidence root sees the audit verdict without having to
    # know the subdirectory convention.  It is skipped when the caller redirected
    # the audit into a scratch directory, where "the root" is not this run's.
    if out_dir is None:
        (RE.confine(base / "FINAL_AUDIT.md", "root audit report")).write_text(
            report, encoding="utf-8")

    print(f"result audit: {payload['result']} "
          f"({len(audit.checks)} checks, {len(audit.failures)} failed)")
    for failure in audit.failures[:20]:
        print(f"  FAIL {failure['group']}/{failure['check']}: "
              f"{json.dumps(failure['detail'], ensure_ascii=False, default=str)[:400]}")
    if audit.failures:
        print(f"\nHCH_SIGNED_MASS_EXECUTION_INVALID: {len(audit.failures)} audit "
              "check(s) failed")
        return 1
    print(f"wrote {json_path}")
    return 0


def _render(payload: Mapping[str, Any], audit: Audit) -> str:
    lines: List[str] = []
    add = lines.append
    add("# Independent result audit")
    add("")
    add(f"- schema: `{payload['schema']}`")
    add(f"- evidence root: `{payload['evidence_root']}`")
    add(f"- checks: {payload['n_checks']} ({payload['n_failed']} failed)")
    add(f"- result: **{payload['result']}**")
    add(f"- derived verdict: `{payload['derived_verdict']}`")
    add(f"- recorded verdict: `{payload['recorded_verdict']}`")
    add("")
    add("## Coverage")
    add("")
    for key, value in payload["coverage"].items():
        add(f"- {key}: `{value}`")
    add("")
    add("## Checks")
    add("")
    for group in sorted({c["group"] for c in audit.checks}):
        add(f"### {group}")
        add("")
        group_checks = [c for c in audit.checks if c["group"] == group]
        add(f"{sum(1 for c in group_checks if c['ok'])}/{len(group_checks)} pass")
        add("")
        for check in group_checks:
            if not check["ok"]:
                add(f"- **FAIL** `{check['check']}` — "
                    f"{json.dumps(check['detail'], ensure_ascii=False, default=str)[:600]}")
        add("")
    add("## Worst recomputation differences")
    add("")
    add("| field | max abs difference | at |")
    add("| --- | --- | --- |")
    for field, worst in sorted(payload["fits"]["worst_absolute_difference_per_field"].items()):
        add(f"| {field} | {worst['abs']:.3e} | {worst['at']} |")
    add("")
    add("## Instrument changes made after the first audit run")
    add("")
    add("Disclosed because they were made *after* seeing this audit fail, which is "
        "the circumstance in which a change to a measuring instrument is most "
        "easily mistaken for a result.  None of them alters a metric, a "
        "component decision or a gate; the first two decide only what this report "
        "can see, and the third was a tolerance error in the report itself.")
    add("")
    add("1. **The Shape-simplex tolerance was wrong, not the method.** The check "
        "required row sums within `1e-9`, a float64 figure, of rows produced by a "
        "float32 `softmax`. It therefore failed all 360 artifacts -- including the "
        "ones whose Shapes are exactly the simplex the model emitted -- and the "
        "observed residuals cluster on float32 epsilon (min 6.0e-8, median "
        "1.6e-7, max 3.8e-7; 0 of 360 above `1e-6`). The tolerance is now derived "
        "as `H * eps32` from the stored horizon, and the check still refuses an "
        "unnormalised Shape, which is wrong by O(1) rather than by 1e-7 (see "
        "`test_the_simplex_check_accepts_a_softmax_and_rejects_a_broken_one`). No "
        "gate reads this flag.")
    add("2. **The frozen method's own fits were not being audited.** For a "
        "derived vector the frozen method is a sixth configuration with its own "
        "status table, and the reported comparison is built from *that* table. "
        "The audit previously recomputed the 300 screen fits only, so it "
        "adjudicated a configuration whose predictions it had never opened. It "
        "now audits both tables (300 + 60 = 360 artifacts), which strengthens the "
        "audit rather than relaxing it.")
    add("3. **The frozen label was reconstructed instead of read.** The verifier "
        "derived the configuration name from the decision table alone, which "
        "yields the placeholder `FROZEN` for a derived vector; the comparison then "
        "found no rows and returned an empty table -- an audit reporting zero "
        "cells compared against zero cells, which reads as 'nothing to report' "
        "rather than as the label mismatch it was. The label is now read from the "
        "frozen record and *checked* against the derived vector.")
    add("")
    return "\n".join(lines) + "\n"
