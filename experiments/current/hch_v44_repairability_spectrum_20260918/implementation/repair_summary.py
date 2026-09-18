"""Cell / market / Host-family headroom summaries and the registered taxonomy.

Reads ``PER_DAY_REPAIRABILITY.parquet`` back (no data path, no model).

Aggregation convention, registered here once and used everywhere downstream:

* every per-day quantity is a **percent of that day's own Host MAE**, so a
  percentage is a per-day ratio first and a median second;
* a cell's value is the median over that seed's eligible VAL days, then the
  median over the three registered seeds ("median across seeds first", the
  project's standing primary-statistic rule);
* a market's (or Host family's) value is the median of its cells' values, so
  every cell carries equal weight and no market wins by having longer series.

Taxonomy (design Sec. 7, evaluated in the registered order, first match wins):

    LOW_HEADROOM          best of ray/B/Shape/b  < 1.0
    DISTANCE_LIMITED      ray   >= 2.0 and ray   - max(B, Shape, b) >= 0.5
    SHAPE_LIMITED         Shape >= 2.0 and Shape - max(B, b)        >= 0.5
    BALANCED_MASS_LIMITED B     >= 2.0 and B     - max(Shape, b)    >= 0.5
    LEVEL_LIMITED         b     >= 2.0 and b     - max(B, Shape)    >= 0.5
    MIXED_HEADROOM        otherwise

A cell whose best single-coordinate headroom lands in [1.0, 2.0) is therefore
``MIXED_HEADROOM`` under the registered fallback; ``best_single_coordinate_
headroom_pct`` is carried on every row so that band stays visible without
inventing an extra label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import repair_common as R  # noqa: E402

FAMILY_OF_COLUMN = {
    "ray_pct": "distance", "B_pct": "balanced_mass", "Shape_pct": "shape", "b_pct": "level",
}

#: The summary's own short names -> the R0 atlas columns that carry them.  ``ob``
#: means "O1 with only b replaced by its oracle", so ``imp_ob_pct`` is exactly the
#: design's "b single-coordinate oracle improvement"; likewise ``oB``/``oS``.
ATLAS_COLUMN = {
    "ray_pct": "increment_ray_pct",
    "b_pct": "imp_ob_pct",
    "B_pct": "imp_oB_pct",
    "Shape_pct": "imp_oS_pct",
    "bB_pct": "imp_obB_pct",
    "bS_pct": "imp_obS_pct",
    "BS_pct": "imp_oBS_pct",
}
#: The subset the taxonomy and the market/Host roll-up consume.
HEADROOM_KEYS = ("ray_pct", "b_pct", "B_pct", "Shape_pct", "bB_pct", "bS_pct", "BS_pct")

CELL_HEADER = [
    "market", "host", "cell", "n_val_days", "n_train_days",
    "o1_gain_vs_host_pct", "ray_pct", "b_pct", "B_pct", "Shape_pct",
    "bB_pct", "bS_pct", "BS_pct",
    "best_single_coordinate_headroom_pct", "best_family", "diagnosis",
    "registered_low_gain_cell", "measured_low_gain_cell", "material_headroom",
    "alpha_star_med", "alpha_frac_gt_1", "alpha_frac_under_1", "alpha_frac_zero_direction",
    "o1_correction_ratio_med", "host_mae_med", "o1_mae_med", "ray_remaining_recall_med",
]


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _cell_value(rows: list[dict], key: str, role: str = "VAL") -> float | None:
    by_seed = {}
    for row in rows:
        if row["role"] != role:
            continue
        by_seed.setdefault(int(row["seed"]), []).append(row.get(key))
    per_seed = [R.median(vals) for vals in by_seed.values()]
    return R.median([v for v in per_seed if v is not None])


def _cell_gain(rows: list[dict], role: str = "VAL") -> float | None:
    by_seed = {}
    for row in rows:
        if row["role"] != role:
            continue
        host = float(row["host_mae"])
        gain = 100.0 * (host - float(row["o1_mae"])) / host if host > 0 else None
        by_seed.setdefault(int(row["seed"]), []).append(gain)
    per_seed = [R.median(vals) for vals in by_seed.values()]
    return R.median([v for v in per_seed if v is not None])


def _frac_alpha(rows: list[dict], predicate) -> float | None:
    vals = [bool(predicate(float(r["alpha_star"]))) for r in rows if r["role"] == "VAL"]
    return float(np.mean(vals)) if vals else None


def classify(values: dict) -> str:
    ray, b, B, S = values["ray_pct"], values["b_pct"], values["B_pct"], values["Shape_pct"]
    known = [v for v in (ray, b, B, S) if v is not None]
    if not known:
        return "UNDEFINED"
    best = max(known)
    if best < 1.0:
        return "LOW_HEADROOM"
    if ray is not None and ray >= 2.0 and ray - max(v for v in (b, B, S) if v is not None) >= 0.5:
        return "DISTANCE_LIMITED"
    if S is not None and S >= 2.0 and S - max(v for v in (b, B) if v is not None) >= 0.5:
        return "SHAPE_LIMITED"
    if B is not None and B >= 2.0 and B - max(v for v in (S, b) if v is not None) >= 0.5:
        return "BALANCED_MASS_LIMITED"
    if b is not None and b >= 2.0 and b - max(v for v in (B, S) if v is not None) >= 0.5:
        return "LEVEL_LIMITED"
    return "MIXED_HEADROOM"


def cell_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["market"], row["host"]), []).append(row)
    out = []
    for (market, host), cell in sorted(grouped.items(), key=lambda kv: (R.MARKETS.index(kv[0][0]), R.HOSTS.index(kv[0][1]))):
        values = {k: _cell_value(cell, ATLAS_COLUMN[k]) for k in HEADROOM_KEYS}
        gain = _cell_gain(cell)
        best_family, best = None, None
        for key, fam in FAMILY_OF_COLUMN.items():
            if values[key] is not None and (best is None or values[key] > best):
                best, best_family = values[key], fam
        diagnosis = classify(values)
        val_days = sorted({r["day"] for r in cell if r["role"] == "VAL"})
        train_days = sorted({r["day"] for r in cell if r["role"] == "TRAIN"})
        out.append({
            "market": market, "host": host, "cell": R.cell_key(market, host),
            "n_val_days": len(val_days), "n_train_days": len(train_days),
            "o1_gain_vs_host_pct": gain,
            **values,
            "best_single_coordinate_headroom_pct": best, "best_family": best_family,
            "diagnosis": diagnosis,
            "registered_low_gain_cell": (market, host) in R.LOW_GAIN_CELLS,
            "measured_low_gain_cell": (gain is not None and gain < 1.0),
            "material_headroom": (best is not None and best >= 2.0),
            "alpha_star_med": _cell_value(cell, "alpha_star"),
            "alpha_frac_gt_1": _frac_alpha(cell, lambda a: a > 1.0 + 1e-9),
            "alpha_frac_under_1": _frac_alpha(cell, lambda a: a < 1.0 - 1e-9),
            "alpha_frac_zero_direction": _frac_alpha(cell, lambda a: a == 0.0),
            "o1_correction_ratio_med": _cell_value(cell, "o1_correction_ratio"),
            "host_mae_med": _cell_value(cell, "host_mae"),
            "o1_mae_med": _cell_value(cell, "o1_mae"),
            "ray_remaining_recall_med": _cell_value(cell, "ray_remaining_recall"),
        })
    return out


CALC_COLUMNS = ("o1_gain_vs_host_pct", "ray_pct", "b_pct", "B_pct", "Shape_pct",
                "best_single_coordinate_headroom_pct")


def grouped_summary(cells: list[dict], group_type: str, key_fn) -> list[dict]:
    buckets = {}
    for cell in cells:
        buckets.setdefault(key_fn(cell), []).append(cell)
    out = []
    for name, members in buckets.items():
        agg = {col: R.median([c[col] for c in members]) for col in CALC_COLUMNS}
        best_family = None
        best = None
        for col, fam in FAMILY_OF_COLUMN.items():
            v = agg[col]
            if v is not None and (best is None or v > best):
                best, best_family = v, fam
        diag_counts = {}
        for c in members:
            diag_counts[c["diagnosis"]] = diag_counts.get(c["diagnosis"], 0) + 1
        out.append({
            "group_type": group_type, "group": name, "n_cells": len(members),
            **agg,
            "best_family": best_family,
            "n_material_headroom_cells": sum(1 for c in members if c["material_headroom"]),
            "dominant_diagnosis": max(diag_counts, key=lambda k: (diag_counts[k], k)),
            "diagnosis_counts": diag_counts,
        })
    return out


def run() -> dict:
    rows = read_parquet(R.EVID / "PER_DAY_REPAIRABILITY.parquet")
    cells = cell_rows(rows)
    R.write_csv(R.EVID / "CELL_HEADROOM_SUMMARY.csv", CELL_HEADER, cells)

    market_rows = grouped_summary(cells, "MARKET", lambda c: c["market"])
    host_rows = grouped_summary(cells, "HOST_FAMILY", lambda c: c["host"])
    overall = grouped_summary(cells, "ALL", lambda c: "ALL")
    R.write_csv(R.EVID / "MARKET_HOST_HEADROOM_SUMMARY.csv",
                ["group_type", "group", "n_cells", *CALC_COLUMNS, "best_family",
                 "n_material_headroom_cells", "dominant_diagnosis", "diagnosis_counts"],
                overall + market_rows + host_rows)

    low = [c for c in cells if (c["market"], c["host"]) in R.LOW_GAIN_CELLS]
    material_low = [c for c in low if c["material_headroom"]]
    families = {}
    for c in material_low:
        families[c["best_family"]] = families.get(c["best_family"], 0) + 1
    out = {
        "n_cells": len(cells),
        "diagnosis_counts": {d: sum(1 for c in cells if c["diagnosis"] == d)
                             for d in sorted({c["diagnosis"] for c in cells})},
        "registered_low_gain_cells": [R.cell_key(m, h) for m, h in R.LOW_GAIN_CELLS],
        "low_gain_cells_material_headroom": [c["cell"] for c in material_low],
        "n_low_gain_with_material_headroom": len(material_low),
        "low_gain_dominant_family_counts": families,
        "measured_vs_registered_low_gain_mismatch": [
            c["cell"] for c in cells if c["registered_low_gain_cell"] != c["measured_low_gain_cell"]
        ],
    }
    R.json_dump(R.EVID / "GATE_A_B_EVIDENCE.json", out)
    return out


if __name__ == "__main__":
    print(run())
