"""Runtime for the HCH v4.4 repairability-spectrum diagnostic (TRAIN+VAL only).

This stage trains nothing and takes no optimizer step.  It reuses the closed
stages' code read-only:

* ``recovery_common``     -- access guard + the registered TRAIN/VAL frames;
* ``recovery_train``      -- ``build_cell_data`` (the identical eligible row sets
                             the 60 frozen runs were trained and scored on);
* ``pipeline``            -- the shadow-OOF history support and ``Row``/``batch``;
* ``core.geometry``       -- the single mathematical authority for b/P/N/B/S+/S-.

The 60 frozen O1 runs are addressed by ``run_dir`` across three evidence roots
and are **never retrained, copied or mutated**.

Everything the oracle analysis needs is recomputed here from those frozen bytes:

* the true residual geometry of every eligible TRAIN/VAL day;
* the selected O1 EMA checkpoint's predicted coordinates and correction
  (``no_grad`` inference -- not an optimizer step);
* the one-coordinate oracle swaps O_b/O_B/O_S and the interactions;
* the exact nonnegative MAE-optimal ray scale alpha*.

Conventions registered by this module (also written to
``LEGAL_DESCRIPTOR_SCHEMA.json`` / ``RESULTS.md``):

* residual, coordinates and corrections are all in **raw price units** --
  ``core.model``'s heads emit raw units and ``CoordinateScales`` only normalises
  the *loss terms* (``src/core/losses.py:144-147``);
* the day MAE is the unmasked mean over the 24 registered hours, which is
  exactly the statistic ``recovery_train._stats`` recorded as
  ``selected_val_mae_ema`` -- reconciliation of the two is a hard check in
  ``repair_r0``;
* every eligible day in this panel has ``hour_valid`` fully true, so the
  per-day valid-hour count ``h`` equals ``HORIZON`` everywhere; the decoder
  below still carries ``h`` explicitly so the identity stays exact if that ever
  changes, and ``repair_r0`` records the count of days with ``h < HORIZON``.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

O1_STAGE = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917"
O1_IMPL = O1_STAGE / "implementation"
FULL8_IMPL = REPO / "experiments/current/hch_v44_o1_full8_completion_20260918/implementation"
REC_IMPL = REPO / "experiments/current/hch_v44_optimization_recovery_20260917/implementation"
V2_IMPL = REPO / "experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/implementation"

for _p in (str(O1_IMPL), str(FULL8_IMPL), str(REC_IMPL), str(V2_IMPL), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import recovery_common as RC  # noqa: E402  (closed stage, read-only)

RC.install_access_guard()

import pipeline as PL  # noqa: E402  (closed stage, read-only)
import recovery_train as RT  # noqa: E402  (closed stage, read-only)

U = RC.U

EVID = REPO / "experiments/evidence/hch_v44_repairability_spectrum_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
FULL8_EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
DOM20_EVID = REPO / "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918"

PROTOCOL_ID = "HCH_V44_REPAIRABILITY_SPECTRUM_20260918"
RECIPE_ID = "O1_AUX_WARMUP_400_THEN_MAE"
SWITCH_STEP = 400
HORIZON = int(U.HORIZON)

MARKETS = list(U.MARKETS)
HOSTS = list(U.HOSTS)
SEEDS = list(U.SEEDS)
DOM_PANEL = [(m, h) for m in MARKETS for h in HOSTS]

#: The eight cells whose runs live in an earlier evidence root.  Identical to the
#: guardrail stage's registry; nothing here is recomputed from results.
REUSED_CELLS = [
    ("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer"),
]
NEW_CELLS = [c for c in DOM_PANEL if c not in REUSED_CELLS]
REUSE_ROOTS = {
    "GANSU_DA__PatchTST": O1_EVID, "SHANDONG_DA__iTransformer": O1_EVID,
    "SHAANXI_DA__TimeMixer": O1_EVID, "NINGXIA_DA__iTransformer": O1_EVID,
    "GANSU_DA__LSTM": FULL8_EVID, "SHANDONG_DA__PatchTST": FULL8_EVID,
    "SHAANXI_DA__PatchTST": FULL8_EVID, "QINGHAI_DA__TimeMixer": FULL8_EVID,
}

RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")

#: The fixed seven low-gain cells named in PROTOCOL.md Sec. 6.  Never changed.
LOW_GAIN_CELLS = [
    ("GANSU_DA", "TimeMixer"), ("GANSU_DA", "iTransformer"),
    ("SHANDONG_DA", "TimeMixer"), ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "iTransformer"),
    ("SHAANXI_DA", "LSTM"),
]

#: HEAD_FAMILIES maps the taxonomy's four named families to the DELTA and the
#: competitors each must beat by 0.5pp (PROTOCOL.md Sec. 7).
HEAD_FAMILIES = ("distance", "shape", "balanced_mass", "level")


# ------------------------------------------------------------------ paths
def cell_key(market: str, host: str) -> str:
    return U.cell_key(market, host)


def run_dir(market: str, host: str, seed: int) -> Path:
    key = cell_key(market, host)
    root = REUSE_ROOTS.get(key, DOM20_EVID)
    return root / "o1_runs" / key / f"seed{seed}"


# ------------------------------------------------------------------ io helpers
def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_dump(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8"
    )


def _csv_cell(value) -> str:
    """RFC4180 quoting, so a list/dict field can never corrupt a row."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if any(ch in text for ch in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_csv(path: Path, header, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(str(h) for h in header)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(h)) for h in header))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_split(line: str) -> list[str]:
    cells, cur, quoted = [], "", False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == "," and not quoted:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    cells.append(cur)
    return cells


def load_csv_rows(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").rstrip("\n").splitlines()
    if not lines:
        return []
    head = _csv_split(lines[0])
    return [dict(zip(head, _csv_split(line))) for line in lines[1:]]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write_table_parquet(path: Path, rows: list[dict], columns=None) -> None:
    """Rectangular parquet writer; ``None`` values become nulls, lists stay lists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    columns = list(columns) if columns is not None else list(rows[0])
    table = pa.table({c: [row.get(c) for row in rows] for c in columns})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_table_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def median(values):
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    return float(st.median(vals)) if vals else None


def quantile(values, q: float):
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    return float(np.quantile(vals, q)) if vals else None


def fnum(value):
    """A float that survives a CSV round trip; None stays empty."""
    if value is None:
        return None
    v = float(value)
    return None if v != v else v


# ------------------------------------------------------------------ geometry
def geometry_np(residual: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """True v4.4 geometry through the single authority (``core.geometry``)."""
    import torch
    from core.geometry import residual_geometry

    with torch.no_grad():
        r = torch.as_tensor(np.asarray(residual, dtype=np.float64), dtype=torch.float32)
        v = None if valid is None else torch.as_tensor(np.asarray(valid), dtype=torch.bool)
        g = residual_geometry(r, v)
        return {
            "b": g.b.numpy().astype(np.float64),
            "P": g.P.numpy().astype(np.float64),
            "N": g.N.numpy().astype(np.float64),
            "B": g.B.numpy().astype(np.float64),
            "m1": g.m1.numpy().astype(np.float64),
            "eta": g.eta.numpy().astype(np.float64),
            "s_plus": g.s_plus.numpy().astype(np.float64),
            "s_minus": g.s_minus.numpy().astype(np.float64),
            "horizon": g.horizon.numpy().astype(np.float64),
            "reconstruct": g.reconstruct().numpy().astype(np.float64),
        }


def decode_np(b_hat, B_hat, s_plus, s_minus, h):
    """Exact decoder with an explicit per-day valid-hour count ``h`` [B].

    Reduces to ``core.geometry.decode_prediction`` byte-for-byte when every
    ``h`` equals ``HORIZON`` (which is the case on this panel; the count of
    exceptions is recorded in the evidence).
    """
    b_hat = np.asarray(b_hat, dtype=np.float64).reshape(-1)
    B_hat = np.asarray(B_hat, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    s_plus = np.asarray(s_plus, dtype=np.float64)
    s_minus = np.asarray(s_minus, dtype=np.float64)
    a_plus = B_hat + np.maximum(h * b_hat, 0.0)
    a_minus = B_hat + np.maximum(-h * b_hat, 0.0)
    return a_plus[:, None] * s_plus - a_minus[:, None] * s_minus


def mae_of(residual: np.ndarray, correction: np.ndarray) -> np.ndarray:
    """Unmasked per-day MAE over the HORIZON hours -- the frozen statistic."""
    return np.abs(np.asarray(residual, dtype=np.float64) - np.asarray(correction, dtype=np.float64)).mean(axis=1)


# ------------------------------------------------------------------ ray oracle
def ray_alpha_star(residual: np.ndarray, correction: np.ndarray) -> tuple:
    """Exact nonnegative MAE-optimal ray scale, day by day.

    ``mae(alpha) = mean_h |r_h - alpha c_h|`` is convex piecewise linear in
    ``alpha``; on the support ``{h : c_h != 0}`` it equals the weighted L1
    ``sum_h |c_h| * |r_h/c_h - alpha|``, so its minimiser set is exactly the set
    of weighted medians of ``{r_h/c_h}`` with weights ``|c_h|``.  The **lower**
    weighted median therefore solves ``min_{alpha >= 0}`` after the clamp
    ``alpha* = max(0, lower weighted median)``: when the whole median interval is
    negative the constrained optimum is 0, which is what the clamp returns.

    Returns ``(alpha, status, n_effective)`` where ``status`` is
    ``ok | zero_direction``.
    """
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    c = np.asarray(correction, dtype=np.float64).reshape(-1)
    mask = c != 0.0
    n_eff = int(mask.sum())
    if n_eff == 0:
        return 0.0, "zero_direction", 0
    z = r[mask] / c[mask]
    w = np.abs(c[mask])
    order = np.argsort(z, kind="mergesort")
    z, w = z[order], w[order]
    cum = np.cumsum(w)
    half = 0.5 * float(w.sum())
    idx = int(np.searchsorted(cum, half, side="left"))
    idx = min(idx, len(z) - 1)
    alpha = float(z[idx])
    return max(0.0, alpha), "ok", n_eff


def ray_alpha_star_all(residual: np.ndarray, correction: np.ndarray) -> tuple:
    n = residual.shape[0]
    alpha = np.zeros(n, dtype=np.float64)
    n_eff = np.zeros(n, dtype=np.int64)
    status = []
    for i in range(n):
        a, s, k = ray_alpha_star(residual[i], correction[i])
        alpha[i], n_eff[i] = a, k
        status.append(s)
    return alpha, status, n_eff


# ------------------------------------------------------------------ descriptors
def _safe_ratio(num: float, den: float, fallback: float = 0.0) -> float:
    return float(num / den) if den > 0 else float(fallback)


def shape_distribution(traj: np.ndarray) -> np.ndarray:
    """Relative-Shape view: nonnegative mass normalised to sum one."""
    v = np.asarray(traj, dtype=np.float64)
    shifted = v - float(np.min(v))
    total = float(shifted.sum())
    if total <= 0:
        return np.full(v.shape, 1.0 / v.shape[0])
    return shifted / total


def shape_entropy(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def sign_change_count(r: np.ndarray) -> int:
    s = np.sign(np.asarray(r, dtype=np.float64))
    nz = s != 0
    count, prev = 0, 0.0
    for h in range(s.shape[0]):
        if not nz[h]:
            continue
        if prev != 0.0 and s[h] != prev:
            count += 1
        prev = float(s[h])
    return count


def w1_ordered(p: np.ndarray, q: np.ndarray) -> float:
    """The frozen W1 (``core.losses.wasserstein1_ordered``) on two [H] vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]).sum())


def history_descriptor_vector(hist_residuals: np.ndarray, s_B: float) -> dict:
    """The registered W=7 legal-history descriptors of one target day.

    ``hist_residuals`` is [7, H] in chronological order (the registered window
    already delivered it that way).
    """
    g = geometry_np(hist_residuals)
    m1 = g["m1"]
    b = g["b"]
    B = g["B"]
    day_mae = np.abs(hist_residuals).mean(axis=1)
    logB = np.log1p(np.maximum(B, 0.0) / max(float(s_B), 1e-12))
    qL = np.array([_safe_ratio(HORIZON * abs(b[i]), m1[i]) for i in range(len(b))])
    qB = np.array([_safe_ratio(2.0 * B[i], m1[i]) for i in range(len(b))])
    sc = np.array([sign_change_count(hist_residuals[i]) for i in range(len(b))], dtype=np.float64)

    weights = g["P"]
    if float(weights.sum()) > 0:
        bary_pos = (weights[:, None] * g["s_plus"]).sum(axis=0) / float(weights.sum())
        coh_pos = float(np.median([w1_ordered(g["s_plus"][i], bary_pos) for i in range(len(b))]))
    else:
        coh_pos = 0.0
    weights_n = g["N"]
    if float(weights_n.sum()) > 0:
        bary_neg = (weights_n[:, None] * g["s_minus"]).sum(axis=0) / float(weights_n.sum())
        coh_neg = float(np.median([w1_ordered(g["s_minus"][i], bary_neg) for i in range(len(b))]))
    else:
        coh_neg = 0.0

    idx = np.arange(len(b), dtype=np.float64)
    return {
        "hs_mae_med": float(np.median(day_mae)),
        "hs_mae_mad": float(np.median(np.abs(day_mae - np.median(day_mae)))),
        "hs_absb_med": float(np.median(np.abs(b))),
        "hs_absb_mad": float(np.median(np.abs(np.abs(b) - np.median(np.abs(b))))),
        "hs_logB_med": float(np.median(logB)),
        "hs_logB_mad": float(np.median(np.abs(logB - np.median(logB)))),
        "hs_qL_med": float(np.median(qL)),
        "hs_qB_med": float(np.median(qB)),
        "hs_trend_b": float(np.polyfit(idx, b, 1)[0]),
        "hs_trend_logB": float(np.polyfit(idx, logB, 1)[0]),
        "hs_signchange_med": float(np.median(sc)),
        "hs_coh_pos": coh_pos,
        "hs_coh_neg": coh_neg,
        "hs_mae_iqr": float(np.quantile(day_mae, 0.75) - np.quantile(day_mae, 0.25)),
        "hs_b_iqr": float(np.quantile(b, 0.75) - np.quantile(b, 0.25)),
        "hs_B_iqr": float(np.quantile(B, 0.75) - np.quantile(B, 0.25)),
    }


def host_descriptor_vector(host_traj: np.ndarray, center: float, scale: float) -> dict:
    """The registered current-Host-trajectory descriptors (robust-scaled)."""
    v = np.asarray(host_traj, dtype=np.float64)
    z = (v - float(center)) / max(float(scale), 1e-12)
    d = np.diff(z)
    p = shape_distribution(v)
    hour_max = int(np.argmax(v))
    hour_min = int(np.argmin(v))
    return {
        "hd_mean": float(z.mean()),
        "hd_std": float(z.std(ddof=0)),
        "hd_iqr": float(np.quantile(z, 0.75) - np.quantile(z, 0.25)),
        "hd_spread": float(z.max() - z.min()),
        "hd_tv": float(np.abs(d).sum()),
        "hd_max_ramp": float(np.abs(d).max()) if d.size else 0.0,
        "hd_shape_entropy": shape_entropy(p),
        "hd_shape_concentration": float(p.max()),
        "hd_hour_max_sin": float(math.sin(2 * math.pi * hour_max / HORIZON)),
        "hd_hour_max_cos": float(math.cos(2 * math.pi * hour_max / HORIZON)),
        "hd_hour_min_sin": float(math.sin(2 * math.pi * hour_min / HORIZON)),
        "hd_hour_min_cos": float(math.cos(2 * math.pi * hour_min / HORIZON)),
    }


#: The frozen descriptor list.  Order is the CSV column order and the probe's
#: column order.  Written verbatim into ``LEGAL_DESCRIPTOR_SCHEMA.json`` BEFORE
#: any association is computed, and never extended afterwards.
HOST_DESCRIPTOR_KEYS = (
    "hd_mean", "hd_std", "hd_iqr", "hd_spread", "hd_tv", "hd_max_ramp",
    "hd_shape_entropy", "hd_shape_concentration",
    "hd_hour_max_sin", "hd_hour_max_cos", "hd_hour_min_sin", "hd_hour_min_cos",
)
HISTORY_DESCRIPTOR_KEYS = (
    "hs_mae_med", "hs_mae_mad", "hs_absb_med", "hs_absb_mad",
    "hs_logB_med", "hs_logB_mad", "hs_qL_med", "hs_qB_med",
    "hs_trend_b", "hs_trend_logB", "hs_signchange_med", "hs_coh_pos", "hs_coh_neg",
    "hs_mae_iqr", "hs_b_iqr", "hs_B_iqr",
)
DESCRIPTOR_KEYS = HOST_DESCRIPTOR_KEYS + HISTORY_DESCRIPTOR_KEYS


# ------------------------------------------------------------------ statistics
def spearman(x, y) -> float | None:
    """Spearman rho with average ranks for ties; None when undefined."""
    a = [float(v) for v in x if v is not None and float(v) == float(v)]
    b = [float(v) for v in y if v is not None and float(v) == float(v)]
    if len(a) != len(b) or len(a) < 3:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    ma, mb = np.mean(ra), np.mean(rb)
    da, db = ra - ma, rb - mb
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    if den <= 0:
        return None
    return float((da * db).sum() / den)


def _rankdata(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def sign_of(value) -> int:
    v = float(value)
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


# ------------------------------------------------------------------ reuse pins
def reuse_provenance() -> dict:
    """Pin all 60 frozen runs by hash.  Nothing is retrained, copied or moved."""
    cells, missing = {}, []
    code_hashes, tree_digests = set(), set()
    for market, host in DOM_PANEL:
        key = cell_key(market, host)
        runs = {}
        for seed in SEEDS:
            d = run_dir(market, host, seed)
            have = {f for f in RUN_FIELDS if (d / f).is_file()}
            if len(have) != len(RUN_FIELDS):
                missing.append(f"{key}__seed{seed}: missing {sorted(set(RUN_FIELDS) - have)}")
                continue
            fr = load_json(d / "freeze.json")
            met = fr.get("selected_metrics", {})
            ckpt = d / "selected_ema.pt"
            observed = sha256_file(ckpt)
            code_hashes.add(fr.get("probe_code_hash"))
            tree_digests.add(json.dumps(fr.get("source_tree_digest"), sort_keys=True))
            runs[str(seed)] = {
                "root": str(d.parent.parent.parent.relative_to(REPO)).replace("\\", "/"),
                "freeze_sha256": sha256_file(d / "freeze.json"),
                "curve_sha256": sha256_file(d / "training_curve.json"),
                "checkpoint_sha256": observed,
                "checkpoint_sha256_matches_own_record": (
                    observed == str(fr.get("selected_ema_checkpoint_sha256", "")).upper()),
                "selected_ema_parameter_hash": fr.get("selected_ema_parameter_hash"),
                "probe_code_hash": fr.get("probe_code_hash"),
                "history_support_hash": fr.get("history_support_hash"),
                "selected_val_mae_ema": fr.get("selected_val_mae_ema"),
                "selected_val_host_mae": met.get("val_host_mae"),
                "selected_val_gain_vs_host_pct": met.get("val_gain_vs_host_pct"),
                "selected_val_correction_ratio": met.get("val_correction_ratio"),
                "selected_step": fr.get("selected_step"),
                "scales_fingerprint": (fr.get("scales") or {}).get("fingerprint"),
                "scales_s_r": (fr.get("scales") or {}).get("s_r"),
                "scales_s_b": (fr.get("scales") or {}).get("s_b"),
                "scales_s_B": (fr.get("scales") or {}).get("s_B"),
                "test_target_read_count": fr.get("test_target_read_count"),
            }
        cells[key] = {"market": market, "host": host, "reused": (market, host) in REUSED_CELLS,
                      "runs": runs}
    return {
        "schema": "hch_v44_repairability_spectrum_reuse_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "role": "read-only reuse of frozen O1 runs; zero new fits, zero optimizer steps",
        "n_cells": len(cells),
        "n_runs": sum(len(c["runs"]) for c in cells.values()),
        "missing": missing,
        "all_present": not missing and len(cells) == len(DOM_PANEL)
        and all(len(c["runs"]) == len(SEEDS) for c in cells.values()),
        "reused_cells": [cell_key(m, h) for m, h in REUSED_CELLS],
        "distinct_probe_code_hash": sorted(h for h in code_hashes if h),
        "one_probe_code_hash": len(code_hashes) == 1,
        "distinct_source_tree_digest": sorted(tree_digests),
        "cells": cells,
        "roots": {
            "hch_v44_objective_alignment_probe_20260917": str(O1_EVID.relative_to(REPO)).replace("\\", "/"),
            "hch_v44_o1_full8_completion_20260918": str(FULL8_EVID.relative_to(REPO)).replace("\\", "/"),
            "hch_v44_o1_domestic20_guardrail_20260918": str(DOM20_EVID.relative_to(REPO)).replace("\\", "/"),
        },
    }


def run_snapshot() -> dict:
    """Byte snapshot of all 60 runs, taken before and after the analysis."""
    snap = {}
    for market, host in DOM_PANEL:
        for seed in SEEDS:
            d = run_dir(market, host, seed)
            for f in RUN_FIELDS:
                p = d / f
                if p.is_file():
                    snap[f"{cell_key(market, host)}__seed{seed}__{f}"] = sha256_file(p)
    return snap


__all__ = [name for name in dir() if not name.startswith("_")]
