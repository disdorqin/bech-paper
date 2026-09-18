"""Runtime for the HCH O1 matched-control closure stage (TRAIN+VAL only).

This stage adds **no new data path, no new architecture and no new trainer
semantics**.  It imports the closed O1 probe stage's ``probe_common`` /
``recovery_common`` / ``recovery_train`` read-only and reuses the frozen v4.4
substrate verbatim; the only scientific delta it introduces is *which* variant
``HCHV44Core`` is instantiated with (and, for the G1-selected branch of Phase B,
the single ``tied_identity`` flag on the existing G3 control).

Everything in this module is bookkeeping:

* the registered 8-cell x 3-seed panel and the two O1/G2 evidence roots;
* hash pins for the 24 reused reference runs;
* the seed-median-first aggregation and the A1/A2/B1/B2 arithmetic;
* the variant registry, including the no-new-parameter
  ``G3_SHARED_DIRECT_CONTEXT_CTRL`` wrapper.

The two closed stage directories are deliberately never edited: every reuse
record pins them by digest.

There is no TEST path anywhere in this stage.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
STAGE = HERE.parents[1]
REPO = HERE.parents[4]

O1_STAGE = REPO / "experiments/current/hch_v44_objective_alignment_probe_20260917"
O1_IMPL = O1_STAGE / "implementation"
FULL8_STAGE = REPO / "experiments/current/hch_v44_o1_full8_completion_20260918"
FULL8_IMPL = FULL8_STAGE / "implementation"

for _p in (str(FULL8_IMPL), str(O1_IMPL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_common as PC  # noqa: E402  (closed stage, read-only)
import probe_train as PT  # noqa: E402  (closed stage, read-only)
import recovery_train as RT  # noqa: E402  (closed stage, read-only)

U = PC.U
RC = PC.RC

EVID = REPO / "experiments/evidence/hch_v44_o1_matched_controls_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
FULL8_EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"

PROTOCOL_ID = "HCH_V44_O1_MATCHED_CONTROLS_20260918"
RECIPE_ID = PC.RECIPE_O1["recipe_id"]
SWITCH_STEP = PC.SWITCH_STEP

#: Variant identifiers.  ``G3_SHARED_DIRECT_CONTEXT_CTRL`` is *not* a src/core
#: variant: it is this stage's name for the existing G3 direct-control model with
#: ``tied_identity`` forced True.  It adds no parameter and no module.
G0 = "G0_GEOM_HOST"
G1 = "G1_GEOM_SHARED_CONTEXT"
G2 = "G2_GEOM_COORD_CONTEXT"
G3 = "G3_DIRECT_CONTEXT_CTRL"
G3_SHARED = "G3_SHARED_DIRECT_CONTEXT_CTRL"

SEEDS = list(U.SEEDS)                       # [7, 17, 37]
MARKETS = list(U.MARKETS)
HOSTS = list(U.HOSTS)

#: PROTOCOL.md section 3, in the order the protocol writes it.
PANEL = [
    ("GANSU_DA", "PatchTST"),
    ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"),
    ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"),
    ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "iTransformer"),
    ("QINGHAI_DA", "TimeMixer"),
]
PANEL_KEYS = [U.cell_key(m, h) for m, h in PANEL]

#: The two evidence roots holding the 24 reused O1/G2 runs.  Every reused cell is
#: addressed in exactly one of them.  This mapping is copied from the closed
#: domestic-20 stage's own ``REUSE_ROOTS``, which is the adjudicated record of
#: where each of these eight cells was fitted under O1.
REUSE_ROOTS = {
    "GANSU_DA__PatchTST": O1_EVID, "SHANDONG_DA__iTransformer": O1_EVID,
    "SHAANXI_DA__TimeMixer": O1_EVID, "NINGXIA_DA__iTransformer": O1_EVID,
    "GANSU_DA__LSTM": FULL8_EVID, "SHANDONG_DA__PatchTST": FULL8_EVID,
    "SHAANXI_DA__PatchTST": FULL8_EVID, "QINGHAI_DA__TimeMixer": FULL8_EVID,
}

RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")

#: The frozen digest the reused O1/G2 runs recorded for the shared substrate.
FROZEN_CORE_DIGEST = "6F18C0E4241A299E32C8D8A617061BAEAA7B8DC98B4EE069DC67199B87531577"
FROZEN_BACKBONES_DIGEST = "79B51A56499860D489C1FE4642F111EBB6BABC0281028F57F683A21484C12E7B"


# ------------------------------------------------------------------ paths
def cell_key(market: str, host: str) -> str:
    return U.cell_key(market, host)


def matched_tree_digest() -> str:
    """Digest of this stage's own source, so each freeze record pins its runner."""
    items = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".md"):
            items.append(f"{p.relative_to(REPO).as_posix()}|{U.sha256(p)}")
    return U.sha256_bytes("\n".join(items).encode())


def new_run_dir(phase: str, market: str, host: str, seed: int) -> Path:
    return EVID / "runs" / phase / cell_key(market, host) / f"seed{seed}"


def reused_run_dir(market: str, host: str, seed: int) -> Path:
    key = cell_key(market, host)
    return REUSE_ROOTS[key] / "o1_runs" / key / f"seed{seed}"


# ------------------------------------------------------------------ io helpers
def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def median(values):
    vals = [float(v) for v in values]
    return float(st.median(vals)) if vals else None


# ------------------------------------------------------------------ reuse pins
def _historical_support_hash(cell: str):
    """Each reused cell's registered shadow-support hash, from the O1 root.

    The O1 reference runs recorded ``history_support_hash``; the shadow-support
    artifacts live in the recovery stage's own evidence root and are the second,
    independent witness that the reused run used the registered history support.
    """
    p = U.EVID / "shadow_oof" / f"{cell}.json"
    return load_json(p).get("support_hash") if p.is_file() else None


def reuse_provenance() -> dict:
    """Pin all 24 reused O1/G2 runs by hash before a single new fit is executed.

    Four independent pins, so a stale, edited or retrained reuse cannot pass:

    * all three run artifacts are present for every registered coordinate;
    * the run's own ``freeze.json`` re-hashes its checkpoint and the result must
      equal the sha256 that run recorded at fit time;
    * all 24 runs must report one and the same ``probe_code_hash`` -- the code
      that produced them is one frozen module tree;
    * each run's ``history_support_hash`` must equal the registered shadow-support
      artifact's own ``support_hash`` on disk.

    Also records the two source digests and the selection each run made.  Nothing
    here writes; the pins are re-checked after the fits, and a byte snapshot is
    taken before and after, so a retrained reuse is caught as a changed byte.
    """
    cells, missing = {}, []
    code_hashes, support_mismatch, digests = set(), [], set()
    for market, host in PANEL:
        key = cell_key(market, host)
        registered_support = _historical_support_hash(key)
        runs = {}
        for seed in SEEDS:
            d = reused_run_dir(market, host, seed)
            have = {f for f in RUN_FIELDS if (d / f).is_file()}
            if len(have) != len(RUN_FIELDS):
                missing.append(f"{key}__seed{seed}: missing {sorted(set(RUN_FIELDS) - have)}")
                continue
            fr = load_json(d / "freeze.json")
            ckpt = d / "selected_ema.pt"
            observed = sha256_file(ckpt)
            code_hashes.add(fr.get("probe_code_hash"))
            digest = fr.get("source_tree_digest") or {}
            digests.add(json.dumps(digest, sort_keys=True))
            support_ok = fr.get("history_support_hash") == registered_support
            if not support_ok:
                support_mismatch.append(f"{key}__seed{seed}")
            runs[str(seed)] = {
                "root": str(d.parent.parent.parent.relative_to(REPO)).replace("\\", "/"),
                "freeze_sha256": sha256_file(d / "freeze.json"),
                "curve_sha256": sha256_file(d / "training_curve.json"),
                "checkpoint_sha256": observed,
                "checkpoint_sha256_matches_own_record": (
                    observed == str(fr.get("selected_ema_checkpoint_sha256", "")).upper()),
                "selected_ema_parameter_hash": fr.get("selected_ema_parameter_hash"),
                "probe_code_hash": fr.get("probe_code_hash"),
                "source_tree_digest": digest,
                "variant": fr.get("variant"),
                "history_support_hash": fr.get("history_support_hash"),
                "history_support_matches_registered": support_ok,
                "recovery_common_sha256": fr.get("recovery_common_sha256"),
                "recovery_train_sha256": fr.get("recovery_train_sha256"),
                "selected_val_mae_ema": fr.get("selected_val_mae_ema"),
                "selected_step": fr.get("selected_step"),
                "selected_check": fr.get("selected_check"),
                "test_target_read_count": fr.get("test_target_read_count"),
                "test_rows_materialised": fr.get("test_rows_materialised"),
                "n_train_rows": fr.get("n_train_rows"),
                "n_val_rows": fr.get("n_val_rows"),
                "train_day_ids_hash": fr.get("train_day_ids_hash"),
                "val_day_ids_hash": fr.get("val_day_ids_hash"),
                "scales": fr.get("scales"),
                "registered_schedule": PC.registered_schedule_of(fr.get("optimization") or {}),
            }
        cells[key] = {"market": market, "host": host,
                      "registered_history_support_hash": registered_support,
                      "runs": runs}
    digest_list = sorted(digests)
    return {
        "schema": "hch_v44_o1_matched_reuse_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "role": "reused by hash; never retrained, never copied, never edited",
        "n_reused_cells": len(cells),
        "n_reused_runs": sum(len(c["runs"]) for c in cells.values()),
        "expected_runs": len(PANEL) * len(SEEDS),
        "missing": missing,
        "all_reused_present": (not missing and len(cells) == len(PANEL)
                               and all(len(c["runs"]) == len(SEEDS) for c in cells.values())),
        "reused_cells": cells,
        "distinct_probe_code_hash_across_reused": sorted(h for h in code_hashes if h),
        "one_probe_code_hash_across_reused": len(code_hashes) == 1,
        "distinct_source_tree_digest_across_reused": digest_list,
        "one_source_tree_digest_across_reused": len(digest_list) == 1,
        "history_support_mismatches": support_mismatch,
        "all_history_support_registered": not support_mismatch,
        "source_roots": {
            "hch_v44_objective_alignment_probe_20260917":
                str(O1_EVID.relative_to(REPO)).replace("\\", "/"),
            "hch_v44_o1_full8_completion_20260918":
                str(FULL8_EVID.relative_to(REPO)).replace("\\", "/"),
        },
    }


def reuse_snapshot() -> dict:
    """Byte snapshot of every reused artifact; taken before and after the fits."""
    snap = {}
    for market, host in PANEL:
        for seed in SEEDS:
            d = reused_run_dir(market, host, seed)
            for f in RUN_FIELDS:
                p = d / f
                if p.is_file():
                    snap[f"{cell_key(market, host)}__seed{seed}__{f}"] = sha256_file(p)
    return snap


# ------------------------------------------------------------------ statistics
def gain_pct(mae_x: float, mae_y: float) -> float:
    """Registered relative gain of method X over method Y, in Y's own units.

    The design authority fixes ``gain(G2,G1) = 100 * (MAE_G1 - MAE_G2) / MAE_G1``
    and ``gain(STRUCT,DIRECT) = 100 * (MAE_DIRECT - MAE_STRUCT) / MAE_DIRECT``;
    both are this function with ``(mae_x, mae_y)`` := (candidate, baseline), the
    denominator being the baseline the candidate is measured against.
    """
    return 100.0 * (float(mae_y) - float(mae_x)) / float(mae_y)


def cell_medians(rows: list[dict]) -> list[dict]:
    """Seed-median-first: median each method's VAL MAE across seeds, *then* gain.

    A median of paired per-seed gains is deliberately never formed.
    """
    out = []
    for market, host in PANEL:
        cell = cell_key(market, host)
        sub = [r for r in rows if r["cell"] == cell]
        if len(sub) != len(SEEDS):
            raise RuntimeError(f"{cell}: {len(sub)} rows, expected {len(SEEDS)}")
        mae_s = median([r["mae_struct"] for r in sub])
        mae_c = median([r["mae_ctrl"] for r in sub])
        host_s = median([r["host_mae_struct"] for r in sub])
        host_c = median([r["host_mae_ctrl"] for r in sub])
        out.append({
            "cell": cell, "market": market, "host": host, "n_seeds": len(sub),
            "mae_struct_median": mae_s, "mae_ctrl_median": mae_c,
            "host_mae_struct_median": host_s, "host_mae_ctrl_median": host_c,
            "gain_pct_from_medians": gain_pct(mae_s, mae_c),
            "per_seed_gain_pct": [round(r["gain_pct"], 6) for r in sub],
            "struct_beats_ctrl": bool(mae_s < mae_c),
            "struct_host_positive": bool(mae_s < host_s),
            "ctrl_host_positive": bool(mae_c < host_c),
            "struct_host_gain_pct": gain_pct(mae_s, host_s),
            "ctrl_host_gain_pct": gain_pct(mae_c, host_c),
            "selected_step_struct_median": median([r["selected_step_struct"] for r in sub]),
            "selected_step_ctrl_median": median([r["selected_step_ctrl"] for r in sub]),
        })
    return out


def _threshold_checks(cells, compliance, *, label_struct, label_ctrl,
                      min_cells, median_threshold, worst_threshold,
                      min_host_positive, gate_prefix):
    """The performance clauses shared verbatim by A1 and B1 (and A2 by symmetry)."""
    gains = [c["gain_pct_from_medians"] for c in cells]
    n_beat = sum(1 for c in cells if c["struct_beats_ctrl"])
    n_host = sum(1 for c in cells if c["struct_host_positive"])
    panel_median = float(st.median(gains))
    return {
        f"{gate_prefix}_cells_{label_struct}_beats_{label_ctrl}_ge_{min_cells}of{len(cells)}":
            n_beat >= min_cells,
        f"{gate_prefix}_panel_median_gain_ge_{median_threshold}pct":
            panel_median >= median_threshold,
        f"{gate_prefix}_worst_cell_ge_{worst_threshold}pct":
            float(min(gains)) >= worst_threshold,
        f"{gate_prefix}_{label_struct}_host_positive_ge_{min_host_positive}of{len(cells)}":
            n_host >= min_host_positive,
        f"{gate_prefix}_no_access_source_numerical_failure": bool(compliance["clean"]),
    }, {
        "cells_beating": n_beat, "cells_required": min_cells,
        "panel_median_gain_pct": panel_median, "median_threshold_pct": median_threshold,
        "worst_cell_gain_pct": float(min(gains)), "worst_threshold_pct": worst_threshold,
        "best_cell_gain_pct": float(max(gains)),
        "struct_host_positive_cells": n_host, "host_positive_required": min_host_positive,
        "gains_by_cell": {c["cell"]: c["gain_pct_from_medians"] for c in cells},
    }


def gate_a(rows: list[dict], cells: list[dict], compliance: dict) -> dict:
    """Design authority sections 7: A1, then A2, then A3.

    ``compliance`` carries the audited access/source/numerical facts; it is
    computed from the freeze records and curves, never asserted by hand.
    """
    a1_checks, a1_detail = _threshold_checks(
        cells, compliance, label_struct="g2", label_ctrl="g1",
        min_cells=6, median_threshold=0.50, worst_threshold=-0.50,
        min_host_positive=7, gate_prefix="a1")
    a1 = all(a1_checks.values()) and bool(compliance["clean"])

    # A2 is evaluated only if A1 fails, but is always recorded for the audit.
    gaps = [abs(c["gain_pct_from_medians"]) for c in cells]          # |G1 vs G2|, G1 units
    g1_gains = [-c["gain_pct_from_medians"] for c in cells]          # G1-vs-G2, G2 units
    within = sum(1 for g in gaps if g <= 0.50)
    a2_checks = {
        "a2_g1_within_0.50pct_of_g2_ge_7of8": within >= 7,
        "a2_panel_median_g1_vs_g2_gain_ge_-0.25pct":
            float(st.median([gain_pct(c["mae_ctrl_median"], c["mae_struct_median"])
                             for c in cells])) >= -0.25,
        "a2_worst_g1_vs_g2_cell_ge_-1.00pct":
            float(min(gain_pct(c["mae_ctrl_median"], c["mae_struct_median"])
                      for c in cells)) >= -1.00,
        "a2_g1_host_positive_ge_7of8":
            sum(1 for c in cells if c["ctrl_host_positive"]) >= 7,
        "a2_no_access_source_numerical_failure": bool(compliance["clean"]),
    }
    a2 = (not a1) and all(a2_checks.values())
    a2_detail = {
        "cells_within_0.50pct": within,
        "panel_median_g1_vs_g2_gain_pct": float(st.median(
            [gain_pct(c["mae_ctrl_median"], c["mae_struct_median"]) for c in cells])),
        "worst_g1_vs_g2_cell_pct": float(min(
            gain_pct(c["mae_ctrl_median"], c["mae_struct_median"]) for c in cells)),
        "best_g1_vs_g2_cell_pct": float(max(
            gain_pct(c["mae_ctrl_median"], c["mae_struct_median"]) for c in cells)),
        "g1_host_positive_cells": sum(1 for c in cells if c["ctrl_host_positive"]),
        "per_cell_abs_gap_pct": {c["cell"]: g for c, g in zip(cells, gaps)},
        "per_cell_g1_vs_g2_gain_pct": {c["cell"]: g for c, g in zip(cells, g1_gains)},
    }
    verdict, selected = "A3_ROUTING_IDENTITY_INCONCLUSIVE", None
    if a1:
        verdict, selected = "A1_COORDINATE_IDENTITY_SUPPORTED", G2
    elif a2:
        verdict, selected = "A2_SHARED_CONTEXT_SUFFICIENT", G1
    return {
        "schema": "hch_v44_o1_matched_phase_a_gate.v1",
        "protocol_id": PROTOCOL_ID,
        "statistic": "3-seed median VAL MAE per method per cell, then gain(candidate,baseline)",
        "a1_coordinate_identity_supported": a1,
        "a1_checks": a1_checks, "a1_detail": a1_detail,
        "a2_shared_context_sufficient": a2,
        "a2_note": ("A2 is evaluated only because A1 failed; its checks are recorded "
                    "either way for audit." if not a1 else
                    "A1 held, so A2 is not the operative clause (recorded for audit)."),
        "a2_checks": a2_checks, "a2_detail": a2_detail,
        "verdict": verdict,
        "selected_structured_variant": selected,
        "n_rows": len(rows), "n_cells": len(cells),
        "compliance": compliance,
    }


def gate_b(rows: list[dict], cells: list[dict], compliance: dict,
           struct_variant: str, direct_variant: str) -> dict:
    """Design authority section 8: B1, else B2."""
    checks, detail = _threshold_checks(
        cells, compliance, label_struct="structured", label_ctrl="direct",
        min_cells=6, median_threshold=0.50, worst_threshold=-0.50,
        min_host_positive=7, gate_prefix="b1")
    b1 = all(checks.values()) and bool(compliance["clean"])
    return {
        "schema": "hch_v44_o1_matched_phase_b_gate.v1",
        "protocol_id": PROTOCOL_ID,
        "struct_variant": struct_variant, "direct_variant": direct_variant,
        "statistic": "3-seed median VAL MAE per method per cell, then "
                     "gain(STRUCT,DIRECT) = 100*(MAE_DIRECT-MAE_STRUCT)/MAE_DIRECT",
        "b1_exact_geometry_supported": b1,
        "b1_checks": checks, "b1_detail": detail,
        "verdict": "B1_EXACT_GEOMETRY_SUPPORTED" if b1 else "B2_EXACT_GEOMETRY_NOT_SUPPORTED",
        "n_rows": len(rows), "n_cells": len(cells),
        "compliance": compliance,
    }


# ------------------------------------------------------------------ variants
def build_model_for_variant(data: dict, device: str, profile, variant: str):
    """``recovery_train.build_model`` with the variant made explicit.

    ``build_model`` hardcodes ``G2_GEOM_COORD_CONTEXT``; this is that function
    with one line parameterised.  ``readout_init`` is fixed at ``None`` by the
    registered O1 recipe, so it is not exposed here.

    ``G3_SHARED_DIRECT_CONTEXT_CTRL`` instantiates the existing
    ``G3_DIRECT_CONTEXT_CTRL`` model and sets ``tied_identity = True``.  That
    attribute is a plain bool read only by ``allocate_coordinates``; it owns no
    parameter, so the model is parameter-for-parameter identical to G3.
    """
    import torch
    from core.model import HCHV44Core

    base = G3 if variant == G3_SHARED else variant
    model = HCHV44Core(base, data["scales"], profile=profile).to(device)
    if variant == G3_SHARED:
        model.tied_identity = True
    return model


def requested_tied_identity(variant: str) -> bool:
    return variant in (G0, G1, G3_SHARED)


__all__ = [name for name in dir() if not name.startswith("_")]
