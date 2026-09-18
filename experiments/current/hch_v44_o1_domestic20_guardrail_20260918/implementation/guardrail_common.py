"""Runtime for the O1 domestic-20 + international-guardrail stage (TRAIN+VAL only).

This stage adds **no new data path and no new trainer**.  It imports the closed
O1 probe stage's ``probe_common`` / ``probe_train`` read-only and calls
``probe_train.train_o1`` unchanged, so the 36 new fits are produced by the
identical code bytes that produced the 24 reused runs.  That is why every one of
the 60 runs must record the same ``probe_code_hash``.

Everything here is bookkeeping: the registered coordinate universe, the reuse
provenance pins, the median-MAE-first aggregation and the D0-D4 arithmetic.

The O1 stage directory is deliberately never edited: its ``probe_code_hash`` is a
digest of that directory, so writing into it would invalidate every frozen run.
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

U = PC.U
RC = PC.RC

EVID = REPO / "experiments/evidence/hch_v44_o1_domestic20_guardrail_20260918"
O1_EVID = REPO / "experiments/evidence/hch_v44_objective_alignment_probe_20260917"
FULL8_EVID = REPO / "experiments/evidence/hch_v44_o1_full8_completion_20260918"
TRANSFER_EVID = REPO / "experiments/evidence/hch_frozen_method_baseline_transfer_20260911"

PROTOCOL_ID = "HCH_V44_O1_DOMESTIC20_GUARDRAIL_20260918"
RECIPE_ID = "O1_AUX_WARMUP_400_THEN_MAE"
SWITCH_STEP = PC.SWITCH_STEP
VARIANT = RC.VARIANT

MARKETS = list(U.MARKETS)
HOSTS = list(U.HOSTS)
SEEDS = list(U.SEEDS)

#: The registered 20-cell domestic universe, in the canonical ``MARKETS x HOSTS``
#: order the V2 substrate itself uses.
DOM_PANEL = [(m, h) for m in MARKETS for h in HOSTS]

#: The eight cells whose 24 O1 runs are reused by hash and never retrained.
REUSED_CELLS = [
    ("GANSU_DA", "PatchTST"), ("GANSU_DA", "LSTM"),
    ("SHANDONG_DA", "PatchTST"), ("SHANDONG_DA", "iTransformer"),
    ("SHAANXI_DA", "TimeMixer"), ("SHAANXI_DA", "PatchTST"),
    ("NINGXIA_DA", "iTransformer"), ("QINGHAI_DA", "TimeMixer"),
]

#: Derived, never hand-written, so the two sets cannot drift apart.
NEW_CELLS = [c for c in DOM_PANEL if c not in REUSED_CELLS]

#: The two evidence roots the reused runs live in.  A reused cell's runs are
#: addressed in exactly one of them.
REUSE_ROOTS = {
    "GANSU_DA__PatchTST": O1_EVID, "SHANDONG_DA__iTransformer": O1_EVID,
    "SHAANXI_DA__TimeMixer": O1_EVID, "NINGXIA_DA__iTransformer": O1_EVID,
    "GANSU_DA__LSTM": FULL8_EVID, "SHANDONG_DA__PatchTST": FULL8_EVID,
    "SHAANXI_DA__PatchTST": FULL8_EVID, "QINGHAI_DA__TimeMixer": FULL8_EVID,
}

RUN_FIELDS = ("freeze.json", "training_curve.json", "selected_ema.pt")

#: The registered international guardrail panel.  Phase I is legal only when
#: D0-D4 all pass, and only these four cells may be fitted.
INTL_CELLS = [
    ("LAGO_DE", "PatchTST"), ("LAGO_DE", "TimeMixer"),
    ("LAGO_PJM", "PatchTST"), ("LAGO_PJM", "TimeMixer"),
]


# ------------------------------------------------------------------ paths
def cell_key(market: str, host: str) -> str:
    return U.cell_key(market, host)


def new_run_dir(market: str, host: str, seed: int) -> Path:
    return EVID / "o1_runs" / cell_key(market, host) / f"seed{seed}"


def reused_run_dir(market: str, host: str, seed: int) -> Path:
    return REUSE_ROOTS[cell_key(market, host)] / "o1_runs" / cell_key(market, host) / f"seed{seed}"


def run_dir(market: str, host: str, seed: int) -> Path:
    return (new_run_dir(market, host, seed) if (market, host) in NEW_CELLS
            else reused_run_dir(market, host, seed))


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


def load_csv_rows(path: Path) -> list[dict]:
    """Read back a CSV this stage wrote, honouring RFC4180 quoting."""
    lines = Path(path).read_text(encoding="utf-8").rstrip("\n").splitlines()
    if not lines:
        return []
    head = _csv_split(lines[0])
    return [dict(zip(head, _csv_split(line))) for line in lines[1:]]


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def median(values):
    vals = [float(v) for v in values]
    return float(st.median(vals)) if vals else None


def quantile(values, q: float):
    import numpy as np

    vals = [float(v) for v in values]
    return float(np.quantile(vals, q)) if vals else None


# ------------------------------------------------------------------ reuse provenance
def reuse_provenance() -> dict:
    """Pin all 24 reused O1 runs by hash before a single new fit is executed.

    Three independent pins, so a stale or edited reuse cannot pass:

    * the run's own ``freeze.json`` re-hashes its checkpoint and the result must
      equal the sha256 the run itself recorded at fit time;
    * every reused run must report the same ``probe_code_hash``;
    * each reused cell's ``history_support_hash`` must equal the shadow-support
      artifact's own ``support_hash`` on disk.

    Nothing here writes; the pins are re-checked after the fits as well, so a
    retrained reuse would be caught as a changed byte.
    """
    cells, missing = {}, []
    code_hashes, support_hashes = set(), {}
    for market, host in [c for c in DOM_PANEL if c in REUSED_CELLS]:
        key = cell_key(market, host)
        support_js = (U.EVID / "shadow_oof" / f"{key}.json")
        support = load_json(support_js) if support_js.is_file() else {}
        support_hashes[key] = support.get("support_hash")
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
            runs[str(seed)] = {
                "root": str(d.parent.parent.parent.relative_to(REPO)).replace("\\", "/"),
                "freeze_sha256": sha256_file(d / "freeze.json"),
                "curve_sha256": sha256_file(d / "training_curve.json"),
                "checkpoint_sha256": observed,
                "checkpoint_sha256_matches_own_record": (
                    observed == str(fr.get("selected_ema_checkpoint_sha256", "")).upper()),
                "selected_ema_parameter_hash": fr.get("selected_ema_parameter_hash"),
                "probe_code_hash": fr.get("probe_code_hash"),
                "source_tree_digest": fr.get("source_tree_digest"),
                "history_support_hash": fr.get("history_support_hash"),
                "history_support_matches_shadow_artifact": (
                    fr.get("history_support_hash") == support.get("support_hash")),
                "recovery_common_sha256": fr.get("recovery_common_sha256"),
                "recovery_train_sha256": fr.get("recovery_train_sha256"),
                "selected_val_mae_ema": fr.get("selected_val_mae_ema"),
                "selected_step": fr.get("selected_step"),
                "test_target_read_count": fr.get("test_target_read_count"),
            }
        cells[key] = {"market": market, "host": host, "runs": runs}
    return {
        "schema": "hch_v44_o1_domestic20_reuse_provenance.v1",
        "protocol_id": PROTOCOL_ID,
        "role": "reused by hash; never retrained, never copied",
        "n_reused_cells": len(cells),
        "n_reused_runs": sum(len(c["runs"]) for c in cells.values()),
        "missing": missing,
        "all_reused_present": not missing and len(cells) == len(REUSED_CELLS),
        "reused_cells": cells,
        "distinct_probe_code_hash_across_reused": sorted(h for h in code_hashes if h),
        "one_probe_code_hash_across_reused": len(code_hashes) == 1,
        "shadow_support_hash_by_cell": support_hashes,
        "source_roots": {
            "hch_v44_objective_alignment_probe_20260917": str(O1_EVID.relative_to(REPO)).replace("\\", "/"),
            "hch_v44_o1_full8_completion_20260918": str(FULL8_EVID.relative_to(REPO)).replace("\\", "/"),
        },
    }


def reuse_snapshot() -> dict:
    """Byte snapshot of every reused artifact; taken before and after the fits."""
    snap = {}
    for market, host in [c for c in DOM_PANEL if c in REUSED_CELLS]:
        for seed in SEEDS:
            d = reused_run_dir(market, host, seed)
            for f in RUN_FIELDS:
                p = d / f
                if p.is_file():
                    snap[f"{cell_key(market, host)}__seed{seed}__{f}"] = sha256_file(p)
    return snap


def coordinate_table() -> list[dict]:
    """The 60 registered coordinates with their origin, before any fit."""
    rows = []
    for market, host in DOM_PANEL:
        for seed in SEEDS:
            d = run_dir(market, host, seed)
            rows.append({
                "market": market, "host": host, "seed": seed,
                "status": "reused" if (market, host) in REUSED_CELLS else "new",
                "run_dir": str(d.relative_to(REPO)).replace("\\", "/"),
                "complete": all((d / f).is_file() for f in RUN_FIELDS),
            })
    return rows


__all__ = [name for name in dir() if not name.startswith("_")]
