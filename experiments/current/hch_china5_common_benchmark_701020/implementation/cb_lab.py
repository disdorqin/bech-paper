"""P1 — isolate the new common-benchmark result laboratory.

Creates `experiments/lab/common_benchmark_results/` per
`docs/current/HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912.md` and
proves the old `experiments/lab/strict_results/` tree is byte-identical before and
after, so no old low-shot row was relabelled, overwritten or silently superseded.

The new lab starts empty on purpose: during PRETEST no coordinate has a TEST numeric
row, and the migration plan forbids populating the registry from the old lab "merely
to fill a table".  Every registry file is written with its header and zero data rows,
and the builder refuses any row whose `protocol_id` is not `COMMON_BENCHMARK_701020_V1`.

Run:  python .../cb_lab.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

ROOT = CB.ROOT
LAB = ROOT / "experiments/lab/common_benchmark_results"
OLD_LAB = ROOT / "experiments/lab/strict_results"
# the lab's own audit record, relative to LAB; excluded from the lab digest because this
# build writes it after computing that digest
AUDIT_RECORD = "audit/build_state.json"
EV = CB.EV

SCHEMA_VERSION = "COMMON_BENCHMARK_V1_20260912"

# ---------------------------------------------------------------- registry shape
RESULT_KEY = ["protocol_id", "stage_id", "market", "host", "method", "setting", "split",
              "seed_contract", "metric"]
RESULT_REQUIRED_NEW = ["split_manifest_hash", "test_target_hash", "host_train_budget",
                       "host_val_budget", "method_train_semantics", "online_or_offline"]
RESULT_VALUE = ["value", "unit", "n_days", "n_values", "subset_definition",
                "coverage_status", "evidence_root", "evidence_path", "hash_type",
                "artifact_sha256", "gain_vs_host_pct", "generated_utc", "notes"]
RESULTS_LONG_COLUMNS = RESULT_KEY + RESULT_REQUIRED_NEW + RESULT_VALUE

COVERAGE_COLUMNS = ["protocol_id", "stage_id", "market", "host", "method", "setting",
                    "admission_status", "coordinate_status", "blocker_id", "blocker_class",
                    "blocker_reason", "blocker_authority", "has_numeric_object",
                    "has_test_numeric_row", "evidence_root", "notes"]

BLOCKER_COLUMNS = ["blocker_id", "protocol_id", "market", "host", "method", "blocker_class",
                   "blocker_reason", "authority", "authority_path", "inherited", "numeric_values_carried"]

HOST_COLUMNS = ["protocol_id", "stage_id", "market", "dataset_id", "host", "host_recipe_id",
                "source_sha256", "split_manifest_hash", "train_id_sha256", "val_id_sha256",
                "host_train_budget", "host_val_budget", "checkpoint_sha256", "array_sha256",
                "scaler_fit_partition", "selection_partition", "gate_passed", "evidence_path"]

METHOD_COLUMNS = ["method", "setting", "online_or_offline", "fidelity_label",
                  "method_train_semantics", "fitting_partition", "selection_partition",
                  "evaluation_partition", "authority_path"]

STAGE_ADMISSION_COLUMNS = ["stage_id", "protocol_id", "status", "admission_status",
                           "numeric_rows", "test_rows", "evidence_root", "admitted_by",
                           "notes"]

DATASET_COLUMNS = ["protocol_id", "market", "dataset_id", "family", "source_path",
                   "source_sha256", "n_open_days", "day_id_convention",
                   "eligibility_contract_sha256", "split_manifest_sha256",
                   "test_target_blind_id_sha256", "sealed_roles", "sealed_role_partitioned"]

EVIDENCE_INDEX_COLUMNS = ["protocol_id", "stage_id", "market", "host", "method", "setting",
                          "artifact_kind", "evidence_path", "hash_type", "artifact_sha256",
                          "contains_test_values", "generated_utc"]

PAPER_TABLE_COLUMNS = {
    "COMMON_OFFLINE_COMPARISON_WIDE.csv": ["market", "host", "setting", "Host", "MDR",
                                           "delta_Adapter", "PIR", "M0", "blocker_status",
                                           "coverage_status"],
    "COMMON_PREQUENTIAL_COMPARISON_WIDE.csv": ["market", "host", "Host_reference", "COSA",
                                               "M1", "M2", "setting", "coverage_status"],
    "COMMON_PROTOCOL_COVERAGE.csv": ["market", "host", "method", "admission_status",
                                     "coordinate_status", "blocker_id", "protocol_id"],
    "COMMON_METHOD_FIDELITY.csv": ["market", "host", "method", "fidelity_label",
                                   "split_protocol_id", "admission_status"],
}

READMES = {
    "README.md": """# Common-Benchmark Results Laboratory

Protocol: `COMMON_BENCHMARK_701020_V1`
Schema version: `COMMON_BENCHMARK_V1_20260912`
Stage of record: `hch_china5_common_benchmark_701020`

This lab indexes results produced **only** under the chronological 70/10/20 common
benchmark (TRAIN 70% / VAL 10% / TEST 20% of each market's open eligible target-day
sequence). It is the intended source of the future main comparison tables.

## Separation from `../strict_results/`

`../strict_results/` remains the frozen authority for the old
`HARD_FROZEN_LOW_SHOT_TRANSFER` (`~45/5/15/5/30`) results. The two labs answer
different data-budget questions, so the same market/Host/method may legitimately carry
different values in each. Never average, rank together, or silently supersede across
`protocol_id`.

No numeric row from `strict_results` is copied here. Only a stage executed under
`COMMON_BENCHMARK_701020_V1` may supply a numeric row. Inherited blockers may be
re-registered as blockers with provenance to their fidelity authority, carrying **no**
numeric value.

## Row admission

A row whose `protocol_id != COMMON_BENCHMARK_701020_V1` cannot become active here.
Every numeric row must also carry `split_manifest_hash`, `test_target_hash`,
`host_train_budget`, `host_val_budget`, `method_train_semantics` and
`online_or_offline`.

## State at PRETEST

The registry is intentionally empty: PRETEST freezes splits, Hosts, baselines and the
method definitions, but produces **no** TEST numeric row. `test_rows` is a hard zero
until the JOINT_TEST phase is separately authorized.
""",
    "REGISTRY_PROTOCOL.md": """# Common-Benchmark Registry Protocol

1. `protocol_id` is mandatory on every row and must equal `COMMON_BENCHMARK_701020_V1`.
2. `split_manifest_hash` must match the hash recorded in
   `experiments/evidence/hch_china5_common_benchmark_701020_20260912/01_splits/SPLIT_INDEX.json`.
3. `split` is `TRAIN` / `VAL` / `TEST`; main scored rows use `TEST`.
4. `setting` separates data-access regimes. `ONLINE_TTA` and `PREQUENTIAL_SAFETY` rows
   must never be ranked against `OFFLINE_*` rows as if data access were identical.
5. Empty subsets are represented by `coverage_status` (e.g. `NOT_APPLICABLE_N0`) and
   never by a numeric zero.
6. `NORMAL_HARM_VS_HOST` and any harm field must declare `unit` — the registry has
   historically confused absolute units with relative percentages; the unit column is
   mandatory for every harm row.
7. Each row references exactly one evidence root under `experiments/evidence/**`.
8. No row may be written while the TEST seal holds. The builder enforces this by
   refusing any row whose `split == TEST` during PRETEST.
9. The old `../strict_results/` tree is read-only from here.
""",
    "SCHEMA.md": """# Common-Benchmark Registry Schema

Schema version: `COMMON_BENCHMARK_V1_20260912`

Inherits the `strict_results` vocabularies that remain valid (coordinate status,
admission status, method identity, setting, hash semantics, gain convention, metric
vocabulary, empty-subset rule, residual convention) and adds the protocol-identity
fields required by `HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912` §3.

## Coordinate status

`NUMERIC` / `BLOCKED` / `NOT_APPLICABLE_N0` / `METRICS_ONLY` / `SUPERSEDED` /
`HISTORICAL_ONLY` / `PENDING_TEST`.

## Admission status

`COMMON_BENCHMARK_ADMITTED` / `COMMON_BENCHMARK_ADMITTED_WITH_SCOPE_CAVEAT` /
`BLOCKER_ONLY` / `PENDING` / `REJECTED_INVALID`.

## Added fields

- `protocol_id` — mandatory, `COMMON_BENCHMARK_701020_V1`
- `split_manifest_hash` — the frozen split manifest hash of the market
- `test_target_hash` — the target-blind TEST-ID hash of the market
- `host_train_budget` / `host_val_budget` — TRAIN / VAL target-day counts of the Host
- `method_train_semantics` — the partition(s) a method is fitted and selected on
- `online_or_offline` — `ONLINE` / `OFFLINE`

## Hash semantics

Unchanged from `strict_results`: `hash_type` must state what was hashed, and an
`ARRAY_CONTENT_SHA256` is never compared with a compressed container hash.

## Gain convention

`gain_vs_host_pct = 100 * (HostMetric - MethodMetric) / HostMetric`; positive gain is
an improvement over the exact corresponding Host of the same protocol. Host rows carry
gain exactly 0.

## Units

Every harm / normal-region field must declare `unit` as `ABSOLUTE_PRICE` or
`PERCENT_OF_HOST_MAE`. No harm value may be published without its unit.

## Integrity invariants

1. A `BLOCKED` coordinate has no numeric artifact.
2. An `ONLINE_TTA` or `PREQUENTIAL_SAFETY` row never enters an offline ranking.
3. Every `NUMERIC` row has an admitted `COMMON_BENCHMARK_701020_V1` source stage.
4. Every row references exactly one evidence root.
5. No TEST row exists before JOINT_TEST authorization.
6. Inherited blockers carry provenance and no numeric value.
""",
}


def tree_digest(root: Path, exclude_dirs=("__pycache__",),
                exclude_files=()) -> tuple[str, int, list[str]]:
    """sha256 over relative POSIX path + file bytes, in case-sensitive ASCII path order.

    `exclude_files` names paths (relative, POSIX) that are not part of the digest.  The
    lab's own audit record is excluded this way: it is written *after* the build it
    describes, so a digest that included it could never be re-derived from the tree on
    disk without deleting the record.
    """
    root = Path(root)
    skip = {str(f) for f in exclude_files}
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        if rel.as_posix() in skip:
            continue
        files.append(p)
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\x0a")
    return h.hexdigest(), len(files), [p.relative_to(root).as_posix() for p in files]


def _write_csv(path: Path, columns, rows=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=columns)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def assert_common_benchmark_row(row: dict) -> None:
    """Row admission gate — the only way a numeric row may enter this lab."""
    if row.get("protocol_id") != CB.PROTOCOL_ID:
        raise ValueError(f"row rejected: protocol_id {row.get('protocol_id')!r} is not "
                         f"{CB.PROTOCOL_ID}")
    missing = [f for f in RESULT_REQUIRED_NEW if not row.get(f)]
    if missing:
        raise ValueError(f"row rejected: missing mandatory common-benchmark fields {missing}")
    if row.get("split") == "TEST" and row.get("test_phase_authorized") is not True:
        raise PermissionError("row rejected: TEST rows may not be written while the "
                              "PRETEST seal holds")


def scaffold() -> dict:
    for name, body in READMES.items():
        (LAB / name).write_text(body, encoding="utf-8")
    _write_csv(LAB / "registry/RESULTS_LONG.csv", RESULTS_LONG_COLUMNS)
    _write_csv(LAB / "registry/COVERAGE_MATRIX.csv", COVERAGE_COLUMNS)
    _write_csv(LAB / "registry/BLOCKERS.csv", BLOCKER_COLUMNS)
    _write_csv(LAB / "registry/HOSTS.csv", HOST_COLUMNS)
    _write_csv(LAB / "registry/METHODS.csv", METHOD_COLUMNS)
    _write_csv(LAB / "registry/STAGE_ADMISSION.csv", STAGE_ADMISSION_COLUMNS)
    _write_csv(LAB / "registry/DATASETS.csv", DATASET_COLUMNS)
    _write_csv(LAB / "registry/EVIDENCE_INDEX.csv", EVIDENCE_INDEX_COLUMNS)
    for fname, cols in PAPER_TABLE_COLUMNS.items():
        _write_csv(LAB / "paper_tables" / fname, cols)
    (LAB / "paper_tables/README.md").write_text(
        "# Common-Benchmark Paper Tables\n\nGenerated from `../registry/` only. Offline and "
        "online/prequential settings are never merged into one ranking; every harm column "
        "declares its unit. Empty subsets carry a coverage status, never a numeric zero.\n",
        encoding="utf-8")
    (LAB / "audit/README.md").write_text(
        "# Common-Benchmark Lab Audit\n\n`build_state.json` records the registry build and the "
        "PRETEST seal state. `REGISTRY_AUDIT.md` is written by the independent verifier.\n",
        encoding="utf-8")
    (LAB / "snapshots/README.md").write_text(
        "# Common-Benchmark Registry Snapshots\n\nImmutable snapshots of `../registry/`. "
        "Snapshots are append-only; none may be deleted or rewritten.\n", encoding="utf-8")
    (LAB / "implementation/README.md").write_text(
        "# Common-Benchmark Lab Implementation\n\nThe registry builder lives in the stage that "
        "produces the rows:\n\n"
        "`experiments/current/hch_china5_common_benchmark_701020/implementation/cb_lab.py`\n\n"
        "It refuses any row whose `protocol_id` is not `COMMON_BENCHMARK_701020_V1`, and any "
        "`split == TEST` row while the PRETEST seal holds.\n", encoding="utf-8")
    return {"files_written": sorted(p.relative_to(LAB).as_posix()
                                    for p in LAB.rglob("*") if p.is_file())}


def main(argv: list[str]) -> int:
    before_hash, before_n, before_files = tree_digest(OLD_LAB)
    scaffold()
    after_hash, after_n, after_files = tree_digest(OLD_LAB)

    unchanged = (before_hash == after_hash and before_n == after_n
                 and before_files == after_files)
    # the audit record is written into the lab after this digest is taken, so it is named
    # as excluded rather than silently absent
    stage_sha, stage_n, stage_files = tree_digest(LAB, exclude_files=(AUDIT_RECORD,))

    state = {
        "schema": "common_benchmark_lab_build_state.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "stage_id": "hch_china5_common_benchmark_701020",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "old_lab": {
            "path": "experiments/lab/strict_results",
            "digest_before": before_hash, "digest_after": after_hash,
            "n_files_before": before_n, "n_files_after": after_n,
            "file_list_identical": before_files == after_files,
            "unchanged": unchanged,
            "interpretation_tag": "HARD_FROZEN_LOW_SHOT_TRANSFER",
            "modification_authorized": False,
        },
        "new_lab": {
            "path": "experiments/lab/common_benchmark_results",
            "digest": stage_sha, "n_files": stage_n,
            "digest_covers": ("every file under the lab except "
                              f"{AUDIT_RECORD}, which this build writes after taking the "
                              "digest; re-derive by hashing relpath+NUL+sha256(file)+LF "
                              "over that set in ASCII path order"),
            "registry_rows": 0, "test_numeric_rows": 0,
            "state": "SCAFFOLD_ONLY_NO_NUMERIC_ROWS",
        },
        "test_label_read_count": CB.PROCESS_SEAL.test_label_read_count,
        "notes": ("the new lab is created empty; PRETEST produces no TEST numeric row, and "
                  "the old strict-results lab is proven byte-identical before and after"),
    }
    CB.dump_json(EV / "07_audits/OLD_LAB_UNCHANGED.json",
                 {"old_lab_digest_before": before_hash, "old_lab_digest_after": after_hash,
                  "n_files_before": before_n, "n_files_after": after_n,
                  "file_list_identical": before_files == after_files,
                  "unchanged": unchanged})
    CB.dump_json(LAB / "audit/build_state.json", state)
    CB.dump_json(EV / "06_registry_export/LAB_SCAFFOLD.json", state)

    print(f"[P1] old lab digest {before_hash[:16]}… ({before_n} files) "
          f"-> {'UNCHANGED' if unchanged else 'CHANGED'}")
    print(f"[P1] new lab digest {stage_sha[:16]}… ({stage_n} files), "
          f"registry_rows=0, test_numeric_rows=0")
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
