"""P0 — freeze the five split manifests before anything is trained.

Writes `01_splits/<MARKET>/split_manifest.json` plus one stage-level index, and runs
the TEST-inaccessibility audit.  The audit is deliberately structural, not a promise:

* the row positions a TRAIN+VAL projected read needs are recomputed and shown to be
  disjoint from every TEST target-day row position, so no parse path can materialise a
  TEST target value;
* the process TEST seal is shown to refuse an attempted reveal;
* the split plan is shown to be chronological and to leave the sealed final role
  untouched.

Run:  python .../cb_p0_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

IMPL = Path(__file__).resolve().parent
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

import cb_contracts as CB                                                    # noqa: E402

OUT = CB.EV / "01_splits"


def _needed_rows(ob: CB.OpenBlock, day_ids) -> set[int]:
    needed: set[int] = set()
    for d in day_ids:
        pos = ob.positions[d]
        needed.update(pos.tolist())
        start = int(pos[0])
        needed.update(range(start - CB.SEQ, start))
    return needed


def audit_test_inaccessibility(market: str) -> dict:
    plan = CB.split_plan(market)
    ob = CB.open_block(market)

    pretest_days = list(plan.train_ids) + list(plan.val_ids)
    materialised = _needed_rows(ob, pretest_days)
    test_target_rows: set[int] = set()
    for d in plan.test_ids:
        test_target_rows.update(ob.positions[d].tolist())

    # the projected reader's keep-set for TRAIN+VAL
    _, audit = CB.load_pretest_windows(market)

    overlap = sorted(materialised & test_target_rows)
    # the seal must refuse while PRETEST holds
    refused = None
    try:
        CB.PROCESS_SEAL.reveal(f"{market}: audit probe")
    except PermissionError as exc:
        refused = str(exc)

    checks = {
        "counts_match_preregistration": plan.counts == CB.EXPECTED_COUNTS[market],
        "no_pretest_row_is_a_test_target_row": len(overlap) == 0,
        "test_ids_disjoint_from_train_and_val": not (set(plan.test_ids)
                                                     & (set(plan.train_ids) | set(plan.val_ids))),
        "test_ids_are_the_chronological_tail": list(plan.day_ids[-len(plan.test_ids):])
                                               == list(plan.test_ids),
        "consecutive_blocks_are_contiguous_in_the_open_sequence": (
            list(plan.train_ids) + list(plan.val_ids) + list(plan.test_ids)
            == list(plan.day_ids)),
        "loader_reported_zero_test_target_reads": int(audit["test_target_values_read"]) == 0,
        "loader_reported_zero_sealed_final_reads": int(audit["sealed_final_target_values_read"]) == 0,
        "test_seal_refuses_before_joint_test": refused is not None,
        "rows_skipped_at_parse_time_cover_every_test_row": (
            audit["rows_skipped_at_parse_time"]
            >= len(test_target_rows) - len(materialised)),
    }
    return {
        "market": market,
        "M": plan.M,
        "counts": {"TRAIN": len(plan.train_ids), "VAL": len(plan.val_ids),
                   "TEST": len(plan.test_ids)},
        "pretest_target_day_rows_materialised": len(
            materialised & set().union(*[set(ob.positions[d].tolist()) for d in pretest_days])),
        "test_target_day_rows": len(test_target_rows),
        "overlap_rows": len(overlap),
        "seal_refusal_message": refused,
        "loader_access_audit": audit,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str]) -> int:
    index, audits = {}, {}
    for market in CB.MARKETS:
        man = CB.split_manifest(market)
        h = CB.manifest_hash(man)
        man["split_manifest_sha256"] = h
        CB.dump_json(OUT / market / "split_manifest.json", man)
        audit = audit_test_inaccessibility(market)
        CB.dump_json(OUT / market / "test_inaccessibility_audit.json", audit)
        index[market] = {
            "split_manifest": f"experiments/evidence/hch_china5_common_benchmark_701020_20260912/01_splits/{market}/split_manifest.json",
            "split_manifest_sha256": h,
            "counts": man["counts"],
            "n_open_days": man["n_open_days"],
            "source_sha256": man["source_sha256"],
            "test_target_blind_id_sha256": man["test_target_blind_id_sha256"],
            "eligibility_contract_sha256": man["eligibility_contract_sha256"],
            "audit_passed": audit["passed"],
        }
        audits[market] = audit["passed"]
        print(f"[P0] {market:12s} M={man['n_open_days']:5d} "
              f"TRAIN/VAL/TEST={man['counts']['TRAIN']}/{man['counts']['VAL']}/{man['counts']['TEST']}"
              f"  sha={h[:16]}…  audit={'PASS' if audit['passed'] else 'FAIL'}")

    CB.dump_json(OUT / "SPLIT_INDEX.json", {
        "schema": "china5_common_benchmark_split_index.v1",
        "protocol_id": CB.PROTOCOL_ID,
        "stage_id": "hch_china5_common_benchmark_701020",
        "generated_before_any_host_training": True,
        "test_label_read_count": CB.PROCESS_SEAL.test_label_read_count,
        "test_seal": CB.PROCESS_SEAL.as_dict(),
        "markets": index,
    })
    n_ok = sum(1 for v in audits.values() if v)
    print(f"\n[P0] {n_ok}/{len(audits)} market audits PASS; "
          f"TEST-label read count = {CB.PROCESS_SEAL.test_label_read_count}")
    return 0 if n_ok == len(audits) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
