"""The deliverables the entry protocol names exist, and do not contradict each other.

The protocol names its artifacts by path, and this harness writes two of them in
two places (the package directory, where the protocol asks for them, and the
evidence root, where the run record belongs).  Two copies of one claim is a
drift risk unless something checks them, so this file does: every mirrored pair
must be byte-identical, and the readiness table must agree with the verdict the
audit reports.

These are checks on *artifacts produced by a previous run*.  They skip cleanly
when the artifacts are absent, because a fresh checkout has not run anything
yet -- but they never skip silently when one copy of a pair exists and the other
does not, which is the state that actually matters.
"""
from __future__ import annotations

import csv
import json

import pytest

from experiments.current.hch_signed_mass_method_entry import config as C
from experiments.current.hch_signed_mass_method_entry.contracts import sha256_file

PACKAGE_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]
AUDIT_DIR = C.EVIDENCE_ROOT / "06_audits"

MIRRORED = ("HOST_INPUT_READINESS.csv", "PREEXECUTION_AUDIT.md",
            "PREEXECUTION_READINESS.json")
EVIDENCE_ONLY = ("SMOKE_AUDIT.json", "HOST_INPUT_READINESS_SUMMARY.json")


def _pair(name):
    """The two locations of an artifact, or ``None`` if neither exists."""
    here, there = PACKAGE_DIR / name, AUDIT_DIR / name
    if not here.exists() and not there.exists():
        return None
    return here, there


@pytest.mark.parametrize("name", MIRRORED)
def test_mirrored_deliverables_exist_in_both_places_and_agree(name):
    pair = _pair(name)
    if pair is None:
        pytest.skip(f"{name} has not been produced in this checkout")
    here, there = pair
    assert here.exists(), f"{name} is missing from the package directory"
    assert there.exists(), f"{name} is missing from {AUDIT_DIR}"
    assert sha256_file(here) == sha256_file(there), (
        f"{name} has drifted: the package copy and the evidence copy differ")


@pytest.mark.parametrize("name", EVIDENCE_ONLY)
def test_evidence_only_artifacts_live_in_the_evidence_root(name):
    path = AUDIT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced in this checkout")
    assert path.parent == AUDIT_DIR


def test_readiness_csv_has_one_row_per_cell_and_no_result_column():
    """20 distinct coordinates, and no column that could carry an outcome."""
    path = PACKAGE_DIR / "HOST_INPUT_READINESS.csv"
    if not path.exists():
        pytest.skip("HOST_INPUT_READINESS.csv has not been produced")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {(r["market"], r["host"]) for r in rows} == set(C.CELLS)
    # The read count is zero in every row.  ``contract_sealed_days`` is a
    # different thing -- the number of days the contract *declares* sealed --
    # and must be positive, because a market with nothing sealed would make the
    # zero read vacuous rather than proved.
    assert "n_protected_final_read" in rows[0]
    assert "contract_sealed_days" in rows[0]
    for row in rows:
        assert row["n_protected_final_read"] == "0", row
        assert int(row["contract_sealed_days"]) > 0, row
        # The sealed days are declared, never materialised: the rows the
        # artifact actually carries are the revealed roles alone.
        revealed = sum(int(row[k]) for k in
                       ("n_host_train", "n_host_val", "n_post_train", "n_dev_eval"))
        assert revealed == int(row["n_episodes"]), row
        assert revealed + int(row["contract_sealed_days"]) > int(row["n_episodes"])
    # And nothing that names a scientific outcome may appear at all.
    for key in rows[0]:
        for token in ("mae", "rmse", "gain", "alpha", "benefit", "improvement"):
            assert token not in key.lower(), key
    # A row that is not READY must say why.  A row that is READY must not.
    for row in rows:
        assert row["status"]
        if row["status"] == "READY_FROZEN":
            assert not row["blocker"], row
        else:
            assert row["blocker"], row


def test_readiness_csv_agrees_with_the_verdict():
    """The audit reports the same counts the table does."""
    table = PACKAGE_DIR / "HOST_INPUT_READINESS.csv"
    verdict = PACKAGE_DIR / "PREEXECUTION_READINESS.json"
    if not (table.exists() and verdict.exists()):
        pytest.skip("readiness artifacts have not been produced")

    with open(table, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    payload = json.loads(verdict.read_text(encoding="utf-8"))
    summary = payload["readiness"]["summary"]
    assert summary["n_coordinates"] == len(rows) == 20
    assert summary["status_counts"] == counts
    assert summary["n_ready_frozen"] == counts.get("READY_FROZEN", 0)
    assert payload["readiness"]["pending"] == [
        {"market": r["market"], "host": r["host"], "blocker": r["blocker"]}
        for r in rows if r["status"] != "READY_FROZEN"]
    assert payload["n_failed"] == 0, payload["failures"]


def test_the_audit_records_a_zero_protected_final_read_count():
    """The protocol's explicit requirement, read back off the artifact.

    Every one of the twenty cells must carry the independent sealing witness,
    must report ``read_count == 0``, and must show that the sealed role was not
    merely skipped but could not be looked up at all.
    """
    verdict = PACKAGE_DIR / "PREEXECUTION_READINESS.json"
    if not verdict.exists():
        pytest.skip("PREEXECUTION_READINESS.json has not been produced")
    payload = json.loads(verdict.read_text(encoding="utf-8"))

    assert len(payload["cells"]) == 20
    for cell in payload["cells"]:
        sealing = cell["sealing"]
        assert sealing["read_count"] == 0, cell["market"]
        assert sealing["ok"] is True, cell["market"]
        assert sealing["sealed_lookup_refused"] is True, cell["market"]
        assert sealing["sealed_labels_present_in_artifact"] == [], cell["market"]
        assert sealing["identity_rows_fully_accounted"] is True, cell["market"]
        # Nothing sealed was ever materialised as a row.
        assert (sealing["artifact_rows"] + sealing["declared_sealed_days"]
                == sealing["declared_role_total"]), cell["market"]
    # And nothing sealed was ever opened.  ``read_set`` is the set of files the
    # verifier itself touched, and it must contain no sealed artifact.
    assert payload["read_set"]
    assert not [p for p in payload["read_set"]
                if "protected_final" in p.lower()]


def test_the_audit_records_a_clean_pass_and_no_scientific_execution():
    """The verdict token, the zero-failure count, and the round's scope."""
    verdict = PACKAGE_DIR / "PREEXECUTION_READINESS.json"
    if not verdict.exists():
        pytest.skip("PREEXECUTION_READINESS.json has not been produced")
    payload = json.loads(verdict.read_text(encoding="utf-8"))

    ready = "SIGNED_MASS_CHINA5_METHOD_HARNESS_READY"
    ready_pending = ready + "_WITH_EXTERNAL_HOSTS_PENDING"
    assert payload["verdict"] in (ready, ready_pending)
    assert payload["n_failed"] == 0, payload["failures"]
    # A floor, not a pin.  The count grew by nine when the method-registration
    # group was added, and it grows whenever a precondition is added -- but it
    # must never *shrink*, because a shrinking count is a group that stopped
    # running while the verdict still reads READY, and that is the only failure
    # a fixed number was ever catching here.
    assert payload["n_checks"] >= 367, payload["n_checks"]
    # What the count cannot say is *which* groups ran, so the newest one is
    # checked by name.
    method = payload["method_switches"]
    assert method["findings"] == [], method["findings"]
    assert sorted(method["switches"]) == sorted(C.SWITCH_NAMES)
    assert sorted(method["configs"]) == sorted(C.CONFIG_VARIANTS)
    assert list(method["seeds"]) == [7, 17, 37]
    assert set(payload["verdict_tokens"].values()) == {ready, ready_pending,
                                                       ready[: -len("READY")]
                                                       + "NOT_READY"}
    # The round fitted nothing: no DEV_EVAL outcome appears anywhere in the
    # audit, and the recorded `read_set` holds inputs only.
    assert "dev_eval" not in json.dumps(payload["readiness"]).lower() or \
        payload["readiness"]["summary"]["cells_with_dev_eval"] == 20
