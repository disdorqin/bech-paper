"""Tests for the aggregation, the package and the independent verifier (A6/E-J).

The mandatory pre-run patch ends in two artifacts that have to be able to
disagree with the experiment: a decision module that derives the smallest
surviving method from the screen, and an auditor that recomputes every reported
number from the raw arrays.  If either of them merely agrees with the run, the
run is unverified and the package is a press release.

So the tests below are mostly about *refusal* and *disagreement*: an incomplete
panel cannot be adjudicated, an unregistered switch vector cannot be fitted by
name, a corrupted number is caught rather than echoed, and the raw artifact can
produce its own index row without being told what it is.  Nothing here reads a
real ``DEV_EVAL`` outcome; the numbers are synthetic and the evidence root is
redirected into ``tmp_path``.
"""
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from experiments.current.hch_signed_mass_method_entry import (aggregate as A,
                                                              config as C,
                                                              evaluate as E,
                                                              grid,
                                                              oof,
                                                              protocol,
                                                              raw_evidence as RE,
                                                              runner,
                                                              synthetic)

# The tests directory is not a package, so the sibling module is reached by its
# plain name.  Importing the three fixtures rather than redeclaring them is
# deliberate: ``synthetic_thresholds`` is autouse and injects the only frozen
# quantile record the synthetic market will accept, and a local copy that drifted
# from the original would silently defuse the injection in one file and not the
# other.
from test_scientific_execution import (PACKAGE_DIR,  # noqa: F401
                                       _SYNTHETIC_THRESHOLDS,
                                       cell,  # noqa: F401 - fixture
                                       sandbox,  # noqa: F401 - fixture
                                       synthetic_thresholds)  # noqa: F401

CELLS = [(m, h) for m, h in C.CELLS]
SEEDS = tuple(int(s) for s in C.TRAINING["seeds"])

#: Measurement fields a synthetic record carries.  The keys are the harness's
#: own, so the cell table exercises the real column derivation rather than a
#: private vocabulary invented for the test.
_MEASURED = ("overall_mae", "tail_mae", "tail_mae_host", "upper_tail_mae",
             "lower_tail_mae", "normal_mae", "normal_mae_host",
             "normal_relative_harm_pct", "negative_price_mae",
             "shape_w1_positive", "shape_w1_negative",
             "mass_positive_l1", "mass_negative_l1",
             "high_mass_positive_l1", "high_mass_negative_l1")


def _records(penalty: float = 0.4, configs=None) -> list:
    """A complete synthetic panel: 20 cells x 5 configs x 3 seeds.

    ``FULL`` is given the best values and every ablation ``penalty`` worse, so
    each component's four conditions hold by construction.  The point is not to
    make the data look good but to make the *decisions* predictable, so a test
    that breaks the decision path fails for the reason it names.
    """
    out = []
    for index, (market, host) in enumerate(CELLS):
        for name in (configs or list(C.CONFIG_VARIANTS)):
            for seed in SEEDS:
                # Identical across the three seeds of a cell on purpose: the cell
                # value must be the median, and a per-seed offset would make the
                # median of three distinct numbers indistinguishable from their
                # mean at the tolerance a test can assert.
                base = 10.0 + 0.017 * index
                pen = 0.0 if name == "FULL" else penalty
                row = {
                    "market": market, "host": host, "cell": f"{market}::{host}",
                    "config": name, "seed": seed, "partition": C.ROLE_DEV_EVAL,
                    "status": "OK", "failure": None,
                    "n_days": 30, "n_entries_evaluated": 720,
                    "n_days_any_valid": 30, "alpha": 0.5,
                    "amplitude_scale": 1.0, "n_parameters": 1000,
                    "fit_seconds": 1.0, "inference_seconds": 0.01, "n_folds": 5,
                    "final_epochs": 3,
                    "switches": json.dumps(dict(C.CONFIG_VARIANTS[name]),
                                           sort_keys=True),
                }
                for field in _MEASURED:
                    row[field] = base + pen
                row["host_mae"] = base + 1.0
                row["tail_mae_host"] = base + 1.0
                row["normal_mae_host"] = base
                row["negative_price_mae_host"] = base + 1.0
                row["normal_relative_harm_pct"] = -0.5 + pen
                out.append(row)
    return out


def _strict(control_mae: str = "12.0", posthoc_mae: str = "10.6") -> list:
    """A frozen-table stand-in, written in the frozen table's own vocabulary.

    The setting strings are load-bearing, not decoration.  The rank admits
    ``OFFLINE_STATIC_POSTHOC`` rows and reports the control beside it, so a
    stand-in that wrote ``"matched"`` here would select nothing at all -- every
    cell would report ``n/a``, and a test asserting only that the column exists
    would pass over an empty rank.

    ``control_mae`` defaults to the real relationship: ``MatchedDirectResidual``
    is registered as ``OFFLINE_STATIC_CONTROL`` and is in fact worse than its own
    Host in most cells, so it is modelled worse than the setting-matched
    comparator.  The parameter exists so one test can invert it -- the rule is
    only worth anything if it still excludes the control when the control's
    number would otherwise win.
    """
    rows = []
    for market, host in CELLS:
        rows.append({"market": market, "host": host, "method": "Host",
                     "setting": C.HOST_SETTING, "result_status": "NUMERIC",
                     "Overall_MAE": "11.5", "Tail_MAE": "11.5",
                     "Normal_region_MAE": "10.0"})
        rows.append({"market": market, "host": host,
                     "method": "MatchedDirectResidual",
                     "setting": C.CONTROL_SETTING, "result_status": "NUMERIC",
                     "Overall_MAE": control_mae, "Tail_MAE": control_mae,
                     "Normal_region_MAE": "10.0"})
        rows.append({"market": market, "host": host, "method": "delta-Adapter",
                     "setting": C.SETTING_MATCHED_STRICT,
                     "result_status": "NUMERIC", "Overall_MAE": posthoc_mae,
                     "Tail_MAE": posthoc_mae, "Normal_region_MAE": "10.0"})
        rows.append({"market": market, "host": host, "method": "PIR",
                     "setting": C.SETTING_MATCHED_STRICT,
                     "result_status": "BLOCKED",
                     "Overall_MAE": None, "Tail_MAE": None,
                     "Normal_region_MAE": None})
    return rows


# ==========================================================================
# Seed aggregation
# ==========================================================================
def test_the_cell_table_medians_measurements_and_keeps_provenance_apart():
    records = _records(penalty=0.4)
    # Located by content rather than by index: the cell the first record belongs
    # to is decided by the table's own sort order, and a test that assumes it
    # would pass or fail on an ordering change instead of on the median.
    outlier = next(r for r in records if r["config"] == "FULL")
    outlier["overall_mae"] = 100.0      # an outlier seed must not survive
    cells = A.cell_table(records)
    assert len(cells) == 20 * len(C.CONFIG_VARIANTS)
    high = next(r for r in cells if r["config"] == "FULL"
                and r["cell"] == outlier["cell"])
    assert high["n_seeds"] == 3 and high["complete"] is True
    assert high["seeds"] == "|".join(str(s) for s in SEEDS)

    # The median, not the mean and not the best seed: one seed at 100 must leave
    # the cell value at the level of the other two, while the mean would be some
    # 40 and a best-seed rule would report the untouched value for the wrong
    # reason.
    seeds = [r for r in records
             if r["cell"] == high["cell"] and r["config"] == "FULL"]
    values = [r["overall_mae"] for r in seeds]
    assert sorted(values)[1] == pytest.approx(high["overall_mae"])
    assert high["overall_mae"] < min(max(values), sum(values) / 3)

    # Provenance is identity, never a measurement: a median of three hashes is
    # not a hash and a median of three byte counts is not a size.
    for field in ("status", "raw_sha256", "switches", "seeds"):
        assert field not in A._numeric_fields(records)
    assert "overall_mae" in A._numeric_fields(records)


def test_the_cell_table_records_a_derived_vector_without_a_registry_lookup():
    """PART F's vector is generally not a registered name.

    ``table`` looking the vector up by configuration name would raise on the
    derived label, and the tempting repair -- silently keeping the mismatch, or
    falling back to ``FULL`` -- would attribute the frozen run's numbers to a
    recipe it never used.
    """
    derived = dict(C.FULL_SWITCHES)
    derived["use_tcn"] = False
    assert "FROZEN_SMALLEST" not in C.CONFIG_VARIANTS

    records = _records(penalty=0.4)
    for row in records:
        if row["config"] == "FULL":
            row["config"] = "FROZEN_SMALLEST"
            row["switches"] = json.dumps(derived, sort_keys=True)
    cells = A.cell_table(records)
    frozen = [r for r in cells if r["config"] == "FROZEN_SMALLEST"]
    assert len(frozen) == 20
    assert all(json.loads(r["switches"]) == derived for r in frozen)


def test_component_decisions_refuse_a_partial_seed_grid():
    records = [r for r in _records() if not (r["seed"] == 37 and r["config"] == "FULL")]
    with pytest.raises(Exception, match="incomplete"):
        A.component_decisions(records)


# ==========================================================================
# The four adjudications and the derived vector
# ==========================================================================
def test_every_component_is_retained_when_its_registered_evidence_holds():
    decisions = A.component_decisions(_records(penalty=0.4))
    assert [d["component"] for d in decisions] == list(C.SWITCH_NAMES)
    for decision in decisions:
        assert decision["decision"] == "RETAINED", decision["reason"]
        for condition in ("condition_1_majority_cells",
                          "condition_2_tail_or_relevant_improves",
                          "condition_3_overall_non_worse",
                          "condition_4_normal_harm_bounded",
                          "condition_5_not_one_cell_or_seed"):
            assert decision[condition] is True, (decision["component"], condition)
    vector = A.frozen_vector(decisions)
    assert dict(vector["switches"]) == dict(C.FULL_SWITCHES)
    assert vector["matches_registered_config"] == "FULL"
    assert vector["deleted_components"] == []


def test_a_component_is_deleted_on_its_own_primary_metric_and_not_rescued():
    """Ablate one component by making its primary metric *worse* on.

    Only ``shape_w1_positive``/``shape_w1_negative`` are moved, and only for the
    ``NO_SHAPE_CONTEXT`` arm.  The component is judged on those two metrics
    alone, so it must be deleted even though every other number in the panel
    still favours it -- the failure this forbids is re-judging a component on
    whichever metric did improve.
    """
    records = _records(penalty=0.4)
    for row in records:
        if row["config"] == "NO_SHAPE_CONTEXT":
            row["shape_w1_positive"] -= 1.0
            row["shape_w1_negative"] -= 1.0
    decisions = A.component_decisions(records)
    by_name = {d["component"]: d for d in decisions}
    assert by_name["shape_semantic_context"]["decision"] == "DELETED"
    assert by_name["shape_semantic_context"]["condition_1_majority_cells"] is False
    assert "primary mechanism" in by_name["shape_semantic_context"]["reason"]

    # The other three are untouched: the deletion is local to its own claim.
    assert all(by_name[n]["decision"] == "RETAINED"
               for n in C.SWITCH_NAMES if n != "shape_semantic_context")

    vector = A.frozen_vector(decisions)
    assert vector["switches"]["shape_semantic_context"] is False
    assert vector["deleted_components"] == ["shape_semantic_context"]
    assert vector["matches_registered_config"] == "NO_SHAPE_CONTEXT"


def test_the_derived_vector_is_not_reported_as_a_registered_match_when_it_is_not():
    records = _records(penalty=0.4)
    for row in records:
        if row["config"] == "NO_SHAPE_CONTEXT":
            row["shape_w1_positive"] -= 1.0
        if row["config"] == "NO_TCN":
            row["tail_mae"] -= 1.0
    vector = A.frozen_vector(A.component_decisions(records))
    assert sorted(vector["deleted_components"]) == ["shape_semantic_context",
                                                    "use_tcn"]
    # Two deletions land on a vector no registered name describes, and saying so
    # is what stops PART F from reusing an ablation's predictions for it.
    assert vector["matches_registered_config"] is None
    assert vector["n_components_retained"] == 2


def _ablate(records, config, **metrics):
    """Move one ablation arm's registered metrics so the screen reads a failure.

    A component is deleted when removing it *helps*, so ``-1`` on the OFF arm's
    primary metric is what a failed component looks like.  Held separately
    because the sign is the whole content of the test that uses it.
    """
    for row in records:
        if row["config"] == config:
            for name, delta in metrics.items():
                row[name] += delta
    return records


# ==========================================================================
# The evidence package
# ==========================================================================
def test_the_package_writes_every_artifact_the_protocol_names(sandbox):
    result = A.run(_records(), strict=_strict(), online=[], write=True)
    assert result["frozen_config"] == "FULL"

    root_files = {"CELL_METRICS.csv", "COMPONENT_DECISIONS.csv",
                  "METHOD_CONFIG.json", "METHOD_SUMMARY.md", "VERDICT.json",
                  "NEW_WINDOW_HANDOFF.md"}
    for name in root_files:
        assert (sandbox / name).is_file(), name
    for name in ("03_component_screen/COMPONENT_SCREEN.csv",
                 "03_component_screen/COMPONENT_DELTAS.csv",
                 "05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.csv",
                 "05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.json"):
        assert (sandbox / name).is_file(), name

    # The root decision table and the screen's copy are the same bytes, so a
    # reader cannot be shown one set of reasons and audited against another.
    assert (sandbox / "COMPONENT_DECISIONS.csv").read_bytes() == \
        (sandbox / "03_component_screen/COMPONENT_SCREEN.csv").read_bytes()

    verdict = json.loads((sandbox / "VERDICT.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == result["verdict"]
    assert verdict["protected_final_authorized"] is False
    assert verdict["protected_final_reads"] == 0
    assert verdict["development_experiment_only"] is True

    method = json.loads((sandbox / "METHOD_CONFIG.json").read_text(encoding="utf-8"))
    assert method["knn_enabled"] is False
    assert method["switches"] == dict(C.FULL_SWITCHES)
    assert method["seeds"] == list(SEEDS)

    # The per-cell table covers the full panel and carries the registered
    # columns, not whatever happened to be in the dicts.
    metrics = A.read_csv(sandbox / "CELL_METRICS.csv")
    assert len(metrics) == 20 * len(C.CONFIG_VARIANTS)
    assert len({r["cell"] for r in metrics}) == 20
    assert set(A.CELL_METRIC_FIELDS) <= set(metrics[0])
    assert set(metrics[0]) <= set(A.CELL_METRIC_FIELDS)


def test_the_package_summary_states_the_panel_the_gate_and_the_inference_setting(sandbox):
    result = A.run(_records(), strict=_strict(), online=[], write=True)
    summary = (sandbox / "METHOD_SUMMARY.md").read_text(encoding="utf-8")

    for market, host in CELLS:
        assert f"{market}::{host}" in summary, (market, host)
    assert C.INFERENCE_SETTING in summary
    assert "**not** online test-time adaptation" in summary
    for name in result["gates"]["gates"]:
        assert name in summary
    assert result["verdict"] in summary
    # A component that failed must be reported as deleted, and no stronger claim
    # than the gate table may appear.
    assert "No rescue variant was run" in summary

    handoff = (sandbox / "NEW_WINDOW_HANDOFF.md").read_text(encoding="utf-8")
    assert result["verdict"] in handoff
    assert "PROTECTED_FINAL" in handoff
    for decision in result["decisions"]:
        assert decision["component"] in handoff


def test_the_component_delta_table_is_per_cell_and_signed_like_the_decision(sandbox):
    A.run(_records(), strict=_strict(), online=[], write=True)
    rows = A.read_csv(sandbox / "03_component_screen/COMPONENT_DELTAS.csv")
    assert {r["component"] for r in rows} == set(C.SWITCH_NAMES)
    assert len({(r["component"], r["metric"], r["cell"]) for r in rows}) == len(rows)

    # delta = ON - OFF, negative favours the component.  FULL is better here, so
    # every primary row must be negative and labelled ON.
    primary = [r for r in rows if r["role"] == "primary"]
    assert primary
    for row in primary:
        assert float(row["delta_on_minus_off"]) < 0.0
        assert row["favours"] == "ON"

    # A metric registered at two ranks -- ``use_tcn`` registers ``tail_mae`` as
    # both its primary and its secondary evidence -- is emitted once per cell, at
    # its strongest rank.  Two rows for one number under two headings would read
    # as two independent confirmations of the same fact.
    for component in C.SWITCH_NAMES:
        ranks: Dict[str, set] = {}
        for row in rows:
            if row["component"] == component:
                ranks.setdefault(row["metric"], set()).add(row["role"])
        assert all(len(v) == 1 for v in ranks.values()), (component, ranks)
        primary = [m for m, v in ranks.items() if v == {"primary"}]
        assert primary, component
    tcn = {r["metric"]: r["role"] for r in rows if r["component"] == "use_tcn"}
    assert tcn["tail_mae"] == "primary"


def test_the_online_view_is_unpooled_and_carries_no_win_column(sandbox):
    online = [{"market": m, "host": h, "method": "COSA",
               "setting": "online-tta", "parity_role": "ONLINE_TTA",
               "Overall_MAE": "9.0"} for m, h in CELLS]
    A.run(_records(), strict=_strict(), online=online, write=True)
    rows = A.read_csv(sandbox / "05_baseline_comparison/ONLINE_SUPPLEMENTARY_COSA.csv")
    assert len(rows) == 20
    for row in rows:
        assert str(row["pooled_with_strict_rank"]).lower() == "false"
    assert "strict_win" not in A.SUPPLEMENTARY_FIELDS
    assert "win" not in " ".join(A.SUPPLEMENTARY_FIELDS).replace(
        "pooled_with_strict_rank", "")

    strict_rows = A.read_csv(
        sandbox / "05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.csv")
    assert len(strict_rows) == 20
    # A blocked comparator row is never substituted for a number.
    assert all(r["best_strict_comparator"] != "PIR" for r in strict_rows)
    assert all(str(r["best_strict_comparator_status"]).startswith("NUMERIC")
               for r in strict_rows)


def test_the_control_is_not_ranked_even_when_its_number_would_win(sandbox):
    """The setting rule, tested where it actually bites.

    ``MatchedDirectResidual`` is a real measurement of a different setting, so its
    number is present and numeric and cannot be excluded for being blocked.  The
    only thing keeping it out of the rank is the setting rule -- and a rule that
    is only exercised on a control that loses anyway is not being tested at all.
    So here the control is made *better* than every setting-matched comparator:
    if the rank admitted it, it would be reported as the best strict comparator
    and the strict margin would be computed against it, loosening the gate while
    every column still looked well-formed.
    """
    A.run(_records(), strict=_strict(control_mae="9.0"), online=[], write=True)
    rows = A.read_csv(
        sandbox / "05_baseline_comparison/STRICT_OFFLINE_WITH_OUR_METHOD.csv")
    assert len(rows) == 20
    for row in rows:
        # 9.0 is the best number in the table and it is still not the comparator.
        assert row["best_strict_comparator"] == "delta-Adapter", row
        assert row["best_strict_comparator_Overall_MAE"] == "10.6", row
        # It is reported, unranked, in its own columns.
        assert row["control_method"] == "MatchedDirectResidual", row
        assert row["control_Overall_MAE"] == "9.0", row
        assert str(row["control_excluded_from_rank"]).lower() == "true", row
        # And the margin is measured against the ranked comparator, so against
        # 9.0 it would have been a different number -- 10.0-ish versus 10.6.
        ours = float(row["our_method_Overall_MAE"])
        assert float(row["strict_margin_pct"]) == pytest.approx(
            100.0 * (10.6 - ours) / 10.6, abs=1e-4), row


def test_the_package_is_refused_when_the_panel_is_not_twenty_cells(sandbox):
    # The dropped coordinate is named as a cell, not as a market string: a
    # comparison against a market literal is forbidden package-wide, because a
    # per-market branch is exactly the kind of special case the panel exists to
    # rule out, and the audit cannot tell a test's use of one from a runner's.
    dropped = CELLS[-1]
    records = [r for r in _records() if (r["market"], r["host"]) != dropped]
    with pytest.raises(Exception, match="20"):
        A.run(records, strict=_strict(), online=[], write=True)


# ==========================================================================
# The independent verifier recomputes and can say no
# ==========================================================================
def _one_fit(sandbox, cell):
    """One real fit with its raw artifact, plus the status row that claims it.

    The row is the table row, so the switch vector is a *string* exactly as
    ``write_csv`` leaves it -- the verifier has to parse the table's own encoding
    rather than a dict a test handed it.
    """
    high_mass = oof.fit_high_mass_thresholds(cell)
    record = E.evaluate_fit(cell, "FULL", 7, high_mass=high_mass, max_epochs=1)
    row = {k: v for k, v in record.items()}
    row["switches"] = json.dumps(dict(C.CONFIG_VARIANTS["FULL"]), sort_keys=True)
    row["seed"] = str(row["seed"])
    row["raw_path"] = str(sandbox / C.RAW_DIR_GRID / Path(record["raw_path"]).name)
    return high_mass, record, row


def _failed(audit, group):
    """The failed checks of one group, as a list of ``(name, detail)``."""
    return [(c["check"], c["detail"]) for c in audit.failures if c["group"] == group]


def _ran(audit, group, name):
    return [c for c in audit.checks if c["group"] == group and c["check"] == name]


def test_the_verifier_catches_a_corrupted_recorded_number(sandbox, cell):
    from experiments.current.hch_signed_mass_method_entry import verify_results as V

    high_mass, record, row = _one_fit(sandbox, cell)
    thresholds = {synthetic.SYNTHETIC_MARKET: dict(_SYNTHETIC_THRESHOLDS)}
    high = {record["cell"]: high_mass}

    audit = V.Audit()
    good = V.audit_fits([row], audit, thresholds, high, host_recheck=False)
    assert good["n_metric_bad"] == 0, good["worst"]
    assert _ran(audit, "metrics", "every_reported_number_recomputes")[0]["ok"] is True

    # Because the honest row was accepted, a rejection below is the corruption
    # and not a verifier that always says no.
    corrupt = dict(row)
    corrupt["overall_mae"] = float(row["overall_mae"]) * 1.05 + 0.25
    audit = V.Audit()
    bad = V.audit_fits([corrupt], audit, thresholds, high, host_recheck=False)
    assert bad["n_metric_bad"] >= 1
    names = [n for n, _ in _failed(audit, "metrics")]
    assert any(n.startswith("recomputed_matches_recorded::") for n in names), names
    assert "every_reported_number_recomputes" in names
    assert any("overall_mae" in json.dumps(d, default=str)
               for _, d in _failed(audit, "metrics"))

    # The recomputed value is the honest one, not the file's: the verifier derives
    # the number rather than reading it back and agreeing with itself.
    assert bad["records"][0]["overall_mae"] == pytest.approx(record["overall_mae"])


def test_the_simplex_check_accepts_a_softmax_and_rejects_a_broken_one(sandbox, cell):
    """The Shape tolerance is float32-derived, and still catches a real defect.

    The check originally demanded 1e-9 -- a float64 figure -- from rows a float32
    ``softmax`` produced, so it failed every artifact ever written, including the
    ones whose Shapes are exactly the simplex the model emitted.  Widening an
    instrument is only honest if it still fires on the failure it exists for, so
    this test does both halves: an honest artifact passes, and one whose Shapes
    are not normalised at all fails.  A tolerance that admitted the second would
    be a rubber stamp.
    """
    from experiments.current.hch_signed_mass_method_entry import verify_results as V

    high_mass, record, row = _one_fit(sandbox, cell)
    thresholds = {synthetic.SYNTHETIC_MARKET: dict(_SYNTHETIC_THRESHOLDS)}
    high = {record["cell"]: high_mass}
    tag = f"{record['cell']}::FULL::s7"

    def _branch_ok(candidate_row, mutate=None):
        path = Path(candidate_row["raw_path"])
        raw = RE.read_raw_artifact(path)
        if mutate is not None:
            mutate(raw)
        numbers = V.recompute_metrics(
            np.asarray(raw["target"], dtype=np.float64),
            np.asarray(raw["host_prediction"], dtype=np.float64),
            np.asarray(raw["prediction"], dtype=np.float64),
            np.asarray(raw["valid_mask"]).astype(bool),
            dict(_SYNTHETIC_THRESHOLDS), high_mass)
        return V._branch_numbers(raw, numbers)

    honest = _branch_ok(row)
    assert honest["shape_simplex_ok"] is True, honest["shape_simplex_max_abs_err"]
    # The tolerance is derived from the stored horizon and float32 epsilon, not
    # chosen to fit the observation.
    h = np.asarray(RE.read_raw_artifact(Path(row["raw_path"]))["shape_positive"]).shape[1]
    assert honest["shape_simplex_tolerance"] == pytest.approx(
        h * float(np.finfo(np.float32).eps))

    # A Shape that is merely mis-scaled -- the defect this check exists for --
    # is wrong by O(1) and must be refused.
    def _denormalise(raw):
        raw["shape_positive"] = np.asarray(raw["shape_positive"]) * 1.5

    broken = _branch_ok(row, _denormalise)
    assert broken["shape_simplex_ok"] is False, broken["shape_simplex_max_abs_err"]
    assert broken["shape_simplex_max_abs_err"] > 0.1


def test_the_verifier_audits_the_frozen_methods_own_fits(sandbox, cell, monkeypatch):
    """The table the verdict rests on is the frozen one, so it must be audited.

    For a derived vector the frozen method is a *sixth* configuration with its own
    status table.  A verifier that read only the screen's ``GRID_STATUS.csv``
    would recompute 300 fits and then adjudicate a configuration whose
    predictions it never opened -- and, because the derived label is not a
    registered name, it would look up that name, find no rows, and report a
    twenty-cell comparison over an empty table.

    The two tables are told apart by *which path was opened*, not by the rows that
    came back: a test that stubbed the reader and only checked the returned rows
    would pass for a verifier reading either file.  Nothing is written here --
    the source audit forbids an unregistered module from opening an artifact for
    writing, and a test that wants to prove a reader works should not have to
    become a writer to do it.
    """
    from experiments.current.hch_signed_mass_method_entry import verify_results as V

    root = RE.evidence_root()
    opened: list = []

    # The missing-record case is an absence, not an error: a reuse route has no
    # frozen fits of its own, and an audit that invented one would be reading a
    # table the run never wrote.  Asserted against the real reader, before the
    # stub below replaces it.
    assert V._frozen_record(root) == {}

    # Nothing frozen yet: the reuse route is not a case that has rows to audit.
    monkeypatch.setattr(V, "_frozen_record", lambda _root: {})
    audit = V.Audit()
    assert V.frozen_rows(root, audit) == []
    assert _ran(audit, "frozen", "frozen_rows_not_applicable")[0]["ok"] is True

    # A fitted derived vector: the frozen status table is the one opened, and it
    # is opened under 04_frozen_method rather than 02_crossfit.
    monkeypatch.setattr(V, "_frozen_record", lambda _root: {
        "route": "FITTED_DERIVED_VECTOR", "frozen_label": "FROZEN_SMALLEST"})
    row = {"cell": f"{synthetic.SYNTHETIC_MARKET}::Synthetic",
           "config": "FROZEN_SMALLEST", "seed": "7", "status": "OK"}

    def _read_csv(path):
        opened.append(Path(path))
        return [row]

    monkeypatch.setattr(V.A, "read_csv", _read_csv)
    audit = V.Audit()
    rows = V.frozen_rows(root, audit)
    assert [r["config"] for r in rows] == ["FROZEN_SMALLEST"]
    assert len(opened) == 1 and opened[0].name == "FROZEN_STATUS.csv", opened
    assert opened[0].parent.name == "04_frozen_method", opened
    assert _ran(audit, "frozen", "frozen_status_table_present")[0]["ok"] is True


def test_the_verifier_flags_an_artifact_that_is_not_the_row_it_claims(sandbox, cell):
    from experiments.current.hch_signed_mass_method_entry import verify_results as V

    high_mass, record, row = _one_fit(sandbox, cell)
    thresholds = {synthetic.SYNTHETIC_MARKET: dict(_SYNTHETIC_THRESHOLDS)}
    high = {record["cell"]: high_mass}
    tag = f"{record['cell']}::FULL::s7"

    def _audit_for(candidate):
        audit = V.Audit()
        V.audit_fits([candidate], audit, thresholds, high, host_recheck=False)
        return audit

    audit = _audit_for(row)
    assert _ran(audit, "artifacts", f"identity::{tag}")[0]["ok"] is True
    assert _ran(audit, "artifacts", f"hash::{tag}")[0]["ok"] is True

    # A row that does not record its switch vector cannot be checked against the
    # artifact, and must fail rather than pass on the strength of its name.
    silent = dict(row)
    silent.pop("switches", None)
    assert _ran(_audit_for(silent), "artifacts",
                f"identity::{tag}")[0]["ok"] is False

    # The name is right and the vector is wrong: exactly the case a registry
    # lookup by name would wave through.
    wrong = dict(row)
    wrong["switches"] = json.dumps({k: not v for k, v in C.FULL_SWITCHES.items()},
                                   sort_keys=True)
    assert _ran(_audit_for(wrong), "artifacts",
                f"identity::{tag}")[0]["ok"] is False

    # And a rewritten artifact is caught by its own bytes, not by its metadata.
    other = dict(row)
    other["raw_sha256"] = "0" * 64
    assert _ran(_audit_for(other), "artifacts", f"hash::{tag}")[0]["ok"] is False


def test_the_verifier_does_not_import_the_metric_layer_it_audits():
    """Independence is structural, not a promise in a docstring.

    A verifier that called ``metrics.compute_metrics`` would reproduce a sign
    error or a transposed axis faithfully and report agreement.  The ban is on
    the metric layer specifically; ``config`` and ``raw_evidence`` are definitions
    and file plumbing, not arithmetic.
    """
    tree = ast.parse((PACKAGE_DIR / "verify_results.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any(name.endswith("metrics") or name.endswith("evaluate")
                   for name in imported), sorted(imported)


def test_the_verifier_derives_the_verdict_before_it_reads_the_recorded_one():
    """``A.verdict_token`` must be called before ``VERDICT.json`` is opened.

    A verdict that is read first and re-derived afterwards can only ever agree;
    the ordering is what makes the comparison a check.
    """
    source = (PACKAGE_DIR / "verify_results.py").read_text(encoding="utf-8")
    body = source[source.index("def audit_gates("):]
    body = body[:body.index("\ndef ")]
    assert body.index("verdict_token") < body.index("VERDICT.json"), body


# ==========================================================================
# A derived vector is fitted through the registered path
# ==========================================================================
def test_evaluate_fit_refuses_an_unregistered_name_without_a_switch_vector(cell):
    with pytest.raises(Exception, match="unknown configuration"):
        E.evaluate_fit(cell, "FROZEN_SMALLEST", 7, max_epochs=1, write=False)


def test_evaluate_fit_fits_a_derived_vector_under_its_own_label(sandbox, cell):
    derived = dict(C.FULL_SWITCHES)
    derived["use_tcn"] = False
    high_mass = oof.fit_high_mass_thresholds(cell)
    record = E.evaluate_fit(cell, "FROZEN_SMALLEST", 7, high_mass=high_mass,
                            max_epochs=1, switches=derived)
    meta = RE.read_raw_meta(Path(record["raw_path"]) if
                            Path(record["raw_path"]).is_absolute()
                            else C.REPO_ROOT / record["raw_path"])
    assert meta["config"] == "FROZEN_SMALLEST"
    assert meta["switches"] == derived
    assert meta["role"] == C.ROLE_OUR_METHOD


def test_a_raw_artifact_can_produce_its_own_index_row(sandbox, cell):
    """A4: the artifact is self-describing.

    The index is built from the artifact alone.  A row assembled from what the
    caller intended to write would keep looking right after the write had
    changed, which is the one failure an index exists to prevent.
    """
    high_mass = oof.fit_high_mass_thresholds(cell)
    record = E.evaluate_fit(cell, "FULL", 7, high_mass=high_mass, max_epochs=1)
    path = (Path(record["raw_path"]) if Path(record["raw_path"]).is_absolute()
            else C.REPO_ROOT / record["raw_path"])
    meta = RE.read_raw_meta(path)
    row = RE.index_rows_from_artifact(meta, {
        "raw_path": record["raw_path"], "raw_sha256": record["raw_sha256"],
        "raw_bytes": record["raw_bytes"]})

    assert set(row) == set(RE.RAW_INDEX_FIELDS)
    assert row["market"] == cell.market and row["host"] == cell.host
    assert row["cell"] == cell.key and row["config"] == "FULL"
    assert row["partition"] == C.ROLE_DEV_EVAL
    assert row["role"] == C.ROLE_OUR_METHOD
    assert row["switches"] == dict(C.FULL_SWITCHES)
    assert row["raw_sha256"] == record["raw_sha256"]
    assert row["raw_bytes"] == record["raw_bytes"]
    # The provenance the harness computes itself is always present.  The Host and
    # dataset digests are absent here only because the synthetic substrate has no
    # frozen Host artifact to point at; the columns exist so a real cell's row
    # carries them, and inventing a value for the synthetic one would be the
    # failure this index exists to prevent.
    for field in ("source_sha256", "core_provenance_sha256", "git_commit",
                  "generated_utc"):
        assert row.get(field), (field, row.get(field))
    for field in ("frozen_host_artifact_sha256", "dataset_contract_sha256"):
        assert field in row
    # The threshold pair travels with the artifact, so a reader can reproduce
    # the high-mass subset without consulting another file.
    assert row["high_mass_q90_positive"] == pytest.approx(
        high_mass["q90_positive"])


def test_the_grid_appends_the_raw_index_and_resumes_by_identity(sandbox, cell,
                                                                monkeypatch):
    """One cell, one config, one seed through the real driver."""
    high_mass = oof.fit_high_mass_thresholds(cell)
    monkeypatch.setattr(grid, "_dataset", lambda market, host: (cell, None))
    monkeypatch.setattr(protocol, "high_mass_for",
                        lambda market, host, path=None: dict(high_mass))

    cells = [(cell.market, cell.host)]
    first = grid.run_grid(cells, configs=["FULL"], seeds=[7], workers=1)
    assert first["summary"]["n_failed"] == 0
    index = sandbox / "RAW_PREDICTION_INDEX.csv"
    assert index.is_file()
    rows = A.read_csv(index)
    assert len(rows) == 1 and rows[0]["config"] == "FULL"
    assert rows[0]["raw_sha256"] == first["records"][0]["raw_sha256"]

    # A re-run resumes rather than re-fits, and the index does not grow: it is a
    # function of the artifacts, not a log of the sessions that produced them.
    second = grid.run_grid(cells, configs=["FULL"], seeds=[7], workers=1)
    assert second["summary"]["n_resumed"] == 1
    assert len(A.read_csv(index)) == 1

    # A resume whose vector disagrees is a collision, not a resume.
    other = dict(C.FULL_SWITCHES)
    other["use_tcn"] = False
    with pytest.raises(Exception, match="collision"):
        grid.run_grid(cells, configs=["FULL"], seeds=[7], workers=1,
                      variants={"FULL": other})


def test_the_grid_serves_a_derived_vector_through_the_registered_path(sandbox, cell,
                                                                      monkeypatch):
    high_mass = oof.fit_high_mass_thresholds(cell)
    monkeypatch.setattr(grid, "_dataset", lambda market, host: (cell, None))
    monkeypatch.setattr(protocol, "high_mass_for",
                        lambda market, host, path=None: dict(high_mass))
    derived = dict(C.FULL_SWITCHES)
    derived["rare_mass_sampling"] = False

    result = grid.run_grid([(cell.market, cell.host)], configs=["FROZEN_SMALLEST"],
                           seeds=[7], workers=1, raw_subdir=C.RAW_DIR_FROZEN,
                           metric_subdir=C.METRIC_DIR_FROZEN,
                           status_name="FROZEN_STATUS.csv",
                           out_subdir="04_frozen_method",
                           variants={"FROZEN_SMALLEST": derived})
    assert result["summary"]["n_failed"] == 0
    raw = result["records"][0]["raw_path"]
    meta = RE.read_raw_meta(Path(raw) if Path(raw).is_absolute()
                            else C.REPO_ROOT / raw)
    assert meta["switches"] == derived
    # The frozen run's evidence lands under its own subtree, never over the
    # screen's.
    assert Path(raw).name.startswith(cell.market)
    assert "04_frozen_method" in Path(result["status_table"]).as_posix() or \
        "04_frozen_method" in str(result["status_table"])


def test_a_switch_vector_with_an_unknown_switch_is_refused():
    with pytest.raises(Exception, match="registered switches"):
        grid.plan_grid([CELLS[0]], configs=["X"],
                       variants={"X": {"use_tcn": True}})


# ==========================================================================
# The closing of the second DEV_EVAL route
# ==========================================================================
def test_the_old_prefix_evaluation_route_is_closed(monkeypatch):
    monkeypatch.setattr(runner.H, "build_readiness_table", lambda: [])
    args = argparse.Namespace(markets=None, hosts=None, configs=["FULL"],
                              dev_eval_days=3, execute=True,
                              authorization_token=None)
    with pytest.raises(Exception, match="--dev-eval-days is closed"):
        runner.cmd_run(args)


def test_the_grid_commands_refuse_to_run_without_their_two_preconditions(monkeypatch):
    with pytest.raises(Exception, match="--dev-eval-all"):
        runner.cmd_grid(argparse.Namespace(dev_eval_all=False,
                                           authorization_token=None))
    with pytest.raises(Exception, match="authorization-token"):
        runner.cmd_grid(argparse.Namespace(dev_eval_all=True,
                                           authorization_token="nope"))
    monkeypatch.setattr(protocol, "load_thresholds", lambda path=None: None)
    with pytest.raises(Exception, match="high-mass threshold"):
        runner.cmd_grid(argparse.Namespace(dev_eval_all=True,
                                           authorization_token=runner.EXECUTION_TOKEN))


def _write_screen(sandbox, records):
    """Persist a synthetic screen through the real table writer.

    Written rather than injected because the point of these tests is what
    survives the round trip: ``cmd_frozen`` reads a file, and a row that loses its
    switch vector on the way to disk is a defect the file is the only witness to.
    """
    from experiments.current.hch_signed_mass_method_entry.contracts import write_csv

    path = sandbox / "02_crossfit" / "GRID_STATUS.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv([{f: r.get(f) for f in grid.GRID_STATUS_FIELDS} for r in records],
              path, grid.GRID_STATUS_FIELDS)
    return path


def _frozen_args():
    return argparse.Namespace(authorization_token=runner.EXECUTION_TOKEN,
                              markets=None, hosts=None, workers=1, no_resume=False)


def test_the_switch_vector_survives_the_round_trip_through_the_status_table(sandbox):
    """The screen table has to record the recipe, not only its name.

    Without the column, ``cmd_frozen``'s identity check and the verifier's both
    see ``None`` and the only way to make them pass is to look the vector up by
    configuration name -- which is precisely the substitution both exist to
    refuse.
    """
    _write_screen(sandbox, _records(penalty=0.4))
    rows = A.read_csv(sandbox / "02_crossfit" / "GRID_STATUS.csv")
    assert "switches" in rows[0]
    for row in rows:
        assert json.loads(row["switches"]) == dict(C.CONFIG_VARIANTS[row["config"]])


def test_the_frozen_command_derives_its_vector_from_the_screen(sandbox, monkeypatch):
    """PART F's vector comes from the decisions, not from a chosen name."""
    records = _ablate(_records(penalty=0.4), "NO_SHAPE_CONTEXT",
                      shape_w1_positive=-1.0)
    _ablate(records, "NO_TCN", tail_mae=-1.0)
    _write_screen(sandbox, records)

    seen = {}

    def _fake_run_grid(cells, configs=None, seeds=None, variants=None, **kwargs):
        seen.update(configs=list(configs), variants=dict(variants or {}),
                    cells=list(cells), kwargs=dict(kwargs))
        return {"summary": {"n_records": 60, "n_resumed": 0, "n_failed": 0},
                "status_table": str(sandbox / "04_frozen_method"
                                    / "FROZEN_STATUS.csv")}

    monkeypatch.setattr(grid, "run_grid", _fake_run_grid)
    monkeypatch.setattr(runner, "_ready_cells",
                        lambda args: [{"market": m, "host": h} for m, h in CELLS])
    assert runner.cmd_frozen(_frozen_args()) == 0

    assert seen["configs"] == [runner._FROZEN_NAME]
    vector = seen["variants"][runner._FROZEN_NAME]
    assert vector["shape_semantic_context"] is False
    assert vector["use_tcn"] is False
    assert vector["untied_amplitude_heads"] is True
    assert vector["rare_mass_sampling"] is True
    assert len(seen["cells"]) == 20
    # The frozen run's evidence lands in its own subtree: writing it over the
    # screen's raw artifacts would destroy the evidence the decision cites.
    assert seen["kwargs"]["raw_subdir"] == C.RAW_DIR_FROZEN
    assert seen["kwargs"]["metric_subdir"] == C.METRIC_DIR_FROZEN
    assert seen["kwargs"]["out_subdir"] == "04_frozen_method"

    payload = json.loads((sandbox / "04_frozen_method"
                          / "FROZEN_METHOD_CONFIG.json").read_text(encoding="utf-8"))
    assert payload["route"] == "FITTED_DERIVED_VECTOR"
    assert payload["frozen_label"] == runner._FROZEN_NAME
    assert payload["registered_config"] is None
    assert payload["switches"] == vector
    assert payload["n_records"] == 60


def test_the_frozen_command_reuses_a_registered_match_only_after_the_identity_check(
        sandbox, monkeypatch):
    """A registered name is a label, and a label can be wrong.

    Reuse is what PART F permits, but the permission is conditional on the
    persisted rows carrying *this* vector.  The table is written by the real
    writer here, so the check is exercised against the same encoding the screen
    will produce.
    """
    records = _ablate(_records(penalty=0.4), "NO_TCN", tail_mae=-1.0)
    path = _write_screen(sandbox, records)
    monkeypatch.setattr(runner, "_ready_cells",
                        lambda args: [{"market": m, "host": h} for m, h in CELLS])

    def _no_fits(*args, **kwargs):
        raise AssertionError("the frozen vector matched a registered "
                             "configuration, so nothing may be re-fitted")

    monkeypatch.setattr(grid, "run_grid", _no_fits)
    assert runner.cmd_frozen(_frozen_args()) == 0
    payload = json.loads((sandbox / "04_frozen_method"
                          / "FROZEN_METHOD_CONFIG.json").read_text(encoding="utf-8"))
    assert payload["route"] == "REUSED_REGISTERED_CONFIG"
    assert payload["frozen_label"] == "NO_TCN"
    assert payload["identity_check"]["all_identities_match"] is True
    assert payload["identity_check"]["n_checked"] == payload["identity_check"]["n_expected"]

    # One row that does not carry the vector it is filed under, and the reuse is
    # refused -- the artifact's name would still have matched.
    from experiments.current.hch_signed_mass_method_entry.contracts import write_csv

    tampered = [dict(r) for r in records]
    for row in tampered:
        if row["config"] == "NO_TCN" and row["seed"] == 7:
            row["switches"] = json.dumps(dict(C.FULL_SWITCHES), sort_keys=True)
    write_csv([{f: r.get(f) for f in grid.GRID_STATUS_FIELDS} for r in tampered],
              path, grid.GRID_STATUS_FIELDS)
    with pytest.raises(Exception, match="do not all carry"):
        runner.cmd_frozen(_frozen_args())

    # A row with no vector at all is the same finding, not a pass by absence.
    silent = [{k: v for k, v in r.items() if k != "switches"} for r in records]
    write_csv([{f: r.get(f) for f in grid.GRID_STATUS_FIELDS} for r in silent],
              path, grid.GRID_STATUS_FIELDS)
    with pytest.raises(Exception, match="do not all carry"):
        runner.cmd_frozen(_frozen_args())
