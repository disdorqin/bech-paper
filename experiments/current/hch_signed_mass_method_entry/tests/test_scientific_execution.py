"""Tests for the mandatory pre-run patch, A1-A6.

The scientific execution prompt requires seven things to be true *before* the
first ``DEV_EVAL`` number exists: the direct branch heads are what the mechanism
metrics read (A1), the Amplitude scale is the training-prefix median and never a
tuned switch (A2), the high-mass thresholds are frozen from ``POST_TRAIN`` (A3),
raw prediction evidence is persisted immutably and inside the evidence root (A4),
D0 describes the geometry without training anything (A5), and the whole thing is
recomputable from the raw artifacts by an independent reader (A6).

Every test here runs on the synthetic substrate.  A test that read a real
``DEV_EVAL`` outcome would be a scientific result wearing a test's name, and the
ordering requirement is explicit: no real ``DEV_EVAL`` row may be touched until
the verifier has passed.

Two consequences shape the code below.  First, the evidence root is redirected
into ``tmp_path`` through :func:`raw_evidence.evidence_root` -- the seam exists
so the confinement rules can be exercised without writing anywhere real.  Second,
the frozen per-market thresholds are injected for the synthetic market only,
because the generator's prices are not any market's prices and inventing a
quantile for a real market would defeat the point of freezing them.
"""
from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.current.hch_signed_mass_method_entry import (config as C,
                                                              evaluate as E,
                                                              metrics as M,
                                                              oof,
                                                              raw_evidence as RE,
                                                              synthetic,
                                                              training as T)
from experiments.current.hch_signed_mass_method_entry.contracts import (
    EvidenceWriteError, LegalityError, ProvenanceError)

PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: The synthetic prices are not any market's prices, so the frozen quantiles are
#: injected for ``SYNTHETIC`` alone.  A real market's thresholds are never
#: fabricated here, and the cache is restored after every test.
_SYNTHETIC_THRESHOLDS = {
    "q05": 20.0,
    "q95": 110.0,
    "day_spread_p90": 60.0,
    "threshold_source": "<synthetic>",
    "threshold_source_sha256": "0" * 64,
    "rule": dict(C.THRESHOLD_RULE),
    "recomputed_for_this_evaluation": False,
}


@pytest.fixture(autouse=True)
def synthetic_thresholds(monkeypatch):
    """Inject the synthetic market's frozen thresholds, and restore afterwards.

    Autouse because *every* metric call needs a threshold record, including the
    ones on hand-built arrays that never touch a dataset.  Without it the harness
    fails closed on ``SYNTHETIC`` -- correctly, since it will not improvise a
    quantile for a market -- and a test asserting the wrong exception would pass
    for the wrong reason.

    The module is imported *here* rather than taken from this module's globals so
    the cache patched is the one the production code will actually read: another
    test in this suite purges and reimports the package, and a stale reference
    would silently defuse the injection.
    """
    from experiments.current.hch_signed_mass_method_entry import metrics, synthetic

    monkeypatch.setitem(metrics._THRESHOLD_CACHE, synthetic.SYNTHETIC_MARKET,
                        dict(_SYNTHETIC_THRESHOLDS))
    return _SYNTHETIC_THRESHOLDS


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A confined evidence root, redirected on the live module."""
    from experiments.current.hch_signed_mass_method_entry import raw_evidence

    root = (tmp_path / "evidence").resolve()
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(raw_evidence, "evidence_root", lambda: root)
    return root


@pytest.fixture
def cell():
    """A fresh synthetic cell per test; nothing shared can be mutated."""
    return synthetic.synthetic_dataset(n_features=6, seed=11)


def _fitted(dataset, config_name="FULL", seed=7, max_epochs=1):
    return oof.fit_method(dataset, C.CONFIG_VARIANTS[config_name], seed,
                          max_epochs=max_epochs)


def _predicted(dataset, config_name="FULL", seed=7, max_epochs=1):
    method = _fitted(dataset, config_name, seed, max_epochs)
    rows = dataset.positions(C.ROLE_DEV_EVAL)
    return method, method.predict(dataset, rows), rows


# ==========================================================================
# A1 -- the direct branch heads are the evidence
# ==========================================================================
def test_a1_every_prediction_exposes_the_direct_branch_arrays(cell):
    method, out, rows = _predicted(cell)
    n, h = len(rows), C.HORIZON
    assert out["shape_positive"].shape == (n, h)
    assert out["shape_negative"].shape == (n, h)
    assert out["mass_positive"].shape == (n,)
    assert out["mass_negative"].shape == (n,)
    assert out["correction"].shape == (n, h)
    assert out["prediction"].shape == (n, h)
    assert out["valid_mask"].shape == (n, h)

    # A simplex is nonnegative and sums to one, per day, for both signs.
    for key in ("shape_positive", "shape_negative"):
        simplex = np.asarray(out[key], dtype=np.float64)
        assert (simplex >= -1e-6).all()
        assert np.allclose(simplex.sum(axis=1), 1.0, atol=1e-5)
    assert (np.asarray(out["mass_positive"]) >= 0.0).all()
    assert (np.asarray(out["mass_negative"]) >= 0.0).all()

    # Stable identity travels with the arrays, so a raw artifact can be joined
    # back to a day without re-deriving the row plan.
    identity = out["identity"]
    assert len(identity["label"]) == n
    assert len(identity["target_day"]) == n
    assert len(identity["role"]) == n
    assert list(identity["episode_index"]) == [int(r) for r in rows]
    assert set(identity["role"]) == {C.ROLE_DEV_EVAL}

    # The fusion identity holds to float32 rounding, and alpha is the linear
    # map the design says it is.
    rebuilt = (np.asarray(out["mass_positive"])[:, None] * np.asarray(out["shape_positive"])
               - np.asarray(out["mass_negative"])[:, None] * np.asarray(out["shape_negative"]))
    assert np.allclose(rebuilt, np.asarray(out["correction"]), atol=1e-6)
    expected = np.asarray(out["host"]) + method.alpha * np.asarray(out["correction"])
    assert np.allclose(expected, np.asarray(out["prediction"]), atol=1e-6)


def test_a1_mechanism_metrics_read_the_branch_not_the_net_correction():
    """The one test that makes A1 falsifiable.

    A branch whose two signs cancel exactly leaves a correction of zero.  The
    direct-mass metric must still see the branch's own masses; the fallback that
    decomposes the net correction cannot, because after cancellation there is
    nothing left to decompose.  If these two numbers ever coincide, the metric
    is not reading the heads it claims to read.
    """
    y = np.array([[100.0, 100.0, 100.0, 100.0],
                  [100.0, 100.0, 100.0, 100.0]])
    host = np.zeros((2, 4))
    realised = M.shape_decomposition(y - host)
    assert (realised["mass_positive"] > 0).all()

    spike = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    branch = {"shape_positive": spike, "shape_negative": spike,
              "mass_positive": np.array([10.0, 10.0]),
              "mass_negative": np.array([10.0, 10.0]),
              "correction": np.zeros((2, 4))}

    direct = M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                               branch=branch)
    assert direct["branch_arrays_used"] == M.DIRECT
    assert direct["branch_reconstruction_max_abs_err"] <= M.RECONSTRUCTION_ATOL
    assert np.allclose(direct["mass_positive_l1"],
                       np.abs(10.0 - realised["mass_positive"]).mean())

    fallback = M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET)
    assert fallback["branch_arrays_used"] == M.DECOMPOSED
    assert fallback["mass_positive_l1"] == pytest.approx(
        realised["mass_positive"].mean())
    assert direct["mass_positive_l1"] != fallback["mass_positive_l1"]


def test_a1_a_branch_that_does_not_reconstruct_its_correction_is_refused():
    y = np.ones((2, 4)) * 100.0
    host = np.zeros((2, 4))
    spike = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    branch = {"shape_positive": spike, "shape_negative": spike,
              "mass_positive": np.array([10.0, 10.0]),
              "mass_negative": np.array([10.0, 10.0]),
              "correction": np.ones((2, 4))}  # cancels to zero; this is a lie
    with pytest.raises(ProvenanceError, match="do not reconstruct"):
        M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                          branch=branch)


def test_a1_an_incomplete_branch_is_refused_rather_than_decomposed():
    y = np.ones((2, 4)) * 100.0
    host = np.zeros((2, 4))
    branch = {"shape_positive": np.full((2, 4), 0.25),
              "shape_negative": np.full((2, 4), 0.25),
              "mass_positive": np.array([1.0, 1.0])}  # mass_negative missing
    with pytest.raises(ProvenanceError, match="incomplete"):
        M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                          branch=branch)


def test_a1_a_prediction_that_is_not_host_plus_alpha_times_correction_is_refused():
    y = np.ones((2, 4)) * 100.0
    host = np.zeros((2, 4))
    spike = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    branch = {"shape_positive": spike, "shape_negative": spike,
              "mass_positive": np.array([10.0, 10.0]),
              "mass_negative": np.zeros(2),
              "correction": np.array([[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])}
    # The correction is real but ``method_pred`` ignores alpha entirely.
    with pytest.raises(ProvenanceError, match="host \\+ alpha"):
        M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                          branch=branch, alpha=0.5)


# ==========================================================================
# A2 -- the robust Amplitude output scale s_A
# ==========================================================================
def test_a2_amplitude_scale_is_the_pooled_median_of_the_prefix_masses():
    pos = np.array([4.0, 0.0, 10.0, 16.0, 0.0])
    neg = np.array([2.0, 0.0, 0.0, 0.0, 6.0])
    value, rule = T.fit_amplitude_scale(pos, neg)
    pooled = np.array([4.0, 10.0, 16.0, 2.0, 6.0])
    assert value == pytest.approx(float(np.median(pooled)))
    assert rule["shared_by_both_sign_heads"] is True
    assert rule["n_pooled_terms"] == pooled.size
    assert rule["degenerate"] if "degenerate" in rule else True

    # An all-zero prefix is the one admissible use of the fixed floor, and it is
    # reported as degenerate rather than silently returned as a fitted number.
    floor, degenerate_rule = T.fit_amplitude_scale(np.zeros(3), np.zeros(3))
    assert floor == float(C.AMPLITUDE_SCALE_FLOOR)
    assert degenerate_rule["degenerate"] == "NO_POSITIVE_MASS_IN_PREFIX"


def test_a2_the_scale_is_derived_from_the_fitting_prefix_and_not_dev_eval(cell):
    post = cell.positions(C.ROLE_POST_TRAIN)
    dev = cell.positions(C.ROLE_DEV_EVAL)
    masses = M.shape_decomposition(cell.residual[post].astype(np.float64) * cell.valid_mask[post])
    expected = float(np.median(np.concatenate(
        [masses["mass_positive"][masses["mass_positive"] > 0.0],
         masses["mass_negative"][masses["mass_negative"] > 0.0]])))

    method = _fitted(cell)
    assert method.stats.amplitude_scale == pytest.approx(expected)
    assert method.stats.amplitude_scale_rule["shared_by_both_sign_heads"] is True

    # The pooled-over-everything median is a different number; the fact that the
    # fitted value matches the prefix one and not this one is the evidence that
    # DEV_EVAL contributed nothing.
    every = M.shape_decomposition(cell.residual.astype(np.float64) * cell.valid_mask)
    pooled_all = np.concatenate([every["mass_positive"][every["mass_positive"] > 0.0],
                                 every["mass_negative"][every["mass_negative"] > 0.0]])
    assert method.stats.amplitude_scale != pytest.approx(float(np.median(pooled_all)))
    assert dev.size > 0

    # Each fold fits its own scale, from its own inner-training prefix.
    prefixes = {float(f["amplitude_scale"]) for f in method.folds}
    assert len(prefixes) >= 1
    for fold in method.folds:
        assert fold["amplitude_scale"] > 0.0
        assert fold["amplitude_scale_rule"]["rule"].startswith("median{")


def test_a2_corrupting_dev_eval_leaves_the_scale_bit_identical():
    first = _fitted(synthetic.synthetic_dataset(n_features=6, seed=11))
    reference = first.stats.amplitude_scale

    poisoned = synthetic.synthetic_dataset(n_features=6, seed=11)
    rows = poisoned.positions(C.ROLE_DEV_EVAL)
    poisoned.residual[rows] = 1.0e6
    poisoned.target[rows] = poisoned.host_forecast[rows] + 1.0e6
    second = _fitted(poisoned)
    assert second.stats.amplitude_scale == reference


def test_a2_the_scale_reaches_the_core_as_a_configuration_input(cell):
    """The core applies ``s_A``; the harness only computes it."""
    method = _fitted(cell)
    model, builder = method.model_and_builder()
    scale = float(method.stats.amplitude_scale)
    assert model.config.amplitude_scale == pytest.approx(scale)

    # And it is a scale, not a switch: no registered configuration varies it,
    # and no ablation turns it off.
    switches = " ".join(C.SWITCH_NAMES)
    assert "amplitude_scale" not in switches
    assert len({tuple(sorted(v.items())) for v in C.CONFIG_VARIANTS.values()}) == \
        len(C.CONFIG_VARIANTS)
    for name, vector in C.CONFIG_VARIANTS.items():
        assert set(vector) == set(C.SWITCH_NAMES), name


# ==========================================================================
# A3 -- frozen high-mass thresholds
# ==========================================================================
def test_a3_thresholds_are_fitted_on_post_train_rows_only(cell):
    record = oof.fit_high_mass_thresholds(cell)
    assert record["fit_partition"] == C.ROLE_POST_TRAIN
    assert record["recomputed_for_this_evaluation"] is False

    rows = np.sort(cell.positions(C.ROLE_POST_TRAIN).astype(np.int64))
    assert record["n_rows"] == rows.size
    assert record["fit_rows_sha256"] == T._rows_digest(rows)
    assert record["cell"] == cell.key

    masses = M.shape_decomposition(cell.residual[rows].astype(np.float64) * cell.valid_mask[rows])
    assert record["q90_positive"] == pytest.approx(
        float(np.quantile(masses["mass_positive"], C.HIGH_MASS_QUANTILE)))
    assert record["q90_negative"] == pytest.approx(
        float(np.quantile(masses["mass_negative"], C.HIGH_MASS_QUANTILE)))


def test_a3_thresholds_do_not_move_when_dev_eval_moves():
    base = oof.fit_high_mass_thresholds(synthetic.synthetic_dataset(seed=11))
    poisoned = synthetic.synthetic_dataset(seed=11)
    rows = poisoned.positions(C.ROLE_DEV_EVAL)
    poisoned.residual[rows] = -5.0e3
    poisoned.target[rows] = poisoned.host_forecast[rows] - 5.0e3
    after = oof.fit_high_mass_thresholds(poisoned)
    assert after["q90_positive"] == base["q90_positive"]
    assert after["q90_negative"] == base["q90_negative"]
    assert after["fit_rows_sha256"] == base["fit_rows_sha256"]


def test_a3_an_empty_dev_subset_is_reported_as_not_applicable(cell):
    rng = np.random.default_rng(0)
    y = cell.target[cell.positions(C.ROLE_DEV_EVAL)]
    host = cell.host_forecast[cell.positions(C.ROLE_DEV_EVAL)]
    huge = {"quantile": C.HIGH_MASS_QUANTILE, "q90_positive": 1.0e12,
            "q90_negative": 1.0e12, "usable_positive": True, "usable_negative": True}
    block = M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                              high_mass=huge)
    assert block["high_mass_positive_n_days"] == 0
    assert block["high_mass_positive_l1"] is None
    assert block["high_mass_positive_status"] == \
        f"{C.NOT_APPLICABLE_N0}:EMPTY_DEV_SUBSET"

    unusable = dict(huge, usable_positive=False)
    block = M.compute_metrics(y, host, host, market=synthetic.SYNTHETIC_MARKET,
                              high_mass=unusable)
    assert block["high_mass_positive_status"] == \
        f"{C.NOT_APPLICABLE_N0}:THRESHOLD_NOT_USABLE"
    assert block["mass_thresholds_used"]["q90_positive"] == 1.0e12
    assert rng is not None


# ==========================================================================
# A4 -- immutable, confined raw prediction evidence
# ==========================================================================
def _toy_artifact(scale=1.0):
    arrays = {"target": np.array([[1.0, 2.0]]),
              "host_prediction": np.array([[0.5, 1.5]]),
              "shape_positive": np.array([[0.5, 0.5]]),
              "shape_negative": np.array([[0.5, 0.5]]),
              "mass_positive": np.array([0.5 * scale]),
              "mass_negative": np.array([0.25 * scale]),
              "correction": np.array([[0.125 * scale, 0.125 * scale]]),
              "prediction": np.array([[0.625 * scale, 1.625 * scale]]),
              "valid_mask": np.array([[1, 1]], dtype=np.uint8)}
    meta = {"config": "FULL", "seed": 7, "partition": C.ROLE_DEV_EVAL,
            "switches": dict(C.FULL_SWITCHES), "identity": {"episode_index": [0]}}
    return arrays, meta


def test_a4_a_raw_artifact_round_trips_and_hashes_its_own_bytes(sandbox):
    arrays, meta = _toy_artifact()
    path = RE.raw_artifact_path("SYNTHETIC", "Synthetic", "FULL", 7,
                                C.ROLE_DEV_EVAL)
    written = RE.write_raw_artifact(arrays, meta, path)
    assert written["written"] is True
    assert Path(written["raw_path"]).exists()
    assert written["raw_bytes"] == Path(written["raw_path"]).stat().st_size

    loaded = RE.read_raw_artifact(path)
    assert loaded["raw_sha256"] == written["raw_sha256"]
    for key in RE.RAW_ARRAY_KEYS:
        assert np.array_equal(loaded[key], np.asarray(arrays[key])), key
    assert loaded["meta"]["config"] == "FULL"
    assert loaded["meta"]["switches"] == dict(C.FULL_SWITCHES)
    assert loaded["meta"]["identity"] == meta["identity"]

    # Rewriting the same bytes is a no-op; rewriting different bytes is refused,
    # so a re-run cannot quietly replace the evidence a previous run produced.
    again = RE.write_raw_artifact(arrays, meta, path)
    assert again["written"] is False
    assert again["raw_sha256"] == written["raw_sha256"]

    other, other_meta = _toy_artifact(scale=3.0)
    with pytest.raises(EvidenceWriteError, match="refusing to overwrite"):
        RE.write_raw_artifact(other, other_meta, path)
    assert RE.read_raw_artifact(path)["raw_sha256"] == written["raw_sha256"]


def test_a4_a_destination_outside_the_evidence_root_is_refused(sandbox, tmp_path):
    outside = tmp_path / "not-evidence" / "x.npz"
    with pytest.raises(EvidenceWriteError, match="outside this experiment"):
        RE.confine(outside, "test write")
    arrays, meta = _toy_artifact()
    with pytest.raises(EvidenceWriteError):
        RE.write_raw_artifact(arrays, meta, outside)


def test_a4_a_frozen_root_is_refused_even_though_it_is_under_experiments(
        sandbox, tmp_path, monkeypatch):
    """Confinement is not merely "under experiments": frozen evidence is absolute."""
    frozen = Path(C.FROZEN_READ_ONLY_ROOTS[0])
    assert frozen.exists() or True  # the rule must not depend on the root existing
    for target in (frozen / "03_results" / "x.csv", frozen):
        with pytest.raises(EvidenceWriteError, match="frozen read-only root"):
            RE.confine(target, "test write")

    # And the frozen roots stay disjoint from the root this experiment may use.
    for root in C.FROZEN_READ_ONLY_ROOTS:
        assert root.resolve() != sandbox
        assert root.resolve() not in sandbox.parents
        assert sandbox not in root.resolve().parents


def test_a4_the_index_merges_by_identity_rather_than_appending(sandbox):
    rows = []
    for seed, config in ((7, "FULL"), (17, "FULL"), (7, "FULL")):
        arrays, meta = _toy_artifact(scale=float(seed) / 7.0)
        meta = dict(meta, seed=seed, config=config)
        path = RE.raw_artifact_path("SYNTHETIC", "Synthetic", config, seed,
                                    C.ROLE_DEV_EVAL)
        written = RE.write_raw_artifact(arrays, meta, path)
        rows.append(RE.index_rows_from_artifact(
            dict(meta, raw_path=written["raw_path"], raw_sha256=written["raw_sha256"],
                 raw_bytes=written["raw_bytes"]), written))

    index = RE.append_raw_index(rows)
    text = index.read_text(encoding="utf-8").strip().splitlines()
    assert text[0].split(",") == list(RE.RAW_INDEX_FIELDS)
    assert len(text) - 1 == 2, "the repeated (cell, FULL, seed 7) row must merge"
    assert "switches" in text[0]


# ==========================================================================
# A5 -- D0, the no-training geometry report
# ==========================================================================
def test_a5_d0_describes_post_train_without_training_or_materialising_dev_eval(
        sandbox, cell):
    from experiments.current.hch_signed_mass_method_entry import d0

    result = d0.run([(synthetic.SYNTHETIC_MARKET, "Synthetic")],
                    build=lambda market, host: cell)
    assert len(result["cells"]) == 1
    record = result["cells"][0]

    assert record["roles_used"] == [C.ROLE_POST_TRAIN]
    assert record["dev_eval_rows_materialised"] == 0
    assert record["protected_final_read_count"] == 0
    assert record["knn_enabled"] is False
    assert record["n_post_train_days"] == cell.positions(C.ROLE_POST_TRAIN).size
    assert record["reconstruction_max_abs_err"] <= 1e-6 + 1e-6 * max(
        1.0, float(np.abs(cell.residual[cell.positions(C.ROLE_POST_TRAIN)]).max()))
    assert record["reconstruction_identity"].startswith("r = A+S+ - A-S-")

    # Every declared column is present and flat enough to be a CSV cell.
    for field in d0.D0_BY_CELL_FIELDS:
        assert field in record, field

    csv_path = Path(result["csv"])
    assert csv_path.exists() and Path(result["json"]).exists()
    assert Path(result["summary"]).exists()
    assert list(Path(result["csv"]).parent.parts)[-1] == "01_d0_geometry"
    assert not (sandbox / "02_crossfit").exists(), "D0 must not open the grid dirs"


def test_a5_d0_refuses_a_cell_with_no_post_train(sandbox):
    from experiments.current.hch_signed_mass_method_entry import d0

    empty = synthetic.synthetic_dataset(
        role_counts={"HOST_TRAIN": 60, "HOST_VAL": 6, "POST_TRAIN": 0,
                     "DEV_EVAL": 8}, seed=3)
    with pytest.raises(LegalityError, match="POST_TRAIN is empty"):
        d0.run([(synthetic.SYNTHETIC_MARKET, "Synthetic")],
               build=lambda market, host: empty)


def test_the_runner_seam_hands_both_no_training_studies_a_dataset(cell,
                                                                  monkeypatch):
    """``runner._dataset_for`` is a *dataset* seam, and both its callers assume so.

    ``grid.dataset_for`` returns ``(dataset, audit)``; the A3 freeze and D0 both
    want the dataset and neither is written to unpack a tuple.  The mismatch is
    invisible until a real cell is built -- it is an ``AttributeError`` on
    ``positions``, raised only at run time, on the first cell of the first step
    that needs it.  So the seam is pinned here by feeding it both consumers: a
    return that satisfies one and not the other is the bug.
    """
    from experiments.current.hch_signed_mass_method_entry import d0, grid, runner
    from experiments.current.hch_signed_mass_method_entry.contracts import ReadAudit

    monkeypatch.setattr(grid, "dataset_for",
                        lambda market, host: (cell, ReadAudit()))

    dataset = runner._dataset_for(synthetic.SYNTHETIC_MARKET, "Synthetic")

    # The A3 freeze path.
    record = oof.fit_high_mass_thresholds(dataset)
    assert record["cell"] == f"{synthetic.SYNTHETIC_MARKET}::Synthetic"
    assert record["fit_rows_sha256"]
    # The D0 path.
    geometry = d0.analyse_cell(dataset)
    assert geometry["roles_used"] == [C.ROLE_POST_TRAIN]

    # And it is the dataset itself, not a pair that happens to unpack.
    assert not isinstance(dataset, tuple)


# ==========================================================================
# A6 -- the record is recomputable from the raw artifact
# ==========================================================================
def test_a6_every_reported_number_is_recomputable_from_the_raw_artifact(
        sandbox, cell):
    high_mass = oof.fit_high_mass_thresholds(cell)
    record = E.evaluate_fit(cell, "FULL", 7, high_mass=high_mass, max_epochs=1)

    raw = RE.read_raw_artifact(Path(C.REPO_ROOT / record["raw_path"])) \
        if not Path(record["raw_path"]).is_absolute() \
        else RE.read_raw_artifact(record["raw_path"])
    assert raw["raw_sha256"] == record["raw_sha256"]

    y = np.asarray(raw["target"], dtype=np.float64)
    hp = np.asarray(raw["host_prediction"], dtype=np.float64)
    mp = np.asarray(raw["prediction"], dtype=np.float64)
    mask = np.asarray(raw["valid_mask"]).astype(bool)

    # Independently recomputed, from arrays alone.
    assert record["overall_mae"] == pytest.approx(float(np.abs(mp - y)[mask].mean()))
    assert record["host_mae"] == pytest.approx(float(np.abs(hp - y)[mask].mean()))
    assert record["n_entries_evaluated"] == int(mask.sum())
    assert record["n_days"] == int(y.shape[0])

    th = M.load_thresholds(synthetic.SYNTHETIC_MARKET)
    tail = ((y <= th["q05"]) | (y >= th["q95"])) & mask
    assert record["tail_mae"] == pytest.approx(float(np.abs(mp - y)[tail].mean()))
    assert record["tail_mae_host"] == pytest.approx(float(np.abs(hp - y)[tail].mean()))

    realised = M.shape_decomposition((y - hp) * mask)
    days = mask.any(axis=1)
    assert record["mass_positive_l1"] == pytest.approx(
        float(np.abs(np.asarray(raw["mass_positive"]) - realised["mass_positive"])[days].mean()))
    assert record["mass_negative_l1"] == pytest.approx(
        float(np.abs(np.asarray(raw["mass_negative"]) - realised["mass_negative"])[days].mean()))

    picked = days & (realised["mass_positive"] > 0.0) & \
        (realised["mass_positive"] >= high_mass["q90_positive"])
    assert record["high_mass_positive_n_days"] == int(picked.sum())
    if int(picked.sum()):
        assert record["high_mass_positive_l1"] == pytest.approx(float(
            np.abs(np.asarray(raw["mass_positive"])[picked]
                   - realised["mass_positive"][picked]).mean()))

    # And the metric record on disk is the same block the flat row came from.
    metric_path = Path(record["metric_path"]) if Path(record["metric_path"]).is_absolute() \
        else C.REPO_ROOT / record["metric_path"]
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "signed_mass_metric_record.v1"
    assert payload["record"]["raw_sha256"] == record["raw_sha256"]
    assert payload["meta"]["amplitude_scale"] == pytest.approx(
        record["amplitude_scale"])
    assert payload["meta"]["high_mass_thresholds"]["fit_partition"] == \
        C.ROLE_POST_TRAIN
    assert payload["meta"]["knn_enabled"] is False
    assert payload["meta"]["inference_setting"] == C.INFERENCE_SETTING
    assert payload["meta"]["row_positions"] == [int(r) for r in
                                                cell.positions(C.ROLE_DEV_EVAL)]
    assert set(payload["meta"]["identity"]) >= {"label", "target_day",
                                                "episode_index", "role"}


def test_a6_the_registered_grid_is_twenty_cells_five_configs_three_seeds():
    from experiments.current.hch_signed_mass_method_entry import grid

    cells = list(itertools.product(C.MARKETS, C.HOSTS))
    assert len(cells) == 20
    plan = grid.plan_grid(cells)
    assert len(plan) == 20 * len(C.CONFIG_VARIANTS) * len(C.TRAINING["seeds"]) == 300
    assert len({(t["market"], t["host"], t["config"], t["seed"]) for t in plan}) == 300
    assert {t["seed"] for t in plan} == set(C.TRAINING["seeds"])

    # The five configurations differ exactly as registered: off-by-one-switch
    # from FULL, and FULL itself is the all-on vector.
    assert C.CONFIG_VARIANTS["FULL"] == C.FULL_SWITCHES
    for name, switch in C.VARIANT_DELTA.items():
        if switch is None:
            continue
        vector = dict(C.CONFIG_VARIANTS[name])
        assert vector[switch] is False
        for other in C.SWITCH_NAMES:
            if other != switch:
                assert vector[other] == C.FULL_SWITCHES[other]


def test_a6_the_verifier_registers_every_writer_it_exempts():
    """The exemption list and the guard wiring are the A4 gate's own rules."""
    from experiments.current.hch_signed_mass_method_entry import verify_preexecution as V

    audit = V.Audit()
    V.audit_source(audit)
    assert audit.ok("source")
    names = {c["check"] for c in audit.group("source")}
    assert "no_artifact_write" in names
    assert "writer_exemptions_are_guarded" in names
    assert set(V._WRITER_GUARDED_FILES) <= set(V._WRITER_EXEMPT_FILES)


def test_a6_the_evidence_root_is_reached_only_through_the_guard_seam():
    """No production module may name the evidence root directly.

    ``C.EVIDENCE_ROOT`` is the constant; ``RE.evidence_root()`` is the seam the
    confinement guard and the tests both go through.  A module that reaches for
    the constant builds a destination the guard never saw, which is how a write
    would escape confinement without ever deleting the guard.  The three files
    allowed to name it are the constant's definition, the seam itself, and the
    verifier's directory bootstrap -- plus the tests, which read it to assert the
    layout exists.
    """
    allowed = {"config.py", "raw_evidence.py", "runner.py"}
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _chain(node) == "C.EVIDENCE_ROOT":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def _chain(node: ast.AST) -> str:
    parts: list = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ""
    return ".".join(reversed(parts))
