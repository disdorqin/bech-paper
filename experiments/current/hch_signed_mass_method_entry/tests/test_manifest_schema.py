"""The run manifest matches its published schema, and refuses to lie.

A provenance record that can be written with a negative alpha, an unauthorized
``DEV_EVAL`` count, or a cell that this round forbids is worse than no record at
all, because it looks authoritative.  So the manifest is built here exactly as
``runner`` builds it and validated against ``RUN_MANIFEST_SCHEMA.json``; the
invalid variants below are negative controls that must be rejected.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from experiments.current.hch_signed_mass_method_entry import config as C
from experiments.current.hch_signed_mass_method_entry import runner as R
from experiments.current.hch_signed_mass_method_entry.core_bridge import (
    CORE_EXPORTS_USED)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((PACKAGE_DIR / "RUN_MANIFEST_SCHEMA.json").read_text(
    encoding="utf-8"))


def _record(market="GANSU_DA", host="PatchTST", config="FULL", seed=7):
    return {
        "config": config,
        "market": market,
        "host": host,
        "seed": seed,
        "alpha": 0.8125,
        "oof_correction_sha256": "A" * 64,
    }


def _manifest(records=None, plan=None, configs=None, dev_days=0, authorized=False):
    records = [_record()] if records is None else records
    plan = [{"market": "GANSU_DA", "host": "PatchTST"}] if plan is None else plan
    configs = ["FULL"] if configs is None else configs
    return R._run_manifest(records, plan, configs, dev_days, authorized)


def test_manifest_matches_the_published_schema():
    jsonschema.validate(_manifest(), SCHEMA)


def test_manifest_matches_the_schema_over_the_whole_registry():
    """All 20 cells x 5 configurations x 3 seeds is a legal manifest."""
    plan = [{"market": m, "host": h} for m, h in C.CELLS]
    records = [_record(m, h, name, seed)
               for m, h in C.CELLS
               for name in C.CONFIG_VARIANTS
               for seed in C.TRAINING["seeds"]]
    manifest = _manifest(records, plan, list(C.CONFIG_VARIANTS))
    jsonschema.validate(manifest, SCHEMA)
    assert manifest["n_fits"] == 20 * 5 * 3 == len(manifest["fits"])


def test_manifest_records_the_registries_rather_than_restating_them():
    manifest = _manifest()
    assert manifest["seeds"] == list(C.TRAINING["seeds"])
    assert manifest["config_switches"] == {k: dict(v)
                                           for k, v in C.CONFIG_VARIANTS.items()}
    assert manifest["thresholds_source"].endswith("THRESHOLD_FREEZE.json")
    assert manifest["core"]["exports_used"] == list(CORE_EXPORTS_USED)
    assert manifest["core"]["core_package"] == "src/core"
    assert manifest["oof_rule"] == {
        "min_train_days": C.OOF_MIN_TRAIN_DAYS,
        "min_holdout_days": C.OOF_MIN_HOLDOUT_DAYS,
        "max_folds": C.OOF_MAX_FOLDS,
        "inner_val_fraction": C.OOF_INNER_VAL_FRACTION,
    }


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m.update({"dev_eval_days": -1}), "negative DEV_EVAL count"),
    (lambda m: m["fits"][0].update({"alpha": -0.5}), "negative alpha"),
    (lambda m: m["fits"][0].update({"market": "SHANXI_DA"}), "山西不在注册表"),
    (lambda m: m["fits"][0].update({"config": "NO_KNN"}), "unregistered config"),
    (lambda m: m.pop("core"), "missing core provenance"),
    (lambda m: m.update({"schema": "something_else.v2"}), "wrong schema id"),
    (lambda m: m["training"].update({"lr": 3.0}), "unregistered learning rate"),
    (lambda m: m["config_switches"]["FULL"].update(
        {"shape_semantic_context": False}), "FULL with a component removed"),
])
def test_invalid_manifests_are_rejected(mutate, reason):
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, SCHEMA)


def test_schema_pins_the_registered_training_defaults():
    """The registered optimiser settings are constraints, not documentation."""
    training = SCHEMA["properties"]["training"]
    assert training["properties"]["lr"] == {"const": C.TRAINING["lr"]}
    assert training["properties"]["batch_size"] == {"const": C.TRAINING["batch_size"]}
    assert training["properties"]["max_epochs"] == {"const": C.TRAINING["max_epochs"]}
    assert training["properties"]["patience"] == {"const": C.TRAINING["patience"]}
    assert SCHEMA["properties"]["oof_rule"]["properties"]["max_folds"][
        "maximum"] == C.OOF_MAX_FOLDS


def test_schema_pins_exactly_the_registered_switch_vectors():
    """Every registered configuration's pinned vector equals the registry's.

    Without this the schema could keep validating a switch a registry edit had
    already changed, which is the drift the manifest exists to expose.
    """
    def resolve(body):
        if "$ref" in body:
            name = body["$ref"].rsplit("/", 1)[-1]
            return SCHEMA["definitions"][name]
        return body

    pinned = {name: {k: v["const"] for k, v in resolve(body)["properties"].items()}
              for name, body in SCHEMA["properties"]["config_switches"][
                  "properties"].items()}
    assert set(pinned) == set(C.CONFIG_VARIANTS)
    assert pinned == {k: dict(v) for k, v in C.CONFIG_VARIANTS.items()}
    assert all(all(v.values()) for k, v in pinned.items() if k == "FULL")
    # Each ablation flips exactly one switch, as the registry says.
    for name, vector in pinned.items():
        delta = [k for k in C.SWITCH_NAMES if vector[k] != C.FULL_SWITCHES[k]]
        assert delta == ([] if name == "FULL" else [C.VARIANT_DELTA[name]]), name


def test_schema_admits_only_the_registered_markets_and_hosts():
    assert set(SCHEMA["properties"]["cells"]["items"]["properties"]["market"][
        "enum"]) == set(C.MARKETS)
    assert set(SCHEMA["properties"]["cells"]["items"]["properties"]["host"][
        "enum"]) == set(C.HOSTS)
    assert set(SCHEMA["properties"]["configurations"]["items"]["enum"]) == set(
        C.CONFIG_VARIANTS)
