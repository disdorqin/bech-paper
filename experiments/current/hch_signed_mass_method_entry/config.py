"""Registered, frozen configuration for the signed-mass China-5 method harness.

This module owns **every** constant the harness is allowed to branch on.  It
contains no model logic, no data logic and no scientific claim.  The scientific
objects live in ``src/core`` and are never re-implemented here.

Authorities
-----------
* method design     : ``docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md``
* method protocol   : ``docs/current/HCH_SIGNED_MASS_METHOD_EXPERIMENT_ENTRY_PROTOCOL_20260912.md``
* source invariants : ``src/core/DESIGN_CONTRACT.md``
* role vocabulary   : ``experiments/evidence/hch_china_host_breadth_expansion_20260912/NEW_WINDOW_HANDOFF.md``

Nothing in this file may be tuned per market or per Host.  Where a value is a
function of the available data (fold count, batch count) the *rule* is global and
is stated once here.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# 1. Canonical role vocabulary
# --------------------------------------------------------------------------
# The repository's own controlling handoff fixes this vocabulary.  The first two
# blocks belong to the frozen Host; only POST_TRAIN fits the repair method and
# only DEV_EVAL may ever evaluate it.
ROLE_HOST_TRAIN = "HOST_TRAIN"
ROLE_HOST_VAL = "HOST_VAL"
ROLE_POST_TRAIN = "POST_TRAIN"
ROLE_DEV_EVAL = "DEV_EVAL"
ROLE_PROTECTED_FINAL = "PROTECTED_FINAL"

ROLE_ORDER = (
    ROLE_HOST_TRAIN,
    ROLE_HOST_VAL,
    ROLE_POST_TRAIN,
    ROLE_DEV_EVAL,
    ROLE_PROTECTED_FINAL,
)

#: Roles the repair method may fit on.  Exactly one.
FITTING_ROLES = (ROLE_POST_TRAIN,)

#: Roles the repair method may evaluate on, and only after explicit
#: authorization from a later scientific-run command.
EVALUATION_ROLES = (ROLE_DEV_EVAL,)

#: Roles this harness must never materialise.  Structurally excluded: the
#: canonical dataset contracts declare them ``closed_roles``.
SEALED_ROLES = (ROLE_PROTECTED_FINAL,)

#: What *our* artifacts are, as opposed to which slice of data they were fitted
#: or evaluated on.  The vocabulary above partitions the series; this names the
#: producer, so a row in ``RAW_PREDICTION_INDEX.csv`` cannot be read as a Host or
#: comparator prediction that happens to live in the same directory tree.
ROLE_OUR_METHOD = "OUR_METHOD"

#: Revealed-history roles.  Residual windows may be drawn from these because
#: their target days are strictly earlier than the current forecast origin.
REVEALED_HISTORY_ROLES = (
    ROLE_HOST_TRAIN,
    ROLE_HOST_VAL,
    ROLE_POST_TRAIN,
    ROLE_DEV_EVAL,
)

# --------------------------------------------------------------------------
# 2. Markets and Hosts
# --------------------------------------------------------------------------
MARKETS = ("GANSU_DA", "SHANDONG_DA", "SHAANXI_DA", "NINGXIA_DA", "QINGHAI_DA")

HOSTS = ("PatchTST", "TimeMixer", "iTransformer", "LSTM")

#: The 20 registered coordinates, in a fixed order so every artifact is stable.
CELLS = tuple((m, h) for m in MARKETS for h in HOSTS)

# --------------------------------------------------------------------------
# 3. Dataset-contract sources  (provenance adaptation only)
# --------------------------------------------------------------------------
PANEL_EVIDENCE = REPO_ROOT / "experiments/evidence/china5_posthoc_baseline_panel_20260912"
BREADTH_EVIDENCE = REPO_ROOT / "experiments/evidence/hch_china_host_breadth_expansion_20260912"
GANSU_EVIDENCE = REPO_ROOT / "experiments/evidence/hch_unified_da_shape_engineering_20260910"

#: CHINA5 markets publish an audited ``DATASET_CONTRACT.json`` produced by the
#: frozen parent panel.  Keyed by ``physical_market_group``.
CHINA5_CONTRACT_DIR = PANEL_EVIDENCE / "01_dataset_contracts"

#: GANSU_DA has **no** ``DATASET_CONTRACT.json`` anywhere in the repository.  Its
#: controlling byte-level provenance lives in the frozen preflight directory of
#: the upstream evidence root; the adapter reads exactly those files and nothing
#: else.  This is a disclosed legacy contract, not a substitute dataset.
GANSU_LEGAL_STATE_MANIFEST = GANSU_EVIDENCE / "preflight/legal_state_manifest.json"
GANSU_SPLIT_MANIFEST = GANSU_EVIDENCE / "preflight/gansu_da_split_manifest.json"

#: The verbatim stage map from the breadth-stage handoff.  ``README`` discipline:
#: two role vocabularies exist in this repository and are never mixed silently.
GANSU_LEGACY_ROLE_MAP = MappingProxyType(
    {
        "S1": (ROLE_HOST_TRAIN, ROLE_HOST_VAL),
        "DIAG_FIT": (ROLE_POST_TRAIN,),
        "DIAG_EVAL": (ROLE_DEV_EVAL,),
        "S3": (ROLE_PROTECTED_FINAL,),
        "S4": (ROLE_PROTECTED_FINAL,),
    }
)

#: Convenience inverse used by the loader.
GANSU_CANONICAL_TO_NATIVE = MappingProxyType(
    {
        ROLE_HOST_TRAIN: ("S1",),
        ROLE_HOST_VAL: ("S1",),
        ROLE_POST_TRAIN: ("DIAG_FIT",),
        ROLE_DEV_EVAL: ("DIAG_EVAL",),
    }
)

#: Frozen Host prediction roots.  PatchTST/TimeMixer for the four CHINA5 markets
#: come from the frozen parent panel; iTransformer/LSTM come from the frozen
#: breadth expansion.  GANSU's four Hosts come from the upstream evidence root
#: plus the breadth expansion.  Never substitute another Host or cache.
HOST_ARTIFACT_ROOTS = MappingProxyType(
    {
        ("SHANDONG_DA", "PatchTST"): PANEL_EVIDENCE / "02_hosts/SHANDONG/PatchTST",
        ("SHANDONG_DA", "TimeMixer"): PANEL_EVIDENCE / "02_hosts/SHANDONG/TimeMixer",
        ("SHANDONG_DA", "iTransformer"): BREADTH_EVIDENCE / "03_hosts/SHANDONG_DA/iTransformer",
        ("SHANDONG_DA", "LSTM"): BREADTH_EVIDENCE / "03_hosts/SHANDONG_DA/LSTM",
        ("SHAANXI_DA", "PatchTST"): PANEL_EVIDENCE / "02_hosts/SHAANXI/PatchTST",
        ("SHAANXI_DA", "TimeMixer"): PANEL_EVIDENCE / "02_hosts/SHAANXI/TimeMixer",
        ("SHAANXI_DA", "iTransformer"): BREADTH_EVIDENCE / "03_hosts/SHAANXI_DA/iTransformer",
        ("SHAANXI_DA", "LSTM"): BREADTH_EVIDENCE / "03_hosts/SHAANXI_DA/LSTM",
        ("NINGXIA_DA", "PatchTST"): PANEL_EVIDENCE / "02_hosts/NINGXIA/PatchTST",
        ("NINGXIA_DA", "TimeMixer"): PANEL_EVIDENCE / "02_hosts/NINGXIA/TimeMixer",
        ("NINGXIA_DA", "iTransformer"): BREADTH_EVIDENCE / "03_hosts/NINGXIA_DA/iTransformer",
        ("NINGXIA_DA", "LSTM"): BREADTH_EVIDENCE / "03_hosts/NINGXIA_DA/LSTM",
        ("QINGHAI_DA", "PatchTST"): PANEL_EVIDENCE / "02_hosts/QINGHAI/PatchTST",
        ("QINGHAI_DA", "TimeMixer"): PANEL_EVIDENCE / "02_hosts/QINGHAI/TimeMixer",
        ("QINGHAI_DA", "iTransformer"): BREADTH_EVIDENCE / "03_hosts/QINGHAI_DA/iTransformer",
        ("QINGHAI_DA", "LSTM"): BREADTH_EVIDENCE / "03_hosts/QINGHAI_DA/LSTM",
        ("GANSU_DA", "PatchTST"): GANSU_EVIDENCE / "host_foundation/GANSU_DA/PatchTST",
        ("GANSU_DA", "TimeMixer"): GANSU_EVIDENCE / "host_foundation/GANSU_DA/TimeMixer",
        ("GANSU_DA", "iTransformer"): BREADTH_EVIDENCE / "03_hosts/GANSU_DA/iTransformer",
        ("GANSU_DA", "LSTM"): BREADTH_EVIDENCE / "03_hosts/GANSU_DA/LSTM",
    }
)

# --------------------------------------------------------------------------
# 4. Frozen thresholds  (never recomputed per Host or per method)
# --------------------------------------------------------------------------
# The breadth-stage file is the superset: it inherits the four CHINA5 records
# verbatim from the parent panel and completes the one missing GANSU record.
THRESHOLD_FREEZE = BREADTH_EVIDENCE / "00_protocol/THRESHOLD_FREEZE.json"

#: The one global rule that turns a frozen (q05, q95) pair into region masks.
#: ``lower``  : y_true <= q05
#: ``upper``  : y_true >= q95
#: ``tail``   : lower | upper
#: ``normal`` : complement
#: ``negative``: y_true < 0 (reported only where the market actually has them)
#: ``high_spread_day``: (max_h y - min_h y) >= day_spread_p90, a per-day flag
THRESHOLD_RULE = MappingProxyType(
    {
        "lower": "y_true <= q05",
        "upper": "y_true >= q95",
        "tail": "lower | upper",
        "normal": "complement of tail",
        "negative": "y_true < 0",
        "high_spread_day": "(max_h y - min_h y) >= day_spread_p90",
    }
)

# --------------------------------------------------------------------------
# 5. Chronology
# --------------------------------------------------------------------------
# The concrete day counts are already frozen inside each dataset contract
# (``split.counts``) and inside the GANSU split manifest.  These ratios are the
# rule that produced them and are recorded here for provenance only; the adapter
# never re-derives boundaries from them.
SPLIT_BOUNDARY_RATIOS = MappingProxyType(
    {
        "s1_end": 0.5,
        "s2_end": 0.7,
        "s3_end": 0.8,
        "host_val_tail_of_s1": 0.10,
        "dev_eval_tail_of_s2": 0.25,
    }
)

HORIZON = 24
SEQ_LEN = 168

# --------------------------------------------------------------------------
# 6. Calendar basis  (deterministic, known at issue time, identical everywhere)
# --------------------------------------------------------------------------
# No calendar contract exists in the audited dataset contracts, so the harness
# derives a fixed, market-independent basis from the target timestamps alone.
# There is no holiday table: a holiday is not deterministic information this
# repository has frozen, so inventing one would add unregistered information.
CALENDAR_CHANNELS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "doy_sin",
    "doy_cos",
)
N_CALENDAR_FEATURES = len(CALENDAR_CHANNELS)

# --------------------------------------------------------------------------
# 7. Historical residual windows
# --------------------------------------------------------------------------
#: W in ``past_residual (B, W, H)``.  History is the previous W *eligible
#: episodes* in chronological order -- not necessarily calendar-contiguous,
#: because the frozen GANSU split contains genuine calendar gaps.  Every such
#: episode's target day ends at or before the current origin, so the residual is
#: fully revealed.
HISTORY_WINDOWS = 7

# --------------------------------------------------------------------------
# 8. Registered training defaults  (never tuned per market/Host)
# --------------------------------------------------------------------------
TRAINING = MappingProxyType(
    {
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "max_epochs": 50,
        "patience": 8,
        "seeds": (7, 17, 37),
    }
)

LOSS_WEIGHTS = MappingProxyType(
    {
        "repair": 1.0,
        "shape": 1.0,
        "amplitude": 1.0,
        "shape_metric": "wasserstein",
    }
)

# --------------------------------------------------------------------------
# 9. Chronological expanding OOF rule  (one global rule, data-size dependent)
# --------------------------------------------------------------------------
# ``POST_TRAIN`` is split into a mandatory initial training prefix of
# ``OOF_MIN_TRAIN_DAYS`` days followed by ``n_folds`` contiguous holdout blocks.
# Fold ``k`` trains on everything before block ``k`` and predicts block ``k``.
# The first ``OOF_MIN_TRAIN_DAYS`` days are therefore never predicted, which is
# the price of a genuinely expanding (never sliding) window.
OOF_MIN_TRAIN_DAYS = 8
OOF_MIN_HOLDOUT_DAYS = 4
OOF_MAX_FOLDS = 5

#: Fraction of a fold's training prefix held out (chronologically last) for
#: early stopping.  It is a suffix of the training prefix, so no fold statistic
#: ever sees its own holdout block.
OOF_INNER_VAL_FRACTION = 0.2
OOF_MIN_INNER_VAL_DAYS = 2


def oof_fold_count(n_post_train_days: int) -> int:
    """Global fold-count rule. Returns 0 when OOF is impossible (a blocker)."""
    remainder = int(n_post_train_days) - OOF_MIN_TRAIN_DAYS
    if remainder < OOF_MIN_HOLDOUT_DAYS:
        return 0
    folds = remainder // OOF_MIN_HOLDOUT_DAYS
    return int(min(OOF_MAX_FOLDS, max(1, folds)))


# --------------------------------------------------------------------------
# 10. The five registered configurations
# --------------------------------------------------------------------------
# Exactly four removable components.  Each variant flips exactly one switch and
# changes nothing else -- no architecture, optimizer, loss, seed or data change.
# KNN is absent by construction (``ModelConfig.knn_enabled`` is False and the
# core forces ``shape_context_dim = 0`` when KNN is off).
SWITCH_NAMES = (
    "shape_semantic_context",
    "use_tcn",
    "untied_amplitude_heads",
    "rare_mass_sampling",
)

FULL_SWITCHES = MappingProxyType(
    {
        "shape_semantic_context": True,
        "use_tcn": True,
        "untied_amplitude_heads": True,
        "rare_mass_sampling": True,
    }
)


def _variant(**overrides: bool) -> MappingProxyType:
    unknown = set(overrides) - set(SWITCH_NAMES)
    if unknown:
        raise ValueError(f"unknown switch(es): {sorted(unknown)}")
    settings = dict(FULL_SWITCHES)
    settings.update(overrides)
    return MappingProxyType(settings)


CONFIG_VARIANTS = MappingProxyType(
    {
        "FULL": FULL_SWITCHES,
        "NO_SHAPE_CONTEXT": _variant(shape_semantic_context=False),
        "NO_TCN": _variant(use_tcn=False),
        "TIED_AMPLITUDE": _variant(untied_amplitude_heads=False),
        "NO_RARE_MASS": _variant(rare_mass_sampling=False),
    }
)

#: Which single switch each ablation is allowed to differ in.
VARIANT_DELTA = MappingProxyType(
    {
        "FULL": None,
        "NO_SHAPE_CONTEXT": "shape_semantic_context",
        "NO_TCN": "use_tcn",
        "TIED_AMPLITUDE": "untied_amplitude_heads",
        "NO_RARE_MASS": "rare_mass_sampling",
    }
)

#: ``FROZEN_SMALLEST`` is **not** selected here.  It is generated only after the
#: four leave-one-out decisions are adjudicated, by deleting every component
#: that failed.  This harness refuses to name it in advance.
FROZEN_SMALLEST_TEMPLATE = None


def frozen_smallest_switches(supported: dict) -> MappingProxyType:
    """Build the post-adjudication configuration from supported switches only.

    Refuses to run without an explicit adjudication mapping: choosing the
    smallest method is a scientific decision this implementation stage is not
    authorized to make.
    """
    if not isinstance(supported, dict) or set(supported) != set(SWITCH_NAMES):
        raise ValueError(
            "frozen_smallest_switches requires an explicit adjudication mapping "
            f"over exactly {sorted(SWITCH_NAMES)}; none is chosen by this stage")
    return MappingProxyType({k: bool(v) for k, v in supported.items()})


# --------------------------------------------------------------------------
# 10b. Scientific-execution contract  (PART A of the execution prompt)
# --------------------------------------------------------------------------
#: A2 -- the robust Amplitude output scale is ``s_A = median{A_d^+, A_d^- :
#: A_d^pm > 0}`` over one legal fitting prefix, shared by both sign heads.  The
#: floor applies **only** when that set is empty or degenerate (a prefix with no
#: positive residual mass anywhere), which the design names as the sole case
#: where a fixed number may be used.
AMPLITUDE_SCALE_FLOOR = 1.0

#: A3 -- the one global rule that turns a fitted mass distribution into the
#: ``high mass`` subset.  The *value* is fitted per market x Host from that
#: cell's own ``POST_TRAIN``; the *rule* is global and identical everywhere.
HIGH_MASS_QUANTILE = 0.90

#: A3 -- what is recorded when a subset has no members in a partition.  A
#: number is never invented for an empty subset.
NOT_APPLICABLE_N0 = "NOT_APPLICABLE_N0"

#: A5 -- a descriptive similarity diagnostic is computed at a fixed seed so the
#: D0 report is byte-reproducible without being random.
D0_RANDOM_PAIR_SEED = 20260912

#: A5 -- a mass below this fraction of the cell's median mass counts as
#: "near zero" in the D0 report.
D0_NEAR_ZERO_FRACTION = 1e-6

#: G -- the already-frozen comparator tables.  Read-only, never rerun.
BREADTH_RESULTS = BREADTH_EVIDENCE / "03_results"
STRICT_OFFLINE_TABLE = BREADTH_RESULTS / "STRICT_OFFLINE_TABLE.csv"
ONLINE_SUPPLEMENTARY_TABLE = BREADTH_RESULTS / "ONLINE_SUPPLEMENTARY_TABLE.csv"

#: G -- the inference-setting description of this method.  It is NOT online TTA:
#: parameters are frozen before the evaluation partition is opened and no
#: gradient step, update or test-time adaptation occurs on it.
INFERENCE_SETTING = "FROZEN_PARAMETER_CAUSAL_PREQUENTIAL_HISTORY_POSTPROCESSING"
INFERENCE_SETTING_NOTE = (
    "model parameters are frozen on POST_TRAIN before DEV_EVAL is opened; no "
    "gradient/update/TTA occurs during DEV_EVAL; each day may use only "
    "historically revealed residual windows from strictly earlier eligible "
    "origins")

#: The inference-setting taxonomy of the frozen comparator panel, taken from its
#: own protocol (`q0_baseline_config.py` §4, "Measurement-setting separation
#: (never pooled)").  It is reproduced here as a *selection rule* rather than as a
#: convenience: the strict-win gate compares our method against the best
#: **setting-matched** strict offline comparator, and that phrase has exactly one
#: reading in the frozen table.
#:
#: Our method post-processes a frozen Host with parameters fitted on POST_TRAIN
#: and never updated during DEV_EVAL -- that is ``OFFLINE_STATIC_POSTHOC``.
#: ``MatchedDirectResidual`` is registered as ``OFFLINE_STATIC_CONTROL``: a
#: control, a deliberately different setting.  The protocol forbids a
#: "mixed best baseline" claim across settings, so the control is reported beside
#: the comparison and never ranked inside it.  The direction of this rule matters
#: and was fixed before any of our numbers existed: the control is far worse than
#: its own Host in most cells, so admitting it to the rank would *loosen* the
#: gate, not tighten it.
SETTING_MATCHED_STRICT = "OFFLINE_STATIC_POSTHOC"
CONTROL_SETTING = "OFFLINE_STATIC_CONTROL"
#: The Host's own setting.  Named because the Host is excluded from the rank by
#: this value and not merely by its method name: the comparison has to be able to
#: say *why* a numeric row was left out, and "it is a Host" and "it is a Host
#: measurement" are the same exclusion only while the table is honest.
HOST_SETTING = "OFFLINE_HOST"

#: H -- the development paper-candidate gate.  Development gates, not final
#: claims; every threshold is registered here once and never relaxed.
DEVELOPMENT_GATES = MappingProxyType(
    {
        "host_nonworse_min_cells": 19,
        "n_cells": 20,
        "median_overall_mae_gain_pct_vs_host": 3.0,
        "strict_win_min_cells": 14,
        "median_tail_mae_gain_pct_vs_host": 5.0,
        "max_cell_median_normal_relative_harm_pct": 1.0,
    }
)

#: The single legal terminal verdicts of the development experiment.
VERDICT_CANDIDATE = "HCH_SIGNED_MASS_DEVELOPMENT_CANDIDATE_SUPPORTED_READY_FOR_HUMAN_FREEZE"
VERDICT_NOT_SUPPORTED = "HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION"
VERDICT_INVALID = "HCH_SIGNED_MASS_EXECUTION_INVALID"

# --------------------------------------------------------------------------
# 11. Evidence layout
# --------------------------------------------------------------------------
EVIDENCE_ROOT = REPO_ROOT / "experiments/evidence/hch_signed_mass_method_20260912"

#: Raw prediction artifacts and the per-fit metric records.  ``02_crossfit``
#: holds the registered five-configuration grid; ``04_frozen_method`` holds the
#: post-adjudication confirmation run.
RAW_DIR_GRID = "02_crossfit/raw"
METRIC_DIR_GRID = "02_crossfit/metrics"
RAW_DIR_FROZEN = "04_frozen_method/raw"
METRIC_DIR_FROZEN = "04_frozen_method/metrics"

#: The frozen artifact roots that already hold Host and baseline evidence.  The
#: raw-evidence writer may never write inside one of these.  They are listed
#: explicitly -- rather than relying on the general "inside ``EVIDENCE_ROOT``"
#: rule alone -- so the refusal names *which* frozen artifact the destination
#: would have corrupted, and so the harness can assert the list stays disjoint
#: from this experiment's own evidence root.
FROZEN_READ_ONLY_ROOTS = tuple(sorted(
    {PANEL_EVIDENCE, BREADTH_EVIDENCE, GANSU_EVIDENCE}, key=str))

#: Subdirectories reserved by the entry protocol.  They are created empty by the
#: pre-execution verifier and are never populated with scientific results here.
EVIDENCE_SUBDIRS = (
    "00_protocol",
    "01_d0_geometry",
    "02_crossfit",
    "03_component_screen",
    "04_frozen_method",
    "05_baseline_comparison",
    "06_audits",
)

# --------------------------------------------------------------------------
# 12. Forbidden content
# --------------------------------------------------------------------------
#: Column names that must never enter ``RawInputs`` from any market.
FORBIDDEN_INPUT_COLUMNS = ("日前电价", "实时电价", "竞价空间预测值")

#: Source-substring bans.  A grep-level assertion, not the primary defence.
FORBIDDEN_SOURCE_TOKENS = (
    "Gate(",
    "TrustGate",
    "RepairabilityGate",
    "BenefitGate",
    "ConfidenceGate",
    "nn.MultiheadAttention",
    "TransformerEncoder",
    "MixtureOfExperts",
)
