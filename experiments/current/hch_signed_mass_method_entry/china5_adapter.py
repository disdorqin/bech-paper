"""One generic dataset adapter for all five DA markets.

The adapter is driven entirely by each market's audited dataset contract.  There
is **no province branch, no market-specific feature subset and no Host-name
branch** anywhere in this file: the only thing that varies between markets is
``F``, the number of legal forecast-known columns the contract admits, and the
model path accepts any ``F`` by construction.

What the adapter guarantees
---------------------------
* only contract-admitted ``预测值`` columns are read from the market source;
* the forbidden columns (``日前电价``, ``实时电价``, ``竞价空间预测值``) are never
  even opened -- ``usecols`` restricts the read, and a guard refuses if the
  contract's legal list ever grows to include one;
* targets come exclusively from the frozen Host artifact's ``y_true``, and are
  never placed in ``RawInputs``;
* historical residual windows are strictly earlier than the forecast origin;
* every materialised role is one the contract declares open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .contracts import (
    ContractError,
    Episode,
    HarnessError,
    HostPanel,
    LegalityError,
    MarketContract,
    ReadAudit,
    load_market_contract,
)
from .host_prediction_loader import (
    context_rows_of,
    load_host_panel,
    target_rows_of,
)

TIMESTAMP_COLUMN = "时刻"


# --------------------------------------------------------------------------
# Legal feature source
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSource:
    """The audited forecast-known feature table for one market.

    ``values`` holds **only** the contract's legal columns, in contract order.
    """

    market: str
    columns: Tuple[str, ...]
    timestamps: np.ndarray          # (n_rows,) datetime64[s]
    values: np.ndarray              # (n_rows, F) float64
    source_path: str
    source_sha256_recorded: Optional[str]
    source_sha256_observed: str
    reader: str
    encoding: Optional[str]
    n_rows: int
    role_day_span: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.columns)

    def row_of(self, label: np.datetime64) -> Optional[int]:
        return self._index.get(np.datetime64(label, "s"))

    def rows_of(self, labels: np.ndarray) -> Optional[np.ndarray]:
        out = [self._index.get(np.datetime64(x, "s")) for x in labels]
        if any(x is None for x in out):
            return None
        return np.asarray(out, dtype=np.int64)

    _index: Mapping[np.datetime64, int] = field(default_factory=dict, repr=False)


def _guard_columns(contract: MarketContract) -> None:
    illegal = [c for c in contract.legal_columns if c in C.FORBIDDEN_INPUT_COLUMNS]
    if illegal:
        raise LegalityError(
            f"{contract.market}: the contract admits forbidden column(s) {illegal} as "
            "legal features; refusing to build the adapter")
    if contract.target_column in contract.legal_columns:
        raise LegalityError(
            f"{contract.market}: the target column is listed among the legal features")


def load_feature_source(contract: MarketContract,
                        audit: Optional[ReadAudit] = None) -> FeatureSource:
    """Read exactly the contract's legal columns from the market source file."""
    audit = audit if audit is not None else ReadAudit()
    _guard_columns(contract)

    path = C.REPO_ROOT / contract.source_path
    if not path.exists():
        raise ContractError(f"{contract.market}: source not found at {path}")

    usecols: List[str] = [TIMESTAMP_COLUMN, *contract.legal_columns]
    if contract.source_reader == "csv":
        import pandas as pd

        frame = pd.read_csv(path, usecols=usecols,
                            encoding=contract.source_encoding or "utf-8")
    elif contract.source_reader == "xlsx":
        import pandas as pd

        frame = pd.read_excel(path, usecols=usecols)
    else:
        raise ContractError(f"{contract.market}: unsupported reader "
                            f"{contract.source_reader!r}")

    missing = [c for c in usecols if c not in frame.columns]
    if missing:
        raise ContractError(f"{contract.market}: source lacks column(s) {missing}")

    stamps = np.array(frame[TIMESTAMP_COLUMN].to_numpy(), dtype="datetime64[s]")
    if stamps.size and np.unique(stamps).size != stamps.size:
        raise ContractError(f"{contract.market}: duplicate timestamps in source")
    if stamps.size > 1 and not np.all(np.diff(stamps) > np.timedelta64(0, "s")):
        raise ContractError(f"{contract.market}: source timestamps are not increasing")

    values = frame[list(contract.legal_columns)].to_numpy(dtype=np.float64, copy=True)

    observed = _sha256_of_table(stamps, values)
    recorded = contract.source_sha256

    index = {np.datetime64(t, "s"): i for i, t in enumerate(stamps)}

    return FeatureSource(
        market=contract.market,
        columns=tuple(contract.legal_columns),
        timestamps=stamps,
        values=values,
        source_path=contract.source_path,
        source_sha256_recorded=recorded,
        source_sha256_observed=observed,
        reader=contract.source_reader or "",
        encoding=contract.source_encoding,
        n_rows=int(stamps.size),
        role_day_span=contract.role_day_span,
        _index=index,
    )


def _sha256_of_table(stamps: np.ndarray, values: np.ndarray) -> str:
    from .contracts import sha256_bytes

    digest = sha256_bytes(np.ascontiguousarray(stamps, dtype="datetime64[s]").tobytes())
    return sha256_bytes(
        (digest + "|" + str(values.shape) + "|"
         + hashlib_bytes(np.ascontiguousarray(values, dtype=np.float64))).encode("ascii"))


def hashlib_bytes(arr: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(arr.tobytes()).hexdigest().upper()


def _sha256_file_or_none(path: Path) -> Optional[str]:
    from .contracts import sha256_file

    try:
        return sha256_file(path)
    except OSError:
        return None


def verify_source_bytes(contract: MarketContract) -> Dict[str, Any]:
    """Compare the source file's raw bytes against the contract's recorded digest.

    The four CHINA5 contracts publish a ``source_sha256``; GANSU's legacy
    preflight does not.  A recorded digest that disagrees is a hard error, because
    the frozen Host artifacts were produced from those exact bytes.
    """
    path = C.REPO_ROOT / contract.source_path
    observed = _sha256_file_or_none(path)
    recorded = contract.source_sha256.upper() if contract.source_sha256 else None
    if recorded is not None and observed is not None and recorded != observed:
        raise ContractError(
            f"{contract.market}: source bytes changed. contract records "
            f"{recorded}, disk has {observed}. The frozen Host artifacts were "
            "produced from the recorded bytes; refusing to adapt a different file.")
    return {
        "market": contract.market,
        "source_path": contract.source_path,
        "recorded_sha256": recorded,
        "observed_sha256": observed,
        "match": (None if recorded is None else recorded == observed),
        "attested": recorded is not None,
    }


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------
def calendar_tensor(target_day: np.datetime64) -> np.ndarray:
    """Deterministic, market-independent calendar basis, shape ``(H, C)``.

    The channels are the delivery hour's time-of-day, weekday and day-of-year for
    each horizon step.  They are known at issue time and identical in mechanism
    for every market.

    Convention note: the delivery hour of step ``h`` is the ``h``-th hour of the
    episode's target day under **both** lead conventions in this repository, so
    ``hour_of_day == h`` needs no market branch.  The two conventions differ in
    where the episode *label* sits (CHINA5 anchors it at ``D 01:00`` under
    hour-ending labels, GANSU at ``D 00:00``), and that one-hour difference
    exactly cancels the difference in interval semantics.  The calendar is
    therefore unambiguous without resolving which reading GANSU intends.

    No holiday channel is included: no holiday table is frozen in this
    repository, and inventing one would add unregistered information.
    """
    day = np.datetime64(target_day, "D")
    dow = int((day - np.datetime64("1970-01-01", "D")) // np.timedelta64(1, "D")) % 7
    doy = int((day - day.astype("datetime64[Y]").astype("datetime64[D]"))
              // np.timedelta64(1, "D")) + 1

    out = np.zeros((C.HORIZON, C.N_CALENDAR_FEATURES), dtype=np.float32)
    for h in range(C.HORIZON):
        out[h, 0] = math.sin(2.0 * math.pi * h / 24.0)
        out[h, 1] = math.cos(2.0 * math.pi * h / 24.0)
        out[h, 2] = math.sin(2.0 * math.pi * dow / 7.0)
        out[h, 3] = math.cos(2.0 * math.pi * dow / 7.0)
        out[h, 4] = 1.0 if dow >= 5 else 0.0
        out[h, 5] = math.sin(2.0 * math.pi * (doy - 1) / 365.25)
        out[h, 6] = math.cos(2.0 * math.pi * (doy - 1) / 365.25)
    return out


# --------------------------------------------------------------------------
# Cell dataset
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CellDataset:
    """Every legal tensor for one ``market x Host`` cell, aligned episode-wise.

    ``residual`` and ``target`` are **target-derived** arrays.  They exist for the
    loss and the metrics and are never part of ``RawInputs``; ``to_raw_inputs``
    is the only sanctioned way to build model input and it cannot see them.
    """

    market: str
    host: str
    contract: MarketContract
    panel: HostPanel
    features: np.ndarray        # (N, H, F) float32
    feature_mask: np.ndarray    # (N, H, F) bool
    calendar: np.ndarray        # (N, H, C) float32
    host_forecast: np.ndarray   # (N, H) float32
    valid_mask: np.ndarray      # (N, H) bool
    target: np.ndarray          # (N, H) float32
    residual: np.ndarray        # (N, H) float32   = y_true - host_pred
    past_residual: np.ndarray   # (N, W, H) float32
    history_index: np.ndarray   # (N, W) int64, -1 during warm-up
    role_positions: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_episodes(self) -> int:
        return int(self.host_forecast.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[-1])

    @property
    def key(self) -> str:
        return f"{self.market}::{self.host}"

    def positions(self, role: str) -> np.ndarray:
        if role in C.SEALED_ROLES:
            raise LegalityError(f"positions({role})")
        try:
            return self.role_positions[role]
        except KeyError:
            raise LegalityError(f"{self.key}: role {role} is not materialised") from None

    def numeric_block(self, rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(B, H, 1+F)`` absolute numeric block and its availability mask.

        The core's scaler is fitted on exactly this block, so the harness must be
        able to produce it for a training prefix without the core exposing an
        internal.  The definition mirrors ``DeterministicFeatureBuilder.build``:
        ``cat([host, features])`` zeroed by ``cat([valid, feature_mask])``.

        The mirror is **verified, not trusted**: ``tests/test_scaler_prefix.py``
        asserts it is bit-identical to
        ``builder(base_scaler=None).build(raw).base_tokens[..., :1+F]`` for every
        market, so any drift in the core's canonicalization fails the suite rather
        than silently changing what the scaler sees.
        """
        idx = np.asarray(rows, dtype=np.int64)
        numeric = np.concatenate(
            [self.host_forecast[idx][..., None], self.features[idx]], axis=-1)
        mask = np.concatenate(
            [self.valid_mask[idx][..., None], self.feature_mask[idx]], axis=-1)
        return numeric.astype(np.float32), mask

    def to_raw_inputs(self, rows: Sequence[int]) -> Any:
        """Build ``src.core.RawInputs`` for a set of episode positions.

        This is the only function in the package that produces model input, and
        its body mentions only the six legal ``RawInputs`` fields.  ``target`` and
        ``residual`` are unreachable from here by construction.
        """
        import torch

        from .core_bridge import attr

        idx = np.asarray(rows, dtype=np.int64)
        RawInputs = attr("RawInputs")
        return RawInputs(
            host=torch.as_tensor(self.host_forecast[idx], dtype=torch.float32),
            valid_mask=torch.as_tensor(self.valid_mask[idx], dtype=torch.bool),
            forecast_features=torch.as_tensor(self.features[idx], dtype=torch.float32),
            forecast_feature_mask=torch.as_tensor(self.feature_mask[idx],
                                                  dtype=torch.bool),
            calendar=torch.as_tensor(self.calendar[idx], dtype=torch.float32),
            past_residual=torch.as_tensor(self.past_residual[idx], dtype=torch.float32),
        )


def build_cell_dataset(market: str, host: str,
                       contract: Optional[MarketContract] = None,
                       panel: Optional[HostPanel] = None,
                       source: Optional[FeatureSource] = None,
                       audit: Optional[ReadAudit] = None,
                       strict: bool = True) -> CellDataset:
    """Assemble the full legal tensor set for one cell.

    ``strict`` mirrors the contracts' own ``non_finite_legal_feature = hard error``
    policy.  Turning it off keeps the missingness explicit in ``feature_mask`` and
    does not substitute anything; it exists so the missingness test can exercise
    the mask path without contradicting the contract.
    """
    contract = contract or load_market_contract(market)
    audit = audit if audit is not None else ReadAudit()
    panel = panel or load_host_panel(market, host, contract=contract, audit=audit)
    source = source or load_feature_source(contract, audit=audit)

    if source.columns != tuple(contract.legal_columns):
        raise ContractError(f"{market}: feature source columns drifted from contract")

    n = panel.n_episodes
    f = source.n_features
    w = C.HISTORY_WINDOWS

    features = np.zeros((n, C.HORIZON, f), dtype=np.float32)
    feature_mask = np.zeros((n, C.HORIZON, f), dtype=bool)
    calendar = np.zeros((n, C.HORIZON, C.N_CALENDAR_FEATURES), dtype=np.float32)
    history_index = np.full((n, w), -1, dtype=np.int64)

    first_label = np.datetime64(panel.timestamp[0], "s")
    last_label = np.datetime64(panel.timestamp[-1], "s")

    for i, episode in enumerate(panel.episodes):
        labels = target_rows_of(episode.label)
        rows = source.rows_of(labels)
        if rows is None:
            raise ContractError(
                f"{market}::{host}: episode {episode.index} ({episode.label}) has "
                "target hours absent from the source; the frozen artifact and the "
                "source disagree and the harness will not interpolate")
        block = source.values[rows]
        features[i] = block.astype(np.float32)
        feature_mask[i] = np.isfinite(block)
        calendar[i] = calendar_tensor(episode.target_day)
        if i >= w:
            history_index[i] = np.arange(i - w, i, dtype=np.int64)

    values_flat = source.values
    if not np.isfinite(values_flat).all():
        bad = int((~np.isfinite(values_flat)).sum())
        if strict:
            raise ContractError(
                f"{market}: {bad} non-finite legal feature value(s). Every CHINA5 "
                "contract declares non_finite_legal_feature = hard error, and "
                "imputation is explicitly forbidden. The mask records the gap; the "
                "harness refuses to fill it.")
        audit.__dict__.setdefault("non_finite_legal_feature_values", bad)

    host_pred = panel.host_pred[:, :, 0].astype(np.float32)
    target = panel.y_true[:, :, 0].astype(np.float32)

    valid_mask = np.isfinite(host_pred) & np.isfinite(target)
    if not valid_mask.all():
        raise ContractError(
            f"{market}::{host}: the frozen artifact contains non-finite host "
            "predictions or targets; the contracts exclude incomplete days, so this "
            "means the artifact is not the one the contract describes")

    residual = target - host_pred

    past_residual = np.zeros((n, w, C.HORIZON), dtype=np.float32)
    for i in range(n):
        src = history_index[i]
        if src[0] >= 0:
            past_residual[i] = residual[src]

    role_positions: Dict[str, np.ndarray] = {}
    for role in C.ROLE_ORDER:
        if role in C.SEALED_ROLES:
            continue
        role_positions[role] = panel.role_indices(role)

    return CellDataset(
        market=market,
        host=host,
        contract=contract,
        panel=panel,
        features=features,
        feature_mask=feature_mask,
        calendar=calendar,
        host_forecast=host_pred,
        valid_mask=valid_mask,
        target=target,
        residual=residual,
        past_residual=past_residual,
        history_index=history_index,
        role_positions=role_positions,
        provenance=_cell_provenance(contract, panel, source, first_label, last_label),
    )


def _cell_provenance(contract: MarketContract, panel: HostPanel,
                     source: FeatureSource, first_label: np.datetime64,
                     last_label: np.datetime64) -> Dict[str, Any]:
    return {
        "market": contract.market,
        "host": panel.host,
        "n_features": source.n_features,
        "legal_feature_names": list(source.columns),
        "feature_names_source": contract.contract_path,
        "contract_source": contract.contract_source,
        "contract_sha256": contract.contract_sha256,
        "role_taxonomy": contract.role_taxonomy,
        "lead_convention": contract.lead_convention,
        "host_artifact_path": panel.artifact_path,
        "host_artifact_array_sha256": panel.artifact_sha256,
        "host_artifact_container_sha256": panel.provenance.get("container_sha256"),
        "host_freeze_manifest": panel.provenance.get("freeze_manifest"),
        "source_path": source.source_path,
        "source_recorded_sha256": source.source_sha256_recorded,
        "source_table_sha256": source.source_sha256_observed,
        "first_episode_label": str(first_label),
        "last_episode_label": str(last_label),
        "calendar_channels": list(C.CALENDAR_CHANNELS),
        "history_windows": C.HISTORY_WINDOWS,
        "history_rule": (
            "the previous W eligible episodes in the artifact's chronological "
            "order; eligibility requires a complete local window, so calendar gaps "
            "are skipped rather than back-filled, and every window ends at or "
            "before the current origin"),
    }


# --------------------------------------------------------------------------
# Causality, asserted rather than assumed
# --------------------------------------------------------------------------
def assert_history_causal(dataset: CellDataset) -> Dict[str, Any]:
    """Independently re-derive that no history window touches its own episode.

    Returns the worst-case margin in hours for the manifest.
    """
    panel = dataset.panel
    labels = np.array([np.datetime64(e.label, "s") for e in panel.episodes])
    worst = None
    for i in range(dataset.n_episodes):
        origin = labels[i] - np.timedelta64(1, "h")
        for j in dataset.history_index[i]:
            if j < 0:
                continue
            # The source episode's last target hour ends at its own label.
            last_target_end = labels[int(j)] + np.timedelta64(0, "h")
            margin = int((origin - last_target_end) // np.timedelta64(1, "h"))
            if worst is None or margin < worst:
                worst = margin
            if margin < 0:
                raise LegalityError(
                    f"{dataset.key}: history episode {int(j)} ends after the origin "
                    f"of episode {i}; the window is not revealed")
    return {
        "history_windows": C.HISTORY_WINDOWS,
        "min_margin_hours": worst,
        "episodes_with_full_history": int((dataset.history_index[:, 0] >= 0).sum()),
        "warm_up_episodes": int((dataset.history_index[:, 0] < 0).sum()),
    }


def assert_fitting_history_complete(dataset: CellDataset) -> Dict[str, int]:
    """Every fitting/evaluation episode must have a full history window.

    No zero-fill and no warm-up exclusion is needed for any of the five markets;
    this asserts that rather than assuming it, so a future re-freeze that broke it
    would fail loudly instead of silently feeding zeroed windows to the model.
    """
    out: Dict[str, int] = {}
    for role in (C.ROLE_POST_TRAIN, C.ROLE_DEV_EVAL):
        rows = dataset.positions(role)
        if rows.size == 0:
            out[role] = 0
            continue
        incomplete = int((dataset.history_index[rows, 0] < 0).sum())
        if incomplete:
            raise LegalityError(
                f"{dataset.key}: {incomplete} {role} episode(s) lack a full "
                f"{C.HISTORY_WINDOWS}-episode history window; the harness has no "
                "sanctioned fill for a warm-up window")
        out[role] = int(rows.size)
    return out


# --------------------------------------------------------------------------
# Independent alignment witness
# --------------------------------------------------------------------------
def verify_target_alignment(market: str, host: str,
                            contract: Optional[MarketContract] = None,
                            panel: Optional[HostPanel] = None) -> Dict[str, Any]:
    """Prove the frozen artifact's targets really are the market's DA prices.

    **Verification only.**  This function opens the source's ``日前电价`` column to
    compare it against the artifact's ``y_true``.  It is never called from the
    training path, its reads are counted under a separate counter, and the values
    it reads cannot reach ``RawInputs``: it returns a diagnostic dictionary and
    nothing else.

    It also re-derives the forecast-origin convention from the artifact's own
    ``context`` field, which is the 168 realised hours ending at the origin.  Both
    checks are independent of the code path under test.
    """
    contract = contract or load_market_contract(market)
    panel = panel or load_host_panel(market, host, contract=contract)

    path = C.REPO_ROOT / contract.source_path
    cols = [TIMESTAMP_COLUMN, contract.target_column]
    if contract.source_reader == "csv":
        import pandas as pd

        frame = pd.read_csv(path, usecols=cols,
                            encoding=contract.source_encoding or "utf-8")
    else:
        import pandas as pd

        frame = pd.read_excel(path, usecols=cols)

    target_series = np.asarray(frame[contract.target_column].to_numpy(), dtype=np.float64)
    stamps = np.array(frame[TIMESTAMP_COLUMN].to_numpy(), dtype="datetime64[s]")
    index = {np.datetime64(t, "s"): i for i, t in enumerate(stamps)}

    target_err = 0.0
    context_err = 0.0
    missing_target = 0
    missing_context = 0
    for episode in panel.episodes:
        rows = [index.get(x) for x in target_rows_of(episode.label)]
        if any(r is None for r in rows):
            missing_target += 1
        else:
            observed = target_series[rows]
            expected = panel.y_true[episode.index, :, 0].astype(np.float64)
            target_err = max(target_err, float(np.max(np.abs(observed - expected))))

        crows = [index.get(x) for x in context_rows_of(episode.label)]
        if any(r is None for r in crows):
            missing_context += 1
        else:
            observed = target_series[crows]
            expected = panel.context[episode.index, :, 0].astype(np.float64)
            context_err = max(context_err, float(np.max(np.abs(observed - expected))))

    return {
        "market": market,
        "host": host,
        "n_episodes": panel.n_episodes,
        "target_rows_missing": missing_target,
        "context_rows_missing": missing_context,
        "max_abs_target_error": target_err,
        "max_abs_context_error": context_err,
        "target_alignment_ok": (missing_target == 0 and target_err < 1e-3),
        "origin_rule_ok": (missing_context == 0 and context_err < 1e-3),
        "origin_rule": "target rows are L..L+23h; origin is L-1h; context is the "
                       "168 realised hours ending at the origin",
        "read_class": "VERIFICATION_ONLY_NOT_A_MODEL_INPUT",
    }
