"""Synthetic substrate for integration smoke tests.

The entry protocol allows this landing stage to run *unit, integration and
synthetic smoke* checks only.  A smoke test that ran on real market outcomes
would be a scientific result wearing a smoke test's name, so the generator here
produces a :class:`CellDataset` whose every number is drawn from a known
generative process and whose market name is ``SYNTHETIC``.

The object it returns is a **real** ``CellDataset``, not a stub: the point of the
smoke test is to run the production code path (fold planning, prefix-only
scaling, the core model, the core loss, the core sampler, the core alpha) rather
than a parallel path that could drift from it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import config as C
from .china5_adapter import CellDataset, calendar_tensor
from .contracts import Episode, HostPanel, MarketContract

__all__ = ["SYNTHETIC_MARKET", "synthetic_dataset", "synthetic_contract",
           "synthetic_panel"]

SYNTHETIC_MARKET = "SYNTHETIC"
_BASE_DAY = np.datetime64("2024-01-01", "D")


def _feature_names(n_features: int) -> Tuple[str, ...]:
    return tuple(f"f{i + 1}预测值" for i in range(n_features))


def synthetic_contract(n_features: int = 6,
                       role_counts: Optional[Dict[str, int]] = None) -> MarketContract:
    """A contract shaped exactly like an audited one, with synthetic provenance."""
    counts = dict(role_counts or {"HOST_TRAIN": 120, "HOST_VAL": 12,
                                  "POST_TRAIN": 40, "DEV_EVAL": 12})
    return MarketContract(
        market=SYNTHETIC_MARKET,
        legal_columns=_feature_names(n_features),
        target_column="日前电价",
        forbidden_columns=tuple(C.FORBIDDEN_INPUT_COLUMNS),
        source_path="<synthetic>",
        source_sha256=None,
        contract_path="<synthetic>",
        contract_sha256="0" * 64,
        contract_source="SYNTHETIC",
        role_taxonomy="CANONICAL_CHINA5",
        lead_convention="synthetic_day_ahead",
        role_counts=counts,
        closed_roles=(C.ROLE_PROTECTED_FINAL,),
        horizon=C.HORIZON,
        seq_len=C.SEQ_LEN,
        target_day_price_is_not_an_input=True,
        open_roles=(C.ROLE_HOST_TRAIN, C.ROLE_HOST_VAL, C.ROLE_POST_TRAIN,
                    C.ROLE_DEV_EVAL),
        declared_sealed_days=0,
        contract_read_audit={"target_day_DA_price_input_reads": 0,
                            "realized_future_input_reads": 0,
                            "protected_final_reads": 0},
        source_reader="synthetic",
        source_encoding="utf-8",
    )


def _role_of(i: int, counts: Dict[str, int]) -> str:
    cursor = 0
    for role in (C.ROLE_HOST_TRAIN, C.ROLE_HOST_VAL, C.ROLE_POST_TRAIN,
                 C.ROLE_DEV_EVAL):
        cursor += counts[role]
        if i < cursor:
            return role
    raise IndexError(i)


def synthetic_panel(contract: MarketContract, seed: int = 0) -> HostPanel:
    """A frozen-artifact-shaped panel with a known generative process."""
    rng = np.random.default_rng(seed)
    counts = dict(contract.role_counts)
    n = sum(counts.values())
    hours = np.arange(C.HORIZON)
    # A daily double-peak shape plus a slow weekly drift: the Host can learn the
    # bulk, so the residual carries real structure for the repair to find.
    daily = (60.0
             + 40.0 * np.sin(2 * np.pi * (hours - 6) / 24.0)
             + 25.0 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2))

    levels = np.array([_role_of(i, counts) for i in range(n)])
    y_true = np.zeros((n, C.HORIZON), dtype=np.float32)
    host_pred = np.zeros((n, C.HORIZON), dtype=np.float32)
    offsets = rng.normal(0.0, 12.0, size=n)
    for i in range(n):
        drift = 6.0 * np.sin(2 * np.pi * i / 30.0)
        truth = daily + drift + offsets[i] + rng.normal(0.0, 4.0, size=C.HORIZON)
        # The Host systematically under-reacts on the evening ramp.
        bias = np.where(hours >= 17, -18.0, 5.0)
        host = truth + bias + rng.normal(0.0, 3.0, size=C.HORIZON)
        y_true[i] = truth.astype(np.float32)
        host_pred[i] = host.astype(np.float32)

    timestamp = (_BASE_DAY + np.arange(n).astype("timedelta64[D]"))
    context = np.zeros((n, C.SEQ_LEN), dtype=np.float32)
    for i in range(n):
        context[i] = np.tile(daily, C.SEQ_LEN // C.HORIZON)

    episodes = tuple(
        Episode(index=i, market=contract.market, host="Synthetic",
                label=timestamp[i], target_day=timestamp[i],
                segment="SYNTHETIC", role=_role_of(i, counts),
                run_length=C.HORIZON)
        for i in range(n))
    return HostPanel(
        market=contract.market, host="Synthetic", artifact_path="<synthetic>",
        artifact_sha256="0" * 64, n_episodes=n, episodes=episodes,
        timestamp=timestamp, context=context, y_true=y_true,
        host_pred=host_pred, segment=np.full(n, "SYNTHETIC", dtype=object),
        provenance={"synthetic": True, "seed": int(seed)},
    )


def synthetic_dataset(n_features: int = 6,
                      role_counts: Optional[Dict[str, int]] = None,
                      history: int = C.HISTORY_WINDOWS,
                      seed: int = 0,
                      host: str = "Synthetic",
                      missing_feature_rows: int = 0) -> CellDataset:
    """Build a complete, real :class:`CellDataset` from synthetic numbers.

    ``missing_feature_rows`` blanks the last legal feature on that many rows and
    clears the corresponding mask bits, so the explicit-missingness path is
    exercised without inventing a substitute value.
    """
    contract = synthetic_contract(n_features, role_counts)
    panel = synthetic_panel(contract, seed=seed)
    rng = np.random.default_rng(seed + 1)

    n = panel.n_episodes
    f = n_features
    features = rng.normal(0.0, 1.0, size=(n, C.HORIZON, f)).astype(np.float32)
    # Make the features mildly informative about the residual the Host misses.
    ramp = np.clip(np.arange(C.HORIZON) - 16.0, 0.0, None)
    ramped = np.broadcast_to(ramp, (n, C.HORIZON)).astype(np.float32)
    features[:, :, 0] += 0.05 * ramped
    if f > 1:
        features[:, :, 1] += 0.02 * ramped ** 2

    feature_mask = np.ones((n, C.HORIZON, f), dtype=bool)
    if missing_feature_rows > 0:
        k = min(int(missing_feature_rows), n)
        features[:k, :, -1] = 0.0
        feature_mask[:k, :, -1] = False

    calendar = np.stack([calendar_tensor(d) for d in panel.timestamp], axis=0)

    target = panel.y_true.astype(np.float32)
    host_forecast = panel.host_pred.astype(np.float32)
    residual = (target - host_forecast).astype(np.float32)
    valid_mask = np.isfinite(target) & np.isfinite(host_forecast)

    past_residual = np.zeros((n, history, C.HORIZON), dtype=np.float32)
    history_index = np.full((n, history), -1, dtype=np.int64)
    for i in range(n):
        for w in range(history):
            j = i - history + w
            if j >= 0:
                past_residual[i, w] = residual[j]
                history_index[i, w] = j

    roles: Dict[str, np.ndarray] = {}
    for role in (C.ROLE_HOST_TRAIN, C.ROLE_HOST_VAL, C.ROLE_POST_TRAIN,
                 C.ROLE_DEV_EVAL):
        roles[role] = np.array([i for i in range(n)
                                if _role_of(i, dict(contract.role_counts)) == role],
                               dtype=np.int64)

    return CellDataset(
        market=contract.market, host=host, contract=contract, panel=panel,
        features=features, feature_mask=feature_mask, calendar=calendar,
        host_forecast=host_forecast, valid_mask=valid_mask, target=target,
        residual=residual, past_residual=past_residual,
        history_index=history_index, role_positions=roles,
        provenance={"synthetic": True, "seed": int(seed),
                    "generator": "experiments/current/hch_signed_mass_method_entry/synthetic.py"})
