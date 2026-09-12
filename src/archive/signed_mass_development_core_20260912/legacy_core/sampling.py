"""Stage 4 (training data) -- rare-mass batch organization.

Design reference:
``docs/current/HCH_SIGNED_MASS_CONTEXT_ROUTED_NETWORK_DESIGN_20260912.md`` section 14.

Large ``A+`` / ``A-`` days are rare, and a uniform batch ordering lets them be
clustered by chance.  Section 14 is explicit about what the first version may do:

* every training sample still appears **once per epoch**;
* rare high-mass days are **spread across batches** rather than clustered randomly;
* **no duplication / oversampling by default**;
* no test-time effect.

That is what :meth:`RareMassSampler.batches` implements: a proportional
stratified *deal*, not a resample.  Oversampling exists only as an explicitly
opt-in mode for the case section 14 permits later -- "only if later evidence
shows severe extreme underfitting may a globally frozen mass-balanced
sampling/weighting rule be evaluated" -- and it is off unless asked for.

This module has no inference-time surface at all: there is deliberately no method
here that transforms a prediction or touches an evaluation batch.  The strata are
computed from the training partition only, and the proportions must be frozen
before the method is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

__all__ = ["RareMassSamplerConfig", "RareMassSampler"]

#: Stratum ids, from rarest to most ordinary.
STRATUM_BOTH = 0
STRATUM_POSITIVE = 1
STRATUM_NEGATIVE = 2
STRATUM_ORDINARY = 3
N_STRATA = 4


@dataclass
class RareMassSamplerConfig:
    """Frozen-before-evaluation sampling contract.  One global rule, no per-market case."""

    enabled: bool = False
    mode: str = "spread"          # "spread" (section 14 default) | "oversample"
    upper_quantile: float = 0.9
    positive_fraction: float = 0.25   # oversample mode only
    negative_fraction: float = 0.25   # oversample mode only
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("spread", "oversample"):
            raise ValueError(f"mode must be 'spread' or 'oversample', got {self.mode!r}")
        if not 0.0 < self.upper_quantile < 1.0:
            raise ValueError("upper_quantile must lie strictly inside (0, 1)")


def _flat(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


class RareMassSampler:
    """Stratified training-batch organization over observed ``A+`` / ``A-`` mass."""

    def __init__(self, config: Optional[RareMassSamplerConfig] = None):
        self.config = config or RareMassSamplerConfig()
        self._labels: Optional[np.ndarray] = None
        self._n_samples: Optional[int] = None

    @property
    def fitted(self) -> bool:
        return self._labels is not None

    def fit(self, mass_positive, mass_negative) -> "RareMassSampler":
        """Assign each training origin to a mass stratum.

        **Training partition only** -- never an evaluation split.  The quantile
        thresholds are computed here and must be frozen before evaluation.
        """
        pos, neg = _flat(mass_positive), _flat(mass_negative)
        if pos.shape != neg.shape:
            raise ValueError("positive and negative mass must have the same shape")
        if pos.size == 0:
            raise ValueError("RareMassSampler needs at least one training origin")

        q = float(self.config.upper_quantile)
        high_pos = pos >= np.quantile(pos, q)
        high_neg = neg >= np.quantile(neg, q)

        labels = np.full(pos.size, STRATUM_ORDINARY, dtype=np.int64)
        labels[high_neg] = STRATUM_NEGATIVE
        labels[high_pos] = STRATUM_POSITIVE
        labels[high_pos & high_neg] = STRATUM_BOTH

        self._labels = labels
        self._n_samples = int(pos.size)
        return self

    def stratum_counts(self) -> dict:
        """Stratum population sizes.  Useful for logging a frozen protocol."""
        if not self.fitted:
            raise RuntimeError("RareMassSampler.fit must run on the training partition first")
        return {name: int((self._labels == sid).sum()) for sid, name in (
            (STRATUM_BOTH, "both"), (STRATUM_POSITIVE, "positive"),
            (STRATUM_NEGATIVE, "negative"), (STRATUM_ORDINARY, "ordinary"))}

    # -- default path: spread, every sample once ---------------------------
    def batches(self, n_batches: int, seed: Optional[int] = None) -> List[np.ndarray]:
        """Partition all training indices into ``n_batches`` stratified batches.

        Every training sample appears exactly once and no sample is duplicated.
        Each stratum is dealt across the batches proportionally, so rare
        high-mass days are spread rather than clustered.
        """
        self._require_enabled()
        if not self.fitted:
            raise RuntimeError("RareMassSampler.fit must run on the training partition first")
        if n_batches < 1:
            raise ValueError("n_batches must be at least 1")

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        buckets: List[List[int]] = [[] for _ in range(n_batches)]

        for stratum in range(N_STRATA):
            members = np.flatnonzero(self._labels == stratum)
            if members.size == 0:
                continue
            rng.shuffle(members)
            base, remainder = divmod(members.size, n_batches)
            # Spread the remainder over evenly spaced batches instead of piling
            # it onto the first few.
            extra = (np.linspace(0, n_batches, remainder, endpoint=False).astype(int)
                     if remainder else np.empty(0, dtype=int))
            cursor = 0
            for b in range(n_batches):
                take = base + int(np.isin(b, extra))
                buckets[b].extend(members[cursor:cursor + take].tolist())
                cursor += take

        # Keep each batch's internal order random so gradient order is not a
        # function of stratum.
        out = []
        for bucket in buckets:
            arr = np.asarray(bucket, dtype=np.int64)
            rng.shuffle(arr)
            out.append(arr)
        return out

    def order(self, seed: Optional[int] = None) -> np.ndarray:
        """One flat training order with rare strata spread through it."""
        return np.concatenate(self.batches(max(self._n_samples or 1, 1), seed=seed))

    # -- opt-in path: oversampling (section 14 permits only after evidence) --
    def oversampled_indices(self, n_samples: int,
                            seed: Optional[int] = None) -> np.ndarray:
        """Draw ``n_samples`` indices with the rare tails over-represented.

        Duplicates samples.  Section 14 allows this only if later evidence shows
        severe extreme underfitting, and only under a globally frozen rule, so it
        refuses to run unless ``config.mode == "oversample"``.
        """
        self._require_enabled()
        if self.config.mode != "oversample":
            raise RuntimeError(
                "oversampling is opt-in; set RareMassSamplerConfig.mode='oversample' "
                "and freeze it before evaluation")
        if not self.fitted:
            raise RuntimeError("RareMassSampler.fit must run on the training partition first")

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        total = self._n_samples
        n_pos = int(round(n_samples * self.config.positive_fraction))
        n_neg = int(round(n_samples * self.config.negative_fraction))
        n_base = max(n_samples - n_pos - n_neg, 0)

        tail_pos = np.flatnonzero((self._labels == STRATUM_POSITIVE)
                                  | (self._labels == STRATUM_BOTH))
        tail_neg = np.flatnonzero((self._labels == STRATUM_NEGATIVE)
                                  | (self._labels == STRATUM_BOTH))
        parts = []
        if n_base:
            parts.append(rng.integers(0, total, size=n_base))
        for count, pool in ((n_pos, tail_pos), (n_neg, tail_neg)):
            if count <= 0:
                continue
            usable = pool if pool.size else np.arange(total)
            parts.append(rng.choice(usable, size=count, replace=True))

        indices = np.concatenate(parts) if parts else np.arange(0)
        rng.shuffle(indices)
        return indices.astype(np.int64)

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise RuntimeError(
                "RareMassSampler is disabled; do not call it in a training loop")
