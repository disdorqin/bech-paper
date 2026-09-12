"""Optional similar-window context for the Shape branch.

Retrieval is a *context constructor*, not the correction mechanism.  The
prototype below is fed to the Shape decoder as an extra input; it never becomes
the final correction by itself, so the method cannot collapse into retrieval
correction.

Legality is enforced here rather than assumed:

* a bank entry may only be added for a window that was **already revealed**
  strictly before the origin it will be retrieved for;
* the caller must supply an ``is_permitted`` predicate declaring the partition
  rule (an entry the caller cannot vouch for is refused);
* retrieval is deterministic under ties, ordered by ``(distance, day_index)``;
* the retrieved indices, distances and weights are all returned, so a caller can
  audit exactly which windows were used.

Disabled by default.  Turning it on is a configuration change, not a code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = ["KNNShapeContext", "ShapeMemoryBank", "KNNShapeContextProvider"]


def _as_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class KNNShapeContext:
    """Retrieved shape prototype plus the evidence behind it."""

    positive_prototype: torch.Tensor   # (B, H)
    negative_prototype: torch.Tensor   # (B, H)
    indices: np.ndarray                # (B, k) bank positions, in rank order
    distances: np.ndarray              # (B, k)
    weights: np.ndarray                # (B, k), sums to 1 along k


class ShapeMemoryBank:
    """A bank of revealed historical Shape vectors.

    ``is_permitted`` is mandatory: the bank cannot know which partition an index
    came from, and silently trusting the caller is exactly how held-out labels
    leak into a context window.
    """

    def __init__(self, is_permitted: Callable[[int], bool], *, enabled: bool = False,
                 k: int = 8, temperature: float = 1.0):
        if not callable(is_permitted):
            raise TypeError("is_permitted must be a callable partition predicate")
        if k < 1:
            raise ValueError("k must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.is_permitted = is_permitted
        self.enabled = bool(enabled)
        self.k = int(k)
        self.temperature = float(temperature)
        self._keys: List[np.ndarray] = []
        self._positive: List[np.ndarray] = []
        self._negative: List[np.ndarray] = []
        self._day_index: List[int] = []
        self._horizon: Optional[int] = None

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def horizon(self) -> Optional[int]:
        return self._horizon

    @property
    def day_indices(self) -> np.ndarray:
        return np.asarray(self._day_index, dtype=np.int64)

    def add(self, key, shape_positive, shape_negative, *,
            revealed_index: int, origin_index: int) -> int:
        """Register one revealed window.  Returns its bank position."""
        if int(revealed_index) >= int(origin_index):
            raise ValueError(
                f"window {revealed_index} is not revealed before origin {origin_index}")
        if not self.is_permitted(int(revealed_index)):
            raise ValueError(f"window {revealed_index} is not in a permitted partition")

        pos = _as_numpy(shape_positive).reshape(-1)
        neg = _as_numpy(shape_negative).reshape(-1)
        if pos.shape != neg.shape:
            raise ValueError("positive and negative Shape must have the same length")
        if self._horizon is None:
            self._horizon = int(pos.size)
        elif int(pos.size) != self._horizon:
            raise ValueError("bank entries must share one horizon length")

        self._keys.append(_as_numpy(key).reshape(-1))
        self._positive.append(pos)
        self._negative.append(neg)
        self._day_index.append(int(revealed_index))
        return len(self._keys) - 1

    def query(self, key, *, k: Optional[int] = None,
              temperature: Optional[float] = None) -> KNNShapeContext:
        """Weighted prototype of the ``k`` nearest permitted entries."""
        if not self._keys:
            raise RuntimeError("ShapeMemoryBank is empty; nothing to retrieve")
        k = self.k if k is None else int(k)
        temperature = self.temperature if temperature is None else float(temperature)
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        keys = np.stack(self._keys)                       # (N, d)
        query = _as_numpy(key)
        if query.ndim == 1:
            query = query[None, :]
        if query.shape[-1] != keys.shape[-1]:
            raise ValueError("query key width does not match the bank")

        deltas = keys[None, :, :] - query[:, None, :]
        distances = np.sqrt((deltas ** 2).sum(axis=-1))   # (B, N)
        n_take = min(k, keys.shape[0])

        day = self.day_indices
        order = np.lexsort((np.broadcast_to(day, distances.shape), distances), axis=-1)
        chosen = order[:, :n_take]                        # (B, k), rank order

        chosen_distances = np.take_along_axis(distances, chosen, axis=-1)
        logits = -chosen_distances / temperature
        logits -= logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True)

        positive = np.stack(self._positive)
        negative = np.stack(self._negative)
        positive_prototype = (weights[:, :, None] * positive[chosen]).sum(axis=1)
        negative_prototype = (weights[:, :, None] * negative[chosen]).sum(axis=1)

        return KNNShapeContext(
            positive_prototype=torch.as_tensor(positive_prototype, dtype=torch.float32),
            negative_prototype=torch.as_tensor(negative_prototype, dtype=torch.float32),
            indices=chosen,
            distances=chosen_distances,
            weights=weights,
        )


class KNNShapeContextProvider(nn.Module):
    """Exposes the retrieved prototype as a ``(B, H, 2)`` decoder context.

    Returns ``None`` while disabled, which is the default.  The provider holds no
    parameters and produces no correction: it can only add context to a Shape
    decoder that already exists.
    """

    def __init__(self, bank: ShapeMemoryBank, *, enabled: bool = False):
        super().__init__()
        self.bank = bank
        self.enabled = bool(enabled)

    def forward(self, keys, *, k: Optional[int] = None,
                temperature: Optional[float] = None) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        context = self.bank.query(keys, k=k, temperature=temperature)
        return torch.stack([context.positive_prototype,
                            context.negative_prototype], dim=-1)
