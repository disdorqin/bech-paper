"""Peer baseline common interface.

All baselines receive frozen backbone predictions and must:
  fit(Z, yhat, y)       # train on S2
  predict(Z, yhat)      # generate corrected predictions

Z: corrector feature matrix (same as HCH)
yhat: frozen base forecast
y: ground truth (S2 training; invisible at inference)
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class PeerBaseline(ABC):
    """Abstract peer baseline."""

    name: str = "PeerBaseline"

    @abstractmethod
    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        ...


class Identity(PeerBaseline):
    """B0: identity baseline (= frozen base)."""
    name = "Identity"

    def fit(self, Z, yhat, y):
        pass

    def predict(self, Z, yhat):
        return yhat.copy()
