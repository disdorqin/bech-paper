"""B1: Quantile Correction — classic statistical post-processing baseline.

Trains quantile regression on frozen-base residuals and uses the predicted
residual distribution to shift the base forecast.
"""
from __future__ import annotations

import numpy as np
try:
    from .base import PeerBaseline
except ImportError:
    from base import PeerBaseline


class QuantileCorrection(PeerBaseline):
    """Quantile-based residual correction.

    On S2: regress y-yhat against Z at quantiles {0.10, 0.50, 0.90}.
    On S4: correct base by the median residual prediction if the 90% interval
    is tight enough; otherwise return base unchanged (abstention heuristic).

    This is a fair but weak baseline — it does NOT model occurrence separately
    from magnitude, and it has no certificate layer.
    """

    name = "QuantileCorrection"

    def __init__(self, quantiles=(0.10, 0.50, 0.90), seed: int = 0):
        self.quantiles = sorted(quantiles)
        self.seed = seed
        self.models: dict = {}
        self._lower_q = min(quantiles)
        self._med_q = quantiles[len(quantiles) // 2]
        self._upper_q = max(quantiles)

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        resid = y - yhat
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                random_state=self.seed,
                n_jobs=8,
                verbose=-1,
            )
            m.fit(Z, resid)
            self.models[q] = m

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        lower = self.models[self._lower_q].predict(Z)
        median = self.models[self._med_q].predict(Z)
        upper = self.models[self._upper_q].predict(Z)

        # abstention heuristic: if interval width > global median width, return base
        width = upper - lower
        threshold = float(np.median(np.abs(width)))
        safe = width < threshold * 3.0

        corrected = yhat.copy()
        corrected[safe] = yhat[safe] + median[safe]
        return corrected
