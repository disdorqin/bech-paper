"""HCH v2 baselines — Identity, Residual-L1, QuantileResidual-LGBM.

Per spec §9:
  Residual-L1: LightGBM L1 on residuals, no selectivity gate
  QuantileResidual-LGBM: q=0.1/0.5/0.9, S3-frozen width cutoff

PIR / delta-Adapter: limited_reimplementation (see official_adapters.py).
Labels MUST include implementation_status per addendum §11.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


class Identity:
    name = "Identity"

    def fit(self, Z, yhat, y):
        pass

    def predict(self, Z, yhat):
        return yhat.copy()


class ResidualL1:
    """LightGBM L1 regression on frozen-base residuals. No selectivity gate."""

    name = "ResidualL1"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = None

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        resid = y - yhat
        self.model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=self.seed,
            n_jobs=8, verbose=-1,
        )
        self.model.fit(Z, resid)

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if self.model is None:
            return yhat.copy()
        return yhat + self.model.predict(Z)


class QuantileResidualLGBM:
    """q=0.1/0.5/0.9 residual quantile regression.

    Width cutoff calibrated on S3, frozen for S4. Uses q=0.5 for main
    correction; abstains when prediction interval too wide.
    """

    name = "QuantileResidualLGBM"

    def __init__(self, quantiles=(0.10, 0.50, 0.90), seed: int = 0):
        self.quantiles = sorted(quantiles)
        self.seed = seed
        self.models: dict = {}
        self._width_cutoff: float | None = None

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        resid = y - yhat
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=300,
                learning_rate=0.05, num_leaves=31, min_child_samples=20,
                random_state=self.seed, n_jobs=8, verbose=-1,
            )
            m.fit(Z, resid)
            self.models[q] = m

    def calibrate(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        lower = self.models[min(self.quantiles)].predict(Z)
        upper = self.models[max(self.quantiles)].predict(Z)
        widths = upper - lower
        self._width_cutoff = float(np.median(np.abs(widths))) * 3.0

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if not self.models:
            return yhat.copy()
        lower = self.models[min(self.quantiles)].predict(Z)
        median = self.models[self.quantiles[len(self.quantiles) // 2]].predict(Z)
        upper = self.models[max(self.quantiles)].predict(Z)
        width = upper - lower
        if self._width_cutoff is not None:
            safe = width < self._width_cutoff
        else:
            safe = width < float(np.median(np.abs(width))) * 3.0
        corrected = yhat.copy()
        corrected[safe] = yhat[safe] + median[safe]
        return corrected


ALL_BASELINES = {
    "Identity": Identity,
    "ResidualL1": ResidualL1,
    "QuantileResidualLGBM": QuantileResidualLGBM,
}
