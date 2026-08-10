"""B2: Vahedi-style — two-stage negative-price prediction.

Based on: Vahedi et al. 2026, IEEE ICCE. "A Hybrid Classification-Regression
Method for Forecasting Negative Electricity Prices".

Core idea:
  Stage 1: binary classifier predicts P(price < 0 | Z)
  Stage 2: conditional regressor predicts E[price | price < 0, Z]
  Final: ŷ = (1-P_neg)*ŷ_base + P_neg*E[price|neg,Z]

This is NOT post-processing — it directly predicts the final price.
We include it as a competitive negative-price baseline.
"""
from __future__ import annotations

import numpy as np
try:
    from .base import PeerBaseline
except ImportError:
    from base import PeerBaseline


class VahediStyle(PeerBaseline):
    """Two-stage classification-regression for negative prices."""

    name = "VahediStyle"

    def __init__(self, neg_thr: float = 0.0, seed: int = 0):
        self.neg_thr = neg_thr
        self.seed = seed
        self.clf = None
        self.reg = None

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        lab = (y < self.neg_thr).astype(int)
        n_pos = int(lab.sum())

        if n_pos < 10 or n_pos == len(lab):
            self.clf = None; self.reg = None
            return

        w = np.where(lab == 1, len(lab)/(2.0*n_pos),
                     len(lab)/(2.0*max(len(lab)-n_pos, 1)))
        self.clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, random_state=self.seed, n_jobs=8, verbose=-1,
        )
        self.clf.fit(Z, lab, sample_weight=w)

        neg_idx = np.where(lab == 1)[0]
        self.reg = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=300, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, random_state=self.seed,
            n_jobs=8, verbose=-1,
        )
        self.reg.fit(Z[neg_idx], y[neg_idx])

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if self.clf is None:
            return yhat.copy()

        p_neg = self.clf.predict_proba(Z)[:, 1]
        y_neg = self.reg.predict(Z)

        hybrid = (1 - p_neg) * yhat + p_neg * y_neg
        return hybrid
