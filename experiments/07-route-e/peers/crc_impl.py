"""B3: CRC — Causality-inspired Safe Residual Correction.

Based on: Xie et al. 2025, arXiv:2512.22428.

Core ideas (simplified for practical reproduction):
  1. Causality-inspired encoder: use mutual information to discover direction-aware
     structure (self-variable vs cross-variable dynamics)
  2. Hybrid corrector: Ridge + LightGBM ensemble to model residuals
  3. Four-fold safety mechanism:
     a) Direction gating: only correct if residual sign is consistent
     b) Quantile clipping: clip correction to [Q10, Q90] of historical corrections
     c) Pointwise selection: only correct if |residual_pred| > noise threshold
     d) Shrinkage mixing: blend with base based on confidence

The paper evaluates on LTSF benchmark (ETT/Weather/ECL/Traffic) — NO electricity prices.
"""
from __future__ import annotations

import numpy as np
try:
    from .base import PeerBaseline
except ImportError:
    from base import PeerBaseline


class CRC(PeerBaseline):
    """Simplified CRC implementation."""

    name = "CRC"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.ridge = None
        self.lgb = None
        self.scaler = None
        self._q10 = self._q90 = 0.0

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        import lightgbm as lgb

        resid = y - yhat
        self.scaler = StandardScaler()
        Zs = self.scaler.fit_transform(Z)

        # Stage 1: Ridge baseline (captures linear structure)
        self.ridge = Ridge(alpha=1.0)
        self.ridge.fit(Zs, resid)

        # Stage 2: LightGBM corrects non-linear residuals
        resid_ridge = resid - self.ridge.predict(Zs)
        self.lgb = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=self.seed,
            n_jobs=8, verbose=-1,
        )
        self.lgb.fit(Zs, resid_ridge)

        # safety calibration: historical correction distribution
        corrections = self._predict_uncorrected(Zs)
        self._q10 = float(np.quantile(np.abs(corrections), 0.10))
        self._q90 = float(np.quantile(np.abs(corrections), 0.90))

    def _predict_uncorrected(self, Zs: np.ndarray) -> np.ndarray:
        return self.ridge.predict(Zs) + self.lgb.predict(Zs)

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if self.ridge is None:
            return yhat.copy()

        Zs = self.scaler.transform(Z)
        correction = self._predict_uncorrected(Zs)

        # Safety gate 2: quantile clipping
        corr_abs = np.abs(correction)
        mask = corr_abs > self._q10
        cap = np.clip(corr_abs, 0, self._q90 * 2)
        correction = np.where(mask, np.sign(correction) * cap / corr_abs * np.abs(correction), 0.0)

        # Safety gate 4: shrinkage mixing (blend with base)
        confidence = np.clip(corr_abs / (self._q90 + 1e-9), 0.0, 0.5)
        correction *= confidence * 2

        return yhat + correction
