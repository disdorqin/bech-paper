"""B4: delta-Adapter — ICLR 2026 post-processing.

Based on: Liang et al. 2026, "The Forecast After the Forecast",
ICLR 2026, arXiv:2601.20280.

Core ideas:
  1. Input adapter: learnable feature mask that selects important covariates
     (sparse, horizon-aware)
  2. Output adapter: residual correction on the backbone's prediction
  3. δ-bounds: both adapters produce bounded corrections (O(δ) drift)
  4. Composition: final = backbone(X ⊙ mask) + output_residual

For frozen-backbone setting (backbone can't be re-run with masked inputs):
  - Input adapter becomes: feature selection for the CORRECTOR (not backbone)
  - Output adapter: residual correction via LightGBM
  - Combined: ŷ = ŷ_base + λ * residual_correction(selected_Z)
"""
from __future__ import annotations

import numpy as np
try:
    from .base import PeerBaseline
except ImportError:
    from base import PeerBaseline


class DeltaAdapter(PeerBaseline):
    """Simplified delta-Adapter for frozen backbone setting."""

    name = "DeltaAdapter"

    def __init__(self, delta: float = 0.1, seed: int = 0):
        self.delta = delta       # O(delta) drift bound
        self.seed = seed
        self.selected_idx = None
        self.model = None

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        from sklearn.feature_selection import mutual_info_regression

        resid = y - yhat

        # Input adapter: select top features by MI with residual
        mi = mutual_info_regression(Z, resid, random_state=self.seed)
        n_select = max(5, int(Z.shape[1] * self.delta))
        self.selected_idx = np.argsort(-mi)[:n_select]
        Z_sel = Z[:, self.selected_idx]

        # Output adapter: residual correction on selected features
        self.model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=200, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=self.seed,
            n_jobs=8, verbose=-1,
        )
        self.model.fit(Z_sel, resid)

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if self.model is None:
            return yhat.copy()

        Z_sel = Z[:, self.selected_idx]
        correction = self.model.predict(Z_sel)

        # δ-bound: clip correction magnitude
        std_resid = float(np.std(np.abs(correction))) if len(correction) else 1.0
        correction = np.clip(correction, -self.delta * std_resid, self.delta * std_resid)

        return yhat + correction
