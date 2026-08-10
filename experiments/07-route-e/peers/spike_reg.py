"""B5: Spike Regularization — penalty for large prediction errors on spikes.

Based on: Ponyuenyong et al. 2026, arXiv:2602.05430 (AAAI 2026 WS).
Core idea: add a spike-aware penalty term to the training loss.
The paper uses TSFMs + spike regularization; we simplify to LightGBM + spike penalty.

Implementation: during training on S2, we add extra weight to samples where |residual| > threshold,
effectively penalizing large errors more heavily. At inference, standard prediction.
"""
from __future__ import annotations

import numpy as np
try:
    from .base import PeerBaseline
except ImportError:
    from base import PeerBaseline


class SpikeRegularization(PeerBaseline):
    """Spike-aware correction via weighted L1 regression.

    Samples with |residual| > spike_threshold get extra weight during training.
    """

    name = "SpikeRegularization"

    def __init__(self, spike_quantile: float = 0.95, spike_weight: float = 3.0, seed: int = 0):
        self.spike_quantile = spike_quantile
        self.spike_weight = spike_weight
        self.seed = seed
        self.model = None

    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        resid = y - yhat
        spike_thr = float(np.quantile(np.abs(resid), self.spike_quantile))
        is_spike = np.abs(resid) > spike_thr
        sample_weight = np.where(is_spike, self.spike_weight, 1.0)

        self.model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=self.seed,
            n_jobs=8,
            verbose=-1,
        )
        self.model.fit(Z, resid, sample_weight=sample_weight)

    def predict(self, Z: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        if self.model is None:
            return yhat.copy()
        return yhat + self.model.predict(Z)
