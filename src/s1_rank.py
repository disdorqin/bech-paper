"""S1 Rank Reference — deterministic continuous rank context (Eq 5).

Math: hch_v2_iah_crps_final_math_core_v0.3 §2.1

u = R_P(z0, hour, market, target) ∈ [0,1]
R is a continuous mid-rank interpolator built from S1 out-of-fold host predictions.
No learnable parameters. No state loss. No target data in construction.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class S1RankReference:
    """Continuous rank interpolator from S1 host predictions.

    Builds per-hour rank pools from S1 z0 values. Returns u ∈ [0,1].
    If no reference pool exists for a given context, returns 0.5 (neutral).
    """

    def __init__(self, s1_z0_values: np.ndarray,
                 s1_hours: Optional[np.ndarray] = None,
                 s1_market_ids: Optional[np.ndarray] = None,
                 s1_target_ids: Optional[np.ndarray] = None):
        """Build rank pools from S1 host predictions.

        Args:
            s1_z0_values: [N] flattened z0 values from S1 host predictions
            s1_hours: [N] hour-of-day (0-23), optional for per-hour pools
            s1_market_ids: [N] market identifiers, optional
            s1_target_ids: [N] target identifiers, optional
        """
        self.global_pool = np.sort(s1_z0_values.astype(np.float64))
        self.per_hour_pools = {}
        self.use_per_hour = s1_hours is not None

        if self.use_per_hour:
            for h in range(24):
                mask = s1_hours == h
                if mask.sum() >= 10:
                    self.per_hour_pools[h] = np.sort(
                        s1_z0_values[mask].astype(np.float64))
                else:
                    self.per_hour_pools[h] = self.global_pool

    def __call__(self, z0: np.ndarray,
                 hours: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute continuous rank u ∈ [0,1] for each z0 value.

        Uses per-hour pool if available, otherwise global pool.
        Linear interpolation between sorted pool values.
        """
        z0 = np.asarray(z0, dtype=np.float64)
        result = np.full_like(z0, 0.5, dtype=np.float64)

        if self.use_per_hour and hours is not None:
            hours = np.asarray(hours, dtype=np.int32)
            for h in range(24):
                mask = hours == h
                if mask.sum() == 0:
                    continue
                pool = self.per_hour_pools.get(h, self.global_pool)
                result[mask] = self._interpolate_rank(z0[mask], pool)
        else:
            result = self._interpolate_rank(z0, self.global_pool)

        return result.astype(np.float32)

    @staticmethod
    def _interpolate_rank(values: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """Continuous mid-rank: fraction of pool values <= each value.
        Linear interpolation between adjacent pool values.
        """
        n = len(pool)
        if n == 0:
            return np.full_like(values, 0.5)
        idx = np.searchsorted(pool, values, side='right')
        # Linear interpolation
        lo = np.clip(idx - 1, 0, n - 1)
        hi = np.clip(idx, 0, n - 1)
        frac = np.where(idx < n,
                        (values - pool[lo]) / np.maximum(pool[hi] - pool[lo], 1e-12),
                        0.0)
        frac = np.clip(frac, 0.0, 1.0)
        rank = (idx + frac) / (n + 1)
        return np.clip(rank, 0.0, 1.0)
