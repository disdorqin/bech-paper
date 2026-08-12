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

    def freeze(self) -> dict:
        """Export full state (including per-hour pools and the use flag)."""
        return {
            "global_pool": self.global_pool,
            "per_hour_pools": {int(k): v for k, v in self.per_hour_pools.items()},
            "use_per_hour": self.use_per_hour,
        }

    @staticmethod
    def from_frozen(state: dict) -> "S1RankReference":
        """Rebuild from a frozen state, preserving per-hour pools."""
        ref = S1RankReference.__new__(S1RankReference)
        ref.global_pool = np.asarray(state["global_pool"], dtype=np.float64)
        ref.per_hour_pools = {int(k): np.asarray(v, dtype=np.float64)
                              for k, v in state.get("per_hour_pools", {}).items()}
        ref.use_per_hour = bool(state.get("use_per_hour", False))
        return ref

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
        """Tie-aware empirical mid-rank (P1-1).

        mid_rank(v) = (count(pool < v) + 0.5 * count(pool == v)) / n

        This is the standard empirical mid-rank convention. A pool of identical
        values maps the identical query exactly to 0.5. No fake epsilon ordering
        within ties. Off-grid values fall back to the step fraction lt/n.
        """
        n = len(pool)
        if n == 0:
            return np.full_like(values, 0.5)
        lt = np.searchsorted(pool, values, side='left')   # count strictly < v
        ge = np.searchsorted(pool, values, side='right')  # count <= v
        eq = ge - lt                                      # count == v (ties)
        rank = (lt + 0.5 * eq) / n
        return np.clip(rank, 0.0, 1.0)
