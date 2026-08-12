"""DVG — Split-conformal action-value calibration (Eq 32-38).

Math: hch_v2_iah_crps_final_math_core_v0.3 §5

S3-C procedure:
  1. Policy P (candidate, state, retrieval, k, proposal) is FROZEN from S3-M
  2. For each S3-C day t (in temporal order):
     a. Before outcome: produce pi_t = P(X_t), A_hat_t
     b. After outcome: compute true A_t (Eq 21)
     c. Record E_t = A_hat_t - A_t (over-optimism error)
  3. Sort E_{(1)} <= ... <= E_{(n)}
  4. q = E_{(r)} with r = ceil((n+1)*(1-alpha)), or +inf if r=n+1
  5. For new query: LCB = A_hat - q; execute iff LCB > 0
"""
from __future__ import annotations

import numpy as np


class DGVSplitConformal:
    """Split-conformal calibrator for whole-day action value."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.errors: list[float] = []
        self.q: float | None = None
        self.n_calibration: int = 0

    def record_error(self, A_hat: float, A_true: float):
        """Record one calibration day's error (Eq 32)."""
        self.errors.append(A_hat - A_true)
        self.n_calibration += 1

    def compute_quantile(self) -> dict:
        """Compute q_{1-alpha} from stored errors (Eq 33-34).

        Must be called AFTER all calibration days are recorded.
        Modifying self is allowed only during S3-C; before S4, freeze.
        """
        n = self.n_calibration
        if n == 0:
            self.q = float('inf')
            return {"n": 0, "r": 0, "q": float('inf'), "status": "no_calibration_data"}

        r = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        r = min(r, n + 1)

        if r > n:
            self.q = float('inf')
        else:
            sorted_errors = np.sort(self.errors)
            self.q = float(sorted_errors[r - 1])  # 0-indexed

        return {
            "n": n,
            "alpha": self.alpha,
            "r": r,
            "q": self.q,
            "errors_min": float(min(self.errors)),
            "errors_max": float(max(self.errors)),
        }

    def lcb(self, A_hat: float) -> dict:
        """Compute LCB for a new query (Eq 35-36).

        Returns:
            {"A_hat": float, "q": float, "lcb": float,
             "execute": bool, "action": "execute"|"identity"}
        """
        if self.q is None:
            raise RuntimeError("Must call compute_quantile() before lcb()")

        lcb_val = A_hat - self.q
        execute = lcb_val > 0.0

        return {
            "A_hat": A_hat,
            "q": self.q,
            "lcb": lcb_val,
            "execute": execute,
            "action": "execute" if execute else "identity",
        }

    def freeze(self) -> dict:
        """Export frozen calibration state for bundle."""
        return {
            "alpha": self.alpha,
            "errors": list(self.errors),
            "q": self.q,
            "n_calibration": self.n_calibration,
        }

    @staticmethod
    def from_frozen(state: dict) -> "DGVSplitConformal":
        """Restore from frozen state."""
        dvg = DGVSplitConformal(alpha=state["alpha"])
        dvg.errors = list(state["errors"])
        dvg.q = state["q"]
        dvg.n_calibration = state["n_calibration"]
        return dvg
