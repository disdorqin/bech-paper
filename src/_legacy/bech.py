"""BECH v0 = BOM-SSC + SCARR : a model-agnostic extreme-price correction head.

BOM-SSC  Bidirectional Occurrence-Magnitude Selective Safe Corrector
---------------------------------------------------------------------
  * two OCCURRENCE heads   : P(y_t < neg_thr)  and  P(y_t > spike_thr)
  * two MAGNITUDE heads    : E[y_t - yhat_t | negative tail] and | positive tail]
  * SELECTIVE gate         : a correction is emitted only when an occurrence
                             probability clears its threshold; otherwise the
                             head is the IDENTITY (delta == 0 exactly), which
                             gives a *provable zero-degradation budget* on the
                             normal regime.
  * SOFT confidence ramp   : delta scales with (p - tau)/(1 - tau); no hard
                             clamp to a fixed floor value.

SCARR  Signed-tail Conformal Action-Risk Routing
------------------------------------------------
  Calibrated on a segment disjoint from both backbone- and corrector-training.
  For each signed branch b independently, pick the largest shrinkage
  lambda_b in [0,1] such that the split-conformal (1-alpha) upper quantile of
  the HARM  H = |y - (yhat + lambda*delta)| - |y - yhat|  stays <= harm_tol.
  If no positive lambda can be certified, the branch ABSTAINS (lambda_b = 0)
  -> the head degenerates to the identity and provably cannot hurt.

Anti-leakage: every corrector feature is a function of (a) the frozen
backbone's own forecast for the target day, (b) information dated <= the
day-ahead cutoff. y_t never enters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# ------------------------------------------------------- corrector features --
def build_corrector_features(X: np.ndarray, names: list[str], yhat: np.ndarray,
                             y: np.ndarray, hour: np.ndarray, dayid: np.ndarray,
                             oos_start: int) -> tuple[np.ndarray, list[str]]:
    """Z for every row; rows < oos_start are backbone-in-sample and must not be
    used for corrector training (caller slices them away).

    Residual-history features use residuals lagged >= 24h, i.e. realised on or
    before day D-1 -- exactly the same information assumption as `price_lag24`
    in the backbone feature set.
    """
    n = len(yhat)
    cols, zn = [], []

    cols.append(yhat); zn.append("yhat")

    # forecast shape within the target day (all 24 values are issued together)
    df = pd.DataFrame(dict(day=dayid, yhat=yhat))
    g = df.groupby("day")["yhat"]
    day_mean = g.transform("mean").to_numpy()
    day_min = g.transform("min").to_numpy()
    day_max = g.transform("max").to_numpy()
    day_std = g.transform("std").fillna(0.0).to_numpy()
    day_rank = g.rank(pct=True).to_numpy()
    cols += [day_mean, day_min, day_max, day_std, day_rank,
             yhat - day_mean, (yhat - day_min) / (day_max - day_min + EPS)]
    zn += ["yhat_daymean", "yhat_daymin", "yhat_daymax", "yhat_daystd",
           "yhat_dayrank", "yhat_dev_daymean", "yhat_daypos"]

    # deviation of the forecast from recent realised level (both cutoff-safe)
    for nm in ("prevday_mean", "prevweek_mean", "prevweek_std", "prevday_min",
               "prevday_max"):
        if nm in names:
            v = X[:, names.index(nm)]
            cols.append(v); zn.append(nm)
            if nm.endswith("mean") or nm.endswith("min") or nm.endswith("max"):
                cols.append(yhat - v); zn.append(f"yhat_minus_{nm}")

    # residual history of the FROZEN backbone (out-of-sample region only)
    resid = np.full(n, np.nan)
    resid[oos_start:] = y[oos_start:] - yhat[oos_start:]
    rs = pd.Series(resid)
    for L in (24, 48, 168):
        cols.append(rs.shift(L).to_numpy()); zn.append(f"resid_lag{L}")
    roll = rs.shift(24).rolling(168, min_periods=24)
    cols += [roll.mean().to_numpy(), roll.std().to_numpy(),
             roll.min().to_numpy(), roll.max().to_numpy()]
    zn += ["resid_w_mean", "resid_w_std", "resid_w_min", "resid_w_max"]

    # calendar + all remaining cutoff-safe exogenous/lag features
    keep = [nm for nm in names if nm.startswith(("fc_", "act_", "hour_", "dow_",
                                                 "mon_", "is_weekend", "price_lag"))]
    for nm in keep:
        cols.append(X[:, names.index(nm)]); zn.append(f"x_{nm}")

    Z = np.column_stack(cols).astype(np.float64)
    avail = np.isfinite(Z).all(axis=1).astype(float)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.column_stack([Z, avail]); zn.append("feat_available")
    return Z, zn


# ------------------------------------------------------------------- heads ---
HARM_RATE_CAP = 0.65          # only used by the `tau_mode="grid_capped"` ablation
MIN_EVENTS_CLF = 10
MIN_EVENTS_REG = 10
BOOT_B = 500                  # block-bootstrap replicates for the efficacy LCB
BOOT_BLOCK = 24               # one day: preserves intra-day error autocorrelation


_BOOT_CACHE: dict = {}


def _boot_index(n: int, B: int, block: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap resampling indices. Cached: they depend only on
    the segment length, not on the quantity being averaged, so the whole
    lambda grid reuses one index matrix."""
    key = (n, B, block, seed)
    if key not in _BOOT_CACHE:
        rng = np.random.default_rng(seed)
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(B, nb))
        offs = np.arange(block)
        _BOOT_CACHE[key] = (starts[:, :, None] + offs[None, None, :]
                            ).reshape(B, -1)[:, :n]
    return _BOOT_CACHE[key]


def block_bootstrap_lcb(g: np.ndarray, alpha: float = 0.10, B: int = BOOT_B,
                        block: int = BOOT_BLOCK, seed: int = 0) -> float:
    """One-sided (1-alpha) lower confidence bound on E[g] for a serially
    correlated series g (here: the per-hour absolute-error REDUCTION on the
    calibration segment, zero on non-routed hours).

    A moving-block bootstrap is used because hourly price-forecast errors are
    strongly autocorrelated within the day; an i.i.d. bootstrap would grossly
    understate the variance and hand out over-confident certificates.
    """
    n = len(g)
    if n == 0:
        return 0.0
    if n <= block:
        return float(g.mean())
    idx = _boot_index(n, B, block, seed)
    return float(np.quantile(g[idx].mean(axis=1), alpha))


def _fit_clf(Z, lab, seed=0):
    import lightgbm as lgb
    pos = int(lab.sum())
    if pos < MIN_EVENTS_CLF or pos == len(lab):
        return None
    w = np.where(lab == 1, len(lab) / (2.0 * pos),
                 len(lab) / (2.0 * max(len(lab) - pos, 1)))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.9, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, n_jobs=8,
                           verbose=-1)
    m.fit(Z, lab, sample_weight=w)
    return m


def _oof_proba(Z, lab, seed=0, folds=3):
    """Time-blocked cross-fitted probabilities: used to build the magnitude
    head's training population out of sample, so the magnitude head sees the
    SAME false-alarm mix it will face at deployment."""
    n = len(lab)
    p = np.zeros(n)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    for i in range(folds):
        te = np.arange(bounds[i], bounds[i + 1])
        tr = np.setdiff1d(np.arange(n), te)
        m = _fit_clf(Z[tr], lab[tr], seed + i)
        p[te] = m.predict_proba(Z[te])[:, 1] if m is not None else 0.0
    return p


def _fit_reg(Z, tgt, seed=0, objective="regression_l1"):
    """Magnitude head.

    IMPORTANT -- the objective is L1, i.e. the head estimates the CONDITIONAL
    MEDIAN of the tail residual, not its conditional mean. Extreme-price
    residual distributions are heavy-tailed and strongly asymmetric (SA1
    negative prices span -0.01 .. -1000 AUD/MWh), so a mean-optimal magnitude
    systematically OVER-corrects and inflates MAE even when it is unbiased on
    average.  Empirically this single change flips the head from net-harmful
    to net-beneficial on the high-negative-price markets.
    """
    import lightgbm as lgb
    if len(tgt) < MIN_EVENTS_REG:
        if not len(tgt):
            return 0.0
        return float(np.mean(tgt) if objective == "regression"
                     else np.median(tgt))
    m = lgb.LGBMRegressor(objective=objective, n_estimators=300,
                          learning_rate=0.05, num_leaves=15,
                          min_child_samples=10, random_state=seed, n_jobs=8,
                          verbose=-1)
    m.fit(Z, tgt)
    return m


def _pred_reg(m, Z):
    if m is None:
        return np.zeros(len(Z))
    if isinstance(m, float):
        return np.full(len(Z), m)
    return m.predict(Z)


# -------------------------------------------------------------------- BECH ---
class BECH:
    """Parameters
    ----------
    neg_thr : absolute negative-price threshold (0 by market convention).
    spike_thr : REPORTING spike threshold (from S1); the positive occurrence
        head is trained with a *segment-relative* label (top `pos_q` of S2) so
        that it stays trainable under extreme-price regime drift, where an
        absolute S1-based threshold can leave almost no positives in S2.
    alpha : miscoverage / confidence level for the action-risk certificate.
    harm_budget_ratio : rho. Tier-2 (safety) certifies that the (1-alpha) upper
        quantile of the PER-POINT harm stays below rho * (branch baseline MAE
        on the routed calibration points). rho = 0 recovers the strict
        "never worse at every point" rule, which is almost never certifiable
        for tail correction and is reported as an ablation.
    tau_mode : "auto"        -- tau maximises the population absolute-error
                                reduction on the held-out slice of S2. All
                                safety is delegated to SCARR. (default)
               "bayes"       -- tau = 0.5, the Bayes-optimal decision threshold
                                for a conditional-MEDIAN correction under
                                absolute loss.
               "grid_capped" -- v0 ablation: grid search with a hard cap on the
                                per-point harm RATE.
        NOTE: with "bayes" the threshold is fixed a priori, so the held-out
        slice S2b is not consumed by any selection step and is legitimately
        available as extra CONFORMAL CALIBRATION data (see `calibrate`). Under
        extreme-price regime drift a 10% calibration window can contain almost
        no tail events, so this is not a cosmetic gain.
    lam_select : "lcb"     -- lambda maximises the bootstrap lower confidence
                             bound on the mean gain among certificate-feasible
                             values (default).
                 "largest" -- v0 ablation: the largest feasible lambda.
    lam_max : upper bound of the shrinkage grid, 1.0 by default. SCARR is thus
        a CONTRACTION: it may only attenuate the action proposed by BOM-SSC,
        never amplify it. Allowing lambda > 1 turns the certificate into a
        magnitude re-scaler and, empirically, produces boundary solutions that
        buy a little MAE at the cost of a worse negative-price sign hit-rate,
        so amplification is disallowed in the default configuration.
    """

    def __init__(self, neg_thr: float = 0.0, spike_thr: float | None = None,
                 alpha: float = 0.10, harm_budget_ratio: float = 0.50,
                 pos_q: float = 0.99, mag_train: str = "events",
                 weight: str = "gate", tau_mode: str = "bayes",
                 lam_select: str = "lcb", lam_max: float = 1.0,
                 mag_objective: str = "regression_l1",
                 reuse_s2b_for_calib: bool = True, seed: int = 0):
        self.mag_objective = mag_objective
        self.tau_mode = tau_mode
        self.lam_select = lam_select
        self.lam_max = lam_max
        self.reuse_s2b = reuse_s2b_for_calib
        self._calib_extra = None
        self.neg_thr = neg_thr
        self.spike_thr = spike_thr
        self.alpha = alpha
        self.rho = harm_budget_ratio
        self.pos_q = pos_q
        self.mag_train = mag_train      # "events" | "events+fa" (ablation)
        self.weight = weight            # "gate" | "prob" | "ramp" (ablation)
        self.seed = seed
        self.tau = {"neg": 1.01, "pos": 1.01}
        self.lam = {"neg": 0.0, "pos": 0.0}
        self.info: dict = {}

    # ---------------- stage 1: corrector training on S2 (backbone frozen) ----
    def fit(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        n = len(y)
        cut = int(n * 0.75)                      # S2a fit heads | S2b pick tau
        a, b = np.arange(cut), np.arange(cut, n)
        if self.spike_thr is None:
            self.spike_thr = float(np.quantile(y, self.pos_q))
        # segment-relative positive label (leak-free: defined on S2 only)
        self.pos_train_thr = float(np.quantile(y, self.pos_q))

        lab = {"neg": (y < self.neg_thr).astype(int),
               "pos": (y > self.pos_train_thr).astype(int)}
        resid = y - yhat

        self.clf, self.reg = {}, {}
        for br in ("neg", "pos"):
            L = lab[br]
            self.clf[br] = _fit_clf(Z[a], L[a], self.seed)
            if self.clf[br] is None:
                self.reg[br] = None
                continue
            ev = a[L[a] == 1]
            if self.mag_train == "events+fa":
                # ablation: also show the head the most likely FALSE ALARMS
                p_oof = _oof_proba(Z[a], L[a], self.seed)
                non = a[L[a] == 0]
                k = min(len(ev), len(non))
                fa = non[np.argsort(-p_oof[L[a] == 0])[:k]] if k > 0 else np.array([], int)
                sel = np.union1d(ev, fa)
            else:
                sel = ev
            self.reg[br] = _fit_reg(Z[sel], resid[sel], self.seed,
                                    objective=self.mag_objective)
            self.info[f"n_{br}_events_S2a"] = int(len(ev))
            self.info[f"n_{br}_magtrain"] = int(len(sel))

        for br in ("neg", "pos"):
            self.tau[br] = self._pick_tau(Z[b], yhat[b], y[b], br)
        # S2b was not used for any selection under the a-priori Bayes gate,
        # so it may join the conformal calibration sample (it directly precedes
        # S3, so chronology -- and the block bootstrap -- stay intact).
        if self.tau_mode == "bayes" and self.reuse_s2b:
            self._calib_extra = (Z[b], yhat[b], y[b])
        self.info["spike_thr_report"] = self.spike_thr
        self.info["pos_train_thr"] = self.pos_train_thr
        return self

    def _prob(self, Z, br):
        m = self.clf.get(br)
        if m is None:
            return np.zeros(len(Z))
        return m.predict_proba(Z)[:, 1]

    def _raw_delta(self, Z, br, p, tau: float | None = None):
        """delta_raw = w(p) * m(Z).

        `weight="prob"` (default) sets w = p, i.e. the correction is the
        OCCURRENCE-probability-weighted expected tail residual
            E[y - yhat | Z] ~= P(event | Z) * E[y - yhat | event, Z],
        which is exactly the occurrence x magnitude decomposition and needs no
        tuning constant.  `weight="ramp"` is the ablation variant
            w = clip((p - tau)/(1 - tau), 0, 1).
        """
        m = _pred_reg(self.reg.get(br), Z)
        if self.weight == "ramp":
            t = self.tau[br] if tau is None else tau
            w = np.clip((p - t) / max(1.0 - t, EPS), 0.0, 1.0)
        elif self.weight == "prob":
            w = p
        else:                     # "gate": apply the full median correction
            w = 1.0
        return m * w

    def _pick_tau(self, Z, yhat, y, br):
        """Select the selective-gate threshold on the held-out slice of S2.

        `tau_mode="auto"` maximises the POPULATION absolute-error reduction
        (sum of per-point gains over routed points == change in segment MAE
        times n, because non-routed points are untouched by construction).

        Crucially this objective carries NO harm-rate constraint. Extreme-price
        correction is intrinsically a "many small losses, few large wins" trade:
        on NEM-SA1 the per-point harm rate sits near 50% at every threshold that
        actually delivers a gain, so a harm-rate cap does not buy safety -- it
        merely pushes tau to 0.98, cutting recall to 8% and discarding ~78% of
        the attainable improvement. Safety is instead delegated to SCARR, which
        controls the *size* of the damage rather than its frequency.
        """
        if self.clf.get(br) is None:
            return 1.01                          # branch disabled
        if self.tau_mode == "bayes":
            return 0.5
        p = self._prob(Z, br)
        other = self._prob(Z, "pos" if br == "neg" else "neg")
        base = np.abs(y - yhat)
        capped = self.tau_mode == "grid_capped"
        best_tau, best_gain, trace = 1.01, 0.0, []
        for tau in np.round(np.arange(0.10, 0.991, 0.02), 3):
            fire = (p > tau) & (p >= other)
            if fire.sum() < 5:
                continue
            d = self._raw_delta(Z, br, p, tau=float(tau))
            new = np.abs(y - (yhat + d))
            gain = float((base[fire] - new[fire]).sum())
            harm_rate = float((new[fire] > base[fire] + 1e-9).mean())
            trace.append((float(tau), int(fire.sum()), gain, harm_rate))
            if gain > best_gain and (not capped or harm_rate <= HARM_RATE_CAP):
                best_tau, best_gain = float(tau), gain
        self.info[f"tau_search_{br}"] = dict(
            mode=self.tau_mode, best_tau=best_tau, best_gain=round(best_gain, 2),
            grid=[dict(tau=t, n_fire=nf, gain=round(g, 2), harm_rate=round(h, 4))
                  for t, nf, g, h in trace[::5]])
        return best_tau

    # ---------------- stage 2: SCARR conformal routing on S3 -----------------
    def calibrate(self, Z: np.ndarray, yhat: np.ndarray, y: np.ndarray):
        """SCARR v2 -- a TWO-TIER action-risk certificate per signed branch.

        Tier-1 EFFICACY (does it actually help?)
            A moving-block bootstrap (1-alpha) LOWER confidence bound on the
            segment-level mean absolute-error reduction must be strictly
            positive. The gain series is defined over EVERY calibration hour
            (zero on non-routed hours), so the bound is a statement about the
            corrector's effect on the deployed segment MAE, not about a
            self-selected subsample.

        Tier-2 SAFETY (how bad can a single hour get?)
            The split-conformal (1-alpha) upper quantile of the per-point harm
            on routed hours must stay below rho * (branch baseline MAE).

        lambda is then chosen to MAXIMISE the Tier-1 bound among feasible
        values -- not simply the largest feasible one. This matters because a
        magnitude head that is well calibrated in-sample can over-shoot out of
        sample (NEM-VIC1), where the mean gain peaks at a small lambda and
        turns negative well before the safety tier would ever bind. lambda is
        therefore exactly the MAE-optimal scalar recalibration of the magnitude
        head, estimated on a segment disjoint from its training data.

        If no lambda > 0 clears both tiers the branch ABSTAINS (lambda = 0) and
        the head degenerates to the identity, which provably cannot hurt.
        """
        if self._calib_extra is not None:
            Ze, he, ye = self._calib_extra
            Z = np.vstack([Ze, Z]); yhat = np.concatenate([he, yhat])
            y = np.concatenate([ye, y])
        base = np.abs(y - yhat)
        n = len(y)
        grid = np.round(np.arange(0.0, self.lam_max + 1e-9, 0.05), 3)
        cal = {"_n_calib": int(n),
               "_s2b_reused": bool(self._calib_extra is not None)}
        for br in ("neg", "pos"):
            if self.clf.get(br) is None:
                self.lam[br] = 0.0
                cal[br] = dict(status="disabled(no training events)", n_fire=0)
                continue
            p = self._prob(Z, br)
            other = self._prob(Z, "pos" if br == "neg" else "neg")
            route = (p > self.tau[br]) & (p >= other)      # signed-tail routing
            nfire = int(route.sum())
            if nfire < 10:
                self.lam[br] = 0.0
                cal[br] = dict(status="abstain(too few calib events)", n_fire=nfire)
                continue
            draw = self._raw_delta(Z, br, p)[route]   # gate applied via `route`
            yy, hh, bb = y[route], yhat[route], base[route]
            budget = float(self.rho * bb.mean())           # branch-relative budget
            k = int(np.ceil((nfire + 1) * (1.0 - self.alpha)))   # conformal rank
            best_lam, best_lcb, rec = 0.0, 0.0, []
            for lam in grid:
                H = np.abs(yy - (hh + lam * draw)) - bb    # per-point harm
                q = float(np.sort(H)[k - 1]) if k <= nfire else np.inf
                g_full = np.zeros(n)                       # population gain series
                g_full[route] = -H
                lcb = block_bootstrap_lcb(g_full, self.alpha, seed=self.seed)
                rec.append((float(lam), q, float(-H.mean()), lcb))
                feasible = (q <= budget) and (lcb > 0)
                if not feasible:
                    continue
                if self.lam_select == "largest":
                    best_lam, best_lcb = float(lam), lcb
                elif lcb > best_lcb:
                    best_lam, best_lcb = float(lam), lcb
            self.lam[br] = best_lam
            cal[br] = dict(
                status="certified" if best_lam > 0 else "abstain(no lambda certified)",
                n_fire=nfire, lam=best_lam, lcb_mean_gain=round(best_lcb, 5),
                conformal_rank=k, branch_base_mae=float(bb.mean()),
                harm_budget=budget, rho=self.rho, alpha=self.alpha,
                grid=[dict(lam=l, harm_q=None if not np.isfinite(q) else round(q, 3),
                           mean_gain=round(g, 4), lcb=round(lc, 5))
                      for l, q, g, lc in rec[::4]])
        self.info["scarr"] = cal
        return self

    # ---------------- stage 3: apply (identity where not routed) -------------
    def apply(self, Z: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, dict]:
        p_neg = self._prob(Z, "neg")
        p_pos = self._prob(Z, "pos")
        delta = np.zeros(len(yhat))
        route = np.zeros(len(yhat), dtype=int)      # 0 none, -1 neg, +1 pos
        fire_neg = (p_neg > self.tau["neg"]) & (p_neg >= p_pos) & (self.lam["neg"] > 0)
        fire_pos = (p_pos > self.tau["pos"]) & (p_pos > p_neg) & (self.lam["pos"] > 0)
        if fire_neg.any():
            d = self._raw_delta(Z, "neg", p_neg)
            delta[fire_neg] = self.lam["neg"] * d[fire_neg]
            route[fire_neg] = -1
        if fire_pos.any():
            d = self._raw_delta(Z, "pos", p_pos)
            delta[fire_pos] = self.lam["pos"] * d[fire_pos]
            route[fire_pos] = 1
        diag = dict(n_fire_neg=int(fire_neg.sum()), n_fire_pos=int(fire_pos.sum()),
                    n_total=int(len(yhat)),
                    fire_rate=float((route != 0).mean()),
                    lam_neg=self.lam["neg"], lam_pos=self.lam["pos"],
                    tau_neg=self.tau["neg"], tau_pos=self.tau["pos"])
        return yhat + delta, diag


def harm_stats(y, base_pred, new_pred) -> dict:
    fired = np.abs(new_pred - base_pred) > 1e-9
    if fired.sum() == 0:
        return dict(n_fired=0, harm_rate=None, mean_gain_on_fired=None,
                    worst_harm=None)
    b = np.abs(y[fired] - base_pred[fired])
    a = np.abs(y[fired] - new_pred[fired])
    return dict(n_fired=int(fired.sum()),
                harm_rate=float((a > b + 1e-9).mean()),
                mean_gain_on_fired=float((b - a).mean()),
                worst_harm=float((a - b).max()))
