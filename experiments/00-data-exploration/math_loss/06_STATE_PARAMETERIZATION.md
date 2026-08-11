# State and Cross-Market Parameterization

> Part of HCH v2 Derived Loss Math Design v0.1

## 7.1 State Definition

The continuous state `s_t` at time `t` conditions the FS skew-t parameters:

```
(mu_t, log_sigma_t, nu_tilde_t, log_gamma_t) = g_theta(Z_t, s_t)
```

where:
- `Z_t`: external features (calendar, exogenous forecasts, etc.)
- `s_t`: continuous state scalar, derived from past-visible information

### 7.1.1 State Construction

Recommended state: the **host prediction robust rank** within a rolling window:

```
s_t = F_hat(host_pred_t),
where F_hat is the empirical CDF of host predictions on past visible data
(excluding S4 and any future data from S2/S3).
```

Alternatively, a low-dimensional combination:
```
s_t = [rank(host_pred_t), rolling_residual_scale_t, hour_sin, hour_cos]
```

### 7.1.2 State Role in Parameters

State enters `g_theta` as an additional feature dimension. Specifically:

- `mu(Z, s)`: When host is predicting very high (`s ≈ 1`), the residual may
  be biased upward (host underestimates extreme prices). `mu` should increase.
- `log_gamma(Z, s)`: Skew may be directionally different at extreme states.
  When `s ≈ 0` (very low host pred), downward residuals may dominate -> gamma < 1.
  When `s ≈ 1` (very high host pred), upward residuals may dominate -> gamma > 1.
- `nu(Z, s)`: Tail heaviness may vary with state. At normal states, residuals
  are tighter (large nu). At extreme states, residuals fan out (small nu).
- `sigma(Z, s)`: Scale variation (heteroskedasticity) is expected.

### 7.1.3 No Discrete Regime Classifier

Do NOT pre-segment into Low/Normal/High bins. The state is continuous,
and the network learns smooth relationships. This avoids:
- Hard thresholds that create discontinuities
- Sparse bins that prevent learning
- Regime definitions that don't generalize across markets

## 7.2 Cross-Market Architecture

### 7.2.1 Shared Backbone

A single `g_theta` network processes `(Z_t, s_t)` for all markets. Parameters
shared across markets capture:
- Calendar effects (hour, day-of-week, month)
- Residual autocorrelation structure
- General "extreme state -> heavier tail" relationship

### 7.2.2 Market-Specific Parameters

Each market `m` gets:
- `bias_mu[m]`, `bias_log_sigma[m]`: market-specific location/scale offsets
- `bias_log_gamma[m]`: market-specific skew direction

Optionally, market-specific `bias_nu[m]` for tail heaviness.

These are initialized at 0 and learned from data. The shared backbone captures
universal geometry; the biases capture market-specific levels.

### 7.2.3 Knowledge Transfer Mechanism

```
mu_final = mu_shared + alpha * bias_mu[m]
sigma_final = sigma_shared * exp(alpha * bias_log_sigma[m])
gamma_final = gamma_shared * exp(alpha * bias_log_gamma[m])
```

where `alpha in [0, 1]` controls transfer strength:
- `alpha = 0`: pure shared model (full transfer)
- `alpha = 1`: full market-specific offsets
- Trainable `alpha` per market or fixed at 0.5

## 7.3 Negative-Price Knowledge Transfer

### 7.3.1 The Transfer Mechanism

Negative prices `Y < 0` create residuals `R < 0` whose magnitude may differ
from ordinary downward residuals (`R < 0` when `Y >= 0`). The proposed transfer:

1. **Shared heavy-tail geometry**: The `nu` parameter (degrees of freedom)
   is shared between negative-price and ordinary downward residuals. The
   mechanism: "extreme events follow similarly heavy tails on both sides."

2. **Market-specific skew**: The `gamma` parameter is market-specific, allowing
   markets with frequent negative prices to have gamma < 1 (left-heavy) while
   markets without have gamma >= 1.

3. **Adaptive low as bridge**: In markets without physical negative prices,
   the "low-price adaptive low" state (`s ≈ 0`) teaches the network about
   downward residual behavior. When applied to a market with negative prices,
   the shared backbone transfers this knowledge.

### 7.3.2 When Transfer Fails (Negative Transfer)

**Counter-example**: If negative prices arise from a fundamentally different
mechanism than ordinary low prices (e.g., must-run constraints on nuclear
vs. solar oversupply), then the residual structure for `R < 0 | Y < 0` differs
qualitatively from `R < 0 | Y > 0`. Sharing `nu` or `gamma` would degrade both.

**Diagnostic**: Compare FS-ST parameter estimates on S2 for:
- Set A: `{t: Y_t < 0, R_t < 0}` (physical negative price hours)
- Set B: `{t: Y_t > 0, R_t < 0, s_t < 0.2}` (adaptive-low hours)

If `|nu_A - nu_B| > 2 * max(SE_A, SE_B)` or `|gamma_A - gamma_B| > 2 * max(...)`,
do not share parameters between A and B.

## 7.4 What Up/Down Share

| Parameter | Shared? | Rationale |
|---|---|---|
| `g_theta` backbone | **Yes** | Common feature extraction; reduces parameters |
| `nu` (df) | **Yes** | Tail heaviness is a property of the forecaster+market pair |
| `sigma` (scale) | **Yes** | Scale should be symmetric; skew handles asymmetry |
| `mu` (mode) | **Yes** | Center of residual distribution |
| `gamma` (skew) | **Yes** | Single skew parameter determines Full distribution asymmetry |
| Day key for CAGM | **Yes** | Same historical days are retrieved for all actions |
| Historical action gains | **No** (action-specific) | Gains differ per action |

The key insight: Up and Down DO NOT have separate distribution heads.
A single FS skew-t with `gamma > 1` makes the upper tail heavier (Up-tilted)
and `gamma < 1` makes the lower tail heavier (Down-tilted). The partial
moment decomposition `M_+, M_-` naturally yields both candidates from
the same distribution.

### 7.4.1 When Separate Distribution Heads May Be Needed

If the empirical audit finds that upper and lower tails have different
degrees of freedom (one is Pareto-like, the other is moderate-t), then:

1. Try M2: two-piece t with `nu_lower != nu_upper`
2. Or maintain single-nu FS-t but add a separate outlier-modeling head
   for the most extreme tail
3. Only escalate to M3 or M4 if M2 cannot fit.

## 7.5 Monotonicity and Smoothness Constraints

Soft constraints on the state-to-parameter mapping:

1. **Monotonicity of mu**: `mu(s)` should be non-decreasing in `s`. When host
   predicts higher prices, residual mean should shift upward (or stay flat).
   Enforce via: `mu(s) = mu_base + softplus(w_mu) * s`.

2. **Smoothness of sigma**: `sigma(s)` should not have sharp transitions.
   Enforce via L2 penalty on second differences of piecewise-linear sigma(s).

3. **Shared nu across states**: Optionally, `nu` does not depend on `s` at all
   (constant across states, varying only per market). This is a strong but
   defensible simplification that reduces identifiability problems.
