# 24-Hour Dependence Decision

> Part of HCH v2 Derived Loss Math Design v0.1

## 8.1 The Three Options

| Option | Description | Complexity | Risk |
|---|---|---|---|
| **W1** | Hourly conditional FS skew-t; day dependence via encoder/CAGM | Low | Missed joint tail events |
| **W2** | Shared day-level latent scale/skew; low-rank multivariate-t | Medium | Additional parameters; harder optimization |
| **W3** | Copula / energy score / path likelihood | High | Sample-inefficient; fragile to misspecification |

## 8.2 Recommendation: W1 with Guarded Upgrade Path

**Recommend W1 as default.** Rationale:

1. **Composite likelihood is defensible**: Under assumption A2 (conditional
   independence given `Z`), `L_day = -sum_h log f(R_h | Z_h)` is a valid
   composite likelihood. It produces consistent parameter estimates even if
   residual hours are not fully independent, as long as the marginal
   specification of each `f(R_h | Z_h)` is correct (Varin et al. 2011).

2. **Encoder absorbs short-range dependence**: The CAGM day-key encoder and
   the residual-lag features in `Z_t` (lags of 24h, 48h, 168h) provide
   sufficient conditioning for most hour-to-hour dependence.

3. **Empirical evidence direction**: The dossier's residual exploration
   found excess kurtosis of ~10.3 (consistent with Student-t) and skewness
   of ~0.63 (moderate). These are marginal statistics; joint dependence
   may still exist but is second-order relative to the marginal heavy tails.

### 8.2.1 When W1 Is Sufficient

W1 is sufficient when the following diagnostics pass:

| Diagnostic | Criterion | Status |
|---|---|---|
| Day-level NLL vs 24*hourly NLL | Difference < 1% of day-level NLL (S2) | Needs S2 audit |
| 24x24 residual correlation | Max off-diagonal |corr| < 0.3 after conditioning on Z (S2) | Needs S2 audit |
| Eigenvalue spectrum | Effective rank > 20 (S2 24x24 corr matrix) | Needs S2 audit |
| S2-to-S3 stability | Above diagnostics stable across splits | Needs S2 audit |

### 8.2.2 W2 Trigger Conditions

Upgrade to W2 if ANY of:

1. Day-level NLL / 24 consistently differs from mean hourly NLL by > 2 SE
   (joint dependence matters at the likelihood level)
2. 24x24 residual correlation has first eigenvalue explaining > 40% variance
   (strong common factor across hours)
3. Tail events cluster at day level: P(tail event at h+1 | tail event at h)
   is significantly > unconditional rate

### 8.2.3 W2 Design (if needed)

Add a shared day-level latent variable `eta_d ~ N(0, 1)` (or `t_kappa`) that
scales or shifts all 24 hours:

```
R_{d,h} | eta_d ~ FS-t(mu_h + beta * eta_d, sigma_h * exp(gamma * eta_d), nu, gamma_skew)
```

The day-level NLL with marginalization:

```
-log f(R_d) = -log integral prod_h f(R_{d,h} | eta) p(eta) d_eta
```

This integral can be approximated by Gauss-Hermite quadrature (5-7 nodes)
for `eta ~ N(0,1)` or by Monte Carlo sampling during training.

**Cost**: ~5x training computation (one forward pass per quadrature node for
each day). Acceptable for S2/S3-scale data (hundreds of days).

### 8.2.4 Why NOT W3

W3 (copula/path likelihood) is rejected because:

1. Copula selection is itself an open problem with 24 dimensions
2. Energy score requires sampling from the joint predictive, which is expensive
3. The added complexity is not justified unless W1 and W2 are both falsified
   by the S2/S3 diagnostic evidence

## 8.3 Relationship with CAGM's 24h Memory

The CAGM treats a day as a single memory episode: it retrieves similar
historical days based on day-level features (a "day key") and computes
action gains aggregated over all 24 hours of those days.

This is **complementary to, not redundant with**, W1/W2:

- **W1/W2**: Statistical modeling of how 24 residuals co-vary within a day
- **CAGM**: Retrieval-based estimation of how much a correction action
  would have helped on similar days

W1 does NOT model joint 24h structure; it assumes conditional independence
given `Z`. CAGM DOES model joint 24h structure implicitly (by retrieving
days where all 24 hours had similar behavior). The division of labor:

```
W1/CAGM: individual hourly distribution with conditioning -> candidates
CAGM:    day-level retrieval -> action-gain estimates -> route to best action
```

No duplication. CAGM's memory is about action outcomes, not about residual
dependence.
