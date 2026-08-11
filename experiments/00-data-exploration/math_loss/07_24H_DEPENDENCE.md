# 24-Hour Dependence Decision

> Part of HCH v2 Derived Loss Math Design v0.1
> **UPDATED 2026-08-11: Audit evidence forces W2 over W1**

## 8.0 Audit Evidence (18 combos, 4 LAGO datasets x 5 backbones)

| Diagnostic | Observed Range | W1 Threshold | Result |
|---|---|---|---|
| Effective rank | **2–5 / 24** | >20 | ❌ W1 fails |
| First eigenvalue ratio | **82–96%** | <40% | ❌ W1 fails |
| Max off-diagonal corr | **0.89–0.99** | <0.3 | ❌ W1 fails |
| ACF lag1 | 0.56–0.78 | <0.3 | ❌ W1 fails |

**Decision: W2 is mandatory.** All 18 combinations show strong day-level dependence.
Hourly conditional independence given `Z` is empirically false.

## 8.1 The Three Options

| Option | Description | Complexity | Risk |
|---|---|---|---|
| **W1** | Hourly conditional FS skew-t; day dependence via encoder/CAGM | Low | Missed joint tail events |
| **W2** | Shared day-level latent scale/skew; low-rank multivariate-t | Medium | Additional parameters; harder optimization |
| **W3** | Copula / energy score / path likelihood | High | Sample-inefficient; fragile to misspecification |

## 8.2 Recommendation: W2 (was W1 before audit)

**Flip from W1 to W2.** The audit shows that W1's conditional independence assumption
is violated on every tested combination. The within-day dependence is not weak residual
correlation — it is a dominant low-rank structure (1-2 factors explain >80% of variance).

## 8.3 W2 Implementation

### 8.3.1 Day-Level Latent Scale Model

Add a shared day-level latent variable `eta_d` with prior `eta_d ~ N(0, 1)` that modulates
the scale of all 24 hourly residuals:

```
R_{d,h} | eta_d ~ Student-t(mu_h, sigma_h * exp(alpha * eta_d), nu)
```

where:
- `mu_h = g_mu(Z_{d,h})` — hour-specific location
- `sigma_h = g_sigma(Z_{d,h})` — hour-specific base scale
- `nu = g_nu(Z_{d,h})` — shared degrees of freedom
- `alpha` — learnable scalar controlling how much the daily latent affects the scale
- `eta_d ~ N(0, 1)` — per-day random effect

**Justification**: A single scale factor captures "good days" (small residuals across all
hours) vs "bad days" (large residuals). The low effective rank (2-5/24) and dominant first
eigenvalue (82-96%) suggest most day-level variation is explained by a shared scale factor.

### 8.3.2 Marginal Likelihood

The day-level NLL requires marginalizing out `eta_d`:

```
L_day = -log integral_{-inf}^{inf} [prod_{h=1}^{24} f_t(R_{d,h} | mu_h, sigma_h*exp(alpha*eta), nu)] * phi(eta) d_eta
```

where `phi` is the standard normal density.

### 8.3.3 Gauss-Hermite Quadrature

Approximate the integral with `K` quadrature nodes:

```
L_day ≈ -log [sum_{k=1}^{K} w_k * prod_{h=1}^{24} f_t(R_{d,h} | mu_h, sigma_h*exp(alpha*eta_k), nu)]
```

where `(eta_k, w_k)_{k=1}^{K}` are Gauss-Hermite nodes and weights.

**Recommended K = 7**: sufficient for normal prior; cost = 7x forward pass per day.

**Log-sum-exp trick for stability**:
```
log_integral = log_sum_exp_{k=1}^{K} [log(w_k) + sum_{h=1}^{24} log f_t(R_{d,h} | ...)]
L_day = -log_integral
```

### 8.3.4 Training Procedure

1. Standardize `eta_d` to have unit variance across days (batch)
2. For each training step, draw K quadrature nodes, compute all 24xK likelihoods
3. Marginalize per day, backprop through the distribution parameters and alpha
4. alpha initialized at 0 (no day-level effect), learned from data

### 8.3.5 Gradient Path

The gradient flows through `alpha`, `mu_h`, `sigma_h`, and `nu` via the log-sum-exp:
```
dL/dtheta = sum_d [1/Z_d * sum_k w_k * S_{d,k} * d/dtheta log S_{d,k}]
```
where `S_{d,k} = prod_h f_t(R_{d,h} | ...)` and `Z_d = sum_k w_k * S_{d,k}`.

This is differentiable and standard for importance-weighted marginalization.

### 8.3.6 Computational Cost

| Component | Without W2 | With W2 (K=7) |
|---|---|---|
| Forward pass per day | 1x | 7x |
| Memory per day | 24 hourly params | 24 x 7 = 168 hourly params |
| Gradient computation | standard | log-sum-exp over 7 nodes |

For S2 with ~400 days, W2 adds ~7x compute — acceptable (~2 min extra training).
For inference (S3/S4), use `eta=0` (prior mode) or MC with K=3.

## 8.4 Why NOT W1 Anymore

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
