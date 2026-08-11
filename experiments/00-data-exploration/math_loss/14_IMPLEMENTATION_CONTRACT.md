# Final Implementation Contract [REVISED v0.2: M0 + W2]

> Part of HCH v2 Derived Loss Math Design v0.2
> **Updated per audit: M1 deferred (skew unstable), W2 mandatory (24h dependence)**

## 15.1 Unique Recommended Design

### 15.1.1 Distribution Module

**Choice**: M0 (state-conditional symmetric Student-t) with W2 (day-level latent scale).

**Network architecture**:
```
Input: Z_t (host_pred, residual_lags, calendar, exogenous -- from build_corrector_features)
       s_t (host_pred robust rank)
Processing: 3-layer MLP [d_in -> 128 -> 64 -> 32], ReLU activation
Output heads:
  mu_raw       : Linear(32, 1)
  log_sigma_raw: Linear(32, 1)
  nu_tilde     : Linear(32, 1)

  mu    = mu_raw
  sigma = softplus(log_sigma_raw) + 1e-6
  nu    = 2.0 + softplus(nu_tilde)
```

**W2 Day-level latent**:
- Global parameter `alpha` (scalar, learnable, init=0)
- Per-day `eta_d ~ N(0,1)` — random effect
- Effective scale: `sigma_eff_{d,h} = sigma_{d,h} * exp(alpha * eta_d)`
- Marginalization: Gauss-Hermite quadrature, K=7 nodes

### 15.1.2 Training

```
Phase: S2 only
Loss: -log sum_k w_k prod_h f_t(R_{d,h} | mu_h, sigma_h*exp(alpha*eta_k), nu)
      via log-sum-exp (K=7 quadrature nodes)
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Batch: 1 day = 24 hours (or mini-batch of 8-16 days)
Epochs: Early stop on validation NLL (S2b holdout)
Seeds: 3 seeds
```

### 15.1.3 Candidate Generation

Theta distribution (unchanged):
```
pi_neg, pi_pos = CDF at 0 (symmetric t)
M_neg, M_pos   = partial moments (analytic formulas)
Delta_down = -lambda_neg * M_neg
Delta_up   =  lambda_pos * M_pos
```

### 15.1.4 What NOT to Implement
- M1 (FS skew-t) until skew stability verified
- W3 (copula/path) -- not needed
- M2, M3, M4 unless M0+W2 fails

### 15.1.5 Checkpoint Order
```
1. Student-t NLL, CDF, partial moments (fs_skewt.py) passes all U-tests
2. Synthetic data recovery (M0 sym-t)
3. W2 marginalization passes numerical test (K=7 vs K=31 integral comparison)
4. M0+W2 trains and converges on S2 residuals
5. M0+W2 vs M0-W1 (no day latent) ablation on S3
6. Candidate generator produces M_neg, M_pos
7. S3 SCARR calibration
8. DVG receives distribution features
9. End-to-end smoke on S2/S3 (NO S4)
```
