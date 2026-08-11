# Final Implementation Contract

> Part of HCH v2 Derived Loss Math Design v0.1
>
> **This document is the single source of truth for code implementation.
> Do NOT let code AI choose between multiple mathematical branches.**

## 15.1 Unique Recommended Design

### 15.1.1 Distribution Module

**Choice**: M1 (state-conditional FS skew-t) with shared `nu` across Up/Down.

**Network architecture**:

```
Input: Z_t (host_pred, residual_lags, calendar, exogenous — from build_corrector_features)
       s_t (host_pred robust rank)
Processing: 3-layer MLP [d_in -> 128 -> 64 -> 32], ReLU activation
Output heads:
  mu_raw       : Linear(32, 1)
  log_sigma_raw: Linear(32, 1)
  nu_tilde     : Linear(32, 1)
  log_gamma_raw: Linear(32, 1)

  mu    = mu_raw
  sigma = softplus(log_sigma_raw) + 1e-6
  nu    = 2.0 + softplus(nu_tilde)
  gamma = exp(log_gamma_raw)
```

### 15.1.2 Training

```
Phase: S2 only
Loss: -log f_FS(r | mu, sigma, nu, gamma)
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Batch: 256 hourly samples (not days)
Epochs: Early stop on validation NLL (S2b holdout, contiguous days)
Seeds: 3 seeds, report per-seed and aggregate
```

### 15.1.3 Candidate Generation

```
From fitted distribution parameters, compute:
  pi_neg, pi_pos = occurrence probabilities (CDF at 0)
  M_neg, M_pos   = partial moments (using analytic formulas)
  m_neg, m_pos   = side-conditional means (= -M_neg/pi_neg for neg)

Candidate actions:
  Delta_down = -lambda_neg * M_neg  (M_neg already <= 0)
  Delta_up   =  lambda_pos * M_pos  (M_pos already >= 0)

where lambda comes from S3 SCARR calibration.
```

### 15.1.4 S3 Calibration

```
Existing SCARR protocol:
  - Bootstrap LCB on MAE reduction
  - Conformal safety constraint
  - Output lambda_neg, lambda_pos in [0, 1]
  - NO changes to this module
```

### 15.1.5 CAGM-DVG Interface

```
Distribution head outputs the following per hour for DVG:
  - pi_neg, pi_pos
  - M_neg, M_pos
  - m_neg, m_pos
  - sigma_aleatoric = sigma * sqrt(Var_FS(gamma, nu))

DVG receives these as ADDITIONAL features alongside existing CAGM features.
DVG's decision rule is unchanged: argmax(V_down, 0, V_up).
```

## 15.2 What NOT to Implement

1. Do NOT implement M2, M3, M4 unless M1 is falsified by S2/S3 diagnostics.
2. Do NOT implement W2 unless W1 dependence diagnostics fail.
3. Do NOT add a separate classification head for negative price detection --
   the distribution head provides pi_neg naturally.
4. Do NOT add timing loss, copula loss, or energy score.
5. Do NOT add conformal prediction or prediction intervals to the distribution
   head -- that is handled by SCARR at the S3 level.

## 15.3 Required Outputs from Implementation

| Output | Description | Format |
|---|---|---|
| `fs_skewt.py` | Pure function module: logpdf, cdf, partial_moments | Python |
| `test_fs_skewt.py` | Unit tests per Section 11 U1-U14 | Python (pytest) |
| `distribution_head.py` | PyTorch nn.Module for the g_theta network | Python |
| `train_distribution_head.py` | S2 training script with logging | Python |
| `candidate_generator.py` | Post-training candidate computation | Python |
| `ablation_results/` | M0 vs M1 comparison per Section 12 | CSV + plots |

## 15.4 Checkpoint Order

```
1. fs_skewt.py passes all U-tests (U1-U14)
2. Synthetic data recovery test passes (S3)
3. M0 (sym-t) trains and converges on S2 residuals
4. M1 (FS skew-t) trains and converges on S2 residuals
5. M0 vs M1 ablation comparison on S3
6. IF M1 > M0: candidate generator produces M_neg, M_pos
7. S3 calibration applies SCARR lambda
8. DVG receives distribution features
9. End-to-end smoke on S2/S3 (NO S4)
10. Hand off to architecture review
```

## 15.5 Freeze Contract

After training on S2:
- Distribution network weights are FROZEN (no gradient updates during S3 or S4)
- S3 only reads distribution parameters as fixed values; never backpropagates
  through the distribution head
- The distribution head outputs are treated as deterministic for CAGM/DVG
