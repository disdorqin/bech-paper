# Unit and Falsification Tests

> Part of HCH v2 Derived Loss Math Design v0.1

## 11.1 Density Normalization Tests

### U1: Integral = 1
```
Test: integral_{-inf}^{inf} f_FS(r | mu, sigma, nu, gamma) dr == 1
Method: Gauss-Legendre quadrature, n=200 nodes, range [-10*sigma, 10*sigma]
Parameters: (mu, sigma, nu, gamma) from grid:
  mu    in [-2, 0, 2]
  sigma in [0.5, 1.0, 2.0]
  nu    in [3, 5, 10, 30]
  gamma in [0.2, 0.5, 1.0, 2.0, 5.0]
Oracle: abs(integral - 1.0) < 1e-6
Failure: Normalization error in density formula or quadrature range too narrow
```

### U2: gamma=1 -> symmetric Student-t
```
Test: f_FS(r | mu, sigma, nu, 1.0) == t_nu((r-mu)/sigma) / sigma
Method: Compare log-pdfs at r = mu + sigma * [-5, -2, -0.5, 0, 0.5, 2, 5]
Oracle: abs(log f_FS - log f_t) < 1e-10 for all r
Failure: Asymmetry leaking in gamma=1 case
```

### U3: nu -> inf -> skew-normal
```
Test: f_FS(r | 0, 1, inf_approx=1e6, gamma) should match FS skew-normal density
Method: Compare with numerical N(0,1) PDF scaled by FS factors
Oracle: abs(difference) < 1e-4 for tail quantiles up to |z|=5
Failure: Student-t heavy-tail not decaying to normal limit
```

## 11.2 CDF Tests

### U4: CDF ranges
```
Test: F_FS(-inf) == 0, F_FS(+inf) == 1, F_FS monotonically increasing
Method: Evaluate at 1000 equispaced points
Oracle: min_val < 1e-15, max_val > 1-1e-15, all diffs >= 0
```

### U5: CDF vs numerical quadrature of PDF
```
Test: CDF matches cumulative integral of PDF
Method: For each test point, compare analytical CDF vs cumulative trapezoid
       of PDF at fine grid
Oracle: max |CDF_analytic - CDF_numerical| < 1e-5
```

### U6: F(mu) = 1/(gamma^2+1)
```
Test: The mass below the mode equals 1/(gamma^2+1)
Oracle: abs(CDF(mu) - 1/(gamma^2+1)) < 1e-10
```

## 11.3 Moment Tests

### U7: Mean for gamma=1
```
Test: E[R] for gamma=1 equals mu (if symmetric t)
Method: Monte Carlo with 100k samples, nu in [3, 5, 10, 30]
Oracle: |mean - mu| < 0.01 * sigma / sqrt(nu) (within sampling error)
```

### U8: Variance for gamma=1
```
Test: Var(R) = sigma^2 * nu/(nu-2) for gamma=1
Method: MC 100k samples, nu > 2
Oracle: |var - sigma^2 * nu/(nu-2)| < 0.1 * sigma^2
```

### U9: Partial moments vs empirical
```
Test: Analytical M_plus, M_minus match MC estimates
Method: MC 100k samples, compare analytical formula
Oracle: |analytical - MC| / |MC| < 0.02
Parameters: All combinations from U1 grid
```

### U10: pi_neg + pi_pos = 1
```
Test: Occurrence probabilities sum to 1 (for continuous distribution)
Oracle: abs(pi_neg + pi_pos - 1.0) < 1e-10
```

### U11: M_neg + M_pos = E[R]
```
Test: Partial moments sum to full mean
Oracle: abs(M_neg + M_pos - E[R]) < 1e-8
```

## 11.4 Gradient Tests

### U12: Finite-difference gradient check
```
Test: NLL gradient matches finite difference
Method: Central difference, step h=1e-5, for each parameter at 10 random test points
Oracle: |autodiff_grad - fd_grad| / max(1, |autodiff_grad|) < 1e-4
Failure: Numerical instability in autodiff (lgamma, betainc grad)
```

### U13: Gradient at nu near 2
```
Test: NLL gradient does not explode as nu -> 2+
Method: Evaluate grad at nu = 2.001, 2.01, 2.1 with random (r, mu, sigma, gamma)
Oracle: |grad_nu| < 100 (not a NaN or inf)
```

### U14: Gradient at extreme values
```
Test: Gradients for |z| = 10, gamma = 0.01, gamma = 100
Oracle: all finite, no NaN
```

## 11.5 Falsification Tests (Statistics)

### F1: NLL improvement of M1 over M0 is significant
```
Test: On S2-OOF residuals, M1 NLL < M0 NLL with p < 0.05 (paired bootstrap)
Oracle: Bootstrap CI on NLL difference does not include 0
Failure: Asymmetry parameter gamma provides no real improvement
```

### F2: Partial moment candidate is not dominated by unconditional correction
```
Test: On S3, V(partial_moment) > V(simple_mean_correction) under MAE
Oracle: Bootstrap CI on value difference > 0
Failure: The partial-moment approach is strictly worse than naive correction
```

### F3: Identifiability check
```
Test: Hessian of NLL wrt (mu, log_sigma, nu_tilde, log_gamma) is full rank
Method: Compute Hessian at optimum on S2; check condition number
Oracle: condition_number(H) < 1e6 (parameters are jointly identifiable)
Failure: Model is overparameterized relative to data; reduce parameters
```

### F4: Cross-market transfer does not degrade
```
Test: Model trained on source markets (DE, NEM) applied to target (FR, BE)
Oracle: Target NLL <= 1.2 * market-specific NLL (transfer loss <= 20%)
```

## 11.6 Implementation Sanity Tests

### S1: Parameter bounds respected
```
Test: After 100 training steps, all sigma > 0, nu > 2, gamma > 0
Oracle: min(sigma) > EPS, min(nu) > 2.0, min(gamma) > EPS
```

### S2: Loss decreases in training
```
Test: Train NLL model for 100 steps on synthetic FS skew-t data
Oracle: Final NLL < 90% of initial NLL
```

### S3: Synthetic data recovery
```
Test: Fit model to 10000 FS-t(mu=0.5, sigma=1.2, nu=5.0, gamma=0.8) samples
Oracle: Fitted mu in [0.45, 0.55], sigma in [1.15, 1.25],
        nu in [4.5, 5.5], gamma in [0.75, 0.85]
```

### S4: Cross-scale equivariance
```
Test: If residuals are multiplied by c, fitted sigma should scale by c,
      other parameters unchanged
Oracle: |fitted_gamma_rescaled - fitted_gamma_original| < 0.05
        abs(fitted_sigma_rescaled / c - fitted_sigma_original) < 0.05 * sigma
```
