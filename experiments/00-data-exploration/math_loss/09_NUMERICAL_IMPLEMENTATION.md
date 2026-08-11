# Numerical Implementation

> Part of HCH v2 Derived Loss Math Design v0.1

## 10.1 Stable Computation Primitives

### 10.1.1 Standard Student-t log-PDF

```
def student_t_logpdf(x, nu):
    """Log-pdf of standard Student-t(nu) at x."""
    half_nu = 0.5 * nu
    half_nu1 = 0.5 * (nu + 1.0)

    log_const = gammaln(half_nu1) - gammaln(half_nu) \
                - 0.5 * log(max(nu, EPS)) - 0.5 * LOG_PI

    log_kernel = -half_nu1 * log1p(x * x / max(nu, EPS))

    return log_const + log_kernel
```

### 10.1.2 FS Skew-t log-PDF

```
def fs_skewt_logpdf(r, mu, sigma, nu, gamma):
    z = (r - mu) / sigma
    s = where(z < 0, gamma, 1.0 / gamma)
    sz = s * z

    # Normalization constant
    log_norm = LOG_2 - log(sigma) - log(gamma + 1.0/gamma)

    return log_norm + student_t_logpdf(sz, nu)
```

### 10.1.3 FS Skew-t NLL (batched)

```
def fs_skewt_nll(r, mu, sigma, nu, gamma):
    """
    r:     (B,) observed residuals
    mu:    (B,) location parameters
    sigma: (B,) scale parameters (>0)
    nu:    (B,) degrees of freedom (>2 recommended)
    gamma: (B,) skew parameters (>0)
    Returns: (B,) negative log-likelihood per sample
    """
    # Clamp parameters to valid ranges
    sigma = clip(sigma, EPS, 1e6)
    nu = clip(nu, 2.0 + EPS, 100.0)
    gamma = clip(gamma, EPS, 1e6)

    return -fs_skewt_logpdf(r, mu, sigma, nu, gamma)
```

### 10.1.4 Standard Student-t CDF

```
def student_t_cdf(x, nu):
    """CDF of standard Student-t(nu) at x."""
    # Using the regularized incomplete beta function
    # F_nu(x) = 0.5 + 0.5 * sign(x) * I_{nu/(nu+x^2)}(nu/2, 1/2)
    a = 0.5 * nu
    b = 0.5
    ratio = nu / (nu + x * x)

    # betainc(a, b, ratio) = I_ratio(a, b)
    I_x = betainc(a, b, ratio)

    return 0.5 + 0.5 * sign(x) * I_x
```

### 10.1.5 FS Skew-t CDF

```
def fs_skewt_cdf(r, mu, sigma, nu, gamma):
    z = (r - mu) / sigma
    c = 1.0 / (gamma * gamma + 1.0)  # mass below mu for FS

    left_mask = z < 0
    cdf = zeros_like(z)

    # z < 0 case
    cdf_left = (2.0 * c) * student_t_cdf(gamma * z[left_mask], nu[left_mask])
    cdf[left_mask] = cdf_left

    # z >= 0 case
    z_right = z[~left_mask]
    nu_right = nu[~left_mask]
    gamma_right = gamma[~left_mask]
    c_right = c[~left_mask]
    F0 = student_t_cdf(z_right / gamma_right, nu_right)
    cdf_right = c_right + (2.0 * gamma_right**2 * c_right) * (F0 - 0.5)
    cdf[~left_mask] = cdf_right

    return cdf
```

### 10.1.6 Standard Student-t Upper Partial Moment

```
def student_t_upper_partial_moment(a, nu):
    """E[X * 1(X > a)] for X ~ t_nu."""
    if nu <= 1.0:
        return full_like(a, inf)  # undefined
    t_density = exp(student_t_logpdf(a, nu))
    return (nu + a * a) / (nu - 1.0) * t_density
```

### 10.1.7 FS Skew-t Partial Moments

```
def fs_skewt_partial_moments(mu, sigma, nu, gamma):
    """Returns (pi_neg, pi_pos, M_neg, M_pos)."""
    z0 = -mu / sigma  # standardized threshold

    c = 1.0 / (gamma * gamma + 1.0)

    # pi_neg = F_FS(0)
    pi_neg = fs_skewt_cdf(zeros_like(mu), mu, sigma, nu, gamma)
    pi_pos = 1.0 - pi_neg

    # M_pos: case split
    M_pos = zeros_like(mu)
    mask = z0 >= 0  # mu <= 0
    if any(mask):
        zr = z0[mask] / gamma[mask]
        Fz = student_t_cdf(zr, nu[mask])
        Ez = student_t_upper_partial_moment(zr, nu[mask])
        M_pos[mask] = (2.0 * gamma[mask]**2 * c[mask]) * (
            mu[mask] * (1.0 - Fz) + sigma[mask] * gamma[mask] * Ez
        )

    mask2 = z0 < 0  # mu > 0
    if any(mask2):
        # Lower half contribution
        a = gamma[mask2] * z0[mask2]
        b = -a  # > 0
        F_neg = student_t_cdf(a, nu[mask2])
        E_lower = student_t_upper_partial_moment(b, nu[mask2])
        E_0 = student_t_upper_partial_moment(0.0, nu[mask2])
        Term_A = (2.0 * c[mask2]) * (
            mu[mask2] * (0.5 - F_neg)
            + (sigma[mask2] / gamma[mask2]) * (E_lower - E_0)
        )
        # Upper half contribution
        Term_B = (2.0 * gamma[mask2]**2 * c[mask2]) * (
            0.5 * mu[mask2] + sigma[mask2] * gamma[mask2] * E_0
        )
        M_pos[mask2] = Term_A + Term_B

    # M_neg = E[R] - M_pos
    E_R = mu + sigma * (gamma - 1.0/gamma) * student_t_expected_abs(nu)
    M_neg = E_R - M_pos

    return pi_neg, pi_pos, M_neg, M_pos


def student_t_expected_abs(nu):
    """E[|X|] for X ~ t_nu. Requires nu > 1."""
    if nu <= 1.0:
        return full_like(nu, inf)
    return (2.0 * sqrt(nu) * gamma_func((nu + 1.0) / 2.0)
            / ((nu - 1.0) * sqrt(pi) * gamma_func(nu / 2.0)))
```

## 10.2 Parameter Activation Functions

```
mu = mu_raw  # unconstrained location
sigma = softplus(sigma_raw) + EPS  # scale > EPS
nu = 2.0 + softplus(nu_tilde)  # df > 2 (variance existence)
gamma = exp(gamma_raw)  # skew > 0
```

**Why nu > 2**: Ensures variance existence for partial moment computation.
If data requires nu < 2, use trimmed estimators or switch to median-based actions.

## 10.3 Stability Checklist

| Issue | Mitigation |
|---|---|
| `1 + x^2/nu` near 0 | `log1p(x^2 / nu)` |
| `lgamma` for large nu | Use asymptotic expansion `lgamma(x) ≈ (x-0.5)*log(x) - x + 0.5*LOG_2PI` for nu > 100 |
| `betainc` gradient | Use scipy `betainc` with `out=` parameter; backprop through autodiff |
| `gamma = 0` or `gamma = inf` | Clip to [EPS, 1e6] |
| `sigma = 0` | Clip to [EPS, inf) |
| `nu close to 2` | Add small penalty `penalty = EPS_nu / (nu - 2.0)` to push nu away from boundary |
| log(gamma + 1/gamma) | For gamma extreme: `approx abs(log_gamma)` when |log_gamma| > 5 |

## 10.4 Gradient Considerations

The partial moments involve `betainc` which must have a gradient for autodiff.
In PyTorch, `torch.special.betainc` or equivalent is needed. Alternative:
use numerical quadrature with precomputed Gauss-Legendre nodes that are
differentiable through the PDF.

For the NLL loss, gradients through `log1p`, `lgamma`, `log` are standard
and well-behaved for all valid parameter ranges.

## 10.5 Mixed-Precision Risk

32-bit (float32) is sufficient for:
- NLL computation (values typically in [0, 20])
- Parameter gradients

16-bit (float16/bfloat16) is NOT recommended for:
- `lgamma` (loses precision for moderate arguments)
- `betainc` (numerical instability near boundaries)
- Partial moments (cancellation risk in `E[R] - M_pos`)

If mixed precision is used, force the NLL and partial moment computation
to float32.

## 10.6 Initialization

```
mu_init ~ N(0, 0.1)          # residuals should center near 0
sigma_log_init ~ N(0, 0.5)   # exp(0) = 1 for unit-variance residuals
nu_tilde_init ~ N(-2, 0.5)   # 2 + softplus(-2) ≈ 2.13 (moderate tail)
gamma_log_init ~ N(0, 0.1)   # exp(0) ≈ 1 (near-symmetric start)
```

## 10.7 Missing Horizon Handling

If a day has fewer than 24 valid hours, normalize the day-loss by the number
of valid hours:

```
valid_mask = ~isnan(residuals)
day_nll = -sum(fs_skewt_logpdf(residuals[valid_mask], ...)) / sum(valid_mask)
```

Do NOT fill missing hours with 0 or the mean -- this introduces bias.
