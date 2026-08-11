# Directional Partial Moments for FS Skew-t

> Part of HCH v2 Derived Loss Math Design v0.1

## 5.1 Definitions

Given FS skew-t with parameters `(mu, sigma, nu, gamma)` and residual `R = y - yhat_host`:

**Occurrence probabilities**:
```
pi_minus(Z) = P(R < 0 | Z) = F_FS(0)
pi_plus(Z)  = P(R > 0 | Z) = 1 - F_FS(0) - P(R = 0 | Z)
```

Since FS skew-t is continuous, `P(R=0) = 0`. So `pi_plus = 1 - pi_minus`.

**Directional partial moments (threshold = 0)**:
```
M_minus(Z) = E[R * 1(R < 0) | Z]  (= -pi_minus * m_minus, always <= 0)
M_plus(Z)  = E[R * 1(R > 0) | Z]  (= pi_plus * m_plus, always >= 0)
```

**Side-conditional magnitudes**:
```
m_minus(Z) = E[-R | R < 0, Z]  (positive)
m_plus(Z)  = E[R | R > 0, Z]   (positive)
```

Relations: `M_minus = -pi_minus * m_minus`, `M_plus = pi_plus * m_plus`.

## 5.2 Derivation of pi_minus for FS Skew-t

Let `z0 = (0 - mu) / sigma = -mu / sigma` be the standardized threshold.

**Case 1: z0 < 0 (mu > 0)**:
```
pi_minus = (2/(gamma^2+1)) * F_nu(gamma * z0)
```

**Case 2: z0 >= 0 (mu <= 0)**:
```
pi_minus = 1/(gamma^2+1)
          + (2*gamma^2/(gamma^2+1)) * (F_nu(z0/gamma) - 0.5)
```

Equivalently: `pi_minus = F_FS(0)` using the CDF formula from Section 4.4.

### 5.2.1 Special Case: mu = 0

When `mu = 0`: `z0 = 0`, so
```
pi_minus = 1/(gamma^2+1)
pi_plus  = gamma^2/(gamma^2+1)
```

The model is **centered at the threshold**. In the HCH context, `mu` represents
the location of the residual distribution: `mu > 0` means the host systematically
overpredicts; `mu < 0` means it underpredicts.

### 5.2.2 Properties

- gamma -> 1: pi_minus = pi_plus = 0.5 (symmetric around mu)
- gamma -> 0: pi_minus -> 1 (all mass below mu). If mu>0, pi_minus may still be < 1.
- gamma -> inf: pi_plus -> 1
- mu -> +inf: pi_minus -> 0 (threshold far in left tail)
- mu -> -inf: pi_minus -> 1

## 5.3 Derivation of Upper Partial Moment M_plus

```
M_plus = integral_{0}^{inf} r * f_FS(r) dr
```

Two cases based on where 0 falls relative to mu.

**Case: 0 >= mu (z0 >= 0, i.e., mu <= 0)**:
The interval [0, inf) lies entirely in the z >= 0 region.

```
M_plus = integral_{0}^{inf} r * (2/(sigma*(gamma+1/gamma))) * t_nu((r-mu)/(sigma*gamma)) dr

Let u = (r - mu) / (sigma * gamma), r = mu + sigma*gamma*u, dr = sigma*gamma du
u ranges from z0/gamma >= 0 to inf
```

```
M_plus = (2/(sigma*(gamma+1/gamma))) * integral_{z0/gamma}^{inf}
         (mu + sigma*gamma*u) * t_nu(u) * (sigma*gamma) du

       = (2*gamma/(gamma+1/gamma)) * [ mu * integral_{z0/gamma}^{inf} t_nu(u) du
         + sigma*gamma * integral_{z0/gamma}^{inf} u * t_nu(u) du ]

       = (2*gamma^2/(gamma^2+1)) * [ mu * (1 - F_nu(z0/gamma))
         + sigma*gamma * E_nu(z0/gamma) ]
```

where `E_nu(a) = integral_{a}^{inf} u * t_nu(u) du` is the upper partial first
moment of the standard Student-t.

**Standard Student-t upper partial moment**:

For standard t_nu:
```
E_nu(a) = integral_{a}^{inf} u * t_nu(u) du
        = (nu + a^2) * t_nu(a) / (nu - 1)      for nu > 1
```

Proof: `t_nu(u) = C_nu * (1 + u^2/nu)^{-(nu+1)/2}`. Then:

```
d/du [-(1 + u^2/nu)^{-(nu-1)/2}] = ((nu-1)/nu) * u * (1+u^2/nu)^{-(nu+1)/2}
                                  = ((nu-1)/(nu*C_nu)) * u * t_nu(u)
```

So `u * t_nu(u) = -(nu*C_nu/(nu-1)) * d/du[(1+u^2/nu)^{-(nu-1)/2}]`.

Integrating:
```
E_nu(a) = (nu*C_nu/(nu-1)) * (1 + a^2/nu)^{-(nu-1)/2}
        = (nu + a^2)/(nu - 1) * t_nu(a)
```

since `t_nu(a) = C_nu * (1 + a^2/nu)^{-(nu+1)/2}`, so:

`C_nu * (1 + a^2/nu)^{-(nu-1)/2} = C_nu * (1+a^2/nu)^{-(nu+1)/2} * (1+a^2/nu) = t_nu(a) * (1+a^2/nu)`.

Thus: `E_nu(a) = (nu/(nu-1)) * t_nu(a) * (1 + a^2/nu) = (nu + a^2)/(nu-1) * t_nu(a)`. ✓

Check limit a -> -inf: `E_nu(-inf) = 0` (t_nu vanishes exponentially). ✓
Check a = 0: `E_nu(0) = nu/(nu-1) * t_nu(0)`. This equals E[|X|]/2 as expected. ✓

**Lower partial moment** `L_nu(b) = integral_{-inf}^{b} u * t_nu(u) du`:
By symmetry: `L_nu(b) = -E_nu(-b)` for the standard t (since t_nu is symmetric).

### 5.3.1 M_plus (mu <= 0 case, continued)

```
M_plus = (2*gamma^2/(gamma^2+1)) * [ mu * (1 - F_nu(z0/gamma))
         + sigma*gamma * E_nu(z0/gamma) ]
```

**Case: 0 < mu (z0 < 0, i.e., mu > 0)**:
The threshold 0 falls in the z < 0 region. We need to split the integral.

```
M_plus = integral_{0}^{mu} r * f_FS(r) dr + integral_{mu}^{inf} r * f_FS(r) dr

Term A [0, mu): r in (0, mu] with z < 0, mu > 0
Term B [mu, inf): r in [mu, inf) with z >= 0
```

**Term A**:

```
Term A = integral_{0}^{mu} r * (2/(sigma*(gamma+1/gamma))) * t_nu(gamma*(r-mu)/sigma) dr

Let u = gamma*(r-mu)/sigma, r = mu + sigma*u/gamma, dr = (sigma/gamma) du
u ranges from gamma*z0 = -gamma*mu/sigma (< 0) to 0
```

```
Term A = (2/(sigma*(gamma+1/gamma))) * integral_{gamma*z0}^{0}
         (mu + sigma*u/gamma) * t_nu(u) * (sigma/gamma) du

       = (2/(gamma*(gamma+1/gamma))) * [ mu * integral_{gamma*z0}^{0} t_nu(u) du
         + (sigma/gamma) * integral_{gamma*z0}^{0} u * t_nu(u) du ]

       = (2/(gamma^2+1)) * [ mu * (1/2 - F_nu(gamma*z0))
         - (sigma/gamma) * E_nu(0)  + (sigma/gamma) * E_nu(-gamma*z0) ]
```

Wait, let me be more careful with the partial moment signs.

For the lower partial moment of standard t:
`L_nu(b) = integral_{-inf}^{b} u * t_nu(u) du = -E_nu(-b)` (since t is symmetric)

So `integral_{a}^{0} u * t_nu(u) du = L_nu(0) - L_nu(a) = -E_nu(0) - (-E_nu(-a)) = E_nu(-a) - E_nu(0)`.

For a = gamma*z0 < 0: let b = -gamma*z0 > 0. Then:
```
integral_{gamma*z0}^{0} u * t_nu(u) du = E_nu(b) - E_nu(0) where b = -gamma*z0 > 0
```

**Term B**:

```
Term B = integral_{mu}^{inf} r * (2/(sigma*(gamma+1/gamma))) * t_nu((r-mu)/(sigma*gamma)) dr

Let v = (r-mu)/(sigma*gamma), r = mu + sigma*gamma*v, dr = sigma*gamma dv
v ranges from 0 to inf
```

```
Term B = (2/(sigma*(gamma+1/gamma))) * integral_{0}^{inf}
         (mu + sigma*gamma*v) * t_nu(v) * (sigma*gamma) dv

       = (2*gamma/(gamma+1/gamma)) * [ mu * integral_{0}^{inf} t_nu(v) dv
         + sigma*gamma * integral_{0}^{inf} v * t_nu(v) dv ]

       = (2*gamma^2/(gamma^2+1)) * [ mu * 0.5 + sigma*gamma * E_nu(0) ]
```

**M_plus total (mu > 0 case)**:

```
M_plus = (2/(gamma^2+1)) * [ mu * (0.5 - F_nu(gamma*z0))
         + (sigma/gamma) * (E_nu(-gamma*z0) - E_nu(0)) ]
       + (2*gamma^2/(gamma^2+1)) * [ 0.5*mu + sigma*gamma * E_nu(0) ]
```

This is getting complex. For implementation, numerical quadrature is more practical
than closed forms with many cases. See Section 5.6.

## 5.4 Derivation of Lower Partial Moment M_minus

`M_minus = integral_{-inf}^{0} r * f_FS(r) dr`. By property of expectation:

```
M_minus + M_plus = E[R] = mu + sigma * (gamma - 1/gamma) * E[|X|]
```

So `M_minus = E[R] - M_plus`, which is simpler than direct computation.

Or we can derive it directly, analogous to M_plus but integrating over (-inf, 0].

## 5.5 Side-Conditional Magnitudes

```
m_minus(Z) = -M_minus / pi_minus      (when pi_minus > 0)
m_plus(Z)  = M_plus / pi_plus          (when pi_plus > 0)
```

These are `E[-R | R < 0]` and `E[R | R > 0]` respectively.

The recommended candidate actions are:
```
Delta_minus(Z) = M_minus(Z)    (= -pi_minus * m_minus)
Delta_plus(Z)  = M_plus(Z)     (= pi_plus * m_plus)
```

**Double-shrinkage concern**: For a rare event with pi_plus = 0.01 and
conditional mean m_plus = 100:
- M_plus = 0.01 * 100 = 1.0 (severely shrunk)
- Side-conditional mean action: delta = 100 (large correction)
- The question is: which is the correct Bayes action? See Section 6.

## 5.6 Numerical Computation Recommendation

For practical implementation, numerical quadrature (Gauss-Legendre or
adaptive Simpson) on the interval [-10*sigma, 10*sigma] is preferred
over case-by-case analytic formulas. Pseudo-code:

```
def fs_student_t_partial_moments(mu, sigma, nu, gamma, n_quad=200):
    # Gauss-Legendre quadrature
    x_quad, w_quad = gauss_legendre(n_quad, a=-10*sigma, b=10*sigma)
    f = fs_skewt_pdf(x_quad, mu, sigma, nu, gamma)

    mask_neg = x_quad < 0
    mask_pos = x_quad > 0

    M_minus = sum(x_quad[mask_neg] * f[mask_neg] * w_quad[mask_neg])
    M_plus  = sum(x_quad[mask_pos] * f[mask_pos] * w_quad[mask_pos])
    pi_minus = sum(f[mask_neg] * w_quad[mask_neg])
    pi_plus  = sum(f[mask_pos] * w_quad[mask_pos])

    return pi_minus, pi_plus, M_minus, M_plus
```

For differentiable computation during training, use the analytic
formulas with `F_nu` and `E_nu` computed via `lbeta` and `lgamma`
-- see Section 10 for stable implementation.

## 5.7 mu != 0 Case Verification

At mu = 0:
- pi_minus = 1/(gamma^2+1), pi_plus = gamma^2/(gamma^2+1)
- M_minus and M_plus follow from the formulas in 5.3 with z0 = 0

At mu >> 0:
- pi_minus -> 0, almost all mass is above 0
- M_minus -> 0, M_plus -> E[R] = mu + sigma*(gamma-1/gamma)*E[|X|]

At mu << 0:
- pi_plus -> 0, almost all mass is below 0
- M_plus -> 0, M_minus -> E[R]

## 5.8 Non-parametric Comparison

As a sanity check, compare the analytical partial moments with:
```
M_minus_emp = mean(R[R < 0]) * mean(R < 0)  # S2 empirical
M_plus_emp  = mean(R[R > 0]) * mean(R > 0)
```

The FS skew-t fitted values should agree within sampling error.
Large discrepancies suggest:
- Distribution misspecification
- Structural breaks between fitting and evaluation periods
- Outlier-dominated parameter estimates
