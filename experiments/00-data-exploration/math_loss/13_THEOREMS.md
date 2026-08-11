# Theorems, Lemmas, and Propositions

> Part of HCH v2 Derived Loss Math Design v0.1

---

## Proposition 1: FS Skew-t Density Normalization

**Statement**: The Fernandez-Steel skew-t density

```
f(r | mu, sigma, nu, gamma) = (2 / (sigma*(gamma + 1/gamma))) *
    { t_nu(gamma * (r-mu)/sigma),     r < mu
    { t_nu((r-mu) / (sigma*gamma)),   r >= mu
```

integrates to 1 over R.

**Assumptions**: `sigma > 0, nu > 0, gamma > 0`. `t_nu` is the standard Student-t
density with nu degrees of freedom.

**Proof**:
Split the integral at `r = mu`.

For `r < mu`: substitute `u = gamma*(r-mu)/sigma`.
Then `r = mu + sigma*u/gamma, dr = (sigma/gamma) du`, and `u` ranges from `-inf` to 0.
The integral becomes:

```
(2/(sigma*(gamma+1/gamma))) * (sigma/gamma) * I_{-inf}^{0} t_nu(u) du
= (2/(gamma*(gamma+1/gamma))) * (1/2) = 1/(gamma^2+1)
```

since `I_{-inf}^{0} t_nu(u) du = 1/2` by symmetry of the standard Student-t.

For `r >= mu`: substitute `v = (r-mu)/(sigma*gamma)`.
Then `r = mu + sigma*gamma*v, dr = sigma*gamma dv`, and `v` ranges from 0 to `inf`.
The integral becomes:

```
(2/(sigma*(gamma+1/gamma))) * (sigma*gamma) * I_{0}^{inf} t_nu(v) dv
= (2*gamma/(gamma+1/gamma)) * (1/2) = gamma^2/(gamma^2+1)
```

Sum: `1/(gamma^2+1) + gamma^2/(gamma^2+1) = 1`. QED.

**Corollary 1.1**: When `gamma = 1`, the density reduces to the standard location-scale
Student-t `(1/sigma) * t_nu((r-mu)/sigma)`.

**Corollary 1.2**: The probability mass below the mode `mu` equals `1/(gamma^2+1)`.
In particular, `mu` is the median if and only if `gamma = 1`.

---

## Proposition 2: Directional Candidate Non-Collapse

**Statement**: For any residual distribution with finite conditional mean and
`0 < pi_neg < 1`, the partial-moment candidate `M_neg = -pi_neg * m_neg` and
the side-conditional candidate `Delta_cond = -m_neg` are distinct. Specifically:

```
|M_neg| < |Delta_cond|  when 0 < pi_neg < 1
```

**Assumptions**: `E[|R|] < inf`.

**Proof**:
Let `pi = pi_neg = P(R < 0)` and `m = m_neg = E[-R | R < 0]`.
Then `M_neg = -pi * m` and `Delta_cond = -m` (both <= 0, since m > 0).

The ratio is `|M_neg| / |Delta_cond| = pi * m / m = pi`.

Since `0 < pi < 1`, we have `|M_neg| = pi * m < m = |Delta_cond|`. QED.

**Corollary 2.1**: The partial-moment correction is always strictly smaller in
magnitude than the side-conditional mean correction for non-degenerate
distributions. The shrinkage factor equals the occurrence probability.

**Corollary 2.2**: As `pi -> 1`, `M_neg -> Delta_cond` (events are certain).
As `pi -> 0`, `M_neg -> 0` while `Delta_cond -> m` remains positive.

**Implication**: The two candidate types encode genuinely different information:
pi (from distribution) vs m (from distribution). The partial moment uses
both multiplicatively; the conditional mean ignores pi.

---

## Proposition 3: Limit Behavior of FS Skew-t

**Statement**: As `nu -> inf`, the FS skew-t density converges pointwise to the
Fernandez-Steel skew-normal density:

```
f_FS(r) -> (2 / (sigma*(gamma+1/gamma))) *
    { phi(gamma * (r-mu)/sigma),     r < mu
    { phi((r-mu) / (sigma*gamma)),   r >= mu
```

where `phi` is the standard normal density.

**Proof**: For any fixed `x`, `t_nu(x) -> phi(x)` as `nu -> inf`
(pointwise convergence of Student-t to normal). The FS construction
is a continuous function of `t_nu`, so pointwise convergence is preserved. QED.

**Corollary 3.1**: As `nu -> inf` and `gamma -> 1`, the density converges
to `N(mu, sigma^2)`.

---

## Proposition 4: Partial Moment from Standard t Upper Moment

**Statement**: The upper partial moment of standard Student-t, defined as
`E_nu(a) = I_{a}^{inf} x * t_nu(x) dx`, has the closed form:

```
E_nu(a) = (nu + a^2) / (nu - 1) * t_nu(a),    for nu > 1
```

**Proof**:
The standard Student-t density is `t_nu(x) = C_nu * (1 + x^2/nu)^{-(nu+1)/2}`
where `C_nu = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi))`.

Consider the derivative:
```
d/dx [ (1 + x^2/nu)^{-(nu-1)/2} ]
    = -(nu-1)/(2nu) * 2x * (1 + x^2/nu)^{-(nu+1)/2}
    = -(nu-1)/nu * x * (1 + x^2/nu)^{-(nu+1)/2}
    = -(nu-1)/(nu*C_nu) * x * t_nu(x)
```

Therefore:
```
x * t_nu(x) = -(nu*C_nu)/(nu-1) * d/dx[(1 + x^2/nu)^{-(nu-1)/2}]
```

Integrating from `a` to `inf`:
```
E_nu(a) = -(nu*C_nu)/(nu-1) * [0 - (1 + a^2/nu)^{-(nu-1)/2}]
        = (nu*C_nu)/(nu-1) * (1 + a^2/nu)^{-(nu-1)/2}
```

Note: `(1 + a^2/nu)^{-(nu-1)/2} = (1 + a^2/nu) * (1 + a^2/nu)^{-(nu+1)/2}
= (1 + a^2/nu) * t_nu(a) / C_nu`.

Substituting:
```
E_nu(a) = (nu*C_nu)/(nu-1) * (1 + a^2/nu) * t_nu(a) / C_nu
        = (nu + a^2) / (nu - 1) * t_nu(a)
```

QED.

**Corollary 4.1**: `E_nu(0) = nu/(nu-1) * t_nu(0) = E[|X|] / 2` as expected.

**Corollary 4.2**: The lower partial moment `L_nu(b) = I_{-inf}^{b} x * t_nu(x) dx`
equals `-E_nu(-b)` by symmetry.

---

## Proposition 5: Sufficient Condition for W1 Composite Likelihood

**Statement**: Under the conditional independence assumption
`R_h perp R_{h'} | Z_h, Z_{h'}` for `h != h'`, the day-level NLL:

```
L_day = -sum_{h=1}^{24} log f(R_{d,h} | Z_{d,h})
```

is a valid composite log-likelihood, yielding consistent parameter estimates
for the marginal model `f(R_h | Z_h)`.

**Proof**: This is a direct application of composite likelihood theory
(Varin et al. 2011, "An overview of composite likelihood methods").
The pairwise conditional independence implies that the product of marginal
likelihoods is a valid composite likelihood. Under standard regularity
conditions, the maximum composite likelihood estimator is consistent
and asymptotically normal. QED.

**Note**: This proposition does NOT claim that the estimates are efficient
or that the standard errors are correct without adjustment. It only
establishes consistency, which is sufficient for exploratory modeling.
