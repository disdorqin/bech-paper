# Distribution Derivation: State-Conditional FS Skew-t

> Part of HCH v2 Derived Loss Math Design v0.1

## 4.1 Preliminaries

### 4.1.1 Standard (Symmetric) Student-t

Let T_nu denote the standard Student-t distribution with nu > 0 df. Its density:

```
t_nu(x) = Gamma((nu+1)/2) / (Gamma(nu/2) * sqrt(nu*pi)) * (1 + x^2/nu)^{-(nu+1)/2}
```

**Normalization**: The integral over R is 1 by the standard construction
(ratio of independent Normal and scaled chi random variables).

**CDF**: Let `F_nu(x) = integral_{-inf}^{x} t_nu(u) du`. No elementary closed form
but computable via regularized incomplete beta function:

```
F_nu(x) = 1/2 + x * 2F1(1/2, (nu+1)/2; 3/2; -x^2/nu) / (sqrt(nu) * Beta(nu/2, 1/2))
```

Equivalently: `F_nu(x) = 0.5 + 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2) * sign(x)` where
`I` is the regularized incomplete beta.

**Moments** (standardized, location 0, scale 1):
- `E[X] = 0` for `nu > 1` (otherwise undefined)
- `Var(X) = nu/(nu-2)` for `nu > 2` (otherwise infinite)
- `E[X^3] = 0` (symmetric)
- `E[X^4] = 3*nu^2 / ((nu-2)(nu-4))` for `nu > 4`
- Excess kurtosis = `6/(nu-4)` for `nu > 4`

**Limit behavior**:
- `nu -> inf`: `t_nu(x) -> N(0,1)` (pointwise convergence)
- `nu -> 1`: `t_nu(x) -> Cauchy(0,1)`
- `nu downarrow 0`: density becomes degenerate (diverges)

### 4.1.2 Location-Scale Student-t

For `R = mu + sigma * X` where `X ~ t_nu`:

```
f(r | mu, sigma, nu) = (1/sigma) * t_nu((r - mu) / sigma)
log f = log t_nu(z) - log sigma
```

where `z = (r - mu) / sigma`.

**NLL contribution**:

```
NLL(r) = -log f(r) = log sigma - log t_nu(z)
        = log sigma + log(Gamma(nu/2) * sqrt(nu*pi) / Gamma((nu+1)/2))
          + ((nu+1)/2) * log(1 + z^2/nu)
```

Using `log(Gamma(...))` for stability.

## 4.2 Fernandez-Steel (FS) Skew-t Construction

### 4.2.1 Density

Following Fernandez & Steel (1998), given `gamma > 0`:

```
f_FS(r | mu, sigma, nu, gamma) = (2 / (sigma * (gamma + 1/gamma))) *
    { t_nu(gamma * (r-mu)/sigma),     r < mu
    { t_nu((r-mu)/(sigma*gamma)),     r >= mu
```

where `z = (r - mu) / sigma`:

```
f_FS(r) = (2 / (sigma * (gamma + 1/gamma))) *
    { t_nu(gamma * z),      z < 0
    { t_nu(z / gamma),      z >= 0
```

### 4.2.2 Normalization Proof

We must verify `integral_R f_FS(r) dr = 1`.

```
integral_R f_FS(r) dr = integral_{-inf}^{mu} (2/sigma(gamma+1/gamma)) * t_nu(gamma*(r-mu)/sigma) dr
                       + integral_{mu}^{inf} (2/sigma(gamma+1/gamma)) * t_nu((r-mu)/(sigma*gamma)) dr
```

Let `u = gamma*(r-mu)/sigma` for r < mu, then `dr = (sigma/gamma) du`, u in (-inf, 0):

```
Term 1 = (2/(sigma(gamma+1/gamma))) * (sigma/gamma) * integral_{-inf}^{0} t_nu(u) du
       = (2/(gamma*(gamma+1/gamma))) * (1/2) = (2/(gamma*(gamma+1/gamma))) * (1/2)
       = 1/(gamma*(gamma+1/gamma))
```

Wait -- carefully: `integral_{-inf}^{0} t_nu(u) du = 1/2` by symmetry.

```
Term 1 = (2/(gamma+1/gamma)) * (1/gamma) * (1/2) = 1/(gamma*(gamma+1/gamma))
```

Hmm, that gives `1/(gamma^2 + 1)`. Let me recalculate.

Actually: `Term 1 = (2/(gamma+1/gamma)) * (1/gamma) * integral_{-inf}^{0} t_nu(u) du`
`= (2/(gamma+1/gamma)) * (1/gamma) * (1/2) = 1/(gamma*(gamma+1/gamma)) = 1/(gamma^2+1)`

For r >= mu, let `v = (r-mu)/(sigma*gamma)`, then `dr = sigma*gamma dv`, v in [0, inf):

```
Term 2 = (2/(sigma(gamma+1/gamma))) * (sigma*gamma) * integral_{0}^{inf} t_nu(v) dv
       = (2*gamma/(gamma+1/gamma)) * (1/2)
       = gamma/(gamma+1/gamma)
       = gamma^2/(gamma^2+1)
```

**Total**: `1/(gamma^2+1) + gamma^2/(gamma^2+1) = 1`. ✓

Note: `gamma+1/gamma = (gamma^2+1)/gamma`.

So rewritten compactly:

```
Term 1 (z < 0) contribution = 1/(gamma^2+1)
Term 2 (z >= 0) contribution = gamma^2/(gamma^2+1)
```

The fraction of probability mass below mu is `1/(gamma^2+1)`.

**Special case gamma=1**: `Term 1 = 1/2, Term 2 = 1/2` -- recovers symmetry. ✓

**Special case gamma -> inf**: All mass goes above mu; density approaches half-t
truncated at mu from below. Term 1 -> 0. ✓

### 4.2.3 Parameter Semantics

- `mu`: **mode** AND **median** of the distribution (the split point).
  When gamma != 1, mu is NOT the mean (unless nu is infinite).
- The mean is:

```
E[R] = mu + sigma * (gamma - 1/gamma) * E[|X|]  for nu > 1
```

where `X ~ t_nu` standard, and `E[|X|] = 2*sqrt(nu)*Gamma((nu+1)/2) / ((nu-1)*sqrt(pi)*Gamma(nu/2))`
for nu > 1.

- `sigma`: scale parameter. NOT equal to standard deviation.
  `Var(R) = sigma^2 * [Constant involving gamma, nu]` for nu > 2.
- `gamma`: skewness parameter. `gamma > 1` means the right tail is heavier;
  `gamma < 1` means the left tail is heavier.

**Identifiability concern**: A shift in `mu` and a change in `gamma` can both
make the distribution more right-heavy. Specifically, `F(mu) = 1/(gamma^2+1)`.
So `gamma` directly controls the mass split at `mu`. If `mu` is shifted by the
network, the interpretation of `gamma` as skew depends on `mu` being the actual
mode/median. This is a practical identifiability risk when both are estimated.

## 4.3 Negative Log-Likelihood

For a single observation `r` with `z = (r - mu)/sigma`:

**Case z < 0:**
```
log f(r) = log 2 - log sigma - log(gamma + 1/gamma)
           + log t_nu(gamma * z)
= log 2 - log sigma - log(gamma + 1/gamma)
  + log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi)
  - ((nu+1)/2) * log(1 + (gamma*z)^2 / nu)
```

**Case z >= 0:**
```
log f(r) = log 2 - log sigma - log(gamma + 1/gamma)
           + log t_nu(z / gamma)
= log 2 - log sigma - log(gamma + 1/gamma)
  + log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5*log(nu*pi)
  - ((nu+1)/2) * log(1 + z^2 / (gamma^2 * nu))
```

**Compact form**:
```
log f(r) = log 2 - log sigma - log(gamma + 1/gamma) + log t_nu(s * z)

where s = gamma     if z < 0
      s = 1/gamma   if z >= 0
```

### 4.3.1 NLL Implementation (Stable)

```
def fs_skewt_nll(r, mu, sigma, nu, gamma):
    z = (r - mu) / sigma  # standardize
    s = where(z < 0, gamma, 1.0 / gamma)
    sz = s * z  # scaled standardized value

    const = log(2.0) - log(sigma) - log(gamma + 1.0/gamma)
    t_logpdf = log_gamma((nu + 1.0)/2.0) - log_gamma(nu/2.0) \
               - 0.5 * log(max(nu, EPS) * pi) \
               - ((nu + 1.0)/2.0) * log1p(sz**2 / max(nu, EPS))

    return -(const + t_logpdf)
```

**Stability notes**:
- `log1p` used for the `1 + sz^2/nu` term to avoid catastrophic cancellation
  when `sz^2/nu` is near 0
- `max(nu, EPS)` prevents log(0) when nu -> 0
- `log_gamma` via `lgamma` for stability
- `log(gamma + 1/gamma)` can use `logaddexp(log_gamma, -log_gamma)` or direct
  computation (stable for gamma in reasonable range)

## 4.4 CDF of FS Skew-t

```
F_FS(r) = integral_{-inf}^{r} f_FS(u) du
```

Derivation via cases:

**Case r < mu**: The entire integral is in the z < 0 region.

```
F_FS(r) = (2/(gamma+1/gamma)) * (1/gamma) * integral_{-inf}^{gamma*z} t_nu(u) du
        = (2/(gamma+1/gamma)) * (1/gamma) * F_nu(gamma * z)
```

where `F_nu` is the standard Student-t CDF.

**Case r >= mu**:

```
F_FS(r) = P(z < 0) + P(0 <= z <= (r-mu)/sigma)
        = 1/(gamma^2+1) + (2/(gamma+1/gamma)) * gamma * [F_nu(z/gamma) - 1/2]
        = 1/(gamma^2+1) + (2*gamma^2/(gamma^2+1)) * [F_nu(z/gamma) - 1/2]
```

Simplified:

```
F_FS(r) = { (2/(gamma^2+1)) * (1/gamma) * F_nu(gamma*z),            r < mu
          { 2/(gamma^2+1) * (1/gamma) * F_nu(gamma*0)
            + gamma^2/(gamma^2+1) when r = mu ... NO

Better:
    F_FS(r) = 2/(gamma^2+1) * [ 1/gamma * F_nu(gamma*z_small) * I(z<0)
                               + gamma * (F_nu(z/gamma) - 1/2) * I(z>=0) ]
             + 1/(gamma^2+1) * I(z>=0)
```

Wait, let me re-derive this cleanly.

Let `G(r) = integral_{-inf}^{r} f_FS(u) du`.

**Case z < 0 (r < mu)**:
With `u = gamma * w` where `w = (r'-mu)/sigma`:

```
G(r) = integral_{-inf}^{r} (2/(sigma*(gamma+1/gamma))) * t_nu(gamma*(v-mu)/sigma) dv
```

Substitute `t = gamma*(v-mu)/sigma`, `dv = (sigma/gamma) dt`:

```
G(r) = (2/(sigma*(gamma+1/gamma))) * (sigma/gamma) * integral_{-inf}^{gamma*z} t_nu(t) dt
     = (2/(gamma*(gamma+1/gamma))) * F_nu(gamma*z)
```

Since `gamma*(gamma+1/gamma) = gamma^2 + 1`:

```
G(r) = (2/(gamma^2+1)) * F_nu(gamma*z)        for z < 0
```

Check at z=0 (r=mu): `G(mu) = (2/(gamma^2+1)) * 1/2 = 1/(gamma^2+1)`. ✓

**Case z >= 0 (r >= mu)**:

```
G(r) = G(mu) + integral_{mu}^{r} f_FS(v) dv
     = 1/(gamma^2+1)
       + integral_{mu}^{r} (2/(sigma*(gamma+1/gamma))) * t_nu((v-mu)/(sigma*gamma)) dv
```

Substitute `t = (v-mu)/(sigma*gamma)`, `dv = sigma*gamma dt`:

```
G(r) = 1/(gamma^2+1)
       + (2/(sigma*(gamma+1/gamma))) * (sigma*gamma) * integral_{0}^{z/gamma} t_nu(t) dt
     = 1/(gamma^2+1) + (2*gamma/(gamma+1/gamma)) * (F_nu(z/gamma) - 1/2)
     = 1/(gamma^2+1) + (2*gamma^2/(gamma^2+1)) * (F_nu(z/gamma) - 1/2)
```

Check at z -> inf: `G(inf) = 1/(gamma^2+1) + (2*gamma^2/(gamma^2+1)) * (1 - 1/2) = 1/(gamma^2+1) + gamma^2/(gamma^2+1) = 1`. ✓

### 4.4.1 Compact CDF

```
F_FS(r | mu, sigma, nu, gamma) =
    (2/(gamma^2+1)) * F_nu(gamma*z)                              for z < 0
    1/(gamma^2+1) + (2*gamma^2/(gamma^2+1))*(F_nu(z/gamma)-0.5) for z >= 0
```

where `z = (r - mu) / sigma`.

## 4.5 Moments of FS Skew-t

### 4.5.1 Mean (requires nu > 1)

Let `X ~ t_nu` standard. The FS skew-t `R = mu + sigma * Y` where Y has the standard
FS skew-t (mu=0, sigma=1). Then:

```
E[Y] = integral y * f_FS(y) dy

     = integral_{-inf}^{0} y * (2/(gamma+1/gamma)) * gamma * t_nu(gamma*y) dy
     + integral_{0}^{inf} y * (2/(gamma+1/gamma)) * (1/gamma) * t_nu(y/gamma) dy
```

Substitute `u = gamma*y` in first term, `v = y/gamma` in second:

```
E[Y] = (2/(gamma*(gamma+1/gamma))) * integral_{-inf}^{0} (u/gamma) * t_nu(u) du
     + (2*gamma/(gamma+1/gamma)) * integral_{0}^{inf} (v*gamma) * t_nu(v) dv

     = (2/(gamma^2*(gamma+1/gamma))) * integral_{-inf}^{0} u * t_nu(u) du
     + (2*gamma^2/(gamma+1/gamma)) * integral_{0}^{inf} v * t_nu(v) dv
```

For standard t_nu, by symmetry: `integral_{-inf}^0 u*t_nu(u) du = -E[|X|]/2`,
`integral_0^inf v*t_nu(v) dv = E[|X|]/2` where `E[|X|] = integral_0^inf 2*x*t_nu(x) dx`.

Known result: `E[|X|] = 2*sqrt(nu)*Gamma((nu+1)/2) / ((nu-1)*sqrt(pi)*Gamma(nu/2))` for nu > 1.

```
E[Y] = (2/(gamma^2*(gamma+1/gamma))) * (-E[|X|]/2)
     + (2*gamma^2/(gamma+1/gamma)) * (E[|X|]/2)

     = (-1/(gamma^2*(gamma+1/gamma)) + gamma^2/(gamma+1/gamma)) * E[|X|]

     = (gamma^2 - 1/gamma^2) / (gamma+1/gamma) * E[|X|]

     = (gamma^4 - 1) / (gamma*(gamma^2+1)) * E[|X|]
```

Or more elegantly: `E[Y] = (gamma - 1/gamma) * E[|X|]`.

Check gamma=1: `E[Y] = (1-1) * E[|X|] = 0`. ✓ (symmetric case).

**Full mean**:
```
E[R] = mu + sigma * (gamma - 1/gamma) * E[|X|],      nu > 1
```

where `E[|X|] = 2*sqrt(nu)*Gamma((nu+1)/2) / ((nu-1)*sqrt(pi)*Gamma(nu/2))`.

### 4.5.2 Variance (requires nu > 2)

From the variance of the standard FS skew-t:

```
Var(Y) = E[Y^2] - E[Y]^2

E[Y^2] = integral_{-inf}^0 y^2 * (2/(gamma+1/gamma)) * gamma * t_nu(gamma*y) dy
       + integral_0^inf y^2 * (2/(gamma+1/gamma)) * (1/gamma) * t_nu(y/gamma) dy

       = (2/(gamma^3*(gamma+1/gamma))) * integral_{-inf}^0 u^2 * t_nu(u) du
       + (2*gamma^3/(gamma+1/gamma)) * integral_0^inf v^2 * t_nu(v) dv

       = (2/(gamma^3*(gamma+1/gamma)) + 2*gamma^3/(gamma+1/gamma)) * (1/2) * Var_std(nu)
```

where `Var_std(nu) = nu/(nu-2)` is the variance of standard t_nu.

```
E[Y^2] = (1/gamma^3 + gamma^3) / (gamma+1/gamma) * Var_std(nu)

       = (1 + gamma^6) / (gamma^3*(gamma^2+1)/gamma) * Var_std(nu)

       = (1 + gamma^6) / (gamma^2*(gamma^2+1)) * Var_std(nu)
```

Hmm, this is getting messy. Let me use a cleaner approach.

Alternative: From Fernandez & Steel (1998) or Wurtz+ (2006), the variance of
FS skew-t is:

```
Var(Y) = (gamma^3 + 1/gamma^3) / (gamma + 1/gamma) * Var_std(nu) - [E(Y)]^2
```

where `Var_std(nu) = nu/(nu-2)` for nu > 2.

**Full variance**:
```
Var(R) = sigma^2 * Var(Y),      nu > 2
```

### 4.5.3 Skewness and Excess Kurtosis

These have closed forms in terms of `gamma` and `nu` (see Wurtz+ 2006 or
Rigby & Stasinopoulos 2005) but are algebraically heavy. For our purposes,
numerical computation (quadrature or MC integration) is sufficient,
with analytic verification at known limits.

### 4.5.4 Limits

- `gamma -> 1`: FS skew-t -> symmetric Student-t(mu, sigma, nu)
  - `E[R] -> mu`, Var -> sigma^2 * nu/(nu-2), skewness -> 0
- `nu -> inf` with gamma fixed: FS skew-t -> FS skew-normal (Azalini-type)
  - `t_nu(x) -> N(0,1)` pointwise
  - All moments exist; mean, variance, skewness, kurtosis converge
- `nu -> inf, gamma -> 1`: FS skew-t -> Normal(mu, sigma)
- `nu -> 1`: Heavy-tailed case approaching Cauchy on each side
  - Mean undefined, variance undefined
  - Partial moments may still exist under certain conditions

## 4.6 Dossier Section 6 Questions Answered

### 4.6.1 Density integral verification

Verified in 4.2.2: integral = 1/(gamma^2+1) + gamma^2/(gamma^2+1) = 1. ✓

### 4.6.2 mu is mode or mean?

**mu is the MODE** (and median). The density f_FS has its peak at r = mu because
t_nu has its peak at 0 and both piecewise segments meet at z=0 with:

```
f_FS at z=0- = (2/(sigma*(gamma+1/gamma))) * t_nu(0)
f_FS at z=0+ = (2/(sigma*(gamma+1/gamma))) * t_nu(0)
```

These are equal (continuous at mu), and t_nu(0) is the global maximum of the
standard t density. So mu is the mode. ✓

The median is also mu because F_FS(mu) = 1/(gamma^2+1) is NOT always 0.5.

**Wait -- this is important.** The median is NOT mu unless gamma = 1!

```
F_FS(mu) = 1/(gamma^2+1)
```

For gamma = 0.5: F(mu) = 1/(0.25+1) = 0.8, so median > mu.
For gamma = 2: F(mu) = 1/(4+1) = 0.2, so median > mu.

Actually for gamma < 1: F(mu) = 1/(gamma^2+1) > 0.5 (gamma^2 < 1, so gamma^2+1 < 2, 1/(...) > 0.5)
For gamma > 1: F(mu) = 1/(gamma^2+1) < 0.5

So mu is the mode, but NOT the median. The median depends on gamma.

**Correction**: The mode = mu. The median is NOT mu (unless gamma=1).

Let me find the median. Need to solve F(r) = 0.5.

Case 1: 0.5 <= 1/(gamma^2+1), i.e., gamma^2+1 <= 2, gamma <= 1. Then median >= mu.
    0.5 = (2/(gamma^2+1)) * F_nu(gamma*z)
    F_nu(gamma*z) = 0.25*(gamma^2+1)
    gamma*z = F_nu^{-1}(0.25*(gamma^2+1))
    median = mu + sigma*z = mu + sigma * F_nu^{-1}(0.25*(gamma^2+1)) / gamma

Case 2: 0.5 > 1/(gamma^2+1), gamma > 1. Then median < mu.
    0.5 = 1/(gamma^2+1) + (2*gamma^2/(gamma^2+1)) * (F_nu(z/gamma) - 0.5)
    F_nu(z/gamma) = 0.5 + [0.5 - 1/(gamma^2+1)] * (gamma^2+1)/(2*gamma^2)
    = 0.5 + (gamma^2-1)/(4*gamma^2)
    median = mu + sigma*z = mu + sigma * gamma * F_nu^{-1}(...)

**Implication**: The `mu` parameter in FS skew-t is the MODE, not the mean or
median. This is important for interpretation and for the DVG interface.

### 4.6.3 sigma and gamma scale confusion

Yes, sigma and gamma both affect the scale. gamma compresses/expands one side
while making the other side symmetric. The effective standard deviation depends
on BOTH sigma and gamma. This is a practical concern:

- sigma: controls overall spread
- gamma: redistributes mass between left and right halves

If gamma >> 1, the right tail is heavy and the left tail is compressed.
Both sigma and gamma contribute to the apparent scale. In neural network
optimization, this means gradients can flow through both parameters and
a change in sigma can be partially compensated by gamma.

**Mitigation**: Use the reparameterization `alpha = log gamma`. Then impose
a mild regularizer on `alpha` to prevent extreme skew.

### 4.6.4 gamma=1 degeneracy

When gamma = 1:
- `f_FS(r) = (2/(sigma*2)) * {t_nu(z), z<0; t_nu(z), z>=0} = (1/sigma) * t_nu(z)` ✓
- This is exactly the Student-t(mu, sigma, nu) density. ✓
- `1/(gamma^2+1) = 1/2`, so F(mu) = 0.5, mu is also the median. ✓

### 4.6.5 nu -> inf limit

As nu -> inf, t_nu(x) -> N(0,1). Then:

```
f_FS -> (2/(sigma*(gamma+1/gamma))) * {N(gamma*z), z<0; N(z/gamma), z>=0}
```

This is the Fernandez-Steel skew-normal distribution. Its mean is:

```
E[R] -> mu + sigma * (gamma - 1/gamma) * sqrt(2/pi)
```

and variance:

```
Var(R) -> sigma^2 * [gamma^3 + 1/gamma^3] / [gamma + 1/gamma]
        - sigma^2 * (gamma - 1/gamma)^2 * (2/pi)
```

### 4.6.6 First/second moment existence conditions

- Mean `E[R]` exists iff `nu > 1`
- Variance `Var(R)` exists iff `nu > 2`
- Skewness exists iff `nu > 3`
- Excess kurtosis exists iff `nu > 4`

When `nu` is between 1 and 2, the mean exists but variance is infinite --
this corresponds to the "very heavy-tailed but not Cauchy-level" regime
observed in some market residuals.

### 4.6.7 Neural network identifiability

With a NN outputting 4 params simultaneously:

```
Problem: (mu, sigma, nu, gamma) all learned from same latent features.
If mu shifts and gamma adjusts, the resulting distribution can look similar
for moderate sample sizes.

Diagnostic: Profile likelihood. Fix all other parameters; vary one;
check if log-likelihood has a sharp peak. Flat profiles indicate
practical non-identifiability.

Mitigation in order of preference:
1. Orthogonal parameterization: let mu = median (not mode) by construction
   (requires solving for mode from median, or using a different skew-t
   parameterization like Azalini-type)
2. Mild regularization: add small penalty |alpha| for log_gamma
3. Shared nu across hours: nu is a global or slowly-varying parameter,
   not hour-specific
4. Separate heads: mu/sigma from one sub-network, gamma/nu from another
   with reduced capacity
```
