# Bayes Decision and Candidate Semantics

> Part of HCH v2 Derived Loss Math Design v0.1

## 6.1 Decision Problem Statement

The HCH selects one of three actions:
- **Identity (I)**: `y_hat = y_hat_host` (delta = 0)
- **Down (D)**: `y_hat = y_hat_host + Delta_minus` (Delta_minus <= 0)
- **Up (U)**: `y_hat = y_hat_host + Delta_plus` (Delta_plus >= 0)

The loss is some function `L(y, y_hat)`. The CAGM-DVG does NOT use the
distributional NLL as the loss -- it uses realized action gain `G = L(y, host) - L(y, corrected)`.
But the candidate corrections `Delta_pm` come from the distribution.

## 6.2 Action Set and Bayes Optimality

### 6.2.1 MSE Loss (L2)

Under MSE: `L(y, y_hat) = (y - y_hat)^2`.

**Lemma 6.1 (MSE Bayes action)**: Given `R = y - y_hat_host ~ FS-t(mu, sigma, nu, gamma)`,
the single best correction is `Delta = E[R] = mu + sigma*(gamma-1/gamma)*E[|X|]`.

This is a single value, not a directional split. The optimal single correction
under MSE is just the conditional mean -- it does NOT decompose into Up and Down.

**Proposition 6.1a (MSE dominance)**: Under MSE, the action "predict E[R]" weakly
dominates any candidate that forces Delta * R <= 0 with probability 1.

*Proof sketch*: The minimizer of `E[(R-Delta)^2]` is `Delta = E[R]`. Any
constraint `Delta >= 0` or `Delta <= 0` can only increase or equal the
expected MSE; the unconstrained optimum may violate the constraint.

**Implication**: If MSE is the evaluation metric, the partial-moment approach
(forcing correction sign) is theoretically suboptimal compared to mean
correction. This is NOT necessarily bad -- the point of HCH is that "do
nothing" (Identity) and "do something in a direction" have different
real-world meanings, and MSE is not the action utility.

### 6.2.2 MAE Loss (L1)

Under MAE: `L(y, y_hat) = |y - y_hat|`.

**Lemma 6.2 (MAE Bayes action)**: The MAE-optimal single correction is
`Delta = median(R) = F_FS^{-1}(0.5)`.

For FS skew-t, the median is NOT mu. It must be solved from:
```
F_FS(r) = 0.5
```

This gives a single scalar, again not directional.

### 6.2.3 Asymmetric Loss (Lin-Lin)

Consider: `L(y, hat_y) = a*(y - hat_y) * I(y > hat_y) + b*(hat_y - y) * I(y < hat_y)`.
This is the newsvendor / pinball loss. Under this loss:

**Lemma 6.3**: The Bayes action is the b/(a+b) quantile of the residual distribution.

With a >> b (severe overprediction penalty): the action is a high quantile, naturally > 0.
With b >> a (severe underprediction penalty): the action is a low quantile, naturally < 0.

This naturally produces directional actions, but the asymmetry is fixed and
does not adapt per context.

### 6.2.4 The HCH 3-Action Problem (non-standard)

The HCH faces a discrete choice + continuous magnitude problem. The utility is:

```
U(action, R) =
    0,                                            if Identity chosen
    |R| - |R - Delta_minus|,                      if Down chosen
    |R| - |R - Delta_plus|,                       if Up chosen
```

(or the equivalent for MSE). This is NOT a standard Bayes decision problem
because the magnitude Delta_pm is chosen BEFORE seeing the action outcome.

**Proposition 6.2**: Under MAE utility, the optimal Down magnitude for the
action "correct downward" is `Delta_minus = median(-R | choose Down)`.
Under MSE, it is `Delta_minus = E[-R | choose Down]`.

The optimal choice of "when to act" depends on the CAGM-DVG routing
and is a sequential decision problem.

## 6.3 Comparison of Candidate Types

| Candidate | Formula | Strengths | Weaknesses | MAE-optimal? |
|---|---|---|---|---|
| Partial moment `M_pm` | `pi_pm * m_pm` | Single distribution; π acts as soft gate | Severe shrinkage for rare events | No (under-corrects) |
| Side-conditional mean `m_pm` | `E[|R| \| sign]` | Full magnitude; no shrinkage | No gate; always corrects when π > 0 | Yes, conditional on acting |
| Side-conditional median | `med(|R| \| sign)` | Robust to outliers | No gate; heavier computation | Yes under MAE, conditional |
| Calibrated partial moment | `lambda * pi_pm * m_pm` | S3 λ adjusts shrinkage | Needs S3; λ adds parameter | With tuned λ |
| Truncated side-conditional | `m_pm * I(pi > tau)` | Hard gate at threshold τ | Threshold discontinuity | Approximately |

## 6.4 Recommended Candidate Design

**For implementation**: Use two candidate types and compare:

1. **Partial moment**: `Delta_pm = pi_pm * m_pm` (distribution-derived, auto-gated)
2. **Side-conditional mean**: `Delta_cond = m_pm` (no gate, full magnitude)

Let CAGM-DVG decide which action to execute, with S3 calibration setting
`lambda in [0, 1]` per branch for the partial-moment candidate.

The DVG should receive BOTH `Delta_pm` and `Delta_cond` as inputs and
learn which candidate type works better in which context.

## 6.5 Why Two Candidates Won't Collapse

**Risk**: If both candidates predict similar values, the Bi-OMC structure is redundant.

**Check**: For `pi ~ 0.5`, `M_pm ≈ 0.5 * m_pm`, so the partial moment is HALF the
side-conditional mean. The ratio `Delta_pm / Delta_cond = pi` is inherently variable.

For `pi -> 0`, the ratio -> 0 (partial moment vanishes, conditional mean may still
be large). For `pi -> 1`, the ratio -> 1 (they converge). The variability comes from
uncertainty in occurrence, which provides genuine structural diversity.

**Proposition 6.3 (Non-collapse)**: For any distribution with `0 < pi < 1` and
`E[|R|] > 0`, the partial-moment candidate and side-conditional candidate differ.
Specifically, `|M_pm| < m_pm` strictly when `0 < pi < 1`.
