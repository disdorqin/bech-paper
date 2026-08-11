# Ablation and Diagnostics Design

> Part of HCH v2 Derived Loss Math Design v0.1

## 12.1 Core Ablation Ladder

| # | Ablation | What Changes | Hypothesis | Minimal N |
|---|---|---|---|---|
| A0 | **M0 (sym-t) vs M1 (FS skew-t)** | gamma replaced by 1.0 | M1 NLL < M0 NLL on S3; gamma improves tail calibration | S2-OOF residuals |
| A1 | **Old loss (L2 + BCE) vs new NLL** | Separate classification/regression heads -> unified FS skew-t head | New NLL yields better S3 candidate quality (higher MAE gain) | S2 + S3 |
| A2 | **Partial moment vs side-conditional mean** | Candidate = M_pm vs m_pm | Which candidate type is selected more by DVG on S3 | S3 DVG |
| A3 | **W1 vs W2** | Add day-level latent scale | W2 S3 NLL < W1 S3 NLL (joint dependence matters) | S2 + S3 |
| A4 | **State conditioning (s_t) vs no state** | Remove s_t from g_theta inputs | State conditioning improves tail calibration (PIT uniformity) | S3 |
| A5 | **Market-specific gamma vs shared gamma** | gamma unified across markets | Shared gamma hurts S3 NLL in markets with different skew directions | S3 |
| A6 | **S3 lambda calibration vs hard threshold** | Replace SCARR with lambda=1 at tau=0.5 | S3 calibration prevents harm while lambda=1 may degrade normal periods | S3 |

## 12.2 Diagnostic Criteria (not "winners")

For each ablation, report:
1. **S3 NLL difference** with block-bootstrap CI (not just point estimate)
2. **Tail calibration**: PIT uniformity in upper/lower 5% quantiles
3. **Candidate gain distribution**: CAGM-DVG selected action gains
4. **Normal-period harm**: MAE degradation for |residual| < IQR thresholds
5. **Parameter stability**: Variance of fitted parameters across S2 bootstrap samples
6. **Computational cost**: Training time and memory (secondary)

## 12.3 Minimum Comparison Effort

Only compare the smallest justified nesting:
- **M0 vs M1** first (is asymmetry needed?)
- If M1 > M0: **M1-cond vs M1-global** (does state conditioning help?)
- If M1 passes: **Partial moment vs conditional mean** (candidate type)
- Only IF W1 diagnostics fail: **W1 vs W2**

Do NOT compare all 6x6 = 36 combinations. Follow the nested order.

## 12.4 Multiplicity Correction

When testing across multiple markets, hosts, and abatations:
- Report ALL results, not cherry-picked
- Use macro-average (equal weight per market) and micro-average (sample-weighted)
- Mark clearly: "exploratory on S3" vs "confirmatory on S4" (S4 prohibited here)
- No p-hacking: declare test plan before seeing S3 results

## 12.5 Table Template

Expected output table for M0 vs M1:

| Market | Host | M0 S3 NLL | M1 S3 NLL | NLL diff | 95% CI | M0 PIT p-val | M1 PIT p-val | gamma_hat | gamma SE |
|---|---|---|---|---|---|---|---|---|---|
| LAGO_DE | Linear | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 12.6 Condorcet / Ensemble of diagnostics

No single metric selects the model. Use a "diagnostic panel":

| Criterion | Threshold | M0 Pass? | M1 Pass? |
|---|---|---|---|
| S3 NLL < baseline | NLL diff CI excludes 0 | ✓/✗ | ✓/✗ |
| PIT KS p-val > 0.05 | p > 0.05 for both tails | ✓/✗ | ✓/✗ |
| gamma 95% CI excludes 1.0 | CI does not contain 1 | ✗ (by design) | ✓ |
| nu > 3 with CI | Lower CI > 3 | ✓/✗ | ✓/✗ |
| condition_number(H) < 1e6 | Identifiable | ✓ | ✓/✗ |
| S2->S3 parameter stability | shift < 0.2 SD | ✓/✗ | ✓/✗ |

A model with >= 4/6 passes is considered adequate. M1 only recommended over
M0 if it passes criteria that M0 fails.
