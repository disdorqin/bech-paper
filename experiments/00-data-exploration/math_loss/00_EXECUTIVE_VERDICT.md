# HCH v2 Loss Math Design — Executive Verdict

> Version: v0.1 | Date: 2026-08-11
> Status: `READY FOR ARCHITECTURE REVIEW`
> Based on: `hch_v2_math_loss_research_dossier_and_prompt_v0.1_2026-08-11.md`

---

## 1.1 Recommended Minimal Model

**M1: State-conditional Fernandez-Steel skew-t (FS-ST)** with shared backbone:

- Network `g_theta(Z)` outputs `(mu, log_sigma, log_nu_tilde, log_gamma)`
- `nu = 2 + softplus(nu_tilde)` and `gamma` in R+ controls skew
- Both Up/Down candidates from same conditional distribution via partial-moment decomposition
- Training: NLL of FS skew-t density; gradient detached to candidate action computation

## 1.2 Is Asymmetric Student-t Really Needed?

**Yes, with precise scope.** Symmetric Student-t (M0) handles heavy tails (kurtosis median ~10.3)
but cannot model directional asymmetry (skewness median ~0.63, peaks of 12.27 for NEM).
The asymmetry is NOT universal -- it is host-dependent and market-dependent.

M1 value proposition:
1. Single NLL generates both occurrence probability `pi_pm(Z)` and conditional magnitude
   `m_pm(Z)` from same density, eliminating separate classification + regression heads
2. `gamma` enables different Up/Down tail behavior while sharing `nu` (natural partial-pooling)
3. `gamma -> 1` degrades to symmetric t; `nu -> inf` to skew-normal: clean nested ablation

## 1.3 Maximum Mathematical Risks

1. **Identifiability**: Neural net estimating `mu, sigma, nu, gamma` simultaneously may be
   practically underdetermined. `mu(Z)` vs `gamma(Z)` can produce similar quantile shifts.
2. **Partial moment shrinkage**: `M_pm(Z) = pi_pm * m_pm` doubles the shrinkage. For rare
   events (pi ~ 0.01), correction collapses near-zero regardless of true magnitude.
3. **nu near 2**: When tails are extremely heavy, `nu` approaches moment-existence boundary,
   causing gradient explosion in NLL.

## 1.4 Whether to Proceed

**Proceed with explicit guardrails:**
- Implement M0 first; add M1 only after M0 passes unit tests
- Compare `M_pm` against side-conditional mean; if shrinkage dominates, switch to
  `sgn(z) * m_|z|` or calibrated quantile action
- Build S2/S3 diagnostics for identifiability failure and `nu` boundary hits

## 1.5 Core Differentiability Claim (Tentative)

> Existing heavy-tail likelihoods primarily target direct probabilistic forecasts. This
  project converts state-conditional heavy-tail residual distributions into directional
  decision quantities (Up/Down candidates) for a frozen host, routed by episodic action-gain
  memory relative to Identity. Innovation, if it holds, comes from coupling
  "distribution-derived bidirectional candidates + episodic action-value routing" --
  NOT from Student-t or skew-t distributions themselves.

**Status**: This claim is **mathematically defensible** but requires the empirical evidence
audit to confirm that the FS skew-t partial-moment candidates are materially different from
separate classifiers + regressors.

## 1.6 Overall Status

`READY FOR ARCHITECTURE REVIEW` -- 14 sections complete. The following are complete:
derivations (FS skew-t density, CDF, NLL, moments, partial moments for threshold=0),
Bayes action analysis, identifiability diagnosis, 24h dependence recommendation (W1),
CAGM-DVG risk decomposition, numerical implementation pseudocode, unit tests,
falsification tests, ablation plan, risk register, theorem/proposition set,
implementation contract, and allowed/forbidden claims.

**Input needed from empirical audit**: confirmation of A2, A4, A5, A6, A7, A8 from the
assumptions table before finalizing M1 over M0.
