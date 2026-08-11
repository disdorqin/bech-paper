# Rejection-Risk Register

> Part of HCH v2 Derived Loss Math Design v0.1

| # | Risk | Severity | What a Reviewer Would Say | Mitigation |
|---|---|---|---|---|
| R1 | **Partial moment = double-shrunk heuristic** | HIGH | "Multiplying occurrence probability by conditional mean yields a correction near zero for rare events. This is ad-hoc and not Bayes-optimal." | Pre-compute M0/M1 with BOTH candidate types (M_pm and m_cond); if DVG consistently prefers m_cond, abandon M_pm. |
| R2 | **FS skew-t is NOT a novel distribution** | HIGH | "Fernandez-Steel skew-t has been used in finance since 1998. Using it for electricity price residuals is an application, not a method contribution." | Position novelty in (a) bidirectional candidate derivation, (b) CAGM-DVG routing, (c) frozen-host post-processing. NOT in the distribution choice. |
| R3 | **Identifiability: mu, sigma, nu, gamma confounded** | MEDIUM | "With only hourly residuals and 4 NN-output parameters, the model may be poorly identified. Small changes in gamma can be absorbed by changes in sigma or mu." | Diagnostic F3 (Hessian rank); if condition number > 1e6, reduce to 3 params (fix nu or use gamma-only skew without mu). |
| R4 | **Price caps/floors violate continuity** | MEDIUM | "More than 1% of residuals sit at price-cap boundaries. A continuous skew-t cannot model these point masses." | Diagnostic A8; if censored mass > 1%, add censored likelihood or exclude boundary observations from fitting. |
| R5 | **nu estimated near boundary (nu ~ 2)** | MEDIUM | "If the best-fitting nu is near 2, variance does not exist. Using variance-based diagnostics (PIT, CRPS) is invalid." | Use quantile-based diagnostics (not variance-based); add penalty to keep nu > 3; report raw nu estimates. |
| R6 | **Cross-market transfer fails** | MEDIUM | "Claims of 'cross-market generalization' are unsupported when the model needs market-specific bias terms for adequate performance." | Only claim cross-market AFTER leave-one-out validation on S3 passes; report per-market results honestly. |
| R7 | **CAGM-DVG risk double-counting** | MEDIUM | "The DVG receives aleatoric variance AND retrieval variance as features. If these are correlated, the model may overweight tail-related risks and never correct." | Diagnostic: compute correlation between U_aleatoric and U_retrieval on S3; if corr > 0.7, combine into single risk feature. |
| R8 | **S3 lambda calibration is S3-optimistic** | HIGH | "Lambda is tuned on S3 and evaluated on S3. This is in-sample optimization, not genuine held-out calibration." | The SCARR protocol uses S3's own held-out folds within the calibration phase. Verify that grid search does not see evaluation data. |
| R9 | **A few outlier days dominate nu/gamma estimates** | MEDIUM | "A single day with extreme residual magnitude drives the heavy-tailed fit. Remove those 3 days and the Student-t advantage disappears." | Diagnostic A5; report trimmed vs raw results; if sensitive, use robust M-estimation or trimmed likelihood. |
| R10 | **Up and Down candidates collapse to similar values** | LOW | "For most predictions, pi_neg ~ 0.5 and pi_pos ~ 0.5, giving M_neg ~ M_pos in magnitude. The directional structure is decorative." | Proposition 6.3 guarantees non-collapse when 0 < pi < 1; verify empirically on S3. |
| R11 | **No baseline comparison with simpler corrector** | HIGH | "Why not just use a single LightGBM that predicts y directly from Z+host? The complex distribution+memory+gate architecture adds unjustified complexity." | Always compare against: (1) direct LightGBM corrector, (2) quantile regression corrector, (3) identity. Distribution-based method must beat (1) or (2) with statistical significance. |
| R12 | **Computational cost obscures marginal gain** | LOW | "The FS skew-t + quadrature + CAGM retrieval takes 10x the compute of a linear correction with 90% of the accuracy." | Report wall-clock times; if >5x slower than linear corrector with <5% gain, note this limitation explicitly. |

## 13.1 Top-3 Kill Risks

1. **R1 (Partial moment is heuristic)**: If DVG consistently selects side-conditional
   mean over partial moment, the distribution-derived candidate is not useful.
2. **R8 (S3 lambda is optimistic)**: If lambda chosen on S3 doesn't hold on a
   genuine holdout within S3, the safety certificate is invalid.
3. **R2 (No novelty in FS skew-t)**: The paper must clearly distinguish "using a
   known distribution" from "a novel decision-theoretic correction framework."
