# Assumptions and Falsification Table

> Part of HCH v2 Derived Loss Math Design v0.1

| # | Assumption | Mathematical Role | S2/S3 Falsification Test | Fallback if Falsified |
|---|---|---|---|---|
| A1 | Host residuals `R` have finite second moment | Enables NLL as valid loss; partial moments well-defined | S2 sample variance stable across year-blocks; Hill tail index alpha_hat > 2 with bootstrap CI | Switch to L1-based NLL (Laplace) or median-based actions |
| A2 | Residuals conditionally independent across hours given `Z` | Enables hourly NLL decomposition (W1) | S2: compare day-level joint NLL vs sum of hourly NLLs; day-NLL/24 differs from mean hourly NLL by >2 SE | Adopt W2 day-level latent scale |
| A3 | Conditional density `R|Z` is unimodal | Skew-t parameterization meaningful | Hartigan dip test per state bin; multi-modal bins flagged | Consider M3 mixture-t for affected bins |
| A4 | Skewness stable across S2 to S3 | M1 generalizes; gamma(Z) holds at test | S2 gamma quantiles vs S3 empirical skewness per state bin; Spearman rho < 0.5 | Fall back to M0 |
| A5 | Tail events not dominated by <5 episodes | Parameter estimation not driven by few extreme days | Remove top-3 |residual| days from S2; re-fit; compare nu, gamma shifts | Use trimmed likelihood or robust re-weighting |
| A6 | 24h dependence captured by Z conditioning | Justifies W1 over W2 | See Section 8 dependence diagnostics | Add day-level random scale factor (W2) |
| A7 | Negative-price residual geometry shares tail with ordinary downward residuals | Cross-market negative-price transfer feasible | S2: compare FS-ST (nu, gamma) fitted to `R<0 | Y<0` vs `R<0 | Y>0`; parameter distance >2 SE | Separate nu_neg from nu_low; no transfer claim |
| A8 | Price caps/floors not creating boundary point masses | Continuous skew-t density adequate | Count exact boundary residuals (cap - yhat); if >1% of total | Add censored likelihood (Tobit-type) |
| A9 | Exogenous features Z cannot see y_t | Leakage-free residual modeling | max|corr(Z_j, y_t)| < 0.3 on S2 (existing protocol) | Hard block: experiment invalid |
| A10 | Network g_theta maintains gradient flow for all four params | Trainable NLL system | Check gradient magnitudes per parameter on S2; nu grad > 100x mu grad | Parameter-specific learning rates; stop-gradient on nu early epochs |
| A11 | Markets are exchangeable after normalization | Shared residual distribution geometry defensible | S2: Wasserstein distances between market-specific FS-ST fits; cluster by distance | Market-specific backbone + shared tail-geometry prior |
| A12 | S2 sample sufficient for 4-parameter distribution per hour-state | Reliable parameter estimation | Min(events per state-bin) > 50; otherwise aggregate neighbor bins | Coarser state grid or Bayesian prior shrinkage |
