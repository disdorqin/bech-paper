# HCH v2 Loss Math Design — Executive Verdict

> Version: v0.2 | Date: 2026-08-11 (updated with audit evidence)
> Status: `READY FOR ARCHITECTURE REVIEW` (audit-backed revision)
> Based on: `hch_v2_math_loss_research_dossier_and_prompt_v0.1_2026-08-11.md`
> Audit: `outputs/00_EXECUTIVE_EVIDENCE_VERDICT.md` (18 combos, 4 datasets x 5 backbones)

---

## 1.1 Recommended Minimal Model [REVISED per audit]

**M0: State-conditional symmetric Student-t** with **W2 day-level latent scale**:

- Network `g_theta(Z)` outputs `(mu, log_sigma, log_nu_tilde)`, with `nu = 2 + softplus(nu_tilde)`
- Both Up/Down candidates from same distribution via partial-moment decomposition
- **W2**: add day-level random scale factor `eta_d ~ N(0,1)` for 24h joint dependence
- Training: NLL of symmetric Student-t (hourly) + day-level latent marginalization
- M1 (FS skew-t) is deferred: audit found S2->S3 skew instability (median |shift| = 3.37)

## 1.2 Distribution Choice: Audit Evidence

**S2/S3 audit on 18 combos (4 LAGO datasets x 5 backbones):**

| Finding | Evidence | Implication |
|---|---|---|
| Heavy tails confirmed | Student-t NLL < Normal NLL: **17/18**; nu median 7.0, range [3,15] | M0 validated |
| Student-t vs Laplace | t better: 12/18 (CI excludes 0) | t over Laplace |
| **Skew NOT stable** | S2->S3 skew shift median **3.37**, max 7.5 | **M1 deferred**; gamma won't generalize |
| **24h dependence strong** | Effective rank 2-5/24, first eval 82-96%, max corr 0.89-0.99 | **W2 required**; hourly NLL not enough |
| Partial moment viable | harm_rate ~10% on LAGO_BE vs ~17-22% for side-cond-mean | Keep M_pm candidate |

## 1.3 Maximum Mathematical Risks [UNCHANGED + new]

1. **Identifiability**: Neural net estimating `mu, sigma, nu` simultaneously may be
   practically underdetermined — but removing gamma helps.
2. **Partial moment shrinkage**: `M_pm(Z) = pi_pm * m_pm` doubles the shrinkage. For rare
   events (pi ~ 0.01), correction collapses near-zero regardless of true magnitude.
3. **nu near 2**: When tails are extremely heavy, `nu` approaches moment-existence boundary,
   causing gradient explosion in NLL.
4. **[NEW] S2->S3 skew instability**: Asymmetry learned on S2 does not hold on S3. Even if M1
   eventually needed, it requires market-specific gamma or stronger regularization.

## 1.4 Whether to Proceed [REVISED]

**Proceed with revised plan:**
- Implement **M0 + W2**: symmetric Student-t with day-level latent scale
- Defer M1 until: (a) 24h dependence is addressed by W2, (b) skew-stability issue is resolved
  via either per-market gamma or hierarchical prior
- Compare `M_pm` against side-conditional mean; audit shows M_pm has lower harm_rate
- Build S2/S3 diagnostics for identifiability failure, nu boundary hits, and W2 necessity

## 1.5 Core Differentiability Claim [UNCHANGED]

> Existing heavy-tail likelihoods primarily target direct probabilistic forecasts. This
  project converts state-conditional heavy-tail residual distributions into directional
  decision quantities (Up/Down candidates) for a frozen host, routed by episodic action-gain
  memory relative to Identity. Innovation, if it holds, comes from coupling
  "distribution-derived bidirectional candidates + episodic action-value routing" --
  NOT from Student-t or skew-t distributions themselves.

**Status**: Mathematically defensible; audit confirms heavy tails (17/18) and shows
partial-moment candidates have favorable harm profile.

## 1.6 Overall Status

`READY FOR ARCHITECTURE REVIEW` — 14 sections complete, **revised per audit evidence**.
Key revisions: M0 over M1 (skew unstable), W2 over W1 (24h dependence strong).
Next step: architecture review → implement M0+W2 → retest with NEM/EPEX/GEFCOM data.
