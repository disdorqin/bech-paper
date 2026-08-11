# Literature and Claim Boundary

> Part of HCH v2 Derived Loss Math Design v0.1

## 3.1 Prior Work and Collision Assessment

| Work | Real Contribution | Overlap | Collision |
|---|---|---|---|
| **TFT** (Lim+, [arxiv:1912.09363](https://arxiv.org/abs/1912.09363)) | Quantile loss for multi-horizon probabilistic forecasts; VSN | Both use heavy-tailed TSF losses | **None.** TFT uses pinball (quantile) loss, NOT Student-t NLL. |
| **N-BEATS** (Oreshkin+, [arxiv:1905.10437](https://arxiv.org/abs/1905.10437)) | DL basis-expansion for point forecasts; sMAPE/MASE training | Both forecast time series | **None.** Main N-BEATS training is sMAPE/MASE, not Student-t NLL. |
| **DeepAR** (Salinas+, [arxiv:1704.04110](https://arxiv.org/abs/1704.04110)) | Parametric distribution output from RNN; NLL training | Both use likelihood-based NN training | Student-t likelihood per se is NOT novel. We must not claim "using Student-t NLL for TSF" as novel. |
| **FS Skewing** (Fernandez & Steel 1998, [doi:10.1080/01621459.1998.10473708](https://doi.org/10.1080/01621459.1998.10473708)) | Two-piece skew-t via inverse-scale factor gamma | Both use FS skew-t density | **None.** FS is a standard tool. We use it; we don't invent it. |
| **Gen. Asymmetric t** (Zhu & Galbraith 2011, [doi:10.1016/j.jeconom.2009.11.001](https://doi.org/10.1016/j.jeconom.2009.11.001)) | Two-piece t with different df per side | Both consider asymmetric t | **Partial.** Two-df version (nu_L ne nu_R) would collide. Use single-nu FS first. |
| **Distributional NNs for EPF** (Marcjasz+, [arxiv:2207.02832](https://arxiv.org/abs/2207.02832)) | DL outputs Johnson's SU distribution params; SOTA EPF forecasts | Both use parametric distributions for EPF | **None on Student-t.** They use Normal/Johnson's SU. Distributional NN for EPF is established. |
| **Tail Risk of Electricity Futures** (Kostrzewska+, [arxiv:2202.01732](https://arxiv.org/abs/2202.01732)) | AR-GARCH with Student-t for VaR/ES on EPF | Both use Student-t for electricity tail risk | **None.** Their use is univariate GARCH, not conditional neural posterior correction. |
| **Taming the Long Tail** (Hoskere+, [arxiv:2202.13418](https://arxiv.org/abs/2202.13418)) | Pareto-tailed loss for deep probabilistic forecasting | Both address heavy tails in probabilistic TSF | **Partial awareness.** Tail-focused loss variants exist. Differentiate via bidirectional structure. |
| **CRC** (Yao+, [arxiv:2512.22428](https://arxiv.org/abs/2512.22428)) | Causality-inspired safe residual correction for MTS | Both do post-hoc correction with safety constraint | **Direct competitor.** CRC is the closest existing work. We differ by (a) distribution-derived candidates vs greedy residual correction, (b) episodic memory routing vs causality constraints. |
| **delta-Adapter** (Liu+, [arxiv:2601.20280](https://arxiv.org/abs/2601.20280)) | Lightweight post-processing adapter for frozen forecasters | Both do post-hoc correction of frozen forecasters | **Direct competitor.** delta-Adapter uses adapter layers; we use distribution-derived partial moments + memory routing. |
| **CRAFTER** (Li+, [arxiv:2608.05207](https://arxiv.org/abs/2608.05207)) | Automated corrective feature discovery on frozen forecaster residuals | Both mine frozen forecaster residuals for correction | **Very recent (Aug 2026).** CRAFTER focuses on feature discovery; we focus on distribution-derived action semantics. Overlap in "frozen + correct" paradigm. |
| **ARC-STAR** (Zhang+, [arxiv:2605.22222](https://arxiv.org/abs/2605.22222)) | Risk-calibrated spatial triage for PDE foundation model correction | Both do frozen-model post-hoc correction with risk calibration | **Different domain.** PDE vs electricity. Shared concept of selective/non-uniform correction. |

## 3.2 What This Project Can Claim (Tentative, Evidence-Dependent)

1. **Bidirectional residual correction from a single conditional distribution** -- eliminating separate classification + regression heads for Up/Down by deriving both occurrence and magnitude from FS skew-t moments.

2. **Episodic action-value routing** -- CAGM retrieves 24h-day historical action gains; DVG selects Identity/Down/Up based on risk-adjusted value, NOT on pointwise NLL.

3. **State-conditioned heavy-tail modeling for frozen hosts** -- the continuous state Z (host prediction rank, residual history, calendar, exogenous forecasts) modulates all four distribution parameters, adapting tail behavior per context.

4. **Normal-harm-constrained selective routing** -- S3 calibration enforces that expected harm in normal periods stays below budget, while tail benefits must clear a bootstrap LCB.

## 3.3 What This Project Cannot Claim

See Section 15 of the dossier for the forbidden claims list. Additionally:
- Cannot claim "first to use Student-t for electricity price forecasting"
- Cannot claim "first distribution-derived correction" (CRC and delta-Adapter predate)
- Cannot claim "model-agnostic" until host diversity (Linear through PatchTST) is empirically verified
- Cannot claim "cross-market generalization" until leave-one-market-out transfer is validated on S3
