# Math Evidence Audit — Executive Verdict
> Date: 2026-08-11 (RE-AUDITED 2026-08-12: CDF/bootstrapping/day-matrix corrected)
>
> **NOTE (2026-08-12): 06_CANDIDATE_ACTIONS.csv is INVALID_ORACLE_DIAGNOSTIC.
> Phase C computed pi/m on S3 itself and used S3 residual sign as gate — not a
> deployable candidate. All "partial moment viable" conclusions are WITHDRAWN.**

## Data Scope
- 4 datasets: LAGO_BE, LAGO_DE, LAGO_FR, LAGO_PJM
- 5 backbones: LSTM, Linear, MLP, PatchTST, TCN
- 18 dataset x backbone combinations

## Distribution Fit Results
- Student-t NLL < Normal NLL (S3, CI excludes 0): 17/18
- Student-t NLL < Laplace NLL (S3, CI excludes 0): 12/18
- Fitted nu range: [3.0, 15.0], median: 7.0

## Tail Stability
- |S3 skew - S2 skew| median: 3.373
- |S3 skew - S2 skew| max: 7.503

## 24h Dependence
- Effective rank median: 3.0 / 24
- First eigenvalue ratio median: 0.904
- Max off-diagonal corr median: 0.949
- W1 (hourly independence) appears adequate: 0/18

## Assumption Verdicts
1. A1 (finite variance): Student-t consistently better than Normal → heavy tails confirmed
2. A4: skew not stable across splits; caution with M1
3. A6: significant day-level dependence; consider W2

## Status
**PARTIAL_BLOCKED** — Core S2/S3 evidence produced.
Missing: NEM/EPEX/GEFCOM markets (need host_cache), FS skew-t (M1), CAGM/DVG artifacts.
