# PILOT_VERDICT (v7-B-R2)

> Runtime: 80.2s | Seed: 42

## Summary

| Dataset | Backbone | Method | Recall | Miss% | MAE | Normal MAE |
|---------|----------|--------|--------|-------|-----|------------|
| LAGO_DE | linear | B0_linear | 0.161 | 83.9% | 24.6045 | 24.1124 |
| LAGO_DE | linear | B1_linear | 0.144 | 85.6% | 24.6010 | 24.0976 |
| LAGO_DE | linear | B2_linear | 0.136 | 86.4% | 24.6032 | 24.1061 |
| LAGO_DE | linear | B3_linear | 0.161 | 83.9% | 24.6045 | 24.1124 |
| LAGO_DE | linear | B4_linear | 0.136 | 86.4% | 24.6012 | 24.1015 |
| LAGO_DE | linear | E0_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| LAGO_DE | linear | E0_NoInsert_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| LAGO_DE | linear | E0_NoShift_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| LAGO_DE | linear | E0_NoMatching_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| LAGO_DE | linear | Oracle_linear | 1.000 | 0.0% | 0.0000 | 0.0000 |
| LAGO_DE | gbdt | B0_gbdt | 0.186 | 81.4% | 39.0675 | 39.0274 |
| LAGO_DE | gbdt | B1_gbdt | 0.102 | 89.8% | 39.0316 | 38.9660 |
| LAGO_DE | gbdt | B2_gbdt | 0.153 | 84.7% | 39.0085 | 38.9211 |
| LAGO_DE | gbdt | B3_gbdt | 0.186 | 81.4% | 39.0675 | 39.0274 |
| LAGO_DE | gbdt | B4_gbdt | 0.127 | 87.3% | 39.0382 | 38.9837 |
| LAGO_DE | gbdt | E0_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| LAGO_DE | gbdt | E0_NoInsert_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| LAGO_DE | gbdt | E0_NoShift_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| LAGO_DE | gbdt | E0_NoMatching_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| LAGO_DE | gbdt | Oracle_gbdt | 1.000 | 0.0% | 0.0000 | 0.0000 |
| NEM_SA1 | linear | B0_linear | 0.000 | 0.0% | 7.4195 | 7.4364 |
| NEM_SA1 | linear | B1_linear | 0.000 | 0.0% | 7.4195 | 7.4364 |
| NEM_SA1 | linear | B2_linear | 0.000 | 0.0% | 7.4195 | 7.4364 |
| NEM_SA1 | linear | B3_linear | 0.000 | 0.0% | 7.4195 | 7.4364 |
| NEM_SA1 | linear | B4_linear | 0.000 | 0.0% | 7.4195 | 7.4364 |
| NEM_SA1 | linear | E0_linear | 0.000 | 0.0% | 7.4390 | 7.4601 |
| NEM_SA1 | linear | E0_NoInsert_linear | 0.000 | 0.0% | 7.4390 | 7.4601 |
| NEM_SA1 | linear | E0_NoShift_linear | 0.000 | 0.0% | 7.4390 | 7.4601 |
| NEM_SA1 | linear | E0_NoMatching_linear | 0.000 | 0.0% | 7.4390 | 7.4601 |
| NEM_SA1 | linear | Oracle_linear | 0.000 | 0.0% | 0.0000 | 0.0000 |
| NEM_SA1 | gbdt | B0_gbdt | 0.000 | 0.0% | 9.5537 | 9.5766 |
| NEM_SA1 | gbdt | B1_gbdt | 0.000 | 0.0% | 9.5501 | 9.5729 |
| NEM_SA1 | gbdt | B2_gbdt | 0.000 | 0.0% | 9.5734 | 9.5964 |
| NEM_SA1 | gbdt | B3_gbdt | 0.000 | 0.0% | 9.5537 | 9.5766 |
| NEM_SA1 | gbdt | B4_gbdt | 0.000 | 0.0% | 9.5537 | 9.5766 |
| NEM_SA1 | gbdt | E0_gbdt | 0.000 | 0.0% | 9.5798 | 9.6016 |
| NEM_SA1 | gbdt | E0_NoInsert_gbdt | 0.000 | 0.0% | 9.5798 | 9.6016 |
| NEM_SA1 | gbdt | E0_NoShift_gbdt | 0.000 | 0.0% | 9.5798 | 9.6016 |
| NEM_SA1 | gbdt | E0_NoMatching_gbdt | 0.000 | 0.0% | 9.5798 | 9.6016 |
| NEM_SA1 | gbdt | Oracle_gbdt | 0.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_DE | linear | B0_linear | 0.161 | 83.9% | 24.6045 | 24.1124 |
| UNIELEC_DE | linear | B1_linear | 0.144 | 85.6% | 24.6010 | 24.0976 |
| UNIELEC_DE | linear | B2_linear | 0.136 | 86.4% | 24.6032 | 24.1061 |
| UNIELEC_DE | linear | B3_linear | 0.161 | 83.9% | 24.6045 | 24.1124 |
| UNIELEC_DE | linear | B4_linear | 0.136 | 86.4% | 24.6012 | 24.1015 |
| UNIELEC_DE | linear | E0_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| UNIELEC_DE | linear | E0_NoInsert_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| UNIELEC_DE | linear | E0_NoShift_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| UNIELEC_DE | linear | E0_NoMatching_linear | 0.161 | 83.9% | 24.6026 | 24.1004 |
| UNIELEC_DE | linear | Oracle_linear | 1.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_DE | gbdt | B0_gbdt | 0.186 | 81.4% | 39.0675 | 39.0274 |
| UNIELEC_DE | gbdt | B1_gbdt | 0.102 | 89.8% | 39.0316 | 38.9660 |
| UNIELEC_DE | gbdt | B2_gbdt | 0.153 | 84.7% | 39.0085 | 38.9211 |
| UNIELEC_DE | gbdt | B3_gbdt | 0.186 | 81.4% | 39.0675 | 39.0274 |
| UNIELEC_DE | gbdt | B4_gbdt | 0.127 | 87.3% | 39.0382 | 38.9837 |
| UNIELEC_DE | gbdt | E0_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| UNIELEC_DE | gbdt | E0_NoInsert_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| UNIELEC_DE | gbdt | E0_NoShift_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| UNIELEC_DE | gbdt | E0_NoMatching_gbdt | 0.186 | 81.4% | 39.0560 | 39.0283 |
| UNIELEC_DE | gbdt | Oracle_gbdt | 1.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_FI | linear | B0_linear | 0.059 | 94.1% | 32.6265 | 33.1746 |
| UNIELEC_FI | linear | B1_linear | 0.059 | 94.1% | 32.6265 | 33.1746 |
| UNIELEC_FI | linear | B2_linear | 0.059 | 94.1% | 32.6265 | 33.1746 |
| UNIELEC_FI | linear | B3_linear | 0.059 | 94.1% | 32.6265 | 33.1746 |
| UNIELEC_FI | linear | B4_linear | 0.059 | 94.1% | 32.6265 | 33.1746 |
| UNIELEC_FI | linear | E0_linear | 0.059 | 94.1% | 32.6415 | 33.1818 |
| UNIELEC_FI | linear | E0_NoInsert_linear | 0.059 | 94.1% | 32.6415 | 33.1818 |
| UNIELEC_FI | linear | E0_NoShift_linear | 0.059 | 94.1% | 32.6415 | 33.1818 |
| UNIELEC_FI | linear | E0_NoMatching_linear | 0.059 | 94.1% | 32.6415 | 33.1818 |
| UNIELEC_FI | linear | Oracle_linear | 1.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_FI | gbdt | B0_gbdt | 0.000 | 100.0% | 37.6157 | 38.3108 |
| UNIELEC_FI | gbdt | B1_gbdt | 0.000 | 100.0% | 37.6157 | 38.3108 |
| UNIELEC_FI | gbdt | B2_gbdt | 0.000 | 100.0% | 37.6157 | 38.3108 |
| UNIELEC_FI | gbdt | B3_gbdt | 0.000 | 100.0% | 37.6157 | 38.3108 |
| UNIELEC_FI | gbdt | B4_gbdt | 0.000 | 100.0% | 37.6157 | 38.3108 |
| UNIELEC_FI | gbdt | E0_gbdt | 0.000 | 100.0% | 37.6302 | 38.3013 |
| UNIELEC_FI | gbdt | E0_NoInsert_gbdt | 0.000 | 100.0% | 37.6302 | 38.3013 |
| UNIELEC_FI | gbdt | E0_NoShift_gbdt | 0.000 | 100.0% | 37.6302 | 38.3013 |
| UNIELEC_FI | gbdt | E0_NoMatching_gbdt | 0.000 | 100.0% | 37.6302 | 38.3013 |
| UNIELEC_FI | gbdt | Oracle_gbdt | 1.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_NL | linear | B0_linear | 0.210 | 79.0% | 24.9872 | 23.7874 |
| UNIELEC_NL | linear | B1_linear | 0.198 | 80.2% | 24.9778 | 23.7767 |
| UNIELEC_NL | linear | B2_linear | 0.142 | 85.8% | 25.0633 | 23.7990 |
| UNIELEC_NL | linear | B3_linear | 0.210 | 79.0% | 24.9872 | 23.7874 |
| UNIELEC_NL | linear | B4_linear | 0.160 | 84.0% | 24.9638 | 23.7469 |
| UNIELEC_NL | linear | E0_linear | 0.210 | 79.0% | 24.9875 | 23.7752 |
| UNIELEC_NL | linear | E0_NoInsert_linear | 0.210 | 79.0% | 24.9875 | 23.7752 |
| UNIELEC_NL | linear | E0_NoShift_linear | 0.210 | 79.0% | 24.9875 | 23.7752 |
| UNIELEC_NL | linear | E0_NoMatching_linear | 0.210 | 79.0% | 24.9875 | 23.7752 |
| UNIELEC_NL | linear | Oracle_linear | 1.000 | 0.0% | 0.0000 | 0.0000 |
| UNIELEC_NL | gbdt | B0_gbdt | 0.000 | 100.0% | 38.0953 | 37.0790 |
| UNIELEC_NL | gbdt | B1_gbdt | 0.000 | 100.0% | 38.0953 | 37.0790 |
| UNIELEC_NL | gbdt | B2_gbdt | 0.000 | 100.0% | 38.0953 | 37.0790 |
| UNIELEC_NL | gbdt | B3_gbdt | 0.000 | 100.0% | 38.0953 | 37.0790 |
| UNIELEC_NL | gbdt | B4_gbdt | 0.000 | 100.0% | 38.0953 | 37.0790 |
| UNIELEC_NL | gbdt | E0_gbdt | 0.000 | 100.0% | 38.0631 | 37.0822 |
| UNIELEC_NL | gbdt | E0_NoInsert_gbdt | 0.000 | 100.0% | 38.0631 | 37.0822 |
| UNIELEC_NL | gbdt | E0_NoShift_gbdt | 0.000 | 100.0% | 38.0631 | 37.0822 |
| UNIELEC_NL | gbdt | E0_NoMatching_gbdt | 0.000 | 100.0% | 38.0631 | 37.0822 |
| UNIELEC_NL | gbdt | Oracle_gbdt | 1.000 | 0.0% | 0.0000 | 0.0000 |

## Stop Conditions

**Verdict: STOP**

- LAGO_DE/linear: E0 recall = B0 recall (0.161)
- LAGO_DE/gbdt: E0 recall = B0 recall (0.186)
- NEM_SA1/linear: E0 recall = B0 recall (0.000)
- NEM_SA1/gbdt: E0 recall = B0 recall (0.000)
- UNIELEC_DE/linear: E0 recall = B0 recall (0.161)
- UNIELEC_DE/gbdt: E0 recall = B0 recall (0.186)
- UNIELEC_FI/linear: E0 recall = B0 recall (0.059)
- UNIELEC_FI/gbdt: E0 recall = B0 recall (0.000)
- UNIELEC_NL/linear: E0 recall = B0 recall (0.210)
- UNIELEC_NL/gbdt: E0 recall = B0 recall (0.000)
