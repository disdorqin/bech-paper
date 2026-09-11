# Host-Relative State Interaction (HRSI) — final structural closure

Token: `HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1`

- R0: `PASS`
- R1: `FAIL`
- R2: `FAIL`
- R3: `FAIL`
- R4: `PASS`

## Method

A causal recent Host-bias direction `b_hat` is built from the seven-day same-hour historical residual
already carried by the frozen S1 repair input, L2-normalized over the 24 horizon hours. The fixed
five-role standardized state table `R in R^(24x5)` multiplies it elementwise into `Z[h,k]=b_hat[h]*R[h,k]`,
and exactly ONE global `beta in R^5` shared by all six cells gives `u_HRSI = normalize(u_S1 + Z beta)`,
`beta=0` replaying S1 exactly. `alpha_HRSI` is refit per cell with the original exact weighted-median rule.

## Cell metrics

| market   | host      | method                            |   Overall_MAE |   Tail_MAE |   Normal_MAE |       MSE |   relative_gain_pct |   shape_cosine |   wrong_hemisphere_rate |     alpha |   oof_calibrated_utility |   parameter_count |   incremental_parameter_count |   training_seconds |   inference_seconds |   inference_seconds_per_day |   inference_overhead_seconds_per_day |
|:---------|:----------|:----------------------------------|--------------:|-----------:|-------------:|----------:|--------------------:|---------------:|------------------------:|----------:|-------------------------:|------------------:|------------------------------:|-------------------:|--------------------:|----------------------------:|-------------------------------------:|
| GANSU_DA | PatchTST  | HRSI_HostRelativeStateInteraction |      70.3256  |   81.7     |     69.2936  | 8004.7    |            2.39792  |       0.227484 |                0.238095 |   1.78873 |                0.043506  |             18717 |                             5 |            3.99792 |          0.00050435 |                 2.40167e-05 |                          7.50714e-06 |
| GANSU_DA | PatchTST  | Host                              |      72.0534  |  102.734   |     68.8239  | 8323.47   |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| GANSU_DA | PatchTST  | S1_DailyPatch_GRU32               |      70.4342  |   79.0745  |     69.0474  | 8005.22   |            2.24732  |       0.227242 |                0.285714 |   1.70942 |                0.0336828 |             18712 |                             5 |           10.1196  |          0.000354   |                 1.68571e-05 |                          0           |
| GANSU_DA | TimeMixer | HRSI_HostRelativeStateInteraction |      70.9695  |  127.295   |     64.3031  | 7827.61   |            6.43676  |       0.320948 |                0.190476 |   1.91307 |                0.0528985 |             18717 |                             5 |            3.99792 |          0.00049105 |                 2.33833e-05 |                          7.00952e-06 |
| GANSU_DA | TimeMixer | Host                              |      75.852   |  157.584   |     67.2486  | 8859.6    |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| GANSU_DA | TimeMixer | S1_DailyPatch_GRU32               |      70.7496  |  129.267   |     64.5898  | 7730.55   |            6.72675  |       0.331464 |                0.190476 |   2.02695 |                0.0454487 |             18712 |                             5 |           10.3743  |          0.0003397  |                 1.61762e-05 |                          0           |
| LAGO_DE  | PatchTST  | HRSI_HostRelativeStateInteraction |       4.27492 |    6.72018 |      3.90479 |   34.6912 |           10.1251   |       0.429054 |                0.137615 |   2.40456 |                0.08415   |             14877 |                             5 |            3.99792 |          0.00100585 |                 9.22798e-06 |                          2.19266e-06 |
| LAGO_DE  | PatchTST  | Host                              |       4.75652 |    7.4155  |      4.34583 |   40.9278 |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_DE  | PatchTST  | S1_DailyPatch_GRU32               |       4.24566 |    6.58172 |      3.88961 |   34.226  |           10.7403   |       0.450648 |                0.100917 |   2.40049 |                0.0884541 |             14872 |                             5 |           16.0155  |          0.0007703  |                 7.06697e-06 |                          0           |
| LAGO_DE  | TimeMixer | HRSI_HostRelativeStateInteraction |       4.33504 |    6.57462 |      3.98912 |   35.3888 |            7.8941   |       0.420172 |                0.174312 |   2.35539 |                0.073507  |             14877 |                             5 |            3.99792 |          0.00094815 |                 8.69862e-06 |                          2.18761e-06 |
| LAGO_DE  | TimeMixer | Host                              |       4.70658 |    7.14155 |      4.33048 |   40.7722 |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_DE  | TimeMixer | S1_DailyPatch_GRU32               |       4.30824 |    6.55801 |      3.96075 |   35.0767 |            8.46346  |       0.438839 |                0.146789 |   2.22927 |                0.0794982 |             14872 |                             5 |           16.0566  |          0.0007097  |                 6.51101e-06 |                          0           |
| LAGO_PJM | PatchTST  | HRSI_HostRelativeStateInteraction |       3.04934 |   10.6288  |      2.81024 |   18.8454 |           -0.984306 |       0.130798 |                0.422018 |   1.20237 |                0.0200131 |             14109 |                             5 |            3.99792 |          0.0008611  |                 7.9e-06     |                          1.85367e-06 |
| LAGO_PJM | PatchTST  | Host                              |       3.01961 |   10.5586  |      2.78179 |   19.0014 |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_PJM | PatchTST  | S1_DailyPatch_GRU32               |       3.04576 |   10.6616  |      2.80551 |   18.8677 |           -0.865822 |       0.123811 |                0.431193 |   1.13853 |                0.0202126 |             14104 |                             5 |           16.4474  |          0.0006527  |                 5.98807e-06 |                          0           |
| LAGO_PJM | TimeMixer | HRSI_HostRelativeStateInteraction |       3.06736 |    8.19044 |      2.90575 |   18.0211 |            4.77441  |       0.421383 |                0.155963 |   2.52746 |                0.101557  |             14109 |                             5 |            3.99792 |          0.00085795 |                 7.8711e-06  |                          1.87156e-06 |
| LAGO_PJM | TimeMixer | Host                              |       3.22116 |    9.62358 |      3.01919 |   20.968  |            0        |     nan        |              nan        | nan       |              nan         |               nan |                           nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_PJM | TimeMixer | S1_DailyPatch_GRU32               |       3.06124 |    8.17437 |      2.89994 |   18.0068 |            4.96469  |       0.422158 |                0.155963 |   2.43097 |                0.097623  |             14104 |                             5 |           16.5627  |          0.0006358  |                 5.83303e-06 |                          0           |

## Shape mechanism by cell

| market   | host      |   S1_shape_cosine |   HRSI_shape_cosine |   S1_wrong_hemisphere_rate |   HRSI_wrong_hemisphere_rate |   shape_cosine_change |   wrong_hemisphere_change |   HIGH_shape_cosine_change |   HIGH_wrong_hemisphere_change |
|:---------|:----------|------------------:|--------------------:|---------------------------:|-----------------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|
| GANSU_DA | PatchTST  |          0.227484 |            0.227242 |                   0.238095 |                     0.285714 |          -0.000241891 |                0.047619   |               -0.0108789   |                      0         |
| GANSU_DA | TimeMixer |          0.320948 |            0.331464 |                   0.190476 |                     0.190476 |           0.010516    |                0          |               -0.024052    |                      0         |
| LAGO_DE  | PatchTST  |          0.429054 |            0.450648 |                   0.137615 |                     0.100917 |           0.0215941   |               -0.0366972  |               -0.00857431  |                      0.0285714 |
| LAGO_DE  | TimeMixer |          0.420172 |            0.438839 |                   0.174312 |                     0.146789 |           0.0186664   |               -0.0275229  |               -0.0204621   |                      0.0555556 |
| LAGO_PJM | PatchTST  |          0.130798 |            0.123811 |                   0.422018 |                     0.431193 |          -0.00698639  |                0.00917431 |                0.00659011  |                      0         |
| LAGO_PJM | TimeMixer |          0.421383 |            0.422158 |                   0.155963 |                     0.155963 |           0.000775069 |                0          |               -0.000253528 |                      0         |

## Final global beta per seed

|   seed |   beta_DEMAND_FC |   beta_RENEWABLE_FC |   beta_SUPPLY_MARGIN_FC |   beta_INTERCHANGE_FC |   beta_MUST_RUN_FC |
|-------:|-----------------:|--------------------:|------------------------:|----------------------:|-------------------:|
|      7 |        0.10913   |            0.172179 |                       0 |             -0.264663 |         -0.133575  |
|     17 |        0.0478109 |            0.153985 |                       0 |             -0.312565 |         -0.0495669 |
|     37 |        0.0535995 |            0.143888 |                       0 |             -0.327526 |         -0.0465948 |

## Causal bias-profile availability

| market   | host      |   fit_rows |   eval_rows |   fit_rows_with_history |   eval_rows_with_history |   fit_rows_without_history |   eval_rows_without_history |   fit_rows_zero_bias |   eval_rows_zero_bias |   fit_hours_with_any_history |   eval_hours_with_any_history |   fit_min_days_per_hour |   fit_max_days_per_hour |   eval_min_days_per_hour |   eval_max_days_per_hour |   fit_mean_raw_bias_norm |   eval_mean_raw_bias_norm |   fit_bhat_norm_min |   fit_bhat_norm_max |
|:---------|:----------|-----------:|------------:|------------------------:|-------------------------:|---------------------------:|----------------------------:|---------------------:|----------------------:|-----------------------------:|------------------------------:|------------------------:|------------------------:|-------------------------:|-------------------------:|-------------------------:|--------------------------:|--------------------:|--------------------:|
| LAGO_DE  | PatchTST  |        327 |         109 |                     327 |                      109 |                          0 |                           0 |                    0 |                     0 |                           24 |                            24 |                       7 |                       7 |                        7 |                        7 |                  2.6995  |                   2.50546 |                   1 |                   1 |
| LAGO_DE  | TimeMixer |        327 |         109 |                     327 |                      109 |                          0 |                           0 |                    0 |                     0 |                           24 |                            24 |                       7 |                       7 |                        7 |                        7 |                  2.53454 |                   2.40058 |                   1 |                   1 |
| LAGO_PJM | PatchTST  |        327 |         109 |                     327 |                      109 |                          0 |                           0 |                    0 |                     0 |                           24 |                            24 |                       7 |                       7 |                        7 |                        7 |                  2.87347 |                   3.58747 |                   1 |                   1 |
| LAGO_PJM | TimeMixer |        327 |         109 |                     327 |                      109 |                          0 |                           0 |                    0 |                     0 |                           24 |                            24 |                       7 |                       7 |                        7 |                        7 |                  3.3889  |                   3.76473 |                   1 |                   1 |
| GANSU_DA | PatchTST  |         62 |          21 |                      56 |                       21 |                          6 |                           0 |                    6 |                     0 |                           24 |                            24 |                       0 |                       7 |                        3 |                        7 |                  3.67205 |                   2.8088  |                   0 |                   1 |
| GANSU_DA | TimeMixer |         62 |          21 |                      56 |                       21 |                          6 |                           0 |                    6 |                     0 |                           24 |                            24 |                       0 |                       7 |                        3 |                        7 |                  3.67912 |                   3.4748  |                   0 |                   1 |

## Required statements

1. **Do both GANSU_DA Hosts improve, under one shared beta?** No: cell-median Shape cosine changes by +0.0002 (PatchTST), -0.0105 (TimeMixer), wrong-hemisphere by -0.0476 (PatchTST), +0.0000 (TimeMixer), with no Market-ID, Host-ID, or per-cell coefficient anywhere in the model.
2. **Is the same five-parameter mechanism non-worse across all six cells?** No: relative Overall-MAE gain vs S1 is nonnegative in 1/6 cell medians, strictly better in 1/6, and Host-nonworse in 5/6 (worst -0.6151 pp vs S1).
3. **Does the Host-relative interaction resolve the HSA sign-transfer conflict?** No: international cell-median delta is -0.6151 pp (worst seed -1.3676 pp) versus HSA's worst cell of -0.87 pp. The bilateral term still does not transfer to every international cell.
4. **Is the method-development target reached?** No: R1=FAIL, R2=FAIL, R3=FAIL, R4=PASS with R0=PASS.

Median relative Overall-MAE gain vs Host: `5.605584%`; median change vs S1: `-0.240137 pp`.
Maximum Normal-MAE harm: `1.022650%`.
Median global beta-fit time: `0.9885 s`; median HRSI inference overhead: `2.1901 us/day`; median total repair inference: `8.9791 us/day`.
Trainable parameters added for the entire six-cell panel: `5`.

## Row-set-alignment diagnostic (not a gate)

HRSI's scalar alpha is calibrated on OOF folds B2..B4 (a beta cannot exist before B2), while frozen
S1 uses B1..B4. The exact weighted median is discrete, so this is a comparison confound rather than
a modelling difference. Rescoring frozen S1 on HRSI's exact row set moves its pooled alpha by up to
`26.3927%`; under that aligned row set the HRSI-minus-S1 Overall-MAE gain is
nonnegative in `3/6` cell medians, worst cell `-0.7848 pp`, worst
single seed `-1.3998 pp`. The registered R2 gate above remains defined against the
frozen S1 numbers; `alpha_rowset_sensitivity.csv` carries the per-cell detail.

Method consequence: `S1 Daily-Patch GRU32 Shape + pooled fixed alpha` is frozen as the final structural candidate. No further structural rescue is opened; the next stage is human adjudication followed by the full public-China / international comparison against the audited baseline suite.
