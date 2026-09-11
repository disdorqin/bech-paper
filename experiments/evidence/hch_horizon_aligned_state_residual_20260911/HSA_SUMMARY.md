# Horizon-Aligned Semantic State Residual (HSA) Shape Closure

Token: `HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`

- H0: `PASS`
- H1: `FAIL`
- H2: `FAIL`
- H3: `FAIL`
- H4: `PASS`

## Method

Frozen S1 OOF Shape directions are reused unchanged. A deterministic horizon-aligned semantic role table `R_d in R^(24x5)` is formed by mean-pooling audited `primary_channel=True` forecast-time state columns into `DEMAND_FC / RENEWABLE_FC / SUPPLY_MARGIN_FC / INTERCHANGE_FC / MUST_RUN_FC` (absent role = structural 0). The only new trainable object is `beta in R^5`, giving `u_HSA = normalize(u_S1 + R_d beta)`; `alpha_HSA` is refit with the original exact weighted-median rule on HSA OOF directions.

## Cell metrics

| market   | host      | method                          |   seed |   Overall_MAE |   Tail_MAE |   Normal_MAE |       MSE |   relative_gain_pct |   shape_cosine |   wrong_hemisphere_rate |     alpha |   oof_calibrated_utility |   parameter_count |   training_seconds |   inference_seconds |   inference_seconds_per_day |   inference_overhead_seconds_per_day |
|:---------|:----------|:--------------------------------|-------:|--------------:|-----------:|-------------:|----------:|--------------------:|---------------:|------------------------:|----------:|-------------------------:|------------------:|-------------------:|--------------------:|----------------------------:|-------------------------------------:|
| GANSU_DA | PatchTST  | HSA_HorizonAlignedStateResidual |     17 |      69.498   |   85.8224  |     67.7797  | 7906.16   |            3.54653  |       0.293865 |               0.238095  |   1.09493 |                0.0233246 |             18717 |            1.5363  |          0.0004056  |                 1.93143e-05 |                          4.87381e-06 |
| GANSU_DA | PatchTST  | Host                            |     17 |      72.0534  |  102.734   |     68.8239  | 8323.47   |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| GANSU_DA | PatchTST  | S1_DailyPatch_GRU32             |     17 |      70.4342  |   79.0745  |     69.0474  | 8005.22   |            2.24732  |       0.227242 |               0.285714  |   1.70942 |                0.0336828 |             18712 |           13.0543  |          0.0003056  |                 1.45524e-05 |                          0           |
| GANSU_DA | TimeMixer | HSA_HorizonAlignedStateResidual |     17 |      67.4804  |  116.101   |     63.1766  | 7142.17   |           11.0366   |       0.44842  |               0.0952381 |   2.2951  |                0.0842002 |             18717 |            1.57953 |          0.0005063  |                 2.41095e-05 |                          5.3381e-06  |
| GANSU_DA | TimeMixer | Host                            |     17 |      75.852   |  157.584   |     67.2486  | 8859.6    |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| GANSU_DA | TimeMixer | S1_DailyPatch_GRU32             |     17 |      70.7496  |  129.267   |     64.5898  | 7730.55   |            6.72675  |       0.331464 |               0.190476  |   2.02695 |                0.0454487 |             18712 |           11.8611  |          0.0003587  |                 1.7081e-05  |                          0           |
| LAGO_DE  | PatchTST  | HSA_HorizonAlignedStateResidual |     17 |       4.28716 |    5.90103 |      4.02812 |   33.0051 |            9.8678   |       0.455953 |               0.137615  |   2.84475 |                0.116181  |             14877 |            1.71401 |          0.0007577  |                 6.95138e-06 |                          1.22294e-06 |
| LAGO_DE  | PatchTST  | Host                            |     17 |       4.75652 |    7.4155  |      4.34583 |   40.9278 |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_DE  | PatchTST  | S1_DailyPatch_GRU32             |     17 |       4.24566 |    6.58172 |      3.88961 |   34.226  |           10.7403   |       0.450648 |               0.100917  |   2.40049 |                0.0884541 |             14872 |           18.3097  |          0.0006228  |                 5.71376e-06 |                          0           |
| LAGO_DE  | TimeMixer | HSA_HorizonAlignedStateResidual |     17 |       4.29726 |    6.08531 |      4.02252 |   34.0725 |            8.69682  |       0.438446 |               0.183486  |   2.8009  |                0.115232  |             14877 |            1.75268 |          0.0007887  |                 7.23578e-06 |                          1.20917e-06 |
| LAGO_DE  | TimeMixer | Host                            |     17 |       4.70658 |    7.14155 |      4.33048 |   40.7722 |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_DE  | TimeMixer | S1_DailyPatch_GRU32             |     17 |       4.30824 |    6.55801 |      3.96075 |   35.0767 |            8.46346  |       0.438839 |               0.146789  |   2.22927 |                0.0794982 |             14872 |           18.5159  |          0.0006146  |                 5.63853e-06 |                          0           |
| LAGO_PJM | PatchTST  | HSA_HorizonAlignedStateResidual |     17 |       3.03923 |   10.3021  |      2.81011 |   18.6439 |           -0.649539 |       0.131433 |               0.422018  |   1.17468 |                0.0208173 |             14109 |            1.80732 |          0.00077735 |                 7.13165e-06 |                          9.19725e-07 |
| LAGO_PJM | PatchTST  | Host                            |     17 |       3.01961 |   10.5586  |      2.78179 |   19.0014 |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_PJM | PatchTST  | S1_DailyPatch_GRU32             |     17 |       3.04576 |   10.6616  |      2.80551 |   18.8677 |           -0.865822 |       0.123811 |               0.431193  |   1.13853 |                0.0202126 |             14104 |           24.0547  |          0.000682   |                 6.25688e-06 |                          0           |
| LAGO_PJM | TimeMixer | HSA_HorizonAlignedStateResidual |     17 |       3.06789 |    8.16087 |      2.90722 |   18.1368 |            4.75821  |       0.420313 |               0.165138  |   2.37902 |                0.0905092 |             14109 |            1.84866 |          0.00081165 |                 7.44633e-06 |                          1.16835e-06 |
| LAGO_PJM | TimeMixer | Host                            |     17 |       3.22116 |    9.62358 |      3.01919 |   20.968  |            0        |     nan        |             nan         | nan       |              nan         |               nan |          nan       |        nan          |               nan           |                        nan           |
| LAGO_PJM | TimeMixer | S1_DailyPatch_GRU32             |     17 |       3.06124 |    8.17437 |      2.89994 |   18.0068 |            4.96469  |       0.422158 |               0.155963  |   2.43097 |                0.097623  |             14104 |           25.9423  |          0.0005937  |                 5.44679e-06 |                          0           |

## Final beta (median across seeds)

| market   | host      |   beta_DEMAND_FC |   beta_RENEWABLE_FC |   beta_SUPPLY_MARGIN_FC |   beta_INTERCHANGE_FC |   beta_MUST_RUN_FC |
|:---------|:----------|-----------------:|--------------------:|------------------------:|----------------------:|-------------------:|
| GANSU_DA | PatchTST  |       -0.0637679 |           -0.297523 |                       0 |              0.118134 |          0.065431  |
| GANSU_DA | TimeMixer |       -0.0438228 |           -0.2384   |                       0 |              0.126669 |          0.0311087 |
| LAGO_DE  | PatchTST  |        0.0995357 |           -0.347019 |                       0 |              0        |          0         |
| LAGO_DE  | TimeMixer |        0.0804792 |           -0.399103 |                       0 |              0        |          0         |
| LAGO_PJM | PatchTST  |        0.085136  |            0        |                       0 |              0        |          0         |
| LAGO_PJM | TimeMixer |        0.0574217 |            0        |                       0 |              0        |          0         |

## Frozen state-tertile behaviour

| market   | host      |   HSA_minus_S1_shape_cosine |   HSA_minus_S1_wrong_hemisphere_rate |   HSA_minus_S1_relative_gain_pct |
|:---------|:----------|----------------------------:|-------------------------------------:|---------------------------------:|
| GANSU_DA | PatchTST  |                  0.0347042  |                           -0.125     |                        -0.885085 |
| GANSU_DA | TimeMixer |                  0.118942   |                           -0.181818  |                         4.2062   |
| LAGO_DE  | PatchTST  |                  0.0177602  |                            0.0481203 |                         0.948497 |
| LAGO_DE  | TimeMixer |                 -0.0448935  |                            0.166667  |                        -1.52164  |
| LAGO_PJM | PatchTST  |                  0.0173547  |                           -0.03125   |                         0.373794 |
| LAGO_PJM | TimeMixer |                 -0.00912961 |                            0.0384615 |                        -0.838405 |

## Required statements (protocol §10)

1. **Does horizon-local semantic state improve Shape across markets?** Only partially: cell-median Shape cosine rises in 4/6 cells and is non-worse in 4/6 (median change +0.0065); wrong-hemisphere is non-worse in 3/6. The improvement is consistent in GANSU_DA but not universal.
2. **Do both GANSU_DA Hosts improve without market-specific tuning?** Yes: Shape cosine rises by +0.0666 (PatchTST), +0.1170 (TimeMixer), using one shared five-role vocabulary and a per-market beta fit on that market's own OOF rows only; no market-specific role selection, IDs, or rules are introduced.
3. **Is the same five-parameter mechanism non-worse across all six cells?** No: relative Overall-MAE gain vs S1 is nonnegative in 4/6 cell medians and vs Host in 5/6 (worst -0.8725 pp vs S1). It regresses on at least one cell.
4. **Is public-China expansion scientifically authorized?** No: H1=FAIL, H2=FAIL, H3=FAIL with H0=PASS, H4=PASS. At least one scientific gate fails, so S1 plus the pooled fixed alpha is retained and no SHAANXI/NINGXIA/QINGHAI or full-panel/protected/final work is authorized by this stage.

Median relative Overall-MAE gain vs Host: `6.727511%`; median gain change vs S1: `0.224821 pp`.
Maximum Normal-MAE harm: `1.018271%`.
Median beta training time: `1.7333 s/cell`; median HSA inference overhead: `1.2161 us/day`; median total HSA inference: `7.0567 us/day`.

Method consequence: keep S1 + pooled fixed alpha; HSA is not a supportable upgrade.
