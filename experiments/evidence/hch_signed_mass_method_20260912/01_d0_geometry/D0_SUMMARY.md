# D0 — signed-mass geometry over `POST_TRAIN`

- cells: 20
- roles used: `POST_TRAIN` only; `DEV_EVAL` never materialised
- `PROTECTED_FINAL` read count: 0
- KNN: disabled (D0 does not enable it)
- training: none — this report is pure arithmetic on frozen residuals

## What D0 is, and is not

D0 describes the object the method factorises: how the Host's residual mass
is distributed across days, whether its signed Shapes carry geometry beyond
uniform, and whether the factorization reconstructs the residual exactly.

It is **descriptive**.  Nothing here gates the run, tunes a parameter or
tests novelty; the only failures that stop execution are structural
(reconstruction, roles, sealed reads).  The adjacent-vs-random W1 column is a
similarity diagnostic with no registered threshold.

## Per cell

| market | host | days | A+ median | A- median | defined + / − | entropy + / − | peak + / − | W1 adj/rand | recon err |
|---|---|---|---|---|---|---|---|---|---|
| GANSU_DA | PatchTST | 62 | 461.9 | 924.1 | 60 / 62 | 1.869 / 2.263 | 21 / 8 | 6.034 / 6.658 | 2.84e-14 |
| GANSU_DA | TimeMixer | 62 | 575.9 | 835.3 | 57 / 61 | 2.038 / 2.221 | 23 / 9 | 6.013 / 5.99 | 2.84e-14 |
| GANSU_DA | iTransformer | 62 | 457.7 | 1028 | 58 / 61 | 1.960 / 2.163 | 23 / 9 | 6.144 / 6.367 | 2.84e-14 |
| GANSU_DA | LSTM | 62 | 9.44 | 3270 | 33 / 62 | 1.290 / 2.894 | 15 / 9 | 5.385 / 3.778 | 5.68e-14 |
| SHANDONG_DA | PatchTST | 248 | 846.6 | 770.1 | 242 / 242 | 2.196 / 2.154 | 0 / 10 | 5.013 / 5.557 | 1.14e-13 |
| SHANDONG_DA | TimeMixer | 248 | 848.5 | 780.1 | 238 / 242 | 2.262 / 2.131 | 10 / 23 | 4.978 / 5.83 | 5.68e-14 |
| SHANDONG_DA | iTransformer | 248 | 787.7 | 821.7 | 236 / 242 | 2.193 / 2.210 | 9 / 5 | 5.148 / 6.078 | 1.14e-13 |
| SHANDONG_DA | LSTM | 248 | 1052 | 753.1 | 242 / 241 | 2.351 / 2.052 | 12 / 8 | 4.516 / 5.437 | 1.14e-13 |
| SHAANXI_DA | PatchTST | 61 | 887.1 | 1083 | 61 / 60 | 2.041 / 2.182 | 0 / 22 | 5.134 / 6.311 | 5.68e-14 |
| SHAANXI_DA | TimeMixer | 61 | 716.6 | 1243 | 60 / 60 | 1.786 / 2.370 | 0 / 23 | 5.022 / 6.897 | 2.84e-14 |
| SHAANXI_DA | iTransformer | 61 | 760 | 1234 | 60 / 59 | 1.890 / 2.337 | 18 / 23 | 4.71 / 5.68 | 1.14e-13 |
| SHAANXI_DA | LSTM | 61 | 584.7 | 1662 | 54 / 61 | 1.716 / 2.438 | 0 / 23 | 5.457 / 7.169 | 5.68e-14 |
| NINGXIA_DA | PatchTST | 22 | 1092 | 505.9 | 22 / 22 | 2.355 / 1.806 | 0 / 16 | 4.559 / 3.966 | 2.84e-14 |
| NINGXIA_DA | TimeMixer | 22 | 1501 | 905.3 | 20 / 22 | 2.388 / 2.162 | 3 / 16 | 3.866 / 4.743 | 2.84e-14 |
| NINGXIA_DA | iTransformer | 22 | 1333 | 563.1 | 22 / 22 | 2.320 / 1.878 | 8 / 16 | 4.735 / 4.073 | 2.84e-14 |
| NINGXIA_DA | LSTM | 22 | 1213 | 356.1 | 19 / 21 | 2.416 / 1.734 | 3 / 16 | 4.695 / 4.61 | 2.84e-14 |
| QINGHAI_DA | PatchTST | 20 | 1461 | 985.4 | 20 / 20 | 2.267 / 2.081 | 19 / 23 | 2.656 / 3.32 | 5.68e-14 |
| QINGHAI_DA | TimeMixer | 20 | 961.2 | 1343 | 20 / 20 | 2.277 / 2.191 | 20 / 23 | 2.443 / 2.922 | 5.68e-14 |
| QINGHAI_DA | iTransformer | 20 | 1251 | 1491 | 20 / 20 | 2.351 / 2.222 | 10 / 23 | 1.865 / 2.144 | 5.68e-14 |
| QINGHAI_DA | LSTM | 20 | 544.9 | 1932 | 20 / 20 | 1.774 / 2.422 | 18 / 23 | 2.796 / 3.197 | 5.68e-14 |

## Structural checks

- max reconstruction error over all cells: `1.137e-13`
- cells with a non-empty positive-Shape day set: 20/20
- cells with a non-empty negative-Shape day set: 20/20
- cells with at least one exactly-zero-mass day in either sign: 14/20
