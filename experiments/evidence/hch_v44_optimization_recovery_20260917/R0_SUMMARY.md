# R0 — why the frozen epoch budget stopped near the Host

Rebuilt read-only from the completed stage's TRAIN/VAL freeze records; **0 new fits, 0 TEST reads**. Authority: `PROTOCOL.md` §6.

## Budget asymmetry (the registered reason for this stage)

- optimizer steps per epoch: **3 .. 29** (max/min = 9.7x)
- total optimizer steps in a full 50-epoch run: **27 .. 1450** (median 68)
- epoch 0 selected in **34/60** runs (56.7%)
- steps per epoch by market: GANSU_DA=[7], SHANDONG_DA=[29], SHAANXI_DA=[7], NINGXIA_DA=[3], QINGHAI_DA=[3]

## Did training ever improve VAL?

- VAL improvement from step 0 to the selected epoch: median **0.0000%**, best **2.3326%**, positive in **26/60** runs
- correction magnitude is not stored in the prior freeze; it is measured at step 0 of every R1 run (same evaluation path), which is the honest pre-training reference.

## Recovery panel (8 cells, optimization-development only)

- 13/24 panel runs selected epoch 0
- median old optimizer steps inside the panel: 205.5

| cell | host | n_train | steps/epoch | old total steps | selected epoch | epoch0? |
|---|---|---|---|---|---|---|
| GANSU_DA | LSTM | 224 | 7 | 350 | 49 | no |
| GANSU_DA | LSTM | 224 | 7 | 350 | 49 | no |
| GANSU_DA | LSTM | 224 | 7 | 350 | 49 | no |
| GANSU_DA | PatchTST | 224 | 7 | 63 | 0 | yes |
| GANSU_DA | PatchTST | 224 | 7 | 63 | 0 | yes |
| GANSU_DA | PatchTST | 224 | 7 | 63 | 0 | yes |
| NINGXIA_DA | iTransformer | 72 | 3 | 150 | 49 | no |
| NINGXIA_DA | iTransformer | 72 | 3 | 150 | 49 | no |
| NINGXIA_DA | iTransformer | 72 | 3 | 150 | 49 | no |
| QINGHAI_DA | TimeMixer | 67 | 3 | 27 | 0 | yes |
| QINGHAI_DA | TimeMixer | 67 | 3 | 27 | 0 | yes |
| QINGHAI_DA | TimeMixer | 67 | 3 | 27 | 0 | yes |
| SHAANXI_DA | PatchTST | 219 | 7 | 63 | 0 | yes |
| SHAANXI_DA | PatchTST | 219 | 7 | 63 | 0 | yes |
| SHAANXI_DA | PatchTST | 219 | 7 | 63 | 0 | yes |
| SHAANXI_DA | TimeMixer | 219 | 7 | 350 | 41 | no |
| SHAANXI_DA | TimeMixer | 219 | 7 | 315 | 36 | no |
| SHAANXI_DA | TimeMixer | 219 | 7 | 336 | 39 | no |
| SHANDONG_DA | PatchTST | 916 | 29 | 1015 | 26 | no |
| SHANDONG_DA | PatchTST | 916 | 29 | 261 | 0 | yes |
| SHANDONG_DA | PatchTST | 916 | 29 | 1450 | 47 | no |
| SHANDONG_DA | iTransformer | 916 | 29 | 261 | 0 | yes |
| SHANDONG_DA | iTransformer | 916 | 29 | 261 | 0 | yes |
| SHANDONG_DA | iTransformer | 916 | 29 | 261 | 0 | yes |

Full per-run table: `R0_DIAGNOSTICS.csv`.
