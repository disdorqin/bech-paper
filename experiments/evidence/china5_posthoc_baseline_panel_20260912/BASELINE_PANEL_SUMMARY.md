# China-5 post-hoc baseline panel — summary

**Terminal state:** `CHINA5_BASELINE_PANEL_PARTIALLY_FROZEN_WITH_DISCLOSED_BLOCKERS`

**Semantics:** day-ahead price -> day-ahead price, next 24h hourly  
**Markets run:** SHANDONG, SHAANXI, NINGXIA, QINGHAI  
**Hosts:** PatchTST, TimeMixer (frozen GANSU_DA recipe, reused verbatim)  
**Strict offline table:** Host / MatchedDirectResidual / δ-Adapter (Ada-Y) / PIR  
**PROTECTED_FINAL:** SEALED_STRUCTURALLY_NOT_READ  
**Host gate all passed:** True

## 1. Strict offline main table

Only the four methods admitted to the strict offline comparison appear here.

| physical_market_group | backbone | method_display | n_days | Overall_MAE | relative_gain_vs_Host_pct | Tail_MAE | Extreme_day_MAE | Normal_harm_vs_Host |
|---|---|---|---|---|---|---|---|---|
| SHANDONG | PatchTST | Host (frozen) | 82 | 104.5452 | 0.0000 | 181.2230 | 134.5908 | 0.0000 |
| SHANDONG | PatchTST | MatchedDirectResidual | 82 | 134.0554 | -28.2272 | 216.8870 | 176.2475 | 28.4979 |
| SHANDONG | PatchTST | delta-Adapter (Ada-Y) | 82 | 102.9680 | 1.5087 | 177.8789 | 132.0340 | -1.2866 |
| SHANDONG | PatchTST | PIR | 82 | 110.7582 | -5.9429 | 183.3617 | 167.9919 | 6.8832 |
| SHANDONG | TimeMixer | Host (frozen) | 82 | 103.7590 | 0.0000 | 184.6171 | 148.4166 | 0.0000 |
| SHANDONG | TimeMixer | MatchedDirectResidual | 82 | 143.0590 | -37.8762 | 226.6725 | 163.1478 | 38.8467 |
| SHANDONG | TimeMixer | delta-Adapter (Ada-Y) | 82 | 103.9035 | -0.1392 | 184.2797 | 147.7562 | 0.2237 |
| SHANDONG | TimeMixer | PIR | 82 | 110.9025 | -6.8847 | 193.9741 | 183.2444 | 6.7794 |
| SHAANXI | PatchTST | Host (frozen) | 20 | 101.6389 | 0.0000 | 143.0109 | 139.3178 | 0.0000 |
| SHAANXI | PatchTST | MatchedDirectResidual | 20 | 132.3328 | -30.1991 | 160.2971 | 123.7517 | 33.2162 |
| SHAANXI | PatchTST | delta-Adapter (Ada-Y) | 20 | 99.9975 | 1.6149 | 141.1125 | 138.1805 | -1.5930 |
| SHAANXI | PatchTST | PIR | 20 | 107.8236 | -6.0850 | 156.7005 | 142.7127 | 4.7729 |
| SHAANXI | TimeMixer | Host (frozen) | 20 | 99.7407 | 0.0000 | 140.6307 | 123.6764 | 0.0000 |
| SHAANXI | TimeMixer | MatchedDirectResidual | 20 | 127.9108 | -28.2434 | 156.5578 | 129.8768 | 30.4733 |
| SHAANXI | TimeMixer | delta-Adapter (Ada-Y) | 20 | 99.6668 | 0.0741 | 140.6600 | 123.4876 | -0.0933 |
| SHAANXI | TimeMixer | PIR | 20 | 108.9754 | -9.2587 | 170.1659 | 138.8868 | 5.4158 |
| NINGXIA | PatchTST | Host (frozen) | 7 | 82.4741 | 0.0000 | 73.1810 |  | 0.0000 |
| NINGXIA | PatchTST | MatchedDirectResidual | 7 | 100.6755 | -22.0693 | 67.7951 |  | 32.3538 |
| NINGXIA | PatchTST | delta-Adapter (Ada-Y) | 7 | 81.6325 | 1.0204 | 72.4204 |  | -0.8901 |
| NINGXIA | PatchTST | PIR | 7 | 98.8108 | -19.8082 | 84.1238 |  | 19.5730 |
| NINGXIA | TimeMixer | Host (frozen) | 7 | 92.5665 | 0.0000 | 99.5850 |  | 0.0000 |
| NINGXIA | TimeMixer | MatchedDirectResidual | 7 | 107.2402 | -15.8520 | 65.8470 |  | 43.7207 |
| NINGXIA | TimeMixer | delta-Adapter (Ada-Y) | 7 | 91.8831 | 0.7383 | 99.1000 |  | -0.8026 |
| NINGXIA | TimeMixer | PIR | 7 | 106.0586 | -14.5755 | 98.7713 |  | 22.0755 |
| QINGHAI | PatchTST | Host (frozen) | 6 | 110.8599 | 0.0000 | 111.5894 | 117.0636 | 0.0000 |
| QINGHAI | PatchTST | MatchedDirectResidual | 6 | 125.5292 | -13.2323 | 123.3766 | 134.6667 | 17.5514 |
| QINGHAI | PatchTST | delta-Adapter (Ada-Y) | 6 | 110.7663 | 0.0844 | 111.5602 | 116.9227 | -0.1578 |
| QINGHAI | PatchTST | PIR | 6 | 146.1911 | -31.8702 | 141.4702 | 141.3611 | 40.7817 |
| QINGHAI | TimeMixer | Host (frozen) | 6 | 114.1517 | 0.0000 | 104.8469 | 117.1963 | 0.0000 |
| QINGHAI | TimeMixer | MatchedDirectResidual | 6 | 145.2221 | -27.2185 | 135.2819 | 154.4808 | 31.7057 |
| QINGHAI | TimeMixer | delta-Adapter (Ada-Y) | 6 | 114.3556 | -0.1786 | 105.1169 | 117.4487 | 0.1378 |
| QINGHAI | TimeMixer | PIR | 6 | 171.9425 | -50.6263 | 162.1588 | 160.0465 | 58.2697 |


## 2. Supplementary (setting-disclosed) table

COSA is online/TTA: it is scored on the same days it adapts on, so its numbers are **not comparable** to the strict offline table. It is never merged into the main table.

| physical_market_group | backbone | method_display | n_days | Overall_MAE | relative_gain_vs_Host_pct | Tail_MAE | Extreme_day_MAE | Normal_harm_vs_Host |
|---|---|---|---|---|---|---|---|---|
| SHANDONG | PatchTST | COSA_online | 82 | 103.8608 | 0.6547 | 179.2910 | 134.4710 | -0.4792 |
| SHANDONG | TimeMixer | COSA_online | 82 | 102.5355 | 1.1792 | 183.9370 | 148.3727 | -1.3130 |
| SHAANXI | PatchTST | COSA_online | 20 | 101.4078 | 0.2273 | 142.2579 | 138.8331 | -0.1328 |
| SHAANXI | TimeMixer | COSA_online | 20 | 99.4724 | 0.2690 | 140.0053 | 123.3557 | -0.2012 |
| NINGXIA | PatchTST | COSA_online | 7 | 82.4681 | 0.0073 | 73.1384 |  | 0.0160 |
| NINGXIA | TimeMixer | COSA_online | 7 | 92.5498 | 0.0181 | 99.5314 |  | 0.0054 |
| QINGHAI | PatchTST | COSA_online | 6 | 110.8494 | 0.0095 | 111.5587 | 117.0510 | 0.0097 |
| QINGHAI | TimeMixer | COSA_online | 6 | 114.0960 | 0.0488 | 104.7538 | 117.1295 | -0.0183 |
| ALL | ALL | UEC_STD | 0 |  |  |  |  |  |
| ALL | ALL | OMPB | 0 |  |  |  |  |  |


## 3. Blocked / not-run tracks

| track | status | proxy_used | retrained |
|---|---|---|---|
| MARKET::SHANXI | ABSENT_FROM_REGISTRY | False | False |
| MARKET::LIAONING | DATA_BLOCKED | False | False |
| BASELINE::UEC_STD | HOST_FIDELITY_BLOCKED / NOT_PAPER_FAITHFULLY_ADMITTED | False | False |
| BASELINE::OMPB | STANDALONE_DATA_BLOCKED / NOT_REPRODUCED | False | False |
| IMPORTED::GANSU_DA::PatchTST::Host | IMPORTED_OK | False | False |
| IMPORTED::GANSU_DA::TimeMixer::Host | IMPORTED_OK | False | False |
| IMPORTED::GANSU_DA::PatchTST::PIR_paper_protocol | IMPORTED_OK | False | False |
| IMPORTED::GANSU_DA::TimeMixer::PIR_paper_protocol | IMPORTED_OK | False | False |
| IMPORTED::GANSU_DA::PatchTST::raw_arrays_absent | RAW_ARRAY_ABSENT_METRICS_ONLY | False | False |
| IMPORTED::GANSU_DA::TimeMixer::raw_arrays_absent | RAW_ARRAY_ABSENT_METRICS_ONLY | False | False |


## 4. Host gate

| cell | passed | DEV_EVAL_MAE | DEV_EVAL_MAE_train_mean_constant | HOST_TRAIN_MAE | HOST_VAL_MAE | n_dev_eval_days |
|---|---|---|---|---|---|---|
| SHANDONG__PatchTST | True | 104.5452 | 150.1011 | 95.4693 | 99.5919 | 82 |
| SHANDONG__TimeMixer | True | 103.7590 | 150.1011 | 97.1213 | 97.3338 | 82 |
| SHAANXI__PatchTST | True | 101.6388 | 136.7429 | 78.2410 | 87.8099 | 20 |
| SHAANXI__TimeMixer | True | 99.7407 | 136.7429 | 78.3989 | 87.5958 | 20 |
| NINGXIA__PatchTST | True | 82.4741 | 131.2150 | 75.6954 | 84.8747 | 7 |
| NINGXIA__TimeMixer | True | 92.5665 | 131.2150 | 90.8861 | 95.1740 | 7 |
| QINGHAI__PatchTST | True | 110.8599 | 200.9550 | 74.3883 | 107.4922 | 6 |
| QINGHAI__TimeMixer | True | 114.1517 | 200.9550 | 87.1518 | 99.8546 | 6 |

