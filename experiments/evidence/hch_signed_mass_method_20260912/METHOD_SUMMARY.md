# Signed-mass method — development experiment summary

Generated: 2026-09-12T10:44:26Z

This is a **development** experiment on the China-5 panel.  Every number below was produced on `DEV_EVAL` of the 20 registered market x Host cells, under the frozen method, with the frozen Host predictions left untouched.  `PROTECTED_FINAL` was not opened and is not authorized.

## Inference setting

`FROZEN_PARAMETER_CAUSAL_PREQUENTIAL_HISTORY_POSTPROCESSING` — model parameters are frozen on POST_TRAIN before DEV_EVAL is opened; no gradient/update/TTA occurs during DEV_EVAL; each day may use only historically revealed residual windows from strictly earlier eligible origins.

This is **not** online test-time adaptation.  A frozen-parameter post-processor and an online TTA method are not the same inference setting, and their numbers are never ranked against each other.

## The frozen method

Configuration: `FROZEN_SMALLEST`  ·  components retained: none  ·  deleted: rare_mass_sampling, shape_semantic_context, untied_amplitude_heads, use_tcn

| switch | state |
|---|---|
| `shape_semantic_context` | OFF |
| `use_tcn` | OFF |
| `untied_amplitude_heads` | OFF |
| `rare_mass_sampling` | OFF |

Seeds (cell median over all three, never a best seed): 7, 17, 37.  KNN: off.  High-mass subset: q90 of realised mass, fitted per cell on `POST_TRAIN` only and frozen before any `DEV_EVAL` metric was computed.

## Component adjudication

| component | decision | primary evidence (median delta, ON − OFF) | cells improved | reason |
|---|---|---|---|---|
| `shape_semantic_context` | **DELETED** | `{"shape_w1_negative": "0.6527713493512111", "shape_w1_positive": "0.8755192911804337"}` | 1 of 20 | primary mechanism does not improve in a majority of cells; median relevant tail/upper/lower metric does not improve; median Overall-MAE is worse; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact |
| `use_tcn` | **DELETED** | `{"tail_mae": "0.6372247874003278"}` | 2 of 20 | primary mechanism does not improve in a majority of cells; median relevant tail/upper/lower metric does not improve; median Overall-MAE is worse; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact |
| `untied_amplitude_heads` | **DELETED** | `{"mass_negative_l1": "-4.539010733878968", "mass_positive_l1": "-0.0018481727290691197"}` | 12 of 20 | median relevant tail/upper/lower metric does not improve; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact |
| `rare_mass_sampling` | **DELETED** | `{"high_mass_negative_l1": "0.03593599732994335", "high_mass_positive_l1": "-0.08208521188726081"}` | 9 of 20 | primary mechanism does not improve in a majority of cells; a cell gains more than 1.0% normal-region relative harm from this component alone; the effect does not survive leave-one-cell-out or is a single-seed artifact |

A component is deleted unless all five pre-registered conditions held on its own registered primary metric.  No rescue variant was run, no per-market exemption was applied, and no metric was substituted after the fact.

## Per-cell results

20 cells.  All values are cell medians over the three registered seeds; every cell was evaluated on its complete `DEV_EVAL` partition.

| cell | Overall MAE | Host | gain vs Host | Tail | Host tail | Upper | Lower | Normal | Host normal | normal harm | neg-price | best strict comparator | margin | control (not-ranked) | gain vs control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GANSU_DA::LSTM | 120.3142 | 121.1592 | +0.697% | 144.0768 | 138.2518 | n/a | 144.0768 | 120.1719 | 121.0568 | -0.731% | n/a | delta-Adapter (121.2417) | +0.765% | MatchedDirectResidual (96.4235) | -24.777% |
| GANSU_DA::PatchTST | 69.0771 | 72.0534 | +4.131% | 9.7274 | 9.3237 | n/a | 9.7274 | 69.4495 | 72.4291 | -4.114% | n/a | delta-Adapter (71.5799) | +3.496% | MatchedDirectResidual (92.7100) | +25.491% |
| GANSU_DA::TimeMixer | 69.6862 | 75.8520 | +8.129% | 44.1802 | 65.2831 | n/a | 44.1802 | 69.8389 | 75.9152 | -8.004% | n/a | delta-Adapter (75.8697) | +8.150% | MatchedDirectResidual (101.8573) | +31.584% |
| GANSU_DA::iTransformer | 72.0913 | 77.1381 | +6.543% | 39.6539 | 55.1348 | n/a | 39.6539 | 72.3003 | 77.2699 | -6.431% | n/a | delta-Adapter (76.7081) | +6.019% | MatchedDirectResidual (101.8047) | +29.187% |
| NINGXIA_DA::LSTM | 70.0166 | 75.7347 | +7.550% | 63.8577 | 70.2318 | n/a | 63.8577 | 73.7722 | 79.0364 | -6.660% | n/a | delta-Adapter (75.7379) | +7.554% | MatchedDirectResidual (133.2966) | +47.473% |
| NINGXIA_DA::PatchTST | 79.5910 | 82.4741 | +3.496% | 72.6642 | 73.1810 | n/a | 72.6642 | 83.5415 | 88.0499 | -5.120% | n/a | delta-Adapter (81.6325) | +2.501% | MatchedDirectResidual (100.6755) | +20.943% |
| NINGXIA_DA::TimeMixer | 82.4224 | 92.5665 | +10.959% | 80.2719 | 99.5850 | n/a | 80.2719 | 84.2848 | 88.3554 | -4.607% | n/a | delta-Adapter (91.8831) | +10.296% | MatchedDirectResidual (107.2402) | +23.142% |
| NINGXIA_DA::iTransformer | 73.4893 | 86.5544 | +15.095% | 62.6967 | 70.4630 | n/a | 62.6967 | 79.9649 | 96.2092 | -16.884% | n/a | delta-Adapter (86.6070) | +15.146% | MatchedDirectResidual (102.5167) | +28.315% |
| QINGHAI_DA::LSTM | 125.8409 | 117.1789 | -7.392% | 114.2384 | 106.6048 | 209.4190 | 53.6690 | 137.4433 | 127.7531 | +7.585% | n/a | delta-Adapter (117.1228) | -7.443% | MatchedDirectResidual (169.5435) | +25.777% |
| QINGHAI_DA::PatchTST | 119.9889 | 110.8599 | -8.235% | 130.0998 | 111.5894 | 160.8116 | 110.5558 | 116.3497 | 110.1303 | +5.647% | n/a | delta-Adapter (110.7663) | -8.326% | MatchedDirectResidual (125.5292) | +4.414% |
| QINGHAI_DA::TimeMixer | 125.3685 | 114.1517 | -9.826% | 128.6886 | 104.8469 | 221.4894 | 69.6335 | 125.6318 | 123.4565 | +1.762% | n/a | delta-Adapter (114.3556) | -9.630% | MatchedDirectResidual (145.2221) | +13.671% |
| QINGHAI_DA::iTransformer | 148.2030 | 116.9818 | -26.689% | 160.7101 | 92.0517 | 224.7654 | 119.9477 | 135.6959 | 141.9118 | -4.380% | n/a | delta-Adapter (116.7359) | -26.956% | MatchedDirectResidual (150.8630) | +1.763% |
| SHAANXI_DA::LSTM | 101.0030 | 110.7256 | +8.781% | 174.2937 | 186.8588 | 340.0525 | 136.8643 | 87.2157 | 96.4035 | -9.531% | n/a | delta-Adapter (111.3188) | +9.267% | MatchedDirectResidual (117.9976) | +14.402% |
| SHAANXI_DA::PatchTST | 90.0699 | 101.6389 | +11.382% | 130.8218 | 143.0109 | 327.3575 | 83.8993 | 83.5760 | 93.8560 | -10.953% | n/a | delta-Adapter (99.9975) | +9.928% | MatchedDirectResidual (132.3328) | +31.937% |
| SHAANXI_DA::TimeMixer | 88.2406 | 99.7407 | +11.530% | 132.2536 | 140.6307 | 350.2667 | 82.4824 | 80.8060 | 92.0485 | -12.214% | n/a | delta-Adapter (99.6668) | +11.464% | MatchedDirectResidual (127.9108) | +31.014% |
| SHAANXI_DA::iTransformer | 99.7983 | 109.3556 | +8.740% | 153.7329 | 161.6882 | 380.5257 | 101.2772 | 89.8432 | 99.5108 | -9.715% | n/a | delta-Adapter (108.2644) | +7.820% | MatchedDirectResidual (130.4160) | +23.477% |
| SHANDONG_DA::LSTM | 96.0932 | 112.0280 | +14.224% | 163.1987 | 210.5180 | 240.3134 | 158.8005 | 85.8547 | 95.8267 | -10.406% | 175.2337 | delta-Adapter (112.0093) | +14.210% | MatchedDirectResidual (132.2931) | +27.363% |
| SHANDONG_DA::PatchTST | 87.8916 | 104.5452 | +15.930% | 125.0652 | 181.2230 | 204.5635 | 120.5311 | 81.4041 | 91.9319 | -11.452% | 132.1237 | delta-Adapter (102.9680) | +14.642% | MatchedDirectResidual (134.0554) | +34.436% |
| SHANDONG_DA::TimeMixer | 90.2499 | 103.7590 | +13.020% | 130.3492 | 184.6171 | 175.4005 | 127.1906 | 83.6537 | 90.4581 | -7.522% | 142.1369 | delta-Adapter (103.9035) | +13.141% | MatchedDirectResidual (143.0590) | +36.914% |
| SHANDONG_DA::iTransformer | 94.9839 | 111.0789 | +14.490% | 143.2740 | 204.0725 | 237.3391 | 137.4846 | 87.1020 | 95.7818 | -9.062% | 150.0266 | delta-Adapter (110.8066) | +14.280% | MatchedDirectResidual (138.9806) | +31.657% |

`gain vs Host` is positive when our method is better; `normal harm` is positive when it is worse on the normal region.  `margin` is our advantage over the best numeric, setting-matched **strict offline** comparator for that cell; blocked comparator rows (a disclosed `Q-FIDELITY`/`Q-DATA`/`Q-INCOMPATIBLE` outcome) never enter that rank.

The `control` column is `OFFLINE_STATIC_CONTROL`, a different measurement setting, and it is reported *beside* the comparison rather than inside it.  Our method post-processes a frozen Host and never updates a parameter during `DEV_EVAL`, which is `OFFLINE_STATIC_POSTHOC`; the control is a deliberately different setting, and the comparator panel's own protocol forbids a cross-setting `mixed best baseline` claim.  The rule fixes the direction of the comparison before any of our numbers existed -- the control is far worse than its own Host in most cells, so admitting it to the rank would *loosen* the gate, not tighten it.

## Where it fails

- cells worse than their own Host on Overall MAE (4): QINGHAI_DA::LSTM, QINGHAI_DA::PatchTST, QINGHAI_DA::TimeMixer, QINGHAI_DA::iTransformer
- cells not strictly better than the best strict offline comparator (4): QINGHAI_DA::LSTM, QINGHAI_DA::PatchTST, QINGHAI_DA::TimeMixer, QINGHAI_DA::iTransformer

- worse than their own Host, by market: QINGHAI_DA 4
- worse than their own Host, by Host: LSTM 1, PatchTST 1, TimeMixer 1, iTransformer 1
- not strictly better than the best strict comparator, by market: QINGHAI_DA 4
- not strictly better than the best strict comparator, by Host: LSTM 1, PatchTST 1, TimeMixer 1, iTransformer 1

## Development gate

| gate | value | required | result |
|---|---|---|---|
| `gate_1_host_nonworse_19_of_20` | 16 | 19 | FAIL |
| `gate_2_median_overall_gain_vs_host_at_least_3pct` | 8.434119377728752 | 3.0 | PASS |
| `gate_3_strict_win_14_of_20` | 16 | 14 | PASS |
| `gate_4_median_tail_gain_vs_host_at_least_5pct` | 7.623787885315595 | 5.0 | PASS |
| `gate_5_max_normal_harm_at_most_1pct` | 7.585116263660587 | 1.0 | FAIL |
| `gate_6_one_recipe` | 20 cells, 1 switch vector(s) |  | PASS |

All gates pass: **no**.

## Verdict

`HCH_SIGNED_MASS_DEVELOPMENT_NOT_SUPPORTED_STOP_FOR_ADJUDICATION`

The independent audit of this run, together with the three corrections made to the audit instrument after its first pass, is reported in `FINAL_AUDIT.md` and `06_audits/RESULT_AUDIT.json`.  Those corrections changed what the audit could see and one tolerance inside it; they changed no metric, no component decision and no gate.

Development experiment only.  No claim here is stronger than the gate table above: a component that failed is deleted rather than rescued, and a gate that failed is reported as failed rather than relabelled.

