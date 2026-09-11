# Frozen HCH-S1 vs admitted offline baselines — transfer verdict

**Token: `HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED`**

international parity (B2) failed; GANSU competitiveness (B1) passed.

Protocol: `docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md`  
Frozen method: `S1 Daily-Patch GRU32 Shape + pooled exact OOF MAE alpha` (replayed, not modified)  
Panel: `GANSU_DA / LAGO_DE / LAGO_PJM x PatchTST / TimeMixer`

## Per-cell result

| market | host | HCH_MAE | Host_MAE | best_baseline | best_baseline_MAE | gap_to_best_baseline_pct | hch_gain_vs_host_pct | HCH_strict_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GANSU_DA | PatchTST | 70.4342 | 72.0534 | delta_Adapter_AdaY | 71.5799 | -1.6006 | 2.2473 | True |
| GANSU_DA | TimeMixer | 70.7496 | 75.8520 | delta_Adapter_AdaY | 75.8697 | -6.7486 | 6.7268 | True |
| LAGO_DE | PatchTST | 4.2457 | 4.7565 | delta_Adapter_AdaY | 4.7413 | -10.4529 | 10.7403 | True |
| LAGO_DE | TimeMixer | 4.3082 | 4.7066 | delta_Adapter_AdaY | 4.7182 | -8.6885 | 8.4635 | True |
| LAGO_PJM | PatchTST | 3.0458 | 3.0196 | delta_Adapter_AdaY | 2.9756 | 2.3564 | -0.8658 | False |
| LAGO_PJM | TimeMixer | 3.0612 | 3.2212 | PIR_paper_protocol | 3.2249 | -5.0763 | 4.9647 | True |

`gap_to_best_baseline_pct = 100 * (MAE_HCH - min(MAE_Direct, MAE_Delta, MAE_PIR)) / min(...)`;
negative means the frozen method beats the cell's best admitted offline baseline.

## All methods, cell medians over seeds

| market | host | method | Overall_MAE | Tail_MAE | Normal_MAE | MSE | RMSE | relative_gain_vs_host_pct | n_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GANSU_DA | PatchTST | Host | 72.0534 | 102.7341 | 68.8239 | 8323.4678 | 91.2330 | 0.0000 | 1 |
| GANSU_DA | PatchTST | MatchedDirectResidual | 92.7100 | 130.4334 | 88.9702 | 14989.8838 | 122.4332 | -28.6684 | 3 |
| GANSU_DA | PatchTST | PIR_paper_protocol | 80.1368 | 153.1646 | 72.4497 | 9738.1914 | 98.6823 | -11.2186 | 1 |
| GANSU_DA | PatchTST | S1_DailyPatch_GRU32 | 70.4342 | 79.0745 | 69.0474 | 8005.2214 | 89.4719 | 2.2473 | 3 |
| GANSU_DA | PatchTST | delta_Adapter_AdaY | 71.5799 | 105.6054 | 67.9978 | 8192.2242 | 90.5109 | 0.6572 | 3 |
| GANSU_DA | TimeMixer | Host | 75.8520 | 157.5844 | 67.2486 | 8859.5996 | 94.1254 | 0.0000 | 1 |
| GANSU_DA | TimeMixer | MatchedDirectResidual | 101.8573 | 146.3390 | 97.5076 | 17991.1409 | 134.1311 | -34.2843 | 3 |
| GANSU_DA | TimeMixer | PIR_paper_protocol | 81.9412 | 162.9480 | 73.4142 | 9999.4883 | 99.9974 | -8.0278 | 1 |
| GANSU_DA | TimeMixer | S1_DailyPatch_GRU32 | 70.7496 | 129.2672 | 64.5898 | 7730.5464 | 87.9235 | 6.7268 | 3 |
| GANSU_DA | TimeMixer | delta_Adapter_AdaY | 75.8697 | 157.3875 | 67.2756 | 8866.0517 | 94.1597 | -0.0234 | 3 |
| LAGO_DE | PatchTST | Host | 4.7565 | 7.4155 | 4.3458 | 40.9278 | 6.3975 | 0.0000 | 1 |
| LAGO_DE | PatchTST | MatchedDirectResidual | 5.6054 | 7.7505 | 5.2764 | 58.0779 | 7.6209 | -17.8467 | 3 |
| LAGO_DE | PatchTST | PIR_paper_protocol | 5.0409 | 8.3165 | 4.5349 | 45.8678 | 6.7726 | -5.9779 | 1 |
| LAGO_DE | PatchTST | S1_DailyPatch_GRU32 | 4.2457 | 6.5817 | 3.8896 | 34.2260 | 5.8503 | 10.7403 | 3 |
| LAGO_DE | PatchTST | delta_Adapter_AdaY | 4.7413 | 7.4683 | 4.3200 | 40.6939 | 6.3792 | 0.3209 | 3 |
| LAGO_DE | TimeMixer | Host | 4.7066 | 7.1416 | 4.3305 | 40.7722 | 6.3853 | 0.0000 | 1 |
| LAGO_DE | TimeMixer | MatchedDirectResidual | 5.8400 | 7.8486 | 5.5395 | 62.4466 | 7.9023 | -24.0810 | 3 |
| LAGO_DE | TimeMixer | PIR_paper_protocol | 5.2705 | 8.4888 | 4.7735 | 48.5505 | 6.9678 | -11.9825 | 1 |
| LAGO_DE | TimeMixer | S1_DailyPatch_GRU32 | 4.3082 | 6.5580 | 3.9607 | 35.0767 | 5.9226 | 8.4635 | 3 |
| LAGO_DE | TimeMixer | delta_Adapter_AdaY | 4.7182 | 7.1875 | 4.3403 | 40.8998 | 6.3953 | -0.2465 | 3 |
| LAGO_PJM | PatchTST | Host | 3.0196 | 10.5586 | 2.7818 | 19.0014 | 4.3591 | 0.0000 | 1 |
| LAGO_PJM | PatchTST | MatchedDirectResidual | 3.6547 | 10.3197 | 3.4446 | 25.0463 | 5.0046 | -21.0332 | 3 |
| LAGO_PJM | PatchTST | PIR_paper_protocol | 3.2761 | 11.7834 | 3.0078 | 21.4048 | 4.6265 | -8.4955 | 1 |
| LAGO_PJM | PatchTST | S1_DailyPatch_GRU32 | 3.0458 | 10.6616 | 2.8055 | 18.8677 | 4.3437 | -0.8658 | 3 |
| LAGO_PJM | PatchTST | delta_Adapter_AdaY | 2.9756 | 11.1464 | 2.7179 | 18.4889 | 4.2999 | 1.4562 | 3 |
| LAGO_PJM | TimeMixer | Host | 3.2212 | 9.6236 | 3.0192 | 20.9680 | 4.5791 | 0.0000 | 1 |
| LAGO_PJM | TimeMixer | MatchedDirectResidual | 3.2902 | 10.8770 | 3.0509 | 21.4295 | 4.6292 | -2.1434 | 3 |
| LAGO_PJM | TimeMixer | PIR_paper_protocol | 3.2249 | 12.5216 | 2.9317 | 20.8207 | 4.5630 | -0.1176 | 1 |
| LAGO_PJM | TimeMixer | S1_DailyPatch_GRU32 | 3.0612 | 8.1744 | 2.8999 | 18.0068 | 4.2434 | 4.9647 | 3 |
| LAGO_PJM | TimeMixer | delta_Adapter_AdaY | 3.2403 | 9.6193 | 3.0399 | 21.2019 | 4.6046 | -0.5948 | 3 |

## Paired day-level block bootstrap (7-day blocks, 1000 replicates, seed 20260911)

Robustness reporting only — no decision in this stage reads this table.

Block count caveat: `GANSU_DA` has 21 DIAG_EVAL days, so a fixed 7-day block partition yields only 3 blocks and hence that few distinct resampled blocks per replicate. The GANSU interval is therefore coarse by construction; it is reported for completeness and carries no gate weight.

Pairing rule: HCH-S1 is taken at the pre-registered middle panel seed, and each baseline at its own contract seed (2021 for the official PIR paper protocol). One row per cell; per-seed variation is reported in `metrics_by_seed.csv`.

| market | host | baseline | HCH_seed | baseline_seed | observed_mean_diff | ci95_low | ci95_high | frac_replicates_negative | n_blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GANSU_DA | PatchTST | delta_Adapter_AdaY | 17 | 17 | -1.1457 | -4.2731 | 3.7252 | 0.7540 | 3 |
| GANSU_DA | TimeMixer | delta_Adapter_AdaY | 17 | 17 | -5.9453 | -6.8885 | -4.7397 | 1.0000 | 3 |
| LAGO_DE | PatchTST | delta_Adapter_AdaY | 17 | 17 | -0.5213 | -0.6913 | -0.3595 | 1.0000 | 16 |
| LAGO_DE | TimeMixer | delta_Adapter_AdaY | 17 | 17 | -0.4087 | -0.4932 | -0.3179 | 1.0000 | 16 |
| LAGO_PJM | PatchTST | delta_Adapter_AdaY | 17 | 17 | 0.0829 | -0.0112 | 0.1691 | 0.0400 | 16 |
| LAGO_PJM | TimeMixer | PIR_paper_protocol | 17 | 2021 | -0.1637 | -0.5383 | 0.1736 | 0.8150 | 16 |

## Gates

- **B0 validity**: `True`
  - frozen S1 replay max abs diff: `0.000e+00` (tol 1e-06)
  - frozen MatchedDirectResidual replay: `True` (max abs diff `0.000e+00` over 18 cell-seed rows)
  - forbidden-path scan clean: `True`
  - no DIAG_EVAL tuning pattern: `True`
  - partition access confined: `True`
- **B1 GANSU hard-case competitiveness**: `True` — strict best in 2/2 cells, median gain vs Host 4.4870% (gate >= 4.0%)
  - `GANSU_TWO_HOST_BEST_OFFLINE`: `True`
- **B2 international parity**: `False` — within 1.0% in 3/4, strict-or-tied best in 3/4
- **B3 overall**: `False` — Host-nonworse in 5/6, median gain vs Host 5.8457%, max gap 2.3564%

### B1 detail

```json
{
  "GANSU_DA/PatchTST": {
    "gap_to_best_pct": -1.600638063562868,
    "best_baseline": "delta_Adapter_AdaY",
    "strict_best": true,
    "gain_vs_host_pct": 2.2473246700798524,
    "normal_harm_vs_host_pct": 0.3247022628784179,
    "no_worse_than_0p5pct": true,
    "normal_harm_ok": true
  },
  "GANSU_DA/TimeMixer": {
    "gap_to_best_pct": -6.748616796614747,
    "best_baseline": "delta_Adapter_AdaY",
    "strict_best": true,
    "gain_vs_host_pct": 6.726754251443086,
    "normal_harm_vs_host_pct": -3.95357608795166,
    "no_worse_than_0p5pct": true,
    "normal_harm_ok": true
  }
}
```

### B2 detail

```json
{
  "LAGO_DE/PatchTST": {
    "gap_to_best_pct": -10.452914461411906,
    "best_baseline": "delta_Adapter_AdaY",
    "strict_best": true,
    "within_0p5pct": true,
    "normal_harm_vs_host_pct": -10.497760772705078,
    "tight_ok": true,
    "loose_ok": true,
    "normal_harm_ok": true
  },
  "LAGO_DE/TimeMixer": {
    "gap_to_best_pct": -8.688511366914197,
    "best_baseline": "delta_Adapter_AdaY",
    "strict_best": true,
    "within_0p5pct": true,
    "normal_harm_vs_host_pct": -8.537912368774414,
    "tight_ok": true,
    "loose_ok": true,
    "normal_harm_ok": true
  },
  "LAGO_PJM/PatchTST": {
    "gap_to_best_pct": 2.3563869656925966,
    "best_baseline": "delta_Adapter_AdaY",
    "strict_best": false,
    "within_0p5pct": false,
    "normal_harm_vs_host_pct": 0.8526921272277832,
    "tight_ok": false,
    "loose_ok": false,
    "normal_harm_ok": true
  },
  "LAGO_PJM/TimeMixer": {
    "gap_to_best_pct": -5.076300630396408,
    "best_baseline": "PIR_paper_protocol",
    "strict_best": true,
    "within_0p5pct": true,
    "normal_harm_vs_host_pct": -3.94970178604126,
    "tight_ok": true,
    "loose_ok": true,
    "normal_harm_ok": true
  }
}
```

## Consequence

The frozen method does not clear the gates against the admitted offline
baselines.

**1. Both GANSU Hosts vs the best admitted baseline.** HCH is strict best on 2/2 GANSU Hosts (`GANSU_TWO_HOST_BEST_OFFLINE = True`), gap `-6.7486%` / `-1.6006%` (negative = HCH better).

| market | host | best_baseline | best_baseline_MAE | HCH_MAE | gap_to_best_baseline_pct |
| --- | --- | --- | --- | --- | --- |
| GANSU_DA | PatchTST | delta_Adapter_AdaY | 71.5799 | 70.4342 | -1.6006 |
| GANSU_DA | TimeMixer | delta_Adapter_AdaY | 75.8697 | 70.7496 | -6.7486 |

**2. Exact HCH gap to the best admitted baseline, every international cell.**

| market | host | best_baseline | best_baseline_MAE | HCH_MAE | gap_to_best_baseline_pct |
| --- | --- | --- | --- | --- | --- |
| LAGO_DE | PatchTST | delta_Adapter_AdaY | 4.7413 | 4.2457 | -10.4529 |
| LAGO_DE | TimeMixer | delta_Adapter_AdaY | 4.7182 | 4.3082 | -8.6885 |
| LAGO_PJM | PatchTST | delta_Adapter_AdaY | 2.9756 | 3.0458 | 2.3564 |
| LAGO_PJM | TimeMixer | PIR_paper_protocol | 3.2249 | 3.0612 | -5.0763 |

The single cell outside the 2.0% band is `LAGO_PJM/PatchTST` at `2.3564%` against `delta_Adapter_AdaY` (`2.9756` vs HCH `3.0458`); the other three international cells are all inside `1.0%`.

**3. Strong enough to justify full-panel transfer.** No. The failure is localised to international parity (B2); the full panel is not run.

- GANSU competitiveness (B1): `True`
- international parity (B2): `False`
- overall (B3): `False`

**4. Branch status.** The current method branch is frozen. A new research
idea and a new window are required. No rescue of S1, alpha, the state design
or any gate is performed or proposed by this stage; see
`NEW_WINDOW_HANDOFF.md` in this evidence root.