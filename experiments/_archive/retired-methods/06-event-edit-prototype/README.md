# v7-B: Event-Edit Minimal Prototype & Falsification Pilot - INVALID / ARCHIVED

> Do not run or cite this prototype. See `INVALID_DO_NOT_USE.md` and `../../MANIFEST.md`.

## Overview

This is the minimal prototype for the event-edit hypothesis: **negative-price episodes should be edited as sets, not points**.

The prototype tests whether explicit `KEEP/SCALE/SHIFT/DELETE/INSERT` edit supervision provides independent value over pointwise correction and ordinary 24-vector networks.

## Methods

| ID | Method | Description |
|----|--------|-------------|
| B0 | Frozen Base | No correction (origin) |
| B1 | Pointwise BECH | Current method: median residual on negative points |
| B2 | Pointwise Residual | Linear residual regression on negative points |
| B3 | 24-Vector MLP | Daily mean correction vector |
| B4 | Contiguous Decoder | Duration-based event detection + correction |
| E0 | Event Editor | Hungarian matching + edit supervision (KEEP/SCALE/SHIFT/DELETE/INSERT) |

## Ablations

| ID | Method | Description |
|----|--------|-------------|
| E0_NoInsert | E0 without INSERT | Remove insertion capability |
| E0_NoShift | E0 without SHIFT | Remove boundary shift |
| E0_PointwiseMask | E0 without edit matching | Pointwise mask instead of event-level |

## Data

- LAGO_DE (Germany, EUR/MWh, hourly)
- NEM_SA1 (South Australia, AUD/MWh, hourly)
- UNIELEC_DE (Germany, EUR/MWh, hourly)
- UNIELEC_FI (Finland, EUR/MWh, hourly)
- UNIELEC_NL (Netherlands, EUR/MWh, hourly)

## Protocol

- Strict chronological split: S1(50%)/S2(20%)/S3(10%)/S4(20%)
- Negative threshold: price < 0
- Features: lag-1..24 + hour-of-day
- Base: lag-1 as frozen base predictor

## Metrics

- Episode recall/precision
- Complete-miss rate
- Boundary L1 error
- Duration absolute error
- Event magnitude MAE
- Point recall/precision
- Overall MAE
- Normal-hour MAE/harm
- Exact-fallback rate

## Stop Conditions

Any of:
1. B3/B4 vs E0 no stable difference
2. Removing INSERT or SHIFT does not degrade
3. Event metric improvement comes at cost of normal-hour harm
4. Only works for NEM, not day-ahead markets
5. S3 passes but S4 fails

## Running

```bash
cd experiments/06-event-edit-prototype
python run_pilot.py
```

## Output

- `results/pilot_metrics.csv`
- `results/ablation_metrics.csv`
- `results/PILOT_VERDICT.md` (PROCEED/REVISE/STOP)
