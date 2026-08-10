# Extreme-Price Episode Audit

> P0 audit only: no new model was tuned. Episodes are maximal hourly runs within a delivery day.

## Label Topology

| Dataset | Regime | Event hours | Episodes | Median h | Multi-hour | >=4h | Max h |
|---|---|---:|---:|---:|---:|---:|---:|
| GEFCOM14P | negative | 0 | 0 | nan | nan% | nan% | 0 |
| LAGO_BE | negative | 2 | 2 | 1.0 | 0.0% | 0.0% | 1 |
| LAGO_DE | negative | 194 | 47 | 3.0 | 72.3% | 42.6% | 17 |
| LAGO_FR | negative | 2 | 1 | 2.0 | 100.0% | 0.0% | 2 |
| LAGO_NP | negative | 0 | 0 | nan | nan% | nan% | 0 |
| LAGO_PJM | negative | 60 | 18 | 3.0 | 83.3% | 38.9% | 7 |
| NEM_NSW1 | negative | 1086 | 236 | 4.0 | 78.4% | 55.9% | 12 |
| NEM_SA1 | negative | 2590 | 378 | 7.0 | 87.8% | 71.7% | 24 |
| NEM_VIC1 | negative | 2054 | 333 | 6.0 | 83.2% | 65.2% | 24 |
| UNIELEC_AT | negative | 318 | 73 | 4.0 | 84.9% | 56.2% | 11 |
| UNIELEC_BE | negative | 616 | 140 | 3.5 | 80.0% | 50.0% | 15 |
| UNIELEC_CZ | negative | 433 | 98 | 4.0 | 83.7% | 52.0% | 14 |
| UNIELEC_DE | negative | 553 | 119 | 4.0 | 83.2% | 60.5% | 16 |
| UNIELEC_DK | negative | 531 | 104 | 4.0 | 90.4% | 62.5% | 24 |
| UNIELEC_FI | negative | 1031 | 214 | 4.0 | 79.9% | 52.3% | 24 |
| UNIELEC_IE | negative | 78 | 18 | 3.5 | 88.9% | 50.0% | 13 |
| UNIELEC_NL | negative | 752 | 164 | 4.0 | 86.6% | 56.7% | 15 |
| GEFCOM14P | spike | 71 | 13 | 3.0 | 76.9% | 46.2% | 16 |
| LAGO_BE | spike | 167 | 47 | 2.0 | 63.8% | 31.9% | 15 |
| LAGO_DE | spike | 198 | 41 | 2.0 | 61.0% | 34.1% | 16 |
| LAGO_FR | spike | 113 | 38 | 2.0 | 57.9% | 23.7% | 14 |
| LAGO_NP | spike | 1278 | 186 | 4.0 | 85.5% | 58.1% | 24 |
| LAGO_PJM | spike | 11 | 6 | 2.0 | 66.7% | 0.0% | 3 |
| NEM_NSW1 | spike | 61 | 33 | 1.0 | 45.5% | 9.1% | 6 |
| NEM_SA1 | spike | 51 | 21 | 1.0 | 47.6% | 23.8% | 7 |
| NEM_VIC1 | spike | 35 | 12 | 2.0 | 75.0% | 41.7% | 6 |
| UNIELEC_AT | spike | 11198 | 1063 | 7.0 | 92.1% | 80.2% | 24 |
| UNIELEC_BE | spike | 4594 | 742 | 4.0 | 83.2% | 50.3% | 24 |
| UNIELEC_CZ | spike | 10868 | 1142 | 7.0 | 90.5% | 76.3% | 24 |
| UNIELEC_DE | spike | 10515 | 1101 | 7.0 | 92.6% | 78.9% | 24 |
| UNIELEC_DK | spike | 9313 | 966 | 7.0 | 91.8% | 76.0% | 24 |
| UNIELEC_FI | spike | 5290 | 536 | 6.0 | 84.1% | 64.7% | 24 |
| UNIELEC_IE | spike | 24 | 18 | 1.0 | 33.3% | 0.0% | 2 |
| UNIELEC_NL | spike | 9957 | 1237 | 6.0 | 89.0% | 71.5% | 24 |

## Weighted Failure Structure

Rates are weighted by the number of true S4 episodes.

| Regime | Predictor | Event recall | Complete miss | Boundary mismatch | Fragmented | Exact boundary |
|---|---|---:|---:|---:|---:|---:|
| negative | base | 27.8% | 72.2% | 24.0% | 2.6% | 1.1% |
| negative | bech | 33.4% | 66.6% | 29.3% | 2.5% | 1.6% |
| spike | base | 55.0% | 45.0% | 37.2% | 6.9% | 10.8% |
| spike | bech | 55.0% | 45.0% | 37.2% | 6.9% | 10.8% |

## Per-Combination Deltas

Positive event-recall delta is better; positive fragmentation delta is worse.

| Dataset | Backbone | Regime | Event recall delta | Fragmentation delta | Event-hour MAE delta |
|---|---|---|---:|---:|---:|
| GEFCOM14P | GBDT | negative | +nan% | +nan% | +nan |
| GEFCOM14P | Linear | negative | +nan% | +nan% | +nan |
| LAGO_BE | GBDT | negative | +0.0% | +0.0% | +0.000 |
| LAGO_BE | Linear | negative | +0.0% | +0.0% | +0.000 |
| LAGO_DE | GBDT | negative | +0.0% | +0.0% | -1.597 |
| LAGO_DE | Linear | negative | +0.0% | +0.0% | +0.000 |
| LAGO_FR | GBDT | negative | +0.0% | +0.0% | +0.000 |
| LAGO_FR | Linear | negative | +0.0% | +0.0% | +0.000 |
| LAGO_NP | GBDT | negative | +nan% | +nan% | +nan |
| LAGO_NP | Linear | negative | +nan% | +nan% | +nan |
| LAGO_PJM | GBDT | negative | +0.0% | +0.0% | +0.000 |
| LAGO_PJM | Linear | negative | +0.0% | +0.0% | +0.000 |
| NEM_NSW1 | GBDT | negative | +12.3% | +3.0% | -6.705 |
| NEM_NSW1 | Linear | negative | +0.0% | +0.0% | +0.000 |
| NEM_SA1 | GBDT | negative | +12.4% | -3.4% | -15.288 |
| NEM_SA1 | Linear | negative | +15.9% | +0.5% | -10.077 |
| NEM_VIC1 | GBDT | negative | +14.7% | -1.2% | -5.294 |
| NEM_VIC1 | Linear | negative | +9.9% | +0.6% | -4.636 |
| UNIELEC_AT | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_AT | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_BE | GBDT | negative | +0.0% | +0.0% | -0.962 |
| UNIELEC_BE | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_CZ | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_CZ | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_DE | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_DE | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_DK | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_DK | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_FI | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_FI | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_IE | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_IE | Linear | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_NL | GBDT | negative | +0.0% | +0.0% | +0.000 |
| UNIELEC_NL | Linear | negative | +0.0% | +0.0% | +0.000 |
| GEFCOM14P | GBDT | spike | +0.0% | +0.0% | +0.000 |
| GEFCOM14P | Linear | spike | +0.0% | +0.0% | +0.000 |
| LAGO_BE | GBDT | spike | +0.0% | +0.0% | +0.000 |
| LAGO_BE | Linear | spike | +0.0% | +0.0% | +0.000 |
| LAGO_DE | GBDT | spike | +0.0% | +0.0% | +0.000 |
| LAGO_DE | Linear | spike | +0.0% | +0.0% | +0.000 |
| LAGO_FR | GBDT | spike | +0.0% | +0.0% | +0.000 |
| LAGO_FR | Linear | spike | +0.0% | +0.0% | +0.000 |
| LAGO_NP | GBDT | spike | +0.0% | +0.0% | +0.000 |
| LAGO_NP | Linear | spike | +0.0% | +0.0% | +0.000 |
| LAGO_PJM | GBDT | spike | +0.0% | +0.0% | +0.000 |
| LAGO_PJM | Linear | spike | +0.0% | +0.0% | +0.000 |
| NEM_NSW1 | GBDT | spike | +0.0% | +0.0% | +0.620 |
| NEM_NSW1 | Linear | spike | +0.0% | +0.0% | +0.000 |
| NEM_SA1 | GBDT | spike | +0.0% | +0.0% | +0.000 |
| NEM_SA1 | Linear | spike | +0.0% | +0.0% | +0.000 |
| NEM_VIC1 | GBDT | spike | +0.0% | +0.0% | +0.000 |
| NEM_VIC1 | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_AT | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_AT | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_BE | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_BE | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_CZ | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_CZ | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_DE | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_DE | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_DK | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_DK | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_FI | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_FI | Linear | spike | +0.2% | +0.0% | -0.022 |
| UNIELEC_IE | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_IE | Linear | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_NL | GBDT | spike | +0.0% | +0.0% | +0.000 |
| UNIELEC_NL | Linear | spike | +0.0% | +0.0% | +0.000 |
