# R1B Domain & Feature-Schema Audit v0.1

**Date:** 2026-08-13  
**Auditor:** Claude (autonomous R1B sprint)  
**Data source:** `src/common.py` (`DATASETS` / `load_dataset` / `build_tabular`) + frozen R1A artifact `host_cache_manifest.csv` + live local computation (NORD_DK1 / NEM_SA1 splits).

---

## 1. Summary table

| Market | Coverage | Currency | Target | n_full | n_valid | neg. price rate | price range | n_exog_fc | n_exog_act | n_features | feature_schema_hash | split_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LAGO_DE | 2012-01-09 → 2017-12-31 | EUR | Day-ahead | 52416 | 52224 | 1.026% | [−221.99, 210.0] | 2 | 0 | **21** | `b879812097b253dc` | `9e3a4dcfb2ed15ff` |
| LAGO_PJM | 2013-01-01 → 2018-12-24 | USD | Zonal COMED DA | 52416 | 52224 | 0.177% | [−6.98, 839.3] | 2 | 0 | **21** | `c5a3a4a2196528ce` | `a98174f9394e1f60` |
| NEM_SA1 | 2024-01-01 → 2024-12-31 | AUD | Spot SA1 | 8785 | 8593 | **25.999%** | [−718.8, 14131.5] | 0 | 1 | **19** | `890b07da29ed93e4` | `460e8e97513bdc9f` |
| NORD_DK1 | 2022-01-01 → 2024-10-04 | EUR | Day-ahead DK1 | 24189 | 23997 | 2.774% | [−440.1, 871.0] | 0 | 0 | **17** | `86e23e9f5c98a018` | `b21f0e32707961a5` |

## 2. Feature schema (from `build_tabular`, `src/common.py`)

Every host gets a **cutoff-safe** tabular design matrix `X` (valid-row compressed):

- **Price lags** (`PRICE_LAGS = (24, 48, 72, 168)`): `price_lag24/48/72/168` — 4 features
- **Prev-day aggregates** (rolling 24h at lag 24): `prevday_mean/min/max/std` — 4
- **Prev-week aggregates** (rolling 168h at lag 24): `prevweek_mean/std` — 2
- **Calendar** (hour/dow/month sin+cos + `is_weekend`) — 7
- **Exogenous forecast** (`exog_fc`, safe at t): each column contributes `fc_{c}` + `fc_{c}_lag24` — 2 per column
- **Exogenous actual** (`exog_act`, must lag): each column contributes `act_{c}_lag24` + `act_{c}_lag168` (`ACT_LAGS=(24,168)`) — 2 per column

**Base = 17 features** for a pure-price market (DK1). Each exogenous column adds 2.

| Market | Exogenous columns | Count |
|---|---|---|
| LAGO_DE | `fc_Ampirion Load Forecast`, `fc_PV+Wind Forecast` (both forecast) | 21 |
| LAGO_PJM | `fc_System load forecast`, `fc_Zonal COMED load forecast` (both forecast) | 21 |
| NEM_SA1 | `act_demand` (actual, lagged 24/168) | 19 |
| NORD_DK1 | none | 17 |

All four feature-schema hashes differ → **heterogeneous host-input schemas confirmed**.

## 3. Split counts (valid rows, per `ExperimentManifest`)

| Market | H0 | S1R | S2T | S2V | S3M | S3C | S4 | excluded dates |
|---|---|---|---|---|---|---|---|---|
| LAGO_DE | 20784 | 5232 | 8376 | 2112 | 2616 | 2616 | 10488 | 0 |
| LAGO_PJM | 20784 | 5232 | 8376 | 2112 | 2616 | 2616 | 10488 | 0 |
| NEM_SA1 | 3312 | 888 | 1416 | 336 | 456 | 432 | 1753 | 1 |
| NORD_DK1 | 9456 | 2400 | 3864 | 984 | 1200 | 1200 | 4893 | 3 |

Note the **S3M evidence asymmetry**: NEM_SA1 has only 456 S3M valid days (19d calendar ≈ the R1A.11 evidence-sparse regime), DK1 has 1200 S3M days (vs LAGO 2616). Local calibration eligibility on DK1 will depend on prequential evidence accumulation exactly as in R1A.11.

## 4. Host-input heterogeneity vs HCH-core heterogeneity

- **Host-input heterogeneity — CONFIRMED.** The four markets feed hosts with genuinely different feature schemas (21/21/19/17), different exog types (forecast vs actual vs none), different currencies, and wildly different negative-price rates (0.18%–26%).
- **HCH-core heterogeneity — LIMITED (as designed).** The universal IAH candidate consumes a fixed contract: the host's price-residual-based signature + raw action utility. It does **not** consume arbitrary rich covariates directly. Therefore R1B can support:

  > "the corrector works across hosts trained from heterogeneous market schemas"

  but **cannot yet** support:

  > "HCH directly consumes arbitrary rich feature schemas"

  The latter belongs to R1C / U2 (role-based optional feature encoder, §E7).

## 5. Negative-price / regime observations

- **NEM_SA1** is the extreme regime: 26% negative hours, −718 lower tail, +14132 spikes, single year of data. This is the "rare + extreme" domain the R1A gates had to protect.
- **NORD_DK1** at 2.77% negative rate sits between PJM (0.18%) and DE (1.03%) — a genuinely intermediate regime, and it is the **only market with zero exogenous inputs** (pure price), making it a clean test of the corrector's price-only generalization.
- **LAGO_PJM** has the highest positive spike tail (839) with almost no negatives — a distinct tail-shape regime.

## 6. Data provenance

- LAGO_DE / LAGO_PJM: `data/raw/lago_benchmark/{DE,PJM}.csv` (GEFCom2014-style Lago benchmark)
- NEM_SA1: `data/raw/nem_aemo/clean/SA1_price.csv` (AEMO-derived)
- NORD_DK1: `data/raw/epex_markets/NordPool_DK1_2022_2024.csv` (Nord Pool day-ahead)

## 7. Open items (for later review)

- Exact DST/excluded-day mechanics per market (LAGO 0, NEM 1, DK1 3) — consistent with manifest/excluded-date counts.
- Host quality (MAE by segment, residual stats) per market×host — see `host_quality_by_domain.csv` (§7 of the sprint).
