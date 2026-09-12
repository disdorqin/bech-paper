# FINAL_AUDIT — China-5 post-hoc baseline panel

## A. What was frozen before any result existed

1. `00_protocol/PROTOCOL_FREEZE.md` — task semantics, legality rules, role layout, Host contract, host gate criteria, metric contract, extreme-day definition, prohibitions.
2. `00_protocol/THRESHOLD_FREEZE.json` — q05 / q95 / S90-day-spread per market, fitted on **HOST_TRAIN targets only**, before any Host or baseline output existed.
3. `00_protocol/P0_DATASET_AUDIT.json` — per-market forecast origin, resolution, timezone, target column, legal forecast-known features, forbidden features, missing/duplicate policy, source hash.

## B. Legality assertions

| assertion | value |
|---|---|
| day-ahead → day-ahead semantics | day-ahead price -> day-ahead price, next 24h hourly |
| random split used | False |
| imputation used | False |
| province-specific tuning | False |
| PROTECTED_FINAL read | False |
| Host gate all passed | True |

PROTECTED_FINAL is sealed **structurally**: every reader projects rows at parse time (`skiprows`) and refuses any coordinate belonging to a closed role, so sealed values are never materialised — not merely never used. Each stage records `protected_final_target_values_read = 0` in its own access audit.

## C. Raw prediction registry

`RAW_PREDICTION_INDEX.csv` is the primary artifact: one row per stored array, with `pred`, `y_true` and `residual` all persisted and hashed, plus a per-cell per-day 24h loss file (`*_daily.csv`, indexed in `DAILY_LOSS_INDEX.csv`).

## D. Imported evidence (never retrained)

| item | verdict |
|---|---|
| GANSU_DA Host caches | re-hashed, hash matched, `S3/S4` absent, target column `日前电价` |
| GANSU_DA PIR | re-hashed, `PAPER_FAITHFUL_EXACT_ACCEPTED`, day count matches Host |
| GANSU_DA metric cross-check | re-derived MAE equals published MAE to 0.0 abs diff |
| δ-Adapter / MatchedDirectResidual raw arrays | **absent** — metrics-only import, disclosed |

No model was retrained to make any table tidier.

## E. Disclosed blockers

| track | status | reason |
|---|---|---|
| MARKET::SHANXI | ABSENT_FROM_REGISTRY | No Shanxi dataset file exists: `data/CHINA/` holds only GANSU/LIAONING/NINGXIA/QINGHAI/SHAANXI/SHANDONG, and `data/MARKET_INDEX.csv` has no Shanxi row. Shanxi appears only as a *candidate* row `SHANXI_DA_RT` in the archived v2.5 research table `docs/archive/v2_5_details/HCH_V2.5_DATA_TRAINING_RESEARCH_20260820/dataset_registry_candidates.csv`, classified `C_diagnostic_only` with 0-1 normalised features and an undisclosed licence (open item D5). It was never admitted to the registry. RESEARCH_STATE forbids inventing or silently substituting a market, so no Shanxi cell is reported. |
| MARKET::LIAONING | DATA_BLOCKED | Registered as public=true but role=candidate with admission unresolved; data/CHINA/LIAONING contains only portal HTML captures, no price dataset. |
| BASELINE::UEC_STD | HOST_FIDELITY_BLOCKED / NOT_PAPER_FAITHFULLY_ADMITTED | instantiating it requires a Host that failed admission, i.e. an unverified proxy |
| BASELINE::OMPB | STANDALONE_DATA_BLOCKED / NOT_REPRODUCED | no committed runner and no reproducible standalone data path in this repo |
| IMPORTED::GANSU_DA::PatchTST::Host | IMPORTED_OK | existing strict GANSU_DA evidence imported by integrity audit; never retrained |
| IMPORTED::GANSU_DA::TimeMixer::Host | IMPORTED_OK | existing strict GANSU_DA evidence imported by integrity audit; never retrained |
| IMPORTED::GANSU_DA::PatchTST::PIR_paper_protocol | IMPORTED_OK | existing strict GANSU_DA evidence imported by integrity audit; never retrained |
| IMPORTED::GANSU_DA::TimeMixer::PIR_paper_protocol | IMPORTED_OK | existing strict GANSU_DA evidence imported by integrity audit; never retrained |
| IMPORTED::GANSU_DA::PatchTST::raw_arrays_absent | RAW_ARRAY_ABSENT_METRICS_ONLY | delta_Adapter_AdaY and MatchedDirectResidual raw arrays were never persisted by the frozen GANSU_DA transfer run; only published metrics are importable. |
| IMPORTED::GANSU_DA::TimeMixer::raw_arrays_absent | RAW_ARRAY_ABSENT_METRICS_ONLY | delta_Adapter_AdaY and MatchedDirectResidual raw arrays were never persisted by the frozen GANSU_DA transfer run; only published metrics are importable. |


## F. Statuses that this panel did NOT alter

PIR remains `PAPER_FAITHFUL_EXACT_ACCEPTED`; δ-Adapter remains `ACCEPTED_WITH_DISCLOSED_PARTIAL_FIDELITY`; COSA remains `ONLINE_PARTIAL_FIDELITY / PAPER_NATIVE_METHOD_UNRESOLVED` (the transferred run carries the weaker label `TRANSFERRED_IMPLEMENTATION / PAPER_FIDELITY_UNRESOLVED`). A successful run on a Chinese province never rewrites a paper-fidelity adjudication.

## G. Artifact-hash caveat (recorded, not repaired)

`np.savez_compressed` stamps the local write time into the zip header, so an `.npz` is *content*-deterministic but not *byte*-deterministic: re-writing the same arrays yields a different file hash. The COSA track was re-executed once, to restore the NINGXIA/QINGHAI legality proofs that a partial invocation had clobbered in `COSA_PREFLIGHT.json`; the re-run reproduced every COSA MAE to 4 decimals (SHANDONG PatchTST 103.8608, TimeMixer 102.5355; SHAANXI 101.4078 / 99.4724; NINGXIA 82.4681 / 92.5498; QINGHAI 110.8494 / 114.0960). Every hash in `RAW_PREDICTION_INDEX.csv` and `DAILY_LOSS_INDEX.csv` was then refreshed and verified to match its on-disk artifact, which is what those hashes attest. They attest artifact identity, not bit-reproducibility of the writer.
