# Common-Benchmark Registry Protocol

1. `protocol_id` is mandatory on every row and must equal `COMMON_BENCHMARK_701020_V1`.
2. `split_manifest_hash` must match the hash recorded in
   `experiments/evidence/hch_china5_common_benchmark_701020_20260912/01_splits/SPLIT_INDEX.json`.
3. `split` is `TRAIN` / `VAL` / `TEST`; main scored rows use `TEST`.
4. `setting` separates data-access regimes. `ONLINE_TTA` and `PREQUENTIAL_SAFETY` rows
   must never be ranked against `OFFLINE_*` rows as if data access were identical.
5. Empty subsets are represented by `coverage_status` (e.g. `NOT_APPLICABLE_N0`) and
   never by a numeric zero.
6. `NORMAL_HARM_VS_HOST` and any harm field must declare `unit` — the registry has
   historically confused absolute units with relative percentages; the unit column is
   mandatory for every harm row.
7. Each row references exactly one evidence root under `experiments/evidence/**`.
8. No row may be written while the TEST seal holds. The builder enforces this by
   refusing any row whose `split == TEST` during PRETEST.
9. The old `../strict_results/` tree is read-only from here.
