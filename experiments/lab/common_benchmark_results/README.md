# Common-Benchmark Results Laboratory

Protocol: `COMMON_BENCHMARK_701020_V1`
Schema version: `COMMON_BENCHMARK_V1_20260912`
Stage of record: `hch_china5_common_benchmark_701020`

This lab indexes results produced **only** under the chronological 70/10/20 common
benchmark (TRAIN 70% / VAL 10% / TEST 20% of each market's open eligible target-day
sequence). It is the intended source of the future main comparison tables.

## Separation from `../strict_results/`

`../strict_results/` remains the frozen authority for the old
`HARD_FROZEN_LOW_SHOT_TRANSFER` (`~45/5/15/5/30`) results. The two labs answer
different data-budget questions, so the same market/Host/method may legitimately carry
different values in each. Never average, rank together, or silently supersede across
`protocol_id`.

No numeric row from `strict_results` is copied here. Only a stage executed under
`COMMON_BENCHMARK_701020_V1` may supply a numeric row. Inherited blockers may be
re-registered as blockers with provenance to their fidelity authority, carrying **no**
numeric value.

## Row admission

A row whose `protocol_id != COMMON_BENCHMARK_701020_V1` cannot become active here.
Every numeric row must also carry `split_manifest_hash`, `test_target_hash`,
`host_train_budget`, `host_val_budget`, `method_train_semantics` and
`online_or_offline`.

## State at PRETEST

The registry is intentionally empty: PRETEST freezes splits, Hosts, baselines and the
method definitions, but produces **no** TEST numeric row. `test_rows` is a hard zero
until the JOINT_TEST phase is separately authorized.
