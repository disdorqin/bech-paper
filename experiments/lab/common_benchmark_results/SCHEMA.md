# Common-Benchmark Registry Schema

Schema version: `COMMON_BENCHMARK_V1_20260912`

Inherits the `strict_results` vocabularies that remain valid (coordinate status,
admission status, method identity, setting, hash semantics, gain convention, metric
vocabulary, empty-subset rule, residual convention) and adds the protocol-identity
fields required by `HCH_COMMON_BENCHMARK_RESULT_MIGRATION_AND_LAB_PLAN_20260912` §3.

## Coordinate status

`NUMERIC` / `BLOCKED` / `NOT_APPLICABLE_N0` / `METRICS_ONLY` / `SUPERSEDED` /
`HISTORICAL_ONLY` / `PENDING_TEST`.

## Admission status

`COMMON_BENCHMARK_ADMITTED` / `COMMON_BENCHMARK_ADMITTED_WITH_SCOPE_CAVEAT` /
`BLOCKER_ONLY` / `PENDING` / `REJECTED_INVALID`.

## Added fields

- `protocol_id` — mandatory, `COMMON_BENCHMARK_701020_V1`
- `split_manifest_hash` — the frozen split manifest hash of the market
- `test_target_hash` — the target-blind TEST-ID hash of the market
- `host_train_budget` / `host_val_budget` — TRAIN / VAL target-day counts of the Host
- `method_train_semantics` — the partition(s) a method is fitted and selected on
- `online_or_offline` — `ONLINE` / `OFFLINE`

## Hash semantics

Unchanged from `strict_results`: `hash_type` must state what was hashed, and an
`ARRAY_CONTENT_SHA256` is never compared with a compressed container hash.

## Gain convention

`gain_vs_host_pct = 100 * (HostMetric - MethodMetric) / HostMetric`; positive gain is
an improvement over the exact corresponding Host of the same protocol. Host rows carry
gain exactly 0.

## Units

Every harm / normal-region field must declare `unit` as `ABSOLUTE_PRICE` or
`PERCENT_OF_HOST_MAE`. No harm value may be published without its unit.

## Integrity invariants

1. A `BLOCKED` coordinate has no numeric artifact.
2. An `ONLINE_TTA` or `PREQUENTIAL_SAFETY` row never enters an offline ranking.
3. Every `NUMERIC` row has an admitted `COMMON_BENCHMARK_701020_V1` source stage.
4. Every row references exactly one evidence root.
5. No TEST row exists before JOINT_TEST authorization.
6. Inherited blockers carry provenance and no numeric value.
