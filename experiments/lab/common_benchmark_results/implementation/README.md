# Common-Benchmark Lab Implementation

The registry builder lives in the stage that produces the rows:

`experiments/current/hch_china5_common_benchmark_701020/implementation/cb_lab.py`

It refuses any row whose `protocol_id` is not `COMMON_BENCHMARK_701020_V1`, and any `split == TEST` row while the PRETEST seal holds.
