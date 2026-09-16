执行 HCH v4.4 五基线补充实验。只处理这 5 个 baseline：

δ-Adapter

COSA

PIR

UEC-STD

OMPB

先严格读取：

experiments/current/hch_v44_five_baseline_supplement_20260916/PROTOCOL.md

docs/current/HCH_V44_BASELINE_COMPLETENESS_AND_PAPER_FREEZE_20260916.md

docs/current/HCH_COMMON_BENCHMARK_701020_FULL_V2_20260913.md（或项目中对应 V2 控制协议）

各 baseline 最新 qualification/config/evidence 文件。

固定 China-5 × 4 Hosts：
GANSU_DA / SHANDONG_DA / SHAANXI_DA / NINGXIA_DA / QINGHAI_DA
×
PatchTST / TimeMixer / iTransformer / LSTM。

历史完成度必须先独立重建并核对：

δ-Adapter：20/20 已完成 -> 全部复用，禁止无意义重跑；

COSA：20/20 已完成 -> 全部复用，禁止无意义重跑；

PIR：10/20 已完成，只缺 5 市场 × {iTransformer,LSTM} = 10 cells；

UEC-STD：0/20 numeric，本轮按用户授权跑全部 20，保留旧 PAPER_FIDELITY_NOT_ADMITTED 历史标签；

OMPB：0/20 numeric，本轮按用户授权跑全部 20，明确标注 supplementary China transfer，不冒充 exact paper reproduction。

所以正常情况下：

frozen reuse = 50 cells；

new execution queue = 50 cells；

final registry = 100 coordinates。

数据严格使用 COMMON_BENCHMARK_701020_FULL_V2：

GANSU 291/42/83

SHANDONG 1156/166/330 registered，329 scored

SHAANXI 284/41/81

NINGXIA 100/16/28

QINGHAI 94/14/27
禁止重切 split，禁止 Host 重训，禁止按 baseline 重新定阈值。

PIR：

补全 iTransformer 5 cells 和 LSTM 5 cells；

iTransformer 优先使用 official PIR iTransformer 路径/config；

LSTM 若无 paper-native backbone，只做保持 PIR revision algorithm 不变的 generic frozen-Host transfer，并明确标 SUPPLEMENTARY_TRANSFER_NON_PAPER_NATIVE_HOST；

不得把 transfer 行写成 paper-faithful。

UEC-STD：

跑 20 cells；

使用已冻结实现/config；

旧 fidelity gate 7.6062% > 7.5% 的失败事实必须保留；

新数字统一标 SUPPLEMENTARY_TRANSFER / PAPER_FIDELITY_NOT_ADMITTED；

不做 per-market tuning。

OMPB：

跑 20 cells；

保持 online certificate/calibration 算法；

V2 TEST 必须严格 predict/persist -> reveal -> update；

标 SUPPLEMENTARY_TRANSFER / PAPER_NATIVE_SHIFT_PROTOCOL_ADAPTED_TO_CHINA_V2；

不冒充 exact paper reproduction。

并行要求：尽量快，但不能改变科学配置。

先一次性 cache 20 个 frozen Host prediction、split、scoring mask；

CPU_WORKERS=min(24,max(4,physical_cores-2))，大量使用多进程做数据准备、retrieval/index、metrics、hash、CPU-only cell；

每个 GPU training process 内 CPU threads 设小值避免 oversubscription；

自动检测 GPU 数量，用 shared job queue 动态分配 PIR/UEC 等 GPU job；

单 GPU 时 CPU 准备和 GPU 训练流水并行；

多 GPU 默认 1 cell/GPU；只有 smoke test 证明显存充足时才允许 2 jobs/GPU；

不准为了并行擅自改 batch size / epochs / lr；

OMPB/COSA 等顺序在线更新必须保持 cell 内 chronology，但不同 cells 可以并行；

每 cell 都支持 hash-based resume，VERIFIED cell 不重跑。

输出到：
experiments/evidence/hch_v44_five_baseline_supplement_20260916/

必须至少有：
HISTORICAL_COMPLETENESS_MATRIX.csv
FROZEN_REUSE_INDEX.csv
NEW_RUN_QUEUE.csv
CELL_STATUS.csv
STRICT_OFFLINE_TABLE.csv
ONLINE_SUPPLEMENTARY_TABLE.csv
PAPER_COMPARATOR_MATRIX.csv
COMPATIBILITY_ADAPTATIONS.md
PARALLEL_EXECUTION_REPORT.json
SOURCE_AND_CONFIG_HASHES.json
VERIFICATION_REPORT.json
RESULTS.md

独立 verifier 必须确认只存在这 5 个 baseline，正常目标为 100 coordinates = 50 frozen reuse + 50 new runs；禁止 v4.4/HCH fit，禁止 Host retrain，禁止 proxy。

全部结束后只返回：
HCH_V44_FIVE_BASELINE_SUPPLEMENT_COMPLETE_FOR_ADJUDICATION
然后停止。