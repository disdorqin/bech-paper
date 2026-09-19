在 D:\作业\science\solar_leak_price_model 执行当前冻结的 S1-B2 国际 benchmark，不做任何方法设计。

先读 RESEARCH_STATE.md、EXPERIMENT_LEDGER.md、HANDOFF.md、experiments/AGENTS.md，
再完整读：
docs/current/HCH_S1_B2_PAPER_READY_ADJUDICATION_20260919.md
docs/current/HCH_S1_B2_INTERNATIONAL_BENCHMARK_FREEZE_20260919.md
experiments/current/hch_s1_b2_international_confirmation_20260919/PROTOCOL.md

严格执行：
LAGO_NP / GEFCOM14P / NORD_DK1 / NEM_SA1 × PatchTST / TimeMixer / iTransformer / LSTM；
HCH seeds 7/17/37；方法固定为 S1-B2。

P0 必须先证明 PROTECTED_FINAL 从未读取，然后才允许一次性打开完整 16-cell confirmation。
比较 Host、delta-Adapter、PIR（仅合法 Host）；MDR 只作内部控制；COSA 单列 ONLINE_TTA；
UEC/OMPB/PIR-new-host blocker 不得用 proxy 补数。

禁止调任何 HCH/baseline 超参数，禁止看结果后 rescue，禁止增加数据集，禁止修改 src/paper。
LAGO_DE/PJM 只复用为 development continuity，不进入 untouched gate。

最后运行独立 verifier，并严格按 freeze 的 11 项 Return format 返回。
