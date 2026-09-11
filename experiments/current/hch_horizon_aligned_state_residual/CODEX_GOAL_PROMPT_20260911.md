在 canonical repo `D:\作业\science\solar_leak_price_model` 工作。

先按项目规则读取：

`RESEARCH_STATE.md`
`EXPERIMENT_LEDGER.md`
`HANDOFF.md`
`AGENTS.md`
`experiments/AGENTS.md`
`.agents/skills/solar-research-executor/SKILL.md`

然后读取并严格执行唯一 controlling protocol：

`docs/current/HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SHAPE_CLOSURE_20260911.md`

并读取直接前置裁决：

`experiments/evidence/hch_state_excursion_shape_20260910/adjudication/INDEPENDENT_STATE_EXCURSION_SHAPE_ADJUDICATION.md`

本轮只测试一个统一、五参数的 `Horizon-Aligned Semantic State Residual`。

严格冻结 S1 Daily-Patch GRU32、Host、split、Shape target/loss、OOF chronology 和 pooled scalar-Amplitude form。不得重新训练或修改 S1。

只能使用已审计 forecast-time state level，不得使用 state excursion/MAD/1.4826/clipping。把 `state_schema_manifest.csv` 中 `primary_channel=True` 的源列映射到固定五个 semantic roles：

`DEMAND_FC / RENEWABLE_FC / SUPPLY_MARGIN_FC / INTERCHANGE_FC / MUST_RUN_FC`。

复用 S1 对每个 horizon×source-channel 的 train-only standardization，然后在 role 内做确定性均值池化，得到每个市场统一的 `R_d in R^{24x5}`；缺失 role 为结构性 0。

唯一新增参数：`beta in R^5`。

实现：

`q_d = R_d @ beta`

`u_HSA = normalize(u_S1 + q_d)`

`beta` 必须全 0 初始化，因此初始输出精确重放 S1。

`beta` 必须按 protocol 做 stacked chronological OOF，自身预测行不得参与自身训练。训练目标仅为原 Shape loss；固定 AdamW lr=1e-3、weight_decay=1e-4、1000 steps、grad clip=1.0，不得搜索。

得到 HSA OOF Shape 后，按原 exact weighted-median 规则重新拟合 pooled `alpha_HSA`，最终仍为：

`Delta = s * alpha_HSA * u_HSA`。

禁止：

- retrain/modify S1；
- state excursion/MAD/1.4826/alternate windows；
- market ID/Host ID；
- market-specific role selection or parameter count；
- per-hour beta、bias、role embedding、MLP、attention、Transformer、CNN/TCN；
- conditional Amplitude；
- learned Verification/Gate；
- threshold/LR/epoch/seed/hyperparameter search；
- 新市场、山东、S3/S4/protected/final；
- baseline 改动。

固定 panel：

`GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`

seeds：`7,17,37`。

新代码只写：

`experiments/current/hch_horizon_aligned_state_residual/**`

新 evidence 只写：

`experiments/evidence/hch_horizon_aligned_state_residual_20260911/**`

严格执行 H0--H4，并返回且只返回三个 registered token 之一：

`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_SUPPORTED_FOR_PUBLIC_CHINA_EXPANSION`

`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`

`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_INVALID`

汇报 H0--H4、六格 S1→HSA Overall/Tail/Normal MAE、relative gain、Shape cosine、wrong hemisphere、两个 GANSU HIGH-state recovery、beta、alpha_HSA、seed stability、5-param/latency audit 和 artifact 路径，然后停止。即使 PASS，也不得自动扩展公开中国 panel。