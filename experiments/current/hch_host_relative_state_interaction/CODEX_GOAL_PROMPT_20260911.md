在 canonical repo `D:\作业\science\solar_leak_price_model` 工作。

先严格读取：

`RESEARCH_STATE.md`
`EXPERIMENT_LEDGER.md`
`HANDOFF.md`
`AGENTS.md`
`experiments/AGENTS.md`
`.agents/skills/solar-research-executor/SKILL.md`

然后读取并严格执行唯一 controlling protocol：

`docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md`

并读取直接前置裁决：

`experiments/evidence/hch_horizon_aligned_state_residual_20260911/adjudication/INDEPENDENT_HSA_ADJUDICATION.md`

本轮是**最后一次结构性方法 closure**。不得自行开启后续 rescue。

固定 panel：

`GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`

固定 seeds：

`7 / 17 / 37`

必须严格冻结 S1 Daily-Patch GRU32、Host、split、S1 OOF/final directions、Shape target/loss、semantic role schema、S1 state normalization，以及 scalar-Amplitude 的 pooled exact MAE calibration form。

唯一新对象：`Host-Relative State Interaction (HRSI)`。

从 S1 已有七天 historical residual/mask 中，对每个 horizon hour 计算 causal recent Host bias：

`b_h = mean(available normalized residuals at the same hour over the existing 7 historical days)`

然后做 24h L2 normalization：

`b_hat = b / (||b||_2 + eps)`

不得更换窗口、加 decay、MAD、clipping 或新 normalization。

复用固定五个 semantic roles 的 standardized role table `R in R^(24x5)`：

`DEMAND_FC`
`RENEWABLE_FC`
`SUPPLY_MARGIN_FC`
`INTERCHANGE_FC`
`MUST_RUN_FC`

构造：

`Z[h,k] = b_hat[h] * R[h,k]`

唯一 trainable parameter：

`beta in R^5`

而且是 **整个六-cell panel 共享的一个 global beta**；不是每个 market/Host 各训练一套。

固定：

`q = Z @ beta`

`u_HRSI = normalize(u_S1 + q)`

`beta=0` 初始化，并验证初始化精确 replay S1。

beta 必须做 cross-market stacked chronological OOF：对每个 fold `Bk`，只汇总六个 cells 各自早于 `Bk` 的合法 OOF rows，按 **cell macro-balanced loss**（每个 cell 总权重相同）训练一个 global beta，再用同一个 beta 预测六个 cells 的当前 `Bk`。任何当前 block label 都不得进入该 fold beta。

最终 DIAG_EVAL 也只能用一个 global beta：在六个 cells 的全部合法 DIAG_FIT OOF rows 上按 cell 宏平均训练，然后同一个 beta 同时应用到六个 DIAG_EVAL cells。

训练目标仍然只允许 Shape MSE：

`||u_HRSI - u_true||_2^2`

固定 optimizer：

AdamW
lr = 1e-3
weight_decay = 1e-4
steps = 1000
grad clip = 1.0

禁止 search。

得到 HRSI OOF Shape 后，用现有 exact weighted-median 规则重新拟合 `alpha_HRSI`，最终仍然只能：

`Delta = s * alpha_HRSI * u_HRSI`

禁止：

- retrain/modify S1；
- alternate bias windows；
- state excursion/MAD；
- role selection；
- per-hour beta / bias / MLP；
- conditional Amplitude；
- Verification/Gate；
- Transformer/attention/CNN/TCN；
- router/expert/retrieval；
- per-market/per-Host beta 或任何 market-specific hyperparameters；
- threshold/LR/epoch/seed search；
- 新市场、山东、S3/S4/protected/final；
- 基于结果放宽 gate。

新代码只能写入：

`experiments/current/hch_host_relative_state_interaction/**`

新 evidence 只能写入：

`experiments/evidence/hch_host_relative_state_interaction_20260911/**`

严格执行 protocol 中 R0–R4。

必须汇报：

- 六格 S1→HRSI Overall/Tail/Normal MAE；
- relative Overall-MAE gain vs Host；
- HRSI-minus-S1 gain pp；
- Shape cosine / wrong-hemisphere；
- 两个 GANSU HIGH-state Shape recovery；
- beta；
- alpha_HRSI；
- bias-profile availability/norm；
- seed stability；
- OOF utility；
- global 5-param beta（每 fold/seed/final）；
- global beta training time；
- absolute inference overhead/day。

完成后只能返回三个 token 之一：

`HCH_HOST_RELATIVE_STATE_INTERACTION_SUPPORTED_METHOD_TARGET_REACHED`

`HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1`

`HCH_HOST_RELATIVE_STATE_INTERACTION_INVALID`

若 PASS，明确指出 method-development target reached 并停止；不得自动跑 public-China/full-panel。

若 FAIL，明确冻结 `S1 + pooled fixed alpha`，不得自动发明下一结构；下一阶段只能等待 human adjudication 后进入完整 public-China/international + 已审计 baseline comparison。
