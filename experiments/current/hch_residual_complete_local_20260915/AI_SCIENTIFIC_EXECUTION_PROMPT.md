# Local AI — RCL v4.1 最终修补 + 科学实验执行提示词

请严格读取并执行，禁止自行扩展：

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
5. `docs/current/HCH_RCL_V4_1_FINAL_PREEXECUTION_AUDIT_AND_EXECUTION_DESIGN_20260915.md`
6. `experiments/current/hch_residual_complete_local_20260915/SCIENTIFIC_EXECUTION_PROTOCOL.md`

用户已在本轮明确授权 RCL v4.1 scientific campaign，但授权是 **conditional**：必须先完成 Phase 0 final patch gate；任何 gate fail 都必须在 scientific fit 之前停止。

## Phase 0 — final bounded patch

只修复 final audit A–H：

- Stage-1 OOF/final normalization 只能由各自合法 `fit_ids` 拟合；Level loss 使用独立 `s_b`，不得复用全体 raw residual scale；
- Stage-2 `seed_everything(seed)` 必须在 `build_candidate()` 之前；
- final Level 保留 seeds 7/17/37 三个 frozen refits，VAL prediction 与 OOF 一样采用 coordinate-wise median；
- B1 只作为 B2 开始处的 causal q-history warm-start，target/loss/stats 仍严格 B2∪B3∪B4；
- R2 按 `q=A+S+−A−S−` 原生坐标训练，显式 A+/A− targets/readouts/loss，不再套用 R1 的 delta/centered-A objective；
- 固化 R0 exact `BalancedAmplitudeBranch` parity、R3 rho-history+q-history+Level condition、所有 variants zero dead trainable parameters、R3<=1.25×R1；
- 修正 `val_metrics()`：所有 MAE 都在 residual-error space，修正 tail/extreme、normal_harm 符号、q reconstruction、Stage-1 floor、delta closure、CAL MAE；
- verifier 同步独立重算以上定义，不得调用 runner 的 metric 函数。

同时确认：非-joint `host_predictions.npz` 只包含 TRAIN+VAL 行；loaded rows 必须等于 metadata `fit_n+valid_n`，segment 中不得出现 TEST；不得打开 `host_predictions_joint.npz`。

新增足够的 adversarial tests。执行：

- RCL/core tests
- parity tests
- execution-contract tests
- execution-harness tests
- 新增 metric arithmetic / seed / scale-support / final-level-ensemble / R2-native-loss / B1-history tests

只有全部 PASS，且：

- scientific fit count = 0
- TEST target read count = 0
- `src/core` 本轮 final patch 若无需修改则必须 hash 不变；若发现确需修改 core，停止并报告，不得自行扩大 patch

才输出：

`RCL_V4_1_FINAL_PREEXECUTION_GATE_PASS`

若失败输出：

`RCL_V4_1_FINAL_PREEXECUTION_GATE_FAIL_STOP`

并立即停止，不运行实验。

## Phase 1 — scientific execution（仅在 Phase 0 PASS 后）

严格按 `SCIENTIFIC_EXECUTION_PROTOCOL.md` 运行：

- 8 fixed cells；
- Stage-1 OOF 4 folds × seeds 7/17/37；
- final Level 3 frozen seeds + median ensemble；
- R0/R1/R2/R3；
- Stage-2 seeds 7/17/37；
- 96 OOF Level fits + 24 final Level fits + 96 Stage-2 fits = **216 registered neural fits**；
- TRAIN+VAL only；
- primary = raw output；
- secondary pooled alpha 只报告，不参与 primary gate；
- 不重训 Host，不重跑 baseline，不开 QINGHAI，不开 TEST/foreign/final，不做 rescue，不做 hyperparameter sweep。

可以使用 CUDA/CPU 并行提高利用率，但并行只允许改变调度，不得改变：数据顺序、seed、fit support、early stopping、聚合规则、模型结构、metric 定义。每个 fit 设置 `torch.set_num_threads(1)`；避免多个进程写同一 artifact。

## Phase 2 — independent verification

科学执行结束后，必须由独立 verifier 从 raw artifacts 重算：

- 216-fit registry；
- causal OOF supports 与 scale-fit IDs；
- seed-before-model-construction state hashes；
- final Level median-ensemble；
- B1 history-only / B2-B4 target-only；
- R2 A+/A− native target/loss；
- raw Overall / q reconstruction / Level floor / delta closure；
- upper/lower/extreme/failure-tail；
- `normal_harm = max(0,new_error-old_error)`；
- R1 vs R0/R2/R3/Host gates；
- read-set 中无 joint/TEST/QINGHAI/foreign/final；
- scientific run 前后 `src/core` hash 不变。

必须生成：

`experiments/evidence/hch_residual_complete_local_v4_1_20260915/`

以及协议要求的 manifests、checkpoints、emissions、tables、RESULTS.md、VERIFICATION_REPORT.json。

只有 verifier PASS 才返回：

`HCH_RCL_V4_1_SCIENTIFIC_EXECUTION_COMPLETE_FOR_ADJUDICATION`

最终回复只需要：

1. Phase-0 gate 与测试数；
2. 216 fits 完成数；
3. 8-cell R1 vs Host / R0 / R2 / R3 median 表；
4. Level/delta closure 核心指标；
5. Tail/extreme/Normal harm 摘要；
6. 三个主要 adjudication labels；
7. verifier PASS/FAIL；
8. evidence root；
9. terminal token。

到此停止，不自动进行 TEST、国外市场、最终 baseline 主表或论文修改。