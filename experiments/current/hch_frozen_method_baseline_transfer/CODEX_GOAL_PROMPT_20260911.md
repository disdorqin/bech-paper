# Codex Goal — Frozen HCH vs Audited Baselines

在 canonical repo `D:\作业\science\solar_leak_price_model` 工作。

先读取：

`RESEARCH_STATE.md`  
`EXPERIMENT_LEDGER.md`  
`HANDOFF.md`  
`AGENTS.md`  
`experiments/AGENTS.md`  
`.agents/skills/solar-research-executor/SKILL.md`

然后严格读取并执行唯一 controlling protocol：

`docs/current/HCH_FROZEN_METHOD_BASELINE_TRANSFER_ADJUDICATION_20260911.md`

同时读取 baseline fidelity authority：

`experiments/evidence/hch_baseline_paper_fidelity_20260911/adjudication/FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md`

以及 PIR / δ-Adapter 的独立 anchor adjudication，确保 transfer runner 来自已审核 paper-fidelity 路径，而不是历史 proxy：

`experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_PIR_ANCHOR_ADJUDICATION.md`

`experiments/evidence/hch_baseline_paper_fidelity_20260910/adjudication/INDEPENDENT_DELTA_ADAPTER_ANCHOR_ADJUDICATION.md`

本轮不再改方法。

冻结 HCH：

`S1 Daily-Patch GRU32 Shape + pooled exact OOF MAE alpha`

固定 panel：

`GANSU_DA / LAGO_DE / LAGO_PJM × PatchTST / TimeMixer`

主 offline comparison 只能包括：

- Host / Identity
- frozen MatchedDirectResidual
- audited δ-Adapter Ada-Y transfer
- audited PIR full-official / paper-protocol transfer
- frozen HCH-S1

严禁使用旧 `PIR_PROXY_10epoch_K10`、历史 `src/baselines/pir.py` ridge/refiner substitute、旧 2-epoch δ wrapper，或将 COSA/UEC/OMPB 混入 main offline best-baseline ranking。

COSA 只有在已有合法 audited transfer runner、无需任何 rescue 的情况下，才可放入单独 online/TTA supplementary table；不得阻塞主结果。

必须保证：

- 相同 dataset rows / DA target / Host family / final metric implementation；
- 每个方法保留自身 paper-native training semantics；
- 无 target-day label leakage；
- 无 DIAG_EVAL hyperparameter tuning；
- HCH、baseline 均不得根据结果调参；
- 不读取 SHAANXI/NINGXIA/QINGHAI/SHANDONG/S3/S4/protected/final。

必须执行 B0–B4，并计算每个 cell：

- Overall MAE
- Tail MAE
- Normal MAE
- MSE/RMSE
- relative gain vs Host
- HCH gap to best admitted offline baseline
- parameter count / training time / inference time
- fidelity label

同时做固定 7-day block bootstrap：1000 replicates，seed `20260911`，比较 HCH vs 当前 cell best admitted baseline 的 day-level MAE 差异。Bootstrap 只做 robustness reporting，不得用于调参。

新代码只能写：

`experiments/current/hch_frozen_method_baseline_transfer/**`

新 evidence 只能写：

`experiments/evidence/hch_frozen_method_baseline_transfer_20260911/**`

完成后只能返回三个 token 之一：

`HCH_BASELINE_TRANSFER_TARGET_MET_FULL_PANEL_ALLOWED`

`HCH_BASELINE_TRANSFER_NOT_COMPETITIVE_NEW_IDEA_REQUIRED`

`HCH_FROZEN_BASELINE_TRANSFER_INVALID`

若 PASS：明确报告 GANSU 两 Host 相对 best baseline 的位置、四个国际 cell 的 gap，并停止；不得自动 full panel。

若 FAIL：明确指出失败来自 GANSU competitiveness、international parity 或两者；冻结本 branch，不得 rescue S1/alpha/state/Gate。然后停止，等待 human adjudication / new-window research reset。
