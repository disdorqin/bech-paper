# Local-AI prompt — RCL v4.1 execution-harness preparation only

先读取：

1. `RESEARCH_STATE.md`
2. `HANDOFF.md`
3. `docs/current/HCH_RCL_FINAL_SOURCE_GATE_20260915.md`
4. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
5. `experiments/current/hch_residual_complete_local_20260915/PROTOCOL.md`
6. active `src/core/**`

当前结论：active `src/core` 已通过主窗口第二轮 source audit；**不得再改动 RCL 主方法数学对象，除非发现 correctness bug 并 fail-closed 报告。** 本任务只完成 experiment-layer harness preparation，严禁运行 scientific fits。

必须完成：

1. **R0 matched control 修正**
   - `R0_STACK_ZERO_SUM` 必须复用与 R1 相同的 `BalancedAmplitudeBranch` 及 `config.amplitude_scale/dropout` semantics；
   - 唯一结构差异应是 R0 没有 `delta` coordinate；
   - 不得借机改变 backbone / Shape conditioning / optimizer recipe。

2. **R3 Direct-Q 公平性修正**
   - 当前 R3 计算 Shape/rho-history branch 但 direct head 不读取它，形成 dead trainable branch；
   - 按 protocol 的“same Local backbone/conditioning/history”实现，direct head 必须显式读取：`shape latent + magnitude latent + normalized coarse Level`，或等价地读取能够证明依赖 q-history 与 rho-history 的同容量 representation；
   - 删除任何仍不进入 R3 reconstruction loss 的 trainable Shape output/head；
   - R3 参数量继续满足 `<=1.25× R1`；
   - 加对抗测试：只改 shape/rho history 时 R3 direct q 输出必须变化；backward 后 R3 不得有 unintended dead trainable parameters。

3. **Level conditioning stats 冻结**
   - 新建清楚的 TRAIN-only `LevelConditionStats` 或 experiment-side等价对象；
   - 仅使用 `LOCAL_TRAIN` 对应的 causal OOF `b_tilde` 拟合；
   - 固定：`center = median(b_tilde)`；`scale = median(abs(b_tilde-center))`，clamp `1e-6`；
   - VAL/EVAL 复用该 TRAIN-fitted stats；
   - 禁止用 q/delta stats 代替 coarse-Level stats；
   - runner 不得调用 `LocalNormalizationStats.normalize_level_condition()` 作为 Level-scaling authority。

4. **正式 runner/trainer/verifier 源码准备**
   需要实现但不执行 scientific fit：
   - Stage-1 causal OOF trainer；
   - exact B1..B4 provenance；
   - final frozen Stage-1 refit；
   - `LOCAL_TRAIN=B2∪B3∪B4` target/history construction；
   - R0/R1/R2/R3 Stage-2 training entry；
   - VAL chronological CAL/EVAL prequential evaluator；
   - metric/result writer schema；
   - independent verifier。

5. **input-width contract**
   - `d_base_tokens` 必须由 active core 的 `base_token_dimension(calendar_channels, F)` 计算；
   - 不允许按市场手写 input width；
   - 5/8/9 等不同 F 只影响输入维度，不允许 province/market-specific branch。

6. **结构性数据边界**
   - scientific TEST reader 不得在 runner 的 development execution path 可达；
   - `test_label_read_count` schema 固定为 0；
   - 不访问 QINGHAI、foreign/final；
   - 不重训 Host / baseline；
   - 不改 split。

7. **必须新增的 tests/smoke**
   至少证明：
   - R0/R1/R2/R3 全部 build；
   - 四个 variant 在非退化 synthetic loss 下均无 unintended dead trainable parameters；
   - R3 对 rho-history 与 q-history 都有非零响应；
   - R0 与 R1 的 amplitude primitive class/registered semantics matched；
   - R1 仍直接 import `core.ResidualCompleteLocalModel`，无 duplicate R1；
   - Stage-1 未 freeze 时 Stage-2 optimizer fail-closed；
   - OOF fold training prefix 严格早于 target block；
   - Local TRAIN target 的 coarse-Level artifact 与 batch coarse-level source ID/hash 一致；
   - final Level architecture fingerprint 与 OOF Level architecture fingerprint 一致；
   - VAL history严格 predict→persist→reveal→append；
   - LevelConditionStats 只从 LOCAL_TRAIN causal OOF levels 拟合；
   - `base_token_dimension()` 对 F=1/4/7/8/9 等宽度正确；
   - TEST path structural zero。

禁止：

- 运行任何 8-cell scientific fit；
- 读取 V2 TEST label；
- 读取 foreign/final；
- 修改 paper；
- 增加 gate/safety/retrieval/TCN/attention/Transformer/MoE/market expert；
- result-conditioned rescue；
- 修改已冻结的 8 cells / 3 seeds / protocol gates。

任务完成后只返回：

- changed/new files；
- source test counts；
- synthetic smoke counts；
- R0/R1/R2/R3 parameter counts；
- scientific fit count（必须 0）；
- TEST read count（必须 0）；
- terminal token：

`RCL_V4_1_EXECUTION_HARNESS_READY_FOR_AUTHORIZATION`

到此停止，等待主窗口再次审核和用户单独授权 scientific experiment。
