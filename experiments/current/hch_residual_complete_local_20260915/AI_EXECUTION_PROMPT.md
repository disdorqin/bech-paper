# Local-AI prompt — HCH Residual-Complete Local v4.1

先阅读：

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
5. `experiments/current/hch_residual_complete_local_20260915/PROTOCOL.md`

当前状态是 **CORE ALIGNMENT BLOCKED / NOT YET EXECUTION-AUTHORIZED**。

`RCL_PREPARATION_TESTS_PASS` 只证明 preparation helpers/guards 通过；active `src/core` 仍是上一代 exact-zero-sum alignment-safe method，与 RCL v4.1 不对应。先执行：

`experiments/current/hch_residual_complete_local_20260915/AI_CORE_ALIGNMENT_REFACTOR_PROMPT.md`

并取得：

`RCL_V4_1_CORE_ALIGNMENT_READY_FOR_EXECUTION_REVIEW`

在主窗口再次人工审核之前，不得运行任何 scientific fit。即使 core-refactor token 已返回，本文件也不构成自动实验授权。

实现必须严格体现：

- Stage-1 coarse Level 先独立训练并冻结；
- TRAIN Local target 使用 causal OOF Level prediction；
- Local 显式看到定义其 target 的 Level prediction；
- Local history 改到 post-Level remainder `q` space；
- 主候选 `RCL_ORTHOGONAL`: `q = delta*1 + A(S+ - S-)`；
- matched controls: `STACK_ZERO_SUM / UNTIED_Q / DIRECT_Q`；
- 每个 variant 使用与自身 target geometry 匹配的 TRAIN-only normalization；
- primary evaluation 不允许 dual-alpha rescue；
- TEST read count 必须保持 0；
- 不改 `src/core`，不跑 baseline，不加 gate/safety/retrieval/attention/TCN/Transformer/MoE/market-specific tuning。

特别审计并写测试证明 v4.0 C2 中以下问题在新实现不存在：

1. target-defining Level prediction 未被 Local 观察；
2. target-defining Level 与 inference Level 不是同一冻结 predictor；
3. coordinate target 与 reconstruction target 双重计入 remainder mean；
4. `M*kappa` mean coupling；
5. raw-residual history 与 post-Level target space 不一致；
6. normalization stats 与 target geometry 不一致。

准备完成后停止，并返回实现审计摘要、计划新增文件、测试清单和“等待执行授权”。
