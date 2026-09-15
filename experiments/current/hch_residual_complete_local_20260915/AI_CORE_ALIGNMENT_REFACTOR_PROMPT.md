# AI prompt — RCL v4.1 core alignment refactor before any scientific run

先严格读取：

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `src/AGENTS.md`
5. `docs/current/HCH_RESIDUAL_COMPLETE_LOCAL_DESIGN.md`
6. `docs/current/HCH_RCL_CORE_ALIGNMENT_REFACTOR_PLAN_20260915.md`
7. `experiments/current/hch_residual_complete_local_20260915/PROTOCOL.md`
8. 当前 `src/core/README.md`
9. 当前 `src/core/DESIGN_CONTRACT.md`

当前结论：`RCL_PREPARATION_TESTS_PASS` 只代表 target/helper/guard 准备通过，**不代表 core 与方法一致，也不代表方法实现完成**。当前 active core 仍是旧的 exact-zero-sum alignment-safe method，并明确禁止 daily-level term；这与 RCL v4.1 的 `q=delta*1+A(S+−S-)` 冲突。因此在任何 scientific fit 前，先完成一次 source-only core alignment refactor。

## 任务

严格执行：

`docs/current/HCH_RCL_CORE_ALIGNMENT_REFACTOR_PLAN_20260915.md`

### A. 先归档旧 active core

把当前 `src/core/**` byte-preserving 归档到：

`src/archive/alignment_safe_core_pre_rcl_20260915/legacy_core/`

生成 per-file SHA256、tree digest、archive manifest，并在改写后复算证明 archive 未变化。

### B. 重构 active `src/core`

active core 必须实现当前主候选：

1. `CoarseLevelModel`
   - current legal facts + past raw Host residual history；
   - shared deterministic stem + magnitude GRU32；
   - one signed scalar coarse Level；
   - target `b*=mean(r)`。

2. `ResidualCompleteLocalModel`
   - 输入必须显式包含 frozen coarse Level prediction；
   - history 必须是 post-Level q/rho history；
   - Local 输出必须是：

   `local = delta*1 + A*(S_plus-S_minus)`

   - `delta` signed；
   - `A>=0`；
   - `S+/S-` masked simplex；
   - only centered Shape part zero-sum；
   - Local overall允许 nonzero mean；
   - final correction = coarse Level + Local。

3. central targets：

   `q=r-b_hat*1`

   `delta=mean(q)`

   `rho=q-delta*1`

   `A=0.5*||rho||_1`

   `q=delta*1+A(S+-S-)`

4. Local normalization 必须只从 legal LOCAL_TRAIN q geometry fit。

5. Stage-2 model **不得持有或调用 Stage-1 model object**。它只接收 frozen coarse-level tensor，保证冻结边界结构化。

### C. experiment package 与 core 必须一一对应

- R1 `RCL_ORTHOGONAL` 不得在 `src/mvp` 复制一份网络；runner 必须直接 import active `src.core.ResidualCompleteLocalModel`。
- R0 `STACK_ZERO_SUM`、R2 `UNTIED_Q`、R3 `DIRECT_Q` 才是 experiment-only comparators。
- 删除/改造当前 preparation-only duplicate target formulas，使 central math authority 只有一份；MVP 可薄封装 core，但不能维护第二套公式。

### D. 旧 safety/history

旧 `safety.py/history.py` 属于上一代 active method。按 plan byte-preserving archive 后从新 active core 移出，不把 safety 接进 RCL，也不要兼容垫片伪装成当前方法。

### E. 六项 v4.0 缺陷必须用结构级测试闭合

不能只靠 `assert_rcl_contract()` 的 caller 声明。测试必须证明：

1. Local 真正观察 target-defining Level prediction；
2. TRAIN OOF Level 与 inference final Level 来自同一个 `CoarseLevelModel` architecture family；
3. oracle coordinates 可让 reconstruction 与 coordinate losses 同时为 0；
4. active R1 无 `kappa` / `M*kappa` coupling；
5. Local history 真正来自 q-space；raw residual-only Local 调用 fail-closed；
6. q/delta/A normalization 与 target geometry 匹配。

### F. 新增/更新测试

至少覆盖 plan §8：

- `test_core_rcl_targets.py`
- `test_core_rcl_preprocessing.py`
- `test_core_rcl_model.py`
- `test_core_rcl_losses.py`
- `test_rcl_core_experiment_parity.py`
- `test_rcl_execution_contract.py`

旧绑定 exact-zero-sum / safety active-object 的测试，先 byte-preserving 归档并记录 superseded mapping，不得为了“全绿”篡改成含义不同的断言。

### G. 文档同步

更新：

- `src/core/README.md`
- `src/core/DESIGN_CONTRACT.md`
- `src/AGENTS.md`
- `docs/current/README.md`
- `RESEARCH_STATE.md`
- `HANDOFF.md`
- `experiments/STAGE_INDEX.md`

不要改 `EXPERIMENT_LEDGER.md`，因为本任务禁止 scientific fit。
不要改 `paper/**`。

## 严格禁止

本任务禁止：

- 任何 scientific training/evaluation fit；
- 读取 V2 TEST / foreign / final；
- Host/baseline 重训；
- safety/gate/retrieval/KNN/attention/Transformer/TCN/MoE/router/market expert；
- per-market architecture；
- loss-weight search；
- 结果条件化 rescue。

可以运行：source/unit tests、compile、static audit、hash audit、纯 synthetic toy tests。

## 最终必须返回

1. old core archive path + tree hash + byte-preserving proof；
2. new active core exact file list；
3. method data-flow：Stage-1 Level -> frozen prediction -> q -> Stage-2 RCL -> final correction；
4. R1 core/experiment parity proof；
5. 六项 v4.0 semantic defect 的逐项 closure；
6. tests 命令与 PASS/FAIL 数；
7. scientific fit count = 0；
8. TEST label read count = 0；
9. 所有修改/新增文件；
10. terminal token：

`RCL_V4_1_CORE_ALIGNMENT_READY_FOR_EXECUTION_REVIEW`

到此停止。**不要自动进入实验。**
