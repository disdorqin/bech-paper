# 新窗口接管提示词 — HCH v4.4 optimization recovery

这是 `solar-model科研` Project 的新窗口。**不要从头重新理解项目，不要重新设计架构。** 当前唯一主任务是：在已经暴露 V2 TEST 的前提下，严格 TEST-quarantined 地恢复 v4.4 G2 的优化路径，得到一个可冻结、可送往 untouched confirmation 的最强合法 TRAIN+VAL recipe。

首先按 Project 规则读取，文件最新事实优先于聊天记忆：

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `experiments/STAGE_INDEX.md`
5. `docs/current/HCH_V44_POSTTEST_REGRESSION_ADJUDICATION_AND_RECOVERY_PLAN_20260917.md`
6. `experiments/current/hch_v44_optimization_recovery_20260917/PROTOCOL.md`
7. `experiments/current/hch_v44_optimization_recovery_20260917/AI_EXECUTION_PROMPT.md`
8. `src/core/DESIGN_CONTRACT.md`
9. 如需 production harness，仅查看上一阶段的 TRAIN/VAL/support 代码：`experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/implementation/**`

## 你必须知道的当前事实

### 1. Baseline 已完成，不再重跑

China-5 × 4 Hosts 下用户指定的五个 baseline 已完整：δ-Adapter、COSA、PIR、UEC-STD、OMPB。不要再做 baseline qualification，也不要重跑 Host/baseline。

### 2. 第一版 v4.4 G2 TEST 结果是有效负/中性结果

同一 `COMMON_BENCHMARK_701020_FULL_V2`、同一冻结 Host、同一可评分日：

- M0 median gain vs Host `+1.83%`
- M1 `+1.78%`
- M2 `+1.78%`
- frozen v4.4 G2 `~+0.00%`
- v4.4 G2 仅 1/20 胜同 setting 最优 offline baseline，median gap `+2.493 MAE`（正值表示更差）

这个 TEST 结果**不能删除、不能改写、不能当作 plumbing bug**。

### 3. 根因已诊断为 frozen recipe 的 optimization-budget failure

`GANSU_DA/PatchTST` 400-step full-batch probe：TRAIN MAE `74.7107 -> 69.3408`，mean|correction| `~0.007 -> 18.73`，说明图可以学习。

但旧 recipe `lr=1e-3 / 50 epochs / patience=8 / bs=32 / EMA=.995` 在不同市场对应的 optimizer step 数差异巨大；60 runs 中 34 个选择 epoch 0，保留 near-zero readout，导致 identity-like correction。

所以当前问题不是“再想一个新架构”，而是：**同一个 v4.4 object 在足够 optimizer steps 下能否形成稳定 VAL improvement？**

## 最高优先级科学边界

**V2 TEST 已经暴露，并且直接触发了本 recovery。现在它必须被隔离。**

Recovery executor 不得读取或使用：

- V2 TEST target
- TEST raw prediction
- TEST cell metrics
- `CELL_MEDIAN_RESULTS.csv`
- `PER_SEED_RESULTS.csv`
- `HCH_V44_MAIN_20260917.csv`
- `RESULTS_LONG.csv`
- `BASELINE_COMPLETION_20260917.csv`
- 任何 TEST 数字做 recipe / lr / init / stopping / seed 决策

允许复用的只有 TRAIN/VAL-only support：shadow OOF、TRAIN-only scales、TRAIN/VAL day IDs、TRAIN/VAL histories/caches、hash/provenance。

如果最后 recovery 成功，**只能冻结 recipe 并申请 untouched confirmation**；不能拿已看过的 V2 TEST 再声称 SOTA/generalization。

## 当前科学对象冻结

不要改：

- `src/core/**`
- exact `b/B/S+/S-` geometry
- exact decoder
- Host/history/calendar evidence
- W=7 revealed history
- coordinate-specific G2 routing
- loss 与 loss weights
- network width/depth
- seeds 7/17/37
- batch 32
- wd 1e-4
- dropout .1
- grad clip 1.0
- EMA .995

不要开：G0/G1/G3、optional target-day China features、retrieval/gate/MoE/Transformer、SAM/PCGrad、calibration rescue、market-specific recipe、best-seed selection、paper edits、foreign/final。

## 今晚的执行路线

### R0 — measurement only

先重建旧 TRAIN/VAL 训练轨迹，确认 steps/epoch、selected step、step0 VAL、best old VAL、epoch0 选择率、correction magnitude。没有的诊断只允许 TRAIN/VAL instrumentation。

### R1 — 主恢复路线

固定 8-cell × 3-seed TRAIN+VAL recovery panel：

- GANSU_DA/PatchTST
- GANSU_DA/LSTM
- SHANDONG_DA/PatchTST
- SHANDONG_DA/iTransformer
- SHAANXI_DA/TimeMixer
- SHAANXI_DA/PatchTST
- NINGXIA_DA/iTransformer
- QINGHAI_DA/TimeMixer

Recipe：

- AdamW
- lr 1e-3
- batch 32
- wd 1e-4
- EMA .995（唯一 primary validation weights）
- max_steps 2000
- VAL every 50 steps
- step0 仍然是合法 checkpoint
- 0..799 禁止 early stop
- step >=800 后 patience = 8 VAL checks = 400 steps
- hard stop 2000
- difficulty interleaver 不变

R1 promotion gate（seed-median VAL）：

- >=6/8 优于 frozen v4.4
- panel median improvement >=0.5%
- >=5/8 selected step>0 且 mean|c|/mean|r| >=0.05
- worst degradation <=1.0%
- 0 leakage / numerical failure

PASS 后复用 24 panel fits，只补剩余 12 cells ×3 seeds，形成完整 20-cell TRAIN+VAL recipe freeze。

### R2 — 只能在 R1 明确 optimization-limited 时触发

若 R1 只是学出了非零 correction 但 VAL 不改善，则**必须停止 v4.4 recovery**，不能继续“调到赢”。

只有 optimization-limited trigger 达标，才可单独测试一个因素：

- R2a：只把 lr `1e-3 -> 3e-3`
- 或 R2b：只有 readout activation 被直接证明 bottleneck 时，把 Level final affine weight init 改成 `N(0,1e-3)`，其余不变

R2a/R2b 首次不能合并。

## 计算要求

目标是一晚上尽快完成，但速度优化不能改变 scientific recipe：

- 先测 CUDA/VRAM/CPU/RAM
- 默认 1 GPU lane；只有实际显存允许才 2 lane
- CPU pool 并行 cache/history verification/summary
- CPU worker 强制 `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- 不同 cell/seed 独立目录
- 已完成合法 artifact 可 resume
- 不得为加速改 batch / seed / sample support / precision contract

## 今晚真正的目标

不是“看 TEST 调到 SOTA”，而是：

> **在 TRAIN+VAL 上找到并冻结最强、最简单、最可辩护的 v4.4 optimization recipe；如果它通过预注册 gate，就把它送到新的 untouched confirmation surface。**

如果 R1/R2 都过不了 VAL gate，就应诚实停止 v4.4 performance recovery，不能无限调参。

## 终止状态

只允许：

- `HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION`
- `HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED`
- `HCH_V44_OPTIMIZATION_RECOVERY_BLOCKED_<REASON>`

拿到任一 token 后停止，并把结果交回主窗口做下一步裁决。