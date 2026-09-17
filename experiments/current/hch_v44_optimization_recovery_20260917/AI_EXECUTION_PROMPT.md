你现在执行 **HCH v4.4 optimization recovery**。这是第一次 v4.4 G2 TEST 结果已经暴露之后的 **TRAIN+VAL-only 优化恢复阶段**，不是新架构阶段，也不是允许“看 TEST 调到 SOTA”的阶段。

先严格读取：

1. `RESEARCH_STATE.md`
2. `EXPERIMENT_LEDGER.md`
3. `HANDOFF.md`
4. `experiments/AGENTS.md`
5. `docs/current/HCH_V44_POSTTEST_REGRESSION_ADJUDICATION_AND_RECOVERY_PLAN_20260917.md`
6. `experiments/current/hch_v44_optimization_recovery_20260917/PROTOCOL.md`
7. `src/core/DESIGN_CONTRACT.md`
8. 上一阶段 TRAIN/VAL runner 与 support-only 代码：
   - `experiments/current/hch_v44_china5_fullpanel_main_eval_20260917/implementation/**`
   - 只读取其中 TRAIN/VAL / shadow-OOF / scale / freeze 相关接口

## 最高优先级边界

**本阶段严禁读取或使用 V2 TEST target / TEST prediction / TEST metric 来做任何决定。**

执行器不得读取用于恢复选择的以下结果文件：

- `experiments/evidence/hch_v44_china5_fullpanel_main_eval_20260917/CELL_MEDIAN_RESULTS.csv`
- `.../PER_SEED_RESULTS.csv`
- `.../BASELINE_COMPARISON.csv`
- `experiments/lab/common_benchmark_results_v2/HCH_V44_MAIN_20260917.csv`
- `experiments/lab/common_benchmark_results_v2/RESULTS_LONG.csv`
- `experiments/lab/common_benchmark_results_v2/BASELINE_COMPLETION_20260917.csv`

这些 TEST 数字已经由主窗口完成裁决，但**不能进入本恢复 executor 的选择逻辑**。

允许读取上一阶段的 TRAIN/VAL-only：

- shadow OOF support；
- TRAIN-only scales；
- TRAIN/VAL day ids；
- TRAIN/VAL freeze / epoch-history；
- deterministic caches；
- hashes / source provenance。

先做 `ACCESS_AUDIT.json`，明确列出实际打开过的文件类别和 TEST-read=0。

## 科学对象完全冻结

不得修改 `src/core/**`。

不得修改：

- `b/B/S+/S-` exact geometry；
- decoder；
- Host/history/calendar evidence；
- W=7；
- feature set；
- loss；
- G2 routing；
- network width/depth；
- batch size 32；
- dropout 0.1；
- wd 1e-4；
- grad clip 1.0；
- EMA 0.995；
- seeds 7/17/37。

禁止 G0/G1/G3、optional China target-day feature、gate、retrieval、MoE、Transformer、SAM、PCGrad、calibration rescue、market-specific tuning、best-seed selection。

## 先验证复用 support，不要重新跑 80 shadow Hosts

优先 hash-验证并只读复用上一阶段已经合法生成的：

`V44_SHADOW_OOF_PREFIX_90_10_V1`

以及 TRAIN-only scales / histories。

只有 artifact 缺失或 hash-invalid 才允许按**原 recipe**重建对应 support。不得因为方便重新训练 frozen comparison Host。

## R0：先把旧训练为什么停在 Host 附近测清楚

从 TRAIN/VAL freeze 历史重建：

- 每个 cell 的 steps/epoch；
- selected epoch / estimated selected optimizer step；
- step-0 VAL MAE；
- best old VAL MAE；
- epoch0 是否被选中；
- 有的话读取 TRAIN/VAL correction magnitude；
- 没有的诊断允许在 TRAIN+VAL 上补 instrumentation，但不能开 TEST。

输出：

- `R0_DIAGNOSTICS.csv`
- `R0_SUMMARY.md`

## R1：只改训练预算语义

先跑固定 8-cell recovery panel × seeds 7/17/37 = 24 fits：

- GANSU_DA / PatchTST
- GANSU_DA / LSTM
- SHANDONG_DA / PatchTST
- SHANDONG_DA / iTransformer
- SHAANXI_DA / TimeMixer
- SHAANXI_DA / PatchTST
- NINGXIA_DA / iTransformer
- QINGHAI_DA / TimeMixer

固定 recipe：

- AdamW
- lr = `1e-3`
- batch = `32`
- wd = `1e-4`
- EMA = `0.995`
- max optimizer steps = `2000`
- validation every `50` optimizer steps
- step 0 保留为合法候选 checkpoint
- `0..799` step 内禁止 early stop
- step >=800 后 patience = `8` validation checks = 400 steps
- 因此最早在 step 1200 才可停止
- hard stop = 2000
- difficulty interleaver 保持 TRAIN-only、每 epoch exact once，不 oversample / reweight
- EMA 仍是唯一 primary validation weights；raw 只做诊断

每 50 step 记录：

- step
- epoch-equivalent / sample exposures
- EMA VAL MAE
- raw VAL MAE（diagnostic only）
- frozen TRAIN diagnostic subset reconstruction MAE
- TRAIN / VAL `mean|c| / mean|r|`
- Level MAE / B MAE / Shape W1
- gradient norms
- LR

selection：**所有 validation checkpoints（包括 step0）里 EMA VAL MAE 最低者**。不得为了强行非零 correction 排除 step0。

## R1 panel gate

按 seed-median VAL 独立重算：

1. >=6/8 cells 优于 frozen v4.4 recipe；
2. panel median improvement >=0.5%；
3. >=5/8 cells selected step >0 且 selected `mean|c|/mean|r| >=0.05`；
4. worst degradation <=1.0%；
5. 0 leakage / 0 numerical failure。

如果 PASS：

- 冻结 R1；
- 不重跑已完成 24 fits；
- 只补剩余 12 cells ×3 seeds = 36 fits；
- 形成完整 20-cell TRAIN+VAL recovery panel。

full-panel gate：

- >=15/20 non-worse vs frozen v4.4；
- >=12/20 improve >=0.5%；
- panel-median VAL gain vs Host >0 且显著高于 frozen recipe；
- 没有任何 market median degradation >1.0%；
- 不能按 market 选不同 recipe。

PASS 后只返回：

`HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION`

然后停止。**不要开 TEST。**

## R2：只有 R1 明确属于 optimization-limited 才能触发

如果 R1 panel 不通过，先按协议自己重算 trigger。

只有 >=50% of 24 runs 满足至少一个 optimization-limited 条件才能继续：

- best 到 2000 ceiling；
- final 400 steps TRAIN 仍持续改善但 VAL 尚无稳定 minimum；
- correction magnitude 仍 <5% residual 且 gradient 非零；
- EMA 持续 >400 steps 明显落后于 raw VAL trajectory。

如果 TRAIN 已经学出明显 correction、训练收敛，而 VAL 就是不改善：**立即停止，不得继续调。** 返回：

`HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED`

### R2a（优先）

只改：

`lr = 3e-3`

其余与 R1 完全相同。只跑 8-cell ×3 seeds。

### R2b（只有 readout activation 被直接证明为 bottleneck 时）

只改 Level final affine weight 初始化：

`N(0, 1e-3)`

Level bias 仍 0；B 初始化保持当前 `~0.01*s_B`；其余全部和 R1 相同。

R2a 和 R2b 第一次比较时不得合并。任何一个通过同样的 8-cell gate，就冻结它并扩到 full 20-cell TRAIN+VAL；两个都失败则停止。

## 并行计算要求

今晚目标是尽可能快，但不能改变科学 recipe：

- 先检测 CUDA / VRAM / CPU / RAM；
- 默认 1 GPU training lane；如果两并行 job 的实测 peak VRAM + 余量 < 可用显存的约 75%，可以 2 GPU lanes；
- CPU pool 同时做 cache preparation、artifact verification、summary 聚合；
- 每个 CPU worker `OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1`；
- 不同 cell/seed 独立目录，禁止并发写同一路径；
- 已完成合法 recovery artifact 可 resume，不得换配置静默重跑；
- 不得为了加速改变 batch size / precision contract / seeds / sample support。

## evidence root

只写：

`experiments/evidence/hch_v44_optimization_recovery_20260917/`

严格按 `PROTOCOL.md` 产出全部文件，并运行一个**不 import runner verdict function** 的 independent verifier。

## 终止 token

只允许三个族：

- `HCH_V44_OPTIMIZATION_RECOVERY_READY_FOR_UNTOUCHED_CONFIRMATION`
- `HCH_V44_OPTIMIZATION_RECOVERY_NOT_SUPPORTED`
- `HCH_V44_OPTIMIZATION_RECOVERY_BLOCKED_<REASON>`

无论哪个 token，返回后立即停止。不要打开 V2 TEST，不要访问 foreign/final，不要修改 `paper/**`。
