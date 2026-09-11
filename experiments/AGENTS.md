# Experiments Working Instructions

- 先读根目录 `RESEARCH_STATE.md`、`EXPERIMENT_LEDGER.md`、`HANDOFF.md`，再读 `STAGE_INDEX.md`。
- `foundation/` 是可复用 benchmark/reproduction 基础设施；改动前检查依赖与路径。
- `current/hch_minimal_repair_m0/`、`current/hch_calibrated_ray_closure/`、`current/hch_compact_verification_closure/`、`current/hch_material_gain_market_audit/`、`current/hch_unified_da_shape_upgrade/`、`current/hch_anchored_compact_amplitude/` 与 `current/hch_market_state_coupling/` 均已完成。Market-State Coupling 独立裁决拒绝 state->distance：仅 1/6 cell 为 `DISTANCE_STATE_COUPLED`，ridge probe 仅 2/6 cell median 改善，`A_state1` 未训练。
- 更强的新证据是 Shape-side：4/6 cells 为 `SHAPE_LIMITED`，包括两个 GANSU_DA Host；高状态区 GANSU Shape cosine 约下降 0.062/0.185，wrong-hemisphere 增加 25/20 pp。
- `current/hch_state_excursion_shape/` 已完成并被独立裁决为 reject：国际四格改善但两个 GANSU_DA cells 均退化。不得通过改 7-day window、MAD `1.4826`、clipping 或 market-specific role selection 救援。
- `current/hch_horizon_aligned_state_residual/` 已完成并被有效拒绝：`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`（H0/H4 PASS，H1/H2/H3 FAIL）。五个共享 `beta` 真实改善两个 GANSU_DA Host（Shape cosine `+0.0666`/`+0.1170`），但不可迁移：vs S1 增益仅在 4/6 cell median 非负，vs Host 仅 3/6 cell 达 `>=5%`，Normal-MAE 最大伤害 `1.02%`。证据 `experiments/evidence/hch_horizon_aligned_state_residual_20260911/`，独立复核 `verify_gates.py`（不 import runner）39/39。不得用 China-only role、按市场选 role、per-hour/conditional beta、加大 step budget 或事后放宽阈值救援；当前无已授权的后继执行。
- baseline-fidelity workstream 已完成并封账，最终解释以 `experiments/evidence/hch_baseline_paper_fidelity_20260911/adjudication/FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md` 为准：δ-Adapter partial accepted、PIR exact accepted、COSA partial online、UEC Host-fidelity blocked、OMPB official-data blocked。当前不得继续 anchor/rescue search；later HCH transfer 需单独授权。
- 本轮冻结 S1 本体、Host/split/OOF、Shape target/loss 与 pooled scalar-Amplitude form；复用 S1 train-only standardized state level，按固定五个 semantic roles 做确定性 role pooling，学习且只学习 `beta in R^5`，`u_HSA=normalize(u_S1 + R beta)`，再在 HSA OOF directions 上重拟合 exact pooled alpha。
- 不得使用 excursion/MAD、训练 conditional Amplitude、恢复 Gate/Verification、改变 GRU depth/hidden size、添加 attention/Transformer/router/retrieval/market expert、按市场选 role、增加新市场/山东或访问 protected/final roles。
- 当前 development comparator 只作开发参照；final baseline admission 以已登记 paper-fidelity workstream verdict 为准，旧 proxy evidence 不得升级为 SOTA 证据。
- `current/hch_repair_diagnostics/` 已完成并 adjudicated，不得自动重跑。其 diagnostic S3/S4 已被 raw loader materialize，不能再作为 untouched final truth。
- 历史 evidence 必须原位只读保留；Market-State Coupling 新 evidence 只能写到 `experiments/evidence/hch_market_state_coupling_20260910/**`。
- Host 必须使用 target-compatible frozen cache；`GANSU_DA` 不得复用 `GANSU_RT` prediction cache。
- 不允许根据结果增加 rescue variant、阈值、seed、超参或 baseline。
- `history/` 只放阶段导航与生命周期说明，不代替原始证据；`archive/` 不得因路径现代化而重写。
- 本目录存在不等同于科学有效性；结论以 canonical state、adjudication 和 stage index 为准。