# Experiments Working Instructions

- 先读根目录 `RESEARCH_STATE.md`、`EXPERIMENT_LEDGER.md`、`HANDOFF.md`，再读 `STAGE_INDEX.md`。
- `foundation/` 是可复用 benchmark/reproduction 基础设施；改动前检查依赖与路径。
- `current/hch_minimal_repair_m0/`、`current/hch_calibrated_ray_closure/`、`current/hch_compact_verification_closure/`、`current/hch_material_gain_market_audit/`、`current/hch_unified_da_shape_upgrade/`、`current/hch_anchored_compact_amplitude/`、`current/hch_market_state_coupling/` 与 `current/hch_host_relative_state_interaction/` 均已完成。Market-State Coupling 独立裁决拒绝 state->distance：仅 1/6 cell 为 `DISTANCE_STATE_COUPLED`，ridge probe 仅 2/6 cell median 改善，`A_state1` 未训练。
- 更强的新证据是 Shape-side：4/6 cells 为 `SHAPE_LIMITED`，包括两个 GANSU_DA Host；高状态区 GANSU Shape cosine 约下降 0.062/0.185，wrong-hemisphere 增加 25/20 pp。
- `current/hch_state_excursion_shape/` 已完成并被独立裁决为 reject：国际四格改善但两个 GANSU_DA cells 均退化。不得通过改 7-day window、MAD `1.4826`、clipping 或 market-specific role selection 救援。
- `current/hch_horizon_aligned_state_residual/` 已完成并被有效拒绝：`HCH_HORIZON_ALIGNED_STATE_RESIDUAL_NOT_SUPPORTED_KEEP_S1`（H0/H4 PASS，H1/H2/H3 FAIL）。它真实改善两个 GANSU_DA Host，但不可迁移；直接 state→residual coefficients 存在跨市场符号冲突。不得用 China-only role、按市场选 role、per-hour/conditional beta、加大 step budget 或事后放宽阈值救援。
- **最终结构 closure** `current/hch_host_relative_state_interaction/` 已完成，严格执行 `docs/current/HCH_HOST_RELATIVE_STATE_INTERACTION_FINAL_CLOSURE_20260911.md` 并返回 `HCH_HOST_RELATIVE_STATE_INTERACTION_NOT_SUPPORTED_FREEZE_S1`（R0/R4 PASS，R1/R2/R3 FAIL）。执行内容：冻结 S1，从现有七天 residual/mask 构造 normalized Host-bias direction `b_hat`，与固定五角色 state table 做 `Z[h,k]=b_hat[h]*R[h,k]`，学习全六-cell 共享的一个 global 5-param beta（cell-macro-balanced Shape loss），`u_HRSI=normalize(u_S1+Z beta)`，随后只按 cell 重拟合现有 exact pooled alpha。证据：`evidence/hch_host_relative_state_interaction_20260911/`，`verify_gates.py` 独立复算 50/50。
- 失败机理：全 panel 共享一个 beta 恰好取消了 HSA 用来拟合 GANSU 的 per-market 系数自由度。GANSU Shape cosine 仅变 `+0.0002`/`-0.0105`（HSA 为 `+0.0666`/`+0.1170`），Overall-MAE 相对 S1 仅在 1/6 cell median 非负、仅 1/6 严格更好，0/4 国际 cell 严格更好，max Normal harm `1.0227%` 超 1% gate。已学习 beta 跨 seed 稳定，但由贡献最多 pooled rows 的四个国际 cell 决定。
- **已冻结 `S1 + pooled fixed alpha` 为最终结构候选**，不得再开任何结构 rescue（含 per-market/per-Host beta、per-hour beta、role selection、conditional Amplitude、换窗口/变换、事后放宽阈值）。下一阶段只能经 human adjudication 后进入完整 public-China/international + audited baseline comparison；未经新的明确授权不得启动该 comparison 或任何后继结构。
- baseline-fidelity workstream 已完成并封账。所有选定比较方法都已完成严格 protocol/official-code/split/training/metric 审计；具体 exact/partial/blocked 边界仍以 `FINAL_BASELINE_FIDELITY_ADJUDICATION_20260911.md` 为准，不能把 blocked/partial 写成 exact reproduced。
- 不得使用 excursion/MAD、新 history window、训练 conditional Amplitude、恢复 Gate/Verification、改变 GRU、添加 attention/Transformer/router/retrieval/market expert、按市场选 role、增加新市场/山东或访问 protected/final roles。
- 当前 development comparator 只作开发参照；final baseline admission 以已登记 paper-fidelity workstream verdict 为准，旧 proxy evidence 不得升级为 SOTA 证据。
- `current/hch_repair_diagnostics/` 已完成并 adjudicated，不得自动重跑。其 diagnostic S3/S4 已被 raw loader materialize，不能再作为 untouched final truth。
- 历史 evidence 必须原位只读保留；Market-State Coupling 新 evidence 只能写到 `experiments/evidence/hch_market_state_coupling_20260910/**`。
- Host 必须使用 target-compatible frozen cache；`GANSU_DA` 不得复用 `GANSU_RT` prediction cache。
- 不允许根据结果增加 rescue variant、阈值、seed、超参或 baseline。
- `history/` 只放阶段导航与生命周期说明，不代替原始证据；`archive/` 不得因路径现代化而重写。
- 本目录存在不等同于科学有效性；结论以 canonical state、adjudication 和 stage index 为准。