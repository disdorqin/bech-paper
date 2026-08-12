# v0.4 三层架构落地设计

> 目标：把 v0.4 Universal Adaptive Architecture 真正接到正式路径，而不是只存在孤立模块。
> 权威文档：`hch_v2_v0.4_universal_adaptive_architecture_design_2026-08-12.md`
> 数学核心：`hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md`

## 1. 问题回顾（审计结论）

- 模块级 v0.4 存在（hch_v2_context.py），端到端 v0.4 不存在。
- pipeline 用裸 IAHCandidateHead(d_context)，smoke 手工拼 context。
- 三层架构（Universal Core + Data-Adaptive Interface + Local Evidence）在正式路径缺席。

## 2. 数据流（端到端）

```
数据层 (hch_v2_data.py)
    host_raw    [B,H,1]   原始价格（IAH 坐标输入）
    core_ctx    [B,H,13]  scale-free 上下文：u(1) + time_feat(7) + lag_sf(5)
    optional    [B,H,N,F] 可选外生（有则用，无则 core-only）
    target_raw  [B,H,1]|None  S2/S3 有，S4 None
        ↓
核心模型 (iah_candidate.py: IAHCandidateHead)
    1. IAH 坐标：s=mean|host|, z0=asinh(host/s)     [内部算，scale-free]
    2. core_input = concat([z0, core_ctx])          [B,H,14]
    3. h_core = CoreContextEncoder(core_input)      [含 DataSignature FiLM 调制]
    4. h_opt  = OptionalCovariateEncoder(optional)  [zero-init 残差，可 None]
    5. h = h_core + h_opt                            [无 optional 时 h=h_core]
    6. (l⁻,l⁺) = mass_head(h), (r⁻,r⁺) = shift_head(h)
    7. w=softmax([l⁻,0,l⁺]), m=ReLU(r), z⁻=z0-m⁻, z⁺=z0+m⁺, xᵃ=s·sinh(zᵃ)
        ↓
pipeline (hch_v2_pipeline.py: HCHV2UniversalPipeline)
    S1 rank → S2 train → S3-M memory+k → S3-C conformal → freeze/from_bundle → S4 predict
```

## 3. 模块边界（不可再脱节）

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `CoreContextEncoder` | 编码 core_input → h_core，含 FiLM 调制 | 不碰价格 z-score，不碰 target |
| `DataSignature` | 从 z0 算确定性描述子 + learned 池化 → (γ,β) | 不做 domain 分类 loss |
| `OptionalCovariateEncoder` | 外生 → h_optional，zero-init 残差 | 不影响 core 参数 |
| `IAHCandidateHead` | 组装三层 + IAH 数学 head | 不含自己的 context_net |
| `HCHV2UniversalPipeline` | 阶段编排 + freeze/load | 不含数学、不含数据 |

## 4. core_ctx 的 13 维（scale-free）

| 维度 | 通道 | 变换 |
|---|---|---|
| 0 | u（S1 rank） | mid-rank ∈ [0,1] |
| 1-7 | time_feat | 循环编码（hour/day-of-week/week 谐波 + night flag） |
| 8-12 | lag_sf | lagged z0(24/48/168h) + 双曲残差(24h) + availability |

z0 不放进 core_ctx（由 head 内部从 host_raw 算），避免重复且保证与 IAH 坐标一致。

## 5. 四个 bug 的修法

| 审计项 | 修法 |
|---|---|
| CRITICAL-1 select_s3m_k 空操作 | 对每个 k，对每个 validation day 用该 k 的邻居做 final replay 得 Â(k,vd)，score=-mean\|Â(k,vd)−realized_A(vd)\|，选误差最小 k。smoke 调它 |
| CRITICAL-2 三层未落地 | IAHCandidateHead 重构，内部组装 CoreContextEncoder+DataSignature+Optional |
| HIGH-3 无 from_bundle | pipeline 加 from_bundle() 还原 candidate/s1_rank/memory/dvg |
| HIGH-4 neighbors 覆盖 | predict_s4 的 neighbors 改 list append |

## 6. 验收（对应文档 §12 architecture acceptance gate）

- formal runner 只走 IAH 路径，不碰 legacy
- core-only 在 price-only 数据可用
- optional 分支可精确关闭
- S3-M/S3-C 分离，select_s3m_k 真选择
- final π replay 贯穿
- bundle round-trip 复现候选/邻居/π/Â/q/LCB/最终预测
- market_id 不作预测 shortcut
