# HCH-v2 v0.3 数学架构融合与代码设计规范

> 日期：2026-08-12  
> 仓库：https://github.com/disdorqin/bech-paper  
> 对照最新 commit：f206de9e8c159c093e6a549092337a15ad37f409  
> 唯一数学依据：hch_v2_iah_crps_final_math_core_v0.3_2026-08-12.md  
> 当前裁决：数学核心允许进入融合；代码必须重写 HCH 主路径后才可实验。

---

## 0. 给代码 AI 的直接指令

请在最新 main@f206de9 上完成 HCH-v2 v0.3 融合。本文是实现规范，数学公式和阶段顺序以数学窗口 v0.3 为准。

目标是形成一个完整、连贯、可冻结、可审计的第二版原型，不是现在追求最终 SOTA、全量实验或消融实验。

必须遵守：

- 宿主始终冻结。
- 只保留一条训练目标：IAH 三原子 CRPS。
- 严格执行：候选生成 → 查询剂量历史回放 → 双事件提案 → 整日动作价值校准 → LCB 门控。
- 不重新引入 BCE occurrence、SmoothL1 magnitude、state loss、Student-t/W2、CARA、KL、temperature、逐小时 confidence sequence、beta-mixing、查询级 rho 或任意多区间 WIS。
- 不使用硬低/正常/高阈值；连续状态只能是结果前上下文。
- S4 不读取目标、残差、动作收益或校准数据。
- 旧路径可以保留，但必须标记 legacy_unvalidated，正式 runner 误调用时 fail closed。
- 优先修改现有文件，不扩展复杂工程目录；最多新增少量数学/策略辅助文件。
- 本轮先跑契约测试、合成数据和一个代表性 smoke，不启动全量实验。

---

## 1. 最新仓库状态：基础修复完成，不等于数学已实现

截至 f206de9，仓库基础层已有进展：host_cache 编译问题、活跃目录 compile、统一 ExperimentManifest、raw/model 双通道、target-free batch、exog mask/type、bundle 字段和 timestamp 对齐已经补齐。

但当前主实现仍是旧 HCH：

| 当前代码 | v0.3 问题 | 必须替换 |
|---|---|---|
| ContinuousStateHead + state_loss | 目标派生状态，不是冻结 S1 中秩 | 删除网络 state head，改确定性 S1 rank context |
| BiOMC occ_d/occ_u | occurrence BCE + magnitude | mass logits + raw bidirectional shifts |
| candidate_loss_fn | BCE + SmoothL1 + W1 组合 | 单一 IAH-CRPS |
| key_net/metric_proj | 旧神经 key，未训练或语义不一致 | residual atomic measure + exact W1 |
| DVG cara_eta/kl_tau | 旧 CARA/KL/temperature | 整日动作价值误差的单侧校准分位 |
| calibrate_s3 | 搜索 eta/tau/k，邻居平均 gain | S3-M 冻结策略，S3-C 只校准 q |
| smoke_v2 | 仍调用旧 loss/state/calibration | 改为 v0.3 五阶段 |

当前状态应记录为 FOUNDATION_READY / IAH_NOT_IMPLEMENTED。

---

## 2. 唯一总体架构

    冻结宿主 + 结果前上下文
                 ↓
       IAH 三原子双尾候选
                 ↓
       CAGM：W1 相似日 + 查询剂量回放
                 ↓
       最多一个 Down 段 + 一个 Up 段
                 ↓
       S3-C 整日动作价值校准
          ↙                 ↘
      LCB > 0             LCB ≤ 0
      执行动作              Identity

三个模块职责：

1. IAH-Candidate：学习候选，不决定是否执行；
2. CAGM-Replay：评估今天候选剂量在历史情景中的价值，不重新训练候选；
3. DVG-Gate：校准完整动作价值，只决定执行或 Identity。

连续低—正常—高状态、周期信息、market/target ID、任意外生变量和 learned-null 是共享条件上下文，不是额外创新。

---

## 3. 数据契约

### 3.1 DailyEpisodeBatch

保留现有 dataclass，但 v0.3 数学路径必须使用：

    host_raw       [B,H,1]       原始货币单位，经济零点保留
    host_model     [B,H,1]       仅供 encoder 的 S1 标准化输入
    target_raw     [B,H,1]|None  S2/S3 使用；S4 必须 None
    target_model   [B,H,1]|None  兼容字段，v0.3 loss 禁止使用
    exog_value     [B,H,N,F]
    exog_type      [B,H,N]
    exog_mask      [B,H,N]       1=valid, 0=null/padding
    lag_context    [B,H,L]
    time_feat      [B,H,T]
    market_id      [B]
    target_id      [B]           山东 DA/RT 必须区分
    timestamps     list
    date_ids       list[str]

target_model 可以保留以减少兼容改动，但严禁用于 IAH loss、尺度、状态、回放或 S4 输出。

### 3.2 日长度与信息边界

数学允许 H_d=23/24/25 或有效小时交集。不得静默把所有日强制 reshape 为 24 点；若实验只保留完整 24 小时日，manifest 必须记录筛选。

候选网络、状态、检索 key、双事件提案和 A_hat 只能使用宿主预测、日历、合法滞后、结果前外生预测和冻结参数。target_raw 只在 S2/S3 存在；S4 batch 必须传 None。

---

## 4. IAH 候选模块

### 4.1 尺度与双曲坐标

对每个预测日计算：

    s  = mean(abs(host_raw), dim=hours)
    z0 = asinh(host_raw / s)

要求：

- s=0、非有限宿主或无有效小时：整日 SCALE_UNIDENTIFIED，直接 Identity；
- 不加数学 epsilon，不做价格 floor，不平移经济零点；
- 用 float64 计算 asinh/sinh；非有限逆变换对应候选无效并回退 Identity；
- s、z0 必须作为证据输出；
- 训练时用同一日 s 计算 zY=asinh(target_raw/s)。

### 4.2 连续共享状态

删除 ContinuousStateHead、compute_state_targets 和 state_loss_fn。新增无可学习参数的：

    u = s1_rank_reference(z0, hour, market_id, target_id)

参考池只由 S1 折外宿主预测构建，保存每个 hour/market/target 的排序结点与中秩插值。无参考池使用 STATE_UNAVAILABLE 和冻结 fallback，不用目标值重建。

u 只作为候选网络上下文，不是三分类标签，不增加 state loss。

### 4.3 候选头

直接重写现有 BiOMC 为 IAHCandidateHead，避免复杂目录：

    mass_logits = Linear(context)       # [B,H,2]
    raw_shift   = Linear(context)       # [B,H,2]
    weights = softmax([l_minus, 0, l_plus])
    m_minus = relu(r_minus)
    m_plus  = relu(r_plus)
    z_minus = z0 - m_minus
    z_plus  = z0 + m_plus
    x_minus = s * sinh(z_minus)
    x_plus  = s * sinh(z_plus)

不要保留 occurrence sigmoid。weights 是预测测度质量，不是事件发生概率和执行概率；m_minus/m_plus 是反双曲候选剂量。

唯一损失：

    L = sum_a w_a * abs(zY-z_a)
        - w_minus*(1-w_minus)*m_minus
        - w_plus*(1-w_plus)*m_plus

按有效小时平均、再按日平均。禁止加入 BCE、SmoothL1、MAE、W1 location、state、tail、market 或 trading loss。

候选输出至少返回 z0/z_minus/z_plus、weights、m_minus/m_plus、raw x_identity/x_down/x_up、scale、state_u 和 valid mask。

---

## 5. CAGM：尺度自由检索与查询剂量回放

### 5.1 residual atomic measure 与 W1

每个日期—小时构造：

    R_h = w_minus*delta(-m_minus)
          + w_zero*delta(0)
          + w_plus*delta(+m_plus)

日期距离：

    D(q,j) = mean_h W1(R_q[h], R_j[h])

质量不等时，合并两个三原子 CDF 的累计质量断点计算一维 W1；禁止使用旧的“三个位置等权绝对差”。有效交集不足或 key 非有限的历史日无效。

retrieval_key_pre_outcome 不得含目标、残差、真实收益或 outcome。结果后字段必须分离存储。

### 5.2 S3-M 顺序

1. 用 S2 历史日或 S3-M 前段建立候选 memory；
2. 用后续 forward validation 选择一个 k；
3. 选择后冻结 memory、W1、k 和双事件提案规则；
4. S3-C 不再搜索或更新这些对象。

同一天不能同时作为 memory 建设样本和 S3-C 校准样本。若使用 S3-M 内部数据，先切 memory_prefix 与 k_validation_suffix。

### 5.3 查询剂量回放

候选提案前必须分别回放两个方向的完整逐时剂量：

    pi_minus[h] = -m_minus_q[h]
    pi_plus[h]  = +m_plus_q[h]

历史回放：

    z_replay_j_a = z0_j + pi_a_q
    g_j[h,a] = abs(zY_j[h]-z0_j[h])
               - abs(zY_j[h]-z_replay_j_a[h])
    A_q_to_j[a] = mean_valid_hours(g_j[:,a])

使用方向回放得到 g_hat[h, down/up] 后，先完成双事件提案。提案确定最终 pi_q 后，再用最终 pi_q 在每个历史日重新回放一次，得到用于 DVG 的完整动作收益 A_q_to_j(pi_q)。必须使用查询日剂量，不能使用历史日自身剂量。回放输出保存查询日期、历史日期、W1 距离、方向剂量、最终 pi_q、逐时 g 和整日 A_q_to_j。

---

## 6. 双事件提案

近邻回放平均得到：

    g_hat[h,down] = mean_j(g_j[h,down])
    g_hat[h,up]   = mean_j(g_j[h,up])

精确选择：最多一个 Down 连续区间、最多一个 Up 连续区间，二者不重叠，任一可为空。

    pi[h] = -m_minus[h] if h in I_down
            +m_plus[h]  if h in I_up
            0            otherwise

用区间前缀和枚举 Down，用 Up 前缀/后缀最大子段和处理不重叠；H=24 时 O(H²)，必须和小 H 穷举对照。不得使用硬尖峰阈值、最短事件长度、固定事件个数或 soft boundary。

边界：两方向无正收益则 Identity；只有一方向则单尾；两个方向重叠则选择总收益最大的非重叠组合；并列采用固定顺序（更短区间、更早起点、Down 优先）。

---

## 7. DVG：整日动作价值单侧校准

### 7.1 S3-C

S3-C 中候选器、S1 state reference、W1、memory、k 和双事件提案均已冻结。每个校准日按顺序：

1. 结果出现前产生 pi_t 和 A_hat_t；
2. 结果出现后计算真实整日双曲收益 A_t；
3. 记录 E_t=A_hat_t-A_t。

校准单位是整日动作，不是小时、区间或 action class。

### 7.2 split-conformal LCB

    r = ceil((n+1)*(1-alpha))
    q = E_sorted[r-1] if r <= n else +inf
    LCB = A_hat_query - q
    final = proposed_action if LCB > 0 else Identity

q=+inf 时必须全日 Identity。交换性下可报告 split-conformal 边际覆盖；真实电价时间序列报告时间顺序经验覆盖、错误放行率、释放率和 Identity 率，不宣称非平稳序列上的分布无关安全。

没有目标市场 S3-C 结果时，zero-shot 不能生成目标市场认证门控。

---

## 8. 冻结 bundle

必须保存并纳入 hash：

- candidate model state、config/version；
- IAH scale 规则；
- S1 rank reference；
- market/target ID；
- residual atom memory、日期/timestamp/data hash；
- W1 版本；
- 冻结 k；
- proposal/tie-break；
- alpha、校准误差或 q；
- split/host commit hash；
- fallback reason codes。

reload 后必须复现 scale、u、candidate atoms、neighbors、pi、A_hat、q、LCB 和 final action，不能只比较模型参数 hash。

---

## 9. 旧代码隔离

正式路径必须删除或替换：

- ContinuousStateHead、compute_state_targets、state_loss_fn；
- occurrence BCE 和旧 candidate_loss_fn；
- 神经 key_net/metric_proj；
- DVG.cara_eta/kl_tau；
- eta/tau 网格校准；
- 旧邻居 gain 和历史自身剂量逻辑。

旧文件可保留，但统一标记 legacy_unvalidated；正式 runner 误调用旧路径必须 fail closed。

---

## 10. 必须替换的测试

至少加入：

1. 正比例缩放后 z0/weights/m/W1/pi 不变，raw action 等比例缩放；
2. all-zero host 返回 SCALE_UNIDENTIFIED + Identity；
3. loss 使用 raw host/target，改变 model scaler 不改变数学输出；
4. mass 三项和为 1，中心 logit 固定为 0；
5. ReLU 可产生精确零位移，Identity 不漂移；
6. x_down <= x_identity <= x_up；
7. 离散 CRPS 与手工公式一致；
8. unequal-mass W1 与手工 CDF 断点一致；
9. 改变 target 不改变 retrieval key；
10. replay 使用 query dose，不使用历史 dose；
11. abs(g) <= abs(pi)；
12. 双事件算法与小 H 穷举一致；
13. Down/Up 不重叠，空提案返回 Identity；
14. conformal rank、q=inf 和 LCB 边界正确；
15. S3-C 不能改变 candidate/memory/k/proposal；
16. target-free S4 可运行且不返回 y_true；
17. bundle round-trip 输出一致；
18. legacy guard 阻止旧 loss/CARA/KL；
19. timestamp join 对顺序扰动不敏感；
20. S1/S2/S3-M/S3-C/S4 日期严格不重叠，target_id 不丢失。

现有测试 07、14、15、17 只证明旧 state/metric/hash，必须改写，不能继续作为 v0.3 证据。

---

## 11. 实现顺序与验收门

Phase 1：实现 raw IAH coordinate、S1 state reference、mass/dose head、CRPS；通过测试 1–8；用合成数据确认有限、符号正确。

Phase 2：实现 residual atom memory、exact W1、query-dose replay；通过测试 9–11；手工核对 2–3 个历史日。

Phase 3：实现 O(H²) double-event proposal；通过测试 12–13；先与穷举对照，不接真实数据。

Phase 4：严格分离 S3-M/S3-C；实现单侧 error quantile 与 LCB；通过测试 14–15；用合成交换性数据检查覆盖。

Phase 5：通过 bundle、target-free S4、timestamp join、legacy guard；只跑一个公开数据集 × 一个 Linear/MLP 基座；输出逐日 candidate/proposal/A_hat/q/LCB/action 证据 JSON。未通过全部验收门前，不开始 half-exp。

---

## 12. 最小证据输出

每个预测日至少记录：

    dataset_id, market_id, target_id, date, timestamp
    host_raw, scale, state_u
    w_minus, w_zero, w_plus, m_minus, m_plus
    proposal_down_start, proposal_down_end
    proposal_up_start, proposal_up_end
    neighbor_dates, neighbor_distances
    A_hat, q_calibration, LCB, final_action, fallback_reason

不要只输出最终 MAE。后续改进要依靠这些证据判断是候选剂量不足、检索失真、双事件提案错误还是价值门控过严。

---

## 13. 主张边界

实现阶段可以记录：IAH 在宿主锚定反双曲坐标中用标准 CRPS 训练结构化三原子候选；条件质量解除固定 1/6、5/6 的表示限制；CAGM 回放查询日剂量；双事件提案最多一个 Down 段和一个 Up 段；DVG 对完整整日动作价值做单侧校准并在 LCB≤0 时 Identity 回退。

暂不允许：自动保证 p99 尖峰、完整分布校准、非平稳电价上的分布无关安全、zero-shot 目标市场安全、sMAPE/套利必然改善，以及 IAH/asinh/CRPS/W1 任一工具的单独首创。

---

## 14. 回传格式

代码 AI 完成后必须回传：

1. 起始/结束 commit；
2. 修改文件及对应章节；
3. compileall 输出；
4. 20 项测试逐项结果；
5. 合成数据结果；
6. 一个公开数据集 × 一个基座 smoke；
7. target-free S4 证据；
8. bundle round-trip 差异；
9. 每日证据 JSON 路径；
10. 未解决问题。

只有 Phase 1–5 全部通过才能标记 IAH_V0.3_IMPLEMENTED；这仍不等同于 PAPER_READY 或 SOTA_CONFIRMED。
