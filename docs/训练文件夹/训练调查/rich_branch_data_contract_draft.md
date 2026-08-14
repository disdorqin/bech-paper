# HCH-v2 §6 Rich Covariate Branch — 数据契约草案 v0.1

**日期：2026-08-14**
**用途：论文 §6（Rich covariate branch）的数据契约草案，供调查报告引用。只调查不实现。**
**上游依据：**
- `src/common.py`：`load_shandong`（L268-312）、`DATASETS`（L34-63）、`build_tabular`（L120-164）、`PRICE_LAGS/ACT_LAGS/SEQ_LEN`（L115-117）
- `src/hch_v2_context.py`：`OptionalCovariateEncoder`（L159-199）、`ROLES`（L166-167）
- `src/iah_candidate.py`：`IAHCandidateHead`（L32-162），`optional_*` 前向参数（L84-118）
- `src/hch_v2_pipeline.py` / `src/universal_trainer.py`：`d_value=0` 主路径
- 架构文档 `docs/paper_prep/v2_final/hch_v2_v0.4_universal_adaptive_architecture_design_2026-08-12.md` §1/§4/§9
- 主计划 `docs/paper_prep/v2_final/hch_v2_universal_training_master_plan_v0.1_2026-08-13.md` §11（U2-Rich）、P1-4/P1-5
- 数据审计 `docs/paper_prep/v2_final_prep/r1b_domain_feature_schema_audit_v0.1.md`、`public_da_rt_dataset_audit_v0.1.md`

**核心事实（当前状态）：** 主实验 `D_VALUE=0`（`experiments/08-hch-v2/r1a_run.py` L78），`IAHCandidateHead` 构造时 `optional_encoder = None`（`iah_candidate.py` L49-50），`DomainBatch` 与 `UniversalCoreTrainer` 均不携带 `optional_values/roles/masks`。**论文目前不能声称 HCH 已利用山东丰富外生变量**；本草案是启用该分支前必须锁定的数据契约。

---

## 1. 山东 DA/RT 特征清单与类型映射

### 1.1 原始文件事实（实测）

文件 `data/raw/provinces/shandong_pmos_hourly.csv`（GBK 编码，23 列，39,816 行，2022-01-01 01:00 → 2026-07-18 00:00，北京时 UTC+8，中国无 DST）。

| # | 列名 | 含义 | 单位* | 缺失(NaN) | `load_shandong` 现行分类 |
|---|---|---|---|---|---|
| 0 | 时刻 | 时间戳（北京时） | datetime | 0 | timestamp |
| 1 | 日前电价 | Day-ahead 出清电价 | 元/MWh | 24（约 1 日） | `price_col`（默认 target） |
| 2 | 实时电价 | Real-time 出清电价 | 元/MWh | 48（约 2 日） | `price_col`（备选 target） |
| 3 | 地方电厂总加预测值 | 地方电厂出力聚合预测 | MW | 0 | exog_fc |
| 4 | 联络线受电负荷预测值 | 省间联络线受电负荷预测 | MW | 0 | exog_fc |
| 5 | 风电总加预测值 | 风电出力聚合预测 | MW | 0 | exog_fc |
| 6 | 光伏总加预测值 | 光伏出力聚合预测 | MW | 0 | exog_fc |
| 7 | 核电总加预测值 | 核电出力预测 | MW | 0 | exog_fc |
| 8 | 自备机组总加预测值 | 自备电厂机组出力预测 | MW | 0 | exog_fc |
| 9 | 试验机组总加预测值 | 试验机组出力预测 | MW | 0 | exog_fc |
| 10 | 直调负荷预测值 | 全网统调（直调）负荷预测 | MW | 0 | exog_fc |
| 11 | 竞价空间预测值 | 市场竞价空间预测 | MW | 0 | exog_fc |
| 12 | 新能源总加预测值 | 新能源（风+光）出力预测 | MW | 0 | exog_fc |
| 13 | 地方电厂总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 14 | 联络线受电负荷实际值 | 对应实际值 | MW | 48 | exog_act |
| 15 | 风电总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 16 | 光伏总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 17 | 核电总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 18 | 自备机组总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 19 | 试验机组总加实际值 | 对应实际值 | MW | 48 | exog_act |
| 20 | 直调负荷实际值 | 对应实际值 | MW | 48 | exog_act |
| 21 | 竞价空间实际值 | 对应实际值 | MW | 48 | exog_act |
| 22 | 新能源总加实际值 | 对应实际值 | MW | 48 | exog_act |

\* 单位推断为 MW：核电 ~2465 MW（与海阳核电站 2×1250 MW 量级一致）、直调负荷 ~60 GW（与山东网架量级一致）、光伏夜间为 0。价格单位 `meta.currency="CNY"`，惯例为 元/MWh。正式论文引用前应再向数据提供方核对单位。

**现行 `load_shandong` 的切分逻辑（`common.py` L288-294）：** `"电价" in c` → 排除出 exog；`"预测" in c` → exog_fc；`"实际" in c` → exog_act。⚠️ 该逻辑把 `日前电价/实时电价` 一律当 target 排除，**不感知 target**。注释里提到的「全省负荷预测总值」在实际文件中不存在，注释已过期。

### 1.2 四类角色映射（信息截断约束）

| 类别 | 山东列 | 截断语义 | 时点规则 |
|---|---|---|---|
| **KNOWN_FUTURE**（DA 已知的未来输入） | 10 个「预测值」列：地方电厂/联络线受电负荷/风电/光伏/核电/自备机组/试验机组/直调负荷/竞价空间/新能源总加 | 均为 D-1 日前预测，对目标时刻 t（D 日）在 cutoff 时已知 | `t` 直接可用（同 `build_tabular` 的 `exog_fc` → `fc_{c}`） |
| **OBSERVED_PAST**（滞后观测） | 10 个「实际值」列 | 实际值在运行后实现，cutoff 前只可得 ≤ D-1 的历史 | 必须滞后 ≥24h（同 `ACT_LAGS=(24,168)` → `act_{c}_lag24/168`） |
| **CALENDAR**（日历） | 时刻派生：`hour_sin/cos, dow_sin/cos, mon_sin/cos, is_weekend`（`build_tabular` L149-157）；`_cyclic_time_features`（`r1a_run.py` L111-118） | 恒已知 | `t` 直接可用 |
| **STATIC**（静态） | 文件中无逐行静态列 | — | 仅市场元数据：`currency=CNY`、tier、负价下限等，来自 `meta`，非逐行特征 |

**无法直接归类的列（明确说明）：**
- `新能源总加预测值/实际值` ≡ `风电 + 光伏`（实测 01:00 风电=新能源=8930.405、光伏=0）。是**派生冗余量**，建议在适配器标记 `derived_from=[wind, solar]`；保留它不改变角色，但若同时喂风电+光伏+新能源会造成信息重复。
- `竞价空间预测值/实际值` = 负荷 − 非市场机组（派生量），语义上仍是 KNOWN_FUTURE / OBSERVED_PAST，但需记录 derivation 关系。
- **节假日不在文件中**：`is_weekend` 只是周末代理，春节/国庆等中国法定节假日需外部日历，属于 CALENDAR 缺口（当前无，必须显式声明缺失，不得用周末代理冒充）。

### 1.3 target 依赖的角色翻转（关键，易泄漏）

同一列的角色随 target 改变，契约必须按 `(market, target)` 绑定角色：

| 条件 | 日前电价 | 实时电价 |
|---|---|---|
| **target = 日前电价（shandong_DA）** | **target，禁止进入当日输入** | OBSERVED_PAST（只可滞后 ≥24h 使用） |
| **target = 实时电价（shandong_RT）** | **KNOWN_FUTURE（D-1 已发布，整张 D 日 DA 计划合法）** | target，禁止进入当日输入 |

依据：DA 出清价于 D-1 固定时刻发布（审计文档 §1：NYISO 11:00 / ERCOT 13:30 / PJM 13:30；山东现货同此惯例）。**当 target=RT 时任务退化为「DA→RT 价差预测」，论文必须如此标注**（`public_da_rt_dataset_audit_v0.1.md` §1.2）。

⚠️ 现行 `load_shandong` 的「电价一律排除」会丢失这条合法路径；启用 Rich 分支前，loader 必须改成 target-aware 列角色映射，且为 `(shandong_DA, shandong_RT)` 各冻结一份 `feature_role_contract`。

---

## 2. 跨市场语义一致性

### 2.1 国外 price-only / 稀疏外生数据如何进入 optional branch

| 市场 | exog_fc | exog_act | optional 规模 | 进入方式 |
|---|---|---|---|---|
| NORD_DK1 | 0 | 0 | N=0（纯价格） | `optional_values=None` → `h_opt=0` → 精确退化为 HCH-Core |
| NEM_SA1 | 0 | 1（demand 实际） | N=1 OBSERVED_PAST | `act_demand_lag24` 作 token |
| LAGO_DE/PJM | 2（load fc, PV+Wind fc） | 0 | N=2 KNOWN_FUTURE | `fc_*` 作 token |
| 山东 DA/RT | 10 | 10 | N=10 KNOWN_FUTURE + N=10 OBSERVED_PAST（+ RT 条件的前日电价） | 见 §1 |

**learned-null 语义（对齐架构 §4.1、主计划 §11.5、架构 §9 不变式 #6）：**
- `OptionalCovariateEncoder` 采用 **valid-count 加权 mean-pooling + 零初始化残差**（`hch_v2_context.py` L176-199）。mask=0 的 token 在 `e = e * mask` 后对 pooled 贡献为 0；若全部 token 缺失，`pooled = 0 / 1 = 0`，`out(0)=0`（零初始化），故 **全部缺失 ⇒ `h_opt ≡ 0` ⇒ `h_final = h_core`**（初始化与训练后均由不变式 #6 约束为「向 HCH-Core 退化」而非「任意 learned-null 行为」）。
- **建议项（设计契约，非当前实现）：** 为每个角色增加一个显式 `NULL` token（value=0, role=r, mask=1），让分支学习「该角色缺失」的条件向量，而不是只会靠 mask 归零。NULL token 在零初始化下同样从恒等出发。

### 2.2 维度一致性

- **d_value（每 token 特征维）：** `OptionalCovariateEncoder.value_proj = nn.Linear(d_value, d_model)`（L172）。所有 token 必须共享同一 `d_value`。最小一致选择 `d_value=1`（每列归一化后的标量值 + role embedding 区分语义）；若需要扩展（如携带 availability/derivative），必须对所有市场统一，不能某市场用 1、另一市场用 3。
- **N（token 数）可变：** 架构按 `[B, H, N, d_value]` + `masks[B,H,N]` 用 **对称 mean-pooling** 聚合，天然对 N 可变、token 顺序排列不变。各市场需对齐到**全局统一 max_N**（按角色分槽位 padding），缺失槽位 mask=0。契约要求：**维度只由 (d_value, max_N) 决定，语义只由 (role, mask) 决定，绝不依赖列名或槽位序号**。
- **CALENDAR/STATIC 跨市场一致：** CALENDAR 由时刻派生（所有市场都有），STATIC 由 market meta 派生（所有市场都有），因此这两类 token 在所有市场维度恒等；差异只在 KNOWN_FUTURE / OBSERVED_PAST 的数量。

### 2.3 山东私有列名清单 + 禁止硬编码声明

以下 22 个中文列名是山东 PMOS 现货平台私有命名（**禁止硬编码为公开数据集必需输入**）：

```
时刻, 日前电价, 实时电价,
地方电厂总加预测值, 联络线受电负荷预测值, 风电总加预测值, 光伏总加预测值,
核电总加预测值, 自备机组总加预测值, 试验机组总加预测值, 直调负荷预测值,
竞价空间预测值, 新能源总加预测值,
地方电厂总加实际值, 联络线受电负荷实际值, 风电总加实际值, 光伏总加实际值,
核电总加实际值, 自备机组总加实际值, 试验机组总加实际值, 直调负荷实际值,
竞价空间实际值, 新能源总加实际值
```

契约规则（对应架构 §4.2、§9 不变式 #2/#5 与提示词 §6「禁止把山东私有列名硬编码成公开数据集的必需输入」）：

1. **通用模块只消费角色化 token**（KNOWN_FUTURE / OBSERVED_PAST / CALENDAR / STATIC），列名在数据适配器边界被抹去，永不进入网络。
2. 山东私有列名**不得**成为 `required`。任何市场缺任何列，走 `mask=0 + learned-null`。
3. 派生/冗余列（新能源总加、竞价空间）在适配器标注 `derived_from`；若某市场无对应量，直接缺省，不允许为对齐维度而伪造零值列。
4. `market_id / target_id` 只作审计元数据，**不作预测 token**（架构 §3.5、`iah_candidate.py` L18-19）。
5. 适配器为每个 `(market, target)` 输出一份冻结的 `feature_role_contract`（列名 → role → 归一化 id → 截断时点），作为 data manifest 的一部分。

---

## 3. 契约草案（代码块）

```text
required:
  - host forecast            # [B,H,1] 冻结宿主对目标小时的预测（→ z0, 每日常量 s_d）
  - time index               # 小时/星期/月（→ CALENDAR; 由时刻派生, 恒可用）
  - scale-free history       # u: 局部 S1 连续秩(z0); lag_sf: z0_lag24/48/168 + 双曲残差(zY-z0)_lag24 + availability
                             # 仅使用 ≤ D-1 的价格/残差历史（PRICE_LAGS=(24,48,72,168), lag≥24h）

optional: typed exogenous tokens   # [B,H,N,d_value], d_value 跨市场统一(建议 1)
  - KNOWN_FUTURE             # D-1 已发布的预测（山东 10 预测值列; LAGO load/PV+Wind fc; target=RT 时含 日前电价）
  - OBSERVED_PAST            # 运行后实现量, 必须滞后 ≥24h（山东 10 实际值列; NEM demand act）
  - CALENDAR                 # 恒已知; 节假日需外部日历, 缺失必须显式声明
  - STATIC                   # 市场元数据(currency/tier/负价下限), 非逐行特征
  - OTHER                    # 保留位, 仅当能给出与上面四类之一等价的信息边界

missing optional: learned-null token   # mask=0; 建议每角色显式 NULL token
  # 不变式: 全部缺失 ⇒ h_opt ≡ 0 ⇒ HCH-Rich ≡ HCH-Core（零初始化残差保证）
  # 不变量: 维度只由 (d_value, max_N) 决定; 语义只由 (role, mask) 决定; 列名绝不进入网络

normalization:
  - 只用训练段统计量          # 每市场每列冻结归一化 profile(median/MAD 或训练段分位数), 见 §5
  - 拒绝用 S3/S4 统计量       # 任何在 S3/S4 上计算的统计量不得烘焙进 profile/模型

target:
  - 永不进入当日输入          # y_t 及任何其函数不得进入特征矩阵(L101 教训, common.py L4-14)
  - 只经滞后历史进入          # lag_sf 中的 (zY-z0)_lag24 是 ≤ D-1 的合法实现价
  - target 角色映射必须按 (market,target) 绑定   # 日前电价: DA-target 时禁止 / RT-target 时 KNOWN_FUTURE

modality dropout（建议项）:
  - U2 训练时按语义组随机丢弃（如整组丢弃 WIND_FC / LOAD_FC / 全部 optional）
  - 防止分支依赖单一市场特征; 对应架构 §9 不变式 #2/#6 与主计划 §11.4
```

---

## 4. HCH-Core vs HCH-Rich 两个正式实验条件论证

### 4.1 为什么拆成两个正式条件

1. **因果隔离（最主要的理由）。** 只拆成两个条件才能回答「Rich 分支的增益到底来自外生变量本身，还是来自 HCH 的修正机制」。外生预测（风电/负荷/日前计划）对任何宿主预测器都有信息量；若只有 HCH-Rich 一个条件，增益无法归因。`HCH-Rich − HCH-Core`（同一批域、同一核心、仅差 optional 分支）是唯一干净的边际估计。
2. **守住已成立的主张边界。** 当前 R1B 主张「跨异构 schema 的 model-agnostic 修正」建立在 `D_VALUE=0`、12 个 source domain 上（feature schema audit §4）。若把 rich covariate 设为必需输入，DK1 这类纯价格市场会被排除，universal 主张（架构 §9 不变式 #4「price-only 是 first-class 输入」）会崩。两个条件让「核心跨市场」与「外生增强」分层主张，互不污染。
3. **可复现性。** HCH-Core 在公开 price-only 基准（LAGO / GEFCom）上完全可复现；HCH-Rich 依赖私有山东文件，无法再分发（审计 §5.2：Lago 之外的中国市场数据为私有）。分离后，论文主 claim 仍可复现，Rich 作为补充业务验证。
4. **论文叙事与架构一致性。** 架构 §1「rich covariates may improve HCH, but their absence must not make the module structurally incomplete」、§4.4「HCH-Rich at init ≈ HCH-Core」、主计划 §11「U2-Rich 是 U1 稳定之后的独立阶段」——两个条件正是这些原则的实验化。
5. **防止 overclaim。** 当前 `D_VALUE=0`，任何「HCH 已利用山东丰富特征」的表述都是假的。两个正式条件使「启用」与「未启用」成为显式、可证伪的实验变量。

### 4.2 每个条件需要的实验组/对照组

**条件 A — HCH-Core（`D_VALUE=0`，= 现行主实验）**

| 组 | 说明 |
|---|---|
| 处理组 | `LearnedSig` UniversalCore + 冻结宿主 + 本地 S1/CAGM/DVG 链（= 现行 B5/B6 管线） |
| 对照 1 | 宿主 identity（B0） |
| 对照 2 | 公开 peer 基线：B1 ResidualL1 / B3 δ-Adapter / B4 PIR（与 paper gate 同口径） |
| 对照 3 | `PlainCore`（无 signature，`iah_candidate.py` 旁路）——分离 signature 贡献 |
| 对照 4 | `Local-Core`（每域独立训练同结构）——universal sharing 上界 |
| 网格 | 12 source domain × 4 host + DK1 unseen + LOHO-host；seed 0/1/2；S4 永不参与选择 |

**条件 B — HCH-Rich（`D_VALUE>0`，U2 阶段，核心冻结）**

| 组 | 说明 |
|---|---|
| 处理组 | 冻结 A 条件核心 + `OptionalCovariateEncoder`（U2 训练，仅 IAH-CRPS）+ 同一本地链；含 modality dropout |
| 对照 1（隔离分支） | 同域、同核心的 **HCH-Core**（optional 关闭）——边际 = B − 对照 1 |
| 对照 2（超出宿主已见信息） | **Host+rich**：宿主已用 `build_tabular` 的 `fc_*/act_*` 特征训练的预测器。关键对照：若 HCH-Rich 不优于它，则外生信息已被宿主吸收，分支无边际价值 |
| 对照 3（缺失退化的不变式） | **All-missing**：HCH-Rich 全 token mask=0，必须复现 HCH-Core（架构 §9 不变式 #6） |
| 对照 4（单特征依赖） | **逐语义组 dropout**：整组丢 WIND_FC / LOAD_FC / SOLAR_FC / 全部，验证不依赖单一市场特征 |
| 域 | source rich 域：LAGO_DE（2 fc）、LAGO_PJM（2 fc）、NEM_SA1（1 act）；**Shandong DA/RT 保持 held-out**（主计划 §16 山东最后做业务验证），作为 primary rich 验证 |

**跨条件一致性要求：** 数据切分、宿主、seed、训练预算完全相同；U2 前后必须断言核心参数 hash 不变（主计划 P1-4）；Shandong 在两个条件中都只作评估，不作训练。

---

## 5. 归一化边界（防 S4 泄漏）

契约的原则：**凡被 S3/S4 复用的统计量，一律只能在训练段计算并冻结。**

| 统计量 | 计算段（唯一允许） | 备注 |
|---|---|---|
| 宿主每日尺度 `s_d = mean(\|host_raw\|)`（`iah_candidate.py` L62-79） | 当日 host 预测（pre-outcome） | 合法：仅用当日冻结宿主输出，非 target |
| Core 输入 `z0 = asinh(host/s_d)` | 同上 | 同上 |
| S1 连续秩 `u`（`S1RankReference`） | S1R（10%）host z0 | 已由 R1B 冻结 |
| DataSignature 确定性描述子（8 维，`compute_domain_descriptors`） | S1R host z0 | 已冻结（`hch_v2_context.py` L28-59） |
| **可选协变量归一化 profile（新增）**：每列 `(μ_c, σ_c)`（建议 median/MAD 或训练段分位数） | **S1∪S2T（或 S1R∪S2T）有效行** | **必须冻结进 local/domain package**，训练/推理用同一 profile，并写入 manifest（含 hash） |
| `lag_sf` 中的双曲残差 `(zY−z0)_lag24` | 用 `y_full[lag24]`（≤ D-1 实现价） | 合法滞后；但若对残差做缩放，缩放统计量只能来自训练段 |
| target 相关任何量 | 当日 target 一律禁止 | `common.py` L4-14；S4 无 target 访问（`predict_s4` 不接受 target） |

**明确禁止：**
1. 用 S3/S4 计算协变量 μ/σ、MAD、min/max 或任何归一化常数；
2. 用 S4 做模态 dropout 强度或任何超参选择；
3. 用 S4 做 CALENDAR 外的「手工阈值」分组（提示词 §5.1 同规则）。
4. 现有 `load_shandong` 的 `ffill().bfill()`（`common.py` L296-298）会**抹掉缺失信号**——Rich 分支要求保留 missingness 为 mask，不得用前向/后向填充伪造观测。

---

## 6. 落地前置条件（给后续实现的最小清单）

1. `load_shandong` 改为 **target-aware 列角色映射**（§1.3），并输出冻结 `feature_role_contract`（含单位、derivation、缺失率）。
2. `DomainBatch` / `UniversalCoreTrainer` 增加 `optional_values/roles/masks` 通道（当前不存在，`universal_trainer.py` L36-47）。
3. 新增 U2 训练器：冻结核心、断言 core hash 不变、仅 IAH-CRPS、modality dropout（主计划 §11/P1-4）。
4. 为每个 rich 域冻结协变量归一化 profile（§5）并入 manifest。
5. 建议每角色显式 `NULL` token（§2.1）；保持零初始化残差不变式。
6. 论文措辞：HCH-Core（price-only, universal）与 HCH-Rich（+optional exogenous）为两个显式条件；target=RT 时日前价作 KNOWN_FUTURE 必须标注为「DA→RT 价差预测」。
