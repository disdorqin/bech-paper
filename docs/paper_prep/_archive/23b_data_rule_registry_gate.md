# 23b · 数据规则注册表审计（P0-C/D）

> 执行：opencode-b2 | 任务单 t819e2cb3 | 2026-08-07
> 目标：逐数据集核对标签语义/规则边界，输出 registry schema、无法证明的字段、停止条件

---

## 一、审计结论

**山东 da_cq_price/rt_cq_price 无法证明是 market clearing price；-100/1500 口径与官方 -0.08/1.3 元/kWh 不一致；NEM/Lago/UniElecPrice 规则时效需逐市场锁定。**

---

## 二、Registry Schema（可执行格式）

```yaml
schema_version: 1
datasets:
  - id: shandong_da
    market: 山东电力现货
    product: 日前电能量
    label_field: da_cq_price
    label_semantics: ???
    unit: 元/MWh
    granularity: 1h (96点/日)
    time_range: [2022-01-01, 2026-07-18]
    floor_cap: ???
    rule_version: ???
    clip_status: ???
    negative_price_pct: 11.1%
    
  - id: shandong_rt
    market: 山东电力现货
    product: 实时电能量
    label_field: rt_cq_price
    label_semantics: ???
    unit: 元/MWh
    granularity: 1h
    time_range: [2022-01-01, 2026-07-18]
    floor_cap: ???
    rule_version: ???
    clip_status: ???
    negative_price_pct: 13.4%
    
  - id: nem_sa1
    market: NEM (South Australia)
    product: 日前电能量
    label_field: price
    label_semantics: settlement price (AUD/MWh)
    unit: AUD/MWh
    granularity: 30min
    floor_cap: [-1000, 20300] (2025-07-01 ~ 2026-06-30)
    rule_version: AEMC 2026-27 update
    clip_status: ???
    negative_price_pct: 24.6%
    
  - id: lago_de
    market: Germany/Luxembourg (ENTSO-E)
    product: 日前电能量
    label_field: price
    label_semantics: day-ahead auction price (EUR/MWh)
    unit: EUR/MWh
    granularity: 1h
    floor_cap: [-500, 4000] (ACER 2026)
    rule_version: ACER Decision 02/2026
    clip_status: ???
    negative_price_pct: 6.0%
    
  - id: unielecprice
    market: 40国
    product: 日前电能量
    label_field: price
    label_semantics: varies by country
    unit: varies (EUR/MWh, AUD/MWh, etc.)
    granularity: varies
    floor_cap: varies
    rule_version: varies
    clip_status: ???
```

---

## 三、逐字段审计

### 3.1 山东 da_cq_price / rt_cq_price

| 字段 | 状态 | 证据 |
|------|------|------|
| **label_semantics** | ⚠️ **未证实** | da_cq_price/rt_cq_price 是否 = market clearing price？07 审计文档未明确说明；22 号审稿指出"在证明它等于受相同 floor/cap 约束的市场级清算标签以前，不能把山东写成 C1 的规则证明数据集" |
| **unit** | ✅ 元/MWh | 07 审计确认 |
| **granularity** | ✅ 1h | 07 审计确认 |
| **floor_cap** | ❌ **口径不一致** | 07 审计显示 min=-100, max=1500；但山东官方 2023 信息写申报价格下限 -0.08 元/kWh = -80 元/MWh、上限 1.3 元/kWh = 1300 元/MWh。-100 与 -80 不一致，1500 与 1300 不一致 |
| **rule_version** | ⚠️ **未锁定** | 22 号审稿："山东官方 2023 信息写的是申报价格下限 -0.08 元/kWh、上限 1.3 元/kWh，并说明可适时调整" |
| **clip_status** | ⚠️ **未说明** | 数据是否被预先截断？未披露 |

**关键问题**：
- da_cq_price 中的 "cq" = ？（不清楚缩写含义）
- 是否 = market clearing price（市场出清价）？还是 unit transaction price（机组成交价）？
- -100/1500 口径 vs 官方 -80/1300 口径：哪个是真实的行政边界？

### 3.2 NEM (South Australia)

| 字段 | 状态 | 证据 |
|------|------|------|
| **label_semantics** | ✅ settlement price | AEMO 标准 |
| **floor_cap** | ✅ [-1000, 20300] | AEMC 2026-27 update |
| **rule_version** | ✅ 已锁定 | 2025-07-01 ~ 2026-06-30 |
| **clip_status** | ⚠️ 未说明 | 数据是否被预先截断？ |

### 3.3 Lago DE/LU

| 字段 | 状态 | 证据 |
|------|------|------|
| **label_semantics** | ✅ day-ahead auction price | ENTSO-E 标准 |
| **floor_cap** | ✅ [-500, 4000] | ACER Decision 02/2026 |
| **rule_version** | ✅ 已锁定 | ACER 2026 |
| **clip_status** | ⚠️ 未说明 | 数据是否被预先截断？ |

### 3.4 UniElecPrice

| 字段 | 状态 | 证据 |
|------|------|------|
| **label_semantics** | ⚠️ varies by country | 需逐市场确认 |
| **unit** | ⚠️ varies | 需逐市场确认 |
| **floor_cap** | ⚠️ varies | 需逐市场确认 |

---

## 四、无法证明的字段（停止条件触发器）

| # | 字段 | 数据集 | 问题 | 严重性 |
|---|------|--------|------|--------|
| 1 | label_semantics | 山东 | da_cq_price 是否 = market clearing price？ | 🔴 致命 |
| 2 | floor_cap 口径 | 山东 | -100/1500 vs 官方 -80/1300 | 🔴 致命 |
| 3 | rule_version | 山东 | 规则版本未锁定，"可适时调整" | 🔴 致命 |
| 4 | clip_status | 全部 | 数据是否被预先截断未说明 | 🟠 高 |
| 5 | label_semantics | UniElecPrice | 逐市场确认 | 🟡 中 |

---

## 五、停止条件（任一成立即停止论文主实验）

1. ✅ 山东 da_cq_price 不是 market clearing price → **停止 C1 实验**
2. ✅ 山东 floor/cap 口径错误 → **停止 C1 实验**
3. ✅ 规则边界用错年份 → **停止所有行政边界实验**
4. ⚠️ 负价被数据提供方预先截断 → **停止所有负价实验**
5. ⚠️ 跨市场单位混用 → **停止跨市场比较**

---

## 六、给 codex 的建议

1. **冻结山东标签语义**：确认 da_cq_price/rt_cq_price 的定义（market clearing price? unit transaction price?）
2. **锁定山东 floor/cap**：确认 -80/-100、1300/1500 的官方来源，按规则版本锁定
3. **补全 clip_status**：所有数据集需说明是否被预先截断
4. **UniElecPrice 逐市场确认**：40 国的 unit/label/floor_cap 需逐市场核实
5. **若山东标签无法证明**：C1 实验线必须删除，论文不能用山东作为行政边界规则的证明数据集
