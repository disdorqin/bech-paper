# R1B §19 — DA/RT 公开数据审计 v0.1

**日期：2026-08-13**
**审计人：Claude（R1B sprint §19 自动审计）**
**方法：桌面审计（desktop audit）。仅用官方/一手来源（市场运营商官网、官方数据门户、官方 API 文档），经 WebSearch/WebFetch 核实；未下载多年原始数据，存储量为估算。**
**范围：NYISO、ERCOT、PJM 三个市场的官方日前（DA）与实时（RT/平衡）电价公开数据。**

---

## 0. 结论速览（TL;DR）

| 市场 | DA 目标 | RT 实现 | 建议角色 |
|---|---|---|---|
| **NYISO** | 11 个 zone 的小时 LBMP（2001/2005–今） | 5 分钟 zone LBMP | **R1C primary** |
| **ERCOT** | 10 负荷区 + 4 hub 的小时 SPP（2010-12–今） | 15 分钟 SPP | **R1C secondary** |
| **PJM** | hub/zone/节点小时 LMP（2008 前后–今，DM2 窗口受限） | 5 分钟 LMP（2018-04–今） | **reference only** |

**一句话建议：用户回来后优先接入 NYISO 的「DA 小时 zonal LBMP（target）+ RT 5 分钟 zonal LBMP（小时化，realization）」对。**
理由详见 §6：NYISO 历史最长且格式稳定（~2005 起）、11 个 zone 标识符稳定、DA 结果固定于 D-1 11:00 发布（截断边界最干净）、公开 CSV 无需注册即可匿名批量下载、负价真实存在；短板是原始数据再分发许可不明确（附录只能放派生统计）。ERCOT 许可最宽松（原始数据可进附录/复现仓库）、DA/RT 对干净（2010-12 节点制以来），列为 secondary。PJM 因 DM2 再分发禁令 + 归档窗口限制 + 5 分钟数据仅自 2018-04-01，列为 reference only。

---

## 1. 背景：预报截断（forecast truncation）的信息学含义

本审计服务于 R1B/R1C：给 HCH 校正层新增「DA/RT 成对」市场，论文的实证需要明确**哪个价格作为「真实值」最干净、且不产生前瞻泄漏（lookahead leakage）**。

三市场的共同市场结构：

- **日前市场（DA）**：在运行日 D 之前一天（D-1）以拍卖方式出清，出清结果（DA 价格）在 D-1 的**固定时刻**公开发布：
  - NYISO：DAM 于 D-1 05:00 截止，**结果于 D-1 11:00 前发布**（NYISO tariff §4.2.5）。
  - PJM：竞价于 D-1 10:30（新手册为 11:00）截止，**结果于 D-1 13:30 前发布**。
  - ERCOT：竞价于 D-1 10:00 截止，**结果于 D-1 13:30 前发布**（Nodal Protocols §4.5.3）。
- **实时市场（RT）**：在运行日 D 之内按 5/15 分钟实时出清，价格在对应时刻之后才实现；最终结算/修正数据在运行日之后发布。

**对「真实值」选择的含义：**

1. **用 DA 价格作 target（最干净）**：DA 价格是 D-1 已知的前向市场出清价。只要特征集只使用「≤ 预报发布时刻」的信息，且预报发布时刻不早于 DA 结果时刻，则 target 本身在预报时刻已定、不存在前瞻泄漏。这是「日前电价预测」论文的标准设定，也与仓库现有 DA-target 数据（LAGO_DE / LAGO_PJM / NORD_DK1）一致。
2. **用 RT 价格作 target（次干净）**：RT 价格在 D-1 不可知、运行日后才实现，天然满足「真实值晚于预报时刻」的无泄漏要求。但此时 **DA 价格整张 D 日计划表变成合法的 KNOWN_FUTURE 特征**（因为 D-1 已发布），任务退化为「DA→RT 价差预测」；且 RT 序列噪声更大（5 分钟尖峰、稀缺定价）。
3. **必须避免的泄漏**：无论用哪种 target，`t` 时刻的特征不得包含任何 `> t` 时刻的 RT 信息（RT 只能用作 target）；DA 价格是否可用取决于预报发布时刻与 DA 结果时刻的相对先后。若预报在 D-1 11:00（NYISO）之后发布，则 D 日整日 DA 价格可作特征；若在之前发布，则 D 日 DA 价格不可见、只能用 D-1 及以前的 DA 价格（即 `price_lag24/48/168` 之类的滞后特征）。

**本审计建议 HCH-R1C 采用：DA 价格为 target（主），RT 价格为 realization 分析（次）。** 理由：① 三市场 DA 都有固定发布时刻，截断边界可写进论文；② DA 价格与现有数据集口径一致（EUR/USD DA 价），便于跨市场宏平均；③ RT 只做 DA–RT 价差/鲁棒性分析，避免把 RT 噪声引入主 target。

---

## 2. NYISO

### 2.1 官方 DA 价格来源
- **MIS 公共档案（主源）**：`mis.nyiso.com/public`，DA zonal LBMP 日 CSV：`https://mis.nyiso.com/public/csv/damlbmp/YYYYMMDDdamlbmp_zone.csv`；月 ZIP：`https://mis.nyiso.com/public/csv/damlbmp/YYYYMM01damlbmp_zone_csv.zip`；索引页：`http://mis.nyiso.com/public/P-2Alist.htm`。
- 人类门户：`https://www.nyiso.com/energy-market-operational-data`。
- OASIS（FERC Order 889 传输透明度 + 市场结果发布）：`https://www.nyiso.com/oasis`；对象存储 `oasis-postings.nyiso.com`（历史可回溯至 1999）。
- 参考：`https://www.nyiso.com/markets`

### 2.2 官方 RT 价格来源
- MIS 公共档案：RT zonal LBMP 日 CSV：`https://mis.nyiso.com/public/csv/realtime/YYYYMMDDrealtime_zone.csv`；索引页：`http://mis.nyiso.com/public/P-24Alist.htm`。
- 字段含 LBMP total + marginal loss + congestion component（$/MWh）。

### 2.3 时间粒度
- **DA：小时**（24 点/日）。
- **RT：5 分钟**（288 点/日）。

### 2.4 档案历史
- 名义上可回溯至 **~2001**，格式在 **~2005** 前后变更（RT 与 DA 都变）；当前 `damlbmp_zone.csv` / `realtime_zone.csv` 的日文件约定稳定可用（gridstatus NYISO PR 备注；MATLAB 论坛有 2005-02 起 `pal` 月 ZIP 实证）。
- 结论：**稳定格式 ≈ 2005–今**；名义起点 ~2001。

### 2.5 zone/hub 标识符
- 11 个内部负荷区 A–K（官方区名/CSV 名）：A WEST、B GENESE、C CENTRAL、D NORTH、E MHK_VL（Mohawk Valley）、F CAPITL、G HUD_VL（Hudson Valley）、H MILLWD、I DUNWOD、J N.Y.C.、K LONGIL。
- 另有外部接口（H_Q、NPX、O_H、PJM 等）与 generator 节点（PTID）。

### 2.6 发布/可用时间线
- DA：D-1 05:00 截止，**11:00 前发布** D 日全部 24 小时 LBMP（tariff §4.2.5；NYISO 培训材料「DAM closes 05:00 — Results posted 11:00」）。
- RT：实时发布；日 CSV 在运行日之后可下载。
- 数据文件以**东部（America/New_York）日期**为键；UTC 侧 20:00–24:00 取日期会错位（第三方客户端踩坑点）。

### 2.7 预报截断含义
- DA LBMP 于 D-1 11:00 发布 → 以 DA 为 target 时截断边界最干净（详见 §1）。
- RT 5 分钟价仅用于 realization/DA–RT 价差。

### 2.8 负价情况
- 存在负价，集中在上州风电场密集区（Zone A West、Zone D North 等）；LBMP CSV 内含负值（$0 以下）。未见官方统一统计页码被本次检索核实，论文引用时建议直接对下载数据统计（如负价小时占比、最负价）。

### 2.9 下载机制
- **匿名 HTTP 文件档案**：日 CSV + 月 ZIP，无需账号/API key/许可点击（`mis.nyiso.com/public`）。
- OASIS 提供市场结果发布；`api.nyiso.com` 下的 Finance/Metering REST API 仅限市场参与者（MIS 账号 + NAESB 证书），公开数据不走此通道。

### 2.10 再分发/许可
- MIS 公开档案**无许可点击门**、可匿名访问。
- 但 NYISO 官网 Terms of Use 声明：访问网站**不授予任何许可/所有权**，NYISO 保留全部权利（catalyst-cooperative 许可讨论引用的条款原文）。
- **结论：原始 CSV 全量重发布进论文附录风险较高；派生统计/图/表 + 引注属通行做法。** 建议只发布派生统计。

### 2.11 DST 处理
- 东部时间（EST/EDT）；文件按东部运行日日期键控。
- 春令时（3 月）日有 **23 小时**，秋令时（11 月）日有 **25 小时**，RT 5 分钟序列在切换时连续。
- 小时化聚合时需按东部本地时钟重建 24h/25h 序列（与 LAGO/Nord Pool 的排除日期机制一致）。

### 2.12 小时化后存储量估算
- 取 11 zone × 2 市场（DA+RT）。DA 小时 96,360 行/年；RT 5 分钟 1,154,880 行/年，小时化后 96,360 行/年。
- CSV 行 ~75 B（含 LBMP+loss+congestion 三列）：DA ≈ 7 MB/年；RT 5 分钟原始 ≈ 87 MB/年；**仅保留小时化 total LBMP 双市场 ≈ 10 MB/年**；保留三成分 ≈ 30 MB/年。

### 2.13 建议角色
**R1C primary** — 见 §6。

---

## 3. ERCOT

### 3.1 官方 DA 价格来源
- **Market Prices 页（主入口）**：`https://www.ercot.com/mktinfo/prices`
- **DAM Settlement Point Prices（全结算点）**：Data Product **NP4-190-CD**，zip/csv/xml：`https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD`
- **历史 DAM 负荷区/hub 价格**：Data Product **NP4-180-ER**（年度 ZIP，含 2010-12-01 起全部修正）。
- Market Information 总入口：`https://www.ercot.com/mktinfo`

### 3.2 官方 RT 价格来源
- **Settlement Point Prices at Resource Nodes, Hubs and Load Zones**：Data Product **np6-905-cd**（由 SCED LMP 每 15 分钟生成），zip/csv/xml：`https://www.ercot.com/mp/data-products/data-product-details?id=np6-905-cd`
- **历史 RTM 负荷区/hub 价格**：Data Product **NP6-785-ER**（zip/xlsx，首次发布 2011-09-30，周更新，按周日晚生成并纳入价格修正）：`https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER`

### 3.3 时间粒度
- **DA：小时**（24 HE/日）。
- **RT：15 分钟**（96 区间/日，SPP 由 SCED LMP 计算）。

### 3.4 档案历史
- 节点制市场 2010-12-01 上线 → **SPP 历史自 2010-12-01 起**；历史年度 ZIP 覆盖 2010 至今，且 2012 年曾重跑纳入全部修正。
- 结论：**2010-12–今**（约 16 年）。

### 3.5 zone/hub 标识符
- 负荷区（Load Zone）：`LZ_WEST`、`LZ_NORTH`、`LZ_SOUTH`、`LZ_HOUSTON`、`LZ_RAILN`、`LZ_RARL`、`LZ_LCRA`、`LZ_AEN`、`LZ_CPS`、`LZ_NRG`。
- 交易 hub（Trading Hub）：`HB_NORTH`、`HB_HOUSTON`、`HB_SOUTH`、`HB_WEST`。
- 另有单个资源节点（resource node）与电力母线。

### 3.6 发布/可用时间线
- DA：竞价 D-1 10:00 截止，**结果不晚于 D-1 13:30 CT 发布**（Nodal Protocols §4.5.3：发布 hourly DASPP per settlement point、hourly LMP per bus 等）；后续活动窗口至 14:30。
- RT：实时发布；历史价格报告每周日晚上更新（含价格修正记录）。

### 3.7 预报截断含义
- DA SPP 于 D-1 13:30 发布 → 以 DA 为 target 时边界干净（同 §1）。
- RT 15 分钟 SPP 为 realization；DA→RT 价差是常见研究设定。

### 3.8 负价情况
- 西区（West / 西德州风电）显著：**2011 年西区 819 小时负价**（Potomac Economics ERCOT State of the Market）；西德州风电节点 2011 年 **RT >15% 小时、DA >6% 小时为负**（LBNL 分析），CREZ 投运后回落至 <2%，随风电增长又回升。
- 市场对 price-taking 资源报价视为 **-$250/MWh**（Potomac SOM 2018）。
- 结论：**负价真实且集中在 West zone/风电节点**；hub 级负价远少于节点级。

### 3.9 下载机制
- **MIS 公开报告**（数据产品页 zip/csv/xml；`ercot.com/mktinfo/prices` 年度 ZIP）。
- **API Explorer**（`apiexplorer.ercot.com`）：免费注册拿 subscription key 后可程序化拉取（第三方 SDK `ercot` / `gridstatus` 均封装此通道）。
- 报告工具单 tab 上限 65,000 行，大月拆多 tab。

### 3.10 再分发/许可
- **ERCOT Terms of Use（2023-07-20 更新）明确允许再分发**：公开部分内容可复制/再分发（不改动并保留声明）；**原始数据「可无需保留声明」地用于编译、图表、分析**：`https://www.ercot.com/help/terms`
- **结论：三市场中许可最宽松，原始数据可进论文附录/复现仓库（建议仍保留来源声明）。**

### 3.11 DST 处理
- **Central Prevailing Time（CPT）**，时间戳 CPT-aware（CDT = UTC-5，CST = UTC-6）。
- 春令时（3 月）：跳过 HE 03:00 → DA **23** 个 HE、RT **92** 个 15 分钟区间；秋令时（11 月）：HE 02:00 重复 → DA **25** 个 HE、RT **100** 个区间。ERCOT 会标记重复区间。
- 曾有 DST 过渡期展示 bug（Market Notice M-A030625-01），处理 25h 日时需显式去重。

### 3.12 小时化后存储量估算
- 取 10 负荷区 + 4 hub = 14 结算点 × 2 市场。DA 小时 122,640 行/年；RT 15 分钟 490,560 行/年，小时化后 122,640 行/年。
- CSV 行 ~55 B：DA ≈ 7 MB/年；RT 15 分钟原始 ≈ 27 MB/年；**仅小时化双市场 ≈ 14 MB/年**。

### 3.13 建议角色
**R1C secondary** — 见 §6。

---

## 4. PJM

### 4.1 官方 DA 价格来源
- **Data Miner 2（主源）**：`https://www.pjm.com/markets-and-operations/etools/data-miner-2`，feed **`da_hrl_lmps`**（Day-Ahead Hourly LMP，全部母线含聚合）。
- OASIS：`https://www.pjm.com/markets-and-operations/etools/oasis`
- 历史数据检索指南（权威口径）：`https://www.pjm.com/-/media/etools/data-miner-2/data-miner-2-historic-data-guide.ashx?la=en`

### 4.2 官方 RT 价格来源
- Data Miner 2 feeds：**`rt_hrl_lmps`**（Real-Time Hourly LMP）、**`rt_fivemin_hrl_lmps` / `rt_hrl_interval_lmps`**（Real-Time 5-Minute LMP）。
- 5 分钟结算自 **2018-04-01** 起（5-minute settlement 实施）。

### 4.3 时间粒度
- **DA：小时**。
- **RT：5 分钟**（另有小时平均 feed）。

### 4.4 档案历史
- Data Miner 2 于 **2017-08-15** 上线（替代 Data Miner 1），上线时载入「全部可得历史」；但 **5 分钟 RT 仅自 2018-04-01**。
- **归档窗口（archive cutoff，滚动）**：RT 5 分钟 = 186 天（~6 个月）；RT 小时与 DA 小时 = 731 天（~2 年）。更早数据进入「historic archive」，查询受限（同日历年内、仅 date/type/row_is_current_version_nbr 过滤、无排序）。
- 结论：DM2 内可靠程序化访问 ≈ 滚动 2 年窗口 + 受限的历史档；PJM 历史 LMP 名义上可回溯至 1998（论文引用需走 OASIS/日报告而非 DM2）。

### 4.5 zone/hub 标识符
- **Hub**：AEP、AEP Dayton、COMED、Dominion（DOM）、Northern Illinois（NI）、Western Hub 等。
- **Zone（约 20 个）**：AEP、ATSI、COMED、DAYTON、DEOK、DOM、DUQ、EKPC、JC、ME、PE、PENELEC、PJM、PS、PL、RECO 等。
- **节点**：pricing node（pnode）上万，含 LMP type = pnode / agg / zone / hub。

### 4.6 发布/可用时间线
- DA：竞价 D-1 10:30（新手册 11:00）截止，**结果于 D-1 13:30 发布**（Manual 11 时间线；含小时计划、LMP、负荷预测等）；其后 14:15 重报价截止、RAC 约 15:00。
- RT：实时发布；DM2 文件运行日后可拉取。

### 4.7 预报截断含义
- DA LMP 于 D-1 13:30 发布 → 以 DA 为 target 时边界干净（同 §1）。
- RT 5 分钟仅作 realization；DA–RT 价差需注意 RT 5 分钟→小时化窗口。

### 4.8 负价情况
- 存在负价，集中在西部风电/核电区（AEP、COMED 等）与核电机组 during 低负荷时段；IMM（Potomac Economics）State of the Market 报告长期记录负价小时。建议论文直接对下载数据统计（本审计未检索到单一官方汇总 URL，引用时以自行统计为准）。

### 4.9 下载机制
- **Data Miner 2 UI**：免费注册，单次导出 ≤ 1,000,000 行。
- **API**：`dataminer2.pjm.com`，单次 ≤ 50,000 行；归档数据跨年度拆分查询。
- OASIS 可查历史 LMP（XML/CSV）。

### 4.10 再分发/许可
- **Data Miner 数据「仅供内部使用，未获 PJM 会员资格禁止再分发」**；非会员禁止 republish；PJM 拥有数据编译版权（Data License Agreement：`https://www.pjm.com/-/media/etools/edatafeed/data-license-agreement-edata-feed-data-miner-2.ashx`）。
- **结论：原始数据禁止重发布（含论文附录）。派生统计/图 + 引注是唯一安全路径；如需附录原始数据，必须申请 PJM（Associate）会员或改走 ERCOT/NYISO。**

### 4.11 DST 处理
- DM2 时间戳**双列**：`datetime_beginning_utc` 与 `datetime_beginning_ept`（Eastern Prevailing Time）。
- 归档查询按 UTC 且同日历年；小时化时建议以 EPT 列重建 23/24/25 小时日。

### 4.12 小时化后存储量估算
- 只取 hub（5 个）+ 少量 zone：DA 小时 43,800 行/年；RT 5 分钟 525,600 行/年（5 hub），小时化后 43,800 行/年。
- CSV 行 ~60 B：DA ≈ 3 MB/年；RT 5 分钟原始（5 hub）≈ 32 MB/年；**仅小时化双市场 ≈ 5 MB/年**。
- ⚠️ 若取全部 pnode 的 RT 5 分钟，数据量达数十 GB/年，**不可行**；必须按 hub/zone 子集。

### 4.13 建议角色
**reference only** — 见 §6。

---

## 5. 对比表

| 维度 | NYISO | ERCOT | PJM |
|---|---|---|---|
| DA 官方来源 | `mis.nyiso.com/public/csv/damlbmp/`（日 CSV/月 ZIP）；OASIS | `ercot.com/mktinfo/prices` + Data Product NP4-190-CD / NP4-180-ER | Data Miner 2 feed `da_hrl_lmps`；OASIS |
| RT 官方来源 | `mis.nyiso.com/public/csv/realtime/` | Data Product np6-905-cd / NP6-785-ER | DM2 feed `rt_hrl_lmps` / `rt_fivemin_hrl_lmps` |
| DA 粒度 | 1 小时 | 1 小时 | 1 小时 |
| RT 粒度 | 5 分钟 | 15 分钟 | 5 分钟 |
| 档案历史 | ~2001 起，稳定格式 ~2005–今 | 2010-12-01（节点制）–今 | DA/RT 小时可回溯较久，但 DM2 归档窗口 2 年；RT 5 分钟仅 2018-04–今 |
| 区域标识 | 11 个 zone（A–K，WEST…LONGIL） | 10 个 LZ + 4 个 HB（LZ_WEST、HB_HOUSTON…） | hub（AEP/COMED/DOM/NI…）+ ~20 zone + 上万 pnode |
| DA 发布时点 | D-1 05:00 截止，11:00 发布 | D-1 10:00 截止，13:30 发布 | D-1 10:30 截止，13:30 发布 |
| RT 可用性 | 实时；日 CSV 次日 | 实时；历史周更新 | 实时；DM2 次日 |
| 下载机制 | 匿名 HTTP 文件（日/月批量），无 key | MIS 报告 zip/csv/xml + API Explorer（免费 key） | DM2 UI（1M 行）/ API（50k 行）；归档分年 |
| 再分发许可 | Terms 不授予许可，原始数据重发布风险高；派生统计 OK | **最宽松**：原始数据可用于编译/图表/分析，无需保留声明 | **最严**：非会员禁止再分发；仅派生统计 |
| DST | 东部时间；23/25 小时日 | CPT；跳/重复 HE，23/25 DA、92/100 RT | UTC + EPT 双时间戳；归档按 UTC 年 |
| 小时化存储（估算） | ~10 MB/年（双市场 total LBMP） | ~14 MB/年（双市场，14 结算点） | ~5 MB/年（5 hub 双市场）；全 pnode 不可行 |
| 负价 | 有（上州风电区） | 有（West 区显著，2011 年 819h） | 有（AEP/COMED 等西部区） |
| 建议角色 | **R1C primary** | **R1C secondary** | **reference only** |

---

## 6. 建议：用户回来后优先接入哪个 DA/RT 对

**推荐接入顺序：NYISO（primary）→ ERCOT（secondary）→ PJM（reference only）。**

### 为什么 NYISO 排第一
1. **截断边界最干净**：DA 结果固定 D-1 11:00 发布（tariff §4.2.5），论文可写出精确的信息边界；DA 小时 + RT 5 分钟天然成对。
2. **历史最长且稳定**：~2005 年至今稳定格式（名义 2001 起），比 ERCOT 的 2010-12 起点多约 5 年样本，便于长序列/跨年泛化实验。
3. **标识符稳定**：11 个 zone（A–K）自市场成立基本不变，跨年无「换区」风险。
4. **下载零门槛**：匿名 HTTP 批量 CSV/月 ZIP，无 key、无归档窗口限制（对比 PJM 的 2 年滚动窗口），最适合作业复现。
5. **负价真实存在**，能与现有 NEM_SA1（26% 负价）形成中等负价率区间。
6. **不足与对策**：原始数据重发布许可不明确 → 论文只发布派生统计 + 引注；如需附录原始数据，改用 ERCOT。

### ERCOT 作为 secondary 的理由
- **许可最宽松**：原始数据可进论文附录/复现仓库（Terms of Use 明文允许），对「可复现论文」的审稿诉求最有利。
- DA/RT 对干净（2010-12–今，DA 小时 + RT 15 分钟）。
- **不足**：历史短于 NYISO；CPT DST 需要 23/25 DA、92/100 RT 区间特判；RT 是 15 分钟而非 5 分钟。

### PJM 降为 reference only 的理由
- 仓库已有 LAGO_PJM（COMED zone DA）数据，**PJM DA 侧已覆盖**；缺的是 RT。
- **DM2 再分发禁令**（非会员禁止 republish）直接挡住「原始数据进附录」；归档窗口（DA/RT 小时 731 天、RT 5 分钟 186 天）使长历史程序化访问受限；RT 5 分钟仅 2018-04 起。
- 定位：用现有 LAGO_PJM DA 数据保持多市场证据，RT 仅作内部鲁棒性检查，不进论文附录。

### 落地建议（R1C 最小集）
1. 主 target：**NYISO Zone J（N.Y.C.）+ Zone A（WEST）DA 小时 LBMP**（两个代表性 zone：负荷中心 vs 风电/负价区），历史 2005–今。
2. realization：对应 zone 的 **RT 5 分钟 → 小时平均**（按东部本地时钟 23/24/25 小时日重建）。
3. 特征侧：DA 价格按「预报发布时刻」规则滞后（`price_lag24/48/168`），RT 特征一律滞后 24h 以上；ERTCOT 作为第二对（LZ_HOUSTON / HB_HOUSTON）验证跨市场迁移。
4. 论文数据声明：ERCOT 原始数据可附；NYISO/PJM 只附派生统计 + 官方数据门户引注。

---

## 7. 来源清单

**NYISO**
- MIS 公开档案（DA/RT 日 CSV、月 ZIP、索引）：`http://mis.nyiso.com/public/`；`http://mis.nyiso.com/public/P-2Alist.htm`；`http://mis.nyiso.com/public/P-24Alist.htm`
- 能源市场运行数据门户：`https://www.nyiso.com/energy-market-operational-data`
- OASIS：`https://www.nyiso.com/oasis`；对象存储 `https://oasis-postings.nyiso.com`
- 市场页面：`https://www.nyiso.com/markets`
- DA 11:00 发布（tariff §4.2.5/培训材料）：`https://www.nyiso.com/documents/20142/3625950/mpug.pdf`；`https://www.nyiso.com/documents/20142/3037451/2-Energy-Marketplace.pdf`
- Zone A–K 官方区名：NYISO 图表/数据页（zone selector）`https://www.nyiso.com/single-chart-page`
- 历史起点 ~2001/格式 ~2005：gridstatus NYISO PR `https://github.com/gridstatus/gridstatus/pull/55`
- Terms of Use 引用与许可讨论：`https://github.com/orgs/catalyst-cooperative/discussions/1750`

**ERCOT**
- Market Prices：`https://www.ercot.com/mktinfo/prices`
- Market Information：`https://www.ercot.com/mktinfo`
- DAM SPP：`https://www.ercot.com/mp/data-products/data-product-details?id=NP4-190-CD`
- RTM SPP：`https://www.ercot.com/mp/data-products/data-product-details?id=np6-905-cd`
- 历史 DAM LZ/Hub：`https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER`
- 历史 RTM LZ/Hub：`https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER`
- Terms of Use：`https://www.ercot.com/help/terms`
- DAM 13:30 发布（Nodal Protocols §4.5.3）：`https://www.ercot.com/files/docs/2020/09/30/October_1__2020_Nodal_Protocols.pdf`
- DST 过渡与展示 bug：`https://www.ercot.com/services/comm/mkt_notices/M-A030625-01`
- 负价统计（2011 西区 819h / -$250 下限）：Potomac Economics 2018 ERCOT State of the Market `https://potomaceconomics.com/wp-content/uploads/2019/06/2018-State-of-the-Market-Report.pdf`；LBNL 风电负价分析 `https://eta-publications.lbl.gov/sites/default/files/lbnl_-_wind_and_solar_impacts_on_wholesale_prices_approved.pdf`

**PJM**
- Data Miner 2：`https://www.pjm.com/markets-and-operations/etools/data-miner-2`
- Data Miner 2 历史数据检索指南（归档窗口/时间戳/查询限制）：`https://www.pjm.com/-/media/etools/data-miner-2/data-miner-2-historic-data-guide.ashx?la=en`
- Data License Agreement（eDataFeed / Data Miner 2，再分发禁令）：`https://www.pjm.com/-/media/etools/edatafeed/data-license-agreement-edata-feed-data-miner-2.ashx`
- OASIS：`https://www.pjm.com/markets-and-operations/etools/oasis`
- DA 竞价/结果时间线（Manual 11）：`https://www.pjm.com/-/media/DotCom/documents/manuals/archive/m11/m11v93-energy-and-ancillary-services-market-operations-04-01-2018.ashx`
- 5 分钟结算 2018-04-01：`https://ftp.pjm.com/-/media/committees-groups/forums/tech-change/20180417/20180417-item-03a-initiative-roadmap-5-minute-settlements.ashx`

---

## 8. 已知局限（audit caveats）
- 本审计为桌面审计，未下载/实测文件；存储量是行数 × 行宽的估算值。
- NYISO/PJM 负价的「官方统一统计 URL」未能在本次检索中锁定，论文引用前应先用下载数据自行统计。
- PJM DM2 历史最早年份（2010 前）无法仅凭检索确认；如需 2010 前 PJM DA 数据需走 OASIS/日报告人工核对。
- 各市场 Terms of Use 以官网最新版为准（本审计引用的 NYISO/PJM 条款来自许可讨论转引，正式引用前请打开官网原文复核）。
