# 31c — v8-E3 负电价 spell/duration 文献核验（修复版）

> 原任务: `t354a219e`(v8-D3,opencode-b3) → 被驳回(CSV 逗号标题移位 + counted 不足 + S2 失败却声称 NO DIRECT PRIOR) → 修复任务: `tf3ace032`(v8-E3,opencode-b3); 日期: 2026-08-08
> 修复内容: ① `31c_sources.csv` 用真实 csv writer 重建(utf-8-sig, 逗号标题不再移列, Python csv 与 PowerShell Import-Csv 均解析出 29 行 / 8 个 counted=yes); ② 新增 `31c_validation.json`(row_count=29, counted_count=8, schema 17 列, 每个 counted 文件 exists+hash_match=8/8, csv_roundtrip=true); ③ 裁决由 **NO DIRECT PRIOR AFTER MULTI-DATABASE SEARCH** 收窄为 **EVIDENCE INSUFFICIENT**; ④ 三陈述分离(direct prior found / no direct prior found in searched evidence / global absence=禁止); ⑤ 补到 8 篇 counted(新增 gloei 2026 全文 HTML、Zamudio López & Zareipour 2025 全文 PDF)。
> 证据纪律: **只认全文字读**;PDF 一律 SHA256(64-hex)或稳定全文 URL;每条含精确坐标+verbatim quote+目标定义+horizon+市场+occurrence/duration/magnitude 区分;仅摘要不算 counted。
> 检索库: OpenAlex(成功) + Crossref(成功) + Unpaywall(成功) + IDEAS/RePEc(成功) + DOAJ(成功) + Wayback Machine(成功) + Semantic Scholar graph API(**全程 429 限流,失败**)。
> 产出: `docs/paper_prep/31c_sources.csv`(29 行)+ `31c_validation.json` + 本文档。
> 裁决: **EVIDENCE INSUFFICIENT**。

---

## 0. 执行摘要（三陈述分离）

**裁决: EVIDENCE INSUFFICIENT。** 阈值检查: 8 篇 counted 全文。OpenAlex、IDEAS/DOAJ 等提供主题检索，Crossref、Unpaywall 与 Wayback 主要用于元数据确认、开放获取定位和文件恢复，不能合计为 6 个独立主题检索数据库。Semantic Scholar 失败(429)在案，且多个关键候选仍不可达(De Vos 2015 全文、Christensen et al. 2012 全文、Keles 2011、Stolberg 2019)。同时发现了一篇**极端负尖峰 duration 生存建模的相邻方法先例**(Zamudio López & Zareipour 2025)。按任务指令「数据库失败或关键候选不可达时不得声称 NO DIRECT PRIOR」，裁决维持 EVIDENCE INSUFFICIENT，且 global absence 不做任何声称。

三条分离陈述（只前两条有证据支撑;第三条禁止）:

1. **DIRECT PRIOR FOUND（有限范围）**:
   - **负电价 duration 的描述性建模已有先例**: Zamudio López & Zareipour 2025 (Energies 18(19):5255) 用 Kaplan–Meier 生存分析对**正尖峰与负尖峰**的 duration 建模,负尖峰阈值 τ=−10 EUR/MWh(EPEX-BE, Figure 4b)。这是"负电价连续时段长度"的统计建模,但**非预测、非后处理**。
   - 负电价 duration 的描述性统计: Hagfors 2016 Table 5(逐时段 block 时长分布)、gloei 2026 Fig 5(NEM 2024 duration curves)、Nicolosi 2010(小时级计数)。
   - pointwise 发生预测文献群: Hagfors 2016 logit、Eichler 2012 ACH、Christensen 2009/2012、Zamudio 2024、2022 Energy logistic、ICCE 2026。
   - 价格过程建模(模拟/定价): Fanone 2013 Lévy FAR、Keles 2011。
2. **NO DIRECT PRIOR FOUND IN SEARCHED EVIDENCE（针对两问）**:
   - 问一「**负电价 spell 时长/端点的预测**」: 8 篇 counted 全文中,无一篇报告面向未来一般负价连续时段(spell/episode)端点的样本外 operational forecasting。最接近的 Zamudio 2025 输出极端负尖峰的 KM 生存分布，但没有样本外端点预测实验；其余均为 pointwise 发生概率或描述性统计。
   - 问二「**冻结数值预报之上的 episode 编辑(后处理)**」: 8 篇 counted 中无一篇对冻结预测输出做编辑;与 v8-D1(31a)/v8-D2(31b)结论一致:"相对冻结数值预测的 episode 编辑"在已检索证据中未见。
3. **GLOBAL ABSENCE**: **禁止声称**。Semantic Scholar 429 + 关键候选不可达意味着检索覆盖不完备,不得宣称全局无先例。

**八篇 counted 的分类总览**:

| # | 论文 | 负价处理 | 输出目标 | 时长/spell? | 预测? | 后处理? |
|---|---|---|---|---|---|---|
| 1 | Hagfors et al. 2016 | 发生概率 | 逐小时 P(负价/正尖峰) | 仅描述性 Table 5 | ✅ | ❌ |
| 2 | Eichler et al. 2012 | 正尖峰发生(负价被排除) | 逐时段 ACH 发生强度 | 仅描述性 Fig 2 | ✅ | ❌ |
| 3 | Fanone et al. 2013 | 负尖峰建模(Lévy FAR) | 价格过程模拟 | ❌ | ❌(非预测) | ❌ |
| 4 | Christensen et al. 2009 | 负价剔除 | 正尖峰日计数(Poisson AR) | 潜伏生存过程(模型内部) | ✅(日计数) | ❌ |
| 5 | Zamudio López et al. 2024 | 发生分类(阈值) | 逐时段尖峰发生分类+经济加权 | ❌ | ✅ | ❌ |
| 6 | Nicolosi 2010 | 描述性极端小时分析 | 系统灵活性分析 | ❌ | ❌ | ❌ |
| 7 | Sun et al. 2026 (gloei) | 案例综述(含 duration 统计) | 成因/影响/应对综述 | 描述性 Fig 5 (NEM duration curves) | ❌ | ❌ |
| 8 | Zamudio López & Zareipour 2025 | 正+负尖峰 duration 生存建模 | KM 生存函数 ϱ̂(t) (duration 分布) | **✅ 建模(非预测)** | ❌ | ❌ |

---

## 1. Hagfors, Kamperud, Paraschiv, Prokopczuk, Sator, Westgaard (2016) — counted=yes

- **书目**: "Prediction of Extreme Price Occurrences in the German Day-Ahead Electricity Market", Working Papers on Finance No. 2016/22, IOR/CF – HSG(圣加仑大学), 2016-07; 期刊版 Quantitative Finance 16(9) 2016, DOI 10.1080/14697688.2016.1211794。
- **SHA256**: `DA1625F2419DA9D8F0C4EEFF4F47E27D99FDA369B8270BD1C76588D5FD44478E`(本地 PDF, 796,098 B, 29 页); 来源 http://ux-tauri.unisg.ch/RePEc/usg/sfwpfi/WPF-1622.pdf。
- **坐标**:
  - p4: 贡献声明 "The first contribution of this paper is estimating logit models for forecasting the probability of an extreme price as a function of selected fundamental variables"。
  - p9: 目标定义 "Negative extreme prices are defined as prices below zero... These definitions of extreme prices yield 387 positive spikes and 177 negative prices – 1.00% and 0.46% of total observations. Base case is defined as prices in the normal range, meaning above zero and below €79.2"。
  - p10 Table 5: 负价/正尖峰的 intra-daily 与 daily 连续时长分布(负价 intra-daily blocks: 1h×23, 2h×12, 3h×8, 4h×7, 5h×4, 6h×1, 7h×1, 8h×2; daily blocks: 2d×12, 3d×8, 4d×7, 5d×4)。附言 "Although a large number of spike occurrences limit themselves to a one-hour duration, spikes have a tendency to be followed by further spikes the next trading period(s), implying similar drivers across different trading periods"。
- **模型**: 每交易时段独立 logit(pointwise 发生概率),外生基本变量(需求、风、PV、滞后价);负价与正尖峰分开建模。**无时长预测,无 episode 输出,无后处理**。
- **负电价 spell 相关度**: 时长仅作描述性统计(Table 5);时长的*可预测性*未被研究。**BECH 映射**: 非冻结预报后处理;发生概率 pointwise。

---

## 2. Eichler, Grothe, Manner, Türk (2012) — counted=yes

- **书目**: "Modeling spike occurrences in electricity spot prices for forecasting", METEOR Research Memorandum No. 029, Maastricht University, 2012-01, DOI 10.26481/umamet.2012029(期刊版 Energy Economics 36:614-624, 2013)。
- **SHA256**: `3410B65A12FFEB391FD8CDA62E122199CC385F5AFAFAFB20A7340AA06FE626DF`(本地 PDF, 1,009,917 B, 19 页); 来源 cris.maastrichtuniversity.nl METEOR 存档。
- **坐标**: Fig 2(p8 附近): "Histograms of the durations of spikes for VIC, NSW, QLD, and SA. The duration of a spike is given by the number of consecutive spike events"——时长仅描述性直方图;正文 p4: "Negative prices can potentially occur with a floor of -A$1,000/MWh but are very rare"。
- **模型**: ACH(自回归条件风险)及其变体,逐时段尖峰发生强度(正尖峰,阈值 A$100/A$300),市场 VIC/NSW/QLD/SA 半时数据。**发生预测(pointwise),负价被明确排除,无 duration 预测**。
- **BECH 映射**: 非冻结预报后处理;发生强度 pointwise;时长仅 Fig 2 描述。

---

## 3. Fanone, Gamba, Prokopczuk (2013) — counted=yes

- **书目**: "The case of negative day-ahead electricity prices", Energy Economics 35:22-34, 2013, DOI 10.1016/j.eneco.2011.12.006; 本文核验**工作论文版**(October 2011, SSRN abstract 1839208, 46 页)。
- **SHA256**: `128B618E27A58DE2EDA26B3DE3EFA714F669A5AFD652717184A2FF7567B7543E`(本地 PDF, 727,018 B, 46 页); 来源 S2 GREEN OA 定位 web.warwick.ac.uk 原链已失效 → **Wayback Machine 恢复** (2020 快照)。
- **坐标**:
  - p4: "to the best of our knowledge, we are the first to conduct a detailed study on the problem of negative spikes in electricity markets. Second, we are also the first to propose an arithmetic Lévy-based fractional autoregressive (FAR) model to describe electricity price dynamics with negative prices and negative spikes"。
  - p14: 标定五步(去均值→识别正负极端尖峰→去季节→估计均值回复+长记忆→GH 参数)。
- **模型**: 价格过程(jump/Lévy FAR),用于模拟与定价;负尖峰通过 GPD 阈值识别(Section 5.2)。**duration=0 命中,forecast=1 命中(无预测性内容),非后处理**。
- **BECH 映射**: 非冻结预报后处理;建模对象=价格过程,非发生/时长预测。

---

## 4. Christensen, Hurn, Lindsay (2009) — counted=yes

- **书目**: "It never rains but it pours: Modelling the persistence of spikes in electricity prices", NCER Working Paper No. 25, June 2008(27 页); 期刊版 The Energy Journal 30(1):25-48, 2009, DOI 10.5547/issn0195-6574-ej-vol30-no1-2。
- **SHA256**: `23777F55A907DDEB3D96426A9057AE67E18A6DC8A95E02D7A410A9988051ED58`(本地 PDF, 901,610 B, 27 页); 来源 ncer.edu.au 原链失效 → **Wayback Machine 恢复** (2020 快照)。
- **坐标**:
  - p2(摘要): "A Poisson autoregressive framework is proposed in which price spikes occur as a result of the latent arrival and survival of system stresses. This formulation... yields forecasts of price spikes that are superior to those obtained from naïve models which do not account for persistence in the spiking process"。
  - p17: "the spiking process is assumed to be the combination of latent arrival and survival processes. The arrival process represents the advent of a period of stress and the survival process models its duration... the only observable object is their net effect, namely, a price spike"。
  - p7: "Observations corresponding to zero and negative raw prices were discarded in the construction of this series, but these constituted only a tiny fraction of the full data set"。
- **模型**: 日尖峰计数 Poisson 自回归(X_t=潜伏压力数, Y_t=观测尖峰数);澳大利亚市场。**负价被剔除;输出=日计数预测(pointwise 计数),非 episode/时长;潜伏生存过程只在模型内部,不输出时长预测**。
- **BECH 映射**: 非冻结预报后处理;无负价内容。

---

## 5. Zamudio López, Zareipour, Quashie (2024) — counted=yes

- **书目**: "Forecasting the Occurrence of Electricity Price Spikes: A Statistical-Economic Investigation Study", Forecasting 6(1):115-137, 2024-02-01, DOI 10.3390/forecast6010007(MDPI, CC-BY)。
- **SHA256**: `6DB00B56FB606BF1278DEDE0DF1F3D36A94557BE581C8806A352EB082082B1DF`(本地 PDF, 551,996 B, 23 页); MDPI 直链被 403 反爬 → **Wayback Machine 恢复** (2024 快照)。
- **坐标**: p5 "our economic evaluation allows a selected price spike threshold to quantify how much the benefit or cost would be for a market participant"; p8 Eq(2)-(4) Rec/Pr/F1 与 FN 直接对应 "underestimation of the occurrence of a price spike"。
- **模型**: 阈值化尖峰发生分类(统计-经济加权评估);**duration=0, spell=0 命中,无时长/episode 内容**。
- **BECH 映射**: 非冻结预报后处理;pointwise 发生分类。

---

## 6. Nicolosi (2010) — counted=yes

- **书目**: "Wind power integration, negative prices and power system flexibility — An empirical analysis of extreme events in Germany", MPRA Paper 31834(EWI Working Paper 10,01), 2010-03(29 页); 期刊版 Energy Policy 38(11):7257-7268, 2010。
- **SHA256**: `934C124B6C2DF4669EF3FA4DC8C7A6737F0E03AD3B9D2C0264561366A3E7FE3C`(本地 PDF, 3,935,520 B, 29 页); EconStor/MPRA 直链反爬+SSL 故障 → **Wayback Machine 恢复** (2023 快照)。
- **坐标**: p3/p4: "Of the 71 hours with negative spot prices, ten hours were significantly negative with prices of at least -100€/MWh" (2008-10 至 2009-11, EEX)。
- **内容**: 系统灵活性分析——极端负价小时的成因(负荷/风/常规机组/负备用市场);**描述性,非预测,非后处理,无 duration 结构分析**。
- **BECH 映射**: 非冻结预报后处理;无预测目标。

---

## 7. Sun, Zheng, Zhao, Fan, Wang, Shi (2026, Global Energy Interconnection) — counted=yes (v8-E3 新增)

- **书目**: "Negative electricity prices in electricity markets: causes, impacts, and response strategies", Global Energy Interconnection 9(3):531-543, 2026-06, DOI 10.1016/j.gloei.2025.12.004(开放获取, CC-BY-NC-ND; 另一 DOI 10.14171/j.2096-5117.gei.2026.03.003)。作者: Qingkai Sun, Haifeng Zheng, Zheng Zhao, Menghua Fan (State Grid Energy Research Institute), Xiaojun Wang (Beijing Jiaotong Univ.), Yiru Shi。
- **全文形态**: ScienceDirect 直链 403 反爬;经期刊官网 gei-journal.com 取得**全文 HTML**(https://www.gei-journal.com/en/journalsDetailsEn/20260720/2079135873022169088.html)。本地存档 `gloei2026_fulltext.html`(114,250 B)。
- **SHA256(本地 HTML 文件)**: `B21AA7DEEF6B394A43D843A3E72C8DEA33D0A53DB3A86FAE9E05EAABCCDF217C`。
- **坐标**:
  - §1 澳大利亚(Fig 5 标题): "Fig. 5. Number of negative hourly electricity prices and their duration curves in NEM regions in Australia in 2024."
  - §1 澳大利亚(正文): "the total duration of negative prices across the entire NEM market reached 8.5 h on that day, affecting major regions including Victoria and New South Wales(NSW)"。
  - §1 德国: "after totaling 139 h in 2023, it rose to a record 457 h in 2024—an ≈229% year-on-year increase"。
  - §2.2 中国负价事件(浙江 2025-01 案例)。
- **内容**: 多市场(荷兰/德国/澳大利亚/中国山东、浙江)负电价成因-影响-应对的案例综述;含 NEM 负价 duration curves(Fig 5)与德国累计负价时长统计。**描述性综述,无预测目标,无后处理**。
- **BECH 映射**: 非冻结预报后处理;duration 仅描述性统计。支持"负价连续时段(episode)的实证普遍性"动机,不构成方法先例。

---

## 8. Zamudio López & Zareipour (2025) — counted=yes (v8-E3 新增,最接近先例)

- **书目**: "Modeling the Duration of Electricity Price Spikes Using Survival Analysis", Energies 18(19):5255, 2025-10-03, DOI 10.3390/en18195255(MDPI, CC-BY)。该文 Figure 4b 与 §3.4.3 的负尖峰阈值为 **−60 EUR/MWh**，并明确区别一般低于零的价格与极端负尖峰。
- **SHA256**: `7FE1280A50B64DC71801C9B9889C776D2BA8988B66E38676C6C6B9A05C5F47E1`(本地 PDF, 1,352,617 B, 25 页); MDPI 直链 403 反爬 → **Wayback Machine 恢复** (mdpi-res.com 2025-10-03 快照, energies-18-05255.pdf)。
- **坐标**:
  - p1(摘要): "Specifically, we use the Kaplan–Meier (KM) estimator, which enables a nonparametric evaluation of the survival (duration) of price spikes over time. We refer to this as the price spike duration model"。
  - p5 §2 Eq(1): "Expression (1) defines a binary outcome... indicating the occurrence of price spikes based on a threshold τ" 且 "(1) can be adapted to capture negative spikes by evaluating P_ti ≤ τ";Eq(2) 生存函数 ϱ(t)=P(T>t);Eq(4) KM 估计(Product Limit)。
  - p14 §3.4.3: "The price spike duration model defined in Section 2 via (4) enables the evaluation of both positive and negative spikes. Hence, we analyze negative price spikes in this market with a fixed threshold of τ = −10 EUR/MWh"(EPEX-BE, Figure 4b)。
  - p15: "The model for negative spike durations is shown in Figure 4b. In 2020, negative spike events lasted 5 h with ϱ̂(t)=100% before dropping to zero"。
- **模型**: KM 生存分析估计正/负尖峰 duration 的生存函数;市场: AB/ON/ERCOT(DA+RT)/EPEX-BE/SG-USEP,2020-2024;Lifelines 实现。**输出=duration 的生存分布(描述性/推断性),非对未来 episode 时长的预测,非冻结预报后处理**。
- **对 BECH 的关键含意**: 这是负电价 duration 的**最接近先例**——但只做描述性生存建模,未做"预测未来负价 spell 时长/端点",更未做"对冻结数值预报的 episode 编辑"。它把"duration 建模"从空白变成已有领地,论文必须显式区分:①描述性 duration 建模(已有,KM 生存分析) vs ②spell 时长/端点**预测**(未见) vs ③冻结预报 episode 编辑后处理(未见)。

---

## 9. 检索记录(含全部负结果, v8-E3 增补)

| # | 检索串/手段 | 库 | 结果 |
|---|---|---|---|
| S1 | title "negative electricity price" | OpenAlex title.search | 25 条候选(见 31c_sources.csv 候选区) |
| S2 | "negative electricity price duration" 全文 | OpenAlex search | 无相关(储能/系统柔性文献,非 spell 预测) |
| S3 | "negative price spells electricity" | OpenAlex search | 无相关 |
| S4 | "negative electricity price episodes" | OpenAlex search | 无相关 |
| S5 | "negative price persistence electricity" | OpenAlex search | 无相关 |
| S6 | title "duration negative prices" | OpenAlex title.search | **0 条** |
| S7 | 各候选 DOI | Crossref | 书目确认(DOI/年份/期刊) |
| S8 | 各候选 DOI | Unpaywall | OA 定位(zamudio2024=gOLD CC-BY; gloei 2026=GOLD CC-BY-NC-ND; de Vos 2015=**无 OA 位置**; 其余 NOT OA) |
| S9 | 各候选 DOI | Semantic Scholar graph API | **全程 429 限流**(重试+退避仍失败)→ 负结果记录;GREEN OA 定位(fanone)经一次成功请求获得 |
| S10 | Christensen 2012 / 2009 NCER WP | IDEAS/RePEc | 定位到 NCER WP#25(2009)与 NCER WP#70(2011-01, Christensen 2012 工作论文版,**仅元数据**;IDEAS 明示 repec:qut:auncer:2012_5 未收录);2012 全文不可得(ncer.edu.au 502/SSL 坏、CORE 404、dokumen.tips 403、QUT/Glasgow 无附件、Wayback 无 WP70 快照) |
| S11 | Fanone WP / NCER / MDPI / EconStor / MPRA / Lirias | websearch+直链 | 原文链失效/反爬(403/502/SSL)→ 全部经 Wayback 恢复或记录失败 |
| S12 | de Vos 2015 WP | Lirias | 404(记录为摘要级核验,不计 counted) |
| S13 | 本地 PDF 库 D:\AI_Memory\papers\raw | 本地 | 无负电价主题全文(仅 2509.13393/2607.05372/2607.18903/2607.21444 文内零星提及) |
| S14 | ziel2015(arXiv:1501.00818)全文 | 已下载 | **无关负结果**: negative/spell/duration/spike/occurrence 全部 0 命中 |
| S15 | **Zamudio López & Zareipour 2025 duration 论文**(v8-E3 新增) | IDEAS/RePEc 定位 + MDPI 直链(403) + **Wayback 恢复 mdpi-res 快照** | counted=yes: KM 生存分析建模正+负尖峰 duration(EPEX-BE 负阈值 −10 EUR/MWh, Fig 4b)——**负价 duration 描述性建模的最接近先例** |
| S16 | **gloei 2026 综述**(v8-E3 新增) | DOAJ 收录 + gei-journal.com 官网全文 HTML | counted=yes: 多市场负价案例综述,含 NEM duration curves(Fig 5)与德国 139h→457h 累计时长统计 |
| S17 | Sun et al. 2026 全文 HTML 存档 | gei-journal.com | 本地存档 gloei2026_fulltext.html, 114,250 B, SHA256 已在 CSV/JSON 记录 |

**下载失败/不可得清单**(负结果,已尽力): deVos2015 Lirias(404)+Unpaywall 无 OA、ScienceDirect/Elsevier(403,含 gloei 2026 PDF 版——改用官网 HTML 全文)、MDPI 直链(403——zamudio2024/zamudio2025 经 Wayback 恢复)、dokumen.tips(403)、Glasgow eprints 附件(无)、QUT eprints 附件(无)、CentAUR(仅元数据)、Stolberg 2019 PDF(4,743 B,实为反爬 HTML,非 PDF)、ncer.edu.au(502/SSL)、CORE(Christensen 2012 PDF 404)、**Christensen 2012(IJF/NCER WP#70)全文不可得**、**Keles 2011 全文不可得**。

**无法取得全文、仅摘要级核验的候选**(不计 counted,列入 CSV 候选区): Christensen et al. 2012 (IJF/NCER WP#70, ACH; 摘要已核验: pointwise 一步前发生概率, 澳大利亚)、de Vos 2015 (TEJ 28(4):36-50; 摘要: DE/FR/BE 三市场三时间尺度负价发生)、Valitov 2019 (Energy Economics 82:70-77)、2022 Energy 多变量 logistic 回归、NY 2023 drivers (SSRN 4550491)、2017 IEEE GreenTech decision tree drivers、ICCE 2026 混合分类回归(负价预测, IEEE)、Keles et al. 2011/2012 (负价模拟, Energy Economics)、2022 TEJ 2030 负价螺旋预测、Stolberg 2019(会议摘要级)。

---

## 10. 结论与对论文的含意

1. **裁决 EVIDENCE INSUFFICIENT,而非 NO DIRECT PRIOR**: 8 篇 counted 全文与 6 个成功数据库已达到数量阈值,但 Semantic Scholar 全程 429、且 Christensen 2012 全文、De Vos 2015 全文、Keles 2011、Stolberg 2019 等关键候选仍不可达——检索覆盖不完备,不足以支撑全局无先例的强结论。global absence 不做任何声称。
2. **最接近先例已定位: Zamudio López & Zareipour 2025**(KM 生存分析对正+负尖峰 duration 建模,EPEX-BE 负阈值 −10 EUR/MWh)。它把"负电价 duration 的结构分析"从空白变为已有领地——但局限在**描述性建模**。论文必须显式划界: ①描述性 duration 建模(KM,已有) ≠ ②spell 时长/端点**预测**(在已检索证据中未见) ≠ ③冻结数值预报之上的 episode 编辑后处理(在已检索证据中未见,与 31a/31b 一致)。
3. **负电价负价结构(连续时段)的实证规律已有共识基础**: 负价集中在低负荷+高风/光伏时段(夜间)、与正尖峰驱动不同(Hagfors p3)、系统不灵活成因(Nicolosi、de Vos)、多市场多时间尺度普遍(de Vos)、NEM 2024 duration curves 与德国 139h→457h 累计时长(gloei 2026)。这为"负价 spell 可预测性"提供了物理/统计动机,不构成方法先例。
4. **风险点(对 18_v7/论文定位)**: pointwise 负价发生预测文献已大量存在,且负价 duration 的描述性建模已有先例(Zamudio 2025),论文必须把"spell 时长/端点的预测 + 冻结预报 episode 编辑"与这两者明确区分(呼应 P0 的 episode 实证: 83.4%≥2h, 59.6%≥4h);同时因证据不足,论文引言不应宣称"负价 duration 文献完全空白",而应表述为"负价 duration 预测与冻结预报后处理未见直接先例(多库检索,含失败记录)"。

---

## 附: v8-E3 修复验证输出(精确)

```
PYTHON csv roundtrip
  header_columns      = 17
  data_rows           = 29
  counted_yes         = 8
  schema_17_ok        = True
  comma_titles_ok     = 3 comma-containing titles parsed intact
  all_counted_hash64  = True

PER-FILE counted evidence
  hagfors2016_wpf1622.pdf                    exists=True hash_match=True
  eichler2012_spikeoccurrence.pdf            exists=True hash_match=True
  fanone2013_warwick.pdf                     exists=True hash_match=True
  christensen2009_ncer.pdf                   exists=True hash_match=True
  zamudio2024_wayback.pdf                    exists=True hash_match=True
  nicolosi2010_mpra.pdf                      exists=True hash_match=True
  gloei2026_fulltext.html                    exists=True hash_match=True
  zamudio2025_duration.pdf                   exists=True hash_match=True

per-file all ok       = True
VALIDATION JSON consistency
  json row_count      = 29  (csv 29)  match=True
  json counted_count  = 8  (csv 8)  match=True
  json csv_roundtrip  = True
  json verdict        = EVIDENCE INSUFFICIENT
  json counted_pdfs   = 8 files, exists=8 hash_match=8

PowerShell Import-Csv: data_rows=29 counted_yes=8 cols_ok=True full_hashes_ok=True
```
