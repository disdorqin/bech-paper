# HCH-v2 U0 / HF 外部语料盘点清单 v0.1

**日期:** 2026-08-13
**对应规划:** `hch_v2_universal_training_master_plan_v0.1` §8.5(HF 数据阶梯)、§8.6(防泄漏);
`hch_v2_r1a7_prior_local_fusion_failure_audit_compute_plan_v0.1` §10(只读盘点任务)
**状态:** 只盘点、未下载大数据。所有条目均给出官方来源 URL。
**用途:** 为 U0 阶段(服务器端 MOMENT 表示蒸馏)挑选外部公开时序语料。

---

## 0. 盘点口径与判定规则

1. **只盘点不下载。** 所有大小/序列数均标注 `≈`(近似),以官方 HF card / 论文 / 源仓库为准;确切值以实际拉取时的 manifest 为准。
2. **建议角色**(每条目取一,来自 R1A.7 §10 的枚举):
   `representation distillation` / `host diversity` / `final benchmark` / `exclude`。
3. **老师(teacher)固定为 `AutonLab/MOMENT-1-small`。** 其官方预训练语料为 `AutonLab/Timeseries-PILE`,由 MOMENT 论文(arXiv:2402.03885)明示的 4 个组件构成:
   (a) **LTSF/Informer 长期基准**(ETTh/ETTm、**ECL/Electricity**、Exchange、Traffic、Weather、ILI);
   (b) **Monash 时序预测档案**(全集);
   (c) UCR/UEA 分类档案;
   (d) TSB-UAD 异常检测基准。
   **重叠判定只依据老师(而非 Chronos/Moirai)官方语料信息。** 因此:
   - 凡属于 Monash 组件或 LTSF 组件的 → 与 MOMENT 预训练重叠 = **明确重叠**;
   - LOTSA 单独**不在** Timeseries-PILE 组件内 → 与 MOMENT 重叠**不声称**(但注明对 Moirai/Chronos 的重叠,供参考);
   - 只有老师官方语料信息未提及的,标注 `不声称/证据不足`。
4. **上下文长度兼容性** 以老师 `seq_len = 512` 为基准(MOMENT-1-small `config.json` 的 `seq_len=512`,`patch_len=8`,约 37.9M 参数,MIT 许可)。能提供大量 ≥512 步窗口的序列才算"兼容长 context"。

---

## 1. LOTSA(Large-scale Open Time Series Archive)energy/electricity 子集

> 总览: `Salesforce/lotsa_data`(HF)。论文 Woo et al., *Unified Training of Universal Time Series Forecasting Transformers*, ICML 2024, arXiv:2402.02592;代码 `SalesforceAIResearch/uni2ts`。
> 全文约 **27B 观测、9 个领域**;仓库整体 **Apache-2.0**,体积约 **925 GB**(含全部领域;只按 config 拉取所需子集)。
> ⚠️ **重要口径**: uni2ts 论文明确将 **ECL/Electricity 与 Solar 从 LOTSA 预训练中剔除**,用作 **OOD 评估**(见 §5.4 处 Lago/ECL 条目)。
> 注: LOTSA 各 config 由 `huggingface-cli download Salesforce/lotsa_data --repo-type=dataset` 拉取(uni2ts README 流程)。

### 1.1 `australian_electricity_demand`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (config `australian_electricity_demand`);底层数据源自 AEMO,与 Monash 同名数据集同源
- **许可:** Apache-2.0(LOTSA 仓库)
- **频率:** 30 分钟(半小时)
- **序列数:** 5(VIC / NSW / QLD / TAS / SA 五个州的电力需求)
- **近似大小:** ≈ 4.62 MB(`dataset_info.json` 记录 `dataset_size=4,614,461` 字节)
- **上下文兼容性:** 每序列数年长、数万点,支持 512-step 窗口 ✅
- **建议角色:** `representation distillation`(U0-A sanity,主计划 §8.5 已点名)
- **与 FM 预训练重叠:** 该序列属于 Monash 组件的同名数据集 → 与 MOMENT 预训练**明确重叠(经由 Monash)**;同时也在 Moirai(LOTSA)预训练内。**作为蒸馏语料没问题;不得同时作为 final benchmark。**

### 1.2 `london_smart_meters_with_missing`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (config 名取自 TimesFM 2.0 的 LOTSA pretrain 清单);同源 Monash `london_smart_meters` / UK Power Networks Low Carbon London 试验
- **许可:** Apache-2.0(LOTSA 仓库)
- **频率:** 30 分钟
- **序列数:** ≈ 5,520(LOTSA 版;Monash 版 5,560)
- **近似大小:** 总观测 ≈ 166M(≈ 30–60 MB Arrow 级)
- **上下文兼容性:** 每序列约 2 年以上半小时值 → 数万点,支持 512-step 窗口 ✅
- **建议角色:** `representation distillation`(U0-B 服务器语料,家庭用电多序列多样性好)
- **与 FM 预训练重叠:** 属于 Monash 组件 → 与 MOMENT 预训练**明确重叠(经由 Monash)**;同时也在 Moirai 预训练内。与 final benchmark(如本地 ECL)不同源,泄漏风险低。

### 1.3 `wind_power`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (config `wind_power`);与 Monash `wind_4_seconds` 同源
- **许可:** Apache-2.0
- **频率:** 4 秒
- **序列数:** 1
- **近似大小:** 总观测 ≈ 7,397,147(单条超长序列,Arrow 约数十 MB)
- **上下文兼容性:** 700 万点单序列 → 极佳,支持 512-step 甚至更长窗口 ✅
- **建议角色:** `representation distillation`(U0-B;单条超长物理序列,利于学习极长程依赖)
- **与 FM 预训练重叠:** LOTSA 不在 MOMENT 组件内 → 对 MOMENT 重叠**不声称**;对 Moirai(自身即 LOTSA)明确重叠(参考)。

### 1.4 `solar_power`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (config `solar_power`);与 Monash `solar`(137 站点)同源
- **许可:** Apache-2.0
- **频率:** 4 秒
- **序列数:** 1
- **近似大小:** 总观测 ≈ 7,397,222
- **上下文兼容性:** 同上,超长单序列 ✅
- **建议角色:** `representation distillation`(U0-B)
- **与 FM 预训练重叠:** 对 MOMENT 重叠**不声称**;对 Moirai 明确重叠(参考)。

### 1.5 `wind_farms`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (config `wind_farms`);与 Monash `wind_farms_minutely` 同源
- **许可:** Apache-2.0
- **频率:** 1 分钟
- **序列数:** ≈ 337
- **近似大小:** 总观测 ≈ 172M(≈ 数百 MB Arrow)
- **上下文兼容性:** 每序列约 50 万点,支持 512-step 窗口 ✅
- **建议角色:** `representation distillation`(U0-B;风机多序列、分钟级)
- **与 FM 预训练重叠:** 对 MOMENT 重叠**不声称**;对 Moirai 明确重叠(参考)。

### 1.6 `residential_load_power` / `residential_pv_power`
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (在 TimesFM 2.0 的 LOTSA pretrain 清单中列出)
- **许可:** Apache-2.0
- **频率:** 未检索到官方明细(估计小时/分钟级);**此条目为低置信度**
- **序列数:** 未检索到
- **近似大小:** 未检索到
- **上下文兼容性:** 待确认(住宅负荷/光伏通常小时级、多年长 → 预计支持)
- **建议角色:** `host diversity`(若确认为负荷/光伏,可作为 U1-Energy+ 的 host 源)
- **与 FM 预训练重叠:** 对 MOMENT 重叠**不声称**;对 Moirai 明确重叠(参考)。

### 1.7 `buildings`(BuildingsBench)
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data ;源自 BuildingsBench(能源/建筑负荷基准)
- **许可:** Apache-2.0
- **频率:** 小时/15 分钟(含多种)
- **序列数:** 900k 级(buildings_900k 系列;按 config 划分)
- **近似大小:** 大型(数百 GB 级,仅流式/按需拉取)
- **上下文兼容性:** 建筑负荷多为多年小时级 → 支持 512-step 窗口 ✅(部分短序列除外)
- **建议角色:** `representation distillation`(U0-B,若需更大负荷多样性)/ 或 `exclude`(体积过大,优先级低)
- **与 FM 预训练重叠:** 对 MOMENT 重叠**不声称**;对 Moirai 明确重叠(参考)。

### 1.8 `cmip6_*`(气候模拟)与 `era5`(再分析)
- **来源:** https://huggingface.co/datasets/Salesforce/lotsa_data (多个 `cmip6_YYYY` config;`era5`)
- **许可:** Apache-2.0
- **频率:** 气候日/月尺度(era5 为再分析格点)
- **序列数:** 极多(格点 × 变量)
- **近似大小:** 合计占 LOTSA 绝大部分(数百 GB 级)
- **上下文兼容性:** 单格点序列极长 → 支持 ✅
- **建议角色:** `exclude`(非电力价格/负荷领域,体量过大;若 U0-C 需要极端规模再考虑流式抽样)
- **与 FM 预训练重叠:** 对 MOMENT 重叠**不声称**;对 Moirai 明确重叠(参考)。

---

## 2. Monash Time Series 里的 electricity 相关

> 总览: HF `Monash-University/monash_tsf`;官网 forecastingdata.org;论文 Godahewa et al., NeurIPS 2021 D&B, arXiv:2105.06643。
> **许可 CC BY 4.0**(档案级)。30 个数据集 / 58 个变体。原始 `.tsf` 打包约 577 MB;HF 仓库另有各 config 拆分。
> **这些数据集全部属于 MOMENT 预训练语料 Timeseries-PILE 的 Monash 组件 → 与老师预训练"明确重叠"。**

### 2.1 `electricity_hourly`(Monash 电力逐时)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf ;Zenodo 4656140;底层 UCI ElectricityLoadDiagrams 2011–2014(Lai et al. 2017)
- **许可:** CC BY 4.0
- **频率:** 1 小时
- **序列数:** 321
- **近似大小:** 321 × 26,304 点 ≈ 31 MB(HF card 口径;主计划 §8.5 引用);原始 TSF 变体更大
- **上下文兼容性:** 每序列约 26k 点 → 支持 512-step 窗口 ✅(也是 ECL 的逐时同源版,见 §5.4)
- **建议角色:** `representation distillation`(U0-A sanity,主计划 §8.5 已点名)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。⚠️ 与本地 final benchmark `ECL` 高度同源(ECL 即该数据集的逐时裁剪版),若 ECL 用于最终基准,蒸馏语料与基准存在**同源泄漏风险**,须在污染注册表标注。

### 2.2 `electricity_weekly`(Monash 电力周频)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf
- **许可:** CC BY 4.0
- **频率:** 1 周
- **序列数:** 321(同源客户,周聚合)
- **近似大小:** 小(每序列约 100–200 点)
- **上下文兼容性:** 每序列点过少(<512)→ **不支持** 512-step 蒸馏窗口
- **建议角色:** `exclude`(对 MOMENT 512 蒸馏无意义;若做短 horizon benchmark 另说)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。

### 2.3 `australian_electricity_demand`(Monash 澳洲电力需求)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf ;Zenodo 4659727;底层 AEMO(经 R `tsibbledata` 包)
- **许可:** CC BY 4.0
- **频率:** 30 分钟
- **序列数:** 5(VIC/NSW/QLD/TAS/SA)
- **近似大小:** 5 × 数万点 ≈ 数 MB
- **上下文兼容性:** 支持 512-step 窗口 ✅(该数据集官方 `prediction_length=336`,即 7 天)
- **建议角色:** `representation distillation`(与 LOTSA §1.1 为同一底层数据;二选一即可,推荐用 LOTSA 版因其 Apache-2.0 且 config 干净)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。⚠️ 澳洲需求与本地 final benchmark `AEMO/NEM 电价`是不同量(需求 vs 价格),但同市场;蒸馏语料含澳洲需求 → "澳洲域"表示已被老师见过,最终 NEM 电价基准需按"representation-pretrained transfer"而非"strict unseen"表述。

### 2.4 `london_smart_meters`(Monash 伦敦智能电表)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf ;Zenodo 4656072(含缺失)/ 4656091(无缺失)
- **许可:** CC BY 4.0(底层 UK Power Networks 数据 OGL)
- **频率:** 30 分钟
- **序列数:** 5,560
- **近似大小:** ≈ 166M 观测(原始 TSF 数百 MB)
- **上下文兼容性:** 支持 512-step 窗口 ✅(官方 `prediction_length=60`)
- **建议角色:** `representation distillation`(U0-B;家庭用电大规模多序列)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。

### 2.5 `solar_weekly` / `solar_10_minutes`(Monash 光伏)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf ;Zenodo 4656144(solar_10_minutes, `solar_10_minutes_dataset.zip` ≈ 4.6 MB)
- **许可:** CC BY 4.0
- **频率:** 周频(weekly)/ 10 分钟(10_minutes)
- **序列数:** 137(2006 年 Alabama 州 137 座 PV 电站,NREL 数据)
- **近似大小:** 10min 版 137 × 52,560 点(24.1 MB TSF);weekly 版小
- **上下文兼容性:** 10min 版支持 512-step ✅;weekly 版(<52 点/年)**不支持**
- **建议角色:** `representation distillation`(U0-B,10min 版)/ `exclude`(weekly 版)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。⚠️ 该 Solar 与本地 final benchmark `Solar-AL`(ts_benchmarks)为同一数据 → 若 Solar 用于最终基准,蒸馏与基准同源泄漏,须标注。

### 2.6 `wind_farms_minutely` / `wind_4_seconds`(Monash 风电)
- **来源:** https://huggingface.co/datasets/Monash-University/monash_tsf
- **许可:** CC BY 4.0
- **频率:** 1 分钟 / 4 秒
- **序列数:** 337(minutely)/ 1(4_seconds)
- **近似大小:** minutely ≈ 172M 观测;4_seconds ≈ 7.4M 观测
- **上下文兼容性:** 均支持 512-step ✅
- **建议角色:** `representation distillation`(U0-B;与 LOTSA §1.3/§1.5 同源,二选一即可)
- **与 FM 预训练重叠:** **明确重叠(Monash 组件)**。

---

## 3. 澳洲电力需求数据集(AEMO 相关公开版)

> 三层来源,按"可直接用作 U0 语料"程度排序:

### 3.1 LOTSA `australian_electricity_demand`(推荐入口)
- 见 §1.1。Apache-2.0,30 分钟,5 序列,≈4.62 MB。**推荐作为 U0 语料入口**(config 干净、许可宽松)。

### 3.2 Monash `australian_electricity_demand`
- 见 §2.3。CC BY 4.0,同源。与 3.1 二选一。

### 3.3 AEMO 官方数据与工具(作为原始源,不作为 U0 直接语料)
- **来源:** https://aemo.com.au/ (AEMO Data Portal / NEM 数据);R 包 `aemo`(CRAN),源码 https://github.com/charlescoverdale/aemo ;另见 R 包获取 5 分钟分区需求与 predispatch 预报
- **许可:** AEMO Copyright Permissions Notice(须标注 "Source: AEMO",**非 CC BY**)
- **频率:** 5 分钟原始 / 半小时平均需求(实际四秒区域运行需求半小时平均,GW)
- **序列数/大小:** 分区 × 多年(仓库本地已拉 `data/nem_aemo/`,300/300 月文件)
- **上下文兼容性:** 5min/30min 多年 → 支持 512-step ✅
- **建议角色:** `final benchmark`(仓库本地 AEMO/NEM 电价+需求已用作最终基准;**不要**再把 AEMO 需求拉进 U0 蒸馏语料,避免与 NEM 最终基准同市场泄漏)
- **与 FM 预训练重叠:** AEMO 需求本身不在 MOMENT 组件名下列举(但经 Monash 澳洲需求间接重叠);AEMO 原始拉取版**不声称**重叠。

---

## 4. 精选的 load / solar / wind 数据集

### 4.1 GEFCom2014 的 L / W / S 三赛道
- **来源:** Hong et al., *Probabilistic energy forecasting: GEFCom2014 and beyond*, IJF 2016(Elsevier 官方补充材料 + Tao Hong 官方分发,仓库本地 `data/gefcom2014/` 已含该 zip)
- **许可:** GEFCom2014 官方条款(Elsevier 分发;商业用途受限)
- **频率:** 小时(负荷 24 点/日;光伏仅 16 个昼间小时/日)
- **序列数:** L: 1 个区域(美国,约 2000 天负荷+25 个温度站点);W: 10 个澳洲风场(2012–2013,每场 16,800 点);S: 3 个澳洲光伏电站(约 720 天)
- **近似大小:** 小(单文件 MB 级)
- **上下文兼容性:** W/L 支持 512-step ✅;S 略短但可用 ⚠️
- **建议角色:** `host diversity`(U2-rich 外生协变量场景:负荷/温度/风资源)+ `final benchmark`(作为 EPF 领域对照)
- **与 FM 预训练重叠:** GEFCom 不在 MOMENT 组件名下列举 → **不声称**。(Chronos 曾用 GEFCom 作零样本评估,非预训练,故也不构成预训练重叠。)

### 4.2 Chronos `electricity_15min`
- **来源:** https://huggingface.co/datasets/autogluon/chronos_datasets (config `electricity_15min`);底层 UCI ElectricityLoadDiagrams 15 分钟版
- **许可:** 仓库级 "other"(config 级多为各数据集原始许可;ECL 家族为 CC BY 4.0 惯例)
- **频率:** 15 分钟
- **序列数:** ≈ 370(客户端)
- **近似大小:** 370 × 约 70k 点(数百 MB 级)
- **上下文兼容性:** 支持 512-step ✅
- **建议角色:** `representation distillation`(U0-B;与 §2.1 ECL 同源但更高频)
- **与 FM 预训练重叠:** 经 ECL/LTSF 组件与 MOMENT 预训练**明确重叠**(同底层 UCI 数据);同时被 Chronos 用于预训练/评估(参考)。

### 4.3 Chronos `ercot`
- **来源:** https://huggingface.co/datasets/autogluon/chronos_datasets (config `ercot`);源自 Kaggle ERCOT 德州负荷竞赛
- **许可:** 仓库级 "other";Kaggle 竞赛条款
- **频率:** 小时
- **序列数:** 数十条(commit 中可见 T88–T220 等 ID;按负荷类别拆分)
- **近似大小:** 数十 MB 级
- **上下文兼容性:** 多年小时级 → 支持 512-step ✅
- **建议角色:** `host diversity`(U1-Energy+ 负荷 host 源;如用于 U0-B 亦可作负荷多样性补充)
- **与 FM 预训练重叠:** ERCOT 不在 MOMENT 组件名下列举 → **不声称**(对 Chronos 明确,参考)。

### 4.4 Chronos `solar` / `solar_1h`
- **来源:** https://huggingface.co/datasets/autogluon/chronos_datasets (config `solar`, `solar_1h`);Solar-AL(137 座 Alabama PV,2006)
- **许可:** 仓库级 "other"(原始数据 NREL/CC BY 惯例)
- **频率:** 10 分钟(solar)/ 1 小时(solar_1h)
- **序列数:** 137
- **近似大小:** solar ≈ 137 × 52,560 点;solar_1h ≈ 137 × 8,760 点
- **上下文兼容性:** 均支持 512-step ✅(solar_1h 每序列约 8.7k 点)
- **建议角色:** `representation distillation`(U0-B,10min 版)
- **与 FM 预训练重叠:** 与 §2.5 Solar 同源(Monash 组件)→ 与 MOMENT **明确重叠**。⚠️ 与本地 final benchmark `Solar-AL` 同源,勿同时用于蒸馏与最终基准。

### 4.5 NREL 原始 PV/Wind 数据(父级源,非独立 U0 语料)
- **来源:** https://www.nrel.gov/grid/solar-power-data.html (Solar Power Data for Integration Studies)
- **许可:** NREL/DOE 公开条款
- **频率:** 5 分钟 / 小时
- **序列数:** 约 6,000 条模拟 PV 序列(2006,全国)
- **上下文兼容性:** 支持 512-step ✅
- **建议角色:** `exclude`(原始文件体积大、格式杂;直接使用其上层的 Solar-AL/Monash 精选版即可)
- **与 FM 预训练重叠:** **不声称**。

---

## 5. 仓库里已被引用的公开数据集(挑选关键项)

> 这些是仓库已在用的数据(`data/` 与 `docs/paper_research/03_..._数据集与对比对象核实.md`)。它们**不进入 U0 蒸馏语料**,但必须明确各自在总管线中的角色与泄漏风险。

### 5.1 Lago 5 市场(EPF 主基准)
- **来源:** Zenodo 4624804(Lago et al. 2021, *Applied Energy*)
- **许可:** 开放数据(CC BY 4.0 惯例)
- **频率:** 小时;NP/PJM/BE/FR/DE,各 52,416 点
- **建议角色:** `final benchmark`(EPF 主实验)。**排除出 U0 蒸馏语料。**
- **与 FM 预训练重叠:** **不声称**(Lago 不在 MOMENT/Chronos 语料名下列举;Chronos 用其评估非预训练)。

### 5.2 GEFCom2014-P(价格赛道)
- **来源:** Elsevier 补充材料 + Tao Hong 官方分发(本地已双源字节校验)
- **许可:** GEFCom2014 条款
- **频率:** 小时,25,968 点
- **建议角色:** `final benchmark`(经典竞赛对照,零负价)。**排除出 U0 蒸馏语料。**
- **与 FM 预训练重叠:** **不声称**。

### 5.3 AEMO / NEM 5 区现货价与需求
- **来源:** AEMO 官方端点(本地 `data/nem_aemo/`,300/300 月)
- **许可:** AEMO Copyright Permissions Notice
- **频率:** 5 分钟原始 → 小时聚合
- **建议角色:** `final benchmark`(当代极端市场主战场)。**排除出 U0 蒸馏语料**(见 §3.3)。
- **与 FM 预训练重叠:** 经 Monash 澳洲需求**间接**重叠(需求≠价格);以"representation-pretrained transfer"表述并记录。

### 5.4 ECL / Electricity(ts_benchmarks)
- **来源:** https://github.com/laiguokun/multivariate-time-series-data (LSTNet);本地 `data/ts_benchmarks/electricity.csv`,26304×321
- **许可:** 惯例 CC BY 4.0(原始 UCI ElectricityLoadDiagrams)
- **频率:** 小时
- **建议角色:** `final benchmark`(广义时序对照)。**排除出 U0 蒸馏语料。**
- **与 FM 预训练重叠:** **明确重叠** —— MOMENT 语料 LTSF 组件明示含 Electricity(Trindade 2015);与 §2.1 Monash electricity_hourly 同源。**论文必须按 §8.6 在污染注册表标注 ECL 属于老师预训练分布。**

### 5.5 Solar-AL / Solar(ts_benchmarks)
- **来源:** 同 laiguokun/multivariate-time-series-data;本地 `solar_AL.csv`,52560×137
- **许可:** 惯例 CC BY 4.0(底层 NREL)
- **频率:** 10 分钟
- **建议角色:** `final benchmark`。**排除出 U0 蒸馏语料。**
- **与 FM 预训练重叠:** uni2ts 论文明确 Solar 被剔除出 LOTSA(作 OOD);MOMENT 组件信息未单列 Solar → **不声称**(但经 §2.5 Monash solar 同源,存在间接重叠;建议按"谨慎标注、默认不声称"处理)。

### 5.6 UniElecPrice(40 国 DA 电价)
- **来源:** Zenodo 16284828(作者自存档镜像;IEEE Data Descriptions 2025)
- **许可:** 作者公开镜像(CC BY 惯例)
- **频率:** 小时;40 国,量纲极端异质
- **建议角色:** `host diversity` + `final benchmark`(尺度不变性对照)。**排除出 U0 蒸馏语料。**
- **与 FM 预训练重叠:** **不声称**。

### 5.7 Weather / Traffic / ETT(ts_benchmarks 通用时序)
- **来源:** laiguokun/multivariate-time-series-data;ETT 来自 zhouhaoyi/ETDataset
- **许可:** 惯例开放数据
- **频率:** Weather 10min / Traffic 1h / ETT 15min–1h
- **建议角色:** `final benchmark`(模型无关通用性对照)。**默认排除出 U0 蒸馏语料**(非电力负荷;若 U0-C 需更大规模可考虑,但 ETT/Weather/Traffic 在 MOMENT 语料内 → 会进一步扩大泄漏面)。
- **与 FM 预训练重叠:** ETT/Weather/Traffic/Exchange/ILI 均属 MOMENT 语料 **LTSF 组件 → 明确重叠**。ECL 同理(已列 §5.4)。

---

## 6. 汇总表

| # | 数据集/配置 | 来源 URL | 许可 | 频率 | 序列数 | 近似大小 | 512-step 长 context | 建议角色 | 与 MOMENT 预训练重叠 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | LOTSA `australian_electricity_demand` | huggingface.co/datasets/Salesforce/lotsa_data | Apache-2.0 | 30min | 5 | ≈4.6MB | ✅ | **representation distillation**(U0-A) | 明确(经 Monash 组件) |
| 2 | LOTSA `london_smart_meters_with_missing` | 同上 | Apache-2.0 | 30min | ≈5,520 | ≈166M obs | ✅ | **representation distillation**(U0-B) | 明确(经 Monash 组件) |
| 3 | LOTSA `wind_power` | 同上 | Apache-2.0 | 4s | 1 | ≈7.4M obs | ✅ | **representation distillation**(U0-B) | 不声称(Moirai 参考:是) |
| 4 | LOTSA `solar_power` | 同上 | Apache-2.0 | 4s | 1 | ≈7.4M obs | ✅ | **representation distillation**(U0-B) | 不声称(Moirai 参考:是) |
| 5 | LOTSA `wind_farms` | 同上 | Apache-2.0 | 1min | ≈337 | ≈172M obs | ✅ | **representation distillation**(U0-B) | 不声称(Moirai 参考:是) |
| 6 | LOTSA `residential_load_power` / `residential_pv_power` | 同上 | Apache-2.0 | 未详 | 未详 | 未详 | 待确认 | **host diversity** | 不声称(Moirai 参考:是) |
| 7 | LOTSA `buildings` | 同上 | Apache-2.0 | 1h/15min | 900k 级 | 大型 | ✅ | representation distillation / exclude | 不声称(Moirai 参考:是) |
| 8 | LOTSA `cmip6_*` / `era5` | 同上 | Apache-2.0 | 日/月/再分析 | 极多 | 数百 GB | ✅ | **exclude** | 不声称(Moirai 参考:是) |
| 9 | Monash `electricity_hourly` | zenodo.org/records/4656140; Monash-University/monash_tsf | CC BY 4.0 | 1h | 321 | ≈31MB(HF card) | ✅ | **representation distillation**(U0-A) | **明确**(Monash 组件) |
| 10 | Monash `electricity_weekly` | monash_tsf | CC BY 4.0 | 1w | 321 | 小 | ❌(<512) | **exclude** | 明确(Monash 组件) |
| 11 | Monash `australian_electricity_demand` | zenodo.org/record/4659727 | CC BY 4.0 | 30min | 5 | 数 MB | ✅ | representation distillation(与 #1 二选一) | **明确**(Monash 组件) |
| 12 | Monash `london_smart_meters` | zenodo.org/record/4656072 | CC BY 4.0 | 30min | 5,560 | ≈166M obs | ✅ | **representation distillation**(U0-B) | **明确**(Monash 组件) |
| 13 | Monash `solar_10_minutes` | zenodo.org/records/4656144 | CC BY 4.0 | 10min | 137 | ≈4.6MB zip | ✅ | **representation distillation**(U0-B) | **明确**(Monash 组件) |
| 14 | Monash `solar_weekly` | monash_tsf | CC BY 4.0 | 1w | 137 | 小 | ❌ | **exclude** | 明确(Monash 组件) |
| 15 | Monash `wind_farms_minutely` / `wind_4_seconds` | monash_tsf | CC BY 4.0 | 1min/4s | 337/1 | 172M/7.4M obs | ✅ | **representation distillation**(U0-B) | **明确**(Monash 组件) |
| 16 | AEMO 官方(NEM 需求/价格) | aemo.com.au | AEMO 版权声明 | 5min→1h | 分区×多年 | 本地已拉 | ✅ | **final benchmark**(本地) | 间接(经 Monash 澳洲需求) |
| 17 | GEFCom2014-L/W/S | Hong et al. IJF 2016 官方补充材料 | GEFCom 条款 | 1h | L1/W10/S3 | MB 级 | W/L ✅,S ⚠️ | **host diversity** | 不声称 |
| 18 | Chronos `electricity_15min` | autogluon/chronos_datasets | 仓库 "other" | 15min | ≈370 | 数百 MB | ✅ | **representation distillation**(U0-B) | **明确**(ECL 同源) |
| 19 | Chronos `ercot` | autogluon/chronos_datasets | 仓库 "other" | 1h | 数十 | 数十 MB | ✅ | **host diversity** | 不声称(Chronos 参考:是) |
| 20 | Chronos `solar` / `solar_1h` | autogluon/chronos_datasets | 仓库 "other" | 10min/1h | 137 | MB–百 MB | ✅ | **representation distillation**(U0-B) | **明确**(Solar-AL 同源) |
| 21 | NREL 原始 PV/Wind | nrel.gov/grid/solar-power-data | NREL/DOE | 5min/1h | ≈6,000 | 大 | ✅ | **exclude** | 不声称 |
| 22 | Lago 5 市场 | zenodo.org/records/4624804 | 开放 | 1h | 5 | 已本地 | ✅ | **final benchmark** | 不声称 |
| 23 | GEFCom2014-P | 官方补充材料 | GEFCom 条款 | 1h | 1 | 已本地 | ✅ | **final benchmark** | 不声称 |
| 24 | AEMO/NEM 5 区 | AEMO | AEMO 版权 | 5min→1h | 5 | 已本地 | ✅ | **final benchmark** | 间接 |
| 25 | UniElecPrice | zenodo.org/records/16284828 | 作者镜像 | 1h | 40 国 | 已本地 | ✅ | **host diversity** + final benchmark | 不声称 |
| 26 | ECL / Electricity | github.com/laiguokun/multivariate-time-series-data | CC BY 4.0 惯例 | 1h | 321 | 已本地 | ✅ | **final benchmark** | **明确**(LTSF 组件) |
| 27 | Solar-AL | 同上 | CC BY 4.0 惯例 | 10min | 137 | 已本地 | ✅ | **final benchmark** | 不声称(谨慎) |
| 28 | Weather / Traffic / ETT | 同上 / ETDataset | 开放 | 10min/1h | 21/862/7×4 | 已本地 | ✅ | **final benchmark** | **明确**(LTSF 组件) |

---

## 7. 建议:哪些值得 U0 用、哪些排除

### 7.1 值得进入 U0(蒸馏语料)

**U0-A sanity(最小验证,与主计划 §8.5 一致)**
1. `Salesforce/lotsa_data : australian_electricity_demand`(Apache-2.0,5 序列,≈4.6MB)—— 最小的端到端验证集。
2. `autogluon/chronos_datasets : monash_electricity_hourly`(CC BY 4.0,321 序列)—— 多序列批量窗口验证。

**U0-B 服务器语料(推荐加入)**
3. `LOTSA : london_smart_meters_with_missing`(家庭用电,5,520 序列)或 Monash 同名版二选一。
4. `LOTSA : wind_power` + `solar_power`(4 秒超长单序列,练极长程依赖)。
5. `LOTSA : wind_farms`(337 条分钟级风场)。
6. `Monash : solar_10_minutes`(137 条光伏,10min)或 Chronos `solar` 二选一。
7. `Chronos : electricity_15min`(15 分钟电力,370 序列,与 ECL 同源但更高频)。

**取舍原则:** 同一底层数据源(LOTSA vs Monash vs Chronos)只保留一个 config,避免语料内自重复;优先许可更宽松、config 更干净的(LOTSA Apache-2.0 > Chronos "other" > Monash CC BY 4.0)。

### 7.2 值得作为 host diversity(不进蒸馏、供 U1-Energy+/U2-rich)
- `Chronos : ercot`(德州负荷,冻结 TTM/小模型作 host 可便宜造域)。
- `GEFCom2014-L/W/S`(负荷/温度/风/光外生协变量齐全,适合 U2-rich)。
- `UniElecPrice`(40 国电价,量纲极端异质,验证尺度不变)。
- `LOTSA : residential_load_power / residential_pv_power`(确认细节后再定)。

### 7.3 排除(exclude)
- **Monash `electricity_weekly`、`solar_weekly`**:序列过短,不满足 512-step 蒸馏窗口。
- **LOTSA `cmip6_*`、`era5`**:非电力负荷领域、体量过大(数百 GB),收益低。
- **NREL 原始 PV/Wind**:父级源,格式杂、体积大;其精选版已足够。
- **所有 final benchmark 数据(Lago / GEFCom2014-P / AEMO-NEM / UniElecPrice / ECL / Solar-AL / Weather / Traffic / ETT)**:一律**排除出 U0 蒸馏语料**,否则直接构成"蒸馏语料 ∩ 最终基准"泄漏。

### 7.4 重叠与泄漏注册(必须写进污染注册表)
1. **明确重叠(老师 MOMENT 官方语料)**:ECL、ETT、Weather、Traffic、ILI、Exchange(MOMENT 的 LTSF 组件);以及一切 Monash 组件数据集(electricity_hourly、australian_electricity_demand、london_smart_meters、solar_10_minutes、wind_farms、wind_4_seconds)。
2. **同源但老师组件信息未单列**:Solar-AL(经 Monash solar 同源)、electricity_15min(经 ECL 同源)——按"谨慎标注、默认不声称"处理,并在论文按 §8.6 区分 `representation-pretrained transfer` 与 `strict unseen-data transfer`。
3. **不声称**:LOTSA 单独(不在 MOMENT 组件内;对 Moirai/Chronos 明确重叠,仅参考)、Lago、GEFCom、ERCOT、UniElecPrice、NREL。
4. **可用的真正 unseen 基准(建议保留给最终验证)**:Lago、GEFCom2014-P、UniElecPrice、以及未进入任何蒸馏语料的 NEM 价格切片 —— 这些可用于 `strict unseen-data transfer` 陈述。

### 7.5 下一步(只读任务内)
- 生成 `data_manifest.csv`(config、license、series count、window count、sampling weight),需在 U0 开始前完成并纳入 run manifest(主计划 §14)。
- 对每个入选 config 记录**确切 HF revision**(拉取时 `state.json` 的 `_commit_hash`),以便复现。
- 确认 LOTSA `residential_load_power`/`residential_pv_power` 与 `buildings` 的频率/序列数后再决定是否纳入 U0-B。

---

## 8. 来源 URL 清单

- LOTSA(HF): https://huggingface.co/datasets/Salesforce/lotsa_data
- uni2ts 论文: https://arxiv.org/abs/2402.02592 ;代码: https://github.com/SalesforceAIResearch/uni2ts
- Monash 档案(HF): https://huggingface.co/datasets/Monash-University/monash_tsf ;官网: https://forecastingdata.org/ ;论文: https://arxiv.org/abs/2105.06643
- Monash electricity hourly(Zenodo): https://zenodo.org/records/4656140
- Monash australian electricity demand(Zenodo): https://zenodo.org/record/4659727
- Monash solar 10-minutes(Zenodo): https://zenodo.org/records/4656144
- Monash london smart meters(Zenodo): https://zenodo.org/record/4656072
- Chronos datasets(HF): https://huggingface.co/datasets/autogluon/chronos_datasets ;extra: https://huggingface.co/datasets/autogluon/chronos_datasets_extra
- Chronos 论文: https://arxiv.org/abs/2403.07815 ;代码: https://github.com/amazon-science/chronos-forecasting
- MOMENT(HF 模型): https://huggingface.co/AutonLab/MOMENT-1-small ;论文: https://arxiv.org/abs/2402.03885 ;代码: https://github.com/moment-timeseries-foundation-model/moment
- MOMENT Timeseries-PILE(HF): https://huggingface.co/datasets/AutonLab/Timeseries-PILE
- TimesFM 2.0 README(LOTSA pretrain 清单,参考): https://huggingface.co/google/timesfm-2.0-500m-jax
- AEMO 官方: https://aemo.com.au/ ;`aemo` R 包源码: https://github.com/charlescoverdale/aemo
- GEFCom2014(Hong et al., IJF 2016): https://www.sciencedirect.com/science/article/abs/pii/S0169207016000133 ;Tao Hong 博客/分发: http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html
- NREL Solar Power Data: https://www.nrel.gov/grid/solar-power-data.html
- ECL/Solar/Weather/Traffic/Exchange/ILI 源: https://github.com/laiguokun/multivariate-time-series-data ;ETT: https://github.com/zhouhaoyi/ETDataset
- Lago(Zenodo): https://zenodo.org/records/4624804
- UniElecPrice(Zenodo): https://zenodo.org/records/16284828
- 仓库本地参考: `data/lago_benchmark/`、`data/gefcom2014/`、`data/nem_aemo/`、`data/unielecprice/`、`data/ts_benchmarks/`;核实文档 `docs/paper_research/03_论文准备工作文档_数据集与对比对象核实.md`
