# HCH-v2 Paper Benchmark Gate — 详细执行计划(协议 v0.1 落地)

**日期**: 2026-08-14 | **环境**: 本地桌面(纯 CPU)| **协议**: `hch_v2_paper_benchmark_gate_comparative_experiment_protocol_v0.1_2026-08-14.md`

---

## 0. 资产盘点结论(计划依据)

### 已有可复用
| 资产 | 位置 | 说明 |
|---|---|---|
| 8 headline 数据集 | `data/raw/lago_benchmark/clean/`(DE/BE/FR/PJM/NP)、`nem_aemo`、`gefcom2014`、`unielecprice/by_country/Denmark.csv` | 全部齐 |
| 8 extended 数据集 | `data/raw/epex_markets/`(DE_EPEX/PJM_2020/EPEX_BE/FR/NL/Finland/NordPool_NO/SE3) | 全部齐 |
| domestic | `data/raw/provinces/`(shandong_pmos_96_full_v2.xlsx、shandong_pmos_hourly.csv、宁/甘/陕/青 xlsx) | 6 文件,需 audit |
| host 骨干 | `src/backbones.py` BACKBONES=(Linear/MLP/LSTM/Transformer/GBDT/TCN/PatchTST) | H1–H4 已实现 |
| HCH 主链 | `src/hch_v2_pipeline.py`(replay 355–425 正确)、`src/iah_candidate.py`、`universal_trainer.py` | 完整 |
| **P0-A 正确 replay** | `experiments/08-hch-v2/r1a9_action_calibration.py:472` `point_metrics()` `pred=s_day·sinh(z⁰+π_eff)` | **与协议公式一字不差,直接集中化** |
| baseline B0–B2 | `experiments/08-hch-v2/baselines_v2.py`(Identity / ResidualL1-LGBM / QuantileResidual-LGBM) | 有 |
| B3 δ-Adapter / B4 PIR | `experiments/07-route-e/peers/delta_adapter.py`、`PIR/`、`repro_delta_adapter.py`、`repro_pir.py` | **复现状态:报告标注 PIR「待复现」,δ 未列入复现报告 → 均需 B1 跑 fidelity** |
| B5/B6 训练逻辑 | `r1b_stage2a_panel.py`(universal trainer)、`r1b_stage2f_localcore.py`(单域) | 有 |
| 旧对比矩阵 | `experiments/01-comparative/results/v2_*.json`(12 市场,旧方法) | 只作参考,不复用 |
| R1B 已 prep 域 | LAGO_DE/PJM、NEM_SA1、GEFCOM14P、NORD_DK1、EPEX_FR、PJM_2020 | 6 市场已 prep |
| 本地依赖 | lightgbm 4.6 / pandas 2.2.3 / sklearn / scipy / statsmodels / openpyxl / pyarrow | 全齐 |

### 缺口(本计划要建)
1. P0-A:final-output 口径集中化 + 3 回归测试 + 2D 重跑
2. P0-B:provenance 打包(`00_RUN_CONFIG/01_CODE_PROVENANCE.json`,去 `git_sha=unknown`)
3. `02_DATASET_MANIFEST.csv` / `03_DOMESTIC_DATA_AUDIT.csv` / `04_HOST_MANIFEST.csv`
4. 新域 prep + host cache 批量(8 headline 未 prep 的 BE/FR/NP + 8 extended + domestic)
5. universal 重训 → public 全部 headline 训练域 + hierarchical balancing + checkpoint guards
6. B1 PIR/δ fidelity smoke → `05_PEER_FIDELITY_REPORT.md`
7. B3/B4 主矩阵 runner(18 项产物全链:`06_PREDICTIONS.parquet` → `17_PAPER_GATE_VERDICT.md`)
8. DM 检验、rank、win-rate 汇总

---

## 1. Work Package 分解

### WP-0 资产基线(P0-B 数据面)—— 1–2h
**目标**:数据/host/版本清单冻结。
1. 遍历 `data/raw/` → `02_DATASET_MANIFEST.csv`(数据集/时段/频率/DA-RT/负价率/信息截止期/缺失/是否已 prep)。
2. 按协议 §5.3 字段生成 `03_DOMESTIC_DATA_AUDIT.csv`;**判定 Shandong DA / Shandong RT 是否 headline 资格**(shandong_pmos_96_full 需核对 96 点语义、时间粒度、信息截止期)。
3. `04_HOST_MANIFEST.csv`(4 host,版本/hash)。
4. 冻结:记录 git SHA,确认 `git_sha=unknown` 全清。
**验收**:3 张 csv + 版本快照;domestic 表能回答"山东 DA/RT 是否合法 headline"。
**风险**:Shandong 语义不确定(96 点=DA 日前?RT 5min?)→ audit 判定可能剔除;宁/甘/陕/青若无负价且信息截止期不清,只能进 secondary。

### WP-1 P0-A final replay 统一 —— 2–4h ⭐关键前置
**目标**:所有 point 指标用真实最终输出 `x_final = s_d·sinh(z⁰ + π_eff)`。
1. 从 `r1a9:472 point_metrics()` 提炼公共函数(建议入 `src/hch_v2_pipeline.py` 或 `experiments/08-hch-v2/_final_point.py`):输入 (rows, released) → final pred / host pred / sMAPE。
2. `r1b_stage2d_action_chain.py:67 forecast_metrics()` 改为调用它;2E/2F 同步。
3. 回归测试 3 条(协议 P0-A 原文):
   - 全 Identity → `x_final == host`
   - released π≠0 → `x_final == s·sinh(z⁰+π)` 精确
   - bundle reload → x_final 逐位一致
4. canonical 16 域以新口径重跑,对比旧 host_day 口径的 MAE/sMAPE/负价/高尾。
**验收**:3 测试过;重跑后 CRPS/A_net 不变(口径只影响 point),MAE/sMAPE 换真实值;写明新旧差异表。
**影响**:R1B 主结论(CRPS/A_net/§30 判定)不受影响;point 指标数值更新。此步不过,后面一切不启动(协议明文)。
**风险**:低——正确实现已存在;注意 `_rows`/`_released` 是否在 2D 路径传递完整。

### WP-2 P0-B/P0-C provenance + 命名 —— 1–2h
1. `00_RUN_CONFIG.json`(协议 §4 全字段)+ `01_CODE_PROVENANCE.json`(git SHA / 架构 / 数学核 / split / host / peer / dataset hashes)。
2. host 命名强制:PatchTST→"PatchTST-style"(协议 P0-C);论文稿前不得宣称官方复现。
**验收**:00/01 齐;全产物无 `git_sha=unknown`。

### WP-3 B1 peer fidelity smoke —— 3–6h
**目标**:PIR / δ-Adapter 官方-复现对照,决定 accept / patch。
1. PIR:官方 reference(优先 `PatchTST × ETTh1`,备选 `PatchTST × Electricity`,用官方 split/horizon/metric)。对比 Host MSE/MAE vs PIR → `PIR_ACCEPT_AS_IS` 或 `PIR_PATCH_RETRIEVAL_REQUIRED`(当前实现缺官方 retrieval,协议已预设 patch 路径)。
2. δ-Adapter:1–2 官方 reference(PostY/Ada-Y 家族 setting)。→ `DELTA_ACCEPT_AS_IS` 或 `DELTA_OFFICIAL_ALIGNMENT_REQUIRED`。
3. 输出 `05_PEER_FIDELITY_REPORT.md`(每个:官方数字、复现数字、相对改善、判定)。
**验收**:两方法都有数字与判定;不 tune 到 HCH 数据集。
**风险**:PIR 若触发 PATCH(retrieval 补齐)耗时最大,但协议要求"不值得长期工程",只补已知 retrieval。

### WP-4 B2 paper smoke —— 2–4h
**目标**:小规模暴露真实 final 指标 + 方法栈全通。
- 域:LAGO_DE / LAGO_PJM / NEM_SA1(+ Shandong DA/RT **若 WP-0 通过 audit**)。
- host:Linear / MLP;方法:B0–B6(7 方法)。
- 复用 R1B 已 prep 的 3 市场 host cache;写 smoke runner(从 stage2d 骨架改)。
**验收**:B0–B6 全跑通;确认 P0-A 后 point 指标真实;发现明显 gap 则进入 WP-6 前先记入 failure map。

### WP-5 主矩阵数据准备 —— 5–10h CPU ⭐最大工作量
1. **新域 prep**:LAGO_BE/FR/NP、8 extended、domestic 通过域(复用 `r1b_cache_batch.py` 管线;每域:时间窗/特征/asinh/签名参考 S1R)。
2. **host cache 批量**:全部 audit 通过域 × Linear/MLP/LSTM/PatchTST → host 预测缓存(分市场保存,seed0)。
3. **universal 重训**:public **全部 headline 训练域**(≈12 域,含新 BE/FR/NP)重训 LearnedSig_main,seed0,`L=L_IAH-CRPS`,hierarchical balancing(market→dataset→host→day)。
4. **checkpoint guards**(协议 §12.2):逐 headline 域 S2V CRPS 回归 ≤2%;worst 域 ≤3%;generalization-anchor ≤2%。失败→ rollback / 调 sampler。
**验收**:host cache 覆盖主表;universal 通过 guards;`training_reports.json` 留档。
**风险**:新市场 prep 数据质量(负价率、异常值、时区);扩域后 worst-domain guard 失败 → Case C 路径(hierarchical sampling / family-balanced)。

### WP-6 B3 foreign headline 矩阵 seed0 —— 8–16h CPU
**目标**:64 primary cells + 全套诊断。
1. runner:8 dataset × 4 host × B0–B6;每 cell 存小时级预测 → `06_PREDICTIONS.parquet`(可重算)。
2. 指标:MAE / sMAPE 主;RMSE / rMAE / CRPS / 负价 MAE / 高尾 MAE 次。
3. **DM 检验**:day-level 配对 Diebold-Mariano(绝对误差损失),B5 vs 各 peer + B5 vs Host(协议 §10.2)。
4. 汇总产物:`07_METRICS_BY_CELL` / `08_RANKS_BY_CELL` / `09_PRIMARY_WIN_RATE` / `10_DATASET_LEVEL_SUMMARY` / `11_HOST_LEVEL_SUMMARY` / `12_DM_TESTS` / `13_HCH_CANDIDATE_DIAGNOSTICS` / `14_HCH_ACTION_DIAGNOSTICS` / `15_FAILURE_MAP` / `16_GENERALIZATION_LEDGER`。
**验收**:64 主 cell 全有数字;ranks 与 DM 齐;FAILURE_MAP 每弱 cell 打标(CANDIDATE/POINT_READOUT/LOCAL_CALIBRATION/DVG/HOST)。
**风险**:计算量大(64×7);B6 每 dataset 单域训练(部分难域弱属正常,标记 business caveat)。

### WP-7 B4 domestic 矩阵 seed0 —— 3–6h
- CN-A:public-universal frozen transfer(山东 DA/RT + 其他通过省份)× 4 host × B0–B6。
- CN-B(可选):山东 dev 标签 few-shot/local adaptation,标 private/business。
**验收**:domestic scorecard(协议 §15.2 目标:山东 DA/RT Top1/tied ≥75%)。

### WP-8 B5 Paper Performance Gate v1 —— 2–4h 🔴硬 STOP
1. 按协议 §15 判层:STRONG_GREEN(Top1/tied≥70% & Top2≥90%)/ GREEN(60%/85%)/ YELLOW(30–60%)/ RED(<30%)。
2. `17_PAPER_GATE_VERDICT.md`:A P0 状态 / B peer fidelity / C foreign scorecard / D domestic / E failure 分解 / F 唯一 verdict / G ≤3 个证据驱动修改建议。
3. **硬 STOP(协议 §20)**:不启动 U0 / 大扩数据 / 新架构 / 新 FM / sealed test。交人类评审。
**验收**:verdict 唯一;后续动作 ≤3 条且不自动执行。

### WP-9 B6 multi-seed(条件触发)—— 额外 2× 计算
- 仅 GREEN 或修复后强 YELLOW:seeds 0/1/2,host cache 固定,隔离 HCH 随机性。

---

## 2. 依赖图与并行策略

```
WP-0 ─┬→ WP-1(P0-A)──→ WP-2(provenance)──→ WP-4(B2 smoke)
      └→ WP-3(B1 fidelity)────────────────→ WP-6(B3)
                                 WP-5(数据准备)──→ WP-6 → WP-7 → WP-8(STOP)→ WP-9
```

- WP-0 与 WP-1 可并行(不同人/不同核)。
- WP-5 的 host cache 一旦 host 骨干定版即可后台批跑,与 WP-3/WP-4 并行。
- 全程本地 CPU 顺序为主;host cache 与 universal 训练可后台,其余单进程。

## 3. 总时间线(本地 CPU,乐观)
| 段 | 累计 |
|---|---|
| WP-0 资产基线 | ~0.5 天 |
| WP-1 P0-A(关键门) | ~1 天 |
| WP-2/3/4(fidelity + smoke) | ~1–1.5 天 |
| WP-5 数据准备(最大) | ~1.5–2 天 |
| WP-6 B3 主矩阵 | ~1.5–2 天 |
| WP-7/8 B4 + Gate v1 | ~0.5–1 天 |
| **合计** | **≈5–8 个工作日(视新市场 prep 与 DM/渲染细节)** |

## 4. 红线(全程不可破)
1. **P0-A 不过,不碰任何 headline 对比**(协议明文)。
2. **Shandong/private 永不混入 public-universal 训练**(协议 §8)。
3. **sealed test 不可反复 tune**(§17):tune 只许 train/val/dev。
4. **B5 硬 STOP**,不自动 multi-seed / 不扩数据 / 不新架构(§20)。
5. **30% 不叫多数**(§15);完整表格必须留,不靠口径把输写赢。
6. host 命名:PatchTST 只能是 style,不冒充官方复现(P0-C)。
7. R1A/R1B 的 S4 不得称 pristine manuscript test;新 Paper Development Test 段另冻结(§11)。

## 5. 风险清单与缓解
| 风险 | 概率 | 缓解 |
|---|---|---|
| P0-A 重跑暴露 2D 指标口径不一致 | 中 | 正确实现已在 r1a9;集中化+测试;新旧口径差异表透明 |
| 新市场 prep 数据质量(负价率/异常/时区) | 中 | 每域 audit manifest;异常域打标;不改架构 |
| universal 扩域后 worst-domain guard 失败 | 中 | Case C:family-balanced sampling;guards 回滚 |
| PIR 缺 retrieval → 需 patch | 中高 | 只补已知 retrieval;不 redesign(协议 §9.1) |
| B6 HCH-Local 单域训练弱(如 NEM_SA1) | 确定 | 标记 business/local 性质,不做 candidate 借口 |
| B3 计算量大(本地 CPU) | 高 | host cache 一次算;预测存 parquet;指标可重算 |
| Shandong 语义审计不过 | 中 | 合法则 headline;否则 secondary + 明确说明 |

## 6. 首个执行段建议(拿到 go 之后)
先做 **WP-0 + WP-1**(资产 audit + P0-A):两者都不依赖大计算、风险最低、且是后续一切的前置。P0-A 的 3 条回归测试与新旧口径差异表是第一个可交付里程碑。
