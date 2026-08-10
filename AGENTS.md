# BECH — 极端电价校正研究项目

> 科研项目 · 论文：*A Model-Agnostic Budgeted Correction Head for Extreme Electricity Prices*

## 当前研究状态（2026-08-09）

项目已从 Phase B 创新审计转为 **Route-E 应用论文**路线。Phase B 判 `NO-GO`（见 `docs/paper_prep/37_PhaseB_commander_verdict.md`）。

当前主线：**Hurdle Correction Head (HCH)** — 冻结基座 + Bi-Hurdle 选择性极端电价校正。

## 入口文档（阅读顺序）

1. `AGENTS.md` (本文件)
2. `docs/paper_info/README.md` — 论文全局信息（数据集/基线/引用）
3. `docs/paper_info/peer_reproduction_log.md` — 同行基线复现记录
4. `docs/paper_prep/38_RouteE_experiment_design.md` — 实验设计
5. `docs/paper_prep/37_PhaseB_commander_verdict.md` — Phase B 裁决
6. `experiments/07-route-e/peers/` — 同行基线代码

## 文件夹公约

```
solar_leak_price_model/
├── AGENTS.md                         # 本文件 — 项目总入口
│
├── data/                             # 数据（大文件git忽略）
│   ├── raw/                          #   不可变的原始数据
│   │   ├── provinces/                #     省级数据（山东/山西/甘肃/陕西/青海/宁夏）
│   │   ├── lago_benchmark/           #     Lago 2021 开源基准 (DE/BE/FR/NP/PJM)
│   │   ├── gefcom2014/               #     GEFCom2014-P
│   │   ├── nem_aemo/                 #     AEMO NEM 5区
│   │   ├── unielecprice/             #     UniElecPrice 40国
│   │   └── ts_benchmarks/            #     ETTh/Weather/ECL/Solar
│   └── processed/                    #   清洗后的建模数据
│
├── src/                              # 核心源代码
│   ├── backbones.py                  #   5 冻结基座 (Linear/MLP/LSTM/Transformer/GBDT)
│   ├── selective_hurdle.py           #   Hurdle Correction Head (fit→calibrate→apply)
│   │                                 #     Bi-Hurdle: occurrence×magnitude + SCARR 证书
│   ├── common.py                     #   数据加载/四段切分/评估/episode指标/防泄漏
│   ├── bech.py                       #   [归档] 旧 BECH v0，已被 selective_hurdle.py 替代
│   ├── run_peer_baselines.py         #   [归档] 旧同行对照脚本
│   ├── audit_peer_gain.py            #   [归档] 旧增益归因
│   └── make_evidence.py              #   [归档] 旧证据生成
│
├── experiments/                      # 按科学状态组织，先读 experiments/README.md
│   ├── 00-data-exploration/          #   数据特征与制度漂移支持证据
│   ├── 01-main-matrix/               #   冻结 Route-E 证据：9数据集 × 5基座
│   ├── 02-ablations/                 #   冻结 Route-E 证据：A0-A4
│   ├── 03-peer-comparison/           #   冻结 Route-E 证据：M0-M9
│   ├── 04-gain-audit/                #   冻结 Route-E 证据：增益归因
│   ├── 05-episode-audit/             #   当前有效问题锚点
│   ├── 07-route-e/                   #   **活跃** Route-E 应用论文实验
│   │   ├── peers/                    #     同行基线代码 (Quantile/CRC/delta-Adapter/SpikeReg/Vahedi)
│   │   ├── run_route_e.py            #     HCH 实验 runner
│   │   └── results/                  #     实验结果
│   ├── figures/                      #   论文图件
│   ├── _support/                     #   数据核验/私有数据盘点工具，不是实验结果
│   └── _archive/                     #   失效原型、退役路线和中间产物，不得作证据
│
├── docs/                             # 文档
│   ├── literature/                   #   文献调研（交叉验证、综述）
│   ├── phase0_evidence_audit/        #   Phase-0：山东失败审计 + RQ1-10
│   ├── phase1_failure_analysis/      #   Phase-1：多省跨市场证据
│   ├── phase2_public_migration/      #   Phase-2：Lago 公开基准落地
│   └── paper_prep/                   #   新旧科研记录并存；先读 README.md
│
└── paper/                            # 论文写作工程 (LaTeX)
```

## 核心代码三步（`src/bech.py`）

| 步骤 | 方法 | 段 | 做什么 |
|---|---|---|---|
| Step 1 训练 | `fit()` | S2 | 用 LightGBM 训练双路分类头（`P(neg\|Z)`, `P(spike\|Z)`）+ 条件中位数幅度头（预测修正量 δ）|
| Step 2 标定 | `calibrate()` | S3 | SCARR v2 两层证书——效能层（bootstrap LCB 确认改善>0）+ 安全层（共形分位约束单点伤害≤ρ×基线）。输出每分支 λ∈[0,1] |
| Step 3 修正 | `apply()` | S4 | `P(event\|Z) > 0.5` 且 `λ > 0` 时：ŷ_bech = ŷ_base + λ·δ；否则恒等返回 |

## 实验协议（硬约束）

- 四段 rolling-origin：S1(50%)基座 → S2(20%)校正头 → S3(10%)标定 → S4(20%)测试
- 全部报告仅来自 S4
- `y_t` 永不进特征；残差历史 ≥24h 滞后
- 外生只用日前可得预测值；尖峰阈值取自 S1 p99
- 建表后 `assert_no_leakage`；校正特征矩阵 Z 独立泄露审计（`max|corr(Z_j, y_t)| < 0.3`）

## 五个基座

| 基座 | 类型 | 归纳偏置 |
|---|---|---|
| Linear | Ridge 回归 | 线性 |
| MLP | sklearn MLPRegressor | 浅层非线性 |
| LSTM | PyTorch 单层 LSTM | 循环/时序记忆 |
| Transformer | PyTorch 小Transformer | 自注意力 |
| GBDT | LightGBM | 提升树 |

**注意**：这五个是自建轻量实现，用来提供异构误差结构检验基座无关性，**不代表 SOTA 精度**。

## 同行基线（Route-E 应用论文）

| # | 基线 | 论文 | 角色 | 状态 |
|---|---|---|---|---|
| B1 | **Quantile Correction** | 经典方法 | 统计基线 | ✅ |
| B2 | **Vahedi 2026** | IEEE ICCE | 负价预测基线 | ✅ |
| B3 | **PIR** | NeurIPS 2025 | 实例感知后处理 | ⏳ 待复现 |
| B4 | **CRC** | arXiv:2512.22428 | 安全残差校正 | ⚠️ 方法论 |
| B5 | **SpikeReg** | AAAI 2026 WS | 尖峰感知 | ⏳ 待深入 |

详细复现记录见 `docs/paper_info/peer_reproduction_report.md`

## 已有数据集

## 已跑的冻结 Route-E 实验（2026-08-06 完成）

| # | 实验 | 规模 | 关键结论 |
|---|---|---|---|
| 01 | 主矩阵 | 9数据集×5基座=45组合 | SCARR触发16/弃权29，弃权组合MAE bit-exact 0，加权漏判率-19.4% |
| 02 | 消融 | 5阶梯×9对=45次 | 活跃组+0.22→+1.69%单调递增，零退化 |
| 03 | 同行对照 | 18组合×11方法=198行 | BECH唯一零退化+漏判率显著↓+最坏伤害低一个数量级 |
| 04 | 增益审计 | 3市场×1基座 | 全局后处理器增益 36.3%来自数据新鲜度、仅0.5%来自特征集 |
| 05 | M9组合实验 | 18组合 | BECH叠重训基座仍0/18退化，NEM_SA1/Linear +2.40%(p<1e-4) |

另有 `experiments/05-episode-audit/` 提供当前问题锚点：负价具有多小时 episode 结构，冻结基座经常完整漏掉事件。该结论不等于算法创新。

## 当前 Phase B 执行闸门

- 第一轮只允许生成 `35a/35b/35c` 三份研究文档。
- Agent A 必须攻击最强组合；Agent B 必须区分已知 T0 与非定义性 T1；Agent C 只能设计证伪计划。
- A/B 未同时通过前，不得新建或运行真实数据 pilot。
- 若获放行，唯一新实验目录为 `experiments/07-episode-relative-pilot/`。
- 已归档的 `experiments/_archive/retired-methods/06-event-edit-prototype/` 不得修补复用。

## 项目规则

- **CPU only**：本机 GPU 路径不可靠（LightGBM GPU 死锁 + CatBoost GPU 崩溃 + 睡眠杀 CUDA context）
- **conda epf-2**：所有 ML 训练用 `D:/computer_download/environment/conda/epf-2/python.exe`
- **负价漏判率必须按事件数加权**：不可在只有个位数负价点的市场（LAGO_BE/FR）上取算术平均——会被 0.000/1.000 退化值带偏
- **数据策略（用户 2026-08-07 定稿，见 16_论文策略决策.md）**：核心实验全用公开数据可复现（Lago/NEM/GEFCom/UniElecPrice）；**山东私有数据仅作动机段落（引郭鸿业公开统计 11%/13%）+ 内部验证（进补充材料）**；公共主战场=负价重市场（NEM SA1/Lago DE/UniElecPrice 负价国）；配可复现包（脚本+配置+seed）
- **论文定位（用户 2026-08-07 定稿）**：先出一篇能中的，但创新点必须真创新、真解决问题；**硬件/加速创新已明确放弃**，效率只写一小节部署声明不作贡献；横向对比正确对象=基座自己+其他后处理，**BECH 叠前沿预测器展示模型无关性**
- **校正模块输入契约**：当前 Z 含基座内部特征（`x_*` 列），正调研是否收敛为最小可移植契约（yhat+日内形状+日历+残差历史）；见 docs/paper_prep/07_校正模块输入契约设计.md
