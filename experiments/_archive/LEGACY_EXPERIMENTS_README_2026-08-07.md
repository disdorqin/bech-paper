# experiments/ 实验总览（BECH 项目）

> 本文件由 opencode 通读本目录全部脚本与结果后整理，生成于 2026-08-07。
> 角色定位：`experiments/` 只放**实验脚本的产物与结果**；核心算法代码在 `src/`（`backbones.py` / `bech.py` / `common.py`）。本文档对照每个脚本说明"它在跑什么、输入输出、得到什么结论"。

---

## 0. 全局：统一实验协议

所有 Phase-3 实验共享同一套四段 rolling-origin 隔离协议（见 `src/common.py`），全部报告指标**只来自 S4**：

| 段 | 占比 | 用途 | 之后状态 |
|---|---|---|---|
| S1 | 50% | 训练基座（5 基座之一） | 基座**冻结** |
| S2 | 20% | 训练 BECH 校正头（S2a 75% 拟合 / S2b 25% 备用） | 校正头**冻结** |
| S3 | 10% | SCARR 共形风险标定，选 λ | λ **冻结** |
| S4 | 20% | 最终测试 | 仅此段出数 |

防泄漏硬约束：`y_t` 永不进特征；残差历史滞后 ≥24h；外生只用日前可得预测值；尖峰阈值取 S1 p99；建表后 `assert_no_leakage`。

### 数据集分层（`meta.tier`）

| 层级 | 含义 | 数据集 |
|---|---|---|
| L1 | 主战场（高负价） | NEM_SA1, LAGO_DE |
| L2 | 泛化 | NEM_VIC1, NEM_NSW1, LAGO_BE, LAGO_FR, LAGO_PJM |
| L3 | 负对照（应不伤害） | LAGO_NP, GEFCOM14P |
| L4/L5 | 规模 / TS 基准 | （规划中） |

---

## 1. 00-data-exploration/ —— 数据极端性 & 制度漂移证据

### `examine_all_price_datasets.py`
- **干什么**：用统一口径统计全部公开电价数据集的极端性（负价占比、p99/中位、max/p99、偏度、峰度、超绝对阈值占比），供选题论证使用。
- **输出**：`results/ALL_price_characteristics.csv` + `.md`。
- **关键结论**：NEM 各区间负价占比 **5.2%–24.6%**（SA1 最高）、最深到 −1000 AUD/MWh；Lago-DE 1.03%、Lago 其余 <0.2%；GEFCom2014-P 0%。正向尖峰长尾极厚（NEM max/p99 达 38–43）。

### `regime_shift_evidence.py`
- **干什么**：逐年统计"经典基准 vs 当代市场"的负价占比，证明**制度漂移**（problem statement 的核心实证）。
- **输出**：`results/regime_shift_yearly.csv` + `results/regime_shift_evidence.md`。
- **关键结论**：Lago-DE（2012–2017）负价 1.03% → 同一德国市场 UniElec 延伸至 2024 达 3.48%（2023/24 年已 6.1–6.6%）；NEM-SA1（2021–2025）负价 24.6%–30.3%；GEFCom2014 全程 0%。→ 只在经典基准上验证过的方法对"当下最主要的困难"结构性未经检验。

---

## 2. 01-main-matrix/ —— 主矩阵（基座无关性）

### `run_bech_matrix.py`
- **干什么**：**9 数据集 × 5 基座（Linear/MLP/LSTM/Transformer/GBDT）× {base, +BECH}** 全矩阵。对每个基座：S1 训练→冻结；构造校正特征 Z（含 yhat、日内形状、日历、≥24h 残差历史）；`BECH.fit(S2)`→`calibrate(S3)`→`apply(S4)`；输出 base/+BECH 全指标 + DM 检验 + 路由诊断。
- **用法**：`--quick` 冒烟 / `--datasets ... --backbones ...` / `--tag`（分片跑时打后缀，产物如 `bech_*_s1.json`）。
- **输出**：`results/bech_{DS}.json`（逐组合全量）+ `bech_matrix_summary.csv` + `bech_matrix_evidence{tag}.md`。
- **关键结论**（聚合 45 组合）：
  - **SCARR 触发 16 组 / 弃权 29 组**；弃权组合 MAE **bit-exact 0**（逐点恒等）。
  - 高负价主战场 NEM_SA1：5 基座全部触发，MAE↓ **1.07%–3.15%**，DM p 全部显著；负价漏判率如 GBDT 68.1%→49.0%、Linear 63.7%→48.9%；负价段 MAE 全面下降。
  - NEM_VIC1/NSW1：4–5 基座触发，MAE↓ 0.09%–1.60%；Transformer/VIC1 为 -0.24%（唯一小退化，DM 不显著，见局限 2）。
  - LAGO_DE：仅 MLP/GBDT 触发（+1.55%/+1.58%）。
  - 无负价市场（NP/GEFCom）与负价极稀疏市场（BE/FR/PJM）**全部弃权**，λ=(0,0)，MAE 逐点不变——负对照行为正确。
  - 全程正向尖峰分支 λ 一律 =0（主动弃权），负价分支才被认证 → 负电价与正尖峰的**可预测性不对称**（第 8 节结论）。

---

## 3. 02-ablations/ —— 消融阶梯 A0–A4

### `run_ablations.py`
- **干什么**：回答"增益到底来自哪个部件"。**累积阶梯**，每行只比上一行多加一项改动：
  - A0 v0（L2 幅度头 + 伤害率网格搜 τ + 最大 λ）
  - A1 +L1 幅度头（条件中位数）
  - A2 +贝叶斯门（τ 先验固定 0.5）
  - A3 +两层证书（SCARR v2，rho=0.50）
  - A4 +S2b 复用 = v1
- 附带两组证书参数敏感性（rho ∈ {0,0.1,0.25,0.5,1.0}、alpha ∈ {0.05,0.1,0.2}）和一个**去掉 SCARR（λ≡1）**的退化对照。只跑快速基座（Linear/GBDT）× 9 对。
- **输出**：`results/bech_ablations.csv`（127 行，含 ladder/rho/alpha/safety 四族）。
- **关键结论**：
  - 阶梯**单调递增**：活跃组平均 MAE↓ 从 A0 **+0.22%** 升到 A4 **+1.69%**，全程**零退化**（n_neg_any=0）。
  - **SCARR 不是"精度税"**：去掉证书后，过度自信的幅度头在样本外过冲——最坏单点伤害放大一个数量级（如 NEM_SA1/GBDT 93.4→1615.5），平均 MAE 反而变差。λ 实质上是幅度头在独立段上的 MAE 最优标量再标定。
  - rho/alpha 越大增益越大但最坏伤害随之上升（ρ 是明确的伤害预算刻度）。

---

## 4. 03-peer-comparison/ —— 同行对照 M0–M9

> 脚本在 `src/run_peer_baselines.py`，本目录只存 `results/`。

- **干什么**：BECH（M7）对比 9 个标准模型无关后处理，**公平规则刻意向对方倾斜**：同一四段切分、同一冻结基座、同一特征矩阵 Z、同一学习器族（LightGBM 同超参）；能用标定段的竞品（M3/M4/M5）主动给 S3；M2 直接赠予本文的 L1 发现。
  - M0 基座在 S1∪S2 重训（**新鲜度对照**，最关键）
  - M1/M2 全局 δ-Adapter（L2/L1）、M3 +标定收缩、M4 分位后处理、M5 EVT 尾部仿射
  - M6 本文门控但去证书（λ≡1）、**M7 BECH v1**
  - M8 δ-Adapter→BECH、M9 重训基座→BECH（**前后兼容性**实验，BECH 叠加在同行方法之上）
- **输出**：`results/bech_peers.csv`（18 组合 × 11 方法）+ `bech_peers_extra.json`（路由细节、收缩系数、EVT 仿射映射）。
- **关键结论**（论文第 7 节，必须如实呈现）：
  1. **总体 MAE 上 BECH 输给全局后处理器（M2/M4），不是小输** —— 但 M2/M4 逐点改写 100% 预测，等价于用更新的数据把预测器重做一遍（见第 5 节审计）。
  2. BECH 平均只改动高负价 NEM 市场 ~23% 的点，其余逐点恒等，最坏伤害被证书压到一个数量级以下（M2/M4 最坏伤害数千 AUD/MWh）。
  3. **M7 是唯一同时满足**：平均 MAE 不变差 + 加权负价漏判率显著下降 + 全组合零退化 + 最坏伤害最低（低一个数量级）。
  4. **M8/M9 叠加实验**：BECH 叠在已被同行方法（M2）或周期性重训（M0，最强现实部署基线）改良过的预测之上，MAE 几乎不变、负价漏判率继续下降、叠加动作自身 0/18 退化；M9 在 NEM_SA1/Linear 上 **+2.40%（p<1e-4）**。证明 BECH 价值不是"冻结基座过时"的副产品。
  5. M5（EVT 尾部缩放）对符号判别维度**变差**——它只能缩放基座已判为尾部的点，救不回符号判错的点。
  6. 方法学铁律：负价漏判率必须**按事件数加权**（LAGO_BE/FR 测试段仅 2 个负价点，算术平均会被 0.000/1.000 退化值带偏）。

---

## 5. 04-gain-audit/ —— 增益归因审计

> 脚本在 `src/audit_peer_gain.py`，本目录只存 `results/`。

- **干什么**：同行后处理器动辄 −30% 的增益**必须先解释清楚才能写进论文**。把差距拆成三部分，全部在同一 S4 上测量：
  - R0 冻结基座 `X[S1]→y`；R1 同特征新数据 `X[S2]→y`（隔离**新鲜度**）；R2 富特征新数据 `Z[S2]→y`（= M4 家族）；R2b 去掉残差历史（量化 `resid_lag*` 贡献）。并审计 Z 与 `y_t` 的最大相关性（排除泄露）。
- **输出**：`results/bech_peer_gain_audit.csv`。
- **关键结论**：
  - **排除泄露**：max|corr(Z_j, y_t)| ≤ 0.869（NEM 上仅 0.21–0.26；LAGO_DE 0.869 是 `yhat` 本身，属基座输出与真值的正常相关，非信息泄漏）；`resid_lag24` 结构性重建误差 = 0。
  - **真正的解释**：NEM_SA1 上 M2/M4 的 −36.9% 增益中 **36.3% 来自数据新鲜度、仅 0.5% 来自特征集**（残差历史贡献 ≈0）。→ 所谓"全局后处理"实际是"在更新的数据上、用更丰富的特征，把预测器重做了一遍"，文献中常被含糊带过。
  - 附带发现：NEM 上"只用近窗 S2 重训"明显优于"扩窗 S1∪S2 重训" → 早期数据来自已失效的价格制度，是**制度漂移的直接证据**；因此冻结基座作评测锚点时必须同时报告重训对照。

---

## 6. 05-episode-audit/ —— 极端事件"片段结构"审计

### `run_episode_audit.py`
- **干什么**：纯审计（P0，不调新方法），复用 S1–S4 协议重新生成 S4 预测。把负价/尖峰**事件定义为"单个交付日内的小时连续段"**（匹配 BECH 的 24 点校正单元），检验：(a) 标签本身的片段拓扑；(b) base vs BECH 的**事件级**召回/边界/碎片化失败结构。
- **输出**：`results/label_episode_summary.csv`、`model_episode_summary.csv`、`event_failure_detail.csv`、`bech_episode_deltas.csv`、`episode_audit.md`。
- **关键结论**：
  - 负价事件**高度片段化**：NEM_SA1 中位时长 7h、87.9% 为多小时段、71.8% ≥4h、最长 24h（一整天）——24 点校正单元是合理粒度。
  - **BECH 显著改善负价事件的事件级召回**：NEM_SA1 GBDT +12.4%、Linear +15.8%；完整漏检率 base 55.4% → bech 44.6%（按事件加权）；事件小时 MAE 下降 5–16 点。
  - 尖峰分支 base 与 bech 完全一致（Δ=0，因 BECH 对尖峰弃权）——符合"主动弃权"设计。

---

## 7. 数据核验脚本（verify_*.py）—— 数据真实性证据链

| 脚本 | 干什么 | 核验内容 |
|---|---|---|
| `verify_lago.py` | Lago 2021 基准 | 本地文件 MD5 vs `zenodo_meta.json` 官方 checksum + 各市场负价占比 |
| `verify_nem.py` | AEMO NEM 5 区 | 时间范围 / 负价占比 / 缺失 |
| `verify_gefcom_unielec.py` | GEFCom2014-P + UniElecPrice | 列结构 / 负价占比 / meta 结构 |
| `verify_unielec_ts.py` | UniElecPrice + TS 基准 | zip MD5 vs 官方、by_country 与 zip 国家目录对账、ETTh/Weather/ECL/Solar 等 TS 文件形状 |
| `verify_unielec_deep.py` | UniElecPrice 深核 | meta.json 结构（title/doi/files）、zip 内容清单、by_country 逐文件 shape |
| `verify_ts_rest.py` | 杂项 | weather/exchange/illness 的 gbk/latin1 编码问题、UniElec 缺 Canada/USA 的原因（zip 内实为大目录，未解包到 by_country） |

> 用途：为论文"数据真实性"提供证据链（下载完整性 + 内容核验），是 Phase-2 公开迁移的可复现性配套。

---

## 8. 省级私有数据审计（audit_provinces*.py）—— 动机段 & 内部验证

> 按 2026-08-07 数据策略：山东等省级数据**仅作动机段 + 内部验证（进补充材料）**，核心实验全用公开数据。

| 脚本 | 干什么 |
|---|---|
| `audit_provinces.py` | 快速盘 宁夏/甘肃/陕西/青海 xlsx + 山东 csv 的 sheet/列结构（输出写入 `docs/paper_prep/07_省级数据集可用性审计.md`） |
| `audit_provinces_full.py` | 深度审计：格式探测（PK→xlsx）、编码回退（utf-8/gbk/gb18030）、时间解析、电价列缺失率/负价/0 值 |
| `audit_shandong_inventory.py` | 山东数据盘点：逐年负价占比、四段切分时间范围、双尾可预测性（S1 训练→S4 测，Logistic AUC/AP）、特征相关性清单 |

---

## 9. `gate_nem_cross_region.py` —— 跨区图结构门禁

- **干什么**：NEM 5 区**跨区图结构**可行性门禁：Pearson/Spearman 相关矩阵 + 手写**格兰杰因果检验**（lag 1/2/3/24）+ demand 与价格的同期/跨区相关。
- **用途**：为"图结构先验"（特征交互图/跨区因果边，GAFE 模块）提供数据支撑，判断跨区信息是否可作为图边。

---

## 10. `src/make_evidence.py` —— 证据文档自动生成

- **干什么**：**不重新训练**，只从 `results/bech_*.json` + `bech_ablations.csv` + `bech_peers.csv` + `bech_peer_gain_audit.csv` 自动拼装论文级证据文档（协议/主表/负价分支/安全表/负对照/消融/同行对照/审计/不对称性/局限/待办）。
- **输出**：`experiments/01-main-matrix/results/bech_matrix_flat.csv` + `docs/paper_prep/04_BECH公开数据基座无关性证据.md`（注：该目标路径当前在仓库中不存在，需确认 docs 结构后重跑）。

---

## 11. figures/ —— 论文用图

- `figure-1-backbone-agnostic.svg` —— 主矩阵：5 基座跨数据集 MAE 增益（基座无关性）。
- `figure-2-ablation-ladder.svg` —— 消融阶梯 A0→A4 单调增益。
- `figure-3-peer-comparison.svg` —— 同行对照（MAE↓ vs 改动点占比 / 最坏伤害）。

---

## 12. 结果文件地图

| 文件 | 来源脚本 | 内容 |
|---|---|---|
| `00-data-exploration/results/ALL_price_characteristics.*` | examine_all_price_datasets.py | 全数据集极端性统计 |
| `00-data-exploration/results/regime_shift_evidence.*` | regime_shift_evidence.py | 制度漂移逐年实证 |
| `01-main-matrix/results/bech_{DS}.json` | run_bech_matrix.py | 逐组合全量结果 |
| `01-main-matrix/results/bech_matrix_evidence*.md` | run_bech_matrix.py（分片 s1–s6 + smoke） | 主表/负价/安全表碎片 |
| `01-main-matrix/results/bech_matrix_flat.csv` | make_evidence.py | 扁平主矩阵 |
| `02-ablations/results/bech_ablations.csv` | run_ablations.py | 消融 + 敏感性 + no-SCARR |
| `03-peer-comparison/results/bech_peers.csv` | src/run_peer_baselines.py | 18 组合 × 11 方法 |
| `04-gain-audit/results/bech_peer_gain_audit.csv` | src/audit_peer_gain.py | 增益分解 + 泄露审计 |
| `05-episode-audit/results/*` | run_episode_audit.py | 事件片段审计 |

---

## 13. 已知问题与注意

1. **孤儿脚本（仓库重组后遗症）**：`01-main-matrix/run_bech_matrix.py` 与 `02-ablations/run_ablations.py` 只把自己的目录加进 `sys.path`，但 `common.py`/`backbones.py`/`bech.py` 现在在 `src/` 下——**直接运行会 `ModuleNotFoundError: No module named 'common'`**。`05-episode-audit/run_episode_audit.py` 已正确适配（插入 `ROOT/src`）。重跑前需给前两者补上 `sys.path` 或改为 symlink。
2. `docs/paper_prep/` 当前不存在（重组后），`make_evidence.py` 的输出目标路径失效，重跑前需确认新文档结构。
3. `bech_matrix_evidence.md`（无后缀）是早期单条碎片；完整证据需看 `_s1`–`_s6` 或重新 `make_evidence.py` 汇总。
4. 单切分 + 单种子（seed=0），尚无跨种子/跨滚动原点的方差估计（局限 6）。
5. 5 个基座是**自建轻量实现**，绝对精度不代表 SOTA；作用是提供异构误差结构检验基座无关性（论文必须明确此区分）。

---

## 14. 一句话总结每条实验的定位

| 实验 | 回答的问题 | 一句话结论 |
|---|---|---|
| 00 数据探索 | 选题立得住吗？ | 制度漂移真实存在：经典基准近零负价 → 当代 NEM 达 24.6% |
| 01 主矩阵 | 基座无关吗？ | 9×5=45 组合：16 触发 / 29 弃权，高负价市场 5 基座全部显著获益，无负价市场 bit-exact 0 |
| 02 消融 | 哪个部件有效？ | A0→A4 单调 +0.22%→+1.69% 零退化；去掉 SCARR 精度与安全**同时**变差 |
| 03 同行对照 | 比标准后处理强在哪？ | 不是 MAE 冠军，而是唯一"零退化 + 漏判率显著↓ + 最坏伤害低一个数量级"且**可叠加** |
| 04 增益审计 | 对方大增益是真是假？ | 无泄露；36.3% 来自数据新鲜度，仅 0.5% 来自特征集 |
| 05 事件审计 | 24 点单元合理吗？ | 负价事件中位 4–7h、≥4h 占六成以上；BECH 提升事件召回 10–16 点 |
