# P1 第三阶段 — 总结果记录报告(国内数据集 + 梯度分配 + 增量数据)

- 日期: 2026-08-14
- 范围: D1(国内数据接入/准入)→ D2(国内基准评估)→ T4(难度软权重采样)→ T5(国内混合比例网格)
- 触发: 用户要求"第三阶段做 T4、T5;补一个国内省份数据集实验(把各省份特征加进实验)"。T2 已 REJECT(equal → full_coverage 配对 3/3 退化),T0 等域采样仍为训练范式基线。
- 主配置(T0)**: 12 国际源域 × HOSTS4,等域采样,`TRAINER_CMP_20260814_seeds012` vA heads —— **全程未变**。
- 状态: 第三阶段全部完成。**主配置 T0 保持不动**;国内数据是"冻结头迁移正、入训混合有条件"。

---

## 0. 一句话结论

第三阶段用 **2 条正交的实验轴**回答了"国内数据怎么用"与"梯度怎么分":

| 轴 | 实验 | 结论 | 对主配置 |
|---|---|---|---|
| **国内数据 = 冻结头转移** | D2 | **DOMESTIC_TRANSFER_POSITIVE**: 40 cell 中 37 帮助,宏平均 correction **+13.9%**(MAE 109.3→91.3) | 不动(国内只做评价域) |
| **梯度分配** | T4 | **REJECT / REJECT**: 上移 NEM 梯度(host_s1r_mae +0.0074 / inv_nbatch +0.0085)3/3 seed 一致更差 → 等权是局部最优 | 不动 |
| **国内数据 = 增量训练数据** | T5 | **INCONCLUSIVE**(r 不一致): r=0.15 国外中性+国内改善(满足 doc KEEP), r=0.30 国外明确负迁移 → 低比例有条件、高比例污染 | 不动 |

三条都指向同一结论:**T0 等域采样 + 12 国际源域的主配置是当前数据下的稳健最优点**,国内数据既不是无条件的资产、也不是无条件的污染。

---

## 1. 实验链(每步触发逻辑)

```
D1 数据接入+准入(10/10 PASS)
   └─> D2 国内基准(冻结头迁移, 37/40 帮助)
         └─> 触发 T5"国内数据入训是否有增量价值"
   └─> T2 REJECT 教训(NEM 重复是必要正则)
         └─> T4 反方向验证(给 NEM 更多梯度)
               └─> REJECT -> T5 国内域用等权(§5 衔接)
```

## 2. 各实验结果

### 2.1 D1 — 国内数据接入 + 准入审计(前置, 10/10 PASS)

- **机制缺口修复**: 山东之外 4 省(宁夏/甘肃/陕西/青海)此前**完全没有加载器** —— 不是特征进不去,是数据根本没进管线。`load_province` 通用加载器 + `PROVINCES` 注册表(8 key)补上,host 层经 `build_tabular` 消费各省自己的 `fc_*`/`act_*` 特征(schema 异质,每省独立 host,无需跨省对齐)。
- **信息泄露护栏**(用户明确要求): `assert_no_future_leakage`(查每列 vs y_{t+k},k≥1 近完美相关/精确碰撞)+ `verify_forecast_vs_actual`(预测值不得是回填实际值)。10 个 (省份×模式) 全部通过;山东旧缓存复检无未来价格泄漏。
- **数据规模如实标注**: 山东 S1R 166d / 甘肃 56d / 陕西 44d / 宁夏 18d(WARN 偏薄)/ 青海 17d(WARN 偏薄)。负价: 仅山东(11~13%),其余 4 省 **0%** → 按"严禁"规则不报 negative-price。
- 产物: `province_admission.py`、`results/domestic/admission_report.csv`、40 个 host 缓存。

### 2.2 D2 — 国内基准评估(冻结头迁移, DOMESTIC_TRANSFER_POSITIVE)

- 设置: round-1 vA seed0 冻结头 × 10 国内模式 × 4 host = 40 cell,`weighted_mean` 读出,S4 冻结确认(标签从未用于训练/选择)。
- 结果: **37/40 cell 修正帮助 host**,宏平均 MAE **109.3 → 91.3(−16.4%)**,逐 cell 平均 correction **+13.9%**。

| 模式 | 帮助 | macro_mae | correction | 备注 |
|---|---|---|---|---|
| shandong_DA | 4/4 | 91.25 | +26.6% | 含负价,8 cell 负价 MAE 已记 |
| shandong_RT | 4/4 | 140.32 | +20.6% | |
| ningxia_DA/RT | 4/4 ×2 | 55.93 / 77.22 | +18.6% / +8.2% | S1R 18d*,WARN |
| gansu_DA/RT | 4/4 ×2 | 67.80 / 84.18 | +13.5% / +9.8% | |
| shaanxi_DA | 4/4 | 101.08 | +5.7% | |
| **shaanxi_RT** | **1/4** | 107.07 | **−0.9%** | 唯一系统性例外(host 已强) |
| qinghai_DA/RT | 4/4 ×2 | 97.20 / 91.14 | +15.5% / +21.4% | S1R 17d*,WARN |
| **宏平均** | **37/40** | **91.32** | **+13.9%** | |

- per-host: MLP +25.4%(host 基线 146.4 最弱,修正空间最大)/ Linear +11.9% / PatchTST +9.8% / LSTM +8.5%。
- 诚实边界: shaanxi_RT 反例如实呈现(不宣称 10/10);宁夏/青海参考薄,幅度读作方向;国内特征仍只进 host(修正头 D_VALUE=0,schema-agnostic),D2 证明"国际训练通用头 + 各省特征 host"的组合在国内有效。
- 交叉验证锚点: 本文档 §2.4 的 T5 r=0 seed0 国内 40 cell 与此表**逐 cell 完全一致(40/40)**,两套独立脚本互相印证。

### 2.3 T4 — 难度软权重采样(REJECT / REJECT)

- 假设 H4: 低数据量/高难度域(NEM)在等权采样下梯度不足,按宿主侧难度统计上调软权重(预算 12×K 不变,只改分配)应同时改善 NEM 且不伤 DE/PJM。
- 结果(主指标 = S2V 宏平均 CRPS 配对 vs T0):

| arm | seed0 | seed1 | seed2 | 宏平均 Δ | 判定 |
|---|---|---|---|---|---|
| host_s1r_mae(training-only) | +0.0096 | +0.0061 | +0.0065 | **+0.0074** | REJECT |
| inv_nbatch(target-free) | +0.0108 | +0.0067 | +0.0079 | **+0.0085** | REJECT |

- **机制闭环**(逐域拆解): 权重"如预期"起作用 —— NEM 被上调到 1.7~2.9× 后 3/4 域改善(证明 NEM 响应更多梯度,与 T2 教训互证);但 DE/PJM 被下调到 ~0.4× 后 **8/8 全部退化**,损失系统性大于 NEM 改善。
- **结论**: T0 等权(NEM 33%)是**梯度分配轴的局部最优** —— T2 压到 8.3% 变差,T4 提到 ~50%+ 也变差,双向证伪,该轴不再正交方向可试。
- 诚实披露: host_s1r_mae 读 S1R 目标价(training-only,§5.2 已标注);其失败削弱"训练-only 难度信息"的价值主张。

### 2.4 T5 — 国内数据混合比例网格(INCONCLUSIVE, r 不一致)

- 设置: 12 国内训练域(shandong_DA/gansu_DA/shaanxi_DA × 4 host)以梯度份额 r 混入 12 国外源域,3 seeds × r∈{0.15, 0.30},648 个 S4 评估。预算 24×K=24×11=264/epoch = T0 精确保留。
- 国外 32-cell transfer(ΔMAE vs T0 同 seed):

| r | seed0 | seed1 | seed2 | 宏平均 |
|---|---|---|---|---|
| 0.15 | +1.17% | −0.84% | +0.07% | **+0.13%**(≤+0.5% 带,无负迁移) |
| 0.30 | +7.85% | −1.05% | +2.24% | **+3.01%**(>+0.5%,2/3 seeds 超带) |

- 国内 40-cell holdout: r=0.15 → seed1/2 **−3.0%/−2.4%**(2/3 seeds 改善);r=0.30 → 类似,不随 r 增加。
- **判定**: r=0.15 满足 doc §3 KEEP 条件(国外宏平均无负迁移 + 国内 ≥2/3 seeds 改善);r=0.30 触发 REJECT 条件(国外 ≥2/3 seeds >+0.5%)。keep_r ∩ reject_r → "r 不一致" → **INCONCLUSIVE**,主配置不动。
- **机制画像**:
  - 国外伤害集中在 **LAGO_NP(挪威,最温和市场,MAE~3)**: r=0.30 时 seed0/2 相对 +18~25% —— 国内高波动分布污染温和市场参考池(scale-free 分布差异)。
  - 国内改善**含未见模式**(shandong_RT/ningxia/qinghai 等 28 cell,unseen −0.2/−2.9/−2.1%)→ 泛化而非记忆。
  - seed0 从不改善(per-seed 不一致源;国外 seed0 也最伤)。
  - 国外-only S2V 同度量: r=0.15 ≈ T0(+0.0004),r=0.30 略差(+0.0011),与 S4 方向一致。
- 诚实披露: 判据操作化修正(KEEP 国外条件按 doc 用"宏平均"而非 per-seed 全过,初版误判 REJECT → 修正为 INCONCLUSIVE,数字不变);交叉验证 40/40 与 D2 一致。

---

## 3. 跨实验综合洞察

1. **梯度分配轴双向证伪(T2+T4)**: 等权采样是鲁棒最优点。任何"按数据量/难度重分配梯度"的方案都被数据否决 —— 低数据量域(NEM)的重复曝光是必要正则,不是要修的失衡。
2. **增量数据轴(T5)**: 国内数据入训是"**有条件资产**"。低比例(r=0.15)不伤国外(宏平均 +0.13%)且改善国内(含未见模式);比例升高(r=0.30)污染国外温和市场。增量数据必须按域分配谨慎引入。
3. **国内 transfer 的双重证据(D2+T5)**: 冻结头直接迁移(D2)是清晰的正(+13.9%);入训混合(T5)只在低 r 成立。论文应写"修正头对未见市场(含国内)有天然迁移力,但增量训练数据需低比例引入"。
4. **主配置稳健性**: T2/T4/T5 三条轴(均衡化/加权/增量)都未能打败 T0 等域 12 源域 —— 第一轮点读出优势、R1B 泛化结论均不受影响。

## 4. 对论文的意义(段落素材)

- **国内业务验证**: D2"国际训练通用头在国内 37/40 cell 改善" → 强化修正头 domain-agnostic 主张;shaanxi_RT 反例作诚实加分项。
- **训练范式**: T4 论文写"梯度分配轴双向证伪,等权采样是鲁棒最优";T5 写"增量数据有条件正(低 r 资产/高 r 污染),温和市场对参考池污染最敏感"。
- **不写**: 不写"国内数据无条件提升泛化"、"重分配梯度能打败等权"。

## 5. 产物清单(git + 结果目录)

**commits**(分支 exp/r1b-screening-20260813):
- D1/D2: `be986f2`(common.py 省份加载/泄露护栏、host_cache.py、province_admission.py、p1_domestic_eval.py + 文档)
- T4: `4da3ebf`(p1_t4.py + 假设/verdict 文档)
- T5: `1dd1aa0`(p1_t5.py + collect_domain info 复用 + 报告骨架)、`f269506`(verdict 修正 + verdict/hypothesis 文档)
- 本总报告: 当前 commit

**结果文件**(`results/`,gitignore 不追踪,结论以 docs 为准):
- `results/domestic/`: admission_report.csv、domestic_s4_metrics.csv/json、domestic_summary.json(D1/D2)
- `results/P1_T4_be986f2/`: verdicts.json、s2v_paired_comparison.csv、per_domain_comparison.csv、每 arm 训练报告(T4)
- `results/P1_T5_1dd1aa0/`: manifest.json、r{0.15,0.30}/ 训练报告+heads、eval_rows.json(648 行)、transfer_matrix.csv、summary.json(T5)
- `results/cache/{mode}/{host}/`: 40 国内 host 缓存(seg.json 含 split_hash + feature_schema_hash 可复检)

**文档**(`docs/训练文件夹/对比实验/`):
- D2: `hch_v2_d2_domestic_benchmark_verdict_report_v0.1_2026-08-14.md`
- T4: `hch_v2_p1_t4_soft_weight_hypothesis_prompt_v0.1_2026-08-14.md` + `hch_v2_p1_t4_soft_weight_verdict_report_v0.1_2026-08-14.md`
- T5: `hch_v2_p1_t5_mixing_grid_hypothesis_prompt_v0.1_2026-08-14.md` + `hch_v2_p1_t5_mixing_grid_verdict_report_v0.1_2026-08-14.md`
- 本总报告: `hch_v2_phase3_results_master_report_v0.1_2026-08-14.md`

## 6. 未解决 / 后续

1. **温和市场(型 LAGO_NP)对参考池污染的脆弱性未建模** —— 机制清楚,无预防手段(不改 CAGM/DVG 核心、不加市场类别门)。
2. **seed0 的国内敏感性未解释** —— 同协议下 seed0 国外最伤且国内不改善,其余 seed 相反。
3. **负价仅山东一个域** —— 无法做国内负价处理的稳健统计。
4. **可选后续(不承诺)**: 国内专用修正头(不混入通用头);按市场类别分层评估 transfer 风险。
5. **T6 维持暂缓**(host 消费外生,HCH 核心不复活)。

---

## 附: 主判据方法一致性与数据可复现性

- 所有配对评估: 同 seed、同 trainer 协议、同 best_epoch 规则(宏平均 S2V 选 checkpoint,禁 S4 调参)。
- S4 冻结确认标签从未用于训练/选择/超参。
- 交叉验证: D2(独立脚本)与 T5 r=0 seed0 国内 40 cell 逐 cell 完全一致 → 无实现漂移。
- 判据操作化偏差(T4、T5 各一次)均已披露并修正到与 doc 字面一致,verdict 数字不变。
