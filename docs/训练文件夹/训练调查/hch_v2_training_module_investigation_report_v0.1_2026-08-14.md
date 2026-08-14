# HCH-v2 训练与模块问题调查报告 v0.1(2026-08-14)

> 驱动文档:`hch_v2_training_module_investigation_prompt_v0.2_2026-08-14.md`
> 性质:**调查/证伪/排序**。按 v0.2 §9,本轮**未修改 src/ 生产代码**,未覆盖旧实验结果,未把 holdout 加入训练。
> 版本基线:架构 v0.4 / 数学链 v0.3(`math_core_v0.3`) / 分支 `exp/r1b-screening-20260813` / HEAD `8b81364`。

---

## 8.1 已核验材料与代码路径

| 类别 | 条目 | 说明 |
|---|---|---|
| 版本 | HEAD `8b81364`(Paper Gate YELLOW_READOUT, hard stop)· `architecture_version="v0.4"` | `hch_v2_bundle.py:28`;git |
| 核心源码 | `src/iah_candidate.py`、`iah_crps_loss.py`、`universal_trainer.py`、`hch_v2_context.py`、`hch_v2_pipeline.py`、`_legacy/` | 全部逐行核验 |
| 动作链件 | `src/double_event.py`、`query_replay.py`、`dvg_calibrate.py`、`w1_retrieval.py`、`s1_rank.py` | 数学引用 `math_core_v0.3` |
| 正式 runner | `experiments/08-hch-v2/r1b_generalization_screen.py`(12 源域主筛选) | R1B/T0-1 确认唯一训练入口 |
| 面板/链 runner | `r1b_stage2a_panel.py`(eval_panel_domain)· `r1b_stage2d_action_chain.py`(A0/A1/A2) | §2.1/§2.3 核验对象 |
| 路径 B | `experiments/08-hch-v2/r1a9_action_calibration.py`(collect_domain/evaluate_days/dvg_and_s4)· `r1a11_prequential_calibration_router.py` | Paper Gate 实际动作链 |
| 点读出 | `experiments/08-hch-v2/_final_point.py`(P0-A replay)· `r1a9_action_calibration.py:472 point_metrics`(旧,仍被 r1a9/r1a10/r1a11 调用) | 三份实现,见 §2.1.4 |
| 矩阵 | `experiments/09-paper-gate/runner/run_matrix.py`(B3/B4)· `run_smoke_matrix.py` | B5 HCH 走 `_final_point.final_metrics` |
| 数据 | `src/common.py`(DATASETS/load_shandong/build_tabular)· host cache 18 域(`results/cache/`) | 含 shandong_DA/RT |
| 实验状态 | R1B 7 阶段 verdict、Paper Gate results、T0-1 审计(两条动作链决策点) | 见文献/附件 |
| 诊断 | `experiments/08-hch-v2/training_investigation_diag.py`(本调查新增,§2.1 toy + §3/§4)· `training_investigation_trainer_cmp.py`(§5) | 只读已有 head(`P0A_RERUN`),不改 src/ |

---

## 8.2 当前问题清单(按 P0-P3)

### P0-1 【评估层】候选点预测从未被真正评估——`x_identity` = 宿主原值

- **代码证据**:
  - `r1b_stage2a_panel.py:159-160` `cand_pred = torch.where(sv.unsqueeze(-1).unsqueeze(-1), out["x_identity"], host)`
  - `src/iah_candidate.py:136` `x_identity = s_safe32 * sinh(z0)`;且 `z0 = asinh(host_raw/s)`(L103)⇒ `s·sinh(z0) ≡ host_raw`(scale-valid 时)。scale-invalid 天全部 fallback `host_raw`(L150-152)。
  - 因此 `cand_mae`、`mae_rel_deg`、`safety`(stage2a_panel)、`cand_mae_panel`(stage2d)、`pan`(run_matrix)全部测的是**宿主原值**。
- **影响当前结论**:是。R1B 报告"MAE 恒等 / 安全零退化"实为**评估层未接候选输出**,不是候选能力结论。
- **类别**:评估 bug(非训练问题)。

### P0-2 【读出层】三原子分布的位置信息(m⁻/m⁺)从未直接进入最终点预测

- **代码证据**:
  - 候选输出含 raw atoms `x_down = s·sinh(z0−m⁻)`、`x_up = s·sinh(z0+m⁺)`(iah_candidate.py:137-138),但**无 `x_point`/`z_point`**,不存在 μ_R = w⁺m⁺−w⁻m⁻ 或加权中位数任何分布→点读出函数。
  - 唯一最终输出 `_final_point.py:44` `x_final = s·sinh(z0 + π_eff)`,`π_eff = π` **仅当 DVG 释放**否则 0。π 由动作链(g̃=mm·s̃ → double_event_proposal → form_final_pi,r1a9:403-406)产生。
  - **toy test 证实**(本调查):π=0 时 `x_final ≡ host` 精确(evaluator_pi0_mae_equals_host_mae=True)。
- **影响当前结论**:是。Paper Gate YELLOW_READOUT 的架构根因:CRPS(分布)改善与 final point(宿主+稀疏动作授权)之间隔着"动作授权开关"。
- **类别**:读出/模块接口问题(非训练未收敛)。

### P0-3 【评估层】三份点读出实现并存

- `_final_point.final_point_metrics`(官方,stage2d/run_matrix)· `r1a9.point_metrics`(旧,等价格式,仍被 r1a9/r1a10/r1a11 调用)· `raw_metrics`(baseline)。
- 官方版对全部小时重放;r1a9 版用 vm mask。语义等价但**双维护易漂移**。
- **类别**:工程债(不影响当前结论,影响复现)。

### P1-1 【训练器】逐域 sequential update,无跨域梯度累积

- `universal_trainer.py:246-272` schedule 打乱后逐 batch `opt.step()`,无宏域累积;`BATCH_DAYS=16`(r1a_run.py:85),最后不足 16 天的块与完整块**等权**。
- K=median(n_batches) 截断长域/重复短域(§5.2 对照目标)。
- **类别**:训练组织方式,§5 Version A vs B 对照。

### P1-2 【数据】`domain_det` 无内容绑定断言(T0-1 §2.1.4 待补)

- `r1a_run.py:37` `domain_det=None` fallback 到 mutable buffer,存在静默填零路径;仅有 shape 检查无"det ↔ market:host"绑定断言。

### P2-1 【动作链】gains 信号两条来源并存(T0-1 决策点)

- 路径 A(权威 `predict_s4`):CAGM 邻居 gains 回放;路径 B(Paper Gate 实际):候选 m × calibrator 缩放,无 CAGM。数学链(v0.3)一致,差异在 gains 证据来源。统一方向待主调研方定(§2.3)。

### P2-2 【训练】配置双处重复定义

- `r1a_run.py` D_MODEL/D_SIG/D_VALUE 与 `r1b_generalization_screen.py` 复制维护,易漂移。

---

## 2. 第一部分:端到端代码审计(§2 明细)

### 2.1 候选输出审计

**结论:候选分布是"活的"(携带位置信息),但 point 层从不消费它。**

核验 5 项:

| 审计问题 | 结果 |
|---|---|
| x_identity/x_down/x_up 是否被当 point forecast | `eval_panel_domain` 用 `x_identity`(=宿主);`x_down/x_up` 无任何评估使用 |
| 是否有 raw-space atom | 有:iah_candidate.py:136-138(scale-valid 天) |
| 是否有明确分布→点预测函数 | **无**(无 μ_R/加权中位数/任何位置读出) |
| 点预测是否仍等于 host identity | DVG 未释放时**精确等于**(toy test 证实) |
| m⁻/m⁺ 是否真正进入最终预测 | 仅经动作链间接进(释放日的 π);非释放日不进入 |
| CRPS 改善与 MAE/sMAPE 改善是否被混为一谈 | R1B 主证据=ΔCRPS(分布),Paper Gate=final MAE/sMAPE,中间缺分布→点评估段 |

**§2.1 toy test(host=0, m=0.5/0.3, w=0.40/0.10/0.50):**
- 五种读出全部产生不同 raw 预测(pairwise |Δ| 0.37–10.64);identity 与 host 差 0.0。
- evaluator π=0 时 `mae == host_mae` 精确 → **candidate m 永不进入 x_final 除非动作授权**。
- 证据:`results/TRAINING_INVESTIGATION_FULL/report.json`(toy_test)。

### 2.2 训练链审计

| 检查项 | 当前实现 | 科学风险 | 是否改 | 证据路径 |
|---|---|---|---|---|
| 域单位 | (market, host) 对,`DomainBatch` 每域 | 无 | 否 | universal_trainer.py:35-55 |
| 每域 update 数 | K=median(N_g)/域/epoch,等域采样 | 长域截断/短域重复,曝光不均 | §5 对照 | universal_trainer.py:203-205,235-245 |
| 最后小 batch 权重 | 每 batch 一次 step,<16 天块与完整块等权 | 末段单次出现,权重等价 | 低 | r1a_run.py:284-298 |
| loss 加权 | batch mean(B×H) | 非逐日/逐小时 | 否 | r1a_run.py:295 |
| 更新方式 | schedule 打乱逐 batch sequential step | 域间振荡(无跨域累积) | §5 B 对照 | universal_trainer.py:246-272 |
| S2V 曲线 | 每 epoch per-domain S2V 记录;train 只 epoch mean | 无逐日 train 曲线 | 否 | universal_trainer.py:284-294 |
| checkpoint | best macro S2V 状态保存并 reload | 无(正确) | 否 | universal_trainer.py:299-311 |
| 健康度 | mass entropy/shift alive/p50/p95/grad norm/NaN/scale-invalid 全记录 | 无 | 否 | universal_trainer.py:110-163 |
| 配置重复 | D_* 在 r1a_run 与 screen 双处复制 | 漂移 | 是(T0-2 断言) | r1a_run.py:60-70 |

### 2.3 Action-chain 审计

| 问题 | 答案 |
|---|---|
| 正式实现 | 权威 `src/hch_v2_pipeline.py::predict_s4`(含 CAGM 邻居 gains 回放)= v0.3 数学链完整实现 |
| R1B/Paper Gate 实际跑哪条 | **路径 B**:r1a9(collect_domain→evaluate_days→dvg_and_s4)+ r1a11 router(C0/C3)。GATING_HURTS 结论基于此条 |
| 是否"文档 v0.3 代码跑旧链" | 数学链一致(均 double_event + 整日 LCB);差异=**gains 来源**(CAGM 回放 vs calibrator 缩放)。无"旧链",有"两条 gains 变体" |
| 是否先协议对齐 | 是。T0-1 已标记待决点(a/b/c),统一前不得混表 |

重点审计项:query dose 用当天 mm/mp(r1a9:403-406,无历史邻居)→ 至多一 Down 一 Up(double_event Kadane)✅ · 整日授权(dvg_and_s4 逐日 released)✅ · S3M(memory+val)/S3C/ S4(dev)严格分离(collect_domain block_of)✅ · DVG q 在 S3C 冻结前完成(dvg_and_s4:418-425)✅ · LCB 仅 S4 执行 ✅ · **GATING_HURTS 归因:EPEX_FR:PatchTST 短证据(50 S3M 日)C3 误授权=证据不足+selector 门槛问题,非 candidate**(与 R1B Stage-2E 结论一致)。

---

## 3. 第二部分:分布到点读出独立调查(§3)

**方法**(`training_investigation_diag.py`,只读 P0A_RERUN universal head):
五种读出 identity / raw weighted mean / raw weighted median / 期望 sMAPE 数值 Bayes action / shrink z0+λ(w⁺m⁺−w⁻m⁻),λ∈{0.25,0.5,0.75,1.0}。
- S2V:读出选择与诊断(candidate 校验段);S4:冻结评估(不碰 S4 调参);S3M/S3C 只用于 action calibration(本调查不动)。
- 域:6 个(LAGO_DE:Linear/MLP、LAGO_PJM:MLP、NEM_SA1:MLP、NORD_DK1:Linear、GEFCOM14P:Linear)

**S2V 读出选择结果**(best readout by MAE;identity ≡ host,因 x_identity 就是宿主原值):

| 域 | identity MAE | **best readout** | best MAE | Δ vs identity | source/holdout |
|---|---:|---|---:|---:|---|
| LAGO_DE:Linear | 6.569 | shrink_0.75 | 5.929 | **−9.7%** | source |
| LAGO_DE:MLP | 7.424 | shrink_0.75 | 5.605 | **−24.5%** | source |
| LAGO_PJM:MLP | 2.361 | shrink_1.0 | 2.211 | **−6.4%** | source |
| NEM_SA1:MLP | 132.3 | shrink_1.0 | 80.48 | **−39.2%** | source |
| NORD_DK1:Linear | 26.73 | shrink_0.5 | 25.64 | **−4.1%** | **holdout** |
| GEFCOM14P:Linear | 12.13 | weighted_mean | 9.889 | **−18.5%** | **holdout** |

**S4 冻结评估**(readout 参数在 S2V 上选,冻结在 S4 执行;S2V Δ 与 S4 Δ 基准不同段,不可直接跨段比较):

| 域 | S4 host MAE | readout | S4 MAE | Δ vs host |
|---|---:|---|---:|---:|
| LAGO_DE:Linear | 6.249 | shrink_0.75 | 6.205 | **+0.70%** |
| LAGO_DE:MLP | 8.417 | shrink_0.75 | 7.111 | **+15.5%** |
| LAGO_PJM:MLP | 4.210 | shrink_1.0 | 4.215 | **−0.11%** |
| NEM_SA1:MLP | 135.9 | shrink_1.0 | 68.96 | **+49.2%** |
| NORD_DK1:Linear | 32.78 | shrink_0.5 | 29.08 | **+11.3%** |
| GEFCOM14P:Linear | 5.686 | weighted_mean | 5.326 | **+6.3%** |

**最终判定:READOUT_SUFFICIENT**。
- 6/6 域 S2V 读出全部改善(−4%~−39%),**5/6 域 S4 冻结改善**(唯一例外 LAGO_PJM −0.11%,该域 host 已极准 MAE≈2.2,S2V best 也只是 −6.4%,无可修正空间)。
- **NEM_SA1:MLP 读出使 S4 MAE 减半(135.9→69.0,+49%)**——难域(高波动)候选位置信息的价值最大。
- **跨域转移有效**:NORD_DK1、GEFCOM14P 均为 holdout 域(不在 12 source 训练域),读出 +11.3%/+6.3% → universal head 学到的是**可迁移的位置修正**,不是记忆 → 与 R1B NATIVE_GENERALIZATION_SUPPORTED 一致。
- 读出 shrink 为 λ 对 (m⁺−m⁻) 的连续缩放,零额外参数、不违反数学链(v0.2 §1"从已有三原子分布导出点预测"属允许候选)。
- **LOCATION_CAPACITY_INSUFFICIENT 被否定**:无需改 IAH 几何(z0+δ 一类),点指标差是"读出从未被连接"而非"位置不存在"。

---

## 4. 第三部分:候选剂量与动作方向(§4)

**方法**:对每个小时 `residual_z = zY − z0`,predicted direction = sign(m⁺−m⁻);分 high_tail/neg_price/normal 期统计;frac_dose_neg_gain = 主导方向剂量小时中动作增益为负的比例。

**S2V 诊断结果**(6 域全量):

| 域 | dir_acc | corr(m⁺,+残差) | corr(m⁻,−残差) | 高尾 corr(m⁺,+) | 负价 corr(m⁺,+) | 负价 mean(m⁺) | frac_dose_neg_gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| LAGO_DE:Linear | 0.640 | **0.774** | −0.011 | **−0.212** | 0.435 | 1.273 | 0.644 |
| LAGO_DE:MLP | 0.631 | **0.819** | 0.133 | 0.048 | 0.677 | 1.599 | 0.673 |
| LAGO_PJM:MLP | 0.605 | 0.272 | 0.263 | 0.392 | 无负价 | — | 0.793 |
| NEM_SA1:MLP | **0.824** | 0.283 | 0.260 | 0.083 | **0.837** | 0.459 | **0.414** |
| NORD_DK1:Linear | 0.584 | 0.187 | 0.074 | −0.069 | −0.811 | 0.145 | 0.663 |
| GEFCOM14P:Linear | **0.741** | 0.197 | 0.296 | 0.178 | 无负价 | — | 0.599 |

**跨域规律**(修正版——LAGO_DE 两域局部现象不得推广为 universal 结论):
1. **方向准确率 6/6 域 >0.5**(0.58–0.82),候选确实携带方向信息;NEM_SA1 最高 0.82。
2. **`corr(m⁺, +残差)` 在 LAGO_DE 强(0.77–0.82),其余域中等(0.19–0.28)** → m⁺ 上行幅值编码在 LAGO_DE 最清晰,非全域强。
3. **`corr(m⁻, −残差)` 全域 0.07–0.30**:LAGO_DE 约 0,但 GEFCOM14P 0.30、LAGO_PJM 0.26 → **"m⁻ 无下行幅值编码"仅限 LAGO_DE,不成立为 universal 规律**(先前结论修正)。
4. **高尾反相关只在 LAGO_DE:Linear(−0.21) 与 NORD_DK1(−0.07)**:极端尖峰上 m⁺ 方向信息衰减是**域相关**的,不是全部域。
5. **frac_dose_neg_gain 全域 0.41–0.79(中位 ~0.66)**:剂量小时多数动作增益为负是跨域一致现象 → **"该不该释放、释放多大"是比"方向对不对"更弱的环节**;LAGO_PJM 最严重 0.79,而 NEM_SA1 最健康 0.41(恰与它 dir_acc 最高一致)。
6. NEM_SA1 负价期 corr(m⁺,+)=0.84 且 mean(m⁺)=0.46(克制):负价→上行回归在 NEM_SA1 上既准又不过度;LAGO_DE 则 mean(m⁺)=1.27–1.60(过度)。

**读出 vs 动作一致性**:§3 显示连续 shrink 读出(soft)在 5/6 域改善 point,§4 显示硬释放剂量(动作层)多数负增益——**连续读出比硬动作更鲁棒地利用 m**。这直接支持"读出层消费 m(而非依赖动作层)"的修复方向,且与 Gating/LCB 防护不冲突。

**动作层六分解标签(4.2)**:candidate distribution quality=S2V CRPS 好、读出改善(§3) / directional proposal=方向>0.5 但幅度编码域相关 / interval selection=double_event 至多一 Down 一 Up 结构正确 / C0-C3 校准=R1B 结论,§5 待补 / DVG=LCB 保守 / final realized value=见 R1B GATING_HURTS。**标签:LOCAL_ACTION_OR_PROPOSAL_LIMITED(动作/剂量层,非 candidate training failed)**。

---

## 5. 第四部分:跨域训练方式(§5)

**对照**(`training_investigation_trainer_cmp.py`):Version A = 现网 UniversalCoreTrainer 逐 batch sequential step(复用 R1B 正式 `train_candidate`,byte-identical);Version B = macro-domain gradient accumulation(每 macro-step 每域一 batch,loss/|G| 累积,统一 step)。**只改一个因素**:epochs=12、lr=3e-4、wd=1e-4、clip=1.0、head 结构、checkpoint=macro S2V 全部一致。12 source 域 = 3 市场 × 4 host。

**seeds 0,1,2 结果**:

| 版本 | seed | best macro S2V | worst@best | macro ΔCRPS | 域间方差 | worst ΔCRPS |
|---|---:|---:|---:|---:|---:|---:|
| A(现网) | 0 | **0.22750** | 0.6761 | **−0.1580** | 0.02914 | −0.0111 |
| A | 1 | **0.22880** | 0.6788 | **−0.1567** | 0.02721 | −0.0126 |
| A | 2 | **0.22376** | **0.6593** | **−0.1617** | 0.02949 | −0.0143 |
| B(accum) | 0 | 0.23568 | 0.6787 | −0.1498 | **0.02599** | −0.0098 |
| B | 1 | 0.23747 | **0.6722** | −0.1480 | **0.02571** | −0.0084 |
| B | 2 | 0.23611 | 0.6752 | −0.1494 | **0.02580** | −0.0090 |
| **A mean** | | **0.22669** | **0.6714** | **−0.1588** | 0.02861 | **−0.0127** |
| **B mean** | | 0.23642 | 0.6754 | −0.1491 | **0.02583** | −0.0091 |

**结论:H4 明确否定,跨 3 种子一致**:
1. **A(sequential)宏观 CRPS 每种子都优于 B**(+0.008~+0.012,均值差 0.0097 ≈ 4%),差距远大于 seed 间噪声(±0.003)→ 非偶然。
2. A 的 macro ΔCRPS(−0.159 vs −0.149)与 worst ΔCRPS(−0.013 vs −0.009)也一致更优。
3. **B 唯一稳定优势是域间方差略低**(0.0258 vs 0.0286,更均衡),但**未转化为 worst 域或宏观改善**(worst 仅 seed1 略好 0.006,其余输)。
4. 解读:等域采样下 sequential 让单域连续吃多个 batch,域内适配更快;macro accumulation 使每步梯度被 12 域平均,域内收敛更钝——"更均衡"以"更钝"为代价,不划算。
5. **macr-domain gradient accumulation 不建议改**(§8.4 已降级);现网 sequential update 保留。当前不构成负迁移源。

**诚实限定**:本对照保持 epochs=12、等域采样 K=median 不变,只改更新组织方式(v0.2 §5 纪律)。未验证 warmup/cosine(§5.1 仅当曲线显示仍在收敛——seed0 曲线 ep10→ep11 仍微降,但 macro 已近饱和 0.23,不足以触发延长训练改动)。

---

## 6. 第五部分:Rich covariate branch(§6)— 数据契约草案已交付

见 `docs/训练文件夹/训练调查/rich_branch_data_contract_draft.md`。要点:
- 山东 23 列全清单 + 四类角色映射;**target 角色翻转**:shandong_DA 时日日前电价=target,shandong_RT 时日日前电价=合法 KNOWN_FUTURE(任务退化为 DA→RT 价差预测),现行 `load_shandong` 丢失该路径。
- 新能源总加=风+光(派生冗余);节假日缺失不得用周末代理冒充。
- 契约:`required: host forecast + time index + scale-free history / optional: typed tokens / missing: learned-null / target 永不入当日输入 / 归一化只用训练段`。
- HCH-Core vs HCH-Rich 两正式条件论证(因果隔离 + 守住 price-only 主张边界)。

---

## 7. 文献调查(§7)

**完整核查表**:`docs/训练文件夹/训练调查/literature_review_draft.md`(29 篇,5 方向,A=9/B=4/C=6/D=5/E=5,全部链接 HTTP 200 + 标题匹配验证,无 UNVERIFIED;G&R 2007 原刊 T&F 付费墙以 Crossref 记录代,如实标注)。

**对 HCH 最相关的 5 篇**:

| 论文 | 为什么相关 | 迁移代价 | 建议 |
|---|---|---|---|
| Gneiting *Making and Evaluating Point Forecasts*(Bundesbank) | **直接命中缺口**:分布层赢≠任意点读出赢;MAE→中位数等一致评分函数 | 低,零代码 | 点读出按 Bayes act 取,不默认取均值 |
| δ-Adapter *The Forecast After the Forecast*(ICLR 2026, arXiv:2601.20280) | 与 HCH 校正头结构同族:frozen host + 输出残差 + 分布校准;O(δ) drift bound | 中 | 结构对标 + 数学保证可借鉴 |
| EasyUQ(arXiv:2212.08376) | 只吃 host 输出→输出校准分布,是 LocalCore 无训练上界 | 低 | LocalCore 对照 baseline |
| BLAST balanced sampling(KDD'25, arXiv:2505.17871) | HCH 等域采样的数据维度文献支撑 | 低 | §5 采样方法论文献 |
| *Beyond RMSE and MAE* electricity econ eval(arXiv:2511.13616) | 统计 vs 经济评价分离,支持 Paper Gate CRPS/MAE 分离判断 | 低 | 评价体系引用 |

**五方向迁移可行性**:A 能迁移(低代价,零违界);B 能迁移(中,检索类需过 leakage 审计);C 能迁移(低,工程补强);D 能迁移(中,只更新校正头为边界);E 能迁移(低,领域对标)。**无一违反当前数学边界(v0.2 §7 要求回答的"是否违反边界"全部为否)。**

**关键提醒(来自检索类文献)**:*Factorize to Generalize*(arXiv:2605.24911) 指出检索增强在平滑序列上反伤——与 R1A 检索价值 audit 结论呼应,支持 HCH"检索只在极端/稀有实例启用"的现有设计。

---

## 8.3 证伪矩阵(§3/§4 已填;§5 待跑)

| 假设 | 支持证据 | 反证方式 | 最小实验 | 当前结论 |
|---|---|---|---|---|
| H1 点指标差=训练未收敛 | R1B ΔCRPS<0 零例外 | 读出(0 参)6/6 域 S2V 改善(−4%~−39%) | §3 五读出 | **否** |
| H2 点指标差=读出错误 | 候选 m 从未进 x_final(toy test);cand_mae 测宿主原值;读出后 5/6 域 S4 改善(唯一例外 LAGO_PJM −0.11%,host 已极准) | — | §3 全域 | **是(P0-1/P0-2)** |
| H3 IAH 分布无位置修正能力 | m 非零(alive 94%);dir_acc 6/6>0.5;读出 S4 大幅改善(NEM_SA1 +49%) | m⁻ 幅值编码域相关(0.07–0.30);高尾反相关仅 2/6 域 | §4 全域 | **否**(位置信息真实且跨域迁移) |
| H4 逐域 sequential 造成负迁移 | — | Version B 对照 | §5 | **否定(seeds 0,1,2 一致:A 宏观 CRPS 每种子优于 B;B 仅域间方差略低)** |
| H5 action-chain 而非 candidate 是瓶颈 | 读出(soft)5/6 域 S4 改善 vs 硬释放剂量 frac_neg_gain 0.41–0.79;GATING_HURTS 短证据误授权 | — | §3 vs §4 对比 | **是(读出路径高效,动作路径低效)** |
| H6 外生接口缺失致山东失败 | D_VALUE=0,分支未参与;山东 0/16 FAIL | — | §6 契约(已交付) | **待 U2-Rich 实验** |

**H3 的细分修正**:不是"m 退化",而是"读出从未被消费"。m⁻ 的下行幅值编码在 LAGO_DE 缺失但在 GEFCOM14P/LAGO_PJM 存在——是**域相关**而非 universal 缺陷,不足以支撑 location shift。

---

## 8.4 修改候选排序

| 修改候选 | 预期收益 | 实现难度 | 论文风险 | 是否马上做 |
|---|---:|---:|---:|---|
| 修正 point readout/evaluator(x_identity 评估 bug,接候选输出) | 高(§3:6/6 S2V、5/6 S4 改善) | 低 | 低 | **是(P1)** |
| shrink/Bayes 连续读出(§3 已验证 shrink λ∈{0.5,0.75,1.0};建议接入 final,非只动作链) | 高 | 中 | 低 | **是(P1)** |
| macro-domain gradient accumulation | **无(§5 seeds 0,1,2 一致更差 +0.010 CRPS,降级为否)** | 中 | 低 | **否** |
| warmup/cosine | 低-中 | 低 | 低 | §5.1 曲线确认后 |
| location shift(改 IAH 几何) | **已否定**(§3=READOUT_SUFFICIENT) | 中 | 中-高 | **否**,除非未来读出审计再次失败 |
| Rich covariate branch | 未知 | 中-高 | 中 | 单独实验(§6 契约已备) |
| 新事件头/辅助 loss | 不确定 | 高 | 高 | 否(v0.2 §1 禁止) |

## 8.5 推荐执行顺序

```
P0 输出/协议审计(已完成,本报告)
→ P1 三原子点读出 + evaluator 修正(§3 证据:READOUT_SUFFICIENT;唯一反例 LAGO_PJM −0.11% 需观察读出是否带门控)
→ P2 候选方向/动作分解(已完成,§4;标签 LOCAL_ACTION_OR_PROPOSAL_LIMITED)
→ P3 跨域训练器对照(已完成,§5;H4 否定,训练器不改)
→ P4 Rich branch 数据契约(已交付)
→ P5 location shift —— 不执行(§3 已否定)
```

**下一步是否允许修改代码(按 v0.2 §9 的陈述项)**:本轮报告完成即**暂停,等待人工确认**。§1 规定报告完成并人工确认前不得修改 src/ 生产代码。若人工确认允许修改,再按 §8.4 排序进入执行:第一步 P1 为"三原子分布到点读出审计"(v0.2 §8.5 默认排序),其落地方式(a) 读出接入推理路径 / (b) 仅修 evaluator 诚实评估候选——**届时由主调研方定夺**,不在本轮决定。

**P1 的两种落地方式**(由主调研方选择,不在本轮决定):
- (a) 读出层直接接入 final:`x_final = s·sinh(z0 + λ(w⁺m⁺−w⁻m⁻))`(λ 在 S2V 上选,S4 冻结)——零新增参数,§3 已验证;
- (b) 读出作为 evaluator 修正(评估候选真实 point 能力),不动推理路径——先诚实评估,再决定是否落地。

## 8.6 停止条件检查

| 条件 | 状态 |
|---|---|
| 点指标当前评估宿主原值 | **命中一半**:panel cand_mae=宿主;headline(P0-A)=宿主+稀疏动作 |
| S4 标签参与 readout/action/checkpoint 选择 | 否(§3 只 S2V 选,S4 冻结) |
| action-chain 与 v0.3 数学链不一致 | 否(两条 gains 变体均属 v0.3 数学) |
| 候选 CRPS 无法复现 | 否(逐位复现 0.2251746) |
| holdout 进训练/调参 | 否 |
| 文献无真实链接 | 否(29 篇全部 HTTP 200 + 标题匹配验证) |

---

## 附:本轮产物清单

- `experiments/08-hch-v2/training_investigation_diag.py`(§2.1 toy + §3/§4 诊断,新增)
- `experiments/08-hch-v2/training_investigation_trainer_cmp.py`(§5 对照,新增)
- `results/TRAINING_INVESTIGATION_FULL/`(诊断输出:report.json + 6 域 S2V/S4 + §4 诊断)
- `results/TRAINER_CMP_20260814_seeds012/`(§5 对照,seeds 0,1,2,A vs B)
- `docs/训练文件夹/训练调查/rich_branch_data_contract_draft.md`(§6)
- `docs/训练文件夹/训练调查/literature_review_draft.md`(§7,29 篇全验证)
- 本报告

**状态:定稿(§2-§8 全部完成)。本轮未修改 src/ 生产代码、未覆盖旧实验结果、未把 holdout 加入训练。下一步是否允许修改代码,由人工确认后决定(v0.2 §9:报告完成后暂停)。**
