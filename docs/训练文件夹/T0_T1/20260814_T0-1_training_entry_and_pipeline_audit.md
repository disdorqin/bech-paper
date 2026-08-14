# T0-1 训练入口与权威链路审计(2026-08-14)

> 阶段:T0 第 1 步 · 驱动文档 `hch_v2_t0_t1_finetune_and_component_diagnosis_prompt_v0.1_2026-08-14.md` §2.1
> 本步只审计与标记,**不改代码**。

---

## 1. 正式 runner 确认(§2.1.1)

**正式 runner = `experiments/08-hch-v2/r1b_generalization_screen.py`**(R1B 12-domain 主筛选)

- `--help` 正常:`--out / --variants / --only-loho / --epochs`
- 12 source domains = **3 source markets × 4 hosts**(`SOURCE_MARKETS=["LAGO_DE","LAGO_PJM","NEM_SA1"]` × `HOSTS=["Linear","MLP","LSTM","PatchTST"]`)
- 超参:`epoch=12, patience=4, lr=3e-4, wd=1e-4, seed=0, d_model=64, d_sig=32, d_value=0`
- 复用链:`import r1a_run as R`(prepare_domain / det_for_variant / D_* 配置)+ `universal_trainer` + `iah_candidate`(src)

结论:训练入口唯一,无第二套入口。**不新建 runner。**

## 2. 训练链路确认(§2.1.2)

- ✅ 正式训练调用 `src/universal_trainer.py::UniversalCoreTrainer.train()`
- ✅ 等域采样 + macro-S2V checkpoint 选择:`universal_trainer.py:176` "True equal-domain sampling + macro S2V checkpoint selection";`universal_trainer.py:299-307` macro 最优保留 + patience early-stop
- ✅ 训练 head = `IAHCandidateHead`(src/iah_candidate.py),weight = IAH-CRPS(src/iah_crps_loss.py)

## 3. ⚠️ 核心发现:存在两条并存的动作链实现(§2.1.3)

| | **路径 A — 权威完整链路** | **路径 B — Paper Gate 实际使用** |
|---|---|---|
| 位置 | `src/hch_v2_pipeline.py::HCHV2UniversalPipeline.predict_s4`(353-425 行) | `run_matrix.py → r1a9_action_calibration` + `r1a11` + `_final_point` |
| 候选 | `candidate_head` | `collect_domain`(同用 candidate_head 输出) |
| 动作提案输入 | `build_retrieval_index → get_neighbors → full_replay_chain`(**CAGM 查询剂量回放,用 memory 邻居的 `build_directional_gains` 计算 g_hat**) | `evaluate_days`:calibrator 对 `sd/su` 缩放 → `g̃ = m·s̃` → `double_event_proposal(g̃d, g̃u)`(**无 CAGM 邻居检索**) |
| 提案 | `double_event_proposal`(same) | `double_event_proposal`(same) |
| π | `form_final_pi`(same) | `form_final_pi`(same) |
| 门控 | `self.dvg.lcb`(整日 LCB) | `dvg_and_s4`:per-calibrator 独立 DVG q(S3C 上 fit,split-conformal α=0.10) |
| 最终输出 | `x_final = x_up/x_down/identity`(execute/fallback) | `_final_point.final_point_metrics`:**P0-A replay** `x_final = s·sinh(z0 + π_eff)` |
| 是否含 CAGM 查询回放 | ✅ 是(邻居 gains) | ❌ 否(calibrator 缩放直接替代) |

**两条路径的差异核心**:动作提案的 gains 信号不同——
- A:gains = CAGM 查询剂量回放(从 memory 邻居的历史 realized A 计算)
- B:gains = 候选 m × calibrator 缩放(无历史邻居信息)

**影响**:Paper Gate 的 B3/B4 矩阵(B5 point MAE/sMAPE)全部基于**路径 B** 测出。文档 §2.1.3 要求"先标记并统一为同一条默认路径;统一前不得把两条路径的结果混在一张表里"——当前矩阵内是统一的(全用路径 B),但**权威 `predict_s4` 与 gate 实际路径不一致**,需要在 T0 统一。

## 4. ⚠️ domain descriptor 绑定缺口(§2.1.4)

- `r1a_run.py:100 det_for_variant` 生成 descriptor;`universal_trainer.py:123` 用 batch 5-tuple 或 `d.det_tensor(shape)` 广播
- **现状只有 shape 检查**(`det.shape[0] != host.shape[0]` 时 expand),**无"det 内容 ↔ 声明的 market:host"绑定断言**
- `r1a_run.py:37`:`domain_det=None` fallback 到 **MUTABLE buffer** —— 存在静默填零/可变路径
- §2.1.4 要求"加入断言,发现错配立即失败,不允许静默填零" → **待补**

## 5. 数据切分现状(§2.3 快照)

- `S1 train → S2T tuning → S2V validation → S3M freeze-memory → S3C calibration → S4 final`
- `S3M_MEM_FRAC=0.75`(前 75% S3M = memory,后 25% = forward-val);`ALPHA=0.10`(DVG split-conformal)
- 时间边界/时区/DST/负价占比/特征 hash 的**训练启动断言未见**(§2.3 要求"启动时输出并断言")→ 并入 T0-2

---

## 决策点(交主调研方)

1. **两条动作链统一方向**:
   - (a) 以权威 `predict_s4`(含 CAGM 查询回放)为默认 —— 但需回归对比路径 B 的 B5 结果是否变化(可能改善 point);
   - (b) 确认路径 B(calibrator 缩放,无 CAGM 检索)为正式生产链,标记 `predict_s4` 为候选/参考,并明确"Paper Gate B5 = 路径 B"的事实写入论文;
   - (c) 两者都保留,但在 manifest 中显式声明每条结果用了哪条路径,禁止混表。
2. descriptor 绑定断言 + 数据边界断言并入 T0-2。

## 下一步

待主调研方决定动作链统一方向后,再进 T0-2(日志/边界断言补齐 + smoke)。
