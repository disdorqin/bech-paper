# HCH-v2 Phase4 — P2 composite retrieval + P3 local observe 实验与实现报告

- 日期: 2026-08-15
- 范围: build spec §7-P2(composite retrieval + E1-E3)、§7-P3(observe_outcome local,默认关闭)
- 分支: `exp/r1b-screening-20260813`
- 依赖: P0 基线记录 `hch_v2_phase4_p0_baseline_record_v0.1_2026-08-15.md`(71 测试断言 + smoke hash)

---

## 1. 修改清单

### 改动文件(全部在本次 Phase4 P1-P3 中)

| 文件 | 改动 | 阶段 |
|---|---|---|
| `src/context_action_memory.py`(新增) | `CAVMExperience` dataclass、`ContextKeyBuilder`(cavm-key-v1)、`context_distance`/`atom_w1_distance`/`composite_distance`、`ContextActionMemory`(CAGM 兼容 + `query` + `freeze`/`from_frozen`) | P1 |
| `src/hch_v2_bundle.py` | 新增 CAVM 可选字段:`memory_mode`/`cavm_key_version`/`cavm_global_state`/`cavm_local_state`/`cavm_update_policy`/`cavm_global_hash`/`cavm_local_hash`/`cavm_retrieval_lambda`;hash 覆盖(global 进 universal 集,local 进 local 集);save/load/extract 全量 `.get()` 兼容 | P1/P2 |
| `src/hch_v2_pipeline.py` | `__init__` 增 `memory_mode` 与 CAVM 状态;`set_cavm_retrieval(λ_a,λ_c)`;`fit_cavm_memory(global_days, local_days)`;`predict_s4` cavm 分支(composite 检索 + `cavm_no_neighbors` 显式 Identity fallback + 预测时记录 `context_keys`);`freeze_bundle`/`from_bundle` 持久化恢复 CAVM;`set_cavm_update_policy(observe=...)`;**`observe_outcome(query_id, target_zY, evidence)`**(P3,默认关闭);模块级 `_cavm_day_view`/`_cavm_experience`/`_resolve_query_id` | P1/P2/P3 |
| `experiments/08-hch-v2/tests/audit_contracts.py` | A12 测试更新为 ID-based self-exclusion(P1-2 语义),并加"两日同原子度量仍合法互为邻居"断言 | P1 |
| `experiments/08-hch-v2/tests/test_cavm.py`(新增) | 12 测试:key builder dim/健康度/禁用字段/确定性/全无效 fallback/距离函数/λ=(1,0) 复现 W1/冻结往返/市场 ID 惰性/optional 维度/同原子邻居合法 | P1 |
| `experiments/08-hch-v2/tests/test_cavm_pipeline.py`(新增) | 5 测试:E1 账本零扰动、P2 λ 切换、bundle 往返、旧 w1 bundle 加载、未揭示日拒绝 | P1/P2 |
| `experiments/08-hch-v2/tests/test_cavm_p3.py`(新增) | 7 测试:observe 默认关、局部追加不改 universal、局部永不消费、key 预测时冻结、post-reveal 守卫、**A_true 与离线 estimate_realized_A 一致**、local ledger bundle 往返 | P3 |
| `experiments/08-hch-v2/p2_cavm_experiment.py`(新增) | E0-E3 真实数据对比(LAGO_DE Linear) | P2 |
| `experiments/08-hch-v2/p2_cavm_analyze.py`(新增) | P2 结果动作级诊断 | P2 |
| `experiments/08-hch-v2/p3_cavm_experiment.py`(新增) | P3 frozen vs streaming + 冷启动/稳态曲线 | P3 |

### 未改动文件(显式声明,§6)

- `src/w1_retrieval.py`(`day_w1_distance`/`CAGMAtomMemory`/`get_neighbors`)
- `src/query_replay.py`(`full_replay_chain`/`estimate_realized_A`)
- `src/double_event.py`(`double_event_proposal`)
- `src/iah_crps_loss.py`、`src/dvg_calibrate.py`
- 以上四个文件给出下述全部公式与行为;CAVM 只复用其接口,不改其实现。

---

## 2. CAVM context key schema(§3, key version `cavm-key-v1`)

维度:`dim = 8 + 5 + 2·d_core_context + D_SIG + 8 + 3·n_optional_roles`。

本报告实验配置:`d_core_context=13, D_SIG=8, n_optional_roles=0` → **dim=55**。

| 段 | 大小 | 内容(仅在 valid hours 上统计) |
|---|---|---|
| c_shape | 8 | host z0 逐日形状/尺度统计 |
| c_dyn | 5 | z0 动态统计 |
| c_time | 2D=26 | 24h 与周内循环时间(正弦/余弦) |
| c_sig | D_SIG=8 | 域描述符 domain_det(来自 S1 签名) |
| c_atom | 8 | 原子权重图统计 |
| c_opt | 3R=0 | 可选协变量(本阶段 R=0,严格退化到 core-only) |

硬性信息隔离:key builder 拒绝 `target_raw/target_zY/residual/action_gain/A_hat/A_true/action_error` 任何一栏进入 candidate dict。NaN/Inf 由 `_nan0` 归零,key 永不含 NaN。

检索距离:`d = λ_atom·norm(W1) + λ_ctx·norm(ctx)`,`norm` = /max 单调归一。`λ=(1,0)` 必须逐日复现 W1-only。

---

## 3. global/local memory 时间范围(LAGO_DE Linear)

| 账本 | 来源 | 天数 | 日期范围 |
|---|---|---|---|
| CAVM global | 与 CAGM atom memory 相同的 S3-M 前缀日(同一 `mem_days` 列表) | 81 | 2016-03-17 → 2016-06-05 |
| CAVM local | P3 streaming 逐日 observe(S4 全段) | 437 | 2016-10-21 → 2017-12-31 |

- global 与 atom memory 的候选/目标来自同一批揭示日,保证 P2 E1 的 `λ=(1,0)` 检索等价性可直接对照。
- local 只增不改;predictions 永不读取 local(build spec §5.2)。

---

## 4. 测试命令与结果(全绿)

### 4.1 脚本式测试(仓库约定 `python tests/test_x.py`)

```text
test_cavm.py        12 passed, 0 failed   (P1 key/memory/query)
test_cavm_pipeline.py  5 passed, 0 failed (P1 E1 + P2 λ)
test_cavm_p3.py       7 passed, 0 failed  (P3 observe 契约)
test_p1.py           18 passed, 0 failed  (universal/local hash 独立)
test_phase1.py        8 passed, 0 failed
test_phase2.py        3 passed, 0 failed
test_phase3.py        2 passed, 0 failed
test_phase45.py       7 passed, 0 failed
test_pipeline.py      3 passed, 0 failed
test_p0_fix.py        5 passed, 0 failed
test_p0a_final_replay.py ALL PASSED
```

合计 **70 断言脚本式全绿**(P1-P3 新增 24 个 CAVM 断言 + 原有套件无回归)。

### 4.2 P0 基线逐日复现(smoke_v4, LAGO_DE Linear)

```text
S2 checkpoint (S2V-selected): 0.1793 (88 validation days)
S3-M memory: 81 days, validation: 28, selected k=5
S3-C calibration: n=109, q=0.0636
selected_k=5, execute_rate=0.050 (22/437 days)
roundtrip_hash_match=True
```

与 P0 记录逐字一致。**w1 路径在 P1+P2+P3 全部改动后逐字未变。**

> **关于 evidence 文件 hash**:当前 `results/v0.3/smoke_v4_lago_de_linear.json` 整体 hash 为 `9107b8c13fa61a0d`,与 P0 记录的 `cbcf88f577b29ccb` 不同。原因是 `bundle_hash` 深度 hash 包含 `training_provenance.code_commit`(git HEAD),P0 在 `8b81364`、现在在 `63b158a`,属 provenance 元数据差异;**所有预测关键指标逐字一致**,w1 路径零漂移由 §4.1 全套 + P2 E1 真数据 0/437 邻居切换独立证明。`results/` 目录 gitignored,不在仓库内。

---

## 5. "是否读到了 target"审计(信息隔离证据链)

1. **key 不含标签**:`ContextKeyBuilder` 硬拒绝 7 个标签/动作衍生字段;测试 `test_cavm.py` 覆盖。
2. **预测时不读 target**:`predict_s4(host_raw, core_context, domain_det)` 签名无 target 参数;S4 target-free。
3. **key 在预测时冻结**:`predict_s4` 在 cavm 分支把逐日 context_key 写入 `evidence["context_keys"]`;`observe_outcome` 只读该预测时 key,绝不回建。测试:"修改 query target → key 逐字节不变"。
4. **A_true 只在揭示后计算**:`observe_outcome` 用揭示的 scale-free `target_zY` 调用 `estimate_realized_A`;内部 A_true 与离线重算 `test_cavm_p3` 断言 `<1e-9` 一致。
5. **当日经验不被当日预测读取**:P3 invariance 0/437 动作切换,streaming 与 frozen 逐日等价。
6. **local ledger 不在 universal 包**:`extract_universal()` 不含 `cavm_local_state`(测试断言)。

---

## 6. 公式未被触碰的显式声明

以下公式在 P1-P3 中**未做任何修改**,由测试锁定:

- **IAH-CRPS**(`src/iah_crps_loss.py`)—— S2 训练损失;
- **W1 精确 CDF 距离**(`src/w1_retrieval.py` Eq 16)—— `test_phase1.py` 8 项断点;
- **double-event proposal**(`src/double_event.py`)—— `test_phase2.py` |g|≤|π|、`test_phase3.py` Down/Up 不重叠;
- **DVG split-conformal**(`src/dvg_calibrate.py`)—— q/LCB 计算与调用点不变;
- **estimate_realized_A / full_replay_chain**(`src/query_replay.py`)—— CAVM 复用原接口。

CAVM 只做**检索选择层的替换**(global ledger 复合检索)与**审计账本追加**(local observe),不改变 candidate 生成、replay 剂量、proposal 构造或 conformal 校准的数学。

---

## 7. P2 实验结果(E0-E3, LAGO_DE Linear, S4=437 天)

结果文件:`results/phase4/p2_cavm_experiment_lago_de_linear.json`(gitignored,本报告记录数字)。

| 模式 | 检索 | exec_rate | MAE | RMSE | mean_A_hat | mean_A_true |
|---|---|---|---|---|---|---|
| E0 | w1(对照) | 0.050 | 6.219 | 8.830 | +0.0319 | −0.0051 |
| E1 | cavm λ=(1,0) | 0.050 | 6.219 | 8.830 | +0.0319 | −0.0051 |
| E2 | cavm λ=(0,1) | 0.128 | **6.046** | **8.704** | +0.0315 | **+0.0129** |
| E3 | cavm λ=(1,1) | 0.073 | 6.192 | 8.836 | +0.0328 | +0.0082 |

逐日差异(vs E0):

| 模式 | 邻居切换 | 动作切换 | MAE Δ | RMSE Δ |
|---|---|---|---|---|
| E1 | **0/437** | **0/437** | +0.0000 | +0.0000 |
| E2 | 437/437 | 56/437 | −0.173 | −0.126 |
| E3 | 434/437 | 28/437 | −0.027 | +0.006 |

### 7.1 解读

- **E1 == E0**:CAVM 账本在 λ=(1,0) 下真数据逐日等价复现 W1,诊断不改预测的核心契约成立。
- **E2(纯上下文)机制有效**:邻居 437/437 切换(上下文距离驱动检索,无退化平局);点预测 MAE −2.8%;动作级净益——identity→execute 45 天中 32 天 A_true 为正(+0.0452 均值),execute→identity 11 天中 8 天正确摘除(E0 当日执行其实 A_true<0)。
- **E3(复合)信号被稀释**:W1 项把上下文信号摊薄,W1 在复合里更像噪声源。
- **A_hat 校准差(如实)**:执行日 A_hat≈+0.11 明显高估 A_true≈+0.06,E2 执行日 corr=−0.21。E2 的改善来自"检索到更好邻居→更准 π",**不是**来自 A_hat 排序。与 R1A.8 `ACTION_SIGNAL_UNRESOLVED` 相互印证,属已知薄弱点。

### 7.2 边界(必须如实标注)

- **单市场单 seed**(LAGO_DE Linear),full-chain 机制探针,**不是泛化判定**。
- **λ 未在 S4 调参**(红线遵守)。E2 优势是机制诊断证据,不是最终配置;真正的 λ 选择应在 S3-M/S2V,本阶段固定网格 {0,1}/{1,1}。

---

## 8. P3 实验结果(frozen vs streaming)

结果文件:`results/phase4/p3_cavm_experiment_lago_de_linear.json`(gitignored)。

### 8.1 不变性(§5.2/§8.3)

```text
max_A_hat_delta   = 1.02e-8   (浮点噪声级)
max_x_final_delta = 1.14e-5   (浮点噪声级)
action_switch     = 0 / 437
```

streaming 逐日 observe 437 天后,437 天预测与严格 frozen **逐日等价**。local 账本 437 天,global 账本 81 天不变。

### 8.2 交叉验证

修正后 P3 逐日 A_true 与 P2 E0 **全部 437 天逐字一致(max|d|=0.0)**。先前一次运行差异源自脚本把**原始电价**而非 scale-free `arcsinh(y/s)` 传入 `observe_outcome`;已(a)在 `observe_outcome` docstring 固化"target_zY 必须 scale-free"约定,(b)新增契约测试锁定内部 A_true 与离线 `estimate_realized_A` 一致。

### 8.3 冷启动/稳态曲线(账本统计收敛)

| 窗口 | mean_action_error | mean_A_true | pos_rate |
|---|---|---|---|
| cold-start(前 50 天) | 0.0483 | −0.0172 | 0.26 |
| steady-state(后 50 天) | 0.0383 | −0.0057 | 0.34 |
| full(437 天) | 0.0369 | −0.0051 | 0.414 |

- 执行日(22 天):mean_A_true **+0.0377**,pos_rate 0.545 —— 与 P2 E0 执行日统计一致。
- 账本统计随样本增大收敛(action_error 0.048→0.038,A_true 上移,pos_rate 0.26→0.34)。这是**审计账本**的统计收敛,P4(action-value state update)的饲料;本阶段 local 不反哺预测。

---

## 9. Phase4 假设与判定

**技术架构假设(v0.1)**:CAVM 用"上下文条件检索 + 揭示后经验账本"补充 W1 的纯原子相似度,改善极端价格日的动作选择,且不触碰 universal 训练。

**P2-P3 结论(机制探针,单域单 seed)**:

1. ✅ CAVM 账本/检索层零扰动契约成立(E1==E0,真数据 0/437)。
2. ✅ 上下文检索确实驱动邻居切换(437/437),且**方向正确**(E2 点预测与动作价值同向改善,满足 §9.2-4"动作与点预测不反向")。
3. ✅ local observe 默认关闭、零反哺、严格 frozen(§9.2-5 的"仅 streaming 有效"前提不触发)。
4. ✅ 信息隔离审计全过(§5)。
5. ⚠️ **尚未满足 §9.2 放行**:(a) 只有 LAGO_DE 单一类型市场,不满足"≥2 种市场方向稳定";(b) 只有 seed0,不满足"3 seed 非单一 seed 支撑"。

**判定**:`COMPOSITE_RETRIEVAL_MECHANISM_SUPPORTED`(P2-P3 阶段),**非** Phase4 全局通过。上下文距离是检索信号的主要来源这一机制假设获得首份支持性证据;进入 §9.3 停止条件检查:**未触发**(上下文检索在单一域即改善、非仅山东负价域、无温和市场污染、action harm 未上升)。

---

## 10. 下一步

1. **P2 扩展**:多市场 × 多 seed(LAGO_DE/PJM/AUS + 山东/甘肃,3 seeds)验证 §9.2;λ 在 S3-M/S2V 上选择(不碰 S4)。
2. **P4 候选**(仅在 P2 扩展通过后):local 账本 → action-value state update 或经验收缩;若 P2 扩展失败,回到候选表达/时间错位,不扩大 memory(§9.3)。
3. 报告收录为论文"CAVM 机制验证"段落素材,正式方法名 HCH-CAVM 需在 §9.2 全绿后。
