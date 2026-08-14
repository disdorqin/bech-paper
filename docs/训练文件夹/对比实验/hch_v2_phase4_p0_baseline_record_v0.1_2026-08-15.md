# HCH-v2 Phase4 — P0 基线复现记录(旧路径健康)

- 日期: 2026-08-15
- 目的: build spec §7-P0 门 —— 在增加任何 CAVM 代码前,证明旧 `v0.4-core + W1 + static DVG` 路径健康,保存旧结果 hash。
- 分支: `exp/r1b-screening-20260813`, git HEAD `8b81364`(Paper Benchmark Gate YELLOW_READOUT)。

---

## 1. P0 门 1: 全部现有测试通过

| 测试文件 | 结果 | 备注 |
|---|---|---|
| `tests/test_pipeline.py` | 3 passed | 完整 IAH 链路 + legacy guard |
| `tests/test_phase1.py` | 8 passed | W1 精确 CDF 断点 |
| `tests/test_phase2.py` | 3 passed | hourly gain bounded |
| `tests/test_phase3.py` | 2 passed | Down/Up 不重叠 |
| `tests/test_phase45.py` | 7 passed | conformal DVG + bundle/legacy |
| `tests/test_p1.py` | 18 passed | universal/local hash 独立 |
| `tests/test_p0_fix.py` | 5 passed | det 展开 batch size |
| `tests/test_p0a_final_replay.py` | ALL PASSED | inverse-asinh 精确 replay / bundle reload |
| `tests/audit_contracts.py` | **25 passed** | 修复 1 个过时契约测试,见 §3 |

**合计: 71 个测试断言全绿(审计契约修复后)。**

## 2. P0 门 2: smoke_v4 端到端复现

`experiments/08-hch-v2/smoke_v4.py`(LAGO_DE Linear,完整 S1→S2→S3M→S3C→freeze→from_bundle→predict_s4):

```text
S2 checkpoint (S2V-selected): 0.1793 (88 validation days)
S3-M memory: 81 days, validation: 28, selected k=5
S3-C calibration: n=109, q=0.0636
selected_k=5, execute_rate=0.050 (22/437 days)
roundtrip_hash_match=True
```

**旧结果 hash(CAVM 改动后必须逐日复现):**
- smoke evidence file: `results/v0.3/smoke_v4_lago_de_linear.json`
- evidence 整体 SHA-256[:16]: **`cbcf88f577b29ccb`**
- bundle_hash: 见该 json(冻结 bundle 深度 hash)
- 关键基线值: `S2_loss=0.1793`, `selected_k=5`, `execute_rate=0.05034`, `q=0.0636`, `roundtrip=True`

## 3. 发现与修复记录(不触碰 v0.4 核心)

**`audit_contracts.py` A12 是过时测试,非核心回归。**

- 历史: commit `66e812e` 时 `get_neighbors` 为 distance-based self-exclusion(自动排除 W1<1e-14),A12 当时 45/45 通过。
- 现状: `w1_retrieval.py:192` 的 `get_neighbors(distances, k, exclude_idx=None)` 为 **ID-based self-exclusion(P1-2 明确契约:两个不同日可合法拥有完全相同的 atom measure / W1=0,仍应为完美邻居)**。真实 pipeline 中 memory 与查询日严格不相交(`r1a_run.py:404` `mem_dates = s3m_all[:n_mem]`, `val_dates = s3m_all[n_mem:]`;S3C/S4 查询日同样不在 memory),故 `exclude_idx=None` 无 self-泄漏。
- 修复: 更新 A12 以匹配 ID-based 契约 —— `get_neighbors(dists, k, exclude_idx=0)` 时排除查询日自身;不传 `exclude_idx` 时距离 0 的日仍是合法邻居。**未改动 `src/w1_retrieval.py`。**
- 修改文件清单(本阶段): `experiments/08-hch-v2/tests/audit_contracts.py`(仅测试)。未修改任何 `src/` 文件。

## 4. Phase4 接入点备忘(实现前的冻结基准)

后续 P1-P3 只允许:
1. 新建 `src/context_action_memory.py`;
2. `HCHV2UniversalPipeline.__init__` 增加可选 `memory_mode="w1"`(默认不变);
3. pipeline 新增 `fit_cavm_memory()` / `observe_outcome()`(不改变 `predict_s4` 签名);
4. bundle 增加可选 CAVM 字段(用 `.get()` defaults 保证旧 bundle 可加载)。

禁止改动: `iah_crps_loss.py`、`IAHCandidateHead` 三原子参数化、`double_event_proposal()`、`DGVSplitConformal` q/LCB 公式、`UniversalCoreTrainer` 等域采样、`memory_mode="w1"` 结果、S4 target-free 接口。
