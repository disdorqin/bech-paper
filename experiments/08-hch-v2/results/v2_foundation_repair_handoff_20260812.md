# HCH v2 Foundation Repair Handoff

> Date: 2026-08-12 | Start: `f73b561` | End: (HEAD) | **22/22 PASS**

## 1. 起止 commit 与环境

- Start: `f73b561` (审计修订后的 math_loss 文档)
- Python: conda epf-2, PyTorch CPU, LightGBM

## 2. P0/P1 修改映射

| ID | 文件/符号 | 修改 | 测试 |
|---|---|---|---|
| P0-1 | `host_cache.py` L29 | 缩进修复；CLI 只保留 `main()` 内一套；`--seed` 参数生效 | `py_compile` PASS |
| P0-2 | `test_contracts.py` test 01 | 扩展编译覆盖至 4 个活跃目录 | `compileall` 全仓 PASS |
| S3+S4 | `hch_v2_data.py` (重写) | raw/model 双通道；target Optional；exog_type 分离；lag_context | test 07/08/13/18 |
| S5 | `hch_v2_data.py` | `fit_exog_scalers()` S1-only 标准化；learned-null mask=0 | test 09/10 |
| S5 | `hch_v2.py encode()` | 显式传递 `batch.exog_type` 到 `HourTokenEncoder` | test 07/08 |
| S6 | `HCHV2Bundle` | 新增 calibration/exog_scalers/split_hash/data_hash; hash 覆盖全部组件 | test 17/19 |
| S7/8 | `hch_v2.py` | `LEGACY_UNTRAINED` flag; `require_not_legacy()` gate | runtime validation |
| S9 | `audit_utils.py` | CDF 修正 (16/16 vs scipy); bootstrap multiplicity; 日矩阵重写 | 单元验证 |
| S9 | `run_math_evidence_audit.py` | 残差符号统一 `r = y - host_pred` | 全局 |
| U1-U5 | `hch_v2.py + smoke_v2.py` | `_to_device()` 13字段映射; encode/forward/calibrate/train_step/build_s3 字段改名 | contract 22/22 |
| U5 | `smoke_v2.py` | trim/pad → strict equality + fail-closed | assertion gate |
| U6 | `test_contracts.py` tests 07/08/13/18 | 6位置参数→13关键字参数; target→target_model | 22/22 PASS |
| S11 | `official_adapters.py` | 类名 `Official→Limited`; `limited_reimplementation` 标签 | test 20 PASS |
| S13 | `AGENTS.md` | v2 canonical docs; 基座/基线表更新 | |

## 3. Compile 与 Contract Tests

```
python -m compileall -q src experiments/08-hch-v2 experiments/00-data-exploration/math_loss
→ 0 errors

python experiments/08-hch-v2/test_contracts.py
→ 22 passed, 0 failed
```

## 4. 数据契约

| 旧 | 新 |
|---|---|
| `host_pred` (norm) | `host_raw` + `host_model` |
| `target` (always present) | `target_raw` + `target_model` (both Optional) |
| `exog [B,H,N,3]` 混合 | `exog_value [B,H,N,1]` + `exog_type [B,H,N]` + `exog_mask [B,H,N]` |
| 无 | `lag_context [B,H,5]` (lag24/168 price + residual + hour) |

## 5. Gate 状态

| Gate | 状态 |
|---|---|
| F0 可运行 | ✅ compileall 0 errors; 22/22 PASS |
| F1 数据契约 | ✅ raw/model/target-free/S1-only exog/lag |
| F2 评估 | ✅ trim/pad 已删除; fail-closed |
| F3 冻结 | ⚠️ calibration persisting pending IAH |
| F4 证据 | ✅ CDF/bootstrap/日矩阵/hash 修正 |
| F5 诚实 | ✅ legacy quarantine; baseline labels |

## 6. 遗留项

| ID | 描述 | 优先级 |
|---|---|---|
| L1 | test 04 仍为 `pass` stub | LOW |
| L2 | `docs/paper_info/README.md` 山东角色 | LOW |
| L3 | Candidate audit csv 标 `INVALID_ORACLE_DIAGNOSTIC` | MEDIUM |

## 7. 最终状态

**READY_FOR_MATH_ARCHITECTURE_FUSION_REVIEW**
