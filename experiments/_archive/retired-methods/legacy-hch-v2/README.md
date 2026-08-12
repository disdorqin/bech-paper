# Legacy HCH v2 (retired)

> 2026-08-12：被 v0.4 IAH-CRPS 架构取代。移入 `_archive`，不再作为正式入口。

这些文件是 v0.3 之前的 legacy HCH v2 原型（BiOMC + CAGM + DVG + CARA/KL 校准），
与 v0.3 数学契约矛盾（用了 BCE occurrence + SmoothL1 magnitude + state loss + eta/tau 网格校准）。

- `smoke_v2.py` — 旧 formal smoke runner（legacy HCH 路径）
- `test_contracts.py` — 旧契约测试（测试 legacy HCH）

**唯一的正式入口现在是 `experiments/08-hch-v2/smoke_v4.py`**，
只走 `hch_v2_pipeline.py`（HCHV2UniversalPipeline）的 IAH-CRPS 路径。

`src/hch_v2.py`（legacy HCH 实现）仍保留在 `src/`，但：
- 标记 `LEGACY_UNTRAINED` + `require_not_legacy()` gate；
- 仅被 `tests/test_phase45.py` 和 `tests/test_pipeline.py` 引用，
  用于验证 legacy fail-closed 是否真实（这是文档 P0-2 的验收测试）。
