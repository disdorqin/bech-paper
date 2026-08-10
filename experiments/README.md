# Experiments: active directory

> 日期：2026-08-09 | 当前阶段：Route-E 应用论文

---

## 活跃实验目录

| 目录 | 作用 |
|---|---|
| **`07-route-e/`** | Route-E 主线：HCH 实验 runner + 同行基线复现 + 对比实验 |
| `00-data-exploration/` | 负价频率、regime drift 等数据特征证据（论文 Introduction 引用） |
| `05-episode-audit/` | P0 问题锚点：episode 结构 + 基座完整漏报率审计 |
| `_archive/` | 归档：旧 BECH 实验 + 退休原型 |
| `_support/` | 数据核验工具（非实验） |

---

## 同行基线

所有基线复现代码在 `07-route-e/peers/`，复现报告在 `docs/paper_info/peer_reproduction_report.md`。

| # | 基线 | 出处 | 状态 |
|---|---|---|---|
| B1 | Quantile Correction | 经典 | ✅ |
| B2 | Vahedi 2026 | IEEE ICCE | ✅ |
| B3 | PIR | NeurIPS 2025 | ✅ |
| B4 | CRC | arXiv:2512.22428 | ⚠️ |
| B5 | SpikeReg | AAAI 2026 WS | ⏳ |

## 执行环境

```powershell
$PY = 'D:/computer_download/environment/conda/epf-2/python.exe'
```
