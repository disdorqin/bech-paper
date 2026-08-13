# R1B 泛化筛选报告 — 服务器冲刺执行记录

> 会话: 2026-08-13 21:10–23:10(本地)· 冲刺文档: `hch_v2_r1b_two_hour_autonomous_research_sprint_v0.1_2026-08-13.md`
> 本地 git SHA: `e3f6e97`(+本报告后的新提交)· 服务器 venv 无 .git → config.json 中 git_sha=unknown
> 本报告为最终执行记录;筛选结果来自 `R1B_SCREEN/`(两轮运行,第二轮为 dk1 写盘修复后完整运行,SCREEN2_EXIT=0)。

## A. 环境 (ENV) — §25.A

| 项 | 值 |
|---|---|
| 服务器 | gpuhome.cn 容器 (hn01-ssh.gpuhome.cc:30581) |
| 系统 | Ubuntu 22.04.5, Linux 5.15.0-174 |
| Python | 3.11.15 (deadsnakes, venv `/root/rivermind-data/.venv-r1b`) |
| torch | 2.13.0+cu126 · GPU RTX 4090 23.5GB (cap 8.9) |
| 依赖 | numpy 2.4.6 / pandas 2.2.3 / scipy 1.17.1 / sklearn 1.5.2 / lightgbm 4.6.0 / matplotlib 3.9.2 |
| 数据 | LAGO_DE 52416h · LAGO_PJM · NEM_SA1 · NORD_DK1 24189h(全部在位并加载) |
| 测试 | test_p1.py **18/18 PASS** · test_p0_fix.py **5/5 PASS**(P0-A/B/C/D, P0-1/3/4, P1-3) |

`environment.txt` 全量 16 行在 `experiments/08-hch-v2/results/environment.txt`。

## B. Host caches — §25.B (16 cells, seed=0, H0-fit 冻结)

全部 OK,split hash:
- LAGO_DE `9e3a4dcfb2ed15ff` · LAGO_PJM `a98174f9394e1f60` · NEM_SA1 `460e8e97513bdc9f` · NORD_DK1 `b21f0e32707961a5`

**§7 host 误差体制**(host_quality_by_domain.csv,S2V):

| market | transformed_host_error | S2V MAE | lag1 ACF | 判读 |
|---|---|---|---|---|
| LAGO_DE | 0.200–0.280 | 5.6–7.4 | 0.91–0.94 | 中等 |
| LAGO_PJM | 0.059–0.120 | 2.4–4.9 | 0.83–0.95 | 低(host 已强) |
| NEM_SA1 | **0.821–0.911** | 74–170 | 0.60–0.72 | **极高(高波动负价市场)** |
| NORD_DK1 | 0.205–0.335 | 26–48 | 0.94–0.96 | 中等偏强序列相关 |

→ R1B 横跨异构 host 误差体制;NEM_SA1 为天然压力测试。

## C. Candidate screening — §25.C

- 训练域: main = 12 (3 源市场 × 4 host); LOHO = 9 (排除 PatchTST,零 S2T 梯度/零 S2V 选择信号)
- 评估域: 16 (12 源 + 4 DK1, DK1 零梯度 §11)
- 候选参数: d_model=64, d_sig=32, seed=0, AdamW 3e-4, wd 1e-4, clip 1.0, epochs=12, patience=4

**训练 (S2V macro CRPS):**

| screen | macro_s2v | worst | epochs |
|---|---|---|---|
| LearnedSig_main | **0.22517** | 0.67937 (NEM:SLSTM) | 12 |
| LearnedSig_LOHO | 0.22729 | 0.68605 | 12 |
| PlainCore_main | 0.23723 | 0.69587 | 9 |
| PlainCore_LOHO | 0.23979 | 0.71188 | 12 |

**§12 四格聚合 (mean delta_crps,负 = 候选优于 host):**

| cell | LearnedSig_main | LearnedSig_LOHO | PlainCore_main | PlainCore_LOHO |
|---|---|---|---|---|
| Seen/Seen (n=9) | **-0.1612** | -0.1598 | -0.1503 | -0.1473 |
| Seen/Unseen_host (n=3, PatchTST) | **-0.1772** | -0.1756 | -0.1617 | -0.1605 |
| Unseen_market/Seen_host (n=3, DK1) | -0.0670 | -0.0674 | -0.0671 | -0.0650 |
| Unseen/Unseen (n=1, DK1:PatchTST) | -0.0735 | -0.0742 | -0.0737 | -0.0733 |

**DK1 全部 16 行 delta_crps 均为负**(-0.037 ~ -0.112),跨 4 host 家族无 MARKET_TRANSFER_COLLAPSE。
**LOHO-PatchTST 全部负**(-0.013 ~ -0.026)于源市场,无 HOST_TRANSFER_COLLAPSE。
**LearnedSig vs PlainCore**: 源域 Seen/Seen -0.161 vs -0.150(sig 更优);未见域 -0.067 vs -0.067(无差)→ 无 SIGNATURE_NEGATIVE_TRANSFER。

**mass health (LearnedSig_main)**: entropy 1.088(≈ln3=1.099 分散良好);w0=0.40;m⁻/m⁺ 存活 0.99/0.98;shift_p95 0.5–0.9。PlainCore δγ=β=0(真旁路),LearnedSig δγ=0.138(学习 FiLM 激活)。

**逐域看点**: NEM_SA1 最大受益(delta -0.23~-0.54,host CRPS 0.82→候选 0.28-0.68);LAGO_PJM 获益最小(host 已强 0.06-0.12,delta -0.012~-0.037);LAGO_PJM:MLP 是 PlainCore 唯一近持平行(-0.0014),LearnedSig 同格 -0.0119(签名保护最强 host cell)。

## D. 泛化裁决 — §25.D (exactly one label)

```
R1B_SCREEN_HEALTHY
```

判据(§14): 四 cell 全部 mean<0;DK1 跨 host 家族全负(无 MARKET_TRANSFER_COLLAPSE);LOHO PatchTST 全负(无 HOST_TRANSFER_COLLAPSE);LearnedSig 源域优于 PlainCore 且未见域无劣化(无 SIGNATURE_NEGATIVE_TRANSFER)。candidate 推理有限、无坍缩、源 macro 健康。
**注意**: 这是 §D 的一次性单种子筛选裁决,**不是 R1B 最终裁决**。

## E. 未做 (not-done) — §25.E

- 无 multi-seed 确认;无 U0;无 R1C;无架构更改;无 S4 调参;无新 loss。
- 无 C3/DVG 改动;无 IAH 重训;无 HCH 重新设计;无 Shandong;无 TCN host。

## F. Artifacts — §25.F

| 类别 | 路径 |
|---|---|
| env | `experiments/08-hch-v2/results/environment.txt` |
| tests | `results/tests_p1.txt` `results/tests_p0_fix.txt` |
| cache | `experiments/08-hch-v2/results/cache/*` (16 cells) + `cache_batch.txt` |
| host quality | `results/host_quality_by_domain.csv` |
| screen | `results/R1B_SCREEN/` (config.json, matrix_*×4, dk1_zero_gradient.csv, four_cell_aggregate.csv, training_reports.json, summary.txt) |
| host fidelity | `docs/paper_prep/v2_final_prep/r1b_host_backbone_fidelity_audit_v0.1.md` |
| feature schema | `docs/paper_prep/v2_final_prep/r1b_domain_feature_schema_audit_v0.1.md` |
| DA/RT | `docs/paper_prep/v2_final_prep/public_da_rt_dataset_audit_v0.1.md` |
| U0 corpus | `docs/paper_prep/v2_final_prep/u0_external_corpus_inventory_v0.1.md` |
| literature | `docs/paper_prep/v2_final_prep/r1b_literature_scout_v0.1.md` |

## G. 建议下一步决策 — §25.G (≤3, 不自动执行)

1. **授权 action-chain 阶段评估**(sprint §14 CONTINUE 路径):LearnedSig 为候选主变体,按 R1A 链(C3 LocalIsotonic → 决策链 → 价值评估)在 6 域 + DK1 上推进,保持 §11「修复不杀 NEM/DE」约束。**(推荐)**
2. **DK1 相似度衰减的预声明解释**:按文献侦察 §7,把 DK1 收益小于源域(-0.067 vs -0.16)解读为「transfer 收益随市场相似度衰减」——负电价率、残差波动、host 误差 regime 的差异已在 §7 台账记录,这是可伪造的诚实结论,而非笼统「transfer 失败」。
3. **R1B 多种子/U0 确认暂缓**:单种子筛查健康不等同于稳健;在 R1C 接入 NYISO DA/RT 对(public_da_rt_audit 推荐)之前,评估是否补 2 个额外种子的只读复现(需新授权,本次未做)。

---
### 服务器执行痕迹(审计用)
- 首轮运行 crashed at dk1_zero_gradient 写盘(`_write_matrix` 固定列缺 `screen`)→ 已修复(extra 列前置)并完整重跑 `SCREEN2_EXIT=0`,全程 ~10 min(14:09→14:19 UTC,一轮)。
- 服务器保留 `/root/rivermind-data/r1b_out/{environment.txt, tests_p1.txt, tests_p0_fix.txt, cache_batch.txt, screen.txt, screen2.txt, R1B_SCREEN/}`。
- 修复 bug 的 runner 已同步本地并待提交。
