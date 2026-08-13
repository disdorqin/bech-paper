# R1B 总结提示词(可整段粘贴给 AI)

> 你正在接手/继续 HCH-v2 项目。以下是 R1B 全程(A→F + SCREEN)的浓缩结论,先读这个,需要细节再点报告。

## 一句话判定
HCH-v2 R1B(2026-08-14)= **NATIVE_GENERALIZATION_SUPPORTED**。冻结 universal 候选(LearnedSig_main,seed=0,12 source 域)跨 4 未见市场 + LOHO host + regime,100+ 评估 ΔCRPS<0 **零例外**;source 与 unseen 互不牺牲,反而互惠。

## 候选
- LearnedSig_main(DataSignature FiLM,8.7K 参数),IAH-CRPS 3-atom 度量(负/零/正原子),frozen,不重训、不改设计、不修 host。
- 对比:PlainCore(true bypass)源域上 0.225 vs 0.237,签名无负迁移。

## 证据链(每阶段一行)
| 阶段 | 面板 | 结果 |
|---|---|---|
| SCREEN | 16 cells | 四格全负;DK1+LOHO 无坍缩;macro_s2v 0.22517 |
| 2A | Linear/MLP 32 | 32/32 ΔCRPS<0;UNSEEN_MARKET n=14 −0.080 |
| 2B | LSTM/PatchTST | UNSEEN_HOST n=3 **−0.176**;无 DEEP_HOST_COLLAPSE |
| 2C | seeds 1/2 ×40 | 3/3 同号,120/120,STABLE |
| 2D | action-chain 16 | source A2 +0.0276、DK1 +0.0539;SAFETY False |
| 2E | 扩展 6(FR/PJM2020/GEFCOM) | 6/6 ΔCRPS<0;ext A2 +0.0155;**GATING_HURTS×1** |
| 2F | Local-Core | **SOURCE_FIT_HELPS**(−0.02545)+**TARGET_TRAINED_FULLSHOT_SURPASSED**(−0.01218) |

## 关键 caveat(不推翻判定)
- **EPEX_FR:PatchTST 短证据 C3 误授权**(≈50 S3M 日,GATING_HURTS,A2 0.041<裸 A1 0.149)→ §22 LOCAL_ADAPTATION_LIMITED,动作链问题非候选问题。若续做:短证据域提高 OOS 门槛或部署前校验 full-fit vs rolling map。
- LAGO_DE:MLP 单域 LocalCore 微赢 +0.0176(±0.02 容差内)。
- R1A.9 遗留:S4 gate rare+harmful release,NEM LCB coverage<0.90。

## 文件索引
- 报告:`docs/训练文件夹/R1B/hch_v2_r1b_{stage2a..2f,final_verdict}_report_2026-08-14.md`
- 协议:`hch_v2_r1b_stage2_broad_frozen_generalization_and_action_chain_v0.1_2026-08-13.md`(§30 标签 / §31 成功标准)
- 代码:`experiments/08-hch-v2/r1b_*.py`;结果:`experiments/08-hch-v2/results/R1B_STAGE2*_2026*.tar`(gitignored,重跑生成)

## 环境状态(重要)
- **R1B 之后所有运行回本地桌面**(用户决定):纯 CPU 可跑,4090 白付。本地已验 paramiko 4.0.0 / torch 2.13.0+cpu / numpy 2.4.6。
- 服务器 hn01-ssh.gpuhome.cc:30581 不再续跑 HCH 流水线;别再租 GPU 机跑本项目。
- 下一步未定,等用户指令。
