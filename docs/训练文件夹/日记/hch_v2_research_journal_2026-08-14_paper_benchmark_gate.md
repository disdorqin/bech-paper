# BECH / HCH-v2 研究推进札记 — 2026-08-14：现在终于要和同行正面打一次了

前面几天很多实验回答的是“这条方法内部合理不合理”。现在终于到了更残酷的问题：它到底打不打得过别人。

这轮我刻意同时保留两个价值观。

第一，追结果。外国 headline benchmark 不是平均提升一点就算过。内部最低线放到 60% 的 dataset×host×metric cell Top-1 或统计并列 Top-1，70% 才算真正强。山东更苛刻。调学习率、宽度、context、sampler、readout 都可以，只要不拿最终 sealed test 反复调。

第二，不能靠口径把输写成赢。30% 不叫多数。完整表格必须留下来。内部 gate 只是告诉我们什么时候继续改方法。

我最期待的其实是“有结构的失败”。例如 CRPS 几乎都赢，但 MAE 输，那说明 IAH probabilistic core 是真的，distribution→point 的读出太保守。三原子自己已有 expected residual：w+ m+ - w- m-。这可以成为第一刀，而不需要动 CRPS 或加新 loss。

如果连 CRPS 都输，才说明问题更深，要看 capacity、context、signature、mixture，最后才考虑三个 atom 是否够。

论文叙事也越来越清楚。v0.4 里的 CAGM/W1 已经被实验淘汰为主线，我们没必要为了早期文档一致性把它硬塞回最终方法。留下来的主线反而更有辨识度：host-relative signed distribution、data-signature modulation、evidence-authorized local calibration、structured event correction、abstention。

还有一个公平问题：公开 benchmark 的 universal 权重不要混山东私有数据。公开模型只吃公开数据；山东另外展示 frozen transfer 和 private adaptation。这样公开主表仍然可复现。

PIR 和 δ-Adapter 也不值得变成一个长期工程。只要在官方 reference setting 上复现相近的改善，就冻结。PIR 对不上就先补已知 retrieval；δ 对不上就核 output correction、normalization、training。够了。

这轮之后会真正分岔。

如果 60–70% 以上主 cell Top-1/tied，我会愿意把数学 core 冻住。后面的扩数据、DA/RT、大规模训练，是给一个已经会赢的模块扩大疆域。

如果四成左右，就找清楚拖后腿的层，再改一刀。

如果连三成都不到，就别拿“R1B 泛化很好”安慰自己。泛化一个不够强的模块没有意义。

从这一轮开始，表格会告诉我们真话。
