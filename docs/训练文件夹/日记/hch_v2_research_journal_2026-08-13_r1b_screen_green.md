# BECH / HCH-v2 研究推进札记 — 2026-08-13 夜：DK1 成功以后，反而更不能急

这次回来看到 `R1B_SCREEN_HEALTHY`，第一感觉当然是高兴。

因为这不是 R1A 那种“在熟悉市场里把机制修好”的 GREEN。DK1 没给 candidate 梯度，LOHO 的 PatchTST 也真的没有进训练，但 candidate 还是能比 host CRPS 好。特别是 NEM 那种非常差的 host regime 和 PJM 那种已经很强的 host regime 同时存在，说明 HCH 确实没有只学一个“弱模型修正器”。

但我越看 runner，越觉得这时候反而不能兴奋过头。

第一个原因是一个很小但很典型的语义 bug。

runner 把 `PatchTST` 这个名字永久写成了 `host_seen=False`。于是 main candidate 明明训练时已经看过 PatchTST，四格汇总里却仍然叫它 unseen host。LOHO 的结论是真的，因为那一组确实删了 PatchTST；main 的两个 unseen-host 格子只是标签错了。

这个 bug 没让模型作弊，也没有改变 CRPS，但它提醒我：泛化实验最容易出问题的地方不一定是神经网络，而是“我们到底给某个结果起什么名字”。

所以 R1B 后面我想把 transfer taxonomy 做得更严格。不是简单 seen/unseen 两个布尔值，而是区分未见市场、同市场新数据集、未见 schema、未见 host，后面还有 DA/RT target category。

第二个原因是 DK1 终究只有一个市场。

报告里说“DK1 16 行都负”，听起来很强，但那其实是四个 host 乘四种 candidate-screen 配置，不是十六个独立市场。数字没有错，但独立证据量不能按 16 算。

如果我们现在直接进入 action-chain，然后把 DK1 做漂亮，我很担心几天以后又变成“DE/PJM/NEM/DK1 已经被我们看得太熟”。

好在仓库里已经有一批一直没参与 universal candidate 训练的市场。EPEX_FR/BE/NL、NordPool FI/NO/SE3，还有 DE_EPEX、PJM_2020、GEFCom。它们现在最大的价值不是加入训练，而是用来攻击我们。

我甚至不想先上所有深模型。

先用 Linear/MLP 快速扫过去，看看一个 source 只见过三个市场的 corrector 到底在多少 holdout 上还能保持 delta CRPS 为负。这个实验很便宜，但如果结果不好，它会比后面十个 action-chain 表格更早告诉我们“universal 还没有成立”。

其中我尤其喜欢把 DE_EPEX/PJM_2020 单独列出来。它们不是完全新市场，而是同 market family 下换数据来源或时间 regime。这正好对应用户一直强调的“不同数据集”。以后论文里我们可以更细地说：市场不变时换数据，和市场一起换，哪个更难。

GEFCom 又是另一种压力。更老的价格 regime、不同 exogenous availability、没有现代负价市场那种尾部。如果 HCH 在那里还能工作，才说明 host-relative geometry 可能真的比“识别负价市场”更一般。

这次同行文献给我的最大提醒也不是某个新模块。

Moirai 在讲 universality 时，本来就在拆 frequency、variate、distribution；GIFT-Eval 又把 non-leaking unseen datasets 做成核心纪律。电价 transfer-learning 论文已经证明跨市场知识可以传，但它们多数允许 target fine-tune。我们的 corrector 如果不更新 candidate 还能 transfer，其实是更难的一件事。

所以现在最重要的是把难度真的做出来，而不是因为一次 DK1 成功就把更多市场加进 source training。

还有一个 provenance 小问题：服务器跑的时候没有 `.git`，config 里 SHA 是 unknown。虽然代码后来同步回本地提交了，Stage-1 不至于失效，但从下一正式 run 开始必须纠正。科研里这种东西很无聊，却会决定半年以后我们能不能重新跑出同一张表。

我现在对 R1B 的节奏想得更明确了：

先修实验语义；
再扩 holdout 面；
再看 seed；
最后才跑 action chain。

如果 broad panel 失败，说明 candidate 泛化还没站稳，那就不应该让 C3/DVG 替它遮羞。

如果 broad candidate 站稳、action chain 失败，反而非常好定位：问题就在 local adaptation，不在 universal core。

如果两层都站稳，那才值得开始 Local-Core 对照、R1C DA/RT，以及后面的 MOMENT distillation。

这次 GREEN 让我第一次真的觉得 HCH 有“跨域模块”的味道。

也正因为如此，我更不想用一个太容易的验证把它提前宣布成功。

现在应该做的事情，是找更多它没见过的东西，认真尝试把它弄坏。
