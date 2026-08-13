# BECH / HCH-v2 研究推进札记 — 2026-08-13 夜：别把“证据不够”误诊成“模型还不够复杂”

R1A.10 出来以后，有一个非常诱人的下一步：regularized isotonic。

因为上一轮文档里本来就写过，如果 router 方向正确但 CI 不稳定，可以考虑给 isotonic 做 shrinkage 或 regularization。PJM:MLP 的结果又恰好是——mean delta 是正的，81.5% bootstrap sample 是正的，LCB 偏偏贴在 0 上。很容易产生一种感觉：只要把 C3 稍微稳定一点，也许这个 0 就过去了。

但我把代码和结果重新对在一起以后，觉得这一步现在还不能做。

原因其实很简单：PJM 的 C3 本身已经非常稳定了。7 天、14 天、30 天、60 天 fit window，它都选择 abstain。我们现在没有看到“换一点数据，isotonic map 就乱跳”的证据。

真正稀缺的是另一种东西——严格 out-of-sample 的 action event。

固定 S3M suffix 里，PJM:MLP 的 C0 只 fire 了两次，而且两次都错。C3 把两次都关掉，于是它在观测层面当然比 C0 好。但绝大多数其他日子两者都是 0，所以 paired delta 是一个极度 zero-inflated 的序列。moving-block bootstrap 重采样时只要没有抽中那两个坏 fire，样本平均差就是 0。LCB 卡在 0，并不奇怪。

这时如果为了让 LCB 变正去换一个 regularized calibrator，其实是在用模型复杂度解决一个“没有足够验证事件”的统计问题。

这是我现在最想避免的事情。

更自然的办法是回到时间序列最基本的评价方式：rolling origin。

我们以前把 S3M 硬切成 prefix 和 suffix，是为了保证 C3 fit 与 selector validation 分开，这个原则是对的。但它牺牲了大量可用历史。对一个一百多天的 S3M，可能只有最后二十来天真正进入 selector。rare-fire domain 当然很容易只留下两次 action。

如果改成 prequential cross-fitting：先拿 30 天 fit，往后 7 天纯 OOS 评价；然后历史扩大，再 fit，再评价下一 7 天……我们仍然没有让任何一天看到自己的 target，却能让剩余的大部分 S3M 日期都贡献真正 forward decision evidence。

这是我很喜欢的一种改动，因为它没有给模型增加任何能力。

IAH 不变。
C3 不变。
DVG 不变。
selector 的统计门槛甚至也不变。

只是我们不再浪费时间序列本来就很宝贵的历史。

而且这一轮还能真正回答“要不要正则 isotonic”。每一次 rolling refit 我们都把 isotonic map 保存下来。如果 PJM 的 map 随历史扩展始终长得差不多，但 CI 还是过不了，那答案非常明确：不是 calibrator variance，是 rare-event evidence 本来就不足。那时应该讨论更长的 local adaptation horizon，甚至部署后 sequential accumulation，而不是继续调 C3。

反过来，如果 map 今天把所有动作关掉，过七天突然又把一大片 utility 拉正，再过七天又完全变样，那才是 regularization 真正应该登场的时候。到那时我们有数据证明它需要被约束，而不是因为文献里有一个漂亮的方法就加进去。

我觉得这也是这篇研究慢慢形成的一个习惯：每次先问“失败来自信息、统计，还是模型”，再决定加不加东西。

R1A.10 其实已经很接近成功。它保护住了 NEM，健康 domain 也没有被乱校准，唯一没做到的就是在 S4 之前正式授权 PJM 的 C3。这个失败没有让我怀疑整个 eligibility gate，反而让我觉得它足够保守——它宁可错过一个可能有用的 local override，也不愿意因为两次坏动作就宣布有统计把握。

从部署角度看，这个性格是对的。

我们现在要做的是给它更多合法证据，而不是让它胆子变大。

所以今天这一轮的总结可以很短：

> 不要把“证据不够”误诊成“模型不够复杂”。

如果 rolling-origin 以后 PJM 真能跨过同一个 LCB>0 门槛，那我会非常愿意结束 R1A 系列。因为那意味着 HCH 不只知道怎么提出 correction、怎么翻译成 action utility，也第一次知道如何用时间序列自己的历史，在不偷看未来的情况下决定一个 local override 是否值得信。

到那时候再去服务器上跑 LSTM、PatchTST 和 DK1，我会觉得我们是在测试一套方法，而不是继续开发一套方法。
