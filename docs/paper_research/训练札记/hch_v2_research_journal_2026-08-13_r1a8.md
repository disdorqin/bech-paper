# BECH / HCH-v2 研究推进札记 — 2026-08-13 夜：从“没有信号”走到“也许只是校准错了”

R1A.8 出来的时候，我第一感觉有一点不舒服。

前面几轮虽然不断否定自己的想法，但问题都还在单向收缩：W1 不行，看 learned key；static memory 不行，看 prequential；local evidence 有点信号，就试着和 IAH prior 融合。到了 R1A.8，PJM:MLP 给出的答案却很暧昧——0.577。

它不是 0.5，不能说完全没有信息；它又远远没到 0.65、0.7，不能理直气壮地说“representation 明明有强 action signal，只是 action head 没读出来”。更麻烦的是，我们原本希望 DVG 做最后一道保险丝，PJM:MLP 却偏偏在少数 fire days 上把错误动作放了出去。

一开始很容易顺着这个结果去造一个 tiny action head。但把动作条件重新推一遍以后，我觉得还应该再忍一轮。

Down 动作真正有收益的条件其实非常具体：

\[
r<-m/2.
\]

而在 IAH 自己的三原子世界里，这个事件的预测概率恰好就是 \(w^-\)。Up 也是一样，预测概率就是 \(w^+\)。

这一下问题突然变得比“要不要 action head”更精确：我们甚至还没有认真问过，\(w^-\) 和 \(w^+\) 在它们真正对应的动作阈值上到底校不校准。

CRPS 好，只能说明整份 distribution 在 proper score 上整体不错。它不保证“w 刚刚跨过 0.5，所以我要真的动手修价格”这个很狭窄、很关键的决策边界也可靠。

PJM:MLP 恰恰特别容易把这个问题暴露出来。它的 host 已经很强，m 很小，多数时候 Identity 本来就是对的。最后成败可能完全取决于那少量接近 action boundary 的 rare fire。只要 0.5 附近存在系统性 overconfidence，就会出现一种很有迷惑性的情况：总体 CRPS 继续改善，真正执行的动作却恰好都错。

所以我现在不想先把问题叫“representation 不够”。我更想先把它当成纯粹的 action calibration 问题。

最保守的实验甚至只需要四个参数：Down 一个 slope/intercept，Up 一个 slope/intercept。把 IAH 原来预测的 normalized action utility \(2w-1\) 做一个单调 affine recalibration。如果这么小的东西就能把 PJM:MLP 那些假阳性 fire 压回去，同时保住 NEM 那些真正有价值的动作，那么整个方法反而会比之前更漂亮：

IAH 负责 distribution；
极小 calibration map 把 distribution 翻译成 action utility；
double-event 负责结构；
DVG 负责剩余风险。

如果共享四参数不行、local 四参数行，也很有意义。那说明“有哪些 correction candidate”可以 universal，但“这个 market-host 到底什么时候值得执行”是 local evidence。这个结论和我们三层框架并不冲突，反而会让分工更清楚。

如果连 isotonic 这种低容量、单调、几乎把“calibration 能做什么”推到上限的东西都救不了，我也会死心，不再把问题叫 calibration。那时才真正有证据说：现有 IAH distribution 缺的是 action discrimination，必须引入新的信息或新的 mapping。

我越来越喜欢现在这种推进方式。每次只给一个假设很小的生存空间。它活下来，就纳入方法；它死掉，就记录下来，不用十个新参数把它抢救回来。

服务器的事情反而简单了。智川云支持实例创建后继续扩盘，而且不能缩容，所以没必要现在就买 500GB。R1B 开始时把总数据盘扩成 200GB 已经很舒服；等 U0 真开始再加到 500GB。最重要的不是省那几块钱，而是不在方法还没走到 U0 时提前给一个不能缩容的大盘付长期成本。

还有一个很实际的提醒：系统盘只有 30GB。后面服务器建立时，Hugging Face 和 Torch cache 绝对不能让它们默认堆在 root 目录，第一天就应该全部指向数据盘。

今天我会把项目状态描述成一句话：

> 我们已经有了一个可信的 correction distribution，但还没有一个可信的 action translation。

这句话其实很好。

它说明前半段已经站住了，后半段仍然在被实验塑形。

下一轮 R1A.9，我最想看到的不是更高的 CRPS，也不是更高的 AUC，而是一张最朴素的图：横轴是 IAH 说“这个动作有多值得做”，纵轴是这个动作实际上到底有多值得做。

如果这条关系只是歪了，我们把它扶正。

如果这条关系根本不存在，我们就承认必须换问题。
