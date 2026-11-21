# 第9章 不是所有历史都重要：attention sink 与 StreamingLLM

**一句话**：`attention sink` 指模型始终把大量注意力放在序列开头少数 token 上，`StreamingLLM` 据此只保留开头 `sink token` 加最近窗口，就能在无限长流上稳定生成。

**为什么需要它**：直觉上，流式生成该缓存最近一段 token，开头的当作无用信息丢掉。实践却相反：一旦丢掉开头，`attention` 分布会崩、生成质量骤降。`attention sink` 揭示了 KV 缓存的关键事实——开头少数 token 承担着稳定注意力的角色，不是所有历史键值都同等重要。理解这一点，才能既省缓存又不崩。

**核心结构**：研究发现模型对序列开头少数 token 有持续的高注意力，即使它们语义上并不关键，这些就是 `attention sink`。`StreamingLLM` 的方案是：保留开头几个 `sink token` 的 `KV cache`，再配合 `sliding window` 缓存最近的 token，中间部分可以丢弃。这样缓存大小恒定，却能在很长流上稳定生成。`sink token` 的作用与位置编码的相位演化、注意力分布有关，是长上下文与流式推理的交叉点。

**关键设计**：综合报告口径下，`attention sink` 说明「保留全部历史」既不必要也不高效，开头 sink 加最近窗口才是主体。这直接启发了 `KV cache` 的选择性保留策略：不是按时间一刀切地全留或全丢，而是按 token 角色区分。它与 `prefix caching` 不同——后者复用相同前缀、追求不重复算；前者决定哪些历史值得留、追求省容量。

**拆解**：
### 1. 开头 token 有特殊职责
`attention sink` 承接了大量注意力，充当分布的稳定锚。把开头当普通历史裁掉，会破坏注意力的归一化行为，导致生成崩溃。
### 2. sink 加滑动窗口构成恒定缓存
`StreamingLLM` 保留开头 `sink token` 与最近 `sliding window`，缓存不再随流长度增长，因此能支持很长的流式生成。
### 3. 这是选择性保留的起点
既然不是所有历史都重要，就可以按角色做更精细的保留与淘汰。`KV cache compression` 的许多思路，都建立在「键值重要性不均」这一观察之上。
### 4. 与前缀缓存是两回事
`prefix caching` 避免重复计算共享前缀，`StreamingLLM` 决定流式场景保留哪些历史。一个管「算不算」，一个管「留不留」，别混为一谈。

**今天怎么做**：在流式或超长场景里，确认框架是否支持 `attention sink` 保留策略；若要裁减历史，先保留开头若干 `sink token` 并维持一个 `sliding window`，再测生成是否稳定。

<div class="chapend" markdown="1">

**容易误解的地方**：以为流式生成只需保留最近 token。丢掉开头 `sink token` 会让注意力分布失稳，比少存一些历史严重得多。

**记住这一句**：开头是锚不是垃圾；sink 加窗口，流式才稳。

</div>
