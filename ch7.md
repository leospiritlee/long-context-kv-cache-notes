# 第7章 两阶段推理：prefill、decode 与显存账本

**一句话**：`prefill` 一次性处理整个提示、计算密集并生成初始 `KV cache`，`decode` 逐 token 生成、受 `memory bandwidth` 限制且不断累积键值——两阶段瓶颈不同，优化手段也不同。

**为什么需要它**：长上下文服务的许多「慢」，定位到最后是阶段混淆：长提示的 `prefill` 拖慢了所有人的首 token，decode 阶段又被 `KV cache` 容量卡住并发。不把两阶段分开度量，就会用错手段——给 `decode` 加算力没用，给 `prefill` 减缓存也没用。

**核心结构**：`prefill` 对整段提示并行计算，产出第一个 token 并写下初始 `KV cache`，是算力受限；`decode` 每步只算一个新 token，却要对全部历史键值做注意力，是带宽与容量受限。`KV cache` 显存由层数、KV 头数、`head dimension`、`sequence length`、batch 与 `precision` 共同决定，长上下文下常超过权重本身。`chunked prefill` 把长提示切块与 decode 请求混合调度，缓解长 prefill 对其他请求的阻塞；`batching` 策略则决定显存在并发请求间如何分摊。

**关键设计**：综合报告口径下，`prefill` 吃算力、`decode` 吃 `memory bandwidth` 与 `KV cache` 容量，这一分工决定了优化路线：前者靠更好的算子与并行，后者靠更小的缓存与更快的显存。首 token 延迟（`latency`）主要受 `prefill` 影响，吞吐（`throughput`）主要受 `decode` 的并发能力影响。`memory management` 因此不是抽象话题，而是「同一张卡上能同时挂几条长请求」的具体账本。

**拆解**：
### 1. 两个阶段两种瓶颈
`prefill` 可高度并行、吃浮点算力；`decode` 每步计算量小却要把大缓存读一遍，吃带宽。诊断慢请求，第一步是分清慢在哪一阶段。
### 2. 缓存随长度与并发线性涨
`KV cache` 对每个活跃请求都要保留一份，长度与并发同时增长时显存呈乘积式压力。长上下文服务的容量上限，往往由这本账决定。
### 3. 切块预填充缓解队头阻塞
`chunked prefill` 把长提示拆成小块，插入 decode 间隙执行，避免一个超长 `prefill` 把其他请求的首 token 全部拖慢。
### 4. 分账度量才有优化方向
把 `latency` 拆成首 token 与后续 token，把 `throughput` 按阶段统计，才能判断该优化算子还是缓存容量。混在一起的平均值会掩盖真正瓶颈。

**今天怎么做**：在压测里分别记录首 token 延迟与 decode 吞吐，按并发数记录 `KV cache` 峰值；确认 `chunked prefill` 是否开启，再决定下一步是压缓存还是加算力。

<div class="chapend" markdown="1">

**容易误解的地方**：以为长上下文慢只是「算得慢」。多数情况慢在 decode 阶段反复读大缓存、以及 `KV cache` 挤占并发额度，单纯加算力不解决问题。

**记住这一句**：prefill 吃算力，decode 吃带宽；先分账，再优化。

</div>
