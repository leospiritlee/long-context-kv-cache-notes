# 第8章 显存分页与前缀复用：PagedAttention 与 vLLM

**一句话**：`PagedAttention` 把 `KV cache` 按固定块管理、用 `block table` 映射逻辑与物理位置，`prefix caching` 复用相同前缀的键值——一个治碎片，一个治重复计算。

**为什么需要它**：连续预留显存给每条序列会造成内部与外部碎片，实际利用率很低；多条请求共享同一段 `system prompt` 时，又会对相同前缀反复 `prefill`。长上下文把这两个浪费都放大：序列越长，预留浪费越大；前缀越长，重复计算越贵。

**核心结构**：`PagedAttention` 借鉴操作系统分页，`KV cache` 被切成固定大小的块，逻辑上连续的序列在物理上不必连续，由 `block table` 做映射。`vLLM` 以此为核心，几乎消除碎片与预留浪费。`prefix caching` 复用相同前缀的键值缓存，`automatic prefix caching` 用类似基数树的结构管理前缀；面向 API 的形态称为 `prompt caching`。`virtual memory` 的类比说明了这一设计的来源：把显存当可被分页管理的地址空间。

**关键设计**：综合报告口径下，`paged attention` 把显存利用率推到接近理论上限并显著提升 `throughput`，是现代大模型服务的基石；`prefix caching` 在多轮对话、共享 `system prompt` 与长文档问答中大幅降低首 token 延迟与重复计算，常是性价比最高的一项。两项优化都属系统层，不改模型权重，因此不带来质量损失；它们的收益取决于请求模式——前缀越重复、长度越参差，收益越大。

**拆解**：
### 1. 分页治的是碎片
按最大长度连续预留会浪费大量显存。`block table` 让序列按需取块，物理上分散、逻辑上连续，显存利用率因此接近理论上限。
### 2. 前缀缓存治的是重复
相同 `system prompt` 或文档前缀在多请求间重复出现，`prefix caching` 只算一次、后续复用。`automatic prefix caching` 把前缀组织成可共享的结构，命中率随重复度上升。
### 3. 两项优化都不改模型
分页与前缀复用发生在服务系统层，注意力数学结果不变。它们与 `quantization`、`offload` 这类以质量或延迟换容量的手段性质不同。
### 4. 收益看请求分布
前缀高度重复、长度差异大的负载最受益；请求彼此毫无共享时，`prefix caching` 几乎无用。上线前应先看真实流量分布。

**今天怎么做**：确认服务框架是否启用 `PagedAttention` 与 `automatic prefix caching`，把共享前缀（如 `system prompt`）放在请求最前面以获得命中；用真实流量回放测命中率与吞吐变化。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `PagedAttention` 会降低精度或改变模型输出。它只改变显存的物理组织与映射，数学计算与结果不变。

**记住这一句**：分页治碎片，前缀复用治重复；都不改模型，收益看流量。

</div>
