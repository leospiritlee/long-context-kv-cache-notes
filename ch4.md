# 第4章 注意力的算力账：FlashAttention 与 IO 感知

**一句话**：`FlashAttention` 是 IO 感知的精确 `attention`，用 `tiling` 把查询、键、值分块载入片上内存，配合 `online softmax` 避免物化整张注意力矩阵，把瓶颈从 `memory bandwidth` 拉回计算。

**为什么需要它**：标准 `self-attention` 会把 N×N 的注意力矩阵写到显存再读回，长序列下这块中间结果是显存与带宽的主要杀手。算力明明够，时间却花在搬运数据上。`FlashAttention` 的洞察是：注意力受内存读写而非浮点运算限制，只要分块在片上算完、不写回大矩阵，就能既精确又省显存。

**核心结构**：`FlashAttention` 把 Q、K、V 切成块，逐块载入 SRAM，在片上完成矩阵乘与 `online softmax`，滚动维护最大值与归一化因子，全程不物化完整注意力矩阵——因此是精确注意力而非近似。`IO-aware` 指它按内存层级设计数据流，减少 HBM 读写。`FlashAttention-2` 在此基础上改进工作划分与并行策略，减少非矩阵乘运算、优化 warp 与线程块分工，在长序列与更大 `head dimension` 上收益更明显。`blockwise attention` 与 `sparse attention` 则是另一类按块或按模式减少计算量的思路。

**关键设计**：综合报告口径下，`FlashAttention` 通过减少显存读写带来数倍级加速并显著降低显存峰值，是现代推理框架默认注意力内核的基础；`FlashAttention-2` 进一步逼近硬件峰值。这里的关键不是「更快的算法」，而是「IO 复杂度更低的精确算法」——它不改变数学结果，因此不需要质量回归。`half precision` 与数值精度设置会影响 `online softmax` 的稳定性，工程上要按实现核对。

**拆解**：
### 1. 瓶颈在搬运不在计算
`attention` 的浮点运算量随序列平方增长，但物化中间矩阵的显存读写同样随平方增长。长序列下后者常常先撞墙，`memory bandwidth` 成为实际瓶颈。
### 2. 分块让矩阵不必落地
`tiling` 把计算拆成能放进片上内存的小块，每块算完即时累积结果。`online softmax` 用增量方式维护归一化，避免先存整行再 softmax。
### 3. 精确注意力不需要近似
它不丢精度、不稀疏化，只是改变了数据流顺序。这与 `sparse attention`、`sliding window` 这类以近似换算力的路线本质不同，前者免费、后者要验质量。
### 4. 二代优化工作划分
`FlashAttention-2` 调整 warp 与线程块的分工、减少非矩阵乘操作，让长序列与更大头部维度上更接近硬件峰值，是许多框架的默认内核。

**今天怎么做**：确认推理框架与训练配置是否启用 `FlashAttention` 或 `FlashAttention-2`，核对支持的 `head dimension` 与精度；升级内核后跑一遍质量冒烟，确认结果与旧路径一致。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `FlashAttention` 是一种近似注意力、会掉质量。它是精确的 IO 优化，数学结果与标准注意力一致；真正需要质量回归的是稀疏与量化路线。

**记住这一句**：注意力先算 IO 账；分块加在线 softmax，精确又省带宽。

</div>
