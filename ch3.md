# 第3章 训练长度之外：插值、外推与频率缩放

**一句话**：`position interpolation` 把超出训练长度的位置压回区间，`NTK-aware scaling` 放大基频，`YaRN` 分频带处理并校正温度，`LongRoPE` 搜索非均匀因子——四条路线都在回答同一个问题：训练长度之外位置感怎么保持。

**为什么需要它**：模型只在有限 `training length` 上见过位置，直接把 `sequence length` 拉到训练区间之外，相位演化超出熟悉范围，`context extrapolation` 迅速退化。重训成本极高，于是产生了一族「不改权重、改配置或少量微调」的长度扩展方法。知道每条路线改的是什么、代价在哪，才能按场景选而不照抄默认配方。

**核心结构**：`position interpolation`（`PI`）按比例缩小 `position index`，让所有位置落回训练范围，属内插派；`NTK-aware scaling` 从 `neural tangent kernel` 视角出发，放大 `base frequency`，让 `high frequency` 基本不变、`low frequency` 被拉伸，属改基频派；`dynamic NTK` 按当前长度动态调整缩放；`YaRN` 在 NTK 基础上按频带分别采用插值与外推，并对注意力 logits 做 `temperature scaling`；`LongRoPE` 与 `LongRoPE2` 用搜索为不同维度与长度找非均匀因子。

**关键设计**：综合报告口径下，四条路线的取舍是：`PI` 可靠但压缩短距离分辨率、通常需要微调；`NTK-aware scaling` 零样本可用但对 `high frequency` 处理不足；`YaRN` 兼顾分频与温度校正，社区实现广泛，常作默认 `rope scaling` 选项；`LongRoPE` 上限高但搜索成本与可迁移性受限。`Fourier` 视角把位置看成不同频率的分量，这解释了为什么统一缩放会同时伤到近处细节与远处依赖，也解释了分频带处理的动机。

**拆解**：
### 1. 内插把位置压回见过的区间
`PI` 的做法最直接：把位置除以一个缩放因子。所有位置回到模型熟悉的相位范围，短距离区分度随之下降，因此常配少量 `fine-tuning` 恢复分辨率。
### 2. 改基频保住高频细节
`NTK-aware scaling` 不碰 `position index`，而是放大 `base frequency`。`high frequency` 维度几乎不变，`low frequency` 被拉伸，从而零样本获得外推能力，`dynamic NTK` 进一步按长度调整。
### 3. 分频带与温度校正
`YaRN` 把频率分段处理，让 `high frequency` 保持、`low frequency` 插值，并用 `temperature scaling` 补偿长序列下注意力分布变平的问题。这是当前零样本与少样本扩展中较稳健的一档。
### 4. 搜索派的上限与成本
`LongRoPE` 与 `LongRoPE2` 不为所有维度套用一个公式，而是搜索非均匀插值因子。上限更高，但搜索本身要算力，且换模型后因子未必迁移。

**今天怎么做**：先确认模型的训练长度与目标长度之比，再选路线：小倍数扩展可先试 `YaRN` 或 `dynamic NTK`；需要稳定且能微调时用 `position interpolation`；追求极限窗口再评估 `LongRoPE`。每次扩展后都回跑 `RULER`，别只看能不能跑通。

<div class="chapend" markdown="1">

**容易误解的地方**：以为改一个 `rope scaling` 参数就万事大吉。位置编码与微调、注意力分布是耦合的，配置改了却不做 `effective context length` 回归，很可能只是让标称数字变大。

**记住这一句**：内插压位置，外推改频率；分频带更稳，搜索派更贵，改完必测。

</div>
