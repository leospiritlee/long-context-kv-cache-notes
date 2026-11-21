# 第2章 把位置写进相位：RoPE 与频率坐标

**一句话**：`RoPE` 把 `absolute position` 编码成 `rotation matrix`，作用在查询与键上，让 `self-attention` 中两 token 的内积只依赖 `relative position`，于是长上下文外推问题被转化成频率问题。

**为什么需要它**：没有 `position encoding`，`attention` 对 token 顺序不敏感，模型分不清「甲 precedes 乙」与「乙 precedes 甲」。早期用 `absolute position` 直接加位置向量，外推困难；`relative position` 方案又难与线性注意力结合。`rotary position embedding` 的价值在于：位置写在相位里，相对位置由旋转角差自然表达，既保留远程衰减，又能处理可变 `sequence length`，还能与高效内核结合。

**核心结构**：RoPE 对每个 `head dimension` 维度对施加一个角度随 `position index` 变化的 `rotation matrix`，频率由 `base frequency` 决定。低维对应 `high frequency`，对相邻位置敏感；高维对应 `low frequency`，承载远程依赖。一个 token 的位置越靠后，旋转相位越大；两个 token 的注意力分数由它们的相对位移决定，这正是 `relative position` 想要的性质。频率与 `wavelength` 一起，构成了后续所有外推方法的操作对象。

**关键设计**：既然位置信息以频率形式存在，扩展窗口就有两条路：改 `position index` 的映射，或改频率坐标本身。前者是 `position interpolation` 的思路，后者是 `NTK-aware scaling` 与 `YaRN` 的思路。理解 `base frequency`、`high frequency`、`low frequency` 与 `wavelength` 的关系，是判断某个 `rope scaling` 配方会伤到哪类信息的前提。`ALiBi` 则代表另一条路：不旋转向量，直接在注意力分数上加随距离线性增长的偏置，用 `attention with linear biases` 换取 `length extrapolation`。

**拆解**：
### 1. 相位编码为什么保留相对位置
两个向量被各自位置对应的旋转矩阵作用后，其内积只与旋转角之差有关，也就是与 `relative position` 有关。这比直接加绝对位置向量更利于泛化到未见长度。
### 2. 高频管近，低频管远
`high frequency` 维度在小小的位移上就转过明显角度，负责区分相邻 token；`low frequency` 维度转得慢，携带远距离的相对关系。外推时伤到谁、保住谁，取决于改的是哪一段频率。
### 3. 相位是周期函数带来的隐患
旋转有周期，`wavelength` 决定周期长度。当 `position index` 超出训练见过的范围，相位演化对模型而言不再熟悉，这就是 `length extrapolation` 困难的根源。
### 4. ALiBi 是另一条外推路线
`attention with linear biases` 不碰向量，只在分数上加线性距离偏置。`ALiBi` 训练短、测试长的性质说明相对偏置本身支持外推，但通用性受限，主流开源模型仍以 `RoPE` 家族为主。

**今天怎么做**：查清你用的模型在配置里如何设置 `rope scaling` 与 `base frequency`，记录训练时的 `training length`；任何长度扩展都要先说明它改的是位置映射还是频率坐标。

<div class="chapend" markdown="1">

**容易误解的地方**：以为 `RoPE` 直接编码绝对位置。它编码的是旋转相位，注意力里体现为相对位置；外推问题因此是频率问题，而不是简单的索引越界。

**记住这一句**：位置即相位，外推即频率；高频管近，低频管远。

</div>
