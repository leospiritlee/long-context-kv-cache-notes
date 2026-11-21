## 序：能装下不等于能用好

把窗口从八千 token 拉到十万 token，模型立刻就会读长文吗？很多时候不会。它能把十万 token 接进 `context window`，却在文档中间的 `retrieval` 上失败；它能跑完一次长提示，却在第二个并发请求到来时被 `KV cache` 挤出显存；它在 `needle in a haystack` 上满分，却在多跳聚合任务上早早退化。这本专题精读写的是：从「窗口数字变大」到「长上下文真的可用」之间，工程上必须过的十二道关。

长上下文是三件事的耦合，而不是一个开关。第一件是位置编码能不能在训练长度之外继续工作，也就是 `length extrapolation` 与 `position interpolation`；第二件是注意力与 `key-value cache` 在长序列下会不会把算力和显存吃掉，也就是 `FlashAttention`、`GQA`、`PagedAttention` 这一层；第三件是评测能不能反映真实能力，也就是 `effective context length` 与 `RULER` 这类压力测试。三层任何一层掉链子，用户看到的都是同一个结果：长上下文不好用。

**分工**：与《Agent记忆与上下文工程精读》不同，那本讲 Agent 该记什么、怎么检索与压缩；与《机器学习系统精读》不同，那本讲训练与服务系统的通用底座；与《投机解码与推理加速精读》不同，那本讲用草稿模型多步验证换延迟。这本只讲一件事：模型这一侧的位置编码、注意力算子、KV 缓存与评测，怎么共同决定一段长文本能不能被稳定地读进去、算出来、服务好。

## 长上下文全景图

```text
+------------------------------------------------------------------+
| LONG CONTEXT & KV CACHE: TWELVE GATES OF ONE LONG REQUEST         |
|                                                                  |
| [1] window  -> [2] RoPE -> [3] extrapolate -> [4] attention kernel|
| [5] KV heads -> [6] cross-device -> [7] prefill/decode -> [8] page|
| [9] sink    -> [10] compress -> [11] evaluate -> [12] train/ship  |
|                                                                  |
| nominal window != effective context   measure before you scale   |
+------------------------------------------------------------------+
```

这不是一张模块清单，而是「一段长提示从进入系统到被正确回答，约束依次落在哪里」：先分清标称窗口与有效上下文，再看位置编码怎么在训练长度之外保持相位，然后才轮到注意力内核、KV 头数、显存分页、缓存压缩与评测。越往后，代价越不是配置项，而是显存账本与可验证性。

## 十二关速查表

| 关 | 主题 | 一句话 |
|---|---|---|
| 1 | 窗口与有效上下文 | `context window` 是容量，`effective context length` 才是能力 |
| 2 | 位置编码 | `RoPE` 把位置写进相位，外推问题变成频率问题 |
| 3 | 外推与插值 | `position interpolation` 压回区间，`NTK-aware scaling` 改基频，`YaRN` 分频带 |
| 4 | 注意力内核 | `FlashAttention` 用 `tiling` 与 `online softmax` 避开物化大矩阵 |
| 5 | KV 头数 | `MQA` 共享到极致，`GQA` 折中，`MLA` 压进 `low-rank` 潜空间 |
| 6 | 跨设备序列 | `ring attention` 让序列长度随设备数扩展 |
| 7 | 两阶段推理 | `prefill` 吃算力，`decode` 吃 `memory bandwidth` 与 `KV cache` 容量 |
| 8 | 显存分页 | `PagedAttention` 用 `block table` 消除碎片，`prefix caching` 复用前缀 |
| 9 | 注意力汇聚 | `attention sink` 说明不是所有历史 `KV cache` 都同等重要 |
| 10 | 压缩与卸载 | `KIVI` 把缓存压到 `2 bit`，`offload` 用传输换容量 |
| 11 | 长上下文评测 | `needle in a haystack` 只测检索，`RULER` 才压聚合与追踪 |
| 12 | 训练与边界 | `LongLoRA` 降微调成本，没有有效上下文就别堆窗口数字 |

## 四层读法：位置、算子、显存、评测

`RoPE`、`position encoding`、`base frequency`、`NTK-aware scaling`、`YaRN`、`LongRoPE` 是位置层，决定模型在训练长度之外还能不能保持可分辨的位置感。`FlashAttention`、`tiling`、`online softmax`、`GQA`、`MLA`、`ring attention` 是算子层，决定注意力在长序列上的算力与显存组织。`KV cache`、`PagedAttention`、`block table`、`prefix caching`、`attention sink`、`KIVI`、`offload` 是显存层，决定长上下文能不能同时服务多个请求。`needle in a haystack`、`RULER`、`LongBench`、`Lost in the Middle`、`context rot` 是评测层，决定你看到的数字是不是真的。

四层不能互相替代：位置编码再优雅，`KV cache` 放不进显存就服务不起来；显存管理再精细，模型早已 `context rot` 也只是更快地产出坏答案。综合报告口径建议按「先评测确认有效长度、再看算子、再算显存、最后调位置配置」的顺序推进。

## 一个可验收的出口

读完十二关，出口标准不是记住某个缩写，而是能对着一个真实的长上下文需求回答四句话：在目标长度上，模型的 `effective context length` 是否被 `RULER` 与 `needle in a haystack` 验证过；注意力内核是否支持 `FlashAttention` 或等价实现，KV 头数与缓存精度按什么 `GPU memory` 预算选定；`PagedAttention` 与 `prefix caching` 是否开启，`attention sink` 与 `sliding window` 在流式场景怎么处理；超过窗口时选外推、微调还是 `offload`，质量回归用什么基准兜底。

对做工程的人还要多一句：如果第一句答不出来，后面所有窗口配置都只是把标称数字做大。这一句答不出来，前面三句答得再完整也算没过。

## 语料边界与口径

一手论文逐字留存，用于机制与术语的精确写法：RoFormer/RoPE、ALiBi、Position Interpolation、YaRN、LongRoPE、StreamingLLM、Lost in the Middle、RULER、LongBench、Ring Attention、FlashAttention 与 FlashAttention-2、MQA、GQA、DeepSeek-V2 的 MLA、PagedAttention/vLLM、KIVI、LongLLMLingua、LongLoRA 与有效上下文长度研究。官方文档给出接入形态：vLLM 的 `PagedAttention` 与前缀缓存文档、Hugging Face 的 KV 缓存与 RoPE 缩放文档、FlashAttention、StreamingLLM、DeepSeek-V2 与 LongRoPE 仓库说明。

综合报告只补工程判断，比例、成本与趋势一律标注「综合报告口径」。十二关的顺序、场景与速查表属于「延伸，非语料原文」，不加行内代码标记；行内代码只用语料中确实出现的标识符。版本与配置以你实际使用的推理框架文档为准。