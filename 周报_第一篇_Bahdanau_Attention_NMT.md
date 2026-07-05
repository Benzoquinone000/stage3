# 周报：Bahdanau Attention NMT 论文复现

日期：2026-07-05

## 一、本周目标

本周围绕第三阶段第一篇论文《Neural Machine Translation by Jointly Learning to Align and Translate》开展较严格的复现实验。该论文提出的 RNNsearch 模型首次将 soft attention 机制系统用于神经机器翻译，使 decoder 在生成每个目标词时可以动态关注源句不同位置，从而缓解传统 encoder-decoder 固定长度向量表示在长句翻译上的瓶颈。

本周主要目标包括：

- 阅读论文与 GroundHog 参考实现，梳理 RNNsearch-50 的数据、模型和训练配置。
- 从官方 WMT14 EN-FR 数据重建论文使用的 348M source words selected corpus。
- 使用 PyTorch 实现并训练接近论文设置的 Bahdanau attention / RNNsearch 模型。
- 在 newstest2014 上计算 BLEU，并与论文 Table 1 中 RNNsearch-50 / RNNsearch-50? 结果对比。
- 分析当前复现实验与原论文之间的差异，以及这些差异对最终 BLEU 的影响。

## 二、论文方法理解

论文的核心贡献是引入 attention 机制。传统 encoder-decoder 模型会将整个源句压缩为一个固定维度向量，然后 decoder 仅依赖该向量生成目标句。该方式在长句上容易丢失信息。Bahdanau attention 则为源句每个位置生成 annotation，并在 decoder 每一步计算当前 hidden state 与所有源端 annotation 的匹配程度，得到 soft alignment。

基本过程如下：

```text
e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij h_j
```

其中 `alpha_ij` 表示生成第 `i` 个目标词时对第 `j` 个源词的关注程度，`c_i` 是根据 attention 权重得到的上下文向量。该机制使模型在翻译时能够动态选择源句信息，因此在长句翻译上明显优于无 attention 的 RNNencdec。

## 三、数据准备与严格复现流程

论文使用 WMT14 English-to-French 任务，训练数据来自多个官方平行语料，并使用 Axelrod / Moore-Lewis 风格的数据选择方法选出约 348M English source words 的训练子集。由于论文中原作者提供的 LIUM selected corpus 链接当前已经失效，本周采用官方 WMT14 数据重新构建 selected corpus。

本周完成的数据准备如下：

| 项目 | 当前复现设置 |
| --- | --- |
| 训练语料来源 | News Commentary v9、Europarl v7、Common Crawl、UN、Giga-Fren |
| 数据选择方法 | 以 newstest2012 + newstest2013 为 in-domain 数据，进行 cross-entropy / Moore-Lewis 风格筛选 |
| selected corpus | 13,538,554 句对，348,000,013 English source words |
| RNNsearch-50 过滤后训练集 | 11,992,626 句对 |
| source words / target words | 257,613,746 / 295,415,338 |
| 验证集 | newstest2012 + newstest2013，共 6,003 句对 |
| 测试集 | newstest2014，共 3,003 句对 |
| 分词 | Moses tokenization，word-level |
| 词表 | source / target 各 30,000 words，另加 4 个 special tokens |
| 训练长度限制 | source 和 target 均 <= 50 |
| dev/test 长度限制 | 不过滤，完整评测 |

为了避免 1,199 万句训练数据一次性加载导致内存压力，本周还实现了 indexed TSV 训练路径。训练时只保存 offset / length sidecar，通过索引按需读取训练样本，同时保留 sort-k-batches 的 padding reduction 策略。

## 四、模型与训练配置

当前模型对齐论文 RNNsearch-50 / GroundHog 配置：

| 配置项 | 论文设置 | 当前复现 |
| --- | ---: | ---: |
| source/target vocab | 30,000 words | 30,000 words + special tokens |
| embedding dim | 620 | 620 |
| encoder hidden | forward/backward 各 1000 | BiGRU 各 1000 |
| decoder hidden | 1000 | 1000 |
| alignment hidden | 1000 | 1000 |
| maxout hidden | 500 | GroundHog-style maxout，1000 / 2 = 500 |
| optimizer | Adadelta | Adadelta |
| learning rate | 1.0 | 1.0 |
| rho / eps | 0.95 / 1e-6 | 0.95 / 1e-6 |
| gradient clipping | L2 norm <= 1 | 1.0 |
| sort-k-batches | 每 20 个 batch 按长度排序 | 20 |
| precision | fp32 | fp32，AMP 关闭 |
| batch size | 80 | 240 |

其中唯一明确的训练超参数偏离是 batch size。论文使用 batch size 80；当前为了充分利用 RTX 5090 显存，将 batch size 提高到 240。学习率没有随 batch size 等比放大，因为论文使用的是 Adadelta，自适应优化器不适合简单套用大 batch SGD 的线性缩放规则。当前学习率仍保持论文值 1.0。

## 五、实验结果

### 5 epoch 主实验

首先按照论文 longer RNNsearch-50? 的数据遍数跑 5 epoch，但 batch size 使用 240。

前 5 个 epoch 的训练曲线如下：

| Epoch | Train Loss | Train PPL | Valid Loss | Valid PPL | Time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.4112 | 30.30 | 2.6895 | 14.72 | 164m 48s |
| 2 | 2.3347 | 10.33 | 2.3980 | 11.00 | 164m 39s |
| 3 | 2.1441 | 8.53 | 2.2925 | 9.90 | 164m 42s |
| 4 | 2.0460 | 7.74 | 2.2360 | 9.36 | 164m 52s |
| 5 | 1.9823 | 7.26 | 2.1995 | 9.02 | 165m 21s |

可以看到，前 5 个 epoch 中 train loss 和 valid loss 都持续下降，没有出现过拟合或训练发散。第 5 个 epoch 结束后，使用 best checkpoint 在 newstest2014 上进行 test loss 和 BLEU 评测：

| Checkpoint | Test Loss | Test PPL | all BLEU |
| --- | ---: | ---: | ---: |
| epoch 5 best | 1.8710 | 6.49 | 27.6856 |

该结果已经超过论文普通 RNNsearch-50 的 all-sentence BLEU：

```text
论文 RNNsearch-50:  26.75 BLEU
当前 5 epoch:       27.69 BLEU
论文 RNNsearch-50?: 28.45 BLEU
```

不过，5 epoch 主实验尚未达到论文 longer RNNsearch-50? 的 28.45 BLEU。

### 继续训练实验

为了验证差距是否主要来自 batch size 变大后 update 数不足，本周继续从 5 epoch checkpoint 逐 epoch 续训，并在每个 epoch 后立即用 beam size 10 在 newstest2014 上计算 BLEU。

结果如下：

| Epoch | Train Loss | Valid Loss | Test Loss | all BLEU |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1.9823 | 2.1995 | 1.8710 | 27.6856 |
| 6 | 1.9359 | 2.1738 | 1.8404 | 27.9523 |
| 7 | 1.9000 | 2.1559 | 1.8191 | 28.1177 |
| 8 | 1.8710 | 2.1422 | 1.8033 | 28.2377 |

可以看到，valid loss 和 BLEU 在继续训练后持续改善。到 epoch 8 时，all-sentence BLEU 已经达到 28.2377，距离论文 RNNsearch-50? 的 28.45 只差约 0.21 BLEU。

### no-UNK 子集结果

论文 Table 1 除了报告全测试集 BLEU，还报告了 no-UNK 子集 BLEU，即源句和参考译文中都不含词表外词的测试句。当前按 30,000 词 shortlist 统计，newstest2014 中符合条件的句子为 737 / 3,003。

| Epoch | all BLEU | no-UNK BLEU |
| ---: | ---: | ---: |
| 5 | 27.6856 | 35.2032 |
| 6 | 27.9523 | 35.3526 |
| 7 | 28.1177 | 35.8757 |
| 8 | 28.2377 | 35.7913 |

与论文结果对比：

| 模型 | all BLEU | no-UNK BLEU |
| --- | ---: | ---: |
| 论文 RNNsearch-50 | 26.75 | 34.16 |
| 当前 epoch 8 | 28.2377 | 35.7913 |
| 当前最佳 no-UNK epoch 7 | 28.1177 | 35.8757 |
| 论文 RNNsearch-50? | 28.45 | 36.15 |

当前复现已经超过论文普通 RNNsearch-50，在 all BLEU 和 no-UNK BLEU 上都接近 longer RNNsearch-50?。

## 六、实验耗时统计

本周实验的主要耗时如下。这里统计的是严格复现实验中已经完成并记录日志的阶段，不包含早期反复尝试、网络下载等待和人工检查时间。

| 阶段 | 时间范围 / 单项耗时 | 说明 |
| --- | ---: | --- |
| selected corpus 构建与 word-level 数据准备 | 约 2小时48分钟 | 从 dev/test 处理、KenLM 打分、sort-select 到 word-level max50 与 vocab 重建 |
| 5 epoch 主训练 | 13小时44分22秒 | epoch 1-5，batch size 240，fp32 |
| epoch 5 BLEU 解码 | 约 22分钟 | beam size 10，newstest2014 全 3,003 句 |
| epoch 6 续训 | 2小时45分23秒 | 从 epoch 5 checkpoint 恢复 |
| epoch 6 BLEU 解码 | 约 12分钟 | beam size 10 |
| epoch 7 续训 | 2小时45分18秒 | 从 epoch 6 checkpoint 恢复 |
| epoch 7 BLEU 解码 | 约 12分钟 | beam size 10 |
| epoch 8 续训 | 2小时45分11秒 | 从 epoch 7 checkpoint 恢复 |
| epoch 8 BLEU 解码 | 约 14分钟 | beam size 10 |

训练部分合计耗时：

```text
前 5 epoch 训练: 13小时44分22秒
epoch 6-8 续训: 8小时15分52秒
epoch 1-8 训练合计: 22小时00分14秒
```

如果加上数据选择与预处理、4 次 BLEU 解码，严格主实验从数据准备完成到 epoch 8 评测结束约为 25 小时量级。相比论文 RNNsearch-50? 在 Quadro K-6000 上约 252 小时的训练耗时，当前通过 RTX 5090 和更大 batch size 显著压缩了实验周期，但代价是相同 epoch 下 update 数少于论文设置。

## 七、与原论文的主要差异

当前实验并不是 bit-level 原论文复现，而是论文设置高度对齐的 PyTorch 复现。产生差异的核心背景是：本周时间窗口有限，而 WMT14 EN-FR 严格复现的训练成本非常高。论文中的 RNNsearch-50? 在 Quadro K-6000 上训练约 252 小时；如果完全按照 batch size 80 和论文 update 数重新跑，单轮实验周期会明显超出本周可用时间。因此，本周优先保证数据、模型、优化器和评测流程完整跑通，并通过增大 batch size 充分利用 RTX 5090 显存，在有限时间内得到可对照的阶段性结果。

主要差异如下。

### 1. Batch size 不同

论文使用 batch size 80，当前使用 batch size 240。这个调整主要是受本周实验时间限制影响：如果完全使用 batch size 80，虽然最严格，但单个 epoch 的 update 数约为当前的 3 倍，训练时间会明显增加；为了在本周内完成 WMT14 规模训练、BLEU 评测和续训验证，本周采用 batch size 240 来提高吞吐并充分利用显卡。

batch size 变大后，同样 5 epoch 的参数更新次数会显著减少。

当前训练集大小为 11,992,626 句对：

```text
batch=240 时，每 epoch 约 49,970 updates
5 epoch 约 249,850 updates
8 epoch 约 399,760 updates
```

论文 longer RNNsearch-50? 表中报告约 667,000 updates。也就是说，即使当前已经跑到 8 epoch，update 数仍低于论文 longer run。这是当前与论文 RNNsearch-50? 仍有 0.21 BLEU 差距的最主要原因。换言之，当前差距不是因为 attention 机制或模型主体实现失败，而是由于时间限制下采用了更大 batch size，导致在相同 epoch 数内完成的参数更新次数不足。

这一判断有实验现象支持：从 epoch 5 到 epoch 8，valid loss 持续下降，BLEU 也从 27.6856 提升到 28.2377，说明模型仍然受益于更多 update，并未完全收敛。

### 2. Selected corpus 是重建版

论文曾经提供过 LIUM selected corpus 下载链接，但该链接目前已经失效。因此当前使用官方 WMT14 数据重新构建 348M source words selected corpus。

重建过程严格遵循论文思路，但仍可能与原作者当年的 selected file 存在差异：

- 原作者具体数据清洗细节无法完全确认。
- Moore-Lewis 数据选择实现细节、语言模型平滑、排序稳定性可能不同。
- 当前 max50 后训练集为 11,992,626 句对，而论文从 update 数和 batch size 反推训练集规模约在 1,000 万句对量级，两者接近但不完全相同。

这类数据选择差异会影响词频、句子分布、UNK 比例和最终 BLEU。

### 3. 框架与实现不同

原论文使用 GroundHog / Theano 实现，当前使用 PyTorch 复现。虽然模型结构、初始化、Adadelta、gradient clipping、maxout readout 等尽量对齐，但仍可能存在细节差异：

- GRU/conditional GRU 的具体门控实现可能与 GroundHog 源码不完全一致。
- 参数初始化已按论文风格实现，但随机数和矩阵布局不可能完全一致。
- beam search、长度处理、UNK 输出和后处理细节可能与论文实现不同。
- 当前没有进行 UNK replacement 或额外 rare word 后处理，输出中仍有 `<unk>`。

这些差异通常会带来小幅 BLEU 波动。

### 4. BLEU 计算口径可能存在微小差别

当前使用 `sacrebleu`，在 tokenized newstest2014 上采用 `tokenize=none`。论文时代常用的是 Moses multi-bleu 风格的 tokenized BLEU。两者整体可比，但不是完全同一个脚本环境。对于追求小数点级别一致的复现，这会带来轻微差异。

### 5. 尚未完整复现 Table 1 的所有对照

受时间限制，本周重点完成了最核心的 RNNsearch-50 主线及 longer training 方向。论文 Table 1 还包括：

- RNNencdec-30
- RNNsearch-30
- RNNencdec-50
- RNNsearch-50
- RNNsearch-50?

其中 RNNencdec 是无 attention baseline，目前项目中尚未实现完整无 attention baseline；RNNsearch-30 也尚未单独训练。因此，当前已经能比较 RNNsearch-50 / RNNsearch-50? 主结果，但还不能声称完整复现论文所有对照实验。这部分不是方法上不可行，而是在本周时间内优先级低于主模型严格训练和 BLEU 对齐，计划放到下一阶段补齐。

## 八、为什么结果与论文仍有差异

综合当前实验，差异主要来自以下几方面。

第一，受时间限制，本周采用了更大的 batch size 以提高训练吞吐，但这也导致 update 数不足。当前 batch size 是论文的 3 倍，同样 epoch 下 update 数约为论文设置的三分之一。5 epoch 时 BLEU 为 27.69，明显低于 RNNsearch-50?；继续训练到 epoch 8 后 BLEU 提升到 28.24，差距大幅缩小。这说明更多 update 对结果仍然有效，因此时间限制下的 batch size 调整是最主要原因。

第二，原始 selected corpus 不可直接获得。虽然当前重建了 348M source words 的 selected corpus，但无法保证与原作者数据完全一致。机器翻译对训练数据分布较敏感，尤其是 rare words、专名、新闻领域句子比例等，会影响 final BLEU。

第三，解码和 UNK 处理仍可能存在差异。论文指出 unknown / rare words 是系统的一个挑战，并额外报告 no-UNK 子集。当前输出中仍有较多 `<unk>`，例如 epoch 8 的 hypothesis 中有 6,103 个 `<unk>` token。no-UNK 子集 BLEU 已经接近论文 RNNsearch-50?，说明 rare word / unknown word 处理仍是影响全测试集 BLEU 的重要因素之一。

第四，PyTorch 复现与 GroundHog 原实现存在工程细节差异。即使宏观配置一致，RNN 内部实现、数值稳定性、初始化顺序、batch 排序方式和 beam search tie-breaking 都可能导致最终模型略有不同。

## 九、本周结论

本周完成了 Bahdanau attention / RNNsearch-50 在 WMT14 EN-FR 上的较严格复现。当前结果可以总结为：

- 数据侧已经从官方 WMT14 语料重建出 348M source words selected corpus。
- 模型侧实现了 paper-aligned RNNsearch-50 配置。
- 训练侧完成了 fp32、Adadelta、clip=1、sort-k-batches=20 的大规模实验。
- 评测侧完成了 all-sentence BLEU 和 no-UNK subset BLEU。
- 当前 epoch 8 all BLEU 为 28.2377，接近论文 RNNsearch-50? 的 28.45。
- 当前最佳 no-UNK BLEU 为 35.8757，接近论文 RNNsearch-50? 的 36.15。

从实验趋势看，当前未完全达到论文 RNNsearch-50? 的主要原因并不是模型结构错误，而是在本周有限时间内为了完成完整流程而放大 batch size，导致 update 数仍不足；此外还存在 selected corpus 重建和实现细节差异。继续增加 update 数、或在时间允许时补跑 batch size 80 的严格实验，有较大希望进一步逼近甚至达到论文 RNNsearch-50?。

## 十、下周计划

下周计划继续推进以下工作：

- 继续从 epoch 8 checkpoint 训练，观察是否能超过 28.45 BLEU。
- 计算更细的句长分桶 BLEU，对应论文 Figure 2 的长句鲁棒性分析。
- 实现无 attention 的 RNNencdec baseline，补齐 Table 1 中 RNNencdec-30 / RNNencdec-50 对照。
- 准备 max length 30 的 RNNsearch-30 数据与训练脚本，复现 RNNsearch-30。
- 分析 `<unk>` 产生原因，尝试 UNK replacement 或 copy-style 后处理，观察对 all BLEU 的影响。
- 整理最终实验报告，包括模型结构图、attention 可视化样例、训练曲线、BLEU 对照和差异分析。
