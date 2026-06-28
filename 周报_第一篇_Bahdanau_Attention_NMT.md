# 周报：Bahdanau Attention NMT 论文复现

日期：2026-06-28

## 一、本周目标

本周围绕第三阶段 NLP 任务中的第一篇论文《Neural Machine Translation by Jointly Learning to Align and Translate》开展复现工作，重点目标是理解并实现 Bahdanau attention / RNNsearch 结构，并在公开英法机器翻译数据上完成可复现实验。

核心目标包括：

- 阅读并对照论文与 GroundHog 开源参考实现，明确 RNNsearch 的模型结构和训练配置。
- 使用 PyTorch 实现双向 RNN 编码器、加性 attention、conditional GRU decoder、maxout readout 和 beam search。
- 搭建 WMT14 EN-FR 数据下载、清洗、子词化、训练、评测、可视化的完整实验流程。
- 使用 W&B 记录训练过程，保存 checkpoint、metadata、BLEU 结果和 attention 图。

## 二、论文与机制理解

Bahdanau 等人的主要创新是将 soft attention 引入神经机器翻译，使 decoder 在生成每个目标词时不再只依赖固定长度句向量，而是动态关注源句不同位置。

本周重点理解了以下机制：

- Encoder 使用双向 GRU/RNN 为源句每个 token 生成 annotation。
- Decoder 每一步用上一时刻 hidden state 与所有 encoder annotation 计算对齐分数。
- 对齐分数经 softmax 得到 attention 权重。
- context vector 是 encoder annotation 的加权和。
- context vector 参与 decoder 状态更新和目标词预测。

attention 公式如下：

```text
e_ij = v_a^T tanh(W_a [s_{i-1}; h_j])
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij h_j
```

通过该机制，模型可以在翻译长句时缓解固定向量瓶颈，并能输出可解释的软对齐权重。

## 三、代码实现与修正

本周对项目做了较完整的工程化整理，主要包括：

- 实现并检查 RNNsearch-style 模型：BiGRU encoder、Bahdanau additive attention、conditional GRU decoder、maxout readout。
- 增加 beam search 解码，支持 length penalty、UNK suppression、no-repeat ngram。
- 增加 SentencePiece/BPE 数据处理流程，支持固定 official validation/test split。
- 增加 WMT SGM 文件转 TSV 脚本，用 newstest2013 作为验证集、newstest2014 作为测试集。
- 修复 News Commentary 中 lone carriage return 导致英法行可能错位的问题。
- 修复训练/验证 loss 统计口径，改为按非 PAD token 加权，更适合报告 PPL。
- 为评测结果增加长度比、UNK 数、平均输出长度等统计。
- 增加 checkpoint warm-start 和 resume 支持，后续大规模训练中断后可以继续。
- 导出 attention CSV 和热力图，便于在报告中展示 attention 对齐效果。

当前最佳 attention 可视化文件：

```text
outputs/nc_all_official_spm16k_30ep_attention.png
```

## 四、数据集准备

已完成的数据准备如下：

| 数据配置 | 训练样本数 | 验证集 | 测试集 | 子词词表 |
| --- | ---: | --- | --- | ---: |
| News Commentary 50k | 49,989 | newstest2013 | newstest2014 | 8k BPE |
| News Commentary 全量 | 173,482 | newstest2013 | newstest2014 | 16k BPE |
| News Commentary + Europarl | 2,011,107 | newstest2013 | newstest2014 | 30k BPE |

其中 News Commentary + Europarl 已完成 TSV 清洗与 30k BPE 编码，是目前最接近论文 30k vocabulary 设置的一档数据。

## 五、实验结果

已完成实验结果如下：

| 实验 | 训练数据 | Epoch | Valid PPL | Test PPL | BLEU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50k NC + 8k BPE | 49,989 | 30 | 96.95 | 99.89 | 0.60 |
| 全量 NC + 16k BPE | 173,482 | 20 | 39.41 | 38.38 | 6.15 |
| 全量 NC + 16k BPE | 173,482 | 30 | 28.55 | 26.02 | 10.34 |

当前最佳已完成模型：

```text
checkpoints/rnnsearch_nc_all_official_spm16k_30ep.pt
```

当前最佳 official newstest2014 BLEU 文件：

```text
outputs/nc_all_official_spm16k_30ep_newstest2014_bleu.json
```

当前正在进行的实验：

```text
News Commentary + Europarl + 30k BPE
```

该实验第 1 个 epoch 已完成，结果为：

```text
Train Loss 4.597
Train PPL 99.21
Valid Loss 4.956
Valid PPL 142.05
```

目前已从第 1 epoch checkpoint 断点恢复，继续训练第 2 至第 5 epoch。该实验数据规模更大，单个 epoch 约 20 分钟以上，预计后续 BLEU 会比 News Commentary-only baseline 更有参考价值。

## 六、问题与分析

目前主要问题如下：

- News Commentary-only 数据规模较小，领域覆盖不足，模型能生成较流畅法语，但专名、新闻事件和长句语义仍容易错位。
- 50k 小数据模型 BLEU 很低，说明 attention 机制跑通不等于论文级复现。
- 全量 News Commentary 训练到 30 epoch 后 BLEU 提升到 10.34，证明数据规模和训练轮数对结果影响很大。
- News+Europarl 数据规模明显更接近论文设置，但 RNN 模型训练速度较慢，需要 checkpoint resume 和 W&B 持续跟踪。
- 当前模型维度为 embedding 256、hidden 512，仍小于论文中的 620/1000；完整论文级复现后续应尝试更大模型。

## 七、下周计划

下周计划继续推进到更接近论文设置：

- 完成 News Commentary + Europarl + 30k BPE 的 5 epoch 训练和 official BLEU 评测。
- 根据 valid loss 曲线决定是否继续训练到 10 或 20 epoch。
- 等 CommonCrawl、UN、Giga-Fren 下载完成后，构建更完整的 WMT14 EN-FR 训练集。
- 尝试更接近论文的模型维度：embedding 620、hidden 1000。
- 对比不同数据规模下 BLEU、PPL、长度比和样例翻译质量。
- 整理 attention heatmap、实验命令、W&B 链接和最终复现报告。

## 八、本周小结

本周已经完成 Bahdanau attention 机制的 PyTorch 复现、数据处理流水线、official BLEU 评测、W&B 记录、attention 可视化和多档数据实验。当前结果说明模型结构和训练流程已经正确跑通，但要达到论文级效果仍需要更大规模 WMT14 数据、更长训练和更接近论文的模型尺寸。下一步重点是完成 News+Europarl 训练，并继续向完整 WMT14 复现推进。
