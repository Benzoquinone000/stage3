# Bahdanau Attention NMT 论文复现

本仓库用于复现论文：

> Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.  
> Neural Machine Translation by Jointly Learning to Align and Translate.

论文提出的 RNNsearch 模型将 soft attention 引入神经机器翻译，使 decoder 在生成每个目标词时动态关注源句不同位置，从而缓解传统 encoder-decoder 固定长度向量在长句翻译中的信息瓶颈。

本项目使用 PyTorch 复现 Bahdanau attention / RNNsearch，并尽量对齐论文在 WMT14 English-French 任务上的设置。

## 当前复现进度

已完成第一篇论文的主线复现：

- 使用官方 WMT14 EN-FR 数据重建 348M source words selected corpus。
- 实现 RNNsearch-50：双向 GRU encoder、additive attention、conditional GRU decoder、GroundHog-style maxout readout。
- 使用论文配置训练：Adadelta、lr=1.0、rho=0.95、eps=1e-6、gradient clip=1、embedding=620、hidden=1000。
- 在 newstest2014 上完成 beam search 翻译和 BLEU 评测。
- 完成 all-sentence BLEU 与 no-UNK subset BLEU 对照。
- 完成中文周报与严格复现实验记录。

## 实验结果

当前最佳结果来自 batch size 240 的继续训练实验。

| 模型 / Checkpoint | all BLEU | no-UNK BLEU |
| --- | ---: | ---: |
| 论文 RNNsearch-50 | 26.75 | 34.16 |
| 当前 epoch 8 | 28.2377 | 35.7913 |
| 当前最佳 no-UNK epoch 7 | 28.1177 | 35.8757 |
| 论文 RNNsearch-50? | 28.45 | 36.15 |

训练曲线摘要：

| Epoch | Train Loss | Valid Loss | Test Loss | all BLEU |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1.9823 | 2.1995 | 1.8710 | 27.6856 |
| 6 | 1.9359 | 2.1738 | 1.8404 | 27.9523 |
| 7 | 1.9000 | 2.1559 | 1.8191 | 28.1177 |
| 8 | 1.8710 | 2.1422 | 1.8033 | 28.2377 |

当前结果已经超过论文普通 RNNsearch-50，并接近 longer RNNsearch-50?。与论文 RNNsearch-50? 仍有约 0.21 BLEU 差距，主要原因是本周受时间限制，将 batch size 从论文的 80 调整到 240，导致相同 epoch 数下 update 数少于论文设置。

## 与论文设置的主要差异

当前复现是论文设置高度对齐的 PyTorch 复现，不是 bit-level GroundHog/Theano 原版复现。主要差异包括：

- batch size：论文为 80，当前主实验为 240。
- update 数：batch size 变大后，同样 5 epoch 的参数更新次数约为论文设置的三分之一。
- selected corpus：论文原始 LIUM selected corpus 链接已失效，当前使用官方 WMT14 数据重建。
- 实现框架：论文使用 GroundHog/Theano，当前使用 PyTorch。
- 解码细节：beam search、UNK 处理、BLEU 脚本可能与原论文存在轻微差别。

更多分析见：

- [周报_第一篇_Bahdanau_Attention_NMT.md](周报_第一篇_Bahdanau_Attention_NMT.md)
- [STRICT_REPRO_STATUS.md](STRICT_REPRO_STATUS.md)
- [STRICT_EXPERIMENT_LOG.md](STRICT_EXPERIMENT_LOG.md)
- [STRICT_PAPER_REPRODUCTION.md](STRICT_PAPER_REPRODUCTION.md)

## 项目结构

```text
.
├── nmt_attention/
│   ├── data.py                  # 词表、TSV 数据集、indexed dataset、bucket sampler
│   ├── model.py                 # Encoder、BahdanauAttention、Decoder、Seq2Seq
│   ├── training.py              # 训练、验证、AMP/optimizer 工具
│   └── wandb_utils.py           # W&B 日志工具
├── scripts/
│   ├── download_wmt14_enfr.py
│   ├── download_moses_tokenizer.py
│   ├── prepare_moore_lewis_selection.py
│   ├── run_paper_except_bs240_rnnsearch50_5ep.sh
│   └── run_continue_3x1ep_with_bleu.sh
├── train.py                     # 训练入口
├── evaluate_bleu.py             # BLEU 评测入口
├── translate.py                 # 单句翻译与 attention 导出
├── STRICT_REPRO_STATUS.md       # 当前严格复现状态
├── STRICT_EXPERIMENT_LOG.md     # 实验日志
└── 周报_第一篇_Bahdanau_Attention_NMT.md
```

本仓库不会上传大文件。以下目录被 `.gitignore` 排除：

```text
data/
checkpoints/
outputs/
wandb/
logs/
tools/
references/
```

## 环境安装

建议使用 Python 3.10+ 与 PyTorch CUDA 环境。

```bash
pip install -r requirements.txt
```

如需使用 W&B：

```bash
wandb login
```

## 快速运行

小规模 debug：

```bash
python train.py --preset debug
```

教程数据训练：

```bash
python train.py --preset tutorial
```

自定义 TSV 训练：

```bash
python train.py \
  --preset paper \
  --data-path path/to/parallel.tsv \
  --source-col 0 \
  --target-col 1
```

## 严格数据准备流程

下载 WMT14 EN-FR 官方数据：

```bash
python scripts/download_wmt14_enfr.py \
  --profile paper-full \
  --output-dir data/wmt14_enfr \
  --yes \
  --extract
```

下载 Moses tokenizer：

```bash
python scripts/download_moses_tokenizer.py
```

构建论文风格 selected corpus：

```bash
python scripts/prepare_moore_lewis_selection.py --stage devtest
python scripts/prepare_moore_lewis_selection.py --stage sample-general --general-sample-lines 100000
python scripts/prepare_moore_lewis_selection.py --stage build-lms --kenlm-memory 4G
python scripts/prepare_moore_lewis_selection.py --stage score
python scripts/prepare_moore_lewis_selection.py --stage sort-select --sort-memory 4G --target-source-words 348000000
python scripts/prepare_moore_lewis_selection.py --stage wordlevel-selected --sort-memory 4G --max-len 50 --vocab-words 30000 --target-source-words 348000000
```

## 训练与评测

5 epoch 主实验：

```bash
bash scripts/run_paper_except_bs240_rnnsearch50_5ep.sh
```

从 epoch 5 checkpoint 继续训练 3 个 epoch，并在每个 epoch 后计算 BLEU：

```bash
bash scripts/run_continue_3x1ep_with_bleu.sh
```

单独计算 BLEU：

```bash
python evaluate_bleu.py \
  --checkpoint checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch8.pt \
  --data-path data/wmt14_enfr/paper_strict/wordlevel/newstest2014.en-fr.tok.tsv \
  --beam-size 10 \
  --max-len 100 \
  --bleu-method sacrebleu \
  --sacrebleu-tokenize none \
  --predictions outputs/strict/rnnsearch50_epoch8_newstest2014_beam10.tsv \
  --metrics outputs/strict/rnnsearch50_epoch8_newstest2014_beam10_bleu.json \
  --device cuda
```

## Attention 机制公式

```text
e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij h_j
```

其中 `s_{i-1}` 是 decoder 上一时刻 hidden state，`h_j` 是 encoder 对第 `j` 个源词的 annotation，`alpha_ij` 是 soft alignment 权重，`c_i` 是 decoder 当前步使用的上下文向量。

## 后续计划

- 继续从 epoch 8 checkpoint 训练，尝试达到或超过论文 RNNsearch-50? 的 28.45 BLEU。
- 实现无 attention 的 RNNencdec baseline，补齐论文 Table 1 对照实验。
- 准备 RNNsearch-30 / RNNencdec-30 的 max length 30 数据和训练脚本。
- 计算按句长分桶 BLEU，复现论文 Figure 2 的长句鲁棒性分析。
- 分析 `<unk>` 输出，尝试 UNK replacement 或 copy-style 后处理。
