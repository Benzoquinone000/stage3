# Bahdanau Attention NMT Reproduction

This folder is a PyTorch reproduction scaffold for:

> Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. 2014.
> Neural Machine Translation by Jointly Learning to Align and Translate.

The original open-source code is Theano/GroundHog:

- https://github.com/lisa-groundhog/GroundHog
- GroundHog NMT directory: `experiments/nmt`

This project implements the RNNsearch idea in PyTorch: bidirectional recurrent
encoder annotations, additive attention, a conditional GRU decoder, maxout
readout, gradient clipping, Adadelta paper preset, length-filtered data, and
bucketed mini-batches.

## What Is Reproduced

- Encoder annotations: a bidirectional GRU encodes every source token.
- Additive attention: every target step scores all source annotations and
  normalizes the scores into alignment weights.
- Conditional decoder: the attention context contributes to GRU reset, update,
  and candidate-state transitions.
- Maxout readout: the output projection uses a deep maxout layer.
- Paper-style training options: Adadelta, gradient clipping, no dropout,
  sequence length filtering, and `sort_k_batches` bucketing.
- Attention export: translation can save alignment weights as CSV or heatmap.

This is not automatically an exact WMT14-scale reproduction. To reproduce the
paper's reported result, you need WMT-scale English-French data, 30k source and
target vocabularies, 1000-dimensional recurrent states, 620-dimensional
embeddings, long training, and BLEU evaluation. The default tutorial data is
small and intended for mechanism verification.

Current best local baseline:

- Data: full News Commentary v9 EN-FR, official `newstest2013` validation,
  official `newstest2014` test.
- Model: shared 16k SentencePiece/BPE, 256 embeddings, 512 GRU states,
  Bahdanau/RNNsearch attention, maxout readout, Adadelta, gradient clip 1.0.
- Result: BLEU 10.34 on `newstest2014`.
- Checkpoint: `checkpoints/rnnsearch_nc_all_official_spm16k_30ep.pt`.
- Metrics: `outputs/nc_all_official_spm16k_30ep_newstest2014_bleu.json`.

## Files

- `nmt_attention/data.py`: tokenization, vocabulary, TSV bitext loader,
  bucketed batching, padding, and DataLoader collate function.
- `nmt_attention/model.py`: Encoder, BahdanauAttention, ConditionalGRUCell,
  MaxoutReadout, Decoder, Seq2Seq.
- `train.py`: downloads tutorial data or reads custom TSV, trains, validates,
  tests, logs metrics, and saves the best checkpoint.
- `evaluate_bleu.py`: evaluates a checkpoint with beam search and corpus BLEU.
- `translate.py`: translates a sentence and exports attention weights.
- `REPRODUCTION_NOTES.md`: official GroundHog settings and experiment notes.
- `EXPERIMENT_PLAN_5090.md`: staged RTX 5090 + W&B experiment plan.

## Quick Start

Run from this directory.

Small debugging run:

```powershell
python train.py --preset debug
```

CPU-friendly tutorial run:

```powershell
python train.py --preset tutorial
```

RTX 5090 + W&B monitored run:

```powershell
python train.py --preset rtx5090 --device cuda --amp --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name rnnsearch-rtx5090
```

Paper-fidelity scaffold, for a custom parallel TSV:

```powershell
python train.py --preset paper --data-path path\to\parallel.tsv --source-col 0 --target-col 1
```

Warm-start from weights only:

```powershell
python train.py --init-checkpoint checkpoints/rnnsearch_nc_all_official_spm16k_30ep.pt ...
```

Resume model and optimizer state when available:

```powershell
python train.py --resume-checkpoint checkpoints/rnnsearch_wmt14_full.pt ...
```

## Download WMT14 Data

The downloader is conservative by default: it prints the plan unless `--yes` is
provided.

Only dev/test sets:

```powershell
python scripts/download_wmt14_enfr.py --profile devtest --output-dir data/wmt14_enfr --yes --extract
```

Small training profile with News Commentary plus dev/test:

```powershell
python scripts/download_wmt14_enfr.py --profile paper-small --output-dir data/wmt14_enfr --yes --extract
```

Full paper-data profile. This downloads several GB of archives:

```powershell
python scripts/download_wmt14_enfr.py --profile paper-full --output-dir data/wmt14_enfr --yes --extract
```

Prepare extracted `.en/.fr` files as TSV:

```powershell
python scripts/prepare_wmt14_enfr.py --input-dir data/wmt14_enfr/extracted --output data/wmt14_enfr/wmt14_enfr.tsv --direction en-fr --max-len 50
```

Translate after training:

```powershell
python translate.py --sentence "je suis content ." --attention-out outputs/attention.csv --plot-out outputs/attention.png
```

## Attention Formula

```text
e_ij = v_a^T tanh(W_a [s_{i-1}; h_j])
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij h_j
```

`s_{i-1}` is the previous decoder hidden state, `h_j` is the encoder annotation
for source token `j`, `alpha_ij` is the soft alignment weight, and `c_i` is the
context vector used during target step `i`.

## Suggested Experiments

- Compare `--preset tutorial` and `--preset paper` on the same small corpus.
- Compare optimizers: `--optimizer adam` vs `--optimizer adadelta`.
- Compare `--sort-k-batches 1`, `10`, and `20`.
- Visualize attention for short sentences and include the heatmap in the report.
