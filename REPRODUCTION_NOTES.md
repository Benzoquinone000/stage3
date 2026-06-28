# Reproduction Notes

## Open-Source Reference

The official/open-source reference for this paper is GroundHog:

- Repository: https://github.com/lisa-groundhog/GroundHog
- NMT code: `experiments/nmt`
- Important files copied locally for inspection:
  - `reference_groundhog_state.py`
  - `reference_groundhog_train.py`
  - `reference_groundhog_encdec.py`

The GroundHog implementation is Theano-based. This project is a PyTorch
reimplementation of the RNNsearch model rather than a line-by-line port.

## GroundHog RNNsearch-50 Settings

From `prototype_search_state()` and the shared prototype settings:

- Model: RNNsearch-50.
- Encoder: bidirectional recurrent encoder.
- Decoder: recurrent layer with search/attention.
- Hidden dimension: `dim = 1000`.
- Embedding/low-rank approximation dimension: `rank_n_approx = 620`.
- Source/target vocabulary: `30000 + EOS`.
- Maximum sequence length: `seqlen = 50`.
- Batch size: `bs = 80`.
- Length bucketing: `sort_k_batches = 20`.
- Optimizer: `SGD_adadelta`.
- Adadelta: `rho = 0.95`, `eps = 1e-6`.
- Gradient clipping cutoff: `1.0`.
- Dropout: disabled in the original state file.
- Weight initialization: recurrent matrices use orthogonal initialization;
  other matrices use small random weights.

## What This PyTorch Version Now Matches

- Additive attention score:

```text
e_ij = v_a^T tanh(W_a [s_{i-1}; h_j])
alpha_ij = softmax(e_ij)
c_i = sum_j alpha_ij h_j
```

- Bidirectional encoder annotations.
- Conditional GRU decoder where the attention context contributes to reset,
  update, and candidate-state transitions.
- Maxout readout before the vocabulary projection.
- Paper-style initialization: small random non-recurrent weights, orthogonal
  recurrent weights, zero attention scoring vector.
- Adadelta paper preset with `rho=0.95`, `eps=1e-6`, and gradient clipping.
- `sort_k_batches` length bucketing to reduce padding.
- Beam search decoding for evaluation.
- BLEU evaluation script for the held-out split.

## Remaining Gap To A True Paper-Scale Reproduction

The paper result depends on training with large English-French data comparable
to WMT14, a 30k vocabulary, 1000 hidden units, 620-dimensional embeddings, and
long training. News Commentary alone is useful for validating the mechanism and
the official evaluation pipeline, but it cannot reproduce the paper's BLEU
score.

To make the report academically honest, describe the result as:

> A PyTorch RNNsearch-style reproduction aligned with the official GroundHog
> architecture and training recipe, evaluated on a smaller public bitext due to
> compute and data constraints.

## Current Experiment Record

All runs below use English-to-French translation, official `newstest2013` as
validation, official `newstest2014` as test, shared SentencePiece/BPE
preprocessing, beam size 5, `length_penalty=1.0`, `<unk>` suppression, and
SacreBLEU `13a` tokenization unless noted.

Important preprocessing fix: News Commentary contains lone carriage returns in
some lines. `scripts/prepare_wmt14_enfr.py` now opens text with `newline="\n"`
and checks unpaired lines with `zip_longest`, preventing silent EN/FR
misalignment.

| Run | Train data | Vocab | Epochs | Valid PPL | Test PPL | BLEU | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rnnsearch_nc50k_official_spm8k_30ep` | 49,989 NC pairs | 8k BPE | 30 | 96.95 | 99.89 | 0.60 | `lyq4pj7w`, `q5x6ch1z` |
| `rnnsearch_nc_all_official_spm16k_20ep` | 173,482 NC pairs | 16k BPE | 20 | 39.41 | 38.38 | 6.15 | `vdrnejpv`, `jzyo1meh` |
| `rnnsearch_nc_all_official_spm16k_30ep` | 173,482 NC pairs | 16k BPE | 30 | 28.55 | 26.02 | 10.34 | `ifqc2csf`, `7iqesq0r` |

The best current checkpoint is:

```text
checkpoints/rnnsearch_nc_all_official_spm16k_30ep.pt
```

Its official test metrics are:

```text
outputs/nc_all_official_spm16k_30ep_newstest2014_bleu.json
```

This is a much stronger News Commentary-only baseline, but it is still not a
paper-scale reproduction. The remaining bottleneck is the amount and domain
coverage of parallel data; the next required run is News Commentary + Europarl,
then the full WMT14 profile.

## Commands

Smoke test:

```powershell
python train.py --preset debug --epochs 1 --limit 200 --batch-size 32 --embedding-dim 16 --encoder-hidden-dim 32 --decoder-hidden-dim 32 --checkpoint checkpoints/smoke_rnnsearch_bahdanau_nmt.pt --metadata checkpoints/smoke_rnnsearch_metadata.json --train-log checkpoints/smoke_rnnsearch_train_log.jsonl
```

CPU-friendly training:

```powershell
python train.py --preset tutorial
```

Paper-style configuration on custom TSV data:

```powershell
python train.py --preset paper --data-path path\to\parallel.tsv --source-col 0 --target-col 1
```

Beam-search translation:

```powershell
python translate.py --checkpoint checkpoints/bahdanau_nmt.pt --sentence "je suis content ." --beam-size 5 --attention-out outputs/attention.csv --plot-out outputs/attention.png
```

BLEU evaluation:

```powershell
python evaluate_bleu.py --checkpoint checkpoints/bahdanau_nmt.pt --beam-size 5 --predictions outputs/predictions.tsv --metrics outputs/bleu_metrics.json
```

## Training Tips From The Reference Implementation

- Use sentence-length filtering. The paper's RNNsearch-50 state uses maximum
  length 50.
- Use length bucketing. Sorting groups of 20 mini-batches reduces padding and
  speeds recurrent training.
- Use gradient clipping. The original cutoff is `1.0`.
- Use Adadelta for paper-fidelity runs. Adam is convenient for small local
  debugging, but GroundHog used Adadelta.
- Keep dropout disabled for strict comparison with the released state.
- Prefer beam search for reporting translations and BLEU.
- Evaluate on a fixed split and save predictions, not just a single loss value.

## Suggested Experiment Record

For each run, record:

- Dataset name and size.
- Train/valid/test split sizes.
- Vocabulary sizes.
- Maximum source and target length.
- Hidden dimension, embedding dimension, optimizer, batch size.
- Training loss, validation loss, test loss, BLEU.
- Decoding mode and beam size.
- At least one attention heatmap for a short sentence.
