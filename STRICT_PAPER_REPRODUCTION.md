# Strict Bahdanau RNNsearch Reproduction

This document pins the data and preprocessing settings for a paper-style
reproduction of Bahdanau et al., "Neural Machine Translation by Jointly Learning
to Align and Translate".

## Target Setting

- Task: WMT14 English-to-French translation.
- Training pool: WMT14 EN-FR parallel corpora:
  - Europarl v7
  - News Commentary v9
  - UN corpus
  - Common Crawl
  - 10^9 French-English corpus / Giga-Fren
- Paper data selection: reduce the combined WMT14 pool to about 348M words with
  the Axelrod et al. cross-entropy data selection procedure; following Cho et al.
  (2014), use `newstest2012` and `newstest2013` as the in-domain development
  material for data selection/tuning.
- Validation: concatenate `newstest2012` and `newstest2013`.
- Test: `newstest2014`, 3003 sentences.
- Preprocessing:
  - Moses tokenization.
  - No lowercasing.
  - No stemming.
  - Word-level vocabulary.
  - 30,000 most frequent source words and 30,000 most frequent target words.
  - Out-of-shortlist words map to `<unk>`.
  - RNNsearch-50 training keeps sentence pairs with source and target length <= 50.
  - RNNsearch-30 training keeps sentence pairs with source and target length <= 30.
  - Dev/test are not length-filtered for final BLEU.

The original LIUM selected-corpus URL in the paper currently returns 404, so the
strict path is to reconstruct the selection process from the official WMT14
parallel data and record that the selected file is a regenerated equivalent, not
the authors' original binary/file dump.

## Data Preparation Commands

Download or resume the official WMT14 archives:

```bash
python scripts/download_wmt14_enfr.py \
  --profile paper-full \
  --output-dir data/wmt14_enfr \
  --yes \
  --extract \
  --print-postprocess
```

Download Moses tokenization scripts:

```bash
python scripts/download_moses_tokenizer.py
```

Prepare word-level paper-format files after all archives are extracted:

```bash
python scripts/prepare_wmt14_paper_wordlevel.py \
  --extracted-dir data/wmt14_enfr/extracted \
  --work-dir data/wmt14_enfr/paper_wordlevel \
  --moses-tokenizer-dir tools/mosesdecoder/scripts/tokenizer \
  --max-len 50 \
  --vocab-words 30000 \
  --threads 8
```

The resulting key files are:

- `data/wmt14_enfr/paper_wordlevel/train.en-fr.tok.max50.tsv`
- `data/wmt14_enfr/paper_wordlevel/valid.newstest2012_2013.en-fr.tok.tsv`
- `data/wmt14_enfr/paper_wordlevel/newstest2014.en-fr.tok.tsv`
- `data/wmt14_enfr/paper_wordlevel/vocab.en.top30000.txt`
- `data/wmt14_enfr/paper_wordlevel/vocab.fr.top30000.txt`

For the final paper-level run, replace the training TSV with the 348M-word
Axelrod-selected TSV once the selection step is regenerated.

## Strict RNNsearch-50 Command

Use `--max-vocab-size 30004` because the paper shortlist is 30,000 lexical words
per language, and the implementation adds four special symbols:
`<pad>`, `<sos>`, `<eos>`, `<unk>`.

```bash
python train.py \
  --preset paper \
  --data-path data/wmt14_enfr/paper_wordlevel/train.en-fr.tok.max50.tsv \
  --valid-data-path data/wmt14_enfr/paper_wordlevel/valid.newstest2012_2013.en-fr.tok.tsv \
  --test-data-path data/wmt14_enfr/paper_wordlevel/newstest2014.en-fr.tok.tsv \
  --source-col 0 \
  --target-col 1 \
  --limit all \
  --max-source-len 50 \
  --max-target-len 50 \
  --eval-max-source-len none \
  --eval-max-target-len none \
  --tokenizer whitespace \
  --subword-type none \
  --max-vocab-size 30004 \
  --batch-size 80 \
  --embedding-dim 620 \
  --encoder-hidden-dim 1000 \
  --decoder-hidden-dim 1000 \
  --optimizer adadelta \
  --lr 1.0 \
  --adadelta-rho 0.95 \
  --adadelta-eps 1e-6 \
  --clip 1.0 \
  --teacher-forcing-ratio 1.0 \
  --sort-k-batches 20 \
  --readout maxout \
  --device cuda \
  --amp \
  --checkpoint checkpoints/rnnsearch50_wmt14_wordlevel_strict.pt \
  --metadata checkpoints/rnnsearch50_wmt14_wordlevel_strict_metadata.json \
  --train-log checkpoints/rnnsearch50_wmt14_wordlevel_strict_log.jsonl \
  --wandb \
  --wandb-project bahdanau-attention-nmt \
  --wandb-run-name rnnsearch50-wmt14-wordlevel-strict
```

## BLEU Evaluation

The paper table reports BLEU on all `newstest2014` sentences and also on the
subset with no unknown words. The all-sentence run:

```bash
python evaluate_bleu.py \
  --checkpoint checkpoints/rnnsearch50_wmt14_wordlevel_strict.pt \
  --data-path data/wmt14_enfr/paper_wordlevel/newstest2014.en-fr.tok.tsv \
  --beam-size 10 \
  --max-len 100 \
  --bleu-method sacrebleu \
  --sacrebleu-tokenize none \
  --predictions outputs/rnnsearch50_wmt14_wordlevel_strict_newstest2014.tsv \
  --metrics outputs/rnnsearch50_wmt14_wordlevel_strict_newstest2014_bleu.json \
  --device cuda
```

## Remaining Gap Before Claiming "Strict"

The blocker is the 348M-word data-selected corpus. Until the Axelrod selection
step is rebuilt, runs on News Commentary, News+Europarl, or the raw full WMT14
pool are useful ablations, but not a strict reproduction of the paper result.
