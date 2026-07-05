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

Prepare the regenerated 348M-source-word selected corpus and word-level
paper-format files after all archives are extracted:

```bash
python scripts/prepare_moore_lewis_selection.py --stage devtest
python scripts/prepare_moore_lewis_selection.py --stage sample-general \
  --general-sample-lines 100000
python scripts/prepare_moore_lewis_selection.py --stage build-lms \
  --kenlm-memory 4G
python scripts/prepare_moore_lewis_selection.py --stage score
python scripts/prepare_moore_lewis_selection.py --stage sort-select \
  --sort-memory 4G \
  --target-source-words 348000000
python scripts/prepare_moore_lewis_selection.py --stage wordlevel-selected \
  --sort-memory 4G \
  --max-len 50 \
  --vocab-words 30000 \
  --target-source-words 348000000
```

The resulting key files are:

- `data/wmt14_enfr/paper_strict/selection/selected/selected_ids.tsv`
- `data/wmt14_enfr/paper_strict/wordlevel/train.en-fr.tok.selected.max50.tsv`
- `data/wmt14_enfr/paper_strict/wordlevel/valid.newstest2012_2013.en-fr.tok.tsv`
- `data/wmt14_enfr/paper_strict/wordlevel/newstest2014.en-fr.tok.tsv`
- `data/wmt14_enfr/paper_strict/wordlevel/vocab.en.top30000.txt`
- `data/wmt14_enfr/paper_strict/wordlevel/vocab.fr.top30000.txt`
- `data/wmt14_enfr/paper_strict/wordlevel/manifest.json`

Local regenerated-selection statistics:

- Selected before RNNsearch-50 filtering: 13,538,554 sentence pairs and
  348,000,013 English/source words.
- Kept after max length 50 on both sides: 11,992,626 sentence pairs,
  257,613,746 source words, and 295,415,338 target words.
- Validation: 6,003 sentence pairs from `newstest2012` + `newstest2013`.
- Test: 3,003 `newstest2014` sentence pairs.
- Vocabulary: 30,000 unique source words and 30,000 unique target words, built
  with a streaming counter that matches `train.py --tokenizer whitespace`.

## Strict RNNsearch-50 Command

Use `--max-vocab-size 30004` because the paper shortlist is 30,000 lexical words
per language, and the implementation adds four special symbols:
`<pad>`, `<sos>`, `<eos>`, `<unk>`.

```bash
python train.py \
  --preset paper \
  --data-path data/wmt14_enfr/paper_strict/wordlevel/train.en-fr.tok.selected.max50.tsv \
  --valid-data-path data/wmt14_enfr/paper_strict/wordlevel/valid.newstest2012_2013.en-fr.tok.tsv \
  --test-data-path data/wmt14_enfr/paper_strict/wordlevel/newstest2014.en-fr.tok.tsv \
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
  --source-vocab-path data/wmt14_enfr/paper_strict/wordlevel/vocab.en.top30000.txt \
  --target-vocab-path data/wmt14_enfr/paper_strict/wordlevel/vocab.fr.top30000.txt \
  --indexed-train-data \
  --train-index-prefix data/wmt14_enfr/paper_strict/wordlevel/train.en-fr.tok.selected.max50.tsv.full_index \
  --indexed-shuffle-buffer-size 1000000 \
  --indexed-shuffle-once \
  --num-workers 0 \
  --eval-batch-size 80 \
  --batch-size 80 \
  --embedding-dim 620 \
  --encoder-hidden-dim 1000 \
  --decoder-hidden-dim 1000 \
  --dropout 0.0 \
  --optimizer adadelta \
  --lr 1.0 \
  --adadelta-rho 0.95 \
  --adadelta-eps 1e-6 \
  --clip 1.0 \
  --teacher-forcing-ratio 1.0 \
  --sort-k-batches 20 \
  --readout groundhog \
  --epochs 5 \
  --device cuda \
  --checkpoint checkpoints/strict/rnnsearch50_wmt14_wordlevel_strict.pt \
  --metadata checkpoints/strict/rnnsearch50_wmt14_wordlevel_strict_metadata.json \
  --train-log checkpoints/strict/rnnsearch50_wmt14_wordlevel_strict_log.jsonl \
  --wandb \
  --wandb-project bahdanau-attention-nmt \
  --wandb-run-name rnnsearch50-wmt14-wordlevel-strict
```

## Current Formal Run

To satisfy the current experiment request, the running job keeps the command
above except for the training batch size. Batch size is 240 instead of 80 so the
RTX 5090 is used efficiently; validation/test still use batch size 80 and AMP is
disabled.

```bash
bash scripts/run_paper_except_bs240_rnnsearch50_5ep.sh
```

- Screen session: `rnnsearch50_paper_except_bs240_5ep`
- Console log: `logs/strict/rnnsearch50_paper_except_bs240_5ep.out`
- Checkpoint:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt`
- W&B run:
  `rnnsearch50-wmt14-paper-except-bs240-fp32-5ep-20260704` / `2q7rl4kr`

## BLEU Evaluation

The paper table reports BLEU on all `newstest2014` sentences and also on the
subset with no unknown words. The all-sentence run:

```bash
python evaluate_bleu.py \
  --checkpoint checkpoints/strict/rnnsearch50_wmt14_wordlevel_strict.pt \
  --data-path data/wmt14_enfr/paper_strict/wordlevel/newstest2014.en-fr.tok.tsv \
  --beam-size 10 \
  --max-len 100 \
  --bleu-method sacrebleu \
  --sacrebleu-tokenize none \
  --predictions outputs/strict/rnnsearch50_wmt14_wordlevel_strict_newstest2014.tsv \
  --metrics outputs/strict/rnnsearch50_wmt14_wordlevel_strict_newstest2014_bleu.json \
  --device cuda
```

## Workspace Audit

Before running the expensive selection/training steps, check the local workspace:

```bash
python scripts/audit_strict_workspace.py
```

`ready_for_selection` must be `true` before starting the 348M-word
Axelrod/Moore-Lewis selection. The expected extra dependency for that step is
KenLM, specifically the `lmplz`, `build_binary`, and `query` executables. The
local workspace build puts these under `tools/kenlm/build/bin/`.

## Moore-Lewis Selection Pipeline

The strict workspace uses a staged script so expensive steps can be audited and
resumed:

```bash
python scripts/prepare_moore_lewis_selection.py --stage devtest
python scripts/prepare_moore_lewis_selection.py --stage sample-general
python scripts/prepare_moore_lewis_selection.py --stage build-lms
python scripts/prepare_moore_lewis_selection.py --stage score --corpora news_commentary
```

KenLM commands are run with explicit memory caps, defaulting to `--kenlm-memory
4G`, and write logs under `logs/strict/`.

## Remaining Gap Before Claiming Full Paper Result

The data-preparation blocker is resolved locally, and `train.py` now has an
indexed TSV training path for the 11,992,626-pair selected corpus. The remaining
gap is the full-cost RNNsearch-50 training run and BLEU evaluation. The selected
corpus is a regenerated equivalent because the authors' original selected-corpus
file is not available.
