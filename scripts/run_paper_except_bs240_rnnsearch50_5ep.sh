#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

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
  --num-workers 2 \
  --batch-size 240 \
  --eval-batch-size 80 \
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
  --checkpoint checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt \
  --metadata checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_metadata.json \
  --train-log checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_log.jsonl \
  --wandb \
  --wandb-project bahdanau-attention-nmt \
  --wandb-run-name rnnsearch50-wmt14-paper-except-bs240-fp32-5ep-20260704 \
  --wandb-watch none
