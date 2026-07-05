#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

BASE_DIR="/root/autodl-tmp/阶段3/bahdanau_attention_nmt"
cd "$BASE_DIR"

mkdir -p checkpoints/strict logs/strict outputs/strict

previous_checkpoint="checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt"

for epoch in 6 7 8; do
  checkpoint="checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch${epoch}.pt"
  metadata="checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch${epoch}_metadata.json"
  train_log="checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch${epoch}_log.jsonl"
  train_console="logs/strict/rnnsearch50_paper_except_bs240_epoch${epoch}_train.out"
  predictions="outputs/strict/rnnsearch50_paper_except_bs240_epoch${epoch}_newstest2014_beam10.tsv"
  metrics="outputs/strict/rnnsearch50_paper_except_bs240_epoch${epoch}_newstest2014_beam10_bleu.json"
  bleu_console="logs/strict/rnnsearch50_paper_except_bs240_epoch${epoch}_bleu_beam10.out"

  echo "===== Continue training epoch ${epoch} from ${previous_checkpoint} ====="
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
    --epochs 1 \
    --device cuda \
    --resume-checkpoint "$previous_checkpoint" \
    --save-last-checkpoint \
    --checkpoint "$checkpoint" \
    --metadata "$metadata" \
    --train-log "$train_log" \
    --wandb \
    --wandb-project bahdanau-attention-nmt \
    --wandb-run-name "rnnsearch50-wmt14-paper-except-bs240-continue-epoch${epoch}-20260705" \
    --wandb-watch none \
    > "$train_console" 2>&1

  echo "===== BLEU evaluation epoch ${epoch} ====="
  python evaluate_bleu.py \
    --checkpoint "$checkpoint" \
    --data-path data/wmt14_enfr/paper_strict/wordlevel/newstest2014.en-fr.tok.tsv \
    --beam-size 10 \
    --max-len 100 \
    --bleu-method sacrebleu \
    --sacrebleu-tokenize none \
    --predictions "$predictions" \
    --metrics "$metrics" \
    --device cuda \
    > "$bleu_console" 2>&1

  previous_checkpoint="$checkpoint"
done
