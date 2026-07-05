# Strict Reproduction Status

This file records the local workspace state for the Bahdanau et al. WMT14
English-to-French strict reproduction.

## Workspace Layout

- Official WMT14 archives: `data/wmt14_enfr/archives/`
- Extracted official corpora: `data/wmt14_enfr/extracted/`
- Strict reproduction workspace: `data/wmt14_enfr/paper_strict/`
- Historical SentencePiece/quick-run data: `data/wmt14_enfr/legacy_prestrict/`
- Historical checkpoints: `checkpoints/legacy_prestrict/`
- Strict checkpoints: `checkpoints/strict/`
- Historical evaluation outputs: `outputs/legacy_prestrict/`
- Strict evaluation outputs: `outputs/strict/`
- Download logs: `logs/downloads/`
- Strict run logs: `logs/strict/`

## Current Readiness

- WMT14 full EN-FR source archives are downloaded.
- The EN-FR training sources needed for the paper pool are extracted:
  News Commentary v9, Europarl v7, Common Crawl, UN, and Giga-Fren.
- The official validation/test SGM files are extracted:
  `newstest2012`, `newstest2013`, and `newstest2014`.
- Moses tokenization scripts are available under
  `tools/mosesdecoder/scripts/tokenizer/`.
- KenLM has been built locally under `tools/kenlm/build/bin/`.
- Previous non-strict SentencePiece experiments are isolated in legacy folders.
- The regenerated Axelrod/Moore-Lewis-style selected corpus is complete:
  13,538,554 selected sentence pairs and 348,000,013 selected English/source
  words before the RNNsearch-50 length filter.
- The RNNsearch-50 word-level training file is complete:
  `data/wmt14_enfr/paper_strict/wordlevel/train.en-fr.tok.selected.max50.tsv`
  with 11,992,626 sentence pairs, 257,613,746 source words, and 295,415,338
  target words after max-length filtering.
- The strict validation/test files and 30,000-word source/target vocabularies
  are complete under `data/wmt14_enfr/paper_strict/wordlevel/`.
- `train.py` has an indexed TSV training path with prebuilt vocabulary support;
  a 512-example CPU smoke test completed successfully with source and target
  vocabulary sizes of 30,004 each.

## Latest Completed Run

The GroundHog-aligned RNNsearch-50 run with paper-aligned settings except for
training batch size completed successfully:

- Console log: `logs/strict/rnnsearch50_paper_except_bs240_5ep.out`
- Training script: `scripts/run_paper_except_bs240_rnnsearch50_5ep.sh`
- Checkpoint:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt`
- Metadata:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_metadata.json`
- Train log:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_log.jsonl`
- W&B run:
  `rnnsearch50-wmt14-paper-except-bs240-fp32-5ep-20260704` / `2q7rl4kr`
- Intentional deviation: training batch size is 240 instead of the paper's 80.
  AMP is disabled and the run is fp32.
- Final epoch 5 metrics:
  train loss 1.9822869533512253, train ppl 7.259325755286307, valid loss
  2.199510244131915, valid ppl 9.020594528310076.
- Test metrics from the best checkpoint:
  test loss 1.87099, test ppl 6.49471.

## Next Strict Step

BLEU evaluation on all `newstest2014` sentences is complete:

- Prediction file:
  `outputs/strict/rnnsearch50_paper_except_bs240_5ep_newstest2014_beam10.tsv`
- Metrics file:
  `outputs/strict/rnnsearch50_paper_except_bs240_5ep_newstest2014_beam10_bleu.json`
- BLEU: 27.685561042648768 with beam size 10 and `sacrebleu:none`.
- Paper comparison on all sentences:
  RNNsearch-50 is 26.75 BLEU and RNNsearch-50? is 28.45 BLEU.

Next optional step: compute the paper's no-unknown-word subset BLEU and/or
continue from the 5-epoch checkpoint toward a batch-240 update-count-matched
run.

## Active Follow-Up Plan

The current hypothesis is that the remaining gap to RNNsearch-50? is mainly
caused by using batch size 240 for only 5 epochs, which gives far fewer
parameter updates than the paper's longer run. The follow-up experiment will
continue from the 5-epoch checkpoint for epochs 6, 7, and 8. Each epoch is
trained as a separate one-epoch continuation, followed immediately by beam-10
BLEU evaluation on all `newstest2014` examples.

Progress:

- Epoch 6 completed. Valid loss improved to 2.1737956961039306 and beam-10
  BLEU improved to 27.95232258637054.
- Epoch 7 completed. Valid loss improved to 2.155866563857671 and beam-10
  BLEU improved to 28.11771641998558.
- Epoch 8 completed. Valid loss improved to 2.14223373809261 and beam-10
  BLEU improved to 28.2377065392939.

Continuation summary:

- Epoch 5 BLEU: 27.685561042648768.
- Epoch 6 BLEU: 27.95232258637054.
- Epoch 7 BLEU: 28.11771641998558.
- Epoch 8 BLEU: 28.2377065392939.
- Paper RNNsearch-50? all-sentence BLEU: 28.45.

The three-epoch continuation reduced the gap to RNNsearch-50? from about 0.76
BLEU to about 0.21 BLEU.

No-UNK subset BLEU was also computed from the existing predictions. The subset
contains 737 / 3,003 `newstest2014` sentences whose source and reference tokens
are all in the 30,000-word shortlists:

- Epoch 5: all BLEU 27.685561042648768, no-UNK BLEU 35.203166199557586.
- Epoch 6: all BLEU 27.95232258637054, no-UNK BLEU 35.35264119481368.
- Epoch 7: all BLEU 28.11771641998558, no-UNK BLEU 35.875656330327836.
- Epoch 8: all BLEU 28.2377065392939, no-UNK BLEU 35.79130254339472.

Paper comparison:

- RNNsearch-50: all BLEU 26.75, no-UNK BLEU 34.16.
- RNNsearch-50?: all BLEU 28.45, no-UNK BLEU 36.15.
