# Strict Experiment Log: Bahdanau RNNsearch

## 2026-07-04

- Goal: reproduce "Neural Machine Translation by Jointly Learning to Align and
  Translate" on WMT14 English-to-French as strictly as possible.
- Strict data basis: official WMT14 EN-FR training pool, official
  `newstest2012` + `newstest2013` validation material, official
  `newstest2014` test set.
- Current blocker for paper-level BLEU: reconstruct the approximately
  348M-word Axelrod/Moore-Lewis selected corpus, because the original selected
  LIUM corpus URL is no longer available.
- Workspace audit passed: official archives, extracted EN-FR corpora, Moses
  tokenizer, sacrebleu, and local KenLM are available.
- Memory safety decision: every KenLM build must use explicit `-S` memory limits
  and a controlled temp directory under `data/wmt14_enfr/paper_strict/tmp`.

### Planned Strict Pipeline

1. Tokenize official dev/test sets with Moses.
2. Build in-domain language models from `newstest2012` + `newstest2013`.
3. Build general-domain language models from the official WMT14 candidate pool.
4. Score candidate sentence pairs with bilingual Moore-Lewis
   cross-entropy difference.
5. Select the best-ranked sentence pairs until the paper target of about
   348M English/source words is reached.
6. Build 30,000-word source and target vocabularies, length-filter RNNsearch-50
   training examples, and train the GroundHog-aligned PyTorch RNNsearch model.
[2026-07-04T12:09:41Z] START stage=devtest
[2026-07-04T12:09:43Z] Prepared tokenized newstest2012/2013/2014.
[2026-07-04T12:09:43Z] END stage=devtest
[2026-07-04T12:10:25Z] START stage=sample-general
[2026-07-04T12:10:38Z] sample-general scanned 1000000 candidate pairs
[2026-07-04T12:10:50Z] sample-general scanned 2000000 candidate pairs
[2026-07-04T12:11:02Z] sample-general scanned 3000000 candidate pairs
[2026-07-04T12:11:14Z] sample-general scanned 4000000 candidate pairs
[2026-07-04T12:11:26Z] sample-general scanned 5000000 candidate pairs
[2026-07-04T12:11:39Z] sample-general scanned 6000000 candidate pairs
[2026-07-04T12:11:52Z] sample-general scanned 7000000 candidate pairs
[2026-07-04T12:12:05Z] sample-general scanned 8000000 candidate pairs
[2026-07-04T12:12:18Z] sample-general scanned 9000000 candidate pairs
[2026-07-04T12:12:31Z] sample-general scanned 10000000 candidate pairs
[2026-07-04T12:12:44Z] sample-general scanned 11000000 candidate pairs
[2026-07-04T12:12:58Z] sample-general scanned 12000000 candidate pairs
[2026-07-04T12:13:12Z] sample-general scanned 13000000 candidate pairs
[2026-07-04T12:13:26Z] sample-general scanned 14000000 candidate pairs
[2026-07-04T12:13:39Z] sample-general scanned 15000000 candidate pairs
[2026-07-04T12:13:53Z] sample-general scanned 16000000 candidate pairs
[2026-07-04T12:14:07Z] sample-general scanned 17000000 candidate pairs
[2026-07-04T12:14:21Z] sample-general scanned 18000000 candidate pairs
[2026-07-04T12:14:35Z] sample-general scanned 19000000 candidate pairs
[2026-07-04T12:14:50Z] sample-general scanned 20000000 candidate pairs
[2026-07-04T12:15:04Z] sample-general scanned 21000000 candidate pairs
[2026-07-04T12:15:20Z] sample-general scanned 22000000 candidate pairs
[2026-07-04T12:15:34Z] sample-general scanned 23000000 candidate pairs
[2026-07-04T12:15:48Z] sample-general scanned 24000000 candidate pairs
[2026-07-04T12:16:04Z] sample-general scanned 25000000 candidate pairs
[2026-07-04T12:16:21Z] sample-general scanned 26000000 candidate pairs
[2026-07-04T12:16:36Z] sample-general scanned 27000000 candidate pairs
[2026-07-04T12:16:52Z] sample-general scanned 28000000 candidate pairs
[2026-07-04T12:17:08Z] sample-general scanned 29000000 candidate pairs
[2026-07-04T12:17:24Z] sample-general scanned 30000000 candidate pairs
[2026-07-04T12:17:40Z] sample-general scanned 31000000 candidate pairs
[2026-07-04T12:17:56Z] sample-general scanned 32000000 candidate pairs
[2026-07-04T12:18:12Z] sample-general scanned 33000000 candidate pairs
[2026-07-04T12:18:29Z] sample-general scanned 34000000 candidate pairs
[2026-07-04T12:18:45Z] sample-general scanned 35000000 candidate pairs
[2026-07-04T12:19:03Z] sample-general scanned 36000000 candidate pairs
[2026-07-04T12:19:26Z] sample-general scanned 37000000 candidate pairs
[2026-07-04T12:19:44Z] sample-general scanned 38000000 candidate pairs
[2026-07-04T12:20:04Z] sample-general scanned 39000000 candidate pairs
[2026-07-04T12:20:24Z] sample-general scanned 40000000 candidate pairs
[2026-07-04T12:20:47Z] Sampled 100000 general-domain lines from 40836715 candidate pairs.
[2026-07-04T12:20:47Z] END stage=sample-general
[2026-07-04T12:21:15Z] START stage=build-lms
[2026-07-04T12:22:05Z] START stage=build-lms
[2026-07-04T12:22:23Z] Built in-domain and general-domain KenLM models.
[2026-07-04T12:22:23Z] END stage=build-lms
[2026-07-04T12:22:36Z] START stage=score
[2026-07-04T12:22:54Z] Scored news_commentary: 182761 rows.
[2026-07-04T12:22:54Z] END stage=score
[2026-07-04T12:24:06Z] START stage=score
[2026-07-04T12:24:25Z] Scored news_commentary: 182761 rows.
[2026-07-04T12:24:25Z] END stage=score
[2026-07-04T12:25:47Z] START stage=score
[2026-07-04T12:30:13Z] Scored europarl: 2002756 rows.
[2026-07-04T12:30:13Z] END stage=score
[2026-07-04T12:30:52Z] START stage=score
[2026-07-04T12:37:22Z] Scored commoncrawl: 3244152 rows.
[2026-07-04T12:37:22Z] END stage=score
[2026-07-04T12:38:23Z] START stage=score
[2026-07-04T13:07:45Z] Scored un: 12886814 rows.
[2026-07-04T13:07:45Z] END stage=score
[2026-07-04T13:08:35Z] START stage=score
[2026-07-04T14:03:42Z] Scored giga_fren: 22519560 rows.
[2026-07-04T14:03:42Z] END stage=score
[2026-07-04T14:07:01Z] START stage=sort-select
[2026-07-04T14:07:52Z] Sorted 40836043 score rows.
[2026-07-04T14:08:06Z] Selected 13538554 rows with 348000013 source words.
[2026-07-04T14:08:06Z] END stage=sort-select
[2026-07-04T14:12:39Z] START stage=wordlevel-selected
[2026-07-04T14:12:54Z] Split selected ids into 5 corpus files.
[2026-07-04T14:13:13Z] Materialized news_commentary: kept 142162 max50 rows.
[2026-07-04T14:15:14Z] Materialized europarl: kept 923521 max50 rows.
[2026-07-04T14:19:40Z] Materialized commoncrawl: kept 2195280 max50 rows.
[2026-07-04T14:23:56Z] Materialized un: kept 2001009 max50 rows.
[2026-07-04T14:40:14Z] Materialized giga_fren: kept 6730654 max50 rows.
[2026-07-04T14:43:18Z] Prepared selected wordlevel max50: 11992626 rows.
[2026-07-04T14:43:18Z] END stage=wordlevel-selected

### Completed Strict Data Preparation Summary

- Scored 40,836,043 candidate sentence pairs from News Commentary v9,
  Europarl v7, Common Crawl, UN, and Giga-Fren.
- Selected 13,538,554 sentence pairs with 348,000,013 English/source words
  before RNNsearch-50 filtering.
- Built the final RNNsearch-50 training TSV:
  `data/wmt14_enfr/paper_strict/wordlevel/train.en-fr.tok.selected.max50.tsv`.
- Final max50 training set: 11,992,626 sentence pairs, 257,613,746 source
  words, and 295,415,338 target words.
- Validation set: 6,003 pairs from `newstest2012` + `newstest2013`.
- Test set: 3,003 pairs from `newstest2014`.
- Source and target vocabularies: 30,000 unique words each. The initial
  external-sort vocabulary was rebuilt with a streaming Python counter to match
  `train.py --tokenizer whitespace` exactly.
- Resource note: KenLM and sort-heavy steps were run with explicit 4G memory
  limits; no long-running data-preparation processes remain.
- Next risk: full training must avoid loading all 11,992,626 examples into
  Python memory at once.
[2026-07-04T14:54:50Z] START stage=vocab-selected
[2026-07-04T14:57:15Z] Rebuilt selected vocabularies with streaming counter: 30000 source / 30000 target words.
[2026-07-04T14:57:15Z] END stage=vocab-selected

### Strict RNNsearch-50 Training Start

- Added an indexed TSV training path to avoid loading all 11,992,626 training
  examples into Python memory.
- Built full index sidecars:
  `train.en-fr.tok.selected.max50.tsv.full_index.offsets.bin` and
  `train.en-fr.tok.selected.max50.tsv.full_index.lengths.bin`.
- CUDA paper-size probe passed with 81,455,964 trainable parameters,
  batch size 80, AMP, and 30,004 source/target vocabulary sizes.
- Full training started in detached screen session `rnnsearch50_strict`.
- Main process command: `scripts/run_strict_rnnsearch50.sh`.
- Console log: `logs/strict/rnnsearch50_full_train.out`.
- W&B run: `rnnsearch50-wmt14-wordlevel-strict-20260704`, run id `fahkw3lc`.
- Initial runtime check: GPU memory approximately 9.9GB / 32GB and training
  entered epoch 1.

### RTX5090 High-Throughput Training Start

- The strict batch-80 run was stopped during epoch 1 before any epoch checkpoint
  was written, because GPU memory usage was only about 9.9GB / 32GB.
- Added `--eval-batch-size` so training can use a larger batch while validation
  and test stay at batch 80 for long unfiltered newstest sentences.
- Probe results:
  - batch 160: about 18.4GB GPU memory.
  - batch 240: about 27.0GB GPU memory, about 85-94% GPU utilization, about
    400W power draw.
- Full RTX5090 high-throughput run started in detached screen session
  `rnnsearch50_bs240`.
- Main process command: `scripts/run_rtx5090_rnnsearch50_bs240.sh`.
- Console log: `logs/strict/rnnsearch50_rtx5090_bs240_train.out`.
- Checkpoint target:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_rtx5090_bs240.pt`.
- W&B run: `rnnsearch50-wmt14-wordlevel-rtx5090-bs240-20260704`, run id
  `sdsipbh9`.
- Note: this is a high-throughput run and intentionally changes the paper's
  original batch size 80 to batch size 240.

### Paper-Strict-Except-Batch Training Start

- The previous high-throughput AMP run was stopped during epoch 1 before any
  epoch checkpoint was written.
- The formal run requested by the user keeps paper-aligned settings except for
  the training batch size:
  - batch size 240 instead of the paper's 80, to use the RTX 5090 memory better.
  - fp32 training, no AMP.
  - Adadelta with lr 1.0, rho 0.95, eps 1e-6.
  - L2 gradient clipping at 1.0.
  - RNNsearch-50 dimensions: embeddings 620, encoder/decoder hidden 1000.
  - GroundHog-style maxout readout.
  - dropout 0.0.
  - sort-k-batches 20 and one-time indexed corpus shuffle.
  - 5 epochs, matching the longer RNNsearch-50 table row by data passes.
- Probe results before launch:
  - fp32 batch 160 was stable at about 20.3GB GPU memory.
  - fp32 batch 240 was stable at about 29.7GB GPU memory.
- Full run started in detached screen session
  `rnnsearch50_paper_except_bs240_5ep`.
- Main process command:
  `scripts/run_paper_except_bs240_rnnsearch50_5ep.sh`.
- Console log: `logs/strict/rnnsearch50_paper_except_bs240_5ep.out`.
- Checkpoint target:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt`.
- W&B run:
  `rnnsearch50-wmt14-paper-except-bs240-fp32-5ep-20260704`, run id
  `2q7rl4kr`.
- Initial runtime check: GPU memory about 29.7GB / 32.6GB, utilization
  about 98-99%, and training entered epoch 1.

### Paper-Strict-Except-Batch Training Complete

- The 5-epoch fp32 run completed successfully. The screen session exited and
  the GPU returned to idle.
- Final epoch metrics:
  - epoch 5 time: 165m 21s.
  - train loss / ppl: 1.9822869533512253 / 7.259325755286307.
  - valid loss / ppl: 2.199510244131915 / 9.020594528310076.
  - best valid loss: 2.199510244131915.
- Test set metrics from the best checkpoint:
  - test loss / ppl: 1.87099 / 6.49471.
- Outputs:
  - checkpoint:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt`
  - metadata:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_metadata.json`
  - train log:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep_log.jsonl`
  - W&B run:
    `rnnsearch50-wmt14-paper-except-bs240-fp32-5ep-20260704` / `2q7rl4kr`.
- Remaining evaluation step: generate translations and compute BLEU on
  `newstest2014`.

### Paper-Strict-Except-Batch BLEU Evaluation Complete

- Decoding/evaluation command:
  `evaluate_bleu.py` with beam size 10, max length 100, and
  `sacrebleu` tokenization mode `none` on tokenized `newstest2014`.
- Evaluation completed on all 3,003 test examples.
- BLEU: 27.685561042648768.
- Prediction file:
  `outputs/strict/rnnsearch50_paper_except_bs240_5ep_newstest2014_beam10.tsv`
- Metrics file:
  `outputs/strict/rnnsearch50_paper_except_bs240_5ep_newstest2014_beam10_bleu.json`
- Paper comparison on all `newstest2014` sentences:
  - RNNsearch-50: 26.75 BLEU.
  - RNNsearch-50?: 28.45 BLEU.
  - This run: 27.69 BLEU.
- Interpretation: this run exceeds the paper's standard RNNsearch-50
  all-sentence BLEU, but is still below the longer RNNsearch-50? result. The
  main known deviation remains batch size 240 instead of 80, which reduces the
  number of parameter updates during 5 epochs.

### Follow-Up Hypothesis: Batch Size and Update Count

- The most likely reason this run did not reach the longer RNNsearch-50?
  28.45 BLEU is the batch-size deviation.
- With batch size 240, 5 epochs produce about 249,850 optimizer updates, while
  the paper's longer RNNsearch-50? row reports about 667,000 updates.
- The validation loss was still decreasing at epoch 5, so the model likely had
  not exhausted the benefit of more updates:
  2.6895 -> 2.3980 -> 2.2925 -> 2.2360 -> 2.1995.
- Follow-up plan: continue from the 5-epoch checkpoint for three one-epoch
  increments. After each epoch, run beam-10 BLEU on all 3,003 `newstest2014`
  examples and record the result.

### Continue Epoch 6 Complete

- Continued from:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_5ep.pt`.
- Epoch 6 train/valid:
  - train loss / ppl: 1.9359050295980218 / 6.930313356710112.
  - valid loss / ppl: 2.1737956961039306 / 8.79159099690464.
  - epoch time: 165m 23s.
  - test loss / ppl: 1.84035 / 6.29873.
- Beam-10 `newstest2014` BLEU:
  27.95232258637054.
- Output files:
  - checkpoint:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch6.pt`
  - metrics:
    `outputs/strict/rnnsearch50_paper_except_bs240_epoch6_newstest2014_beam10_bleu.json`
- Interpretation: BLEU improved from 27.6856 at epoch 5 to 27.9523 at epoch 6,
  supporting the update-count hypothesis. Epoch 7 training has started.

### Continue Epoch 7 Complete

- Continued from:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch6.pt`.
- Epoch 7 train/valid:
  - train loss / ppl: 1.9000041951556048 / 6.685922490705646.
  - valid loss / ppl: 2.155866563857671 / 8.635370037185123.
  - epoch time: 165m 18s.
  - test loss / ppl: 1.81906 / 6.16607.
- Beam-10 `newstest2014` BLEU:
  28.11771641998558.
- Output files:
  - checkpoint:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch7.pt`
  - metrics:
    `outputs/strict/rnnsearch50_paper_except_bs240_epoch7_newstest2014_beam10_bleu.json`
- Interpretation: BLEU improved again, from 27.9523 at epoch 6 to 28.1177 at
  epoch 7. This is now close to the paper's longer RNNsearch-50? all-sentence
  BLEU of 28.45. Epoch 8 training has started.

### Continue Epoch 8 Complete

- Continued from:
  `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch7.pt`.
- Epoch 8 train/valid:
  - train loss / ppl: 1.8709781635757412 / 6.494646119519343.
  - valid loss / ppl: 2.14223373809261 / 8.518444366617596.
  - epoch time: 165m 11s.
  - test loss / ppl: 1.80333 / 6.06982.
- Beam-10 `newstest2014` BLEU:
  28.2377065392939.
- Output files:
  - checkpoint:
    `checkpoints/strict/rnnsearch50_wmt14_wordlevel_paper_except_bs240_epoch8.pt`
  - metrics:
    `outputs/strict/rnnsearch50_paper_except_bs240_epoch8_newstest2014_beam10_bleu.json`
- Three-epoch continuation summary:
  - epoch 5 BLEU: 27.6856.
  - epoch 6 BLEU: 27.9523.
  - epoch 7 BLEU: 28.1177.
  - epoch 8 BLEU: 28.2377.
  - paper RNNsearch-50?: 28.45.
- Interpretation: continuing training consistently improved BLEU and reduced
  the gap to RNNsearch-50? from 0.76 to 0.21 BLEU, strongly supporting the
  update-count explanation. The remaining gap may require more updates, exact
  batch size 80, original selected corpus, or closer decoder/UNK handling.

### No-UNK Subset BLEU Summary

- Computed the paper-style no-UNK subset from existing `newstest2014`
  predictions by keeping sentences where both the source sentence and reference
  sentence contain only words in the 30,000-word shortlists.
- Subset size: 737 / 3,003 test sentences.
- Results:
  - epoch 5: all BLEU 27.6856, no-UNK BLEU 35.2032.
  - epoch 6: all BLEU 27.9523, no-UNK BLEU 35.3526.
  - epoch 7: all BLEU 28.1177, no-UNK BLEU 35.8757.
  - epoch 8: all BLEU 28.2377, no-UNK BLEU 35.7913.
- Paper comparison:
  - RNNsearch-50: all 26.75, no-UNK 34.16.
  - RNNsearch-50?: all 28.45, no-UNK 36.15.
- Interpretation: the reproduction exceeds the standard RNNsearch-50 on both
  all-sentence and no-UNK BLEU, and approaches the longer RNNsearch-50? on both
  metrics. The best no-UNK BLEU so far is epoch 7, while the best all-sentence
  BLEU so far is epoch 8.
