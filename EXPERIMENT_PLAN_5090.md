# RTX 5090 Experiment Plan

This plan is for a high-compute reproduction of Bahdanau et al. RNNsearch with
Weights & Biases tracking.

## 0. Environment Check

Before training, make sure PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected: `cuda.is_available()` should be `True`, and the device name should be
an RTX 5090 or the GPU assigned by your server.

Install project dependencies:

```powershell
pip install -r requirements.txt
```

Log in to W&B once:

```powershell
wandb login
```

For offline logging:

```powershell
$env:WANDB_MODE="offline"
```

## 1. Smoke Test

Goal: verify CUDA, dataloading, W&B logging, checkpointing, and evaluation.

```powershell
python train.py --preset debug --device cuda --amp --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name smoke-debug-rtx5090 --wandb-tags smoke debug rtx5090
```

Evaluate the smoke checkpoint:

```powershell
python evaluate_bleu.py --checkpoint checkpoints/bahdanau_nmt.pt --beam-size 2 --limit-examples 50 --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name smoke-debug-bleu
```

Expected: the run completes. BLEU may be poor because this is only a pipeline
test.

## 2. Tutorial Baseline

Goal: get a stable baseline on the small English-French tutorial corpus.

```powershell
python train.py --preset tutorial --device cuda --amp --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name tutorial-baseline --wandb-tags tutorial baseline rnnsearch
```

Evaluate:

```powershell
python evaluate_bleu.py --checkpoint checkpoints/bahdanau_nmt.pt --beam-size 5 --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name tutorial-baseline-bleu
```

Export an attention example:

```powershell
python translate.py --checkpoint checkpoints/bahdanau_nmt.pt --sentence "je suis content ." --beam-size 5 --attention-out outputs/tutorial_attention.csv --plot-out outputs/tutorial_attention.png
```

## 3. WMT14 Paper-Small

Goal: move from toy data to official WMT14 resources without downloading the
entire paper-scale corpus.

Download:

```powershell
python scripts/download_wmt14_enfr.py --profile paper-small --output-dir data/wmt14_enfr --yes --extract
```

Prepare TSV:

```powershell
python scripts/prepare_wmt14_enfr.py --input-dir data/wmt14_enfr/extracted --include-path /training/ --name-contains fr-en --output data/wmt14_enfr/wmt14_enfr_small.tsv --direction en-fr --max-len 50
```

Create fixed official valid/test splits:

```powershell
python scripts/sgm_to_tsv.py --source-sgm data/wmt14_enfr/extracted/dev/newstest2013-src.en.sgm --target-sgm data/wmt14_enfr/extracted/dev/newstest2013-ref.fr.sgm --output data/wmt14_enfr/newstest2013_enfr.tsv
python scripts/sgm_to_tsv.py --source-sgm data/wmt14_enfr/extracted/test-full/newstest2014-fren-src.en.sgm --target-sgm data/wmt14_enfr/extracted/test-full/newstest2014-fren-ref.fr.sgm --output data/wmt14_enfr/newstest2014_enfr.tsv
python scripts/prepare_sentencepiece_bitext.py --input data/wmt14_enfr/wmt14_enfr_small.tsv --valid-input data/wmt14_enfr/newstest2013_enfr.tsv --test-input data/wmt14_enfr/newstest2014_enfr.tsv --output-dir data/wmt14_enfr/spm_small_official --name small_official_16k --vocab-size 16000 --max-source-pieces 100 --max-target-pieces 100
```

Train:

```powershell
python train.py --preset rtx5090 --device cuda --amp --data-path data/wmt14_enfr/spm_small_official/small_official_16k_train.tsv --valid-data-path data/wmt14_enfr/spm_small_official/small_official_16k_valid.tsv --test-data-path data/wmt14_enfr/spm_small_official/small_official_16k_test.tsv --source-col 0 --target-col 1 --tokenizer whitespace --subword-type sentencepiece --subword-model data/wmt14_enfr/spm_small_official/small_official_16k.model --checkpoint checkpoints/rnnsearch_wmt14_small.pt --metadata checkpoints/rnnsearch_wmt14_small_metadata.json --train-log checkpoints/rnnsearch_wmt14_small_log.jsonl --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name wmt14-small-rnnsearch-rtx5090 --wandb-tags wmt14 paper-small rnnsearch rtx5090 --wandb-log-artifact
```

Evaluate:

```powershell
python evaluate_bleu.py --checkpoint checkpoints/rnnsearch_wmt14_small.pt --data-path data/wmt14_enfr/spm_small_official/small_official_16k_test.tsv --beam-size 5 --length-penalty 1.0 --suppress-unk --no-repeat-ngram-size 2 --predictions outputs/wmt14_small_predictions.tsv --metrics outputs/wmt14_small_bleu.json --bleu-method sacrebleu --sacrebleu-tokenize 13a --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name wmt14-small-bleu
```

## 4. WMT14 Full

Goal: closest available version of the paper-data setup. This requires several
GB of archives, much larger extracted data, and long training.

Download:

```powershell
python scripts/download_wmt14_enfr.py --profile paper-full --output-dir data/wmt14_enfr --yes --extract
```

Prepare TSV:

```powershell
python scripts/prepare_wmt14_enfr.py --input-dir data/wmt14_enfr/extracted --include-path /training/ --name-contains fr-en --output data/wmt14_enfr/wmt14_enfr_full.tsv --direction en-fr --max-len 50
```

Train:

```powershell
python train.py --preset rtx5090 --device cuda --amp --data-path data/wmt14_enfr/wmt14_enfr_full.tsv --source-col 0 --target-col 1 --checkpoint checkpoints/rnnsearch_wmt14_full.pt --metadata checkpoints/rnnsearch_wmt14_full_metadata.json --train-log checkpoints/rnnsearch_wmt14_full_log.jsonl --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name wmt14-full-rnnsearch-rtx5090 --wandb-tags wmt14 paper-full rnnsearch rtx5090 --wandb-log-artifact
```

Evaluate:

```powershell
python evaluate_bleu.py --checkpoint checkpoints/rnnsearch_wmt14_full.pt --beam-size 5 --predictions outputs/wmt14_full_predictions.tsv --metrics outputs/wmt14_full_bleu.json --wandb --wandb-project bahdanau-attention-nmt --wandb-run-name wmt14-full-bleu
```

## Metrics To Track

W&B logs:

- `loss/train`, `loss/valid`, `loss/test`
- `ppl/train`, `ppl/valid`, `ppl/test`
- `time/epoch_seconds`
- `best/valid_loss`
- `bleu`
- checkpoint artifacts when `--wandb-log-artifact` is enabled

For strict FP32 comparison with the original paper recipe, omit `--amp`. For
throughput-oriented RTX 5090 runs, keep `--amp`.

Record in the final report:

- dataset profile and size
- max sequence length
- vocabulary sizes
- hidden dimension and embedding dimension
- optimizer and batch size
- best validation loss
- test loss and BLEU
- beam size
- one attention heatmap
