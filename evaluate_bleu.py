from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch

from nmt_attention.data import (
    UNK_TOKEN,
    Vocabulary,
    detokenize_tokens,
    pad_sequences,
    read_parallel_tsv,
    split_examples,
)
from translate import load_model
from train import download_tutorial_data


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: list[list[str]],
    references: list[list[str]],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    hyp_len = 0
    ref_len = 0

    for hypothesis, reference in zip(hypotheses, references):
        hyp_len += len(hypothesis)
        ref_len += len(reference)
        for order in range(1, max_order + 1):
            hyp_ngrams = ngram_counts(hypothesis, order)
            ref_ngrams = ngram_counts(reference, order)
            overlap = hyp_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(0, len(hypothesis) - order + 1)

    precisions = [0.0] * max_order
    for idx in range(max_order):
        if smooth:
            precisions[idx] = (matches_by_order[idx] + 1.0) / (
                possible_matches_by_order[idx] + 1.0
            )
        elif possible_matches_by_order[idx] > 0:
            precisions[idx] = matches_by_order[idx] / possible_matches_by_order[idx]

    if min(precisions) <= 0 or hyp_len == 0:
        return 0.0

    geo_mean = math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )
    brevity_penalty = (
        1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    )
    return 100.0 * geo_mean * brevity_penalty


def compute_bleu(
    hypotheses: list[list[str]],
    references: list[list[str]],
    method: str,
    sacrebleu_tokenize: str,
    detokenizer: str,
) -> tuple[float, str]:
    if method == "internal":
        return corpus_bleu(hypotheses, references), "internal"

    hypothesis_text = [
        detokenize_tokens(hypothesis, tokenizer=detokenizer) for hypothesis in hypotheses
    ]
    reference_text = [
        detokenize_tokens(reference, tokenizer=detokenizer) for reference in references
    ]
    try:
        import sacrebleu
    except ImportError:
        return corpus_bleu(hypotheses, references), "internal_fallback"

    score = sacrebleu.corpus_bleu(
        hypothesis_text,
        [reference_text],
        tokenize=sacrebleu_tokenize,
    )
    return float(score.score), f"sacrebleu:{sacrebleu_tokenize}"


def corpus_stats(
    hypotheses: list[list[str]], references: list[list[str]]
) -> dict[str, float | int]:
    hyp_tokens = sum(len(hypothesis) for hypothesis in hypotheses)
    ref_tokens = sum(len(reference) for reference in references)
    hyp_unk = sum(token == UNK_TOKEN for hypothesis in hypotheses for token in hypothesis)
    count = max(1, len(hypotheses))
    return {
        "hyp_tokens": hyp_tokens,
        "ref_tokens": ref_tokens,
        "hyp_unk_tokens": hyp_unk,
        "avg_hyp_len": hyp_tokens / count,
        "avg_ref_len": ref_tokens / count,
        "length_ratio": hyp_tokens / max(1, ref_tokens),
    }


def get_columns(train_args: dict[str, object]) -> tuple[int, int]:
    if (
        train_args.get("source_col") is not None
        and train_args.get("target_col") is not None
    ):
        return int(train_args["source_col"]), int(train_args["target_col"])
    if train_args.get("direction", "fra-eng") == "fra-eng":
        return 1, 0
    return 0, 1


def load_eval_examples(
    train_args: dict[str, object],
    data_path: Path | None,
    data_dir: Path,
) -> list[tuple[list[str], list[str]]]:
    source_col, target_col = get_columns(train_args)
    if data_path is not None:
        pairs_path = data_path
    elif train_args.get("test_data_path"):
        pairs_path = Path(str(train_args["test_data_path"]))
    elif train_args.get("data_path"):
        pairs_path = Path(str(train_args["data_path"]))
    else:
        pairs_path = download_tutorial_data(data_dir)

    examples = read_parallel_tsv(
        pairs_path,
        source_col=source_col,
        target_col=target_col,
        limit=None if train_args.get("test_data_path") else train_args.get("limit"),
        max_source_len=int(train_args["max_source_len"]),
        max_target_len=int(train_args["max_target_len"]),
        tokenizer=str(train_args.get("tokenizer", "legacy")),
    )
    if train_args.get("test_data_path") or data_path is not None:
        return examples
    _, _, test_examples = split_examples(examples, seed=int(train_args["seed"]))
    return test_examples


def translate_tokens(
    model,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    source_tokens: list[str],
    device: torch.device,
    max_len: int,
    beam_size: int,
    length_penalty: float,
    suppress_unk: bool,
    min_len: int,
    no_repeat_ngram_size: int,
) -> list[str]:
    source_ids = source_vocab.encode(source_tokens)
    source, source_lengths = pad_sequences([source_ids], source_vocab.pad_idx)
    source = source.to(device)
    source_lengths = source_lengths.to(device)

    if beam_size > 1:
        generated_ids, _ = model.beam_search(
            source,
            source_lengths,
            sos_idx=target_vocab.sos_idx,
            eos_idx=target_vocab.eos_idx,
            unk_idx=target_vocab.unk_idx if suppress_unk else None,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            min_len=min_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        generated_ids, _ = model.translate(
            source,
            source_lengths,
            sos_idx=target_vocab.sos_idx,
            eos_idx=target_vocab.eos_idx,
            max_len=max_len,
        )
    return target_vocab.decode(generated_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NMT checkpoint with corpus BLEU."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bahdanau_nmt.pt")
    )
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--min-len", type=int, default=0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--suppress-unk",
        action="store_true",
        help="Prevent beam search from emitting the vocabulary <unk> token.",
    )
    parser.add_argument(
        "--bleu-method",
        choices=["sacrebleu", "internal"],
        default="sacrebleu",
    )
    parser.add_argument(
        "--sacrebleu-tokenize",
        default="none",
        help="Use none for already tokenized TSV references; use 13a for detokenized text.",
    )
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument(
        "--predictions", type=Path, default=Path("outputs/predictions.tsv")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("outputs/bleu_metrics.json")
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log BLEU metrics and prediction table to W&B.",
    )
    parser.add_argument("--wandb-project", type=str, default="bahdanau-attention-nmt")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    model, source_vocab, target_vocab = load_model(args.checkpoint, device)
    test_examples = load_eval_examples(train_args, args.data_path, args.data_dir)
    if args.limit_examples is not None:
        test_examples = test_examples[: args.limit_examples]

    hypotheses: list[list[str]] = []
    references: list[list[str]] = []
    detokenizer = (
        "sentencepiece"
        if train_args.get("subword_type") == "sentencepiece"
        else str(train_args.get("tokenizer", "legacy"))
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["source", "reference", "hypothesis"])
        for source_tokens, target_tokens in test_examples:
            hypothesis = translate_tokens(
                model,
                source_vocab,
                target_vocab,
                source_tokens,
                device,
                args.max_len,
                args.beam_size,
                args.length_penalty,
                args.suppress_unk,
                args.min_len,
                args.no_repeat_ngram_size,
            )
            hypotheses.append(hypothesis)
            references.append(target_tokens)
            writer.writerow(
                [
                    detokenize_tokens(source_tokens, tokenizer=detokenizer),
                    detokenize_tokens(target_tokens, tokenizer=detokenizer),
                    detokenize_tokens(hypothesis, tokenizer=detokenizer),
                ]
            )

    bleu, bleu_method = compute_bleu(
        hypotheses,
        references,
        method=args.bleu_method,
        sacrebleu_tokenize=args.sacrebleu_tokenize,
        detokenizer=detokenizer,
    )
    metrics = {
        "checkpoint": str(args.checkpoint),
        "test_examples": len(test_examples),
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "min_len": args.min_len,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "suppress_unk": args.suppress_unk,
        "bleu_method": bleu_method,
        "bleu": bleu,
        "predictions": str(args.predictions),
    }
    metrics.update(corpus_stats(hypotheses, references))
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed. Install it with `pip install wandb` or omit `--wandb`."
            ) from exc
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config={
                "checkpoint": str(args.checkpoint),
                "beam_size": args.beam_size,
                "length_penalty": args.length_penalty,
                "max_len": args.max_len,
                "test_examples": len(test_examples),
            },
        )
        run.log({"bleu": bleu, "test_examples": len(test_examples)})
        artifact = wandb.Artifact(
            name=f"{run.name or 'bahdanau-nmt'}-predictions", type="evaluation"
        )
        artifact.add_file(str(args.predictions))
        artifact.add_file(str(args.metrics))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
