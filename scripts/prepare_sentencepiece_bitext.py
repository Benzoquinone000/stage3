from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def read_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            source = " ".join(row[0].strip().split())
            target = " ".join(row[1].strip().split())
            if source and target:
                pairs.append((source, target))
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    valid_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    test = shuffled[:test_size]
    valid = shuffled[test_size : test_size + valid_size]
    train = shuffled[test_size + valid_size :]
    return train, valid, test


def write_raw_training_text(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for source, target in pairs:
            file.write(source + "\n")
            file.write(target + "\n")


def train_sentencepiece(
    raw_text: Path,
    model_prefix: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
) -> Path:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Install sentencepiece first: `python -m pip install sentencepiece`."
        ) from exc

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(raw_text),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        input_sentence_size=0,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


def encode_pairs(
    pairs: list[tuple[str, str]],
    model_path: Path,
    output_path: Path,
    max_source_pieces: int | None,
    max_target_pieces: int | None,
) -> dict[str, int | str | None]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_long = 0
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        for source, target in pairs:
            source_pieces = processor.encode(source, out_type=str)
            target_pieces = processor.encode(target, out_type=str)
            if max_source_pieces is not None and len(source_pieces) > max_source_pieces:
                skipped_long += 1
                continue
            if max_target_pieces is not None and len(target_pieces) > max_target_pieces:
                skipped_long += 1
                continue
            writer.writerow([" ".join(source_pieces), " ".join(target_pieces)])
            kept += 1
    return {
        "output": str(output_path),
        "kept": kept,
        "skipped_long": skipped_long,
        "max_source_pieces": max_source_pieces,
        "max_target_pieces": max_target_pieces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create shared SentencePiece/BPE train/valid/test TSV files."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--valid-input",
        type=Path,
        default=None,
        help="Optional fixed validation TSV. When set, --input is used entirely for training.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=None,
        help="Optional fixed test TSV. Must be set together with --valid-input.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", type=str, default="spm_bpe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--max-source-pieces", type=int, default=80)
    parser.add_argument("--max-target-pieces", type=int, default=80)
    args = parser.parse_args()

    pairs = read_pairs(args.input, limit=args.limit)
    if (args.valid_input is None) != (args.test_input is None):
        raise ValueError("Set both --valid-input and --test-input, or omit both.")

    if args.valid_input is not None and args.test_input is not None:
        train = pairs
        valid = read_pairs(args.valid_input)
        test = read_pairs(args.test_input)
        split_mode = "fixed"
    else:
        if len(pairs) <= args.valid_size + args.test_size:
            raise ValueError("Not enough sentence pairs for the requested split sizes.")
        train, valid, test = split_pairs(
            pairs, valid_size=args.valid_size, test_size=args.test_size, seed=args.seed
        )
        split_mode = "random"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = args.output_dir / f"{args.name}_train_raw.txt"
    model_prefix = args.output_dir / args.name
    write_raw_training_text(raw_text, train)
    model_path = train_sentencepiece(
        raw_text=raw_text,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )

    split_outputs = {
        "train": encode_pairs(
            train,
            model_path,
            args.output_dir / f"{args.name}_train.tsv",
            args.max_source_pieces,
            args.max_target_pieces,
        ),
        "valid": encode_pairs(
            valid,
            model_path,
            args.output_dir / f"{args.name}_valid.tsv",
            args.max_source_pieces,
            args.max_target_pieces,
        ),
        "test": encode_pairs(
            test,
            model_path,
            args.output_dir / f"{args.name}_test.tsv",
            args.max_source_pieces,
            args.max_target_pieces,
        ),
    }

    manifest = {
        "input": str(args.input),
        "valid_input": str(args.valid_input) if args.valid_input else None,
        "test_input": str(args.test_input) if args.test_input else None,
        "limit": args.limit,
        "seed": args.seed,
        "split_mode": split_mode,
        "subword_type": "sentencepiece",
        "model_type": args.model_type,
        "vocab_size": args.vocab_size,
        "model": str(model_path),
        "raw_training_text": str(raw_text),
        "original_pairs": len(pairs),
        "split_sizes_before_piece_filter": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "outputs": split_outputs,
    }
    manifest_path = args.output_dir / f"{args.name}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
