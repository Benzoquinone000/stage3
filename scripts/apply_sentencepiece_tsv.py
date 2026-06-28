from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a SentencePiece model to TSV bitext.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-col", type=int, default=0)
    parser.add_argument("--target-col", type=int, default=1)
    parser.add_argument("--max-source-pieces", type=int, default=None)
    parser.add_argument("--max-target-pieces", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Install sentencepiece first: `python -m pip install sentencepiece`."
        ) from exc

    processor = spm.SentencePieceProcessor(model_file=str(args.model))
    total = 0
    kept = 0
    skipped_long = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", newline="") as input_file, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        reader = csv.reader(input_file, delimiter="\t")
        writer = csv.writer(output_file, delimiter="\t")
        for row in reader:
            if len(row) <= max(args.source_col, args.target_col):
                continue
            total += 1
            source_pieces = processor.encode(row[args.source_col], out_type=str)
            target_pieces = processor.encode(row[args.target_col], out_type=str)
            if (
                args.max_source_pieces is not None
                and len(source_pieces) > args.max_source_pieces
            ):
                skipped_long += 1
                continue
            if (
                args.max_target_pieces is not None
                and len(target_pieces) > args.max_target_pieces
            ):
                skipped_long += 1
                continue
            writer.writerow([" ".join(source_pieces), " ".join(target_pieces)])
            kept += 1

    manifest = {
        "input": str(args.input),
        "model": str(args.model),
        "output": str(args.output),
        "total": total,
        "kept": kept,
        "skipped_long": skipped_long,
        "max_source_pieces": args.max_source_pieces,
        "max_target_pieces": args.max_target_pieces,
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
