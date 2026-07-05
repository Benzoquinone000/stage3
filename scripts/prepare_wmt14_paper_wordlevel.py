from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
import shlex
import subprocess
from collections import Counter
from itertools import zip_longest
from pathlib import Path


SEGMENT_PATTERN = re.compile(r"<seg[^>]*>(.*?)</seg>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def read_sgm_segments(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    segments: list[str] = []
    for match in SEGMENT_PATTERN.finditer(text):
        segment = TAG_PATTERN.sub("", match.group(1))
        segment = html.unescape(segment)
        segment = normalize_space(segment)
        if segment:
            segments.append(segment)
    return segments


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n")
    return path.open("r", encoding="utf-8", errors="replace", newline="\n")


def moses_script_dir(path: Path) -> Path:
    required = (
        "normalize-punctuation.perl",
        "tokenizer.perl",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Moses script(s) under {path}: {', '.join(missing)}. "
            "Run scripts/download_moses_tokenizer.py first."
        )
    return path


def tokenize_file(
    input_path: Path,
    output_path: Path,
    lang: str,
    scripts_dir: Path,
    threads: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] tokenized {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix == ".gz":
        input_command = f"gzip -cd {shlex.quote(str(input_path))}"
    else:
        input_command = f"cat {shlex.quote(str(input_path))}"

    command = (
        f"{input_command} "
        f"| perl {shlex.quote(str(scripts_dir / 'normalize-punctuation.perl'))} -l {shlex.quote(lang)} "
    )
    remove_script = scripts_dir / "remove-non-printing-char.perl"
    if remove_script.exists():
        command += f"| perl {shlex.quote(str(remove_script))} "
    command += (
        f"| perl {shlex.quote(str(scripts_dir / 'tokenizer.perl'))} "
        f"-l {shlex.quote(lang)} -threads {int(threads)} -no-escape "
        f"> {shlex.quote(str(output_path))}"
    )
    print(f"[tokenize] {input_path} -> {output_path}")
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for line in lines:
            file.write(f"{line}\n")


def pair_for_en_file(en_path: Path) -> Path:
    if en_path.name.endswith(".en.gz"):
        return en_path.with_name(en_path.name[: -len(".en.gz")] + ".fr.gz")
    return en_path.with_suffix(".fr")


def is_training_path(path: Path) -> bool:
    parts = set(path.parts)
    if {"dev", "test", "test-full"} & parts:
        return False
    return True


def find_training_pairs(extracted_dir: Path) -> list[tuple[Path, Path]]:
    english_files = list(extracted_dir.rglob("*.en")) + list(
        extracted_dir.rglob("*.en.gz")
    )
    pairs: list[tuple[Path, Path]] = []
    for en_path in sorted(set(english_files)):
        if not is_training_path(en_path):
            continue
        fr_path = pair_for_en_file(en_path)
        if fr_path.exists():
            pairs.append((en_path, fr_path))
    return pairs


def stable_stem(path: Path) -> str:
    name = path.name
    for suffix in (".en.gz", ".fr.gz", ".en", ".fr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parent = path.parent.name
    return f"{parent}_{name}" if parent not in {"extracted", "training"} else name


def tokenized_training_pairs(
    pairs: list[tuple[Path, Path]],
    tokenized_dir: Path,
    scripts_dir: Path,
    threads: int,
    overwrite: bool,
) -> list[tuple[Path, Path, Path, Path]]:
    tokenized: list[tuple[Path, Path, Path, Path]] = []
    for en_path, fr_path in pairs:
        stem = stable_stem(en_path)
        tok_en = tokenized_dir / f"{stem}.tok.en"
        tok_fr = tokenized_dir / f"{stem}.tok.fr"
        tokenize_file(en_path, tok_en, "en", scripts_dir, threads, overwrite)
        tokenize_file(fr_path, tok_fr, "fr", scripts_dir, threads, overwrite)
        tokenized.append((tok_en, tok_fr, en_path, fr_path))
    return tokenized


def write_train_tsv(
    tokenized_pairs: list[tuple[Path, Path, Path, Path]],
    output_path: Path,
    max_len: int,
    max_pairs: int | None,
) -> dict[str, object]:
    total = 0
    kept = 0
    skipped_empty = 0
    skipped_long = 0
    skipped_unpaired = 0
    source_words = 0
    target_words = 0
    file_records: list[dict[str, object]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for tok_en, tok_fr, raw_en, raw_fr in tokenized_pairs:
            file_total = 0
            file_kept = 0
            with open_text(tok_en) as en_file, open_text(tok_fr) as fr_file:
                for en_line, fr_line in zip_longest(en_file, fr_file):
                    total += 1
                    file_total += 1
                    if en_line is None or fr_line is None:
                        skipped_unpaired += 1
                        continue
                    en = normalize_space(en_line)
                    fr = normalize_space(fr_line)
                    if not en or not fr:
                        skipped_empty += 1
                        continue
                    en_len = len(en.split())
                    fr_len = len(fr.split())
                    if en_len > max_len or fr_len > max_len:
                        skipped_long += 1
                        continue
                    writer.writerow([en, fr])
                    kept += 1
                    file_kept += 1
                    source_words += en_len
                    target_words += fr_len
                    if max_pairs is not None and kept >= max_pairs:
                        break
            file_records.append(
                {
                    "raw_en": str(raw_en),
                    "raw_fr": str(raw_fr),
                    "tokenized_en": str(tok_en),
                    "tokenized_fr": str(tok_fr),
                    "total_seen": file_total,
                    "kept": file_kept,
                }
            )
            if max_pairs is not None and kept >= max_pairs:
                break

    return {
        "path": str(output_path),
        "max_len": max_len,
        "max_pairs": max_pairs,
        "total_seen": total,
        "kept": kept,
        "skipped_empty": skipped_empty,
        "skipped_long": skipped_long,
        "skipped_unpaired": skipped_unpaired,
        "source_words": source_words,
        "target_words": target_words,
        "files": file_records,
    }


def write_sgm_pair_raw(source_sgm: Path, target_sgm: Path, raw_prefix: Path) -> tuple[Path, Path, int]:
    source_segments = read_sgm_segments(source_sgm)
    target_segments = read_sgm_segments(target_sgm)
    if len(source_segments) != len(target_segments):
        raise ValueError(
            f"Segment count mismatch: {source_sgm} has {len(source_segments)}, "
            f"{target_sgm} has {len(target_segments)}"
        )
    raw_en = raw_prefix.with_suffix(".raw.en")
    raw_fr = raw_prefix.with_suffix(".raw.fr")
    write_lines(raw_en, source_segments)
    write_lines(raw_fr, target_segments)
    return raw_en, raw_fr, len(source_segments)


def write_tokenized_tsv(source_path: Path, target_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="\n") as source_file:
        with target_path.open("r", encoding="utf-8", newline="\n") as target_file:
            with output_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(output_file, delimiter="\t")
                for source_line, target_line in zip_longest(source_file, target_file):
                    if source_line is None or target_line is None:
                        raise ValueError(
                            f"Tokenized segment mismatch: {source_path} vs {target_path}"
                        )
                    source = normalize_space(source_line)
                    target = normalize_space(target_line)
                    if source and target:
                        writer.writerow([source, target])
                        count += 1
    return count


def prepare_eval_set(
    name: str,
    source_sgm: Path,
    target_sgm: Path,
    work_dir: Path,
    scripts_dir: Path,
    threads: int,
    overwrite: bool,
) -> dict[str, object]:
    raw_prefix = work_dir / "raw_eval" / name
    raw_en, raw_fr, raw_count = write_sgm_pair_raw(source_sgm, target_sgm, raw_prefix)
    tok_en = work_dir / "tokenized_eval" / f"{name}.tok.en"
    tok_fr = work_dir / "tokenized_eval" / f"{name}.tok.fr"
    tokenize_file(raw_en, tok_en, "en", scripts_dir, threads, overwrite)
    tokenize_file(raw_fr, tok_fr, "fr", scripts_dir, threads, overwrite)
    output_path = work_dir / f"{name}.en-fr.tok.tsv"
    count = write_tokenized_tsv(tok_en, tok_fr, output_path)
    return {
        "name": name,
        "source_sgm": str(source_sgm),
        "target_sgm": str(target_sgm),
        "raw_segments": raw_count,
        "tokenized_pairs": count,
        "path": str(output_path),
    }


def concatenate_tsv(inputs: list[Path], output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for input_path in inputs:
            with input_path.open("r", encoding="utf-8", newline="") as input_file:
                for line in input_file:
                    output_file.write(line)
                    count += 1
    return count


def build_vocab(train_tsv: Path, source_vocab: Path, target_vocab: Path, size: int) -> dict[str, object]:
    source_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    rows = 0
    with train_tsv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            source_counter.update(row[0].split())
            target_counter.update(row[1].split())
            rows += 1

    source_vocab.parent.mkdir(parents=True, exist_ok=True)
    target_vocab.parent.mkdir(parents=True, exist_ok=True)
    source_items = source_counter.most_common(size)
    target_items = target_counter.most_common(size)
    with source_vocab.open("w", encoding="utf-8", newline="\n") as file:
        for token, count in source_items:
            file.write(f"{token}\t{count}\n")
    with target_vocab.open("w", encoding="utf-8", newline="\n") as file:
        for token, count in target_items:
            file.write(f"{token}\t{count}\n")

    return {
        "train_rows": rows,
        "vocab_words_per_language": size,
        "source_vocab": str(source_vocab),
        "target_vocab": str(target_vocab),
        "source_unique_tokens": len(source_counter),
        "target_unique_tokens": len(target_counter),
        "source_top_words": len(source_items),
        "target_top_words": len(target_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare paper-style WMT14 EN-FR word-level data: Moses tokenization, "
            "max-length filtered training TSV, newstest2012+2013 validation, "
            "newstest2014 test, and 30k word shortlists."
        )
    )
    parser.add_argument(
        "--extracted-dir", type=Path, default=Path("data/wmt14_enfr/extracted")
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/wmt14_enfr/paper_strict/wordlevel"),
    )
    parser.add_argument(
        "--moses-tokenizer-dir",
        type=Path,
        default=Path("tools/mosesdecoder/scripts/tokenizer"),
    )
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--vocab-words", type=int, default=30000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--overwrite-tokenized", action="store_true")
    args = parser.parse_args()

    scripts_dir = moses_script_dir(args.moses_tokenizer_dir)
    pairs = find_training_pairs(args.extracted_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No EN-FR training pairs found under {args.extracted_dir}. "
            "Download/extract WMT14 training archives first."
        )

    print("[training pairs]")
    for en_path, fr_path in pairs:
        print(f"  - {en_path} || {fr_path}")

    tokenized_pairs = tokenized_training_pairs(
        pairs=pairs,
        tokenized_dir=args.work_dir / "tokenized_train",
        scripts_dir=scripts_dir,
        threads=args.threads,
        overwrite=args.overwrite_tokenized,
    )
    train_path = args.work_dir / f"train.en-fr.tok.max{args.max_len}.tsv"
    train_manifest = write_train_tsv(
        tokenized_pairs=tokenized_pairs,
        output_path=train_path,
        max_len=args.max_len,
        max_pairs=args.max_pairs,
    )

    eval_sets = [
        prepare_eval_set(
            "newstest2012",
            args.extracted_dir / "dev" / "newstest2012-src.en.sgm",
            args.extracted_dir / "dev" / "newstest2012-ref.fr.sgm",
            args.work_dir,
            scripts_dir,
            args.threads,
            args.overwrite_tokenized,
        ),
        prepare_eval_set(
            "newstest2013",
            args.extracted_dir / "dev" / "newstest2013-src.en.sgm",
            args.extracted_dir / "dev" / "newstest2013-ref.fr.sgm",
            args.work_dir,
            scripts_dir,
            args.threads,
            args.overwrite_tokenized,
        ),
        prepare_eval_set(
            "newstest2014",
            args.extracted_dir / "test-full" / "newstest2014-fren-src.en.sgm",
            args.extracted_dir / "test-full" / "newstest2014-fren-ref.fr.sgm",
            args.work_dir,
            scripts_dir,
            args.threads,
            args.overwrite_tokenized,
        ),
    ]
    valid_path = args.work_dir / "valid.newstest2012_2013.en-fr.tok.tsv"
    valid_count = concatenate_tsv(
        [Path(eval_sets[0]["path"]), Path(eval_sets[1]["path"])], valid_path
    )

    vocab_manifest = build_vocab(
        train_path,
        args.work_dir / f"vocab.en.top{args.vocab_words}.txt",
        args.work_dir / f"vocab.fr.top{args.vocab_words}.txt",
        args.vocab_words,
    )

    manifest = {
        "paper": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "direction": "English-to-French",
        "preprocessing": {
            "tokenizer": "Moses tokenizer.perl",
            "lowercase": False,
            "stemming": False,
            "training_max_len": args.max_len,
            "vocab_words_per_language": args.vocab_words,
            "train_vocab_size_for_train_py": args.vocab_words + 4,
            "special_tokens": ["<pad>", "<sos>", "<eos>", "<unk>"],
        },
        "note": (
            "This prepares the paper-style word-level corpus. The original paper also "
            "uses Axelrod et al. data selection to reduce the full WMT14 pool to about "
            "348M words; use the selected TSV as train input for the final strict run."
        ),
        "training": train_manifest,
        "validation": {
            "path": str(valid_path),
            "sources": ["newstest2012", "newstest2013"],
            "pairs": valid_count,
        },
        "test": eval_sets[2],
        "eval_sets": eval_sets,
        "vocabulary": vocab_manifest,
    }
    manifest_path = args.work_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
