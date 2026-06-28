from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import zip_longest
from pathlib import Path


EN_PATTERNS = (
    "*.en",
    "*.en.gz",
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def word_count(text: str) -> int:
    return len(text.split())


def find_parallel_pairs(
    input_dir: Path,
    include_path: str | None = None,
    name_contains: str | None = None,
) -> list[tuple[Path, Path]]:
    english_files = []
    for pattern in EN_PATTERNS:
        english_files.extend(input_dir.rglob(pattern))

    pairs: list[tuple[Path, Path]] = []
    for en_path in sorted(set(english_files)):
        if include_path is not None and include_path not in en_path.as_posix():
            continue
        if name_contains is not None and name_contains not in en_path.name:
            continue
        if en_path.name.endswith(".en.gz"):
            fr_path = en_path.with_name(en_path.name[:-6] + ".fr.gz")
        else:
            fr_path = en_path.with_suffix(".fr")
        if fr_path.exists():
            pairs.append((en_path, fr_path))
    return pairs


def open_text(path: Path):
    if path.suffix == ".gz":
        import gzip

        return gzip.open(
            path, "rt", encoding="utf-8", errors="replace", newline="\n"
        )
    return path.open("r", encoding="utf-8", errors="replace", newline="\n")


def write_tsv(
    pairs: list[tuple[Path, Path]],
    output_path: Path,
    max_len: int,
    limit: int | None,
    direction: str,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    skipped_empty = 0
    skipped_long = 0
    skipped_unpaired = 0
    files: list[dict[str, object]] = []

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for en_path, fr_path in pairs:
            file_total = 0
            file_kept = 0
            file_unpaired = 0
            with open_text(en_path) as en_file, open_text(fr_path) as fr_file:
                for en_line, fr_line in zip_longest(en_file, fr_file):
                    total += 1
                    file_total += 1
                    if en_line is None or fr_line is None:
                        skipped_unpaired += 1
                        file_unpaired += 1
                        continue
                    en = normalize_space(en_line)
                    fr = normalize_space(fr_line)
                    if not en or not fr:
                        skipped_empty += 1
                        continue
                    if word_count(en) > max_len or word_count(fr) > max_len:
                        skipped_long += 1
                        continue
                    if direction == "en-fr":
                        writer.writerow([en, fr])
                    else:
                        writer.writerow([fr, en])
                    kept += 1
                    file_kept += 1
                    if limit is not None and kept >= limit:
                        break
            files.append(
                {
                    "en": str(en_path),
                    "fr": str(fr_path),
                    "total_seen": file_total,
                    "kept": file_kept,
                    "skipped_unpaired": file_unpaired,
                }
            )
            if limit is not None and kept >= limit:
                break

    return {
        "output": str(output_path),
        "direction": direction,
        "max_len": max_len,
        "limit": limit,
        "total_seen": total,
        "kept": kept,
        "skipped_empty": skipped_empty,
        "skipped_long": skipped_long,
        "skipped_unpaired": skipped_unpaired,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare extracted WMT14 EN-FR files as a TSV for train.py."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/wmt14_enfr/extracted")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/wmt14_enfr/wmt14_enfr.tsv")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/wmt14_enfr/prepared_manifest.json")
    )
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-path",
        type=str,
        default=None,
        help="Only include parallel files whose path contains this substring, e.g. /training/.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Only include parallel files whose filename contains this substring, e.g. fr-en.",
    )
    parser.add_argument(
        "--direction",
        choices=["en-fr", "fr-en"],
        default="en-fr",
        help="Column order in the generated TSV.",
    )
    args = parser.parse_args()

    pairs = find_parallel_pairs(
        args.input_dir,
        include_path=args.include_path,
        name_contains=args.name_contains,
    )
    if not pairs:
        raise FileNotFoundError(
            f"No .en/.fr parallel files found under {args.input_dir}. "
            "Run download_wmt14_enfr.py with --extract first."
        )

    print("Found parallel files:")
    for en_path, fr_path in pairs:
        print(f"  - {en_path} || {fr_path}")

    manifest = write_tsv(
        pairs=pairs,
        output_path=args.output,
        max_len=args.max_len,
        limit=args.limit,
        direction=args.direction,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
