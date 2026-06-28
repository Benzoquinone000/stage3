from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path


SEGMENT_PATTERN = re.compile(r"<seg[^>]*>(.*?)</seg>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")


def read_sgm_segments(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    segments = []
    for match in SEGMENT_PATTERN.finditer(text):
        segment = TAG_PATTERN.sub("", match.group(1))
        segment = html.unescape(segment)
        segment = re.sub(r"\s+", " ", segment).strip()
        if segment:
            segments.append(segment)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert paired WMT SGM files to TSV.")
    parser.add_argument("--source-sgm", type=Path, required=True)
    parser.add_argument("--target-sgm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_segments = read_sgm_segments(args.source_sgm)
    target_segments = read_sgm_segments(args.target_sgm)
    if len(source_segments) != len(target_segments):
        raise ValueError(
            f"Segment count mismatch: {args.source_sgm} has {len(source_segments)}, "
            f"{args.target_sgm} has {len(target_segments)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        for source, target in zip(source_segments, target_segments):
            writer.writerow([source, target])
    print(f"Wrote {len(source_segments)} pairs to {args.output}")


if __name__ == "__main__":
    main()
