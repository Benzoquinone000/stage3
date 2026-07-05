from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_TRAINING_PAIRS = {
    "news_commentary": (
        "extracted/training/news-commentary-v9.fr-en.en",
        "extracted/training/news-commentary-v9.fr-en.fr",
    ),
    "europarl": (
        "extracted/training/europarl-v7.fr-en.en",
        "extracted/training/europarl-v7.fr-en.fr",
    ),
    "commoncrawl": (
        "extracted/commoncrawl.fr-en.en",
        "extracted/commoncrawl.fr-en.fr",
    ),
    "un": (
        "extracted/un/undoc.2000.fr-en.en",
        "extracted/un/undoc.2000.fr-en.fr",
    ),
    "giga_fren": (
        "extracted/giga-fren.release2.fixed.en.gz",
        "extracted/giga-fren.release2.fixed.fr.gz",
    ),
}

REQUIRED_EVAL_FILES = {
    "newstest2012_src": "extracted/dev/newstest2012-src.en.sgm",
    "newstest2012_ref": "extracted/dev/newstest2012-ref.fr.sgm",
    "newstest2013_src": "extracted/dev/newstest2013-src.en.sgm",
    "newstest2013_ref": "extracted/dev/newstest2013-ref.fr.sgm",
    "newstest2014_src": "extracted/test-full/newstest2014-fren-src.en.sgm",
    "newstest2014_ref": "extracted/test-full/newstest2014-fren-ref.fr.sgm",
}

REQUIRED_ARCHIVES = (
    "archives/training-parallel-nc-v9.tgz",
    "archives/training-parallel-europarl-v7.tgz",
    "archives/training-parallel-commoncrawl.tgz",
    "archives/training-parallel-un.tgz",
    "archives/training-giga-fren.tar",
    "archives/dev.tgz",
    "archives/test-full.tgz",
)

STRICT_DIRS = (
    "paper_strict/devtest",
    "paper_strict/manifests",
    "paper_strict/selected",
    "paper_strict/selection",
    "paper_strict/tmp",
    "paper_strict/wordlevel",
)


def find_executable(name: str, local_bin: Path) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    local_path = local_bin / name
    if local_path.exists():
        return str(local_path)
    return None


def exists_report(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit readiness for the strict Bahdanau WMT14 reproduction."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/wmt14_enfr"))
    parser.add_argument(
        "--moses-tokenizer-dir",
        type=Path,
        default=Path("tools/mosesdecoder/scripts/tokenizer"),
    )
    parser.add_argument(
        "--kenlm-bin",
        type=Path,
        default=Path("tools/kenlm/build/bin"),
        help="Local KenLM bin directory used when commands are not on PATH.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    report: dict[str, object] = {
        "data_dir": str(data_dir),
        "archives": {
            name: exists_report(data_dir, name) for name in REQUIRED_ARCHIVES
        },
        "training_pairs": {
            name: {
                "source": exists_report(data_dir, source),
                "target": exists_report(data_dir, target),
            }
            for name, (source, target) in REQUIRED_TRAINING_PAIRS.items()
        },
        "eval_files": {
            name: exists_report(data_dir, path)
            for name, path in REQUIRED_EVAL_FILES.items()
        },
        "strict_dirs": {
            name: {
                "path": str(data_dir / name),
                "exists": (data_dir / name).is_dir(),
            }
            for name in STRICT_DIRS
        },
        "moses_tokenizer": {
            name: exists_report(args.moses_tokenizer_dir, name)
            for name in (
                "normalize-punctuation.perl",
                "remove-non-printing-char.perl",
                "tokenizer.perl",
                "detokenizer.perl",
            )
        },
        "executables": {
            name: find_executable(name, args.kenlm_bin)
            for name in ("perl", "sacrebleu", "lmplz", "build_binary", "query")
        },
    }

    missing: list[str] = []
    for group_name in ("archives", "eval_files", "strict_dirs", "moses_tokenizer"):
        group = report[group_name]
        assert isinstance(group, dict)
        for item_name, item in group.items():
            assert isinstance(item, dict)
            if not item.get("exists"):
                missing.append(f"{group_name}.{item_name}")
    training_pairs = report["training_pairs"]
    assert isinstance(training_pairs, dict)
    for pair_name, pair in training_pairs.items():
        assert isinstance(pair, dict)
        for side_name, side in pair.items():
            assert isinstance(side, dict)
            if not side.get("exists"):
                missing.append(f"training_pairs.{pair_name}.{side_name}")
    executables = report["executables"]
    assert isinstance(executables, dict)
    for name in ("lmplz", "build_binary", "query"):
        if not executables.get(name):
            missing.append(f"executables.{name}")

    report["ready_for_selection"] = len(missing) == 0
    report["missing"] = missing
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
