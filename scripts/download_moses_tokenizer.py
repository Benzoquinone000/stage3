from __future__ import annotations

import argparse
import json
import shutil
import stat
import time
import urllib.request
from pathlib import Path


BASE_URLS = (
    "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer",
    "https://cdn.jsdelivr.net/gh/moses-smt/mosesdecoder@master/scripts/tokenizer",
)

FILES = (
    "normalize-punctuation.perl",
    "remove-non-printing-char.perl",
    "tokenizer.perl",
    "detokenizer.perl",
)


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_from_mosestokenizer(output_dir: Path, overwrite: bool) -> dict[str, object]:
    try:
        import mosestokenizer
    except ImportError as exc:
        raise RuntimeError(
            "mosestokenizer is not installed. Install it with "
            "`python -m pip install mosestokenizer` or use --source raw."
        ) from exc

    package_dir = Path(mosestokenizer.__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = {
        "normalize-punctuation.perl": "normalize-punctuation.perl",
        "tokenizer-v1.1.perl": "tokenizer.perl",
        "detokenizer.perl": "detokenizer.perl",
    }
    records: list[dict[str, object]] = []
    for source_name, target_name in source_map.items():
        source = package_dir / source_name
        target = output_dir / target_name
        if target.exists() and not overwrite:
            records.append(
                {
                    "source": str(source),
                    "path": str(target),
                    "status": "skipped_existing",
                }
            )
            continue
        shutil.copy2(source, target)
        make_executable(target)
        records.append(
            {"source": str(source), "path": str(target), "status": "copied"}
        )

    prefixes_source = package_dir / "nonbreaking_prefixes"
    prefixes_target = output_dir / "nonbreaking_prefixes"
    if prefixes_source.exists() and (overwrite or not prefixes_target.exists()):
        if prefixes_target.exists():
            shutil.rmtree(prefixes_target)
        shutil.copytree(prefixes_source, prefixes_target)
        records.append(
            {
                "source": str(prefixes_source),
                "path": str(prefixes_target),
                "status": "copied",
            }
        )

    return {
        "source": "mosestokenizer package",
        "package_path": str(package_dir),
        "files": records,
    }


def download(
    urls: list[str], output: Path, overwrite: bool, retries: int, timeout: int
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return {"urls": urls, "path": str(output), "status": "skipped_existing"}

    last_error: Exception | None = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                print(f"[download] {url} -> {output} (attempt {attempt}/{retries})")
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    output.write_bytes(response.read())
                make_executable(output)
                return {"url": url, "path": str(output), "status": "downloaded"}
            except Exception as exc:
                last_error = exc
                print(f"[retry] {type(exc).__name__}: {exc}")
                time.sleep(min(10, attempt * 2))

    raise RuntimeError(f"Failed to download {output.name}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Moses tokenizer scripts used for paper-style WMT preprocessing."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/mosesdecoder/scripts/tokenizer"),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "mosestokenizer", "raw"],
        default="auto",
        help=(
            "auto first copies scripts from the installed mosestokenizer package, "
            "then falls back to raw GitHub/CDN downloads."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.source in {"auto", "mosestokenizer"}:
        try:
            manifest = copy_from_mosestokenizer(args.output_dir, args.overwrite)
            manifest_path = args.output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[manifest] {manifest_path}")
            return
        except RuntimeError:
            if args.source == "mosestokenizer":
                raise
            print("[fallback] mosestokenizer unavailable; trying raw URLs")

    records = [
        download(
            [f"{base_url}/{name}" for base_url in BASE_URLS],
            args.output_dir / name,
            args.overwrite,
            args.retries,
            args.timeout,
        )
        for name in FILES
    ]
    manifest = {
        "source": "moses-smt/mosesdecoder tokenizer scripts",
        "base_urls": BASE_URLS,
        "files": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
