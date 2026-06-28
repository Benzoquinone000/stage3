from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


BASE_URL = "https://statmt.org/wmt14"
LEGACY_BASE_URL = "https://statmt.org"


@dataclass(frozen=True)
class Resource:
    name: str
    url: str
    size_hint: str
    description: str
    profile: tuple[str, ...]


RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="dev",
        url=f"{BASE_URL}/dev.tgz",
        size_hint="17 MB",
        description="WMT14 development sets, including news-test2012 and news-test2013.",
        profile=("devtest", "paper-small", "paper-full"),
    ),
    Resource(
        name="test-filtered",
        url=f"{BASE_URL}/test-filtered.tgz",
        size_hint="3.2 MB",
        description="Filtered official WMT14 test sets used for evaluation.",
        profile=("devtest", "paper-small", "paper-full"),
    ),
    Resource(
        name="test-full",
        url=f"{BASE_URL}/test-full.tgz",
        size_hint="3.2 MB",
        description="Cleaned WMT14 test sets with later minor fixes.",
        profile=("devtest", "paper-small", "paper-full"),
    ),
    Resource(
        name="news-commentary-v9",
        url=f"{BASE_URL}/training-parallel-nc-v9.tgz",
        size_hint="77 MB",
        description="News Commentary v9 parallel corpus. Smallest FR-EN training corpus.",
        profile=("paper-small", "paper-full"),
    ),
    Resource(
        name="europarl-v7",
        url=f"{LEGACY_BASE_URL}/wmt13/training-parallel-europarl-v7.tgz",
        size_hint="628 MB",
        description="Europarl v7 parallel corpus.",
        profile=("paper-full",),
    ),
    Resource(
        name="commoncrawl",
        url=f"{LEGACY_BASE_URL}/wmt13/training-parallel-commoncrawl.tgz",
        size_hint="876 MB",
        description="Common Crawl parallel corpus.",
        profile=("paper-full",),
    ),
    Resource(
        name="un",
        url=f"{LEGACY_BASE_URL}/wmt13/training-parallel-un.tgz",
        size_hint="2.3 GB",
        description="UN parallel corpus.",
        profile=("paper-full",),
    ),
    Resource(
        name="giga-fren",
        url=f"{LEGACY_BASE_URL}/wmt10/training-giga-fren.tar",
        size_hint="2.3 GB",
        description="10^9 French-English corpus, same as WMT10 release.",
        profile=("paper-full",),
    ),
)


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_resources(
    profile: str, include: list[str], exclude: list[str]
) -> list[Resource]:
    if include:
        names = set(include)
        selected = [resource for resource in RESOURCES if resource.name in names]
        missing = sorted(names - {resource.name for resource in selected})
        if missing:
            raise ValueError(f"Unknown resource name(s): {', '.join(missing)}")
    else:
        selected = [resource for resource in RESOURCES if profile in resource.profile]

    excluded = set(exclude)
    return [resource for resource in selected if resource.name not in excluded]


def download_file(
    resource: Resource, archive_dir: Path, overwrite: bool
) -> dict[str, object]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / Path(resource.url).name
    part = target.with_suffix(target.suffix + ".part")

    if target.exists() and not overwrite:
        print(
            f"[skip] {target.name} already exists ({human_bytes(target.stat().st_size)})"
        )
        return {
            "name": resource.name,
            "url": resource.url,
            "archive": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "status": "skipped_existing",
        }

    resume_from = 0 if overwrite or not part.exists() else part.stat().st_size
    print(f"[download] {resource.name}: {resource.url} ({resource.size_hint})")
    if resume_from:
        print(f"[resume] {part.name} from {human_bytes(resume_from)}")
    start = time.time()
    request = urllib.request.Request(resource.url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    mode = "ab" if resume_from else "wb"
    with urllib.request.urlopen(request, timeout=60) as response:
        if resume_from and response.status != 206:
            print("[resume] server ignored Range header; restarting download")
            mode = "wb"
            resume_from = 0
        with part.open(mode) as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)
    part.replace(target)
    elapsed = time.time() - start
    size = target.stat().st_size
    print(f"[done] {target.name}: {human_bytes(size)} in {elapsed:.1f}s")
    return {
        "name": resource.name,
        "url": resource.url,
        "archive": str(target),
        "bytes": size,
        "sha256": sha256_file(target),
        "status": "downloaded",
    }


def safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        root = extract_dir.resolve()
        for member in archive.getmembers():
            destination = (extract_dir / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        archive.extractall(extract_dir)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    print(f"[extract] {archive_path.name} -> {extract_dir}")
    if archive_path.suffix == ".tar":
        safe_extract_tar(archive_path, extract_dir)
        return
    if archive_path.name.endswith(".tgz") or archive_path.name.endswith(".tar.gz"):
        safe_extract_tar(archive_path, extract_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def maybe_run(command: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return
    print(f"[run] {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def write_manifest(
    output_dir: Path,
    profile: str,
    resources: list[Resource],
    downloaded: list[dict[str, object]],
) -> None:
    manifest = {
        "dataset": "WMT14 English-French",
        "profile": profile,
        "source_page": "https://statmt.org/wmt14/translation-task.html",
        "resources": [asdict(resource) for resource in resources],
        "files": downloaded,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[manifest] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official WMT14 English-French resources used by Bahdanau et al. (2014)."
    )
    parser.add_argument(
        "--profile",
        choices=["devtest", "paper-small", "paper-full"],
        default="devtest",
        help=(
            "devtest downloads only dev/test; paper-small adds News Commentary; "
            "paper-full adds Europarl/CommonCrawl/UN/Giga-Fren."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/wmt14_enfr"))
    parser.add_argument(
        "--include", nargs="*", default=[], help="Explicit resource names to download."
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="Resource names to skip."
    )
    parser.add_argument(
        "--extract", action="store_true", help="Extract downloaded tar/tgz archives."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-download existing archives."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually download. Without this, only prints the plan.",
    )
    parser.add_argument(
        "--print-postprocess",
        action="store_true",
        help="Print example preprocessing commands after download.",
    )
    args = parser.parse_args()

    resources = selected_resources(args.profile, args.include, args.exclude)
    archive_dir = args.output_dir / "archives"
    extract_dir = args.output_dir / "extracted"

    print("Selected WMT14 EN-FR resources:")
    for resource in resources:
        print(
            f"  - {resource.name:18s} {resource.size_hint:>8s}  {resource.description}"
        )

    if not args.yes:
        print("\nDry run only. Add --yes to download.")
        print(
            "For the complete paper profile, expect several GB of archives and much larger extracted data."
        )
        write_manifest(args.output_dir, args.profile, resources, [])
        return

    downloaded: list[dict[str, object]] = []
    for resource in resources:
        downloaded.append(download_file(resource, archive_dir, args.overwrite))

    if args.extract:
        for item in downloaded:
            extract_archive(Path(str(item["archive"])), extract_dir)

    write_manifest(args.output_dir, args.profile, resources, downloaded)

    if args.print_postprocess:
        print("\nExample follow-up commands:")
        print(
            "  python scripts/prepare_wmt14_enfr.py --input-dir data/wmt14_enfr/extracted --output data/wmt14_enfr/wmt14_enfr.tsv"
        )
        print(
            "  python train.py --preset paper --data-path data/wmt14_enfr/wmt14_enfr.tsv --source-col 0 --target-col 1"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
