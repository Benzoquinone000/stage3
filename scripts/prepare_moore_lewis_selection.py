from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from prepare_wmt14_paper_wordlevel import (
    concatenate_tsv,
    moses_script_dir,
    normalize_space,
    open_text,
    prepare_eval_set,
    tokenize_file,
    write_lines,
)


SCORE_PATTERN = re.compile(r"Total:\s+(-?\d+(?:\.\d+)?)\s+OOV:\s+(\d+)")


@dataclass(frozen=True)
class CandidatePair:
    name: str
    source: Path
    target: Path


def candidate_pairs(extracted_dir: Path) -> list[CandidatePair]:
    return [
        CandidatePair(
            "news_commentary",
            extracted_dir / "training/news-commentary-v9.fr-en.en",
            extracted_dir / "training/news-commentary-v9.fr-en.fr",
        ),
        CandidatePair(
            "europarl",
            extracted_dir / "training/europarl-v7.fr-en.en",
            extracted_dir / "training/europarl-v7.fr-en.fr",
        ),
        CandidatePair(
            "commoncrawl",
            extracted_dir / "commoncrawl.fr-en.en",
            extracted_dir / "commoncrawl.fr-en.fr",
        ),
        CandidatePair(
            "un",
            extracted_dir / "un/undoc.2000.fr-en.en",
            extracted_dir / "un/undoc.2000.fr-en.fr",
        ),
        CandidatePair(
            "giga_fren",
            extracted_dir / "giga-fren.release2.fixed.en.gz",
            extracted_dir / "giga-fren.release2.fixed.fr.gz",
        ),
    ]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def open_write_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="\n")
    return path.open("w", encoding="utf-8", newline="\n")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{now_utc()}] {message}\n")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{now_utc()}] $ {shlex.join(command)}\n")
        log_file.flush()
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def run_command_env(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{now_utc()}] $ {shlex.join(command)}\n")
        log_file.flush()
        subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )


def resolve_kenlm_tool(kenlm_bin: Path, name: str) -> Path:
    path = kenlm_bin / name
    if not path.exists():
        raise FileNotFoundError(f"Missing KenLM tool: {path}")
    return path


def prepare_devtest(args: argparse.Namespace) -> dict[str, object]:
    scripts_dir = moses_script_dir(args.moses_tokenizer_dir)
    work_dir = args.work_dir / "devtest"
    records = [
        prepare_eval_set(
            "newstest2012",
            args.extracted_dir / "dev/newstest2012-src.en.sgm",
            args.extracted_dir / "dev/newstest2012-ref.fr.sgm",
            work_dir,
            scripts_dir,
            args.threads,
            args.overwrite,
        ),
        prepare_eval_set(
            "newstest2013",
            args.extracted_dir / "dev/newstest2013-src.en.sgm",
            args.extracted_dir / "dev/newstest2013-ref.fr.sgm",
            work_dir,
            scripts_dir,
            args.threads,
            args.overwrite,
        ),
        prepare_eval_set(
            "newstest2014",
            args.extracted_dir / "test-full/newstest2014-fren-src.en.sgm",
            args.extracted_dir / "test-full/newstest2014-fren-ref.fr.sgm",
            work_dir,
            scripts_dir,
            args.threads,
            args.overwrite,
        ),
    ]
    valid_path = work_dir / "valid.newstest2012_2013.en-fr.tok.tsv"
    valid_count = concatenate_tsv(
        [Path(records[0]["path"]), Path(records[1]["path"])], valid_path
    )

    indomain_en = work_dir / "indomain.newstest2012_2013.tok.en"
    indomain_fr = work_dir / "indomain.newstest2012_2013.tok.fr"
    concatenate_plain(
        [
            work_dir / "tokenized_eval/newstest2012.tok.en",
            work_dir / "tokenized_eval/newstest2013.tok.en",
        ],
        indomain_en,
    )
    concatenate_plain(
        [
            work_dir / "tokenized_eval/newstest2012.tok.fr",
            work_dir / "tokenized_eval/newstest2013.tok.fr",
        ],
        indomain_fr,
    )

    manifest = {
        "stage": "devtest",
        "created_at": now_utc(),
        "records": records,
        "validation": {
            "path": str(valid_path),
            "pairs": valid_count,
            "sources": ["newstest2012", "newstest2013"],
        },
        "indomain_lm_text": {
            "en": str(indomain_en),
            "fr": str(indomain_fr),
        },
    }
    write_json(args.work_dir / "manifests/devtest_manifest.json", manifest)
    append_log(args.experiment_log, "Prepared tokenized newstest2012/2013/2014.")
    return manifest


def concatenate_plain(inputs: list[Path], output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for input_path in inputs:
            with input_path.open("r", encoding="utf-8", newline="\n") as input_file:
                for line in input_file:
                    output_file.write(line)
                    count += 1
    return count


def iter_candidate_rows(pairs: list[CandidatePair]):
    for pair in pairs:
        with open_text(pair.source) as source_file, open_text(pair.target) as target_file:
            for line_no, (source_line, target_line) in enumerate(
                zip_longest(source_file, target_file), start=1
            ):
                if source_line is None or target_line is None:
                    continue
                source = normalize_space(source_line)
                target = normalize_space(target_line)
                if source and target:
                    yield pair.name, line_no, source, target


def sample_general(args: argparse.Namespace) -> dict[str, object]:
    scripts_dir = moses_script_dir(args.moses_tokenizer_dir)
    pairs = candidate_pairs(args.extracted_dir)
    rng = random.Random(args.seed)
    reservoir: list[tuple[str, int, str, str]] = []
    seen = 0
    for item in iter_candidate_rows(pairs):
        seen += 1
        if len(reservoir) < args.general_sample_lines:
            reservoir.append(item)
        else:
            replace_idx = rng.randrange(seen)
            if replace_idx < args.general_sample_lines:
                reservoir[replace_idx] = item
        if args.max_scan_lines is not None and seen >= args.max_scan_lines:
            break
        if seen % args.progress_every == 0:
            message = f"sample-general scanned {seen} candidate pairs"
            print(f"[progress] {message}", flush=True)
            append_log(args.experiment_log, message)

    sample_dir = args.work_dir / "selection/general_sample"
    raw_en = sample_dir / "general_sample.raw.en"
    raw_fr = sample_dir / "general_sample.raw.fr"
    meta_path = sample_dir / "general_sample.rows.tsv"
    write_lines(raw_en, [item[2] for item in reservoir])
    write_lines(raw_fr, [item[3] for item in reservoir])
    with meta_path.open("w", encoding="utf-8", newline="") as meta_file:
        writer = csv.writer(meta_file, delimiter="\t")
        writer.writerow(["sample_index", "corpus", "line_no"])
        for idx, (name, line_no, _, _) in enumerate(reservoir):
            writer.writerow([idx, name, line_no])

    tok_en = sample_dir / "general_sample.tok.en"
    tok_fr = sample_dir / "general_sample.tok.fr"
    tokenize_file(raw_en, tok_en, "en", scripts_dir, args.threads, args.overwrite)
    tokenize_file(raw_fr, tok_fr, "fr", scripts_dir, args.threads, args.overwrite)
    manifest = {
        "stage": "general_sample",
        "created_at": now_utc(),
        "seed": args.seed,
        "requested_lines": args.general_sample_lines,
        "seen_lines": seen,
        "sampled_lines": len(reservoir),
        "max_scan_lines": args.max_scan_lines,
        "raw": {"en": str(raw_en), "fr": str(raw_fr)},
        "tokenized": {"en": str(tok_en), "fr": str(tok_fr)},
        "rows": str(meta_path),
    }
    write_json(args.work_dir / "manifests/general_sample_manifest.json", manifest)
    append_log(
        args.experiment_log,
        f"Sampled {len(reservoir)} general-domain lines from {seen} candidate pairs.",
    )
    return manifest


def build_lm(
    args: argparse.Namespace,
    text_path: Path,
    prefix: Path,
    log_path: Path,
) -> dict[str, object]:
    lmplz = resolve_kenlm_tool(args.kenlm_bin, "lmplz")
    build_binary = resolve_kenlm_tool(args.kenlm_bin, "build_binary")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    arpa_path = Path(f"{prefix}.arpa")
    binary_path = Path(f"{prefix}.binary")
    tmp_dir = args.work_dir / "tmp/kenlm"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite or not arpa_path.exists():
        run_command(
            [
                str(lmplz),
                "-o",
                str(args.lm_order),
                "--text",
                str(text_path),
                "--arpa",
                str(arpa_path),
                "--discount_fallback",
                "-S",
                args.kenlm_memory,
                "-T",
                str(tmp_dir),
                "--vocab_estimate",
                str(args.kenlm_vocab_estimate),
            ],
            log_path,
        )
    if args.overwrite or not binary_path.exists():
        run_command([str(build_binary), str(arpa_path), str(binary_path)], log_path)
    return {
        "text": str(text_path),
        "arpa": str(arpa_path),
        "binary": str(binary_path),
        "order": args.lm_order,
        "memory": args.kenlm_memory,
        "vocab_estimate": args.kenlm_vocab_estimate,
    }


def build_lms(args: argparse.Namespace) -> dict[str, object]:
    lm_dir = args.work_dir / "selection/lms"
    log_path = args.log_dir / "kenlm_build.log"
    devtest_dir = args.work_dir / "devtest"
    sample_dir = args.work_dir / "selection/general_sample"
    records = {
        "indomain_en": build_lm(
            args, devtest_dir / "indomain.newstest2012_2013.tok.en", lm_dir / "indomain.en", log_path
        ),
        "indomain_fr": build_lm(
            args, devtest_dir / "indomain.newstest2012_2013.tok.fr", lm_dir / "indomain.fr", log_path
        ),
        "general_en": build_lm(
            args, sample_dir / "general_sample.tok.en", lm_dir / "general.en", log_path
        ),
        "general_fr": build_lm(
            args, sample_dir / "general_sample.tok.fr", lm_dir / "general.fr", log_path
        ),
    }
    manifest = {
        "stage": "build_lms",
        "created_at": now_utc(),
        "kenlm_bin": str(args.kenlm_bin),
        "records": records,
    }
    write_json(args.work_dir / "manifests/lm_manifest.json", manifest)
    append_log(args.experiment_log, "Built in-domain and general-domain KenLM models.")
    return manifest


def run_query_scores(args: argparse.Namespace, lm_path: Path, text_path: Path, score_path: Path) -> int:
    query = resolve_kenlm_tool(args.kenlm_bin, "query")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "kenlm_query.log"
    count = 0
    with text_path.open("r", encoding="utf-8", newline="\n") as input_file:
        with score_path.open("w", encoding="utf-8", newline="\n") as output_file:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{now_utc()}] $ {query} -v sentence {lm_path} < {text_path}\n")
                process = subprocess.Popen(
                    [str(query), "-v", "sentence", str(lm_path)],
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    text=True,
                    encoding="utf-8",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    match = SCORE_PATTERN.search(line)
                    if not match:
                        raise RuntimeError(f"Could not parse KenLM query line: {line!r}")
                    output_file.write(f"{match.group(1)}\t{match.group(2)}\n")
                    count += 1
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, process.args)
    return count


def score_one_corpus(args: argparse.Namespace, pair: CandidatePair) -> dict[str, object]:
    scripts_dir = moses_script_dir(args.moses_tokenizer_dir)
    tmp_dir = args.work_dir / "tmp/scoring" / pair.name
    score_dir = args.work_dir / "selection/scores"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    output_path = score_dir / f"{pair.name}.scores.tsv"
    if output_path.exists() and not args.overwrite:
        rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))
        append_log(args.experiment_log, f"Skipped existing score file for {pair.name}: {rows} rows.")
        return {
            "corpus": pair.name,
            "source": str(pair.source),
            "target": str(pair.target),
            "score_path": str(output_path),
            "rows": rows,
            "status": "skipped_existing",
        }

    clean_en = tmp_dir / f"{pair.name}.clean.raw.en.gz"
    clean_fr = tmp_dir / f"{pair.name}.clean.raw.fr.gz"
    line_map = tmp_dir / f"{pair.name}.line_map.tsv"
    clean_rows = write_clean_candidate_pair(pair, clean_en, clean_fr, line_map)
    tok_en = tmp_dir / f"{pair.name}.tok.en"
    tok_fr = tmp_dir / f"{pair.name}.tok.fr"
    tokenize_file(clean_en, tok_en, "en", scripts_dir, args.threads, args.overwrite)
    tokenize_file(clean_fr, tok_fr, "fr", scripts_dir, args.threads, args.overwrite)

    lm_dir = args.work_dir / "selection/lms"
    score_files = {
        "in_en": tmp_dir / "in.en.score",
        "gen_en": tmp_dir / "gen.en.score",
        "in_fr": tmp_dir / "in.fr.score",
        "gen_fr": tmp_dir / "gen.fr.score",
    }
    counts = {
        "in_en": run_query_scores(args, lm_dir / "indomain.en.binary", tok_en, score_files["in_en"]),
        "gen_en": run_query_scores(args, lm_dir / "general.en.binary", tok_en, score_files["gen_en"]),
        "in_fr": run_query_scores(args, lm_dir / "indomain.fr.binary", tok_fr, score_files["in_fr"]),
        "gen_fr": run_query_scores(args, lm_dir / "general.fr.binary", tok_fr, score_files["gen_fr"]),
    }
    rows = combine_scores(pair.name, tok_en, tok_fr, line_map, score_files, output_path)
    if args.keep_scoring_tmp is False:
        for path in [clean_en, clean_fr, line_map, tok_en, tok_fr, *score_files.values()]:
            path.unlink(missing_ok=True)
    record = {
        "corpus": pair.name,
        "source": str(pair.source),
        "target": str(pair.target),
        "score_path": str(output_path),
        "clean_rows": clean_rows,
        "rows": rows,
        "query_counts": counts,
    }
    append_log(args.experiment_log, f"Scored {pair.name}: {rows} rows.")
    return record


def write_clean_candidate_pair(
    pair: CandidatePair,
    clean_en: Path,
    clean_fr: Path,
    line_map: Path,
) -> int:
    rows = 0
    with open_text(pair.source) as source_file, open_text(pair.target) as target_file:
        with open_write_text(clean_en) as clean_en_file:
            with open_write_text(clean_fr) as clean_fr_file:
                with line_map.open("w", encoding="utf-8", newline="\n") as map_file:
                    for original_line_no, (source_line, target_line) in enumerate(
                        zip_longest(source_file, target_file), start=1
                    ):
                        if source_line is None or target_line is None:
                            continue
                        source = normalize_space(source_line)
                        target = normalize_space(target_line)
                        if not source or not target:
                            continue
                        rows += 1
                        clean_en_file.write(f"{source}\n")
                        clean_fr_file.write(f"{target}\n")
                        map_file.write(f"{rows}\t{original_line_no}\n")
                        if rows % 1_000_000 == 0:
                            print(
                                f"[progress] clean {pair.name}: kept {rows} rows",
                                flush=True,
                            )
    return rows


def read_total_score(line: str) -> float:
    return float(line.split("\t", 1)[0])


def combine_scores(
    corpus: str,
    tok_en: Path,
    tok_fr: Path,
    line_map: Path,
    score_files: dict[str, Path],
    output_path: Path,
) -> int:
    rows = 0
    with tok_en.open("r", encoding="utf-8", newline="\n") as en_file:
        with tok_fr.open("r", encoding="utf-8", newline="\n") as fr_file:
            with line_map.open("r", encoding="utf-8") as map_file:
                with score_files["in_en"].open("r", encoding="utf-8") as in_en:
                    with score_files["gen_en"].open("r", encoding="utf-8") as gen_en:
                        with score_files["in_fr"].open("r", encoding="utf-8") as in_fr:
                            with score_files["gen_fr"].open("r", encoding="utf-8") as gen_fr:
                                with output_path.open("w", encoding="utf-8", newline="\n") as out:
                                    for values in zip_longest(
                                        en_file,
                                        fr_file,
                                        map_file,
                                        in_en,
                                        gen_en,
                                        in_fr,
                                        gen_fr,
                                    ):
                                        if any(value is None for value in values):
                                            raise RuntimeError(
                                                f"Line mismatch while scoring {corpus}"
                                            )
                                        (
                                            en_line,
                                            fr_line,
                                            map_line,
                                            in_en_line,
                                            gen_en_line,
                                            in_fr_line,
                                            gen_fr_line,
                                        ) = values
                                        clean_line_no, original_line_no = map_line.rstrip("\n").split("\t")
                                        en_tokens = normalize_space(en_line).split()
                                        fr_tokens = normalize_space(fr_line).split()
                                        if not en_tokens or not fr_tokens:
                                            continue
                                        en_len = len(en_tokens) + 1
                                        fr_len = len(fr_tokens) + 1
                                        score = (
                                            (-read_total_score(in_en_line) / en_len)
                                            - (-read_total_score(gen_en_line) / en_len)
                                            + (-read_total_score(in_fr_line) / fr_len)
                                            - (-read_total_score(gen_fr_line) / fr_len)
                                        )
                                        out.write(
                                            f"{score:.8f}\t{corpus}\t{original_line_no}\t"
                                            f"{clean_line_no}\t{len(en_tokens)}\t{len(fr_tokens)}\n"
                                        )
                                        rows += 1
                                        if rows % 1_000_000 == 0:
                                            print(
                                                f"[progress] combine {corpus}: wrote {rows} scores",
                                                flush=True,
                                            )
    return rows


def score_corpora(args: argparse.Namespace) -> dict[str, object]:
    selected = set(args.corpora) if args.corpora else None
    records = []
    for pair in candidate_pairs(args.extracted_dir):
        if selected is not None and pair.name not in selected:
            continue
        records.append(score_one_corpus(args, pair))
    manifest = {
        "stage": "score_corpora",
        "created_at": now_utc(),
        "records": records,
    }
    write_json(args.work_dir / "manifests/score_manifest.json", manifest)
    return manifest


def sort_scores(args: argparse.Namespace) -> dict[str, object]:
    score_dir = args.work_dir / "selection/scores"
    sorted_dir = args.work_dir / "selection/sorted"
    tmp_dir = args.work_dir / "tmp/sort"
    sorted_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    score_files = sorted(score_dir.glob("*.scores.tsv"))
    if not score_files:
        raise FileNotFoundError(f"No score files found under {score_dir}")
    output_path = sorted_dir / "all_scores.sorted.tsv"
    if args.overwrite or not output_path.exists():
        command = [
            "sort",
            "-S",
            args.sort_memory,
            "-T",
            str(tmp_dir),
            "-k1,1g",
            "-o",
            str(output_path),
            *[str(path) for path in score_files],
        ]
        env = dict(os.environ)
        env["LC_ALL"] = "C"
        run_command_env(command, args.log_dir / "sort_scores.log", env)
    rows = sum(1 for _ in output_path.open("r", encoding="utf-8"))
    manifest = {
        "stage": "sort_scores",
        "created_at": now_utc(),
        "input_files": [str(path) for path in score_files],
        "output": str(output_path),
        "rows": rows,
        "sort_memory": args.sort_memory,
    }
    write_json(args.work_dir / "manifests/sort_manifest.json", manifest)
    append_log(args.experiment_log, f"Sorted {rows} score rows.")
    return manifest


def select_scores(args: argparse.Namespace) -> dict[str, object]:
    sorted_path = args.work_dir / "selection/sorted/all_scores.sorted.tsv"
    if not sorted_path.exists():
        raise FileNotFoundError(f"Missing sorted score file: {sorted_path}")
    output_path = args.work_dir / "selection/selected/selected_ids.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    source_words = 0
    target_words = 0
    by_corpus: dict[str, dict[str, int]] = {}
    last_score: str | None = None
    with sorted_path.open("r", encoding="utf-8", newline="\n") as sorted_file:
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for line in sorted_file:
                score, corpus, original_line, clean_line, en_words, fr_words = line.rstrip("\n").split("\t")
                en_count = int(en_words)
                fr_count = int(fr_words)
                if rows > 0 and source_words >= args.target_source_words:
                    break
                source_words += en_count
                target_words += fr_count
                rows += 1
                last_score = score
                stats = by_corpus.setdefault(
                    corpus,
                    {"rows": 0, "source_words": 0, "target_words": 0},
                )
                stats["rows"] += 1
                stats["source_words"] += en_count
                stats["target_words"] += fr_count
                output_file.write(
                    f"{score}\t{corpus}\t{original_line}\t{clean_line}\t"
                    f"{en_words}\t{fr_words}\t{source_words}\n"
                )
                if rows % 1_000_000 == 0:
                    print(
                        f"[progress] select wrote {rows} rows, source_words={source_words}",
                        flush=True,
                    )
    manifest = {
        "stage": "select_scores",
        "created_at": now_utc(),
        "sorted_scores": str(sorted_path),
        "selected_ids": str(output_path),
        "target_source_words": args.target_source_words,
        "rows": rows,
        "source_words": source_words,
        "target_words": target_words,
        "last_score": last_score,
        "by_corpus": by_corpus,
    }
    write_json(args.work_dir / "manifests/selection_manifest.json", manifest)
    append_log(
        args.experiment_log,
        f"Selected {rows} rows with {source_words} source words.",
    )
    return manifest


def split_selected_ids(args: argparse.Namespace) -> dict[str, object]:
    selected_path = args.work_dir / "selection/selected/selected_ids.tsv"
    split_dir = args.work_dir / "selection/selected/by_corpus"
    tmp_dir = args.work_dir / "tmp/sort_selected"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected ids file: {selected_path}")

    split_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    corpus_names = [pair.name for pair in candidate_pairs(args.extracted_dir)]
    sorted_paths = {name: split_dir / f"{name}.selected_lines.tsv" for name in corpus_names}
    if (
        not args.overwrite
        and all(path.exists() for path in sorted_paths.values())
    ):
        records = {}
        for name, path in sorted_paths.items():
            records[name] = {
                "path": str(path),
                "rows": sum(1 for _ in path.open("r", encoding="utf-8")),
                "status": "skipped_existing",
            }
        manifest = {
            "stage": "split_selected_ids",
            "created_at": now_utc(),
            "selected_ids": str(selected_path),
            "records": records,
        }
        write_json(args.work_dir / "manifests/split_selected_manifest.json", manifest)
        append_log(args.experiment_log, "Skipped split-selected; by-corpus files already exist.")
        return manifest

    unsorted_paths = {
        name: split_dir / f"{name}.selected_lines.unsorted.tsv"
        for name in corpus_names
    }
    handles = {
        name: path.open("w", encoding="utf-8", newline="\n")
        for name, path in unsorted_paths.items()
    }
    counts = {name: 0 for name in corpus_names}
    try:
        with selected_path.open("r", encoding="utf-8", newline="\n") as selected_file:
            for row_no, line in enumerate(selected_file, start=1):
                score, corpus, original_line, clean_line, en_words, fr_words, cumulative = (
                    line.rstrip("\n").split("\t")
                )
                if corpus not in handles:
                    raise ValueError(f"Unknown corpus {corpus!r} at selected row {row_no}")
                handles[corpus].write(
                    f"{original_line}\t{clean_line}\t{score}\t"
                    f"{en_words}\t{fr_words}\t{cumulative}\n"
                )
                counts[corpus] += 1
                if row_no % args.progress_every == 0:
                    print(f"[progress] split-selected read {row_no} rows", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    env = dict(os.environ)
    env["LC_ALL"] = "C"
    records = {}
    for name in corpus_names:
        unsorted_path = unsorted_paths[name]
        sorted_path = sorted_paths[name]
        run_command_env(
            [
                "sort",
                "-S",
                args.sort_memory,
                "-T",
                str(tmp_dir),
                "-n",
                "-k1,1",
                "-o",
                str(sorted_path),
                str(unsorted_path),
            ],
            args.log_dir / "split_selected_sort.log",
            env,
        )
        if not args.keep_materialize_tmp:
            unsorted_path.unlink(missing_ok=True)
        records[name] = {
            "path": str(sorted_path),
            "rows": counts[name],
        }

    manifest = {
        "stage": "split_selected_ids",
        "created_at": now_utc(),
        "selected_ids": str(selected_path),
        "records": records,
        "sort_memory": args.sort_memory,
    }
    write_json(args.work_dir / "manifests/split_selected_manifest.json", manifest)
    append_log(args.experiment_log, f"Split selected ids into {len(records)} corpus files.")
    return manifest


def next_selected_line(selected_file) -> tuple[int, list[str]] | None:
    line = selected_file.readline()
    if not line:
        return None
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 6:
        raise ValueError(f"Malformed selected line record: {line!r}")
    return int(parts[0]), parts


def extract_selected_raw(
    pair: CandidatePair,
    selected_lines_path: Path,
    raw_en: Path,
    raw_fr: Path,
    progress_every: int,
) -> dict[str, object]:
    raw_en.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with selected_lines_path.open("r", encoding="utf-8", newline="\n") as selected_file:
        selected = next_selected_line(selected_file)
        with open_text(pair.source) as source_file, open_text(pair.target) as target_file:
            with open_write_text(raw_en) as raw_en_file, open_write_text(raw_fr) as raw_fr_file:
                for original_line_no, (source_line, target_line) in enumerate(
                    zip_longest(source_file, target_file), start=1
                ):
                    if selected is None:
                        break
                    wanted_line_no, _ = selected
                    if wanted_line_no < original_line_no:
                        raise RuntimeError(
                            f"Selected line order error for {pair.name}: "
                            f"wanted {wanted_line_no}, passed {original_line_no}"
                        )
                    if wanted_line_no != original_line_no:
                        continue
                    if source_line is None or target_line is None:
                        raise RuntimeError(
                            f"Selected line {wanted_line_no} is unpaired in {pair.name}"
                        )
                    source = normalize_space(source_line)
                    target = normalize_space(target_line)
                    if not source or not target:
                        raise RuntimeError(
                            f"Selected line {wanted_line_no} is empty after normalization in {pair.name}"
                        )
                    raw_en_file.write(f"{source}\n")
                    raw_fr_file.write(f"{target}\n")
                    rows += 1
                    if rows % progress_every == 0:
                        print(
                            f"[progress] materialize {pair.name}: extracted {rows} rows",
                            flush=True,
                        )
                    selected = next_selected_line(selected_file)
        if selected is not None:
            wanted_line_no, _ = selected
            raise RuntimeError(
                f"Could not find selected line {wanted_line_no} in corpus {pair.name}"
            )
    return {
        "corpus": pair.name,
        "selected_lines": str(selected_lines_path),
        "raw_en": str(raw_en),
        "raw_fr": str(raw_fr),
        "rows": rows,
    }


def append_tokenized_training_rows(
    tok_en: Path,
    tok_fr: Path,
    writer: csv.writer,
    max_len: int,
) -> dict[str, int]:
    total = 0
    kept = 0
    skipped_empty = 0
    skipped_long = 0
    skipped_unpaired = 0
    source_words = 0
    target_words = 0
    with tok_en.open("r", encoding="utf-8", newline="\n") as en_file:
        with tok_fr.open("r", encoding="utf-8", newline="\n") as fr_file:
            for en_line, fr_line in zip_longest(en_file, fr_file):
                total += 1
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
                source_words += en_len
                target_words += fr_len
    return {
        "total": total,
        "kept": kept,
        "skipped_empty": skipped_empty,
        "skipped_long": skipped_long,
        "skipped_unpaired": skipped_unpaired,
        "source_words": source_words,
        "target_words": target_words,
    }


def merge_count_stats(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def build_vocab_with_external_sort(
    args: argparse.Namespace,
    train_tsv: Path,
    source_vocab: Path,
    target_vocab: Path,
) -> dict[str, object]:
    tmp_dir = args.work_dir / "tmp/vocab"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LC_ALL"] = "C"

    def build_side(field: int, vocab_path: Path, lang: str) -> dict[str, object]:
        counts_path = tmp_dir / f"vocab.{lang}.counts"
        sorted_counts_path = tmp_dir / f"vocab.{lang}.counts.sorted"
        extract_command = (
            "set -o pipefail; "
            "awk -F '\\t' -v field="
            f"{field} "
            "'{ n=split($field, a, \" \"); for (i=1; i<=n; i++) if (a[i] != \"\") print a[i] }' "
            f"{shlex.quote(str(train_tsv))} "
            f"| sort -S {shlex.quote(args.sort_memory)} -T {shlex.quote(str(tmp_dir))} "
            f"| uniq -c > {shlex.quote(str(counts_path))}"
        )
        run_command_env(
            ["bash", "-lc", extract_command],
            args.log_dir / f"vocab_{lang}.log",
            env,
        )
        run_command_env(
            [
                "sort",
                "-S",
                args.sort_memory,
                "-T",
                str(tmp_dir),
                "-k1,1nr",
                "-o",
                str(sorted_counts_path),
                str(counts_path),
            ],
            args.log_dir / f"vocab_{lang}.log",
            env,
        )
        unique_tokens = 0
        top_words = 0
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with sorted_counts_path.open("r", encoding="utf-8", newline="\n") as counts_file:
            with vocab_path.open("w", encoding="utf-8", newline="\n") as vocab_file:
                for line in counts_file:
                    unique_tokens += 1
                    if top_words >= args.vocab_words:
                        continue
                    stripped = line.strip()
                    if not stripped:
                        continue
                    count, token = stripped.split(maxsplit=1)
                    vocab_file.write(f"{token}\t{count}\n")
                    top_words += 1
        if not args.keep_materialize_tmp:
            counts_path.unlink(missing_ok=True)
            sorted_counts_path.unlink(missing_ok=True)
        return {
            "vocab": str(vocab_path),
            "unique_tokens": unique_tokens,
            "top_words": top_words,
        }

    source = build_side(1, source_vocab, "en")
    target = build_side(2, target_vocab, "fr")
    return {
        "train_tsv": str(train_tsv),
        "vocab_words_per_language": args.vocab_words,
        "source_vocab": source["vocab"],
        "target_vocab": target["vocab"],
        "source_unique_tokens": source["unique_tokens"],
        "target_unique_tokens": target["unique_tokens"],
        "source_top_words": source["top_words"],
        "target_top_words": target["top_words"],
        "method": "external_sort",
        "sort_memory": args.sort_memory,
    }


def build_vocab_with_streaming_counter(
    args: argparse.Namespace,
    train_tsv: Path,
    source_vocab: Path,
    target_vocab: Path,
) -> dict[str, object]:
    def build_side(field: int, vocab_path: Path, lang: str) -> dict[str, object]:
        counter: Counter[str] = Counter()
        rows = 0
        with train_tsv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                if len(row) <= field:
                    continue
                counter.update(row[field].split())
                rows += 1
                if rows % args.progress_every == 0:
                    print(f"[progress] vocab {lang}: read {rows} rows", flush=True)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        top_items = counter.most_common(args.vocab_words)
        with vocab_path.open("w", encoding="utf-8", newline="\n") as vocab_file:
            for token, count in top_items:
                vocab_file.write(f"{token}\t{count}\n")
        return {
            "vocab": str(vocab_path),
            "rows": rows,
            "unique_tokens": len(counter),
            "top_words": len(top_items),
        }

    source = build_side(0, source_vocab, "en")
    target = build_side(1, target_vocab, "fr")
    return {
        "train_tsv": str(train_tsv),
        "vocab_words_per_language": args.vocab_words,
        "source_vocab": source["vocab"],
        "target_vocab": target["vocab"],
        "source_unique_tokens": source["unique_tokens"],
        "target_unique_tokens": target["unique_tokens"],
        "source_top_words": source["top_words"],
        "target_top_words": target["top_words"],
        "train_rows": source["rows"],
        "method": "streaming_python_counter",
        "tokenizer": "python str.split whitespace, matching train.py tokenizer=whitespace",
    }


def rebuild_selected_vocab(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.work_dir / "wordlevel"
    train_path = output_dir / f"train.en-fr.tok.selected.max{args.max_len}.tsv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing selected training TSV: {train_path}")
    manifest = build_vocab_with_streaming_counter(
        args,
        train_path,
        output_dir / f"vocab.en.top{args.vocab_words}.txt",
        output_dir / f"vocab.fr.top{args.vocab_words}.txt",
    )
    wordlevel_manifest_path = output_dir / "manifest.json"
    if wordlevel_manifest_path.exists():
        payload = json.loads(wordlevel_manifest_path.read_text(encoding="utf-8"))
        payload["vocabulary"] = manifest
        write_json(wordlevel_manifest_path, payload)
    write_json(args.work_dir / "manifests/vocab_selected_manifest.json", manifest)
    append_log(
        args.experiment_log,
        f"Rebuilt selected vocabularies with streaming counter: "
        f"{manifest['source_top_words']} source / {manifest['target_top_words']} target words.",
    )
    return manifest


def prepare_selected_wordlevel(args: argparse.Namespace) -> dict[str, object]:
    split_dir = args.work_dir / "selection/selected/by_corpus"
    if not all(
        (split_dir / f"{pair.name}.selected_lines.tsv").exists()
        for pair in candidate_pairs(args.extracted_dir)
    ):
        split_selected_ids(args)

    scripts_dir = moses_script_dir(args.moses_tokenizer_dir)
    output_dir = args.work_dir / "wordlevel"
    raw_dir = args.work_dir / "selection/selected/raw"
    tok_dir = args.work_dir / "selection/selected/tokenized"
    train_path = output_dir / f"train.en-fr.tok.selected.max{args.max_len}.tsv"
    if train_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{train_path} already exists. Pass --overwrite to regenerate selected wordlevel data."
        )

    total_stats: dict[str, int] = {}
    corpus_records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with train_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for pair in candidate_pairs(args.extracted_dir):
            selected_lines = split_dir / f"{pair.name}.selected_lines.tsv"
            raw_en = raw_dir / f"{pair.name}.selected.raw.en.gz"
            raw_fr = raw_dir / f"{pair.name}.selected.raw.fr.gz"
            tok_en = tok_dir / f"{pair.name}.selected.tok.en"
            tok_fr = tok_dir / f"{pair.name}.selected.tok.fr"
            extract_record = extract_selected_raw(
                pair,
                selected_lines,
                raw_en,
                raw_fr,
                args.progress_every,
            )
            tokenize_file(raw_en, tok_en, "en", scripts_dir, args.threads, args.overwrite)
            tokenize_file(raw_fr, tok_fr, "fr", scripts_dir, args.threads, args.overwrite)
            stats = append_tokenized_training_rows(tok_en, tok_fr, writer, args.max_len)
            merge_count_stats(total_stats, stats)
            corpus_records.append(
                {
                    "corpus": pair.name,
                    "extraction": extract_record,
                    "tokenized_en": str(tok_en),
                    "tokenized_fr": str(tok_fr),
                    "training_filter": stats,
                }
            )
            append_log(
                args.experiment_log,
                f"Materialized {pair.name}: kept {stats['kept']} max{args.max_len} rows.",
            )
            if not args.keep_materialize_tmp:
                raw_en.unlink(missing_ok=True)
                raw_fr.unlink(missing_ok=True)
                tok_en.unlink(missing_ok=True)
                tok_fr.unlink(missing_ok=True)

    valid_path = output_dir / "valid.newstest2012_2013.en-fr.tok.tsv"
    test_path = output_dir / "newstest2014.en-fr.tok.tsv"
    newstest2012_path = output_dir / "newstest2012.en-fr.tok.tsv"
    newstest2013_path = output_dir / "newstest2013.en-fr.tok.tsv"
    devtest_dir = args.work_dir / "devtest"
    valid_count = concatenate_tsv(
        [devtest_dir / "valid.newstest2012_2013.en-fr.tok.tsv"],
        valid_path,
    )
    test_count = concatenate_tsv(
        [devtest_dir / "newstest2014.en-fr.tok.tsv"],
        test_path,
    )
    newstest2012_count = concatenate_tsv(
        [devtest_dir / "newstest2012.en-fr.tok.tsv"],
        newstest2012_path,
    )
    newstest2013_count = concatenate_tsv(
        [devtest_dir / "newstest2013.en-fr.tok.tsv"],
        newstest2013_path,
    )
    vocab_manifest = build_vocab_with_streaming_counter(
        args,
        train_path,
        output_dir / f"vocab.en.top{args.vocab_words}.txt",
        output_dir / f"vocab.fr.top{args.vocab_words}.txt",
    )
    manifest = {
        "stage": "materialize_selected_wordlevel",
        "created_at": now_utc(),
        "paper": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "direction": "English-to-French",
        "selection": {
            "selected_ids": str(args.work_dir / "selection/selected/selected_ids.tsv"),
            "target_source_words_before_length_filter": args.target_source_words,
            "method": "bilingual Moore-Lewis/Axelrod-style cross-entropy difference",
        },
        "preprocessing": {
            "tokenizer": "Moses tokenizer.perl",
            "lowercase": False,
            "stemming": False,
            "training_max_len": args.max_len,
            "vocab_words_per_language": args.vocab_words,
            "train_vocab_size_for_train_py": args.vocab_words + 4,
            "special_tokens": ["<pad>", "<sos>", "<eos>", "<unk>"],
        },
        "training": {
            "path": str(train_path),
            **total_stats,
            "corpora": corpus_records,
        },
        "validation": {
            "path": str(valid_path),
            "pairs": valid_count,
            "sources": ["newstest2012", "newstest2013"],
        },
        "test": {
            "path": str(test_path),
            "pairs": test_count,
            "source": "newstest2014",
        },
        "eval_sets": {
            "newstest2012": {"path": str(newstest2012_path), "pairs": newstest2012_count},
            "newstest2013": {"path": str(newstest2013_path), "pairs": newstest2013_count},
            "newstest2014": {"path": str(test_path), "pairs": test_count},
        },
        "vocabulary": vocab_manifest,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(args.work_dir / "manifests/materialize_selected_manifest.json", manifest)
    append_log(
        args.experiment_log,
        f"Prepared selected wordlevel max{args.max_len}: {total_stats.get('kept', 0)} rows.",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare strict Moore-Lewis data selection for Bahdanau WMT14 EN-FR."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "devtest",
            "sample-general",
            "build-lms",
            "score",
            "sort-scores",
            "select",
            "sort-select",
            "split-selected",
            "materialize-selected",
            "wordlevel-selected",
            "vocab-selected",
            "all-small",
        ],
        required=True,
    )
    parser.add_argument("--extracted-dir", type=Path, default=Path("data/wmt14_enfr/extracted"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/wmt14_enfr/paper_strict"))
    parser.add_argument(
        "--moses-tokenizer-dir",
        type=Path,
        default=Path("tools/mosesdecoder/scripts/tokenizer"),
    )
    parser.add_argument("--kenlm-bin", type=Path, default=Path("tools/kenlm/build/bin"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/strict"))
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=Path("STRICT_EXPERIMENT_LOG.md"),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--general-sample-lines", type=int, default=50000)
    parser.add_argument("--max-scan-lines", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--lm-order", type=int, default=5)
    parser.add_argument("--kenlm-memory", default="4G")
    parser.add_argument("--kenlm-vocab-estimate", type=int, default=500000)
    parser.add_argument("--sort-memory", default="4G")
    parser.add_argument("--target-source-words", type=int, default=348_000_000)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--vocab-words", type=int, default=30000)
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-scoring-tmp", action="store_true")
    parser.add_argument("--keep-materialize-tmp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    append_log(args.experiment_log, f"START stage={args.stage}")
    if args.stage == "devtest":
        manifest = prepare_devtest(args)
    elif args.stage == "sample-general":
        manifest = sample_general(args)
    elif args.stage == "build-lms":
        manifest = build_lms(args)
    elif args.stage == "score":
        manifest = score_corpora(args)
    elif args.stage == "sort-scores":
        manifest = sort_scores(args)
    elif args.stage == "select":
        manifest = select_scores(args)
    elif args.stage == "sort-select":
        manifest = {
            "sort": sort_scores(args),
            "select": select_scores(args),
        }
    elif args.stage == "split-selected":
        manifest = split_selected_ids(args)
    elif args.stage == "materialize-selected":
        manifest = prepare_selected_wordlevel(args)
    elif args.stage == "wordlevel-selected":
        manifest = {
            "split_selected": split_selected_ids(args),
            "wordlevel": prepare_selected_wordlevel(args),
        }
    elif args.stage == "vocab-selected":
        manifest = rebuild_selected_vocab(args)
    else:
        manifest = {
            "devtest": prepare_devtest(args),
            "general_sample": sample_general(args),
            "lms": build_lms(args),
        }
    append_log(args.experiment_log, f"END stage={args.stage}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
