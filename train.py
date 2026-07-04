from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nmt_attention.config import PRESETS
from nmt_attention.data import (
    BucketBatchSampler,
    ParallelTextDataset,
    TOKENIZER_CHOICES,
    Vocabulary,
    make_collate_fn,
    read_parallel_tsv,
    split_examples,
)
from nmt_attention.model import build_model
from nmt_attention.training import build_optimizer, evaluate, get_amp_dtype, train_epoch
from nmt_attention.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_checkpoint_artifact,
    log_wandb,
    maybe_watch_model,
)


DATA_URL = "https://download.pytorch.org/tutorial/data.zip"


def optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "all"}:
        return None
    return int(value)


def serialize_namespace(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def download_tutorial_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = data_dir / "eng-fra.txt"
    if pairs_path.exists():
        return pairs_path

    archive_path = data_dir / "pytorch_tutorial_data.zip"
    print(f"Downloading small translation corpus to {archive_path} ...")
    urllib.request.urlretrieve(DATA_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(data_dir)

    extracted = data_dir / "data" / "eng-fra.txt"
    if not extracted.exists():
        raise FileNotFoundError(f"Expected {extracted} after extracting {archive_path}")
    extracted.replace(pairs_path)
    return pairs_path


def epoch_time(start_time: float, end_time: float) -> tuple[int, int]:
    elapsed = int(end_time - start_time)
    return elapsed // 60, elapsed % 60


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    args: argparse.Namespace,
    best_valid_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "source_vocab": source_vocab.to_dict(),
        "target_vocab": target_vocab.to_dict(),
        "args": serialize_namespace(args),
        "best_valid_loss": best_valid_loss,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def write_metadata(
    metadata_path: Path,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    train_size: int,
    valid_size: int,
    test_size: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "paper": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "model": "Bidirectional GRU encoder + Bahdanau attention conditional GRU decoder",
        "implementation_details": {
            "encoder": "bidirectional GRU annotations",
            "attention": "additive Bahdanau attention over all source annotations",
            "decoder": "RNNsearch-style conditional GRU with context contributions to reset/update/candidate gates",
            "readout": args.readout,
            "optimizer": args.optimizer,
            "gradient_clip": args.clip,
            "sort_k_batches": args.sort_k_batches,
        },
        "source_language": args.source_language,
        "target_language": args.target_language,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "source_vocab_size": len(source_vocab),
        "target_vocab_size": len(target_vocab),
        "args": serialize_namespace(args),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=PRESETS.keys(), default="tutorial")
    preset_args, _ = preset_parser.parse_known_args()
    preset = PRESETS[preset_args.preset]

    parser = argparse.ArgumentParser(
        description="Train a Bahdanau/RNNsearch attention NMT model."
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default=preset_args.preset)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional custom TSV bitext path. If omitted, the PyTorch eng-fra tutorial data is downloaded.",
    )
    parser.add_argument(
        "--valid-data-path",
        type=Path,
        default=None,
        help="Optional validation TSV. When set, --data-path is used only for training.",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=None,
        help="Optional test TSV. When set, --data-path is used only for training.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bahdanau_nmt.pt")
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model weights initialize this run.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model and optimizer states resume this run.",
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("checkpoints/metadata.json")
    )
    parser.add_argument("--limit", type=optional_int, default=preset["limit"])
    parser.add_argument("--max-source-len", type=int, default=preset["max_source_len"])
    parser.add_argument("--max-target-len", type=int, default=preset["max_target_len"])
    parser.add_argument(
        "--eval-max-source-len",
        type=optional_int,
        default=None,
        help=(
            "Optional length filter for fixed validation/test files. By default, "
            "paper-style runs filter training only and evaluate full newstest sets."
        ),
    )
    parser.add_argument(
        "--eval-max-target-len",
        type=optional_int,
        default=None,
        help=(
            "Optional length filter for fixed validation/test files. By default, "
            "paper-style runs filter training only and evaluate full newstest sets."
        ),
    )
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=preset["max_vocab_size"])
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_CHOICES,
        default=preset["tokenizer"],
        help=(
            "legacy matches the original tutorial normalization; whitespace preserves "
            "pre-tokenized WMT text for paper-style experiments."
        ),
    )
    parser.add_argument(
        "--subword-type",
        choices=["none", "sentencepiece"],
        default="none",
        help="Records the subword encoding used by preprocessed TSV files.",
    )
    parser.add_argument(
        "--subword-model",
        type=Path,
        default=None,
        help="SentencePiece model used to create preprocessed TSV files.",
    )
    parser.add_argument("--batch-size", type=int, default=preset["batch_size"])
    parser.add_argument("--epochs", type=int, default=preset["epochs"])
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Offset used only for logging when continuing from an earlier run.",
    )
    parser.add_argument("--embedding-dim", type=int, default=preset["embedding_dim"])
    parser.add_argument(
        "--encoder-hidden-dim", type=int, default=preset["encoder_hidden_dim"]
    )
    parser.add_argument(
        "--decoder-hidden-dim", type=int, default=preset["decoder_hidden_dim"]
    )
    parser.add_argument("--dropout", type=float, default=preset["dropout"])
    parser.add_argument(
        "--optimizer", choices=["adam", "adadelta", "sgd"], default=preset["optimizer"]
    )
    parser.add_argument("--lr", type=float, default=preset["lr"])
    parser.add_argument("--adadelta-rho", type=float, default=0.95)
    parser.add_argument("--adadelta-eps", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--teacher-forcing-ratio", type=float, default=preset["teacher_forcing_ratio"]
    )
    parser.add_argument("--sort-k-batches", type=int, default=preset["sort_k_batches"])
    parser.add_argument(
        "--readout", choices=["maxout", "linear"], default=preset["readout"]
    )
    parser.add_argument("--maxout-dim", type=int, default=None)
    parser.add_argument(
        "--amp", action="store_true", help="Enable CUDA automatic mixed precision."
    )
    parser.add_argument(
        "--amp-dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--train-log", type=Path, default=Path("checkpoints/train_log.jsonl")
    )
    parser.add_argument(
        "--direction",
        choices=["fra-eng", "eng-fra"],
        default="fra-eng",
        help="Use the TSV columns as French->English or English->French for tutorial data.",
    )
    parser.add_argument("--source-col", type=int, default=None)
    parser.add_argument("--target-col", type=int, default=None)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument("--wandb-project", type=str, default="bahdanau-attention-nmt")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb-watch",
        choices=["none", "gradients", "all"],
        default="none",
        help="Optionally log model gradients/parameters. Use sparingly on large runs.",
    )
    parser.add_argument("--wandb-watch-freq", type=int, default=200)
    parser.add_argument(
        "--wandb-log-artifact",
        action="store_true",
        help="Upload/log the best checkpoint as a W&B model artifact at the end.",
    )
    return parser


def get_data_columns(args: argparse.Namespace) -> tuple[int, int]:
    if args.source_col is not None and args.target_col is not None:
        args.source_language = f"column_{args.source_col}"
        args.target_language = f"column_{args.target_col}"
        return args.source_col, args.target_col
    if args.direction == "fra-eng":
        args.source_language = "French"
        args.target_language = "English"
        return 1, 0
    args.source_language = "English"
    args.target_language = "French"
    return 0, 1


def build_vocabularies(
    args: argparse.Namespace,
    train_examples: list[tuple[list[str], list[str]]],
) -> tuple[Vocabulary, Vocabulary]:
    if args.subword_type == "sentencepiece":
        if args.subword_model is None:
            raise ValueError("--subword-model is required for sentencepiece training.")
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "Install sentencepiece first: `python -m pip install sentencepiece`."
            ) from exc
        processor = spm.SentencePieceProcessor(model_file=str(args.subword_model))
        pieces = [processor.id_to_piece(idx) for idx in range(processor.vocab_size())]
        vocab = Vocabulary.from_tokens(pieces)
        return vocab, vocab

    source_vocab = Vocabulary.build(
        [source for source, _ in train_examples],
        min_freq=args.min_freq,
        max_size=args.max_vocab_size,
    )
    target_vocab = Vocabulary.build(
        [target for _, target in train_examples],
        min_freq=args.min_freq,
        max_size=args.max_vocab_size,
    )
    return source_vocab, target_vocab


def build_dataloaders(
    args: argparse.Namespace,
    pairs_path: Path,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Vocabulary,
    Vocabulary,
    int,
    int,
    int,
    int,
]:
    source_col, target_col = get_data_columns(args)
    train_examples = read_parallel_tsv(
        pairs_path,
        source_col=source_col,
        target_col=target_col,
        limit=args.limit,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        tokenizer=args.tokenizer,
    )
    if args.valid_data_path is not None or args.test_data_path is not None:
        if args.valid_data_path is None or args.test_data_path is None:
            raise ValueError(
                "Set both --valid-data-path and --test-data-path, or omit both."
            )
        valid_examples = read_parallel_tsv(
            args.valid_data_path,
            source_col=source_col,
            target_col=target_col,
            limit=None,
            max_source_len=args.eval_max_source_len,
            max_target_len=args.eval_max_target_len,
            tokenizer=args.tokenizer,
        )
        test_examples = read_parallel_tsv(
            args.test_data_path,
            source_col=source_col,
            target_col=target_col,
            limit=None,
            max_source_len=args.eval_max_source_len,
            max_target_len=args.eval_max_target_len,
            tokenizer=args.tokenizer,
        )
        examples = train_examples + valid_examples + test_examples
    else:
        examples = train_examples
        train_examples, valid_examples, test_examples = split_examples(
            examples, seed=args.seed
        )

    if min(len(train_examples), len(valid_examples), len(test_examples)) < 1:
        raise ValueError(
            "Too few examples after filtering. Relax length or limit settings."
        )

    source_vocab, target_vocab = build_vocabularies(args, train_examples)

    train_dataset = ParallelTextDataset(train_examples, source_vocab, target_vocab)
    valid_dataset = ParallelTextDataset(valid_examples, source_vocab, target_vocab)
    test_dataset = ParallelTextDataset(test_examples, source_vocab, target_vocab)
    collate_fn = make_collate_fn(source_vocab.pad_idx, target_vocab.pad_idx)
    train_sampler = BucketBatchSampler(
        train_examples,
        batch_size=args.batch_size,
        sort_k_batches=args.sort_k_batches,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    return (
        train_loader,
        valid_loader,
        test_loader,
        source_vocab,
        target_vocab,
        len(examples),
        len(train_dataset),
        len(valid_dataset),
        len(test_dataset),
    )


def build_run_config(
    args: argparse.Namespace,
    train_size: int,
    valid_size: int,
    test_size: int,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    model: nn.Module,
    use_amp: bool,
) -> dict[str, object]:
    param_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_param_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run_config = serialize_namespace(args)
    run_config.update(
        {
            "paper": "Neural Machine Translation by Jointly Learning to Align and Translate",
            "implementation": "PyTorch RNNsearch-style Bahdanau attention",
            "implementation/encoder": "bidirectional GRU annotations",
            "implementation/attention": "additive Bahdanau attention",
            "implementation/decoder": "conditional GRU with attended context in gates",
            "implementation/readout": args.readout,
            "train_examples": train_size,
            "valid_examples": valid_size,
            "test_examples": test_size,
            "source_vocab_size": len(source_vocab),
            "target_vocab_size": len(target_vocab),
            "parameters": param_count,
            "trainable_parameters": trainable_param_count,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "amp_enabled": use_amp,
            "amp_dtype": args.amp_dtype,
        }
    )
    return run_config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    pairs_path = args.data_path or download_tutorial_data(args.data_dir)
    (
        train_loader,
        valid_loader,
        test_loader,
        source_vocab,
        target_vocab,
        total_size,
        train_size,
        valid_size,
        test_size,
    ) = build_dataloaders(args, pairs_path)

    model = build_model(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        source_pad_idx=source_vocab.pad_idx,
        target_pad_idx=target_vocab.pad_idx,
        device=device,
        embedding_dim=args.embedding_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        dropout=args.dropout,
        readout=args.readout,
        maxout_dim=args.maxout_dim,
    )
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError("Use either --init-checkpoint or --resume-checkpoint, not both.")

    initial_best_valid_loss = float("inf")
    resume_start_epoch = args.epoch_offset
    pending_optimizer_state = None
    checkpoint_to_load = args.resume_checkpoint or args.init_checkpoint
    if checkpoint_to_load is not None:
        checkpoint = torch.load(
            checkpoint_to_load,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        initial_best_valid_loss = float(
            checkpoint.get("best_valid_loss", initial_best_valid_loss)
        )
        if args.resume_checkpoint is not None:
            pending_optimizer_state = checkpoint.get("optimizer_state_dict")
            if args.epoch_offset == 0 and checkpoint.get("epoch") is not None:
                resume_start_epoch = int(checkpoint["epoch"])
        if not args.checkpoint.exists():
            save_checkpoint(
                args.checkpoint,
                model,
                source_vocab,
                target_vocab,
                args,
                initial_best_valid_loss,
                epoch=resume_start_epoch,
            )
        action = "Resumed" if args.resume_checkpoint is not None else "Initialized"
        print(f"{action} model weights from {checkpoint_to_load}")
    optimizer = build_optimizer(args, model)
    if pending_optimizer_state is not None:
        optimizer.load_state_dict(pending_optimizer_state)
        print(f"Loaded optimizer state from {args.resume_checkpoint}")
    criterion = nn.CrossEntropyLoss(
        ignore_index=target_vocab.pad_idx,
        reduction="sum",
    )
    amp_dtype = get_amp_dtype(args.amp_dtype)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    run_config = build_run_config(
        args,
        train_size,
        valid_size,
        test_size,
        source_vocab,
        target_vocab,
        model,
        use_amp,
    )
    wandb_run = init_wandb(args, run_config)
    maybe_watch_model(args, wandb_run, model)

    print(
        f"Loaded {total_size} examples "
        f"({train_size} train / {valid_size} valid / {test_size} test)"
    )
    print(f"Vocab sizes: source={len(source_vocab)}, target={len(target_vocab)}")
    print(
        f"Parameters: {run_config['parameters']:,} "
        f"({run_config['trainable_parameters']:,} trainable)"
    )
    print(
        f"Preset: {args.preset} | Optimizer: {args.optimizer} | sort_k_batches={args.sort_k_batches}"
    )
    print(f"AMP: {use_amp} ({args.amp_dtype})")
    print(f"Device: {device}")

    best_valid_loss = initial_best_valid_loss
    args.train_log.parent.mkdir(parents=True, exist_ok=True)
    args.train_log.write_text("", encoding="utf-8")
    try:
        for local_epoch in range(1, args.epochs + 1):
            epoch = resume_start_epoch + local_epoch
            start_time = time.time()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                args.clip,
                args.teacher_forcing_ratio,
                device,
                amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            valid_loss = evaluate(
                model, valid_loader, criterion, device, amp=use_amp, amp_dtype=amp_dtype
            )
            end_time = time.time()
            mins, secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(
                    args.checkpoint,
                    model,
                    source_vocab,
                    target_vocab,
                    args,
                    best_valid_loss,
                    optimizer=optimizer,
                    epoch=epoch,
                )

            train_ppl = math.exp(train_loss)
            valid_ppl = math.exp(valid_loss)
            epoch_seconds = int(end_time - start_time)
            print(
                f"Epoch {epoch:02d} | Time {mins}m {secs}s | "
                f"Train Loss {train_loss:.3f} | Train PPL {train_ppl:.2f} | "
                f"Valid Loss {valid_loss:.3f} | Valid PPL {valid_ppl:.2f}"
            )
            log_item = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_ppl": train_ppl,
                "valid_ppl": valid_ppl,
                "seconds": epoch_seconds,
                "best_valid_loss": best_valid_loss,
            }
            with args.train_log.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_item, ensure_ascii=False) + "\n")
            log_wandb(
                wandb_run,
                {
                    "epoch": epoch,
                    "loss/train": train_loss,
                    "loss/valid": valid_loss,
                    "ppl/train": train_ppl,
                    "ppl/valid": valid_ppl,
                    "time/epoch_seconds": epoch_seconds,
                    "best/valid_loss": best_valid_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
    except Exception:
        finish_wandb(wandb_run)
        raise

    if args.checkpoint.exists():
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(
        model, test_loader, criterion, device, amp=use_amp, amp_dtype=amp_dtype
    )
    test_ppl = math.exp(test_loss)
    print(f"Test Loss {test_loss:.3f} | Test PPL {test_ppl:.2f}")
    log_wandb(
        wandb_run,
        {
            "loss/test": test_loss,
            "ppl/test": test_ppl,
            "best/valid_loss_final": best_valid_loss,
        },
        step=resume_start_epoch + args.epochs,
    )
    write_metadata(
        args.metadata,
        source_vocab,
        target_vocab,
        train_size,
        valid_size,
        test_size,
        args,
    )
    print(f"Best checkpoint saved to {args.checkpoint}")
    print(f"Metadata saved to {args.metadata}")
    if args.wandb_log_artifact:
        log_checkpoint_artifact(wandb_run, args.checkpoint, args.metadata)
    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
