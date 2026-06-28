from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from nmt_attention.data import (
    EOS_TOKEN,
    SOS_TOKEN,
    TOKENIZER_CHOICES,
    Vocabulary,
    detokenize_tokens,
    pad_sequences,
    tokenize,
)
from nmt_attention.model import build_model


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_vocab = Vocabulary.from_dict(checkpoint["source_vocab"])
    target_vocab = Vocabulary.from_dict(checkpoint["target_vocab"])
    train_args = checkpoint["args"]

    model = build_model(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        source_pad_idx=source_vocab.pad_idx,
        target_pad_idx=target_vocab.pad_idx,
        device=device,
        embedding_dim=int(train_args["embedding_dim"]),
        encoder_hidden_dim=int(train_args["encoder_hidden_dim"]),
        decoder_hidden_dim=int(train_args["decoder_hidden_dim"]),
        dropout=float(train_args["dropout"]),
        readout=train_args.get("readout", "maxout"),
        maxout_dim=train_args.get("maxout_dim"),
        init="default",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.train_args = train_args
    return model, source_vocab, target_vocab


def get_sentencepiece_processor(train_args: dict[str, object]):
    if train_args.get("subword_type") != "sentencepiece":
        return None
    model_path = train_args.get("subword_model")
    if not model_path:
        raise ValueError("Checkpoint uses sentencepiece but has no subword_model path.")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "Install sentencepiece first: `python -m pip install sentencepiece`."
        ) from exc
    return spm.SentencePieceProcessor(model_file=str(model_path))


def encode_source_sentence(
    sentence: str,
    train_args: dict[str, object],
    tokenizer: str,
) -> list[str]:
    processor = get_sentencepiece_processor(train_args)
    if processor is not None:
        return processor.encode(sentence, out_type=str)
    return tokenize(sentence, mode=tokenizer)


def decode_target_tokens(tokens: list[str], train_args: dict[str, object]) -> str:
    processor = get_sentencepiece_processor(train_args)
    if processor is not None:
        return processor.decode(tokens)
    return detokenize_tokens(tokens, tokenizer=str(train_args.get("tokenizer", "legacy")))


def save_attention_csv(
    path: Path,
    source_tokens: list[str],
    target_tokens: list[str],
    attentions: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target_token", *source_tokens])
        for token, weights in zip(target_tokens, attentions.tolist()):
            writer.writerow(
                [token, *[f"{weight:.6f}" for weight in weights[: len(source_tokens)]]]
            )


def maybe_save_attention_plot(
    path: Path | None,
    source_tokens: list[str],
    target_tokens: list[str],
    attentions: torch.Tensor,
) -> None:
    if path is None or attentions.numel() == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or omit --plot-out.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6, len(source_tokens) * 0.6), max(4, len(target_tokens) * 0.45))
    )
    image = ax.imshow(
        attentions[:, : len(source_tokens)].numpy(), cmap="viridis", aspect="auto"
    )
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticklabels(target_tokens)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated target tokens")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate a sentence with a trained attention model."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/bahdanau_nmt.pt")
    )
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--min-len", type=int, default=0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument(
        "--suppress-unk",
        action="store_true",
        help="Prevent beam search from emitting the vocabulary <unk> token.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=TOKENIZER_CHOICES,
        default=None,
        help="Defaults to the tokenizer stored in the checkpoint.",
    )
    parser.add_argument("--attention-out", type=Path, default=None)
    parser.add_argument("--plot-out", type=Path, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, source_vocab, target_vocab = load_model(args.checkpoint, device)

    train_args = getattr(model, "train_args", {})
    tokenizer = args.tokenizer or train_args.get("tokenizer", "legacy")
    source_tokens = encode_source_sentence(args.sentence, train_args, tokenizer)
    source_ids = source_vocab.encode(source_tokens)
    source, source_lengths = pad_sequences([source_ids], source_vocab.pad_idx)
    source = source.to(device)
    source_lengths = source_lengths.to(device)

    if args.beam_size > 1:
        generated_ids, attentions = model.beam_search(
            source,
            source_lengths,
            sos_idx=target_vocab.sos_idx,
            eos_idx=target_vocab.eos_idx,
            unk_idx=target_vocab.unk_idx if args.suppress_unk else None,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            min_len=args.min_len,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    else:
        generated_ids, attentions = model.translate(
            source,
            source_lengths,
            sos_idx=target_vocab.sos_idx,
            eos_idx=target_vocab.eos_idx,
            max_len=args.max_len,
        )
    target_tokens = target_vocab.decode(generated_ids)
    print("Source:", " ".join(source_tokens))
    print("Translation:", decode_target_tokens(target_tokens, train_args))

    source_attention_tokens = [SOS_TOKEN, *source_tokens, EOS_TOKEN]
    if args.attention_out:
        save_attention_csv(
            args.attention_out,
            source_attention_tokens,
            target_tokens,
            attentions,
        )
        print(f"Attention CSV saved to {args.attention_out}")
    maybe_save_attention_plot(
        args.plot_out,
        source_attention_tokens,
        target_tokens,
        attentions,
    )
    if args.plot_out:
        print(f"Attention plot saved to {args.plot_out}")


if __name__ == "__main__":
    main()
