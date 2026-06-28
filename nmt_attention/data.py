from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
TOKENIZER_CHOICES = ("legacy", "unicode", "whitespace")


def unicode_to_ascii(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def normalize_text(text: str) -> str:
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"([.!?,;:])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.!?,;:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"([.!?,;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, mode: str = "legacy") -> list[str]:
    if mode == "legacy":
        return normalize_text(text).split()
    if mode == "unicode":
        return normalize_unicode_text(text).split()
    if mode == "whitespace":
        return re.sub(r"\s+", " ", text.strip()).split()
    raise ValueError(
        f"Unsupported tokenizer mode: {mode}. Choose from {', '.join(TOKENIZER_CHOICES)}."
    )


def detokenize_sentencepiece(tokens: Iterable[str]) -> str:
    return "".join(tokens).replace("▁", " ").strip()


def detokenize_tokens(tokens: Iterable[str], tokenizer: str = "legacy") -> str:
    if tokenizer == "sentencepiece":
        return detokenize_sentencepiece(tokens)
    return " ".join(tokens)


@dataclass
class Vocabulary:
    token_to_idx: dict[str, int]
    idx_to_token: list[str]

    @classmethod
    def build(
        cls,
        tokenized_sentences: Iterable[list[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for sentence in tokenized_sentences:
            counter.update(sentence)

        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        words = [
            word
            for word, count in counter.most_common()
            if count >= min_freq and word not in specials
        ]
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]

        idx_to_token = specials + words
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "Vocabulary":
        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        idx_to_token = specials[:]
        seen = set(specials)
        for token in tokens:
            if token not in seen:
                idx_to_token.append(token)
                seen.add(token)
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token_to_idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token_to_idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, tokens: list[str], add_sos_eos: bool = True) -> list[int]:
        ids = [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        if add_sos_eos:
            return [self.sos_idx, *ids, self.eos_idx]
        return ids

    def decode(self, ids: Iterable[int], skip_specials: bool = True) -> list[str]:
        specials = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}
        tokens: list[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.idx_to_token):
                token = UNK_TOKEN
            else:
                token = self.idx_to_token[idx]
            if skip_specials and token in specials:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": self.idx_to_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Vocabulary":
        return cls(
            token_to_idx=dict(payload["token_to_idx"]),
            idx_to_token=list(payload["idx_to_token"]),
        )


def read_parallel_tsv(
    path: str | Path,
    source_col: int = 0,
    target_col: int = 1,
    limit: int | None = None,
    max_source_len: int | None = None,
    max_target_len: int | None = None,
    tokenizer: str = "legacy",
) -> list[tuple[list[str], list[str]]]:
    examples: list[tuple[list[str], list[str]]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) <= max(source_col, target_col):
                continue
            source_tokens = tokenize(row[source_col], mode=tokenizer)
            target_tokens = tokenize(row[target_col], mode=tokenizer)
            if not source_tokens or not target_tokens:
                continue
            if max_source_len is not None and len(source_tokens) > max_source_len:
                continue
            if max_target_len is not None and len(target_tokens) > max_target_len:
                continue
            examples.append((source_tokens, target_tokens))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def split_examples(
    examples: list[tuple[list[str], list[str]]],
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
) -> tuple[
    list[tuple[list[str], list[str]]],
    list[tuple[list[str], list[str]]],
    list[tuple[list[str], list[str]]],
]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    test_size = int(len(shuffled) * test_ratio)
    valid_size = int(len(shuffled) * valid_ratio)
    test = shuffled[:test_size]
    valid = shuffled[test_size : test_size + valid_size]
    train = shuffled[test_size + valid_size :]
    return train, valid, test


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        examples: list[tuple[list[str], list[str]]],
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        self.examples = examples
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        source_tokens, target_tokens = self.examples[index]
        return (
            self.source_vocab.encode(source_tokens),
            self.target_vocab.encode(target_tokens),
        )


class BucketBatchSampler(Sampler[list[int]]):
    """Approximate GroundHog's sort_k_batches padding-reduction trick."""

    def __init__(
        self,
        examples: list[tuple[list[str], list[str]]],
        batch_size: int,
        sort_k_batches: int = 20,
        shuffle: bool = True,
        seed: int = 13,
    ) -> None:
        self.examples = examples
        self.batch_size = batch_size
        self.sort_k_batches = max(1, sort_k_batches)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.examples)))
        if self.shuffle:
            rng.shuffle(indices)

        chunk_size = self.batch_size * self.sort_k_batches
        batches: list[list[int]] = []
        for start in range(0, len(indices), chunk_size):
            chunk = indices[start : start + chunk_size]
            chunk.sort(
                key=lambda idx: max(
                    len(self.examples[idx][0]),
                    len(self.examples[idx][1]),
                )
            )
            for batch_start in range(0, len(chunk), self.batch_size):
                batch = chunk[batch_start : batch_start + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.examples) + self.batch_size - 1) // self.batch_size


def pad_sequences(sequences: list[list[int]], pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for idx, sequence in enumerate(sequences):
        batch[idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch, lengths


def make_collate_fn(source_pad_idx: int, target_pad_idx: int):
    def collate(batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        source_sequences, target_sequences = zip(*batch)
        source, source_lengths = pad_sequences(list(source_sequences), source_pad_idx)
        target, target_lengths = pad_sequences(list(target_sequences), target_pad_idx)
        return {
            "source": source,
            "source_lengths": source_lengths,
            "target": target,
            "target_lengths": target_lengths,
        }

    return collate
