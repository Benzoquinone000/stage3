from __future__ import annotations

import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "adadelta":
        return optim.Adadelta(
            model.parameters(),
            lr=args.lr,
            rho=args.adadelta_rho,
            eps=args.adadelta_eps,
        )
    if args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def get_amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    teacher_forcing_ratio: float,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    epoch_loss_sum = 0.0
    epoch_tokens = 0
    use_amp = amp and device.type == "cuda"
    for batch in dataloader:
        source = batch["source"].to(device)
        source_lengths = batch["source_lengths"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output, _ = model(source, source_lengths, target, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)
            loss_sum = criterion(output, target_flat)
            token_count = target_flat.ne(criterion.ignore_index).sum().clamp_min(1)
            loss = loss_sum / token_count

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        epoch_loss_sum += float(loss_sum.item())
        epoch_tokens += int(token_count.item())

    return epoch_loss_sum / max(1, epoch_tokens)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    model.eval()
    epoch_loss_sum = 0.0
    epoch_tokens = 0
    use_amp = amp and device.type == "cuda"
    for batch in dataloader:
        source = batch["source"].to(device)
        source_lengths = batch["source_lengths"].to(device)
        target = batch["target"].to(device)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output, _ = model(source, source_lengths, target, teacher_forcing_ratio=1.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)
            loss_sum = criterion(output, target_flat)
            token_count = target_flat.ne(criterion.ignore_index).sum().clamp_min(1)
        epoch_loss_sum += float(loss_sum.item())
        epoch_tokens += int(token_count.item())

    return epoch_loss_sum / max(1, epoch_tokens)
