from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch import nn


def init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or omit `--wandb`."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )


def log_wandb(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(payload, step=step)


def maybe_watch_model(args: argparse.Namespace, run, model: nn.Module) -> None:
    if run is None or args.wandb_watch == "none":
        return
    import wandb

    wandb.watch(model, log=args.wandb_watch, log_freq=args.wandb_watch_freq)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def log_checkpoint_artifact(run, checkpoint_path: Path, metadata_path: Path) -> None:
    if run is None or not checkpoint_path.exists():
        return
    import wandb

    artifact = wandb.Artifact(
        name=f"{run.name or 'bahdanau-nmt'}-best-checkpoint",
        type="model",
        metadata={
            "checkpoint": str(checkpoint_path),
            "metadata": str(metadata_path),
        },
    )
    artifact.add_file(str(checkpoint_path))
    if metadata_path.exists():
        artifact.add_file(str(metadata_path))
    run.log_artifact(artifact)
