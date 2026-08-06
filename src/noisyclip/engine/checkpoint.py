"""Atomic training checkpoint save and restore with RNG and ELR state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from noisyclip.engine.seed import capture_rng_state, restore_rng_state
from noisyclip.utils.atomic import atomic_save_with_writer

CHECKPOINT_FORMAT_VERSION = 1


class StatefulLoss(Protocol):
    """Protocol for checkpointable loss state such as ELR history."""

    def state_dict(self) -> Mapping[str, object]:
        """Return a checkpointable mapping."""

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a mapping produced by `state_dict`."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Minimal checkpoint identity and progress metadata."""

    epoch: int
    global_step: int
    sample_state_epoch: int | None
    config_digest: str | None = None
    data_digest: str | None = None
    best_metric: float | None = None
    early_best_metric: float | None = None
    epochs_without_improvement: int = 0

    def __post_init__(self) -> None:
        """Validate non-negative epoch and step fields."""

        if self.epoch < 0:
            raise ValueError("checkpoint epoch must be non-negative.")
        if self.global_step < 0:
            raise ValueError("checkpoint global_step must be non-negative.")
        if self.sample_state_epoch is not None and self.sample_state_epoch < 0:
            raise ValueError("sample_state_epoch must be non-negative or None.")
        if self.epochs_without_improvement < 0:
            raise ValueError("epochs_without_improvement must be non-negative.")


def save_checkpoint(
    destination: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler_state: Mapping[str, object] | None,
    metadata: CheckpointMetadata,
    loss_state: Mapping[str, object] | None = None,
    minimum_free_bytes: int = 0,
) -> Path:
    """Atomically save model, optimizer, scheduler, scaler, RNG, and loss state.

    Args:
        destination: Final `.pt` checkpoint path.
        model: Trainable model.
        optimizer: Optimizer whose state is saved.
        scheduler: Optional scheduler with `state_dict`.
        scaler_state: Optional precision/GradScaler state.
        metadata: Epoch, global step, and sample-state reference.
        loss_state: Optional mapping such as `{"elr": ELRLoss.state_dict()}`.
        minimum_free_bytes: Disk-space safety margin checked before writing.

    Returns:
        Final checkpoint path.

    Raises:
        OSError: If disk space is insufficient or atomic writing fails. The old
            checkpoint remains untouched on failure.
    """

    payload: dict[str, object] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": dict(scaler_state or {}),
        "epoch": metadata.epoch,
        "global_step": metadata.global_step,
        "sample_state_epoch": metadata.sample_state_epoch,
        "config_digest": metadata.config_digest,
        "data_digest": metadata.data_digest,
        "best_metric": metadata.best_metric,
        "early_best_metric": metadata.early_best_metric,
        "epochs_without_improvement": metadata.epochs_without_improvement,
        "rng": capture_rng_state().state_dict(),
        "loss_state": dict(loss_state or {}),
    }

    def _writer(tmp_path: Path) -> None:
        torch.save(payload, tmp_path)

    final_path, _ = atomic_save_with_writer(
        destination,
        _writer,
        overwrite=True,
        minimum_free_bytes=minimum_free_bytes,
    )
    return final_path


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    precision_manager: Any | None = None,
    loss_objects: Mapping[str, StatefulLoss] | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    """Load a checkpoint and restore all supplied mutable training objects.

    Args:
        checkpoint_path: Path produced by `save_checkpoint`.
        model: Model receiving checkpointed parameters.
        optimizer: Optional optimizer receiving state.
        scheduler: Optional scheduler receiving state when present.
        precision_manager: Optional object exposing `load_state_dict`.
        loss_objects: Optional mapping whose values restore matching loss states.
        map_location: Torch load location.

    Returns:
        `CheckpointMetadata` from the checkpoint.

    Raises:
        ValueError: If checkpoint format or requested loss state is invalid.
        OSError: If the checkpoint cannot be read.
    """

    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')!r}.")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if precision_manager is not None:
        precision_manager.load_state_dict(payload.get("scaler", {}))
    restore_rng_state(payload["rng"])
    raw_loss_state = payload.get("loss_state", {})
    if loss_objects:
        if not isinstance(raw_loss_state, Mapping):
            raise ValueError("checkpoint loss_state must be a mapping.")
        for name, loss_object in loss_objects.items():
            state = raw_loss_state.get(name)
            if state is not None:
                if not isinstance(state, Mapping):
                    raise ValueError(f"checkpoint loss_state[{name!r}] must be a mapping.")
                loss_object.load_state_dict(state)
    sample_epoch_raw = payload.get("sample_state_epoch")
    sample_state_epoch = None if sample_epoch_raw is None else int(sample_epoch_raw)
    return CheckpointMetadata(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        sample_state_epoch=sample_state_epoch,
        config_digest=(
            None if payload.get("config_digest") is None else str(payload.get("config_digest"))
        ),
        data_digest=None if payload.get("data_digest") is None else str(payload.get("data_digest")),
        best_metric=_optional_float(payload.get("best_metric")),
        early_best_metric=_optional_float(payload.get("early_best_metric")),
        epochs_without_improvement=int(payload.get("epochs_without_improvement", 0)),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Checkpoint metric state must be numeric or None.")
    return float(value)
