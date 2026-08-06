"""Unit tests for checkpoint atomicity and state restoration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from noisyclip.engine.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from noisyclip.engine.seed import set_seed


def test_checkpoint_restores_model_optimizer_rng_and_loss_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Checkpoint load restores mutable training state and RNG streams."""

    set_seed(10)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_state = {"elr": {"targets": {"a": torch.tensor([0.2, 0.8])}, "num_classes": 2}}
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={},
        metadata=CheckpointMetadata(epoch=1, global_step=3, sample_state_epoch=1),
        loss_state=loss_state,
    )
    expected_next = torch.rand(2)
    with torch.no_grad():
        model.weight.add_(1.0)
    restored_loss = _LossState()
    metadata = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        loss_objects={"elr": restored_loss},
    )
    assert metadata.global_step == 3
    assert restored_loss.loaded["num_classes"] == 2
    assert torch.equal(torch.rand(2), expected_next)


def test_checkpoint_space_failure_preserves_old_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Disk-space preflight fails before replacing the existing checkpoint."""

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "last.pt"
    path.write_bytes(b"old")
    with pytest.raises(OSError, match="Insufficient free disk space"):
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler_state={},
            metadata=CheckpointMetadata(epoch=0, global_step=0, sample_state_epoch=None),
            minimum_free_bytes=10**30,
        )
    assert path.read_bytes() == b"old"


class _LossState:
    """Tiny checkpointable loss-state fake."""

    def __init__(self) -> None:
        self.loaded: dict[str, object] = {}

    def state_dict(self) -> dict[str, object]:
        """Return unused fake state."""

        return {}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Record loaded fake state."""

        self.loaded = dict(state_dict)
