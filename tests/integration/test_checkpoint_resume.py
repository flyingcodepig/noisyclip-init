"""Checkpoint, ELR-state, SampleState, and RNG resume integration tests."""

from __future__ import annotations

import pytest
import torch
from test_two_batch_train import tiny_components
from torch import nn

from noisyclip.engine.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from noisyclip.engine.seed import set_seed
from noisyclip.losses.elr import ELRLoss
from noisyclip.noise.state import JsonSampleStateStore


def test_checkpoint_restores_elr_sample_state_and_rng_for_next_step(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Restored state reproduces the next loss and parameter update."""

    set_seed(123)
    model_a = nn.Linear(2, 2)
    model_b = nn.Linear(2, 2)
    model_b.load_state_dict(model_a.state_dict())
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
    elr = ELRLoss(enabled=True, start_epoch=0)
    state_store = JsonSampleStateStore(tmp_path / "sample_state", ["s0"])
    state_store.stage_epoch([_state("s0", epoch=0)], 0)
    state_store.commit_epoch(0)
    save_checkpoint(
        tmp_path / "last.pt",
        model=model_a,
        optimizer=optimizer_a,
        scheduler=None,
        scaler_state={},
        metadata=CheckpointMetadata(epoch=0, global_step=1, sample_state_epoch=0),
        loss_state={"elr": elr.state_dict()},
    )
    expected_random = torch.rand(1)

    restored_elr = ELRLoss(enabled=True, start_epoch=0)
    metadata = load_checkpoint(
        tmp_path / "last.pt",
        model=model_b,
        optimizer=optimizer_b,
        loss_objects={"elr": restored_elr},
    )
    assert metadata.sample_state_epoch == 0
    assert state_store.load(["s0"])[0].seen_count == 1
    assert torch.equal(torch.rand(1), expected_random)

    x = torch.tensor([[0.25, 0.75]])
    y = torch.tensor([1])
    loss_a = torch.nn.functional.cross_entropy(model_a(x), y)
    loss_b = torch.nn.functional.cross_entropy(model_b(x), y)
    loss_a.backward()
    loss_b.backward()
    optimizer_a.step()
    optimizer_b.step()
    assert torch.allclose(loss_a, loss_b)
    for left, right in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(left, right)


def test_checkpoint_save_failure_does_not_commit_sample_state(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """SampleState commit occurs only after checkpoint save succeeds."""

    from noisyclip.engine import trainer as trainer_module

    config, components = tiny_components(tmp_path)

    def fail_save(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr(trainer_module, "save_checkpoint", fail_save)
    with pytest.raises(trainer_module.TrainingFailedError):
        trainer_module.Trainer(config=config, components=components, device="cpu").fit()
    assert not (tmp_path / "run" / "sample_state" / "manifest.json").exists()
    assert (tmp_path / "run" / "FAILED").is_file()


def test_same_epoch_sample_state_cannot_be_overwritten(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The state store rejects staging an already committed epoch."""

    store = JsonSampleStateStore(tmp_path / "state", ["s0"])
    store.stage_epoch([_state("s0", epoch=0)], 0)
    store.commit_epoch(0)
    with pytest.raises(ValueError, match="latest committed epoch"):
        store.stage_epoch([_state("s0", epoch=0)], 0)


def _state(sample_id: str, *, epoch: int):
    from noisyclip.noise.state import SampleState

    return SampleState(
        sample_id=sample_id,
        seen_count=1,
        ema_loss=0.1,
        ema_probs=[0.5, 0.5],
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=1.0,
        prototype_margin=1.0,
        trust_score=1.0,
        supervised_weight=1.0,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=epoch,
    )
