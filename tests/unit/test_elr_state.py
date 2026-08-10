"""Unit tests for ELR historical state behavior."""

from __future__ import annotations

import pytest
import torch

from noisyclip.data.records import Batch
from noisyclip.losses.elr import ELRLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState


def _state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=1,
        ema_loss=0.0,
        ema_probs=None,
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=1.0,
        prototype_margin=0.5,
        trust_score=1.0,
        supervised_weight=1.0,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )


def _batch(sample_ids: list[str]) -> Batch:
    images = torch.zeros((len(sample_ids), 3, 224, 224), dtype=torch.float32)
    return Batch(
        sample_ids=sample_ids,
        image_weak=images,
        image_strong=None,
        targets=torch.zeros((len(sample_ids),), dtype=torch.int64),
        class_ids=None,
    )


def _output(logits: torch.Tensor) -> ModelOutput:
    return ModelOutput(
        logits=logits,
        embedding=torch.zeros((logits.shape[0], 2), dtype=logits.dtype),
        temperature=None,
    )


def _history(state: dict[str, object]) -> dict[str, torch.Tensor]:
    sample_ids = state["sample_ids"]
    target_tensor = state["target_tensor"]
    assert isinstance(sample_ids, list)
    assert isinstance(target_tensor, torch.Tensor)
    return {
        sample_id: row
        for sample_id, row in zip(sample_ids, target_tensor, strict=True)
    }


def test_elr_history_is_keyed_by_sample_id_under_reordering() -> None:
    """Reordered batches produce the same ELR scalar for the same ID/logit pairs."""

    ids = ["a", "b", "c"]
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]])
    elr = ELRLoss(target_momentum=0.5, start_epoch=0)
    first = elr(_batch(ids), _output(logits), [_state(item) for item in ids], epoch=0)

    order = [2, 0, 1]
    reordered_ids = [ids[index] for index in order]
    reordered_logits = logits[order].clone()
    second = elr(
        _batch(reordered_ids),
        _output(reordered_logits),
        [_state(item) for item in reordered_ids],
        epoch=1,
    )

    fresh = ELRLoss(target_momentum=0.5, start_epoch=0)
    fresh.load_state_dict(elr.state_dict())
    restored = fresh(
        _batch(reordered_ids),
        _output(reordered_logits),
        [_state(item) for item in reordered_ids],
        epoch=2,
    )

    assert torch.isfinite(first)
    assert torch.allclose(second, restored)


def test_elr_warmup_and_disabled_do_not_create_history() -> None:
    """Warmup and disabled modes return zero without storing per-position state."""

    ids = ["a", "b"]
    logits = torch.zeros((2, 2), dtype=torch.float32)

    warmup = ELRLoss(start_epoch=5)
    assert warmup(_batch(ids), _output(logits), [_state("a"), _state("b")], epoch=0).item() == 0.0
    assert _history(warmup.state_dict()) == {}

    disabled = ELRLoss(enabled=False)
    assert (
        disabled(_batch(ids), _output(logits), [_state("a"), _state("b")], epoch=10).item() == 0.0
    )
    assert _history(disabled.state_dict()) == {}


def test_elr_teacher_history_is_detached_from_student_logits() -> None:
    """The history target is detached while current probabilities keep gradients."""

    ids = ["a", "b"]
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    elr = ELRLoss(start_epoch=0)

    value = elr(_batch(ids), _output(logits), [_state("a"), _state("b")], epoch=0)
    value.backward()

    assert logits.grad is not None
    for item in _history(elr.state_dict()).values():
        assert not item.requires_grad


def test_elr_uses_sample_weights_for_loss_and_history() -> None:
    """A zero-weight sample has no gradient and does not create ELR history."""

    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
    states = [_state("a"), _state("b")]
    states[1].supervised_weight = 0.0
    elr = ELRLoss(start_epoch=0)

    value = elr(_batch(["a", "b"]), _output(logits), states, epoch=0)
    value.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0]).item() > 0
    assert torch.count_nonzero(logits.grad[1]).item() == 0
    assert set(_history(elr.state_dict())) == {"a"}


def test_elr_state_dict_uses_one_compact_history_tensor() -> None:
    """Checkpoint state avoids one serialized tensor object per sample."""

    elr = ELRLoss(start_epoch=0)
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    elr(_batch(["b", "a"]), _output(logits), [_state("b"), _state("a")], epoch=0)

    state = elr.state_dict()

    assert state["format_version"] == 2
    assert state["sample_ids"] == ["a", "b"]
    target_tensor = state["target_tensor"]
    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == (2, 2)
    assert "targets" not in state


def test_elr_loads_legacy_mapping_checkpoint() -> None:
    """Existing mapping-based checkpoints remain readable."""

    legacy_target = torch.tensor([0.75, 0.25])
    legacy: dict[str, object] = {
        "target_momentum": 0.7,
        "start_epoch": 0,
        "enabled": True,
        "epsilon": 1e-6,
        "num_classes": 2,
        "targets": {"a": legacy_target},
    }
    elr = ELRLoss()

    elr.load_state_dict(legacy)

    assert torch.equal(_history(elr.state_dict())["a"], legacy_target)


def test_preallocated_elr_history_stays_compact_and_restores() -> None:
    """Assembled ELR uses one device table while preserving ID-keyed state."""

    ids = ["a", "b", "c"]
    elr = ELRLoss(start_epoch=0, sample_ids=ids, history_device="cpu")
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    elr(_batch(["c", "a"]), _output(logits), [_state("c"), _state("a")], epoch=0)

    state = elr.state_dict()
    assert state["sample_ids"] == ["a", "c"]
    target_tensor = state["target_tensor"]
    assert isinstance(target_tensor, torch.Tensor)
    assert target_tensor.shape == (2, 2)

    restored = ELRLoss(start_epoch=0, sample_ids=ids, history_device="cpu")
    restored.load_state_dict(state)
    assert _history(restored.state_dict()).keys() == _history(state).keys()
    assert torch.equal(
        restored.state_dict()["target_tensor"],
        state["target_tensor"],
    )


def test_preallocated_and_mapping_elr_paths_have_matching_updates() -> None:
    """The no-sync table preserves the established ELR mathematics."""

    ids = ["a", "b", "c"]
    mapping = ELRLoss(target_momentum=0.5, start_epoch=0)
    table = ELRLoss(
        target_momentum=0.5,
        start_epoch=0,
        sample_ids=ids,
        history_device="cpu",
    )
    first_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]])
    second_order = [2, 0, 1]
    second_ids = [ids[index] for index in second_order]
    second_logits = torch.tensor([[2.0, 1.0], [2.5, 0.5], [0.5, 2.5]])
    states = [_state(sample_id) for sample_id in ids]
    states[1].supervised_weight = 0.4

    first_mapping = mapping(_batch(ids), _output(first_logits), states, epoch=0)
    first_table = table(_batch(ids), _output(first_logits), states, epoch=0)
    second_states = [states[index] for index in second_order]
    second_mapping = mapping(
        _batch(second_ids), _output(second_logits), second_states, epoch=1
    )
    second_table = table(
        _batch(second_ids), _output(second_logits), second_states, epoch=1
    )

    assert torch.allclose(first_mapping, first_table, atol=1e-7, rtol=0.0)
    assert torch.allclose(second_mapping, second_table, atol=1e-7, rtol=0.0)
    mapping_history = _history(mapping.state_dict())
    table_history = _history(table.state_dict())
    assert mapping_history.keys() == table_history.keys()
    for sample_id in mapping_history:
        assert torch.allclose(
            mapping_history[sample_id], table_history[sample_id], atol=1e-7, rtol=0.0
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("target_momentum", 1.0, "target_momentum"),
        ("start_epoch", -1, "start_epoch"),
        ("epsilon", 0.0, "epsilon"),
        ("num_classes", 0, "num_classes"),
    ],
)
def test_elr_restore_rejects_invalid_checkpoint_hyperparameters(
    field: str,
    value: object,
    message: str,
) -> None:
    """Corrupt checkpoint metadata cannot silently alter ELR behavior."""

    state = ELRLoss().state_dict()
    state[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        ELRLoss().load_state_dict(state)
