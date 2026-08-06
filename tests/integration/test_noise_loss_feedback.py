"""Integration test for the loss-to-trust-to-next-loss feedback loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from noisyclip.config.schema import LossConfig
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.losses.composite import RobustCompositeLoss
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.partition import apply_partitions, partition_by_class
from noisyclip.noise.signals import update_prediction_history
from noisyclip.noise.state import JsonSampleStateStore, SampleState
from noisyclip.noise.trust import ClasswiseTrustAggregator


def _state(sample_id: str) -> SampleState:
    return SampleState(
        sample_id=sample_id,
        seen_count=0,
        ema_loss=0.0,
        ema_probs=None,
        prediction_stability=0.0,
        augmentation_agreement=0.0,
        prototype_similarity=0.0,
        prototype_margin=0.0,
        trust_score=0.5,
        supervised_weight=1.0,
        partition="uncertain",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )


def test_per_sample_loss_updates_trust_state_and_next_epoch_weights(tmp_path: Path) -> None:
    """Agent C and D interfaces form a persistent sample-ID keyed feedback loop."""

    sample_ids = ["a", "b", "c", "d"]
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    batch = Batch(
        sample_ids=sample_ids,
        image_weak=torch.zeros((4, 3, 224, 224)),
        image_strong=None,
        targets=targets,
        class_ids=["0000", "0000", "0001", "0001"],
    )
    logits = torch.tensor(
        [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]],
        requires_grad=True,
    )
    output = ModelOutput(
        logits=logits,
        embedding=F.normalize(torch.ones((4, 2)), dim=1),
        temperature=None,
    )
    states = [_state(sample_id) for sample_id in sample_ids]
    loss = RobustCompositeLoss(LossConfig())

    first = loss(batch, output, None, None, states, epoch=0)
    assert first.per_sample_supervised is not None
    history = update_prediction_history(states, logits, epoch=0, momentum=0.9)
    records = [
        SampleRecord(
            sample_id=sample_id,
            relative_path=f"{target:04d}/{sample_id}.jpg",
            split="train",
            class_id=f"{target:04d}",
            target=target,
            file_sha256=None,
            width=224,
            height=224,
            readable=True,
        )
        for sample_id, target in zip(sample_ids, targets.tolist(), strict=True)
    ]
    trusted = ClasswiseTrustAggregator(
        {"ema_loss": 1.0},
        supervised_weight_min=0.1,
        supervised_weight_max=1.0,
    ).update_epoch(
        records,
        {"ema_loss": first.per_sample_supervised},
        history,
        epoch=0,
    )
    partitions = partition_by_class(
        sample_ids,
        targets,
        torch.tensor([state.trust_score for state in trusted]),
        trusted_quantile=0.65,
        uncertain_quantile=0.90,
    )
    trusted = apply_partitions(trusted, partitions, epoch=0)

    assert trusted[0].ema_loss == pytest.approx(float(first.per_sample_supervised[0]))
    assert trusted[1].ema_loss == pytest.approx(float(first.per_sample_supervised[1]))
    assert trusted[0].supervised_weight > trusted[1].supervised_weight
    assert trusted[2].supervised_weight > trusted[3].supervised_weight
    assert {state.partition for state in trusted} == {"trusted", "suspicious"}

    store = JsonSampleStateStore(tmp_path / "state", expected_sample_ids=sample_ids)
    store.stage_epoch(trusted, 0)
    store.commit_epoch(0)
    restored = store.load(["d", "b", "a", "c"])
    assert [state.sample_id for state in restored] == ["d", "b", "a", "c"]

    second = loss(batch, output, None, None, trusted, epoch=1)
    assert torch.isfinite(second.total)
    assert second.components["loss/effective_supervised_weight"].item() < 4.0
