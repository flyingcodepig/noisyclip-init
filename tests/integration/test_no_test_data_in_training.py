"""Compliance tests preventing test manifests from entering training."""

from __future__ import annotations

import pytest
import torch
from test_two_batch_train import tiny_batches, tiny_components, tiny_records

from noisyclip.data.records import Batch
from noisyclip.engine.trainer import Trainer, TrainingPreflightError


def test_test_manifest_in_training_records_fails_preflight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A test split record cannot enter train_records."""

    config, components = tiny_components(tmp_path)
    components.train_records = tiny_records(split="test")
    components.train_loader = tiny_batches(tiny_records())
    with pytest.raises(TrainingPreflightError, match="Test/unlabeled"):
        Trainer(config=config, components=components, device="cpu").preflight()


def test_unlabeled_test_batch_in_training_path_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A batch without labels is rejected before loss/backward."""

    config, components = tiny_components(tmp_path)
    batch = components.train_loader[0]  # type: ignore[index]
    components.train_loader = [
        Batch(
            sample_ids=batch.sample_ids,
            image_weak=batch.image_weak,
            image_strong=None,
            targets=None,
            class_ids=None,
        )
    ]
    with pytest.raises(Exception, match="Training batch cannot be unlabeled"):
        Trainer(config=config, components=components, device="cpu").fit()


def test_training_tests_use_only_synthetic_tensors() -> None:
    """The integration fixtures are artificial tensors, not real data paths."""

    batch = tiny_batches(tiny_records())[0]
    assert isinstance(batch.image_weak, torch.Tensor)
    assert batch.image_weak.shape == (3, 3, 224, 224)
