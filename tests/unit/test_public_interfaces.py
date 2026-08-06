"""Unit tests for F01 public data and context interfaces."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import pytest
import torch
from pydantic import ValidationError

from noisyclip.config.loader import load_config_from_mapping
from noisyclip.data.records import Batch, SampleRecord
from noisyclip.engine.context import EpochContext, RunContext
from noisyclip.losses.outputs import LossOutput
from noisyclip.models.outputs import ModelOutput
from noisyclip.noise.state import SampleState

REQUIRED_CONFIG = {
    "experiment": {},
    "paths": {},
    "data": {},
    "model": {},
    "noise": {},
    "loss": {},
    "trainer": {},
    "evaluation": {},
    "tracking": {},
    "submission": {},
}


def test_sample_record_contract_fields_and_freezing() -> None:
    """SampleRecord exposes exactly the architecture fields and is immutable."""

    expected = [
        "sample_id",
        "relative_path",
        "split",
        "class_id",
        "target",
        "file_sha256",
        "width",
        "height",
        "readable",
    ]
    assert is_dataclass(SampleRecord)
    assert [field.name for field in fields(SampleRecord)] == expected

    record = SampleRecord(
        sample_id="abc",
        relative_path="0001/image.jpg",
        split="train",
        class_id="0001",
        target=0,
        file_sha256=None,
        width=224,
        height=224,
        readable=True,
    )
    with pytest.raises(FrozenInstanceError):
        record.split = "test"


def test_batch_model_loss_state_and_context_are_instantiable() -> None:
    """Core public records can be instantiated with contract-shaped tensors."""

    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.int64)
    batch = Batch(
        sample_ids=["a", "b"],
        image_weak=images,
        image_strong=None,
        targets=targets,
        class_ids=["0001", "0002"],
    )
    model_output = ModelOutput(
        logits=torch.zeros((2, 2)),
        embedding=torch.nn.functional.normalize(torch.ones((2, 4)), dim=1),
        temperature=None,
    )
    state = SampleState(
        sample_id="a",
        seen_count=1,
        ema_loss=0.5,
        ema_probs=None,
        prediction_stability=1.0,
        augmentation_agreement=1.0,
        prototype_similarity=0.8,
        prototype_margin=0.2,
        trust_score=0.9,
        supervised_weight=0.9,
        partition="trusted",
        pseudo_target=None,
        pseudo_confidence=None,
        updated_epoch=0,
    )
    loss = LossOutput(
        total=torch.tensor(1.0),
        components={"ce": torch.tensor(1.0)},
        per_sample_supervised=torch.tensor([0.5, 0.7]),
    )
    run = RunContext(
        run_id="run",
        run_dir=Path("runs/run"),
        seed=42,
        num_classes=2,
        class_to_idx={"0001": 0, "0002": 1},
        config_digest="config",
        data_digest="data",
    )
    epoch = EpochContext(run=run, epoch=1, global_step=2)

    assert batch.image_weak.shape == (2, 3, 224, 224)
    assert model_output.logits.shape == (2, 2)
    assert state.partition == "trusted"
    assert loss.total.ndim == 0
    assert epoch.run.run_id == "run"


def test_public_records_have_type_hints_and_docstrings() -> None:
    """Public dataclasses include annotations and docstrings for later agents."""

    for public_type in [SampleRecord, Batch, ModelOutput, SampleState, LossOutput, RunContext]:
        assert public_type.__doc__
        assert get_type_hints(public_type)


def test_config_rejects_unknown_top_level_and_nested_fields() -> None:
    """Strict config validation fails on misspelled or unexpected fields."""

    assert load_config_from_mapping(REQUIRED_CONFIG)

    with_top_level_unknown = dict(REQUIRED_CONFIG)
    with_top_level_unknown["unexpected"] = {}
    with pytest.raises(ValidationError):
        load_config_from_mapping(with_top_level_unknown)

    with_nested_unknown = dict(REQUIRED_CONFIG)
    with_nested_unknown["model"] = {"clip_name": "ViT-B/32"}
    with pytest.raises(ValidationError):
        load_config_from_mapping(with_nested_unknown)
