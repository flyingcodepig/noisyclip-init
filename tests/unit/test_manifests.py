"""Tests for manifest serialization, digests, and dataset boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noisyclip.data.dataset import ManifestImageDataset, collate_batch
from noisyclip.data.manifests import (
    ManifestError,
    data_digest,
    make_sample_id,
    read_manifest,
    write_manifest,
)
from noisyclip.data.records import SampleRecord


def _record(relative_path: str, split: str = "train") -> SampleRecord:
    is_test = split == "test"
    return SampleRecord(
        sample_id=make_sample_id(relative_path),
        relative_path=relative_path,
        split=split,
        class_id=None if is_test else "0001",
        target=None if is_test else 0,
        file_sha256="0" * 64,
        width=16,
        height=16,
        readable=True,
    )


def test_manifest_round_trip_preserves_fields_and_sorting(tmp_path: Path) -> None:
    """Serialization and reload preserve SampleRecord semantics."""

    records = [_record("0001/b.png"), _record("0001/a.png")]
    path = write_manifest(records, tmp_path / "manifest.json")

    loaded = read_manifest(path)

    assert [record.relative_path for record in loaded] == ["0001/a.png", "0001/b.png"]
    assert loaded[0] == _record("0001/a.png")


def test_duplicate_sample_id_fails(tmp_path: Path) -> None:
    """Duplicate sample IDs are rejected before writing."""

    first = _record("0001/a.png")
    with pytest.raises(ManifestError, match="Duplicate sample_id"):
        write_manifest([first, first], tmp_path / "bad.json")


def test_absolute_path_and_illegal_split_fail(tmp_path: Path) -> None:
    """Manifests never serialize absolute paths or unknown split names."""

    with pytest.raises(ManifestError, match="relative_path"):
        write_manifest([_record(str(tmp_path / "x.png"))], tmp_path / "abs.json")

    bad = SampleRecord(
        sample_id=make_sample_id("0001/a.png"),
        relative_path="0001/a.png",
        split="dev",
        class_id="0001",
        target=0,
        file_sha256=None,
        width=16,
        height=16,
        readable=True,
    )
    with pytest.raises(ManifestError, match="Illegal split"):
        write_manifest([bad], tmp_path / "bad_split.json")


def test_data_digest_changes_when_file_hash_changes() -> None:
    """Changing one image hash changes the data digest."""

    record = _record("0001/a.png")
    changed = SampleRecord(
        sample_id=record.sample_id,
        relative_path=record.relative_path,
        split=record.split,
        class_id=record.class_id,
        target=record.target,
        file_sha256="1" * 64,
        width=record.width,
        height=record.height,
        readable=record.readable,
    )

    assert data_digest([record], {"0001": 0}) != data_digest([changed], {"0001": 0})


def test_test_manifest_cannot_be_loaded_for_training() -> None:
    """A test manifest entering training Dataset fails before image loading."""

    with pytest.raises(ManifestError, match="Training dataset"):
        ManifestImageDataset(
            [_record("sample.png", split="test")],
            data_root=Path("unused"),
            split="test",
            image_weak_transform=lambda image, *, sample_id=None: torch.zeros((3, 224, 224)),
            training=True,
        )


def test_collate_batch_builds_public_batch() -> None:
    """Dataset item dictionaries collate into the public Batch structure."""

    batch = collate_batch(
        [
            {
                "sample_id": "a",
                "image_weak": torch.zeros((3, 224, 224), dtype=torch.float32),
                "image_strong": None,
                "target": 0,
                "class_id": "0001",
            }
        ]
    )

    assert batch.image_weak.shape == (1, 3, 224, 224)
    assert batch.targets is not None
    assert batch.targets.dtype == torch.int64
