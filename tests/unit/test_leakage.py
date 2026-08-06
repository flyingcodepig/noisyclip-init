"""Tests for test-data leakage detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from noisyclip.data.leakage import LeakageError, assert_no_leakage, check_manifest_leakage
from noisyclip.data.manifests import make_sample_id
from noisyclip.data.records import SampleRecord


def _record(relative_path: str, split: str, file_hash: str) -> SampleRecord:
    is_test = split == "test"
    return SampleRecord(
        sample_id=make_sample_id(relative_path),
        relative_path=relative_path,
        split=split,
        class_id=None if is_test else "0001",
        target=None if is_test else 0,
        file_sha256=file_hash,
        width=10,
        height=10,
        readable=True,
    )


def test_path_and_filename_intersections_fail() -> None:
    """Shared relative paths or basenames between train/val and test leak."""

    train = [_record("0001/shared.png", "train", "0" * 64)]
    test = [_record("shared.png", "test", "1" * 64)]

    report = check_manifest_leakage(train, [], test)

    assert report.filename_intersections == ("shared.png",)
    with pytest.raises(LeakageError, match="filename intersections"):
        assert_no_leakage(report)


def test_hash_intersection_fails() -> None:
    """Shared file hashes between labeled and test manifests leak."""

    shared_hash = "a" * 64
    report = check_manifest_leakage(
        [_record("0001/train.png", "train", shared_hash)],
        [],
        [_record("test.png", "test", shared_hash)],
    )

    assert report.hash_intersections == (shared_hash,)
    with pytest.raises(LeakageError, match="hash intersections"):
        assert_no_leakage(report)


def test_nested_train_test_roots_fail(tmp_path: Path) -> None:
    """Train and test roots cannot resolve to the same or nested directories."""

    train_root = tmp_path / "train"
    test_root = train_root / "test"
    test_root.mkdir(parents=True)

    report = check_manifest_leakage([], [], [], train_root=train_root, test_root=test_root)

    assert report.root_issue is not None
    with pytest.raises(LeakageError, match="inside"):
        assert_no_leakage(report)
