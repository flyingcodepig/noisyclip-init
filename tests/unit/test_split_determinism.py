"""Tests for deterministic stratified splitting."""

from __future__ import annotations

import random

import pytest

from noisyclip.data.manifests import make_sample_id, manifest_digest
from noisyclip.data.records import SampleRecord
from noisyclip.data.split import SplitError, stratified_train_val_split


def _record(class_id: str, index: int) -> SampleRecord:
    relative_path = f"{class_id}/img_{index}.png"
    return SampleRecord(
        sample_id=make_sample_id(relative_path),
        relative_path=relative_path,
        split="train",
        class_id=class_id,
        target=int(class_id),
        file_sha256=None,
        width=32,
        height=32,
        readable=True,
    )


def test_same_seed_produces_identical_split_and_digest() -> None:
    """Same data, seed, and fraction produce identical assignments."""

    records = [_record("0000", index) for index in range(5)] + [
        _record("0001", index) for index in range(5)
    ]

    first = stratified_train_val_split(records, seed=123, val_fraction=0.2)
    second = stratified_train_val_split(records, seed=123, val_fraction=0.2)

    assert first == second
    assert manifest_digest(first[0] + first[1]) == manifest_digest(second[0] + second[1])


def test_input_order_does_not_change_split() -> None:
    """The split is invariant to caller-provided record order."""

    records = [_record("0000", index) for index in range(6)] + [
        _record("0001", index) for index in range(6)
    ]
    shuffled = list(records)
    random.Random(99).shuffle(shuffled)

    ordered_split = stratified_train_val_split(records, seed=7, val_fraction=0.25)
    shuffled_split = stratified_train_val_split(shuffled, seed=7, val_fraction=0.25)

    assert ordered_split == shuffled_split


def test_each_splittable_class_keeps_train_samples() -> None:
    """Every class with at least two samples retains at least one train row."""

    records = [_record("0000", index) for index in range(2)] + [
        _record("0001", index) for index in range(3)
    ]

    train, val = stratified_train_val_split(records, seed=4, val_fraction=0.49)

    assert {record.class_id for record in train} == {"0000", "0001"}
    assert {record.class_id for record in val} == {"0000", "0001"}


def test_tiny_one_sample_class_fails() -> None:
    """A one-sample class has no valid train/val split and fails clearly."""

    with pytest.raises(SplitError, match="at least 2"):
        stratified_train_val_split([_record("0000", 0)], seed=1, val_fraction=0.1)
