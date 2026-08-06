"""Deterministic stratified train/validation splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from noisyclip.data.records import SampleRecord


class SplitError(ValueError):
    """Raised when a train/validation split cannot satisfy safety rules."""


def stratified_train_val_split(
    records: list[SampleRecord],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split labeled records by class with deterministic, order-invariant logic.

    Args:
        records: Readable labeled samples. Each row must have `class_id` and
            `target`; image tensors are not inspected, so no content, loss, or
            test-set information can affect the split.
        seed: Non-negative integer seed. The same records, seed, and validation
            fraction produce the same assignments and downstream digest.
        val_fraction: Fraction in `(0, 0.5)`. For each class with at least two
            samples, at least one validation sample and one training sample are
            retained.

    Returns:
        `(train_records, val_records)` with updated `split` fields.

    Raises:
        SplitError: If `val_fraction` is outside `(0, 0.5)`, a sample is
            unlabeled, or a class has fewer than two readable samples.
    """

    if seed < 0:
        raise SplitError("split seed must be non-negative.")
    if not 0.0 < val_fraction < 0.5:
        raise SplitError(f"val_fraction must be in (0, 0.5), got {val_fraction}.")

    by_class: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.class_id is None or record.target is None:
            raise SplitError(f"Labeled split requires class_id and target: {record.sample_id}")
        if record.split == "test":
            raise SplitError(f"Test sample cannot enter train/val split: {record.sample_id}")
        by_class[record.class_id].append(record)

    train_records: list[SampleRecord] = []
    val_records: list[SampleRecord] = []
    for class_id in sorted(by_class):
        class_records = sorted(
            by_class[class_id], key=lambda row: (row.relative_path, row.sample_id)
        )
        sample_count = len(class_records)
        if sample_count < 2:
            raise SplitError(
                f"Class {class_id} has {sample_count} readable sample; at least 2 are required "
                "to create both train and val splits."
            )
        val_count = max(1, round(sample_count * val_fraction))
        val_count = min(val_count, sample_count - 1)

        rng = random.Random(f"{seed}:{class_id}")
        shuffled = list(class_records)
        rng.shuffle(shuffled)
        val_ids = {record.sample_id for record in shuffled[:val_count]}
        for record in class_records:
            if record.sample_id in val_ids:
                val_records.append(_with_split(record, "val"))
            else:
                train_records.append(_with_split(record, "train"))

    return (
        sorted(train_records, key=lambda row: (row.relative_path, row.sample_id)),
        sorted(val_records, key=lambda row: (row.relative_path, row.sample_id)),
    )


def _with_split(record: SampleRecord, split: str) -> SampleRecord:
    return SampleRecord(
        sample_id=record.sample_id,
        relative_path=record.relative_path,
        split=split,
        class_id=record.class_id,
        target=record.target,
        file_sha256=record.file_sha256,
        width=record.width,
        height=record.height,
        readable=record.readable,
    )
