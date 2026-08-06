"""Public data records shared by data, training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """A stable manifest row for one image sample.

    `sample_id` is the SHA256 of the relative path, never an absolute machine
    path. `split` must be `train`, `val`, or `test`. `class_id` and `target`
    are `None` for test samples and populated for train/val samples.
    """

    sample_id: str
    relative_path: str
    split: str
    class_id: str | None
    target: int | None
    file_sha256: str | None
    width: int | None
    height: int | None
    readable: bool


@dataclass(slots=True)
class Batch:
    """A batch handed to models and losses.

    `image_weak` has shape `[B, 3, 224, 224]` and is float32 or AMP dtype.
    `image_strong`, when present, has the same shape. `targets` is an int64
    tensor with shape `[B]`, and is `None` for unlabeled test-only flows.
    """

    sample_ids: list[str]
    image_weak: Tensor
    image_strong: Tensor | None
    targets: Tensor | None
    class_ids: list[str] | None
