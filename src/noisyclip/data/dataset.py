"""Manifest-backed image dataset construction."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset

from noisyclip.data.image_io import load_rgb_image
from noisyclip.data.manifests import ManifestError, normalize_relative_path, read_manifest
from noisyclip.data.records import Batch, SampleRecord

ImageTransform = Callable[..., Tensor]


class ManifestImageDataset(Dataset[dict[str, Any]]):
    """Dataset that loads images only through a precomputed manifest.

    Args:
        records: Manifest records. No directory scanning is performed here.
        data_root: Official train or test root used to resolve relative paths.
        split: Expected split, one of `train`, `val`, or `test`.
        image_weak_transform: Callable returning a `[3, 224, 224]` tensor.
        image_strong_transform: Optional callable returning a second tensor for
            train consistency paths.
        training: If true, only `train` split is accepted and every record must
            have a target. A test manifest fails immediately.

    Raises:
        ManifestError: If records are mislabeled, test records are loaded for
            training, or train/val targets are missing.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        data_root: Path | str,
        split: Literal["train", "val", "test"],
        image_weak_transform: ImageTransform,
        image_strong_transform: ImageTransform | None = None,
        training: bool = False,
    ) -> None:
        """Initialize the dataset from manifest rows only."""

        if split not in {"train", "val", "test"}:
            raise ManifestError(f"Illegal dataset split: {split}")
        if training and split != "train":
            raise ManifestError(f"Training dataset cannot load {split} manifest.")
        self.records = sorted(records, key=lambda row: (row.relative_path, row.sample_id))
        self.data_root = Path(data_root)
        self.split = split
        self.image_weak_transform = image_weak_transform
        self.image_strong_transform = image_strong_transform
        self.training = training
        self._validate_records()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        data_root: Path | str,
        split: Literal["train", "val", "test"],
        image_weak_transform: ImageTransform,
        image_strong_transform: ImageTransform | None = None,
        training: bool = False,
    ) -> ManifestImageDataset:
        """Construct a dataset from a serialized manifest file.

        Args:
            manifest_path: JSON manifest produced by `write_manifest`.
            data_root: Official train or test root.
            split: Expected split.
            image_weak_transform: Callable returning `[3, 224, 224]`.
            image_strong_transform: Optional strong-view callable.
            training: Whether the dataset will feed the training loop.

        Returns:
            A manifest-backed dataset.

        Raises:
            ManifestError: If manifest rows violate split/target rules.
        """

        return cls(
            read_manifest(manifest_path),
            data_root=data_root,
            split=split,
            image_weak_transform=image_weak_transform,
            image_strong_transform=image_strong_transform,
            training=training,
        )

    def __len__(self) -> int:
        """Return the number of manifest rows."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample and return fields compatible with `collate_batch`.

        Args:
            index: Integer in `[0, len(self))`.

        Returns:
            Dict containing `sample_id`, `[3, 224, 224]` weak image tensor,
            optional strong image tensor, optional target, and optional class ID.

        Raises:
            IndexError: If the index is outside the dataset.
            ImageAuditError: If the image cannot be decoded.
            ValueError: If transforms return tensors with invalid shape.
        """

        record = self.records[index]
        image = load_rgb_image(self.data_root / normalize_relative_path(record.relative_path))
        image_weak = self.image_weak_transform(image, sample_id=record.sample_id)
        image_strong = (
            self.image_strong_transform(image, sample_id=record.sample_id)
            if self.image_strong_transform is not None
            else None
        )
        return {
            "sample_id": record.sample_id,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "target": record.target,
            "class_id": record.class_id,
        }

    def _validate_records(self) -> None:
        for record in self.records:
            if record.split != self.split:
                raise ManifestError(
                    f"Dataset split {self.split} cannot load record {record.sample_id} "
                    f"from split {record.split}."
                )
            if self.training and record.split == "test":
                raise ManifestError(
                    f"Test manifest cannot be loaded in training mode: {record.sample_id}"
                )
            if record.split in {"train", "val"} and (
                record.target is None or record.class_id is None
            ):
                raise ManifestError(
                    f"{record.split} record requires target/class_id: {record.sample_id}"
                )
            if record.split == "test" and (
                record.target is not None or record.class_id is not None
            ):
                raise ManifestError(
                    f"test record must not include target/class_id: {record.sample_id}"
                )


def collate_batch(items: list[dict[str, Any]]) -> Batch:
    """Collate dataset items into the public `Batch` interface.

    Args:
        items: Non-empty list of dictionaries returned by
            `ManifestImageDataset.__getitem__`.

    Returns:
        `Batch` with `image_weak` shape `[B, 3, 224, 224]`,
        optional `image_strong` with the same shape, optional int64 targets
        shape `[B]`, and stable sample IDs.

    Raises:
        ValueError: If tensors are missing, non-finite, have wrong shape/dtype,
            or if target presence is inconsistent across rows.
    """

    if not items:
        raise ValueError("Cannot collate an empty batch.")
    weak_tensors = [_require_image_tensor(item["image_weak"], "image_weak") for item in items]
    strong_values = [item["image_strong"] for item in items]
    has_strong = [value is not None for value in strong_values]
    if any(has_strong) and not all(has_strong):
        raise ValueError("All batch rows must agree on image_strong presence.")
    image_strong = None
    if all(has_strong):
        image_strong = torch.stack(
            [_require_image_tensor(value, "image_strong") for value in strong_values]
        )

    targets = [item["target"] for item in items]
    target_tensor = None
    if any(target is not None for target in targets):
        if not all(isinstance(target, int) for target in targets):
            raise ValueError("All labeled batch rows require integer targets.")
        target_tensor = torch.tensor(targets, dtype=torch.int64)

    class_ids = [item["class_id"] for item in items]
    class_id_list = None
    if any(class_id is not None for class_id in class_ids):
        if not all(isinstance(class_id, str) for class_id in class_ids):
            raise ValueError("All labeled batch rows require class_id strings.")
        class_id_list = list(class_ids)

    return Batch(
        sample_ids=[str(item["sample_id"]) for item in items],
        image_weak=torch.stack(weak_tensors),
        image_strong=image_strong,
        targets=target_tensor,
        class_ids=class_id_list,
    )


def _require_image_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if value.shape != (3, 224, 224):
        raise ValueError(f"{name} must have shape [3, 224, 224], got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have floating dtype, got {value.dtype}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return value
