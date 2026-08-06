"""Deterministic single-model test inference and submission orchestration."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import DataLoader

from noisyclip.data.dataset import ManifestImageDataset, collate_batch
from noisyclip.data.manifests import read_manifest
from noisyclip.data.transforms import ClipImageTransform
from noisyclip.models.clip_loader import ClipBackend
from noisyclip.models.export import load_exported_model_auto
from noisyclip.submission.mapping import ClassMapping, load_class_mapping
from noisyclip.submission.package import load_exported_model_package
from noisyclip.submission.validator import ValidationReport, validate_submission_csv
from noisyclip.submission.writer import PREDICTION_FILENAME, write_prediction_csv


def run_packaged_submission_inference(
    model_path: Path | str,
    output_dir: Path | str,
    *,
    test_manifest_path: Path | str,
    test_root: Path | str,
    class_mapping_path: Path | str | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    num_workers: int = 0,
    cache_dir: Path | str | None = None,
    backend: ClipBackend | None = None,
    overwrite: bool = False,
) -> ValidationReport:
    """Run one exported student over the test manifest, then write and validate CSV.

    Test records are loaded only for inference. The function runs under
    `torch.inference_mode`, never updates parameters, and supports no TTA or
    multiple-model inputs.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    package = load_exported_model_package(model_path)
    mapping = _load_runtime_mapping(package.mapping, class_mapping_path)
    package.require_compatible_mapping(mapping)
    records = read_manifest(test_manifest_path)
    filenames = [_flat_test_filename(record.relative_path) for record in records]
    resize_short_side = int(package.preprocess["resize_short_side"])
    transform = ClipImageTransform(
        mode="eval",
        image_size=224,
        resize_short_side=resize_short_side,
    )
    dataset = ManifestImageDataset(
        records,
        data_root=test_root,
        split="test",
        image_weak_transform=transform,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    model = load_exported_model_auto(
        model_path,
        device=device,
        cache_dir=cache_dir,
        backend=backend,
    )
    model.eval()
    prediction_indices: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            if batch.targets is not None or batch.class_ids is not None:
                raise ValueError("Test inference batch unexpectedly contains labels.")
            output = model(batch.image_weak.to(device))
            if output.logits.shape[1] != mapping.num_classes:
                raise ValueError(
                    f"Model logits have {output.logits.shape[1]} classes; "
                    f"mapping has {mapping.num_classes}."
                )
            prediction_indices.extend(output.logits.argmax(dim=1).cpu().tolist())

    output_path = Path(output_dir) / PREDICTION_FILENAME
    write_prediction_csv(
        filenames,
        prediction_indices,
        mapping,
        output_path,
        overwrite=overwrite,
    )
    return validate_submission_csv(output_path, filenames, mapping)


def _load_runtime_mapping(
    embedded: ClassMapping,
    class_mapping_path: Path | str | None,
) -> ClassMapping:
    if class_mapping_path is None:
        return embedded
    return load_class_mapping(class_mapping_path)


def _flat_test_filename(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    if path.name != relative_path:
        raise ValueError(f"Competition test manifest must be flat, got {relative_path!r}.")
    return path.name
