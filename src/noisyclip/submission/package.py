"""Validation adapter for one exported PyTorch inference package."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noisyclip.models.export import load_export_package
from noisyclip.submission.mapping import ClassMapping, MappingError, validate_class_mapping


class ExportPackageError(ValueError):
    """Raised when an exported model package violates F02 submission rules."""


@dataclass(frozen=True, slots=True)
class ExportedModelPackage:
    """Validated metadata for exactly one student inference model."""

    path: Path
    mapping: ClassMapping
    mapping_digest: str
    preprocess: Mapping[str, Any]
    num_classes: int

    def require_compatible_mapping(self, mapping: ClassMapping) -> None:
        """Fail when runtime and exported mappings differ in size or digest."""

        if self.mapping.num_classes != mapping.num_classes:
            raise MappingError(
                f"Model package class count {self.mapping.num_classes} differs from "
                f"runtime mapping class count {mapping.num_classes}."
            )
        mapping.require_digest(self.mapping_digest, field="model mapping_digest")
        if self.mapping.digest != mapping.digest:
            raise MappingError(
                f"Model embedded mapping digest {self.mapping.digest} differs from "
                f"runtime mapping digest {mapping.digest}."
            )


def load_exported_model_package(path: Path | str) -> ExportedModelPackage:
    """Load single-model metadata from an Agent B `.pt` export artifact."""

    package_path = Path(path)
    raw = load_export_package(package_path)
    class_to_idx = raw.get("class_to_idx")
    if not isinstance(class_to_idx, Mapping):
        raise ExportPackageError("Submission model must embed class_to_idx metadata.")
    mapping = validate_class_mapping(class_to_idx)
    mapping_digest = raw.get("mapping_digest")
    if not isinstance(mapping_digest, str) or not mapping_digest:
        raise ExportPackageError("Submission model must embed a mapping_digest.")
    mapping.require_digest(mapping_digest, field="mapping_digest")

    num_classes = raw.get("num_classes")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ExportPackageError("Export package num_classes must be an integer.")
    if num_classes != mapping.num_classes:
        raise ExportPackageError(
            f"Model class count {num_classes} differs from mapping size {mapping.num_classes}."
        )
    preprocess = raw.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise ExportPackageError("Export package preprocess must be an object.")
    _validate_preprocess(preprocess)
    _validate_clip_weight_metadata(raw.get("clip_weight_metadata"))
    return ExportedModelPackage(
        path=package_path,
        mapping=mapping,
        mapping_digest=mapping_digest,
        preprocess=preprocess,
        num_classes=num_classes,
    )


def _validate_preprocess(preprocess: Mapping[str, Any]) -> None:
    if preprocess.get("test_time_augmentation", False):
        raise ExportPackageError("Test-time augmentation is forbidden for final inference.")
    if preprocess.get("image_size") != 224 or preprocess.get("center_crop") != 224:
        raise ExportPackageError("Export package must use one 224x224 center crop.")
    resize = preprocess.get("resize_short_side")
    if isinstance(resize, bool) or not isinstance(resize, int) or resize < 224:
        raise ExportPackageError("resize_short_side must be an integer of at least 224.")
    if preprocess.get("normalization") != "openai_clip_official":
        raise ExportPackageError("Export package must use official OpenAI CLIP normalization.")


def _validate_clip_weight_metadata(raw: Any) -> None:
    if not isinstance(raw, Mapping):
        raise ExportPackageError("Submission model must include CLIP weight metadata.")
    if raw.get("model_name") != "ViT-B/32":
        raise ExportPackageError("Submission model backbone must be CLIP ViT-B/32.")
    if raw.get("source") not in {"openai", "openai_clip_official"}:
        raise ExportPackageError("Submission model must use official OpenAI CLIP weights.")
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ExportPackageError("Submission model must record a lowercase CLIP weight SHA256.")
