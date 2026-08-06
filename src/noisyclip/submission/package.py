"""Small adapter around a single exported inference package."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noisyclip.submission.mapping import ClassMapping, MappingError, validate_class_mapping


class ExportPackageError(ValueError):
    """Raised when an exported model package violates F02 submission rules."""


@dataclass(frozen=True, slots=True)
class ExportedModelPackage:
    """Validated metadata for one single-model inference package.

    Args:
        path: JSON export-package path.
        mapping: Validated `ClassMapping` embedded in the package.
        mapping_digest: Stable mapping digest recorded by the exporter.
        preprocess: Test preprocessing metadata. Expected values are deterministic
            CLIP evaluation preprocessing such as resize `256` and center crop
            `224`; TTA must be absent or `False`.
        expected_filenames: Optional official filename sequence of shape `[N]`.
        predictions: Optional internal prediction indices. A sequence has shape
            `[N]`; a mapping is keyed by filename and stores indices in `[0, C)`.

    Raises:
        ExportPackageError: Raised by loaders when the package contains teacher,
            ensemble, or multiple-model metadata, or misses required fields.
    """

    path: Path
    mapping: ClassMapping
    mapping_digest: str
    preprocess: Mapping[str, Any]
    expected_filenames: tuple[str, ...]
    predictions: tuple[int, ...] | Mapping[str, int] | None

    def require_compatible_mapping(self, mapping: ClassMapping) -> None:
        """Fail if this package's recorded mapping digest differs from `mapping`.

        Args:
            mapping: Runtime mapping loaded from the caller-provided class
                mapping file, with index lookup shape `[C]`.

        Raises:
            MappingError: If the class count, digest, or embedded mapping digest
                differs, which catches swapped class-order metadata.
        """

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
    """Load JSON metadata for one exported inference package.

    Args:
        path: UTF-8 JSON file produced by a compatible `export_single_model`
            implementation. The package must describe exactly one student
            inference model and must not include teacher or ensemble metadata.

    Returns:
        An `ExportedModelPackage` ready for deterministic prediction writing.

    Raises:
        ExportPackageError: If the package is missing required fields, contains
            teacher/ensemble/multiple-model metadata, or enables TTA.
        MappingError: If embedded class mapping metadata is malformed.
        OSError, json.JSONDecodeError: If the package cannot be loaded.
    """

    package_path = Path(path)
    raw = json.loads(package_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ExportPackageError("Export package root must be a JSON object.")
    _reject_forbidden_model_metadata(raw)

    class_to_idx = raw.get("class_to_idx")
    if not isinstance(class_to_idx, Mapping):
        raise ExportPackageError("Export package must include class_to_idx mapping.")
    mapping = validate_class_mapping(class_to_idx)
    mapping_digest = raw.get("mapping_digest", mapping.digest)
    if not isinstance(mapping_digest, str) or not mapping_digest:
        raise ExportPackageError("Export package mapping_digest must be a non-empty string.")
    mapping.require_digest(mapping_digest, field="mapping_digest")

    preprocess = raw.get("preprocess", {})
    if not isinstance(preprocess, Mapping):
        raise ExportPackageError("Export package preprocess must be an object.")
    _validate_preprocess(preprocess)

    expected_filenames = _parse_optional_filenames(raw.get("expected_filenames", ()))
    predictions = _parse_optional_predictions(raw.get("predictions"))
    return ExportedModelPackage(
        path=package_path,
        mapping=mapping,
        mapping_digest=mapping_digest,
        preprocess=preprocess,
        expected_filenames=tuple(expected_filenames),
        predictions=predictions,
    )


def predictions_for_filenames(
    package: ExportedModelPackage,
    filenames: Sequence[str],
) -> list[int]:
    """Return internal prediction indices ordered to match `filenames`.

    Args:
        package: Single-model package with either a sequence of prediction
            indices shaped `[N]` or a filename-keyed prediction mapping.
        filenames: Expected official test filenames with shape `[N]`.

    Returns:
        Prediction indices ordered by `filenames`.

    Raises:
        ExportPackageError: If the package has no compatible prediction source,
            length differs, filenames are missing, or values are not integers.
    """

    if package.predictions is None:
        raise ExportPackageError(
            "Export package does not expose a compatible local inference backend yet; "
            "Agent B must provide a callable single-model inference interface."
        )
    if isinstance(package.predictions, Mapping):
        missing = [filename for filename in filenames if filename not in package.predictions]
        if missing:
            raise ExportPackageError(f"Predictions are missing filenames: {missing}.")
        return [_require_index(package.predictions[filename], filename) for filename in filenames]
    if len(package.predictions) != len(filenames):
        raise ExportPackageError(
            f"Prediction count {len(package.predictions)} differs from filename count "
            f"{len(filenames)}."
        )
    return [
        _require_index(value, f"position {index}")
        for index, value in enumerate(package.predictions)
    ]


def _reject_forbidden_model_metadata(raw: Mapping[str, Any]) -> None:
    models = raw.get("models", [{"role": raw.get("role", "student")}])
    if not isinstance(models, list) or len(models) != 1:
        raise ExportPackageError("Export package must contain exactly one inference model.")
    model = models[0]
    if not isinstance(model, Mapping):
        raise ExportPackageError("Export package model entry must be an object.")
    role = model.get("role")
    if role != "student":
        raise ExportPackageError("Export package must contain exactly one student inference model.")
    forbidden_keys = ("teacher", "teachers", "ensemble", "ensembles", "voting", "fusion")
    for key in forbidden_keys:
        if raw.get(key):
            raise ExportPackageError(f"Export package must not include {key} metadata.")
    if raw.get("artifact_type") == "ensemble":
        raise ExportPackageError("Export package must not be an ensemble artifact.")


def _validate_preprocess(preprocess: Mapping[str, Any]) -> None:
    if preprocess.get("test_time_augmentation", False):
        raise ExportPackageError("Test-time augmentation is forbidden for final inference.")
    if "center_crop" in preprocess and preprocess["center_crop"] != 224:
        raise ExportPackageError("Export package must use a single 224 center crop.")


def _parse_optional_filenames(raw: Any) -> list[str]:
    if raw in (None, ()):
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise ExportPackageError("expected_filenames must be a list of non-empty strings.")
    return raw


def _parse_optional_predictions(raw: Any) -> tuple[int, ...] | Mapping[str, int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return tuple(_require_index(item, f"position {index}") for index, item in enumerate(raw))
    if isinstance(raw, Mapping):
        predictions: dict[str, int] = {}
        for filename, value in raw.items():
            if not isinstance(filename, str) or not filename:
                raise ExportPackageError("prediction mapping keys must be non-empty filenames.")
            predictions[filename] = _require_index(value, filename)
        return predictions
    raise ExportPackageError("predictions must be a list or filename-keyed object when present.")


def _require_index(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportPackageError(f"Prediction for {field} must be an integer internal index.")
    if value < 0:
        raise ExportPackageError(f"Prediction for {field} must be non-negative.")
    return value
