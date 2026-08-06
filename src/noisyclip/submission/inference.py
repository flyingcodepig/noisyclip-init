"""Core deterministic submission inference orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from noisyclip.submission.mapping import ClassMapping, load_class_mapping
from noisyclip.submission.package import load_exported_model_package, predictions_for_filenames
from noisyclip.submission.validator import (
    ValidationReport,
    load_expected_filenames,
    validate_submission_csv,
)
from noisyclip.submission.writer import PREDICTION_FILENAME, write_prediction_csv


def run_packaged_submission_inference(
    model_path: Path | str,
    output_dir: Path | str,
    *,
    class_mapping_path: Path | str | None = None,
    test_manifest_path: Path | str | None = None,
    overwrite: bool = False,
) -> ValidationReport:
    """Write and validate `pred_results.csv` from one exported model package.

    Args:
        model_path: Path to exactly one compatible local export package. The
            package records a single student model, deterministic preprocessing,
            class mapping digest, and either local predictions or a future Agent
            B compatible inference backend.
        output_dir: Directory where `pred_results.csv` will be created.
        class_mapping_path: Optional runtime class mapping JSON. When supplied,
            its digest must match the model package metadata; otherwise the
            embedded package mapping is used.
        test_manifest_path: Optional test manifest or filename-list path. When
            absent, package `expected_filenames` are used.
        overwrite: Whether an existing `pred_results.csv` may be replaced.

    Returns:
        A `ValidationReport` for the generated CSV.

    Raises:
        ValueError: If the model package, mapping, manifest, prediction indices,
            or generated CSV violate F02 constraints.
        OSError, json.JSONDecodeError: If required files cannot be read.
    """

    package = load_exported_model_package(model_path)
    mapping = _load_runtime_mapping(package.mapping, class_mapping_path)
    package.require_compatible_mapping(mapping)
    filenames = _load_runtime_filenames(package.expected_filenames, test_manifest_path)
    predictions = predictions_for_filenames(package, filenames)
    output_path = Path(output_dir) / PREDICTION_FILENAME
    write_prediction_csv(filenames, predictions, mapping, output_path, overwrite=overwrite)
    return validate_submission_csv(output_path, filenames, mapping)


def _load_runtime_mapping(
    embedded: ClassMapping,
    class_mapping_path: Path | str | None,
) -> ClassMapping:
    if class_mapping_path is None:
        return embedded
    return load_class_mapping(class_mapping_path)


def _load_runtime_filenames(
    embedded: Sequence[str],
    test_manifest_path: Path | str | None,
) -> list[str]:
    if test_manifest_path is not None:
        return load_expected_filenames(test_manifest_path)
    if embedded:
        return list(embedded)
    raise ValueError("A test manifest or package expected_filenames list is required.")
