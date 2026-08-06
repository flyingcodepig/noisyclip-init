"""Deterministic writer for official `pred_results.csv` files."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from noisyclip.submission.mapping import ClassMapping, MappingError, validate_unique_ids

PREDICTION_FILENAME = "pred_results.csv"


class PredictionWriteError(ValueError):
    """Raised when prediction CSV inputs or destination paths are invalid."""


def write_prediction_csv(
    filenames: Sequence[str],
    prediction_indices: Sequence[int],
    mapping: ClassMapping,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write official two-column prediction CSV with no header.

    Args:
        filenames: Sequence of official test filenames with shape `[N]`.
            Entries must be non-empty strings and unique.
        prediction_indices: Sequence of internal class indices with shape `[N]`.
            Each index must be an integer in `[0, C)`, where `C` is the mapping
            class count.
        mapping: Validated `ClassMapping` used to convert indices to four-digit
            class ids.
        output_path: Destination path. Its basename must be `pred_results.csv`.
        overwrite: When `False`, an existing destination fails before writing.

    Returns:
        The written `pred_results.csv` path.

    Raises:
        PredictionWriteError: If lengths differ, filenames are empty or
            duplicated, output name is wrong, or destination exists without
            explicit overwrite.
        MappingError: If any prediction index is invalid for `mapping`.
        OSError: If the destination cannot be written.
    """

    destination = Path(output_path)
    if destination.name != PREDICTION_FILENAME:
        raise PredictionWriteError(
            f"Submission file must be named {PREDICTION_FILENAME}, got {destination.name!r}."
        )
    if destination.exists() and not overwrite:
        raise PredictionWriteError(
            f"Refusing to overwrite existing prediction file: {destination}."
        )
    if len(filenames) != len(prediction_indices):
        raise PredictionWriteError(
            f"filenames length {len(filenames)} differs from predictions length "
            f"{len(prediction_indices)}."
        )

    checked_filenames = _validate_filenames(filenames)
    rows = [
        (filename, mapping.class_id_for_index(index))
        for filename, index in zip(checked_filenames, prediction_indices, strict=True)
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _write_temporary_csv(destination.parent, rows)
    try:
        if overwrite:
            os.replace(temp_path, destination)
        else:
            if destination.exists():
                raise PredictionWriteError(
                    f"Refusing to overwrite existing prediction file: {destination}."
                )
            os.link(temp_path, destination)
            Path(temp_path).unlink()
    except OSError:
        Path(temp_path).unlink(missing_ok=True)
        raise
    except PredictionWriteError:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return destination


def _validate_filenames(filenames: Sequence[str]) -> list[str]:
    checked: list[str] = []
    for position, filename in enumerate(filenames):
        if not isinstance(filename, str) or not filename:
            raise PredictionWriteError(f"filename at row {position} must be a non-empty string.")
        checked.append(filename)
    try:
        validate_unique_ids(checked, field="filenames")
    except MappingError as exc:
        raise PredictionWriteError(str(exc)) from exc
    return checked


def _write_temporary_csv(directory: Path, rows: Sequence[tuple[str, str]]) -> str:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=directory,
        prefix=".pred_results.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)
        return handle.name
