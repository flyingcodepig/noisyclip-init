"""Machine-readable validation for official prediction CSV files."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from noisyclip.submission.mapping import CLASS_ID_PATTERN, ClassMapping, load_class_mapping


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured submission-validation failure.

    Args:
        code: Stable machine-readable error code.
        message: Human-readable explanation including the affected field.
        row: One-based CSV row number when applicable, otherwise `None`.
        field: Field name such as `filename`, `class_id`, or `row`.
    """

    code: str
    message: str
    row: int | None = None
    field: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured report for a prediction CSV.

    Args:
        valid: `True` only when the CSV has exactly two columns per row, no
            header, row count `[N]` equal to expected test images, exact filename
            set equality, and class ids that are four digits present in mapping.
        row_count: Number of non-empty parsed CSV rows.
        expected_count: Number of expected test filenames.
        issues: Tuple of `ValidationIssue` entries. Empty means success.
    """

    valid: bool
    row_count: int
    expected_count: int
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable machine-readable report."""

        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "expected_count": self.expected_count,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def to_json(self) -> str:
        """Serialize the report as stable UTF-8 JSON text."""

        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, indent=2)


def validate_submission_csv(
    csv_path: Path | str,
    expected_filenames: Sequence[str],
    mapping: ClassMapping,
) -> ValidationReport:
    """Validate a headerless official prediction CSV.

    Args:
        csv_path: UTF-8 CSV path. Each non-empty row must have exactly two
            columns: `filename,class_id`; no header row is allowed.
        expected_filenames: Official test filename list with shape `[N]`. The
            CSV filename multiset must match this list exactly, including path,
            extension, and case.
        mapping: Validated class mapping; every class id must be one of its
            four-digit keys.

    Returns:
        A `ValidationReport` with machine-readable issue codes.

    Raises:
        OSError: If the CSV file cannot be read.
        ValueError: If the expected filename list itself is empty, duplicated, or
            contains empty values.
    """

    expected = _validate_expected_filenames(expected_filenames)
    issues: list[ValidationIssue] = []
    rows = _read_csv_rows(Path(csv_path), issues)

    if rows and rows[0] == ["filename", "class_id"]:
        issues.append(
            ValidationIssue(
                code="HEADER_ROW",
                message="CSV must not contain a header row.",
                row=1,
                field="row",
            )
        )

    seen: dict[str, int] = {}
    present: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 2:
            issues.append(
                ValidationIssue(
                    code="COLUMN_COUNT",
                    message=f"Row must contain exactly two columns, got {len(row)}.",
                    row=row_number,
                    field="row",
                )
            )
            continue
        filename, class_id = row
        _validate_filename(filename, row_number, expected, seen, present, issues)
        _validate_class_id(class_id, row_number, mapping, issues)

    _validate_filename_set(present, expected, issues)
    if len(rows) != len(expected):
        issues.append(
            ValidationIssue(
                code="ROW_COUNT",
                message=f"CSV row count {len(rows)} differs from expected {len(expected)}.",
                field="row",
            )
        )

    return ValidationReport(
        valid=not issues,
        row_count=len(rows),
        expected_count=len(expected),
        issues=tuple(issues),
    )


def load_expected_filenames(path: Path | str) -> list[str]:
    """Load expected test filenames from JSON, JSONL, CSV, or plain text.

    Args:
        path: Manifest path. JSON can be a list of strings, a list of objects
            containing `filename` or `relative_path`, or an object with a
            `filenames`, `files`, `samples`, or `records` list. CSV uses the
            `filename` or `relative_path` header when present, otherwise the
            first column. Plain text treats each non-empty line as one filename.

    Returns:
        A list of filenames with shape `[N]`.

    Raises:
        ValueError: If no filenames can be extracted or duplicates are present.
        OSError: If the manifest cannot be read.
        json.JSONDecodeError: If a `.json` manifest is malformed.
    """

    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        return _filenames_from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
    if suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _filenames_from_json(records)
    if suffix == ".csv":
        return _filenames_from_csv(manifest_path)
    return _validate_expected_filenames(
        [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )


def load_validation_inputs(
    csv_path: Path | str,
    expected_path: Path | str,
    mapping_path: Path | str,
) -> ValidationReport:
    """Load manifest and mapping paths, then validate a submission CSV.

    Args:
        csv_path: Prediction CSV path.
        expected_path: Test manifest or filename-list path.
        mapping_path: Class mapping JSON path.

    Returns:
        A machine-readable `ValidationReport`.

    Raises:
        OSError, ValueError, json.JSONDecodeError: Propagated from loading or
            strict validation of input files.
    """

    expected = load_expected_filenames(expected_path)
    mapping = load_class_mapping(mapping_path)
    return validate_submission_csv(csv_path, expected, mapping)


def _read_csv_rows(csv_path: Path, issues: list[ValidationIssue]) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                issues.append(
                    ValidationIssue(
                        code="EMPTY_ROW",
                        message="CSV must not contain empty rows.",
                        row=row_number,
                        field="row",
                    )
                )
                continue
            rows.append(row)
    return rows


def _validate_expected_filenames(values: Sequence[str]) -> list[str]:
    if not values:
        raise ValueError("Expected test filename list must not be empty.")
    checked: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for position, filename in enumerate(values):
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Expected filename at position {position} must be a non-empty string."
            )
        checked.append(filename)
        if filename in seen:
            duplicates.add(filename)
        seen.add(filename)
    if duplicates:
        raise ValueError(f"Expected test filename list contains duplicates: {sorted(duplicates)}.")
    return checked


def _validate_filename(
    filename: str,
    row_number: int,
    expected: Sequence[str],
    seen: dict[str, int],
    present: list[str],
    issues: list[ValidationIssue],
) -> None:
    expected_set = set(expected)
    lower_to_expected = {item.lower(): item for item in expected}
    if not filename:
        issues.append(
            ValidationIssue(
                code="EMPTY_FILENAME",
                message="filename must be a non-empty string.",
                row=row_number,
                field="filename",
            )
        )
        return
    if filename in seen:
        issues.append(
            ValidationIssue(
                code="DUPLICATE_FILENAME",
                message=f"filename {filename!r} duplicates row {seen[filename]}.",
                row=row_number,
                field="filename",
            )
        )
    seen.setdefault(filename, row_number)
    present.append(filename)
    if filename not in expected_set:
        matched = lower_to_expected.get(filename.lower())
        code = "FILENAME_CASE_MISMATCH" if matched is not None else "EXTRA_FILENAME"
        detail = f"; expected exact spelling {matched!r}" if matched is not None else ""
        issues.append(
            ValidationIssue(
                code=code,
                message=f"filename {filename!r} is not in the official test list{detail}.",
                row=row_number,
                field="filename",
            )
        )


def _validate_class_id(
    class_id: str,
    row_number: int,
    mapping: ClassMapping,
    issues: list[ValidationIssue],
) -> None:
    if CLASS_ID_PATTERN.fullmatch(class_id) is None:
        issues.append(
            ValidationIssue(
                code="CLASS_ID_FORMAT",
                message=f"class_id {class_id!r} must be exactly four digits.",
                row=row_number,
                field="class_id",
            )
        )
        return
    if class_id not in mapping.class_to_idx:
        issues.append(
            ValidationIssue(
                code="UNKNOWN_CLASS_ID",
                message=f"class_id {class_id!r} is absent from class mapping.",
                row=row_number,
                field="class_id",
            )
        )


def _validate_filename_set(
    present: Sequence[str],
    expected: Sequence[str],
    issues: list[ValidationIssue],
) -> None:
    present_set = set(present)
    expected_set = set(expected)
    for filename in sorted(expected_set - present_set):
        issues.append(
            ValidationIssue(
                code="MISSING_FILENAME",
                message=f"Expected filename {filename!r} is missing from CSV.",
                field="filename",
            )
        )
    for filename in sorted(present_set - expected_set):
        if filename.lower() in {item.lower() for item in expected_set}:
            continue
        issues.append(
            ValidationIssue(
                code="EXTRA_FILENAME",
                message=f"CSV contains unexpected filename {filename!r}.",
                field="filename",
            )
        )


def _filenames_from_json(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return _filenames_from_records(raw)
    if isinstance(raw, Mapping):
        for key in ("filenames", "files", "samples", "records"):
            value = raw.get(key)
            if isinstance(value, list):
                return _filenames_from_records(value)
    raise ValueError("JSON manifest must contain a list of filenames or sample records.")


def _filenames_from_records(records: Iterable[Any]) -> list[str]:
    filenames: list[str] = []
    for position, item in enumerate(records):
        if isinstance(item, str):
            filenames.append(item)
        elif isinstance(item, Mapping):
            value = item.get("filename", item.get("relative_path"))
            if not isinstance(value, str):
                raise ValueError(f"Manifest record {position} lacks filename or relative_path.")
            filenames.append(value)
        else:
            raise ValueError(f"Manifest record {position} must be a string or object.")
    return _validate_expected_filenames(filenames)


def _filenames_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV manifest must not be empty.")
    header = rows[0]
    lower_header = [column.strip().lower() for column in header]
    if "filename" in lower_header or "relative_path" in lower_header:
        key = "filename" if "filename" in lower_header else "relative_path"
        index = lower_header.index(key)
        return _validate_expected_filenames([row[index] for row in rows[1:] if row])
    return _validate_expected_filenames([row[0] for row in rows if row])
