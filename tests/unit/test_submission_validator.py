"""Unit tests for strict prediction CSV validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from noisyclip.submission.mapping import validate_class_mapping
from noisyclip.submission.validator import validate_submission_csv


def write_csv(path: Path, text: str) -> Path:
    """Write raw CSV text for validator tests."""

    path.write_text(text, encoding="utf-8", newline="")
    return path


def codes(path: Path, expected: list[str]) -> set[str]:
    """Return validation issue codes for a temporary CSV."""

    mapping = validate_class_mapping({"0001": 0, "0002": 1})
    return {issue.code for issue in validate_submission_csv(path, expected, mapping).issues}


def test_valid_headerless_csv_passes(tmp_path: Path) -> None:
    """A normal two-column headerless CSV with four-digit ids validates."""

    csv_path = write_csv(tmp_path / "pred_results.csv", "a.jpg,0001\nb.JPG,0002\n")
    mapping = validate_class_mapping({"0001": 0, "0002": 1})

    report = validate_submission_csv(csv_path, ["a.jpg", "b.JPG"], mapping)

    assert report.valid
    assert report.row_count == 2
    assert report.issues == ()


@pytest.mark.parametrize(
    "text, expected_code",
    [
        ("filename,class_id\na.jpg,0001\n", "HEADER_ROW"),
        ("a.jpg,0001,extra\n", "COLUMN_COUNT"),
        ("a.jpg\n", "COLUMN_COUNT"),
        ("a.jpg,0001\n\n", "EMPTY_ROW"),
    ],
)
def test_csv_shape_errors_fail(tmp_path: Path, text: str, expected_code: str) -> None:
    """Header rows, wrong column counts, and empty rows are rejected."""

    csv_path = write_csv(tmp_path / "pred_results.csv", text)

    assert expected_code in codes(csv_path, ["a.jpg"])


@pytest.mark.parametrize(
    "text, expected, expected_code",
    [
        ("a.jpg,1\n", ["a.jpg"], "CLASS_ID_FORMAT"),
        ("a.jpg,9999\n", ["a.jpg"], "UNKNOWN_CLASS_ID"),
        ("a.jpg,0001\na.jpg,0002\n", ["a.jpg"], "DUPLICATE_FILENAME"),
        ("a.jpg,0001\n", ["a.jpg", "b.jpg"], "MISSING_FILENAME"),
        ("a.jpg,0001\nextra.jpg,0001\n", ["a.jpg"], "EXTRA_FILENAME"),
        ("A.JPG,0001\n", ["a.jpg"], "FILENAME_CASE_MISMATCH"),
    ],
)
def test_content_errors_fail(
    tmp_path: Path,
    text: str,
    expected: list[str],
    expected_code: str,
) -> None:
    """Class id and exact filename set violations are rejected."""

    csv_path = write_csv(tmp_path / "pred_results.csv", text)

    assert expected_code in codes(csv_path, expected)
