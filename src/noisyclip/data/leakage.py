"""Leakage checks across train, validation, and test manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from noisyclip.data.records import SampleRecord


class LeakageError(ValueError):
    """Raised when test data leaks into training or validation flows."""


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Summary of leakage intersections between split manifests.

    Args:
        path_intersections: Relative paths shared between test and train/val.
        filename_intersections: Basenames shared between test and train/val.
        hash_intersections: File SHA256 values shared between test and train/val.
        root_issue: Root-boundary violation such as identical or nested roots.

    Raises:
        LeakageError: `assert_no_leakage` raises when any field is non-empty.
    """

    path_intersections: tuple[str, ...]
    filename_intersections: tuple[str, ...]
    hash_intersections: tuple[str, ...]
    root_issue: str | None = None

    @property
    def ok(self) -> bool:
        """Return true when no leakage evidence was found."""

        return not (
            self.path_intersections
            or self.filename_intersections
            or self.hash_intersections
            or self.root_issue
        )


def check_root_boundaries(train_root: Path | str, test_root: Path | str) -> str | None:
    """Check that train and test roots resolve to distinct, non-nested paths.

    Args:
        train_root: Official training root.
        test_root: Official test root.

    Returns:
        `None` when the roots are distinct and neither contains the other;
        otherwise a human-readable issue string.

    Raises:
        OSError: If path resolution fails.
    """

    train = Path(train_root).resolve()
    test = Path(test_root).resolve()
    train_text = str(train).casefold()
    test_text = str(test).casefold()
    if train_text == test_text:
        return f"train_root and test_root resolve to the same directory: {train}"
    if _is_relative_to(train, test):
        return f"train_root is inside test_root: {train} <= {test}"
    if _is_relative_to(test, train):
        return f"test_root is inside train_root: {test} <= {train}"
    return None


def check_manifest_leakage(
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    test_records: list[SampleRecord],
    *,
    train_root: Path | str | None = None,
    test_root: Path | str | None = None,
) -> LeakageReport:
    """Find path, filename, and hash intersections with the test split.

    Args:
        train_records: Training manifest rows with targets.
        val_records: Validation manifest rows with targets.
        test_records: Test manifest rows without targets.
        train_root: Optional root for nested/same-directory boundary checks.
        test_root: Optional root for nested/same-directory boundary checks.

    Returns:
        `LeakageReport` containing sorted intersections.

    Raises:
        LeakageError: If any test record carries labels or any train/val record
            is marked as `test`.
    """

    labeled = train_records + val_records
    for record in labeled:
        if record.split == "test":
            raise LeakageError(f"Test sample entered train/val records: {record.sample_id}")
    for record in test_records:
        if record.split != "test" or record.target is not None or record.class_id is not None:
            raise LeakageError(f"Test manifest record has label fields: {record.sample_id}")

    labeled_paths = {record.relative_path for record in labeled}
    test_paths = {record.relative_path for record in test_records}
    labeled_names = {Path(record.relative_path).name for record in labeled}
    test_names = {Path(record.relative_path).name for record in test_records}
    labeled_hashes = {record.file_sha256 for record in labeled if record.file_sha256 is not None}
    test_hashes = {record.file_sha256 for record in test_records if record.file_sha256 is not None}

    root_issue = None
    if train_root is not None and test_root is not None:
        root_issue = check_root_boundaries(train_root, test_root)

    return LeakageReport(
        path_intersections=tuple(sorted(labeled_paths & test_paths)),
        filename_intersections=tuple(sorted(labeled_names & test_names)),
        hash_intersections=tuple(sorted(labeled_hashes & test_hashes)),
        root_issue=root_issue,
    )


def assert_no_leakage(report: LeakageReport) -> None:
    """Fail fast if a leakage report contains any intersection.

    Args:
        report: Result from `check_manifest_leakage`.

    Raises:
        LeakageError: If train/val and test share paths, basenames, hashes, or
            invalid root boundaries.
    """

    if report.ok:
        return
    issues: list[str] = []
    if report.root_issue:
        issues.append(report.root_issue)
    if report.path_intersections:
        issues.append(f"path intersections: {list(report.path_intersections)}")
    if report.filename_intersections:
        issues.append(f"filename intersections: {list(report.filename_intersections)}")
    if report.hash_intersections:
        issues.append(f"hash intersections: {list(report.hash_intersections)}")
    raise LeakageError("; ".join(issues))


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
