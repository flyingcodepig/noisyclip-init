"""Stable manifest serialization, validation, and digest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from noisyclip.data.records import SampleRecord

SplitName = Literal["train", "val", "test"]
VALID_SPLITS: set[str] = {"train", "val", "test"}


class ManifestError(ValueError):
    """Raised when a manifest row violates the public `SampleRecord` contract."""


def canonical_json_digest(payload: Any) -> str:
    """Return a SHA256 digest for canonical JSON data.

    Args:
        payload: JSON-serializable data. Dict keys are sorted and separators are
            compact to make the digest stable across processes.

    Returns:
        Lowercase hexadecimal SHA256 digest.

    Raises:
        TypeError: If `payload` is not JSON serializable.
    """

    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def make_sample_id(relative_path: str) -> str:
    """Create a stable sample ID from a relative POSIX path.

    Args:
        relative_path: Path relative to an official data root. It must not be
            absolute and must not contain `..` components.

    Returns:
        SHA256 digest of the normalized relative path.

    Raises:
        ManifestError: If the path is absolute or escapes its data root.
    """

    normalized = normalize_relative_path(relative_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_relative_path(relative_path: str) -> str:
    """Validate and normalize a manifest relative path.

    Args:
        relative_path: POSIX-style path relative to a train or test root.

    Returns:
        A normalized POSIX path string with no absolute prefix.

    Raises:
        ManifestError: If the path is absolute, empty, drive-qualified, or uses
            `..` to escape the data root.
    """

    candidate = relative_path.replace("\\", "/")
    pure = PurePosixPath(candidate)
    if not candidate or pure.is_absolute() or ":" in pure.parts[0]:
        raise ManifestError(f"relative_path must be relative, got: {relative_path}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ManifestError(f"relative_path contains an illegal component: {relative_path}")
    return pure.as_posix()


def validate_record(record: SampleRecord) -> None:
    """Validate one manifest row for split, IDs, dimensions, and label semantics.

    Args:
        record: A `SampleRecord` row. Image tensors are not stored in manifests;
            width and height are positive pixel counts when `readable` is true.

    Raises:
        ManifestError: If split, label fields, dimensions, hashes, or relative
            path semantics are invalid.
    """

    normalize_relative_path(record.relative_path)
    if record.split not in VALID_SPLITS:
        raise ManifestError(f"Illegal split for {record.sample_id}: {record.split}")
    if make_sample_id(record.relative_path) != record.sample_id:
        raise ManifestError(f"sample_id does not match relative_path: {record.relative_path}")
    if record.readable:
        if record.width is None or record.height is None or record.width <= 0 or record.height <= 0:
            raise ManifestError(f"Readable sample has invalid dimensions: {record.sample_id}")
    if record.file_sha256 is not None:
        if len(record.file_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in record.file_sha256
        ):
            raise ManifestError(f"Invalid file_sha256 for {record.sample_id}")
    if record.split == "test":
        if record.class_id is not None or record.target is not None:
            raise ManifestError(f"Test sample must not have class_id or target: {record.sample_id}")
    else:
        if record.class_id is None or record.target is None:
            raise ManifestError(
                f"{record.split} sample requires class_id and target: {record.sample_id}"
            )
        if record.target < 0:
            raise ManifestError(f"target must be non-negative for {record.sample_id}")


def validate_records(records: list[SampleRecord]) -> None:
    """Validate a full manifest and reject duplicate IDs or relative paths.

    Args:
        records: Manifest rows to validate. The list may contain any legal
            split, but each `sample_id` and `relative_path` must be unique.

    Raises:
        ManifestError: If any row is invalid or duplicates another row.
    """

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for record in records:
        validate_record(record)
        if record.sample_id in seen_ids:
            raise ManifestError(f"Duplicate sample_id: {record.sample_id}")
        if record.relative_path in seen_paths:
            raise ManifestError(f"Duplicate relative_path: {record.relative_path}")
        seen_ids.add(record.sample_id)
        seen_paths.add(record.relative_path)


def sorted_records(records: list[SampleRecord]) -> list[SampleRecord]:
    """Return manifest rows in a stable split/path/sample order.

    Args:
        records: Manifest rows with legal `SampleRecord` fields.

    Returns:
        A new list sorted by split, relative path, and sample ID.

    Raises:
        ManifestError: If any input record is invalid.
    """

    validate_records(records)
    return sorted(records, key=lambda row: (row.split, row.relative_path, row.sample_id))


def manifest_payload(records: list[SampleRecord]) -> dict[str, Any]:
    """Convert records to a deterministic JSON manifest payload.

    Args:
        records: Manifest rows to serialize.

    Returns:
        A dict containing `schema_version`, sorted `records`, and a manifest
        digest over the serialized row semantics.

    Raises:
        ManifestError: If any record violates the manifest contract.
    """

    rows = [asdict(record) for record in sorted_records(records)]
    digest = canonical_json_digest(rows)
    return {"schema_version": 1, "records": rows, "manifest_digest": digest}


def manifest_digest(records: list[SampleRecord]) -> str:
    """Digest the stable semantic content of a manifest.

    Args:
        records: Manifest rows.

    Returns:
        Lowercase SHA256 digest over canonical row JSON.

    Raises:
        ManifestError: If any record is invalid.
    """

    return str(manifest_payload(records)["manifest_digest"])


def data_digest(records: list[SampleRecord], class_to_idx: dict[str, int]) -> str:
    """Digest data bytes and class mapping used for an audited dataset.

    Args:
        records: Train, validation, and test manifest rows. File hashes may be
            `None` only when hashing is disabled.
        class_to_idx: Stable mapping from four-digit class IDs to targets.

    Returns:
        Lowercase SHA256 digest. Changing a file hash, relative path, split, or
        class mapping changes the digest.

    Raises:
        ManifestError: If records are invalid.
    """

    rows = [asdict(record) for record in sorted_records(records)]
    return canonical_json_digest({"class_to_idx": class_to_idx, "records": rows})


def write_manifest(records: list[SampleRecord], destination: Path | str) -> Path:
    """Write records as a deterministic JSON manifest.

    Args:
        records: Manifest rows to write.
        destination: Output path. Parent directories are created.

    Returns:
        The written path.

    Raises:
        ManifestError: If records are invalid or include absolute paths.
        OSError: If the destination cannot be written.
    """

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload(records)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_manifest(path: Path | str) -> list[SampleRecord]:
    """Read a deterministic JSON manifest into `SampleRecord` objects.

    Args:
        path: Manifest file produced by `write_manifest`. A plain list of rows
            is also accepted for backward-compatible tests.

    Returns:
        Sorted `SampleRecord` rows with identical field semantics after
        serialization and reload.

    Raises:
        ManifestError: If JSON shape, digest, or row semantics are invalid.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("records")
        expected_digest = raw.get("manifest_digest")
    else:
        rows = raw
        expected_digest = None
    if not isinstance(rows, list):
        raise ManifestError("Manifest must contain a list of records.")

    records = [SampleRecord(**row) for row in rows]
    ordered = sorted_records(records)
    if expected_digest is not None and expected_digest != manifest_digest(ordered):
        raise ManifestError("Manifest digest does not match record content.")
    return ordered
