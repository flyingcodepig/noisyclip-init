"""Strict class-id mapping for competition submissions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASS_ID_PATTERN = re.compile(r"^[0-9]{4}$")


class MappingError(ValueError):
    """Raised when a class mapping is malformed or conflicts with metadata."""


@dataclass(frozen=True, slots=True)
class ClassMapping:
    """Validated mapping between raw class ids and internal indices.

    Args:
        class_to_idx: Mapping from four-digit class ids shaped as strings matching
            `^[0-9]{4}$` to integer indices covering `[0, C)` exactly.
        index_to_class_id: Tuple where element `i` is the four-digit class id for
            internal prediction index `i`; shape is `[C]`.
        digest: SHA256 hex digest of the canonical `(class_id, index)` mapping.

    Raises:
        MappingError: Raised by constructors when ids are malformed, indices are
            duplicated, non-integral, negative, or do not cover `[0, C)`.
    """

    class_to_idx: Mapping[str, int]
    index_to_class_id: tuple[str, ...]
    digest: str

    @property
    def num_classes(self) -> int:
        """Return the class count `C` represented by indices `[0, C)`."""

        return len(self.index_to_class_id)

    def class_id_for_index(self, index: int) -> str:
        """Return the four-digit class id for one internal prediction index.

        Args:
            index: Integer class index in `[0, C)`.

        Returns:
            The original class id as a four-character digit string.

        Raises:
            MappingError: If `index` is not an integer or is outside `[0, C)`.
        """

        if isinstance(index, bool) or not isinstance(index, int):
            raise MappingError(f"Prediction index must be an integer, got {index!r}.")
        if index < 0 or index >= self.num_classes:
            raise MappingError(
                f"Prediction index {index} is outside valid range [0, {self.num_classes})."
            )
        return self.index_to_class_id[index]

    def require_digest(self, expected_digest: str | None, *, field: str) -> None:
        """Fail if `expected_digest` is present and differs from this mapping.

        Args:
            expected_digest: Metadata digest to compare against this mapping's
                stable SHA256 digest, or `None` when absent.
            field: Name of the metadata field used in error messages.

        Raises:
            MappingError: If `expected_digest` is empty or different.
        """

        if expected_digest is None:
            return
        if not isinstance(expected_digest, str) or not expected_digest:
            raise MappingError(f"{field} must be a non-empty string when provided.")
        if expected_digest != self.digest:
            raise MappingError(
                f"{field} mismatch: expected {expected_digest}, computed {self.digest}."
            )


def load_class_mapping(path: Path | str) -> ClassMapping:
    """Read and validate a JSON `class_to_idx` mapping.

    Args:
        path: UTF-8 JSON object whose keys are four-digit class ids and values are
            integer internal indices covering `[0, C)` with no gaps.

    Returns:
        A `ClassMapping` containing an index-to-class-id tuple of shape `[C]`.

    Raises:
        MappingError: If duplicate JSON keys, malformed ids, invalid indices, or
            index gaps are found.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """

    mapping_path = Path(path)
    text = mapping_path.read_text(encoding="utf-8")
    raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(raw, Mapping):
        raise MappingError(f"class mapping root must be a JSON object: {mapping_path}")
    return validate_class_mapping(raw)


def validate_class_mapping(mapping: Mapping[str, Any]) -> ClassMapping:
    """Validate a raw `class_id -> internal index` mapping.

    Args:
        mapping: Mapping with `C` entries. Keys must match `^[0-9]{4}$`; values
            must be non-negative integers that cover `[0, C)` exactly.

    Returns:
        A validated `ClassMapping` with stable digest and `[C]` index lookup.

    Raises:
        MappingError: If the mapping is empty, contains malformed class ids,
            duplicate indices, non-integer indices, negative values, or gaps.
    """

    if not mapping:
        raise MappingError("class_to_idx mapping must not be empty.")

    normalized: dict[str, int] = {}
    seen_indices: dict[int, str] = {}
    for class_id, index in mapping.items():
        if not isinstance(class_id, str) or CLASS_ID_PATTERN.fullmatch(class_id) is None:
            raise MappingError(
                f"class_id {class_id!r} must be a four-digit string matching ^[0-9]{{4}}$."
            )
        if isinstance(index, bool) or not isinstance(index, int):
            raise MappingError(f"class_to_idx[{class_id!r}] must be an integer index.")
        if index < 0:
            raise MappingError(f"class_to_idx[{class_id!r}] must be non-negative.")
        if index in seen_indices:
            other = seen_indices[index]
            raise MappingError(
                f"Internal index {index} is assigned to both {other!r} and {class_id!r}."
            )
        normalized[class_id] = index
        seen_indices[index] = class_id

    expected = set(range(len(normalized)))
    actual = set(seen_indices)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MappingError(
            "Internal indices must cover [0, C) with no gaps; "
            f"missing={missing}, out_of_range={extra}."
        )

    index_to_class_id = tuple(seen_indices[index] for index in range(len(normalized)))
    digest = _digest_validated_mapping(normalized)
    return ClassMapping(
        class_to_idx=dict(normalized), index_to_class_id=index_to_class_id, digest=digest
    )


def mapping_digest(mapping: Mapping[str, int] | ClassMapping) -> str:
    """Compute a stable SHA256 digest for a class mapping.

    Args:
        mapping: Either a raw `class_id -> index` mapping or an already validated
            `ClassMapping`; class ids must be four-digit strings and indices must
            be integers in `[0, C)`.

    Returns:
        A SHA256 hex string over canonical UTF-8 JSON sorted by class id.

    Raises:
        MappingError: If a raw mapping is malformed.
    """

    if isinstance(mapping, ClassMapping):
        return _digest_validated_mapping(mapping.class_to_idx)
    else:
        validated = validate_class_mapping(mapping)
        return validated.digest


def _digest_validated_mapping(mapping: Mapping[str, int]) -> str:
    canonical_pairs = [
        {"class_id": class_id, "index": mapping[class_id]} for class_id in sorted(mapping)
    ]
    payload = json.dumps(canonical_pairs, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def class_ids_from_mapping(mapping: ClassMapping) -> set[str]:
    """Return the valid four-digit class-id set for membership checks."""

    return set(mapping.class_to_idx)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise MappingError(f"Duplicate class_id key in mapping JSON: {key!r}.")
        seen.add(key)
        result[key] = value
    return result


def validate_unique_ids(values: Iterable[str], *, field: str) -> None:
    """Validate that string ids are unique.

    Args:
        values: Iterable of ids with shape `[N]`.
        field: Field name included in failure messages.

    Raises:
        MappingError: If any value appears more than once.
    """

    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise MappingError(f"{field} contains duplicate ids: {sorted(duplicates)}.")
