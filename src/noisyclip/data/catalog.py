"""Class-directory scanning and stable class mapping utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from noisyclip.data.manifests import canonical_json_digest


class CatalogError(ValueError):
    """Raised when class directories or mappings violate the data contract."""


@dataclass(frozen=True, slots=True)
class ClassCatalog:
    """Stable bidirectional mapping between raw class IDs and internal targets.

    Args:
        class_to_idx: Mapping from four-digit class IDs to contiguous internal
            indices in `[0, C)`.
        idx_to_class: Reverse mapping from each internal index to the original
            four-digit class ID.
        digest: SHA256 digest of the canonical mapping JSON.

    Raises:
        CatalogError: Raised by factory functions when IDs are duplicated,
            malformed, non-bijective, or not contiguous on the internal side.
    """

    class_to_idx: Mapping[str, int]
    idx_to_class: Mapping[int, str]
    digest: str


def build_class_catalog(
    class_ids: list[str],
    *,
    class_id_regex: str = r"^[0-9]{4}$",
    expected_num_classes: int | None = None,
) -> ClassCatalog:
    """Build a deterministic class catalog from raw class IDs.

    Args:
        class_ids: Raw directory names. Each value must match
            `class_id_regex`, typically `^[0-9]{4}$`.
        class_id_regex: Full-match regular expression for legal IDs.
        expected_num_classes: Optional configured class count. The competition
            default is 500, but tests may set smaller synthetic values.

    Returns:
        A `ClassCatalog` whose internal indices are `[0, C)` and whose reverse
        mapping preserves leading zeroes such as `0001`.

    Raises:
        CatalogError: If any ID is malformed, duplicated, or if the configured
            expected class count does not match the discovered count.
    """

    pattern = re.compile(class_id_regex)
    duplicates = sorted({class_id for class_id in class_ids if class_ids.count(class_id) > 1})
    if duplicates:
        raise CatalogError(f"Duplicate class_id values: {duplicates}")

    invalid = sorted(class_id for class_id in class_ids if not pattern.fullmatch(class_id))
    if invalid:
        raise CatalogError(f"Illegal class_id directory names: {invalid}")

    sorted_ids = sorted(class_ids)
    if expected_num_classes is not None and len(sorted_ids) != expected_num_classes:
        raise CatalogError(
            f"Expected {expected_num_classes} classes, found {len(sorted_ids)} under train_root."
        )

    class_to_idx = {class_id: index for index, class_id in enumerate(sorted_ids)}
    return catalog_from_mapping(class_to_idx, class_id_regex=class_id_regex)


def catalog_from_mapping(
    class_to_idx: Mapping[str, int],
    *,
    class_id_regex: str = r"^[0-9]{4}$",
) -> ClassCatalog:
    """Validate and freeze an existing class mapping.

    Args:
        class_to_idx: Mapping from class IDs to integer targets. Targets must be
            unique and contiguous in `[0, C)`.
        class_id_regex: Full-match regular expression for legal class IDs.

    Returns:
        A `ClassCatalog` with a canonical SHA256 digest.

    Raises:
        CatalogError: If the class IDs, targets, or reverse mapping are invalid.
    """

    pattern = re.compile(class_id_regex)
    invalid_ids = sorted(class_id for class_id in class_to_idx if not pattern.fullmatch(class_id))
    if invalid_ids:
        raise CatalogError(f"Illegal class_id values in mapping: {invalid_ids}")

    values = list(class_to_idx.values())
    if any(not isinstance(value, int) for value in values):
        raise CatalogError("All class_to_idx targets must be integers.")
    if len(set(values)) != len(values):
        raise CatalogError("class_to_idx targets must be unique.")
    expected_values = list(range(len(values)))
    if sorted(values) != expected_values:
        raise CatalogError(f"class_to_idx targets must be contiguous {expected_values}.")

    frozen_mapping = dict(sorted(class_to_idx.items()))
    idx_to_class = {index: class_id for class_id, index in frozen_mapping.items()}
    if len(idx_to_class) != len(frozen_mapping):
        raise CatalogError("class_to_idx is not bijective.")

    digest = canonical_json_digest({"class_to_idx": frozen_mapping})
    return ClassCatalog(class_to_idx=frozen_mapping, idx_to_class=idx_to_class, digest=digest)


def scan_class_catalog(
    train_root: Path | str,
    *,
    class_id_regex: str = r"^[0-9]{4}$",
    expected_num_classes: int | None = None,
) -> ClassCatalog:
    """Scan one official train root for four-digit class directories.

    Args:
        train_root: Official training root containing one directory per class.
        class_id_regex: Full-match regular expression for class directory names.
        expected_num_classes: Optional configured class count. A mismatch fails
            before any manifest is written.

    Returns:
        Deterministic `ClassCatalog` sorted lexicographically by class ID.

    Raises:
        CatalogError: If the root is missing, contains illegal directories, or
            the configured class count does not match.
    """

    root = Path(train_root)
    if not root.is_dir():
        raise CatalogError(f"train_root is not a directory: {root}")

    class_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    files = sorted(path.name for path in root.iterdir() if path.is_file())
    if files:
        raise CatalogError(f"Unexpected files in train_root: {files}")
    return build_class_catalog(
        class_dirs,
        class_id_regex=class_id_regex,
        expected_num_classes=expected_num_classes,
    )


def write_class_mapping(catalog: ClassCatalog, destination: Path | str) -> Path:
    """Write `class_to_idx.json` with a stable digest payload.

    Args:
        catalog: Validated class catalog.
        destination: Output JSON path under a derived artifact directory.

    Returns:
        The written path.

    Raises:
        OSError: If the destination cannot be created or written.
    """

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "class_to_idx": dict(catalog.class_to_idx),
        "idx_to_class": {str(key): value for key, value in sorted(catalog.idx_to_class.items())},
        "digest": catalog.digest,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
