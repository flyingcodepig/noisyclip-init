"""Tests for class catalog scanning and stable mappings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noisyclip.data.catalog import (
    CatalogError,
    build_class_catalog,
    scan_class_catalog,
    write_class_mapping,
)


def test_non_contiguous_four_digit_class_ids_are_lexicographic() -> None:
    """Legal four-digit IDs need not be numerically contiguous."""

    catalog = build_class_catalog(["0010", "0001", "0420"], expected_num_classes=3)

    assert catalog.class_to_idx == {"0001": 0, "0010": 1, "0420": 2}
    assert catalog.idx_to_class[0] == "0001"
    assert len(catalog.digest) == 64


def test_illegal_class_directory_fails(tmp_path: Path) -> None:
    """Class directories must match the configured four-digit regex."""

    (tmp_path / "0001").mkdir()
    (tmp_path / "1").mkdir()

    with pytest.raises(CatalogError, match="Illegal class_id"):
        scan_class_catalog(tmp_path, expected_num_classes=2)


def test_expected_class_count_is_enforced(tmp_path: Path) -> None:
    """The competition default can require exactly 500 classes."""

    (tmp_path / "0001").mkdir()

    with pytest.raises(CatalogError, match="Expected 500"):
        scan_class_catalog(tmp_path, expected_num_classes=500)


def test_class_mapping_write_preserves_reverse_mapping(tmp_path: Path) -> None:
    """Written mapping keeps leading zeroes and stores the stable digest."""

    catalog = build_class_catalog(["0002", "0007"], expected_num_classes=2)
    path = write_class_mapping(catalog, tmp_path / "class_to_idx.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["class_to_idx"] == {"0002": 0, "0007": 1}
    assert payload["idx_to_class"] == {"0": "0002", "1": "0007"}
    assert payload["digest"] == catalog.digest
