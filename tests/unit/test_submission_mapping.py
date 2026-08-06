"""Unit tests for strict class mapping validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noisyclip.submission.mapping import MappingError, load_class_mapping, mapping_digest
from noisyclip.submission.package import load_exported_model_package


def write_json(path: Path, payload: object) -> Path:
    """Write UTF-8 JSON test data and return its path."""

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_mapping_provides_index_lookup_and_digest(tmp_path: Path) -> None:
    """A gapless mapping returns four-digit ids in internal-index order."""

    path = write_json(tmp_path / "class_to_idx.json", {"0002": 1, "0001": 0})

    mapping = load_class_mapping(path)

    assert mapping.index_to_class_id == ("0001", "0002")
    assert mapping.class_id_for_index(1) == "0002"
    assert mapping.digest == mapping_digest({"0001": 0, "0002": 1})


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"1": 0}, "four"),
        ({"0001": 1}, "missing"),
        ({"0001": 0, "0002": 0}, "both"),
        ({"0001": -1}, "non-negative"),
        ({"0001": True}, "integer"),
    ],
)
def test_invalid_mapping_fails_fast(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    """Malformed ids and non-gapless internal indices are rejected."""

    path = write_json(tmp_path / "class_to_idx.json", payload)

    with pytest.raises(MappingError, match=message):
        load_class_mapping(path)


def test_swapped_mapping_order_changes_digest_and_package_check_fails(tmp_path: Path) -> None:
    """Swapping internal indices is detectable through digest metadata."""

    original = {"0001": 0, "0002": 1}
    swapped = {"0001": 1, "0002": 0}
    package_path = write_json(
        tmp_path / "model.json",
        {
            "artifact_type": "noisyclip_single_model_export",
            "models": [{"role": "student"}],
            "class_to_idx": swapped,
            "mapping_digest": mapping_digest(swapped),
            "preprocess": {"center_crop": 224, "test_time_augmentation": False},
            "expected_filenames": ["a.jpg"],
            "predictions": [0],
        },
    )
    runtime_mapping = load_class_mapping(write_json(tmp_path / "class_to_idx.json", original))
    package = load_exported_model_package(package_path)

    assert mapping_digest(original) != mapping_digest(swapped)
    with pytest.raises(MappingError, match="differs|mismatch"):
        package.require_compatible_mapping(runtime_mapping)
