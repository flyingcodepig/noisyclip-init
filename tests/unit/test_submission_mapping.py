"""Unit tests for strict class mapping validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noisyclip.submission.mapping import MappingError, load_class_mapping, mapping_digest


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


def test_loads_agent_a_wrapped_mapping_and_checks_digest(tmp_path: Path) -> None:
    """Submission mapping consumes the metadata wrapper emitted by data audit."""

    class_to_idx = {"0001": 0, "0002": 1}
    digest = mapping_digest(class_to_idx)
    path = write_json(
        tmp_path / "class_to_idx.json",
        {
            "schema_version": 1,
            "class_to_idx": class_to_idx,
            "idx_to_class": {"0": "0001", "1": "0002"},
            "digest": digest,
        },
    )

    mapping = load_class_mapping(path)

    assert mapping.class_to_idx == class_to_idx
    assert mapping.digest == digest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["digest"] = "0" * 64
    write_json(path, payload)
    with pytest.raises(MappingError, match="digest"):
        load_class_mapping(path)


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


def test_swapped_mapping_order_changes_digest(tmp_path: Path) -> None:
    """Swapping internal indices is detectable through the canonical digest."""

    original = {"0001": 0, "0002": 1}
    swapped = {"0001": 1, "0002": 0}
    runtime_mapping = load_class_mapping(write_json(tmp_path / "class_to_idx.json", original))

    assert mapping_digest(original) != mapping_digest(swapped)
    with pytest.raises(MappingError, match="differs|mismatch"):
        runtime_mapping.require_digest(mapping_digest(swapped), field="model mapping_digest")
