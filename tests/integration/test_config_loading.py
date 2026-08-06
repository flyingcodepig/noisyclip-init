"""Integration smoke tests for strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from noisyclip.config.loader import load_config, write_resolved_config

VALID_CONFIG_TEXT = """
experiment: {}
paths: {}
data: {}
model: {}
noise: {}
loss: {}
trainer: {}
evaluation: {}
tracking: {}
submission: {}
"""


def test_yaml_config_load_and_resolved_write(tmp_path: Path) -> None:
    """A minimal strict YAML config loads and can be written back deterministically."""

    config_path = tmp_path / "base.yaml"
    config_path.write_text(VALID_CONFIG_TEXT, encoding="utf-8")

    config = load_config(config_path)
    resolved_path = write_resolved_config(config, tmp_path / "resolved_config.yaml")

    assert config.model_dump()["experiment"] == {}
    assert resolved_path.exists()
    assert "submission" in resolved_path.read_text(encoding="utf-8")


def test_yaml_config_unknown_field_fails(tmp_path: Path) -> None:
    """Unexpected YAML fields fail during Pydantic validation."""

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(VALID_CONFIG_TEXT + "extra: {}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)
