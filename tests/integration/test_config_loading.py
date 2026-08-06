"""Integration smoke tests for strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from noisyclip.config.loader import ConfigInheritanceError, load_config, write_resolved_config

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

    assert config.experiment.name == "base"
    assert config.model.backbone.name == "ViT-B/32"
    assert resolved_path.exists()
    assert "submission" in resolved_path.read_text(encoding="utf-8")


def test_yaml_config_unknown_field_fails(tmp_path: Path) -> None:
    """Unexpected YAML fields fail during Pydantic validation."""

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(VALID_CONFIG_TEXT + "extra: {}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_declared_b0_config_resolves_and_validates() -> None:
    """The checked-in B0 overlay resolves through its parent and strict schema."""

    root = Path(__file__).resolve().parents[2]
    config = load_config(
        root
        / "init_build"
        / "04_scripts_and_configs"
        / "configs"
        / "experiments"
        / "b0_frozen_linear.yaml"
    )

    assert config.experiment.name == "b0_frozen_linear"
    assert config.data.expected_num_classes == 500
    assert config.model.backbone.name == "ViT-B/32"
    assert config.model.backbone.pretrained == "openai"
    assert config.submission.include_header is False


def test_config_inheritance_cycle_is_rejected(tmp_path: Path) -> None:
    """Recursive config inheritance fails with an explicit cycle error."""

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("inherits: second.yaml\n", encoding="utf-8")
    second.write_text("inherits: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigInheritanceError, match="cycle"):
        load_config(first)
