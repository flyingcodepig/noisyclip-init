"""Configuration loading utilities with strict unknown-field validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from noisyclip.config.invariants import validate_config_invariants
from noisyclip.config.schema import ProjectConfig


def load_config(path: Path | str) -> ProjectConfig:
    """Load a YAML config file into an immutable `ProjectConfig`.

    Environment variables inside string values are expanded with
    `os.path.expandvars`. Unknown top-level or nested fields raise Pydantic
    validation errors. Missing files, non-mapping YAML documents, and YAML parse
    errors are surfaced to callers.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        msg = f"Config root must be a mapping, got {type(raw).__name__}."
        raise TypeError(msg)
    return load_config_from_mapping(raw)


def load_config_from_mapping(raw: Mapping[str, Any]) -> ProjectConfig:
    """Validate a mapping as `ProjectConfig` after environment expansion."""

    expanded = _expand_env(raw)
    config = ProjectConfig.model_validate(expanded)
    validate_config_invariants(config)
    return config


def write_resolved_config(config: ProjectConfig, destination: Path | str) -> Path:
    """Write a validated config as deterministic YAML and return the path."""

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            handle,
            allow_unicode=True,
            sort_keys=True,
        )
    return output_path


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value
