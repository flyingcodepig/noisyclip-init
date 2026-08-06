"""Configuration loading with recursive inheritance and strict validation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from noisyclip.config.invariants import validate_config_invariants
from noisyclip.config.schema import ProjectConfig

_OMEGACONF_ENV = re.compile(r"^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigInheritanceError(ValueError):
    """Raised for missing parents, invalid `inherits`, or inheritance cycles."""


def load_config(path: Path | str) -> ProjectConfig:
    """Load, inherit, expand, validate, and freeze a YAML configuration.

    `inherits` is a relative path to one parent YAML file. Parent mappings are
    recursively deep-merged, while lists and scalar values are replaced as a
    whole. OmegaConf-style exact environment references are resolved when the
    variable exists and remain explicit placeholders otherwise; preflight is
    responsible for requiring path variables before data access.
    """

    config_path = Path(path).expanduser().resolve()
    raw = _load_inherited_mapping(config_path, stack=())
    return load_config_from_mapping(raw)


def load_config_from_mapping(raw: Mapping[str, Any]) -> ProjectConfig:
    """Validate one already-resolved mapping as an immutable `ProjectConfig`."""

    expanded = _expand_env(raw)
    config = ProjectConfig.model_validate(expanded)
    validate_config_invariants(config)
    return config


def write_resolved_config(config: ProjectConfig, destination: Path | str) -> Path:
    """Write a validated config as deterministic YAML and return its path."""

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


def _load_inherited_mapping(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigInheritanceError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigInheritanceError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        msg = f"Config root must be a mapping, got {type(raw).__name__}: {path}"
        raise TypeError(msg)

    current = dict(raw)
    parent_ref = current.pop("inherits", None)
    if parent_ref is None:
        return current
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise ConfigInheritanceError(f"`inherits` must be a relative path string: {path}")

    parent_path = (path.parent / parent_ref).resolve()
    parent = _load_inherited_mapping(parent_path, stack=(*stack, path))
    return _deep_merge(parent, current)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = _OMEGACONF_ENV.fullmatch(value)
        if match:
            return os.environ.get(match.group(1), value)
        return os.path.expandvars(value)
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    return value
