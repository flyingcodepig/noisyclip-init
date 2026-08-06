from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(ValueError):
    """Raised when configuration loading or inheritance is invalid."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalars are replaced as a whole."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _resolve_env(value: Any, *, strict: bool) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item, strict=strict) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item, strict=strict) for item in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if strict:
                raise ConfigError(f"required environment variable is missing: {name}")
    return value


def load_config(
    path: str | Path,
    *,
    strict_env: bool = False,
    _stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load YAML, recursively resolve `inherits`, and optionally resolve env vars."""
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path in _stack:
        cycle = " -> ".join(str(item) for item in (*_stack, resolved_path))
        raise ConfigError(f"configuration inheritance cycle: {cycle}")
    if not resolved_path.is_file():
        raise ConfigError(f"configuration file not found: {resolved_path}")

    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"top-level YAML must be a mapping: {resolved_path}")
    parent_ref = raw.pop("inherits", None)
    if parent_ref is None:
        merged: dict[str, Any] = raw
    else:
        if not isinstance(parent_ref, str):
            raise ConfigError("inherits must be a single relative path string")
        parent = (resolved_path.parent / parent_ref).resolve()
        merged = deep_merge(
            load_config(parent, strict_env=False, _stack=(*_stack, resolved_path)), raw
        )
    return _resolve_env(merged, strict=strict_env)


def canonical_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, child))
    else:
        result[prefix] = value
    return result

