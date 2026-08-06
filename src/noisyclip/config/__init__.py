"""Strict configuration schema, loading, and invariant checks."""

from noisyclip.config.loader import load_config, load_config_from_mapping, write_resolved_config
from noisyclip.config.schema import ProjectConfig

__all__ = ["ProjectConfig", "load_config", "load_config_from_mapping", "write_resolved_config"]
