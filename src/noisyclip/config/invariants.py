"""Cross-field configuration invariants for preflight safety."""

from __future__ import annotations

from noisyclip.config.schema import ProjectConfig


def validate_config_invariants(config: ProjectConfig) -> None:
    """Validate cross-field safety rules for a loaded `ProjectConfig`.

    F01 has no concrete path fields yet, so this function is intentionally a
    no-op after receiving a fully validated config. Future agents must add
    rules here before enabling training or data access.
    """

    _ = config
