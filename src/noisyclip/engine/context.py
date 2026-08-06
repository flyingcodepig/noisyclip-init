"""Runtime context records shared by engine and CLI layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable metadata binding config, data, classes, seed, and run path."""

    run_id: str
    run_dir: Path
    seed: int
    num_classes: int
    class_to_idx: Mapping[str, int]
    config_digest: str
    data_digest: str


@dataclass(frozen=True, slots=True)
class EpochContext:
    """Immutable metadata for one training epoch lifecycle step."""

    run: RunContext
    epoch: int
    global_step: int
