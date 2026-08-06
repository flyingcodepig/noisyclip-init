"""Centralized run artifact path management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from noisyclip.utils.paths import fail_if_exists, resolve_under_root


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    """Manage every writable path below one run directory.

    Args:
        run_dir: Unique run directory. It is created during initialization.

    Raises:
        OSError: If the directory cannot be created.
    """

    run_dir: Path

    def __init__(self, run_dir: Path | str) -> None:
        object.__setattr__(self, "run_dir", Path(run_dir).resolve())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for relative in (
            "artifacts",
            "checkpoints",
            "data",
            "environment",
            "logs",
            "metrics",
            "sample_state",
        ):
            (self.run_dir / relative).mkdir(exist_ok=True)

    def path(self, relative_path: str) -> Path:
        """Return a path under this run directory.

        Args:
            relative_path: Relative artifact path such as
                `checkpoints/last.pt`.

        Returns:
            Absolute path under `run_dir`.

        Raises:
            PathSafetyError: If the path escapes the run directory.
        """

        return resolve_under_root(self.run_dir, relative_path)

    def checkpoint(self, name: str = "last.pt") -> Path:
        """Return a checkpoint path under `checkpoints/`."""

        return self.path(f"checkpoints/{name}")

    def metric(self, name: str) -> Path:
        """Return a metrics path under `metrics/`."""

        return self.path(f"metrics/{name}")

    def artifact(self, name: str) -> Path:
        """Return a final artifact path under `artifacts/`."""

        return self.path(f"artifacts/{name}")

    def sample_state_dir(self) -> Path:
        """Return the sample-state directory path."""

        return self.path("sample_state")


def create_run_dir(run_root: Path | str, run_id: str, *, fail_if_run_exists: bool = True) -> Path:
    """Create and return a unique run directory.

    Args:
        run_root: Parent directory for all runs.
        run_id: Non-empty run identifier.
        fail_if_run_exists: When true, existing run directories are refused.

    Returns:
        Absolute run directory.

    Raises:
        ValueError: If `run_id` is empty or contains path separators.
        FileExistsError: If the run already exists and overwrite is forbidden.
    """

    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"run_id must be a single path component, got {run_id!r}.")
    root = Path(run_root).resolve()
    run_dir = root / run_id
    if fail_if_run_exists:
        fail_if_exists(run_dir, field_name="run directory")
    run_dir.mkdir(parents=True, exist_ok=not fail_if_run_exists)
    return run_dir
