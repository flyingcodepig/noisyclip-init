"""Run identity, status markers, and manifest persistence."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from noisyclip.tracking.artifacts import create_run_dir
from noisyclip.utils.atomic import atomic_write_bytes

RunStatus = Literal[
    "CREATED",
    "PREFLIGHT_OK",
    "DATA_READY",
    "MODEL_READY",
    "TRAINING",
    "VALIDATING",
    "CHECKPOINTED",
    "COMPLETED",
    "FAILED",
]


def generate_run_id(prefix: str = "run") -> str:
    """Return a unique run ID safe for one directory name.

    Args:
        prefix: Non-empty ASCII-ish label placed before timestamp and random
            suffix.

    Returns:
        Run ID with no path separators.

    Raises:
        ValueError: If `prefix` is empty or path-like.
    """

    if not prefix or Path(prefix).name != prefix:
        raise ValueError("run_id prefix must be a single non-empty path component.")
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


class RunManifest:
    """Persist run metadata and DONE/FAILED status markers.

    Args:
        run_dir: Existing run directory.
        metadata: JSON-serializable metadata written to `manifest.json`.
    """

    def __init__(self, run_dir: Path | str, metadata: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.metadata = dict(metadata)
        self.status: RunStatus = "CREATED"

    @classmethod
    def create(
        cls,
        *,
        run_root: Path | str,
        run_id: str,
        metadata: Mapping[str, Any],
        fail_if_run_exists: bool = True,
    ) -> RunManifest:
        """Create a unique run directory and manifest instance."""

        run_dir = create_run_dir(run_root, run_id, fail_if_run_exists=fail_if_run_exists)
        manifest = cls(run_dir, metadata)
        manifest.write()
        return manifest

    def transition(self, status: RunStatus, *, extra: Mapping[str, Any] | None = None) -> None:
        """Update manifest status and persist it.

        Args:
            status: New state-machine status.
            extra: Optional JSON-serializable metadata merged into the manifest.

        Raises:
            TypeError: If extra metadata is not JSON-serializable.
        """

        self.status = status
        if extra:
            self.metadata.update(dict(extra))
        self.write()

    def mark_done(self) -> Path:
        """Create the `DONE` marker after successful completion.

        Returns:
            Path to the `DONE` marker.
        """

        self.transition("COMPLETED")
        marker = self.run_dir / "DONE"
        atomic_write_bytes(marker, b"done\n", overwrite=False)
        return marker

    def mark_failed(self, reason: str, *, stage: str | None = None) -> Path:
        """Create the `FAILED` marker with a diagnostic reason.

        Args:
            reason: Human-readable failure reason.
            stage: Optional last completed or active stage.

        Returns:
            Path to the `FAILED` marker.
        """

        self.transition("FAILED", extra={"failure_reason": reason, "failure_stage": stage})
        marker = self.run_dir / "FAILED"
        payload = json.dumps({"reason": reason, "stage": stage}, sort_keys=True).encode("utf-8")
        atomic_write_bytes(marker, payload + b"\n", overwrite=True)
        return marker

    def write(self) -> Path:
        """Atomically write `manifest.json` with status and metadata."""

        payload = {
            "status": self.status,
            "metadata": self.metadata,
        }
        data = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        path = self.run_dir / "manifest.json"
        atomic_write_bytes(path, data + b"\n", overwrite=True)
        _fsync_parent(path)
        return path


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
