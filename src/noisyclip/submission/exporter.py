"""Copy and validate exactly one exported model from a completed run."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from noisyclip.submission.package import load_exported_model_package


class ExportError(ValueError):
    """Raised when an export source or destination violates F02 rules."""


def export_single_model_from_run(run_dir: Path | str, output_path: Path | str) -> Path:
    """Copy one validated `.pt` student artifact without overwriting output."""

    source_dir = Path(run_dir)
    destination = Path(output_path)
    if not source_dir.is_dir():
        raise ExportError(f"run-dir does not exist or is not a directory: {source_dir}.")
    if destination.exists():
        raise ExportError(f"Refusing to overwrite existing export artifact: {destination}.")
    source = _find_export_model(source_dir)
    load_exported_model_package(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=".model.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if destination.exists():
            raise ExportError(f"Refusing to overwrite existing export artifact: {destination}.")
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    load_exported_model_package(destination)
    return destination


def _find_export_model(run_dir: Path) -> Path:
    candidates = (
        run_dir / "artifacts" / "model.pt",
        run_dir / "exported" / "model.pt",
        run_dir / "model.pt",
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise ExportError(
            f"Run must contain exactly one model.pt export in approved locations; found {existing}."
        )
    return existing[0]
