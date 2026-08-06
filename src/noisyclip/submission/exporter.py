"""Compatible export wrapper that refuses multi-model artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from noisyclip.submission.package import load_exported_model_package


class ExportError(ValueError):
    """Raised when an export source or destination violates F02 rules."""


def export_single_model_from_run(run_dir: Path | str, output_path: Path | str) -> Path:
    """Copy one compatible single-model export package out of a run directory.

    Args:
        run_dir: Run directory containing `export_metadata.json` either at the
            root or under `artifacts/`. This is a compatibility surface until
            Agent B exposes a concrete `StudentModel.export_single_model` loader.
        output_path: Destination JSON package path. Existing files are never
            overwritten.

    Returns:
        The written package path.

    Raises:
        ExportError: If `run_dir` is missing, no compatible export metadata is
            present, or `output_path` already exists.
        ValueError: If metadata describes teacher, ensemble, or multiple models.
        OSError, json.JSONDecodeError: If files cannot be read or written.
    """

    source_dir = Path(run_dir)
    destination = Path(output_path)
    if not source_dir.is_dir():
        raise ExportError(f"run-dir does not exist or is not a directory: {source_dir}.")
    if destination.exists():
        raise ExportError(f"Refusing to overwrite existing export artifact: {destination}.")
    source = _find_export_metadata(source_dir)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ExportError("Export metadata root must be a JSON object.")
    _write_json_no_overwrite(dict(raw), destination)
    try:
        load_exported_model_package(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _find_export_metadata(run_dir: Path) -> Path:
    candidates = [run_dir / "export_metadata.json", run_dir / "artifacts" / "export_metadata.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ExportError(
        "No compatible export metadata found; expected export_metadata.json in "
        "run-dir or artifacts."
    )


def _write_json_no_overwrite(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=".export.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        temp_path = handle.name
    try:
        if destination.exists():
            raise ExportError(f"Refusing to overwrite existing export artifact: {destination}.")
        os.link(temp_path, destination)
        Path(temp_path).unlink()
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
