"""Atomic file replacement with fsync and disk-space preflight checks."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from noisyclip.utils.paths import ensure_parent_directory, fail_if_exists

T = TypeVar("T")


def ensure_free_space(path: Path | str, required_bytes: int) -> None:
    """Require at least `required_bytes` free bytes near `path`.

    Args:
        path: Destination file or directory used to select the filesystem.
        required_bytes: Non-negative byte count.

    Raises:
        ValueError: If `required_bytes` is negative.
        OSError: If available disk space is lower than requested.
    """

    if required_bytes < 0:
        raise ValueError("required_bytes must be non-negative.")
    target = Path(path)
    probe = target if target.is_dir() else target.parent
    probe.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(probe).free
    if free < required_bytes:
        raise OSError(
            f"Insufficient free disk space for {target}: need {required_bytes} bytes, "
            f"available {free} bytes."
        )


def atomic_write_bytes(
    destination: Path | str,
    data: bytes,
    *,
    overwrite: bool = True,
    minimum_free_bytes: int = 0,
) -> Path:
    """Write bytes through a temporary file, fsync, and atomic replace.

    Args:
        destination: Final file path.
        data: Bytes to persist exactly.
        overwrite: When false, an existing final destination raises before the
            temporary file is created.
        minimum_free_bytes: Additional disk-space requirement checked before
            writing. Use this for checkpoint safety margins.

    Returns:
        Final destination path.

    Raises:
        FileExistsError: If `overwrite` is false and destination exists.
        OSError: If writing, fsync, space checks, or replacement fail.
    """

    output_path = ensure_parent_directory(destination)
    if not overwrite:
        fail_if_exists(output_path)
    ensure_free_space(output_path, max(minimum_free_bytes, len(data)))
    tmp_path = _temporary_path(output_path)
    if tmp_path.exists():
        raise FileExistsError(f"Temporary artifact already exists: {tmp_path}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(tmp_path.parent)
        os.replace(tmp_path, output_path)
        _fsync_directory(output_path.parent)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def atomic_save_with_writer(
    destination: Path | str,
    writer: Callable[[Path], T],
    *,
    overwrite: bool = True,
    minimum_free_bytes: int = 0,
) -> tuple[Path, T]:
    """Persist a file by asking `writer` to fill a temporary path.

    Args:
        destination: Final artifact path.
        writer: Callable receiving a temporary path and returning arbitrary
            metadata after writing the file.
        overwrite: When false, existing final destinations are refused.
        minimum_free_bytes: Disk-space requirement checked before `writer`.

    Returns:
        Tuple of final path and `writer` return value.

    Raises:
        FileExistsError: If destination or temporary file would be overwritten.
        OSError: If space checks, writer I/O, fsync, or replace fails.
    """

    output_path = ensure_parent_directory(destination)
    if not overwrite:
        fail_if_exists(output_path)
    ensure_free_space(output_path, minimum_free_bytes)
    tmp_path = _temporary_path(output_path)
    if tmp_path.exists():
        raise FileExistsError(f"Temporary artifact already exists: {tmp_path}")
    try:
        result = writer(tmp_path)
        with tmp_path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(tmp_path.parent)
        os.replace(tmp_path, output_path)
        _fsync_directory(output_path.parent)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path, result


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
