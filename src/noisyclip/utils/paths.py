"""Safe path validation and overwrite guards for run artifacts."""

from __future__ import annotations

from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a path escapes its intended root or would overwrite output."""


def resolve_under_root(root: Path | str, candidate: Path | str) -> Path:
    """Resolve `candidate` and require it to stay under `root`.

    Args:
        root: Directory that owns all writable artifacts.
        candidate: Relative path under `root`, or an absolute path already
            inside `root`.

    Returns:
        Absolute resolved candidate path.

    Raises:
        PathSafetyError: If `candidate` resolves outside `root`.
    """

    root_path = Path(root).expanduser().resolve()
    candidate_path = Path(candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = root_path / candidate_path
    resolved = candidate_path.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PathSafetyError(
            f"Path escapes artifact root: {resolved} not under {root_path}"
        ) from exc
    return resolved


def ensure_parent_directory(path: Path | str) -> Path:
    """Create the parent directory for a future file and return its path.

    Args:
        path: File path whose parent should exist.

    Returns:
        The input path as a `Path`.

    Raises:
        OSError: If the parent directory cannot be created.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def fail_if_exists(path: Path | str, *, field_name: str = "artifact") -> None:
    """Reject overwriting an existing file or directory.

    Args:
        path: Important artifact path that must not already exist.
        field_name: Human-readable field used in error messages.

    Raises:
        FileExistsError: If `path` exists.
    """

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing {field_name}: {target}")
