"""Environment and code-version snapshots for run manifests."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def collect_environment_snapshot(*, cwd: Path | str | None = None) -> dict[str, Any]:
    """Collect Python, torch, CUDA, GPU, and Git metadata.

    Args:
        cwd: Optional repository root for Git commands.

    Returns:
        JSON-serializable environment snapshot. Missing Git or GPU details are
        represented as `None` or diagnostic strings, never by raising.
    """

    root = Path(cwd).resolve() if cwd is not None else Path.cwd()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": _gpu_info(),
        "git": _git_info(root),
    }


def _gpu_info() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_mib": int(props.total_memory / (1024 * 1024)),
            }
        )
    return devices


def _git_info(cwd: Path) -> dict[str, str | None]:
    def run_git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "branch": run_git("branch", "--show-current"),
        "head": run_git("rev-parse", "HEAD"),
        "status_short": run_git("status", "--short"),
    }
