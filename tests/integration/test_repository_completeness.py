"""Repository-level checks that protect clean-clone deployability."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only Git query in the repository root."""

    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_required_source_and_config_files_are_not_ignored() -> None:
    """Package data interfaces and root B0 config must be eligible for Git."""

    required = [
        "src/noisyclip/data/__init__.py",
        "src/noisyclip/data/records.py",
        "configs/base.yaml",
    ]
    for relative_path in required:
        result = git("check-ignore", "-q", relative_path)
        assert result.returncode != 0, f"required file is ignored: {relative_path}"


def test_generated_metadata_and_sensitive_artifacts_are_not_tracked() -> None:
    """Generated package metadata, secrets, data, and weights stay out of Git."""

    tracked = git("ls-files")
    assert tracked.returncode == 0, tracked.stderr
    forbidden_suffixes = (".pt", ".pth", ".ckpt", ".safetensors")
    for relative_path in tracked.stdout.splitlines():
        normalized = relative_path.replace("\\", "/")
        assert ".egg-info/" not in normalized
        assert not normalized.endswith(forbidden_suffixes)
        assert normalized != ".env"
        assert not normalized.startswith(("data/", "dataset/", "datasets/"))
