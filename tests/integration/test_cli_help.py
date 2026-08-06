"""Integration smoke tests for F01 command-line entry help."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI_MODULES = [
    "noisyclip.cli.audit_data",
    "noisyclip.cli.train",
    "noisyclip.cli.evaluate",
    "noisyclip.cli.infer",
    "noisyclip.cli.export",
    "noisyclip.cli.validate_submission",
]


def test_cli_help_succeeds_for_all_entry_modules() -> None:
    """Every F01 CLI module imports and returns zero for `--help`."""

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    for module in CLI_MODULES:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
