from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def capture(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return (completed.stdout + completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite environment snapshot: {output}")
    output.mkdir(parents=True)

    (output / "pip_freeze.txt").write_text(capture([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    (output / "nvidia_smi.txt").write_text(capture(["nvidia-smi"]), encoding="utf-8")
    (output / "git_state.txt").write_text(
        capture(["git", "status", "--short"]) + "\n" + capture(["git", "rev-parse", "HEAD"]),
        encoding="utf-8",
    )
    system = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    (output / "system.json").write_text(json.dumps(system, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

