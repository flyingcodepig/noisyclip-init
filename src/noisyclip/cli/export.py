"""CLI skeleton for future LoRA merge and single-model export."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for export arguments."""

    parser = argparse.ArgumentParser(description="Export one inference model artifact.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to export.")
    parser.add_argument("--output", required=True, type=Path, help="Destination model artifact.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse export arguments and fail clearly until export exists."""

    _ = build_parser().parse_args(argv)
    msg = "Model export is not implemented in the F01 project skeleton."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
