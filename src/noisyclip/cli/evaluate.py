"""CLI skeleton for validation-set evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for evaluation arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a completed run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to evaluate.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse evaluation arguments and fail clearly until evaluation exists."""

    _ = build_parser().parse_args(argv)
    msg = "Evaluation is not implemented in the F01 project skeleton."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
