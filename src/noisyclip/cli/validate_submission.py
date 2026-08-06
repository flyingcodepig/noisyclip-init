"""CLI skeleton for submission CSV validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for submission-validation arguments."""

    parser = argparse.ArgumentParser(description="Validate a submission CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Prediction CSV path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse validation arguments and fail clearly until validation exists."""

    _ = build_parser().parse_args(argv)
    msg = "Submission validation is not implemented in the F01 project skeleton."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
