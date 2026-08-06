"""CLI skeleton for test-set inference with one exported model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from noisyclip.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for inference arguments."""

    parser = argparse.ArgumentParser(description="Run test-only inference.")
    parser.add_argument("--model", required=True, type=Path, help="Single exported model path.")
    parser.add_argument("--config", required=True, help="Path to a strict YAML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse inference arguments, validate config, and fail clearly until F02."""

    args = build_parser().parse_args(argv)
    load_config(args.config)
    msg = "Inference is not implemented in the F01 project skeleton."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
