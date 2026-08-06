"""CLI skeleton for the unique training entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from noisyclip.config.loader import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for training arguments."""

    parser = argparse.ArgumentParser(description="Train a NoisyCLIP experiment.")
    parser.add_argument("--config", required=True, help="Path to a strict YAML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse training arguments, validate config, and fail clearly until F02."""

    args = build_parser().parse_args(argv)
    load_config(args.config)
    msg = "Training is not implemented in the F01 project skeleton."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
