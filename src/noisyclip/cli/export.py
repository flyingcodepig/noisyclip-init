"""CLI for exporting exactly one inference model package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from noisyclip.submission.exporter import export_single_model_from_run


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for single-model export arguments."""

    parser = argparse.ArgumentParser(description="Export one inference model artifact.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to export.")
    parser.add_argument("--output", required=True, type=Path, help="Destination model artifact.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export one compatible student inference package.

    Returns:
        Exit code `0` on success, `2` for argument shape errors, and `3` for
        data/compliance errors such as existing outputs or teacher/ensemble
        metadata.
    """

    args = build_parser().parse_args(argv)
    try:
        exported = export_single_model_from_run(args.run_dir, args.output)
    except Exception as exc:
        print(f"EXPORT_INVALID: {exc}")
        return 3
    print(f"EXPORT_OK: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
