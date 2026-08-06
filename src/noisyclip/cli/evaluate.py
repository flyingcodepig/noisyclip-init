"""CLI for fixed-validation checkpoint evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from noisyclip.engine.checkpoint import CHECKPOINT_FORMAT_VERSION


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for evaluation arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a completed run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to <run-dir>/checkpoints/last.pt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate evaluation inputs and require engine-level evaluator assembly.

    Args:
        argv: Optional command-line arguments excluding program name.

    Returns:
        Exit code `0` on success, `2` for invalid arguments/config, `3` for
        missing data/checkpoint/compliance errors, and `4` when model assembly
        is unavailable to the thin CLI.
    """

    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or args.run_dir / "checkpoints" / "last.pt"
    try:
        _validate_evaluation_inputs(args.run_dir, checkpoint)
    except ValueError as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2
    except (FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"DATA_OR_COMPLIANCE_INVALID: {exc}\n")
        return 3
    sys.stderr.write(
        "EVALUATION_ASSEMBLY_REQUIRED: load the checkpointed model and call "
        "noisyclip.engine.evaluator.Evaluator on the fixed val manifest.\n"
    )
    return 4


def _validate_evaluation_inputs(run_dir: Path, checkpoint: Path) -> None:
    """Reject missing run/checkpoint paths and accidental test-evaluation inputs.

    Args:
        run_dir: Existing run directory containing fixed validation artifacts.
        checkpoint: Existing checkpoint that will be reloaded for metrics.

    Raises:
        FileNotFoundError: If required paths are absent.
        ValueError: If a path name suggests test labels or checkpoint selection
            from test data.
    """

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    normalized = checkpoint.as_posix().lower()
    if "test" in normalized:
        raise ValueError("evaluation checkpoint path must not be selected from test data.")
    _ = CHECKPOINT_FORMAT_VERSION


if __name__ == "__main__":
    raise SystemExit(main())
