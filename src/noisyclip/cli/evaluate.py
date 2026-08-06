"""CLI for fixed-validation checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from noisyclip.engine.assembly import AssemblyError, evaluate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for evaluation arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a completed run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path; defaults to <run-dir>/checkpoints/best_top1.pt.",
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
    checkpoint = args.checkpoint or args.run_dir / "checkpoints" / "best_top1.pt"
    try:
        _validate_evaluation_inputs(args.run_dir, checkpoint)
        result = evaluate_checkpoint(args.run_dir, checkpoint)
    except AssemblyError as exc:
        sys.stderr.write(f"DATA_OR_COMPLIANCE_INVALID: {exc}\n")
        return 3
    except (FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"DATA_OR_COMPLIANCE_INVALID: {exc}\n")
        return 3
    except ValueError as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2
    except RuntimeError as exc:
        sys.stderr.write(f"EVALUATION_FAILED: {exc}\n")
        return 4
    sys.stdout.write(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    return 0


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
    if checkpoint.stem.lower().startswith("test"):
        raise ValueError("evaluation checkpoint path must not be selected from test data.")


if __name__ == "__main__":
    raise SystemExit(main())
