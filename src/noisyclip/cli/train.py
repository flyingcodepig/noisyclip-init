"""CLI for the unique NoisyCLIP training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from noisyclip.engine.assembly import AssemblyError, run_training
from noisyclip.tracking.manifest import generate_run_id


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for training arguments."""

    parser = argparse.ArgumentParser(description="Train a NoisyCLIP experiment.")
    parser.add_argument("--config", required=True, help="Path to a strict YAML config.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Explicit checkpoint used to resume the same run directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, validate, preflight output paths, and delegate training assembly.

    Args:
        argv: Optional command-line arguments excluding program name.

    Returns:
        Exit code `0` on success, `2` for config errors, `3` for data/compliance
        errors, and `4` for training failures. The CLI intentionally contains no
        optimization or model-update algorithm.
    """

    args = build_parser().parse_args(argv)
    try:
        from noisyclip.config.loader import load_config

        config = load_config(args.config)
        run_id = args.run_id or generate_run_id(config.experiment.name)
        if args.resume is not None and args.run_id is None:
            raise ValueError("--resume requires the original --run-id.")
        result = run_training(
            args.config,
            run_id=run_id,
            resume_checkpoint=args.resume,
        )
    except (AssemblyError, FileExistsError, FileNotFoundError, OSError) as exc:
        sys.stderr.write(f"DATA_OR_COMPLIANCE_INVALID: {exc}\n")
        return 3
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2
    except RuntimeError as exc:
        sys.stderr.write(f"TRAINING_FAILED: {exc}\n")
        return 4

    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "run_id": run_id,
                "epochs_completed": result.epochs_completed,
                "global_step": result.global_step,
                "last_checkpoint": str(result.last_checkpoint),
                "exported_model": (
                    None if result.exported_model is None else str(result.exported_model)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
