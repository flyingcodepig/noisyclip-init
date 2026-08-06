"""CLI for the unique NoisyCLIP training entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from noisyclip.config.loader import load_config
from noisyclip.tracking.artifacts import create_run_dir
from noisyclip.tracking.manifest import generate_run_id


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for training arguments."""

    parser = argparse.ArgumentParser(description="Train a NoisyCLIP experiment.")
    parser.add_argument("--config", required=True, help="Path to a strict YAML config.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id.")
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
        config = load_config(args.config)
        run_id = args.run_id or generate_run_id(config.experiment.name)
        _preflight_run_directory(config.paths.run_root, run_id, config.tracking.fail_if_run_exists)
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2
    except (FileExistsError, OSError) as exc:
        sys.stderr.write(f"DATA_OR_COMPLIANCE_INVALID: {exc}\n")
        return 3

    sys.stderr.write(
        "TRAINING_ASSEMBLY_REQUIRED: pass validated components to "
        "noisyclip.engine.trainer.Trainer; "
        "the CLI does not construct CLIP weights or datasets by itself.\n"
    )
    return 4


def _preflight_run_directory(run_root: str, run_id: str, fail_if_run_exists: bool) -> None:
    """Validate and create the run directory before expensive model construction.

    Args:
        run_root: Configured run root. Unresolved env placeholders are rejected.
        run_id: Explicit or generated run identifier.
        fail_if_run_exists: Whether to refuse an existing run directory.

    Raises:
        ValueError: If `run_root` still contains an unresolved environment
            placeholder.
        FileExistsError: If the run directory already exists and overwrite is
            forbidden.
        OSError: If the directory cannot be created.
    """

    if run_root.startswith("${oc.env:"):
        raise ValueError("paths.run_root is unresolved; set NOISYCLIP_RUN_ROOT.")
    create_run_dir(Path(run_root), run_id, fail_if_run_exists=fail_if_run_exists)


if __name__ == "__main__":
    raise SystemExit(main())
