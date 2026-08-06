"""CLI for compliant single-model test-set inference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from noisyclip.config.loader import load_config
from noisyclip.submission.inference import run_packaged_submission_inference


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for compliant inference arguments."""

    parser = argparse.ArgumentParser(description="Run test-only inference.")
    parser.add_argument(
        "--model", required=True, action="append", type=Path, help="Single exported model path."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a strict YAML config.")
    parser.add_argument(
        "--test-manifest", type=Path, default=None, help="Test manifest or filename list."
    )
    parser.add_argument(
        "--class-mapping", type=Path, default=None, help="class_to_idx JSON mapping."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for pred_results.csv."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace an existing pred_results.csv."
    )
    parser.add_argument("--tta", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one-model inference and validate the resulting CSV.

    Returns:
        Exit code `0` on success, `2` for argument/configuration errors, `3`
        for data or compliance errors, and `5` when generated artifact
        validation fails.
    """

    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"CONFIG_INVALID: {exc}")
        return 2

    if len(args.model) != 1:
        print("COMPLIANCE_INVALID: exactly one exported model path is allowed.")
        return 3
    if args.tta or config.evaluation.test_time_augmentation:
        print("COMPLIANCE_INVALID: test-time augmentation is forbidden.")
        return 3

    class_mapping = args.class_mapping or _optional_path(config.paths.class_mapping)
    test_manifest = args.test_manifest or _optional_path(config.paths.test_manifest)
    try:
        report = run_packaged_submission_inference(
            args.model[0],
            args.output_dir,
            class_mapping_path=class_mapping,
            test_manifest_path=test_manifest,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"COMPLIANCE_INVALID: {exc}")
        return 3
    if not report.valid:
        print(f"SUBMISSION_INVALID: {len(report.issues)} issue(s)")
        print(report.to_json())
        return 5
    print(f"SUBMISSION_OK: wrote {args.output_dir / 'pred_results.csv'} ({report.row_count} rows)")
    return 0


def _optional_path(value: str | None) -> Path | None:
    if value is None or value.startswith("${"):
        return None
    return Path(value)


if __name__ == "__main__":
    raise SystemExit(main())
