"""CLI for validating official prediction CSV files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from noisyclip.submission.validator import load_validation_inputs


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for submission-validation arguments."""

    parser = argparse.ArgumentParser(description="Validate a submission CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Prediction CSV path.")
    parser.add_argument(
        "--test-manifest",
        "--test-files",
        dest="test_manifest",
        required=True,
        type=Path,
        help="Test manifest or filename list.",
    )
    parser.add_argument(
        "--class-mapping", required=True, type=Path, help="class_to_idx JSON mapping."
    )
    parser.add_argument(
        "--report-json", type=Path, default=None, help="Optional validation report path."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a CSV and return stable submission CLI exit codes.

    Returns:
        Exit code `0` on success, `2` for configuration/input-loading errors,
        and `5` for product validation errors.
    """

    args = build_parser().parse_args(argv)
    try:
        report = load_validation_inputs(args.csv, args.test_manifest, args.class_mapping)
    except Exception as exc:
        print(f"VALIDATION_CONFIG_INVALID: {exc}")
        return 2

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(report.to_json() + "\n", encoding="utf-8")

    if report.valid:
        print(f"SUBMISSION_OK: {report.row_count} rows")
        return 0
    print(f"SUBMISSION_INVALID: {len(report.issues)} issue(s)")
    print(json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True))
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
