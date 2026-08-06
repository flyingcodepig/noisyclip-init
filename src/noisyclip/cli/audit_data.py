"""Command-line entry point for data auditing and manifest generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from noisyclip.config.loader import load_config
from noisyclip.data.audit import AuditConfigError, run_data_audit
from noisyclip.data.catalog import CatalogError
from noisyclip.data.image_io import ImageAuditError
from noisyclip.data.leakage import LeakageError
from noisyclip.data.manifests import ManifestError
from noisyclip.data.split import SplitError

CONFIG_EXCEPTIONS = (AuditConfigError,)
DATA_EXCEPTIONS = (CatalogError, ImageAuditError, LeakageError, ManifestError, SplitError, OSError)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for data-audit arguments."""

    parser = argparse.ArgumentParser(description="Audit data and build manifests.")
    parser.add_argument("--config", required=True, help="Path to a strict YAML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data audit CLI.

    Args:
        argv: Optional command-line arguments, excluding program name.

    Returns:
        Exit code `0` on success, `2` for configuration errors, and `3` for
        data/compliance errors. The CLI only parses, validates, and assembles;
        scan, hash, split, and leakage algorithms live in `noisyclip.data`.
    """

    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_data_audit(config)
    except CONFIG_EXCEPTIONS as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2
    except DATA_EXCEPTIONS as exc:
        sys.stderr.write(f"DATA_AUDIT_FAILED: {exc}\n")
        return 3
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"CONFIG_INVALID: {exc}\n")
        return 2

    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "class_mapping": str(result.class_mapping_path),
                "train_manifest": str(result.train_manifest_path),
                "val_manifest": str(result.val_manifest_path),
                "test_manifest": str(result.test_manifest_path),
                "data_digest": result.data_digest,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
