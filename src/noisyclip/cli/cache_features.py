"""CLI for provenance-bound B0/B1 frozen feature cache generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from noisyclip.engine.assembly import AssemblyError, build_feature_cache_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact frozen CLIP features for B0/B1.")
    parser.add_argument("--config", required=True, help="Path to a strict cache-enabled config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = build_feature_cache_from_config(args.config)
    except (
        AssemblyError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"FEATURE_CACHE_FAILED: {exc}\n")
        return 3
    sys.stdout.write(json.dumps({"status": "ok", "feature_cache": str(root)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
