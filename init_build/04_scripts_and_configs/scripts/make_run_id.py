from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone

from config_tools import config_digest, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config, strict_env=False)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config["experiment"]["name"]).strip("-")
    seed = int(config["experiment"]["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = config_digest(config)[:8]
    print(f"{stamp}-{name}-s{seed}-{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

