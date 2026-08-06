from __future__ import annotations

import argparse
import json

from config_tools import flatten, load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    base = flatten(load_config(args.base, strict_env=False))
    candidate = flatten(load_config(args.candidate, strict_env=False))
    changes = []
    for key in sorted(set(base) | set(candidate)):
        if base.get(key) != candidate.get(key):
            changes.append({"path": key, "base": base.get(key), "candidate": candidate.get(key)})
    algorithm_changes = [item for item in changes if not item["path"].startswith("experiment.")]
    print(
        json.dumps(
            {"all_changes": changes, "algorithm_change_count": len(algorithm_changes)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
