from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "resolved_config.yaml",
    "manifest.json",
    "data/class_to_idx.json",
    "data/manifest_digest.json",
    "metrics/epoch_metrics.jsonl",
    "metrics/best_metrics.json",
    "metrics/per_class_metrics.csv",
    "checkpoints/last.pt",
    "checkpoints/best_top1.pt",
    "logs/train.log",
    "DONE",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (run_dir / relative).is_file()]
    empty = [
        relative
        for relative in REQUIRED_FILES
        if (run_dir / relative).is_file() and (run_dir / relative).stat().st_size == 0 and relative != "DONE"
    ]
    if missing or empty:
        print(json.dumps({"status": "invalid", "missing": missing, "empty": empty}, indent=2))
        return 5

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        print(json.dumps({"status": "invalid", "reason": "manifest status is not completed"}, indent=2))
        return 5
    print(json.dumps({"status": "ok", "run_id": run_dir.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

