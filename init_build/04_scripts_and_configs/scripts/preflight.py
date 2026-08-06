from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from config_tools import ConfigError, config_digest, load_config
from validate_config import cross_field_errors


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    try:
        config = load_config(args.config, strict_env=True)
        errors = cross_field_errors(config)
        if errors:
            raise ConfigError("; ".join(errors))

        paths = config["paths"]
        train_root = Path(paths["train_root"]).resolve()
        test_root = Path(paths["test_root"]).resolve()
        run_root = Path(paths["run_root"]).resolve()
        for name, path in (("train", train_root), ("test", test_root), ("run", run_root)):
            if not path.is_dir():
                raise ConfigError(f"{name} root does not exist: {path}")
        if len({train_root, test_root, run_root}) != 3:
            raise ConfigError("train, test, and run roots must be distinct")
        if is_relative_to(run_root, train_root) or is_relative_to(run_root, test_root):
            raise ConfigError("run root cannot be nested inside raw data")
        if os.access(train_root, os.W_OK) or os.access(test_root, os.W_OK):
            raise ConfigError("raw train/test roots must be mounted read-only")
        if not os.access(run_root, os.W_OK):
            raise ConfigError("run root is not writable")

        pattern = re.compile(config["data"]["class_id_regex"])
        class_ids = sorted(item.name for item in train_root.iterdir() if item.is_dir())
        expected = int(config["data"]["expected_num_classes"])
        if len(class_ids) != expected:
            raise ConfigError(f"expected {expected} class folders, found {len(class_ids)}")
        invalid = [item for item in class_ids if not pattern.fullmatch(item)]
        if invalid:
            raise ConfigError(f"invalid class folder IDs: {invalid[:10]}")

        free_gib = shutil.disk_usage(run_root).free / 2**30
        minimum = float(config["tracking"]["minimum_free_disk_gib"])
        if free_gib < minimum:
            raise ConfigError(f"free disk {free_gib:.1f} GiB is below {minimum:.1f} GiB")

        gpu_report: dict[str, object] = {"checked": False}
        if not args.allow_cpu:
            import torch

            if not torch.cuda.is_available():
                raise ConfigError("CUDA is unavailable")
            index = int(config["trainer"]["device"].split(":", 1)[1])
            if index >= torch.cuda.device_count():
                raise ConfigError(f"configured GPU {index} is not visible")
            props = torch.cuda.get_device_properties(index)
            gpu_report = {
                "checked": True,
                "name": props.name,
                "memory_gib": round(props.total_memory / 2**30, 2),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }

        report = {
            "status": "ok",
            "config_digest": config_digest(config),
            "class_count": len(class_ids),
            "first_class_id": class_ids[0],
            "last_class_id": class_ids[-1],
            "run_free_gib": round(free_gib, 2),
            "gpu": gpu_report,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"PREFLIGHT_FAILED: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
