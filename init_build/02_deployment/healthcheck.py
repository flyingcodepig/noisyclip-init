from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch


def require_dir(env_name: str, *, writable: bool) -> dict[str, object]:
    raw = os.environ.get(env_name)
    if not raw:
        raise RuntimeError(f"missing environment variable: {env_name}")
    path = Path(raw).resolve()
    if not path.is_dir():
        raise RuntimeError(f"directory does not exist: {env_name}={path}")
    if writable and not os.access(path, os.W_OK):
        raise RuntimeError(f"directory is not writable: {path}")
    if not writable and os.access(path, os.W_OK):
        raise RuntimeError(f"raw data mount must be read-only: {path}")
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "writable": writable,
        "free_gib": round(usage.free / 2**30, 2),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError(f"unexpected torch version: {torch.__version__}")

    devices: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_gib": round(properties.total_memory / 2**30, 2),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    if not devices or max(float(d["total_memory_gib"]) for d in devices) < 20.0:
        raise RuntimeError("at least one GPU with approximately 24GB VRAM is required")

    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
        "mounts": {
            "train": require_dir("NOISYCLIP_TRAIN_ROOT", writable=False),
            "test": require_dir("NOISYCLIP_TEST_ROOT", writable=False),
            "runs": require_dir("NOISYCLIP_RUN_ROOT", writable=True),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

