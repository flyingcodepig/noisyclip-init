"""Machine-readable JSONL logging for training and evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append sorted JSON records to one log file.

    Args:
        path: JSONL file path. Parent directories are created.

    Raises:
        OSError: If the log cannot be opened or written.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        """Append one JSON object and fsync it.

        Args:
            record: JSON-serializable mapping.

        Raises:
            TypeError: If `record` is not JSON-serializable.
            OSError: If the write fails.
        """

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
