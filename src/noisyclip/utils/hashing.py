"""Stable SHA256 helpers for configs, manifests, tensors, and files."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def stable_json_dumps(payload: Any) -> str:
    """Return canonical JSON for hashable metadata.

    Args:
        payload: JSON-serializable metadata containing strings, numbers,
            booleans, lists, dictionaries, or `None`.

    Returns:
        Compact UTF-8-compatible JSON with sorted object keys.

    Raises:
        TypeError: If `payload` cannot be serialized as JSON.
    """

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_hash(payload: Any) -> str:
    """Return a lowercase SHA256 digest for canonical JSON metadata.

    Args:
        payload: JSON-serializable object. Tensor values are not accepted here;
            use `hash_state_dict` for model weights.

    Returns:
        Hexadecimal SHA256 digest string of length `64`.

    Raises:
        TypeError: If `payload` is not JSON-serializable.
    """

    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def hash_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest for one existing file.

    Args:
        path: File path to read in binary mode.
        chunk_size: Positive read chunk size in bytes.

    Returns:
        Hexadecimal SHA256 digest.

    Raises:
        ValueError: If `chunk_size` is not positive.
        FileNotFoundError: If `path` is not a regular file.
        OSError: If the file cannot be read.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist for hashing: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_state_dict(state_dict: Mapping[str, Tensor]) -> str:
    """Return a stable SHA256 digest for a tensor state dict.

    Args:
        state_dict: Mapping from parameter/buffer names to tensors. Tensors may
            have any shape or dtype, and are serialized from detached CPU copies.

    Returns:
        Hexadecimal SHA256 digest over sorted tensor entries.

    Raises:
        TypeError: If a key is not a string or a value is not a tensor.
    """

    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("state_dict keys must be strings.")
        if not isinstance(value, Tensor):
            raise TypeError(f"state_dict value for {key!r} must be a torch.Tensor.")
    buffer = io.BytesIO()
    torch.save({key: state_dict[key].detach().cpu() for key in sorted(state_dict)}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()
