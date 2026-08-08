"""Safe image inspection and CLIP-compatible image loading."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

try:
    from PIL import Image, ImageFile, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - exercised only without Pillow installed.
    raise ImportError(
        "Pillow is required for noisyclip.data.image_io; install Pillow>=10."
    ) from exc


UnreadablePolicy = Literal["fail_audit", "skip_with_record"]


class ImageAuditError(ValueError):
    """Raised when image audit fails under the configured strict policy."""


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Readability metadata for one image file.

    Args:
        relative_path: POSIX-style path relative to the official train or test
            root. Absolute paths are never serialized into manifests.
        width: Pixel width when readable, otherwise `None`.
        height: Pixel height when readable, otherwise `None`.
        readable: Whether Pillow could load and verify pixel data.
        file_sha256: Optional SHA256 of file bytes.
        mode: Pillow image mode such as `RGB`, `L`, or `RGBA` when readable.
        error: Error text for unreadable files, otherwise `None`.

    Raises:
        ImageAuditError: Factory functions raise it for unreadable files when
            `unreadable_policy` is `fail_audit`.
    """

    relative_path: str
    width: int | None
    height: int | None
    readable: bool
    file_sha256: str | None
    mode: str | None
    error: str | None


def file_sha256(path: Path | str) -> str:
    """Compute a SHA256 digest for one file without modifying it.

    Args:
        path: File path to read in binary mode.

    Returns:
        Lowercase hexadecimal SHA256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image(
    path: Path | str,
    *,
    relative_path: str,
    hash_file: bool,
    allow_truncated_images: bool,
    unreadable_policy: UnreadablePolicy,
) -> ImageInfo:
    """Inspect image dimensions, readability, mode, and optional file hash.

    Args:
        path: Image file to inspect. The function reads only; it never moves,
            overwrites, deletes, or rewrites source images.
        relative_path: Stable POSIX-style path stored in manifests.
        hash_file: Whether to include file-byte SHA256.
        allow_truncated_images: If true, Pillow may load truncated files; if
            false, truncated files are unreadable.
        unreadable_policy: `fail_audit` raises immediately, while
            `skip_with_record` returns an unreadable `ImageInfo`.

    Returns:
        Image metadata. Readable RGB, grayscale, and RGBA files keep their
        original mode here; conversion to RGB happens in transforms.

    Raises:
        ImageAuditError: If the image cannot be read and the policy is
            `fail_audit`, or if dimensions are non-positive.
        OSError: If hashing fails.
    """

    file_path = Path(path)
    digest = file_sha256(file_path) if hash_file else None
    previous_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated_images
    try:
        with Image.open(file_path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        message = f"Unreadable image {relative_path}: {exc}"
        if unreadable_policy == "fail_audit":
            raise ImageAuditError(message) from exc
        return ImageInfo(
            relative_path=relative_path,
            width=None,
            height=None,
            readable=False,
            file_sha256=digest,
            mode=None,
            error=message,
        )
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_setting

    if width <= 0 or height <= 0:
        message = f"Image {relative_path} has non-positive dimensions: {width}x{height}"
        if unreadable_policy == "fail_audit":
            raise ImageAuditError(message)
        return ImageInfo(
            relative_path=relative_path,
            width=None,
            height=None,
            readable=False,
            file_sha256=digest,
            mode=mode,
            error=message,
        )

    return ImageInfo(
        relative_path=relative_path,
        width=width,
        height=height,
        readable=True,
        file_sha256=digest,
        mode=mode,
        error=None,
    )


def load_rgb_image(path: Path | str) -> Image.Image:
    """Load one image as RGB for transforms.

    Args:
        path: Image file path from a manifest-relative lookup.

    Returns:
        A Pillow RGB image. Grayscale and RGBA inputs are converted to exactly
        three channels.

    Raises:
        ImageAuditError: If Pillow cannot decode the file.
    """

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageAuditError(f"Could not load RGB image: {path}") from exc


def load_rgb_tensor(path: Path | str) -> Tensor:
    """Decode to contiguous uint8 ``[3, H, W]`` with a full-format fallback."""

    try:
        from torchvision.io import ImageReadMode, decode_image

        tensor = decode_image(
            str(Path(path)),
            mode=ImageReadMode.RGB,
            apply_exif_orientation=False,
        )
        if tensor.dtype != torch.uint8 or tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Unexpected torchvision decode result: {tuple(tensor.shape)}")
        return tensor.contiguous()
    except (ImportError, OSError, RuntimeError, ValueError):
        image = load_rgb_image(path)
        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return raw.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()


def build_image_loader(backend: str) -> Callable[[Path | str], Any]:
    """Return the configured decoder while retaining Pillow fallback coverage."""

    if backend == "pillow":
        return load_rgb_image
    if backend == "torchvision_fallback":
        return load_rgb_tensor
    raise ValueError(f"Unsupported image decode backend: {backend!r}.")
