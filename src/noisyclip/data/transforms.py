"""Conservative CLIP-compatible image transforms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as torch_functional

from noisyclip.config.schema import DataConfig

try:
    from PIL import Image, ImageEnhance
except ImportError as exc:  # pragma: no cover - exercised only without Pillow installed.
    raise ImportError(
        "Pillow is required for noisyclip.data.transforms; install Pillow>=10."
    ) from exc

CLIP_IMAGE_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
TransformMode = Literal["train_weak", "train_strong", "eval"]


@dataclass(slots=True)
class ClipImageTransform:
    """Transform a Pillow image to normalized CLIP tensor shape `[3, 224, 224]`.

    Args:
        mode: `train_weak`, `train_strong`, or deterministic `eval`.
        image_size: Output crop size. Competition configs require 224.
        resize_short_side: Deterministic evaluation resize short side.
        random_resized_crop_scale: Conservative crop scale range for train
            modes. Values must lie in `(0, 1]`.
        horizontal_flip_probability: Probability in `[0, 1]`.
        color_jitter_strength: Conservative brightness/contrast jitter
            strength. It is deliberately small by default to preserve local
            fine-grained structure.
        seed: Optional deterministic seed. If provided with `sample_id`,
            train-time randomness is stable per sample.

    Raises:
        ValueError: If parameters are outside their allowed ranges.
    """

    mode: TransformMode
    image_size: int = 224
    resize_short_side: int = 256
    random_resized_crop_scale: tuple[float, float] = (0.75, 1.0)
    horizontal_flip_probability: float = 0.5
    color_jitter_strength: float = 0.1
    seed: int | None = None
    epoch: int = 0
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate transform ranges before any image is processed.

        Raises:
            ValueError: If output size, resize size, crop scale, or probabilities
                are invalid.
        """

        low, high = self.random_resized_crop_scale
        if self.image_size != 224:
            raise ValueError(
                f"CLIP ViT-B/32 data pipeline requires image_size=224, got {self.image_size}"
            )
        if self.resize_short_side < self.image_size:
            raise ValueError("resize_short_side must be at least image_size.")
        if not 0.0 < low <= high <= 1.0:
            raise ValueError("random_resized_crop_scale values must satisfy 0 < low <= high <= 1.")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        if self.color_jitter_strength < 0.0:
            raise ValueError("color_jitter_strength must be non-negative.")

    def __call__(self, image: Image.Image | Tensor, *, sample_id: str | None = None) -> Tensor:
        """Apply the transform and return a normalized tensor.

        Args:
            image: Pillow image in any decodable mode. It is converted to RGB.
            sample_id: Optional stable ID used to seed per-sample train
                randomness.

        Returns:
            Float32 tensor with shape `[3, 224, 224]`. Values are finite and
            CLIP-normalized with OpenAI CLIP mean/std constants.

        Raises:
            ValueError: If the transformed tensor has wrong shape, dtype, or
                non-finite values.
        """

        if isinstance(image, Tensor):
            return self._transform_tensor(image, sample_id=sample_id)
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        rng = self._rng(sample_id)
        if self.mode == "eval":
            transformed = _resize_short_side(rgb, self.resize_short_side)
            transformed = _center_crop(transformed, self.image_size)
        else:
            transformed = _random_resized_crop(
                rgb, self.image_size, self.random_resized_crop_scale, rng
            )
            if rng.random() < self.horizontal_flip_probability:
                transformed = transformed.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.color_jitter_strength > 0.0:
                transformed = _jitter_color(transformed, self.color_jitter_strength, rng)

        tensor = _pil_to_tensor(transformed)
        if self.normalize:
            tensor = normalize_clip_tensor(tensor)
        _validate_tensor(tensor, normalized=self.normalize)
        return tensor

    def _transform_tensor(self, image: Tensor, *, sample_id: str | None) -> Tensor:
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Decoded tensor must be uint8 with shape [3, H, W].")
        rng = self._rng(sample_id)
        if self.mode == "eval":
            transformed = _resize_short_side_tensor(image, self.resize_short_side)
            transformed = _center_crop_tensor(transformed, self.image_size)
        else:
            transformed = _random_resized_crop_tensor(
                image, self.image_size, self.random_resized_crop_scale, rng
            )
            if rng.random() < self.horizontal_flip_probability:
                transformed = transformed.flip(-1)
            if self.color_jitter_strength > 0.0:
                transformed = _jitter_color_tensor(transformed, self.color_jitter_strength, rng)
        transformed = transformed.contiguous()
        output = normalize_clip_tensor(transformed) if self.normalize else transformed
        _validate_tensor(output, normalized=self.normalize)
        return output

    def _rng(self, sample_id: str | None) -> random.Random:
        if self.seed is None or sample_id is None:
            return random.Random()
        return random.Random(f"{self.seed}:{self.epoch}:{self.mode}:{sample_id}")

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic epoch-specific training augmentation."""

        if epoch < 0:
            raise ValueError("transform epoch must be non-negative.")
        self.epoch = epoch


def build_transform(
    data_config: DataConfig,
    *,
    split: Literal["train", "val", "test"],
    strong: bool = False,
    seed: int | None = None,
    normalize: bool | None = None,
) -> ClipImageTransform:
    """Build the configured transform for train, validation, or test records.

    Args:
        data_config: Strict project data configuration.
        split: Manifest split. Validation and test always use deterministic
            resize plus center crop and never test-time augmentation.
        strong: If true for train, build the optional strong view. It must be
            explicitly enabled in config.
        seed: Optional deterministic transform seed.

    Returns:
        A `ClipImageTransform` producing shape `[3, 224, 224]`.

    Raises:
        ValueError: If a strong train transform is requested while disabled, or
            if `split` is illegal.
    """

    should_normalize = True if normalize is None else normalize
    if split in {"val", "test"}:
        return ClipImageTransform(
            mode="eval",
            image_size=data_config.image_size,
            resize_short_side=data_config.eval_transform.resize_short_side,
            seed=seed,
            normalize=should_normalize,
        )
    if split != "train":
        raise ValueError(f"Illegal split for transform: {split}")
    if strong:
        if not data_config.strong_transform.enabled:
            raise ValueError(
                "Strong transform requested but data.strong_transform.enabled is false."
            )
        return ClipImageTransform(
            mode="train_strong",
            image_size=data_config.image_size,
            random_resized_crop_scale=data_config.strong_transform.random_resized_crop_scale,
            color_jitter_strength=float(data_config.strong_transform.randaugment_magnitude) / 100.0,
            seed=seed,
            normalize=should_normalize,
        )
    return ClipImageTransform(
        mode="train_weak",
        image_size=data_config.image_size,
        random_resized_crop_scale=data_config.train_transform.random_resized_crop_scale,
        horizontal_flip_probability=data_config.train_transform.horizontal_flip_probability,
        color_jitter_strength=data_config.train_transform.color_jitter_strength,
        seed=seed,
        normalize=should_normalize,
    )


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    scale = short_side / min(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size))


def _random_resized_crop(
    image: Image.Image,
    size: int,
    scale_range: tuple[float, float],
    rng: random.Random,
) -> Image.Image:
    width, height = image.size
    scale = rng.uniform(scale_range[0], scale_range[1])
    crop_side = max(1, min(width, height, round(min(width, height) * scale)))
    left = rng.randint(0, max(0, width - crop_side))
    top = rng.randint(0, max(0, height - crop_side))
    cropped = image.crop((left, top, left + crop_side, top + crop_side))
    return cropped.resize((size, size), Image.Resampling.BICUBIC)


def _jitter_color(image: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    factor = 1.0 + rng.uniform(-strength, strength)
    bright = ImageEnhance.Brightness(image).enhance(factor)
    contrast = 1.0 + rng.uniform(-strength, strength)
    return ImageEnhance.Contrast(bright).enhance(contrast)


def _pil_to_tensor(image: Image.Image) -> Tensor:
    raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return raw.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()


def normalize_clip_tensor(tensor: Tensor) -> Tensor:
    """Normalize one uint8 or floating CLIP image without Python pixel expansion."""

    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32).div_(255.0)
    else:
        tensor = tensor.float()
    mean = torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _validate_tensor(tensor: Tensor, *, normalized: bool) -> None:
    if tensor.shape != (3, 224, 224):
        raise ValueError(
            f"Transform output must have shape [3, 224, 224], got {tuple(tensor.shape)}"
        )
    expected_dtype = torch.float32 if normalized else torch.uint8
    if tensor.dtype != expected_dtype:
        raise ValueError(f"Transform output must be {expected_dtype}, got {tensor.dtype}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise ValueError("Transform output contains NaN or Inf values.")


def _resize_short_side_tensor(image: Tensor, short_side: int) -> Tensor:
    height, width = image.shape[-2:]
    scale = short_side / min(width, height)
    return _resize_tensor(image, (round(height * scale), round(width * scale)))


def _center_crop_tensor(image: Tensor, size: int) -> Tensor:
    height, width = image.shape[-2:]
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    return image[:, top : top + size, left : left + size]


def _random_resized_crop_tensor(
    image: Tensor,
    size: int,
    scale_range: tuple[float, float],
    rng: random.Random,
) -> Tensor:
    height, width = image.shape[-2:]
    scale = rng.uniform(scale_range[0], scale_range[1])
    crop_side = max(1, min(width, height, round(min(width, height) * scale)))
    left = rng.randint(0, max(0, width - crop_side))
    top = rng.randint(0, max(0, height - crop_side))
    cropped = image[:, top : top + crop_side, left : left + crop_side]
    return _resize_tensor(cropped, (size, size))


def _resize_tensor(image: Tensor, size: tuple[int, int]) -> Tensor:
    resized = torch_functional.interpolate(
        image.unsqueeze(0).float(),
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    return resized.round_().clamp_(0, 255).to(torch.uint8)


def _jitter_color_tensor(image: Tensor, strength: float, rng: random.Random) -> Tensor:
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    value = image.float().mul_(brightness)
    gray = value[0] * 0.299 + value[1] * 0.587 + value[2] * 0.114
    value = value.mul(contrast).add_(gray.mean() * (1.0 - contrast))
    return value.round_().clamp_(0, 255).to(torch.uint8)
