"""Pinned-memory transfer, GPU normalization, and one-batch CUDA prefetch."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch import Tensor

from noisyclip.data.records import Batch
from noisyclip.data.transforms import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD


class BatchDeviceIterator(Iterator[Batch]):
    """Move batches asynchronously and overlap the next transfer on CUDA."""

    def __init__(self, loader: Iterable[Batch], device: torch.device | str) -> None:
        self._source = iter(loader)
        self.device = torch.device(device)
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._next: Batch | None = None
        self._preload()

    def __iter__(self) -> BatchDeviceIterator:
        return self

    def __next__(self) -> Batch:
        batch = self._next
        if batch is None:
            raise StopIteration
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
            _record_batch_stream(batch, torch.cuda.current_stream(self.device))
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self._source)
        except StopIteration:
            self._next = None
            return
        if self._stream is None:
            self._next = move_batch_to_device(batch, self.device)
            return
        with torch.cuda.stream(self._stream):
            self._next = move_batch_to_device(batch, self.device, non_blocking=True)


def move_batch_to_device(
    batch: Batch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Batch:
    """Move a batch and normalize compact uint8 image tensors on the destination."""

    target = torch.device(device)
    return Batch(
        sample_ids=batch.sample_ids,
        image_weak=(
            None
            if batch.image_weak is None
            else _prepare_images(batch.image_weak, target, non_blocking)
        ),
        image_strong=(
            None
            if batch.image_strong is None
            else _prepare_images(batch.image_strong, target, non_blocking)
        ),
        targets=(
            None if batch.targets is None else batch.targets.to(target, non_blocking=non_blocking)
        ),
        class_ids=batch.class_ids,
        embedding_weak=(
            None
            if batch.embedding_weak is None
            else batch.embedding_weak.to(target, non_blocking=non_blocking)
        ),
        embedding_strong=(
            None
            if batch.embedding_strong is None
            else batch.embedding_strong.to(target, non_blocking=non_blocking)
        ),
    )


def _prepare_images(images: Tensor, device: torch.device, non_blocking: bool) -> Tensor:
    images = images.to(device, non_blocking=non_blocking)
    if images.dtype != torch.uint8:
        return images
    images = images.to(torch.float32).div_(255.0)
    mean = images.new_tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)


def _record_batch_stream(batch: Batch, stream: torch.cuda.Stream) -> None:
    for tensor in (
        batch.image_weak,
        batch.image_strong,
        batch.embedding_weak,
        batch.embedding_strong,
        batch.targets,
    ):
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(stream)
