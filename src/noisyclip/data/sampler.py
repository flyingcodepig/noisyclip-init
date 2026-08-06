"""Deterministic sampling helpers for manifest-backed datasets."""

from __future__ import annotations

import random
from collections.abc import Iterator

from torch.utils.data import Sampler

from noisyclip.data.records import SampleRecord


class DeterministicSampler(Sampler[int]):
    """Sampler with stable sequential or seeded-shuffle order.

    Args:
        records: Manifest rows. The sampler uses only row count and sample IDs;
            it never inspects image content or test statistics.
        shuffle: Whether to shuffle indices using `seed`.
        seed: Non-negative deterministic seed.

    Raises:
        ValueError: If `seed` is negative.
    """

    def __init__(self, records: list[SampleRecord], *, shuffle: bool, seed: int) -> None:
        """Initialize a deterministic sampler over manifest indices."""

        if seed < 0:
            raise ValueError("sampler seed must be non-negative.")
        self._records = list(records)
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic, epoch-specific shuffle order."""

        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative.")
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield dataset indices in stable order.

        Returns:
            Iterator of integer indices in `[0, len(records))`.
        """

        indices = list(range(len(self._records)))
        if self._shuffle:
            keyed = [(self._records[index].sample_id, index) for index in indices]
            keyed.sort()
            shuffled = [index for _, index in keyed]
            random.Random(f"{self._seed}:{self._epoch}").shuffle(shuffled)
            return iter(shuffled)
        return iter(indices)

    def __len__(self) -> int:
        """Return the number of sampled indices."""

        return len(self._records)


def build_sampler(
    records: list[SampleRecord],
    *,
    shuffle: bool,
    seed: int,
    class_balanced_enabled: bool = False,
) -> DeterministicSampler:
    """Build the default B0 sampler without changing data distribution.

    Args:
        records: Manifest rows.
        shuffle: Whether to seed-shuffle sample order.
        seed: Non-negative deterministic seed.
        class_balanced_enabled: Reserved explicit opt-in for class-balanced
            sampling. The F02 Agent A implementation leaves it disabled so the
            default B0 data distribution is unchanged.

    Returns:
        A `DeterministicSampler`.

    Raises:
        ValueError: If class-balanced sampling is requested without an
            implementation owned by a later task.
    """

    if class_balanced_enabled:
        raise ValueError(
            "Class-balanced sampling is not implemented and must be explicitly integrated."
        )
    return DeterministicSampler(records, shuffle=shuffle, seed=seed)
